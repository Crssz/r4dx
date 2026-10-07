// tests/model/test_gdn_write_once.cpp -- kernel level, GPU (device 1): the write-once GDN recurrent update
// (r4d_gdn_recurrent_update_wo_*, r4d_gdn_state_replay_*, GdnStateManager's write-once mode;
// docs/gdn-write-once.md) against the legacy window-slot kernel, bit for bit. Needs only a HIP device.
//
// Both sides are fed the same random bf16 q / k / v / a / b / z and the same random fp32 seed state; the
// legacy side stores one state per row into its window slots and picks the next call's seed through
// num_accepted, the write-once side logs the rows, keeps one state B and replays the committed prefix
// in the next call. They are driven exactly as GdnLayer::Forward drives them (see Launch below), and after
// every call must agree:
//   a. `o`, the outputs of every row: bitwise equal (the register chain never changes);
//   b. the LOGICAL state after committing n rows -- legacy window slot n-1, write-once B + the first n logged
//      rows replayed into a scratch slot (MaterializeTo): bitwise equal for EVERY n in 1..T;
//   c. chains: scripted and random sequences of (T, n) rounds, T = 1 plain steps (the direct path, with and
//      without a pending prefix), a full-window commit, a commit of 1, and a final in-place Materialize;
//   d. a call without a pending prefix leaves B untouched (the log is the only thing it writes); a pending
//      prefix of the full window replays correctly; the ping-pong never overwrites the log a pending prefix
//      still needs;
//   e. the shapes: the real rank shapes H/Hg = 48/16 (TP=1) and 24/8 (TP=2) with the fused norm (one
//      workgroup per head), and a no-norm path with VS > 1 (H/Hg = 48/16 -> VS 2, 16/8 -> VS 4) with fp32 a/b;
//   f. refusals: N != 1, a pending prefix without a log, a missing log_out; a NULL_BLOCK_ID slot is skipped.
//
// Exit: 1 on any mismatch; 77 (CTest SKIPPED) without a HIP device.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gdn_state.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"

using r4dx::core::DeviceBuffer;
using r4dx::model::GdnStateManager;
namespace r4d = r4dx::core::r4d;

namespace {

constexpr int kK = 128, kV = 128, kWindow = 8;
constexpr float kSoftplusThr = 20.0f, kEps = 1e-6f;

int g_fail = 0;
#define CHECKF(cond, ...)                                       \
  do {                                                          \
    if (!(cond)) {                                              \
      ++g_fail;                                                 \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                        \
      std::fprintf(stderr, "\n");                               \
    }                                                           \
  } while (0)

struct Shape {
  const char* name;
  int H, Hg;
  bool fused_norm;  // false: the no-norm path (VS > 1), fp32 a / b
};

std::vector<uint16_t> RandBf16(std::mt19937& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = r4dx::core::FloatToBf16(d(rng));
  return v;
}
std::vector<float> RandF32(std::mt19937& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// The kernels' fixed per-shape weights and one call's inputs.
struct Weights {
  DeviceBuffer<float> A_log, dt_bias, norm_w;
};
struct CallInputs {
  DeviceBuffer<uint16_t> q, k, v, z;            // [T, Hg, K], [T, Hg, K], [T, H, V], [T, H, V]
  DeviceBuffer<uint16_t> a16, b16;              // [T, H] bf16 (fused-norm shapes)
  DeviceBuffer<float> a32, b32;                 // [T, H] fp32 (the no-norm shape)
  DeviceBuffer<int32_t> cu;                     // {0, T}
  DeviceBuffer<uint16_t> out_legacy, out_wo;    // [T, H, V] bf16 outputs
};

CallInputs MakeInputs(std::mt19937& rng, const Shape& sh, int T) {
  CallInputs in;
  const size_t H = static_cast<size_t>(sh.H), Hg = static_cast<size_t>(sh.Hg), t = static_cast<size_t>(T);
  in.q = DeviceBuffer<uint16_t>(t * Hg * kK);
  in.k = DeviceBuffer<uint16_t>(t * Hg * kK);
  in.v = DeviceBuffer<uint16_t>(t * H * kV);
  in.z = DeviceBuffer<uint16_t>(t * H * kV);
  in.q.CopyFromHost(RandBf16(rng, t * Hg * kK, -1.0f, 1.0f));
  in.k.CopyFromHost(RandBf16(rng, t * Hg * kK, -1.0f, 1.0f));
  in.v.CopyFromHost(RandBf16(rng, t * H * kV, -1.0f, 1.0f));
  in.z.CopyFromHost(RandBf16(rng, t * H * kV, -2.0f, 2.0f));
  if (sh.fused_norm) {
    in.a16 = DeviceBuffer<uint16_t>(t * H);
    in.b16 = DeviceBuffer<uint16_t>(t * H);
    in.a16.CopyFromHost(RandBf16(rng, t * H, -1.0f, 1.0f));
    in.b16.CopyFromHost(RandBf16(rng, t * H, -2.0f, 2.0f));
  } else {
    in.a32 = DeviceBuffer<float>(t * H);
    in.b32 = DeviceBuffer<float>(t * H);
    in.a32.CopyFromHost(RandF32(rng, t * H, -1.0f, 1.0f));
    in.b32.CopyFromHost(RandF32(rng, t * H, -2.0f, 2.0f));
  }
  in.cu = DeviceBuffer<int32_t>(2);
  const int32_t cu[2] = {0, T};
  in.cu.CopyFromHost(cu, 2);
  in.out_legacy = DeviceBuffer<uint16_t>(t * H * kV);
  in.out_wo = DeviceBuffer<uint16_t>(t * H * kV);
  return in;
}

// One side's state: a manager (legacy or write-once), the device arrays a launch needs.
struct Side {
  std::unique_ptr<GdnStateManager> st;
  DeviceBuffer<int32_t> sidx;     // {1, 2, ..., window}: the stable ascending array GdnControlCache::SidxBase makes
  DeviceBuffer<int32_t> count;    // the acceptance thread, mtp_num_accepted_dev_
  DeviceBuffer<int32_t> sidx_null;  // {0, ...}: a NULL_BLOCK_ID base slot
  // host bookkeeping (what Model keeps)
  bool count_valid = false;  // legacy: naccept armed
  int committed = 1;         // the value in `count`
  bool pending = false;      // write-once: a committed prefix of the last T > 1 call's log is unapplied
  int last_T = 1;
};

Side MakeSide(const Shape& sh, bool write_once) {
  Side s;
  const int64_t conv_dim = 2 * sh.Hg * kK + static_cast<int64_t>(sh.H) * kV;
  s.st = std::make_unique<GdnStateManager>(1, sh.H, kV, kK, conv_dim, 4, kWindow, write_once);
  s.sidx = DeviceBuffer<int32_t>(kWindow);
  std::vector<int32_t> ids(kWindow);
  for (int i = 0; i < kWindow; ++i) ids[static_cast<size_t>(i)] = s.st->SlotForSeq(0) + i;
  s.sidx.CopyFromHost(ids);
  s.sidx_null = DeviceBuffer<int32_t>(kWindow);
  std::vector<int32_t> zeros(kWindow, 0);
  s.sidx_null.CopyFromHost(zeros);
  s.count = DeviceBuffer<int32_t>(1);
  return s;
}

void SetCount(Side& s, int n) {
  s.count.CopyFromHost(&n, 1);
  s.committed = n;
  s.count_valid = true;
}

float* LiveSlot(Side& s) { return s.st->RecurrentSlotPtr(s.st->SlotForSeq(0)); }

// The launch GdnLayer::Forward makes for this side. legacy: the window-slot kernel, seeded by naccept.
// write-once: T == 1 without a pending prefix = the legacy kernel in place; T == 1 with one = DIRECT; T > 1 = LOG.
void Launch(Side& s, const Shape& sh, const Weights& w, const CallInputs& in, int T, uint16_t* out,
            hipStream_t stream, bool null_slot = false) {
  GdnStateManager& st = *s.st;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kK));
  const int32_t* sidx = null_slot ? s.sidx_null.data() : s.sidx.data();
  const void* a = sh.fused_norm ? static_cast<const void*>(in.a16.data()) : static_cast<const void*>(in.a32.data());
  const void* b = sh.fused_norm ? static_cast<const void*>(in.b16.data()) : static_cast<const void*>(in.b32.data());
  const int bf16 = sh.fused_norm ? 1 : 0;
  const void* z = sh.fused_norm ? static_cast<const void*>(in.z.data()) : nullptr;
  const void* nw = sh.fused_norm ? static_cast<const void*>(w.norm_w.data()) : nullptr;
  auto legacy = [&](const int32_t* naccept) {
    r4d::GdnRecurrentUpdate(in.q.data(), in.k.data(), in.v.data(), a, b, sh.H, bf16, w.A_log.data(), w.dt_bias.data(),
                            st.RecurrentBase(), st.RecurrentSlotStride(), st.RecurrentHeadStride(), out, in.cu.data(),
                            sidx, kWindow, naccept, z, nw, kEps, /*act=*/0, 1, sh.H, sh.Hg, kK, kV, scale,
                            kSoftplusThr, stream);
  };
  if (!st.WriteOnce()) {
    legacy(s.count_valid ? s.count.data() : nullptr);
    return;
  }
  const int32_t* pend = s.pending ? s.count.data() : nullptr;
  if (T == 1 && pend == nullptr) {
    legacy(nullptr);
    return;
  }
  const bool direct = T == 1;
  r4d::GdnRecurrentUpdateWo(in.q.data(), in.k.data(), in.v.data(), a, b, sh.H, bf16, w.A_log.data(),
                            w.dt_bias.data(), st.RecurrentBase(), st.RecurrentSlotStride(), st.RecurrentHeadStride(),
                            out, in.cu.data(), sidx, z, nw, kEps, 0, pend != nullptr ? st.LogIn() : nullptr,
                            direct ? nullptr : st.LogOut(), pend, kWindow,
                            direct ? R4D_GDN_WO_DIRECT : R4D_GDN_WO_LOG, 1, sh.H, sh.Hg, kK, kV, scale,
                            kSoftplusThr, stream);
  if (!direct) st.FlipParity();
}

std::vector<float> DeviceFloats(const float* p, size_t n) {
  std::vector<float> v(n);
  R4DX_HIP_CHECK(hipMemcpy(v.data(), p, n * sizeof(float), hipMemcpyDeviceToHost));
  return v;
}
bool SameBytes(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The logical live state of a side into `scratch` (stride floats): legacy window slot committed-1 (window 0
// when the thread is not armed), write-once B + the pending prefix replayed (nothing replayed when none).
std::vector<float> Logical(Side& s, DeviceBuffer<float>& scratch, r4dx::core::Stream& stream) {
  const size_t n = static_cast<size_t>(s.st->RecurrentSlotStride());
  stream.Synchronize();
  if (!s.st->WriteOnce()) {
    const int slot = s.st->WindowSlot(0, s.count_valid ? s.committed - 1 : 0);
    return DeviceFloats(s.st->RecurrentSlotPtr(slot), n);
  }
  s.st->MaterializeTo(0, s.pending ? s.committed : 0, scratch.data(), stream);
  stream.Synchronize();
  return DeviceFloats(scratch.data(), n);
}

void SeedState(Side& s, const std::vector<float>& seed, r4dx::core::Stream& stream) {
  stream.Synchronize();
  R4DX_HIP_CHECK(hipMemcpy(LiveSlot(s), seed.data(), seed.size() * sizeof(float), hipMemcpyHostToDevice));
}

struct Setup {
  Shape sh;
  Weights w;
  Side legacy, wo;
  DeviceBuffer<float> scratch;
  std::vector<float> seed;
};

Setup MakeSetup(std::mt19937& rng, const Shape& sh) {
  Setup S{sh, {}, MakeSide(sh, false), MakeSide(sh, true), {}, {}};
  S.w.A_log = DeviceBuffer<float>(static_cast<size_t>(sh.H));
  S.w.dt_bias = DeviceBuffer<float>(static_cast<size_t>(sh.H));
  S.w.norm_w = DeviceBuffer<float>(kV);
  std::vector<float> alog(static_cast<size_t>(sh.H));
  std::uniform_real_distribution<float> ad(0.05f, 4.0f);
  for (auto& x : alog) x = std::log(ad(rng));
  S.w.A_log.CopyFromHost(alog);
  S.w.dt_bias.CopyFromHost(RandF32(rng, static_cast<size_t>(sh.H), -1.0f, 1.0f));
  S.w.norm_w.CopyFromHost(RandF32(rng, kV, 0.5f, 1.5f));
  S.scratch = DeviceBuffer<float>(static_cast<size_t>(sh.H) * kV * kK);
  S.seed = RandF32(rng, static_cast<size_t>(sh.H) * kV * kK, -0.1f, 0.1f);
  return S;
}

void ResetBoth(Setup& S, std::mt19937& rng, r4dx::core::Stream& stream) {
  S.seed = RandF32(rng, S.seed.size(), -0.1f, 0.1f);
  stream.Synchronize();
  S.legacy.st->ZeroAll(stream);
  S.wo.st->ZeroAll(stream);
  stream.Synchronize();
  SeedState(S.legacy, S.seed, stream);
  SeedState(S.wo, S.seed, stream);
  S.legacy.count_valid = false;
  S.legacy.committed = 1;
  S.wo.count_valid = false;
  S.wo.pending = false;
  S.wo.committed = 1;
  S.wo.last_T = 1;
}

// One round on both sides: verify T rows then commit n (the Model's CommitVerifiedWindow). `plain` marks a
// plain decode step (commit 1, nothing pending afterwards, as RunChunk's tail does).
void Round(Setup& S, std::mt19937& rng, int T, int n, bool plain, r4dx::core::Stream& stream, const char* what) {
  CallInputs in = MakeInputs(rng, S.sh, T);
  const size_t out_elems = static_cast<size_t>(T) * S.sh.H * kV;
  Launch(S.legacy, S.sh, S.w, in, T, in.out_legacy.data(), stream.get());
  Launch(S.wo, S.sh, S.w, in, T, in.out_wo.data(), stream.get());
  stream.Synchronize();
  // (a) outputs
  const auto ol = in.out_legacy.CopyToHost(), ow = in.out_wo.CopyToHost();
  CHECKF(ol.size() == out_elems && std::memcmp(ol.data(), ow.data(), out_elems * sizeof(uint16_t)) == 0,
         "[%s %s] T=%d: row outputs differ between the window-slot and the write-once kernel", S.sh.name, what, T);
  // the call consumed the pending prefix (write-once); the legacy side keeps its thread
  if (S.wo.st->WriteOnce()) {
    S.wo.pending = false;
    S.wo.last_T = T;
  }
  // commit
  SetCount(S.legacy, n);
  SetCount(S.wo, n);
  S.wo.pending = !plain && T > 1;
  if (plain) {  // a plain decode step: the legacy side threads 1; the write-once side has nothing pending
    SetCount(S.legacy, 1);
    SetCount(S.wo, 1);
    S.wo.pending = false;
  }
  // (b) the logical state
  const auto sl = Logical(S.legacy, S.scratch, stream), sw = Logical(S.wo, S.scratch, stream);
  CHECKF(SameBytes(sl, sw), "[%s %s] T=%d n=%d: the logical state after the commit differs (window slot %d vs "
                            "B + %d replayed row(s))", S.sh.name, what, T, n, S.legacy.committed - 1, S.wo.pending ? n : 0);
}

// (a) + (b) for every n of one call from a fresh seed, the pending-less path.
void TestEveryN(Setup& S, std::mt19937& rng, r4dx::core::Stream& stream) {
  for (int T : {1, 2, 5, 8}) {
    ResetBoth(S, rng, stream);
    CallInputs in = MakeInputs(rng, S.sh, T);
    Launch(S.legacy, S.sh, S.w, in, T, in.out_legacy.data(), stream.get());
    Launch(S.wo, S.sh, S.w, in, T, in.out_wo.data(), stream.get());
    stream.Synchronize();
    const size_t out_elems = static_cast<size_t>(T) * S.sh.H * kV;
    const auto ol = in.out_legacy.CopyToHost(), ow = in.out_wo.CopyToHost();
    CHECKF(std::memcmp(ol.data(), ow.data(), out_elems * sizeof(uint16_t)) == 0, "[%s] T=%d: outputs differ", S.sh.name, T);
    const size_t n_state = static_cast<size_t>(S.legacy.st->RecurrentSlotStride());
    if (T > 1) {
      // (d) no pending prefix: B is untouched by the logging call
      const auto b = DeviceFloats(LiveSlot(S.wo), n_state);
      CHECKF(SameBytes(b, S.seed), "[%s] T=%d: a call without a pending prefix wrote B", S.sh.name, T);
      for (int n = 1; n <= T; ++n) {
        S.wo.st->MaterializeTo(0, n, S.scratch.data(), stream);
        stream.Synchronize();
        const auto rep = DeviceFloats(S.scratch.data(), n_state);
        const auto want = DeviceFloats(S.legacy.st->RecurrentSlotPtr(S.legacy.st->WindowSlot(0, n - 1)), n_state);
        CHECKF(SameBytes(rep, want), "[%s] T=%d: replaying %d logged row(s) is not window slot %d", S.sh.name, T, n, n - 1);
      }
      // an in-place Materialize of n rows leaves B equal to that replay
      const int n = T;
      S.wo.st->Materialize(0, n, stream);
      stream.Synchronize();
      const auto b2 = DeviceFloats(LiveSlot(S.wo), n_state);
      const auto want = DeviceFloats(S.legacy.st->RecurrentSlotPtr(S.legacy.st->WindowSlot(0, n - 1)), n_state);
      CHECKF(SameBytes(b2, want), "[%s] T=%d: in-place Materialize(%d) differs from window slot %d", S.sh.name, T, n, n - 1);
    } else {
      // T == 1 without a pending prefix: the state is stored in place, exactly the legacy window 0 slot
      const auto b = DeviceFloats(LiveSlot(S.wo), n_state);
      const auto want = DeviceFloats(LiveSlot(S.legacy), n_state);
      CHECKF(SameBytes(b, want), "[%s] T=1: the in-place state differs from the legacy window 0 slot", S.sh.name);
    }
  }
  std::printf("[ok] %s: outputs equal; replay of n logged rows == window slot n-1 for every n of T in {1,2,5,8}\n", S.sh.name);
}

struct Step {
  int T, n;
  bool plain;
};

void RunScript(Setup& S, std::mt19937& rng, r4dx::core::Stream& stream, const std::vector<Step>& script, const char* label) {
  ResetBoth(S, rng, stream);
  int i = 0;
  for (const Step& st : script) Round(S, rng, st.T, st.n, st.plain, stream, label), ++i;
  // the final materialisation: the in-place flush equals the legacy live slot
  const auto want = Logical(S.legacy, S.scratch, stream);
  if (S.wo.pending) {
    S.wo.st->Materialize(0, S.wo.committed, stream);
    S.wo.pending = false;
  }
  stream.Synchronize();
  const auto got = DeviceFloats(LiveSlot(S.wo), want.size());
  CHECKF(SameBytes(want, got), "[%s %s] the final in-place Materialize differs from the legacy live state", S.sh.name, label);
}

void TestChains(Setup& S, std::mt19937& rng, r4dx::core::Stream& stream) {
  // the design's script, then the corner cases: full-window commits back to back, commit 1 after a T > 1 round,
  // a plain step after a multi-token commit, rounds with T < the window, a T = 1 verify between rounds
  RunScript(S, rng, stream, {{8, 3, false}, {8, 8, false}, {8, 1, false}, {4, 2, false}, {1, 1, true}, {5, 5, false}}, "design chain");
  RunScript(S, rng, stream, {{8, 8, false}, {8, 8, false}, {8, 8, false}}, "full-window commits");
  RunScript(S, rng, stream, {{8, 1, false}, {8, 1, false}, {8, 1, false}, {1, 1, true}}, "anchor-only commits");
  RunScript(S, rng, stream, {{3, 3, false}, {1, 1, true}, {1, 1, true}, {8, 5, false}, {2, 2, false}}, "plain between rounds");
  RunScript(S, rng, stream, {{2, 2, false}, {1, 1, false}, {8, 4, false}}, "T = 1 verify between rounds");
  RunScript(S, rng, stream, {{1, 1, true}, {1, 1, true}, {8, 7, false}}, "plain first");
  int steps = 0;
  for (int seed = 1; seed <= 12; ++seed) {
    std::mt19937 r(static_cast<uint32_t>(seed) * 7919u + 3);
    std::vector<Step> script;
    for (int i = 0; i < 10; ++i) {
      if (r() % 4 == 0) {
        script.push_back({1, 1, true});
      } else {
        const int T = 1 + static_cast<int>(r() % kWindow);
        script.push_back({T, 1 + static_cast<int>(r() % static_cast<unsigned>(T)), false});
      }
    }
    steps += static_cast<int>(script.size());
    RunScript(S, rng, stream, script, ("random script " + std::to_string(seed)).c_str());
  }
  std::printf("[ok] %s: scripted chains + 12 random scripts (%d rounds): outputs, logical state and the final flush equal\n",
              S.sh.name, steps);
}

// f. refusals and the null slot
void TestRefusals(Setup& S, std::mt19937& rng, r4dx::core::Stream& stream) {
  ResetBoth(S, rng, stream);
  CallInputs in = MakeInputs(rng, S.sh, 4);
  GdnStateManager& st = *S.wo.st;
  const float scale = 0.088f;
  const void* a = S.sh.fused_norm ? static_cast<const void*>(in.a16.data()) : static_cast<const void*>(in.a32.data());
  const void* b = S.sh.fused_norm ? static_cast<const void*>(in.b16.data()) : static_cast<const void*>(in.b32.data());
  const void* z = S.sh.fused_norm ? static_cast<const void*>(in.z.data()) : nullptr;
  const void* nw = S.sh.fused_norm ? static_cast<const void*>(S.w.norm_w.data()) : nullptr;
  auto call = [&](const void* log_in, void* log_out, const void* pend, int mode, int N, const int32_t* sidx) {
    r4d::GdnRecurrentUpdateWo(in.q.data(), in.k.data(), in.v.data(), a, b, S.sh.H, S.sh.fused_norm ? 1 : 0, S.w.A_log.data(),
                              S.w.dt_bias.data(), st.RecurrentBase(), st.RecurrentSlotStride(), st.RecurrentHeadStride(),
                              in.out_wo.data(), in.cu.data(), sidx, z, nw, kEps, 0, log_in, log_out, pend, kWindow, mode, N,
                              S.sh.H, S.sh.Hg, kK, kV, scale, kSoftplusThr, stream.get());
  };
  const auto throws = [&](auto&& f) {
    try {
      f();
    } catch (const r4dx::core::R4dError&) {
      return true;
    }
    return false;
  };
  CHECKF(throws([&] { call(nullptr, st.LogOut(), nullptr, R4D_GDN_WO_LOG, 2, S.wo.sidx.data()); }), "[%s] N = 2 was accepted", S.sh.name);
  CHECKF(throws([&] { call(nullptr, st.LogOut(), S.wo.count.data(), R4D_GDN_WO_LOG, 1, S.wo.sidx.data()); }),
         "[%s] a pending prefix without a log_in was accepted", S.sh.name);
  CHECKF(throws([&] { call(nullptr, nullptr, nullptr, R4D_GDN_WO_LOG, 1, S.wo.sidx.data()); }), "[%s] LOG without a log_out was accepted", S.sh.name);
  CHECKF(throws([&] { call(nullptr, nullptr, nullptr, 7, 1, S.wo.sidx.data()); }), "[%s] an unknown mode was accepted", S.sh.name);
  // a NULL_BLOCK_ID base slot: the kernel does nothing (the output keeps its sentinel, B is untouched)
  std::vector<uint16_t> sentinel(in.out_wo.size(), 0xABCDu);
  in.out_wo.CopyFromHost(sentinel);
  const auto before = DeviceFloats(LiveSlot(S.wo), static_cast<size_t>(st.RecurrentSlotStride()));
  call(nullptr, st.LogOut(), nullptr, R4D_GDN_WO_LOG, 1, S.wo.sidx_null.data());
  stream.Synchronize();
  const auto after = DeviceFloats(LiveSlot(S.wo), before.size());
  CHECKF(SameBytes(before, after) && in.out_wo.CopyToHost() == sentinel, "[%s] a NULL_BLOCK_ID slot was not skipped", S.sh.name);
  // the manager's own contract
  CHECKF(st.WriteOnce() && st.RecurrentElems() == static_cast<size_t>(2 * st.RecurrentSlotStride()),
         "[%s] a write-once manager must hold B plus the null slot", S.sh.name);
  bool threw = false;
  try {
    (void)st.WindowSlot(0, 1);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECKF(threw, "[%s] WindowSlot on a write-once manager did not throw", S.sh.name);
  CHECKF(S.legacy.st->RecurrentElems() == static_cast<size_t>((kWindow + 1) * S.legacy.st->RecurrentSlotStride()),
         "[%s] the legacy manager lost a window slot", S.sh.name);
  std::printf("[ok] %s: refusals, NULL_BLOCK_ID skip, manager geometry (%.1f MiB legacy vs %.1f MiB + logs)\n", S.sh.name,
              S.legacy.st->RecurrentElems() * 4 / 1048576.0, st.RecurrentElems() * 4 / 1048576.0);
}

int Run() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    std::fprintf(stderr, "[SKIP] no HIP device\n");
    return 77;
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  r4dx::core::Stream stream;
  std::mt19937 rng(20261008);
  const Shape shapes[] = {
      {"H48/Hg16 fused norm (TP=1)", 48, 16, true},
      {"H24/Hg8 fused norm (TP=2 rank)", 24, 8, true},
      {"H48/Hg16 no norm, VS 2, fp32 a/b", 48, 16, false},
      {"H16/Hg8 no norm, VS 4, fp32 a/b", 16, 8, false},
  };
  for (const Shape& sh : shapes) {
    Setup S = MakeSetup(rng, sh);
    TestEveryN(S, rng, stream);
    TestChains(S, rng, stream);
    TestRefusals(S, rng, stream);
  }
  if (g_fail != 0) {
    std::fprintf(stderr, "test_gdn_write_once: %d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("test_gdn_write_once: OK\n");
  return 0;
}

}  // namespace

int main() {
  try {
    return Run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gdn_write_once: exception: %s\n", e.what());
    return 1;
  }
}
