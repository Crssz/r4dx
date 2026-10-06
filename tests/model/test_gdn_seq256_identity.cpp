// tests/model/test_gdn_seq256_identity.cpp -- the GDN prefill sequence ops (conv prep, kkt solve, chunk scan
// + state commit, gated rmsnorm) run ONCE over a whole prefill chunk (the default inside a 256-row
// super-chunk, GdnLayerParams::seq_slice 0) against the old path, the same rows as consecutive 64-row calls
// with has_init true after the first and the fp32 state handed on through the slot (R4DX_GDN_SLICE=64),
// byte for byte; and r4d_gdn_conv_prep2 (the default; R4DX_GDN_CONV=1 opts out) against r4d_gdn_conv_prep on both paths
// (docs/trellis-m256.md "GDN sequence ops").
//
// Part A (synthetic, needs only a HIP device): the real per-rank shape (H 48, Hg 16, K = V = 128, conv
// width 4, conv_dim 10240), random inputs, driven exactly the way gdn_layer.cpp drives the kernels, with
// four variants each owning its own GdnStateManager:
//   ref      = 64-row calls, r4d_gdn_conv_prep      (today's 64-row path, byte for byte)
//   one/v1   = one call,     r4d_gdn_conv_prep      (a wide Model under R4DX_GDN_CONV=1)
//   one/v2   = one call,     r4d_gdn_conv_prep2     (the default)
//   s64/v2   = 64-row calls, r4d_gdn_conv_prep2
// over a sequence of calls: a fresh 256-row call (has_init false), a 256-row continuation, 320 and 200 rows
// (not multiples of 256, and 200 not of 64: the reference splits it 64/64/64/8 as the 64-row chunk grid
// does), short calls of 17, 2 (fewer rows than the conv history: the cache rewrite reads the old cache)
// and 64 rows, a 256-row call from a random nonzero recurrent state and conv cache, and a fresh 200-row
// call. After every call each variant's q, k, v, g, beta, A, the scan output, the gated-norm output, the
// whole recurrent-state buffer and the whole conv-state buffer must equal ref's, byte for byte.
//
// Part B (needs the 4-layer bf16 test container, as test_gdn_layer): GdnLayer::Forward on layer 0 at
// T = 256, a fresh call then a has_init continuation, seq_slice 64 + conv_prep 1 (the old path) against
// seq_slice 0 with conv_prep 1 and 2: the layer output and both state buffers, byte for byte.
//
// Exit: 1 on any mismatch; 77 (CTest SKIPPED) without a HIP device, or -- after part A has passed --
// without the container, so a skip never hides a part-A regression.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "container.h"
#include "gdn_layer.h"
#include "gdn_state.h"
#include "prefill_chunk.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"
#include "test_common.h"

using r4dx::core::DeviceBuffer;
using r4dx::model::GdnControlCache;
using r4dx::model::GdnStateManager;
namespace r4d = r4dx::core::r4d;

namespace {

constexpr int kH = 48, kHg = 16, kK = 128, kV = 128, kW = 4, kBT = 64;
constexpr int64_t kConvDim = 2 * kHg * kK + kH * kV;  // 10240
constexpr int64_t kMaxT = 320;
constexpr float kSoftplusThr = 20.0f;
constexpr float kEps = 1e-6f;

int g_fail = 0;

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

template <typename T>
bool SameBytes(const char* what, const std::vector<T>& ref, const std::vector<T>& got, size_t n) {
  if (std::memcmp(ref.data(), got.data(), n * sizeof(T)) == 0) return true;
  size_t first = n, bad = 0;
  for (size_t j = 0; j < n; ++j) {
    if (std::memcmp(&ref[j], &got[j], sizeof(T)) != 0) {
      if (first == n) first = j;
      ++bad;
    }
  }
  std::printf("    %s: %zu of %zu elements differ, first at %zu\n", what, bad, n, first);
  return false;
}

// The layer weights part A uses (device), and one call's inputs.
struct Weights {
  DeviceBuffer<uint16_t> conv_w;  // [conv_dim, 4] bf16
  DeviceBuffer<float> A_log, dt_bias, norm_w;
};
struct CallInputs {
  DeviceBuffer<uint16_t> x, a, b, z;  // [T, conv_dim], [T, H], [T, H], [T, H*V]
};

// One variant's outputs for one call (sized kMaxT rows, compared over T).
struct Outputs {
  DeviceBuffer<uint16_t> q{static_cast<size_t>(kMaxT * kHg * kK)}, k{static_cast<size_t>(kMaxT * kHg * kK)},
      v{static_cast<size_t>(kMaxT * kH * kV)}, A{static_cast<size_t>(kMaxT * kH * kBT)},
      o{static_cast<size_t>(kMaxT * kH * kV)}, out{static_cast<size_t>(kMaxT * kH * kV)};
  DeviceBuffer<float> g{static_cast<size_t>(kMaxT * kH)}, beta{static_cast<size_t>(kMaxT * kH)},
      ht{static_cast<size_t>(kH * kV * kK)};
};

struct Variant {
  const char* name;
  bool sliced;  // true: consecutive <= 64-row calls (the 64-row chunk grid); false: one call
  int conv;     // r4dx::model::kGdnConvV1 / kGdnConvV2
  std::unique_ptr<GdnStateManager> st;
  Outputs out;
};

// gdn_layer.cpp's prefill block for rows [0, T) of `in`, as one call or as consecutive 64-row calls (the
// last one shorter when 64 does not divide T -- the chunks Model's 64-row grid makes). A is kept per slice
// (offset r0) so the whole T x H x 64 tensor can be compared; the scan reads exactly what kkt wrote either
// way.
void RunSeq(Variant& var, const Weights& w, const CallInputs& in, int64_t T, bool has_init,
            GdnControlCache& control, hipStream_t s) {
  GdnStateManager& st = *var.st;
  const int32_t slot = st.SlotForSeq(0);
  const int32_t* cache_idx = control.CacheIdx(slot);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kK));
  const int64_t step = var.sliced ? kBT : T;
  const auto conv = var.conv == r4dx::model::kGdnConvV1 ? &r4d::GdnConvPrep : &r4d::GdnConvPrep2;
  Outputs& o = var.out;
  float* h0 = st.RecurrentSlotPtr(slot);
  for (int64_t r0 = 0; r0 < T; r0 += step) {
    const int64_t len = std::min<int64_t>(step, T - r0);
    const int32_t* cu = control.CuPair(len);
    const uint8_t* hi = (has_init || r0 > 0) ? control.HasInitTrue() : nullptr;
    const int lt = static_cast<int>(len);
    conv(in.x.data() + r0 * kConvDim, kConvDim, w.conv_w.data(), nullptr, st.ConvBase(), st.ConvSeqStride(),
         st.ConvDimStride(), st.ConvTokStride(), cache_idx, 1, hi, in.a.data() + r0 * kH,
         in.b.data() + r0 * kH, kH, 1, w.A_log.data(), w.dt_bias.data(), o.q.data() + r0 * kHg * kK,
         o.k.data() + r0 * kHg * kK, o.v.data() + r0 * kH * kV, o.g.data() + r0 * kH,
         o.beta.data() + r0 * kH, cu, 1, lt, kH, kHg, kK, kV, kW, kSoftplusThr, s);
    r4d::GdnKktSolve(o.k.data() + r0 * kHg * kK, o.beta.data() + r0 * kH, o.g.data() + r0 * kH,
                     o.A.data() + r0 * kH * kBT, cu, 1, lt, kH, kHg, kK, kBT, s);
    r4d::GdnChunkScan(o.q.data() + r0 * kHg * kK, o.k.data() + r0 * kHg * kK, o.v.data() + r0 * kH * kV,
                      o.A.data() + r0 * kH * kBT, o.g.data() + r0 * kH, o.beta.data() + r0 * kH, h0,
                      o.o.data() + r0 * kH * kV, o.ht.data(), cu, 1, kH, kHg, kK, kV, kBT, scale, s);
    R4DX_HIP_CHECK(hipMemcpyAsync(h0, o.ht.data(), o.ht.bytes(), hipMemcpyDeviceToDevice, s));
    r4d::GdnGatedRmsNorm(o.o.data() + r0 * kH * kV, in.z.data() + r0 * kH * kV, w.norm_w.data(),
                         o.out.data() + r0 * kH * kV, len * kH, kV, kV, kV, kV, kEps, /*act=*/0, s);
  }
}

bool CompareVariant(const Variant& ref, const Variant& got, int64_t T) {
  bool ok = true;
  const size_t tq = static_cast<size_t>(T * kHg * kK), tv = static_cast<size_t>(T * kH * kV);
  const size_t tg = static_cast<size_t>(T * kH), ta = static_cast<size_t>(T * kH * kBT);
  ok &= SameBytes("q", ref.out.q.CopyToHost(), got.out.q.CopyToHost(), tq);
  ok &= SameBytes("k", ref.out.k.CopyToHost(), got.out.k.CopyToHost(), tq);
  ok &= SameBytes("v", ref.out.v.CopyToHost(), got.out.v.CopyToHost(), tv);
  ok &= SameBytes("g", ref.out.g.CopyToHost(), got.out.g.CopyToHost(), tg);
  ok &= SameBytes("beta", ref.out.beta.CopyToHost(), got.out.beta.CopyToHost(), tg);
  ok &= SameBytes("A", ref.out.A.CopyToHost(), got.out.A.CopyToHost(), ta);
  ok &= SameBytes("scan out", ref.out.o.CopyToHost(), got.out.o.CopyToHost(), tv);
  ok &= SameBytes("gated norm out", ref.out.out.CopyToHost(), got.out.out.CopyToHost(), tv);
  {
    std::vector<float> a(ref.st->RecurrentElems()), b(got.st->RecurrentElems());
    R4DX_HIP_CHECK(hipMemcpy(a.data(), ref.st->RecurrentBase(), a.size() * sizeof(float), hipMemcpyDeviceToHost));
    R4DX_HIP_CHECK(hipMemcpy(b.data(), got.st->RecurrentBase(), b.size() * sizeof(float), hipMemcpyDeviceToHost));
    ok &= SameBytes("recurrent state", a, b, a.size());
  }
  {
    std::vector<uint16_t> a(ref.st->ConvElems()), b(got.st->ConvElems());
    R4DX_HIP_CHECK(hipMemcpy(a.data(), ref.st->ConvBase(), a.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    R4DX_HIP_CHECK(hipMemcpy(b.data(), got.st->ConvBase(), b.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    ok &= SameBytes("conv state", a, b, a.size());
  }
  return ok;
}

int RunPartA() {
  std::printf("part A: synthetic GDN sequence ops, H=%d Hg=%d K=V=%d conv_dim=%lld\n", kH, kHg, kK,
              static_cast<long long>(kConvDim));
  std::mt19937 rng(20261006);
  Weights w;
  w.conv_w = r4dx_test::UploadBf16(RandBf16(rng, static_cast<size_t>(kConvDim * kW), -0.6f, 0.6f));
  {
    std::vector<float> alog(kH), dtb = RandF32(rng, kH, -1.0f, 1.0f), nw = RandF32(rng, kV, 0.5f, 1.5f);
    std::uniform_real_distribution<float> ad(0.05f, 4.0f);
    for (auto& x : alog) x = std::log(ad(rng));
    w.A_log = DeviceBuffer<float>(kH);
    w.A_log.CopyFromHost(alog);
    w.dt_bias = DeviceBuffer<float>(kH);
    w.dt_bias.CopyFromHost(dtb);
    w.norm_w = DeviceBuffer<float>(kV);
    w.norm_w.CopyFromHost(nw);
  }

  r4dx::core::Stream stream;
  const hipStream_t s = stream.get();
  GdnControlCache control;
  Variant vars[4] = {
      {"ref (64-row calls, conv v1)", true, r4dx::model::kGdnConvV1, nullptr, {}},
      {"one call, conv v1", false, r4dx::model::kGdnConvV1, nullptr, {}},
      {"one call, conv v2", false, r4dx::model::kGdnConvV2, nullptr, {}},
      {"64-row calls, conv v2", true, r4dx::model::kGdnConvV2, nullptr, {}},
  };
  for (Variant& v : vars) {
    v.st = std::make_unique<GdnStateManager>(/*max_seqs=*/1, kH, kV, kK, kConvDim, kW, /*max_decode_window=*/1);
    v.st->ZeroAll(stream);
  }

  struct Step {
    const char* name;
    int64_t T;
    bool has_init;
    int reset;  // 0 keep the state, 1 zero it, 2 a random nonzero recurrent state and conv cache
  };
  const Step steps[] = {
      {"fresh 256 (has_init false)", 256, false, 1},
      {"continue 256", 256, true, 0},
      {"continue 320", 320, true, 0},
      {"continue 200 (64/64/64/8)", 200, true, 0},
      {"continue 17", 17, true, 0},
      {"continue 2 (shorter than the conv history)", 2, true, 0},
      {"continue 64", 64, true, 0},
      {"random state, 256", 256, true, 2},
      {"continue 256 after random state", 256, true, 0},
      {"fresh 200 (has_init false)", 200, false, 1},
  };

  bool all_ok = true;
  for (const Step& st : steps) {
    if (st.reset == 1) {
      for (Variant& v : vars) v.st->ZeroAll(stream);
    } else if (st.reset == 2) {
      const auto rec = RandF32(rng, vars[0].st->RecurrentElems(), -0.05f, 0.05f);
      const auto cst = RandBf16(rng, vars[0].st->ConvElems(), -2.0f, 2.0f);
      for (Variant& v : vars) {
        R4DX_HIP_CHECK(hipMemcpy(v.st->RecurrentBase(), rec.data(), rec.size() * sizeof(float), hipMemcpyHostToDevice));
        R4DX_HIP_CHECK(hipMemcpy(v.st->ConvBase(), cst.data(), cst.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
      }
    }
    CallInputs in;
    in.x = r4dx_test::UploadBf16(RandBf16(rng, static_cast<size_t>(st.T * kConvDim), -2.0f, 2.0f));
    in.a = r4dx_test::UploadBf16(RandBf16(rng, static_cast<size_t>(st.T * kH), -2.0f, 2.0f));
    in.b = r4dx_test::UploadBf16(RandBf16(rng, static_cast<size_t>(st.T * kH), -3.0f, 3.0f));
    in.z = r4dx_test::UploadBf16(RandBf16(rng, static_cast<size_t>(st.T * kH * kV), -2.0f, 2.0f));
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    for (Variant& v : vars) {
      // poison the outputs so a row a variant fails to write cannot match by accident
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.q.data(), 0xA5, v.out.q.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.k.data(), 0xA5, v.out.k.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.v.data(), 0xA5, v.out.v.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.A.data(), 0xA5, v.out.A.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.o.data(), 0xA5, v.out.o.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.out.data(), 0xA5, v.out.out.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.g.data(), 0xA5, v.out.g.bytes(), s));
      R4DX_HIP_CHECK(hipMemsetAsync(v.out.beta.data(), 0xA5, v.out.beta.bytes(), s));
      RunSeq(v, w, in, st.T, st.has_init, control, s);
    }
    stream.Synchronize();
    for (int i = 1; i < 4; ++i) {
      const bool ok = CompareVariant(vars[0], vars[i], st.T);
      std::printf("  [%s] %-44s T=%3lld has_init=%d: %s\n", ok ? "PASS" : "FAIL", st.name,
                  static_cast<long long>(st.T), st.has_init ? 1 : 0, vars[i].name);
      all_ok = all_ok && ok;
    }
  }
  return all_ok ? 0 : 1;
}

// ---- part B: GdnLayer::Forward on the real layer 0 ---------------------------------------------------
const char* kContainerPath = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-bf16.r4dx");

int RunPartB() {
  using r4dx::model::Container;
  using r4dx::model::GdnLayer;
  using r4dx::model::GdnLayerParams;
  using r4dx::model::Layout;
  if (!r4dx_test::FileExists(kContainerPath)) return r4dx_test::SkipMissing(kContainerPath);
  std::printf("part B: GdnLayer::Forward, layer 0 of %s\n", kContainerPath);
  const Container container = Container::Load(kContainerPath, Layout::kBf16, Layout::kBf16, /*layer_limit=*/1);
  const auto& cfg = container.Config();
  if (!cfg.IsGdnLayer(0)) {
    std::fprintf(stderr, "test_gdn_seq256_identity: layer 0 is not a GDN layer\n");
    return 1;
  }
  const auto& layer0 = container.Layer(0);
  const int64_t hidden = cfg.hidden_size;
  constexpr int64_t T = 256;

  r4dx::core::Stream stream;
  r4dx::core::Arena arena(256ull << 20);
  GdnControlCache control;
  GdnLayer gdn(cfg, layer0.input_layernorm, *layer0.gdn);

  std::mt19937 rng(77);
  const std::vector<uint16_t> x1 = RandBf16(rng, static_cast<size_t>(T * hidden), -1.0f, 1.0f);
  const std::vector<uint16_t> x2 = RandBf16(rng, static_cast<size_t>(T * hidden), -1.0f, 1.0f);

  struct Run {
    const char* name;
    int64_t seq_slice;
    int conv_prep;
    std::vector<uint16_t> out1, out2, conv;
    std::vector<float> rec;
  };
  Run runs[] = {
      {"ref: seq_slice 64, conv v1", 64, r4dx::model::kGdnConvV1, {}, {}, {}, {}},
      {"seq_slice 0, conv v1", 0, r4dx::model::kGdnConvV1, {}, {}, {}, {}},
      {"seq_slice 0, conv v2", 0, r4dx::model::kGdnConvV2, {}, {}, {}, {}},
      {"seq_slice 64, conv v2", 64, r4dx::model::kGdnConvV2, {}, {}, {}, {}},
  };
  for (Run& r : runs) {
    GdnStateManager states(1, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim,
                           cfg.ConvDim(), cfg.linear_conv_kernel_dim, /*max_decode_window=*/1);
    states.ZeroAll(stream);
    auto call = [&](const std::vector<uint16_t>& xin, bool has_init, std::vector<uint16_t>* out) {
      auto x_dev = r4dx_test::UploadBf16(xin);
      DeviceBuffer<uint16_t> y(static_cast<size_t>(T * hidden));
      GdnLayerParams p;
      p.slot = states.SlotForSeq(0);
      p.is_prefill = true;
      p.has_init = has_init;
      p.seq_slice = r.seq_slice;
      p.conv_prep = r.conv_prep;
      gdn.Forward(stream, arena, states, control, x_dev.data(), y.data(), T, p);
      stream.Synchronize();
      arena.Reset();
      *out = y.CopyToHost();
    };
    call(x1, /*has_init=*/false, &r.out1);
    call(x2, /*has_init=*/true, &r.out2);
    r.rec.resize(states.RecurrentElems());
    r.conv.resize(states.ConvElems());
    R4DX_HIP_CHECK(hipMemcpy(r.rec.data(), states.RecurrentBase(), r.rec.size() * sizeof(float), hipMemcpyDeviceToHost));
    R4DX_HIP_CHECK(hipMemcpy(r.conv.data(), states.ConvBase(), r.conv.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
  }
  bool all_ok = true;
  for (int i = 1; i < 4; ++i) {
    bool ok = SameBytes("layer out, fresh call", runs[0].out1, runs[i].out1, runs[0].out1.size());
    ok &= SameBytes("layer out, continuation", runs[0].out2, runs[i].out2, runs[0].out2.size());
    ok &= SameBytes("recurrent state", runs[0].rec, runs[i].rec, runs[0].rec.size());
    ok &= SameBytes("conv state", runs[0].conv, runs[i].conv, runs[0].conv.size());
    std::printf("  [%s] T=256 fresh + continuation: %s\n", ok ? "PASS" : "FAIL", runs[i].name);
    all_ok = all_ok && ok;
  }
  return all_ok ? 0 : 1;
}

int RunTest() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    std::fprintf(stderr, "[SKIP] no HIP device\n");
    return r4dx_test::kSkipReturnCode;
  }
  R4DX_HIP_CHECK(hipSetDevice(0));  // HIP_VISIBLE_DEVICES=1 remaps physical device 1 to index 0
  const int a = RunPartA();
  if (a != 0) {
    std::printf("test_gdn_seq256_identity: part A FAILED\nFAIL\n");
    return 1;
  }
  const int b = RunPartB();
  if (b == r4dx_test::kSkipReturnCode) {
    std::printf("test_gdn_seq256_identity: part A PASS, part B skipped (no container)\n");
    return b;
  }
  std::printf(b == 0 ? "PASS\n" : "test_gdn_seq256_identity: part B FAILED\nFAIL\n");
  return b == 0 ? 0 : 1;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_gdn_seq256_identity", RunTest); }
