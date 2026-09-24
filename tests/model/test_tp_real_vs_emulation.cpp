// tests/model/test_tp_real_vs_emulation.cpp -- tensor parallel P4 (docs/tp.md 10.1; ctest LABEL
// tp2gpu, OPT-IN): the same binary and inputs through TpModel in --tp-mode emulate (both ranks on one
// device, EmulatedComm) and in --tp-mode real (one rank per GPU, HostMailboxComm) must give
// BYTE-IDENTICAL results (docs/tp.md 1.1 goal 4: same kernels, same tunings, the same exact fp32
// add). On qwen38-27b-l4-allmtp.r4dx, layouts w4a16 and mxfp4, one fixed script:
//
//   1. Reset, Prefill 70 tokens (two chunks), 16 teacher-forced DecodeStep rows (full gathered
//      logits);
//   2. 16 DecodeStepGreedy tokens continuing from there (the merged greedy pair, 7.3);
//   3. Reset, Prefill 70, 16 seeded DecodeStepSampled tokens (T 0.7, top_k 20, top_p 0.8; the merged
//      row summaries, 7.4) -- tokens and the final rng state;
//   4. Reset, Prefill 200 tokens (four chunks) -- the multi-chunk prefill;
//   5. Reset, Prefill 70, one sampled VerifyWindow of 8 rows (logits_out + row summaries, 7.6),
//      through RunCollectiveForTest (the ranks are sized with dflash_draft_k = 7, no drafter).
//
// The prefill submission bounding (docs/tp.md Appendix B N57) is ON in every run and forced to its
// finest setting (a unit after EVERY layer: the 4-layer container has no unit boundary at the default
// of 32 layers). Per layout: emulate with max_inflight 1; real with 1 -- the shipped default's
// hipStreamSynchronize path, which emulation cannot exercise (EmulatedComm already host-waits at
// every all-reduce) -- and real with 2 (the event path); for w4a16 one more real run with the
// bounding OFF. All must match byte for byte: the bounding changes when the runtime submits, never
// what the kernels compute (N64).
//
// Then, on the real w4a16 group of the K = 1 run, fault injection and recovery (docs/tp.md 2.4,
// 2.5) on real GPUs:
// rank 1 throws at its 37th all-reduce (kind 0), then rank 1 stalls 700 ms (kind 1, so rank 0's
// kernel -- on the HEADLESS device 1 -- times out after 500 ms; device 0 never spins). Each time the
// fault reaches the caller, the group is kNeedsRecovery, Reset() recovers (SelfTest included), both
// ranks agree on CallCounts(), and the rerun equals the fresh real run byte for byte. Every case ends
// with core::g_tp_collective_allocs == 0.
//
// Opt-in (docs/tp.md 10.1): unless R4DX_TP2GPU is exactly "1" it prints SKIP and exits 77; with
// fewer than two visible HIP devices it exits 77; SKIP 77 without the container; then the per-device
// free-VRAM pre-flight (exit 1). `tests\run_tests.ps1 -TwoGpu` runs it (HIP_VISIBLE_DEVICES unset:
// emulate uses the last visible ordinal = physical device 1; real: rank 0 = device 1, rank 1 =
// device 0). The main thread is the TpModel facade and makes no HIP call; the pre-flight runs on a
// helper thread.
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "r4dx/kernels/summary_sampler.hpp"
#include "test_common.h"
#include "tp_model.h"

namespace {

using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace core = r4dx::core;
namespace kernels = r4dx::kernels;

const char* kContainerPath = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx");
constexpr int64_t kLayers = 4;
constexpr int kRows = 16;
constexpr double kNeedGiB = 8.0;  // both emulated ranks sit on one device (~3.5 GiB each)

int g_failures = 0;

#define CHECK(cond, ...)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                     \
      std::fprintf(stderr, "\n");                                            \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

std::string GetEnv(const char* name) {
  char* v = nullptr;
  size_t len = 0;
  std::string out;
  if (_dupenv_s(&v, &len, name) == 0 && v != nullptr) out = v;
  std::free(v);
  return out;
}

// Deterministic in-vocabulary token ids (the test asserts engine agreement, not meaning).
std::vector<int32_t> Tokens(size_t n, uint32_t seed) {
  std::vector<int32_t> v(n);
  uint32_t x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = static_cast<int32_t>(1000u + (x >> 8) % 150000u);
  }
  return v;
}

ModelOptions BaseOptions(const std::string& layout) {
  ModelOptions o;
  o.container_path = kContainerPath;
  o.layout = r4dx::model::LayoutFromName(layout);
  o.max_ctx = 512;
  o.layer_limit = kLayers;
  o.vision = ModelOptions::VisionMode::kOff;
  o.dflash_draft_k = 7;  // sizes an 8-row verify window; no drafter is loaded
  return o;
}

TpOptions Options(TpOptions::Mode mode, int submit_layers, int max_inflight) {
  TpOptions t;
  t.world = 2;
  t.mode = mode;
  t.submit_layers = submit_layers;
  t.max_inflight_units = max_inflight;
  return t;
}

struct ScriptOut {
  std::vector<std::vector<float>> rows;  // prefill 70 + 16 DecodeStep
  std::vector<int32_t> greedy, sampled;
  std::mt19937_64 rng_after;
  std::vector<float> long_prefill;
  std::vector<int32_t> verify_preds;
  std::vector<float> verify_logits;
  std::vector<kernels::RowSummary> verify_sums;
};

ScriptOut RunScript(TpModel& m) {
  const std::vector<int32_t> p70 = Tokens(70, 23), forced = Tokens(kRows, 37), p200 = Tokens(200, 41);
  const int64_t V = m.Config().vocab_size;
  ScriptOut o;
  // 1-2. teacher-forced full rows, then greedy
  m.Reset();
  o.rows.push_back(m.Prefill(p70));
  for (int i = 0; i < kRows; ++i) o.rows.push_back(m.DecodeStep(forced[static_cast<size_t>(i)]));
  int32_t t = kernels::Argmax(o.rows.back().data(), V);
  for (int i = 0; i < kRows; ++i) {
    t = m.DecodeStepGreedy(t);
    o.greedy.push_back(t);
  }
  // 3. seeded sampled
  kernels::SampleParams sp;
  sp.temperature = 0.7f;
  sp.top_k = 20;
  sp.top_p = 0.8f;
  sp.seed = 1;
  std::mt19937_64 rng = kernels::MakeRng(1);
  m.Reset();
  const std::vector<float> lg = m.Prefill(p70);
  int32_t s = kernels::Sample(lg.data(), V, sp, rng);
  for (int i = 0; i < kRows; ++i) {
    s = m.DecodeStepSampled(s, sp, rng);
    o.sampled.push_back(s);
  }
  o.rng_after = rng;
  // 4. multi-chunk prefill
  m.Reset();
  o.long_prefill = m.Prefill(p200);
  // 5. a sampled verify window
  m.Reset();
  const std::vector<float> lv = m.Prefill(p70);
  std::vector<int32_t> cands = {kernels::Argmax(lv.data(), V)};
  for (int i = 0; i < 7; ++i) cands.push_back(forced[static_cast<size_t>(i)]);
  struct RankOut {
    std::vector<int32_t> preds;
    std::vector<float> logits;
    std::vector<kernels::RowSummary> sums;
  };
  auto outs = std::make_shared<std::vector<RankOut>>(2);
  m.RunCollectiveForTest([outs, cands](Model& mm, int rank) {
    RankOut& ro = (*outs)[static_cast<size_t>(rank)];
    ro.preds = mm.VerifyWindow(cands, &ro.logits, &ro.sums, 1.0f / 0.7f);
  });
  o.verify_preds = (*outs)[0].preds;
  o.verify_logits = (*outs)[0].logits;
  o.verify_sums = (*outs)[0].sums;
  return o;
}

bool SameFloats(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
// Field by field (a RowSummary's padding bytes carry no meaning), every float by its bits.
bool SameSums(const std::vector<kernels::RowSummary>& a, const std::vector<kernels::RowSummary>& b) {
  if (a.size() != b.size()) return false;
  for (size_t r = 0; r < a.size(); ++r) {
    const kernels::RowSummary &x = a[r], &y = b[r];
    if (x.k != y.k || x.vocab != y.vocab || std::memcmp(&x.inv_temperature, &y.inv_temperature, 4) != 0 ||
        std::memcmp(&x.lse, &y.lse, 4) != 0 ||
        std::memcmp(x.ids, y.ids, sizeof(int32_t) * static_cast<size_t>(x.k)) != 0 ||
        std::memcmp(x.vals, y.vals, sizeof(float) * static_cast<size_t>(x.k)) != 0) {
      return false;
    }
  }
  return true;
}

// Returns the number of differing parts (0 = byte-identical), printing each.
int Compare(const ScriptOut& a, const ScriptOut& b, const std::string& label) {
  int diffs = 0;
  const auto part = [&](bool same, const char* what) {
    if (!same) {
      std::fprintf(stderr, "[%s] %s differ\n", label.c_str(), what);
      ++diffs;
    }
  };
  bool rows_same = a.rows.size() == b.rows.size();
  for (size_t i = 0; rows_same && i < a.rows.size(); ++i) rows_same = SameFloats(a.rows[i], b.rows[i]);
  part(rows_same, "prefill-70 + 16 DecodeStep logits");
  part(a.greedy == b.greedy, "DecodeStepGreedy tokens");
  part(a.sampled == b.sampled, "DecodeStepSampled tokens");
  part(a.rng_after == b.rng_after, "rng states after the sampled run");
  part(SameFloats(a.long_prefill, b.long_prefill), "200-token (4-chunk) prefill logits");
  part(a.verify_preds == b.verify_preds, "VerifyWindow preds");
  part(SameFloats(a.verify_logits, b.verify_logits), "VerifyWindow logits_out");
  part(SameSums(a.verify_sums, b.verify_sums), "VerifyWindow row summaries");
  std::printf("[%s] %s\n", label.c_str(), diffs == 0 ? "byte-identical (8/8 parts)" : "DIFFERENT");
  return diffs;
}

void CheckNoCollectiveAllocs(const char* where) {
  const uint64_t n = core::g_tp_collective_allocs.load();
  CHECK(n == 0, "%s: %llu device allocation(s)/free(s) inside collective commands", where,
        static_cast<unsigned long long>(n));
}

ScriptOut RunMode(const std::string& layout, TpOptions::Mode mode, int submit_layers, int max_inflight,
                  std::unique_ptr<TpModel>* keep = nullptr) {
  std::unique_ptr<TpModel> m = TpModel::Load(BaseOptions(layout), Options(mode, submit_layers, max_inflight));
  ScriptOut o = RunScript(*m);
  const auto ss = m->SubmitStats();
  std::printf("[%s] %s submit %d/%d: ran; forced submissions %llu / %llu, cap waits %llu / %llu\n", layout.c_str(),
              mode == TpOptions::Mode::kReal ? "real" : "emulate", submit_layers, max_inflight,
              static_cast<unsigned long long>(ss[0].units), static_cast<unsigned long long>(ss[1].units),
              static_cast<unsigned long long>(ss[0].waits), static_cast<unsigned long long>(ss[1].waits));
  if (submit_layers > 0) {
    CHECK(ss[0].units > 0 && ss[0].units == ss[1].units, "[%s] the bounding forced %llu / %llu submissions",
          layout.c_str(), static_cast<unsigned long long>(ss[0].units), static_cast<unsigned long long>(ss[1].units));
  }
  if (submit_layers > 0 && max_inflight == 1) {
    // The synchronize path waits at every unit (tp_submit.cpp); the event path never counts that way.
    CHECK(ss[0].waits == ss[0].units && ss[1].waits == ss[1].units,
          "[%s] K = 1 did not synchronize at every unit (waits %llu / %llu)", layout.c_str(),
          static_cast<unsigned long long>(ss[0].waits), static_cast<unsigned long long>(ss[1].waits));
  }
  CHECK(m->GetState() == TpModel::State::kReady, "[%s] group not ready after the script", layout.c_str());
  if (keep != nullptr) *keep = std::move(m);
  return o;
}

// Real-mode faults on `m` (fresh = its fault-free run): the fault reaches the caller, recovery
// works, the rerun is byte-identical.
void FaultCycle(TpModel& m, const ScriptOut& fresh, int kind) {
  const std::string label = kind == 0 ? "real fault kind 0 (rank 1 throws at AR #37)"
                                      : "real fault kind 1 (rank 1 stalls 700 ms at AR #37)";
  m.ArmFaultInjection(1, 37, kind);
  std::string what;
  try {
    (void)RunScript(m);
  } catch (const std::exception& e) {
    what = e.what();
  }
  CHECK(!what.empty(), "[%s] the injected fault did not reach the caller", label.c_str());
  std::printf("[%s] caller saw: %s\n", label.c_str(), what.c_str());
  if (kind == 0) {
    CHECK(what.find("tp fault injection") != std::string::npos, "[%s] expected the injected exception, got: %s",
          label.c_str(), what.c_str());
  } else {
    CHECK(what.find("timeout") != std::string::npos, "[%s] expected rank 0's all-reduce timeout, got: %s",
          label.c_str(), what.c_str());
  }
  CHECK(m.GetState() == TpModel::State::kNeedsRecovery, "[%s] state after the fault is not kNeedsRecovery",
        label.c_str());
  try {
    m.Reset();
  } catch (const std::exception& e) {
    CHECK(false, "[%s] Reset() did not recover: %s", label.c_str(), e.what());
    return;
  }
  CHECK(m.GetState() == TpModel::State::kReady, "[%s] state after Reset() is not kReady", label.c_str());
  const auto counts = m.CallCounts();
  CHECK(counts.size() == 2 && counts[0] == counts[1], "[%s] CallCounts differ after recovery", label.c_str());
  const ScriptOut rerun = RunScript(m);
  CHECK(Compare(fresh, rerun, label + ": rerun vs fresh") == 0, "[%s] the rerun after recovery differs from a fresh run",
        label.c_str());
}

// On a helper thread (the main thread is the TpModel facade): "" or the pre-flight failure.
std::string Preflight(int* visible_out) {
  std::string msg;
  int visible = 0;
  std::thread t([&] {
    if (hipGetDeviceCount(&visible) != hipSuccess) visible = 0;
    for (int d = 0; d < visible && msg.empty(); ++d) {
      size_t free_b = 0, total_b = 0;
      if (hipSetDevice(d) != hipSuccess || hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        msg = "cannot query HIP device " + std::to_string(d);
      } else if (static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0) < kNeedGiB) {
        char b[200];
        std::snprintf(b, sizeof b, "need %.1f GiB free on HIP device %d, have %.2f GiB -- is the production server running?",
                      kNeedGiB, d, static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0));
        msg = b;
      }
    }
    (void)hipGetLastError();
  });
  t.join();
  *visible_out = visible;
  return msg;
}

int RunTest() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep stdout in order with the facade's stderr lines
  if (GetEnv("R4DX_TP2GPU") != "1") {
    std::printf("SKIP: two-GPU test; run tests\\run_tests.ps1 -TwoGpu\n");
    return 77;
  }
  if (!r4dx_test::FileExists(kContainerPath)) return r4dx_test::SkipMissing(kContainerPath);
  int visible = 0;
  const std::string pre = Preflight(&visible);
  if (visible < 2) {
    std::printf("SKIP: two-GPU test needs two visible HIP devices, HIP_VISIBLE_DEVICES exposes %d\n", visible);
    return 77;
  }
  if (!pre.empty()) {
    std::fprintf(stderr, "test_tp_real_vs_emulation: %s\n", pre.c_str());
    return 1;
  }

  using M = TpOptions::Mode;
  for (const char* layout : {"w4a16", "mxfp4"}) {
    std::printf("==== layout %s ====\n", layout);
    const ScriptOut emu = RunMode(layout, M::kEmulate, 1, 1);
    std::unique_ptr<TpModel> real_group;
    const ScriptOut real = RunMode(layout, M::kReal, 1, 1, &real_group);
    CHECK(Compare(emu, real, std::string(layout) + ": emulate (submit 1/1) vs real (submit 1/1)") == 0,
          "[%s] real two-GPU results (synchronize path) differ from emulation", layout);
    const bool w4a16 = std::string(layout) == "w4a16";
    if (w4a16) {
      FaultCycle(*real_group, real, 0);
      FaultCycle(*real_group, real, 1);
    }
    real_group.reset();
    const ScriptOut real_ev = RunMode(layout, M::kReal, 1, 2);
    CHECK(Compare(emu, real_ev, std::string(layout) + ": emulate (submit 1/1) vs real (submit 1/2)") == 0,
          "[%s] real two-GPU results (event path) differ from emulation", layout);
    if (w4a16) {
      const ScriptOut off = RunMode(layout, M::kReal, 0, 0);
      CHECK(Compare(real, off, std::string(layout) + ": real bounded (1/1) vs real unbounded (0/0)") == 0,
            "[%s] the submission bounding changed the results", layout);
    }
    CheckNoCollectiveAllocs(layout);
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "test_tp_real_vs_emulation: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("[PASS] test_tp_real_vs_emulation\n");
  return 0;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_tp_real_vs_emulation", [] { return RunTest(); }); }
