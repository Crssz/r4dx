// tests/model/test_batch_decode_tp.cpp -- batched decode through the TextModel interface under TENSOR PARALLELISM, and in the HYBRID mode
// (docs/batch-decode.md 5): TpModel::BatchImport / DecodeBatch against the same TpModel decoding each sequence alone.
//
//   --rig emulate   TpModel in --tp-mode emulate (both ranks on ONE device, host-synchronized exact all-reduces): the lockstep fingerprint, the
//                   merged greedy pairs, the merged row summaries and the per-row rng copies of a batched step, on the 4-layer container.
//                   HIP_VISIBLE_DEVICES=1 (CMake).
//   --rig hybrid    TpModel with the hybrid engaged (`--tp 2 --pp 2`, two cards, min rows 1: EVERY Prefill runs the PP-2 pipeline and is resharded
//                   into the ranks): the prompt is prefilled by the pipeline, imported into a slot, and decoded as TP=2 -- chains equal the alone chains
//                   of the same hybrid prefill. Both cards visible (CMake: HIP_VISIBLE_DEVICES=0,1); SKIPs (77) when fewer than two devices are.
//
// The comparison is exact in both: the alone run and the batched run share every rank's numerics (same shards, same all-reduces), and a row's bits
// do not depend on the other rows (docs/batch-decode.md 6). Checked per rig:
//   1. lockstep: three prompts (40, 70, 130 tokens) imported into slots 0..2 and decoded together == each decoded alone (TextModel::DecodeStepGreedy);
//   2. a greedy row batched with two sampled ones (summary path, different params) == DecodeStepSampled alone, same seeds, and the callers'
//      generators end exactly where the alone ones do (TpModel hands each rank a copy and takes rank 0's back);
//   3. the group stays kReady throughout; the single-sequence state is where the imports left it (PositionCount) and still decodes;
//   4. (hybrid) the prompts really went through the pipeline: HybridStats::pipelined_calls grew by one per prefill.
//
// Written, NOT run by its author (CPU-only session): it has never met a GPU. Output: "[PASS] ..." lines; exit 0 on success, 1 on a failure, 77
// when the container or the second device is missing.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "r4dx/kernels/sampler.hpp"
#include "test_common.h"
#include "text_model.h"
#include "tp_model.h"

using namespace r4dx_test;
using r4dx::model::BatchDecodeRow;
using r4dx::model::Layout;
using r4dx::model::ModelOptions;
using r4dx::model::PpOptions;
using r4dx::model::TextModel;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;

namespace {

const char* kContainerPath = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-bf16.r4dx");

int g_failures = 0;

#define CHECK(cond, ...)                                                                  \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s -- ", __FILE__, __LINE__, #cond);  \
      std::fprintf(stderr, __VA_ARGS__);                                                  \
      std::fprintf(stderr, "\n");                                                         \
      ++g_failures;                                                                       \
    }                                                                                     \
  } while (0)

std::vector<int32_t> MakePrompt(int n, int salt) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 100 + (i * 37 + salt * 211 + (i / 5) * 13) % 5000;
  return ids;
}

int32_t Argmax(const std::vector<float>& v) {
  return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

struct SeqSpec {
  std::vector<int32_t> prompt;
  r4dx::kernels::SampleParams params;
  uint64_t seed = 0;
};

std::vector<int32_t> Alone(TextModel& m, const SeqSpec& s, int n) {
  m.Reset();
  const std::vector<float> logits = m.Prefill(s.prompt);
  std::mt19937_64 rng = r4dx::kernels::MakeRng(s.seed);
  const bool greedy = !(s.params.temperature > 0.0f);
  int32_t tok = greedy ? Argmax(logits) : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()), s.params, rng);
  std::vector<int32_t> out;
  for (int i = 0; i < n; ++i) {
    out.push_back(tok);
    if (i + 1 == n) break;
    tok = greedy ? m.DecodeStepGreedy(tok) : m.DecodeStepSampled(tok, s.params, rng);
  }
  return out;
}

struct Live {
  int slot = 0;
  SeqSpec spec;
  std::mt19937_64 rng;
  std::vector<int32_t> chain;
  bool greedy() const { return !(spec.params.temperature > 0.0f); }
};

Live Start(TextModel& m, int slot, const SeqSpec& spec) {
  Live l;
  l.slot = slot;
  l.spec = spec;
  l.rng = r4dx::kernels::MakeRng(spec.seed);
  m.Reset();
  const std::vector<float> logits = m.Prefill(spec.prompt);
  l.chain.push_back(l.greedy() ? Argmax(logits)
                               : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()), spec.params, l.rng));
  m.BatchImport(slot);
  return l;
}

void Step(TextModel& m, std::vector<Live>& all) {
  std::vector<BatchDecodeRow> rows;
  for (Live& l : all) {
    BatchDecodeRow r;
    r.slot = l.slot;
    r.token = l.chain.back();
    r.params = l.spec.params;
    r.rng = l.greedy() ? nullptr : &l.rng;
    rows.push_back(r);
  }
  const std::vector<int32_t> next = m.DecodeBatch(rows);
  CHECK(next.size() == rows.size(), "DecodeBatch returned %zu tokens for %zu rows", next.size(), rows.size());
  for (size_t k = 0; k < all.size() && k < next.size(); ++k) all[k].chain.push_back(next[k]);
}

void Compare(const char* tag, const char* what, int seq, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
  const size_t n = std::min(got.size(), want.size());
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "[FAIL] %s %s: sequence %d diverges at token %zu (batched %d, alone %d)\n", tag, what, seq, i, got[i], want[i]);
      ++g_failures;
      return;
    }
  }
  CHECK(got.size() == want.size(), "%s %s: sequence %d has %zu tokens, want %zu", tag, what, seq, got.size(), want.size());
}

// `pipelined`: a function returning how many prefills so far ran as the hybrid's pipeline (nullptr for the emulate rig).
void RunRig(TextModel& m, const char* tag, const std::function<int64_t()>& pipelined) {
  constexpr int kSteps = 20;
  CHECK(m.BatchSlots() == 3 && m.BatchSlotCtx() == 512, "[%s] batch geometry %d x %lld", tag, m.BatchSlots(), static_cast<long long>(m.BatchSlotCtx()));
  const int lens[3] = {40, 70, 130};
  std::vector<SeqSpec> specs(3);
  for (int i = 0; i < 3; ++i) {
    specs[static_cast<size_t>(i)].prompt = MakePrompt(lens[i], i + 1);
    specs[static_cast<size_t>(i)].params.temperature = 0.0f;
  }
  std::vector<std::vector<int32_t>> want(3);
  for (int i = 0; i < 3; ++i) want[static_cast<size_t>(i)] = Alone(m, specs[static_cast<size_t>(i)], kSteps);

  // 1. lockstep
  const int64_t pipelined_before = pipelined ? pipelined() : 0;
  std::vector<Live> live;
  for (int i = 0; i < 3; ++i) live.push_back(Start(m, i, specs[static_cast<size_t>(i)]));
  if (pipelined) {
    CHECK(pipelined() - pipelined_before == 3, "[%s] %lld of the 3 prefills ran as the hybrid pipeline", tag,
          static_cast<long long>(pipelined() - pipelined_before));
  }
  CHECK(m.PositionCount() == lens[2], "[%s] the single-sequence state is at %lld after the last import, want %d", tag,
        static_cast<long long>(m.PositionCount()), lens[2]);
  for (int s = 0; s + 1 < kSteps; ++s) Step(m, live);
  for (int i = 0; i < 3; ++i) Compare(tag, "lockstep", i, live[static_cast<size_t>(i)].chain, want[static_cast<size_t>(i)]);
  CHECK(m.PositionCount() == lens[2], "[%s] batch steps moved the single-sequence state to %lld", tag, static_cast<long long>(m.PositionCount()));
  for (int i = 0; i < 3; ++i) m.BatchRelease(i);

  // 2. greedy + two sampled rows
  std::vector<SeqSpec> mix = specs;
  mix[1].params.temperature = 0.8f;
  mix[1].params.top_k = 40;
  mix[1].seed = 11;
  mix[2].params.temperature = 1.0f;
  mix[2].params.top_p = 0.9f;
  mix[2].seed = 12;
  std::vector<std::vector<int32_t>> mwant(3);
  for (int i = 0; i < 3; ++i) mwant[static_cast<size_t>(i)] = Alone(m, mix[static_cast<size_t>(i)], kSteps);
  std::vector<Live> mixed;
  for (int i = 0; i < 3; ++i) mixed.push_back(Start(m, i, mix[static_cast<size_t>(i)]));
  for (int s = 0; s + 1 < kSteps; ++s) Step(m, mixed);
  for (int i = 0; i < 3; ++i) Compare(tag, "sampled mix", i, mixed[static_cast<size_t>(i)].chain, mwant[static_cast<size_t>(i)]);
  std::mt19937_64 solo1 = r4dx::kernels::MakeRng(11), solo2 = r4dx::kernels::MakeRng(12);
  for (int d = 0; d < kSteps; ++d) {
    (void)r4dx::kernels::DrawUniform01(solo1);
    (void)r4dx::kernels::DrawUniform01(solo2);
  }
  CHECK(mixed[1].rng == solo1 && mixed[2].rng == solo2, "[%s] a sampled row's generator did not end where one draw per token leaves it", tag);

  // 3. the facade is healthy and the single-sequence state still decodes
  const int32_t next = m.DecodeStepGreedy(want[2][0]);
  CHECK(next == want[2][1], "[%s] after batched decode the single-sequence state gave %d, want %d", tag, next, want[2][1]);
  for (int i = 0; i < 3; ++i) m.BatchRelease(i);
  if (g_failures == 0) std::fprintf(stderr, "[PASS] %s: lockstep and sampled-mix chains == alone chains%s\n", tag, pipelined ? " (prompts prefilled by the hybrid pipeline)" : "");
}

ModelOptions Options() {
  ModelOptions o;
  o.container_path = kContainerPath;
  o.layout = Layout::kBf16;
  o.max_ctx = 512;
  o.layer_limit = 4;
  o.vision = ModelOptions::VisionMode::kOff;
  o.batch_slots = 3;
  o.batch_ctx = 512;
  return o;
}

}  // namespace

static int RunTest(int argc, char** argv) {
  std::string rig = "emulate";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--rig") rig = argv[i + 1];
  }
  if (!FileExists(kContainerPath)) return SkipMissing(kContainerPath);
  if (rig == "emulate") {
    TpOptions tp;
    tp.world = 2;
    tp.mode = TpOptions::Mode::kEmulate;
    std::unique_ptr<TextModel> m = r4dx::model::LoadTextModel(Options(), tp);
    RunRig(*m, "tp emulate", nullptr);
  } else if (rig == "hybrid") {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) {
      std::fprintf(stderr, "[SKIP] the hybrid rig needs two visible devices, found %d\n", devices);
      return kSkipReturnCode;
    }
    ModelOptions o = Options();
    o.pp = 2;
    TpOptions tp;
    tp.world = 2;
    tp.mode = TpOptions::Mode::kReal;
    PpOptions pp;
    pp.split = 2;
    pp.min_rows = 1;  // pipeline every prefill
    pp.min_rows_given = true;
    pp.hybrid = 1;
    pp.hybrid_ctx = 512;
    std::unique_ptr<TpModel> m = TpModel::Load(o, tp, pp);
    if (!m->HybridEngaged()) throw std::runtime_error("the hybrid did not engage: " + m->HybridRefusal());
    RunRig(*m, "tp2 pp2 hybrid", [&] { return m->GetHybridStats().pipelined_calls; });
  } else {
    std::fprintf(stderr, "unknown --rig '%s' (emulate|hybrid)\n", rig.c_str());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) { return r4dx_test::RunGuardedMain("test_batch_decode_tp", [&] { return RunTest(argc, argv); }); }
