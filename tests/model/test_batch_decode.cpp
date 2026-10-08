// tests/model/test_batch_decode.cpp -- Model::BatchImport / DecodeBatch (docs/batch-decode.md): several sequences decoded in ONE forward pass per
// token leave every sequence exactly where decoding it alone would. Against the real 4-layer test container (layers 0-2 GDN, layer 3 full
// attention), layouts bf16 and w4a16, a Model loaded with 3 batch slots:
//
//   1. lockstep: three prompts of different lengths, prefilled one after another into the single-sequence state, imported into slots 0..2, then
//      decoded together -- each slot's token chain equals the chain the same prompt gives when decoded alone (DecodeStepGreedy on the
//      single-sequence state, same Model);
//   2. irregular schedule: steps over subsets and orders of the slots ({2,0}, {1}, {0,1,2}, ...) -- a slot's chain does not depend on who it
//      shares a step with, or on missing a step;
//   3. sampled rows: a greedy row batched with two sampled ones (different temperature, top-k, top-p) equals DecodeStepSampled alone for the
//      same seeds, i.e. one draw per token and the same token whichever path (device summary or full-row fallback) resolves it;
//   4. the single-sequence state is untouched by an import and by batch steps (PositionCount, and its own next token);
//   5. slot reuse: a released slot re-imported with another prompt decodes that prompt, nothing of the old one;
//   6. refusals: an inactive slot, a repeated slot, no rows, a prompt that leaves no room in the slot, a full slot.
//
// Token chains are compared EXACTLY. They should be: every row-independent kernel gives a row the same bits at any row count, the attention
// core is the single-sequence launch per row (AttnBatchView), and the GDN kernels loop per sequence. The one thing that could differ is the GEMM
// tuning table choosing another kernel for M = 3 than for M = 1 (docs/batch-decode.md 6); a mismatch is therefore printed with the first
// diverging step and the top-2 logit gap of the alone run at it, so a near-tie can be told from a real bug.
//
// SKIPs (CTest SKIPPED) if the container is missing; an exception from a present one is a FAIL. Device 1.
// Written, NOT run by its author (a CPU-only session): it has never met a GPU.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "r4dx/kernels/sampler.hpp"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::BatchDecodeRow;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

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

// The gap between the best and second-best logit: how much room a rounding difference would need to flip the argmax.
double Top2Gap(const std::vector<float>& v) {
  float a = -1e30f, b = -1e30f;
  for (float x : v) {
    if (x > a) { b = a; a = x; }
    else if (x > b) { b = x; }
  }
  return static_cast<double>(a) - static_cast<double>(b);
}

struct SeqSpec {
  std::vector<int32_t> prompt;
  r4dx::kernels::SampleParams params;  // temperature <= 0: greedy
  uint64_t seed = 0;
};

// The chain `n` tokens long a sequence gives when decoded ALONE on the single-sequence state: token 0 comes from the prefill logits, token i + 1
// from feeding token i. `gap_out`: the top-2 logit gap at each step (greedy only), for the diagnostics.
std::vector<int32_t> Alone(Model& m, const SeqSpec& s, int n, std::vector<double>* gap_out = nullptr) {
  m.Reset();
  std::vector<float> logits = m.Prefill(s.prompt);
  std::mt19937_64 rng = r4dx::kernels::MakeRng(s.seed);
  const bool greedy = !(s.params.temperature > 0.0f);
  int32_t tok = greedy ? Argmax(logits)
                       : r4dx::kernels::Sample(logits.data(), static_cast<int64_t>(logits.size()), s.params, rng);
  if (gap_out != nullptr) gap_out->push_back(Top2Gap(logits));
  std::vector<int32_t> out;
  for (int i = 0; i < n; ++i) {
    out.push_back(tok);
    if (i + 1 == n) break;
    if (greedy && gap_out != nullptr) {
      const std::vector<float> row = m.DecodeStep(tok);  // the full row, for the gap (same forward as DecodeStepGreedy)
      gap_out->push_back(Top2Gap(row));
      tok = Argmax(row);
    } else {
      tok = greedy ? m.DecodeStepGreedy(tok) : m.DecodeStepSampled(tok, s.params, rng);
    }
  }
  return out;
}

// One sequence being decoded in a batch slot.
struct Live {
  int slot = 0;
  SeqSpec spec;
  std::mt19937_64 rng;
  std::vector<int32_t> chain;  // tokens so far; chain.back() is the next one to feed
  bool greedy() const { return !(spec.params.temperature > 0.0f); }
};

// Prefills `spec` into the single-sequence state (Reset first), takes its first token from the prefill logits and imports it into `slot`.
Live Start(Model& m, int slot, const SeqSpec& spec) {
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

// One DecodeBatch call over the given sequences (in this order): feed each one's last token, append the token that comes back.
void Step(Model& m, std::vector<Live>& all, const std::vector<int>& which) {
  std::vector<BatchDecodeRow> rows;
  for (int i : which) {
    Live& l = all[static_cast<size_t>(i)];
    BatchDecodeRow r;
    r.slot = l.slot;
    r.token = l.chain.back();
    r.params = l.spec.params;
    r.rng = l.greedy() ? nullptr : &l.rng;
    rows.push_back(r);
  }
  const std::vector<int32_t> next = m.DecodeBatch(rows);
  CHECK(next.size() == rows.size(), "DecodeBatch returned %zu tokens for %zu rows", next.size(), rows.size());
  for (size_t k = 0; k < which.size() && k < next.size(); ++k) all[static_cast<size_t>(which[k])].chain.push_back(next[k]);
}

void CompareChain(const char* tag, int seq, const std::vector<int32_t>& got, const std::vector<int32_t>& want,
                  const std::vector<double>* gap = nullptr) {
  const size_t n = std::min(got.size(), want.size());
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "[FAIL] %s: sequence %d diverges at token %zu (batched %d, alone %d)", tag, seq, i, got[i], want[i]);
      if (gap != nullptr && i > 0 && i - 1 < gap->size()) {
        std::fprintf(stderr, "; the alone run's top-2 logit gap at that step: %.4g", (*gap)[i - 1]);
      }
      std::fprintf(stderr, "\n");
      ++g_failures;
      return;
    }
  }
  CHECK(got.size() >= want.size(), "%s: sequence %d produced %zu tokens, want %zu", tag, seq, got.size(), want.size());
}

void RunLayout(Layout layout) {
  const std::string tag_s = std::string("layout=") + LayoutName(layout);
  const char* tag = tag_s.c_str();
  ModelOptions opts;
  opts.container_path = kContainerPath;
  opts.layout = layout;
  opts.max_ctx = 512;
  opts.layer_limit = 4;
  opts.batch_slots = 3;
  opts.batch_ctx = 512;
  Model m = Model::Load(opts);
  CHECK(m.BatchSlots() == 3 && m.BatchSlotCtx() == 512 && m.BatchBytes() > 0, "[%s] batch geometry %d x %lld, %lld bytes", tag,
        m.BatchSlots(), static_cast<long long>(m.BatchSlotCtx()), static_cast<long long>(m.BatchBytes()));

  constexpr int kSteps = 24;
  std::vector<SeqSpec> specs(3);
  const int lens[3] = {40, 70, 130};  // below, around and across the 64-row chunk grid
  for (int i = 0; i < 3; ++i) {
    specs[static_cast<size_t>(i)].prompt = MakePrompt(lens[i], i + 1);
    specs[static_cast<size_t>(i)].params.temperature = 0.0f;
  }
  std::vector<std::vector<int32_t>> want(3);
  std::vector<std::vector<double>> gaps(3);
  for (int i = 0; i < 3; ++i) want[static_cast<size_t>(i)] = Alone(m, specs[static_cast<size_t>(i)], kSteps, &gaps[static_cast<size_t>(i)]);

  // 1. lockstep
  {
    std::vector<Live> live;
    for (int i = 0; i < 3; ++i) live.push_back(Start(m, i, specs[static_cast<size_t>(i)]));
    for (int i = 0; i < 3; ++i) {
      CHECK(m.BatchSlotActive(i) && m.BatchSlotPosition(i) == lens[i], "[%s] slot %d after import: active %d pos %lld, want %d", tag, i,
            static_cast<int>(m.BatchSlotActive(i)), static_cast<long long>(m.BatchSlotPosition(i)), lens[i]);
    }
    for (int s = 0; s + 1 < kSteps; ++s) Step(m, live, {0, 1, 2});
    for (int i = 0; i < 3; ++i) CompareChain(tag, i, live[static_cast<size_t>(i)].chain, want[static_cast<size_t>(i)], &gaps[static_cast<size_t>(i)]);
    for (int i = 0; i < 3; ++i) {
      CHECK(m.BatchSlotPosition(i) == lens[i] + kSteps - 1, "[%s] slot %d position %lld after %d steps", tag, i,
            static_cast<long long>(m.BatchSlotPosition(i)), kSteps - 1);
    }
    for (int i = 0; i < 3; ++i) m.BatchRelease(i);
    CHECK(!m.BatchSlotActive(0) && !m.BatchSlotActive(1) && !m.BatchSlotActive(2), "[%s] release", tag);
  }

  // 2. an irregular schedule over subsets and orders; every slot ends with kSteps tokens
  {
    std::vector<Live> live;
    for (int i = 0; i < 3; ++i) live.push_back(Start(m, i, specs[static_cast<size_t>(i)]));
    const std::vector<std::vector<int>> schedule = {{2, 0}, {1}, {0, 1, 2}, {2}, {1, 0}, {0}, {2, 1, 0}, {1, 2}};
    for (int round = 0; round < 40; ++round) {
      std::vector<int> which;
      for (int i : schedule[static_cast<size_t>(round) % schedule.size()]) {
        if (static_cast<int>(live[static_cast<size_t>(i)].chain.size()) < kSteps) which.push_back(i);
      }
      if (!which.empty()) Step(m, live, which);
    }
    for (int i = 0; i < 3; ++i) {
      CHECK(static_cast<int>(live[static_cast<size_t>(i)].chain.size()) == kSteps, "[%s] irregular schedule: sequence %d has %zu tokens", tag, i,
            live[static_cast<size_t>(i)].chain.size());
      CompareChain(tag, i, live[static_cast<size_t>(i)].chain, want[static_cast<size_t>(i)], &gaps[static_cast<size_t>(i)]);
    }
    for (int i = 0; i < 3; ++i) m.BatchRelease(i);
  }

  // 3. a greedy row batched with two sampled ones, against DecodeStepSampled alone with the same seeds
  {
    std::vector<SeqSpec> mix = specs;
    mix[1].params.temperature = 0.8f;
    mix[1].params.top_k = 40;
    mix[1].seed = 11;
    mix[2].params.temperature = 1.0f;
    mix[2].params.top_p = 0.9f;
    mix[2].seed = 12;
    std::vector<std::vector<int32_t>> mwant(3);
    for (int i = 0; i < 3; ++i) mwant[static_cast<size_t>(i)] = Alone(m, mix[static_cast<size_t>(i)], kSteps);
    std::vector<Live> live;
    for (int i = 0; i < 3; ++i) live.push_back(Start(m, i, mix[static_cast<size_t>(i)]));
    for (int s = 0; s + 1 < kSteps; ++s) Step(m, live, {0, 1, 2});
    for (int i = 0; i < 3; ++i) CompareChain(tag, i, live[static_cast<size_t>(i)].chain, mwant[static_cast<size_t>(i)]);
    // one draw per sampled token: the batched generator and the alone one agree on the next draw
    std::mt19937_64 solo1 = r4dx::kernels::MakeRng(11);
    std::mt19937_64 solo2 = r4dx::kernels::MakeRng(12);
    // (token 0 took a draw from the prefill logits, then one per decode step: kSteps draws in all)
    for (int d = 0; d < kSteps; ++d) {
      (void)r4dx::kernels::DrawUniform01(solo1);
      (void)r4dx::kernels::DrawUniform01(solo2);
    }
    CHECK(live[1].rng == solo1 && live[2].rng == solo2, "[%s] a sampled row consumed a different number of draws than one per token", tag);
    for (int i = 0; i < 3; ++i) m.BatchRelease(i);
  }

  // 4. the single-sequence state is untouched by an import and by batch steps
  {
    Live l = Start(m, 1, specs[0]);
    CHECK(m.PositionCount() == lens[0], "[%s] an import moved the single-sequence state: position %lld, want %d", tag,
          static_cast<long long>(m.PositionCount()), lens[0]);
    std::vector<Live> v;
    v.push_back(l);
    for (int s = 0; s < 5; ++s) Step(m, v, {0});
    CHECK(m.PositionCount() == lens[0], "[%s] batch steps moved the single-sequence state: position %lld, want %d", tag,
          static_cast<long long>(m.PositionCount()), lens[0]);
    // ... and it still decodes its own next token: the chain a lone run gives (want[0][1]) follows from feeding want[0][0]
    const int32_t next = m.DecodeStepGreedy(want[0][0]);
    CHECK(next == want[0][1], "[%s] the single-sequence state decoded %d after an import + batch steps, want %d", tag, next, want[0][1]);
    m.BatchRelease(1);
  }

  // 5. slot reuse: the slot that held a long sequence now holds another prompt
  {
    std::vector<Live> live;
    live.push_back(Start(m, 2, specs[2]));            // 130 tokens in slot 2
    for (int s = 0; s < 6; ++s) Step(m, live, {0});
    m.BatchRelease(2);
    std::vector<Live> again;
    again.push_back(Start(m, 2, specs[0]));           // 40 tokens in the same slot
    for (int s = 0; s + 1 < kSteps; ++s) Step(m, again, {0});
    CompareChain(tag, 0, again[0].chain, want[0], &gaps[0]);
    m.BatchRelease(2);
  }

  // 6. refusals
  {
    const auto throws = [&](const std::vector<BatchDecodeRow>& rows) {
      try {
        (void)m.DecodeBatch(rows);
      } catch (const std::exception&) {
        return true;
      }
      return false;
    };
    BatchDecodeRow r;
    r.slot = 0;
    r.token = 5;
    r.params.temperature = 0.0f;
    CHECK(throws({r}), "[%s] a step on an inactive slot was accepted", tag);
    CHECK(throws({}), "[%s] a step with no rows was accepted", tag);
    std::vector<Live> live;
    live.push_back(Start(m, 0, specs[0]));
    CHECK(throws({r, r}), "[%s] a step naming one slot twice was accepted", tag);
    BatchDecodeRow bad = r;
    bad.slot = 3;
    CHECK(throws({bad}), "[%s] a step on a slot beyond the last was accepted", tag);
    bool sampled_without_rng = false;
    BatchDecodeRow s = r;
    s.params.temperature = 0.7f;
    try {
      (void)m.DecodeBatch({s});
    } catch (const std::invalid_argument&) {
      sampled_without_rng = true;
    }
    CHECK(sampled_without_rng, "[%s] a sampled row without its rng was accepted", tag);
    CHECK(m.BatchSlotPosition(0) == lens[0], "[%s] a refused step moved the slot", tag);
    m.BatchRelease(0);

    // a prompt that leaves no room in the slot cannot be imported; one that leaves exactly one position decodes once, then the slot is full
    m.Reset();
    (void)m.Prefill(MakePrompt(512, 4));
    bool too_long = false;
    try {
      m.BatchImport(0);
    } catch (const std::exception&) {
      too_long = true;
    }
    CHECK(too_long && !m.BatchSlotActive(0), "[%s] a prompt that fills the whole slot was imported", tag);
    SeqSpec near_full;
    near_full.prompt = MakePrompt(511, 5);
    near_full.params.temperature = 0.0f;
    std::vector<Live> nf;
    nf.push_back(Start(m, 0, near_full));
    Step(m, nf, {0});  // position 511 -> 512: the slot is now full
    bool full_refused = false;
    BatchDecodeRow f;
    f.slot = 0;
    f.token = nf[0].chain.back();
    f.params.temperature = 0.0f;
    try {
      (void)m.DecodeBatch({f});
    } catch (const std::exception&) {
      full_refused = true;
    }
    CHECK(full_refused, "[%s] a step on a full slot was accepted", tag);
    m.BatchRelease(0);
  }

  if (g_failures == 0) std::fprintf(stderr, "[PASS] %s: batched chains == alone chains (lockstep, irregular, sampled mix), state untouched, slot reuse, refusals\n", tag);
}

}  // namespace

static int RunTest() {
  if (!FileExists(kContainerPath)) return SkipMissing(kContainerPath);
  RunLayout(Layout::kBf16);
  RunLayout(Layout::kW4a16);
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_batch_decode", [] { return RunTest(); }); }
