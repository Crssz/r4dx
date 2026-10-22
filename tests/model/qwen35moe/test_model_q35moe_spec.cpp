// Speculative verify + rollback of the qwen3_5_moe model (docs/qwen35moe-plan.md section 6): the write-once GDN window state, the
// uncommitted KV rows of a verify window, the speculation-to-plain / speculation-to-prefill transitions, and the DFlash2 driver.
// On the TRELLIS tiny container (hidden 3072, 4 layers: 3 GDN + 1 attention, 16 experts).
//
// Part 1 (no drafter; scripted windows). A reference run teacher-forces a token stream S through Prefill + plain DecodeStep and keeps
// the logits behind every position. The speculative run feeds the SAME stream through random-length verify windows whose leading
// rows are the true tokens and whose trailing rows are garbage, commits a random number of leading rows (so the garbage rows and
// some true ones are "rejected"), and interleaves plain decode steps and a mid-stream Prefill. A plain step (or the prefill's last
// row) must reproduce the reference logits for its position -- that only holds when the GDN state B, the conv history, the pending
// log replay and the KV rows are right after every rollback -- and the verify rows' argmax must agree with the reference tokens
// (a handful of near-tie flips between the M = rows and M = 1 GEMMs are tolerated and counted).
//
// Part 2 (needs the DFlash2 drafter container): DecodeStepDflashGreedy rounds on the tiny target (the drafter reads layer inputs
// 0..3 here -- the tiny model has 4 layers -- so its drafts are poor, which exercises every partial-accept length) against plain greedy
// decode: the generated streams must be identical.
//
//   R4DX_Q35MOE_TRELLIS   the trellis container (default E:\models\r4dx\q35moe-tiny\tiny-mix.r4dx)
//   R4DX_Q35MOE_DFLASH    the drafter container (default E:\models\r4dx\ornith-1.5-122b-dflash2-bf16.r4dx)
//   R4DX_Q35_STEPS        tokens per run (default 96)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "q35moe_model_util.h"

using namespace q35m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string TrellisPath() {
  const char* e = std::getenv("R4DX_Q35MOE_TRELLIS");
  return e != nullptr && e[0] != 0 ? e : std::string("E:\\models\\r4dx\\q35moe-tiny\\tiny-mix.r4dx");
}
std::string DflashPath() {
  const char* e = std::getenv("R4DX_Q35MOE_DFLASH");
  return e != nullptr && e[0] != 0 ? e : std::string("E:\\models\\r4dx\\ornith-1.5-122b-dflash2-bf16.r4dx");
}

struct Ref {
  std::vector<int32_t> tokens;                // tokens[i]: the argmax after position (prompt_len - 1 + i)
  std::vector<std::vector<float>> logits;     // the logits row behind tokens[i]
};

// Prefill `ids`, then `steps` plain greedy decode steps (free run).
Ref FreeRef(Qwen35MoeModel& m, const std::vector<int32_t>& ids, int steps) {
  m.Reset();
  Ref r;
  std::vector<float> lg = m.Prefill(ids);
  int32_t tok = static_cast<int32_t>(ArgMax(lg.data(), static_cast<int64_t>(lg.size())));
  r.tokens.push_back(tok);
  r.logits.push_back(lg);
  for (int i = 0; i < steps; ++i) {
    lg = m.DecodeStep(tok);
    tok = static_cast<int32_t>(ArgMax(lg.data(), static_cast<int64_t>(lg.size())));
    r.tokens.push_back(tok);
    r.logits.push_back(lg);
  }
  return r;
}

double MeanNorm(const std::vector<float>& got, const std::vector<float>& ref) {
  return CompareF32(got.data(), ref.data(), static_cast<int64_t>(ref.size())).mean_norm;
}

// ---- Part 1 -----------------------------------------------------------------------------------------------------------------
void ScriptedWindows(Qwen35MoeModel& m, const Golden& g, int steps, uint64_t seed) {
  const int T = static_cast<int>(g.ids().size());
  const Ref ref = FreeRef(m, g.ids(), steps);  // the stream S = ref.tokens[0..steps-1] follows the prompt
  std::printf("-- prompt T=%d, %d tokens, seed %llu\n", T, steps, static_cast<unsigned long long>(seed));

  m.Reset();
  std::mt19937_64 rng(seed);
  const int64_t vocab = m.Config().vocab_size;
  std::vector<float> lg = m.Prefill(g.ids());  // the logits behind ref.tokens[0]
  Check(MeanNorm(lg, ref.logits[0]) <= 1e-6, "T" + std::to_string(T) + ": prefill reproduces itself");
  int i = 0;  // ref.tokens[i] is the next token to feed (position T + i)
  long rows_checked = 0, flips = 0, plain_steps = 0, windows = 0, prefills = 0;
  double worst_plain = 0, worst_verify = 0;
  bool exact = true;  // no prefill chunk has run since the start: the state is the pure decode run's, so every row must be bit-identical
  while (i + 1 < steps) {
    const int kind = static_cast<int>(rng() % 10);
    if (kind == 0) {  // a plain step (speculation -> plain)
      const std::vector<float> got = m.DecodeStep(ref.tokens[static_cast<size_t>(i)]);
      const double e = MeanNorm(got, ref.logits[static_cast<size_t>(i) + 1]);
      worst_plain = std::max(worst_plain, e);
      // BIT-exact: the state a verify window + commit leaves (GDN B + replay, conv history, KV) is the state plain decode steps leave,
      // and the plain step is the same kernels as the reference run's.
      Check(exact ? e == 0.0 : e <= 0.03, "T" + std::to_string(T) + ": plain step after speculation at token " + std::to_string(i) + " is not bit-identical to plain decode (mean/rms " +
                           std::to_string(e) + ")");
      ++plain_steps;
      ++i;
      continue;
    }
    if (kind == 1 && i + 12 < steps) {  // a mid-stream prefill of 2..11 true tokens (speculation -> prefill)
      const int n = 2 + static_cast<int>(rng() % 10);
      std::vector<int32_t> chunk(ref.tokens.begin() + i, ref.tokens.begin() + i + n);
      const std::vector<float> got = m.Prefill(chunk);
      const double e = MeanNorm(got, ref.logits[static_cast<size_t>(i + n)]);
      worst_plain = std::max(worst_plain, e);
      Check(e <= 0.03, "T" + std::to_string(T) + ": prefill after speculation at token " + std::to_string(i) + " drifts (mean/rms " +
                           std::to_string(e) + ")");
      ++prefills;
      exact = false;  // a prefill chunk's GEMM / chunked-scan numerics differ from decode's: the state is only noise-level close from here
      i += n;
      continue;
    }
    // a verify window: `w` rows, the first `good` are the true tokens ref.tokens[i ...], the rest garbage; commit `commit` leading rows
    const int w = 1 + static_cast<int>(rng() % static_cast<uint64_t>(m.SpecWindow()));
    const int good = std::min<int>(w, steps - i);
    std::vector<int32_t> cands(static_cast<size_t>(w));
    for (int r = 0; r < w; ++r) {
      cands[static_cast<size_t>(r)] = r < good ? ref.tokens[static_cast<size_t>(i + r)] : static_cast<int32_t>(rng() % static_cast<uint64_t>(vocab));
    }
    const std::vector<int32_t> am = m.VerifyWindow(cands);
    const std::vector<float> vl = m.LastVerifyLogits();
    const int checkable = std::min(good, steps - i - 1);  // am[r] should be ref.tokens[i + r + 1]
    for (int r = 0; r < checkable; ++r) {
      ++rows_checked;
      const std::vector<float>& want = ref.logits[static_cast<size_t>(i + r + 1)];
      const std::vector<float> row(vl.begin() + static_cast<ptrdiff_t>(r) * vocab, vl.begin() + static_cast<ptrdiff_t>(r + 1) * vocab);
      const double e = MeanNorm(row, want);
      worst_verify = std::max(worst_verify, e);
      // A verify row is the plain decode step of its position, bit for bit (the M-row MoE runs the M = 1 kernel's split groups, the attention
      // runs the decode kernel one row per launch, and the batched linears give a row the same bits at any M).
      Check(exact ? e == 0.0 : e <= 0.04, "T" + std::to_string(T) + ": verify row " + std::to_string(r) + " of the window at token " + std::to_string(i) +
                           " is not bit-identical to the plain decode logits (mean/rms " + std::to_string(e) + ")");
      if (am[static_cast<size_t>(r)] != ref.tokens[static_cast<size_t>(i + r + 1)]) {
        ++flips;
        std::vector<float> sorted = want;
        std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
        Check(sorted[0] - sorted[1] <= 0.126, "T" + std::to_string(T) + ": a verify row flips its argmax although the reference margin (" +
                                                 std::to_string(sorted[0] - sorted[1]) + ") exceeds two bf16 steps of a top logit");
      }
    }
    const int commit = 1 + static_cast<int>(rng() % static_cast<uint64_t>(good));
    m.CommitVerifiedWindow(commit);
    i += commit;
    ++windows;
  }
  // a final plain step must still agree
  if (i < steps) {
    const std::vector<float> got = m.DecodeStep(ref.tokens[static_cast<size_t>(i)]);
    const double e = MeanNorm(got, ref.logits[static_cast<size_t>(i) + 1 < ref.logits.size() ? static_cast<size_t>(i) + 1 : ref.logits.size() - 1]);
    if (static_cast<size_t>(i) + 1 < ref.logits.size()) Check(e <= 0.03, "T" + std::to_string(T) + ": final plain step drifts");
  }
  std::printf("   %ld windows, %ld plain steps, %ld mid-stream prefills; verify rows %ld (worst mean/rms %.5f), argmax flips %ld; worst plain/prefill mean/rms %.5f\n",
              windows, plain_steps, prefills, rows_checked, worst_verify, flips, worst_plain);
}

// ---- Part 2 -----------------------------------------------------------------------------------------------------------------
void DflashRounds(const std::string& container, const std::string& drafter, const Golden& g, int steps, int k, bool oracle) {
  const int T = static_cast<int>(g.ids().size());
  Qwen35MoeModelOptions o;
  o.container_path = container;
  o.max_ctx = 4096;
  o.devices = {TestDevice(), TestDevice()};
  o.dflash_container = drafter;
  o.dflash_draft_k = k;
  o.dflash_target_layers = {0, 1, 2, 3, 0, 1, 2, 3};
  o.dflash_small_vocab = true;
  auto m = Qwen35MoeModel::Load(o);
  const Ref ref = FreeRef(*m, g.ids(), steps);
  std::printf("-- dflash k=%d, prompt T=%d, %d tokens%s\n", k, T, steps, oracle ? ", scripted drafts (true continuation, random cut)" : "");

  m->Reset();
  std::vector<float> lg = m->Prefill(g.ids());
  std::vector<int32_t> out;
  out.push_back(static_cast<int32_t>(ArgMax(lg.data(), static_cast<int64_t>(lg.size()))));
  std::mt19937_64 rng(1234 + static_cast<uint64_t>(T) * 31 + static_cast<uint64_t>(k));
  const int32_t vocab = static_cast<int32_t>(m->Config().vocab_size);
  if (oracle) {
    // The true continuation of the anchor (= out.back(), reference index out.size() - 1), cut at a random length and then wrong.
    m->SetDraftOverride([&](int32_t, int64_t kk) {
      std::vector<int32_t> d;
      const size_t base = out.size();  // ref.tokens[base + j] follows the anchor's successor j times
      const int64_t cut = static_cast<int64_t>(rng() % static_cast<uint64_t>(kk + 1));
      for (int64_t j = 0; j < kk && base + static_cast<size_t>(j) < ref.tokens.size(); ++j) {
        const int32_t t = ref.tokens[base + static_cast<size_t>(j)];
        d.push_back(j < cut ? t : (t + 1) % vocab);
      }
      return d;
    });
  }
  long rounds = 0, drafted = 0, accepted = 0;
  std::vector<long> hist(static_cast<size_t>(k) + 2, 0);
  bool tolerated = false;
  size_t agree = 1;  // out[0] is the prefill's argmax, identical by construction
  while (static_cast<int>(out.size()) < steps && !tolerated) {
    int64_t walk = 0;
    const std::vector<int32_t> round = m->DecodeStepDflashGreedy(out.back(), k, 0.0f, 0, &walk);
    ++rounds;
    drafted += walk;
    accepted += static_cast<long>(round.size()) - 1;
    ++hist[round.size()];
    for (int32_t t : round) {
      const size_t j = out.size();
      if (j < ref.tokens.size() && t != ref.tokens[j]) {
        // A different token from plain greedy: only a near tie of the reference logits (a few bf16 steps of a top logit) may flip
        // between the M = k + 1 verify GEMMs and the M = 1 decode ones.
        std::vector<float> sorted = ref.logits[j];
        std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
        const double margin = sorted[0] - sorted[1];
        std::printf("   token %zu differs from plain greedy (reference margin top1-top2 = %.4g)\n", j, margin);
        Check(margin <= 0.126, "dflash T" + std::to_string(T) + " k" + std::to_string(k) + ": token " + std::to_string(j) +
                                   " differs from plain greedy although the reference margin (" + std::to_string(margin) +
                                   ") exceeds two bf16 steps of a top logit");
        tolerated = true;
        break;
      }
      out.push_back(t);
      ++agree;
    }
  }
  std::printf("   %ld rounds, drafted %ld, accepted %ld; %zu of %d tokens identical to plain greedy\n", rounds, drafted, accepted,
              std::min<size_t>(agree, static_cast<size_t>(steps)), steps);
  std::printf("   tokens per round:");
  for (size_t n = 1; n < hist.size(); ++n) std::printf(" %zu:%ld", n, hist[n]);
  std::printf("\n");
  if (oracle) Check(accepted > rounds, "scripted drafts: fewer than one accepted draft per round on average");
}

}  // namespace

int RunTest();
int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}

int RunTest() {
  Unbuffer();
  const std::string path = TrellisPath();
  if (!Exists(path)) {
    std::printf("SKIP: %s missing\n", path.c_str());
    return kSkip;
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  const int steps = std::getenv("R4DX_Q35_STEPS") != nullptr ? std::atoi(std::getenv("R4DX_Q35_STEPS")) : 96;
  const int prompts[] = {1, 7, 70};
  for (int T : prompts) {
    if (!Exists(GoldenPath(T))) {
      std::printf("SKIP: %s missing (prompt ids)\n", GoldenPath(T).c_str());
      return kSkip;
    }
  }

  {
    Qwen35MoeModelOptions o;
    o.container_path = path;
    o.max_ctx = 4096;
    o.devices = {TestDevice(), TestDevice()};
    o.spec_window = 16;
    auto m = Qwen35MoeModel::Load(o);
    std::printf("trellis container %s on device %d: %lld layer(s), verify window %lld\n", path.c_str(), TestDevice(),
                static_cast<long long>(m->NumLoadedLayers()), static_cast<long long>(m->SpecWindow()));
    for (int T : prompts) {
      const Golden g(T, GoldenPath(T));
      for (uint64_t seed : {1ull, 2ull}) ScriptedWindows(*m, g, steps, seed * 7919ull + static_cast<uint64_t>(T));
    }
  }

  if (!Exists(DflashPath())) {
    std::printf("\nSKIP part 2: %s missing\n", DflashPath().c_str());
  } else {
    for (int T : prompts) {
      const Golden g(T, GoldenPath(T));
      for (int k : {15, 4}) {
        DflashRounds(path, DflashPath(), g, steps, k, /*oracle=*/false);
        DflashRounds(path, DflashPath(), g, steps, k, /*oracle=*/true);
      }
    }
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}


