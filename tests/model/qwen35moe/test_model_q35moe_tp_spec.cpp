// --tp 2 speculative decoding of the qwen3_5_moe model, EMULATE mode (two rank models on ONE device), docs/qwen35moe-tp-design.md P6.
// test_model_q35moe_spec.cpp's two parts on a TENSOR-PARALLEL pair of rank models (each rank runs the whole script; they meet in the all-reduces, the
// vocab merges of the verify rows and the drafter's top-16 merge):
//   Part 1 (no drafter): random-length verify windows (true tokens + garbage), random commits, plain steps and mid-stream prefills against the rank's own
//          plain decode run -- the TP verify rows go through the M-row grouped kernels at expert width 512 and the write-once GDN window state of 32 value heads.
//   Part 2 (DFlash2 drafter replicated on both ranks): DecodeStepDflashGreedy rounds against plain greedy decode of the same model: identical streams.
// Both ranks must also produce the same tokens (the facade would throw on a divergence). On the TRELLIS tiny container; fixtures as the spec test.
// The tolerances of the noise-limited comparisons (a prefill after speculation, a flipped verify-row argmax) are wider than the single-device test's (0.05 mean/rms,
// 4 bf16 steps of the top logit): the tensor-parallel partial sums add one more bf16 rounding per sublayer. The bit-exact checks (a plain step after speculation, a DFlash2
// stream against plain greedy) are NOT relaxed.
//   R4DX_Q35MOE_TRELLIS, R4DX_Q35MOE_DFLASH, R4DX_Q35_STEPS (default 64)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "q35moe_model_util.h"
#include "tp/tp_group.h"

using namespace q35m;
namespace tp = r4dx::model::tp;

namespace {

std::atomic<int> g_failures{0};
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

class Barrier {
 public:
  explicit Barrier(int n) : n_(n) {}
  void Wait() {
    std::unique_lock<std::mutex> lk(m_);
    const int gen = gen_;
    if (++count_ == n_) {
      count_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen != gen_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  int n_, count_ = 0, gen_ = 0;
};
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
      Check(exact ? e == 0.0 : e <= 0.05, "T" + std::to_string(T) + ": plain step after speculation at token " + std::to_string(i) + " is not bit-identical to plain decode (mean/rms " +
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
      Check(e <= 0.05, "T" + std::to_string(T) + ": prefill after speculation at token " + std::to_string(i) + " drifts (mean/rms " +
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
        Check(sorted[0] - sorted[1] <= 0.51, "T" + std::to_string(T) + ": a verify row flips its argmax although the reference margin (" +
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
    if (static_cast<size_t>(i) + 1 < ref.logits.size()) Check(e <= 0.05, "T" + std::to_string(T) + ": final plain step drifts");
  }
  std::printf("   %ld windows, %ld plain steps, %ld mid-stream prefills; verify rows %ld (worst mean/rms %.5f), argmax flips %ld; worst plain/prefill mean/rms %.5f\n",
              windows, plain_steps, prefills, rows_checked, worst_verify, flips, worst_plain);
}

// ---- Part 2 -----------------------------------------------------------------------------------------------------------------
void DflashRounds(Qwen35MoeModel* m, const Golden& g, int steps, int k, bool oracle) {
  const int T = static_cast<int>(g.ids().size());
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

// Runs `script(model, rank)` on two TP ranks (emulate, one device). `configure` fills the per-rank options.
void RunRanks(const std::function<void(Qwen35MoeModelOptions&)>& configure, const std::function<void(Qwen35MoeModel&, int)>& script) {
  auto group = tp::TpGroup::Create(tp::TpGroup::Mode::kEmulate, 2, tp::TpGroup::Geometry{}, 500);
  std::vector<std::exception_ptr> err(2);
  Barrier loaded(2);
  const std::string drafter = DflashPath();
  std::shared_ptr<const r4dx::model::DflashHostCodebooks> codebooks;
  std::vector<std::thread> threads;
  for (int rank = 0; rank < 2; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        R4DX_HIP_CHECK(hipSetDevice(TestDevice()));
        auto ep = group->CreateEndpoint(rank, nullptr);
        Qwen35MoeModelOptions o;
        o.container_path = TrellisPath();
        o.max_ctx = 4096;
        o.tp_world = 2;
        o.tp_rank = rank;
        o.tp_device = TestDevice();
        o.tp_comm = ep.get();
        configure(o);
        if (!o.dflash_container.empty()) o.dflash_codebooks = r4dx::model::DflashDraft::LoadHostCodebooks(o.dflash_container);
        auto m = Qwen35MoeModel::Load(o);
        loaded.Wait();
        script(*m, rank);
        m.reset();
        R4DX_HIP_CHECK(hipDeviceSynchronize());
      } catch (...) {
        err[rank] = std::current_exception();
      }
    });
  }
  for (auto& t : threads) t.join();
  for (int r = 0; r < 2; ++r) {
    if (err[r]) std::rethrow_exception(err[r]);
  }
}

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
  const int steps = std::getenv("R4DX_Q35_STEPS") != nullptr ? std::atoi(std::getenv("R4DX_Q35_STEPS")) : 64;
  const int prompts[] = {1, 7, 70};
  for (int T : prompts) {
    if (!Exists(GoldenPath(T))) {
      std::printf("SKIP: %s missing (prompt ids)\n", GoldenPath(T).c_str());
      return kSkip;
    }
  }

  // Part 1
  RunRanks([](Qwen35MoeModelOptions& o) { o.spec_window = 16; },
           [&](Qwen35MoeModel& m, int rank) {
             if (rank == 0) std::printf("TP=2 emulate, trellis container %s: verify window %lld, grouped layers %lld\n", path.c_str(), static_cast<long long>(m.SpecWindow()),
                                        static_cast<long long>(m.GroupedDecodeLayers()));
             for (int T : prompts) {
               const Golden g(T, GoldenPath(T));
               for (uint64_t seed : {1ull, 2ull}) ScriptedWindows(m, g, steps, seed * 7919ull + static_cast<uint64_t>(T));
             }
           });

  // Part 2
  if (!Exists(DflashPath())) {
    std::printf("\nSKIP part 2: %s missing\n", DflashPath().c_str());
  } else {
    for (int k : {15, 4}) {
      RunRanks(
          [&](Qwen35MoeModelOptions& o) {
            o.dflash_container = DflashPath();
            o.dflash_draft_k = k;
            o.dflash_target_layers = {0, 1, 2, 3, 0, 1, 2, 3};
            o.dflash_small_vocab = true;
          },
          [&](Qwen35MoeModel& m, int rank) {
            for (int T : prompts) {
              const Golden g(T, GoldenPath(T));
              if (rank == 0) {
                DflashRounds(&m, g, steps, k, /*oracle=*/false);
                DflashRounds(&m, g, steps, k, /*oracle=*/true);
              } else {
                DflashRounds(&m, g, steps, k, false);
                DflashRounds(&m, g, steps, k, true);
              }
            }
          });
    }
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures.load());
  return g_failures == 0 ? 0 : 1;
}
