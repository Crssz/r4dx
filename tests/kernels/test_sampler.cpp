// tests/kernels/test_sampler.cpp -- CPU-only (no GPU): r4dx::kernels::Argmax/Sample and
// EmbeddingGatherHost.
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "r4dx/kernels/embedding.hpp"
#include "r4dx/kernels/sampler.hpp"

using namespace r4dx::kernels;

namespace {
int g_failures = 0;
void Check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    g_failures++;
  } else {
    std::printf("PASS: %s\n", what);
  }
}
}  // namespace

int main() {
  // Argmax.
  std::vector<float> logits = {0.1f, 5.0f, -2.0f, 4.9f, 3.0f};
  Check(Argmax(logits.data(), logits.size()) == 1, "Argmax picks the true maximum");

  // Greedy sampling (temperature <= 0) must equal Argmax.
  auto rng = MakeRng(0);
  SampleParams greedy;
  greedy.temperature = 0.0f;
  Check(Sample(logits.data(), logits.size(), greedy, rng) == 1, "temperature<=0 is greedy argmax");

  // top_k=1 must also always return the argmax regardless of temperature/seed.
  SampleParams top1;
  top1.temperature = 1.0f;
  top1.top_k = 1;
  bool all_argmax = true;
  for (uint64_t seed = 0; seed < 20; ++seed) {
    auto r = MakeRng(seed);
    all_argmax &= (Sample(logits.data(), logits.size(), top1, r) == 1);
  }
  Check(all_argmax, "top_k=1 always returns the argmax");

  // A one-hot distribution (one logit vastly larger) must sample that token regardless of seed.
  std::vector<float> one_hot = {-100.0f, -100.0f, 100.0f, -100.0f};
  SampleParams free_params;
  bool all_peak = true;
  for (uint64_t seed = 0; seed < 20; ++seed) {
    auto r = MakeRng(seed);
    all_peak &= (Sample(one_hot.data(), one_hot.size(), free_params, r) == 2);
  }
  Check(all_peak, "an overwhelmingly peaked distribution always samples its peak");

  // Determinism: same seed -> same draw, over a flatter distribution where randomness matters.
  std::vector<float> flat = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  auto r1 = MakeRng(1234);
  auto r2 = MakeRng(1234);
  int32_t a = Sample(flat.data(), flat.size(), free_params, r1);
  int32_t b = Sample(flat.data(), flat.size(), free_params, r2);
  Check(a == b, "same seed -> same draw (determinism)");

  // min_p / top_p must not crash and must return an in-range token.
  SampleParams filtered;
  filtered.top_p = 0.9f;
  filtered.min_p = 0.05f;
  filtered.top_k = 3;
  auto r3 = MakeRng(99);
  int32_t tok = Sample(logits.data(), logits.size(), filtered, r3);
  Check(tok >= 0 && tok < static_cast<int32_t>(logits.size()), "combined filters stay in range");

  // top_k combined with min_p against KNOWN expected survivors: 5 logits with temperature=1 so
  // scaled==logits. Unnormalized prob relative to the global max (logit 5.0, at index 1) is
  // exp(logit - 5.0): idx1=1.0, idx3=exp(-0.1)=0.9048, idx4=exp(-2.0)=0.1353, idx0=exp(-4.9)=
  // 0.0074, idx2=exp(-7.0)=9.1e-4. top_k=3 keeps {1,3,4} (the three largest logits); min_p=0.5
  // then additionally drops idx4 (0.1353 < 0.5), leaving only {1,3} as possible draws -- so
  // every draw over many seeds must land in {1,3} and idx4's mass (~12% pre-min_p) must never be
  // picked once min_p=0.5 is applied.
  {
    SampleParams topk_minp;
    topk_minp.temperature = 1.0f;
    topk_minp.top_k = 3;
    topk_minp.min_p = 0.5f;
    bool all_in_survivors = true;
    bool saw_idx3 = false;
    for (uint64_t seed = 0; seed < 200; ++seed) {
      auto r = MakeRng(seed);
      int32_t t = Sample(logits.data(), logits.size(), topk_minp, r);
      all_in_survivors &= (t == 1 || t == 3);
      saw_idx3 |= (t == 3);
    }
    Check(all_in_survivors, "top_k=3 + min_p=0.5 restricts draws to the two known survivors");
    Check(saw_idx3, "top_k=3 + min_p=0.5 still draws the non-argmax survivor sometimes");
  }

  // min_p alone (no top_k, no top_p) must apply against the UNTRUNCATED distribution's max, per
  // vLLM/HF semantics -- not a rank-truncated max, which is what the previous implementation did
  // when top_k was also active. This case has no top_k, so it specifically exercises the no-sort
  // min_p-only fast path.
  {
    SampleParams minp_only;
    minp_only.temperature = 1.0f;
    minp_only.min_p = 0.5f;
    bool all_in_survivors = true;
    for (uint64_t seed = 0; seed < 200; ++seed) {
      auto r = MakeRng(seed);
      int32_t t = Sample(logits.data(), logits.size(), minp_only, r);
      all_in_survivors &= (t == 1 || t == 3);  // same survivor set as above: min_p=0.5 threshold
    }
    Check(all_in_survivors, "min_p=0.5 alone matches the untruncated-max threshold semantics");
  }

  // Pure temperature (no top_k/top_p/min_p) must exercise the no-sort fast path and still only
  // ever draw the overwhelmingly-peaked distribution's peak.
  {
    SampleParams pure_temp;
    pure_temp.temperature = 1.0f;
    bool all_peak2 = true;
    for (uint64_t seed = 0; seed < 20; ++seed) {
      auto r = MakeRng(seed);
      all_peak2 &= (Sample(one_hot.data(), one_hot.size(), pure_temp, r) == 2);
    }
    Check(all_peak2, "pure-temperature fast path samples the peak of an overwhelming distribution");
  }

  // Argmax on an empty vocab must not read logits[0] out of bounds.
  Check(Argmax(logits.data(), 0) == -1, "Argmax(vocab=0) returns -1, no OOB read");

  // EmbeddingGatherHost.
  const int64_t vocab = 100;
  const int64_t hidden = 8;
  std::vector<uint16_t> table(static_cast<size_t>(vocab) * hidden);
  std::iota(table.begin(), table.end(), static_cast<uint16_t>(0));
  std::vector<int32_t> ids = {5, 0, 42, 99};
  std::vector<uint16_t> out(ids.size() * hidden);
  EmbeddingGatherHost(table.data(), vocab, hidden, ids, out.data());
  bool gather_ok = true;
  for (size_t i = 0; i < ids.size(); ++i) {
    for (int64_t d = 0; d < hidden; ++d) {
      gather_ok &= (out[i * hidden + d] == table[ids[i] * hidden + d]);
    }
  }
  Check(gather_ok, "EmbeddingGatherHost gathers the right rows");

  bool threw_oob = false;
  try {
    std::vector<int32_t> bad_ids = {vocab};  // == vocab, out of [0, vocab)
    std::vector<uint16_t> bad_out(hidden);
    EmbeddingGatherHost(table.data(), vocab, hidden, bad_ids, bad_out.data());
  } catch (const std::out_of_range&) {
    threw_oob = true;
  }
  Check(threw_oob, "EmbeddingGatherHost throws on an out-of-range token id");

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
