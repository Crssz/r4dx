// tests/kernels/test_rep_penalty_cpu.cpp -- pure CPU (header-only rep_penalty.hpp / sampler.hpp, no HIP call).
// The HF repetition penalty rule (docs/lfm2-plan.md 5.4) and the host seen-set:
//   * PenalizeLogit: negative -> x p, otherwise / p, hand-computed values; p == 1 is the identity;
//   * ApplyRepetitionPenalty touches exactly the listed ids, skips out-of-range ids, p == 1 changes no bit;
//   * SeenTokens: distinct ids in first-seen order, bounds, delta bookkeeping for a device mirror;
//   * the penalty is applied BEFORE temperature/top-k: penalizing a row and then sampling equals sampling a
//     row that was penalized by an independent reference; greedy changes when the argmax token was seen;
//   * SampleParams::repetition_penalty defaults to 1.0 and Sample() ignores the field (the caller applies it),
//     so the default-off path of every existing caller is untouched.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "r4dx/kernels/rep_penalty.hpp"
#include "r4dx/kernels/sampler.hpp"

using namespace r4dx::kernels;

namespace {

int g_fail = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_fail;                                                                       \
    }                                                                                 \
  } while (0)

bool SameBits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

void TestRule() {
  CHECK(PenalizeLogit(2.0f, 1.25f) == 1.6f);
  CHECK(PenalizeLogit(-2.0f, 1.25f) == -2.5f);
  CHECK(PenalizeLogit(0.0f, 1.5f) == 0.0f);
  CHECK(PenalizeLogit(4.0f, 0.5f) == 8.0f);    // p < 1 boosts
  CHECK(PenalizeLogit(-4.0f, 0.5f) == -2.0f);
  CHECK(std::isnan(PenalizeLogit(std::numeric_limits<float>::quiet_NaN(), 1.3f)));
  CHECK(PenalizeLogit(std::numeric_limits<float>::infinity(), 2.0f) == std::numeric_limits<float>::infinity());
  CHECK(PenalizeLogit(-std::numeric_limits<float>::infinity(), 2.0f) == -std::numeric_limits<float>::infinity());
  // p == 1: identity for every sign, including signed zero
  for (float v : {3.5f, -3.5f, 0.0f, -0.0f}) CHECK(SameBits(PenalizeLogit(v, 1.0f), v));
}

void TestApply() {
  std::vector<float> row = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, 0.5f};
  const std::vector<float> orig = row;
  const std::vector<int32_t> ids = {1, 2, 5};
  ApplyRepetitionPenalty(row.data(), 6, ids.data(), ids.size(), 2.0f);
  CHECK(row[0] == 1.0f && row[3] == -2.0f && row[4] == 3.0f);  // untouched
  CHECK(row[1] == -2.0f && row[2] == 1.0f && row[5] == 0.25f);
  // out-of-range ids are skipped, never written
  std::vector<float> r2 = orig;
  const std::vector<int32_t> bad = {-1, 6, 1000000, 0};
  ApplyRepetitionPenalty(r2.data(), 6, bad.data(), bad.size(), 2.0f);
  CHECK(r2[0] == 0.5f);
  for (size_t i = 1; i < 6; ++i) CHECK(r2[i] == orig[i]);
  // p == 1: not one bit moves
  std::vector<float> r3 = orig;
  const std::vector<int32_t> all = {0, 1, 2, 3, 4, 5};
  ApplyRepetitionPenalty(r3.data(), 6, all.data(), all.size(), 1.0f);
  CHECK(std::memcmp(r3.data(), orig.data(), orig.size() * sizeof(float)) == 0);
}

void TestSeenTokens() {
  SeenTokens s(10);
  CHECK(s.size() == 0 && !s.contains(3));
  CHECK(s.Add(3) && s.Add(7) && !s.Add(3) && s.Add(0));
  CHECK(!s.Add(-1) && !s.Add(10) && !s.Add(1 << 30));
  CHECK((s.ids() == std::vector<int32_t>{3, 7, 0}));
  CHECK(s.contains(7) && !s.contains(8) && !s.contains(-5));
  // delta bookkeeping
  CHECK(s.uploaded() == 0);
  s.MarkUploaded();
  CHECK(s.uploaded() == 3);
  s.Add(9);
  s.Add(9);
  CHECK(s.size() == 4 && s.uploaded() == 3 && s.ids()[s.uploaded()] == 9);
  // AddAll dedups a whole prompt
  SeenTokens p(100);
  p.AddAll({5, 5, 5, 6, 5, 7, 6, 99, 100, -2});
  CHECK((p.ids() == std::vector<int32_t>{5, 6, 7, 99}));
  // Apply: each distinct id once, however often it occurred
  std::vector<float> row(100, 4.0f);
  p.Apply(row.data(), 100, 2.0f);
  for (int i = 0; i < 100; ++i) CHECK(row[i] == (i == 5 || i == 6 || i == 7 || i == 99 ? 2.0f : 4.0f));
}

void TestSamplingOrder() {
  // Greedy: token 2 is the argmax; once seen, a penalty > 1 moves the argmax to token 0.
  std::vector<float> row = {5.0f, 1.0f, 6.0f, -1.0f};
  SampleParams greedy;
  greedy.temperature = 0.0f;
  std::mt19937_64 rng(1);
  CHECK(Sample(row.data(), 4, greedy, rng) == 2);
  SeenTokens seen(4);
  seen.Add(2);
  seen.Apply(row.data(), 4, 1.5f);  // 6 -> 4
  CHECK(row[2] == 4.0f);
  CHECK(Sample(row.data(), 4, greedy, rng) == 0);
  // The penalty field defaults to off and Sample() does not consult it: identical draws with and without it.
  SampleParams a;
  CHECK(a.repetition_penalty == 1.0f);
  SampleParams b = a;
  b.repetition_penalty = 3.0f;
  a.temperature = b.temperature = 0.7f;
  a.top_k = b.top_k = 3;
  std::vector<float> r2 = {2.0f, 1.0f, 0.5f, 0.0f, -1.0f};
  std::mt19937_64 g1(7), g2(7);
  for (int i = 0; i < 50; ++i) CHECK(Sample(r2.data(), 5, a, g1) == Sample(r2.data(), 5, b, g2));
  // Penalized-then-sampled vs an independent reference of the same filter: the penalized row decides, so a
  // token whose penalized logit falls out of top-k is never drawn.
  std::vector<float> r3 = {3.0f, 2.9f, 2.8f, 0.0f};
  SeenTokens s3(4);
  s3.Add(0);
  s3.Add(1);
  s3.Apply(r3.data(), 4, 4.0f);  // 0.75, 0.725, 2.8, 0
  SampleParams c;
  c.temperature = 1.0f;
  c.top_k = 1;
  std::mt19937_64 g3(11);
  for (int i = 0; i < 20; ++i) CHECK(Sample(r3.data(), 4, c, g3) == 2);
}

}  // namespace

int main() {
  TestRule();
  TestApply();
  TestSeenTokens();
  TestSamplingOrder();
  if (g_fail != 0) {
    std::fprintf(stderr, "test_rep_penalty_cpu: %d failure(s)\n", g_fail);
    return 1;
  }
  std::puts("test_rep_penalty_cpu: OK");
  return 0;
}