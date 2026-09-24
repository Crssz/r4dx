// tests/model/test_tp_vocab_merge.cpp -- CPU-only (no GPU). docs/tp.md 7.2-7.4, 8.2, 10.1.
//
// Under TP=2 the lm_head is vocab-split: rank r owns ids [r*V/2, (r+1)*V/2) and every per-row
// device result covers one shard. src/model/tp/tp_vocab.h merges the per-rank results on the host;
// this test proves each merge reproduces the FULL-row result:
//   * MergeArgmax == kernels::Argmax(full row) -- lowest-index tie-break across the shard boundary,
//     -inf rows and half-rows;
//   * MergeRowSummaries == the full row's summary: ids/vals exact, lse within 2 fp32 ulp (ABSOLUTE,
//     + 1e-7) of the exact full-row logsumexp, on random, tie-heavy, peaked, flat rows and a row
//     whose second shard is all -inf, V in {4096, 248320}, T down to 0.005;
//   * SampleFromSummary(merged) == SampleCanonical(full row) for 100000 u x the six filter configs
//     of tests/kernels/test_summary_sampler.cpp whenever it resolves (V = 4096; a 400-u spot check
//     at the real V = 248320), and never needs a fallback where the filter closes inside the top-K;
//   * MergeTop16 == the full row's top-16 under r4dx_topk16_f32's total order, ties included.
// The per-shard inputs come from CPU stand-ins of the device kernels (same total order, same
// logsumexp definition); tests/kernels/test_topk_lse.cpp / test_topk16.cpp pin the kernels to them.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tp/tp_vocab.h"

using namespace r4dx::model::tp;
using r4dx::kernels::RowSummary;
using r4dx::kernels::SampleParams;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int kK = r4dx::kernels::kRowSummaryMaxK;

bool Canon(const float* l, int32_t a, int32_t b) {
  return l[a] > l[b] || (l[a] == l[b] && a < b);
}

// ---- CPU stand-ins of the device kernels, over logits[0, n), ids offset by `off` ----------------

ArgmaxPair ShardArgmax(const float* logits, int64_t n, int64_t off) {
  const int32_t i = r4dx::kernels::Argmax(logits, n);  // lowest index among equal maxima
  return {static_cast<int32_t>(off + i), logits[i]};
}

// One fp32 ulp at |x|: the gap from float(|x|) to the next float up.
double UlpF32(double x) {
  const float f = static_cast<float>(std::fabs(x));
  return static_cast<double>(std::nextafter(f, std::numeric_limits<float>::infinity()) - f);
}

// Exact logsumexp of logits * inv_t, in double (Kahan), -inf for an all -inf row.
double ExactLse(const float* logits, int64_t n, float inv_t) {
  double m = -std::numeric_limits<double>::infinity();
  for (int64_t i = 0; i < n; ++i) m = std::max(m, static_cast<double>(logits[i]));
  if (m == -std::numeric_limits<double>::infinity()) return m;
  double sum = 0.0, comp = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double term = std::exp((static_cast<double>(logits[i]) - m) * static_cast<double>(inv_t));
    const double y = term - comp;
    const double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  return m * static_cast<double>(inv_t) + std::log(sum);
}

// r4dx_topk_lse_f32's contract over one shard: top-min(64, n) in canonical order (raw values), ids
// made global, lse rounded to fp32.
RowSummary ShardSummary(const float* logits, int64_t n, int64_t off, float inv_t) {
  RowSummary s;
  s.k = static_cast<int>(std::min<int64_t>(kK, n));
  s.vocab = n;
  s.inv_temperature = inv_t;
  std::vector<int32_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + s.k, order.end(),
                    [&](int32_t a, int32_t b) { return Canon(logits, a, b); });
  for (int j = 0; j < s.k; ++j) {
    s.ids[j] = static_cast<int32_t>(off + order[j]);
    s.vals[j] = logits[order[j]];
  }
  s.lse = static_cast<float>(ExactLse(logits, n, inv_t));
  return s;
}

// r4dx_topk16_f32's contract over one shard.
void ShardTop16(const float* logits, int64_t n, int64_t off, int32_t* ids, float* vals) {
  std::vector<int32_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                    [&](int32_t a, int32_t b) { return Canon(logits, a, b); });
  for (int j = 0; j < 16; ++j) {
    ids[j] = static_cast<int32_t>(off + order[j]);
    vals[j] = logits[order[j]];
  }
}

// The per-rank summaries of a vocab-split row, merged.
RowSummary MergedSummary(const std::vector<float>& row, float inv_t) {
  const int64_t V = static_cast<int64_t>(row.size()), half = V / 2;
  const RowSummary per[2] = {ShardSummary(row.data(), half, 0, inv_t),
                             ShardSummary(row.data() + half, half, half, inv_t)};
  return MergeRowSummaries(per, 2, V);
}

// ---- rows ---------------------------------------------------------------------------------------

std::vector<float> RandomRow(int64_t V, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 3.0f);
  std::vector<float> l(static_cast<size_t>(V));
  for (float& x : l) x = d(rng);
  return l;
}

// Integers in about [-8, 8]: hundreds of exact ties at every rank, including across the boundary.
std::vector<float> TieHeavyRow(int64_t V, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 2.0f);
  std::vector<float> l(static_cast<size_t>(V));
  for (float& x : l) x = std::round(d(rng));
  return l;
}

// test_summary_sampler.cpp's Zipf-like decode row, with the ranks scattered over both shards.
std::vector<float> PeakedRow(int64_t V, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> jitter(-0.3f, 0.3f);
  std::vector<float> l(static_cast<size_t>(V));
  for (int64_t i = 0; i < V; ++i) {
    l[static_cast<size_t>(i)] =
        static_cast<float>(10.0 - 2.4 * std::log(static_cast<double>(i + 1))) + jitter(rng);
  }
  std::shuffle(l.begin(), l.end(), rng);
  return l;
}

std::vector<float> FlatRow(int64_t V, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
  std::vector<float> l(static_cast<size_t>(V));
  for (float& x : l) x = 1.0f + jitter(rng);
  return l;
}

struct Shape {
  const char* name;
  std::vector<float> (*make)(int64_t, uint32_t);
};
const Shape kShapes[] = {
    {"random", RandomRow}, {"tie-heavy", TieHeavyRow}, {"peaked", PeakedRow}, {"flat", FlatRow}};

// ---- MergeArgmax --------------------------------------------------------------------------------

bool ArgmaxAgrees(const std::vector<float>& row) {
  const int64_t V = static_cast<int64_t>(row.size()), half = V / 2;
  const ArgmaxPair per[2] = {ShardArgmax(row.data(), half, 0),
                             ShardArgmax(row.data() + half, half, half)};
  return MergeArgmax(per, 2) == r4dx::kernels::Argmax(row.data(), V);
}

void TestMergeArgmax() {
  for (int64_t V : {int64_t{4096}, int64_t{248320}}) {
    const std::string v = " (V=" + std::to_string(V) + ")";
    const int64_t half = V / 2;
    bool ok = true;
    for (const Shape& s : kShapes) {
      for (uint32_t seed = 1; seed <= 4; ++seed) ok = ok && ArgmaxAgrees(s.make(V, seed));
    }
    Check(ok, "MergeArgmax == full-row Argmax on random / tie-heavy / peaked / flat rows" + v);

    std::vector<float> row = RandomRow(V, 9);
    row[static_cast<size_t>(half - 1)] = 100.0f;  // equal maxima either side of the boundary
    row[static_cast<size_t>(half)] = 100.0f;
    Check(ArgmaxAgrees(row) && r4dx::kernels::Argmax(row.data(), V) == half - 1,
          "a tie straddling the shard boundary resolves to the lower id (rank 0)" + v);
    row = RandomRow(V, 10);
    row[5] = 50.0f;
    row[static_cast<size_t>(half)] = 50.0f;  // rank 1's LOCAL index 0 vs rank 0's global 5
    Check(ArgmaxAgrees(row) && r4dx::kernels::Argmax(row.data(), V) == 5,
          "rank 1's local index 0 does not beat an equal value at rank 0's id 5" + v);
    row = RandomRow(V, 11);
    row[static_cast<size_t>(V - 1)] = 70.0f;
    Check(ArgmaxAgrees(row), "a strict maximum on rank 1 wins" + v);

    std::vector<float> all_inf(static_cast<size_t>(V), kNegInf);
    Check(ArgmaxAgrees(all_inf) && r4dx::kernels::Argmax(all_inf.data(), V) == 0,
          "an all -inf row returns id 0" + v);
    std::vector<float> half_inf = RandomRow(V, 12);
    std::fill(half_inf.begin(), half_inf.begin() + half, kNegInf);
    Check(ArgmaxAgrees(half_inf), "rank 0 all -inf: rank 1's maximum wins" + v);
    half_inf = RandomRow(V, 13);
    std::fill(half_inf.begin() + half, half_inf.end(), kNegInf);
    Check(ArgmaxAgrees(half_inf), "rank 1 all -inf: rank 0's maximum wins" + v);
  }
}

// ---- LogAddExp, MergeRowSummaries ---------------------------------------------------------------

void TestLogAddExp() {
  const double inf = std::numeric_limits<double>::infinity();
  Check(LogAddExp(-inf, -inf) == -inf, "LogAddExp(-inf, -inf) == -inf");
  Check(LogAddExp(-inf, 3.5) == 3.5 && LogAddExp(3.5, -inf) == 3.5, "-inf is LogAddExp's identity");
  Check(std::fabs(LogAddExp(2.0, 2.0) - (2.0 + std::log(2.0))) < 1e-15,
        "LogAddExp(x, x) = x + ln 2");
  Check(std::fabs(LogAddExp(1000.0, 999.0) - (1000.0 + std::log1p(std::exp(-1.0)))) < 1e-12,
        "LogAddExp is stable at large magnitudes");
  Check(LogAddExp(1.25, -7.5) == LogAddExp(-7.5, 1.25), "LogAddExp is symmetric (these inputs)");
}

bool SummaryMatches(const std::vector<float>& row, float inv_t, std::string* why) {
  const int64_t V = static_cast<int64_t>(row.size());
  const RowSummary merged = MergedSummary(row, inv_t);
  const RowSummary full = ShardSummary(row.data(), V, 0, inv_t);
  if (merged.k != full.k || merged.vocab != V || merged.inv_temperature != inv_t) {
    *why = "k/vocab/inv_temperature";
    return false;
  }
  for (int j = 0; j < full.k; ++j) {
    if (merged.ids[j] != full.ids[j] || merged.vals[j] != full.vals[j]) {
      *why = "entry " + std::to_string(j) + ": id " + std::to_string(merged.ids[j]) + " vs " +
             std::to_string(full.ids[j]);
      return false;
    }
  }
  const double exact = ExactLse(row.data(), V, inv_t);
  const double lse = static_cast<double>(merged.lse);
  if (std::isinf(exact)) {
    if (lse != exact) {
      *why = "lse of an all -inf row";
      return false;
    }
    return true;
  }
  // ABSOLUTE, in fp32 ulps: SampleFromSummary reads lse only through
  // S_full = exp(lse - vals[0] * inv_t), so an absolute lse error IS the relative S_full error that
  // kRowSummaryLseRelTol bounds -- a tolerance relative to |lse| would accept several times the
  // whole band at T = 0.005 (|lse| ~ 2000-2700 here). An exact merge carries two fp32 roundings
  // (the dominant shard's lse and the merged value, <= 0.5 ulp each), so it lands within ~1 ulp.
  const double tol = 2.0 * UlpF32(exact) + 1e-7;
  if (tol >= r4dx::kernels::kRowSummaryLseRelTol) {
    *why = "|lse| " + std::to_string(exact) + " is too large for the check to mean anything";
    return false;
  }
  const double err = std::fabs(lse - exact);
  if (err > tol) {
    *why = "lse " + std::to_string(lse) + " vs exact " + std::to_string(exact) + " (error " +
           std::to_string(err) + " > " + std::to_string(tol) + ")";
    return false;
  }
  return true;
}

void TestMergeRowSummaries() {
  const float inv_ts[] = {1.0f, 1.0f / 0.7f, 1.0f / 0.6f, 1.0f / 0.8f, 1.0f / 0.005f};
  for (int64_t V : {int64_t{4096}, int64_t{248320}}) {
    const std::string v = " (V=" + std::to_string(V) + ")";
    for (const Shape& s : kShapes) {
      bool ok = true;
      std::string why;
      for (uint32_t seed = 1; seed <= 3 && ok; ++seed) {
        const std::vector<float> row = s.make(V, 100 + seed);
        for (float inv_t : inv_ts) {
          if (!SummaryMatches(row, inv_t, &why)) {
            ok = false;
            break;
          }
        }
      }
      Check(ok, std::string("MergeRowSummaries == full-row summary on ") + s.name +
                    " rows (ids/vals exact, lse within 2 fp32 ulp)" + v + (ok ? "" : ": " + why));
    }
    std::vector<float> row = RandomRow(V, 200);
    std::fill(row.begin() + V / 2, row.end(), kNegInf);
    std::string why;
    Check(SummaryMatches(row, 1.0f, &why),
          "rank 1 all -inf: its lse (-inf) is LogAddExp's identity" + v);
    std::vector<float> all_inf(static_cast<size_t>(V), kNegInf);
    Check(SummaryMatches(all_inf, 1.0f, &why), "an all -inf row merges to lse -inf" + v);
  }

  // A tie band straddling both the shard boundary and the 64th place.
  {
    const int64_t V = 4096;
    std::vector<float> row = RandomRow(V, 300);
    for (float& x : row) x = std::min(x, 4.0f);
    for (int64_t i = 0; i < 40; ++i) row[static_cast<size_t>(V / 2 - 20 + i)] = 9.0f;
    for (int64_t i = 0; i < 60; ++i) row[static_cast<size_t>(i * 67)] = 9.0f;
    std::string why;
    const bool ok = SummaryMatches(row, 1.0f, &why);
    Check(ok, "100 equal maxima across both shards: the top-64 keeps the 64 lowest ids" +
                  (ok ? std::string() : ": " + why));
  }

  RowSummary a = ShardSummary(RandomRow(256, 1).data(), 256, 0, 1.0f);
  RowSummary b = ShardSummary(RandomRow(256, 2).data(), 256, 256, 2.0f);
  const RowSummary pair[2] = {a, b};
  bool threw = false;
  try {
    MergeRowSummaries(pair, 2, 512);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "MergeRowSummaries refuses ranks summarized at different temperatures");
  const RowSummary one = MergeRowSummaries(&a, 1, 256);
  Check(one.k == a.k && std::equal(a.ids, a.ids + a.k, one.ids) && one.lse == a.lse,
        "world 1: the merge is the identity");
}

// ---- SampleFromSummary(merged) == SampleCanonical(full) -----------------------------------------

struct Combo {
  const char* name;
  SampleParams p;
  bool must_always_resolve;
};

std::vector<Combo> Combos() {  // tests/kernels/test_summary_sampler.cpp's six
  const auto P = [](float t, int k, float top_p, float min_p) {
    SampleParams p;
    p.temperature = t;
    p.top_k = k;
    p.top_p = top_p;
    p.min_p = min_p;
    return p;
  };
  return {{"T=1.0 (pure temperature)", P(1.0f, 0, 1.0f, 0.0f), false},
          {"T=0.7 top_k=20 top_p=0.8", P(0.7f, 20, 0.8f, 0.0f), true},
          {"T=0.6 top_k=20 top_p=0.95", P(0.6f, 20, 0.95f, 0.0f), true},
          {"T=1.0 top_p=0.9", P(1.0f, 0, 0.9f, 0.0f), false},
          {"T=0.8 min_p=0.05", P(0.8f, 0, 1.0f, 0.05f), false},
          {"T=0.7 top_k=50 min_p=0.02 top_p=0.9", P(0.7f, 50, 0.9f, 0.02f), true}};
}

std::vector<double> MakeUs(int n) {  // half on a grid, half random (test_summary_sampler.cpp)
  std::vector<double> us;
  for (int i = 0; i < n / 2; ++i) us.push_back(static_cast<double>(i) / (n / 2));
  std::mt19937_64 rng(777);
  for (int i = 0; i < n - n / 2; ++i) us.push_back(r4dx::kernels::DrawUniform01(rng));
  return us;
}

struct Cell {
  std::string name;
  bool must_resolve = false;
  int64_t resolved = 0, fallback = 0, mismatch = 0;
};

// One (row, combo) cell: every u through the merged summary, and the full row wherever it resolves.
void RunCell(const std::vector<float>& row, const Combo& c, const std::vector<double>& us,
             Cell* out) {
  const int64_t V = static_cast<int64_t>(row.size());
  const RowSummary merged = MergedSummary(row, 1.0f / c.p.temperature);
  for (double u : us) {
    const auto r = r4dx::kernels::SampleFromSummary(merged, c.p, u);
    if (!r.resolved) {
      ++out->fallback;
      continue;
    }
    ++out->resolved;
    if (r.token != r4dx::kernels::SampleCanonical(row.data(), V, c.p, u)) ++out->mismatch;
  }
}

void TestSamplingEquivalence(int64_t V, int n_us, uint32_t seed) {
  const std::vector<double> us = MakeUs(n_us);
  const std::vector<Combo> combos = Combos();
  std::vector<std::vector<float>> rows;
  std::vector<Cell> cells;
  for (const Shape& s : kShapes) {
    rows.push_back(s.make(V, seed));
    for (const Combo& c : combos) {
      Cell cell;
      cell.name = std::string(s.name) + " / " + c.name;
      cell.must_resolve = c.must_always_resolve;
      cells.push_back(cell);
    }
  }
  std::vector<std::thread> threads;
  for (size_t i = 0; i < cells.size(); ++i) {
    threads.emplace_back([&, i] {
      RunCell(rows[i / combos.size()], combos[i % combos.size()], us, &cells[i]);
    });
  }
  for (auto& t : threads) t.join();

  std::printf("\nV=%lld, %d values of u per cell\n%-48s %9s %10s %9s\n", static_cast<long long>(V),
              n_us, "shape / filters", "resolved", "fallback%", "mismatch");
  int64_t mismatches = 0;
  bool closed_ok = true;
  for (const Cell& c : cells) {
    std::printf("%-48s %9lld %9.2f%% %9lld\n", c.name.c_str(), static_cast<long long>(c.resolved),
                100.0 * static_cast<double>(c.fallback) / n_us, static_cast<long long>(c.mismatch));
    mismatches += c.mismatch;
    if (c.must_resolve && c.fallback != 0) closed_ok = false;
  }
  const std::string v = " (V=" + std::to_string(V) + ", " + std::to_string(n_us) +
                        " u x 6 configs x 4 shapes)";
  Check(mismatches == 0,
        "every resolved SampleFromSummary(merged) token == SampleCanonical(full row)" + v);
  Check(closed_ok, "filters that close inside the top-K never need the full row" + v);
}

// ---- MergeTop16 ---------------------------------------------------------------------------------

void TestMergeTop16() {
  for (int64_t V : {int64_t{4096}, int64_t{248320}}) {
    const int64_t half = V / 2;
    constexpr int kRows = 8;
    std::vector<std::vector<float>> rows;
    for (int i = 0; i < 4; ++i) rows.push_back(kShapes[i].make(V, 40 + i));
    rows.push_back(std::vector<float>(static_cast<size_t>(V), 0.5f));  // all equal: ids 0..15
    {
      std::vector<float> r = RandomRow(V, 45);  // an 8-way tie for places 12..19, both shards
      for (float& x : r) x = std::min(x, 5.0f);
      for (int i = 0; i < 12; ++i) r[static_cast<size_t>(i * 97 + 3)] = 20.0f - i;
      for (int i = 0; i < 4; ++i) r[static_cast<size_t>(half - 4 + 2 * i)] = 7.0f;
      for (int i = 0; i < 4; ++i) r[static_cast<size_t>(half + 3 * i)] = 7.0f;
      rows.push_back(r);
    }
    {
      std::vector<float> r = RandomRow(V, 46);  // rank 1 holds the whole top-16
      for (int i = 0; i < 16; ++i) r[static_cast<size_t>(half + 100 + i)] = 30.0f + i;
      rows.push_back(r);
    }
    {
      std::vector<float> r = TieHeavyRow(V, 47);
      std::fill(r.begin() + half, r.end(), kNegInf);  // rank 1 all -inf
      rows.push_back(r);
    }

    std::vector<int32_t> ids(2 * kRows * 16);
    std::vector<float> vals(2 * kRows * 16);
    for (int row = 0; row < kRows; ++row) {
      for (int r = 0; r < 2; ++r) {
        const size_t at = (static_cast<size_t>(r) * kRows + row) * 16;
        ShardTop16(rows[row].data() + r * half, half, r * half, ids.data() + at, vals.data() + at);
      }
    }
    std::vector<int32_t> out_ids(kRows * 16);
    std::vector<float> out_vals(kRows * 16);
    MergeTop16(ids.data(), vals.data(), 2, kRows, out_ids.data(), out_vals.data());

    bool ok = true;
    for (int row = 0; row < kRows; ++row) {
      int32_t want_ids[16];
      float want_vals[16];
      ShardTop16(rows[row].data(), V, 0, want_ids, want_vals);
      for (int j = 0; j < 16; ++j) {
        const bool same = out_ids[row * 16 + j] == want_ids[j] &&
                          out_vals[row * 16 + j] == want_vals[j];
        if (!same) {
          std::fprintf(stderr, "  V=%lld row %d place %d: merged (%d, %g) full (%d, %g)\n",
                       static_cast<long long>(V), row, j, out_ids[row * 16 + j],
                       out_vals[row * 16 + j], want_ids[j], want_vals[j]);
        }
        ok = ok && same;
      }
    }
    Check(ok, "MergeTop16 == the full row's top-16 under the (value desc, id asc) order, 8 rows "
              "incl. all-equal, cross-shard ties, one-shard top-16, -inf half (V=" +
                  std::to_string(V) + ")");
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  TestMergeArgmax();
  TestLogAddExp();
  TestMergeRowSummaries();
  TestMergeTop16();
  TestSamplingEquivalence(4096, 100000, 7);
  TestSamplingEquivalence(248320, 400, 8);
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
