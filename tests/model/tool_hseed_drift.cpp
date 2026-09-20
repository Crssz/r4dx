// tests/model/tool_hseed_drift.cpp -- one-off diagnostic (NOT a ctest test -- built by
// tests/model/CMakeLists.txt but never registered via add_test) for docs/mtp.md's "Acceptance gap
// investigation" step 1: "a golden-referenced, per-layout comparison of h_seed itself -- the main
// model's pre-final-norm hidden state that the draft head consumes."
//
// Uses r4dx::model::Model::DebugSeedHiddenBf16() (model.h) to read back mtp_seed_hidden_ -- the
// EXACT [hidden] bf16 row MtpHead::Draft's first step consumes -- at several positions through a
// fixed, deterministic token stream fed IDENTICALLY (same token ids, same order) to a bf16
// (exact-arithmetic reference, per this codebase's own convention -- docs/status.md: "bf16 survives
// only as the exact-arithmetic reference in the 4-layer test containers") Model and to each
// quantized layout's Model, all against the SAME newly-converted 4-layer, all-4-layout, mtp-on
// container (D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx -- the 4-layer container this investigation
// needed but that did not previously exist side by side in one file; the pre-existing
// qwen38-27b-l4-mtp.r4dx only has bf16+w4a16). Feeding the SAME fixed token ids to every layout
// (rather than each layout's own greedy continuation) keeps the comparison apples-to-apples -- a
// layout's own generated tokens would otherwise diverge and confound later positions with a
// different input, not just different arithmetic on the same input.
//
// Prints, per layout, per sampled position: cosine similarity and relative L2 vs the bf16
// reference's h_seed, plus the worst-drifting components by absolute difference. Prints a final
// summary table (mean cosine / mean rel L2 per layout) to correlate against the independently-
// measured acceptance ranking (docs/mtp.md's K=3: w4a16 51.9-54.3%, mxfp4 40.0-41.9%, w4a8
// 32.5-34.2%).
//
// CAVEAT: this container is a 4-layer truncation of the real 64-layer model (the only form a bf16
// exact-arithmetic reference exists in, per this project's bf16-retirement rule). The RELATIVE
// h_seed drift ordering across layouts is the evidence this tool produces, on the assumption that
// per-layer quantization error compounds monotonically with depth for a fixed layout -- not that
// the absolute numbers transfer to 64 layers.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "test_common.h"

using r4dx_test::FileExists;
using r4dx_test::RelL2;
using r4dx_test::SkipMissing;
using r4dx_test::WidenBf16;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

const char* kContainerPath = "D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx";

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  if (na <= 0 || nb <= 0) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<size_t> WorstComponents(const std::vector<float>& a, const std::vector<float>& ref,
                                     size_t k) {
  std::vector<size_t> idx(a.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + std::min(k, idx.size()), idx.end(),
                     [&](size_t x, size_t y) {
                       return std::fabs(a[x] - ref[x]) > std::fabs(a[y] - ref[y]);
                     });
  idx.resize(std::min(k, idx.size()));
  return idx;
}

std::vector<int32_t> FixedTokenStream(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 200 + (i * 53) % 6000;
  return ids;
}

// Feeds `tokens` in `chunk`-sized pieces, capturing DebugSeedHiddenBf16() (widened to fp32) after
// EVERY chunk -- one sample per position sub-range boundary.
std::vector<std::vector<float>> SampleHSeedTrajectory(Model& model,
                                                       const std::vector<int32_t>& tokens,
                                                       int chunk) {
  std::vector<std::vector<float>> samples;
  for (size_t off = 0; off < tokens.size(); off += static_cast<size_t>(chunk)) {
    const size_t end = std::min(tokens.size(), off + static_cast<size_t>(chunk));
    const std::vector<int32_t> piece(tokens.begin() + static_cast<ptrdiff_t>(off),
                                      tokens.begin() + static_cast<ptrdiff_t>(end));
    model.Prefill(piece);
    samples.push_back(WidenBf16(model.DebugSeedHiddenBf16()));
  }
  return samples;
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) return SkipMissing(kContainerPath);

  const std::vector<int32_t> tokens = FixedTokenStream(64);
  constexpr int kChunk = 8;

  ModelOptions bf16_opts;
  bf16_opts.container_path = kContainerPath;
  bf16_opts.layout = Layout::kBf16;
  bf16_opts.max_ctx = 256;
  bf16_opts.layer_limit = 4;
  bf16_opts.mtp_draft_k = 1;  // only needs to be >0 to size mtp_seed_hidden_

  Model bf16_ref = Model::Load(bf16_opts);
  const std::vector<std::vector<float>> ref_traj = SampleHSeedTrajectory(bf16_ref, tokens, kChunk);

  const Layout layouts[] = {Layout::kW4a16, Layout::kW4a8, Layout::kMxfp4};
  std::fprintf(stderr, "[hseed_drift] container=%s, %zu positions sampled every %d tokens\n",
               kContainerPath, ref_traj.size(), kChunk);

  for (Layout layout : layouts) {
    ModelOptions opts = bf16_opts;
    opts.layout = layout;
    Model m = Model::Load(opts);
    const std::vector<std::vector<float>> traj = SampleHSeedTrajectory(m, tokens, kChunk);

    double sum_cos = 0.0, sum_rel = 0.0, min_cos = 1.0, max_rel = 0.0;
    for (size_t p = 0; p < traj.size(); ++p) {
      const double cos = Cosine(traj[p], ref_traj[p]);
      const double rel = RelL2(traj[p], ref_traj[p]);
      sum_cos += cos;
      sum_rel += rel;
      min_cos = std::min(min_cos, cos);
      max_rel = std::max(max_rel, rel);
      std::fprintf(stderr, "[hseed_drift] layout=%-6s pos=%3zu cosine=%.6f rel_L2=%.4e\n",
                   LayoutName(layout), (p + 1) * kChunk, cos, rel);
      if (p + 1 == traj.size()) {  // last (deepest) position: print worst components too
        const auto worst = WorstComponents(traj[p], ref_traj[p], 5);
        for (size_t wi : worst) {
          std::fprintf(stderr,
                       "[hseed_drift]   worst component[%zu]: ref=%+.5f got=%+.5f diff=%+.5f\n", wi,
                       ref_traj[p][wi], traj[p][wi], traj[p][wi] - ref_traj[p][wi]);
        }
      }
    }
    std::fprintf(stderr,
                 "[hseed_drift] SUMMARY layout=%-6s mean_cosine=%.6f min_cosine=%.6f "
                 "mean_rel_L2=%.4e max_rel_L2=%.4e\n\n",
                 LayoutName(layout), sum_cos / static_cast<double>(traj.size()), min_cos,
                 sum_rel / static_cast<double>(traj.size()), max_rel);
  }

  return 0;
}
