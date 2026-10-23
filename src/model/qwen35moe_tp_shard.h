// r4dx::model::qwen35moe_tp -- CPU-only (HIP-free, header-only) byte plans for one rank's slice of a qwen3_5_moe trellis
// linear, docs/qwen35moe-tp-design.md 2.1-2.3.
//
// ShardLoader::TrellisSlice (shard_loader.h) cuts a trellis linear into this rank's words / sign vectors and uploads
// them in one go. The routed experts cannot use it as is: LoadExpertClass keeps ONE stacked buffer per (layer, class)
// that the grouped decode kernel indexes with `all_w + expert * w_per` (moe_trellis_decode.hip), so the sharded loader
// gathers every expert's slice into that stack at `e * w_per_rank`. This header is the arithmetic of one slice, shared
// by that loader (P1) and by tests/model/qwen35moe/test_q35moe_tp_shard.cpp, which proves it on the real container's
// bytes. The plans are the same tp::PlanRows / tp::PlanCols ShardLoader runs; nothing here knows a device.
//
//   column-parallel (kRows): words by rows, suh replicated (every part, full K), svh cut by the rank's rows;
//   row-parallel    (kCols): words by K tiles, suh cut by the rank's K range (a single part), svh replicated;
//   replicated:             everything whole.
// The sign vectors are fp16 on disk (the loader widens them to fp32), so their runs are in fp16 elements * 2 bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tp/tp_shard.h"
#include "trellis_meta.h"

namespace r4dx::model::qwen35moe_tp {

// One rank's slice of a trellis linear [N, K] (GLOBAL shape), in the on-disk parts' byte runs.
struct TrellisSlicePlan {
  std::vector<tp::ByteRun> w;    // into <base>.trellis.w
  std::vector<tp::ByteRun> suh;  // into <base>.trellis.suh (fp16)
  std::vector<tp::ByteRun> svh;  // into <base>.trellis.svh (fp16)
  int64_t N = 0, K = 0;          // the rank's linear
  int64_t part_n[2] = {0, 0};    // the rank's rows inside each part (QuantLinear::trellis_part_n); {N, 0} for one part
  int parts = 1;                 // number of input transforms (suh vectors) the rank holds
  size_t w_bytes = 0;            // sum of w runs == N * K * bits / 8
  size_t suh_elems = 0;          // fp16 elements in the rank's suh (parts * K for a column-parallel / replicated linear)
  size_t svh_elems = 0;          // fp16 elements in the rank's svh (N)
};

inline size_t RunBytes(const std::vector<tp::ByteRun>& runs) {
  size_t t = 0;
  for (const tp::ByteRun& r : runs) t += r.bytes;
  return t;
}

// `rule`: tp::RuleFor of the linear's base name. `parts`: the container's quant.trellis `parts` of the linear (empty: one part
// of all N rows). `world == 1` returns the whole linear (the replicated plan). Throws std::invalid_argument for a rank range
// that is not whole 128-blocks, a row range that crosses a part boundary, a part with no rows on the rank, or a row-parallel
// linear with two input transforms -- the same refusals ShardLoader::TrellisSlice makes.
inline TrellisSlicePlan PlanTrellisLinear(const tp::ShardRule& rule, int64_t N, int64_t K, int bits,
                                          const std::vector<int64_t>& parts, int world, int rank) {
  const auto fail = [&](const std::string& why) {
    throw std::invalid_argument("r4dx::model::qwen35moe_tp::PlanTrellisLinear [" + std::to_string(N) + ", " +
                                std::to_string(K) + "] KB " + std::to_string(bits) + " at TP=" + std::to_string(world) +
                                " rank " + std::to_string(rank) + ": " + why);
  };
  std::vector<tp::Range> rows;
  tp::Range cols{0, K};
  if (rule.split == tp::Split::kRows) {
    int64_t total = 0;
    for (const tp::Segment& s : rule.segments) total += s.rows;
    if (total != N) fail("the rule splits " + std::to_string(total) + " rows");
    rows = tp::RankRows(rule, world, rank);
  } else if (rule.split == tp::Split::kCols) {
    if (rule.k_total != K) fail("the rule splits K = " + std::to_string(rule.k_total));
    cols = tp::RankCols(rule, world, rank);
  } else if (rule.split != tp::Split::kReplicate) {
    fail("a rank-0-only rule");
  }
  if (rule.split != tp::Split::kRows) rows = {tp::Range{0, N}};

  for (const tp::Range& r : rows) {
    if (r.begin % kTrellisBlock != 0 || r.count % kTrellisBlock != 0) fail("rows are not whole 128-blocks");
  }
  if (cols.begin % kTrellisBlock != 0 || cols.count % kTrellisBlock != 0) fail("the K range is not whole 128-blocks");

  const int nparts = parts.empty() ? 1 : static_cast<int>(parts.size());
  if (nparts > 2) fail("more than two parts");
  tp::PartShape w;
  w.part = tp::Part::kTrellisW;
  w.N = N;
  w.K = K;
  w.rate = bits;
  const auto elem = [](int64_t n) {
    tp::PartShape s;
    s.part = tp::Part::kElem;
    s.N = n;
    s.row_bytes = 2;
    return s;
  };

  TrellisSlicePlan p;
  p.parts = nparts;
  p.N = 0;
  for (const tp::Range& r : rows) p.N += r.count;
  p.K = cols.count;
  if (rule.split == tp::Split::kCols) {
    p.w = tp::PlanCols(w, cols);
  } else {
    p.w = tp::PlanRows(w, rows);
  }
  if (rule.split == tp::Split::kCols) {
    if (nparts != 1) fail("a row-parallel linear with two input transforms");
    p.suh = tp::PlanRows(elem(K), {cols});
  } else {
    p.suh = {tp::ByteRun{0, static_cast<size_t>(nparts) * static_cast<size_t>(K) * 2}};
  }
  if (rule.split == tp::Split::kRows) {
    p.svh = tp::PlanRows(elem(N), rows);
  } else {
    p.svh = {tp::ByteRun{0, static_cast<size_t>(N) * 2}};
  }

  const int64_t global_n[2] = {nparts > 1 ? parts[0] : N, nparts > 1 ? parts[1] : 0};
  if (rule.split != tp::Split::kRows) {
    p.part_n[0] = global_n[0];
    p.part_n[1] = global_n[1];
  } else {
    int last_part = 0;
    for (const tp::Range& r : rows) {
      const int pi = (nparts > 1 && r.begin >= global_n[0]) ? 1 : 0;
      const int64_t part_end = pi == 0 ? global_n[0] : global_n[0] + global_n[1];
      if (r.begin + r.count > part_end || pi < last_part) fail("rows cross a part boundary");
      p.part_n[pi] += r.count;
      last_part = pi;
    }
    if (nparts > 1 && (p.part_n[0] == 0 || p.part_n[1] == 0)) fail("a part has no rows on this rank");
  }
  p.w_bytes = RunBytes(p.w);
  p.suh_elems = RunBytes(p.suh) / 2;
  p.svh_elems = RunBytes(p.svh) / 2;
  return p;
}

// The routed experts of one (layer, class) stacked for one rank: expert e's slice starts at `e * w_per_rank` uint32 words
// (and `e * suh_per_rank` / `e * svh_per_rank` floats) of the stacked buffers, exactly the strides the grouped decode kernel
// is instantiated with. The per-expert plan is identical for every expert of a class (same shape, same KB), so one plan
// describes all of them; the caller adds each expert's own tensor offsets.
struct ExpertClassPlan {
  TrellisSlicePlan slice;      // the rank's slice of ONE expert
  size_t w_per_rank = 0;       // uint32 words of one expert's slice
  size_t suh_per_rank = 0;     // floats
  size_t svh_per_rank = 0;     // floats
  int64_t experts = 0;
  size_t StackWords() const { return static_cast<size_t>(experts) * w_per_rank; }
};

// `global`: Qwen35MoeConfig::ToModelConfig() of the UNSHARDED config. `gate_up`: the [2 * I, H] class (two parts of I rows,
// own sign vectors each); else the [H, I] down class.
inline ExpertClassPlan PlanExpertClass(const ModelConfig& global, bool gate_up, int bits, int world, int rank) {
  const int64_t H = global.hidden_size, I = global.moe_intermediate_size;
  ExpertClassPlan c;
  c.experts = global.num_experts;
  const std::string base = std::string("text.layers.0.moe.experts.0.") + (gate_up ? "gate_up" : "down");
  const tp::ShardRule rule = tp::RuleFor(base, global);
  c.slice = gate_up ? PlanTrellisLinear(rule, 2 * I, H, bits, {I, I}, world, rank)
                    : PlanTrellisLinear(rule, H, I, bits, {}, world, rank);
  c.w_per_rank = c.slice.w_bytes / sizeof(uint32_t);
  c.suh_per_rank = c.slice.suh_elems;
  c.svh_per_rank = c.slice.svh_elems;
  return c;
}

}  // namespace r4dx::model::qwen35moe_tp
