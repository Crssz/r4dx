// HIP-free address arithmetic of the hybrid mode's reshard EXECUTOR (docs/pp-tp2-hybrid.md 3, 7): the strided copies of a
// reshard_plan.h ReshardPlan, cut into row-aligned slices that fit the pinned host ring, and the rectangle each slice covers in a
// holder's per-layer buffer. Model::ReshardExport / ReshardImport / ReshardCopyLocal (hybrid_model.cpp) take the plan's CopyOp
// plus an OpSlice and address the device with RectOf -- nothing else decides where a byte goes -- and the CPU test
// (tests/model/test_reshard_exec_cpu.cpp) runs the same RectOf over host buffers against ReshardRef, so the CPU-tested plan IS
// what runs, down to the pointer arithmetic.
//
// A CopyOp is a list of Run2D (hipMemcpy2D's shape: `height` rows of `width` bytes, `*_pitch` apart, from `*_off`). The bytes of an
// op as they cross the host are its runs' rows in order, compact (row after row, run after run): a slice is a row range of one
// run, so any ring piece boundary lands on a row boundary (a KV row is a 16 KiB-per-head block run, a conv row is the 6 live bytes
// of one channel; neither is ever split).
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "reshard_plan.h"

namespace r4dx::model::hybrid {

// Which holder of a CopyOp a buffer belongs to: the stage's full-head layer buffers or the TP rank's head-split ones.
enum class Side { kFull, kRank };

// Rows [row0, row0 + rows) of run `run` of an op.
struct OpSlice {
  size_t run = 0;
  uint64_t row0 = 0, rows = 0;
};
inline uint64_t SliceBytes(const CopyOp& op, const OpSlice& s) { return op.runs.at(s.run).width * s.rows; }

// Where a slice lives in one holder's per-layer buffer: `rows` rows of `width` bytes, `pitch` apart, the first at byte `off`.
struct Rect {
  uint64_t off = 0, pitch = 0, width = 0, rows = 0;
  uint64_t Bytes() const { return width * rows; }
  // One past the last byte touched (0 for an empty rectangle).
  uint64_t End() const { return rows == 0 ? 0 : off + (rows - 1) * pitch + width; }
};
inline Rect RectOf(const CopyOp& op, const OpSlice& s, Side side) {
  const Run2D& r = op.runs.at(s.run);
  if (s.row0 > r.height || s.rows > r.height - s.row0) {
    throw std::out_of_range("hybrid::RectOf: rows [" + std::to_string(s.row0) + ", " + std::to_string(s.row0 + s.rows) + ") of a run of " +
                            std::to_string(r.height));
  }
  Rect q;
  q.width = r.width;
  q.rows = s.rows;
  q.pitch = side == Side::kFull ? r.full_pitch : r.rank_pitch;
  q.off = (side == Side::kFull ? r.full_off : r.rank_off) + s.row0 * q.pitch;
  return q;
}

// Every row of every run of an op, one slice per run (empty runs are dropped).
inline std::vector<OpSlice> WholeOp(const CopyOp& op) {
  std::vector<OpSlice> out;
  for (size_t i = 0; i < op.runs.size(); ++i) {
    if (op.runs[i].height > 0 && op.runs[i].width > 0) out.push_back({i, 0, op.runs[i].height});
  }
  return out;
}

// An op cut into slices of at most `max_bytes` (at least one row each), in stream order.
inline std::vector<OpSlice> SplitOp(const CopyOp& op, uint64_t max_bytes) {
  if (max_bytes == 0) throw std::invalid_argument("hybrid::SplitOp: max_bytes must be positive");
  std::vector<OpSlice> out;
  for (size_t i = 0; i < op.runs.size(); ++i) {
    const Run2D& r = op.runs[i];
    if (r.height == 0 || r.width == 0) continue;
    const uint64_t per = std::max<uint64_t>(1, max_bytes / r.width);
    for (uint64_t row = 0; row < r.height; row += per) out.push_back({i, row, std::min(per, r.height - row)});
  }
  return out;
}

// ---- the ring batches -------------------------------------------------------------------------------------------------------
// The cross-card bytes leaving one card are one stream (the plan's cross ops in order); a batch is the part of it that goes through
// one pinned ring slot in one go: at most `piece_bytes` (reshard_plan.h kRingPieceBytes), cut at ROW boundaries, so a batch is
// short of a full piece by less than one row width (<= 16 KiB for the KV runs the pieces are made of). D2H of batch i on the source
// card overlaps H2D of batch i-1 on the other (design 3, 7).
struct SliceRef {
  size_t op = 0;  // index into ReshardPlan::ops
  OpSlice slice;
};
struct RingBatch {
  std::vector<SliceRef> slices;
  uint64_t bytes = 0;
  int slot = 0;  // round robin over `slots`
};
inline std::vector<RingBatch> PackCrossOps(const ReshardPlan& plan, int source_stage, uint64_t piece_bytes = kRingPieceBytes,
                                           int slots = kRingSlots) {
  if (piece_bytes == 0 || slots < 1) throw std::invalid_argument("hybrid::PackCrossOps: piece_bytes and slots must be positive");
  std::vector<RingBatch> out;
  RingBatch cur;
  const auto flush = [&] {
    if (cur.bytes == 0) return;
    cur.slot = static_cast<int>(out.size() % static_cast<size_t>(slots));
    out.push_back(std::move(cur));
    cur = RingBatch{};
  };
  for (size_t i = 0; i < plan.ops.size(); ++i) {
    const CopyOp& op = plan.ops[i];
    if (op.local || SourceStage(op, plan.dir) != source_stage) continue;
    for (size_t ri = 0; ri < op.runs.size(); ++ri) {
      const Run2D& r = op.runs[ri];
      if (r.height == 0 || r.width == 0) continue;
      if (r.width > piece_bytes) throw std::invalid_argument("hybrid::PackCrossOps: a row of " + std::to_string(r.width) + " bytes does not fit a piece of " + std::to_string(piece_bytes));
      uint64_t row = 0;
      while (row < r.height) {
        const uint64_t fit = (piece_bytes - cur.bytes) / r.width;
        if (fit == 0) {
          flush();
          continue;
        }
        const uint64_t take = std::min(fit, r.height - row);
        cur.slices.push_back({i, {ri, row, take}});
        cur.bytes += take * r.width;
        row += take;
      }
    }
  }
  flush();
  return out;
}

}  // namespace r4dx::model::hybrid
