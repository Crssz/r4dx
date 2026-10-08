// HIP-free rules of the hybrid mode's DFlash tail (docs/pp-tp2-hybrid.md 3 "DFlash ring", 6): the drafter ring is REPLICATED, not
// resharded, and holds the last `window` (2048) injected positions, so after a pipelined prefill each TP rank injects only the
// last >= `window` rows of the call, from features the pipeline stage captured, in the same 64-row slices Model::RunChunk injects.
//   * the start s = max(p0, chunk-aligned n - window): aligned to the call's own 64-row grid (chunks start at p0 + 64 j), so
//     every slice has the same (start, rows) as in a full run -- the fc / k_proj / v_proj GEMMs see identical M and the ring
//     ends up byte-equal;
//   * the rope temporal row of each slice goes with the features (an image can straddle a slice; the delta shortcut only holds
//     past the prompt);
//   * the stage captures slices as its RunChunk drains them (TailCover checks they arrive contiguous and complete).
// Header-only so tests/model/test_dflash_tail_cpu.cpp runs the rules, with a simulated ring, without a device.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::model::hybrid {

constexpr int64_t kDflashSliceRows = 64;     // DflashDraftOptions::max_inject_rows, Model's max_chunk_
constexpr int64_t kDflashWindowRows = 2048;  // the drafter's sliding window == its ring slots

// First position the tail injection covers for a call over rows [p0, n): max(p0, the last grid point at or before n - window).
inline int64_t DflashTailStart(int64_t p0, int64_t n, int64_t window = kDflashWindowRows, int64_t slice = kDflashSliceRows) {
  if (p0 < 0 || n <= p0) throw std::invalid_argument("hybrid::DflashTailStart: needs 0 <= p0 < n");
  if (window < 1 || slice < 1) throw std::invalid_argument("hybrid::DflashTailStart: window and slice must be positive");
  const int64_t want = n - window;
  if (want <= p0) return p0;
  return p0 + (want - p0) / slice * slice;
}

// The 64-row slices of a tail of `rows` rows: {first row, rows}.
struct TailSlice {
  int64_t row0 = 0, rows = 0;
};
inline std::vector<TailSlice> TailSlices(int64_t rows, int64_t slice = kDflashSliceRows) {
  if (slice < 1) throw std::invalid_argument("hybrid::TailSlices: slice must be positive");
  std::vector<TailSlice> out;
  for (int64_t r = 0; r < rows; r += slice) out.push_back({r, std::min(slice, rows - r)});
  return out;
}

// The temporal row of a compact [3, n_rows] rope block (Model::PrefillMultimodal's rope_rows_out: t row, h row, w row), rows
// [from, from + count) of it.
inline std::vector<int32_t> TemporalRopeRows(const std::vector<int32_t>& rope3, int64_t n_rows, int64_t from, int64_t count) {
  if (n_rows < 0 || static_cast<int64_t>(rope3.size()) != 3 * n_rows) throw std::invalid_argument("hybrid::TemporalRopeRows: the block is not [3, n_rows]");
  if (from < 0 || count < 0 || from + count > n_rows) throw std::out_of_range("hybrid::TemporalRopeRows: rows outside the block");
  return std::vector<int32_t>(rope3.begin() + from, rope3.begin() + from + count);
}

// What Model::TpInjectDflashTail requires, as a pure function: "" or the reason it refuses. `injected` is the drafter's frontier,
// `pos` the Model's position (the tail ends there), the tail is rows [start, start + rows).
//   * monotonic: start >= injected (DflashDraft::InjectFeatures throws below it);
//   * a gap (start > injected) leaves everything below `start` unreadable, which is only right when the tail already covers a whole
//     window -- otherwise the visible store would be shorter than the full run's.
inline std::string CheckTailInject(int64_t injected, int64_t pos, int64_t start, int64_t rows, int64_t window = kDflashWindowRows) {
  if (rows < 1) return "the tail has no rows";
  if (start < 0) return "negative start position";
  if (start + rows != pos) {
    return "the tail [" + std::to_string(start) + ", " + std::to_string(start + rows) + ") must end at the Model's position " + std::to_string(pos);
  }
  if (start < injected) return "start " + std::to_string(start) + " is below the drafter's frontier " + std::to_string(injected) + " (injection is monotonic)";
  if (start > injected && rows < window) {
    return "a gap before " + std::to_string(start) + " with only " + std::to_string(rows) + " tail rows leaves a visible store shorter than the window of " +
           std::to_string(window);
  }
  return "";
}

// Bookkeeping of the stage-side capture: the call's slices arrive in order (Model's capture observer: features, rows, start_pos) and
// the ones inside [start, end) are copied to consecutive rows of the host buffer. Throws on a gap or an overlap -- a hole in the
// tail would silently skew every later draft.
class TailCover {
 public:
  TailCover(int64_t start, int64_t end) : start_(start), end_(end), next_(start) {
    if (start < 0 || end <= start) throw std::invalid_argument("hybrid::TailCover: needs 0 <= start < end");
  }
  struct Take {
    int64_t src_row0 = 0;  // first row of the offered slice to copy
    int64_t rows = 0;      // 0: nothing of this slice is wanted
    int64_t dst_row0 = 0;  // row of the tail buffer it lands on
  };
  Take Offer(int64_t slice_start, int64_t slice_rows) {
    Take t;
    if (slice_rows <= 0) return t;
    const int64_t slice_end = slice_start + slice_rows;
    if (slice_end <= start_) return t;  // entirely below the tail
    if (slice_start >= end_) {
      if (next_ != end_) throw std::runtime_error("hybrid::TailCover: the call ended before the tail was complete");
      return t;
    }
    if (next_ == start_ && slice_start > start_) {
      throw std::runtime_error("hybrid::TailCover: the first slice inside the tail starts at " + std::to_string(slice_start) +
                               ", after the tail start " + std::to_string(start_));
    }
    const int64_t from = std::max(slice_start, start_);
    if (from != next_) {
      throw std::runtime_error("hybrid::TailCover: slice at " + std::to_string(slice_start) + " is not contiguous with the rows taken so far (next " +
                               std::to_string(next_) + ")");
    }
    const int64_t upto = std::min(slice_end, end_);
    t.src_row0 = from - slice_start;
    t.rows = upto - from;
    t.dst_row0 = from - start_;
    next_ = upto;
    return t;
  }
  bool Complete() const { return next_ == end_; }
  int64_t Taken() const { return next_ - start_; }
  int64_t Start() const { return start_; }
  int64_t End() const { return end_; }

 private:
  int64_t start_, end_, next_;
};

}  // namespace r4dx::model::hybrid
