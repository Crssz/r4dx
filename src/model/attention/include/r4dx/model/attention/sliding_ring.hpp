// r4dx::model::attention::SlidingRingGeometry -- the host-only arithmetic of a sliding layer's KV
// ring (docs/gemma4-plan.md 3.3): how many blocks it has, the one block table every sliding layer
// shares, the write slots of a chunk, and what chunk sizes it serves. No HIP, no device state, so
// tests/kernels/test_ring_math.cpp runs it on the CPU; SlidingKvCache (sliding_kv_cache.hpp) owns
// the device storage built on it. The per-key math lives in third_party/libr4d/r4d_attn_window.h,
// shared with the windowed attention kernels.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4d_attn_window.h"

namespace r4dx::model::attention {

class SlidingRingGeometry {
 public:
  // window: the sliding window W (1024 for Gemma 4). max_chunk_rows: the largest chunk written at
  // once (prefill chunk 256, an image block 288, a verify window 16 -- the ring must hold W plus
  // it). block_size: the paged block size (16). max_ctx_tokens: the longest context the shared block
  // table must index. ring_tokens: 0 = the default, W + max_chunk_rows rounded up to whole blocks
  // plus `slack_blocks` spare blocks (Gemma 4: 1024 + 288 -> 1312 -> 82 blocks, + 14 = 96 blocks =
  // 1536 tokens, the plan's ring); otherwise the exact ring size (a multiple of block_size).
  SlidingRingGeometry(int window, int max_chunk_rows, int block_size, int max_ctx_tokens,
                      int ring_tokens = 0, int slack_blocks = 14)
      : window_(window), max_chunk_(max_chunk_rows), bs_(block_size), max_ctx_(max_ctx_tokens) {
    if (window_ <= 0 || max_chunk_ <= 0 || bs_ <= 0 || max_ctx_ <= 0) {
      throw std::invalid_argument("SlidingRingGeometry: all sizes must be positive");
    }
    if (ring_tokens == 0) {
      const int need = window_ + max_chunk_;
      ring_tokens = ((need + bs_ - 1) / bs_ + slack_blocks) * bs_;
    }
    if (ring_tokens % bs_ != 0) {
      throw std::invalid_argument("SlidingRingGeometry: ring_tokens must be a multiple of block_size");
    }
    ring_tokens_ = ring_tokens;
    if (!r4d_ring_fits(ring_tokens_, window_, max_chunk_)) {
      throw std::invalid_argument("SlidingRingGeometry: ring of " + std::to_string(ring_tokens_) +
                                  " tokens cannot hold window " + std::to_string(window_) +
                                  " plus a chunk of " + std::to_string(max_chunk_));
    }
  }

  int Window() const { return window_; }
  int BlockSize() const { return bs_; }
  int RingTokens() const { return ring_tokens_; }
  int RingBlocks() const { return ring_tokens_ / bs_; }
  int MaxChunkRows() const { return r4d_ring_max_chunk(ring_tokens_, window_); }
  bool ChunkFits(int rows) const { return r4d_ring_fits(ring_tokens_, window_, rows); }
  // Entries of the shared block table: bt[kpos / block_size] for every key below max_ctx, plus
  // kTableSlackEntries. The attention kernels prefetch the block-table entries of a whole tile (up to
  // 48 keys = 3 blocks) one tile ahead, so the last tile of a context that ends mid-tile reads up to
  // 2 entries past the last block; those values are never used, but the read must stay inside the
  // allocation (a table that is an exact multiple of the allocator's page would fault).
  static constexpr int kTableSlackEntries = 4;
  int BlockTableEntries() const { return (max_ctx_ + bs_ - 1) / bs_ + kTableSlackEntries; }

  // T[i] = i % RingBlocks(): block i of the sequence lives in ring block i % RB.
  std::vector<int32_t> BuildBlockTable() const {
    std::vector<int32_t> t(static_cast<size_t>(BlockTableEntries()));
    for (size_t i = 0; i < t.size(); ++i) {
      t[i] = r4d_ring_block(static_cast<int64_t>(i), RingBlocks());
    }
    return t;
  }

  // Write slots of a chunk of `rows` keys at absolute positions [pos, pos + rows): slot t =
  // (pos + t) % ring_tokens, which is ((p / bs) % RB) * bs + p % bs, i.e. the slot the shared block
  // table makes the kernels read for key p -- so r4dx_kv_write_paged_fp8_hnd (slot = block_id *
  // block_size + offset) is used unchanged. Throws if the chunk does not fit the ring.
  void FillSlots(int64_t pos, int rows, int32_t* out) const {
    if (pos < 0 || rows < 0) throw std::invalid_argument("SlidingRingGeometry::FillSlots: negative pos / rows");
    if (!ChunkFits(rows)) {
      throw std::out_of_range("SlidingRingGeometry::FillSlots: a chunk of " + std::to_string(rows) +
                              " rows does not fit the ring (max " + std::to_string(MaxChunkRows()) + ")");
    }
    for (int t = 0; t < rows; ++t) out[t] = r4d_ring_slot(pos + t, ring_tokens_);
  }
  std::vector<int32_t> Slots(int64_t pos, int rows) const {
    std::vector<int32_t> s(static_cast<size_t>(rows));
    FillSlots(pos, rows, s.data());
    return s;
  }

 private:
  int window_, max_chunk_, bs_, max_ctx_, ring_tokens_ = 0;
};

}  // namespace r4dx::model::attention
