// r4dx::model::attention::SlidingKvCache -- one sliding layer's fp8 e4m3 KV ring, laid out exactly as
// R4DArgs.kv expects ((num_blocks, kv_heads, block_size, 2*head_dim), K then V per slot; same as
// PagedKvCache) but with only RingBlocks() blocks, addressed through the ONE int32 block table every
// sliding layer shares (docs/gemma4-plan.md 3.3). The geometry arithmetic is SlidingRingGeometry
// (sliding_ring.hpp, CPU-testable).
//
// Storage is ZERO-INITIALISED and must stay so at construction: the windowed kernels stage tiles that
// start below the window (up to TILE - 1 keys), whose slots alias newer keys; any finite fp8 value is
// harmless there (masked P = 0 times it), a NaN pattern would not be.
//
// Checkpoint/rollback (speculative decode): a verify window of at most max_chunk_rows rows leaves
// every key a rollback still needs intact (the ring holds window + chunk), so a rolled-back position
// is simply overwritten; SaveCheckpoint-style whole-ring copies (D2D, 252 MB across 40 rings, ~1 ms)
// are CopyFrom().
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/model/attention/sliding_ring.hpp"

namespace r4dx::model::attention {

class SlidingKvCache {
 public:
  SlidingKvCache(int kv_heads, int head_dim, const SlidingRingGeometry& geometry)
      : kv_heads_(kv_heads), head_dim_(head_dim), geo_(geometry) {
    if (kv_heads_ <= 0 || head_dim_ <= 0) {
      throw std::invalid_argument("SlidingKvCache: kv_heads and head_dim must be positive");
    }
    const size_t bytes = static_cast<size_t>(geo_.RingBlocks()) * kv_heads_ * geo_.BlockSize() * 2 *
                         static_cast<size_t>(head_dim_);
    cache_.Resize(bytes);
    cache_.Zero();
  }

  uint8_t* Data() { return cache_.data(); }
  const uint8_t* Data() const { return cache_.data(); }
  const SlidingRingGeometry& Geometry() const { return geo_; }
  int KvHeads() const { return kv_heads_; }
  int HeadDim() const { return head_dim_; }
  int64_t KvHeadStride() const { return static_cast<int64_t>(geo_.BlockSize()) * 2 * head_dim_; }
  int64_t KvBlockStride() const { return static_cast<int64_t>(kv_heads_) * KvHeadStride(); }
  size_t Bytes() const { return cache_.bytes(); }

  // Checks that a chunk of `rows` keys fits (see SlidingRingGeometry::FillSlots) and uploads its
  // write slots into `slots` (device int32 [rows]).
  void UploadSlots(int64_t pos, int rows, r4dx::core::DeviceBuffer<int32_t>* slots) const {
    const std::vector<int32_t> s = geo_.Slots(pos, rows);
    if (slots->size() < s.size()) slots->Resize(s.size());
    slots->CopyFromHost(s);
  }

  // Whole-ring D2D copy (a checkpoint / restore); both rings must have the same geometry.
  void CopyFrom(const SlidingKvCache& other) {
    if (other.cache_.size() != cache_.size() || other.kv_heads_ != kv_heads_ ||
        other.head_dim_ != head_dim_) {
      throw std::invalid_argument("SlidingKvCache::CopyFrom: geometry mismatch");
    }
    cache_.CopyFromDevice(other.cache_, cache_.size());
  }

 private:
  int kv_heads_, head_dim_;
  SlidingRingGeometry geo_;
  r4dx::core::DeviceBuffer<uint8_t> cache_;
};

// The shared block table of every sliding layer: int32 [BlockTableEntries()], T[i] = i % RingBlocks().
class SlidingBlockTable {
 public:
  explicit SlidingBlockTable(const SlidingRingGeometry& geometry) {
    const std::vector<int32_t> t = geometry.BuildBlockTable();
    table_.Resize(t.size());
    table_.CopyFromHost(t);
  }
  const int32_t* Data() const { return table_.data(); }
  int Entries() const { return static_cast<int>(table_.size()); }

 private:
  r4dx::core::DeviceBuffer<int32_t> table_;
};

}  // namespace r4dx::model::attention
