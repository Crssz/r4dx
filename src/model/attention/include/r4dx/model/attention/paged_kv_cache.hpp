// r4dx::model::attention::PagedKvCache -- a single-sequence fp8 e4m3 paged KV cache, sized and
// laid out exactly as R4DArgs.kv expects (docs/architecture.md "fp8 KV paging";
// third_party/libr4d/r4d.h:44-58): (num_blocks, kv_heads, block_size, 2*head_dim), K then V per
// slot. This component (src/model/attention/**) owns its definition per the task brief ("a
// PagedKvCache object you define (num_blocks, block_table per sequence, slot mapping)").
//
// SCOPE: single sequence only. Blocks are allocated contiguously
// (r4dx::core::r4d::BuildContiguousBlockTable(1, max_blocks)), so slot(t) == t for the token
// written at position t -- the same convention tests/kernels/test_attn_decode.cpp's single-
// sequence case and this component's own golden test rely on. A multi-sequence batching layer
// (paging multiple sequences into one cache, per-sequence block tables, eviction) is the
// serving-layer's job, not this component's (out of scope for a single decoder layer's math).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/r4d.hpp"

namespace r4dx::model::attention {

class PagedKvCache {
 public:
  PagedKvCache(int kv_heads, int head_dim, int block_size, int max_context_tokens)
      : kv_heads_(kv_heads), head_dim_(head_dim), block_size_(block_size) {
    if (kv_heads_ <= 0 || head_dim_ <= 0 || block_size_ <= 0 || max_context_tokens <= 0) {
      throw std::invalid_argument("PagedKvCache: all dims must be positive");
    }
    max_blocks_ = (max_context_tokens + block_size_ - 1) / block_size_;
    const size_t elems =
        static_cast<size_t>(max_blocks_) * kv_heads_ * block_size_ * 2 * head_dim_;
    cache_.Resize(elems);
    cache_.Zero();
    std::vector<int32_t> table = r4dx::core::r4d::BuildContiguousBlockTable(1, max_blocks_);
    block_table_.Resize(table.size());
    block_table_.CopyFromHost(table);
  }

  uint8_t* Data() { return cache_.data(); }
  const int32_t* BlockTable() const { return block_table_.data(); }
  int MaxBlocks() const { return max_blocks_; }
  int BlockSize() const { return block_size_; }
  int KvHeads() const { return kv_heads_; }
  int HeadDim() const { return head_dim_; }
  int64_t KvHeadStride() const { return static_cast<int64_t>(block_size_) * 2 * head_dim_; }
  int64_t KvBlockStride() const { return static_cast<int64_t>(kv_heads_) * KvHeadStride(); }
  int CapacityTokens() const { return max_blocks_ * block_size_; }

  // Throws if [start_pos, start_pos+T) would run past the cache's allocated capacity -- callers
  // must size the cache (max_context_tokens above) for the whole sequence up front, matching
  // r4dx::core::r4d::BuildContiguousBlockTable's contiguous-per-sequence block layout (no dynamic
  // growth once constructed).
  void CheckCapacity(int start_pos, int T) const {
    if (start_pos < 0 || T < 0 || start_pos + T > CapacityTokens()) {
      throw std::out_of_range(
          "PagedKvCache: start_pos+T exceeds the max_context_tokens this cache was sized for");
    }
  }

 private:
  int kv_heads_, head_dim_, block_size_, max_blocks_ = 0;
  r4dx::core::DeviceBuffer<uint8_t> cache_;
  r4dx::core::DeviceBuffer<int32_t> block_table_;
};

}  // namespace r4dx::model::attention
