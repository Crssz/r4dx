// tests/kernels/gemma_attn_fixture.hpp -- shared setup of the Gemma 4 sliding-attention GPU tests
// (test_attn_prefill_gqa2, test_attn_decode_gqa2; docs/gemma4-plan.md 3.3, task M1-32): a random bf16
// KV history written through the production fp8 write kernel into a sliding RING (96 blocks, the
// shared block table T[i] = i % RB) or a contiguous cache, a random query chunk, and the stage-0
// reference kernel (r4dx_gemma_attn_ref_fp8kv, validated against a CPU reference by
// test_attn_ref_gemma) run on the SAME cache bytes as the oracle for the libr4d windowed kernels.
//
// Geometry: Gemma 4's sliding layer -- 16 q heads, 8 KV heads (gqa 2), head_dim 256, block 16, scale 1.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_ref.h"
#include "r4dx/model/attention/sliding_ring.hpp"

namespace gemma_attn {

using namespace r4dx::core;
using r4dx::model::attention::SlidingRingGeometry;

constexpr int kQHeads = 16, kKvHeads = 8, kHeadDim = 256, kBs = 16;
constexpr int kRingTokens = 1536;

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Shape {
  int ctx, q_len, window;
  bool ring;                       // sliding ring + shared table, else a contiguous cache
  bool ext;                        // an image block inside the chunk (prefill only)
};

class Fixture {
 public:
  Fixture(const Shape& s, std::mt19937& rng) : s_(s) {
    const float sigma = std::sqrt(1.5f / std::sqrt(static_cast<float>(kHeadDim)));
    std::normal_distribution<float> nd(0.0f, sigma);
    const size_t row = static_cast<size_t>(kKvHeads) * kHeadDim;
    std::vector<uint16_t> k_hist(static_cast<size_t>(s.ctx) * row), v_hist(k_hist.size());
    for (auto& v : k_hist) v = FloatToBf16(nd(rng));
    for (auto& v : v_hist) v = FloatToBf16(nd(rng));
    std::vector<float> kd(kKvHeads), vd(kKvHeads);
    for (int h = 0; h < kKvHeads; ++h) { kd[h] = 0.01f * (1.0f + 0.15f * h); vd[h] = 0.012f * (1.0f + 0.1f * h); }
    kd_.Resize(kKvHeads); vd_.Resize(kKvHeads);
    kd_.CopyFromHost(kd); vd_.CopyFromHost(vd);

    if (s.ring) {
      geo_ = std::make_unique<SlidingRingGeometry>(1024, 288, kBs, s.ctx, kRingTokens);
      table_h_ = geo_->BuildBlockTable();
      blocks_ = geo_->RingBlocks();
    } else {
      blocks_ = (s.ctx + kBs - 1) / kBs;
      table_h_.resize(blocks_ + SlidingRingGeometry::kTableSlackEntries);
      for (size_t i = 0; i < table_h_.size(); ++i) table_h_[i] = static_cast<int32_t>(std::min<size_t>(i, blocks_ - 1));
    }
    head_stride_ = static_cast<int64_t>(kBs) * 2 * kHeadDim;
    block_stride_ = static_cast<int64_t>(kKvHeads) * head_stride_;
    cache_.Resize(static_cast<size_t>(blocks_) * block_stride_);
    cache_.Zero();   // the windowed kernels stage tiles below the window: finite slots only
    for (int pos = 0; pos < s.ctx; pos += 256) {
      const int T = std::min(256, s.ctx - pos);
      DeviceBuffer<uint16_t> kn(static_cast<size_t>(T) * row), vn(kn.size());
      DeviceBuffer<int32_t> sm(T);
      kn.CopyFromHost(k_hist.data() + static_cast<size_t>(pos) * row, kn.size());
      vn.CopyFromHost(v_hist.data() + static_cast<size_t>(pos) * row, vn.size());
      std::vector<int32_t> slots(T);
      for (int t = 0; t < T; ++t) slots[t] = s.ring ? (pos + t) % kRingTokens : pos + t;
      sm.CopyFromHost(slots);
      r4dx_kv_write_paged_fp8_hnd(P(kn.data()), P(vn.data()), P(sm.data()), P(kd_.data()), P(vd_.data()),
                                   P(cache_.data()), T, kKvHeads, kHeadDim, kBs, block_stride_,
                                   head_stride_, 0);
    }
    q_h_.resize(static_cast<size_t>(s.q_len) * kQHeads * kHeadDim);
    for (auto& v : q_h_) v = FloatToBf16(nd(rng));
    ext_h_.assign(s.q_len, -1);
    if (s.ext) {
      const int a = std::min(10, s.q_len - 1), b = std::min(50, s.q_len - 1);
      for (int t = a; t <= b; ++t) ext_h_[t] = s.ctx - s.q_len + b;
    }
    q_.Resize(q_h_.size()); out_ref_.Resize(q_h_.size()); out_k_.Resize(q_h_.size());
    q_.CopyFromHost(q_h_);
    table_.Resize(table_h_.size()); table_.CopyFromHost(table_h_);
    ext_.Resize(ext_h_.size()); ext_.CopyFromHost(ext_h_);
    seqused_.Resize(1); seqused_.CopyFromHost(std::vector<int32_t>{s.ctx});
    R4DX_HIP_CHECK(hipDeviceSynchronize());
  }

  // The oracle: the stage-0 kernel over the same cache bytes. Returns its output (bf16 bits).
  std::vector<uint16_t> Reference() {
    R4dxGemmaAttnRefArgs a{};
    a.q = P(q_.data()); a.kv = P(cache_.data()); a.block_table = P(table_.data()); a.out = P(out_ref_.data());
    a.k_descale = P(kd_.data()); a.v_descale = P(vd_.data()); a.klimit_ext = s_.ext ? P(ext_.data()) : 0;
    a.q_len = s_.q_len; a.q_heads = kQHeads; a.kv_heads = kKvHeads; a.head_dim = kHeadDim; a.block_size = kBs;
    a.ctx = s_.ctx; a.window = s_.window; a.scale = 1.0f;
    a.kv_block_stride = block_stride_; a.kv_head_stride = head_stride_;
    out_ref_.Zero();
    const int rc = r4dx_gemma_attn_ref_fp8kv(&a, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    if (rc != 0) throw std::runtime_error("r4dx_gemma_attn_ref_fp8kv returned " + std::to_string(rc));
    return out_ref_.CopyToHost();
  }

  // The libr4d windowed arguments for this fixture. q_len_override / q_offset let the decode test run
  // a verify window's rows one at a time (a q_len = 1 launch at row t's own position needs ctx' =
  // ctx - q_len + t + 1 and the query row t).
  R4DArgsW Args(uint16_t* out, const uint16_t* q, int q_len, int ctx, const int32_t* seqused) {
    R4DArgsW a{};
    a.q = q; a.kv = cache_.data(); a.block_table = table_.data(); a.seqused_k = seqused; a.out = out;
    a.k_descale = kd_.data(); a.v_descale = vd_.data(); a.q_descale = nullptr; a.scratch = nullptr;
    a.num_seqs = 1; a.q_len = q_len; a.q_heads = kQHeads; a.kv_heads = kKvHeads; a.head_dim = kHeadDim;
    a.block_size = kBs; a.max_blocks = static_cast<int>(table_h_.size());
    a.kv_block_stride = block_stride_; a.kv_head_stride = head_stride_;
    a.scale = 1.0f; a.splits = 0; a.max_ctx = ctx;
    a.window = s_.window; a.klimit_ext = s_.ext ? ext_.data() : nullptr;
    return a;
  }

  const Shape& shape() const { return s_; }
  uint16_t* Q() { return q_.data(); }
  uint16_t* OutK() { return out_k_.data(); }
  const int32_t* Seqused() const { return seqused_.data(); }
  std::vector<uint16_t> OutKHost() const { return out_k_.CopyToHost(); }
  void ZeroOutK() { out_k_.Zero(); }

 private:
  Shape s_;
  std::unique_ptr<SlidingRingGeometry> geo_;
  std::vector<int32_t> table_h_, ext_h_;
  std::vector<uint16_t> q_h_;
  int blocks_ = 0;
  int64_t head_stride_ = 0, block_stride_ = 0;
  DeviceBuffer<uint8_t> cache_;
  DeviceBuffer<float> kd_, vd_;
  DeviceBuffer<uint16_t> q_, out_ref_, out_k_;
  DeviceBuffer<int32_t> table_, ext_, seqused_;
};

struct Err {
  double norm_rel = 0, max_abs = 0, max_ref = 0;
};
// Rows [row0, row0 + rows) of two [*, q_heads, head_dim] bf16 outputs.
inline Err Compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref, int row0, int rows) {
  Err e;
  double num = 0, den = 0;
  for (size_t i = static_cast<size_t>(row0) * kQHeads * kHeadDim;
       i < static_cast<size_t>(row0 + rows) * kQHeads * kHeadDim; ++i) {
    const double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
    e.max_abs = std::max(e.max_abs, std::abs(g - r));
    e.max_ref = std::max(e.max_ref, std::abs(r));
    num += (g - r) * (g - r);
    den += r * r;
  }
  e.norm_rel = std::sqrt(num / std::max(den, 1e-30));
  return e;
}

// libr4d runs f16 operands (P and V in f16, fp8 K widened exactly): ~2.3e-3 relative against fp32.
inline bool Within(const Err& e) { return e.norm_rel < 8e-3 && e.max_abs < 3e-2 * std::max(e.max_ref, 1e-3); }

}  // namespace gemma_attn
