// tests/kernels/test_attn_decode.cpp -- r4d_attn_decode_h256_gqa6_fp8kv (via r4dx::core::r4d)
// against a CPU fp32 attention reference: bf16 q, K/V dequantized from the SAME fp8 cache the
// kernel reads (so the comparison isolates the attention algorithm, not the KV quantization --
// per the task brief). Builds a synthetic paged cache for 1 sequence of 300 tokens using r4dx's
// own r4dx_kv_write_paged_fp8_hnd kernel (doubles as an integration check of that kernel against
// a real r4d consumer). q_head -> kv_head mapping confirmed against
// third_party/libr4d/r4d_attn_decode_h256_gqa6.hip:103 (`qhead = kvh * GQA + hi`, contiguous
// grouping, not interleaved).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  r4d::AttnDims dims = r4d::GetAttnDims();
  const int head_dim = dims.head_dim, gqa = dims.gqa, block_size = dims.block_size;
  const int kv_heads = 4, q_heads = kv_heads * gqa;
  const int seqlen = 300;
  const int max_blocks = (seqlen + block_size - 1) / block_size;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::mt19937 rng(31);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> descale_dist(0.5f, 2.0f);

  // ---- synthetic KV: seqlen tokens, written into a paged fp8 cache via r4dx's own kernel -----
  std::vector<uint16_t> k_h(static_cast<size_t>(seqlen) * kv_heads * head_dim);
  std::vector<uint16_t> v_h(static_cast<size_t>(seqlen) * kv_heads * head_dim);
  for (auto& v : k_h) v = FloatToBf16(dist(rng));
  for (auto& v : v_h) v = FloatToBf16(dist(rng));
  std::vector<float> k_descale(kv_heads), v_descale(kv_heads);
  for (auto& d : k_descale) d = descale_dist(rng);
  for (auto& d : v_descale) d = descale_dist(rng);
  std::vector<int32_t> slot_mapping(seqlen);
  for (int t = 0; t < seqlen; ++t) slot_mapping[t] = t;  // block t/16, offset t%16, contiguous

  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * 2 * head_dim;
  const int64_t kv_block_stride = static_cast<int64_t>(kv_heads) * kv_head_stride;

  DeviceBuffer<uint16_t> k_d(k_h.size()), v_d(v_h.size());
  DeviceBuffer<int32_t> slot_d(seqlen);
  DeviceBuffer<float> kd_d(kv_heads), vd_d(kv_heads);
  DeviceBuffer<uint8_t> cache_d(static_cast<size_t>(max_blocks) * kv_heads * block_size * 2 *
                                 head_dim);
  k_d.CopyFromHost(k_h);
  v_d.CopyFromHost(v_h);
  slot_d.CopyFromHost(slot_mapping);
  kd_d.CopyFromHost(k_descale);
  vd_d.CopyFromHost(v_descale);
  cache_d.Zero();

  r4dx_kv_write_paged_fp8_hnd(reinterpret_cast<int64_t>(k_d.data()),
                               reinterpret_cast<int64_t>(v_d.data()),
                               reinterpret_cast<int64_t>(slot_d.data()),
                               reinterpret_cast<int64_t>(kd_d.data()),
                               reinterpret_cast<int64_t>(vd_d.data()),
                               reinterpret_cast<int64_t>(cache_d.data()), seqlen, kv_heads,
                               head_dim, block_size, kv_block_stride, kv_head_stride, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint8_t> cache_h = cache_d.CopyToHost();

  // ---- one decode query token ----------------------------------------------------------------
  std::vector<uint16_t> q_h(static_cast<size_t>(q_heads) * head_dim);
  for (auto& v : q_h) v = FloatToBf16(dist(rng));

  DeviceBuffer<uint16_t> q_d(q_h.size()), out_d(q_h.size());
  q_d.CopyFromHost(q_h);

  std::vector<int32_t> block_table = r4d::BuildContiguousBlockTable(1, max_blocks);
  DeviceBuffer<int32_t> block_table_d(block_table.size());
  block_table_d.CopyFromHost(block_table);
  std::vector<int32_t> seqused_k = {seqlen};
  DeviceBuffer<int32_t> seqused_k_d(1);
  seqused_k_d.CopyFromHost(seqused_k);

  R4DArgs a{};
  a.q = q_d.data();
  a.kv = cache_d.data();
  a.block_table = block_table_d.data();
  a.seqused_k = seqused_k_d.data();
  a.out = out_d.data();
  a.k_descale = kd_d.data();
  a.v_descale = vd_d.data();
  a.q_descale = nullptr;
  a.num_seqs = 1;
  a.q_len = 1;
  a.q_heads = q_heads;
  a.kv_heads = kv_heads;
  a.head_dim = head_dim;
  a.block_size = block_size;
  a.max_blocks = max_blocks;
  a.kv_block_stride = kv_block_stride;
  a.kv_head_stride = kv_head_stride;
  a.scale = scale;
  a.splits = 0;
  a.max_ctx = 512;

  int64_t scratch_bytes = r4d::AttnDecodeScratchBytes(a);
  DeviceBuffer<uint8_t> scratch_d(scratch_bytes > 0 ? static_cast<size_t>(scratch_bytes) : 1);
  a.scratch = scratch_bytes > 0 ? scratch_d.data() : nullptr;

  r4d::AttnDecodeFp8Kv(a, nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_h = out_d.CopyToHost();

  // ---- CPU fp32 reference: dequantize the SAME fp8 cache, plain softmax attention ------------
  std::vector<float> ref_out(static_cast<size_t>(q_heads) * head_dim, 0.0f);
  for (int qh = 0; qh < q_heads; ++qh) {
    const int kvh = qh / gqa;
    std::vector<float> logits(seqlen);
    float max_logit = -1e30f;
    for (int t = 0; t < seqlen; ++t) {
      int block_id = t / block_size, offset = t % block_size;
      int64_t base = static_cast<int64_t>(block_id) * kv_block_stride +
                      static_cast<int64_t>(kvh) * kv_head_stride +
                      static_cast<int64_t>(offset) * 2 * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        float kd = Fp8E4M3ToFloat(cache_h[base + d]) * k_descale[kvh];
        float qv = Bf16ToFloat(q_h[static_cast<size_t>(qh) * head_dim + d]);
        dot += qv * kd;
      }
      logits[t] = dot * scale;
      max_logit = std::max(max_logit, logits[t]);
    }
    double sum = 0.0;
    std::vector<double> probs(seqlen);
    for (int t = 0; t < seqlen; ++t) {
      probs[t] = std::exp(static_cast<double>(logits[t] - max_logit));
      sum += probs[t];
    }
    for (int t = 0; t < seqlen; ++t) probs[t] /= sum;
    for (int d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (int t = 0; t < seqlen; ++t) {
        int block_id = t / block_size, offset = t % block_size;
        int64_t base = static_cast<int64_t>(block_id) * kv_block_stride +
                        static_cast<int64_t>(kvh) * kv_head_stride +
                        static_cast<int64_t>(offset) * 2 * head_dim;
        float vd = Fp8E4M3ToFloat(cache_h[base + head_dim + d]) * v_descale[kvh];
        acc += probs[t] * vd;
      }
      ref_out[static_cast<size_t>(qh) * head_dim + d] = static_cast<float>(acc);
    }
  }

  double max_rel = 0.0;
  for (size_t i = 0; i < ref_out.size(); ++i) {
    float got = Bf16ToFloat(out_h[i]);
    float ref = ref_out[i];
    max_rel = std::max(max_rel, static_cast<double>(std::abs(got - ref) / std::max(1e-2f, std::abs(ref))));
  }
  std::printf("attn_decode_h256_gqa6_fp8kv max rel err=%.4e (seqlen=%d, q_heads=%d, kv_heads=%d)\n",
              max_rel, seqlen, q_heads, kv_heads);
  bool ok = max_rel < 5e-2;

  // ---- num_seqs==2: exercise Args::k_descale/v_descale's [num_seqs, kv_heads] broadcast --------
  // r4d_attn_decode_h256_gqa6.hip indexes descales as `seq * kv_heads + kvh` (a [num_seqs,
  // kv_heads] table), which the single-sequence check above cannot distinguish from the writer's
  // own [kv_heads] table (see r4dx::core::r4d::AttnDecodeFp8Kv's comment in r4d.hpp, and
  // kernels.h's kv_write doc). Two sequences with DIFFERENT descales each (not a trivially
  // identical replicate) makes a broken broadcast -- e.g. every sequence silently reading
  // sequence 0's descale row -- show up as a real numerical mismatch rather than an accidental
  // pass.
  {
    const int num_seqs2 = 2;
    const int seqlen2 = 180;  // deliberately different from seqlen (300) above
    const int max_blocks2 = (std::max(seqlen, seqlen2) + block_size - 1) / block_size;
    const int64_t kv_head_stride2 = static_cast<int64_t>(block_size) * 2 * head_dim;
    const int64_t kv_block_stride2 = static_cast<int64_t>(kv_heads) * kv_head_stride2;

    std::vector<uint16_t> k2_h(static_cast<size_t>(seqlen2) * kv_heads * head_dim);
    std::vector<uint16_t> v2_h(static_cast<size_t>(seqlen2) * kv_heads * head_dim);
    for (auto& v : k2_h) v = FloatToBf16(dist(rng));
    for (auto& v : v2_h) v = FloatToBf16(dist(rng));
    std::vector<float> k_descale2(kv_heads), v_descale2(kv_heads);
    for (auto& d : k_descale2) d = descale_dist(rng);
    for (auto& d : v_descale2) d = descale_dist(rng);

    // Combined cache: sequence 0's blocks are [0, max_blocks2), sequence 1's are
    // [max_blocks2, 2*max_blocks2), matching r4d::BuildContiguousBlockTable(2, max_blocks2).
    DeviceBuffer<uint8_t> cache2_d(static_cast<size_t>(num_seqs2) * max_blocks2 * kv_heads *
                                    block_size * 2 * head_dim);
    cache2_d.Zero();

    DeviceBuffer<uint16_t> k_seq0_d(k_h.size()), v_seq0_d(v_h.size());
    DeviceBuffer<int32_t> slot_seq0_d(seqlen);
    DeviceBuffer<float> kd_seq0_d(kv_heads), vd_seq0_d(kv_heads);
    k_seq0_d.CopyFromHost(k_h);
    v_seq0_d.CopyFromHost(v_h);
    slot_seq0_d.CopyFromHost(slot_mapping);  // slot t, block base 0 -- same mapping as the single-seq case
    kd_seq0_d.CopyFromHost(k_descale);
    vd_seq0_d.CopyFromHost(v_descale);
    r4dx_kv_write_paged_fp8_hnd(
        reinterpret_cast<int64_t>(k_seq0_d.data()), reinterpret_cast<int64_t>(v_seq0_d.data()),
        reinterpret_cast<int64_t>(slot_seq0_d.data()), reinterpret_cast<int64_t>(kd_seq0_d.data()),
        reinterpret_cast<int64_t>(vd_seq0_d.data()), reinterpret_cast<int64_t>(cache2_d.data()),
        seqlen, kv_heads, head_dim, block_size, kv_block_stride2, kv_head_stride2, 0);

    const int64_t seq1_block_base_slot = static_cast<int64_t>(max_blocks2) * block_size;
    std::vector<int32_t> slot_mapping2(seqlen2);
    for (int t = 0; t < seqlen2; ++t) slot_mapping2[t] = static_cast<int32_t>(seq1_block_base_slot) + t;
    DeviceBuffer<uint16_t> k2_d(k2_h.size()), v2_d(v2_h.size());
    DeviceBuffer<int32_t> slot2_d(seqlen2);
    DeviceBuffer<float> kd2_d(kv_heads), vd2_d(kv_heads);
    k2_d.CopyFromHost(k2_h);
    v2_d.CopyFromHost(v2_h);
    slot2_d.CopyFromHost(slot_mapping2);
    kd2_d.CopyFromHost(k_descale2);
    vd2_d.CopyFromHost(v_descale2);
    r4dx_kv_write_paged_fp8_hnd(
        reinterpret_cast<int64_t>(k2_d.data()), reinterpret_cast<int64_t>(v2_d.data()),
        reinterpret_cast<int64_t>(slot2_d.data()), reinterpret_cast<int64_t>(kd2_d.data()),
        reinterpret_cast<int64_t>(vd2_d.data()), reinterpret_cast<int64_t>(cache2_d.data()),
        seqlen2, kv_heads, head_dim, block_size, kv_block_stride2, kv_head_stride2, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> cache2_h = cache2_d.CopyToHost();

    // Broadcast each sequence's own [kv_heads] descale row into the [num_seqs, kv_heads] table
    // R4DArgs expects.
    std::vector<float> k_descale_bcast(static_cast<size_t>(num_seqs2) * kv_heads);
    std::vector<float> v_descale_bcast(static_cast<size_t>(num_seqs2) * kv_heads);
    for (int h = 0; h < kv_heads; ++h) {
      k_descale_bcast[0 * static_cast<size_t>(kv_heads) + h] = k_descale[h];
      v_descale_bcast[0 * static_cast<size_t>(kv_heads) + h] = v_descale[h];
      k_descale_bcast[1 * static_cast<size_t>(kv_heads) + h] = k_descale2[h];
      v_descale_bcast[1 * static_cast<size_t>(kv_heads) + h] = v_descale2[h];
    }
    DeviceBuffer<float> kd_bcast_d(k_descale_bcast.size()), vd_bcast_d(v_descale_bcast.size());
    kd_bcast_d.CopyFromHost(k_descale_bcast);
    vd_bcast_d.CopyFromHost(v_descale_bcast);

    std::vector<uint16_t> q2_h(static_cast<size_t>(num_seqs2) * q_heads * head_dim);
    for (auto& v : q2_h) v = FloatToBf16(dist(rng));
    DeviceBuffer<uint16_t> q2_d(q2_h.size()), out2_d(q2_h.size());
    q2_d.CopyFromHost(q2_h);

    std::vector<int32_t> block_table2 = r4d::BuildContiguousBlockTable(num_seqs2, max_blocks2);
    DeviceBuffer<int32_t> block_table2_d(block_table2.size());
    block_table2_d.CopyFromHost(block_table2);
    std::vector<int32_t> seqused_k2 = {seqlen, seqlen2};
    DeviceBuffer<int32_t> seqused_k2_d(num_seqs2);
    seqused_k2_d.CopyFromHost(seqused_k2);

    R4DArgs a2{};
    a2.q = q2_d.data();
    a2.kv = cache2_d.data();
    a2.block_table = block_table2_d.data();
    a2.seqused_k = seqused_k2_d.data();
    a2.out = out2_d.data();
    a2.k_descale = kd_bcast_d.data();
    a2.v_descale = vd_bcast_d.data();
    a2.q_descale = nullptr;
    a2.num_seqs = num_seqs2;
    a2.q_len = 1;
    a2.q_heads = q_heads;
    a2.kv_heads = kv_heads;
    a2.head_dim = head_dim;
    a2.block_size = block_size;
    a2.max_blocks = max_blocks2;
    a2.kv_block_stride = kv_block_stride2;
    a2.kv_head_stride = kv_head_stride2;
    a2.scale = scale;
    a2.splits = 0;
    a2.max_ctx = 512;

    int64_t scratch_bytes2 = r4d::AttnDecodeScratchBytes(a2);
    DeviceBuffer<uint8_t> scratch2_d(scratch_bytes2 > 0 ? static_cast<size_t>(scratch_bytes2) : 1);
    a2.scratch = scratch_bytes2 > 0 ? scratch2_d.data() : nullptr;

    r4d::AttnDecodeFp8Kv(a2, nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> out2_h = out2_d.CopyToHost();

    auto RefAttnForSeq = [&](int this_seqlen, int64_t block_base_slot, const uint16_t* q_ptr,
                              const std::vector<float>& kdesc, const std::vector<float>& vdesc) {
      std::vector<float> out(static_cast<size_t>(q_heads) * head_dim, 0.0f);
      for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / gqa;
        std::vector<float> logits(this_seqlen);
        float max_logit = -1e30f;
        for (int t = 0; t < this_seqlen; ++t) {
          int slot = static_cast<int>(block_base_slot) + t;
          int block_id = slot / block_size, offset = slot % block_size;
          int64_t base = static_cast<int64_t>(block_id) * kv_block_stride2 +
                          static_cast<int64_t>(kvh) * kv_head_stride2 +
                          static_cast<int64_t>(offset) * 2 * head_dim;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float kd = Fp8E4M3ToFloat(cache2_h[base + d]) * kdesc[kvh];
            float qv = Bf16ToFloat(q_ptr[static_cast<size_t>(qh) * head_dim + d]);
            dot += qv * kd;
          }
          logits[t] = dot * scale;
          max_logit = std::max(max_logit, logits[t]);
        }
        double sum = 0.0;
        std::vector<double> probs(this_seqlen);
        for (int t = 0; t < this_seqlen; ++t) {
          probs[t] = std::exp(static_cast<double>(logits[t] - max_logit));
          sum += probs[t];
        }
        for (int t = 0; t < this_seqlen; ++t) probs[t] /= sum;
        for (int d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int t = 0; t < this_seqlen; ++t) {
            int slot = static_cast<int>(block_base_slot) + t;
            int block_id = slot / block_size, offset = slot % block_size;
            int64_t base = static_cast<int64_t>(block_id) * kv_block_stride2 +
                            static_cast<int64_t>(kvh) * kv_head_stride2 +
                            static_cast<int64_t>(offset) * 2 * head_dim;
            float vd = Fp8E4M3ToFloat(cache2_h[base + head_dim + d]) * vdesc[kvh];
            acc += probs[t] * vd;
          }
          out[static_cast<size_t>(qh) * head_dim + d] = static_cast<float>(acc);
        }
      }
      return out;
    };

    std::vector<float> ref0 = RefAttnForSeq(seqlen, 0, q2_h.data(), k_descale, v_descale);
    std::vector<float> ref1 =
        RefAttnForSeq(seqlen2, seq1_block_base_slot, q2_h.data() + static_cast<size_t>(q_heads) * head_dim,
                      k_descale2, v_descale2);

    double max_rel2 = 0.0;
    for (size_t i = 0; i < ref0.size(); ++i) {
      float got = Bf16ToFloat(out2_h[i]);
      max_rel2 = std::max(max_rel2,
                           static_cast<double>(std::abs(got - ref0[i]) / std::max(1e-2f, std::abs(ref0[i]))));
    }
    for (size_t i = 0; i < ref1.size(); ++i) {
      float got = Bf16ToFloat(out2_h[static_cast<size_t>(q_heads) * head_dim + i]);
      max_rel2 = std::max(max_rel2,
                           static_cast<double>(std::abs(got - ref1[i]) / std::max(1e-2f, std::abs(ref1[i]))));
    }
    std::printf(
        "attn_decode_h256_gqa6_fp8kv num_seqs=2 (seqlens=%d,%d) descale-broadcast max rel err=%.4e\n",
        seqlen, seqlen2, max_rel2);
    ok = ok && (max_rel2 < 5e-2);
  }

  // ---- a verify window (q_len=T) row vs the q_len=1 launch at that row's own position -----------
  // Speculative decode is exact only if every row of a verify window rounds exactly like the plain
  // decode step at its position (docs/mtp.md, "Sampled rounds are bit-exact"), so this is a BIT
  // comparison. Each window straddles a 16-key tile boundary where the split-KV merge's segment
  // count reaches a multiple of its 4 accumulation chains (ctx 48 -> 49, 112 -> 113, ...): the
  // window's first 8 rows see one segment fewer than its last row. Before the merge counted each
  // row's own segments (r4d_attn_splitkv_combine_kernel), those rows went through a different chain
  // split than their own decode step and differed in their last bits. Same cache, same max_ctx,
  // and no window crosses a change of tiles-per-segment (ctx 256 here), which reshapes the segments
  // themselves and is not this check's subject.
  const int T = 10;  // the widest window this kernel serves (q_len * gqa <= 64)
  std::vector<uint16_t> qw_h(static_cast<size_t>(T) * q_heads * head_dim);
  for (auto& v : qw_h) v = FloatToBf16(dist(rng));
  DeviceBuffer<uint16_t> qw_d(qw_h.size()), outw_d(qw_h.size()), out1_d(qw_h.size());
  qw_d.CopyFromHost(qw_h);
  DeviceBuffer<int32_t> ctx_d(1);
  // The window at positions p..p+T-1 over `cache`'s 300-token cache (a's block table, descales and
  // max_ctx) against the T q_len=1 launches at each row's own position; returns the number of bf16
  // outputs that differ.
  auto WindowVsSteps = [&](const DeviceBuffer<uint8_t>& cache, int p) -> size_t {
    R4DArgs w = a;
    w.kv = cache.data();
    w.q = qw_d.data();
    w.seqused_k = ctx_d.data();
    w.q_len = T;
    const std::vector<int32_t> ctx_window = {p + T};
    ctx_d.CopyFromHost(ctx_window);
    const int64_t wbytes = r4d::AttnDecodeScratchBytes(w);
    DeviceBuffer<uint8_t> wscratch_d(wbytes > 0 ? static_cast<size_t>(wbytes) : 1);
    w.scratch = wbytes > 0 ? wscratch_d.data() : nullptr;
    w.out = outw_d.data();
    r4d::AttnDecodeFp8Kv(w, nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint16_t> outw_h = outw_d.CopyToHost();

    for (int i = 0; i < T; ++i) {
      R4DArgs s = a;
      const size_t row_off = static_cast<size_t>(i) * q_heads * head_dim;
      s.kv = cache.data();
      s.q = qw_d.data() + row_off;
      s.out = out1_d.data() + row_off;
      s.seqused_k = ctx_d.data();
      s.q_len = 1;
      const std::vector<int32_t> ctx_row = {p + i + 1};
      ctx_d.CopyFromHost(ctx_row);
      r4d::AttnDecodeFp8Kv(s, nullptr);  // a.scratch is sized for q_len=1
      R4DX_HIP_CHECK(hipDeviceSynchronize());
    }
    const std::vector<uint16_t> out1_h = out1_d.CopyToHost();
    size_t differing = 0;
    for (size_t e = 0; e < out1_h.size(); ++e) differing += (out1_h[e] != outw_h[e]);
    return differing;
  };
  for (const int p : {40, 104, 168, 232}) {
    const size_t differing = WindowVsSteps(cache_d, p);
    std::printf("attn_decode verify window at positions %d..%d (ctx %d) vs q_len=1 at each row's "
                "own position: %zu/%zu bf16 outputs differ\n",
                p, p + T - 1, p + T, differing, qw_h.size());
    ok = ok && differing == 0;
  }

  // ---- the same comparison where a segment's second tile rescales one row ----------------------
  // At max_ctx 512 the merge has 16 segments, so past context 256 each spans 2 tiles, and a row's
  // lazy rescale (its tile max above m_ref + GROW, 14 octaves on the f16 P path) can fire on a
  // segment's SECOND tile. That decision used to be the whole wave's: one row's jump rescaled every
  // row of its wave whose tile max had risen at all. A q_len=1 launch's wave holds the heads of one
  // position, a verify window's the heads of 2-3, so the other positions' rows rescaled in the
  // window and not in their own decode step. Here one key in the second tile of each of segments
  // 5-8 is planted along the query of head 0 of every KV group at window position 1, 3, 6 or 8 --
  // a row in each of the 4 waves -- so that those rows' max jumps by ~30 octaves there; nothing in
  // the uniform(-1,1) data does. Window 280..289: it and every row's own launch have 2 tiles per
  // segment, so the partition is the same and only the rescale decision can differ.
  {
    const int planted_key[] = {176, 208, 240, 272};  // 2nd tile of segments 5, 6, 7, 8
    const int target_pos[] = {1, 3, 6, 8};            // window rows 6, 18, 36, 48: waves 0-3
    const float plant_scale = 4.0f;                   // score ~ 4 |q|^2 / 16 * log2(e) ~ 30 octaves
    const int n_plant = 4;
    std::vector<uint16_t> kp_h(static_cast<size_t>(n_plant) * kv_heads * head_dim);
    std::vector<uint16_t> vp_h(kp_h.size());
    std::vector<int32_t> slot_p(n_plant);
    for (int j = 0; j < n_plant; ++j) {
      slot_p[j] = planted_key[j];
      for (int kvh = 0; kvh < kv_heads; ++kvh) {
        const size_t dst = (static_cast<size_t>(j) * kv_heads + kvh) * head_dim;
        const size_t src_v = (static_cast<size_t>(planted_key[j]) * kv_heads + kvh) * head_dim;
        const size_t src_q = (static_cast<size_t>(target_pos[j]) * q_heads + kvh * gqa) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          kp_h[dst + d] = FloatToBf16(plant_scale * Bf16ToFloat(qw_h[src_q + d]));
          vp_h[dst + d] = v_h[src_v + d];  // the token's own V
        }
      }
    }
    DeviceBuffer<uint8_t> planted_d(cache_h.size());
    planted_d.CopyFromHost(cache_h);
    DeviceBuffer<uint16_t> kp_d(kp_h.size()), vp_d(vp_h.size());
    DeviceBuffer<int32_t> slotp_d(n_plant);
    kp_d.CopyFromHost(kp_h);
    vp_d.CopyFromHost(vp_h);
    slotp_d.CopyFromHost(slot_p);
    r4dx_kv_write_paged_fp8_hnd(reinterpret_cast<int64_t>(kp_d.data()),
                                 reinterpret_cast<int64_t>(vp_d.data()),
                                 reinterpret_cast<int64_t>(slotp_d.data()),
                                 reinterpret_cast<int64_t>(kd_d.data()),
                                 reinterpret_cast<int64_t>(vd_d.data()),
                                 reinterpret_cast<int64_t>(planted_d.data()), n_plant, kv_heads,
                                 head_dim, block_size, kv_block_stride, kv_head_stride, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const int p = 280;
    const size_t differing = WindowVsSteps(planted_d, p);
    std::printf("attn_decode verify window at positions %d..%d (ctx %d, 2 tiles per segment, one "
                "row per wave rescaled on a second tile) vs q_len=1: %zu/%zu bf16 outputs differ\n",
                p, p + T - 1, p + T, differing, qw_h.size());
    ok = ok && differing == 0;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
