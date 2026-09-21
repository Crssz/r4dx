#include "vision_tower.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"
#include "vision_index.h"

namespace r4dx::vision {

namespace {

// r4d_gemm_bf16_nt_m64's own M cap (r4d_gemm_bf16_nt_m64_max_m()); a wider call is several of
// them back to back, exactly as src/model/linear.cpp's ApplyLinear does for the text side.
constexpr int64_t kMaxGemmM = 64;

struct GemmTuning {
  int WV, SK, MB;
};

// The tower's K values are not the text side's, so src/model/linear.cpp's PickTuning table (and
// its FallbackTuning, which requires K % 512 == 0) does not apply: hidden_size 1152 and
// merger_hidden 4608 are multiples of 512, but patch_dim 1536 is not, and intermediate_size 4304
// is 16 * 269 with 269 PRIME -- so mlp.linear_fc2's K admits SK = 1 and nothing else.
//
// The kernel's only K constraint is `K % (SK * 16) == 0`; WV * SK * 32 must stay under the 1024
// thread cap and WV * SK * 1 KiB under the 64 KiB LDS cap. SK is therefore the largest legal split
// (4 wherever K allows it, 1 for mlp.linear_fc2).
//
// WV = 4 / MB = 1 is measured, not assumed -- tool_vision_bench, tower only, best of 3, encode ms
// for 448 / 1024 / 1536-pixel square images:
//     WV=4  MB=1   28.1 / 151.2 / 399.8   <- shipped
//     WV=2  MB=1   29.0 / 156.4 / 409.7
//     WV=8  MB=1   30.3 / 161.6 / 421.0
//     WV=4  MB=4   35.0 / 183.3 / 464.1
// MB > 1 loses because it amortizes the weight fragment over more row tiles at the cost of grid.y
// blocks, and at M <= 64 (mtiles <= 4) this tower is block-starved long before it is
// weight-bandwidth-bound: N = 1152 is only 72 column tiles, so grid.x is 18 at WV = 4.
GemmTuning PickTuning(int64_t K) {
  for (int sk : {4, 2, 1}) {
    if (K % (static_cast<int64_t>(sk) * 16) == 0) return GemmTuning{/*WV=*/4, sk, /*MB=*/1};
  }
  throw std::runtime_error("r4dx::vision: K=" + std::to_string(K) +
                            " is not a multiple of 16 -- r4d_gemm_bf16_nt_m64 cannot serve it");
}

int64_t AlignUp16(int64_t v) { return (v + 15) & ~static_cast<int64_t>(15); }

}  // namespace

// Every per-encode buffer, in one place so PlanScratchBytes and EnsureScratch cannot drift apart.
// `P`-sized buffers are the residual stream and the attention operands (the attention kernel needs
// every row of q/k/v at once, so those cannot be chunked); everything else is `C`-sized, where C is
// the row chunk.
struct VisionTower::Scratch {
  // [P, hidden]
  uint16_t* x = nullptr;        // the residual stream
  uint16_t* normed = nullptr;   // LayerNorm output, and the merger's [P/4, 4608] view
  uint16_t* q = nullptr;
  uint16_t* k = nullptr;
  uint16_t* v = nullptr;
  uint16_t* attn_o = nullptr;
  // [P, head_dim] fp32
  float* cos = nullptr;
  float* sin = nullptr;
  // [P, taps]
  int32_t* interp_idx = nullptr;
  float* interp_w = nullptr;
  // [num_segments + 1]
  int32_t* cu_seqlens = nullptr;
  // [C, ...] chunked
  uint16_t* pixel = nullptr;    // [C, patch_dim]
  uint16_t* qkv = nullptr;      // [C, 3*hidden]
  uint16_t* fc1 = nullptr;      // [C, intermediate]  (also [C, merger_hidden] for the merger)
  uint16_t* chunk_out = nullptr;  // [C, hidden]
  // Only allocated while a trace is attached: the fp32 `pos_embeds` the golden dumps.
  float* pos_embeds = nullptr;
};

int64_t VisionTower::PlanScratchBytes(const VisionConfig& cfg, int64_t total_patches,
                                       int64_t row_chunk) {
  const int64_t P = total_patches;
  const int64_t C = std::min(row_chunk, P);
  const int64_t h = cfg.hidden_size;
  int64_t b = 0;
  b += 6 * AlignUp16(P * h * 2);                                  // x, normed, q, k, v, attn_o
  b += 2 * AlignUp16(P * cfg.HeadDim() * 4);                      // cos, sin
  b += AlignUp16(P * 4 * 4) + AlignUp16(P * 4 * 4);               // interp idx + weights (4 taps)
  b += AlignUp16(1024 * 4);                                        // cu_seqlens (generous)
  b += AlignUp16(C * cfg.PatchDim() * 2);                          // pixel
  b += AlignUp16(C * 3 * h * 2);                                   // qkv
  b += AlignUp16(C * std::max(cfg.intermediate_size, cfg.MergerHidden()) * 2);  // fc1 / merger fc1
  b += AlignUp16(C * h * 2);                                       // chunk_out
  return b;
}

void VisionTower::EnsureScratch(const VisionConfig& cfg, int64_t total_patches, int64_t row_chunk,
                                 bool tracing) {
  // Capacity is the whole of this arena's state: every Encode allocates the same buffers in the
  // same order, so `PlanScratchBytes` (which sums exactly those aligned sizes) is a sufficient
  // bound, and a call that needs no more than what is already reserved just resets. Only grow --
  // Arena::Reserve reallocates whenever the size DIFFERS, so handing it a smaller number would
  // free and re-hipMalloc the arena on every small image after a large one.
  int64_t need = PlanScratchBytes(cfg, total_patches, row_chunk);
  // The fp32 `pos_embeds` the golden dumps exists only under a trace (tests/vision), and it is
  // P * 1152 * 4 -- 40 MiB at a 1536x1536 image, a fifth of the arena. A production encode must
  // not reserve it; a traced one after an untraced one simply grows the arena once.
  if (tracing) need += AlignUp16(total_patches * cfg.hidden_size * 4);
  if (need > static_cast<int64_t>(arena_.capacity_bytes())) {
    arena_.Reserve(static_cast<size_t>(need));
  }
  arena_.Reset();
}

void VisionTower::Gemm(const VisionLinear& w, const uint16_t* x, uint16_t* y, int64_t M) {
  const GemmTuning t = PickTuning(w.K);
  for (int64_t m0 = 0; m0 < M; m0 += kMaxGemmM) {
    const int m = static_cast<int>(std::min(kMaxGemmM, M - m0));
    core::r4d::GemmBf16NtM64(x + m0 * w.K, w.weight.data(), y + m0 * w.N, m,
                              static_cast<int>(w.K), static_cast<int>(w.N), t.WV, t.SK, t.MB,
                              stream_.get());
  }
}

void VisionTower::Emit(const VisionTrace* trace, const std::string& name, const void* dev,
                        int64_t elems, bool is_fp32) {
  if (trace == nullptr || *trace == nullptr) return;
  const size_t bytes = static_cast<size_t>(elems) * (is_fp32 ? 4 : 2);
  stream_.Synchronize();
  trace_staging_.resize(bytes);
  R4DX_HIP_CHECK(hipMemcpy(trace_staging_.data(), dev, bytes, hipMemcpyDeviceToHost));
  (*trace)(name, trace_staging_.data(), elems, is_fp32);
}

void VisionTower::Encode(const VisionWeights& weights, const float* pixel_values,
                          int64_t total_patches, const std::vector<GridThw>& grids,
                          core::DeviceBuffer<uint16_t>* out, VisionEncodeStats* stats,
                          const VisionTrace* trace, const VisionPreBlock* pre_block) {
  const auto t_start = std::chrono::steady_clock::now();
  const VisionConfig& cfg = weights.config;
  const int64_t P = total_patches;
  const int64_t hidden = cfg.hidden_size;
  const int64_t head_dim = cfg.HeadDim();
  const int64_t merge = cfg.spatial_merge_size;
  const int64_t merger_hidden = cfg.MergerHidden();
  const bool tracing = (trace != nullptr && *trace != nullptr);

  if (P <= 0) throw std::runtime_error("VisionTower::Encode: total_patches must be > 0");
  int64_t expect_patches = 0, merged_tokens = 0;
  for (const GridThw& g : grids) {
    if (g.h % merge != 0 || g.w % merge != 0) {
      throw std::runtime_error(
          "VisionTower::Encode: grid " + std::to_string(g.h) + "x" + std::to_string(g.w) +
          " is not a multiple of spatial_merge_size -- smart_resize guarantees it is, so this "
          "grid did not come from PreprocessImages");
    }
    expect_patches += g.PatchCount();
    merged_tokens += g.MergedTokenCount(merge);
  }
  if (expect_patches != P) {
    throw std::runtime_error("VisionTower::Encode: grid_thw implies " +
                              std::to_string(expect_patches) + " patches but " + std::to_string(P) +
                              " were supplied");
  }

  // A traced encode materializes every intermediate whole (see VisionTrace's doc comment); the
  // result is bit-identical either way.
  const int64_t row_chunk = tracing ? P : std::min(kDefaultRowChunk, P);
  EnsureScratch(cfg, P, row_chunk, tracing);
  const int64_t C = row_chunk;

  // ---- host-side index math (all of it derived from grid_thw alone) --------------------------
  const PosEmbedInterpolation interp =
      BuildPosEmbedInterpolation(grids, static_cast<int>(cfg.NumGridPerSide()),
                                  static_cast<int>(merge));
  const std::vector<int32_t> rope_pos =
      BuildVisionRopePositionIds(grids, static_cast<int>(merge));
  const std::vector<int32_t> cu = BuildCuSeqlens(grids);
  std::vector<float> cos_host, sin_host;
  BuildVisionRopeCosSin(rope_pos, static_cast<int>(head_dim), cfg.rope_theta, &cos_host,
                        &sin_host);
  const int64_t num_segments = static_cast<int64_t>(cu.size()) - 1;
  int64_t max_seqlen = 0;
  for (int64_t s = 0; s < num_segments; ++s) {
    max_seqlen = std::max<int64_t>(max_seqlen, cu[static_cast<size_t>(s + 1)] -
                                                    cu[static_cast<size_t>(s)]);
  }

  Scratch sc;
  sc.x = arena_.Alloc<uint16_t>(static_cast<size_t>(P * hidden), 16);
  sc.normed = arena_.Alloc<uint16_t>(static_cast<size_t>(P * hidden), 16);
  sc.q = arena_.Alloc<uint16_t>(static_cast<size_t>(P * hidden), 16);
  sc.k = arena_.Alloc<uint16_t>(static_cast<size_t>(P * hidden), 16);
  sc.v = arena_.Alloc<uint16_t>(static_cast<size_t>(P * hidden), 16);
  sc.attn_o = arena_.Alloc<uint16_t>(static_cast<size_t>(P * hidden), 16);
  sc.cos = arena_.Alloc<float>(static_cast<size_t>(P * head_dim), 16);
  sc.sin = arena_.Alloc<float>(static_cast<size_t>(P * head_dim), 16);
  sc.interp_idx = arena_.Alloc<int32_t>(interp.indices.size(), 16);
  sc.interp_w = arena_.Alloc<float>(interp.weights.size(), 16);
  sc.cu_seqlens = arena_.Alloc<int32_t>(cu.size(), 16);
  sc.pixel = arena_.Alloc<uint16_t>(static_cast<size_t>(C * cfg.PatchDim()), 16);
  sc.qkv = arena_.Alloc<uint16_t>(static_cast<size_t>(C * 3 * hidden), 16);
  sc.fc1 = arena_.Alloc<uint16_t>(
      static_cast<size_t>(C * std::max(cfg.intermediate_size, merger_hidden)), 16);
  sc.chunk_out = arena_.Alloc<uint16_t>(static_cast<size_t>(C * hidden), 16);
  if (tracing) sc.pos_embeds = arena_.Alloc<float>(static_cast<size_t>(P * hidden), 16);

  const hipStream_t s = stream_.get();
  auto up = [&](void* dev, const void* host, size_t bytes) {
    R4DX_HIP_CHECK(hipMemcpyAsync(dev, host, bytes, hipMemcpyHostToDevice, s));
  };
  up(sc.cos, cos_host.data(), cos_host.size() * 4);
  up(sc.sin, sin_host.data(), sin_host.size() * 4);
  up(sc.interp_idx, interp.indices.data(), interp.indices.size() * 4);
  up(sc.interp_w, interp.weights.data(), interp.weights.size() * 4);
  up(sc.cu_seqlens, cu.data(), cu.size() * 4);

  // ---- 1. patch embed: a dense [patch_dim -> hidden] matmul + bias per patch row --------------
  // The reference casts `pixel_values` to the projection's own dtype BEFORE the matmul
  // (Qwen3_5VisionPatchEmbed.forward), so the bf16 rounding happens here, on the host, not inside
  // the GEMM. Kept alive for the whole call: the H2D copies below are async.
  std::vector<uint16_t> pixel_bf16(static_cast<size_t>(P) * static_cast<size_t>(cfg.PatchDim()));
  for (size_t i = 0; i < pixel_bf16.size(); ++i) {
    pixel_bf16[i] = core::FloatToBf16(pixel_values[i]);
  }
  for (int64_t r0 = 0; r0 < P; r0 += C) {
    const int64_t rows = std::min(C, P - r0);
    up(sc.pixel, pixel_bf16.data() + r0 * cfg.PatchDim(),
       static_cast<size_t>(rows * cfg.PatchDim()) * 2);
    Gemm(weights.patch_embed, sc.pixel, sc.x + r0 * hidden, rows);
    r4dx_bias_add_bf16(reinterpret_cast<int64_t>(sc.x + r0 * hidden),
                       reinterpret_cast<int64_t>(weights.patch_embed.bias.data()),
                       reinterpret_cast<int64_t>(sc.x + r0 * hidden), rows, hidden,
                       reinterpret_cast<int64_t>(s));
  }
  Emit(trace, "patch_embed_out", sc.x, P * hidden, false);

  // ---- 2. learned position embedding, gathered and added in place ----------------------------
  r4dx_vision_pos_embed_bf16(
      reinterpret_cast<int64_t>(weights.pos_embed_table.data()),
      reinterpret_cast<int64_t>(sc.interp_idx), reinterpret_cast<int64_t>(sc.interp_w),
      reinterpret_cast<int64_t>(sc.x), reinterpret_cast<int64_t>(sc.pos_embeds),
      reinterpret_cast<int64_t>(sc.x), P, hidden, /*taps=*/4, cfg.num_position_embeddings,
      reinterpret_cast<int64_t>(s));
  if (tracing) Emit(trace, "pos_embeds", sc.pos_embeds, P * hidden, true);
  Emit(trace, "block_input", sc.x, P * hidden, false);
  Emit(trace, "rope_cos", sc.cos, P * head_dim, true);
  Emit(trace, "rope_sin", sc.sin, P * head_dim, true);

  // ---- 3. the encoder blocks -----------------------------------------------------------------
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int64_t b = 0; b < cfg.depth; ++b) {
    const VisionBlockWeights& bw = weights.blocks[static_cast<size_t>(b)];
    const bool trace_block0 = tracing && b == 0;
    if (pre_block != nullptr && *pre_block != nullptr) {
      stream_.Synchronize();
      (*pre_block)(b, sc.x, P * hidden);
    }

    // --- attention half: h = h + attn(LayerNorm(h)) ---
    r4dx_layernorm_bf16(reinterpret_cast<int64_t>(sc.x),
                        reinterpret_cast<int64_t>(bw.norm1.weight.data()),
                        reinterpret_cast<int64_t>(bw.norm1.bias.data()),
                        reinterpret_cast<int64_t>(sc.normed), P, hidden,
                        static_cast<float>(cfg.layer_norm_eps), reinterpret_cast<int64_t>(s));
    if (trace_block0) Emit(trace, "block0_norm1_out", sc.normed, P * hidden, false);

    for (int64_t r0 = 0; r0 < P; r0 += C) {
      const int64_t rows = std::min(C, P - r0);
      Gemm(bw.qkv, sc.normed + r0 * hidden, sc.qkv, rows);
      r4dx_bias_add_bf16(reinterpret_cast<int64_t>(sc.qkv),
                         reinterpret_cast<int64_t>(bw.qkv.bias.data()),
                         reinterpret_cast<int64_t>(sc.qkv), rows, 3 * hidden,
                         reinterpret_cast<int64_t>(s));
      if (trace_block0) Emit(trace, "block0_attn_qkv_raw", sc.qkv, rows * 3 * hidden, false);
      r4dx_vision_qkv_rope_bf16(
          reinterpret_cast<int64_t>(sc.qkv), reinterpret_cast<int64_t>(sc.cos + r0 * head_dim),
          reinterpret_cast<int64_t>(sc.sin + r0 * head_dim),
          reinterpret_cast<int64_t>(sc.q + r0 * hidden),
          reinterpret_cast<int64_t>(sc.k + r0 * hidden),
          reinterpret_cast<int64_t>(sc.v + r0 * hidden), static_cast<int>(rows),
          static_cast<int>(cfg.num_heads), static_cast<int>(head_dim),
          reinterpret_cast<int64_t>(s));
    }
    if (trace_block0) {
      Emit(trace, "attn_q_post_rope", sc.q, P * hidden, false);
      Emit(trace, "attn_k_post_rope", sc.k, P * hidden, false);
    }

    // Dense, non-causal, one segment per image -- the whole batch in one launch.
    core::r4d::AttnVitBf16(sc.q, sc.k, sc.v, sc.attn_o, sc.cu_seqlens,
                            static_cast<int>(num_segments), static_cast<int>(max_seqlen),
                            static_cast<int>(cfg.num_heads), static_cast<int>(head_dim),
                            attn_scale, s);

    for (int64_t r0 = 0; r0 < P; r0 += C) {
      const int64_t rows = std::min(C, P - r0);
      Gemm(bw.proj, sc.attn_o + r0 * hidden, sc.chunk_out, rows);
      r4dx_bias_add_bf16(reinterpret_cast<int64_t>(sc.chunk_out),
                         reinterpret_cast<int64_t>(bw.proj.bias.data()),
                         reinterpret_cast<int64_t>(sc.chunk_out), rows, hidden,
                         reinterpret_cast<int64_t>(s));
      if (trace_block0) Emit(trace, "block0_attn_proj_out", sc.chunk_out, rows * hidden, false);
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(sc.x + r0 * hidden),
                             reinterpret_cast<int64_t>(sc.chunk_out),
                             reinterpret_cast<int64_t>(sc.x + r0 * hidden), rows * hidden,
                             reinterpret_cast<int64_t>(s));
    }

    // --- MLP half: h = h + mlp(LayerNorm(h)), a plain GELU MLP (NOT the text side's SwiGLU) ---
    r4dx_layernorm_bf16(reinterpret_cast<int64_t>(sc.x),
                        reinterpret_cast<int64_t>(bw.norm2.weight.data()),
                        reinterpret_cast<int64_t>(bw.norm2.bias.data()),
                        reinterpret_cast<int64_t>(sc.normed), P, hidden,
                        static_cast<float>(cfg.layer_norm_eps), reinterpret_cast<int64_t>(s));
    if (trace_block0) Emit(trace, "block0_norm2_out", sc.normed, P * hidden, false);

    for (int64_t r0 = 0; r0 < P; r0 += C) {
      const int64_t rows = std::min(C, P - r0);
      Gemm(bw.fc1, sc.normed + r0 * hidden, sc.fc1, rows);
      r4dx_bias_add_bf16(reinterpret_cast<int64_t>(sc.fc1),
                         reinterpret_cast<int64_t>(bw.fc1.bias.data()),
                         reinterpret_cast<int64_t>(sc.fc1), rows, cfg.intermediate_size,
                         reinterpret_cast<int64_t>(s));
      if (trace_block0) Emit(trace, "block0_mlp_fc1_out", sc.fc1, rows * cfg.intermediate_size, false);
      r4dx_gelu_tanh_bf16(reinterpret_cast<int64_t>(sc.fc1), reinterpret_cast<int64_t>(sc.fc1),
                          rows * cfg.intermediate_size, reinterpret_cast<int64_t>(s));
      Gemm(bw.fc2, sc.fc1, sc.chunk_out, rows);
      r4dx_bias_add_bf16(reinterpret_cast<int64_t>(sc.chunk_out),
                         reinterpret_cast<int64_t>(bw.fc2.bias.data()),
                         reinterpret_cast<int64_t>(sc.chunk_out), rows, hidden,
                         reinterpret_cast<int64_t>(s));
      if (trace_block0) Emit(trace, "block0_mlp_fc2_out", sc.chunk_out, rows * hidden, false);
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(sc.x + r0 * hidden),
                             reinterpret_cast<int64_t>(sc.chunk_out),
                             reinterpret_cast<int64_t>(sc.x + r0 * hidden), rows * hidden,
                             reinterpret_cast<int64_t>(s));
    }

    if (tracing) {
      char name[32];
      std::snprintf(name, sizeof(name), "block_%02d_output", static_cast<int>(b));
      Emit(trace, name, sc.x, P * hidden, false);
    }
  }
  Emit(trace, "last_hidden_state", sc.x, P * hidden, false);

  // ---- 4. merger: LayerNorm(hidden) -> view [-1, 4*hidden] -> fc1 -> exact GELU -> fc2 --------
  // The view is a pure reshape because preprocessing already emitted patches in 2x2
  // spatial-merge-block-major order (docs/vision.md step 1): four consecutive rows of `normed` ARE
  // one merged token's 4608 values, contiguous, so no permute happens anywhere.
  r4dx_layernorm_bf16(reinterpret_cast<int64_t>(sc.x),
                      reinterpret_cast<int64_t>(weights.merger_norm.weight.data()),
                      reinterpret_cast<int64_t>(weights.merger_norm.bias.data()),
                      reinterpret_cast<int64_t>(sc.normed), P, hidden,
                      static_cast<float>(cfg.layer_norm_eps), reinterpret_cast<int64_t>(s));

  out->Resize(static_cast<size_t>(merged_tokens * cfg.out_hidden_size));
  for (int64_t r0 = 0; r0 < merged_tokens; r0 += C) {
    const int64_t rows = std::min(C, merged_tokens - r0);
    Gemm(weights.merger_fc1, sc.normed + r0 * merger_hidden, sc.fc1, rows);
    r4dx_bias_add_bf16(reinterpret_cast<int64_t>(sc.fc1),
                       reinterpret_cast<int64_t>(weights.merger_fc1.bias.data()),
                       reinterpret_cast<int64_t>(sc.fc1), rows, merger_hidden,
                       reinterpret_cast<int64_t>(s));
    // nn.GELU() with approximate='none' -- the EXACT erf form, not the encoder MLP's tanh one.
    r4dx_gelu_erf_bf16(reinterpret_cast<int64_t>(sc.fc1), reinterpret_cast<int64_t>(sc.fc1),
                       rows * merger_hidden, reinterpret_cast<int64_t>(s));
    Gemm(weights.merger_fc2, sc.fc1, out->data() + r0 * cfg.out_hidden_size, rows);
    r4dx_bias_add_bf16(reinterpret_cast<int64_t>(out->data() + r0 * cfg.out_hidden_size),
                       reinterpret_cast<int64_t>(weights.merger_fc2.bias.data()),
                       reinterpret_cast<int64_t>(out->data() + r0 * cfg.out_hidden_size), rows,
                       cfg.out_hidden_size, reinterpret_cast<int64_t>(s));
  }
  stream_.Synchronize();
  Emit(trace, "merger_output", out->data(), merged_tokens * cfg.out_hidden_size, false);

  if (stats != nullptr) {
    stats->total_patches = P;
    stats->merged_tokens = merged_tokens;
    stats->num_segments = num_segments;
    stats->max_seqlen = max_seqlen;
    stats->row_chunk = C;
    stats->scratch_bytes = ScratchBytes();
    stats->encode_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_start)
                            .count();
  }
}

}  // namespace r4dx::vision




