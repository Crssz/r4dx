// src/vision/gemma_vision.h -- host-side half of Gemma 4 (gemma4_unified) vision (docs/gemma4-plan.md M2): the
// image preprocessor (Gemma4UnifiedImageProcessor), the per-image block / chunk planning of the bidirectional
// image mask, and a host reference of the encoder-free embedder. CPU only, no HIP: like the rest of
// r4dx_vision this runs in ordinary always-on unit tests.
//
// What the checkpoint's processor does (transformers 5.18, docs/gemma4-semantics.md section 3):
//   1. bicubic ANTIALIASED resize (the torch uint8 kernel, i.e. exactly Qwen's ResizeU8) to the largest
//      (h, w), both multiples of 48, with h*w <= max_soft_tokens * 9 * 256 (aspect preserving, floor; images
//      are grown as well as shrunk -- the budget is a target, not a cap). No resize at all when the image
//      already has that size.
//   2. rescale 1/255 in fp32 (float(u8) * float(1/255)); no mean/std.
//   3. 16 px patches, merged 3x3 into 48x48x3 HWC rasters (6912 values) in row-major order of the MERGED
//      grid; position of a merged patch is (x, y) in merged-grid cells. Soft tokens = (h/48) * (w/48).
// The HF processor pads every image to max_soft_tokens rows with position (-1, -1); the model drops the pad
// rows again before scattering, so this API returns only the real rows.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_decode.h"
#include "preprocess.h"

namespace r4dx::vision {

constexpr int kGemmaPatchSize = 16;      // teacher patch
constexpr int kGemmaPooling = 3;         // 3 x 3 teacher patches per merged patch
constexpr int kGemmaMergedSide = 48;     // kGemmaPatchSize * kGemmaPooling
constexpr int kGemmaPatchDim = 6912;     // 48 * 48 * 3
constexpr int kGemmaDefaultSoftTokens = 280;
// The runtime splices one image block per prefill chunk, and the sliding KV ring holds window + 288 rows
// (docs/gemma4-plan.md 3.3), so a single image may carry at most this many soft tokens. 560 / 1120 are valid
// processor budgets (preprocessing supports them) but the engine refuses them.
constexpr int kGemmaMaxImageBlockTokens = 288;
constexpr int kGemmaBoiTokenId = 255999;   // <|image>
constexpr int kGemmaEoiTokenId = 258882;   // <image|>
constexpr int kGemmaImageTokenId = 258880; // <|image|>

struct GemmaImageConfig {
  int patch_size = kGemmaPatchSize;
  int pooling = kGemmaPooling;
  int max_soft_tokens = kGemmaDefaultSoftTokens;  // 70 | 140 | 280 | 560 | 1120
  int MergedSide() const { return patch_size * pooling; }
  int PatchDim() const { return MergedSide() * MergedSide() * 3; }
};

// Throws std::runtime_error unless max_soft_tokens is one of the five processor budgets.
void CheckGemmaSoftTokens(int max_soft_tokens);

// transformers' get_aspect_ratio_preserving_size, including its errors (0 x 0 target, budget overflow).
void GemmaTargetSize(int64_t height, int64_t width, const GemmaImageConfig& cfg, int64_t* out_height,
                     int64_t* out_width);

struct GemmaPreprocessed {
  int64_t n = 0;                    // soft tokens == merged patches == rows of pixel_values
  int64_t grid_h = 0, grid_w = 0;   // merged grid (rows, cols); n == grid_h * grid_w
  int64_t resized_h = 0, resized_w = 0;
  std::vector<float> pixel_values;  // [n, 6912] float32, merged-patch raster order
  std::vector<int32_t> positions;   // [n, 2] (x, y)
};

GemmaPreprocessed PreprocessGemmaImage(const DecodedImage& img, const GemmaImageConfig& cfg = GemmaImageConfig());

// ---- the bidirectional image mask (docs/gemma4-semantics.md section 2) ----------------------------------------
// A block is a maximal run of image soft tokens (boi / eoi are not part of one), given here as absolute
// [start, end) positions. Sliding layers only: query q inside a block may additionally attend every key up to
// end - 1 (and the window still applies from below); full layers and decode stay causal.
struct ImageBlock {
  int64_t start = 0;  // absolute position of the first soft token
  int64_t end = 0;    // one past the last
};

// Blocks of a token vector: maximal runs of `image_token_id`. Positions are indices into `tokens` + base.
std::vector<ImageBlock> FindImageBlocks(const std::vector<int32_t>& tokens, int32_t image_token_id, int64_t base = 0);

// One prefill chunk: rows [start, start + len) of the call's token vector (relative indices).
struct PrefillChunk {
  int64_t start = 0;
  int64_t len = 0;
};

// Splits `total` rows into chunks of at most `max_rows` such that no image block (relative [start, end)) is
// ever split: a block is a chunk's whole bidirectional neighbourhood, and its keys must all be written before
// its first query runs. A block starting inside the current chunk range but not fitting ends the chunk at the
// block's start; a block that starts the chunk extends it to the block's end (a block is <= max_block rows).
// Throws if a block is longer than max_block or lies outside [0, total].
std::vector<PrefillChunk> PlanPrefillChunks(int64_t total, const std::vector<ImageBlock>& blocks, int64_t max_rows,
                                            int64_t max_block);

// klimit_ext for the rows [start, start + len): for a row inside a block, end - 1 (the absolute last key of its
// block), else -1 (causal). Positions are absolute: `abs_start` is the absolute position of row `start`, the
// blocks are in the same absolute coordinates. The result is non-decreasing in the effective bound
// max(qpos, ext), the contract of the sliding kernels.
std::vector<int32_t> BuildKlimitExt(int64_t abs_start, int64_t len, const std::vector<ImageBlock>& blocks);

// Dense reference mask [T, T] (1 = attend) of one layer type, the CPU oracle the kernels and the HF dumps are
// compared against: causal, AND window for a sliding layer (window <= 0: none), OR same-block when
// `bidirectional` (the generate() path: AND(window, OR(causal, same_block))).
std::vector<uint8_t> BuildDenseMask(int64_t T, int64_t window, bool bidirectional, const std::vector<ImageBlock>& blocks);

// ---- prompt expansion ------------------------------------------------------------------------------------------
// The Gemma chat template renders one `<|image|>` per image; the processor expands it to
// `<|image>` + N x `<|image|>` + `<image|>`. Same contract as image_prompt.h's ExpandImagePlaceholders (spans carry
// the offset of the FIRST soft token, i.e. after the boi) but with the boi / eoi wrapper.
struct GemmaExpandedImage {
  int64_t offset = 0;  // index of the first soft token in the expanded vector
  int64_t tokens = 0;
};
struct GemmaExpandedPrompt {
  std::vector<int32_t> tokens;
  std::vector<GemmaExpandedImage> spans;
};
// `soft_tokens[i]`: the i-th image's token count. Throws std::runtime_error when the number of placeholders in
// `raw_tokens` differs from soft_tokens.size().
GemmaExpandedPrompt ExpandGemmaImagePlaceholders(const std::vector<int32_t>& raw_tokens,
                                                 const std::vector<int64_t>& soft_tokens,
                                                 int32_t image_token_id = kGemmaImageTokenId,
                                                 int32_t boi_id = kGemmaBoiTokenId, int32_t eoi_id = kGemmaEoiTokenId);

// ---- embedder host reference ----------------------------------------------------------------------------------
// The HF Gemma4UnifiedVisionEmbedder + multimodal embedder chain on the host, with bf16 rounding exactly where
// the bf16 torch module chain rounds (every op output is rounded to bf16; LayerNorm / RMSNorm / GEMM accumulate
// in fp32). Weights are float vectors holding bf16-representable values, row-major as in the checkpoint.
//   x = bf16(pixel);  h = LN1(x);  h = dense(h) [+bias, one rounding];  h = LN2(h)
//   pos = bf16(pos_embedding[x][0] + pos_embedding[y][1]);  h = bf16(h + pos);  h = pos_norm(h)
//   h = RMSNorm_noscale(h, eps);  out = proj(h) (no bias)
struct GemmaEmbedderWeights {
  int patch_dim = 0;     // 6912
  int mm_dim = 0;        // 3840 (mm_embed_dim == output_proj_dims)
  int out_dim = 0;       // text hidden, 3840
  int posemb_size = 0;   // 1120
  float ln_eps = 1e-5f;
  float rms_eps = 1e-6f;
  std::vector<float> ln1_w, ln1_b;        // [patch_dim]
  std::vector<float> dense_w, dense_b;    // [mm_dim, patch_dim], [mm_dim]
  std::vector<float> ln2_w, ln2_b;        // [mm_dim]
  std::vector<float> pos_table;           // [posemb_size, 2, mm_dim]
  std::vector<float> pos_norm_w, pos_norm_b;  // [mm_dim]
  std::vector<float> proj_w;              // [out_dim, mm_dim]
};

// `pixel_values` [n, patch_dim] fp32, `positions` [n, 2] (x, y) all >= 0. Returns [n, out_dim] floats that are
// bf16-representable.
std::vector<float> GemmaEmbedHost(const GemmaEmbedderWeights& w, const float* pixel_values, const int32_t* positions,
                                  int64_t n);

// fp32 -> bf16 -> fp32 (round to nearest even), the rounding every op above applies.
float RoundBf16(float x);

}  // namespace r4dx::vision
