#include "gemma_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace r4dx::vision {

void CheckGemmaSoftTokens(int max_soft_tokens) {
  switch (max_soft_tokens) {
    case 70:
    case 140:
    case 280:
    case 560:
    case 1120:
      return;
    default:
      throw std::runtime_error("max_soft_tokens must be one of 70, 140, 280, 560, 1120, got " +
                               std::to_string(max_soft_tokens));
  }
}

void GemmaTargetSize(int64_t height, int64_t width, const GemmaImageConfig& cfg, int64_t* out_height,
                     int64_t* out_width) {
  if (height <= 0 || width <= 0) throw std::runtime_error("gemma preprocess: image has a zero side");
  const int64_t pool = cfg.pooling;
  const int64_t max_patches = static_cast<int64_t>(cfg.max_soft_tokens) * pool * pool;
  const int64_t side = pool * cfg.patch_size;
  const double total_px = static_cast<double>(height) * static_cast<double>(width);
  const int64_t target_px = max_patches * cfg.patch_size * cfg.patch_size;
  const double factor = std::sqrt(static_cast<double>(target_px) / total_px);
  const double ideal_h = factor * static_cast<double>(height);
  const double ideal_w = factor * static_cast<double>(width);
  int64_t th = static_cast<int64_t>(std::floor(ideal_h / static_cast<double>(side))) * side;
  int64_t tw = static_cast<int64_t>(std::floor(ideal_w / static_cast<double>(side))) * side;
  if (th == 0 && tw == 0) {
    throw std::runtime_error("Attempting to resize to a 0 x 0 image. Resized height should be divisible by " +
                             std::to_string(side));
  }
  const int64_t max_side = (max_patches / (pool * pool)) * side;
  if (th == 0) {
    th = side;
    tw = std::min<int64_t>((width / height) * side, max_side);  // Python floor of a positive ratio
  } else if (tw == 0) {
    tw = side;
    th = std::min<int64_t>((height / width) * side, max_side);
  }
  if (th * tw > target_px) {
    throw std::runtime_error("Resizing [" + std::to_string(height) + "x" + std::to_string(width) + "] to [" +
                             std::to_string(th) + "x" + std::to_string(tw) + "] but this exceeds " +
                             std::to_string(max_patches) + " patches");
  }
  *out_height = th;
  *out_width = tw;
}

GemmaPreprocessed PreprocessGemmaImage(const DecodedImage& img, const GemmaImageConfig& cfg) {
  CheckGemmaSoftTokens(cfg.max_soft_tokens);
  if (img.rgb.size() != img.PixelCount() * 3) {
    throw std::runtime_error("PreprocessGemmaImage: DecodedImage buffer size does not match w*h*3");
  }
  int64_t th = 0, tw = 0;
  GemmaTargetSize(img.height, img.width, cfg, &th, &tw);
  const DecodedImage rs = ResizeU8(img, static_cast<int>(tw), static_cast<int>(th));  // returns img when equal

  const int side = cfg.MergedSide();
  GemmaPreprocessed out;
  out.resized_h = th;
  out.resized_w = tw;
  out.grid_h = th / side;
  out.grid_w = tw / side;
  out.n = out.grid_h * out.grid_w;
  const int64_t dim = cfg.PatchDim();
  out.pixel_values.resize(static_cast<size_t>(out.n * dim));
  out.positions.resize(static_cast<size_t>(out.n) * 2);
  // float32(1/255): the rescale is float32(u8) * float32(rescale_factor) (bit-checked against HF,
  // tools/reference/gemma/vision_preproc_golden.py --check).
  const float kRescale = static_cast<float>(1.0 / 255.0);
  int64_t idx = 0;
  for (int64_t gy = 0; gy < out.grid_h; ++gy) {
    for (int64_t gx = 0; gx < out.grid_w; ++gx, ++idx) {
      float* dst = out.pixel_values.data() + static_cast<size_t>(idx * dim);
      // The merged patch is the plain side x side HWC raster of its merged-grid cell.
      for (int py = 0; py < side; ++py) {
        const uint8_t* src = rs.rgb.data() + (static_cast<size_t>(gy * side + py) * rs.width +
                                              static_cast<size_t>(gx * side)) * 3;
        float* d = dst + static_cast<size_t>(py) * side * 3;
        for (int k = 0; k < side * 3; ++k) d[k] = static_cast<float>(src[k]) * kRescale;
      }
      out.positions[static_cast<size_t>(idx) * 2 + 0] = static_cast<int32_t>(gx);
      out.positions[static_cast<size_t>(idx) * 2 + 1] = static_cast<int32_t>(gy);
    }
  }
  return out;
}

// ---- masks and chunking -------------------------------------------------------------------------------------

std::vector<ImageBlock> FindImageBlocks(const std::vector<int32_t>& tokens, int32_t image_token_id, int64_t base) {
  std::vector<ImageBlock> blocks;
  const int64_t n = static_cast<int64_t>(tokens.size());
  for (int64_t i = 0; i < n;) {
    if (tokens[static_cast<size_t>(i)] != image_token_id) {
      ++i;
      continue;
    }
    int64_t j = i;
    while (j < n && tokens[static_cast<size_t>(j)] == image_token_id) ++j;
    blocks.push_back({base + i, base + j});
    i = j;
  }
  return blocks;
}

void CheckImageSpans(const std::vector<int32_t>& tokens, int32_t image_token_id, const std::vector<ImageBlock>& blocks,
                     int64_t max_block) {
  if (tokens.empty()) throw std::invalid_argument("image spans: token_ids is empty");
  const int64_t total = static_cast<int64_t>(tokens.size());
  int64_t prev_end = 0;
  for (const ImageBlock& b : blocks) {
    if (b.end <= b.start || b.end - b.start > max_block || b.start < prev_end || b.end > total) {
      throw std::invalid_argument("image span [" + std::to_string(b.start) + ", " + std::to_string(b.end) +
                                  ") is out of order, empty, over " + std::to_string(max_block) +
                                  " tokens, or outside the call's tokens");
    }
    prev_end = b.end;
  }
  const std::vector<ImageBlock> runs = FindImageBlocks(tokens, image_token_id, 0);
  bool same = runs.size() == blocks.size();
  for (size_t i = 0; same && i < runs.size(); ++i) same = runs[i].start == blocks[i].start && runs[i].end == blocks[i].end;
  if (!same) {
    throw std::invalid_argument("the image placeholder runs in token_ids do not match the image spans one to one");
  }
}

void CheckAudioSpans(int64_t total, const std::vector<ImageBlock>& blocks) {
  if (total <= 0) throw std::invalid_argument("audio spans: token_ids is empty");
  for (const ImageBlock& b : blocks) {
    if (b.start < 0 || b.end <= b.start || b.end > total) {
      throw std::invalid_argument("audio span [" + std::to_string(b.start) + ", " + std::to_string(b.end) +
                                  ") is empty or outside the " + std::to_string(total) + " tokens fed");
    }
  }
}

std::vector<PrefillChunk> PlanPrefillChunks(int64_t total, const std::vector<ImageBlock>& blocks, int64_t max_rows,
                                            int64_t max_block) {
  if (total < 0 || max_rows < 1) throw std::invalid_argument("PlanPrefillChunks: bad total / max_rows");
  int64_t prev_end = 0;
  for (const ImageBlock& b : blocks) {
    if (b.start < prev_end || b.end <= b.start || b.end > total) {
      throw std::invalid_argument("PlanPrefillChunks: image blocks must be ordered, non-empty and inside [0, total]");
    }
    if (b.end - b.start > max_block) {
      throw std::invalid_argument("PlanPrefillChunks: an image block of " + std::to_string(b.end - b.start) +
                                  " rows exceeds the " + std::to_string(max_block) + "-row limit");
    }
    prev_end = b.end;
  }
  std::vector<PrefillChunk> chunks;
  size_t bi = 0;
  int64_t off = 0;
  while (off < total) {
    int64_t end = std::min(off + max_rows, total);
    while (bi < blocks.size() && blocks[bi].end <= off) ++bi;
    // The first block that reaches into (off, ...): find one that straddles `end`.
    for (size_t k = bi; k < blocks.size() && blocks[k].start < end; ++k) {
      if (blocks[k].end > end) {          // straddles the cut
        end = blocks[k].start > off ? blocks[k].start : blocks[k].end;
        break;
      }
    }
    chunks.push_back({off, end - off});
    off = end;
  }
  return chunks;
}

std::vector<int32_t> BuildKlimitExt(int64_t abs_start, int64_t len, const std::vector<ImageBlock>& blocks) {
  std::vector<int32_t> ext(static_cast<size_t>(len), -1);
  for (const ImageBlock& b : blocks) {
    const int64_t lo = std::max(b.start, abs_start), hi = std::min(b.end, abs_start + len);
    for (int64_t p = lo; p < hi; ++p) ext[static_cast<size_t>(p - abs_start)] = static_cast<int32_t>(b.end - 1);
  }
  return ext;
}

std::vector<uint8_t> BuildDenseMask(int64_t T, int64_t window, bool bidirectional, const std::vector<ImageBlock>& blocks) {
  std::vector<int64_t> block_of(static_cast<size_t>(T), -1);
  for (size_t b = 0; b < blocks.size(); ++b) {
    for (int64_t p = blocks[b].start; p < blocks[b].end && p < T; ++p) block_of[static_cast<size_t>(p)] = static_cast<int64_t>(b);
  }
  std::vector<uint8_t> m(static_cast<size_t>(T * T), 0);
  for (int64_t q = 0; q < T; ++q) {
    for (int64_t k = 0; k < T; ++k) {
      const bool causal = k <= q;
      const bool same = bidirectional && block_of[static_cast<size_t>(q)] >= 0 &&
                        block_of[static_cast<size_t>(q)] == block_of[static_cast<size_t>(k)];
      const bool in_window = window <= 0 || k > q - window;
      m[static_cast<size_t>(q * T + k)] = (in_window && (causal || same)) ? 1 : 0;
    }
  }
  return m;
}

GemmaExpandedPrompt ExpandGemmaImagePlaceholders(const std::vector<int32_t>& raw_tokens,
                                                 const std::vector<int64_t>& soft_tokens, int32_t image_token_id,
                                                 int32_t boi_id, int32_t eoi_id) {
  GemmaExpandedPrompt out;
  out.tokens.reserve(raw_tokens.size());
  size_t next = 0;
  for (const int32_t id : raw_tokens) {
    if (id != image_token_id) {
      out.tokens.push_back(id);
      continue;
    }
    if (next >= soft_tokens.size()) {
      throw std::runtime_error("rendered prompt has more image placeholders than images were supplied");
    }
    out.tokens.push_back(boi_id);
    GemmaExpandedImage sp;
    sp.offset = static_cast<int64_t>(out.tokens.size());
    sp.tokens = soft_tokens[next];
    out.tokens.insert(out.tokens.end(), static_cast<size_t>(sp.tokens), image_token_id);
    out.tokens.push_back(eoi_id);
    out.spans.push_back(sp);
    ++next;
  }
  if (next != soft_tokens.size()) {
    throw std::runtime_error("rendered prompt has fewer image placeholders than images were supplied");
  }
  return out;
}

// ---- embedder host reference ---------------------------------------------------------------------------------

float RoundBf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7F800000u) == 0x7F800000u) return x;  // inf / nan pass through
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

namespace {

// torch.nn.LayerNorm over the last dim (biased variance), fp32 math, one bf16 rounding of the output.
void LayerNormRow(const float* x, const float* w, const float* b, float eps, int64_t d, float* out) {
  double mean = 0.0;
  for (int64_t i = 0; i < d; ++i) mean += x[i];
  mean /= static_cast<double>(d);
  double var = 0.0;
  for (int64_t i = 0; i < d; ++i) {
    const double t = x[i] - mean;
    var += t * t;
  }
  var /= static_cast<double>(d);
  const float rstd = static_cast<float>(1.0 / std::sqrt(var + static_cast<double>(eps)));
  const float m = static_cast<float>(mean);
  for (int64_t i = 0; i < d; ++i) out[i] = RoundBf16((x[i] - m) * rstd * w[i] + b[i]);
}

}  // namespace

std::vector<float> GemmaEmbedHost(const GemmaEmbedderWeights& w, const float* pixel_values, const int32_t* positions,
                                  int64_t n) {
  const int64_t P = w.patch_dim, D = w.mm_dim, O = w.out_dim;
  std::vector<float> out(static_cast<size_t>(n * O));
  std::vector<float> a(static_cast<size_t>(P)), h(static_cast<size_t>(D)), t(static_cast<size_t>(D));
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t i = 0; i < P; ++i) a[static_cast<size_t>(i)] = RoundBf16(pixel_values[r * P + i]);
    std::vector<float> ln1(static_cast<size_t>(P));
    LayerNormRow(a.data(), w.ln1_w.data(), w.ln1_b.data(), w.ln_eps, P, ln1.data());
    for (int64_t o = 0; o < D; ++o) {
      double acc = 0.0;
      const float* wr = w.dense_w.data() + o * P;
      for (int64_t i = 0; i < P; ++i) acc += static_cast<double>(ln1[static_cast<size_t>(i)]) * wr[i];
      h[static_cast<size_t>(o)] = RoundBf16(static_cast<float>(acc) + w.dense_b[static_cast<size_t>(o)]);
    }
    LayerNormRow(h.data(), w.ln2_w.data(), w.ln2_b.data(), w.ln_eps, D, t.data());
    const int64_t px = positions[r * 2 + 0], py = positions[r * 2 + 1];
    if (px < 0 || py < 0 || px >= w.posemb_size || py >= w.posemb_size) {
      throw std::out_of_range("GemmaEmbedHost: position outside the pos_embedding table");
    }
    const float* tx = w.pos_table.data() + (px * 2 + 0) * D;
    const float* ty = w.pos_table.data() + (py * 2 + 1) * D;
    for (int64_t i = 0; i < D; ++i) {
      const float pe = RoundBf16(tx[i] + ty[i]);
      h[static_cast<size_t>(i)] = RoundBf16(t[static_cast<size_t>(i)] + pe);
    }
    LayerNormRow(h.data(), w.pos_norm_w.data(), w.pos_norm_b.data(), w.ln_eps, D, t.data());
    // RMSNorm without a weight (Gemma4UnifiedRMSNorm with_scale=False): x * rsqrt(mean(x^2) + eps).
    double ss = 0.0;
    for (int64_t i = 0; i < D; ++i) ss += static_cast<double>(t[static_cast<size_t>(i)]) * t[static_cast<size_t>(i)];
    const float rs = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(D) + static_cast<double>(w.rms_eps)));
    for (int64_t i = 0; i < D; ++i) h[static_cast<size_t>(i)] = RoundBf16(t[static_cast<size_t>(i)] * rs);
    for (int64_t o = 0; o < O; ++o) {
      double acc = 0.0;
      const float* wr = w.proj_w.data() + o * D;
      for (int64_t i = 0; i < D; ++i) acc += static_cast<double>(h[static_cast<size_t>(i)]) * wr[i];
      out[static_cast<size_t>(r * O + o)] = RoundBf16(static_cast<float>(acc));
    }
  }
  return out;
}

}  // namespace r4dx::vision
