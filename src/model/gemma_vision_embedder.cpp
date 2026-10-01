#include "gemma_vision_embedder.h"

#include <cstring>
#include <stdexcept>

#include "container_load_util.h"
#include "linear.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {

constexpr const char* kEmb = "vision.vision_embedder.";
constexpr const char* kProj = "vision.embed_vision.embedding_projection.weight";

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

uint16_t F32ToBf16(float f) {  // RNE, finite inputs (pixels are in [0, 1])
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

// Container bf16 tensors carry a trailing byte-pair dim: shape [..., 2] (r4dx-convert's add_bf16).
std::vector<int64_t> Dims(const r4dx_convert::SafetensorsReader& r, const std::string& name) {
  std::vector<int64_t> s = r.Meta(name).shape;
  if (s.empty() || s.back() != 2) {
    throw std::runtime_error("r4dx::model::GemmaContainer: '" + name + "' is not a bf16 passthrough tensor");
  }
  s.pop_back();
  return s;
}

std::string Str(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
  return out + "]";
}

void Need(const r4dx_convert::SafetensorsReader& r, const std::string& name, const std::vector<int64_t>& want,
          const std::string& path) {
  if (!r.Has(name)) {
    throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": vision container without '" + name +
                             "' (convert with r4dx-convert --vision on)");
  }
  const std::vector<int64_t> got = Dims(r, name);
  if (got != want) {
    throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": vision tensor '" + name + "' has shape " +
                             Str(got) + ", expected " + Str(want));
  }
}

}  // namespace

bool GemmaContainerHasVisionTensors(const r4dx_convert::SafetensorsReader& r) { return r.Has(kProj); }

GemmaVisionWeights LoadGemmaVisionWeights(const r4dx_convert::SafetensorsReader& r, int64_t hidden,
                                          const std::string& path) {
  using container_util::UploadRawU16;
  GemmaVisionWeights w;
  const std::string e = kEmb;
  // The dense weight fixes patch_dim and mm_dim; everything else is checked against them.
  if (!r.Has(e + "patch_dense.weight")) {
    throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": vision container without '" + e +
                             "patch_dense.weight' (convert with r4dx-convert --vision on)");
  }
  const std::vector<int64_t> dense = Dims(r, e + "patch_dense.weight");
  if (dense.size() != 2) throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": patch_dense.weight is not 2-D");
  w.mm_dim = dense[0];
  w.patch_dim = dense[1];
  w.out_dim = hidden;
  if (w.patch_dim != 6912 || w.mm_dim != hidden) {
    throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": patch_dense is " + Str(dense) +
                             ", expected [" + std::to_string(hidden) + ",6912] (48x48x3 patches, mm_embed_dim == hidden)");
  }
  const std::vector<int64_t> table = r.Has(e + "pos_embedding") ? Dims(r, e + "pos_embedding") : std::vector<int64_t>{};
  if (table.size() != 3 || table[1] != 2 || table[2] != w.mm_dim || table[0] < 1) {
    throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": vision pos_embedding must be [posemb, 2, " +
                             std::to_string(w.mm_dim) + "], got " + Str(table));
  }
  w.posemb_size = table[0];
  Need(r, e + "patch_dense.bias", {w.mm_dim}, path);
  Need(r, e + "patch_ln1.weight", {w.patch_dim}, path);
  Need(r, e + "patch_ln1.bias", {w.patch_dim}, path);
  Need(r, e + "patch_ln2.weight", {w.mm_dim}, path);
  Need(r, e + "patch_ln2.bias", {w.mm_dim}, path);
  Need(r, e + "pos_norm.weight", {w.mm_dim}, path);
  Need(r, e + "pos_norm.bias", {w.mm_dim}, path);
  Need(r, kProj, {hidden, w.mm_dim}, path);

  w.dense.layout = Layout::kBf16;
  w.dense.N = w.mm_dim;
  w.dense.K = w.patch_dim;
  w.dense.bf16_w = UploadRawU16(r, e + "patch_dense.weight");
  w.proj.layout = Layout::kBf16;
  w.proj.N = hidden;
  w.proj.K = w.mm_dim;
  w.proj.bf16_w = UploadRawU16(r, kProj);
  w.dense_b = UploadRawU16(r, e + "patch_dense.bias");
  w.ln1_w = UploadRawU16(r, e + "patch_ln1.weight");
  w.ln1_b = UploadRawU16(r, e + "patch_ln1.bias");
  w.ln2_w = UploadRawU16(r, e + "patch_ln2.weight");
  w.ln2_b = UploadRawU16(r, e + "patch_ln2.bias");
  w.pos_norm_w = UploadRawU16(r, e + "pos_norm.weight");
  w.pos_norm_b = UploadRawU16(r, e + "pos_norm.bias");
  w.pos_table = UploadRawU16(r, e + "pos_embedding");
  return w;
}

std::vector<int32_t> GemmaGridPositions(int64_t grid_h, int64_t grid_w) {
  std::vector<int32_t> pos(static_cast<size_t>(grid_h * grid_w) * 2);
  for (int64_t y = 0; y < grid_h; ++y) {
    for (int64_t x = 0; x < grid_w; ++x) {
      const size_t i = static_cast<size_t>(y * grid_w + x) * 2;
      pos[i] = static_cast<int32_t>(x);
      pos[i + 1] = static_cast<int32_t>(y);
    }
  }
  return pos;
}

void GemmaVisionEmbed(core::Stream& stream, core::Arena& arena, const GemmaVisionWeights& w, const float* pixels,
                      const int32_t* positions, int64_t n, core::DeviceBuffer<uint16_t>* out) {
  if (n < 1) throw std::invalid_argument("GemmaVisionEmbed: n must be >= 1");
  const int64_t Pd = w.patch_dim, D = w.mm_dim, O = w.out_dim;
  const hipStream_t s_raw = stream.get();
  const int64_t s = P(s_raw);
  arena.Reset();

  // Host staging: bf16 pixels, and the position-table taps (row x*2 of the [posemb, 2, D] table flattened to
  // [2*posemb, D] rows is axis 0 at column x; row y*2+1 is axis 1 at y), both with weight 1.0.
  std::vector<uint16_t> pix(static_cast<size_t>(n * Pd));
  for (size_t i = 0; i < pix.size(); ++i) pix[i] = F32ToBf16(pixels[i]);
  std::vector<int32_t> taps(static_cast<size_t>(n) * 2);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t x = positions[i * 2 + 0], y = positions[i * 2 + 1];
    if (x < 0 || y < 0 || x >= w.posemb_size || y >= w.posemb_size) {
      throw std::out_of_range("GemmaVisionEmbed: image position (" + std::to_string(x) + ", " + std::to_string(y) +
                              ") is outside the " + std::to_string(w.posemb_size) + "-entry position table");
    }
    taps[static_cast<size_t>(i) * 2 + 0] = static_cast<int32_t>(x * 2);
    taps[static_cast<size_t>(i) * 2 + 1] = static_cast<int32_t>(y * 2 + 1);
  }
  const std::vector<float> ones(static_cast<size_t>(n) * 2, 1.0f);

  uint16_t* d_pix = arena.Alloc<uint16_t>(static_cast<size_t>(n * Pd), 16);
  int32_t* d_taps = arena.Alloc<int32_t>(static_cast<size_t>(n) * 2, 16);
  float* d_ones = arena.Alloc<float>(static_cast<size_t>(n) * 2, 16);
  uint16_t* ln1 = arena.Alloc<uint16_t>(static_cast<size_t>(n * Pd), 16);
  uint16_t* h = arena.Alloc<uint16_t>(static_cast<size_t>(n * D), 16);
  R4DX_HIP_CHECK(hipMemcpyAsync(d_pix, pix.data(), pix.size() * 2, hipMemcpyHostToDevice, s_raw));
  R4DX_HIP_CHECK(hipMemcpyAsync(d_taps, taps.data(), taps.size() * 4, hipMemcpyHostToDevice, s_raw));
  R4DX_HIP_CHECK(hipMemcpyAsync(d_ones, ones.data(), ones.size() * 4, hipMemcpyHostToDevice, s_raw));

  // LN1 -> dense (+bias) -> LN2
  r4dx_layernorm_bf16(P(d_pix), P(w.ln1_w.data()), P(w.ln1_b.data()), P(ln1), n, Pd, w.ln_eps, s);
  ApplyLinear(s_raw, arena, w.dense, ln1, h, n);
  r4dx_bias_add_bf16(P(h), P(w.dense_b.data()), P(h), n, D, s);
  r4dx_layernorm_bf16(P(h), P(w.ln2_w.data()), P(w.ln2_b.data()), P(h), n, D, w.ln_eps, s);
  // h = bf16(h + bf16(table[x*2] + table[y*2+1]))
  r4dx_vision_pos_embed_bf16(P(w.pos_table.data()), P(d_taps), P(d_ones), P(h), /*out_f32=*/0, /*out_bf16=*/P(h), n,
                             D, /*taps=*/2, /*table_rows=*/w.posemb_size * 2, s);
  // pos_norm -> RMSNorm (no weight) -> projection
  r4dx_layernorm_bf16(P(h), P(w.pos_norm_w.data()), P(w.pos_norm_b.data()), P(h), n, D, w.ln_eps, s);
  r4dx_rmsnorm_noscale_bf16(P(h), P(h), n, D, w.rms_eps, s);
  out->Resize(static_cast<size_t>(n * O));
  ApplyLinear(s_raw, arena, w.proj, h, out->data(), n);
  stream.Synchronize();  // the host staging vectors and the arena scratch die here
  arena.Reset();
}

}  // namespace r4dx::model
