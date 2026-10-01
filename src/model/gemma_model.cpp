#include "gemma_model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/rotate_residual.h"

namespace r4dx::model {

namespace {

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

// Below this temperature the device summary is skipped (the full-logits path takes the draw): the same
// guard Model::DecodeStepSampled applies (summary_sampler.hpp's tolerance was measured above it).
constexpr float kMinSummaryTemperature = 0.01f;

float SummaryInvTemperature(const kernels::SampleParams& params) {
  if (!(params.temperature > 0.0f)) return 0.0f;
  if (params.temperature < kMinSummaryTemperature) return 0.0f;
  const float inv = 1.0f / params.temperature;
  if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity())) return 0.0f;
  return inv;
}

float Bf16ToF32(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t F32ToBf16(float f) {  // RNE, finite inputs
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

}  // namespace

void ApplyGemmaEnv(GemmaModelOptions* o) {
  if (const char* e = std::getenv("R4DX_GEMMA_KV"); e != nullptr && *e != '\0') {
    if (std::strcmp(e, "fp8") == 0) o->kv = GemmaKvMode::kFp8;
    else if (std::strcmp(e, "bf16_full") == 0) o->kv = GemmaKvMode::kBf16Full;
    else if (std::strcmp(e, "bf16") == 0) o->kv = GemmaKvMode::kBf16;
    else throw std::invalid_argument(std::string("R4DX_GEMMA_KV='") + e + "' (want fp8|bf16_full|bf16)");
  }
  if (const char* e = std::getenv("R4DX_GEMMA_ATTN"); e != nullptr && *e != '\0') {
    o->attn = attention::ParseGemmaAttnBackend(e);
  }
  if (const char* e = std::getenv("R4DX_GEMMA_RESID"); e != nullptr && *e != '\0') {
    if (std::strcmp(e, "fp32") == 0) o->resid = GemmaResid::kFp32;
    else if (std::strcmp(e, "bf16") == 0) o->resid = GemmaResid::kBf16;
    else throw std::invalid_argument(std::string("R4DX_GEMMA_RESID='") + e + "' (want fp32|bf16)");
  }
}

GemmaModel GemmaModel::Load(const GemmaModelOptions& opts) {
  if (opts.prefill_chunk < 1 || opts.prefill_chunk > 256) {
    throw std::invalid_argument("GemmaModel::Load: prefill_chunk must be in [1, 256] (the sliding ring holds window + 288)");
  }
  GemmaModel m;
  m.opts_ = opts;
  GemmaLoadOptions lo;
  lo.layout = opts.layout;
  lo.lm_head_layout = opts.layout;
  lo.layer_limit = opts.layer_limit;
  m.container_ = GemmaContainer::Load(opts.container_path, lo);
  const GemmaConfig& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t layers = m.container_.NumLoadedLayers();
  m.max_ctx_ = cfg.ResolveMaxCtx(opts.max_ctx, opts.allow_extended_ctx);
  m.max_chunk_ = opts.prefill_chunk;
  if (m.container_.LmHead().N != cfg.vocab_size) throw std::runtime_error("GemmaModel::Load: lm_head rows != vocab_size");

  // Per-chunk buffers.
  const int64_t rows = m.max_chunk_;
  m.ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows));
  m.ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.positions_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.ring_slots_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.seqused_dev_ = core::DeviceBuffer<int32_t>(1);
  // Residual stream: fp32 (4 B/elem, rows x hidden = 3.9 MB at 256 rows) by default, bf16 for A/B (R4DX_GEMMA_RESID).
  if (opts.resid == GemmaResid::kFp32) m.buf_a32_ = core::DeviceBuffer<float>(static_cast<size_t>(rows * hidden));
  else m.buf_a_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * hidden));
  m.buf_normed_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * hidden));
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(cfg.vocab_size));
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  m.summary_ids_dev_ = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
  m.summary_vals_dev_ = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
  m.summary_lse_dev_ = core::DeviceBuffer<float>(1);
  // A 256-row layer's widest scratch: gate_up 15.7 MB + h 7.9 MB + the GEMM activation scratch (the trellis
  // M = 256 path's partials ~ 90 MB) + attention q / out; 256 MiB leaves ample headroom.
  m.arena_.Reserve(256ull * 1024 * 1024);

  // KV: the ring geometry (window + a chunk of up to 288 rows, the plan's 1536-token ring) and the one block
  // table every sliding layer shares; a contiguous cache for each full layer.
  m.geo_.emplace(static_cast<int>(cfg.sliding_window), /*max_chunk_rows=*/288, attention::GemmaKvCache::kBlockSize,
                 static_cast<int>(m.max_ctx_));
  m.ring_table_.emplace(*m.geo_);
  m.kv_.resize(static_cast<size_t>(layers));
  m.kv_ckpt_.resize(static_cast<size_t>(layers));
  for (int64_t i = 0; i < layers; ++i) {
    const bool full = cfg.IsFullLayer(i);
    const attention::GemmaKvDtype dt = (opts.kv == GemmaKvMode::kBf16 || (opts.kv == GemmaKvMode::kBf16Full && full))
                                           ? attention::GemmaKvDtype::kBf16
                                           : attention::GemmaKvDtype::kFp8;
    const int kvh = static_cast<int>(cfg.NumKvHeads(i)), hd = static_cast<int>(cfg.HeadDim(i));
    if (full) {
      m.kv_[static_cast<size_t>(i)].emplace(attention::GemmaKvCache::Contiguous(kvh, hd, m.max_ctx_, dt));
    } else {
      m.kv_[static_cast<size_t>(i)].emplace(attention::GemmaKvCache::Ring(kvh, hd, *m.geo_, m.ring_table_->Data(),
                                                                          m.ring_table_->Entries(), dt));
      if (opts.prompt_checkpoint) {
        m.kv_ckpt_[static_cast<size_t>(i)].emplace(attention::GemmaKvCache::Ring(
            kvh, hd, *m.geo_, m.ring_table_->Data(), m.ring_table_->Entries(), dt));
      }
    }
    m.kv_bytes_ += static_cast<int64_t>(m.kv_[static_cast<size_t>(i)]->Bytes());
    attention::GemmaAttnConfig ac;
    ac.hidden = static_cast<int>(hidden);
    ac.heads = static_cast<int>(cfg.num_attention_heads);
    ac.kv_heads = kvh;
    ac.head_dim = hd;
    ac.rope_pairs = static_cast<int>(cfg.RotaryAngles(i));
    ac.rope_freq_dim = hd;
    ac.rope_theta = static_cast<float>(cfg.RopeTheta(i));
    ac.rms_eps = static_cast<float>(cfg.rms_norm_eps);
    ac.window = full ? 0 : static_cast<int>(cfg.sliding_window);
    ac.k_eq_v = full;
    ac.v_norm = cfg.v_norm_all_layers;
    m.attn_.emplace_back(ac, opts.attn);
  }
  m.stream_.Synchronize();
  std::cerr << "[r4dx::model::GemmaModel] " << layers << " layers, max_ctx " << m.max_ctx_ << ", KV "
            << (opts.kv == GemmaKvMode::kFp8 ? "fp8" : opts.kv == GemmaKvMode::kBf16Full ? "bf16 (full layers) + fp8 (sliding)" : "bf16")
            << " = " << (static_cast<double>(m.kv_bytes_) / (1024.0 * 1024.0 * 1024.0)) << " GiB, attention "
            << (opts.attn == attention::GemmaAttnBackend::kReference ? "reference" : "libr4d (sliding) + reference (full)")
            << (std::getenv("R4DX_GEMMA_KV") != nullptr && *std::getenv("R4DX_GEMMA_KV") != '\0'
                    ? " [R4DX_GEMMA_KV set]" : " [default; R4DX_GEMMA_KV=fp8 for fp8]")
            << (m.container_.HasRotation() ? ", rotated residual" : "") << (m.container_.HasTrellis() ? ", trellis body" : "")
            << ", residual "
            << (opts.resid == GemmaResid::kFp32 ? "fp32" : "bf16") << "\n";
  return m;
}

void GemmaModel::Reset() {
  // The KV caches have no length bookkeeping of their own: every write lands at its position before
  // anything can read it (a stale byte at or past the new position is unreachable), so the counters are the
  // reset. Trellis tickets are zeroed in case a faulted launch left them non-zero.
  container_.ZeroTrellisTickets(stream_.get());
  stream_.Synchronize();
  pos_ = 0;
  started_ = false;
  arena_.Reset();
  ckpt_pos_ = -1;
  feature_rows_ = 0;
}

void GemmaModel::SaveCheckpoint() {
  if (!opts_.prompt_checkpoint) throw std::runtime_error("GemmaModel::SaveCheckpoint: loaded without prompt_checkpoint");
  if (!started_) throw std::runtime_error("GemmaModel::SaveCheckpoint: nothing has been fed yet");
  stream_.Synchronize();
  // The 40 sliding rings are overwritten in place by later positions; the full layers are positional (a
  // restore rewinds pos_ and the next writes overwrite) so they need no copy.
  for (size_t i = 0; i < kv_.size(); ++i) {
    if (kv_ckpt_[i]) kv_ckpt_[i]->CopyFrom(*kv_[i]);
  }
  ckpt_pos_ = pos_;
}

void GemmaModel::RestoreCheckpoint() {
  if (!opts_.prompt_checkpoint) throw std::runtime_error("GemmaModel::RestoreCheckpoint: loaded without prompt_checkpoint");
  if (ckpt_pos_ < 0) throw std::runtime_error("GemmaModel::RestoreCheckpoint: no checkpoint since Load()/Reset()");
  stream_.Synchronize();
  for (size_t i = 0; i < kv_.size(); ++i) {
    if (kv_ckpt_[i]) kv_[i]->CopyFrom(*kv_ckpt_[i]);
  }
  pos_ = ckpt_pos_;
  started_ = true;
  arena_.Reset();
  feature_rows_ = 0;
}

void GemmaModel::AttachFeatureCapture(std::vector<int64_t> layers) {
  if (layers.empty()) {
    DetachFeatureCapture();
    return;
  }
  const int64_t n = container_.NumLoadedLayers();
  for (size_t i = 0; i < layers.size(); ++i) {
    if (layers[i] < 0 || layers[i] > n) throw std::out_of_range("GemmaModel::AttachFeatureCapture: layer index out of range");
    if (i > 0 && layers[i] <= layers[i - 1]) throw std::invalid_argument("GemmaModel::AttachFeatureCapture: layers must be strictly ascending");
  }
  feature_layers_ = std::move(layers);
  const size_t feat_elems = static_cast<size_t>(max_chunk_) * feature_layers_.size() * static_cast<size_t>(container_.Config().hidden_size);
  features_dev_.Resize(feat_elems);
  if (opts_.resid == GemmaResid::kFp32) features_f32_.Resize(feat_elems);
  feature_rows_ = 0;
}

void GemmaModel::DetachFeatureCapture() {
  feature_layers_.clear();
  features_dev_.Resize(0);
  features_f32_.Resize(0);
  feature_rows_ = 0;
}

void GemmaModel::RotateResidual(void* x, int64_t rows, bool inverse) {
  if (!container_.HasRotation() || rows <= 0) return;
  const RotationWeights& r = container_.Rotation();
  if (opts_.resid == GemmaResid::kFp32) {
    r4dx_rotate_residual_f32(P(x), rows, container_.Config().hidden_size, P(r.signs.data()), P(r.mix5.data()),
                             inverse ? 1 : 0, P(stream_.get()));
  } else {
    r4dx_rotate_residual_bf16(P(x), rows, container_.Config().hidden_size, P(r.signs.data()), P(r.mix5.data()),
                              inverse ? 1 : 0, P(stream_.get()));
  }
}

void GemmaModel::UploadChunkMeta(int64_t start_pos, int64_t T) {
  std::vector<int32_t> positions_h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(start_pos + t);
  positions_dev_.CopyFromHost(positions_h.data(), positions_h.size());
  const std::vector<int32_t> ring_slots = geo_->Slots(start_pos, static_cast<int>(T));
  ring_slots_dev_.CopyFromHost(ring_slots.data(), ring_slots.size());
  const int32_t seqused_h = static_cast<int32_t>(start_pos + T);
  seqused_dev_.CopyFromHost(&seqused_h, 1);
}

// One decoder layer over `cur` (the residual stream, updated in place) whose input_layernorm output `normed` is
// already computed. Leaves the NEXT layer's input norm in `normed` when has_next.
void GemmaModel::RunLayer(int64_t i, void* cur, uint16_t* normed, int64_t T, int64_t start_pos, bool has_next,
                          SpanAccumulator* prof) {
  const GemmaConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);
  const bool rotated = container_.HasRotation();
  const RotationWeights* rot = rotated ? &container_.Rotation() : nullptr;
  const bool rot_had = rotated && rot->spec.Hadamard();
  const GemmaLayerWeights& lw = container_.Layer(i);
  const bool f32 = opts_.resid == GemmaResid::kFp32;
  // Pre-norms read the residual (fp32 or bf16) and emit the bf16 GEMM input.
  const auto plain_norm = [&](const void* x, const uint16_t* w, uint16_t* out) {
    if (f32) r4dx_rmsnorm_plain_f32in_bf16(P(x), P(w), P(out), T, hidden, eps, s);
    else r4dx_rmsnorm_plain_bf16(P(x), P(w), P(out), T, hidden, eps, 0, s);
  };

  // ---- attention half ----
  attention::GemmaAttnWeights aw;
  aw.q = &lw.attn.q;
  aw.k = &lw.attn.k;
  aw.v = lw.full ? nullptr : &lw.attn.v;
  aw.o = &lw.attn.o;
  aw.q_norm = lw.attn.q_norm.data();
  aw.k_norm = lw.attn.k_norm.data();
  aw.k_descale = lw.attn.k_descale.data();
  aw.v_descale = lw.attn.v_descale.data();
  if (rot_had) aw.o_had_signs = lw.full ? rot->had_o_full_signs.data() : rot->had_o_signs.data();
  uint16_t* o_out = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hidden), 16);
  attn_[static_cast<size_t>(i)].Forward(arena_, normed, o_out, aw, *kv_[static_cast<size_t>(i)], static_cast<int>(T),
                                        static_cast<int>(start_pos), positions_dev_.data(),
                                        lw.full ? positions_dev_.data() : ring_slots_dev_.data(), seqused_dev_.data(),
                                        s_raw, prof);
  ProfiledCall(prof, s_raw, "layer.post_attn", [&] {
    if (rotated) {
      if (f32) {
        r4dx_post_rmsnorm_rotate_add_f32res(P(cur), P(o_out), P(lw.post_attention_layernorm.data()), P(rot->signs.data()),
                                            P(rot->mix5.data()), T, hidden, eps, 1.0f, s);
      } else {
        r4dx_post_rmsnorm_rotate_add_bf16(P(cur), P(o_out), P(lw.post_attention_layernorm.data()), P(rot->signs.data()),
                                          P(rot->mix5.data()), T, hidden, eps, 1.0f, s);
      }
      plain_norm(cur, lw.pre_feedforward_layernorm.data(), normed);
    } else if (f32) {
      r4dx_gemma_postnorm_residual_rmsnorm_f32res(P(cur), P(o_out), P(lw.post_attention_layernorm.data()),
                                                  P(lw.pre_feedforward_layernorm.data()), eps, P(cur), P(normed),
                                                  1.0f, T, hidden, s);
    } else {
      // h = h + post_attention_layernorm(o); normed = pre_feedforward_layernorm(h). scalar 1.0: layer_scalar
      // is applied once, after the MLP residual add.
      r4dx_gemma_postnorm_residual_rmsnorm_bf16(P(cur), P(o_out), P(lw.post_attention_layernorm.data()),
                                                P(lw.pre_feedforward_layernorm.data()), eps, P(cur), P(normed),
                                                1.0f, T, hidden, s);
    }
  });

  // ---- MLP half ----
  uint16_t* mlp_out = o_out;  // the attention output is dead: reuse its arena slot
  GemmaMlp mlp(cfg, lw.mlp, rot_had ? rot->had_down_signs.data() : nullptr);
  mlp.Forward(stream_, arena_, normed, mlp_out, T, prof);
  ProfiledCall(prof, s_raw, "layer.post_mlp", [&] {
    if (rotated) {
      if (f32) {
        r4dx_post_rmsnorm_rotate_add_f32res(P(cur), P(mlp_out), P(lw.post_feedforward_layernorm.data()), P(rot->signs.data()),
                                            P(rot->mix5.data()), T, hidden, eps, lw.layer_scalar, s);
      } else {
        r4dx_post_rmsnorm_rotate_add_bf16(P(cur), P(mlp_out), P(lw.post_feedforward_layernorm.data()), P(rot->signs.data()),
                                          P(rot->mix5.data()), T, hidden, eps, lw.layer_scalar, s);
      }
      if (has_next) plain_norm(cur, container_.Layer(i + 1).input_layernorm.data(), normed);
    } else if (f32) {
      r4dx_gemma_postnorm_residual_rmsnorm_f32res(
          P(cur), P(mlp_out), P(lw.post_feedforward_layernorm.data()),
          has_next ? P(container_.Layer(i + 1).input_layernorm.data()) : 0, eps, P(cur), has_next ? P(normed) : 0,
          lw.layer_scalar, T, hidden, s);
    } else {
      // h = (h + post_feedforward_layernorm(m)) * layer_scalar; normed = next layer's input_layernorm(h).
      r4dx_gemma_postnorm_residual_rmsnorm_bf16(
          P(cur), P(mlp_out), P(lw.post_feedforward_layernorm.data()),
          has_next ? P(container_.Layer(i + 1).input_layernorm.data()) : 0, eps, P(cur), has_next ? P(normed) : 0,
          lw.layer_scalar, T, hidden, s);
    }
  });
  arena_.Reset();
}

std::vector<uint16_t> GemmaModel::DebugLayerForward(int64_t layer, const std::vector<uint16_t>& x_rows, int64_t T,
                                                    int64_t start_pos) {
  const int64_t hidden = container_.Config().hidden_size;
  if (layer < 0 || layer >= container_.NumLoadedLayers()) throw std::out_of_range("GemmaModel::DebugLayerForward: layer");
  if (T < 1 || T > max_chunk_ || static_cast<int64_t>(x_rows.size()) != T * hidden) {
    throw std::invalid_argument("GemmaModel::DebugLayerForward: x_rows must be [T, hidden] with T <= prefill_chunk");
  }
  stream_.Synchronize();
  const bool f32 = opts_.resid == GemmaResid::kFp32;
  void* cur = f32 ? static_cast<void*>(buf_a32_.data()) : static_cast<void*>(buf_a_.data());
  if (f32) {
    std::vector<float> xf(x_rows.size());
    for (size_t k = 0; k < xf.size(); ++k) xf[k] = Bf16ToF32(x_rows[k]);
    buf_a32_.CopyFromHost(xf.data(), xf.size());
  } else {
    buf_a_.CopyFromHost(x_rows.data(), x_rows.size());
  }
  UploadChunkMeta(start_pos, T);
  const GemmaLayerWeights& lw = container_.Layer(layer);
  const float eps = static_cast<float>(container_.Config().rms_norm_eps);
  if (f32) {
    r4dx_rmsnorm_plain_f32in_bf16(P(cur), P(lw.input_layernorm.data()), P(buf_normed_.data()), T, hidden, eps, P(stream_.get()));
  } else {
    r4dx_rmsnorm_plain_bf16(P(cur), P(lw.input_layernorm.data()), P(buf_normed_.data()), T, hidden, eps, 0, P(stream_.get()));
  }
  RunLayer(layer, cur, buf_normed_.data(), T, start_pos, /*has_next=*/false, nullptr);
  stream_.Synchronize();
  std::vector<uint16_t> out(x_rows.size());
  if (f32) {
    std::vector<float> of(x_rows.size());
    buf_a32_.CopyToHost(of.data(), of.size());
    for (size_t k = 0; k < of.size(); ++k) out[k] = F32ToBf16(of[k]);  // the layer output, rounded once for the API
  } else {
    buf_a_.CopyToHost(out.data(), out.size());
  }
  return out;
}
std::vector<float> GemmaModel::RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                                        const SummaryRequest* summary_out, SpanAccumulator* prof) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("GemmaModel::RunChunk: token_ids.size() must be in [1, " + std::to_string(max_chunk_) + "]");
  }
  if (pos_ + T > max_ctx_) {
    throw std::runtime_error("GemmaModel::RunChunk: position " + std::to_string(pos_ + T) + " exceeds max_ctx " +
                             std::to_string(max_ctx_));
  }
  const GemmaConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);
  const bool rotated = container_.HasRotation();
  const RotationWeights* rot = rotated ? &container_.Rotation() : nullptr;
  const bool rot_had = rotated && rot->spec.Hadamard();
  const bool capture = !feature_layers_.empty();

  // ---- scaled embedding gather (device table, always) ---------------------------------------------------
  std::copy(token_ids.begin(), token_ids.end(), ids_host_.data());
  ids_dev_.CopyFromHostAsync(ids_host_.data(), static_cast<size_t>(T), stream_);
  const bool f32 = opts_.resid == GemmaResid::kFp32;
  void* cur = f32 ? static_cast<void*>(buf_a32_.data()) : static_cast<void*>(buf_a_.data());
  ProfiledCall(prof, s_raw, "embed", [&] {
    if (f32) {
      r4dx_embedding_gather_scaled_f32(P(container_.EmbedTokensDevice().data()), P(ids_dev_.data()), P(cur), T, hidden,
                                       cfg.vocab_size, cfg.EmbedScale(), s);
    } else {
      r4dx_embedding_gather_scaled_bf16(P(container_.EmbedTokensDevice().data()), P(ids_dev_.data()), P(cur), T,
                                        hidden, cfg.vocab_size, cfg.EmbedScale(), s);
    }
  });
  // Residual rotation: x Q after the (text-row) scale, before layer 0 (docs/gemma4-plan.md 4.4).
  if (rotated) ProfiledCall(prof, s_raw, "rotate.entry", [&] { RotateResidual(cur, T, false); });

  // ---- per-chunk metadata (the device is idle here: the previous call ended synchronized) ------------------
  UploadChunkMeta(pos_, T);

  uint16_t* normed = buf_normed_.data();
  const size_t elem = f32 ? sizeof(float) : sizeof(uint16_t);
  const auto plain_norm = [&](const void* x, const uint16_t* w, uint16_t* out) {
    if (f32) r4dx_rmsnorm_plain_f32in_bf16(P(x), P(w), P(out), T, hidden, eps, s);
    else r4dx_rmsnorm_plain_bf16(P(x), P(w), P(out), T, hidden, eps, 0, s);
  };
  const auto capture_layer_input = [&](int64_t layer) {
    if (!capture) return;
    const auto it = std::find(feature_layers_.begin(), feature_layers_.end(), layer);
    if (it == feature_layers_.end()) return;
    const int64_t col = std::distance(feature_layers_.begin(), it);
    const int64_t ncols = static_cast<int64_t>(feature_layers_.size());
    // fp32 mode stages the capture in fp32 (rotated basis); it is un-rotated and narrowed once after the last layer.
    void* dst = f32 ? static_cast<void*>(features_f32_.data() + col * hidden)
                    : static_cast<void*>(features_dev_.data() + col * hidden);
    R4DX_HIP_CHECK(hipMemcpy2DAsync(dst, static_cast<size_t>(ncols * hidden) * elem, cur,
                                    static_cast<size_t>(hidden) * elem, static_cast<size_t>(hidden) * elem,
                                    static_cast<size_t>(T), hipMemcpyDeviceToDevice, s_raw));
  };

  // layer 0's input norm (every later layer's is fused into the previous layer's last kernel)
  ProfiledCall(prof, s_raw, "layer0.input_norm",
               [&] { plain_norm(cur, container_.Layer(0).input_layernorm.data(), normed); });

  for (int64_t i = 0; i < num_layers; ++i) {
    capture_layer_input(i);
    RunLayer(i, cur, normed, T, pos_, /*has_next=*/i + 1 < num_layers, prof);
  }
  capture_layer_input(num_layers);
  // Residual rotation exit: x Q^T before the final norm (and on the captured features).
  if (rotated) {
    ProfiledCall(prof, s_raw, "rotate.exit", [&] { RotateResidual(cur, T, true); });
    if (capture) {
      RotateResidual(f32 ? static_cast<void*>(features_f32_.data()) : static_cast<void*>(features_dev_.data()),
                     T * static_cast<int64_t>(feature_layers_.size()), true);
    }
  }
  if (capture && f32) {
    r4dx_f32_to_bf16(P(features_f32_.data()), P(features_dev_.data()),
                     T * static_cast<int64_t>(feature_layers_.size()) * hidden, s);
  }

  // ---- final norm + tied lm_head + softcap on the LAST row only ----------------------------------------------
  if (want_logits) {
    const char* last = static_cast<const char*>(cur) + (T - 1) * hidden * static_cast<int64_t>(elem);
    uint16_t* xn = arena_.Alloc<uint16_t>(static_cast<size_t>(hidden), 16);
    uint16_t* logits_bf16 = arena_.Alloc<uint16_t>(static_cast<size_t>(cfg.vocab_size), 16);
    ProfiledCall(prof, s_raw, "final_norm", [&] {
      plain_norm(last, container_.FinalNorm().data(), xn);
    });
    ProfiledCall(prof, s_raw, "gemm:lm_head", [&] { ApplyLinear(stream_, arena_, container_.LmHead(), xn, logits_bf16, 1); });
    ProfiledCall(prof, s_raw, "lm_head.softcap", [&] {
      if (cfg.final_logit_softcapping > 0.0) {
        r4dx_model_widen_softcap_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), cfg.vocab_size,
                                             static_cast<float>(cfg.final_logit_softcapping), s);
      } else {
        r4dx_model_widen_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), cfg.vocab_size, s);
      }
    });
    if (greedy_out != nullptr) {
      r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), cfg.vocab_size, s);
    } else if (summary_out != nullptr) {
      r4dx_topk_lse_f32(P(logits_dev_.data()), P(summary_ids_dev_.data()), P(summary_vals_dev_.data()),
                        P(summary_lse_dev_.data()), 1, cfg.vocab_size, summary_out->inv_temperature, s);
    }
    arena_.Reset();
  }
  // The plain (blocking) copies below and the next call's metadata uploads need an idle device.
  if (prof == nullptr) stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits && prof == nullptr) {
    if (greedy_out != nullptr) {
      argmax_dev_.CopyToHost(greedy_out, 1);
    } else if (summary_out != nullptr) {
      FetchRowSummary(summary_out->inv_temperature, summary_out->out);
    } else {
      logits.resize(static_cast<size_t>(cfg.vocab_size));
      logits_dev_.CopyToHost(logits.data(), logits.size());
    }
  }
  if (capture) {
    feature_rows_ = T;
    feature_pos_ = pos_;
  }
  pos_ += T;
  started_ = true;
  return logits;
}

void GemmaModel::FetchRowSummary(float inv_temperature, kernels::RowSummary* out) {
  std::vector<int32_t> ids(R4DX_TOPK_LSE_K);
  std::vector<float> vals(R4DX_TOPK_LSE_K);
  float lse = 0.0f;
  summary_ids_dev_.CopyToHost(ids.data(), ids.size());
  summary_vals_dev_.CopyToHost(vals.data(), vals.size());
  summary_lse_dev_.CopyToHost(&lse, 1);
  out->k = R4DX_TOPK_LSE_K;
  out->vocab = container_.Config().vocab_size;
  out->inv_temperature = inv_temperature;
  out->lse = lse;
  for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
    out->ids[j] = ids[static_cast<size_t>(j)];
    out->vals[j] = vals[static_cast<size_t>(j)];
  }
}

std::vector<float> GemmaModel::Prefill(const std::vector<int32_t>& token_ids,
                                       const std::function<void()>& on_chunk_captured) {
  if (token_ids.empty()) throw std::runtime_error("GemmaModel::Prefill: token_ids is empty");
  std::vector<float> logits;
  for (size_t off = 0; off < token_ids.size();) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                     token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const bool last = off + n == token_ids.size();
    off += n;
    std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr);
    if (on_chunk_captured) on_chunk_captured();
    if (last) logits = std::move(l);
  }
  return logits;
}

std::vector<float> GemmaModel::DecodeStep(int32_t token_id) { return RunChunk({token_id}, true, nullptr, nullptr, nullptr); }

int32_t GemmaModel::DecodeStepGreedy(int32_t token_id) {
  int32_t next = -1;
  RunChunk({token_id}, true, &next, nullptr, nullptr);
  return next;
}

int32_t GemmaModel::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
  const float inv_t = SummaryInvTemperature(params);
  if (params.temperature <= 0.0f) return DecodeStepGreedy(token_id);  // no draw, as Model's design point E
  const int64_t vocab = container_.Config().vocab_size;
  if (inv_t == 0.0f) {
    const std::vector<float> logits = DecodeStep(token_id);
    return kernels::SampleCanonical(logits.data(), vocab, params, kernels::DrawUniform01(rng));
  }
  kernels::RowSummary summary;
  const SummaryRequest req{inv_t, &summary};
  RunChunk({token_id}, true, nullptr, &req, nullptr);
  const double u = kernels::DrawUniform01(rng);
  const kernels::SummarySampleResult r = kernels::SampleFromSummary(summary, params, u);
  if (r.resolved) return r.token;
  // Unresolved row: the step's own fp32 row is still in logits_dev_ (no second lm_head pass).
  ++sampled_fallback_rows_;
  sampled_row_scratch_.resize(static_cast<size_t>(vocab));
  logits_dev_.CopyToHost(sampled_row_scratch_.data(), static_cast<size_t>(vocab));
  return kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
}

namespace {
std::vector<ProfileEntry> ToEntries(std::vector<SpanEntry> raw) {
  std::vector<ProfileEntry> out;
  out.reserve(raw.size());
  for (auto& e : raw) out.push_back({std::move(e.name), e.ms, e.count});
  return out;
}
}  // namespace

StepProfile GemmaModel::DecodeStepProfiled(int32_t token_id) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  r4dx_kernel_launch_counter_reset();
  SpanAccumulator acc;
  RunChunk({token_id}, true, nullptr, nullptr, &acc);  // a profiled call returns with the stream still busy
  const auto enq = Clock::now();
  StepProfile sp;
  sp.entries = ToEntries(acc.Finish());  // blocks until the GPU is idle
  const auto fin = Clock::now();
  for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
  sp.host_enqueue_ms = std::chrono::duration<double, std::milli>(enq - t0).count();
  sp.finish_wait_ms = std::chrono::duration<double, std::milli>(fin - enq).count();
  sp.wall_ms = std::chrono::duration<double, std::milli>(fin - t0).count();
  sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();
  return sp;
}

StepProfile GemmaModel::PrefillProfiled(const std::vector<int32_t>& token_ids) {
  using Clock = std::chrono::steady_clock;
  if (token_ids.empty()) throw std::runtime_error("GemmaModel::PrefillProfiled: token_ids is empty");
  const auto t0 = Clock::now();
  r4dx_kernel_launch_counter_reset();
  SpanAccumulator acc;
  std::vector<SpanEntry> merged;
  for (size_t off = 0; off < token_ids.size();) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                     token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    off += n;
    RunChunk(chunk, /*want_logits=*/false, nullptr, nullptr, &acc);
    // The next chunk's metadata uploads are plain copies: the device must be idle first (as Model does).
    for (SpanEntry& e : acc.Finish()) {
      auto it = std::find_if(merged.begin(), merged.end(), [&](const SpanEntry& m) { return m.name == e.name; });
      if (it == merged.end()) merged.push_back(std::move(e));
      else {
        it->ms += e.ms;
        it->count += e.count;
      }
    }
  }
  StepProfile sp;
  sp.entries = ToEntries(std::move(merged));
  for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
  sp.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();
  return sp;
}

}  // namespace r4dx::model
