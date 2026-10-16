#include "lfm2_model.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

// Below this temperature the device summary is skipped (the full-logits path takes the draw), as GemmaModel / Model.
constexpr float kMinSummaryTemperature = 0.01f;

float SummaryInvTemperature(const kernels::SampleParams& params) {
  if (!(params.temperature > 0.0f)) return 0.0f;
  if (params.temperature < kMinSummaryTemperature) return 0.0f;
  const float inv = 1.0f / params.temperature;
  if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity())) return 0.0f;
  return inv;
}

int64_t ResolveRealVocab(const Lfm2ModelOptions& o, int64_t vocab) {
  int64_t rv = o.real_vocab;
  if (const char* e = std::getenv("R4DX_LFM2_REAL_VOCAB"); e != nullptr && *e != '\0') rv = std::stoll(e);
  if (rv < 0) rv = vocab == 128000 ? kLfm2RealVocab : 0;
  if (rv >= vocab) rv = 0;  // nothing to mask
  return rv;
}

constexpr uint32_t kNegInfBits = 0xFF800000u;

}  // namespace

// ---- load -----------------------------------------------------------------------------------------------------------

Lfm2Model Lfm2Model::Load(const Lfm2ModelOptions& opts) {
  if (opts.prefill_chunk < 1 || opts.prefill_chunk > 256) {
    throw std::invalid_argument("Lfm2Model::Load: prefill_chunk must be in [1, 256]");
  }
  const bool is_tp = opts.tp_world > 1;
  if (opts.tp_world < 1 || opts.tp_world > 2 || opts.tp_rank < 0 || opts.tp_rank >= opts.tp_world) {
    throw std::invalid_argument("Lfm2Model::Load: tp_world must be 1 or 2 and 0 <= tp_rank < tp_world, got " +
                                std::to_string(opts.tp_world) + ", " + std::to_string(opts.tp_rank));
  }
  if (!is_tp && (opts.tp_comm != nullptr || opts.tp_submit_layers != 0 || opts.tp_max_inflight_units != 0)) {
    throw std::invalid_argument("Lfm2Model::Load: tp_comm / tp_submit_layers / tp_max_inflight_units are tensor-parallel "
                                "options (tp_world > 1 only)");
  }
  if (is_tp && (opts.tp_comm == nullptr || opts.tp_comm->World() != opts.tp_world || opts.tp_comm->Rank() != opts.tp_rank)) {
    throw std::invalid_argument("Lfm2Model::Load: tp_world > 1 needs a TpComm endpoint of the same world and rank");
  }
  if (is_tp) RefuseLfm2TrellisTp(opts.container_path, opts.layout, opts.layer_limit, "Lfm2Model::Load");
  if (is_tp) RefuseLfm2W8a16Tp(opts.layout, "Lfm2Model::Load");
  if (opts.tp_submit_layers < 0 || opts.tp_submit_layers > 64 || opts.tp_max_inflight_units < 0 ||
      opts.tp_max_inflight_units > 64) {
    throw std::invalid_argument("Lfm2Model::Load: tp_submit_layers and tp_max_inflight_units must be in [0, 64]");
  }
  // The per-rank tuning table for a TP rank's thread (docs/tp.md 2.7), cleared again if this load throws.
  SetTp2TuningForThisThread(is_tp);
  struct Tp2FlagOnThrow {
    int uncaught = std::uncaught_exceptions();
    ~Tp2FlagOnThrow() {
      if (std::uncaught_exceptions() > uncaught) SetTp2TuningForThisThread(false);
    }
  } tp2_flag_on_throw;

  Lfm2Model m;
  m.opts_ = opts;
  m.comm_ = opts.tp_comm;
  Lfm2LoadOptions lo;
  lo.layout = opts.layout;
  lo.lm_head_layout = opts.layout;
  lo.layer_limit = opts.layer_limit;
  lo.tp_world = opts.tp_world;
  lo.tp_rank = opts.tp_rank;
  m.container_ = Lfm2Container::Load(opts.container_path, lo);
  const Lfm2Config& cfg = m.container_.Config();  // rank-local
  const int64_t hidden = cfg.hidden_size;
  const int64_t layers = m.container_.NumLoadedLayers();
  m.max_ctx_ = cfg.ResolveMaxCtx(opts.max_ctx);
  m.max_chunk_ = opts.prefill_chunk;
  m.vocab_local_ = m.container_.LmHead().N;
  m.vocab_offset_ = cfg.VocabShardBegin();
  if (m.vocab_local_ * opts.tp_world != cfg.vocab_size) {
    throw std::runtime_error("Lfm2Model::Load: lm_head has " + std::to_string(m.vocab_local_) +
                             " rows, not vocab_size / tp_world = " + std::to_string(cfg.vocab_size / opts.tp_world));
  }
  m.real_vocab_ = ResolveRealVocab(opts, cfg.vocab_size);

  const int64_t rows = m.max_chunk_;
  // The warm-up's forced-routing MoE pass runs 64 rows, so the pinned readback holds at least that many.
  const int64_t moe_rows = std::max<int64_t>(rows, 64);
  m.ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows));
  m.ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.positions_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(moe_rows));
  m.buf_h_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(moe_rows * hidden));
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(m.vocab_local_));
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  if (is_tp) {
    m.argmax_pair_dev_ = core::DeviceBuffer<int32_t>(2);
    m.topk_lse_ws_ = core::DeviceBuffer<uint8_t>(static_cast<size_t>(r4dx_topk_lse_workspace_bytes()));
    if (opts.tp_submit_layers > 0) {
      m.submit_ = tp::SubmitBounder(opts.tp_max_inflight_units);
      m.submit_layers_ = opts.tp_submit_layers;
    }
  }
  m.summary_ids_dev_ = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
  m.summary_vals_dev_ = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
  m.summary_lse_dev_ = core::DeviceBuffer<float>(1);
  m.moe_host_.Allocate(moe_rows, cfg.num_experts_per_tok, cfg.num_experts);
  // The widest 256-row layer scratch: the MoE sorted path (x gathered 1024 x 2048 + slot outputs 1024 x 2048 = 8 MiB,
  // gate_up 256 x 3072) or the dense gate_up 256 x 23552 (12 MiB) + act, plus attention q/k/v/out and GEMM scratch.
  // A trellis body adds per GEMM call its input transform (parts x 64 x K f16: 512 KiB for an expert gate_up, 1.5 MiB
  // for the dense down) and split-group partials (< 1 MiB at the fallback tunings); Lfm2MoeFfn rewinds the arena after
  // each expert, so that is one expert's worth at a time, well inside the reservation.
  m.arena_.Reserve(256ull * 1024 * 1024);

  // Attention layers: a contiguous bf16 cache each; conv layers: a slot in the ShortConv state.
  m.attn_.resize(static_cast<size_t>(layers));
  m.kv_.resize(static_cast<size_t>(layers));
  m.conv_slot_.assign(static_cast<size_t>(layers), -1);
  for (int64_t i = 0; i < layers; ++i) {
    if (cfg.IsAttentionLayer(i)) {
      const int kvh = static_cast<int>(cfg.num_key_value_heads), hd = static_cast<int>(cfg.head_dim);
      m.kv_[static_cast<size_t>(i)].emplace(attention::GemmaKvCache::Contiguous(kvh, hd, m.max_ctx_,
                                                                                attention::GemmaKvDtype::kBf16));
      m.kv_bytes_ += static_cast<int64_t>(m.kv_[static_cast<size_t>(i)]->Bytes());
      attention::Lfm2AttnConfig ac;
      ac.hidden = static_cast<int>(hidden);
      ac.heads = static_cast<int>(cfg.num_attention_heads);
      ac.kv_heads = kvh;
      ac.head_dim = hd;
      ac.rope_theta = static_cast<float>(cfg.rope_theta);
      ac.rms_eps = static_cast<float>(cfg.norm_eps);
      m.attn_[static_cast<size_t>(i)].emplace(ac);
    } else {
      m.conv_slot_[static_cast<size_t>(i)] = m.n_conv_++;
    }
  }
  const size_t state_elems = static_cast<size_t>(std::max<int64_t>(m.n_conv_, 1) * 2 * cfg.conv_dim);
  m.conv_state_ = core::DeviceBuffer<uint16_t>(state_elems);
  m.conv_state_.Zero();
  if (opts.prompt_checkpoint) {
    m.conv_state_ckpt_ = core::DeviceBuffer<uint16_t>(state_elems);
    m.conv_state_ckpt_.Zero();
  }
  m.stream_.Synchronize();
  std::cerr << "[r4dx::model::Lfm2Model] " << layers << " layers (" << m.n_conv_ << " conv), max_ctx " << m.max_ctx_
            << ", KV bf16 = " << (static_cast<double>(m.kv_bytes_) / (1024.0 * 1024.0 * 1024.0))
            << " GiB, attention reference, residual bf16, MoE staged (host-sync)"
            << (m.container_.HasTrellis() ? ", trellis body" : "") << ", logit mask "
            << (m.real_vocab_ > 0 ? ">= " + std::to_string(m.real_vocab_) : std::string("off"));
  if (is_tp) std::cerr << ", TP rank " << opts.tp_rank << "/" << opts.tp_world << " (vocab shard " << m.vocab_local_ << ")";
  std::cerr << "\n";
  return m;
}

// ---- sequence state -------------------------------------------------------------------------------------------------

void Lfm2Model::Reset() {
  // KV is positional (every write lands before anything reads it); the ShortConv state is recurrent and must be zero
  // (HF's zero causal padding) for a new sequence.
  container_.ZeroTrellisTickets(stream_.get());
  conv_state_.ZeroAsync(stream_);
  stream_.Synchronize();
  pos_ = 0;
  started_ = false;
  arena_.Reset();
  ckpt_pos_ = -1;
  submit_.Reset();
}

void Lfm2Model::SaveCheckpoint() {
  if (!opts_.prompt_checkpoint) throw std::runtime_error("Lfm2Model::SaveCheckpoint: loaded without prompt_checkpoint");
  if (!started_) throw std::runtime_error("Lfm2Model::SaveCheckpoint: nothing has been fed yet");
  stream_.Synchronize();
  R4DX_HIP_CHECK(hipMemcpy(conv_state_ckpt_.data(), conv_state_.data(), conv_state_.bytes(), hipMemcpyDeviceToDevice));
  ckpt_pos_ = pos_;
}

void Lfm2Model::RestoreCheckpoint() {
  if (!opts_.prompt_checkpoint) throw std::runtime_error("Lfm2Model::RestoreCheckpoint: loaded without prompt_checkpoint");
  if (ckpt_pos_ < 0) throw std::runtime_error("Lfm2Model::RestoreCheckpoint: no checkpoint since Load()/Reset()");
  stream_.Synchronize();
  R4DX_HIP_CHECK(hipMemcpy(conv_state_.data(), conv_state_ckpt_.data(), conv_state_.bytes(), hipMemcpyDeviceToDevice));
  pos_ = ckpt_pos_;
  started_ = true;
  arena_.Reset();
}

uint16_t* Lfm2Model::ConvState(int64_t layer) {
  const int64_t slot = conv_slot_.at(static_cast<size_t>(layer));
  if (slot < 0) throw std::logic_error("Lfm2Model::ConvState: layer " + std::to_string(layer) + " is not a conv layer");
  return conv_state_.data() + slot * 2 * container_.Config().conv_dim;
}

void Lfm2Model::UploadPositions(int64_t start_pos, int64_t T) {
  std::vector<int32_t> h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) h[static_cast<size_t>(t)] = static_cast<int32_t>(start_pos + t);
  positions_dev_.CopyFromHost(h.data(), h.size());
}

void Lfm2Model::MaskPaddingLogits(float* logits_local, int64_t rows, int64_t row_stride) {
  if (real_vocab_ <= 0) return;
  const int64_t first = std::max<int64_t>(0, real_vocab_ - vocab_offset_);  // first masked LOCAL column
  if (first >= vocab_local_) return;
  for (int64_t r = 0; r < rows; ++r) {
    R4DX_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(logits_local + r * row_stride + first),
                                     static_cast<int>(kNegInfBits), static_cast<size_t>(vocab_local_ - first),
                                     stream_.get()));
  }
}

// ---- one decoder layer ----------------------------------------------------------------------------------------------

void Lfm2Model::RunFfn(int64_t i, const uint16_t* x_normed, uint16_t* out, int64_t T, SpanAccumulator* prof,
                       Lfm2MoePath path, Lfm2MoeDebug* dbg, const std::vector<int32_t>* forced_ids,
                       const std::vector<float>* forced_w) {
  const Lfm2Config& cfg = container_.Config();
  const Lfm2LayerWeights& lw = container_.Layer(i);
  if (lw.dense) {
    Lfm2DenseMlp(cfg, lw.mlp, comm_).Forward(stream_.get(), arena_, x_normed, out, T, prof);
  } else {
    Lfm2MoeFfn(cfg, lw.moe, &moe_host_, comm_)
        .Forward(stream_.get(), arena_, x_normed, out, T, prof, path, dbg, forced_ids, forced_w);
  }
}

void Lfm2Model::RunLayer(int64_t i, uint16_t* h, int64_t T, int64_t start_pos, SpanAccumulator* prof, Lfm2LayerDebug* dbg,
                         Lfm2MoePath moe_path) {
  const Lfm2Config& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);
  const Lfm2LayerWeights& lw = container_.Layer(i);
  const size_t n = static_cast<size_t>(T * hidden);
  const auto take = [&](std::vector<uint16_t>* dst, const uint16_t* src) {
    if (dbg == nullptr) return;
    stream_.Synchronize();
    dst->resize(n);
    R4DX_HIP_CHECK(hipMemcpy(dst->data(), src, n * sizeof(uint16_t), hipMemcpyDeviceToHost));
  };

  uint16_t* x = arena_.Alloc<uint16_t>(n, 16);
  uint16_t* sub = arena_.Alloc<uint16_t>(n, 16);
  // ---- mixer half ----
  ProfiledCall(prof, s_raw, "layer.operator_norm",
               [&] { r4dx_rmsnorm_plain_bf16(P(h), P(lw.operator_norm.data()), P(x), T, hidden, eps, 0, s); });
  take(dbg ? &dbg->operator_norm_out : nullptr, x);
  if (lw.attn_layer) {
    attention::Lfm2AttnLayerWeights aw;
    aw.q = &lw.attn.q;
    aw.k = &lw.attn.k;
    aw.v = &lw.attn.v;
    aw.o = &lw.attn.o;
    aw.q_norm = lw.attn.q_norm.data();
    aw.k_norm = lw.attn.k_norm.data();
    attn_[static_cast<size_t>(i)]->Forward(arena_, x, sub, aw, *kv_[static_cast<size_t>(i)], static_cast<int>(T),
                                           static_cast<int>(start_pos), positions_dev_.data(), s_raw, prof, comm_);
  } else {
    Lfm2ConvLayer(cfg, lw.conv, comm_).Forward(s_raw, arena_, x, sub, ConvState(i), T, prof);
  }
  take(dbg ? &dbg->mixer_out : nullptr, sub);
  ProfiledCall(prof, s_raw, "layer.residual_add", [&] { r4dx_residual_add_bf16(P(h), P(sub), P(h), T * hidden, s); });
  take(dbg ? &dbg->mid_residual : nullptr, h);
  // ---- FFN half ----
  ProfiledCall(prof, s_raw, "layer.ffn_norm",
               [&] { r4dx_rmsnorm_plain_bf16(P(h), P(lw.ffn_norm.data()), P(x), T, hidden, eps, 0, s); });
  take(dbg ? &dbg->ffn_norm_out : nullptr, x);
  RunFfn(i, x, sub, T, prof, moe_path, dbg ? &dbg->moe : nullptr);
  take(dbg ? &dbg->ffn_out : nullptr, sub);
  ProfiledCall(prof, s_raw, "layer.residual_add", [&] { r4dx_residual_add_bf16(P(h), P(sub), P(h), T * hidden, s); });
  arena_.Reset();
}

std::vector<uint16_t> Lfm2Model::DebugLayerForward(int64_t layer, const std::vector<uint16_t>& x_rows, int64_t T,
                                                   int64_t start_pos, Lfm2LayerDebug* dbg, Lfm2MoePath moe_path) {
  RequireNotTp("DebugLayerForward");
  const int64_t hidden = container_.Config().hidden_size;
  if (layer < 0 || layer >= container_.NumLoadedLayers()) throw std::out_of_range("Lfm2Model::DebugLayerForward: layer");
  if (T < 1 || T > max_chunk_ || static_cast<int64_t>(x_rows.size()) != T * hidden) {
    throw std::invalid_argument("Lfm2Model::DebugLayerForward: x_rows must be [T, hidden] with 1 <= T <= prefill_chunk");
  }
  if (start_pos < 0 || start_pos + T > max_ctx_) throw std::invalid_argument("Lfm2Model::DebugLayerForward: position");
  stream_.Synchronize();
  if (start_pos == 0 && conv_slot_[static_cast<size_t>(layer)] >= 0) {
    R4DX_HIP_CHECK(hipMemset(ConvState(layer), 0, static_cast<size_t>(2 * container_.Config().conv_dim) * sizeof(uint16_t)));
  }
  buf_h_.CopyFromHost(x_rows.data(), x_rows.size());
  UploadPositions(start_pos, T);
  RunLayer(layer, buf_h_.data(), T, start_pos, nullptr, dbg, moe_path);
  stream_.Synchronize();
  std::vector<uint16_t> out(x_rows.size());
  buf_h_.CopyToHost(out.data(), out.size());
  return out;
}

std::vector<uint16_t> Lfm2Model::DebugFfnForward(int64_t layer, const std::vector<uint16_t>& x_normed_rows, int64_t T,
                                                 Lfm2MoePath moe_path, Lfm2MoeDebug* dbg) {
  RequireNotTp("DebugFfnForward");
  const int64_t hidden = container_.Config().hidden_size;
  if (layer < 0 || layer >= container_.NumLoadedLayers()) throw std::out_of_range("Lfm2Model::DebugFfnForward: layer");
  if (T < 1 || T > max_chunk_ || static_cast<int64_t>(x_normed_rows.size()) != T * hidden) {
    throw std::invalid_argument("Lfm2Model::DebugFfnForward: rows must be [T, hidden] with 1 <= T <= prefill_chunk");
  }
  stream_.Synchronize();
  buf_h_.CopyFromHost(x_normed_rows.data(), x_normed_rows.size());
  uint16_t* out = arena_.Alloc<uint16_t>(x_normed_rows.size(), 16);
  RunFfn(layer, buf_h_.data(), out, T, nullptr, moe_path, dbg);
  stream_.Synchronize();
  std::vector<uint16_t> host(x_normed_rows.size());
  R4DX_HIP_CHECK(hipMemcpy(host.data(), out, host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
  arena_.Reset();
  return host;
}

std::vector<uint16_t> Lfm2Model::DebugConvState(int64_t layer) {
  stream_.Synchronize();
  std::vector<uint16_t> h(static_cast<size_t>(2 * container_.Config().conv_dim));
  R4DX_HIP_CHECK(hipMemcpy(h.data(), ConvState(layer), h.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
  return h;
}

// ---- one chunk ------------------------------------------------------------------------------------------------------

std::vector<float> Lfm2Model::RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                                       const SummaryRequest* summary_out, SpanAccumulator* prof, bool is_prefill) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Lfm2Model::RunChunk: token_ids.size() must be in [1, " + std::to_string(max_chunk_) + "]");
  }
  if (pos_ + T > max_ctx_) {
    throw std::runtime_error("Lfm2Model::RunChunk: position " + std::to_string(pos_ + T) + " exceeds max_ctx " +
                             std::to_string(max_ctx_));
  }
  const Lfm2Config& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const float eps = static_cast<float>(cfg.norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);
  for (int32_t id : token_ids) {
    if (id < 0 || id >= cfg.vocab_size) {
      throw std::invalid_argument("Lfm2Model::RunChunk: token id " + std::to_string(id) + " outside [0, " +
                                  std::to_string(cfg.vocab_size) + ")");
    }
  }

  // ---- embedding gather (device table, NO scale) ----
  std::copy(token_ids.begin(), token_ids.end(), ids_host_.data());
  ids_dev_.CopyFromHostAsync(ids_host_.data(), static_cast<size_t>(T), stream_);
  uint16_t* h = buf_h_.data();
  ProfiledCall(prof, s_raw, "embed", [&] {
    r4dx_embedding_gather_bf16(P(container_.EmbedTokensDevice().data()), P(ids_dev_.data()), P(h), T, hidden,
                               cfg.vocab_size, s);
  });
  // Positions (the device is idle here: the previous call ended synchronized; the copy below is a plain one).
  UploadPositions(pos_, T);

  // Bounded prefill submission under TP (tp/tp_submit.h), cut the same way on both ranks.
  const bool bounded = comm_ != nullptr && is_prefill && submit_.Active();
  int64_t unit_layers = bounded ? tp::UnitLayersForContext(submit_layers_, pos_ + T) : 0;
  if (bounded && T > 64) unit_layers = std::max<int64_t>(1, unit_layers * 64 / T);
  if (bounded) submit_.Reset();
  for (int64_t i = 0; i < num_layers; ++i) {
    if (bounded && i > 0 && i % unit_layers == 0) submit_.EndUnit(s_raw);
    RunLayer(i, h, T, pos_, prof);
  }

  // ---- final norm + tied lm_head on the LAST row ----
  if (want_logits) {
    const uint16_t* last = h + (T - 1) * hidden;
    uint16_t* xn = arena_.Alloc<uint16_t>(static_cast<size_t>(hidden), 16);
    uint16_t* logits_bf16 = arena_.Alloc<uint16_t>(static_cast<size_t>(vocab_local_), 16);
    ProfiledCall(prof, s_raw, "final_norm", [&] {
      r4dx_rmsnorm_plain_bf16(P(last), P(container_.FinalNorm().data()), P(xn), 1, hidden, eps, 0, s);
    });
    ProfiledCall(prof, s_raw, "gemm:lm_head", [&] { ApplyLinear(stream_, arena_, container_.LmHead(), xn, logits_bf16, 1); });
    ProfiledCall(prof, s_raw, "lm_head.widen", [&] {
      r4dx_model_widen_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), vocab_local_, s);
      MaskPaddingLogits(logits_dev_.data(), 1, vocab_local_);
    });
    if (greedy_out != nullptr && comm_ != nullptr) {
      r4dx_argmax_val_f32(P(logits_dev_.data()), P(argmax_pair_dev_.data()), P(argmax_pair_dev_.data() + 1), vocab_local_, s);
    } else if (greedy_out != nullptr) {
      r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), cfg.vocab_size, s);
    } else if (summary_out != nullptr) {
      r4dx_topk_lse_f32_ws(P(logits_dev_.data()), P(summary_ids_dev_.data()), P(summary_vals_dev_.data()),
                           P(summary_lse_dev_.data()), 1, vocab_local_, summary_out->inv_temperature, s,
                           comm_ != nullptr ? P(topk_lse_ws_.data()) : 0);
    }
    arena_.Reset();
  }
  if (prof == nullptr) stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits && prof == nullptr && comm_ != nullptr) {
    if (greedy_out != nullptr) {
      *greedy_out = MergeGreedyPair();
    } else if (summary_out != nullptr) {
      FetchRowSummary(summary_out->inv_temperature, summary_out->out);
      MergeSummary(summary_out->out);
    } else {
      GatherVocabRow(&logits);
    }
  } else if (want_logits && prof == nullptr) {
    if (greedy_out != nullptr) {
      argmax_dev_.CopyToHost(greedy_out, 1);
    } else if (summary_out != nullptr) {
      FetchRowSummary(summary_out->inv_temperature, summary_out->out);
    } else {
      logits.resize(static_cast<size_t>(cfg.vocab_size));
      logits_dev_.CopyToHost(logits.data(), logits.size());
    }
  }
  pos_ += T;
  started_ = true;
  return logits;
}

void Lfm2Model::FetchRowSummary(float inv_temperature, kernels::RowSummary* out) {
  std::vector<int32_t> ids(R4DX_TOPK_LSE_K);
  std::vector<float> vals(R4DX_TOPK_LSE_K);
  float lse = 0.0f;
  summary_ids_dev_.CopyToHost(ids.data(), ids.size());
  summary_vals_dev_.CopyToHost(vals.data(), vals.size());
  summary_lse_dev_.CopyToHost(&lse, 1);
  out->k = R4DX_TOPK_LSE_K;
  out->vocab = vocab_local_;
  out->inv_temperature = inv_temperature;
  out->lse = lse;
  for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
    out->ids[j] = ids[static_cast<size_t>(j)];
    out->vals[j] = vals[static_cast<size_t>(j)];
  }
}

// ---- public forward -------------------------------------------------------------------------------------------------

std::vector<float> Lfm2Model::Prefill(const std::vector<int32_t>& token_ids) {
  if (token_ids.empty()) throw std::runtime_error("Lfm2Model::Prefill: token_ids is empty");
  std::vector<float> logits;
  for (size_t off = 0; off < token_ids.size();) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                     token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const bool last = off + n == token_ids.size();
    off += n;
    std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr, /*is_prefill=*/true);
    if (last) logits = std::move(l);
  }
  return logits;
}

std::vector<float> Lfm2Model::DecodeStep(int32_t token_id) { return RunChunk({token_id}, true, nullptr, nullptr, nullptr); }

int32_t Lfm2Model::DecodeStepGreedy(int32_t token_id) {
  int32_t next = -1;
  RunChunk({token_id}, true, &next, nullptr, nullptr);
  return next;
}

int32_t Lfm2Model::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
  const float inv_t = SummaryInvTemperature(params);
  if (params.temperature <= 0.0f) return DecodeStepGreedy(token_id);
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
  ++sampled_fallback_rows_;
  if (comm_ != nullptr) {
    GatherVocabRow(&sampled_row_scratch_);
  } else {
    sampled_row_scratch_.resize(static_cast<size_t>(vocab));
    logits_dev_.CopyToHost(sampled_row_scratch_.data(), static_cast<size_t>(vocab));
  }
  return kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
}

// ---- tensor parallel ------------------------------------------------------------------------------------------------

void Lfm2Model::RequireNotTp(const char* what) const {
  if (comm_ != nullptr) {
    throw core::TpUnsupportedError(std::string("Lfm2Model::") + what + ": not supported under tensor parallelism");
  }
}

int32_t Lfm2Model::MergeGreedyPair() {
  tp::ArgmaxPair mine{};
  static_assert(sizeof(tp::ArgmaxPair) == 2 * sizeof(int32_t), "{idx, value} pair is two 32-bit words");
  R4DX_HIP_CHECK(hipMemcpy(&mine, argmax_pair_dev_.data(), sizeof(mine), hipMemcpyDeviceToHost));
  mine.idx += static_cast<int32_t>(vocab_offset_);
  std::array<tp::ArgmaxPair, 2> all{};
  comm_->HostAllGather(&mine, sizeof(mine), all.data());
  return tp::MergeArgmax(all.data(), comm_->World());
}

void Lfm2Model::MergeSummary(kernels::RowSummary* out) {
  static_assert(std::is_trivially_copyable_v<kernels::RowSummary>, "row summaries cross the host exchange as raw bytes");
  const int world = comm_->World();
  for (int j = 0; j < out->k; ++j) out->ids[j] += static_cast<int32_t>(vocab_offset_);
  std::array<kernels::RowSummary, 2> all{};
  comm_->HostAllGather(out, sizeof(kernels::RowSummary), all.data());
  *out = tp::MergeRowSummaries(all.data(), world, container_.Config().vocab_size);
}

void Lfm2Model::GatherVocabRow(std::vector<float>* full) {
  gather_shard_host_.resize(static_cast<size_t>(vocab_local_));
  R4DX_HIP_CHECK(hipMemcpy(gather_shard_host_.data(), logits_dev_.data(), static_cast<size_t>(vocab_local_) * sizeof(float),
                           hipMemcpyDeviceToHost));
  full->resize(static_cast<size_t>(container_.Config().vocab_size));
  comm_->HostAllGather(gather_shard_host_.data(), static_cast<size_t>(vocab_local_) * sizeof(float), full->data());
}

int64_t Lfm2Model::WarmupPositions(const Lfm2ModelOptions& o) {
  // TpWarmup: one full chunk, then (after a Reset) a 64-row chunk and three decode steps; the forced-routing MoE pass
  // is a DebugFfn-style call that does not use positions.
  return std::max<int64_t>(o.prefill_chunk, 64 + 3);
}

void Lfm2Model::TpWarmup() {
  if (comm_ == nullptr) throw std::logic_error("Lfm2Model::TpWarmup: only a tensor-parallel rank (tp_world > 1) warms up");
  const int64_t warm_positions = WarmupPositions(opts_);
  if (max_ctx_ < warm_positions) {
    throw std::invalid_argument("Lfm2Model::TpWarmup: max_ctx " + std::to_string(max_ctx_) + " < " +
                                std::to_string(warm_positions) + " warm-up positions");
  }
  const Lfm2Config& cfg = container_.Config();
  const auto ids = [&](int64_t n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = static_cast<int32_t>(i % cfg.vocab_size);
    return v;
  };
  (void)Prefill(ids(max_chunk_));  // the M = 256 GEMMs, the sliced all-reduces, the per-layer MoE readbacks
  Reset();
  // Every expert of the first MoE layer, through the sorted path, with forced routing: 64 rows x top_k pairs spread
  // round-robin, so each expert's gate_up / down runs (M = 64 * top_k / n_experts rows each) and no first-use GEMM
  // launch of an expert shape lands inside a request. Zero input rows, a lockstep collective (the MoE all-reduce):
  // every rank does the same.
  for (int64_t i = 0; i < container_.NumLoadedLayers(); ++i) {
    if (!cfg.IsMoeLayer(i)) continue;
    const int64_t T = 64, K = cfg.num_experts_per_tok, E = cfg.num_experts;
    // Pair p = t * K + j -> expert p % E: K consecutive (distinct) experts per row, every expert hit when T * K >= E.
    std::vector<int32_t> fid(static_cast<size_t>(T * K));
    std::vector<float> fw(static_cast<size_t>(T * K), 1.0f / static_cast<float>(K));
    for (int64_t p = 0; p < T * K; ++p) fid[static_cast<size_t>(p)] = static_cast<int32_t>(p % E);
    stream_.Synchronize();
    R4DX_HIP_CHECK(hipMemsetAsync(buf_h_.data(), 0, static_cast<size_t>(T * cfg.hidden_size) * sizeof(uint16_t), stream_.get()));
    uint16_t* out = arena_.Alloc<uint16_t>(static_cast<size_t>(T * cfg.hidden_size), 16);
    RunFfn(i, buf_h_.data(), out, T, nullptr, Lfm2MoePath::kSorted, nullptr, &fid, &fw);
    stream_.Synchronize();
    arena_.Reset();
    break;
  }
  (void)Prefill(ids(64));
  (void)DecodeStepGreedy(0);
  kernels::SampleParams sp;  // temperature 1: the summary path and its merge
  std::mt19937_64 rng(0x7e57);
  (void)DecodeStepSampled(1, sp, rng);
  (void)DecodeStep(2);  // the full-row gather
  Reset();
}

// ---- profiling ------------------------------------------------------------------------------------------------------

namespace {
std::vector<ProfileEntry> ToEntries(std::vector<SpanEntry> raw) {
  std::vector<ProfileEntry> out;
  out.reserve(raw.size());
  for (auto& e : raw) out.push_back({std::move(e.name), e.ms, e.count});
  return out;
}
}  // namespace

StepProfile Lfm2Model::DecodeStepProfiled(int32_t token_id) {
  RequireNotTp("DecodeStepProfiled");
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  r4dx_kernel_launch_counter_reset();
  SpanAccumulator acc;
  RunChunk({token_id}, true, nullptr, nullptr, &acc);
  const auto enq = Clock::now();
  StepProfile sp;
  sp.entries = ToEntries(acc.Finish());
  const auto fin = Clock::now();
  for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
  sp.host_enqueue_ms = std::chrono::duration<double, std::milli>(enq - t0).count();
  sp.finish_wait_ms = std::chrono::duration<double, std::milli>(fin - enq).count();
  sp.wall_ms = std::chrono::duration<double, std::milli>(fin - t0).count();
  sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();
  return sp;
}

StepProfile Lfm2Model::PrefillProfiled(const std::vector<int32_t>& token_ids) {
  RequireNotTp("PrefillProfiled");
  using Clock = std::chrono::steady_clock;
  if (token_ids.empty()) throw std::runtime_error("Lfm2Model::PrefillProfiled: token_ids is empty");
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
