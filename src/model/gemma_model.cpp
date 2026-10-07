#include "gemma_model.h"

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

#include "dflash_draft_weights.h"
#include "kernels/model_kernels.h"
#include "linear.h"
#include "prefill_int8.h"  // R4DX_PREFILL_INT8: ignored here, said once at load
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/rotate_residual.h"
#include "gemma_vision.h"  // src/vision: image blocks, chunk planner, klimit_ext

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
  // ---- tensor parallel (docs/gemma4-plan.md M1b-1): validate this rank's options ----------------------------------------
  const bool is_tp = opts.tp_world > 1;
  if (opts.tp_world < 1 || opts.tp_world > 2 || opts.tp_rank < 0 || opts.tp_rank >= opts.tp_world) {
    throw std::invalid_argument("GemmaModel::Load: tp_world must be 1 or 2 and 0 <= tp_rank < tp_world, got " +
                                std::to_string(opts.tp_world) + ", " + std::to_string(opts.tp_rank));
  }
  if (!is_tp && (opts.tp_comm != nullptr || opts.tp_submit_layers != 0 || opts.tp_max_inflight_units != 0)) {
    throw std::invalid_argument("GemmaModel::Load: tp_comm / tp_submit_layers / tp_max_inflight_units are tensor-parallel "
                                "options (tp_world > 1 only)");
  }
  if (is_tp && (opts.tp_comm == nullptr || opts.tp_comm->World() != opts.tp_world || opts.tp_comm->Rank() != opts.tp_rank)) {
    throw std::invalid_argument("GemmaModel::Load: tp_world > 1 needs a TpComm endpoint of the same world and rank");
  }
  if (opts.tp_submit_layers < 0 || opts.tp_submit_layers > 64 || opts.tp_max_inflight_units < 0 ||
      opts.tp_max_inflight_units > 64) {
    throw std::invalid_argument("GemmaModel::Load: tp_submit_layers and tp_max_inflight_units must be in [0, 64]");
  }
  // A TP rank's thread consults the per-rank tuning table first (docs/tp.md 2.7); set on EVERY load so the flag follows this
  // thread's latest load, and cleared again if this one throws (Model::Load's Tp2FlagOnThrow).
  SetTp2TuningForThisThread(is_tp);
  struct Tp2FlagOnThrow {
    int uncaught = std::uncaught_exceptions();
    ~Tp2FlagOnThrow() {
      if (std::uncaught_exceptions() > uncaught) SetTp2TuningForThisThread(false);
    }
  } tp2_flag_on_throw;
  GemmaModel m;
  m.opts_ = opts;
  m.comm_ = opts.tp_comm;
  GemmaLoadOptions lo;
  lo.layout = opts.layout;
  lo.lm_head_layout = opts.layout;
  lo.layer_limit = opts.layer_limit;
  lo.tp_world = opts.tp_world;
  lo.tp_rank = opts.tp_rank;
  lo.vision = opts.vision;
  m.container_ = GemmaContainer::Load(opts.container_path, lo);
  const GemmaConfig& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t layers = m.container_.NumLoadedLayers();
  m.max_ctx_ = cfg.ResolveMaxCtx(opts.max_ctx, opts.allow_extended_ctx);
  m.max_chunk_ = opts.prefill_chunk;
  // This rank's lm_head rows and their first global id: the whole vocabulary and 0 at TP=1 (docs/tp.md 7.2).
  m.vocab_local_ = m.container_.LmHead().N;
  m.vocab_offset_ = cfg.VocabShardBegin();
  if (m.vocab_local_ * opts.tp_world != cfg.vocab_size) {
    throw std::runtime_error("GemmaModel::Load: lm_head has " + std::to_string(m.vocab_local_) +
                             " rows, not vocab_size / tp_world = " + std::to_string(cfg.vocab_size / opts.tp_world));
  }

  // Per-chunk buffers. With the vision embedder loaded they hold kGemmaVisionBufferRows (320) rows so that an
  // image block of up to 288 soft tokens is one chunk; without it they are exactly the prefill_chunk rows.
  m.buf_rows_ = m.container_.HasVision() ? std::max<int64_t>(m.max_chunk_, kGemmaVisionBufferRows) : m.max_chunk_;
  const int64_t rows = m.buf_rows_;
  m.ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows));
  m.ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.positions_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.ring_slots_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  if (m.container_.HasVision()) m.klimit_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.seqused_dev_ = core::DeviceBuffer<int32_t>(1);
  // Residual stream: fp32 (4 B/elem, rows x hidden = 3.9 MB at 256 rows) by default, bf16 for A/B (R4DX_GEMMA_RESID).
  if (opts.resid == GemmaResid::kFp32) m.buf_a32_ = core::DeviceBuffer<float>(static_cast<size_t>(rows * hidden));
  else m.buf_a_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * hidden));
  m.buf_normed_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * hidden));
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(m.vocab_local_));  // this rank's vocab shard under TP
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  if (is_tp) {
    // Allocated here, never inside a collective (docs/tp.md 6.3.7): the greedy {idx, value} pair, this rank's own
    // row-summary workspace (two rank threads -- one device under emulation -- must not share the module scratch), and the
    // prefill submission bounding's events.
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
  // ---- DFlash drafter (docs/gemma4-plan.md 6.4): its own weights + ring, fed by the feature capture ----------------
  if (opts.dflash_draft_k < 0) throw std::invalid_argument("GemmaModel::Load: dflash_draft_k must be >= 0");
  if (!opts.dflash_container.empty()) {
    if (opts.dflash_draft_k <= 0) {
      throw std::invalid_argument("GemmaModel::Load: dflash_container is set but dflash_draft_k <= 0");
    }
    if (m.max_chunk_ < 16) {
      throw std::invalid_argument("GemmaModel::Load: DFlash needs prefill_chunk >= 16 (the verify window / capture rows)");
    }
    const DflashDraftWeights peek = DflashDraftWeights::Open(opts.dflash_container);  // CPU: metadata + directory only
    const Dflash2Config& dc = peek.Config();
    if (dc.layout.empty()) {
      throw std::runtime_error("GemmaModel::Load: dflash container has no layout in its __metadata__.dflash2 block");
    }
    if (dc.hidden_size != hidden) {
      throw std::runtime_error("GemmaModel::Load: dflash container hidden_size " + std::to_string(dc.hidden_size) +
                               " != the target's " + std::to_string(hidden));
    }
    if (dc.vocab_size != cfg.vocab_size) {
      throw std::runtime_error("GemmaModel::Load: dflash container vocab_size " + std::to_string(dc.vocab_size) +
                               " != the target's " + std::to_string(cfg.vocab_size) + " (it shares the target embedding)");
    }
    if (opts.dflash_draft_k > 15) {
      throw std::invalid_argument("GemmaModel::Load: dflash_draft_k must be <= 15 (the verify window is k + 1 <= 16 rows)");
    }
    if (opts.dflash_draft_k > dc.block_size - 1) {
      throw std::invalid_argument("GemmaModel::Load: dflash_draft_k " + std::to_string(opts.dflash_draft_k) +
                                  " exceeds the drafter's block_size - 1 = " + std::to_string(dc.block_size - 1));
    }
    // target_layers are layer-INPUT indices (the converter stored z-lab's hidden_states[id + 1] as id + 1);
    // AttachFeatureCapture's own convention is the same: "the residual stream ENTERING layer L".
    for (int64_t l : dc.target_layers) {
      if (l < 0 || l > layers) {
        throw std::runtime_error("GemmaModel::Load: dflash target layer " + std::to_string(l) + " is outside [0, " +
                                 std::to_string(layers) + "]");
      }
    }
    DflashDraftOptions d;
    d.container_path = opts.dflash_container;
    d.layout = LayoutFromName(dc.layout);
    d.max_inject_rows = 64;
    // The TARGET lm_head's rows: the whole vocabulary at TP=1, this rank's slice under TP, where the drafter merges its
    // per-rank top-16s with the peer's (docs/tp.md 8.2) -- exactly Model::Load's wiring.
    d.lm_head_vocab = m.vocab_local_;
    if (is_tp) {
      d.vocab_offset = m.vocab_offset_;
      d.global_vocab = cfg.vocab_size;
      d.comm = m.comm_;
      d.shared_codebooks = opts.dflash_codebooks;
    }
    m.dflash_ = DflashDraft::Load(d);
    m.dflash_draft_k_ = opts.dflash_draft_k;
    m.dflash_embed_scale_ = dc.embed_scale;
    if (const char* e = std::getenv("R4DX_DFLASH_EMBED_SCALE"); e != nullptr && *e != '\0') {
      // The A/B knob of docs/gemma4-plan.md D-7: the drafter's embed scale {1 = raw rows (z-lab default), 62 =
      // Gemma's sqrt(hidden)-scaled rows} without reconverting. Not a flag: it never changes a shipped default.
      m.dflash_embed_scale_ = std::stod(e);
      if (!(m.dflash_embed_scale_ > 0.0)) throw std::invalid_argument("R4DX_DFLASH_EMBED_SCALE must be > 0");
    }
    m.AttachFeatureCapture(dc.target_layers);
    std::cerr << "[r4dx::model::GemmaModel] DFlash drafter loaded: " << opts.dflash_container << " (layout " << dc.layout
              << ", block " << dc.block_size << ", k " << opts.dflash_draft_k << ", target layers [";
    for (size_t i = 0; i < dc.target_layers.size(); ++i) std::cerr << (i ? "," : "") << dc.target_layers[i];
    std::cerr << "], softcap " << dc.logit_softcap << ", embed scale " << m.dflash_embed_scale_ << ")\n";
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
            << (opts.resid == GemmaResid::kFp32 ? "fp32" : "bf16");
  if (is_tp) std::cerr << ", TP rank " << opts.tp_rank << "/" << opts.tp_world << " (vocab shard " << m.vocab_local_ << ")";
  std::cerr << "\n";
  // R4DX_PREFILL_INT8 (docs/int8-prefill.md "Production path"): the Gemma 4 model has its own RunChunk, which does
  // not open the int8 scope (and no scale table is built), so the switch does nothing here -- say so once per process.
  if (PrefillInt8Request() == kPrefillInt8On) {
    static const bool once = [] {
      std::cerr << "[r4dx::model::GemmaModel] R4DX_PREFILL_INT8=1 ignored: the Gemma 4 model's prefill is not wired to the int8 "
                   "GEMM (Qwen only); its trellis linears run the f16 kernels\n";
      return true;
    }();
    (void)once;
  }
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
  submit_.Reset();  // no unit is outstanding after the synchronize above
  verified_rows_ = 0;
  if (dflash_.has_value()) dflash_->Reset();
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
  verified_rows_ = 0;
  if (dflash_.has_value() && dflash_->InjectedCount() > pos_) dflash_->Rewind(pos_);
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
  const size_t feat_elems = static_cast<size_t>(buf_rows_) * feature_layers_.size() * static_cast<size_t>(container_.Config().hidden_size);
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

void GemmaModel::UploadChunkMeta(int64_t start_pos, int64_t T, const std::vector<int32_t>* klimit_ext) {
  std::vector<int32_t> positions_h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(start_pos + t);
  positions_dev_.CopyFromHost(positions_h.data(), positions_h.size());
  const std::vector<int32_t> ring_slots = geo_->Slots(start_pos, static_cast<int>(T));
  ring_slots_dev_.CopyFromHost(ring_slots.data(), ring_slots.size());
  const int32_t seqused_h = static_cast<int32_t>(start_pos + T);
  seqused_dev_.CopyFromHost(&seqused_h, 1);
  if (klimit_ext != nullptr) {
    if (static_cast<int64_t>(klimit_ext->size()) != T || klimit_dev_.size() < static_cast<size_t>(T)) {
      throw std::runtime_error("GemmaModel: klimit_ext needs the vision buffers and exactly one entry per row");
    }
    klimit_dev_.CopyFromHost(klimit_ext->data(), klimit_ext->size());
  }
}

// One decoder layer over `cur` (the residual stream, updated in place) whose input_layernorm output `normed` is
// already computed. Leaves the NEXT layer's input norm in `normed` when has_next.
void GemmaModel::RunLayer(int64_t i, void* cur, uint16_t* normed, int64_t T, int64_t start_pos, bool has_next,
                          SpanAccumulator* prof, const int32_t* klimit_ext) {
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
                                        s_raw, prof, lw.full ? nullptr : klimit_ext,  // bidirectional: sliding only
                                        comm_);
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
  GemmaMlp mlp(cfg, lw.mlp, rot_had ? rot->had_down_signs.data() : nullptr, comm_);
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
                                                    int64_t start_pos, const std::vector<int32_t>* klimit_ext) {
  RequireNotTp("DebugLayerForward");
  const int64_t hidden = container_.Config().hidden_size;
  if (layer < 0 || layer >= container_.NumLoadedLayers()) throw std::out_of_range("GemmaModel::DebugLayerForward: layer");
  if (T < 1 || T > buf_rows_ || static_cast<int64_t>(x_rows.size()) != T * hidden) {
    throw std::invalid_argument("GemmaModel::DebugLayerForward: x_rows must be [T, hidden] with T <= the chunk buffer rows");
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
  UploadChunkMeta(start_pos, T, klimit_ext);
  // Basis-agnostic: `x_rows` and the result are in the ORIGINAL basis; a rotated container runs the layer on x Q
  // and the output is rotated back (so the HF layer goldens apply unchanged).
  RotateResidual(cur, T, /*inverse=*/false);
  const GemmaLayerWeights& lw = container_.Layer(layer);
  const float eps = static_cast<float>(container_.Config().rms_norm_eps);
  if (f32) {
    r4dx_rmsnorm_plain_f32in_bf16(P(cur), P(lw.input_layernorm.data()), P(buf_normed_.data()), T, hidden, eps, P(stream_.get()));
  } else {
    r4dx_rmsnorm_plain_bf16(P(cur), P(lw.input_layernorm.data()), P(buf_normed_.data()), T, hidden, eps, 0, P(stream_.get()));
  }
  RunLayer(layer, cur, buf_normed_.data(), T, start_pos, /*has_next=*/false, nullptr,
           klimit_ext != nullptr ? klimit_dev_.data() : nullptr);
  RotateResidual(cur, T, /*inverse=*/true);
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
                                        const SummaryRequest* summary_out, SpanAccumulator* prof, bool is_prefill,
                                        const ChunkVision* vision, std::vector<int32_t>* verify_argmax) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (verify_argmax != nullptr && (T > 16 || prof != nullptr || want_logits || greedy_out != nullptr || summary_out != nullptr ||
                                   vision != nullptr)) {
    throw std::logic_error("GemmaModel::RunChunk: a verify window is 1..16 rows, unprofiled, with no other logits request");
  }
  if (T < 1 || T > buf_rows_ || (vision == nullptr && T > max_chunk_)) {
    throw std::runtime_error("GemmaModel::RunChunk: token_ids.size() must be in [1, " +
                             std::to_string(vision != nullptr ? buf_rows_ : max_chunk_) + "]");
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
  // Audio soft-token rows (PrefillAudio): overwrite the gather result, unscaled, before the rotation.
  if (splice_spans_ != nullptr) {
    bool spliced = false;
    for (const AudioRowSpan& sp : *splice_spans_) {
      const int64_t lo = std::max<int64_t>(sp.offset, splice_chunk_off_);
      const int64_t hi = std::min<int64_t>(sp.offset + sp.tokens, splice_chunk_off_ + T);
      if (lo >= hi) continue;
      const int64_t rows = hi - lo;
      const uint16_t* src = sp.rows + (lo - sp.offset) * hidden;
      const int64_t dst_row = lo - splice_chunk_off_;
      if (f32) {
        std::vector<float> wide(static_cast<size_t>(rows * hidden));
        for (size_t k = 0; k < wide.size(); ++k) wide[k] = Bf16ToF32(src[k]);
        R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(cur) + dst_row * hidden * 4, wide.data(), wide.size() * 4,
                                      hipMemcpyHostToDevice, s_raw));
        stream_.Synchronize();  // `wide` is pageable host memory: finish the copy before it goes out of scope
      } else {
        R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(cur) + dst_row * hidden * 2, src,
                                      static_cast<size_t>(rows * hidden) * 2, hipMemcpyHostToDevice, s_raw));
        spliced = true;
      }
    }
    if (spliced) stream_.Synchronize();  // the bf16 path copies straight from the caller's rows
  }
  // ---- image rows over the gather result (UNSCALED: only text rows carry the sqrt(hidden) scale) -------------
  // The placeholder rows were gathered from the embedding table above (any in-vocab id is harmless) and are now
  // overwritten, as HF's masked_scatter does. bf16 [rows, hidden] -> the residual dtype, contiguous per block.
  if (vision != nullptr) {
    for (const ChunkVision::Splice& sp : vision->splices) {
      if (sp.row < 0 || sp.rows < 1 || sp.row + sp.rows > T || sp.embeds == nullptr) {
        throw std::runtime_error("GemmaModel::RunChunk: bad image splice");
      }
      if (sp.host) {
        // TP (docs/tp.md 8.3): the rows are host bf16 (GemmaTpModel's ImageRows::host, one copy every rank splices). Same
        // pageable-host path as the audio rows above: widen on the host for the fp32 residual, H2D, finish before the
        // buffer can go away. No device allocation, so it is legal inside a collective.
        const uint16_t* src = sp.embeds;
        const size_t n = static_cast<size_t>(sp.rows * hidden);
        if (f32) {
          std::vector<float> wide(n);
          for (size_t k = 0; k < n; ++k) wide[k] = Bf16ToF32(src[k]);
          R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(cur) + sp.row * hidden * 4, wide.data(), n * 4,
                                        hipMemcpyHostToDevice, s_raw));
        } else {
          R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(cur) + sp.row * hidden * 2, src, n * 2, hipMemcpyHostToDevice,
                                        s_raw));
        }
        stream_.Synchronize();
        continue;
      }
      ProfiledCall(prof, s_raw, "embed.image_splice", [&] {
        if (f32) {
          r4dx_model_widen_bf16_to_f32(P(sp.embeds), P(static_cast<float*>(cur) + sp.row * hidden), sp.rows * hidden, s);
        } else {
          R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<uint16_t*>(cur) + sp.row * hidden, sp.embeds,
                                        static_cast<size_t>(sp.rows * hidden) * sizeof(uint16_t),
                                        hipMemcpyDeviceToDevice, s_raw));
        }
      });
    }
  }
  // Residual rotation: x Q after the (text-row) scale and the audio / image splices, before layer 0 (plan 4.4).
  if (rotated) ProfiledCall(prof, s_raw, "rotate.entry", [&] { RotateResidual(cur, T, false); });

  // ---- per-chunk metadata (the device is idle here: the previous call ended synchronized) ------------------
  UploadChunkMeta(pos_, T, vision != nullptr && !vision->klimit_ext.empty() ? &vision->klimit_ext : nullptr);
  const int32_t* klimit_dev = vision != nullptr && !vision->klimit_ext.empty() ? klimit_dev_.data() : nullptr;

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

  // Tensor parallel: bounded submission of a prefill chunk (tp/tp_submit.h, docs/tp.md Appendix B N57): every `unit_layers`
  // layers force a submission and wait until at most max_inflight_units - 1 earlier units are unfinished. Decode steps are
  // not split. A 256-row chunk costs ~4x a 64-row one per layer, so its unit is scaled down (at least one layer), exactly as
  // Model::RunChunk does. Both ranks see the same positions, so they cut the same units.
  const bool bounded = comm_ != nullptr && is_prefill && submit_.Active();
  int64_t unit_layers = bounded ? tp::UnitLayersForContext(submit_layers_, pos_ + T) : 0;
  if (bounded && T > 64) unit_layers = std::max<int64_t>(1, unit_layers * 64 / T);
  if (bounded) submit_.Reset();
  for (int64_t i = 0; i < num_layers; ++i) {
    if (bounded && i > 0 && i % unit_layers == 0) submit_.EndUnit(s_raw);
    capture_layer_input(i);
    RunLayer(i, cur, normed, T, pos_, /*has_next=*/i + 1 < num_layers, prof, klimit_dev);
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

  // ---- verify window: final norm + lm_head + softcap + argmax on EVERY row (VerifyWindow) ---------------------------
  if (verify_argmax != nullptr) {
    const int64_t vocab = vocab_local_;  // this rank's lm_head rows (== vocab_size at TP=1)
    uint16_t* xn = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hidden), 16);
    uint16_t* logits_bf16 = arena_.Alloc<uint16_t>(static_cast<size_t>(T * vocab), 16);
    plain_norm(cur, container_.FinalNorm().data(), xn);  // T rows
    ApplyLinear(stream_, arena_, container_.LmHead(), xn, logits_bf16, T);
    if (cfg.final_logit_softcapping > 0.0) {
      r4dx_model_widen_softcap_bf16_to_f32(P(logits_bf16), P(verify_logits_dev_.data()), T * vocab,
                                           static_cast<float>(cfg.final_logit_softcapping), s);
    } else {
      r4dx_model_widen_bf16_to_f32(P(logits_bf16), P(verify_logits_dev_.data()), T * vocab, s);
    }
    for (int64_t t = 0; t < T; ++t) {
      if (comm_ != nullptr) {
        // TP: argmax this rank's slice of row t keeping the winning value; MergeVerifyPairs merges across ranks below.
        r4dx_argmax_val_f32(P(verify_logits_dev_.data() + t * vocab), P(verify_pair_dev_.data() + 2 * t),
                            P(verify_pair_dev_.data() + 2 * t + 1), vocab, s);
      } else {
        r4dx_argmax_f32(P(verify_logits_dev_.data() + t * vocab), P(verify_argmax_dev_.data() + t), vocab, s);
      }
    }
    arena_.Reset();
  }

  // ---- final norm + tied lm_head + softcap on the LAST row only ----------------------------------------------
  if (want_logits) {
    const char* last = static_cast<const char*>(cur) + (T - 1) * hidden * static_cast<int64_t>(elem);
    uint16_t* xn = arena_.Alloc<uint16_t>(static_cast<size_t>(hidden), 16);
    uint16_t* logits_bf16 = arena_.Alloc<uint16_t>(static_cast<size_t>(vocab_local_), 16);  // this rank's shard under TP
    ProfiledCall(prof, s_raw, "final_norm", [&] {
      plain_norm(last, container_.FinalNorm().data(), xn);
    });
    ProfiledCall(prof, s_raw, "gemm:lm_head", [&] { ApplyLinear(stream_, arena_, container_.LmHead(), xn, logits_bf16, 1); });
    ProfiledCall(prof, s_raw, "lm_head.softcap", [&] {
      if (cfg.final_logit_softcapping > 0.0) {
        r4dx_model_widen_softcap_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), vocab_local_,
                                             static_cast<float>(cfg.final_logit_softcapping), s);
      } else {
        r4dx_model_widen_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), vocab_local_, s);
      }
    });
    if (greedy_out != nullptr && comm_ != nullptr) {
      // TP (docs/tp.md 7.3): argmax THIS rank's shard, keeping the winning value too; the host merges the pairs below.
      r4dx_argmax_val_f32(P(logits_dev_.data()), P(argmax_pair_dev_.data()), P(argmax_pair_dev_.data() + 1), vocab_local_, s);
    } else if (greedy_out != nullptr) {
      r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), cfg.vocab_size, s);
    } else if (summary_out != nullptr) {
      // workspace 0 == the module scratch (exactly r4dx_topk_lse_f32); a TP rank uses its own.
      r4dx_topk_lse_f32_ws(P(logits_dev_.data()), P(summary_ids_dev_.data()), P(summary_vals_dev_.data()),
                           P(summary_lse_dev_.data()), 1, vocab_local_, summary_out->inv_temperature, s,
                           comm_ != nullptr ? P(topk_lse_ws_.data()) : 0);
    }
    arena_.Reset();
  }
  // The plain (blocking) copies below and the next call's metadata uploads need an idle device.
  if (prof == nullptr) stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits && prof == nullptr && comm_ != nullptr) {
    // Tensor parallel, H3 (docs/tp.md 6.2, 7.3-7.5): merge the vocab shards on the host -- the greedy pair (8 B per rank), the
    // row summary (536 B per rank) or the full row in global id order. Every rank ends with the same answer.
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
  if (verify_argmax != nullptr) {
    verify_argmax->resize(static_cast<size_t>(T));
    if (comm_ != nullptr) {
      MergeVerifyPairs(T, verify_argmax);  // collective: every rank reaches here with the same T
    } else {
      verify_argmax_dev_.CopyToHost(verify_argmax->data(), static_cast<size_t>(T));
    }
  }
  if (capture) {
    feature_rows_ = T;
    feature_pos_ = pos_;
  }
  if (verify_argmax != nullptr) {
    // A verify window commits nothing: pos_ stays put until CommitVerifiedWindow. Its K/V rows are written, and
    // (see VerifyWindow) the rejected ones are simply overwritten by whatever is fed next.
    started_ = true;
    return logits;
  }
  // Feed the drafter every committed row (a plain decode step, a prefill chunk). Skipped for a profiled call
  // (the stream is still busy there) and while injection is off (the drafter then sees a gap, later).
  if (capture && dflash_.has_value() && dflash_injection_enabled_ && prof == nullptr) InjectDflashRows(T, pos_);
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
  out->vocab = vocab_local_;  // the width the device summarized (this rank's shard under TP; MergeSummary makes it global)
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
    std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr, /*is_prefill=*/true);
    if (on_chunk_captured) on_chunk_captured();
    if (last) logits = std::move(l);
  }
  return logits;
}

std::vector<float> GemmaModel::PrefillAudio(const std::vector<int32_t>& token_ids, const std::vector<AudioRowSpan>& spans,
                                            const std::function<void()>& on_chunk_captured) {
  if (spans.empty()) return Prefill(token_ids, on_chunk_captured);
  if (token_ids.empty()) throw std::runtime_error("GemmaModel::PrefillAudio: token_ids is empty");
  const int64_t total = static_cast<int64_t>(token_ids.size());
  for (const AudioRowSpan& sp : spans) {
    if (sp.rows == nullptr || sp.tokens < 1 || sp.offset < 0 || sp.offset + sp.tokens > total) {
      throw std::runtime_error("GemmaModel::PrefillAudio: audio span [" + std::to_string(sp.offset) + ", +" +
                               std::to_string(sp.tokens) + ") is outside the " + std::to_string(total) + " tokens fed");
    }
  }
  struct SpliceGuard {  // the pointer must never outlive the caller's spans, even on a throw
    GemmaModel* m;
    ~SpliceGuard() { m->splice_spans_ = nullptr; m->splice_chunk_off_ = 0; }
  } guard{this};
  splice_spans_ = &spans;
  std::vector<float> logits;
  for (size_t off = 0; off < token_ids.size();) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                     token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const bool last = off + n == token_ids.size();
    splice_chunk_off_ = static_cast<int64_t>(off);
    off += n;
    std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr, /*is_prefill=*/true);
    if (on_chunk_captured) on_chunk_captured();
    if (last) logits = std::move(l);
  }
  return logits;
}

void GemmaModel::EncodeImages(const float* pixel_values, int64_t total_patches,
                              const std::vector<vision::GridThw>& grids, core::DeviceBuffer<uint16_t>* out) {
  if (!container_.HasVision()) {
    throw std::runtime_error("GemmaModel::EncodeImages: no vision embedder loaded (container without vision.* "
                             "tensors, or loaded with vision off)");
  }
  if (grids.empty() || pixel_values == nullptr || out == nullptr) {
    throw std::invalid_argument("GemmaModel::EncodeImages: need pixel values, at least one grid and an output");
  }
  int64_t sum = 0;
  std::vector<int32_t> pos;
  for (const vision::GridThw& g : grids) {
    const int64_t n = g.h * g.w;
    if (g.t != 1 || g.h < 1 || g.w < 1) throw std::invalid_argument("GemmaModel::EncodeImages: bad grid");
    if (n > vision::kGemmaMaxImageBlockTokens) {
      throw std::invalid_argument("GemmaModel::EncodeImages: an image of " + std::to_string(n) + " soft tokens exceeds the " +
                                  std::to_string(vision::kGemmaMaxImageBlockTokens) +
                                  "-token block limit (one prefill chunk, sliding ring = window + 288); use at most "
                                  "280 soft tokens per image");
    }
    const std::vector<int32_t> p = GemmaGridPositions(g.h, g.w);
    pos.insert(pos.end(), p.begin(), p.end());
    sum += n;
  }
  if (sum != total_patches) throw std::invalid_argument("GemmaModel::EncodeImages: total_patches != sum of grid sizes");
  stream_.Synchronize();
  GemmaVisionEmbed(stream_, arena_, container_.Vision(), pixel_values, pos.data(), sum, out);
}

std::vector<float> GemmaModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                 const std::vector<ImageSpan>& images,
                                                 const std::function<void()>& on_chunk_captured) {
  if (images.empty()) return Prefill(token_ids, on_chunk_captured);
  if (token_ids.empty()) throw std::runtime_error("GemmaModel::PrefillMultimodal: token_ids is empty");
  if (!container_.HasVision()) {
    throw std::runtime_error("GemmaModel::PrefillMultimodal: image spans given but no vision embedder is loaded "
                             "(its buffers hold the 288-row image chunk)");
  }
  const int64_t total = static_cast<int64_t>(token_ids.size());
  const int32_t image_id = static_cast<int32_t>(container_.Info().image_token_id);
  std::vector<vision::ImageBlock> blocks;
  int64_t prev_end = 0;
  for (const ImageSpan& sp : images) {
    if (sp.embeds == nullptr) {
      throw std::runtime_error("GemmaModel::PrefillMultimodal: an image span has no embeds");
    }
    if (sp.embeds_on_host != (comm_ != nullptr)) {
      // TP: host rows (one copy, every rank's H2D splice); TP=1: device rows (the D2D splice, unchanged).
      throw std::runtime_error(comm_ != nullptr ? "GemmaModel::PrefillMultimodal: image rows must be host-resident under "
                                                  "tensor parallelism (ImageSpan::embeds_on_host)"
                                                : "GemmaModel::PrefillMultimodal: an image span needs device embeds");
    }
    if (sp.tokens < 1 || sp.tokens > vision::kGemmaMaxImageBlockTokens || sp.offset < prev_end ||
        sp.offset + sp.tokens > total) {
      throw std::runtime_error("GemmaModel::PrefillMultimodal: image span [" + std::to_string(sp.offset) + ", +" +
                               std::to_string(sp.tokens) + ") is out of order, empty, over " +
                               std::to_string(vision::kGemmaMaxImageBlockTokens) + " tokens, or outside the call's tokens");
    }
    blocks.push_back({sp.offset, sp.offset + sp.tokens});
    prev_end = sp.offset + sp.tokens;
  }
  // Every maximal run of image placeholders must be exactly one span (a placeholder without rows would read the
  // embedding table row of a control token; a span shorter than its run would leave placeholders behind).
  const std::vector<vision::ImageBlock> runs = vision::FindImageBlocks(token_ids, image_id, 0);
  bool same = runs.size() == blocks.size();
  for (size_t i = 0; same && i < runs.size(); ++i) same = runs[i].start == blocks[i].start && runs[i].end == blocks[i].end;
  if (!same) {
    throw std::runtime_error("GemmaModel::PrefillMultimodal: the image placeholder runs in token_ids do not match the "
                             "image spans one to one");
  }
  const std::vector<vision::PrefillChunk> chunks =
      vision::PlanPrefillChunks(total, blocks, max_chunk_, vision::kGemmaMaxImageBlockTokens);
  std::vector<float> logits;
  size_t next_block = 0;
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    const vision::PrefillChunk& c = chunks[ci];
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(c.start),
                                     token_ids.begin() + static_cast<ptrdiff_t>(c.start + c.len));
    ChunkVision cv;
    std::vector<vision::ImageBlock> abs_blocks;  // this chunk's blocks in ABSOLUTE positions
    while (next_block < blocks.size() && blocks[next_block].start < c.start + c.len) {
      const vision::ImageBlock& b = blocks[next_block];
      cv.splices.push_back(
          {b.start - c.start, b.end - b.start, images[next_block].embeds, images[next_block].embeds_on_host});
      abs_blocks.push_back({pos_ + (b.start - c.start), pos_ + (b.end - c.start)});
      ++next_block;
    }
    if (!abs_blocks.empty()) cv.klimit_ext = vision::BuildKlimitExt(pos_, c.len, abs_blocks);
    const bool last = ci + 1 == chunks.size();
    std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr, /*is_prefill=*/true,
                                    cv.splices.empty() ? nullptr : &cv);
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
  if (comm_ != nullptr) {
    // TP, H5 (docs/tp.md 7.4 "unresolved row"): gather this step's full row in global id order. The merged summary is
    // identical on every rank, so every rank takes this branch together; SampleCanonical with the SAME u.
    GatherVocabRow(&sampled_row_scratch_);
  } else {
    sampled_row_scratch_.resize(static_cast<size_t>(vocab));
    logits_dev_.CopyToHost(sampled_row_scratch_.data(), static_cast<size_t>(vocab));
  }
  return kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
}

// ---- tensor parallel (docs/tp.md 7.2-7.5; docs/gemma4-plan.md M1b-1) ---------------------------------------------------

void GemmaModel::RequireNotTp(const char* what) const {
  if (comm_ != nullptr) {
    throw core::TpUnsupportedError(std::string("GemmaModel::") + what + ": not supported under tensor parallelism");
  }
}

int32_t GemmaModel::MergeGreedyPair() {
  tp::ArgmaxPair mine{};
  static_assert(sizeof(tp::ArgmaxPair) == 2 * sizeof(int32_t), "{idx, value} pair is two 32-bit words");
  R4DX_HIP_CHECK(hipMemcpy(&mine, argmax_pair_dev_.data(), sizeof(mine), hipMemcpyDeviceToHost));
  mine.idx += static_cast<int32_t>(vocab_offset_);  // local -> global id
  std::array<tp::ArgmaxPair, 2> all{};              // world <= 2 (Load)
  comm_->HostAllGather(&mine, sizeof(mine), all.data());
  return tp::MergeArgmax(all.data(), comm_->World());
}

void GemmaModel::MergeVerifyPairs(int64_t rows, std::vector<int32_t>* out) {
  static_assert(sizeof(tp::ArgmaxPair) == 8, "8 B per greedy row (docs/tp.md 7.3)");
  if (rows < 1 || rows > 16) throw std::logic_error("GemmaModel::MergeVerifyPairs: 1..16 rows");
  std::array<tp::ArgmaxPair, 16> mine{};
  R4DX_HIP_CHECK(hipMemcpy(mine.data(), verify_pair_dev_.data(), static_cast<size_t>(rows) * sizeof(tp::ArgmaxPair),
                           hipMemcpyDeviceToHost));
  for (int64_t t = 0; t < rows; ++t) mine[static_cast<size_t>(t)].idx += static_cast<int32_t>(vocab_offset_);  // local -> global
  const int world = comm_->World();
  std::array<tp::ArgmaxPair, 32> all{};  // [world <= 2][16]: every rank contributes a fixed 16-pair block
  comm_->HostAllGather(mine.data(), mine.size() * sizeof(tp::ArgmaxPair), all.data());
  // Repack to [world][rows] (the gathered blocks are 16 wide) and merge each row exactly (lowest global id on ties).
  std::array<tp::ArgmaxPair, 32> packed{};
  for (int r = 0; r < world; ++r) {
    for (int64_t t = 0; t < rows; ++t) {
      packed[static_cast<size_t>(r * rows + t)] = all[static_cast<size_t>(r * 16 + t)];
    }
  }
  out->resize(static_cast<size_t>(rows));
  tp::MergeArgmaxRows(packed.data(), world, static_cast<int>(rows), out->data());
}

void GemmaModel::MergeSummary(kernels::RowSummary* out) {
  static_assert(std::is_trivially_copyable_v<kernels::RowSummary>,
                "row summaries cross the host exchange as raw bytes (same process, same layout)");
  const int world = comm_->World();
  for (int j = 0; j < out->k; ++j) out->ids[j] += static_cast<int32_t>(vocab_offset_);  // local -> global ids first
  std::array<kernels::RowSummary, 2> all{};
  comm_->HostAllGather(out, sizeof(kernels::RowSummary), all.data());
  *out = tp::MergeRowSummaries(all.data(), world, container_.Config().vocab_size);
}

void GemmaModel::GatherVocabRow(std::vector<float>* full) {
  // Plain (blocking, null-stream) D2H of this rank's shard: the caller has synchronized stream_ (RunChunk's rule). Staged in
  // a separate vector so the all-gather's source and destination never alias.
  gather_shard_host_.resize(static_cast<size_t>(vocab_local_));
  R4DX_HIP_CHECK(hipMemcpy(gather_shard_host_.data(), logits_dev_.data(), static_cast<size_t>(vocab_local_) * sizeof(float),
                           hipMemcpyDeviceToHost));
  full->resize(static_cast<size_t>(container_.Config().vocab_size));
  comm_->HostAllGather(gather_shard_host_.data(), static_cast<size_t>(vocab_local_) * sizeof(float), full->data());
}

int64_t GemmaModel::WarmupPositions(const GemmaModelOptions& o, bool has_vision) {
  // TpWarmup: one full chunk, then (after a Reset) a 64-row chunk and three decode steps.
  // With a drafter, one more greedy DFlash round (a draft + a 16-row verify window) runs after them.
  // A vision-loaded model (the container really has vision.* and the load is not off) also warms one 4 + 280-row image
  // block; a text-only container under kAuto pays no floor.
  const int64_t vision_rows = has_vision && o.vision != GemmaVisionLoad::kOff ? 4 + 280 : 0;
  return std::max<int64_t>({o.prefill_chunk, 64 + 3 + (o.dflash_container.empty() ? 0 : 16), vision_rows});
}

void GemmaModel::TpWarmup() {
  if (comm_ == nullptr) throw std::logic_error("GemmaModel::TpWarmup: only a tensor-parallel rank (tp_world > 1) warms up");
  const int64_t warm_positions = WarmupPositions(opts_, container_.HasVision());
  if (max_ctx_ < warm_positions) {
    throw std::invalid_argument("GemmaModel::TpWarmup: max_ctx " + std::to_string(max_ctx_) + " < " +
                                std::to_string(warm_positions) + " warm-up positions");
  }
  // A full-width chunk first: the first launch of the M = 256 GEMMs at this rank's shard shapes, the sliced 85-row
  // all-reduces per collective site and the 256-row activations must not meet their first touch inside a request. Fixed ids
  // 0..n-1: the warm-up is a lockstep collective, so every rank must feed the same tokens.
  const auto ids = [](int64_t n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    return v;
  };
  (void)Prefill(ids(max_chunk_));
  Reset();
  if (container_.HasVision()) {
    // One full-width image block (4 text rows + 280 soft tokens, the processor's maximum): the 288-row chunk's GEMM / sliced
    // all-reduce shapes, the klimit_ext sliding-attention path and the host-row splice, all first-use otherwise. Zero rows
    // (a lockstep collective: every rank feeds the same ones).
    constexpr int64_t kWarmImageTokens = 280, kWarmTextRows = 4;
    std::vector<int32_t> toks = ids(kWarmTextRows);
    toks.insert(toks.end(), static_cast<size_t>(kWarmImageTokens), static_cast<int32_t>(container_.Info().image_token_id));
    const std::vector<uint16_t> zero_rows(static_cast<size_t>(kWarmImageTokens * container_.Config().hidden_size), 0);
    ImageSpan sp;
    sp.offset = kWarmTextRows;
    sp.tokens = kWarmImageTokens;
    sp.grid = vision::GridThw{1, 1, kWarmImageTokens};
    sp.embeds = zero_rows.data();
    sp.embeds_on_host = true;
    (void)PrefillMultimodal(toks, {sp});
    Reset();
  }
  (void)Prefill(ids(64));
  (void)DecodeStepGreedy(0);                       // channel 0 all-reduces + the greedy pair merge
  kernels::SampleParams sp;                         // temperature 1: the summary path and its merge
  std::mt19937_64 rng(0x7e57);
  (void)DecodeStepSampled(1, sp, rng);
  (void)DecodeStep(2);                              // the full-row gather
  if (dflash_.has_value()) {
    // The drafter's first-use kernels, the per-round top-16 host merge and the verify window's pair merge (the mirror of
    // Model::TpWarmup's DFlash round); n_min 0 so the walk always drafts dflash_draft_k_ tokens.
    (void)DecodeStepDflashGreedy(3, dflash_draft_k_, /*p_min=*/0.0f, /*n_min=*/0);
  }
  Reset();
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
  RequireNotTp("DecodeStepProfiled");
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
  RequireNotTp("PrefillProfiled");
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

// ---- speculative verification + DFlash decode (docs/gemma4-plan.md D-6) -----------------------------------------

void GemmaModel::InjectDflashRows(int64_t rows, int64_t start_pos) {
  const int64_t cols = static_cast<int64_t>(feature_layers_.size()) * container_.Config().hidden_size;
  for (int64_t row0 = 0; row0 < rows; row0 += 64) {
    const int64_t S = std::min<int64_t>(64, rows - row0);
    dflash_->InjectFeatures(stream_, arena_, features_dev_.data() + row0 * cols, S, start_pos + row0);
    // InjectFeatures stages the slice's rope positions in ONE pinned host array and uploads them asynchronously:
    // the next slice must not overwrite it before this copy ran (as Model::RunChunk does). Also keeps RunChunk's
    // "device idle on return" contract.
    stream_.Synchronize();
  }
  arena_.Reset();
}

void GemmaModel::SetDflashInjectionEnabled(bool enabled) { dflash_injection_enabled_ = enabled; }

std::vector<int32_t> GemmaModel::VerifyWindow(const std::vector<int32_t>& candidates) {
  const int64_t W = static_cast<int64_t>(candidates.size());
  // Under TP the per-row argmax is over this rank's vocab slice; RunChunk merges it across ranks (MergeVerifyPairs).
  if (W < 1 || W > 16) throw std::invalid_argument("GemmaModel::VerifyWindow: 1..16 candidates");
  if (W > max_chunk_) throw std::invalid_argument("GemmaModel::VerifyWindow: window exceeds prefill_chunk");
  const int64_t vocab = vocab_local_;
  if (verify_logits_dev_.size() < static_cast<size_t>(16 * vocab)) {
    verify_logits_dev_.Resize(static_cast<size_t>(16 * vocab));
    verify_argmax_dev_.Resize(16);
    verify_pair_dev_.Resize(32);
  }
  std::vector<int32_t> argmax;
  RunChunk(candidates, /*want_logits=*/false, nullptr, nullptr, nullptr, /*is_prefill=*/false, /*vision=*/nullptr,
           &argmax);
  verified_rows_ = W;
  return argmax;
}

void GemmaModel::CommitVerifiedWindow(int64_t n) {
  if (verified_rows_ < 1 || n < 1 || n > verified_rows_) {
    throw std::invalid_argument("GemmaModel::CommitVerifiedWindow: n must be in [1, rows of the last VerifyWindow]");
  }
  pos_ += n;
  verified_rows_ = 0;
}

std::vector<int32_t> GemmaModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                        int64_t* walk_len_out) {
  if (!dflash_.has_value()) throw std::runtime_error("GemmaModel::DecodeStepDflashGreedy: no drafter (GemmaModelOptions::dflash_container)");
  if (k < 0 || k > dflash_draft_k_) {
    throw std::runtime_error("GemmaModel::DecodeStepDflashGreedy: k must be in [0, " + std::to_string(dflash_draft_k_) + "]");
  }
  if (!dflash_injection_enabled_) {
    throw std::runtime_error("GemmaModel::DecodeStepDflashGreedy: drafter injection is disabled (SetDflashInjectionEnabled(false))");
  }
  if (dflash_->InjectedCount() != pos_) {
    throw std::runtime_error("GemmaModel::DecodeStepDflashGreedy: drafter frontier (" + std::to_string(dflash_->InjectedCount()) +
                             ") != PositionCount() (" + std::to_string(pos_) + ")");
  }
  // The verify window is k + 1 rows: near the end of the context, draft fewer (k = 0 still verifies the anchor).
  if (pos_ + 1 > max_ctx_) {
    throw std::runtime_error("GemmaModel::DecodeStepDflashGreedy: position " + std::to_string(pos_ + 1) +
                             " exceeds max_ctx " + std::to_string(max_ctx_));
  }
  k = std::min<int64_t>(k, max_ctx_ - pos_ - 1);
  const GemmaConfig& cfg = container_.Config();
  const DflashEmbeddingProvider embed = MakeEmbeddingProviderFromTable(
      container_.EmbedTokensDevice().data(), cfg.hidden_size, cfg.vocab_size, static_cast<float>(dflash_embed_scale_));
  const DflashLmHeadProvider lm_head = MakeLmHeadProviderFromLinear(&container_.LmHead());

  const DflashDraftResult draft = dflash_->DraftRound(stream_, arena_, token_id, k, p_min, n_min, embed, lm_head);
  arena_.Reset();
  if (walk_len_out != nullptr) *walk_len_out = draft.walk_len;

  std::vector<int32_t> cands;
  cands.reserve(draft.tokens.size() + 1);
  cands.push_back(token_id);
  cands.insert(cands.end(), draft.tokens.begin(), draft.tokens.end());
  const std::vector<int32_t> argmax = VerifyWindow(cands);

  // Greedy acceptance: draft i (candidates[i + 1]) is confirmed iff the target's argmax after candidates[i] equals it.
  int64_t accepted = 0;
  while (accepted < static_cast<int64_t>(draft.tokens.size()) && argmax[static_cast<size_t>(accepted)] == draft.tokens[static_cast<size_t>(accepted)]) {
    ++accepted;
  }
  const int64_t committed = accepted + 1;  // the anchor + the accepted drafts
  std::vector<int32_t> result(argmax.begin(), argmax.begin() + committed);  // accepted drafts, then the bonus token

  // Inject the committed rows' features (rows 0..committed-1 of the window just captured) at the drafter's frontier,
  // THEN commit. The rejected rows' features are never injected.
  if (feature_rows_ < committed) throw std::runtime_error("GemmaModel::DecodeStepDflashGreedy: internal error, capture short");
  InjectDflashRows(committed, dflash_->InjectedCount());
  CommitVerifiedWindow(committed);
  return result;
}

std::vector<int32_t> GemmaModel::DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                         const kernels::SampleParams& params, std::mt19937_64& rng,
                                                         int64_t* walk_len_out) {
  if (params.temperature > 0.0f) {
    // A sampled DFlash round (rejection sampling over the drafter) is not implemented for Gemma 4 yet: take one plain
    // sampled step instead. It commits one row and RunChunk injects it, so the drafter frontier stays == pos_ and a
    // later greedy round on the same request still works. Correct output, no speculation speedup.
    if (walk_len_out != nullptr) *walk_len_out = 0;
    return {DecodeStepSampled(token_id, params, rng)};
  }
  return DecodeStepDflashGreedy(token_id, k, p_min, n_min, walk_len_out);
}

}  // namespace r4dx::model
