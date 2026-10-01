// r4dx::model::GemmaModel -- the single-device Gemma 4 (gemma4_unified) text engine (docs/gemma4-plan.md 3.2,
// task M1-20). A separate class from Model on purpose: Model is full of GDN, mrope, MTP and 64-row-slice
// logic; this one is 48 layers of {sliding | full} attention + GeGLU with sandwich norms, one sequence, no
// speculation. The Qwen Model / TpModel / AttentionLayer / Mlp are not touched.
//
// Per decoder layer (docs/gemma4-semantics.md; every step rounds to bf16 where the HF module chain does):
//     normed   = input_layernorm(h)                              (fused into the previous layer's last kernel)
//     o        = attention(normed)            GemmaAttnLayer  (sliding: window 1024 ring; full: head_dim 512)
//     h        = h + post_attention_layernorm(o);  normed = pre_feedforward_layernorm(h)   (one fused kernel)
//     m        = mlp(normed)                  GemmaMlp (GeGLU)
//     h        = (h + post_feedforward_layernorm(m)) * layer_scalar;  normed = next input_layernorm(h)
// layer_scalar is applied ONCE, after the MLP residual add (the attention half passes 1.0).
// Residual dtype (GemmaResid, R4DX_GEMMA_RESID, default fp32): the stream h is fp32; norm outputs (GEMM inputs) and
// sublayer outputs are bf16, and the post-norm / add / layer_scalar are fp32 with no intermediate bf16 rounding
// (so the bf16 rounding points listed above apply only to R4DX_GEMMA_RESID=bf16). Under TP the all-reduce acts
// on the bf16 sublayer outputs (o_proj / down_proj), never on the fp32 residual, which stays replicated.
// A rotated container (`__metadata__.rotation`, Gemma option A) runs the residual stream in the rotated basis:
// x Q after the scaled embedding gather, the fused r4dx_post_rmsnorm_rotate_add_bf16 for both post-norm
// residual adds, the Hadamard on the o_proj / down_proj inputs, x Q^T before the final norm (and on captures).
//
// KV: bf16 by default (R4DX_GEMMA_KV unset); GemmaKvMode::kFp8 (R4DX_GEMMA_KV=fp8) is fp8 e4m3 with the container's
// static descales for every layer; GemmaKvMode::kBf16Full keeps
// the 8 full layers bf16 (docs/gemma4-plan.md section 9.5, the fallback if the fp8 KL gate fails); kBf16 is the
// all-bf16 reference run. Sliding layers share ONE block table and each owns a ring of 1536 tokens.
//
// Context: GemmaConfig::ResolveMaxCtx (default 131072, 262144 opt-in).
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "dflash_draft.h"
#include "gemma_container.h"
#include "gemma_mlp.h"
#include "model_types.h"
#include "prefill_chunk.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/summary_sampler.hpp"
#include "r4dx/model/attention/gemma_attention_layer.hpp"
#include "r4dx/model/attention/sliding_kv_cache.hpp"
#include "tp/tp_submit.h"  // tp::SubmitBounder (inert at TP=1)
#include "tp/tp_vocab.h"   // tp::ArgmaxPair, the vocab-split merges (header-only, HIP-free)

namespace r4dx::model {

enum class GemmaKvMode { kFp8, kBf16Full, kBf16 };
// Residual-stream dtype. kFp32 (default): the residual is fp32 [rows, hidden], GEMM inputs (norm outputs) and
// sublayer outputs stay bf16. kBf16: the original all-bf16 stream, kept for A/B measurement
// (R4DX_GEMMA_RESID=bf16).
enum class GemmaResid { kFp32, kBf16 };

struct GemmaModelOptions {
  std::string container_path;
  Layout layout = Layout::kBf16;
  int64_t layer_limit = -1;     // tiny fixtures
  int64_t max_ctx = 0;          // <= 0: the config's max_position_embeddings (131072)
  bool allow_extended_ctx = false;  // opt-in to 262144
  GemmaKvMode kv = GemmaKvMode::kBf16;  // default bf16 (fp8 only when R4DX_GEMMA_KV=fp8, docs/gemma4-plan.md 9)
  attention::GemmaAttnBackend attn = attention::GemmaAttnBackend::kReference;
  int prefill_chunk = 256;      // rows per prefill chunk (<= 256: the ring holds window + 288)
  GemmaResid resid = GemmaResid::kFp32;
  bool prompt_checkpoint = false;  // spare copies of the sliding rings for SaveCheckpoint/RestoreCheckpoint
  // Tensor parallel (docs/gemma4-plan.md M1b-1; GemmaTpModel fills these in per rank, nothing else should): this Model is
  // rank `tp_rank` of `tp_world` (1 or 2) and all-reduces its row-parallel sublayer outputs (attn.o, mlp.down) through
  // `tp_comm` (non-owning; non-null iff tp_world == 2). Every rank's Model must be driven with the same sequence of calls.
  int tp_world = 1;
  int tp_rank = 0;
  core::TpComm* tp_comm = nullptr;
  // Bounded prefill submission (tp/tp_submit.h, docs/tp.md Appendix B N57): both 0 at TP=1.
  int tp_submit_layers = 0;
  int tp_max_inflight_units = 0;
  // Vision embedder (docs/gemma4-plan.md M2). kAuto: load vision.* iff the container has it; kOn: require it;
  // kOff: never (the buffers then stay at prefill_chunk rows, the exact pre-M2 sizes).
  GemmaVisionLoad vision = GemmaVisionLoad::kOff;
  // DFlash drafter (docs/gemma4-plan.md section 6.4, D-6). Non-empty: load this dflash2 container (a DFlash v1
  // checkpoint converted with `r4dx-convert --dflash-hf`), attach the target feature capture at its
  // `target_layers` and inject every committed row into the drafter. `dflash_draft_k` (1..block_size-1) is the
  // most tokens one round drafts. Both are inert at their defaults.
  std::string dflash_container;
  int64_t dflash_draft_k = 0;
};

// Rows the per-chunk buffers hold when vision is loaded: an image block (<= 288 soft tokens, the sliding ring
// holds window + 288) is one chunk, never split (docs/gemma4-plan.md 3.2 "Bidirectional image block").
constexpr int64_t kGemmaVisionBufferRows = 320;

// Reads R4DX_GEMMA_KV (fp8|bf16_full|bf16), R4DX_GEMMA_ATTN (ref|r4d) and R4DX_GEMMA_RESID (fp32|bf16) over the
// defaults; throws on junk.
void ApplyGemmaEnv(GemmaModelOptions* o);

class GemmaModel {
 public:
  static GemmaModel Load(const GemmaModelOptions& opts);

  GemmaModel(GemmaModel&&) = default;
  GemmaModel& operator=(GemmaModel&&) = default;

  const GemmaConfig& Config() const { return container_.Config(); }
  const GemmaContainer& GetContainer() const { return container_; }
  int64_t MaxCtx() const { return max_ctx_; }
  int64_t PositionCount() const { return pos_; }
  int64_t SampledFallbackRows() const { return sampled_fallback_rows_; }
  GemmaKvMode KvMode() const { return opts_.kv; }
  GemmaResid ResidDtype() const { return opts_.resid; }
  // Bytes of KV the load allocated (sliding rings + full layers), for the load log and tests.
  int64_t KvBytes() const { return kv_bytes_; }
  int TpWorld() const { return opts_.tp_world; }
  int TpRank() const { return opts_.tp_rank; }

  // TP only (comm != nullptr, else std::logic_error): the load-time warm-up of docs/tp.md 2.9 step 9 for this model -- one
  // full prefill chunk (the M = 256 GEMMs and the sliced all-reduces), a 64-row prefill, a greedy, a sampled and a
  // full-logits decode step -- so no first-use kernel load lands inside a request, where a rank stalled in it would let
  // its peer's spin timeout fire. Ends with Reset(). A lockstep collective: every rank must call it.
  void TpWarmup();
  // The positions TpWarmup() needs in the KV caches (max_ctx must be at least this under TP).
  static int64_t WarmupPositions(const GemmaModelOptions& o);
  // Prefill submission bounding counters (TP; zeros at TP=1).
  tp::SubmitBounder::Stats TpSubmitStats() const { return submit_.GetStats(); }
  int64_t VocabLocal() const { return vocab_local_; }

  void Reset();
  void SaveCheckpoint();     // needs prompt_checkpoint; D2D copies of the sliding rings
  void RestoreCheckpoint();
  int64_t CheckpointPosition() const { return ckpt_pos_; }

  // Feeds `token_ids` in chunks of <= prefill_chunk rows; returns fp32 logits[vocab] (softcapped) of the last.
  // `on_chunk_captured` runs once per chunk after it finished (the feature rows of that chunk are valid).
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids,
                             const std::function<void()>& on_chunk_captured = nullptr);
  // Prefill() with host-computed soft-token rows (audio, docs/gemma4-audio.md) spliced over the scaled embedding
  // gather: every span's [offset, offset + tokens) rows (offsets relative to `token_ids`) are replaced, unscaled,
  // by `rows` (host bf16 [tokens, hidden]) before the residual rotation, chunk by chunk (a span may straddle
  // chunks). With an empty `spans` this is exactly Prefill(). Throws on a span outside `token_ids`.
  std::vector<float> PrefillAudio(const std::vector<int32_t>& token_ids, const std::vector<AudioRowSpan>& spans,
                                  const std::function<void()>& on_chunk_captured = nullptr);
  // ---- vision (M2; needs the container's vision.* tensors, GemmaModelOptions::vision) --------------------------
  bool HasVision() const { return container_.HasVision(); }
  // `pixel_values`: [total_patches, 6912] fp32 on the host (r4dx::vision::PreprocessImages with
  // ImageProcessorConfig::gemma, or PreprocessGemmaImage); `grids`: one {t=1, h, w} per image in MERGED-grid
  // cells, h*w rows each, summing to total_patches. Writes [total_patches, hidden] bf16 UNSCALED rows to `out`
  // (device) and returns synchronized. Each image must have <= 288 soft tokens (one prefill chunk).
  void EncodeImages(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
                    core::DeviceBuffer<uint16_t>* out);
  // Prefill with image blocks. `images`: spans in `token_ids` (offset = first soft token, relative to this
  // call's tokens), in order and non-overlapping; the placeholder tokens there must be the config's image token.
  // Their `embeds` rows (device bf16, [tokens, hidden]) overwrite the gathered embedding rows UNSCALED, before the
  // residual rotation. Each block is one prefill chunk (never split) and, on the SLIDING layers only, is
  // attended bidirectionally (klimit_ext = the block's last position); full layers and every later decode step
  // stay causal. Empty `images` is Prefill().
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids, const std::vector<ImageSpan>& images,
                                       const std::function<void()>& on_chunk_captured = nullptr);

  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);
  StepProfile DecodeStepProfiled(int32_t token_id);
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids);

  // ---- rung-3 test hook (tests/model/attention/test_gemma_attn_layer.cpp) ---------------------------------
  // Runs decoder layer `layer` alone on `x_rows` ([T, hidden] bf16 host, T <= prefill_chunk) as the chunk at
  // absolute position `start_pos` of that layer's own KV cache (successive calls with start_pos = the rows fed
  // so far continue it; call Reset-free, the layer's cache is only ever written at its own positions), and
  // returns the layer output [T, hidden]. Does not touch PositionCount(). `klimit_ext` (optional, [T] absolute
  // key limits, -1 = causal) is passed to a SLIDING layer's attention only, as PrefillMultimodal does; T may be
  // up to the vision buffer rows (320) when vision is loaded.
  std::vector<uint16_t> DebugLayerForward(int64_t layer, const std::vector<uint16_t>& x_rows, int64_t T,
                                          int64_t start_pos, const std::vector<int32_t>* klimit_ext = nullptr);

  // ---- target hidden-state capture (the DFlash drafter's input, later) ---------------------------------
  // Captures the residual stream ENTERING each listed layer (index == num_layers: the last layer's output,
  // i.e. the final-norm input) for every row of every chunk, in the original (un-rotated) basis, into
  // FeatureBuffer(): [rows, layers.size() * hidden] bf16 (always bf16, in both residual modes: the drafter consumes bf16;
  // with the fp32 residual the capture is taken in fp32, un-rotated in fp32, then rounded once to bf16), row-major, rows = FeatureRows() <= prefill_chunk.
  // Valid after a Prefill chunk callback / after a decode step; rewritten from row 0 by the next chunk.
  // `layers` must be sorted strictly ascending and in [0, num_layers].
  void AttachFeatureCapture(std::vector<int64_t> layers);
  void DetachFeatureCapture();
  const uint16_t* FeatureBuffer() const { return features_dev_.data(); }
  // ---- speculative verification + DFlash decode (docs/gemma4-plan.md D-6) ------------------------------------
  // Runs `candidates` (W = 1..16 rows: [anchor, d1..dk]) as ONE chunk at PositionCount() WITHOUT committing, and
  // returns the target's greedy argmax of every row (softcapped fp32 logits, the same kernel DecodeStepGreedy
  // uses): row i's argmax is the token that follows candidates[i]. K/V of ALL W rows are written (sliding rings
  // and full layers); the feature capture (if attached) holds all W rows. Follow with CommitVerifiedWindow(n).
  //
  // ROLLBACK. Rejected rows need no undo, on either kind of layer: a verify row at position p writes slot
  // p % ring and clobbers position p - ring, which lies below frontier - window for every ring of at least
  // window + W - 1 slots (Gemma: 1536 >= 1024 + 15), so a clobbered key is outside every future query's window;
  // the stale bytes of the rejected positions are overwritten by the next window's own chunk write before any
  // query can read them (a query only reads kpos <= its own position). Full layers are positional. Proved by
  // simulation in tests/kernels/test_ring_verify_rollback.cpp. The one thing that DOES need care is bookkeeping:
  // PositionCount() and the drafter's frontier advance only in CommitVerifiedWindow / the injection.
  std::vector<int32_t> VerifyWindow(const std::vector<int32_t>& candidates);
  // Commits the first `n` rows (1..rows of the last VerifyWindow): PositionCount() += n. Nothing else.
  void CommitVerifiedWindow(int64_t n);
  bool DflashEnabled() const { return dflash_.has_value(); }
  int64_t DflashDraftK() const { return dflash_draft_k_; }
  const DflashDraft& Drafter() const { return dflash_.value(); }
  // Off: the feature capture and the drafter injection are skipped together (a request that never drafts);
  // the next injection then opens a gap in the drafter's ring (DflashDraft::InjectFeatures), correct but cold.
  void SetDflashInjectionEnabled(bool enabled);
  // One DFlash round, greedy: draft up to `k` tokens from `token_id` (the last committed token, NOT yet fed to
  // the target), verify [token_id, drafts] in one window, commit the longest prefix the target's own argmax
  // confirms plus the bonus token, inject the committed rows' features into the drafter. Returns the committed
  // tokens that FOLLOW token_id: accepted drafts then the bonus (1..k+1 tokens), byte-identical to as many plain
  // DecodeStepGreedy calls (up to the verify-vs-decode GEMM numerics, a GPU gate). `walk_len_out`: tokens the
  // selector walk produced before the n_min discard.
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                              int64_t* walk_len_out = nullptr);
  // temperature <= 0 is DecodeStepDflashGreedy; a sampled DFlash round is not implemented for Gemma (throws).
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                               const kernels::SampleParams& params, std::mt19937_64& rng,
                                               int64_t* walk_len_out = nullptr);

  int64_t FeatureRows() const { return feature_rows_; }
  int64_t FeatureStartPosition() const { return feature_pos_; }
  const std::vector<int64_t>& FeatureLayers() const { return feature_layers_; }

 private:
  GemmaModel() = default;
  struct SummaryRequest {
    float inv_temperature = 1.0f;
    kernels::RowSummary* out = nullptr;
  };
  // What a chunk that carries an image block adds: the rows to overwrite with image embeddings (chunk-relative)
  // and the per-row bidirectional key limit for the sliding layers.
  struct ChunkVision {
    struct Splice {
      int64_t row = 0;
      int64_t rows = 0;
      const uint16_t* embeds = nullptr;  // device bf16 [rows, hidden]
    };
    std::vector<Splice> splices;
    std::vector<int32_t> klimit_ext;  // [T] absolute key limit, -1 = causal
  };
  // `verify_argmax` non-null: a verify window (VerifyWindow) -- per-row argmax of all rows into it, pos_ NOT advanced.
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                              const SummaryRequest* summary_out, SpanAccumulator* prof, bool is_prefill = false,
                              const ChunkVision* vision = nullptr, std::vector<int32_t>* verify_argmax = nullptr);
  // Tensor parallel (docs/tp.md 7.2-7.4): the vocab-split merges of RunChunk's result, in global id order. Host
  // rendezvous (HostAllGather): the device must be synchronized. Every rank ends with the same answer.
  int32_t MergeGreedyPair();
  void MergeSummary(kernels::RowSummary* out);
  void GatherVocabRow(std::vector<float>* full);
  void RequireNotTp(const char* what) const;
  void InjectDflashRows(int64_t rows, int64_t start_pos);  // feeds features_dev_ rows [0, rows) to the drafter
  void RotateResidual(void* x, int64_t rows, bool inverse);  // x is fp32 or bf16 per opts_.resid
  void UploadChunkMeta(int64_t start_pos, int64_t T, const std::vector<int32_t>* klimit_ext = nullptr);
  void RunLayer(int64_t i, void* cur, uint16_t* normed, int64_t T, int64_t start_pos, bool has_next,
                SpanAccumulator* prof, const int32_t* klimit_ext = nullptr);
  void FetchRowSummary(float inv_temperature, kernels::RowSummary* out);  // device -> host; LOCAL ids / vocab under TP

  GemmaModelOptions opts_;
  GemmaContainer container_;
  core::Stream stream_;
  core::Arena arena_;
  int64_t max_ctx_ = 0;
  int max_chunk_ = 256;       // rows per ordinary prefill chunk
  int64_t buf_rows_ = 256;    // rows the per-chunk buffers hold: max_chunk_, or 320 with vision loaded
  int64_t pos_ = 0;
  bool started_ = false;
  int64_t sampled_fallback_rows_ = 0;
  int64_t kv_bytes_ = 0;

  // Tensor parallel: this rank's lm_head rows (== vocab_size at TP=1) and their first global id; the comm endpoint.
  core::TpComm* comm_ = nullptr;
  int64_t vocab_local_ = 0, vocab_offset_ = 0;
  tp::SubmitBounder submit_;
  int submit_layers_ = 0;
  core::DeviceBuffer<int32_t> argmax_pair_dev_;  // TP greedy: {local idx, float bits of the value}
  core::DeviceBuffer<uint8_t> topk_lse_ws_;      // TP: this rank's own r4dx_topk_lse_f32_ws workspace
  std::vector<float> gather_shard_host_;

  std::vector<attention::GemmaAttnLayer> attn_;                  // per layer
  std::optional<attention::SlidingRingGeometry> geo_;
  std::optional<attention::SlidingBlockTable> ring_table_;       // shared by every sliding layer
  std::vector<std::optional<attention::GemmaKvCache>> kv_;       // per layer
  std::vector<std::optional<attention::GemmaKvCache>> kv_ckpt_;  // sliding layers, if prompt_checkpoint
  int64_t ckpt_pos_ = -1;

  core::PinnedBuffer<int32_t> ids_host_;
  core::DeviceBuffer<int32_t> ids_dev_, positions_dev_, ring_slots_dev_, seqused_dev_, klimit_dev_;
  core::DeviceBuffer<uint16_t> buf_a_, buf_normed_;  // buf_a_: bf16 residual (R4DX_GEMMA_RESID=bf16 only)
  core::DeviceBuffer<float> buf_a32_;                // fp32 residual [max_chunk, hidden] (default; 4 B/elem)
  core::DeviceBuffer<float> logits_dev_;
  core::DeviceBuffer<int32_t> argmax_dev_, summary_ids_dev_;
  core::DeviceBuffer<float> summary_vals_dev_, summary_lse_dev_;
  std::vector<float> sampled_row_scratch_;
  // PrefillAudio's current chunk: the spans of the whole call and the index of this chunk's first token in it
  // (RunChunk splices the overlap after the embedding gather); null outside PrefillAudio.
  const std::vector<AudioRowSpan>* splice_spans_ = nullptr;
  int64_t splice_chunk_off_ = 0;

  std::vector<int64_t> feature_layers_;
  core::DeviceBuffer<uint16_t> features_dev_;
  core::DeviceBuffer<float> features_f32_;  // fp32 resid mode: capture staging (rotated basis), un-rotated then narrowed
  int64_t feature_rows_ = 0, feature_pos_ = 0;

  std::optional<DflashDraft> dflash_;
  int64_t dflash_draft_k_ = 0;
  bool dflash_injection_enabled_ = true;
  double dflash_embed_scale_ = 1.0;  // the drafter container's embed_scale (R4DX_DFLASH_EMBED_SCALE overrides, A/B)
  core::DeviceBuffer<float> verify_logits_dev_;    // [16, vocab] softcapped fp32 (lazily sized by VerifyWindow)
  core::DeviceBuffer<int32_t> verify_argmax_dev_;  // [16]
  int64_t verified_rows_ = 0;                      // rows of the last VerifyWindow not yet committed (0 == none)
};

}  // namespace r4dx::model
