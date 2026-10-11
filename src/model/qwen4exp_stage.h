// r4dx::model::Qwen4ExpStage -- the compute engine of one qwen4_exp pipeline stage (docs/qwen4exp-plan.md 4.11 item 4):
// the layers [first, end) resident on ONE device with their own stream, arena, per-layer state and scratch, plus (the
// last stage) the final gated-residual mixer and the lm_head.
//
//   embedding rows -> repeat x4 -> [PLE] -> (hyper-connection, GDN | QSA, inject) -> (hyper-connection, MoE, inject)
//   per layer ... -> final mixer -> lm_head
//
// Qwen4ExpModel (qwen4exp_model.h) owns the containers, builds the borrowed weight references below and orchestrates the
// stages in sequence. This file deliberately includes NO weight-owning header (qwen4exp_weights.h): that header and the
// kernel headers (moe_block.h, ple_layer.h, qsa.h) define the structs qwen4exp::MoeWeights / PleWeights / QsaWeights
// twice (owning vs borrowed), so no translation unit can see both; the stage (kernel side) and the model (container side)
// meet at these plain pointer structs instead.
//
// Every public method makes the stage's device the calling thread's device for its duration (and restores the previous
// one). Single sequence; one chunk of <= 256 rows at a time; every method returns with the stage's stream idle.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qwen4exp_config.h"
#include "qwen4exp_gdn.h"
#include "quant_linear.h"
#include "r4dx/kernels/summary_sampler.hpp"

namespace r4dx::model {

// Makes `device` the calling thread's HIP device; restores the previous one on exit.
class Qwen4ExpDeviceScope {
 public:
  explicit Qwen4ExpDeviceScope(int device);
  ~Qwen4ExpDeviceScope();
  Qwen4ExpDeviceScope(const Qwen4ExpDeviceScope&) = delete;
  Qwen4ExpDeviceScope& operator=(const Qwen4ExpDeviceScope&) = delete;

 private:
  int previous_ = -1;
  bool changed_ = false;
};

// ---- borrowed weights (device pointers into the container; the container outlives the stage) ---------------------
struct Qwen4ExpHcRefs {
  const uint16_t* norm = nullptr;    // bf16 [4 * hidden], raw checkpoint weight
  const uint16_t* down = nullptr;    // bf16 [lowrank, 4 * hidden]
  const uint16_t* up = nullptr;      // bf16 [4 * hidden, lowrank]
  const uint16_t* inject = nullptr;  // bf16 [4, 4 * hidden]; null for the final mixer
};

struct Qwen4ExpQsaRefs {
  const QuantLinear* qg = nullptr;
  const QuantLinear* k = nullptr;
  const QuantLinear* v = nullptr;
  const QuantLinear* o = nullptr;
  const uint16_t* q_norm = nullptr;
  const uint16_t* k_norm = nullptr;
  const uint16_t* indexer_qk = nullptr;  // bf16 [(index_heads + 1) * index_dim, hidden]
  const uint16_t* indexer_q_norm = nullptr;
  const uint16_t* indexer_k_norm = nullptr;
  const float* k_descale = nullptr;  // fp32 [kv_heads]
  const float* v_descale = nullptr;
};

struct Qwen4ExpMoeTrellisPointers {
  const uint32_t* gate_up_w = nullptr;
  const float* gate_up_suh = nullptr;
  const float* gate_up_svh = nullptr;
  int gate_up_bits = 0;
  const uint32_t* down_w = nullptr;
  const float* down_suh = nullptr;
  const float* down_svh = nullptr;
  int down_bits = 0;
};

struct Qwen4ExpMoeRefs {
  const uint16_t* router = nullptr;                   // bf16 [experts, hidden]
  const QuantLinear* shared_gate_proj = nullptr;      // [shared_inter, hidden]
  const QuantLinear* shared_up_proj = nullptr;
  const QuantLinear* shared_down_proj = nullptr;      // [hidden, shared_inter]
  const uint16_t* shared_gate = nullptr;              // bf16 [1, hidden]
  std::vector<const QuantLinear*> expert_gate_up;     // [experts]: [2 * inter, hidden]
  std::vector<const QuantLinear*> expert_down;        // [experts]: [hidden, inter]
  Qwen4ExpMoeTrellisPointers trellis_ptrs;
};

struct Qwen4ExpPleRefs {
  const uint16_t* key_proj = nullptr;    // bf16 [4 * hidden, hidden]
  const uint16_t* value_proj = nullptr;  // bf16 [hidden, hidden]
  const uint16_t* norm_key = nullptr;    // bf16 [4 * hidden]
  const uint16_t* norm_query = nullptr;
  const uint16_t* norm_conv = nullptr;
  const uint16_t* conv1d = nullptr;      // bf16 [4 * hidden, 4]
};

struct Qwen4ExpLayerRefs {
  int64_t index = 0;  // global layer index
  bool qsa = false;
  Qwen4ExpHcRefs hc_attn, hc_mlp;
  Qwen4ExpGdnRefs gdn;    // !qsa
  Qwen4ExpQsaRefs attn;   // qsa
  Qwen4ExpMoeRefs moe;
  std::optional<Qwen4ExpPleRefs> ple;
};

struct Qwen4ExpHeadRefs {
  Qwen4ExpHcRefs hc_final;                // inject null
  const QuantLinear* lm_head = nullptr;   // [vocab, hidden]
};

struct Qwen4ExpStageOptions {
  int device = 0;
  Qwen4ExpConfig config;
  int64_t max_ctx = 0;       // KV / QSA capacity in tokens
  int64_t max_rows = 256;    // rows per chunk (<= QsaLayerCore::kMaxChunk)
  bool prompt_checkpoint = false;
};

// Test hook: a named intermediate of the forward pass, handed to the callback with a DEVICE pointer that is valid on
// `stream` (the callback copies it out and synchronizes the stream itself). Names follow the tiny goldens
// (tools/reference/qwen4exp/tiny_golden.py): `layer{i}.residual_in | residual_ple | attn_mix | attn_inject | attn_out |
// residual_attn | mlp_mix | mlp_inject | moe_out | residual_out | attn_pre_o (QSA layers) | topk_ids`, `mixer_out`,
// `logits`. The chunk's rows are sequence positions [first_pos, first_pos + rows).
struct Qwen4ExpCapture {
  enum class Dtype { kBf16, kI32, kF32 };
  std::string name;
  const void* dev = nullptr;
  Dtype dtype = Dtype::kBf16;
  int64_t rows = 0, cols = 0;
  int64_t first_pos = 0;
  hipStream_t stream = nullptr;
  bool on_host = false;  // `dev` points into HOST memory (topk_ids)
};
using Qwen4ExpCaptureFn = std::function<void(const Qwen4ExpCapture&)>;

struct Qwen4ExpChunk {
  int64_t rows = 0;       // 1..max_rows
  int64_t first_pos = 0;  // sequence position of row 0
  bool is_prefill = true; // false: a decode step (rows == 1) -- the GDN recurrent kernels, not the chunked scan
  // GDN layers of a prefill chunk take the recurrent decode kernels token by token instead of the chunked scan
  // (Qwen4ExpGdnCall::prefill_recurrent): slow, the reference path the chunked scan is tested against. Ignored for a decode step.
  bool gdn_prefill_recurrent = false;
  // The n-gram rows [rows, ple_embed_dim] bf16 of this chunk, in HOST memory (pinned), once they are ready; called when the
  // stage reaches a PLE layer (once per PLE layer; it may block on the n-gram reader and must be idempotent). Required by
  // a stage that holds a PLE layer.
  std::function<const uint16_t*()> ple_rows;
  // Test hook (layer-isolated runs): >= 0 runs ONLY the layer with this global index over the loaded residual, leaving the
  // other layers' state untouched. -1 = every layer of the stage.
  int64_t only_layer = -1;
};

struct Qwen4ExpHeadRequest {
  enum class Kind { kLogits, kArgmax, kSummary, kAllArgmax };
  Kind kind = Kind::kLogits;
  float inv_temperature = 0.0f;  // kSummary
};
struct Qwen4ExpHeadResult {
  std::vector<float> logits;  // kLogits: the last row's fp32 logits [vocab]
  int32_t token = -1;         // kArgmax
  std::vector<int32_t> tokens; // kAllArgmax: argmax token for each row
  kernels::RowSummary summary;  // kSummary
};

class Qwen4ExpStage {
 public:
  // `layers` are the stage's layers in order (global indices first, first + 1, ...); `head` makes it the last stage.
  // Allocates everything on options.device (stream, arena, residual / scratch buffers, per-layer state), zeroes the
  // state and synchronizes. Throws on a missing / mis-shaped weight.
  Qwen4ExpStage(const Qwen4ExpStageOptions& options, std::vector<Qwen4ExpLayerRefs> layers,
                std::optional<Qwen4ExpHeadRefs> head);
  ~Qwen4ExpStage();
  Qwen4ExpStage(const Qwen4ExpStage&) = delete;
  Qwen4ExpStage& operator=(const Qwen4ExpStage&) = delete;

  int Device() const;
  hipStream_t Stream() const;
  int64_t FirstLayer() const;
  int64_t EndLayer() const;
  bool HasHead() const;
  bool HasPle() const;  // a PLE layer lives here

  // ---- chunk input: exactly one of the two per chunk, before RunLayers ----------------------------------------------
  // The first stage: the gathered embedding rows [rows, hidden] (host, pinned preferred) -> device, repeated hc times
  // into the residual [rows, hc * hidden].
  void LoadEmbedding(const uint16_t* host_rows, int64_t rows);
  // Any later stage: the previous stage's residual [rows, hc * hidden] (host, pinned) -> device (async; the host buffer
  // must stay valid until the next method that synchronizes returns -- RunLayers does).
  void LoadResidual(const uint16_t* host_rows, int64_t rows);

  // Runs the stage's layers over the loaded residual (arena reset between blocks) and synchronizes the stream.
  void RunLayers(const Qwen4ExpChunk& chunk);

  // The residual [rows, hc * hidden] -> host (pinned preferred); synchronizes.
  void StoreResidual(uint16_t* host_rows, int64_t rows);

  // ---- the head (last stage): final mixer + lm_head on the LAST row of the chunk just run ----------------------------
  Qwen4ExpHeadResult Head(int64_t rows, const Qwen4ExpHeadRequest& request);
  // The fp32 logits row of the last kLogits / kSummary head call (the sampler's row-granular fallback), D2H.
  std::vector<float> LastLogitsRow();

  // ---- sequence state -----------------------------------------------------------------------------------------------
  void Reset();  // GDN state, PLE conv state, QSA bookkeeping; synchronous
  // Copies the live GDN + PLE state into the spare copy (needs options.prompt_checkpoint). Synchronous.
  void SaveCheckpoint();
  // Copies it back and rewinds the QSA layers to `pos` positions. Synchronous.
  void RestoreCheckpoint(int64_t pos);

  // Test hook (see Qwen4ExpCapture). Setting it also makes Head() compute the mixer and the lm_head over EVERY row of the
  // chunk so `mixer_out` / `logits` carry all rows; requires vocab * max_rows * 2 bytes of extra device memory.
  void SetCapture(Qwen4ExpCaptureFn fn);

  // Wall milliseconds of the stage's last RunLayers / Head call (host clock, includes the synchronization).
  double LastRunMs() const;
  double LastHeadMs() const;
  // Last RunLayers' wall time spent inside the PLE rows callback (waiting for the n-gram reader), in milliseconds.
  double LastPleWaitMs() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace r4dx::model
