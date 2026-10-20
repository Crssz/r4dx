// r4dx::model::Qwen35MoeStage -- the compute engine of one qwen3_5_moe pipeline stage (docs/qwen35moe-plan.md M3): the
// layers [first, end) resident on ONE device with their own stream, arena, per-layer state and scratch, plus (the last
// stage) the final RMSNorm and the lm_head.
//
//   embedding rows -> per layer:  h' = h + mixer(rmsnorm(h, input_layernorm))            mixer = GDN | gated attention
//                                  h  = h' + moe(rmsnorm(h', post_attention_layernorm))   moe = 256-expert top-8 + shared expert
//                  -> rmsnorm(final_norm) -> lm_head
//
// The mixers are the ones the Qwen3.5 27B runs: GDN is Qwen4ExpGdn (silu gate; the same libr4d kernels, the span-safe chunked
// scan) and full attention is attention::AttentionLayer (fp8 paged KV, partial mrope), both with their residual add; the MoE
// is qwen4exp::MoeBlockBase (host-routed). Like Qwen4ExpStage this file includes NO weight-owning header
// (qwen4exp_weights.h): the stage (kernel side) and the model (container side) meet at these plain pointer structs.
//
// Every public method makes the stage's device the calling thread's device for its duration (and restores the previous
// one). Single sequence. A chunk is T <= 64 rows, or a multiple of 64 up to 256 rows (the attention layer's sliced path).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qwen35moe_config.h"
#include "qwen35moe_grouped_decode.h"
#include "qwen4exp_gdn.h"
#include "quant_linear.h"
#include "r4dx/kernels/summary_sampler.hpp"

namespace r4dx::model {

// Makes `device` the calling thread's HIP device; restores the previous one on exit.
class Qwen35MoeDeviceScope {
 public:
  explicit Qwen35MoeDeviceScope(int device);
  ~Qwen35MoeDeviceScope();
  Qwen35MoeDeviceScope(const Qwen35MoeDeviceScope&) = delete;
  Qwen35MoeDeviceScope& operator=(const Qwen35MoeDeviceScope&) = delete;

 private:
  int previous_ = -1;
  bool changed_ = false;
};

// ---- borrowed weights (device pointers into the container; the container outlives the stage) -----------------------
struct Qwen35MoeAttnRefs {
  const QuantLinear* qg = nullptr;
  const QuantLinear* k = nullptr;
  const QuantLinear* v = nullptr;
  const QuantLinear* o = nullptr;
  const uint16_t* q_norm = nullptr;
  const uint16_t* k_norm = nullptr;
  const float* k_descale = nullptr;  // fp32 [kv_heads]
  const float* v_descale = nullptr;
};

struct Qwen35MoeMoeRefs {
  const uint16_t* router = nullptr;               // bf16 [experts, hidden]
  const QuantLinear* shared_gate_proj = nullptr;  // [shared_inter, hidden]
  const QuantLinear* shared_up_proj = nullptr;
  const QuantLinear* shared_down_proj = nullptr;  // [hidden, shared_inter]
  const uint16_t* shared_gate = nullptr;          // bf16 [1, hidden]
  std::vector<const QuantLinear*> expert_gate_up; // [experts]: [2 * inter, hidden]
  std::vector<const QuantLinear*> expert_down;    // [experts]: [hidden, inter]
  // The same experts as stacked trellis buffers (the grouped decode path); all null when the experts are not trellis.
  Qwen35MoeTrellisPtrs trellis;
};

struct Qwen35MoeLayerRefs {
  int64_t index = 0;  // global layer index
  bool full_attention = false;
  const uint16_t* input_layernorm = nullptr;           // bf16 [hidden], raw (the kernels apply 1 + w)
  const uint16_t* post_attention_layernorm = nullptr;  // bf16 [hidden]
  Qwen4ExpGdnRefs gdn;       // !full_attention
  Qwen35MoeAttnRefs attn;    // full_attention
  Qwen35MoeMoeRefs moe;
};

struct Qwen35MoeHeadRefs {
  const uint16_t* final_norm = nullptr;  // bf16 [hidden]
  const QuantLinear* lm_head = nullptr;  // [vocab, hidden]
};

struct Qwen35MoeStageOptions {
  int device = 0;
  Qwen35MoeConfig config;
  int64_t max_ctx = 0;       // KV capacity in tokens
  int64_t max_rows = 256;    // rows per chunk (<= 256)
  bool prompt_checkpoint = false;
  // Decode (one row, no capture) runs the MoE on the device without a host sync per layer (qwen35moe_grouped_decode.h) when the
  // layer's geometry and trellis experts allow it. -1 = R4DX_Q35_GROUPED (default on; 0 = the host-routed MoE), 0 / 1 force it.
  int grouped_decode = -1;
};

// Test hook: a named intermediate of the forward pass, handed to the callback with a DEVICE pointer that is valid on
// `stream` (the callback copies it out and synchronizes the stream itself). Names follow the tiny goldens
// (tools/reference/qwen35moe/tiny_golden.py): `layer{i}.residual_in | attn_in | attn_out | residual_attn | mlp_in |
// topk_ids | moe_out | residual_out`, `norm_out`, `logits`. The chunk's rows are sequence positions [first_pos, first_pos + rows).
struct Qwen35MoeCapture {
  enum class Dtype { kBf16, kI32, kF32 };
  std::string name;
  const void* dev = nullptr;
  Dtype dtype = Dtype::kBf16;
  int64_t rows = 0, cols = 0;
  int64_t first_pos = 0;
  hipStream_t stream = nullptr;
  bool on_host = false;  // `dev` points into HOST memory (topk_ids)
};
using Qwen35MoeCaptureFn = std::function<void(const Qwen35MoeCapture&)>;

struct Qwen35MoeChunk {
  int64_t rows = 0;       // 1..max_rows; <= 64, or a multiple of 64
  int64_t first_pos = 0;  // sequence position of row 0
  bool is_prefill = true; // false: a decode step (rows == 1) -- the GDN recurrent kernels, not the chunked scan
  // GDN layers of a prefill chunk take the recurrent decode kernels token by token instead of the chunked scan
  // (Qwen4ExpGdnCall::prefill_recurrent): the slow reference path. Ignored for a decode step.
  bool gdn_prefill_recurrent = false;
  // Test hook (layer-isolated runs): >= 0 runs ONLY the layer with this global index over the loaded residual, leaving the
  // other layers' state untouched. -1 = every layer of the stage.
  int64_t only_layer = -1;
};

struct Qwen35MoeHeadRequest {
  enum class Kind { kLogits, kArgmax, kSummary };
  Kind kind = Kind::kLogits;
  float inv_temperature = 0.0f;  // kSummary
};
struct Qwen35MoeHeadResult {
  std::vector<float> logits;  // kLogits: the last row's fp32 logits [vocab]
  int32_t token = -1;         // kArgmax
  kernels::RowSummary summary;  // kSummary
};

class Qwen35MoeStage {
 public:
  // `layers` are the stage's layers in order (global indices first, first + 1, ...); `head` makes it the last stage.
  // Allocates everything on options.device (stream, arena, residual / scratch buffers, per-layer state), zeroes the
  // state and synchronizes. Throws on a missing / mis-shaped weight.
  Qwen35MoeStage(const Qwen35MoeStageOptions& options, std::vector<Qwen35MoeLayerRefs> layers,
                 std::optional<Qwen35MoeHeadRefs> head);
  ~Qwen35MoeStage();
  Qwen35MoeStage(const Qwen35MoeStage&) = delete;
  Qwen35MoeStage& operator=(const Qwen35MoeStage&) = delete;

  int Device() const;
  hipStream_t Stream() const;
  int64_t FirstLayer() const;
  int64_t EndLayer() const;
  bool HasHead() const;

  // ---- chunk input: exactly one of the two per chunk, before RunLayers ----------------------------------------------
  // The first stage: the gathered embedding rows [rows, hidden] (host, pinned preferred) -> device residual.
  void LoadEmbedding(const uint16_t* host_rows, int64_t rows);
  // Any later stage: the previous stage's residual [rows, hidden] (host, pinned) -> device (async; the host buffer must
  // stay valid until the stage's stream has been synchronized: a prefill RunLayers does, a decode step's Head does).
  void LoadResidual(const uint16_t* host_rows, int64_t rows);

  // Runs the stage's layers over the loaded residual (arena reset between blocks). A prefill chunk synchronizes the stream
  // before returning; a decode step (rows == 1, !is_prefill) only ENQUEUES -- the caller's next synchronizing call
  // (StoreResidual, Head) orders everything after it.
  void RunLayers(const Qwen35MoeChunk& chunk);

  // The residual [rows, hidden] -> host (pinned preferred); synchronizes.
  void StoreResidual(uint16_t* host_rows, int64_t rows);
  // ---- the head (last stage): final norm + lm_head on the LAST row of the chunk just run ----------------------------
  Qwen35MoeHeadResult Head(int64_t rows, const Qwen35MoeHeadRequest& request);
  // The fp32 logits row of the last kLogits / kSummary head call (the sampler's row-granular fallback), D2H.
  std::vector<float> LastLogitsRow();

  // ---- sequence state -----------------------------------------------------------------------------------------------
  void Reset();  // GDN state; synchronous
  // Copies the live GDN state into the spare copy (needs options.prompt_checkpoint). Synchronous.
  void SaveCheckpoint();
  // Copies it back. The attention KV rows past the restored position are simply overwritten by the next chunk. Synchronous.
  void RestoreCheckpoint(int64_t pos);

  // Test hook (see Qwen35MoeCapture). Setting it also makes Head() compute the norm and the lm_head over EVERY row of the
  // chunk so `norm_out` / `logits` carry all rows; requires vocab * max_rows * 2 bytes of extra device memory.
  void SetCapture(Qwen35MoeCaptureFn fn);

  // Test hook: switches the grouped decode MoE on / off at run time (no effect on layers that do not qualify).
  // with_capture: a capture installed does NOT send the one-row MoE back to the host-routed path (it records moe_out, not topk_ids).
  void SetGroupedDecode(bool on, bool with_capture = false);
  // How many of this stage's layers take the grouped decode MoE (0 when it is off).
  int64_t GroupedDecodeLayers() const;

  // Wall milliseconds of the stage's last RunLayers / Head call (host clock).
  double LastRunMs() const;
  double LastHeadMs() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace r4dx::model
