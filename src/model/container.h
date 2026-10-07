// r4dx::model::Container -- loads one .r4dx weight container (docs/container-format.md) into
// device memory and exposes it by name/layer. Reader half of the writer/reader contract
// src/convert/include/r4dx_convert/container_writer.hpp implements the writer half of.
//
// Layout selection happens once, at Load() time: every quantized linear in the model (attn.qg/o,
// gdn.in_proj_qkv/out_proj, mlp.gate_up/down) is uploaded in the SAME chosen `Layout`, except
// lm_head, which gets its own independent `lm_head_layout` (the task's "lm_head loaded in the
// chosen layout (bf16 variant if requested)" -- logits are usually kept at higher precision than
// the rest of the model). Tensors that only ever have one on-disk form (embeddings, norms,
// attn.k/v, descales, A_log/dt_bias, conv1d_weight) are uploaded as that form regardless of
// `layout`.
//
// Trellis (docs/trellis-kernel.md 2, 5.1): a container carrying `__metadata__.quant.trellis` has a
// trellis BODY (the 400 decoder linears) and loads only with `layout` kTrellis, and kTrellis only
// loads such a container. Its lm_head and MTP head stay w4a16 (or bf16), so a requested
// lm_head_layout / mtp_head_layout of kTrellis is read as kW4a16 by every Load -- the same answer
// for every caller, whether it passes the body layout for the heads (Model::Load) or not; a head
// with no w4a16 form (the `--lm-head bf16` twin) then takes the ordinary bf16 fallback.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model_config.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "rotation_meta.h"   // quant2 __metadata__.rotation (docs/quant2.md section 3.1)
#include "trellis_meta.h"    // __metadata__.quant.trellis (docs/trellis-kernel.md 2.3)
#include "vision_weights.h"  // src/vision: the vision.* tower weights (docs/vision.md)

namespace r4dx::model {

// Layout and QuantLinear live in quant_linear.h (shared with r4dx_model_attention -- see that
// file's comment for why).

// One decoder layer's weights (docs/container-format.md "Tensor naming"). Exactly one of
// {attn, gdn} is populated, selected by ModelConfig::IsGdnLayer(layer_idx).
struct AttnWeights {
  QuantLinear qg;   // fused q_proj + output gate, [num_heads*head_dim*2, hidden]
  // k/v (R1, docs/r9700.md): now QuantLinear like every other layout-eligible linear --
  // text.layers.{i}.attn.{k,v} carry w4a16/trellis/bf16 like attn.qg/o when the container was
  // converted with the new converter; Container::Load falls back to bf16 (or, for the oldest
  // pre-R1 containers, the bare single-tensor form) when the requested layout's tensors are
  // absent -- see container.cpp's LoadQuantLinearWithFallback. mtp.attn.k/v are always loaded
  // bf16 regardless of the requested body layout (docs/r9700.md R1 task: "mtp.* stay as they
  // are"), reusing this same field type.
  QuantLinear k, v;                               // [kv_heads*head_dim, hidden]
  QuantLinear o;                                  // [hidden, num_heads*head_dim]
  core::DeviceBuffer<uint16_t> q_norm, k_norm;    // bf16 [head_dim]
  core::DeviceBuffer<float> k_descale, v_descale;  // fp32 [kv_heads]
};

struct GdnWeights {
  QuantLinear in_proj_qkv;                         // [2*key_dim+value_dim, hidden]
  // in_proj_z (R1, docs/r9700.md): now QuantLinear -- 3.02 GB/token of what used to be forced
  // bf16, now eligible for w4a16/trellis like in_proj_qkv/out_proj. Same
  // LoadQuantLinearWithFallback fallback chain as attn.k/v above.
  QuantLinear in_proj_z;                            // [value_dim, hidden]
  core::DeviceBuffer<uint16_t> in_proj_b, in_proj_a;  // bf16 [num_v_heads, hidden] -- deliberately
                                                     // left bf16 (docs/r9700.md R1: "too small to
                                                     // matter, feed the decay path")
  // decode-t1 item 2 (docs/perf.md): in_proj_a's rows then in_proj_b's, bf16 [2 * num_v_heads, hidden],
  // built once at load (BuildGdnAb) so GdnLayer runs the one bf16 GEMM with N = 2 * num_v_heads instead
  // of two launches. A column's reduction does not depend on N (r4d_gemm_bf16_nt_m64 doc comment), so
  // each output is the same bits. Empty when a or b is absent or they differ in size.
  core::DeviceBuffer<uint16_t> in_proj_ab;
  core::DeviceBuffer<uint16_t> conv1d_weight;       // bf16 [conv_dim, width] (container's
                                                     // trailing singleton axis dropped)
  core::DeviceBuffer<float> A_log, dt_bias;         // fp32 [num_v_heads]
  core::DeviceBuffer<float> norm_weight;            // fp32 [head_v_dim] -- widened from the
                                                     // container's bf16 storage: both consumers
                                                     // (r4d_gdn_gated_rmsnorm_h128_bf16's `w` and
                                                     // r4d_gdn_recurrent_update_*'s `norm_weight`)
                                                     // take `const float*`, not bf16.
  QuantLinear out_proj;                             // [hidden, value_dim]
};

struct MlpWeights {
  QuantLinear gate_up;  // [2*intermediate, hidden]
  QuantLinear down;     // [hidden, intermediate]
};

struct LayerWeights {
  core::DeviceBuffer<uint16_t> input_layernorm;           // bf16 [hidden]
  core::DeviceBuffer<uint16_t> post_attention_layernorm;  // bf16 [hidden]
  std::optional<AttnWeights> attn;  // populated iff !IsGdnLayer(i)
  std::optional<GdnWeights> gdn;    // populated iff IsGdnLayer(i)
  MlpWeights mlp;
};

// MTP self-speculation head (docs/mtp.md, docs/container-format.md "mtp.*"): `layer` is the exact
// tensor set of one text full-attention decoder layer (src/convert/main.cpp's converter comment:
// "mtp.layers.0 is a COMPLETE full-attention decoder layer", verified against the real
// checkpoint's shard header -- NOT a GDN layer, so `layer.gdn` is always empty here and `layer.attn`
// always populated), plus the four tensors with no per-text-layer analogue: the hidden/embedding
// pre-norms and the fc projection that combines them (modeling_qwen3_5.py's Qwen3_5MTPLayer,
// cross-checked against the converter's own tensor-shape comment), and mtp's own final norm before
// the SHARED (main-model) lm_head. `fc` is a plain bf16 linear (container-format.md lists it with
// no `.{layout}` suffix, bf16-only) -- [hidden, 2*hidden], row-major, matching every other raw bf16
// linear this codebase calls via core::r4d::GemmBf16NtM64 (gdn_layer.cpp's in_proj_a/b/z).
struct MtpWeights {
  LayerWeights layer;
  core::DeviceBuffer<uint16_t> fc;                     // bf16 [hidden, 2*hidden]
  core::DeviceBuffer<uint16_t> norm;                   // bf16 [hidden]
  core::DeviceBuffer<uint16_t> pre_fc_norm_hidden;     // bf16 [hidden]
  core::DeviceBuffer<uint16_t> pre_fc_norm_embedding;  // bf16 [hidden]

  // Reduced-vocab draft head (docs/r9700.md R9, docs/mtp.md "reduced-vocab draft head"): an
  // OPTIONAL sliced lm_head over a subset of the real vocabulary, used ONLY to speed up
  // MtpHead::Draft's own drafting -- VERIFICATION always runs the real model's full-vocab lm_head
  // (Model::VerifyWindow, container_.LmHead()), never this one, which is what keeps the technique
  // lossless: a draft token this head would have produced but that the real model's full-vocab
  // verify pass does not confirm is simply rejected, exactly like a wrong full-vocab-head draft
  // would be -- an out-of-subset "true" next token is a lower ACCEPTANCE RATE, never a wrong
  // accepted answer. Both members are empty/unpopulated (draft_lm_head.N == 0) when the container
  // was converted without --draft-vocab-ids (old containers, or a run that chose not to build one)
  // -- MtpHead::Draft falls back to the full-vocab head unconditionally in that case, the same
  // "absent optional tensor -> old behavior" contract LoadQuantLinearWithFallback's callers use
  // elsewhere in this file, just simpler here (no on-disk fallback CHAIN, only "present or absent",
  // since this is a genuinely new/optional feature rather than a tensor that used to have one
  // fixed on-disk form).
  QuantLinear draft_lm_head;                    // [draft_vocab_size, hidden], N==0 if absent
  core::DeviceBuffer<int32_t> draft_vocab_ids;  // [draft_vocab_size]: subset index -> real vocab id
  bool HasDraftHead() const { return draft_lm_head.N > 0; }
};

// quant2 residual rotation (docs/quant2.md sections 3, 3.1, 4; docs/container-format.md "Residual
// rotation"): the container's rotation.* tensors, uploaded as stored (fp32, never regenerated from
// the seed). Present iff `__metadata__.rotation` is (rotation_meta.h's ParseRotationMetadata); the
// text layers' weights were then folded by the converter and are only correct together with Model's
// online ops -- Q at the stack entry, Q^T at the stack exit and on DFlash2 feature captures, and for
// q2ab the three input Hadamards below. Everything else (embed, final_norm, lm_head, mtp.*, vision.*)
// is untouched by the fold.
struct RotationWeights {
  RotationSpec spec;
  core::DeviceBuffer<float> signs;  // fp32 [hidden] (+-1): Q's diagonal D. Replicated under TP
  // fp32 [nblk * nblk], R[c][b] at c*nblk + b (rotation.mix5 when nblk == 5, rotation.mix otherwise;
  // the member keeps its historical name). Replicated under TP
  core::DeviceBuffer<float> mix5;
  // q2ab only; empty (data() == nullptr) for q2a. Under tensor parallelism each is this RANK's
  // slice, cut with exactly the K range the slicer gives the matching linear's columns (tp::RuleFor's
  // rotation.* rules), so a kernel indexes it with the rank-local column: [Config().intermediate_size]
  // for mlp.down, [num_attention_heads * head_dim] for attn.o, [ValueDim()] for gdn.out_proj.
  core::DeviceBuffer<float> had_down_signs;
  core::DeviceBuffer<float> had_o_signs;
  core::DeviceBuffer<float> had_gdn_out_signs;
  core::DeviceBuffer<float> had_o_full_signs;  // Gemma 4 option A only (spec.has_o_full)
};

// Every Container::Load knob in one struct (docs/tp.md 3.3). The positional Load below forwards to
// Load(path, ContainerLoadOptions) with tp_world = 1, so every pre-TP caller is unchanged.
//
// Tensor parallel (tp_world > 1, docs/tp.md 5.1): the container is loaded as rank `tp_rank`'s SHARD
// -- every tensor is classified by tp::RuleFor (src/model/tp/tp_shard.h) and only this rank's byte
// ranges are uploaded, straight from the mmap when they are one contiguous range and through a
// reusable host staging buffer otherwise. Config() is then the rank-local config
// (ModelConfig::Shard) and every QuantLinear carries the RANK's N/K; GlobalConfig() is the
// container's own. The TP-only fields below must keep their defaults at tp_world == 1 (Load throws
// std::invalid_argument otherwise): that path is the pre-TP loader, untouched.
struct ContainerLoadOptions {
  Layout layout = Layout::kBf16, lm_head_layout = Layout::kBf16, mtp_head_layout = Layout::kBf16;
  int64_t layer_limit = -1;
  bool embed_device_resident = true;
  // TP: >= 0 overrides the free-VRAM heuristic for the embedding device mirror (1 = mirror it, 0 =
  // host-only), so every rank takes the same gather path (docs/tp.md 2.9 step 5). -1: the heuristic.
  int embed_device_resident_decided = -1;
  bool load_vision = false;          // upload vision.* weights on THIS rank (TP: rank 0 only)
  bool parse_vision_config = false;  // TP: parse vision_config even when not uploading (rank > 0)
  int tp_world = 1, tp_rank = 0;
  // TP: the process's ONE pinned host copy of text.embed_tokens (LoadEmbedTokensHost), shared by
  // every rank instead of each rank pinning its own 2.37 GiB. nullptr: this Load pins its own.
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host;
};

class Container {
 public:
  // Loads `path` onto the current HIP device (caller must have already selected device 1 per the
  // project's GPU rule), synchronously (Container::Load is a startup-path call, not a hot-path
  // one -- every upload below is a plain synchronous DeviceBuffer::CopyFromHost). `layer_limit`,
  // when >= 0, loads only layers [0, layer_limit) -- for the 4-layer test container -- and leaves
  // Layers() sized to layer_limit rather than Config().num_hidden_layers. `mtp_head_layout`
  // (docs/mtp.md "MTP head layout"): the layout used ONLY for the MTP head's four quantized
  // linears (mtp.attn.qg/o, mtp.mlp.gate_up/down) -- independent of `layout`, since the draft
  // head's errors compound across chained draft steps (ModelOptions::mtp_head_layout's own
  // comment). Ignored when the container has no mtp.* weights. `embed_device_resident` (docs/
  // mtp.md "device-resident draft loop", r9700.md P3): if true (default), also mirrors
  // text.embed_tokens into VRAM (~2.54 GB bf16) so RunChunk/MtpHead::Draft can gather embedding
  // rows entirely on-device (EmbedTokensDevice() below) -- checked against free VRAM at the point
  // this call uploads it (before any layer, i.e. against the WHOLE GPU's free memory, not the
  // eventual steady-state footprint after every layer/KV cache is also loaded) with a conservative
  // 2x-headroom heuristic; falls back to host-only (a stderr warning, EmbedTokensDevice() returns
  // nullptr) if that heuristic says it would not fit. EmbedTokensHost() is always populated
  // regardless of this flag -- device residency is purely an additional mirror, never a
  // replacement, so every existing host-gather call site keeps working unchanged.
  // `load_vision` (docs/vision.md "Load policy"): upload the container's 333 `vision.*` tensors
  // (~0.90 GiB bf16) as well. Default FALSE, so a text-only run pays exactly what it paid before
  // this parameter existed -- the "auto" policy lives one level up, in Model::Load/ModelOptions,
  // where it can see whether the caller asked for vision at all. Silently a no-op (not an error)
  // when the container has no vision.* tensors: the 4-layer test container is exactly that case,
  // and HasVision() below is how a caller finds out.
  static Container Load(const std::string& path, Layout layout, Layout lm_head_layout,
                         int64_t layer_limit = -1, Layout mtp_head_layout = Layout::kBf16,
                         bool embed_device_resident = true, bool load_vision = false);
  // The same, with every knob in ContainerLoadOptions (docs/tp.md 3.3) -- the only entry point for
  // a tensor-parallel shard (o.tp_world > 1).
  static Container Load(const std::string& path, const ContainerLoadOptions& o);

  // text.embed_tokens of `path`, read into ONE pinned host buffer allocated with
  // hipHostMallocPortable (so every device in the process can DMA from it) -- the tensor-parallel
  // process loads it once and hands it to every rank (ContainerLoadOptions::shared_embed_host).
  static std::shared_ptr<const core::PinnedBuffer<uint16_t>> LoadEmbedTokensHost(
      const std::string& path);

  // The RANK-local config under tensor parallelism (ModelConfig::Shard of GlobalConfig()); the
  // container's own config at tp_world == 1, where the two are equal.
  const ModelConfig& Config() const { return config_; }
  // The container's own (unsharded) config, whatever tp_world this Load used (docs/tp.md 3.2).
  const ModelConfig& GlobalConfig() const { return global_config_; }
  const std::string& ModelId() const { return model_id_; }
  const std::string& ConfigSha256() const { return config_sha256_; }

  // text.embed_tokens: host-resident (docs/architecture.md), [vocab, hidden] bf16, row-major. The
  // shared tensor-parallel copy (ContainerLoadOptions::shared_embed_host) when one was given.
  const uint16_t* EmbedTokensHost() const {
    return shared_embed_host_ ? shared_embed_host_->data() : embed_tokens_.data();
  }

  // text.embed_tokens' device mirror (docs/mtp.md "device-resident draft loop") -- nullptr when
  // Load() was called with embed_device_resident=false, or when it was true but the free-VRAM
  // heuristic decided it would not fit (see Load()'s own comment); check
  // EmbedTokensDeviceResident() rather than relying on this being non-null implicitly.
  const uint16_t* EmbedTokensDevice() const { return embed_tokens_dev_.data(); }
  bool EmbedTokensDeviceResident() const { return !embed_tokens_dev_.empty(); }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const LayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }

  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }

  // True iff `path` was converted with --mtp on (docs/container-format.md "mtp.*") -- checked once
  // at Load() time via SafetensorsReader::Has, not inferred from `__metadata__.r4dx_convert_run.mtp`
  // (a JSON round-trip the loader would otherwise need just to answer this), so it stays correct
  // even against a hand-built or metadata-stripped test container.
  bool HasMtp() const { return mtp_.has_value(); }
  const MtpWeights& Mtp() const { return mtp_.value(); }

  // True iff the container carries `__metadata__.rotation` (quant2, docs/quant2.md section 3.1) --
  // decided by the metadata key alone, never by probing for rotation.* tensors: a container without
  // the key gets no rotation op anywhere, and one with a kind this binary does not implement already
  // failed Load(). Rotation() is valid iff this is.
  bool HasRotation() const { return rotation_.has_value(); }
  const RotationWeights& Rotation() const { return rotation_.value(); }

  // True iff the container carries `__metadata__.quant.trellis` (docs/trellis-kernel.md 2.3), i.e.
  // its body linears are trellis (Load refuses any other body layout for it). Trellis() is valid
  // iff this is.
  bool HasTrellis() const { return trellis_.has_value(); }
  const TrellisSpec& Trellis() const { return trellis_.value(); }
  // docs/trellis-kernel.md 4.5: every trellis linear's split-group tickets live in ONE device
  // buffer of this Container (each QuantLinear::trellis_tickets is its slice), zeroed at load. A
  // launch that completes leaves its tickets zero again, so this only matters after one that did
  // not (a device fault or TDR); Model::Reset() calls it. One hipMemsetAsync on `stream`; a no-op
  // for a container without trellis linears.
  void ZeroTrellisTickets(hipStream_t stream);
  // R4DX_FAKEQ_W (fake_quant_w.h, docs/int8-prefill.md), Model::Load only, and only when the switch is on:
  // BuildTrellisWScale (linear.h) for every trellis linear -- the per-(column, k group) scale tables the
  // rounding prefill GEMMs read. No-op for mode 0 or a container without trellis linears.
  void BuildTrellisWScales(int mode, hipStream_t stream);
  // R4DX_PREFILL_INT8 (prefill_int8.h, docs/int8-prefill.md "Production path"), Model::Load only, and only when the
  // switch is on and the Model uses it: BuildTrellisI8Scale (linear.h) for every trellis linear that has an int8
  // plan -- the weight scale tables the int8 prefill GEMM quantizes with (K * N / 32 bytes each). Returns how many
  // tables were built; *trellis_total (if non-null) receives how many trellis linears there are. Stream-ordered.
  int BuildTrellisI8Scales(hipStream_t stream, int* trellis_total = nullptr);

  // The vision tower's weights (docs/vision.md), present only when Load() was called with
  // load_vision=true AND the container actually carries vision.* tensors. HasVisionTensors() is
  // the second of those two questions on its own -- a caller that wants to say "this container
  // COULD do vision but vision is off" needs to distinguish them.
  bool HasVision() const { return vision_.has_value(); }
  const vision::VisionWeights& Vision() const { return vision_.value(); }
  bool ContainerHasVisionTensors() const { return container_has_vision_tensors_; }
  // True iff this Container knows the vision geometry: the tower was uploaded (HasVision()), or --
  // a tensor-parallel rank that does not hold the tower (ContainerLoadOptions::parse_vision_config)
  // -- only its vision_config was parsed (docs/tp.md 4.2, 8.3). VisionCfg() is valid iff this is.
  bool HasVisionConfig() const { return vision_.has_value() || vision_config_.has_value(); }
  const vision::VisionConfig& VisionCfg() const {
    return vision_.has_value() ? vision_->config : vision_config_.value();
  }

  // The multimodal placeholder token ids, from the TOP level of `__metadata__.model_config` (they
  // sit next to "text_config"/"vision_config", not inside either -- docs/vision.md "Model facts").
  // Read from the container rather than hardcoded so a differently-tokenized checkpoint splices at
  // its own ids; the defaults are this checkpoint's, for a container converted before these keys
  // were carried. -1 is never a valid token id, so a caller can tell "absent" from "0".
  int64_t ImageTokenId() const { return image_token_id_; }
  int64_t VideoTokenId() const { return video_token_id_; }

 private:
  Container() = default;
  // Model (model.h/model.cpp) default-constructs a Model whose Container member is filled in by
  // Container::Load() right after -- same "private default ctor, public static Load() factory"
  // pattern Container itself uses, one level up.
  friend class Model;

  // The tensor-parallel shard path of Load(path, ContainerLoadOptions) (docs/tp.md 5.1).
  static Container LoadShard(const std::string& path, const ContainerLoadOptions& o);

  // Calls `fn(QuantLinear&)` for every linear this Container holds (body, lm_head, MTP head).
  template <class Fn>
  void ForEachLinear(Fn&& fn);
  // Allocates trellis_tickets_ (zeroed) with one N/128 slice per trellis linear and points each
  // linear's trellis_tickets at its slice. Called once, at the end of both loaders.
  void AssignTrellisTickets();

  ModelConfig config_;
  ModelConfig global_config_;  // == config_ at tp_world == 1
  int64_t image_token_id_ = 248056;  // C:\AI\models\Qwen3.8-27B\config.json, top level
  int64_t video_token_id_ = 248057;
  std::string model_id_, config_sha256_;
  core::PinnedBuffer<uint16_t> embed_tokens_;  // empty when shared_embed_host_ is set
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host_;
  core::DeviceBuffer<uint16_t> embed_tokens_dev_;  // empty iff not device-resident (Load's own
                                                    // comment) -- see EmbedTokensDeviceResident()
  std::vector<LayerWeights> layers_;
  core::DeviceBuffer<uint16_t> final_norm_;
  QuantLinear lm_head_;
  std::optional<MtpWeights> mtp_;
  std::optional<RotationWeights> rotation_;  // iff __metadata__.rotation
  std::optional<TrellisSpec> trellis_;       // iff __metadata__.quant.trellis
  core::DeviceBuffer<uint32_t> trellis_tickets_;  // every trellis linear's tickets (4.5)
  std::optional<vision::VisionWeights> vision_;
  std::optional<vision::VisionConfig> vision_config_;  // parsed-only (TP rank without the tower)
  bool container_has_vision_tensors_ = false;
};

}  // namespace r4dx::model
