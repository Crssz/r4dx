// r4dx-convert's virtual per-expert source for HF 5.x fused MoE checkpoints (docs/qwen35moe-plan.md 1.1, M3).
//
// Ornith-1.5-122B (`qwen3_5_moe`) stores each layer's routed experts as TWO fused 3-D bf16 tensors
//
//   <p>.mlp.experts.gate_up_proj   bf16 [E, 2 * I, H]    expert e = rows [e * 2I, (e + 1) * 2I): gate rows [0, I), up rows [I, 2I)
//   <p>.mlp.experts.down_proj      bf16 [E, H, I]
//
// (HF Qwen3_5MoeExperts: `gate, up = linear(x, gate_up_proj[e]).chunk(2, -1)`), while the converter's qwen4_exp-style layout
// registers one linear per expert and projection. This overlay (installed on the ShardedModel with SetOverlay) presents each
// expert as three ordinary 2-D bf16 tensors with the per-expert names qwen4_exp's checkpoint uses
//
//   <p>.mlp.experts.{e}.gate_proj.weight  bf16 [I, H]     <p>.mlp.experts.{e}.up_proj.weight  bf16 [I, H]
//   <p>.mlp.experts.{e}.down_proj.weight  bf16 [H, I]
//
// and hides the two fused names from ShardedModel::AllNames() (the coverage audit then sees the virtual ones, so a missed
// expert shows up as unconsumed). Every virtual tensor is a CONTIGUOUS byte range of a fused tensor, so Data() / DataRows()
// return pointers straight into the shard mapping: no copy, no per-thread buffer, nothing mutable after Create().
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx_convert::fused_experts {

class FusedExpertsOverlay final : public TensorOverlay {
 public:
  // Scans model.AllNames() for `<p>.mlp.experts.gate_up_proj`; every such tensor must be bf16 [E, 2I, H] with a sibling
  // `<p>.mlp.experts.down_proj` bf16 [E, H, I], and no per-expert `<p>.mlp.experts.0.*` of its own. Throws
  // std::runtime_error naming the tensor otherwise. Installs nothing.
  static std::shared_ptr<FusedExpertsOverlay> Create(ShardedModel& model);

  // Number of fused (layer) groups found; 0 means the checkpoint has no fused experts.
  size_t GroupCount() const { return groups_.size(); }
  // Number of virtual per-expert tensors (3 per expert per group).
  size_t VirtualCount() const { return virt_.size(); }

  bool Provides(const std::string& name) const override;
  const TensorMeta& Meta(const std::string& name) const override;
  const uint8_t* Data(const std::string& name) const override;
  const uint8_t* DataRows(const std::string& name, int64_t row0, int64_t nrows,
                          std::vector<uint8_t>& scratch) const override;
  void RewriteNames(std::vector<std::string>& names) const override;

 private:
  struct Group {
    std::string prefix;  // "<p>.mlp.experts."
    int64_t E = 0, I = 0, H = 0;
  };
  enum class Kind { kGate, kUp, kDown };
  struct Virt {
    size_t group = 0;
    int64_t expert = 0;
    Kind kind = Kind::kGate;
    TensorMeta meta;
  };
  explicit FusedExpertsOverlay(ShardedModel& model) : model_(model) {}

  const Virt& Find(const std::string& name) const;
  const uint8_t* Base(const Virt& v) const;

  ShardedModel& model_;
  std::vector<Group> groups_;
  std::unordered_map<std::string, Virt> virt_;  // virtual name -> description
};

// Installs the overlay on `model` when the checkpoint has fused experts; returns the number of virtual expert tensors
// (0 = nothing fused, nothing installed).
size_t InstallFusedExpertsSource(ShardedModel& model);

}  // namespace r4dx_convert::fused_experts
