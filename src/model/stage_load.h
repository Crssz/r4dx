// HIP-free policy of the hybrid mode's stage-only Model loads (docs/pp-tp2-hybrid.md 1, 7, 9 "P0 prerequisites"): which layers
// of the container a stage holds, what a request for such a load has to satisfy, and the "is this layer range resident" check
// the stage entry points assert. Header-only so a CPU test (tests/model/test_stage_load_cpu.cpp) covers it without a device;
// Container (hole-prefix + norm-only layers) and Model::Load (ModelOptions::stage_only) are the two users.
//
// The two roles (X = HIP device 0 = PP stage A, Y = HIP device 1 = PP stage B / TP rank 0, split layer k):
//   kFront (X): layers [0, k) full, layer k NORM-ONLY (the input_layernorm that layer k-1's Mlp fuses -- the one thing the
//               PP stage A loads layer k for, pp_model.cpp's `reserve_split + 1`), no lm_head, no MTP head, no drafter, no
//               vision tower, no KV / GDN state for layer k.
//   kBack  (Y): layers [k, N) full, layers [0, k) HOLES (default-empty LayerWeights: Container::layers_ stays globally
//               indexed, NumLoadedLayers() stays N), final norm, the FULL lm_head and the FULL MTP head, no vision tower, no
//               drafter (the drafters live on the TP ranks).
// Both are off unless ModelOptions::stage_only.role says so; every other Model is kFull everywhere, as before.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r4dx::model::stage {

enum class Role { kOff, kFront, kBack };
inline const char* RoleName(Role r) { return r == Role::kFront ? "front (stage X)" : r == Role::kBack ? "back (stage Y)" : "off"; }

// What a container slot holds. kHole: nothing at all (a default-constructed LayerWeights). kNormOnly: input_layernorm only.
// kFull: every weight of the layer.
enum class LayerKind : uint8_t { kHole, kNormOnly, kFull };
inline const char* KindName(LayerKind k) { return k == LayerKind::kHole ? "a hole" : k == LayerKind::kNormOnly ? "norm-only" : "full"; }

inline LayerKind KindOf(int64_t layer, int64_t first_layer, int64_t norm_only_from) {
  if (layer < first_layer) return LayerKind::kHole;
  if (norm_only_from >= 0 && layer >= norm_only_from) return LayerKind::kNormOnly;
  return LayerKind::kFull;
}

inline std::vector<LayerKind> LayerKinds(int64_t num_layers, int64_t first_layer, int64_t norm_only_from) {
  std::vector<LayerKind> k(static_cast<size_t>(num_layers < 0 ? 0 : num_layers));
  for (int64_t i = 0; i < num_layers; ++i) k[static_cast<size_t>(i)] = KindOf(i, first_layer, norm_only_from);
  return k;
}

// The container-level request (ContainerLoadOptions::first_layer / norm_only_from) against the layers being loaded. "" = fine.
// norm_only_from: -1 = none, else layers [norm_only_from, num_layers) are norm-only and must leave at least one full layer.
inline std::string ValidateLayerRequest(int64_t num_layers, int64_t first_layer, int64_t norm_only_from) {
  if (num_layers < 1) return "no layers are loaded";
  if (first_layer < 0 || first_layer >= num_layers) {
    return "first_layer " + std::to_string(first_layer) + " must be in [0, " + std::to_string(num_layers) + ")";
  }
  if (norm_only_from < -1) return "norm_only_from must be -1 (none) or a layer index, got " + std::to_string(norm_only_from);
  if (norm_only_from >= 0 && (norm_only_from <= first_layer || norm_only_from >= num_layers)) {
    return "norm_only_from " + std::to_string(norm_only_from) + " must be in (first_layer " + std::to_string(first_layer) + ", " +
           std::to_string(num_layers) + ")";
  }
  return "";
}

// "" when every layer in [first, last) is kFull; else the reason, naming the first layer that is not. The stage entry points
// (Model::CheckLayersLoaded) throw with it: a hole executed silently would run null weights.
inline std::string CheckRangeLoaded(const std::vector<LayerKind>& kinds, int64_t first, int64_t last) {
  if (first < 0 || last < first || last > static_cast<int64_t>(kinds.size())) {
    return "layer range [" + std::to_string(first) + ", " + std::to_string(last) + ") is outside the " +
           std::to_string(kinds.size()) + " layers of the container";
  }
  for (int64_t i = first; i < last; ++i) {
    const LayerKind k = kinds[static_cast<size_t>(i)];
    if (k != LayerKind::kFull) {
      return "layer " + std::to_string(i) + " of the requested range [" + std::to_string(first) + ", " + std::to_string(last) +
             ") is " + KindName(k) + " in this stage-only Model";
    }
  }
  return "";
}

// ---- ModelOptions::stage_only -> the container request ------------------------------------------------------------------
struct RoleRequest {
  Role role = Role::kOff;
  int64_t split = 0;            // k
  int64_t layer_limit = -1;     // ModelOptions::layer_limit
  int tp_world = 1;
  int64_t mtp_draft_k = 0;
  int64_t dflash_draft_k = 0;
  bool dflash_container = false;
};

// "" when the combination is allowed. Stage-only Models are TP=1 Models (the TP rank Model next to them is the one with a
// comm); the drafters live on the ranks; the front stage runs no speculation at all (as the PP stage A: mtp_draft_k = 0).
inline std::string ValidateRole(const RoleRequest& q) {
  if (q.role == Role::kOff) return "";
  const char* who = q.role == Role::kFront ? "stage_only front (X)" : "stage_only back (Y)";
  if (q.tp_world != 1) return std::string(who) + " is a TP=1 Model (the TP rank Model is a separate one)";
  if (q.split < 1) return std::string(who) + " needs stage_only.split k >= 1, got " + std::to_string(q.split);
  if (q.dflash_container) return std::string(who) + " loads no drafter (dflash_container must be empty: the drafters live on the TP ranks)";
  if (q.role == Role::kFront) {
    if (q.mtp_draft_k != 0 || q.dflash_draft_k != 0) return std::string(who) + " runs no speculation (mtp_draft_k and dflash_draft_k must be 0)";
    if (q.layer_limit != -1 && q.layer_limit != q.split + 1) {
      return std::string(who) + " loads layers [0, k] (layer_limit k + 1 = " + std::to_string(q.split + 1) + "), got layer_limit " +
             std::to_string(q.layer_limit);
    }
  } else if (q.layer_limit >= 0 && q.layer_limit <= q.split) {
    return std::string(who) + " holds layers [k, N): layer_limit " + std::to_string(q.layer_limit) + " leaves none above k = " +
           std::to_string(q.split);
  }
  return "";
}

struct ContainerArgs {
  int64_t first_layer = 0;
  int64_t norm_only_from = -1;
  int64_t layer_limit = -1;
  bool skip_heads = false;  // no lm_head, no mtp.*
};
inline ContainerArgs ContainerArgsFor(const RoleRequest& q) {
  ContainerArgs a;
  a.layer_limit = q.layer_limit;
  if (q.role == Role::kFront) {
    a.norm_only_from = q.split;
    a.layer_limit = q.split + 1;
    a.skip_heads = true;
  } else if (q.role == Role::kBack) {
    a.first_layer = q.split;
  }
  return a;
}

}  // namespace r4dx::model::stage
