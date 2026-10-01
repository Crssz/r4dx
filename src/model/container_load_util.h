// r4dx::model::container_util -- the container-loading helpers Container::Load (the Qwen loader, container.cpp)
// and GemmaContainer::Load (gemma_container.cpp) share: the `__metadata__` read, the w4a16-group and
// trellis checks, the raw-tensor uploads, the per-linear layout fallback chain (LoadQuantLinearWithFallback)
// and the rotation.* tensor loader.
//
// MECHANICAL EXTRACTION (docs/gemma4-plan.md M1-19): every function and struct below was moved verbatim out
// of container.cpp's anonymous namespace -- bodies unchanged, only their home and linkage (now external, in
// r4dx::model::container_util) differ -- so the Qwen loader runs exactly the code it ran before. The one
// addition is LoadRotationWeights' trailing `o_full_elems` parameter (Gemma's full-attention o_proj K,
// default 0 = RotationTensors' own default), which Container::Load never passes.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "container.h"  // RotationWeights
#include "nlohmann/json.hpp"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "rotation_meta.h"
#include "trellis_meta.h"
#include "w4a16_group_meta.h"

namespace r4dx::model::container_util {

using r4dx_convert::SafetensorsReader;
using r4dx_convert::Utf8ToWide;

// Reads just the safetensors-shell header (8-byte length + JSON) to pull out `__metadata__`.
nlohmann::json ReadMetadata(const std::string& path);
// Refuses a container whose w4a16 group disagrees with the group this build's
// r4d_gemm_w4a16_nt_m64 was compiled with -- see CheckW4a16Group's comment in quant_linear.h for
// why a mismatch is silent-wrong-numbers rather than a crash.
//
// A container written before the group was recorded has no `quant` block at all; those are group
// 128 by construction (it was the only group that ever existed), which no build reads any more.
// This check skips them; CheckW4a16Shape (and the TP loader's part-size check) refuses their scales
// instead, since both size them at the group the kernel will read (W4a16LoadGroups::KernelGroup),
// not at the unrecorded default.
//
// SCOPE (adversarial-review fix): the check fires only when THIS load will actually read
// `.w4a16.wsz` bytes, i.e. when one of the three layout selections below is `kW4a16`. The hazard
// the guard exists to stop is a w4a16 GEMM striding the scales wrongly; a `--layout bf16` run
// never touches a `.w4a16.*` tensor (LoadQuantLinear reads only the requested layout's tensors,
// and LoadQuantLinearWithFallback's fallback chain is requested -> bf16 -> bare, never -> w4a16).
// Refusing those runs bought no safety and cost real capability.
//
// quant2 Q3 (docs/quant2.md section 5.1): per-tensor groups. The map (`quant.w4a16.groups`) is
// parsed on every load -- ParseW4a16Groups is structural and never throws for a container without
// one -- and the result travels with the load as W4a16LoadGroups, which every w4a16 read resolves
// its linear's group and scale-tensor name through. The kernel checks stay gated as above:
//   - no map (every container converted without --w4a16-group-rule): exactly the check above, up
//     front, unchanged;
//   - a map: every mapped group must be one r4d_gemm_w4a16_nt_m64_g instantiates, up front
//     (CheckW4a16MappedGroup); the default-group equality then applies only to the linears WITHOUT
//     a map entry, so it runs when the first such linear is actually read as w4a16
//     (W4a16LoadGroups::Resolve), with the same W4a16GroupMismatch.
struct W4a16LoadGroups {
  W4a16Groups groups;
  bool check_default_per_linear = false;
  std::string path;

  // QuantLinear::w4a16_group for `base`, about to be read as w4a16.
  int Resolve(const std::string& base) const {
    if (check_default_per_linear && !groups.Mapped(base)) {
      CheckW4a16Group(groups.default_group, path);
    }
    return groups.QuantLinearGroup(base);
  }

  // The group the kernel will stride `base`'s scales at: its mapped group, else this build's
  // default (QuantLinear::w4a16_group 0 -> r4d_gemm_w4a16_nt_m64_group()). The scale tensor's size
  // is checked (CheckW4a16Shape) and its TP slice cut at THIS group, not at the container's
  // recorded default: a container with no `quant.w4a16.group` parses to default 128 and skips
  // CheckW4a16Group, so only the size check stands between its N*K/128 scale
  // dwords and a kernel that reads N*K/64 of them.
  int KernelGroup(const std::string& base) const {
    return groups.Mapped(base) ? groups.GroupFor(base) : r4d_gemm_w4a16_nt_m64_group();
  }
};

W4a16LoadGroups CheckQuantGroups(const nlohmann::json& metadata, const std::string& path,
                                 Layout layout, Layout lm_head_layout, Layout mtp_head_layout);
// Everything a linear's load reads from `__metadata__` besides its shape: the w4a16 groups (quant2
// Q3) and the trellis block (docs/trellis-kernel.md 2.3; nullopt for a container without one).
struct LinearLoadMeta {
  W4a16LoadGroups w4a16;
  std::optional<TrellisSpec> trellis;
  std::string path;
};

// docs/trellis-kernel.md 2.5, the refusals that need only the metadata and the requested body layout.
// `allow_rotated`: the Gemma loader accepts trellis + __metadata__.rotation (option A, docs/gemma4-plan.md 10.2);
// Qwen's refuses the pair (docs/trellis-kernel.md 3.4).
void CheckTrellisChoice(const std::optional<TrellisSpec>& trellis, bool rotated, Layout layout,
                        const std::string& path, bool allow_rotated = false);
// Every `.trellis.*` tensor belongs to a `linears` entry and every entry has its three tensors.
void CheckTrellisTensors(const SafetensorsReader& r, const std::optional<TrellisSpec>& trellis,
                         const std::string& path);
// One trellis linear's entry, which CheckTrellisTensors has matched with its tensors.
const TrellisLinearSpec& TrellisFor(const LinearLoadMeta& meta, const std::string& base);
void CheckTrellisShape(const SafetensorsReader& r, const std::string& base,
                       const TrellisLinearSpec& t, int64_t N, int64_t K, const std::string& path);
void SetTrellisFields(QuantLinear& q, const TrellisLinearSpec& t);
void LogTrellis(const TrellisSpec& spec, const std::string& path);
bool TakeTrellisBf16Head(const LinearLoadMeta& meta, const QuantLinear& head, int fallbacks_before,
                         int* fallbacks);
void LogTrellisBf16Head(const std::string& path);

int64_t ElemCountBySize(const SafetensorsReader& r, const std::string& name, int64_t elem_bytes);
core::DeviceBuffer<uint16_t> UploadRawU16(const SafetensorsReader& r, const std::string& name);
core::DeviceBuffer<float> UploadRawF32(const SafetensorsReader& r, const std::string& name);
core::DeviceBuffer<uint8_t> UploadRawU8(const SafetensorsReader& r, const std::string& name);
core::DeviceBuffer<uint32_t> UploadRawU32(const SafetensorsReader& r, const std::string& name);
core::DeviceBuffer<int32_t> UploadRawI32(const SafetensorsReader& r, const std::string& name);
// bf16 on disk, fp32 on device (gdn.norm_weight).
core::DeviceBuffer<float> UploadWidenedF32(const SafetensorsReader& r, const std::string& name);
// fp16 bytes -> fp32 device (trellis suh / svh).
core::DeviceBuffer<float> UploadWidenedF16Bytes(const uint8_t* src, size_t bytes);
core::DeviceBuffer<float> UploadWidenedF16(const SafetensorsReader& r, const std::string& name);

void CheckW4a16Shape(const SafetensorsReader& r, const std::string& base, const std::string& wsz,
                     int64_t N, int64_t K, int group);
void CheckW4a16GroupTensors(const SafetensorsReader& r, const W4a16Groups& groups,
                            const std::string& path);
void LogW4a16Groups(const W4a16Groups& groups, const std::string& path);

QuantLinear LoadQuantLinear(const SafetensorsReader& r, const LinearLoadMeta& meta,
                            const std::string& base, Layout layout, int64_t N, int64_t K);
// True iff `r` carries every tensor LoadQuantLinear(r, meta, base, layout, ...) would read.
bool HasLayout(const SafetensorsReader& r, const LinearLoadMeta& meta, const std::string& base,
               Layout layout);
// The three-tier fallback chain (requested layout -> bf16 -> bare tensor).
QuantLinear LoadQuantLinearWithFallback(const SafetensorsReader& r, const LinearLoadMeta& meta,
                                        const std::string& base, Layout requested, int64_t N,
                                        int64_t K, int* fallbacks = nullptr);
// quant2 (docs/quant2.md section 3.1): the rotation.* tensors a container with
// `__metadata__.rotation` must carry. Each is checked on disk against the GLOBAL shape
// (rotation_meta.h's RotationTensors), its values are checked on the host (every sign exactly +-1,
// mix5 orthogonal -- the kernels multiply by them, so a corrupt value would silently scale or skew
// the residual rather than fail), then it is uploaded through `upload` (UploadRawF32 at TP=1,
// ShardLoader::Raw<float> under tensor parallelism, which cuts the q2ab sign vectors by
// tp::RuleFor) and the upload is checked against the LOCAL shape (`local` = the rank config).
template <class UploadF32>
RotationWeights LoadRotationWeights(const SafetensorsReader& r, const RotationSpec& spec,
                                    const ModelConfig& global, const ModelConfig& local,
                                    const std::string& path, UploadF32&& upload,
                                    int64_t o_full_elems = 0, int64_t o_full_elems_local = -1) {
  // `o_full_elems_local` (Gemma TP, M1b-1): the o_full sign vector's length on this rank (heads/world * global_head_dim);
  // -1 = the global length, i.e. TP=1 and every Qwen caller.
  const std::vector<RotationTensor> want = RotationTensors(spec, global, o_full_elems);
  const std::vector<RotationTensor> want_local =
      RotationTensors(spec, local, o_full_elems_local >= 0 ? o_full_elems_local : o_full_elems);
  RotationWeights w;
  w.spec = spec;
  // Destination per tensor name (RotationTensors' list is signs, mix, then the q2ab sign vectors the
  // spec carries; the mix tensor is rotation.mix5 or rotation.mix by nblk).
  const auto dest = [&w](const std::string& name) -> core::DeviceBuffer<float>* {
    if (name == kRotationSigns) return &w.signs;
    if (name == kRotationMix5 || name == kRotationMix) return &w.mix5;
    if (name == kRotationHadDownSigns) return &w.had_down_signs;
    if (name == kRotationHadOSigns) return &w.had_o_signs;
    if (name == kRotationHadGdnOutSigns) return &w.had_gdn_out_signs;
    if (name == kRotationHadOFullSigns) return &w.had_o_full_signs;
    throw std::logic_error("LoadRotationWeights: no destination for '" + name + "'");
  };
  const std::string kind = RotationKindName(spec.kind);
  for (size_t i = 0; i < want.size(); ++i) {
    const std::string name = want[i].name;
    core::DeviceBuffer<float>* const dst_i = dest(name);
    if (!r.Has(name)) {
      throw std::runtime_error("r4dx::model::Container: " + path + " has __metadata__.rotation kind " +
                               kind + " but no tensor '" + name + "'");
    }
    const int64_t n = ElemCountBySize(r, name, 4);
    if (n != want[i].elems) {
      throw std::runtime_error("r4dx::model::Container: rotation tensor '" + name + "' in " + path +
                               " has " + std::to_string(n) + " fp32 elements, the model needs " +
                               std::to_string(want[i].elems));
    }
    std::vector<float> host(static_cast<size_t>(n));
    std::memcpy(host.data(), r.Data(name), static_cast<size_t>(n) * sizeof(float));
    if (name == kRotationMix5 || name == kRotationMix) {
      const int64_t nb = spec.nblk;
      for (int64_t a = 0; a < nb; ++a) {
        for (int64_t b = 0; b < nb; ++b) {
          double dot = 0.0;
          for (int64_t k = 0; k < nb; ++k) {
            dot += static_cast<double>(host[a * nb + k]) * host[b * nb + k];
          }
          if (!(std::fabs(dot - (a == b ? 1.0 : 0.0)) <= 1e-5)) {
            throw std::runtime_error("r4dx::model::Container: " + name + " in " + path +
                                     " is not orthogonal (R R^T deviates from I by more than 1e-5)");
          }
        }
      }
    } else {
      for (int64_t j = 0; j < n; ++j) {
        const float v = host[static_cast<size_t>(j)];
        if (v != 1.0f && v != -1.0f) {
          throw std::runtime_error("r4dx::model::Container: rotation tensor '" + name + "' in " +
                                   path + " has element " + std::to_string(j) + " = " +
                                   std::to_string(v) + ", not +-1");
        }
      }
    }
    *dst_i = upload(name);
    if (static_cast<int64_t>(dst_i->size()) != want_local[i].elems) {
      throw std::logic_error("r4dx::model::Container: this rank's slice of '" + name + "' has " +
                             std::to_string(dst_i->size()) + " elements, the rank config needs " +
                             std::to_string(want_local[i].elems));
    }
  }
  return w;
}

void LogRotation(const RotationSpec& spec, const std::string& path);

}  // namespace r4dx::model::container_util
