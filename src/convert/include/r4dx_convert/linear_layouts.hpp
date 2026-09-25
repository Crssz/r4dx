// r4dx_convert::linear_layouts -- plans and emits the `.{layout}` tensor family
// (docs/container-format.md "Quantized layout tensors") for one linear weight `W[N,K]`: `bf16`
// always, plus whichever of `mxfp4` / `w4a16` / `w4a8` the run requested.
//
// Two free functions mirroring ContainerWriter's two phases: PlanLinearLayouts() registers every
// output tensor's name/shape/byte-size (pure function of N,K -- no data needed, so the whole
// model's header can be finalized before any shard is read), EmitLinearLayouts() does the actual
// quantize+pack+write for one already-loaded-as-float weight.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace r4dx_convert {

// kW4A16Group / kW4A8Group (quant_int4.hpp) and kMxfp4Group (quant_mxfp4.hpp) are the group sizes
// used below. w4a16's is a build option (R4DX_W4A16_GROUP) and need NOT equal w4a8's, so every
// site below names the one belonging to the layout it is emitting. w4a16's is additionally
// PER LINEAR: LayoutSet::w4a16_group, the build default unless r4dx-convert's
// --w4a16-group-rule (w4a16_groups.hpp) resolved another one for this base.

struct LayoutSet {
  bool mxfp4 = false, w4a16 = false, w4a8 = false, bf16 = true;
  // K per (scale, zero) pair of THIS linear's w4a16 layout (docs/quant2.md section 5). Anything but
  // kW4A16Group -- the container's default, __metadata__.quant.w4a16.group -- also renames the
  // scale tensor (W4a16WszName) and must be listed in __metadata__.quant.w4a16.groups. Ignored
  // when `w4a16` is false.
  int w4a16_group = kW4A16Group;
};

// The name of a w4a16 linear's scale tensor. At the container's default group it is the historical
// `<base>.w4a16.wsz`; at any other group it is `<base>.w4a16.wsz.g<group>` (docs/container-format.md,
// w4a16, "Per-tensor groups"). The rename is the compatibility guard, not decoration: a binary that
// predates per-tensor groups looks for the bare `.w4a16.wsz`, does not find it, and refuses the
// tensor, instead of reading its scales at the default group's stride -- wrong numbers with no
// other symptom. (r4dx-convert refuses a non-default group on a linear that also keeps a
// `.bf16.w`, which such a binary would silently load instead; see W4a16GroupRules::Plan.)
inline std::string W4a16WszName(const std::string& base, int group) {
  return group == kW4A16Group ? base + ".w4a16.wsz"
                              : base + ".w4a16.wsz.g" + std::to_string(group);
}

// The LayoutSet a `--keep-bf16`-matched linear is written in: `<base>.bf16.w` and NOTHING else
// (src/convert/main.cpp's --keep-bf16, docs/validation.md "Milestone 11 / sensitivity"). Named here
// rather than spelled out at the call site because it is a contract with the LOADER, not a local
// choice: src/model/container.cpp's LoadQuantLinearWithFallback recognizes exactly this on-disk
// shape -- a base carrying only the bf16 form -- and falls that one linear back to bf16 while every
// other linear in the same container still loads in the requested quantized layout.
inline LayoutSet KeptBf16LayoutSet() {
  LayoutSet ls;
  ls.mxfp4 = ls.w4a16 = ls.w4a8 = false;
  ls.bf16 = true;
  return ls;
}

// Exactly the byte total PlanLinearLayouts() below plans for one [N,K] linear in `layouts`. Same
// formulas, derived once: --keep-bf16's "extra bytes vs 4-bit" accounting has to answer "what would
// this linear have cost in the layouts it is NOT being written in", and a second hand-written copy
// of these expressions would silently drift the moment a layout's scale tensor changes shape (as
// w4a16.wsz just did when R4DX_W4A16_GROUP became a build option).
inline uint64_t LinearLayoutBytes(int N, int K, const LayoutSet& layouts) {
  const uint64_t NK = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
  uint64_t bytes = 0;
  if (layouts.bf16) bytes += NK * 2;
  if (layouts.w4a16) bytes += NK / 2 + NK / layouts.w4a16_group * 4;
  if (layouts.w4a8) bytes += NK / 2 + NK / kW4A8Group * 4;
  if (layouts.mxfp4) {
    bytes += NK / 2 + static_cast<uint64_t>(K) / kMxfp4Group * static_cast<uint64_t>(N) +
             static_cast<uint64_t>(N);
  }
  return bytes;
}

// How the quantized layouts pick their (scale, zero) values. The BYTE LAYOUT is identical either
// way -- see quant_search.hpp. `kRtn` is the historical round-to-nearest min/max grid and is the
// default here so every caller that does not opt in (tests/convert, the DFlash2 path's own
// defaults) keeps producing byte-identical containers; r4dx-convert's CLI defaults to it too, and
// `kSearch` is reached only via an explicit `--quant search` (src/convert/main.cpp's header comment
// has the measurements behind that choice).
enum class QuantMode { kRtn, kSearch };

struct QuantOptions {
  QuantMode mode = QuantMode::kRtn;
  // Per-input-channel importance weights of length K, or empty for unweighted MSE. Ignored
  // entirely when mode == kRtn.
  ImportanceVector importance;
  // LDLQ error-feedback rounding (docs/quant2.md 2.2, quant_ldlq.hpp): when non-null, EVERY
  // quantized layout this call emits goes through the *Ldlq quantizer against this factor instead,
  // and `mode`/`importance` are ignored for this linear (the per-group grid inside LDLQ is weighted
  // by the factor's own diag(H), which is what the imatrix approximates anyway). Same byte layout,
  // same packers -- only the q/scale/zero values change, exactly like --quant. Non-owning: the
  // factor lives in r4dx_convert::HessianStore's cache (src/convert/main.cpp's --ldlq), and must
  // outlive the EmitLinearLayouts call. bf16 emission is unaffected.
  const LdlqFactor* ldlq = nullptr;
};

inline void PlanLinearLayouts(ContainerWriter& writer, const std::string& base, int N, int K,
                               const LayoutSet& layouts) {
  // Fail before planning a single byte rather than truncating silently in the quantizers/packers
  // later (review finding, minor -- see quant_int4.hpp's RequireDivisible).
  if (layouts.w4a16 || layouts.w4a8 || layouts.mxfp4) RequireDivisible(N, 16, "N", base.c_str());
  const int g16 = layouts.w4a16_group;
  if (layouts.w4a16) {
    // The build default is whatever this converter was compiled for (checked against the kernel at
    // startup); any other group must be one r4d_gemm_w4a16_nt_m64_g instantiates.
    if (g16 != kW4A16Group && !IsW4A16GroupSupported(g16)) {
      throw std::runtime_error("r4dx_convert: w4a16 group " + std::to_string(g16) + " for " + base +
                               " is not one the kernel instantiates (32, 64, 128) nor this "
                               "build's default (" + std::to_string(kW4A16Group) + ")");
    }
    RequireDivisible(K, g16, "K", base.c_str());
    // PackW4Nibbles' 64-K block: implied by the line above for any group >= 64, NOT for 32.
    RequireDivisible(K, 64, "K", base.c_str());
  }
  if (layouts.w4a8) RequireDivisible(K, kW4A8Group, "K", base.c_str());
  if (layouts.mxfp4) RequireDivisible(K, kMxfp4Group, "K", base.c_str());
  if (layouts.bf16)
    writer.Plan(base + ".bf16.w", {N, K, 2}, static_cast<uint64_t>(N) * K * 2);
  if (layouts.w4a16) {
    writer.Plan(base + ".w4a16.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(W4a16WszName(base, g16), {static_cast<int64_t>(N) * K / g16, 4},
                static_cast<uint64_t>(N) * K / g16 * 4);
  }
  if (layouts.w4a8) {
    writer.Plan(base + ".w4a8.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(base + ".w4a8.ws", {static_cast<int64_t>(N) * K / kW4A8Group, 4},
                static_cast<uint64_t>(N) * K / kW4A8Group * 4);
  }
  if (layouts.mxfp4) {
    writer.Plan(base + ".mxfp4.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(base + ".mxfp4.ws", {static_cast<int64_t>(K) / kMxfp4Group * N},
                static_cast<uint64_t>(K) / kMxfp4Group * N);
    writer.Plan(base + ".mxfp4.wref", {N}, static_cast<uint64_t>(N));
  }
}

inline void EmitLinearLayouts(ContainerWriter& writer, const std::string& base,
                               const std::vector<float>& w, int N, int K, const LayoutSet& layouts,
                               int nthreads, const QuantOptions& opts = QuantOptions{}) {
  const LdlqFactor* ldlq = opts.ldlq;
  const bool search = (ldlq == nullptr) && (opts.mode == QuantMode::kSearch);
  // A factor for a different K would read U out of bounds (or, worse, in bounds with the wrong
  // stride). The CLI already checks the manifest's K during planning; this is the backstop for any
  // other caller.
  if (ldlq != nullptr && (layouts.w4a16 || layouts.w4a8 || layouts.mxfp4) && ldlq->K != K) {
    throw std::runtime_error("EmitLinearLayouts: '" + base + "' has K=" + std::to_string(K) +
                             " but its LDLQ factor was built for K=" + std::to_string(ldlq->K));
  }
  if (layouts.bf16) {
    auto bytes = EncodeBf16(w);
    writer.WriteTensor(base + ".bf16.w", bytes.data(), bytes.size());
  }
  if (layouts.w4a16) {
    const int g16 = layouts.w4a16_group;  // this linear's own group (PlanLinearLayouts checked it)
    std::vector<uint8_t> q, zero;
    std::vector<float> scale;
    if (ldlq != nullptr)
      QuantizeInt4AsymmetricLdlq(w.data(), N, K, g16, *ldlq, nthreads, q, scale, zero);
    else if (search)
      QuantizeInt4AsymmetricSearch(w.data(), N, K, g16, opts.importance, nthreads, q, scale, zero);
    else
      QuantizeInt4Asymmetric(w.data(), N, K, g16, nthreads, q, scale, zero);
    auto wq = PackW4Nibbles(q, N, K, nthreads);
    auto wsz = PackW4A16Scales(scale, zero, N, K, g16);
    writer.WriteTensor(base + ".w4a16.wq", wq.data(), wq.size() * 4);
    writer.WriteTensor(W4a16WszName(base, g16), wsz.data(), wsz.size() * 4);
  }
  if (layouts.w4a8) {
    std::vector<uint8_t> q;
    std::vector<float> scale;
    if (ldlq != nullptr)
      QuantizeInt4Pinned8Ldlq(w.data(), N, K, kW4A8Group, *ldlq, nthreads, q, scale);
    else if (search)
      QuantizeInt4Pinned8Search(w.data(), N, K, kW4A8Group, opts.importance, nthreads, q, scale);
    else
      QuantizeInt4SymmetricPinned8(w.data(), N, K, kW4A8Group, nthreads, q, scale);
    auto wq = PackW4Nibbles(q, N, K, nthreads);
    auto ws = PackW4A8Scales(scale, N, K, kW4A8Group);
    writer.WriteTensor(base + ".w4a8.wq", wq.data(), wq.size() * 4);
    writer.WriteTensor(base + ".w4a8.ws", ws.data(), ws.size() * 4);
  }
  if (layouts.mxfp4) {
    Mxfp4Quantized mq =
        ldlq != nullptr ? QuantizeMxfp4Ldlq(w.data(), N, K, kMxfp4Group, *ldlq, nthreads)
        : search        ? QuantizeMxfp4Search(w.data(), N, K, kMxfp4Group, opts.importance,
                                              nthreads)
                        : QuantizeMxfp4(w.data(), N, K, kMxfp4Group, nthreads);
    auto wq = PackMxfp4Wq(mq.packed, N, K, nthreads);
    auto ws = PackMxfp4Ws(mq.escale, N, K, kMxfp4Group);
    writer.WriteTensor(base + ".mxfp4.wq", wq.data(), wq.size());
    writer.WriteTensor(base + ".mxfp4.ws", ws.data(), ws.size());
    writer.WriteTensor(base + ".mxfp4.wref", mq.wref.data(), mq.wref.size());
  }
}

}  // namespace r4dx_convert
