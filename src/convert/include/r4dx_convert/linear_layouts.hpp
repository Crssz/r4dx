// r4dx_convert::linear_layouts -- plans and emits the `.{layout}` tensor family
// (docs/container-format.md "Quantized layout tensors") for one linear weight `W[N,K]`: `bf16`
// always, plus `w4a16` when the run requested it (mxfp4 and w4a8 are retired).
//
// Two free functions mirroring ContainerWriter's two phases: PlanLinearLayouts() registers every
// output tensor's name/shape/byte-size (pure function of N,K -- no data needed, so the whole
// model's header can be finalized before any shard is read), EmitLinearLayouts() does the actual
// quantize+pack+write for one already-loaded-as-float weight.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace r4dx_convert {

// kW4A16Group (quant_int4.hpp) is the w4a16 group size, a build option (R4DX_W4A16_GROUP). It is
// additionally PER LINEAR: LayoutSet::w4a16_group, the build default unless r4dx-convert's
// --w4a16-group-rule (w4a16_groups.hpp) resolved another one for this base.

struct LayoutSet {
  bool w4a16 = false, bf16 = true;
  // K per (scale, zero) pair of THIS linear's w4a16 layout (docs/quant2.md section 5). Anything but
  // kW4A16Group -- the container's default, __metadata__.quant.w4a16.group -- also renames the
  // scale tensor (W4a16WszName) and must be listed in __metadata__.quant.w4a16.groups. Ignored
  // when `w4a16` is false.
  int w4a16_group = kW4A16Group;
  // The trellis layout (docs/trellis-kernel.md 2; r4dx-convert --trellis-from, trellis_import.hpp):
  // `<base>.trellis.w` / `.suh` / `.svh`, IMPORTED from an oracle directory, never quantized here.
  // Exclusive -- a trellis linear has no other layout (3.4: no `.bf16.w` may defeat the old-binary
  // guard). `trellis_bits` is KB (4 or 5); `trellis_parts` the N of each fused part in container
  // row order ({17408, 17408} for mlp.gate_up; empty = one part of N). Ignored when `trellis` is
  // false.
  bool trellis = false;
  int trellis_bits = 0;
  std::vector<int64_t> trellis_parts;
};

// The trellis LayoutSet of one imported linear: that layout and nothing else.
inline LayoutSet TrellisLayoutSet(int bits, std::vector<int64_t> parts = {}) {
  LayoutSet ls;
  ls.w4a16 = ls.bf16 = false;
  ls.trellis = true;
  ls.trellis_bits = bits;
  ls.trellis_parts = std::move(parts);
  return ls;
}

// Number of fused parts of a trellis linear (1 when trellis_parts is empty).
inline int64_t TrellisPartCount(const LayoutSet& ls) {
  return ls.trellis_parts.empty() ? 1 : static_cast<int64_t>(ls.trellis_parts.size());
}

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
  ls.w4a16 = false;
  ls.bf16 = true;
  return ls;
}

// The tensor names PlanLinearLayouts() below plans for `base` in `layouts`, in its order (the shapes
// and sizes are PlanLinearLayouts'; tests/convert/test_reuse.cpp holds the two to the same names).
// r4dx-convert --reuse-tensors-from uses it to name exactly the tensors a BASELINE holds for a linear
// it recomputes, from the baseline's recorded LayoutSetId.
inline std::vector<std::string> LinearLayoutTensorNames(const std::string& base,
                                                        const LayoutSet& layouts) {
  std::vector<std::string> n;
  if (layouts.bf16) n.push_back(base + ".bf16.w");
  if (layouts.w4a16) {
    n.push_back(base + ".w4a16.wq");
    n.push_back(W4a16WszName(base, layouts.w4a16_group));
  }
  if (layouts.trellis) {
    n.push_back(base + ".trellis.w");
    n.push_back(base + ".trellis.suh");
    n.push_back(base + ".trellis.svh");
  }
  return n;
}

// One linear's RESOLVED layout set as a canonical string -- what r4dx-convert's reuse guard records
// per linear (reuse_guard.linears, docs/quant2.md 5.2): the layouts in the fixed order bf16, w4a16
// (with its own group, always spelled out), joined by '+', or "none". So a
// --keep-bf16 linear is "bf16", the Q3 sweep recipe's quantized linear "w4a16.g64". Everything a linear's tensors depend on beyond the run-wide flags is in it:
// its tensor names and sizes, which quantizers run (LDLQ applies iff a 4-bit layout is present), and
// at which w4a16 group. A trellis linear is "trellis.k4" / "trellis.k5" (its parts follow from its
// HF names, not from a flag, so they are not part of the id).
inline std::string LayoutSetId(const LayoutSet& ls) {
  std::string s;
  auto add = [&](const std::string& t) { s += (s.empty() ? "" : "+") + t; };
  if (ls.bf16) add("bf16");
  if (ls.w4a16) add("w4a16.g" + std::to_string(ls.w4a16_group));
  if (ls.trellis) add("trellis.k" + std::to_string(ls.trellis_bits));
  return s.empty() ? std::string("none") : s;
}

// The inverse of LayoutSetId, strict: only the canonical spelling of a layout set this build can write
// (a w4a16 group of 32/64/128 or the build default) parses; anything else returns false.
inline bool ParseLayoutSetId(const std::string& id, LayoutSet* out) {
  LayoutSet ls;
  ls.bf16 = false;
  if (id != "none") {
    size_t pos = 0;
    while (pos <= id.size()) {
      const size_t plus = id.find('+', pos);
      const std::string t = id.substr(pos, plus == std::string::npos ? std::string::npos : plus - pos);
      if (t == "bf16") {
        ls.bf16 = true;
      } else if (t.compare(0, 7, "w4a16.g") == 0 && t.size() > 7 && t.size() <= 10) {
        for (size_t i = 7; i < t.size(); ++i)
          if (t[i] < '0' || t[i] > '9') return false;
        ls.w4a16 = true;
        ls.w4a16_group = std::stoi(t.substr(7));
        if (ls.w4a16_group != kW4A16Group && !IsW4A16GroupSupported(ls.w4a16_group)) return false;
      } else if (t == "trellis.k4" || t == "trellis.k5") {
        ls.trellis = true;
        ls.trellis_bits = t.back() - '0';
      } else {
        return false;
      }
      if (plus == std::string::npos) break;
      pos = plus + 1;
    }
  }
  // Canonical only: the fixed order, each layout once, no empty token, no leading zeros.
  if (LayoutSetId(ls) != id) return false;
  // Trellis is exclusive (TrellisLayoutSet): "bf16+trellis.k4" is not a set this converter writes.
  if (ls.trellis && (ls.bf16 || ls.w4a16)) return false;
  *out = ls;
  return true;
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
  if (layouts.trellis) {
    bytes += NK * static_cast<uint64_t>(layouts.trellis_bits) / 8 +
             static_cast<uint64_t>(TrellisPartCount(layouts)) * static_cast<uint64_t>(K) * 2 +
             static_cast<uint64_t>(N) * 2;
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

// The trellis layout's shape rules (docs/trellis-kernel.md 2.1, 2.4, 2.5): KB in {4, 5}; K, N and
// every part a multiple of 128 (both Hadamards and the scale vectors work in 128-blocks, and every
// TP = 2 rank range must be 128-aligned); the parts sum to N; no other layout next to it.
inline void CheckTrellisLayout(const std::string& base, int64_t N, int64_t K, const LayoutSet& ls) {
  auto fail = [&](const std::string& why) {
    throw std::runtime_error("r4dx_convert: trellis linear " + base + " [" + std::to_string(N) +
                             "," + std::to_string(K) + "]: " + why);
  };
  if (ls.bf16 || ls.w4a16)
    fail("a trellis linear is written in the trellis layout only (docs/trellis-kernel.md 3.4)");
  if (ls.trellis_bits != 4 && ls.trellis_bits != 5)
    fail("KB=" + std::to_string(ls.trellis_bits) + " is not a rate the kernel instantiates (4, 5)");
  if (K % 128 != 0) fail("K is not a multiple of 128");
  if (N % 128 != 0) fail("N is not a multiple of 128");
  int64_t sum = 0;
  for (int64_t p : ls.trellis_parts) {
    if (p <= 0 || p % 128 != 0)
      fail("part N=" + std::to_string(p) + " is not a positive multiple of 128");
    sum += p;
  }
  if (!ls.trellis_parts.empty() && sum != N)
    fail("its parts sum to " + std::to_string(sum) + ", not N");
}

inline void PlanLinearLayouts(ContainerWriter& writer, const std::string& base, int N, int K,
                               const LayoutSet& layouts) {
  if (layouts.trellis) {
    CheckTrellisLayout(base, N, K, layouts);
    const uint64_t NK = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const int64_t P = TrellisPartCount(layouts);
    writer.Plan(base + ".trellis.w", {static_cast<int64_t>(NK * layouts.trellis_bits / 8)},
                NK * static_cast<uint64_t>(layouts.trellis_bits) / 8);
    writer.Plan(base + ".trellis.suh", {P * K, 2}, static_cast<uint64_t>(P * K) * 2);
    writer.Plan(base + ".trellis.svh", {N, 2}, static_cast<uint64_t>(N) * 2);
    return;
  }
  // Fail before planning a single byte rather than truncating silently in the quantizers/packers
  // later (review finding, minor -- see quant_int4.hpp's RequireDivisible).
  if (layouts.w4a16) RequireDivisible(N, 16, "N", base.c_str());
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
  if (layouts.bf16)
    writer.Plan(base + ".bf16.w", {N, K, 2}, static_cast<uint64_t>(N) * K * 2);
  if (layouts.w4a16) {
    writer.Plan(base + ".w4a16.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(W4a16WszName(base, g16), {static_cast<int64_t>(N) * K / g16, 4},
                static_cast<uint64_t>(N) * K / g16 * 4);
  }
}

inline void EmitLinearLayouts(ContainerWriter& writer, const std::string& base,
                               const std::vector<float>& w, int N, int K, const LayoutSet& layouts,
                               int nthreads, const QuantOptions& opts = QuantOptions{}) {
  // Trellis bits are imported from an oracle directory (TrellisSource::Emit), never computed from
  // `w`: a caller that gets here with a trellis LayoutSet lost track of which linear it holds.
  if (layouts.trellis)
    throw std::logic_error("EmitLinearLayouts: '" + base +
                           "' is a trellis linear -- its tensors are imported "
                           "(TrellisSource::Emit), not quantized");
  const LdlqFactor* ldlq = opts.ldlq;
  const bool search = (ldlq == nullptr) && (opts.mode == QuantMode::kSearch);
  // A factor for a different K would read U out of bounds (or, worse, in bounds with the wrong
  // stride). The CLI already checks the manifest's K during planning; this is the backstop for any
  // other caller.
  if (ldlq != nullptr && layouts.w4a16 && ldlq->K != K) {
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
}

}  // namespace r4dx_convert
