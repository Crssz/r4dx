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
#include <string>
#include <vector>

#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace r4dx_convert {

// kW4A16Group / kW4A8Group (quant_int4.hpp) and kMxfp4Group (quant_mxfp4.hpp) are the group sizes
// used below. w4a16's is a build option (R4DX_W4A16_GROUP) and need NOT equal w4a8's, so every
// site below names the one belonging to the layout it is emitting.

struct LayoutSet {
  bool mxfp4 = false, w4a16 = false, w4a8 = false, bf16 = true;
};

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
  if (layouts.w4a16) bytes += NK / 2 + NK / kW4A16Group * 4;
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
};

inline void PlanLinearLayouts(ContainerWriter& writer, const std::string& base, int N, int K,
                               const LayoutSet& layouts) {
  // Fail before planning a single byte rather than truncating silently in the quantizers/packers
  // later (review finding, minor -- see quant_int4.hpp's RequireDivisible).
  if (layouts.w4a16 || layouts.w4a8 || layouts.mxfp4) RequireDivisible(N, 16, "N", base.c_str());
  if (layouts.w4a16) RequireDivisible(K, kW4A16Group, "K", base.c_str());
  if (layouts.w4a8) RequireDivisible(K, kW4A8Group, "K", base.c_str());
  if (layouts.mxfp4) RequireDivisible(K, kMxfp4Group, "K", base.c_str());
  if (layouts.bf16)
    writer.Plan(base + ".bf16.w", {N, K, 2}, static_cast<uint64_t>(N) * K * 2);
  if (layouts.w4a16) {
    writer.Plan(base + ".w4a16.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(base + ".w4a16.wsz", {static_cast<int64_t>(N) * K / kW4A16Group, 4},
                static_cast<uint64_t>(N) * K / kW4A16Group * 4);
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
  const bool search = (opts.mode == QuantMode::kSearch);
  if (layouts.bf16) {
    auto bytes = EncodeBf16(w);
    writer.WriteTensor(base + ".bf16.w", bytes.data(), bytes.size());
  }
  if (layouts.w4a16) {
    std::vector<uint8_t> q, zero;
    std::vector<float> scale;
    if (search)
      QuantizeInt4AsymmetricSearch(w.data(), N, K, kW4A16Group, opts.importance, nthreads, q, scale,
                                   zero);
    else
      QuantizeInt4Asymmetric(w.data(), N, K, kW4A16Group, nthreads, q, scale, zero);
    auto wq = PackW4Nibbles(q, N, K, nthreads);
    auto wsz = PackW4A16Scales(scale, zero, N, K, kW4A16Group);
    writer.WriteTensor(base + ".w4a16.wq", wq.data(), wq.size() * 4);
    writer.WriteTensor(base + ".w4a16.wsz", wsz.data(), wsz.size() * 4);
  }
  if (layouts.w4a8) {
    std::vector<uint8_t> q;
    std::vector<float> scale;
    if (search)
      QuantizeInt4Pinned8Search(w.data(), N, K, kW4A8Group, opts.importance, nthreads, q, scale);
    else
      QuantizeInt4SymmetricPinned8(w.data(), N, K, kW4A8Group, nthreads, q, scale);
    auto wq = PackW4Nibbles(q, N, K, nthreads);
    auto ws = PackW4A8Scales(scale, N, K, kW4A8Group);
    writer.WriteTensor(base + ".w4a8.wq", wq.data(), wq.size() * 4);
    writer.WriteTensor(base + ".w4a8.ws", ws.data(), ws.size() * 4);
  }
  if (layouts.mxfp4) {
    Mxfp4Quantized mq = search
                            ? QuantizeMxfp4Search(w.data(), N, K, kMxfp4Group, opts.importance,
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
