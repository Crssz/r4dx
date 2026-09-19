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
#include "r4dx_convert/tensor_codec.hpp"

namespace r4dx_convert {

// kInt4Group (quant_int4.hpp) and kMxfp4Group (quant_mxfp4.hpp) are the group sizes used below.

struct LayoutSet {
  bool mxfp4 = false, w4a16 = false, w4a8 = false, bf16 = true;
};

inline void PlanLinearLayouts(ContainerWriter& writer, const std::string& base, int N, int K,
                               const LayoutSet& layouts) {
  // Fail before planning a single byte rather than truncating silently in the quantizers/packers
  // later (review finding, minor -- see quant_int4.hpp's RequireDivisible).
  if (layouts.w4a16 || layouts.w4a8 || layouts.mxfp4) RequireDivisible(N, 16, "N", base.c_str());
  if (layouts.w4a16 || layouts.w4a8) RequireDivisible(K, kInt4Group, "K", base.c_str());
  if (layouts.mxfp4) RequireDivisible(K, kMxfp4Group, "K", base.c_str());
  if (layouts.bf16)
    writer.Plan(base + ".bf16.w", {N, K, 2}, static_cast<uint64_t>(N) * K * 2);
  if (layouts.w4a16) {
    writer.Plan(base + ".w4a16.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(base + ".w4a16.wsz", {static_cast<int64_t>(N) * K / kInt4Group, 4},
                static_cast<uint64_t>(N) * K / kInt4Group * 4);
  }
  if (layouts.w4a8) {
    writer.Plan(base + ".w4a8.wq", {static_cast<int64_t>(N) * K / 2},
                static_cast<uint64_t>(N) * K / 2);
    writer.Plan(base + ".w4a8.ws", {static_cast<int64_t>(N) * K / kInt4Group, 4},
                static_cast<uint64_t>(N) * K / kInt4Group * 4);
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
                               int nthreads) {
  if (layouts.bf16) {
    auto bytes = EncodeBf16(w);
    writer.WriteTensor(base + ".bf16.w", bytes.data(), bytes.size());
  }
  if (layouts.w4a16) {
    std::vector<uint8_t> q, zero;
    std::vector<float> scale;
    QuantizeInt4Asymmetric(w.data(), N, K, kInt4Group, nthreads, q, scale, zero);
    auto wq = PackW4Nibbles(q, N, K, nthreads);
    auto wsz = PackW4A16Scales(scale, zero, N, K, kInt4Group);
    writer.WriteTensor(base + ".w4a16.wq", wq.data(), wq.size() * 4);
    writer.WriteTensor(base + ".w4a16.wsz", wsz.data(), wsz.size() * 4);
  }
  if (layouts.w4a8) {
    std::vector<uint8_t> q;
    std::vector<float> scale;
    QuantizeInt4SymmetricPinned8(w.data(), N, K, kInt4Group, nthreads, q, scale);
    auto wq = PackW4Nibbles(q, N, K, nthreads);
    auto ws = PackW4A8Scales(scale, N, K, kInt4Group);
    writer.WriteTensor(base + ".w4a8.wq", wq.data(), wq.size() * 4);
    writer.WriteTensor(base + ".w4a8.ws", ws.data(), ws.size() * 4);
  }
  if (layouts.mxfp4) {
    Mxfp4Quantized mq = QuantizeMxfp4(w.data(), N, K, kMxfp4Group, nthreads);
    auto wq = PackMxfp4Wq(mq.packed, N, K, nthreads);
    auto ws = PackMxfp4Ws(mq.escale, N, K, kMxfp4Group);
    writer.WriteTensor(base + ".mxfp4.wq", wq.data(), wq.size());
    writer.WriteTensor(base + ".mxfp4.ws", ws.data(), ws.size());
    writer.WriteTensor(base + ".mxfp4.wref", mq.wref.data(), mq.wref.size());
  }
}

}  // namespace r4dx_convert
