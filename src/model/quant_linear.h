// r4dx::model::Layout / QuantLinear -- the on-disk-layout tag and one quantized linear weight's
// device buffers (docs/container-format.md "Quantized layout tensors"). Split out of container.h
// (decode-perf pass, 2026-09-19) so this type can be shared by both r4dx_model (Container/linear.h)
// and r4dx_model_attention (AttentionLayer's qg/o linears) without either depending on the other's
// full target -- see src/model/CMakeLists.txt's r4dx_model_linear target, which owns this file plus
// linear.h/.cpp and is linked by both.
#pragma once

#include <cstdint>
#include <string>

#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

enum class Layout { kBf16, kMxfp4, kW4a16, kW4a8 };

const char* LayoutName(Layout l);
Layout LayoutFromName(const std::string& name);  // throws on an unrecognized name

// One linear weight W[N,K], uploaded in exactly one of the four on-disk layouts
// (docs/container-format.md "Quantized layout tensors"). linear.cpp's ApplyLinear is the only
// thing that reads the layout-specific buffers below; Container's job stops at "the right bytes
// are on the device in the container's documented byte order".
struct QuantLinear {
  Layout layout = Layout::kBf16;
  int64_t N = 0, K = 0;  // W is [N, K]: N output features, K input features

  // layout == kBf16: W itself, row-major [N, K], bf16.
  core::DeviceBuffer<uint16_t> bf16_w;

  // layout == kW4a16 or kW4a8: nibble-packed, WMMA-fragment-permuted weight, uint8[N*K/2].
  // Byte-identical between the two layouts is NOT assumed here (src/convert's quant_int4.hpp
  // quantizes w4a16 and w4a8 separately -- see its file comment) -- each QuantLinear holds only
  // the one layout it was loaded as.
  core::DeviceBuffer<uint8_t> wq;
  // layout == kW4a16: uint32[N*K/128], low16 = f16 scale, high16 = f16(-(1024+zero)).
  core::DeviceBuffer<uint32_t> w4a16_wsz;
  // layout == kW4a8: uint32[N*K/128], low16 = f16 scale (high16 unused).
  core::DeviceBuffer<uint32_t> w4a8_ws;

  // layout == kMxfp4: OCP MXFP4 weight.
  core::DeviceBuffer<uint8_t> mxfp4_wq;    // uint8[N*K/2], fragment-permuted e2m1 pairs
  core::DeviceBuffer<uint8_t> mxfp4_ws;    // uint8[(K/32)*N], E8M0 exponent per (group, row)
  core::DeviceBuffer<int8_t> mxfp4_wref;   // int8[N], per-row reference exponent
};

}  // namespace r4dx::model
