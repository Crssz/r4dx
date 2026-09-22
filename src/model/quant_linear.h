// r4dx::model::Layout / QuantLinear -- the on-disk-layout tag and one quantized linear weight's
// device buffers (docs/container-format.md "Quantized layout tensors"). Split out of container.h
// (decode-perf pass, 2026-09-19) so this type can be shared by both r4dx_model (Container/linear.h)
// and r4dx_model_attention (AttentionLayer's qg/o linears) without either depending on the other's
// full target -- see src/model/CMakeLists.txt's r4dx_model_linear target, which owns this file plus
// linear.h/.cpp and is linked by both.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d.h"  // r4d_gemm_w4a16_nt_m64_group() -- see CheckW4a16Group below
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

enum class Layout { kBf16, kMxfp4, kW4a16, kW4a8 };

const char* LayoutName(Layout l);
Layout LayoutFromName(const std::string& name);  // throws on an unrecognized name

// The w4a16 group size (K per (scale, zero) pair) is a BUILD OPTION -- R4DX_W4A16_GROUP, which
// reaches r4d_gemm_w4a16_nt_m64 as -DR4D_GEMM_W4_GROUP and r4dx-convert as kW4A16Group (root
// CMakeLists.txt has the full story). The kernel derives the .w4a16.wsz stride from the group it
// was COMPILED with; a container carries the group it was PACKED with in
// __metadata__.quant.w4a16.group. If the two disagree every w4a16 GEMM silently reads scales for
// the wrong K range -- no crash, no NaN, just wrong numbers -- so every container loader calls
// this before touching a byte and it throws naming BOTH numbers. Callers gate it on the w4a16
// bytes actually being read: Container::Load only when a requested layout is kW4a16,
// DflashDraftWeights::Open only when the container carries a `.w4a16.*` tensor. A load that never
// touches a `.w4a16.wsz` (a `--layout mxfp4`/`w4a8`/`bf16` run, a bf16 drafter) has no stride to
// get wrong, and refusing it would reject containers this build reads correctly. `what` identifies
// the container in the message. Takes the already-extracted int rather than the JSON so this
// header stays free of nlohmann/json (r4dx_model_attention links this target too), and is inline
// rather than a quant_linear.cpp symbol so a header-only container reader
// (dflash_draft_weights.h) can call it from a target that links r4d_core but not
// r4dx_model_linear.
inline void CheckW4a16Group(int container_group, const std::string& what) {
  const int kernel_group = r4d_gemm_w4a16_nt_m64_group();
  if (container_group == kernel_group) return;
  throw std::runtime_error(
      "r4dx::model: " + what + " was packed with w4a16 group=" + std::to_string(container_group) +
      " but this build's r4d_gemm_w4a16_nt_m64 kernel reads group=" +
      std::to_string(kernel_group) +
      " -- the .w4a16.wsz scales would be read at the wrong stride, producing wrong numbers with "
      "no other symptom. Reconfigure with -DR4DX_W4A16_GROUP=" + std::to_string(container_group) +
      " in its own build directory, or re-convert the container with this build's r4dx-convert.");
}

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
  // layout == kW4a16: uint32[N*K/g], low16 = f16 scale, high16 = f16(-(1024+zero)), where g is the
  // w4a16 group this build was configured with (R4DX_W4A16_GROUP, default 64 since Milestone 11)
  // -- see CheckW4a16Group above, which is what guarantees the container agrees with the kernel.
  core::DeviceBuffer<uint32_t> w4a16_wsz;
  // layout == kW4a8: uint32[N*K/128], low16 = f16 scale (high16 unused). w4a8's group is fixed at
  // 128 by third_party/CMakeLists.txt and is NOT tied to w4a16's.
  core::DeviceBuffer<uint32_t> w4a8_ws;

  // layout == kMxfp4: OCP MXFP4 weight.
  core::DeviceBuffer<uint8_t> mxfp4_wq;    // uint8[N*K/2], fragment-permuted e2m1 pairs
  core::DeviceBuffer<uint8_t> mxfp4_ws;    // uint8[(K/32)*N], E8M0 exponent per (group, row)
  core::DeviceBuffer<int8_t> mxfp4_wref;   // int8[N], per-row reference exponent
};

}  // namespace r4dx::model
