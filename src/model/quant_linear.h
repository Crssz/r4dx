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

#include "r4d.h"  // r4d_gemm_w4a16_nt_m64_group()/_has_group() -- see CheckW4a16Group below
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

// The numeric values are load-bearing (PickTuning's cache key in linear.cpp, the tuning tables,
// TP part serialisation) and predate the retirement of the mxfp4 (1) and w4a8 (3) layouts, so
// they are pinned explicitly: kBf16 = 0, kW4a16 = 2, kTrellis = 4. Do not renumber.
enum class Layout { kBf16 = 0, kW4a16 = 2, kTrellis = 4 };

const char* LayoutName(Layout l);
Layout LayoutFromName(const std::string& name);  // throws on an unrecognized name

// The w4a16 default group size (K per (scale, zero) pair) is 64, a compile-time constant of
// r4d_gemm_w4a16_nt_m64 (R4D_GEMM_W4_GROUP) and of r4dx-convert (kW4A16Group; root CMakeLists.txt has
// the full story). The kernel derives the .w4a16.wsz stride from the group it was COMPILED with; a container carries the group it was PACKED with in
// __metadata__.quant.w4a16.group. If the two disagree every w4a16 GEMM silently reads scales for
// the wrong K range -- no crash, no NaN, just wrong numbers -- so every container loader calls
// this before touching a byte and it throws naming BOTH numbers. Callers gate it on the w4a16
// bytes actually being read: Container::Load only when a requested layout is kW4a16,
// DflashDraftWeights::Open only when the container carries a `.w4a16.*` tensor. A load that never
// touches a `.w4a16.wsz` (a `--layout bf16` run, a bf16 drafter) has no stride to
// get wrong, and refusing it would reject containers this build reads correctly. `what` identifies
// the container in the message. Takes the already-extracted int rather than the JSON so this
// header stays free of nlohmann/json (r4dx_model_attention links this target too), and is inline
// rather than a quant_linear.cpp symbol so a header-only container reader
// (dflash_draft_weights.h) can call it from a target that links r4d_core but not
// r4dx_model_linear.
//
// Throws W4a16GroupMismatch, a std::runtime_error, so every existing catch site is unaffected; the
// distinct type exists for callers that tolerate OTHER load failures (a test that skips a layout a
// container does not carry) but must never swallow this one.
struct W4a16GroupMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void CheckW4a16Group(int container_group, const std::string& what) {
  const int kernel_group = r4d_gemm_w4a16_nt_m64_group();
  if (container_group == kernel_group) return;
  throw W4a16GroupMismatch(
      "r4dx::model: " + what + " was packed with w4a16 group=" + std::to_string(container_group) +
      " but this build's r4d_gemm_w4a16_nt_m64 kernel reads group=" +
      std::to_string(kernel_group) +
      " -- the .w4a16.wsz scales would be read at the wrong stride, producing wrong numbers with "
      "no other symptom. Re-convert the container with this build's r4dx-convert.");
}

// quant2 Q3 (docs/quant2.md section 5.1): a linear named in __metadata__.quant.w4a16.groups is packed
// at its OWN group and runs through r4d_gemm_w4a16_nt_m64_g, so what it needs from this build is
// not "same group as the build default" (CheckW4a16Group, which still governs every unmapped
// linear) but "the kernel is instantiated at that group". Same exception type, for the same catch
// sites.
inline void CheckW4a16MappedGroup(int group, const std::string& base, const std::string& what) {
  if (r4d_gemm_w4a16_nt_m64_has_group(group) != 0) return;
  throw W4a16GroupMismatch("r4dx::model: " + what + " packs '" + base + "' at w4a16 group=" +
                           std::to_string(group) +
                           " (__metadata__.quant.w4a16.groups), which this build's "
                           "r4d_gemm_w4a16_nt_m64_g does not instantiate");
}

// Trellis (docs/trellis-kernel.md 2.5): a linear's per-linear rate (`bits`, 4 or 5) must be one
// r4d_gemm_trellis_nt_m64 instantiates, or every GEMM of it would throw mid-decode; the loader asks
// the kernel once per linear instead. Inline for the same reason as the two checks above.
inline void CheckTrellisRate(int bits, const std::string& base, const std::string& what) {
  if (r4d_gemm_trellis_nt_m64_has_rate(bits) != 0) return;
  throw std::runtime_error("r4dx::model: " + what + " packs trellis linear '" + base + "' at " +
                           std::to_string(bits) +
                           " bits per weight (__metadata__.quant.trellis.linears), which this "
                           "build's r4d_gemm_trellis_nt_m64 does not instantiate");
}

// One linear weight W[N,K], uploaded in exactly one of the three on-disk layouts
// (docs/container-format.md "Quantized layout tensors"). linear.cpp's ApplyLinear is the only
// thing that reads the layout-specific buffers below; Container's job stops at "the right bytes
// are on the device in the container's documented byte order".
struct QuantLinear {
  Layout layout = Layout::kBf16;
  int64_t N = 0, K = 0;  // W is [N, K]: N output features, K input features

  // layout == kBf16: W itself, row-major [N, K], bf16.
  core::DeviceBuffer<uint16_t> bf16_w;

  // layout == kW4a16: nibble-packed, WMMA-fragment-permuted weight, uint8[N*K/2].
  core::DeviceBuffer<uint8_t> wq;
  // layout == kW4a16: uint32[N*K/g], low16 = f16 scale, high16 = f16(-(1024+zero)), where g is
  // w4a16_group below -- by default the group this build's kernel is compiled for (64); see
  // CheckW4a16Group above, which is what guarantees the container agrees with the kernel.
  core::DeviceBuffer<uint32_t> w4a16_wsz;
  // layout == kW4a16: this linear's own group (docs/quant2.md section 5.1), or 0 = "this build's
  // default", r4d_gemm_w4a16_nt_m64_group(). 0 is what every QuantLinear built before per-tensor
  // groups -- and every one not named in __metadata__.quant.w4a16.groups -- carries, and it keeps
  // ApplyLinear on the historical r4d_gemm_w4a16_nt_m64 entry. A non-zero group other than the
  // default dispatches r4d_gemm_w4a16_nt_m64_g (linear.cpp).
  int w4a16_group = 0;
  // layout == kTrellis (docs/trellis-kernel.md 2.1, 5.1): W^T = diag(suh) H Q H diag(svh), Q the
  // decoded trellis tiles, H the natural-order 128-point Hadamard / sqrt(128). Every size here is
  // the RANK's under tensor parallelism (N, K and part N are the rank-local shape).
  //   trellis_w: the uint32 ring words in the pair grid, N*K*bits/32 of them.
  //   trellis_suh: fp32 [parts][K], the input-side scale (signs folded in), widened from fp16.
  //   trellis_svh: fp32 [N], the output-side scale, in container row order (gate then up).
  //   trellis_tickets: u32 [N/128], this linear's slice of its Container's one ticket buffer (the
  //     split 128-group finish, docs/trellis-kernel.md 4.5); Container::ZeroTrellisTickets clears
  //     it.
  //   trellis_part_n: output columns of each part (mlp.gate_up: gate's then up's; one part
  //     otherwise, trellis_part_n[0] == N). Part 1 reads its own input transform (its own suh).
  //   trellis_prescale_log2: the power-of-two shift s of the input side (A = f16(H(x*suh) *
  //     2^s/sqrt(128))), undone by the GEMM's out_scale; from __metadata__.quant.trellis.
  core::DeviceBuffer<uint32_t> trellis_w;
  core::DeviceBuffer<float> trellis_suh;
  core::DeviceBuffer<float> trellis_svh;
  uint32_t* trellis_tickets = nullptr;
  int trellis_bits = 0;
  int trellis_parts = 0;
  int64_t trellis_part_n[2] = {0, 0};
  int trellis_prescale_log2 = 0;
  //   trellis_wscale: R4DX_FAKEQ_W (fake_quant_w.h, docs/int8-prefill.md), empty unless the switch is on. fp32
  //     [K / 16 >> trellis_wscale_gsh][N], the per-(output column, k group) scale of the decoded weight
  //     (libr4d's r4d_trellis_wscale_f32), built once at Model::Load; the prefill GEMMs read it to round
  //     the decoded weights to int8. A measurement hook: nothing else touches it.
  core::DeviceBuffer<float> trellis_wscale;
  int trellis_wscale_gsh = 0;
  //   trellis_i8_sw: R4DX_PREFILL_INT8 (prefill_int8.h, docs/int8-prefill.md "Production path"), empty unless the
  //     switch is on and this linear has an int8 plan (PlanTrellisI8). fp32 [K / 128][N], the weight scale of
  //     (128 k, output column n of Q) the int8 prefill GEMM quantizes the decoded weights with (libr4d's
  //     r4d_trellis_i8_wscale), K * N / 32 bytes, built once at Model::Load from this rank's own words. Separate
  //     from trellis_wscale (R4DX_FAKEQ_W's fp32 s = amax / 127 table): this one holds s_eff = 1 / f16(1 / s).
  core::DeviceBuffer<float> trellis_i8_sw;
  //   trellis_i8_swc: R4DX_PREFILL_INT8_SCALES=coarse (docs/int8-prefill.md "Coarse scales"), instead of trellis_i8_sw (a
  //     linear carries one of the two, never both: the per-128 table is not even allocated in coarse mode). fp32 [N], the
  //     weight scale of output column n of Q over the WHOLE K (libr4d's r4d_trellis_i8_wscale_col), 4 N bytes. Its being
  //     non-empty is what makes the linear's int8 calls coarse (A per row too): TrellisI8Coarse(w) in linear.h.
  core::DeviceBuffer<float> trellis_i8_swc;
};

}  // namespace r4dx::model
