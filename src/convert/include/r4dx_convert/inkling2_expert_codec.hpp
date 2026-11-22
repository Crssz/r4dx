// r4dx_convert::inkling2 -- decoding the pre-quantized expert tensors of an inkling2_loop checkpoint to f32, one
// expert slice at a time (docs/inkling2-plan.md 3.2, 4).
//
// The semantics are those of transformers' `Inkling2LoopDequantizeExperts` (core_model_loading.py, class at line 772
// of the copy under the session scratchpad), reproduced op for op IN FLOAT32 (HF rounds to the param dtype at the
// end; the converter keeps f32 for the re-encode):
//   NVFP4 (the group has `*_weight_amax`):  weight U8 [E, rows, cols/2], two E2M1 codes per byte, the EVEN element
//     (2j) in the LOW nibble and 2j+1 in the HIGH nibble (`byte_to_values = stack(e2m1[b & 15], e2m1[b >> 4])`);
//     scale F8_E4M3 [E, rows, cols/16]; value = (e2m1 * f32(scale)) * (amax[e] / outer_scale_denominator).
//     The two products are separate f32 roundings, in that order (`values * block_scale`, then `* expert_factor`).
//   MXFP8 (no amax):  weight F8_E4M3 [E, rows, cols], scale U8 [E, rows, cols/32] (E8M0);
//     value = f32(e4m3) * 2^(scale - 127)   (`torch.exp2(scale - 127)`, an exact power of two in f32).
//   w13 ONLY: the HF mapping chains `Interleave(dim=1)` after the dequantization
//     (conversion_mapping.py line 2666; core_model_loading.py `class Interleave`, lines 232-251, forward direction:
//     `reshape(rows/2, 2, ...)`, `transpose`, `reshape`). That sends source row 2r to row r and source row 2r + 1
//     to row rows/2 + r, i.e. the checkpoint's w13 rows are INTERLEAVED gate, up, gate, up, ... and the converter
//     de-interleaves to [gate rows; up rows]. The experts forward then does `gate, up = gate_up.chunk(2, dim=-1)`
//     (modeling_inkling2_loop.py `_apply_gate`, line 809), so the first half is gate. w2 is not interleaved.
//   F8_E4M3 values: mxfp8::DecodeE4m3 (bias 7, subnormals m * 2^-9); a NaN code (0x7F / 0xFF) anywhere is refused,
//   as is E8M0 scale byte 255.
//
// Pure functions over raw pointers: no ShardedModel, no I/O, so tests drive them with hand-built bytes. The
// checkpoint-facing view is built in inkling2_layout.cpp (the only place that accepts these source dtypes;
// tensor_codec.hpp keeps rejecting everything but BF16 / F32).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/mxfp8_format.hpp"

namespace r4dx_convert {
namespace inkling2 {

enum class ExpertFormat { kNvfp4, kMxfp8 };

inline constexpr int kNvfp4Group = 16;  // columns per F8_E4M3 block scale
inline constexpr int kMxfp8Group = mxfp8::kGroup;

// E2M1 values by 4-bit code (sign in bit 3). Exactly Inkling2LoopDequantizeExperts.E2M1_VALUES.
inline constexpr float kE2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// One (layer, w13 | w2) group of expert tensors, as raw views. `weight` / `scale` point at the whole tensors
// (all experts); `amax` is the [E] f32 vector (NVFP4 only; little-endian bytes, possibly unaligned).
struct ExpertTensors {
  ExpertFormat format = ExpertFormat::kMxfp8;
  int64_t experts = 0, rows = 0, cols = 0;  // cols = logical K after decoding
  const uint8_t* weight = nullptr;          // NVFP4: [E, rows, cols/2]; MXFP8: [E, rows, cols]
  const uint8_t* scale = nullptr;           // NVFP4: [E, rows, cols/16] e4m3; MXFP8: [E, rows, cols/32] e8m0
  const uint8_t* amax = nullptr;            // NVFP4: E f32
  float denominator = 1.0f;                 // NVFP4: outer_scale_denominator

  int64_t WeightBytesPerExpert() const { return format == ExpertFormat::kNvfp4 ? rows * (cols / 2) : rows * cols; }
  int64_t ScaleBytesPerExpert() const { return rows * (cols / (format == ExpertFormat::kNvfp4 ? kNvfp4Group : kMxfp8Group)); }
  int64_t GroupsPerRow() const { return cols / (format == ExpertFormat::kNvfp4 ? kNvfp4Group : kMxfp8Group); }
};

inline float ReadF32(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, 4);
  return v;
}

// f32(amax[e]) / f32(denominator): HF's `expert_factor = amax.float() / denominator.float()`.
inline float ExpertFactor(const ExpertTensors& t, int64_t e) { return ReadF32(t.amax + 4 * e) / t.denominator; }

namespace detail {
[[noreturn]] inline void Bad(const std::string& what) { throw std::runtime_error("inkling2 expert codec: " + what); }
}  // namespace detail

// Decodes expert `e` to f32 [rows, cols] row-major, in the CHECKPOINT row order (w13: still interleaved).
inline void DecodeExpert(const ExpertTensors& t, int64_t e, float* out) {
  if (e < 0 || e >= t.experts) detail::Bad("expert index out of range");
  const uint8_t* w = t.weight + e * t.WeightBytesPerExpert();
  const uint8_t* s = t.scale + e * t.ScaleBytesPerExpert();
  const int64_t gpr = t.GroupsPerRow();
  if (t.format == ExpertFormat::kNvfp4) {
    const float factor = ExpertFactor(t, e);
    for (int64_t r = 0; r < t.rows; ++r)
      for (int64_t g = 0; g < gpr; ++g) {
        const uint8_t sc = s[r * gpr + g];
        if ((sc & 0x7F) == 0x7F) detail::Bad("NaN F8_E4M3 block scale");
        const float block_scale = mxfp8::DecodeE4m3(sc);
        const uint8_t* wb = w + (r * gpr + g) * (kNvfp4Group / 2);
        float* o = out + r * t.cols + g * kNvfp4Group;
        for (int j = 0; j < kNvfp4Group / 2; ++j) {
          const uint8_t b = wb[j];
          float lo = kE2m1[b & 0xF] * block_scale;  // `values * block_scale`
          float hi = kE2m1[b >> 4] * block_scale;
          o[2 * j] = lo * factor;                   // `values * expert_factor`
          o[2 * j + 1] = hi * factor;
        }
      }
  } else {
    for (int64_t r = 0; r < t.rows; ++r)
      for (int64_t g = 0; g < gpr; ++g) {
        const uint8_t sc = s[r * gpr + g];
        if (sc == 255) detail::Bad("E8M0 scale byte 255 (NaN)");
        const float scale = mxfp8::ScaleValue(sc);
        const uint8_t* wb = w + (r * gpr + g) * kMxfp8Group;
        float* o = out + r * t.cols + g * kMxfp8Group;
        for (int j = 0; j < kMxfp8Group; ++j) {
          if ((wb[j] & 0x7F) == 0x7F) detail::Bad("NaN e4m3 code");
          o[j] = mxfp8::DecodeE4m3(wb[j]) * scale;
        }
      }
  }
}

// MXFP8 sources only: expert `e`'s raw codes [rows * cols] and E8M0 bytes [rows * cols/32], unchanged, in the
// checkpoint row order. NaN codes / scale 255 are refused here too.
inline void RawMxfp8Expert(const ExpertTensors& t, int64_t e, uint8_t* codes, uint8_t* scales) {
  if (t.format != ExpertFormat::kMxfp8) detail::Bad("RawMxfp8Expert on an NVFP4 group");
  if (e < 0 || e >= t.experts) detail::Bad("expert index out of range");
  const uint8_t* w = t.weight + e * t.WeightBytesPerExpert();
  const uint8_t* s = t.scale + e * t.ScaleBytesPerExpert();
  const int64_t nw = t.WeightBytesPerExpert(), ns = t.ScaleBytesPerExpert();
  for (int64_t i = 0; i < nw; ++i)
    if ((w[i] & 0x7F) == 0x7F) detail::Bad("NaN e4m3 code");
  for (int64_t i = 0; i < ns; ++i)
    if (s[i] == 255) detail::Bad("E8M0 scale byte 255 (NaN)");
  std::memcpy(codes, w, static_cast<size_t>(nw));
  std::memcpy(scales, s, static_cast<size_t>(ns));
}

// The Interleave(dim=1) of a w13 slice, as a row permutation: `src` [rows, rowlen] with interleaved gate / up rows
// -> `dst` [rows, rowlen] = [gate rows (src rows 0, 2, 4, ...); up rows (src rows 1, 3, 5, ...)]. Element type T
// is float (decoded values), or uint8_t (raw codes with rowlen = cols, raw scale bytes with rowlen = cols/32).
template <typename T>
inline void DeinterleaveGateUp(const T* src, T* dst, int64_t rows, int64_t rowlen) {
  if (rows % 2 != 0) detail::Bad("w13 row count is odd");
  const int64_t half = rows / 2;
  for (int64_t r = 0; r < half; ++r) {
    std::memcpy(dst + r * rowlen, src + (2 * r) * rowlen, static_cast<size_t>(rowlen) * sizeof(T));
    std::memcpy(dst + (half + r) * rowlen, src + (2 * r + 1) * rowlen, static_cast<size_t>(rowlen) * sizeof(T));
  }
}

}  // namespace inkling2
}  // namespace r4dx_convert
