// r4dx_convert::w8a16_format -- the on-disk constants of the w8a16 layout (docs/w8a16.md 2, 3).
//
// Header-only, dependency-free (no r4dx headers), so the converter, the loader (src/model) and the
// kernel tests can all include it. This is the single source of truth for every literal the
// container format spells: group sizes, the scale bias, tensor suffixes and the metadata block.
// The kernel exports the same bias (r4dx_gemm_w8a16_scale_log2_bias()) and the converter/loader
// assert the two agree.
#pragma once

namespace r4dx_convert {
namespace w8a16 {

// Default group: K per (row, scale). 64 is also instantiated by the kernel.
inline constexpr int kDefaultGroup = 128;
inline constexpr int kGroup64 = 64;
inline constexpr int kGroup128 = 128;
inline constexpr bool IsGroupSupported(int group) { return group == kGroup64 || group == kGroup128; }

// The stored scale is f16(amax/127 * 2^kScaleLog2Bias); the kernel multiplies the final
// accumulator by 2^-kScaleLog2Bias (exact). A FORMAT constant: a loader refuses any other value.
inline constexpr int kScaleLog2Bias = 12;
// Largest amax whose biased scale stays finite in f16: 65504 / 2^12 * 127 = 2031.2.
inline constexpr float kMaxAmax = 2031.0f;

// Codes: symmetric -127..127, stored offset-binary (q + 128, so 1..255); 128 is zero.
inline constexpr int kCodeMax = 127;
inline constexpr int kCodeOffset = 128;

// Kernel fragment geometry (docs/w8a16.md 3.2): 16 rows x 32 K = 512 bytes per packed block.
inline constexpr int kTileRows = 16;
inline constexpr int kKBlock = 32;

// Tensor name suffixes appended to a linear's base name.
inline constexpr const char* kWqSuffix = ".w8a16.wq";  // u8 {N*K}
inline constexpr const char* kWsSuffix = ".w8a16.ws";  // u8 {N*K/G, 2} (f16 pairs, LE)

// __metadata__.quant.w8a16 (version 1).
inline constexpr const char* kMetaQuantKey = "quant";
inline constexpr const char* kMetaBlockKey = "w8a16";
inline constexpr int kMetaVersion = 1;
inline constexpr const char* kMetaScaleFormat = "f16";
inline constexpr const char* kMetaCode = "int8_sym127_offset_binary_x80";
inline constexpr const char* kMetaFragmentPermutation = "wmma16x16x16_f16_b_lane16_k32block";

// LayoutSetId token prefix: "w8a16.g128".
inline constexpr const char* kLayoutIdPrefix = "w8a16.g";

}  // namespace w8a16
}  // namespace r4dx_convert
