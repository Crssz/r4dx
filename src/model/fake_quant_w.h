// r4dx::model fake int8 WEIGHT quantization of the trellis linears in prefill (docs/int8-prefill.md):
// R4DX_FAKEQ_W and the fp32 reference of what its kernels compute.
//
// The other half of R4DX_FAKEQ_ACT (fake_quant_act.h). An int8 x int8 prefill GEMM needs the weights on an
// int8 grid, and a trellis weight is decoded from its code to an f16 value, not an int8 one. This switch
// answers "what does it cost in accuracy to put the DECODED weight on an int8 grid?" before anyone writes
// that GEMM: libr4d's trellis GEMMs (r4d_gemm_trellis_nt_m64, r4d_gemm_trellis_nt_m256, in their `_wq`
// units) round every decoded fragment to symmetric int8 and back right before the WMMA, with one scale per
// output column of Q and per group of k. The GEMM itself stays f16 WMMA with fp32 accumulation, so only the
// weight rounding is measured; with R4DX_FAKEQ_ACT on as well the numerics are the full int8 x int8 ones,
// except the accumulation (int32 in a real kernel).
//
// R4DX_FAKEQ_W (read once per process, at Model load):
//   - unset, empty or "off": nothing -- the code path, the launches, the buffers and the kernels are the ones of
//     a build without this file;
//   - "col128": one scale per (output column, 128-element k block) -- the rotation's own block, matching
//     R4DX_FAKEQ_ACT=blk128;
//   - "col32": one scale per (output column, 32-element k block), matching blk32 (a 4x larger scale table);
//   - anything else: std::invalid_argument at Model load (an experiment must not silently run unquantized).
//
// "Output column" is the column n of the regularized Q[K][N] the GEMM multiplies (W^T = diag(suh) H Q H
// diag(svh)): the values the WMMA sees, before the output-side Hadamard, svh and out_scale.
//
// Header-only and free of HIP so the parser and the reference have a CPU unit test
// (tests/model/test_fake_quant_act_cpu.cpp); the kernels are tested against the same reference by
// tests/kernels/test_fake_quant_w.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::model {

inline constexpr int kFakeQuantWOff = 0;
inline constexpr int kFakeQuantWCol128 = 1;
inline constexpr int kFakeQuantWCol32 = 2;

// false for a spelling that is not one of the above (*mode is then left alone).
inline bool ParseFakeQuantW(const char* e, int* mode) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "off") == 0) {
    *mode = kFakeQuantWOff;
  } else if (std::strcmp(e, "col128") == 0) {
    *mode = kFakeQuantWCol128;
  } else if (std::strcmp(e, "col32") == 0) {
    *mode = kFakeQuantWCol32;
  } else {
    return false;
  }
  return true;
}

inline const char* FakeQuantWName(int mode) {
  switch (mode) {
    case kFakeQuantWCol128: return "col128";
    case kFakeQuantWCol32: return "col32";
    default: return "off";
  }
}

// k elements that share one scale (0 for off).
inline int FakeQuantWGroup(int mode) {
  switch (mode) {
    case kFakeQuantWCol128: return 128;
    case kFakeQuantWCol32: return 32;
    default: return 0;
  }
}

// log2 of the k-tiles (16 k) per scale group: the `gsh` argument of libr4d's `_wq` entries (3 = 128 k, 1 = 32 k).
inline int FakeQuantWGroupShift(int mode) {
  switch (mode) {
    case kFakeQuantWCol128: return 3;
    case kFakeQuantWCol32: return 1;
    default: return 0;
  }
}

// R4DX_FAKEQ_W, parsed once per process. Throws std::invalid_argument for an unrecognized value
// (Model::Load calls it first, so the error surfaces at load).
inline int FakeQuantWRequest() {
  static const int v = [] {
    const char* e = std::getenv("R4DX_FAKEQ_W");
    int mode = kFakeQuantWOff;
    if (!ParseFakeQuantW(e, &mode)) {
      throw std::invalid_argument(std::string("R4DX_FAKEQ_W='") + e + "' not recognized (off|col128|col32)");
    }
    return mode;
  }();
  return v;
}

// ---- reference (fp32), what the `_wq` kernels compute ------------------------------------------------------
// A scale group is the `n` values (f16 widened to fp32) of ONE column of Q over one k group.
//
// The scale: s = max|w| / 127 (fp32 division), or 1.0f for an all-zero group (any scale would do, and 1 keeps
// 1 / s finite). Every decoded weight is a finite f16 (the codebook's range is a few units), so there is no
// non-finite case to pass through, unlike the activation's.
inline float FakeQuantWScaleRef(const float* w, int n) {
  float amax = 0.0f;
  for (int i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(w[i]));
  return amax > 0.0f ? amax / 127.0f : 1.0f;
}

// One value under scale s: rs = 1 / s (IEEE division, as the kernel's), q = clamp(rint(w * rs), -127, 127)
// (rint = round half to even), and the fp32 product q * s, which the kernel then rounds to f16 (the product is
// rounded to fp32 first, never fused into the conversion: libr4d is built with -ffp-contract=off).
inline float FakeQuantWRoundRef(float w, float s) {
  const float rs = 1.0f / s;
  float q = std::rint(w * rs);
  q = std::fmin(std::fmax(q, -127.0f), 127.0f);
  return q * s;
}

// The scale table of a whole Q[K][N] (row-major, f16 values widened to fp32) at `group` k per scale, laid out
// as the kernels' [K / group][N] (libr4d's r4d_trellis_wscale_f32).
inline std::vector<float> FakeQuantWTableRef(const float* Q, int64_t K, int64_t N, int group) {
  std::vector<float> t(static_cast<size_t>(K / group) * static_cast<size_t>(N));
  std::vector<float> col(static_cast<size_t>(group));
  for (int64_t g = 0; g < K / group; ++g)
    for (int64_t n = 0; n < N; ++n) {
      for (int i = 0; i < group; ++i) col[static_cast<size_t>(i)] = Q[(g * group + i) * N + n];
      t[static_cast<size_t>(g * N + n)] = FakeQuantWScaleRef(col.data(), group);
    }
  return t;
}

// The fp32 values (before the f16 rounding) the kernels feed the WMMA: Q[K][N] -> Qq[K][N] with the table above.
inline std::vector<float> FakeQuantWApplyRef(const float* Q, int64_t K, int64_t N, int group,
                                             const std::vector<float>& table) {
  std::vector<float> out(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k)
    for (int64_t n = 0; n < N; ++n)
      out[static_cast<size_t>(k * N + n)] =
          FakeQuantWRoundRef(Q[k * N + n], table[static_cast<size_t>((k / group) * N + n)]);
  return out;
}

}  // namespace r4dx::model
