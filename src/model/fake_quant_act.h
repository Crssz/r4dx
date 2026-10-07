// r4dx::model fake int8 activation quantization of the prefill (docs/int8-prefill.md): R4DX_FAKEQ_ACT and the
// fp32 reference of what its kernel computes.
//
// A research switch that answers "what would int8 activations cost in prefill?" before anyone writes an
// int8 GEMM. When it is on, the f16 activation tile a TRELLIS linear's GEMM consumes -- the operand AFTER
// the 128-block Hadamard rotation (r4dx_trellis_input_bf16 and its fused producers), exactly what an int8
// kernel would quantize -- is rounded to symmetric int8 and back, in place, right before the GEMM
// (ApplyLinear): v = rint(x / s) * s with s = max|x| / 127 over the scale group, in fp32, written back as
// f16. The GEMM itself is unchanged (f16 WMMA), so this measures the activation rounding only.
//
// R4DX_FAKEQ_ACT (read once per process, at Model load):
//   - unset, empty or "off": nothing -- the code path, the launches and the buffers are the ones of a build
//     without this file;
//   - "row": one scale per row of the operand (the old w4a8's granularity, but on the rotated operand);
//   - "blk128": one scale per row x 128-column block (the rotation's own block; the paper's estimate);
//   - "blk32": one scale per row x 32-column block (what an int8 WMMA tile k-slice could carry cheaply);
//   - anything else: std::invalid_argument at Model load (an experiment must not silently run unquantized).
//
// Header-only and free of HIP so the parser and the reference have a CPU unit test
// (tests/model/test_fake_quant_act_cpu.cpp); the kernel (src/kernels/src/trellis_transform.hip,
// r4dx_fake_quant_act_f16) is tested against the same reference by tests/kernels/test_fake_quant_act.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace r4dx::model {

// The values are the `mode` argument of r4dx_fake_quant_act_f16 (kernels.h); 0 means off.
inline constexpr int kFakeQuantOff = 0;
inline constexpr int kFakeQuantRow = 1;
inline constexpr int kFakeQuantBlk128 = 2;
inline constexpr int kFakeQuantBlk32 = 3;

// false for a spelling that is not one of the above (*mode is then left alone).
inline bool ParseFakeQuantAct(const char* e, int* mode) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "off") == 0) {
    *mode = kFakeQuantOff;
  } else if (std::strcmp(e, "row") == 0) {
    *mode = kFakeQuantRow;
  } else if (std::strcmp(e, "blk128") == 0) {
    *mode = kFakeQuantBlk128;
  } else if (std::strcmp(e, "blk32") == 0) {
    *mode = kFakeQuantBlk32;
  } else {
    return false;
  }
  return true;
}

inline const char* FakeQuantActName(int mode) {
  switch (mode) {
    case kFakeQuantRow: return "row";
    case kFakeQuantBlk128: return "blk128";
    case kFakeQuantBlk32: return "blk32";
    default: return "off";
  }
}

// Columns that share one scale; 0 = the whole row.
inline int FakeQuantActGroup(int mode) {
  switch (mode) {
    case kFakeQuantBlk128: return 128;
    case kFakeQuantBlk32: return 32;
    default: return 0;
  }
}

// R4DX_FAKEQ_ACT, parsed once per process. Throws std::invalid_argument for an unrecognized value
// (Model::Load calls it first, so the error surfaces at load).
inline int FakeQuantActRequest() {
  static const int v = [] {
    const char* e = std::getenv("R4DX_FAKEQ_ACT");
    int mode = kFakeQuantOff;
    if (!ParseFakeQuantAct(e, &mode)) {
      throw std::invalid_argument(std::string("R4DX_FAKEQ_ACT='") + e + "' not recognized (off|row|blk128|blk32)");
    }
    return mode;
  }();
  return v;
}

// ---- reference (fp32), what r4dx_fake_quant_act_f16 computes per scale group ------------------------------
// `x` are the group's f16 values widened to fp32. Writes the fp32 value the kernel rounds to f16 (q * s, the
// product rounded to fp32 first, never fused into the f16 conversion). A group whose max |x| is 0 gives
// zeros; one with a non-finite element is left as it is (the kernel's pass-through; an f16 operand only
// overflows to inf by a bug upstream, and a diagnostic must not hide it). rint is round-half-even.
// *scale_out (if non-null) receives s (0 for the zero group).
inline void FakeQuantGroupRef(const float* x, int n, float* v_out, float* scale_out = nullptr) {
  float amax = 0.0f;
  bool finite = true;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(x[i])) finite = false;
    else amax = std::fmax(amax, std::fabs(x[i]));
  }
  if (scale_out != nullptr) *scale_out = amax / 127.0f;
  if (!finite) {
    for (int i = 0; i < n; ++i) v_out[i] = x[i];
    return;
  }
  if (amax == 0.0f) {
    for (int i = 0; i < n; ++i) v_out[i] = 0.0f;
    return;
  }
  const float s = amax / 127.0f;
  for (int i = 0; i < n; ++i) {
    float q = std::rint(x[i] / s);
    q = std::fmin(std::fmax(q, -127.0f), 127.0f);
    v_out[i] = q * s;
  }
}

// One row of K columns under `mode` (kFakeQuantRow: one group; blk128 / blk32: K / group groups).
inline void FakeQuantRowRef(const float* x, int K, int mode, float* v_out) {
  const int g = FakeQuantActGroup(mode);
  if (mode == kFakeQuantOff) {
    for (int i = 0; i < K; ++i) v_out[i] = x[i];
    return;
  }
  if (g == 0) {
    FakeQuantGroupRef(x, K, v_out);
    return;
  }
  for (int c = 0; c < K; c += g) FakeQuantGroupRef(x + c, g, v_out + c);
}

}  // namespace r4dx::model
