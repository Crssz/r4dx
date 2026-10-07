// tests/kernels/int8_gemm_proto_ref.h -- the int8 x int8 prefill GEMM's quantizer definitions and CPU
// references (docs/int8-gemm-proto.md, docs/int8-prefill.md "Production path"). Header-only; compiled by hipcc
// as part of the bench and the production kernel test (the layout helpers are __host__ __device__ there, so the
// kernels and the CPU tests run ONE definition of every index formula) and by a plain C++ compiler in
// test_int8_gemm_proto_cpu (no HIP).
//
// The layouts (A8, SA, W8, SW and the fragment position map) are third_party/libr4d/r4d_trellis_i8_layout.h's:
// the production kernel and these CPU references share ONE definition of every index formula. What is here is
// the CPU side: software f16, the quantizer definitions, the references and the whole-matrix packers.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "r4d_trellis_i8_layout.h"

namespace i8p {
// ---- f16 <-> float, software, RNE -----------------------------------------------------------------
inline float F16ToF32(uint16_t h) {
  const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  float v;
  if (e == 0) v = std::ldexp((float)m, -24);
  else if (e == 31) v = m ? std::nanf("") : INFINITY;
  else v = std::ldexp((float)(m | 0x400u), (int)e - 25);
  return s ? -v : v;
}
// one rounding, from double (so a double-rounding through float cannot happen)
inline uint16_t F64ToF16(double x) {
  if (std::isnan(x)) return 0x7E00;
  const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
  x = std::fabs(x);
  if (std::isinf(x)) return (uint16_t)(sign | 0x7C00);
  if (x == 0.0) return sign;
  int ex;
  std::frexp(x, &ex);
  int e = ex - 1;                                       // x = 1.m * 2^e
  if (e < -14) {                                        // subnormal (or rounds up to the smallest normal)
    const double r = std::nearbyint(std::ldexp(x, 24));
    return (uint16_t)(sign | (uint16_t)r);
  }
  double r = std::nearbyint(std::ldexp(x, 10 - e));     // in [1024, 2048]
  if (r >= 2048.0) { r = 1024.0; ++e; }
  if (e > 15) return (uint16_t)(sign | 0x7C00);
  return (uint16_t)(sign | (uint16_t)(((e + 15) << 10) | ((int)r - 1024)));
}
inline uint16_t F32ToF16(float f) { return F64ToF16((double)f); }

// ---- quantizers -----------------------------------------------------------------------------------------
// Weight side. rs = the f16 the decoded f16 weight is multiplied by: RsHalf(s) = f16(min(1 / s, 60000)); the
// table stores s_eff = 1 / rs (SEff), so dequantization (q * s_eff) is the inverse of exactly that grid. The
// device's quantizer takes rs from the TABLE with the hardware reciprocal (idempotent: 1 / s_eff is within a
// few f32 ulps of rs, far inside the f16 rounding interval, so it returns rs); the table builder uses the
// IEEE division below, so a CPU table is bit-identical to the GPU's.
inline uint16_t RsHalf(float s) {
  float r = 1.0f / s;
  if (r > 60000.0f) r = 60000.0f;
  return F32ToF16(r);
}
inline float SEff(float s) { return 1.0f / F16ToF32(RsHalf(s)); }
// s0 = amax / 127 of a (column, 128 k) group, 1 for an all-zero group (r4d_trellis_wscale_f32's rule)
inline float WScale0(float amax) { return amax > 0.f ? amax / 127.0f : 1.0f; }

// q = rint(w * rs): what v_pk_fma_f16(w, rs, 1536) followed by the low byte of its f16 bits computes (see
// QuantWTrick). |w * rs| <= 127.07 for every w of the group the scale was taken over, so no clamp.
inline int8_t QuantW(uint16_t wbits, uint16_t rsbits) {
  const double y = (double)F16ToF32(wbits) * (double)F16ToF32(rsbits);
  return (int8_t)(int)std::nearbyint(y);
}
// the device sequence in software: fma(w, rs, 1536) rounded once to f16 (the exact sum is below 2048 and
// above 1024, where the f16 spacing is 1, so the rounding IS the integer rounding), low byte of the bits.
inline int8_t QuantWTrick(uint16_t wbits, uint16_t rsbits) {
  const double v = (double)F16ToF32(wbits) * (double)F16ToF32(rsbits) + 1536.0;
  return (int8_t)(uint8_t)(F64ToF16(v) & 0xFFu);
}
// The fp32 rule of R4DX_FAKEQ_W (docs/int8-prefill.md), for the accuracy comparison: q = clamp(rint(w / s)).
inline int8_t QuantWFp32(uint16_t wbits, float s) {
  const float q = std::nearbyintf(F16ToF32(wbits) * (1.0f / s));
  return (int8_t)(int)(q < -127.f ? -127.f : (q > 127.f ? 127.f : q));
}
// Activation side (R4DX_FAKEQ_ACT blk128's rule): s = amax / 127 (1 for all zero), q = clamp(rint(x / s)).
inline float ActScale(float amax) { return amax > 0.f ? amax / 127.0f : 1.0f; }
inline int8_t QuantAct(float x, float s) {
  const float q = std::nearbyintf(x / s);
  return (int8_t)(int)(q < -127.f ? -127.f : (q > 127.f ? 127.f : q));
}

// ---- references -------------------------------------------------------------------------------------------
// C[row][n] = sum_kb sA(kb, row) * sW(kb, n) * (sum_{k in kb} A[row][k] * W[n][k]) in fp64, the integer sums
// exact. mask selects the coarse speed-bound variants: bit 0 = the activation scale of kb 0 stands for every kb,
// bit 1 = the weight scale of kb 0 (mask 3 is the RESC = 1 kernel's math, 1 is RESC 2's, 2 is RESC 3's; 0 is the
// per-128 math). A plain [M][K], W plain [N][K], sa [K/128][M], sw [K/128][N].
inline void RefGemmInt8(const int8_t* A, const float* sa, const int8_t* W, const float* sw, int M, int N, int K,
                        int mask, double* C) {
  const int NKB = K / 128;
  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int kb = 0; kb < NKB; ++kb) {
        long long s = 0;
        for (int k = kb * 128; k < kb * 128 + 128; ++k) s += (int)A[(size_t)r * K + k] * (int)W[(size_t)n * K + k];
        const float sa_v = (mask & 1) ? sa[r] : sa[(size_t)kb * M + r];
        const float sw_v = (mask & 2) ? sw[n] : sw[(size_t)kb * N + n];
        acc += (double)sa_v * (double)sw_v * (double)s;
      }
      C[(size_t)r * N + n] = acc;
    }
}
// the unnormalized natural-order 128-point Walsh-Hadamard transform, in fp64
inline void Fwht128(double* v) {
  for (int h = 1; h < 128; h <<= 1)
    for (int i = 0; i < 128; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const double a = v[j], b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
}

// ---- whole-matrix CPU quantizers and packers (what the device kernels compute, for the tests) --------------
// Q: the decoded f16 weights [K][N] (trellis_ref DecodePairGrid's layout). sw -> [K/128][N] (s_eff), Wp [N][K].
inline void QuantizeWeightsRef(const uint16_t* Q, int K, int N, std::vector<float>& sw, std::vector<int8_t>& Wp) {
  const int NKB = K / 128;
  sw.assign((size_t)NKB * N, 0.f);
  Wp.assign((size_t)N * K, 0);
  for (int kb = 0; kb < NKB; ++kb)
    for (int n = 0; n < N; ++n) {
      float amax = 0.f;
      for (int k = kb * 128; k < kb * 128 + 128; ++k) amax = std::fmax(amax, std::fabs(F16ToF32(Q[(size_t)k * N + n])));
      const uint16_t rs = RsHalf(WScale0(amax));
      sw[(size_t)kb * N + n] = 1.0f / F16ToF32(rs);
      for (int k = kb * 128; k < kb * 128 + 128; ++k) Wp[(size_t)n * K + k] = QuantWTrick(Q[(size_t)k * N + n], rs);
    }
}
// X: f16 activations as floats [256][K]. sa -> [K/128][256], Ap [256][K].
inline void QuantizeActRef(const float* X, int K, std::vector<float>& sa, std::vector<int8_t>& Ap) {
  const int NKB = K / 128;
  sa.assign((size_t)NKB * kM, 0.f);
  Ap.assign((size_t)kM * K, 0);
  for (int r = 0; r < kM; ++r)
    for (int kb = 0; kb < NKB; ++kb) {
      float amax = 0.f;
      for (int k = kb * 128; k < kb * 128 + 128; ++k) amax = std::fmax(amax, std::fabs(X[(size_t)r * K + k]));
      const float s = ActScale(amax);
      sa[(size_t)kb * kM + r] = s;
      for (int k = kb * 128; k < kb * 128 + 128; ++k) Ap[(size_t)r * K + k] = QuantAct(X[(size_t)r * K + k], s);
    }
}
// ---- the COARSE scales (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse scales") --------------------
// One scale per output COLUMN of Q over the whole K, by the per-128 table's own f16-rs rule (RsHalf / SEff /
// QuantWTrick on amax over all of K): sw -> [N] (s_eff), Wp [N][K]. |w rs| <= 127.07 holds for the whole column, so
// there is still no clamp.
inline void QuantizeWeightsColRef(const uint16_t* Q, int K, int N, std::vector<float>& sw, std::vector<int8_t>& Wp) {
  sw.assign((size_t)N, 0.f);
  Wp.assign((size_t)N * K, 0);
  for (int n = 0; n < N; ++n) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(F16ToF32(Q[(size_t)k * N + n])));
    const uint16_t rs = RsHalf(WScale0(amax));
    sw[(size_t)n] = 1.0f / F16ToF32(rs);
    for (int k = 0; k < K; ++k) Wp[(size_t)n * K + k] = QuantWTrick(Q[(size_t)k * N + n], rs);
  }
}
// One scale per ROW over the whole K, the per-128 quantizer's rule (s = amax / 127, 1 for an all-zero row, q =
// clamp(rint(x / s))): sa -> [256], Ap [256][K].
inline void QuantizeActRowRef(const float* X, int K, std::vector<float>& sa, std::vector<int8_t>& Ap) {
  sa.assign((size_t)kM, 0.f);
  Ap.assign((size_t)kM * K, 0);
  for (int r = 0; r < kM; ++r) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(X[(size_t)r * K + k]));
    const float s = ActScale(amax);
    sa[(size_t)r] = s;
    for (int k = 0; k < K; ++k) Ap[(size_t)r * K + k] = QuantAct(X[(size_t)r * K + k], s);
  }
}
inline std::vector<int8_t> PackA8(const std::vector<int8_t>& Ap, int K) {
  std::vector<int8_t> a8((size_t)kM * K, 0);
  for (int r = 0; r < kM; ++r)
    for (int k = 0; k < K; ++k) a8[A8Offset(r, k, K)] = Ap[(size_t)r * K + k];
  return a8;
}
inline std::vector<int8_t> PackW8(const std::vector<int8_t>& Wp, int K, int N) {
  std::vector<int8_t> w8((size_t)N * K, 0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) w8[W8Offset(n, k, K)] = Wp[(size_t)n * K + k];
  return w8;
}

// ---- software WMMA (the CPU test's) ---------------------------------------------------------------------
// D[lane][e] (+)= sum over the tile's 16 k of A[row][k] B[k][col], the operands as 32 lanes x 8 bytes of
// fragments in the layout above: D lane L element e is row 8 (L >> 4) + e, column L & 15; A's row r is held
// by lanes r and r + 16, B's column c by lanes c and c + 16, half h carrying the same 8 k in both.
inline void EmuWmmaI8(const int8_t (&a)[32][8], const int8_t (&b)[32][8], int32_t (&d)[32][8]) {
  for (int L = 0; L < 32; ++L)
    for (int e = 0; e < 8; ++e) {
      const int row = 8 * (L >> 4) + e, col = L & 15;
      int32_t s = d[L][e];
      for (int h = 0; h < 2; ++h)
        for (int p = 0; p < 8; ++p) s += (int)a[row + 16 * h][p] * (int)b[col + 16 * h][p];
      d[L][e] = s;
    }
}

}  // namespace i8p
