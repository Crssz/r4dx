// r4d_trellis_i8_fused.h -- the activation quantizer of libr4d's int8 prefill GEMM (i8g_quant_act in
// r4d_trellis_i8.h) as a device function for a PRODUCER that holds the f16 values of one 128-block in registers: the
// trellis input transform and its fused producers (src/kernels/src/trellis_transform.hip) call it on their own f16 A
// instead of storing it for r4d_trellis_i8_quant_act to read back (R4DX_PREFILL_INT8_FUSEDQ, docs/int8-prefill.md
// "The fused quantizer"). Header only, includes just the layout: the emulation test (tests/kernels/
// test_int8_gemm_proto_emu.cpp, I8G_EMU) compiles the very same source on the CPU and holds it byte for byte to
// i8g_quant_act.
//
// WHAT IT IS.  One wave32 owns one (row, 128-block) of a 256-row A; lane l holds the block's f16 elements
// k = l + 32 r, r = 0..3 (the register layout the transform's FWHT leaves, r4d_fwht128_wave). The arithmetic is
// i8g_quant_act's, step for step, on the same f16 values: amax = max |x| over the 128 elements (exact, so the order of
// the reduction is free), s = amax / 127 (1 for an all-zero block), q = clamp(rint(x / s), -127, 127) with IEEE
// division; the byte is stored straight into the A8 fragment layout (one byte per element, i8p::A8Offset), and lane 0
// stores s into SA[kb][row]. The unfused quantizer reads 4 consecutive k per lane and stores one dword; the bytes it
// writes are the bytes written here, element for element.
#pragma once

#ifdef I8G_EMU
#include "i8g_host_emu.h"
#else
#include <hip/hip_runtime.h>
#endif

#include <cstddef>

#include "r4d_trellis_i8_layout.h"

// hb: this lane's four f16 values as bits (hb[r] = element lane + 32 r of the block). A8 / SA: the base of this A's
// A8 buffer (256 K bytes) and SA buffer (K / 128 x 256 fp32); a second part of a two-part linear is passed its own
// base by the caller. K: the whole row length (a multiple of 128). row 0..255, kb = block index k / 128.
__device__ __forceinline__ void i8g_quant_wave_block(const unsigned short (&hb)[4], int lane, int row, int kb, int K,
                                                     signed char* __restrict__ A8, float* __restrict__ SA) {
  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = (float)__builtin_bit_cast(_Float16, hb[r]);
  float amax = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(v[0]), __builtin_fabsf(v[1])),
                               __builtin_fmaxf(__builtin_fabsf(v[2]), __builtin_fabsf(v[3])));
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) amax = __builtin_fmaxf(amax, __shfl_xor(amax, m, 32));
  const float s = amax > 0.f ? amax / 127.0f : 1.0f;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float q = __builtin_rintf(v[r] / s);
    q = __builtin_fminf(__builtin_fmaxf(q, -127.f), 127.f);
    A8[i8p::A8Offset(row, kb * 128 + lane + 32 * r, K)] = (signed char)(int)q;
  }
  if (lane == 0) SA[(size_t)kb * i8p::kM + row] = s;
}

// ---- the per-ROW activation scale (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse scales") ----------------
// One workgroup of 256 threads (8 waves of 32) owns ONE row of the 256-row A: `xl` holds the row's K f16 values as bits
// (workgroup shared memory in a producer, which writes the row there block by block after its transform; global memory
// in the stand-alone quantizer i8g_quant_act_row) and every thread of the workgroup can read all of it; K a multiple of
// 128. The arithmetic is the per-128 quantizer's with the group widened to the row: amax = max |x| over the whole row
// (exact, so the reduction order is free), s = amax / 127 (1 for an all-zero row), q = clamp(rint(x / s), -127, 127) with
// IEEE division; s goes to SA[row] (one fp32 per row, a part's SA is 256 floats). The bytes are stored straight into the
// A8 fragment layout, one 8-byte segment per (k-tile, half): no two threads share a byte, no atomics.
// `red`: at least 8 floats of workgroup shared memory. Begins after the caller's barrier (xl must be complete) and ends
// with one, so xl and red can be reused right after the call.
__device__ __forceinline__ void i8g_quant_row_wg(const unsigned short* xl, int K, int row, signed char* __restrict__ A8,
                                                 float* __restrict__ SA, float* red) {
  const int tid = (int)threadIdx.x;
  float amax = 0.f;
  for (int i = tid; i < (K >> 1); i += 256) {
    const unsigned p = ((const unsigned*)xl)[i];
    const float a = (float)__builtin_bit_cast(_Float16, (unsigned short)(p & 0xFFFFu));
    const float b = (float)__builtin_bit_cast(_Float16, (unsigned short)(p >> 16));
    amax = __builtin_fmaxf(amax, __builtin_fmaxf(__builtin_fabsf(a), __builtin_fabsf(b)));
  }
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) amax = __builtin_fmaxf(amax, __shfl_xor(amax, m, 32));
  if ((tid & 31) == 0) red[tid >> 5] = amax;
  __syncthreads();
  float mx = red[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) mx = __builtin_fmaxf(mx, red[w]);
  const float s = mx > 0.f ? mx / 127.0f : 1.0f;
  if (tid == 0) SA[row] = s;
  const int KT = K >> 4;
  // segment sg = 2 kt + h holds, as bytes e = 0..7, k = 16 kt + 8 (e >> 2) + 4 h + (e & 3) (i8p::FragK16): two runs of four
  // consecutive f16 (8 bytes each) at 16 kt + 4 h and 16 kt + 8 + 4 h
  for (int sg = tid; sg < (K >> 3); sg += 256) {
    const int kt = sg >> 1, h = sg & 1;
    unsigned long long out = 0;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const unsigned long long raw = *(const unsigned long long*)(xl + kt * 16 + 8 * half + 4 * h);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float v = (float)__builtin_bit_cast(_Float16, (unsigned short)((raw >> (16 * j)) & 0xFFFFull));
        float q = __builtin_rintf(v / s);
        q = __builtin_fminf(__builtin_fmaxf(q, -127.f), 127.f);
        out |= (unsigned long long)((unsigned)(int)q & 0xFFu) << (8 * (4 * half + j));
      }
    }
    *(unsigned long long*)(A8 + i8p::A8FragOffset(row >> 6, kt, (row >> 4) & 3, (row & 15) + 16 * h, KT)) = out;
  }
  __syncthreads();
}
