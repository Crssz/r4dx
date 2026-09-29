// r4d_fwht128.h - the 128-point fast Walsh-Hadamard transform held in one wave32, fp32, shared by
// r4d_gemm_trellis_nt_m64's output transform and r4dx's trellis input transform (r4dx includes
// this header from src/kernels, so both sides run one definition).
//
// Lane l holds v[r] = x[l + 32 r], r = 0..3. On return v[r] = (H x)[l + 32 r], with H the
// UNNORMALIZED natural-order (Sylvester) Hadamard of order 128, H[i][j] = (-1)^popcount(i & j); the
// caller applies 1/sqrt(128) (and whatever else) itself.
//
// Stages lg = 0..6 in that order, each pair (i, i + 2^lg) becoming (a + b, a - b): the same fp32
// operations in the same order as r4dx's FwhtLds (src/kernels/include/r4dx/kernels/hadamard_device.h),
// so the two agree bit for bit and a row's result depends on nothing but its own 128 inputs.
//   lg = 0..3  cross-lane inside a row of 16 lanes: DPP row_xmask:2^lg brings the partner's value;
//   lg = 4     across the two rows: v_permlanex16 with the identity selects (lane l reads l ^ 16);
//   lg = 5, 6  in-register, over r (i ^ 32 is r ^ 1, i ^ 64 is r ^ 2).
// A cross-lane stage's lower lane (bit lg of l clear) holds a = x[i] and computes a + b; the upper
// lane holds b = x[i + 2^lg] and computes partner - own = a - b. Both are one fma(+-1, own,
// partner): the product by +-1 is exact, so the fma rounds once, exactly like the add or subtract,
// and IEEE addition commutes bit for bit.
// About 48 VALU plus 20 lane moves per 128 points. Every lane of the wave must call it (DPP and
// permlane read other lanes' registers).
#pragma once
#include <hip/hip_runtime.h>

// The value of `v` on lane l ^ MASK (MASK = 1, 2, 4, 8 or 16).
template <int MASK>
__device__ __forceinline__ float r4d_fwht_partner(float v) {
  const int x = __builtin_bit_cast(int, v);
  if constexpr (MASK < 16) {
    // DPP16 row_xmask:MASK (0x160 | MASK): every lane of every row reads lane (l ^ MASK) of its row.
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, x, 0x160 | MASK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(
        float, (int)__builtin_amdgcn_permlanex16((unsigned)x, (unsigned)x, 0x76543210u, 0xFEDCBA98u,
                                                 false, false));
  }
}

template <int LG>
__device__ __forceinline__ void r4d_fwht128_lane_stage(float (&v)[4], int lane) {
  const float sgn = (lane >> LG) & 1 ? -1.f : 1.f;
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = __builtin_fmaf(sgn, v[r], r4d_fwht_partner<1 << LG>(v[r]));
}

__device__ __forceinline__ void r4d_fwht128_wave(float (&v)[4], int lane) {
  r4d_fwht128_lane_stage<0>(v, lane);
  r4d_fwht128_lane_stage<1>(v, lane);
  r4d_fwht128_lane_stage<2>(v, lane);
  r4d_fwht128_lane_stage<3>(v, lane);
  r4d_fwht128_lane_stage<4>(v, lane);
  {  // lg = 5: (r, r + 1) for r = 0, 2
    const float a0 = v[0], b0 = v[1], a1 = v[2], b1 = v[3];
    v[0] = a0 + b0;
    v[1] = a0 - b0;
    v[2] = a1 + b1;
    v[3] = a1 - b1;
  }
  {  // lg = 6: (r, r + 2) for r = 0, 1
    const float a0 = v[0], b0 = v[2], a1 = v[1], b1 = v[3];
    v[0] = a0 + b0;
    v[2] = a0 - b0;
    v[1] = a1 + b1;
    v[3] = a1 - b1;
  }
}
