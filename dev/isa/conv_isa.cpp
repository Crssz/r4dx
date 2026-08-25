// Standalone gfx1201 codegen harness for the DFlash2 conv body. Not part of r4d.so.
//   clang++ -O3 -std=c++17 --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -S conv_isa.cpp
#define R4D_DEV static __attribute__((always_inline))
#include "../../r4d_dflash_conv_body.h"

extern "C" __attribute__((amdgpu_kernel))
void k_dflash_conv_t2_g16(const r4d_u16* __restrict__ x,
                          const r4d_u16* __restrict__ delta,
                          const r4d_u16* __restrict__ base,
                          r4d_u16* __restrict__ out,
                          int H, int dpitch, int NG, int vpr, int blockmask) {
    const int hv = __builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x();
    const int t  = __builtin_amdgcn_workgroup_id_y();
    r4d_dflash_conv_body<2, 16>(x, delta, base, out, H, dpitch, NG, vpr, blockmask, t, hv);
}
