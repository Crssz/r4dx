// r4d_gemm_trellis_nt_m256_wq.hip - R4DX_FAKEQ_W's half of r4d_gemm_trellis_nt_m256.hip (docs/int8-prefill.md
// in r4dx): the same kernel template at WQ = true, so the wave that decodes a weight block rounds its
// fragments to symmetric int8 and back before it publishes them to the row groups, and the `_wq` entry point
// of r4d.h. A MEASUREMENT tool.
//
// A unit of its own for the reason r4d_gemm_trellis_nt_m64_wq.hip gives (the shipped kernels' code must not
// depend on a rounding instantiation sharing their translation unit). The kernel source is the shipped
// file's, included with R4D_T256_WQ_UNIT defined, which leaves out the shipped entry points and launches the
// rounding kernel. The 190-VGPR / zero-scratch rule of third_party/check_trellis_isa.cmake does not apply to
// this unit's instantiations (it reports them).
#define R4D_T256_WQ_UNIT 1
#include "r4d_gemm_trellis_nt_m256.hip"
