// r4d_gemm_trellis_nt_m64_wq.hip - R4DX_FAKEQ_W's half of r4d_gemm_trellis_nt_m64.hip (docs/int8-prefill.md in
// r4dx): the same kernel template at WQ = true, so every decoded weight fragment is rounded to symmetric
// int8 and back before it enters the WMMA, plus the scale-table builder (r4d_trellis_wscale_f32) and the
// `_wq` entry points of r4d.h. A MEASUREMENT tool: it answers "what do int8 weights cost in accuracy?" before
// an int8 x int8 GEMM exists.
//
// A unit of its own, not a few more instantiations in r4d_gemm_trellis_nt_m64.hip, because the shipped
// kernels' code must not change: with a WQ = true instantiation in the same translation unit the shipped
// ones compiled to a few different instructions (checked against the pre-change listing), without it they
// are identical. The kernel source is the shipped file's, included with R4D_TQ_WQ_UNIT defined, which
// leaves out the shipped entry points (they live in the shipped unit) and launches the rounding kernel.
// The 190-VGPR / zero-scratch rule of third_party/check_trellis_isa.cmake does not apply to this unit's
// instantiations (it reports them): the scale loads and the rounding may cost registers.
#define R4D_TQ_WQ_UNIT 1
#include "r4d_gemm_trellis_nt_m64.hip"
