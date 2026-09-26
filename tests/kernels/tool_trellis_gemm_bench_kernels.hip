// tests/kernels/tool_trellis_gemm_bench_kernels.hip -- the device side of tool_trellis_gemm_bench
// (docs/trellis-kernel.md 6 "Benches", milestone M1): weight fills, a read-only cache flush, a clock
// probe, op-rate and dependency-distance loops and the WMMA / VALU / memory pipe loops. Built by
// hipcc into its own object (tests/kernels/CMakeLists.txt), never linked into anything but the
// bench.
//
// Every kernel here is bounded: fixed trip counts, no spin on memory, so a bug cannot hang the GPU.
//
// CLOCKS. clock64() is s_getreg SHADER_CYCLES (the shader clock, per wave) and wall_clock64() is
// s_sendmsg_rtn GET_REALTIME (a constant ~100 MHz), so d(clock64) / d(wall_clock64) * wall rate is
// the shader clock over that interval. The clock probe's loop is a dependent v_add chain, a fixed
// number of shader cycles per add whatever the clock, so its cycles per add -- which the bench
// records for every probe -- is constant if and only if clock64 counts real shader cycles.
// HW_REG_HW_ID1 says where a wave ran: WAVE_ID [4:0], SIMD_ID [9:8], WGP_ID [13:10], SA_ID [16],
// SE_ID [20:18] (gfx11/gfx12 layout); the host decodes it to check how many waves really shared a
// SIMD.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"

namespace {

__device__ __forceinline__ uint32_t Hash32(uint64_t i, uint32_t seed) {
  uint32_t h = static_cast<uint32_t>(i) * 0x9E3779B9u ^ static_cast<uint32_t>(i >> 32) * 0x85EBCA6Bu ^ seed;
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return h;
}

__global__ void FillU32Kernel(uint32_t* p, int64_t n, uint32_t seed, uint32_t and_mask, uint32_t or_mask) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = (Hash32(static_cast<uint64_t>(i), seed) & and_mask) | or_mask;
}

// Reads n 16-byte vectors and keeps them alive through a condition that never holds: a flush that leaves the
// cache full of clean lines (a write flush leaves dirty ones, whose write-back the next timed
// chain would pay). Plain loads: a non-temporal read does not allocate in the last-level cache, so
// it evicts nothing (measured: 8-layer attn.o chains still ran at 1 TB/s after one).
__global__ void ReadKernel(const r4d_u32x4* p, int64_t n, uint32_t* sink) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  uint32_t s = 0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const r4d_u32x4 v = p[i];
    s ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (s == 0x9E3779B9u) sink[0] = s;
}

__device__ __forceinline__ uint32_t HwId1() {
  uint32_t v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(v));
  return v;
}

// out[0] = wall_clock64 when this one-thread kernel runs: the bench's timer. Kernels of one stream
// run in order, so two stamps bracket everything launched between them.
__global__ void StampKernel(unsigned long long* out) { out[0] = wall_clock64(); }

// out[0..3] = clock64, wall_clock64 before and after a dependent chain of 8 * iters v_add.
__global__ void ClockProbeKernel(unsigned long long* out, int iters) {
  const unsigned long long c0 = clock64(), w0 = wall_clock64();
  uint32_t x = threadIdx.x;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) asm volatile("v_add_nc_u32 %0, %0, 1" : "+v"(x));
  }
  const unsigned long long c1 = clock64(), w1 = wall_clock64();
  if (threadIdx.x == 0) {
    out[0] = c0;
    out[1] = w0;
    out[2] = c1;
    out[3] = w1;
    out[4] = x;
  }
}

// ---- op rates -----------------------------------------------------------------------------------
// D dependency chains per lane, advanced round-robin, so every instruction reads the result of the
// one D instructions before it: D = 8 measures issue rate (the original op-rate loop), D = 1..4 the
// cost of a near dependency the way the first decode had them -- behind inline asm, so no
// s_delay_alu -- and DELAY = true the same with the s_delay_alu instid0(VALU_DEP_D) the compiler
// would put in front of an instruction it generated itself. 24 ops per iteration per lane.
// The forms: the first nine as before (register operands), then the forms the decode compiles to.
enum Op {
  kMadU32U16, kPkMadU16, kSadU8, kSadHiU8, kMulLoU32, kAlignbit, kPkFmaF16, kAddU32, kFmaF32,
  kMadU32U16Hi,   // v_mad_u32_u16 d, x, s, 0 op_sel:[1,0,0,0]       (the decode's F0 half)
  kPkMadU16Hi,    // v_pk_mad_u16 d, s, x, x op_sel_hi:[1,1,1]
  kPkMadU16Lo,    // v_pk_mad_u16 d, s, x, x op_sel_hi:[1,0,1]
  kSadU8Lit,      // v_sad_u8 d, x, 0, 0x64006400
  kPkFmaF16Lit,   // v_pk_fma_f16 d, x, s, 0xc931 op_sel_hi:[1,0,0]
  kOpCount
};

template <int OP>
__device__ __forceinline__ void Step(uint32_t& x, uint32_t y) {
  const uint32_t k = 0x3C003C01u;
  if constexpr (OP == kMadU32U16) asm volatile("v_mad_u32_u16 %0, %0, %1, %0" : "+v"(x) : "s"(k));
  else if constexpr (OP == kPkMadU16) asm volatile("v_pk_mad_u16 %0, %1, %0, %0" : "+v"(x) : "s"(k));
  else if constexpr (OP == kSadU8) asm volatile("v_sad_u8 %0, %0, 0, %0" : "+v"(x));
  else if constexpr (OP == kSadHiU8) asm volatile("v_sad_hi_u8 %0, %0, 0, %0" : "+v"(x));
  else if constexpr (OP == kMulLoU32) asm volatile("v_mul_lo_u32 %0, %0, %1" : "+v"(x) : "s"(k));
  else if constexpr (OP == kAlignbit) asm volatile("v_alignbit_b32 %0, %0, %1, 4" : "+v"(x) : "v"(y));
  else if constexpr (OP == kPkFmaF16) asm volatile("v_pk_fma_f16 %0, %0, %1, %0" : "+v"(x) : "s"(k));
  else if constexpr (OP == kAddU32) asm volatile("v_add_nc_u32 %0, %0, %1" : "+v"(x) : "s"(k));
  else if constexpr (OP == kFmaF32) asm volatile("v_fma_f32 %0, %0, %1, %0" : "+v"(x) : "s"(k));
  else if constexpr (OP == kMadU32U16Hi)
    asm volatile("v_mad_u32_u16 %0, %0, %1, 0 op_sel:[1,0,0,0]" : "+v"(x) : "s"(R4D_TRELLIS_MUL1_KMLO));
  else if constexpr (OP == kPkMadU16Hi)
    asm volatile("v_pk_mad_u16 %0, %1, %0, %0 op_sel_hi:[1,1,1]" : "+v"(x) : "s"(R4D_TRELLIS_MUL1_KMHI));
  else if constexpr (OP == kPkMadU16Lo)
    asm volatile("v_pk_mad_u16 %0, %1, %0, %0 op_sel_hi:[1,0,1]" : "+v"(x) : "s"(R4D_TRELLIS_MUL1_KMHI));
  else if constexpr (OP == kSadU8Lit) asm volatile("v_sad_u8 %0, %0, 0, 0x64006400" : "+v"(x));
  else
    asm volatile("v_pk_fma_f16 %0, %0, %1, 0xc931 op_sel_hi:[1,0,0]" : "+v"(x) : "s"(R4D_TRELLIS_KINV_F16X2));
}

template <int D>
__device__ __forceinline__ void DelayAlu() {
  if constexpr (D == 1) asm volatile("s_delay_alu instid0(VALU_DEP_1)");
  else if constexpr (D == 2) asm volatile("s_delay_alu instid0(VALU_DEP_2)");
  else if constexpr (D == 3) asm volatile("s_delay_alu instid0(VALU_DEP_3)");
  else if constexpr (D == 4) asm volatile("s_delay_alu instid0(VALU_DEP_4)");
}

template <int OP, int D, bool DELAY>
__global__ void OpRateKernel(unsigned long long* cyc, uint32_t* hwid, uint32_t* sink, int iters) {
  static_assert(24 % D == 0, "24 ops per iteration");
  uint32_t x[D];
#pragma unroll
  for (int c = 0; c < D; ++c) x[c] = threadIdx.x * 8 + c;
  const uint32_t y = threadIdx.x ^ 0x5A5A5A5Au;
  __syncthreads();
  const unsigned long long c0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 24 / D; ++u)
#pragma unroll
      for (int c = 0; c < D; ++c) {
        if constexpr (DELAY) DelayAlu<D>();
        Step<OP>(x[c], y);
      }
  }
  const unsigned long long c1 = clock64();
  const int wave = threadIdx.x / 32 + blockIdx.x * (blockDim.x / 32);
  if ((threadIdx.x & 31) == 0) {
    cyc[wave] = c1 - c0;
    hwid[wave] = HwId1();
  }
  uint32_t s = 0;
#pragma unroll
  for (int c = 0; c < D; ++c) s ^= x[c];
  if (s == 0x12345678u) sink[0] = s;          // keeps the chains alive; never true in practice
}

// ---- pipes --------------------------------------------------------------------------------------
// The GEMM's K loop taken apart (NP = 1, U = 2, MT = 1, one tile pair per wave): per step a lane
// loads its two (A, B) b64 and P b32 of two k-tiles and the two activation fragments (two b64 each,
// from a small L2-resident A, as the GEMM's M = 1 row does), decodes the words
// (r4d_trellis_k4_decode, the GEMM's own device code, so every mode has the GEMM's schedule of it)
// and feeds 4 WMMA. MODE selects which of the three parts run:
enum PipeMode {
  kPipeMem,        // weight and activation loads only (sunk)
  kPipeValu,       // decode only (inputs made opaque each step, outputs sunk)
  kPipeWmma,       // WMMA only, on fixed fragments
  kPipeAlu,        // decode feeding WMMA, no loads: the kernel's compute
  kPipeAluIndep,   // decode and WMMA on independent data in the same wave
  kPipeAll,        // loads + decode + WMMA: the kernel's loop, one step ahead like the GEMM
  kPipeCount
};

__device__ __forceinline__ void SinkV(uint32_t v) { asm volatile("" ::"v"(v)); }
__device__ __forceinline__ void SinkH(v8h v) { asm volatile("" ::"v"(v)); }
__device__ __forceinline__ void Opaque(uint32_t& v) { asm volatile("" : "+v"(v)); }

typedef __attribute__((address_space(1))) const unsigned char* gbyte;

template <typename T, bool NT>
__device__ __forceinline__ T Ld(gbyte base, unsigned off, int imm) {
  auto p = (__attribute__((address_space(1))) const T*)(base + off + imm);
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// One step's activation fragments: k-tiles 2i and 2i + 1 of an M = 1 row (every lane of a half
// reads the same row, as the GEMM's clamped rows do), from a 64 KiB window of `a`.
struct PipeA {
  unsigned long long w[2][2];
};
__device__ __forceinline__ PipeA LoadA(gbyte a, unsigned aoff, int step) {
  const gbyte b = a + static_cast<unsigned>((step & 1023) * 64);
  PipeA f;
  f.w[0][0] = Ld<unsigned long long, false>(b, aoff, 0);
  f.w[0][1] = Ld<unsigned long long, false>(b, aoff, 16);
  f.w[1][0] = Ld<unsigned long long, false>(b, aoff, 32);
  f.w[1][1] = Ld<unsigned long long, false>(b, aoff, 48);
  return f;
}
__device__ __forceinline__ v8h Frag(const unsigned long long (&w)[2]) {
  union { unsigned long long w[2]; v8h h; } u;
  u.w[0] = w[0];
  u.w[1] = w[1];
  return u.h;
}

// One step's operands: the lane's words of two k-tiles and their activation fragments.
struct PipeStage {
  uint32_t a[2], b[2], p[2];   // the (A, B) b64 and P b32 of k-tiles 0 and 1
  PipeA act;
};

template <int MODE>
__global__ __launch_bounds__(1024) void PipeKernel(const unsigned char* __restrict__ w,
                                                  const unsigned char* __restrict__ a,
                                                  int64_t bytes_per_wave, int iters,
                                                  unsigned long long* cyc, uint32_t* hwid,
                                                  float* sink, unsigned long long* clk) {
  constexpr bool kMem = MODE == kPipeMem || MODE == kPipeAll;
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32 + blockIdx.x * (blockDim.x / 32)));
  const bool probe = blockIdx.x == 0 && threadIdx.x == 0;
  if (probe) {
    clk[0] = clock64();
    clk[1] = wall_clock64();
  }
  unsigned off_ab = r4d_trellis_k4_off_ab(lane), off_p = r4d_trellis_k4_off_p(lane);
  unsigned aoff = 8u * static_cast<unsigned>(lane >> 4);
  const gbyte base = (gbyte)w + static_cast<int64_t>(wave) * bytes_per_wave;
  const gbyte abase = (gbyte)a;
  v8f acc0 = (v8f)(0.f), acc1 = (v8f)(0.f);
  union { uint32_t u[4]; v8h h; } af;
#pragma unroll
  for (int i = 0; i < 4; ++i) af.u[i] = 0x3C003C00u ^ (lane * 0x00010001u);
  v8h fixed0, fixed1;
  r4d_trellis_k4_decode(lane * 0x9E3779B9u, lane * 0x85EBCA6Bu, lane * 0xC2B2AE35u, fixed0, fixed1);

  // Stage s holds step s's operands: loaded one step ahead in the memory modes, as the GEMM's two
  // named stages are (no copies at the loop edge); synthetic and made opaque in the others.
  auto load = [&](PipeStage& s, int step) {
    const gbyte b = base + static_cast<int64_t>(step) * 512;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const unsigned long long ab = Ld<unsigned long long, true>(b, off_ab, 256 * u);
      s.a[u] = static_cast<uint32_t>(ab);
      s.b[u] = static_cast<uint32_t>(ab >> 32);
      s.p[u] = Ld<uint32_t, true>(b, off_p, 256 * u);
    }
    s.act = LoadA(abase, aoff, step);
  };
  auto work = [&](PipeStage& s) {
    if constexpr (!kMem) {
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        Opaque(s.a[u]); Opaque(s.b[u]); Opaque(s.p[u]);
      }
    }
    const uint32_t A0 = s.a[0], B0 = s.b[0], A1 = s.a[1], B1 = s.b[1];
    if constexpr (MODE == kPipeMem) {
      SinkV(s.p[0] ^ A0 ^ B0);                 // a few VALU per step: the loads must be consumed
      SinkV(s.p[1] ^ A1 ^ B1);
      SinkV(static_cast<uint32_t>(s.act.w[0][0] ^ s.act.w[0][1] ^ s.act.w[1][0] ^ s.act.w[1][1]));
    } else if constexpr (MODE == kPipeValu) {
      v8h f0, f1, g0, g1;
      r4d_trellis_k4_decode(s.p[0], A0, B0, f0, f1);
      r4d_trellis_k4_decode(s.p[1], A1, B1, g0, g1);
      SinkH(f0); SinkH(f1); SinkH(g0); SinkH(g1);
    } else if constexpr (MODE == kPipeWmma) {
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed0, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed1, acc1);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed1, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed0, acc1);
    } else if constexpr (MODE == kPipeAluIndep) {
      v8h f0, f1, g0, g1;
      r4d_trellis_k4_decode(s.p[0], A0, B0, f0, f1);
      r4d_trellis_k4_decode(s.p[1], A1, B1, g0, g1);
      SinkH(f0); SinkH(f1); SinkH(g0); SinkH(g1);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed0, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed1, acc1);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed1, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af.h, fixed0, acc1);
    } else {                                   // kPipeAlu (fixed A), kPipeAll (the loaded A)
      v8h f0, f1, g0, g1;
      r4d_trellis_k4_decode(s.p[0], A0, B0, f0, f1);
      r4d_trellis_k4_decode(s.p[1], A1, B1, g0, g1);
      const v8h a0 = MODE == kPipeAll ? Frag(s.act.w[0]) : af.h;
      const v8h a1 = MODE == kPipeAll ? Frag(s.act.w[1]) : af.h;
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, f0, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, f1, acc1);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, g0, acc0);
      acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, g1, acc1);
    }
  };

  PipeStage s0{}, s1{};
  if constexpr (kMem) {
    load(s0, 0);
  } else {
    s0.a[0] = lane * 0x85EBCA6Bu; s0.b[0] = lane * 0xC2B2AE35u; s0.p[0] = lane * 0x9E3779B9u;
    s0.a[1] = s0.a[0] ^ 0x5678u; s0.b[1] = s0.b[0] ^ 0x9ABCu; s0.p[1] = s0.p[0] ^ 0x1234u;
    s1 = s0;
  }
  __syncthreads();
  const unsigned long long c0 = clock64();
  for (int it = 0; it < iters; it += 2) {      // iters is even (the host rounds it)
    if constexpr (kMem) load(s1, it + 1);
    __builtin_amdgcn_sched_barrier(0);
    work(s0);
    if constexpr (kMem) load(s0, it + 2 < iters ? it + 2 : it + 1);
    __builtin_amdgcn_sched_barrier(0);
    work(s1);
    Opaque(off_ab);
    Opaque(off_p);
    Opaque(aoff);
  }
  const unsigned long long c1 = clock64();
  if (lane == 0) {
    cyc[wave] = c1 - c0;
    hwid[wave] = HwId1();
  }
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) s += acc0[e] + acc1[e];
  if (s == 1234.5f) sink[wave] = s;            // keeps the accumulators alive
  if (probe) {
    clk[2] = clock64();
    clk[3] = wall_clock64();
  }
}

template <int OP, int D, bool DELAY>
void LaunchOp(int wgs, int waves_per_wg, int iters, unsigned long long* cyc, uint32_t* hwid,
              uint32_t* sink, hipStream_t st) {
  hipLaunchKernelGGL((OpRateKernel<OP, D, DELAY>), dim3(wgs), dim3(waves_per_wg * 32), 0, st, cyc, hwid,
                     sink, iters);
}

template <int OP>
void LaunchOpD(int dist, bool delay, int wgs, int waves_per_wg, int iters, unsigned long long* cyc,
               uint32_t* hwid, uint32_t* sink, hipStream_t st) {
  switch (dist * 2 + (delay ? 1 : 0)) {
    case 2: LaunchOp<OP, 1, false>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 3: LaunchOp<OP, 1, true>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 4: LaunchOp<OP, 2, false>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 5: LaunchOp<OP, 2, true>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 6: LaunchOp<OP, 3, false>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 7: LaunchOp<OP, 3, true>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 8: LaunchOp<OP, 4, false>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 9: LaunchOp<OP, 4, true>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    case 16: LaunchOp<OP, 8, false>(wgs, waves_per_wg, iters, cyc, hwid, sink, st); return;
    default:
      throw std::runtime_error("r4dx_tq_bench_op_rate: distance " + std::to_string(dist) +
                               (delay ? " with" : " without") + " s_delay_alu");
  }
}

template <int MODE>
void LaunchPipe(int wgs, int waves_per_wg, int iters, const unsigned char* w, const unsigned char* a,
                int64_t bpw, unsigned long long* cyc, uint32_t* hwid, float* sink,
                unsigned long long* clk, hipStream_t st) {
  hipLaunchKernelGGL(PipeKernel<MODE>, dim3(wgs), dim3(waves_per_wg * 32), 0, st, w, a, bpw, iters, cyc,
                     hwid, sink, clk);
}

}  // namespace

extern "C" {

void r4dx_tq_bench_fill_u32(int64_t p, int64_t n, uint32_t seed, uint32_t and_mask, uint32_t or_mask,
                            int64_t stream) {
  hipLaunchKernelGGL(FillU32Kernel, dim3(4096), dim3(256), 0, (hipStream_t)stream, (uint32_t*)p, n,
                     seed, and_mask, or_mask);
}

void r4dx_tq_bench_read(int64_t p, int64_t bytes, int64_t sink, int64_t stream) {
  hipLaunchKernelGGL(ReadKernel, dim3(4096), dim3(256), 0, (hipStream_t)stream, (const r4d_u32x4*)p,
                     bytes / 16, (uint32_t*)sink);
}

void r4dx_tq_bench_stamp(int64_t out, int64_t stream) {
  hipLaunchKernelGGL(StampKernel, dim3(1), dim3(1), 0, (hipStream_t)stream, (unsigned long long*)out);
}

void r4dx_tq_bench_clock_probe(int64_t out, int iters, int64_t stream) {
  hipLaunchKernelGGL(ClockProbeKernel, dim3(1), dim3(32), 0, (hipStream_t)stream,
                     (unsigned long long*)out, iters);
}

// op: 0 v_mad_u32_u16, 1 v_pk_mad_u16, 2 v_sad_u8, 3 v_sad_hi_u8, 4 v_mul_lo_u32, 5 v_alignbit_b32,
// 6 v_pk_fma_f16, 7 v_add_nc_u32, 8 v_fma_f32 (register forms), then the decode's compiled forms:
// 9 v_mad_u32_u16 op_sel:[1,0,0,0], 10 v_pk_mad_u16 op_sel_hi:[1,1,1], 11 the same [1,0,1],
// 12 v_sad_u8 with the 0x64006400 literal, 13 v_pk_fma_f16 with the 0xc931 literal. dist: 8 (issue
// rate) or 1..4 (each op reads the one dist before it); delay: an s_delay_alu instid0(VALU_DEP_dist)
// before each op (dist <= 4). Each wave issues iters * 24 of the op.
void r4dx_tq_bench_op_rate(int op, int dist, int delay, int wgs, int waves_per_wg, int iters, int64_t cyc,
                           int64_t hwid, int64_t sink, int64_t stream) {
  auto c = (unsigned long long*)cyc;
  auto h = (uint32_t*)hwid;
  auto s = (uint32_t*)sink;
  auto st = (hipStream_t)stream;
  const bool d = delay != 0;
  switch (op) {
    case kMadU32U16: LaunchOpD<kMadU32U16>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kPkMadU16: LaunchOpD<kPkMadU16>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kSadU8: LaunchOpD<kSadU8>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kSadHiU8: LaunchOpD<kSadHiU8>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kMulLoU32: LaunchOpD<kMulLoU32>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kAlignbit: LaunchOpD<kAlignbit>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kPkFmaF16: LaunchOpD<kPkFmaF16>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kAddU32: LaunchOpD<kAddU32>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kFmaF32: LaunchOpD<kFmaF32>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kMadU32U16Hi: LaunchOpD<kMadU32U16Hi>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kPkMadU16Hi: LaunchOpD<kPkMadU16Hi>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kPkMadU16Lo: LaunchOpD<kPkMadU16Lo>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kSadU8Lit: LaunchOpD<kSadU8Lit>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    case kPkFmaF16Lit: LaunchOpD<kPkFmaF16Lit>(dist, d, wgs, waves_per_wg, iters, c, h, s, st); return;
    default: throw std::runtime_error("r4dx_tq_bench_op_rate: op " + std::to_string(op));
  }
}

// mode: 0 mem, 1 valu, 2 wmma, 3 alu (decode -> wmma), 4 alu_indep, 5 all. One step = 2 k-tiles of
// one tile pair per wave (512 B of words per wave, 32 weights per lane) and their two activation
// fragments from `a` (>= 64 KiB + 64 B, L2-resident); bytes_per_wave >= iters * 512 for the memory
// modes.
void r4dx_tq_bench_pipe(int mode, int wgs, int waves_per_wg, int iters, int64_t w, int64_t a,
                        int64_t bytes_per_wave, int64_t cyc, int64_t hwid, int64_t sink, int64_t clk,
                        int64_t stream) {
  auto wp = (const unsigned char*)w;
  auto ap = (const unsigned char*)a;
  auto c = (unsigned long long*)cyc;
  auto h = (uint32_t*)hwid;
  auto s = (float*)sink;
  auto k = (unsigned long long*)clk;
  auto st = (hipStream_t)stream;
  switch (mode) {
    case kPipeMem: LaunchPipe<kPipeMem>(wgs, waves_per_wg, iters, wp, ap, bytes_per_wave, c, h, s, k, st); return;
    case kPipeValu: LaunchPipe<kPipeValu>(wgs, waves_per_wg, iters, wp, ap, bytes_per_wave, c, h, s, k, st); return;
    case kPipeWmma: LaunchPipe<kPipeWmma>(wgs, waves_per_wg, iters, wp, ap, bytes_per_wave, c, h, s, k, st); return;
    case kPipeAlu: LaunchPipe<kPipeAlu>(wgs, waves_per_wg, iters, wp, ap, bytes_per_wave, c, h, s, k, st); return;
    case kPipeAluIndep: LaunchPipe<kPipeAluIndep>(wgs, waves_per_wg, iters, wp, ap, bytes_per_wave, c, h, s, k, st); return;
    case kPipeAll: LaunchPipe<kPipeAll>(wgs, waves_per_wg, iters, wp, ap, bytes_per_wave, c, h, s, k, st); return;
    default: throw std::runtime_error("r4dx_tq_bench_pipe: mode " + std::to_string(mode));
  }
}

}  // extern "C"
