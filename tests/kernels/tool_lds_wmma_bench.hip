// tests/kernels/tool_lds_wmma_bench.hip -- standalone (host + device) microbench for the LDS ceiling
// ("Q12", docs/r9700.md 5 / docs/trellis-kernel.md 4.7 "M8") on gfx1201, plus the interaction of LDS
// traffic with WMMA and VALU on one SIMD. Answers: how many bytes per clock per WGP can ds_load_b128 /
// ds_store_b128 move (conflict-free lane-contiguous 16 B, the pattern a B fragment in its register
// layout has), and how many WMMA per clock per SIMD survive when LDS reads (B fragments staged by other
// waves) and, separately, decode-sized VALU work are interleaved with them.
//
// Build (no CMake target; a hipcc one-liner, see docs/perf notes in the A1 report):
//   hipcc.exe -O3 -std=c++17 --offload-arch=gfx1201 tool_lds_wmma_bench.hip -o tool_lds_wmma_bench.exe
// Run (ONE device; the caller sets HIP_VISIBLE_DEVICES):
//   tool_lds_wmma_bench.exe [--out x.json] [--reps 5]
//
// Method. One workgroup per WGP (grid = the WGP count, 4 SIMDs each, waves = 4 * waves_per_simd, the
// HW_ID1 of every wave is read back to check that no WGP got two workgroups). Time is clock64() (the
// shader clock) between two workgroup barriers; the shader clock rate is d(clock64) / d(wall_clock64)
// against the ~100 MHz REALTIME counter. Every kernel is bounded (fixed trip counts).
//
// The LDS traffic is inline-asm ds_load_b128 / ds_store_b128 with s_wait_dscnt only where noted, so the
// compiler cannot merge, hoist or drop it and adds no VALU. WMMA is the same builtin the GEMM uses.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); \
  std::exit(1); } } while (0)

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ uint32_t HwId1() {
  uint32_t v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(v));
  return v;
}

static __device__ __forceinline__ v8f Mma(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// ds_load_b128 / ds_store_b128 with a compile-time byte offset off a per-lane 32-bit LDS address.
#define DS_LOAD(dst, addr, off) asm volatile("ds_load_b128 %0, %1 offset:%2" : "=v"(dst) : "v"(addr), "i"(off))
#define DS_STORE(addr, src, off) asm volatile("ds_store_b128 %0, %1 offset:%2" :: "v"(addr), "v"(src), "i"(off))
#define DS_WAIT() asm volatile("s_wait_dscnt 0x0" ::: "memory")
// The compiler must consider the loaded registers used, without a VALU op that would perturb the
// measurement: an empty asm that "reads" them.
#define KEEP(x) asm volatile("" :: "v"(x))
#define KEEP_RW(x) asm volatile("" : "+v"(x))

enum Kind {
  kRead,        // ds_load_b128 only, kLoads per burst, wait, repeat
  kWrite,       // ds_store_b128 only
  kReadWrite,   // 3 reads : 1 write (the staged-fragment ratio at MT = 4 is 4 : 1)
  kWmmaOnly,    // kMma WMMA per iteration, A and B in registers
  kWmmaLds,     // kMma WMMA per iteration + kLoads ds_load_b128 feeding the NEXT iteration's B
  kWmmaLdsAB,   // ... and kLoads more that feed the A operands (both operands staged)
  kValuOnly,    // kValu independent-ish v_add per iteration
  kWmmaValu,    // WMMA + VALU (no overlap probe)
  kFull,        // kValu VALU + ds_store (1 per 4 WMMA... ) + barrier-free ds_load + WMMA: the M8 inner loop
};

template <int KIND, int LOADS, int MMA, int VALU, int STORES>
__global__ __launch_bounds__(1024) void Bench(unsigned long long* out, uint32_t* hwid, float* sink,
                                              int iters) {
  extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  // lane-contiguous 16 B per lane: a conflict-free 512 B per wave instruction. Each wave gets its own
  // 8 KiB window (up to 32 waves = 256 KiB would not fit): windows wrap inside the 64 KiB allocation.
  const uint32_t base = (uint32_t)(((wave & 1) * 32768) + lane * 16);
  {
    uint32_t* p = reinterpret_cast<uint32_t*>(lds);
    for (int i = threadIdx.x; i < 16384; i += blockDim.x) p[i] = 0x3C003C00u ^ (uint32_t)i;
  }
  constexpr int RN = KIND == kFull ? LOADS + STORES : (LOADS > 8 ? LOADS : 8);
  v4u r[RN];
#pragma unroll
  for (int j = 0; j < RN; ++j) r[j] = v4u{(unsigned)j, (unsigned)lane, 3u, 4u};
  v8f acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = (v8f)(0.f);
  v8h a[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    a[j] = __builtin_bit_cast(v8h, v4u{0x3C003C00u ^ (unsigned)lane, 0x3C003C00u, 0x3C003C00u ^ (unsigned)j, 0x3C00u});
  v8h b[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    b[j] = __builtin_bit_cast(v8h, v4u{0x3C003C00u, 0x3C003C00u ^ (unsigned)lane, 0x3C00u, 0x3C003C00u ^ (unsigned)j});
  uint32_t vx[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) vx[j] = (uint32_t)lane * 2654435761u + j;
  __syncthreads();
  const unsigned long long c0 = clock64();
  const unsigned long long w0 = wall_clock64();
  for (int it = 0; it < iters; ++it) {
    if constexpr (KIND == kRead) {
#pragma unroll
      for (int j = 0; j < LOADS; ++j) DS_LOAD(r[j], base, (j * 512) & 0x3FFF);
      DS_WAIT();
#pragma unroll
      for (int j = 0; j < LOADS; ++j) KEEP(r[j]);
    } else if constexpr (KIND == kWrite) {
#pragma unroll
      for (int j = 0; j < LOADS; ++j) DS_STORE(base, r[j], (j * 512) & 0x3FFF);
      DS_WAIT();
    } else if constexpr (KIND == kReadWrite) {
#pragma unroll
      for (int j = 0; j < LOADS; ++j) {
        if (j % 4 == 3) DS_STORE(base, r[j], (j * 512) & 0x3FFF);
        else DS_LOAD(r[j], base, (j * 512) & 0x3FFF);
      }
      DS_WAIT();
#pragma unroll
      for (int j = 0; j < LOADS; ++j) KEEP(r[j]);
    } else if constexpr (KIND == kWmmaOnly) {
      // MMA WMMA on independent accumulators (MMA <= 16), A[j&3] x B[(j>>2)&3]
#pragma unroll
      for (int j = 0; j < MMA; ++j) acc[j] = Mma(a[j & 1], b[(j >> 1) & 1], acc[j]);
    } else if constexpr (KIND == kWmmaLds || KIND == kWmmaLdsAB) {
      // issue the loads for the next iteration (into r), run this iteration's WMMA on the current
      // fragments, then wait and swap in the loaded registers as the next B (and A) operands
#pragma unroll
      for (int j = 0; j < LOADS; ++j) DS_LOAD(r[j], base, (j * 512) & 0x3FFF);
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int j = 0; j < MMA; ++j) acc[j] = Mma(a[j & 1], b[(j >> 1) & 1], acc[j]);
      __builtin_amdgcn_sched_barrier(0);
      DS_WAIT();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // the register file is reused as the next operands: an empty asm that ties b[j] to r[j]
        // (no move is emitted when the allocator can place them together; at worst one v_mov per reg)
        KEEP_RW(b[j]);
        KEEP(r[j]);
      }
    } else if constexpr (KIND == kValuOnly) {
#pragma unroll
      for (int j = 0; j < VALU; ++j) asm volatile("v_add_nc_u32 %0, %0, %1" : "+v"(vx[j & 7]) : "v"(vx[(j + 3) & 7]));
    } else if constexpr (KIND == kWmmaValu) {
#pragma unroll
      for (int j = 0; j < MMA; ++j) acc[j] = Mma(a[j & 1], b[(j >> 1) & 1], acc[j]);
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int j = 0; j < VALU; ++j) asm volatile("v_add_nc_u32 %0, %0, %1" : "+v"(vx[j & 7]) : "v"(vx[(j + 3) & 7]));
      __builtin_amdgcn_sched_barrier(0);
    } else {   // kFull: STORES ds_store + LOADS ds_load + MMA WMMA + VALU v_add per iteration
#pragma unroll
      for (int j = 0; j < LOADS; ++j) DS_LOAD(r[j], base, (j * 512) & 0x3FFF);
#pragma unroll
      for (int j = 0; j < STORES; ++j) DS_STORE(base, r[LOADS + j], 0x4000 + ((j * 512) & 0x1FFF));
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int j = 0; j < VALU; ++j) asm volatile("v_add_nc_u32 %0, %0, %1" : "+v"(vx[j & 7]) : "v"(vx[(j + 3) & 7]));
#pragma unroll
      for (int j = 0; j < MMA; ++j) acc[j] = Mma(a[j & 1], b[(j >> 1) & 1], acc[j]);
      __builtin_amdgcn_sched_barrier(0);
      DS_WAIT();
#pragma unroll
      for (int j = 0; j < LOADS; ++j) KEEP(r[j]);
    }
  }
  __syncthreads();
  const unsigned long long c1 = clock64();
  const unsigned long long w1 = wall_clock64();
  if (threadIdx.x == 0) {
    out[blockIdx.x * 4 + 0] = c1 - c0;
    out[blockIdx.x * 4 + 1] = w1 - w0;
  }
  if (lane == 0) hwid[blockIdx.x * 32 + wave] = HwId1();
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) s += acc[j][0] + acc[j][7];
#pragma unroll
  for (int j = 0; j < RN; ++j) s += (float)r[j].x;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += (float)vx[j];
  if (s == 1234.5f) sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  (void)nw;
}

struct Row {
  std::string name;
  int waves_per_simd;
  double clk_per_iter_per_wave;     // wave-iteration cost in shader clocks with all waves running
  double lds_bytes_per_clk_wgp;     // 0 when no LDS traffic
  double wmma_per_clk_simd;         // 0 when no WMMA
  double valu_per_clk_simd;
  double sclk_mhz;
  int wgps_distinct;
  int wgs;
};

static double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

int main(int argc, char** argv) {
  std::string outp;
  int reps = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
  }
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  const int cus = prop.multiProcessorCount;
  const int wgps = cus / 2;
  std::printf("device %s: %d CUs -> %d WGPs, clock rate attr %d kHz, shared/block %zu\n", prop.name, cus, wgps,
              prop.clockRate, prop.sharedMemPerBlock);
  const int kMaxWaves = 32;
  unsigned long long* d_out;
  uint32_t* d_hw;
  float* d_sink;
  CK(hipMalloc(&d_out, (size_t)wgps * 4 * 8));
  CK(hipMalloc(&d_hw, (size_t)wgps * 32 * 4));
  CK(hipMalloc(&d_sink, (size_t)wgps * 1024 * 4));
  std::vector<Row> rows;

  auto run = [&](const char* name, auto kernel, int wps, int iters, double lds_bytes_per_iter_per_wave,
                 double wmma_per_iter, double valu_per_iter) {
    const int waves = 4 * wps;
    // 64 KiB of LDS per workgroup: one workgroup per WGP by construction only if it also uses the
    // registers; the HW_ID check below says what happened.
    const size_t smem = 64 * 1024;
    CK(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    std::vector<double> clks, sclks;
    int distinct = 0;
    for (int rep = 0; rep < reps + 2; ++rep) {
      hipLaunchKernelGGL(kernel, dim3(wgps), dim3(waves * 32), smem, 0, d_out, d_hw, d_sink, iters);
      CK(hipDeviceSynchronize());
      if (rep < 2) continue;   // warm-up
      std::vector<unsigned long long> o((size_t)wgps * 4);
      std::vector<uint32_t> hw((size_t)wgps * 32);
      CK(hipMemcpy(o.data(), d_out, o.size() * 8, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hw.data(), d_hw, hw.size() * 4, hipMemcpyDeviceToHost));
      std::set<uint32_t> keys;
      for (int g = 0; g < wgps; ++g) {
        const uint32_t v = hw[g * 32];
        const uint32_t wgp = (v >> 10) & 0xF, sa = (v >> 16) & 1, se = (v >> 18) & 7;
        keys.insert((se << 8) | (sa << 4) | wgp);
      }
      distinct = (int)keys.size();
      for (int g = 0; g < wgps; ++g) {
        clks.push_back((double)o[g * 4]);
        sclks.push_back((double)o[g * 4] / (double)o[g * 4 + 1] * 100.0);   // MHz at 100 MHz REALTIME
      }
    }
    const double c = Median(clks);
    Row r;
    r.name = name;
    r.waves_per_simd = wps;
    r.clk_per_iter_per_wave = c / iters;
    // all `waves` waves of a WGP run the same loop concurrently: WGP-level rates
    r.lds_bytes_per_clk_wgp = lds_bytes_per_iter_per_wave * waves * iters / c;
    r.wmma_per_clk_simd = wmma_per_iter * wps * iters / c;
    r.valu_per_clk_simd = valu_per_iter * wps * iters / c;
    r.sclk_mhz = Median(sclks);
    r.wgps_distinct = distinct;
    r.wgs = wgps;
    rows.push_back(r);
    std::printf("%-34s wps=%d  %9.1f clk/iter  LDS %7.2f B/clk/WGP  WMMA %.4f /clk/SIMD (%.2f clk each)  VALU %.3f /clk/SIMD  sclk %.0f MHz  wgps %d/%d\n",
                name, wps, r.clk_per_iter_per_wave, r.lds_bytes_per_clk_wgp, r.wmma_per_clk_simd,
                r.wmma_per_clk_simd > 0 ? 1.0 / r.wmma_per_clk_simd : 0.0, r.valu_per_clk_simd, r.sclk_mhz, distinct, wgps);
  };

  char nm[128];
  // ---- pure LDS ----
  for (int wps : {1, 2, 4, 8}) {
    std::snprintf(nm, sizeof nm, "read_b128 x16");
    run(nm, Bench<kRead, 16, 0, 0, 0>, wps, 4000, 16 * 512.0, 0, 0);
  }
  for (int wps : {1, 2, 4, 8}) run("read_b128 x8", Bench<kRead, 8, 0, 0, 0>, wps, 8000, 8 * 512.0, 0, 0);
  for (int wps : {1, 2, 4, 8}) run("write_b128 x16", Bench<kWrite, 16, 0, 0, 0>, wps, 4000, 16 * 512.0, 0, 0);
  for (int wps : {1, 2, 4, 8}) run("read3:write1 x16", Bench<kReadWrite, 16, 0, 0, 0>, wps, 4000, 16 * 512.0, 0, 0);
  // ---- WMMA alone ----
  for (int wps : {1, 2, 4, 8}) run("wmma x16 (regs)", Bench<kWmmaOnly, 0, 16, 0, 0>, wps, 2000, 0, 16, 0);
  for (int wps : {1, 2, 4, 8}) run("wmma x8 (regs)", Bench<kWmmaOnly, 0, 8, 0, 0>, wps, 4000, 0, 8, 0);
  // ---- WMMA + LDS: B fragments staged in LDS (1 ds_load per 4 WMMA = MT 4), then 2 per 4 (MT 2), 1:1 ----
  for (int wps : {1, 2, 4, 8}) run("wmma16 + 4 lds (B, MT=4)", Bench<kWmmaLds, 4, 16, 0, 0>, wps, 2000, 4 * 512.0, 16, 0);
  for (int wps : {1, 2, 4, 8}) run("wmma16 + 8 lds (A+B)", Bench<kWmmaLdsAB, 8, 16, 0, 0>, wps, 2000, 8 * 512.0, 16, 0);
  for (int wps : {1, 2, 4, 8}) run("wmma8 + 4 lds (B, MT=2)", Bench<kWmmaLds, 4, 8, 0, 0>, wps, 4000, 4 * 512.0, 8, 0);
  for (int wps : {1, 2, 4, 8}) run("wmma8 + 8 lds (MT=1)", Bench<kWmmaLds, 8, 8, 0, 0>, wps, 4000, 8 * 512.0, 8, 0);
  // ---- VALU and no-overlap sanity ----
  for (int wps : {1, 2, 4, 8}) run("valu x62", Bench<kValuOnly, 0, 0, 62, 0>, wps, 4000, 0, 0, 62);
  for (int wps : {1, 2, 4, 8}) run("wmma16 + valu62", Bench<kWmmaValu, 0, 16, 62, 0>, wps, 2000, 0, 16, 62);
  // ---- the M8 inner loop for one pair-block per 256 rows, per wave: 62 VALU decode / 4 waves = 16 per
  //      wave, 2 ds_store (this wave's share of the fragments), 4 ds_load (B) , 16 WMMA ----
  for (int wps : {1, 2, 4, 8})
    run("m8 mock: v16 st2 ld4 mma16", Bench<kFull, 4, 16, 16, 2>, wps, 2000, (4 + 2) * 512.0, 16, 16);
  for (int wps : {1, 2, 4, 8})
    run("m8 mock: v62 st2 ld4 mma16", Bench<kFull, 4, 16, 62, 2>, wps, 2000, (4 + 2) * 512.0, 16, 62);

  if (!outp.empty()) {
    FILE* f = std::fopen(outp.c_str(), "wb");
    std::fprintf(f, "{\"device\":\"%s\",\"cus\":%d,\"wgps\":%d,\"rows\":[\n", prop.name, cus, wgps);
    for (size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      std::fprintf(f, " {\"name\":\"%s\",\"waves_per_simd\":%d,\"clk_per_iter_per_wave\":%.2f,"
                      "\"lds_bytes_per_clk_wgp\":%.3f,\"wmma_per_clk_simd\":%.5f,\"valu_per_clk_simd\":%.4f,"
                      "\"sclk_mhz\":%.1f,\"wgps_distinct\":%d,\"wgs\":%d}%s\n",
                   r.name.c_str(), r.waves_per_simd, r.clk_per_iter_per_wave, r.lds_bytes_per_clk_wgp,
                   r.wmma_per_clk_simd, r.valu_per_clk_simd, r.sclk_mhz, r.wgps_distinct, r.wgs,
                   i + 1 < rows.size() ? "," : "");
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
    std::printf("wrote %s\n", outp.c_str());
  }
  return 0;
}
