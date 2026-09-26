// tools/reference/trellis_viterbi.hip -- the Viterbi encoder of the EXL3 trellis oracle
// (tools/reference/trellis_quant.py, spec docs/trellis.md section 6), in two native builds of this
// one file:
//
//   CPU  clang++ -x c++ -DTRELLIS_CPU -O3 -march=native -ffp-contract=off -shared   (std::thread)
//   HIP  clang++ -x hip --offload-arch=gfx1201 -O3 -ffp-contract=off -shared -lamdhip64
//
// trellis_quant.py builds both on first use (into tools/reference/.trellis_build/, keyed by this
// file's sha256) and calls them through ctypes; its pure-torch `viterbi_torch` is the reference both
// are tested against (tests/reference/test_trellis_quant.py): same objective, same arithmetic, same
// tie rules, so the three produce the SAME states bit for bit.
//
// The trellis (docs/trellis.md 3.3, 5, 6; QTIP arXiv 2406.11235 3.1-3.2; exllamav3 @ 6b84a21
// quantize_tiles_kernel.cuh / quantize_tiles_frac_kernel.cuh, MIT, (c) turboderp -- the recursion
// and the two-pass tail-biting below follow that kernel's structure):
//
//   * a tile is a ring of 256 positions p; position p owns D(p) bits, D(p) = KA + ((MASK >> (p & 15))
//     & 1) (integer K: MASK = 0; K + 0.5: MASK = 0xAAAA); its 16-bit state is the window of the bit
//     ring ENDING at its own bits, so state(p) = ((state(p-1) << D(p)) & 0xFFFF) | own(p).
//   * between positions p and p+1 the node ("edge") is their 16 - D(p+1) overlapping bits: the LOW
//     16 - D(p+1) bits of state(p) (its out-edge) = the HIGH bits of state(p+1) (its in-edge,
//     state(p+1) >> D(p+1)).
//   * step p, with a = D(p), b = D(p+1): for every out-edge o, over the 2^b candidate states
//     s = (h << (16 - b)) | o,   C_p(o) = min_h C_{p-1}(s >> a) + (cb[s] - x_p)^2.
//     Writing o = (g << a) | o_lo gives s >> a = (h << (16 - a - b)) | g: every out-edge of "group"
//     g reads the same 2^b predecessors, which is how both builds below lay the work out.
//   * arithmetic: fp32, d = cb[s] - x, t = C + d*d with two roundings (no FMA: -ffp-contract=off and
//     __fmul_rn/__fadd_rn on the device), strict < while h ascends, so ties go to the smallest h,
//     i.e. the smallest predecessor (EXL3's rule). The final argmin over edges takes the smallest
//     edge on a tie (EXL3 uses a bit-reversed rank there; recorded deviation, docs/trellis.md 6.1).
//     EXL3 keeps fp16 path costs; fp32 is the spec's allowed deviation (6.5: <= 0.04% MSE).
//   * tail-biting, EXL3's two passes: pass 1 starts at position 128 with every in-edge free (cost 0),
//     runs 256 steps, takes the best final edge and traces back only to position 0; the in-edge found
//     there is `end`. Pass 2 starts at position 0 with only in-edge `end` allowed (cost 0, others
//     +inf), runs 256 steps and traces back from out-edge `end` at position 255, which closes the
//     ring exactly. The returned cost is pass 2's C_255(end) = the path's total squared error.
//
// Outputs per tile: 256 int32 states (the 16-bit state of every position, in the order the targets
// were given, i.e. EXL3's tensor-core order) and the float cost.
//
// Supported widths: CPU any D in 1..8 with D(p) + D(p+1) <= 16; HIP D in 3..6 (the LDS holds one
// cost per edge, 2^(16 - Dmin) floats: Dmin = 3 is 32 KiB; K = 2 would need 64 KiB), which covers
// every rate the oracle uses (3.5 = 3/4, 4, 4/5, 5, 6). trellis_quant.py falls back to the torch
// path for anything else.

#include <cstdint>
#include <cstring>
#include <utility>

#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)
#define R4DX_EXPORT extern "C" __declspec(dllexport)
#else
#define R4DX_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define R4DX_TRELLIS_ABI 1

namespace {

constexpr int kSteps = 256;
constexpr int kRoll = kSteps / 2;

// D(p) for the (KA, MASK) width pattern.
#if defined(__HIP__)
__host__ __device__
#endif
inline int width_of(int ka, unsigned mask, int p) {
  return ka + static_cast<int>((mask >> (p & 15)) & 1u);
}

}  // namespace

// ================================================================================================
// CPU build
// ================================================================================================
#if defined(TRELLIS_CPU)

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#pragma clang fp contract(off)

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

using StepFn = void (*)(const float*, float*, uint8_t*, const float*, float);

// One trellis step, a = A, b = B. prev: 2^(16-A) in-edge costs; cur: 2^(16-B) out-edge costs;
// bp: the chosen h per out-edge; cb: the 65536-entry codebook; w: this position's target.
template <int A, int B>
void cpu_step(const float* __restrict prev, float* __restrict cur, uint8_t* __restrict bp,
              const float* __restrict cb, float w) {
  constexpr int NA = 1 << A, NB = 1 << B;
  constexpr int GB = 16 - A - B;
  constexpr int G = 1 << GB;
  for (int g = 0; g < G; ++g) {
    float best[NA];
    int32_t bh[NA];
    for (int o = 0; o < NA; ++o) {
      best[o] = kInf;
      bh[o] = 0;
    }
    for (int h = 0; h < NB; ++h) {
      const float pc = prev[(h << GB) | g];
      const float* row = cb + ((h << (16 - B)) | (g << A));
      for (int o = 0; o < NA; ++o) {
        const float d = row[o] - w;
        const float sq = d * d;
        const float t = pc + sq;
        const bool lt = t < best[o];
        best[o] = lt ? t : best[o];
        bh[o] = lt ? h : bh[o];
      }
    }
    for (int o = 0; o < NA; ++o) {
      cur[(g << A) | o] = best[o];
      bp[(g << A) | o] = static_cast<uint8_t>(bh[o]);
    }
  }
}

template <int A, int B>
constexpr StepFn step_entry() {
  if constexpr (A + B <= 16) {
    return &cpu_step<A, B>;
  } else {
    return nullptr;
  }
}

template <int A, int... Bs>
constexpr void fill_row(StepFn* row, std::integer_sequence<int, Bs...>) {
  ((row[Bs + 1] = step_entry<A, Bs + 1>()), ...);
}

struct StepTable {
  StepFn fn[9][9] = {};
  StepTable() { fill(std::make_integer_sequence<int, 8>{}); }
  template <int... As>
  void fill(std::integer_sequence<int, As...>) {
    (fill_row<As + 1>(fn[As + 1], std::make_integer_sequence<int, 8>{}), ...);
  }
};

const StepTable& steps() {
  static const StepTable t;
  return t;
}

struct TileScratch {
  std::vector<float> a, b;
  std::vector<uint8_t> bp;
};

// The whole two-pass tail-biting Viterbi for one tile. Returns pass 2's path cost.
float cpu_tile(const float* x, int ka, unsigned mask, const float* cb, int32_t* states, int emax,
               TileScratch& s) {
  const StepTable& tab = steps();
  float* prev = s.a.data();
  float* cur = s.b.data();
  uint8_t* bp = s.bp.data();

  // pass 1: ring rolled by 128, every in-edge of position 128 free
  const int nin1 = 1 << (16 - width_of(ka, mask, kRoll));
  std::fill(prev, prev + nin1, 0.0f);
  for (int i = 0; i < kSteps; ++i) {
    const int p = (i + kRoll) & (kSteps - 1);
    const int a = width_of(ka, mask, p), b = width_of(ka, mask, (p + 1) & (kSteps - 1));
    tab.fn[a][b](prev, cur, bp + static_cast<size_t>(p) * emax, cb, x[p]);
    std::swap(prev, cur);
  }
  const int nout1 = 1 << (16 - width_of(ka, mask, kRoll));  // out-edges of position 127
  int e = 0;
  float bestc = prev[0];
  for (int i = 1; i < nout1; ++i) {
    if (prev[i] < bestc) {
      bestc = prev[i];
      e = i;
    }
  }
  for (int p = kRoll - 1; p >= 0; --p) {  // trace back to position 0 only
    const int a = width_of(ka, mask, p), b = width_of(ka, mask, (p + 1) & (kSteps - 1));
    const int h = bp[static_cast<size_t>(p) * emax + e];
    const int st = (h << (16 - b)) | e;
    e = st >> a;
  }
  const int end = e;  // in-edge of position 0 in pass 1's path

  // pass 2: position 0's in-edge pinned to `end`
  const int nin2 = 1 << (16 - width_of(ka, mask, 0));
  std::fill(prev, prev + nin2, kInf);
  prev[end] = 0.0f;
  for (int p = 0; p < kSteps; ++p) {
    const int a = width_of(ka, mask, p), b = width_of(ka, mask, (p + 1) & (kSteps - 1));
    tab.fn[a][b](prev, cur, bp + static_cast<size_t>(p) * emax, cb, x[p]);
    std::swap(prev, cur);
  }
  const float cost = prev[end];
  e = end;
  for (int p = kSteps - 1; p >= 0; --p) {
    const int a = width_of(ka, mask, p), b = width_of(ka, mask, (p + 1) & (kSteps - 1));
    const int h = bp[static_cast<size_t>(p) * emax + e];
    const int st = (h << (16 - b)) | e;
    states[p] = st;
    e = st >> a;
  }
  return cost;
}

bool widths_ok_cpu(int ka, unsigned mask) {
  if (mask > 0xFFFFu) return false;
  for (int p = 0; p < 16; ++p) {
    const int a = width_of(ka, mask, p), b = width_of(ka, mask, p + 1);
    if (a < 1 || a > 8 || b < 1 || b > 8 || a + b > 16) return false;
  }
  return true;
}

}  // namespace

R4DX_EXPORT int r4dx_trellis_abi() { return R4DX_TRELLIS_ABI; }

// targets: T x 256 fp32 (already rounded as the caller wants); codebook: 65536 fp32;
// out_states: T x 256 int32; out_cost: T fp32 (may be null). Returns 0, or a negative error code.
R4DX_EXPORT int r4dx_trellis_viterbi_cpu(const float* targets, int64_t T, int ka, unsigned mask,
                                         const float* codebook, int32_t* out_states,
                                         float* out_cost, int nthreads) {
  if (T < 0 || !targets || !codebook || !out_states) return -1;
  if (!widths_ok_cpu(ka, mask)) return -2;
  int dmin = 8;
  for (int p = 0; p < 16; ++p) dmin = std::min(dmin, width_of(ka, mask, p));
  const int emax = 1 << (16 - dmin);
  if (nthreads <= 0) nthreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  nthreads = static_cast<int>(std::min<int64_t>(nthreads, std::max<int64_t>(T, 1)));
  std::atomic<int64_t> next{0};
  auto worker = [&]() {
    TileScratch s;
    s.a.assign(emax, 0.0f);
    s.b.assign(emax, 0.0f);
    s.bp.assign(static_cast<size_t>(kSteps) * emax, 0);
    for (;;) {
      const int64_t t = next.fetch_add(1);
      if (t >= T) break;
      const float c = cpu_tile(targets + t * kSteps, ka, mask, codebook, out_states + t * kSteps,
                               emax, s);
      if (out_cost) out_cost[t] = c;
    }
  };
  if (nthreads == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (int i = 0; i < nthreads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }
  return 0;
}

// ================================================================================================
// HIP build
// ================================================================================================
#elif defined(__HIP__)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kThreads = 256;

constexpr uint32_t kMul1 = 0x83DCD12Du;

// mul1 (docs/trellis.md 4.1) from x = s * 0x83DCD12D: 1024 + bytesum(x) is exact in fp16, and
// (1024 + sum) * fp16(0x1EEE) + fp16(0xC931) is exact in fp32 (both operands are multiples of 2^-18
// below 2^4), so ONE fp32 -> fp16 round-to-nearest-even reproduces EXL3's __hfma bit for bit.
// Callers pass x, not s: along a candidate loop s grows by a constant, so x does too (mod 2^32),
// which turns the quarter-rate v_mul_lo_u32 into an add.
__device__ __forceinline__ float mul1_from_x(uint32_t x) {
  constexpr float kInvF = 0.00676727294921875f;  // fp16 0x1EEE
  constexpr float kBiasF = -10.3828125f;         // fp16 0xC931
  const uint32_t h = __builtin_amdgcn_udot4(x, 0x01010101u, 1024u, false);
  const float v = __builtin_fmaf(static_cast<float>(h), kInvF, kBiasF);
  return __half2float(__float2half_rn(v));
}

template <bool TABLE>
__device__ __forceinline__ float cb_value(uint32_t s, uint32_t x, const float* __restrict__ cbt) {
  if constexpr (TABLE) {
    return cbt[s];
  } else {
    return mul1_from_x(x);
  }
}

// One step on a workgroup. `cost` (LDS) holds the in-edge costs on entry and the out-edge costs on
// exit (in place: every thread stages its predecessors in registers before the first barrier).
template <int A, int B, bool TABLE>
__device__ __forceinline__ void gpu_step(float* cost, uint8_t* __restrict__ bp,
                                         const float* __restrict__ cbt, float w, int tid) {
  constexpr int NA = 1 << A, NB = 1 << B;
  constexpr int GB = 16 - A - B;
  constexpr int G = 1 << GB;
  constexpr uint32_t kHStep = 1u << (16 - B);   // s = s0 + h * kHStep (h is the top B bits)
  constexpr uint32_t kXStep = kHStep * kMul1;   // so x = s * kMul1 = x0 + h * kXStep (mod 2^32)
  constexpr float kInf = __builtin_huge_valf();
  if constexpr (G >= kThreads) {
    constexpr int GPT = G / kThreads;
    float pc[GPT][NB];
#pragma unroll
    for (int gi = 0; gi < GPT; ++gi) {
      const int g = tid + gi * kThreads;
#pragma unroll
      for (int h = 0; h < NB; ++h) pc[gi][h] = cost[(h << GB) | g];
    }
    __syncthreads();
#pragma unroll
    for (int gi = 0; gi < GPT; ++gi) {
      const int g = tid + gi * kThreads;
#pragma unroll 1
      for (int o = 0; o < NA; ++o) {
        const uint32_t s0 = static_cast<uint32_t>((g << A) | o);
        const uint32_t x0 = s0 * kMul1;
        float best = kInf;
        int bh = 0;
#pragma unroll
        for (int h = 0; h < NB; ++h) {
          const uint32_t st = s0 + static_cast<uint32_t>(h) * kHStep;
          const uint32_t x = x0 + static_cast<uint32_t>(h) * kXStep;
          const float d = __fsub_rn(cb_value<TABLE>(st, x, cbt), w);
          const float t = __fadd_rn(pc[gi][h], __fmul_rn(d, d));
          if (t < best) {
            best = t;
            bh = h;
          }
        }
        cost[(g << A) | o] = best;
        bp[(g << A) | o] = static_cast<uint8_t>(bh);
      }
    }
    __syncthreads();
  } else {
    constexpr int TPG = kThreads / G;  // threads per group
    constexpr int OPT = NA / TPG;      // out-edges per thread
    static_assert(OPT >= 1, "too few out-edges per group for 256 threads");
    const int g = tid / TPG, part = tid % TPG;
    float pc[NB];
#pragma unroll
    for (int h = 0; h < NB; ++h) pc[h] = cost[(h << GB) | g];
    __syncthreads();
#pragma unroll 1
    for (int oi = 0; oi < OPT; ++oi) {
      const int o = part * OPT + oi;
      const uint32_t s0 = static_cast<uint32_t>((g << A) | o);
      const uint32_t x0 = s0 * kMul1;
      float best = kInf;
      int bh = 0;
#pragma unroll
      for (int h = 0; h < NB; ++h) {
        const uint32_t st = s0 + static_cast<uint32_t>(h) * kHStep;
        const uint32_t x = x0 + static_cast<uint32_t>(h) * kXStep;
        const float d = __fsub_rn(cb_value<TABLE>(st, x, cbt), w);
        const float t = __fadd_rn(pc[h], __fmul_rn(d, d));
        if (t < best) {
          best = t;
          bh = h;
        }
      }
      cost[(g << A) | o] = best;
      bp[(g << A) | o] = static_cast<uint8_t>(bh);
    }
    __syncthreads();
  }
}

// The step at position p of a rate KA (+0.5 when HALF: odd positions carry KA + 1 bits, the 0xAAAA
// mask) -- one kernel per rate, so each holds only the one or two (a, b) shapes it runs (merging all
// sixteen into one kernel made the register allocator spill).
template <int KA, bool HALF, bool TABLE>
__device__ __forceinline__ void rate_step(int p, float* cost, uint8_t* bp, const float* cbt, float w,
                                          int tid) {
  if constexpr (HALF) {
    if (p & 1) {
      gpu_step<KA + 1, KA, TABLE>(cost, bp, cbt, w, tid);
    } else {
      gpu_step<KA, KA + 1, TABLE>(cost, bp, cbt, w, tid);
    }
  } else {
    gpu_step<KA, KA, TABLE>(cost, bp, cbt, w, tid);
  }
}

template <int KA, bool HALF, bool TABLE>
__global__ __launch_bounds__(kThreads) void viterbi_kernel(
    const float* __restrict__ targets, int64_t T, const float* __restrict__ cbt,
    int32_t* __restrict__ out_states, float* __restrict__ out_cost, uint8_t* __restrict__ scratch,
    int emax) {
  constexpr int ka = KA;
  constexpr unsigned mask = HALF ? 0xAAAAu : 0u;
  extern __shared__ float lds[];
  float* cost = lds;              // emax
  float* xs = lds + emax;         // 256 targets
  float* rcost = xs + kSteps;     // 256 reduction costs
  int* ridx = reinterpret_cast<int*>(rcost + kThreads);  // 256 reduction indices
  __shared__ int s_edge;
  constexpr float kInf = __builtin_huge_valf();
  const int tid = threadIdx.x;
  uint8_t* bpw = scratch + static_cast<size_t>(blockIdx.x) * kSteps * emax;

  for (int64_t tile = blockIdx.x; tile < T; tile += gridDim.x) {
    xs[tid] = targets[tile * kSteps + tid];
    const int nin1 = 1 << (16 - width_of(ka, mask, kRoll));
    for (int i = tid; i < nin1; i += kThreads) cost[i] = 0.0f;
    __syncthreads();

    // ---- pass 1: rolled by 128, free start ----
    for (int i = 0; i < kSteps; ++i) {
      const int p = (i + kRoll) & (kSteps - 1);
      rate_step<KA, HALF, TABLE>(p, cost, bpw + static_cast<size_t>(p) * emax, cbt, xs[p], tid);
    }
    // argmin over position 127's out-edges, smallest edge on ties
    const int nout1 = 1 << (16 - width_of(ka, mask, kRoll));
    float bc = kInf;
    int bi = 0x7fffffff;
    for (int i = tid; i < nout1; i += kThreads) {
      const float c = cost[i];
      if (c < bc || (c == bc && i < bi)) {
        bc = c;
        bi = i;
      }
    }
    rcost[tid] = bc;
    ridx[tid] = bi;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
      if (tid < off) {
        const float c2 = rcost[tid + off];
        const int i2 = ridx[tid + off];
        if (c2 < rcost[tid] || (c2 == rcost[tid] && i2 < ridx[tid])) {
          rcost[tid] = c2;
          ridx[tid] = i2;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      int e = ridx[0] == 0x7fffffff ? 0 : ridx[0];  // all +inf -> edge 0
      for (int p = kRoll - 1; p >= 0; --p) {
        const int a = width_of(ka, mask, p), b = width_of(ka, mask, (p + 1) & (kSteps - 1));
        const int h = bpw[static_cast<size_t>(p) * emax + e];
        e = ((h << (16 - b)) | e) >> a;
      }
      s_edge = e;
    }
    __syncthreads();
    const int end = s_edge;

    // ---- pass 2: position 0's in-edge pinned ----
    const int nin2 = 1 << (16 - width_of(ka, mask, 0));
    for (int i = tid; i < nin2; i += kThreads) cost[i] = (i == end) ? 0.0f : kInf;
    __syncthreads();
    for (int p = 0; p < kSteps; ++p) {
      rate_step<KA, HALF, TABLE>(p, cost, bpw + static_cast<size_t>(p) * emax, cbt, xs[p], tid);
    }
    if (tid == 0) {
      if (out_cost) out_cost[tile] = cost[end];
      int e = end;
      for (int p = kSteps - 1; p >= 0; --p) {
        const int a = width_of(ka, mask, p), b = width_of(ka, mask, (p + 1) & (kSteps - 1));
        const int h = bpw[static_cast<size_t>(p) * emax + e];
        const int st = (h << (16 - b)) | e;
        out_states[tile * kSteps + p] = st;
        e = st >> a;
      }
    }
    __syncthreads();
  }
}

// The HIP build runs K in {3, 4, 5, 6} and K + 0.5 in {3.5, 4.5, 5.5} (mask 0 or 0xAAAA only).
bool widths_ok_hip(int ka, unsigned mask, int* dmin) {
  if (mask != 0u && mask != 0xAAAAu) return false;
  if (ka < 3 || ka > (mask ? 5 : 6)) return false;
  *dmin = ka;
  return true;
}

template <bool TABLE>
hipError_t launch_rate(int ka, bool half, dim3 grid, size_t lds, hipStream_t s, const float* targets,
                       int64_t T, const float* cbt, int32_t* out_states, float* out_cost,
                       uint8_t* scratch, int emax) {
#define R4DX_LAUNCH(KA_, HALF_)                                                   \
  viterbi_kernel<KA_, HALF_, TABLE><<<grid, dim3(kThreads), lds, s>>>(targets, T, cbt, \
                                                                      out_states, out_cost, \
                                                                      scratch, emax)
  switch (ka * 2 + (half ? 1 : 0)) {
    case 6: R4DX_LAUNCH(3, false); break;
    case 7: R4DX_LAUNCH(3, true); break;
    case 8: R4DX_LAUNCH(4, false); break;
    case 9: R4DX_LAUNCH(4, true); break;
    case 10: R4DX_LAUNCH(5, false); break;
    case 11: R4DX_LAUNCH(5, true); break;
    case 12: R4DX_LAUNCH(6, false); break;
    default: return hipErrorInvalidValue;
  }
#undef R4DX_LAUNCH
  return hipGetLastError();
}

}  // namespace

R4DX_EXPORT int r4dx_trellis_abi() { return R4DX_TRELLIS_ABI; }

// Bytes of `scratch` r4dx_trellis_viterbi_hip needs for `grid` workgroups (0 = unsupported widths).
R4DX_EXPORT int64_t r4dx_trellis_scratch_bytes(int ka, unsigned mask, int grid) {
  int dmin = 0;
  if (!widths_ok_hip(ka, mask, &dmin) || grid <= 0) return 0;
  return static_cast<int64_t>(grid) * kSteps * (int64_t{1} << (16 - dmin));
}

// All pointers are device pointers from the caller's (torch's) HIP runtime; `stream` is a
// hipStream_t. cb_mode 0 = mul1 computed on the fly (codebook may be null), 1 = read `codebook`
// (65536 fp32). Returns 0 or a negative error / a positive hipError_t.
R4DX_EXPORT int r4dx_trellis_viterbi_hip(const float* targets, int64_t T, int ka, unsigned mask,
                                         int cb_mode, const float* codebook, int32_t* out_states,
                                         float* out_cost, uint8_t* scratch, int64_t scratch_bytes,
                                         int grid, void* stream) {
  int dmin = 0;
  if (T < 0 || !targets || !out_states || !scratch) return -1;
  if (!widths_ok_hip(ka, mask, &dmin)) return -2;
  if (cb_mode == 1 && !codebook) return -3;
  if (T == 0) return 0;
  const int emax = 1 << (16 - dmin);
  if (grid <= 0) return -4;
  if (static_cast<int64_t>(grid) > T) grid = static_cast<int>(T);
  if (scratch_bytes < static_cast<int64_t>(grid) * kSteps * emax) return -5;
  const size_t lds = sizeof(float) * (static_cast<size_t>(emax) + kSteps + kThreads) +
                     sizeof(int) * kThreads;
  hipStream_t s = static_cast<hipStream_t>(stream);
  const bool half = mask != 0u;
  const hipError_t err =
      cb_mode == 1 ? launch_rate<true>(ka, half, dim3(grid), lds, s, targets, T, codebook,
                                       out_states, out_cost, scratch, emax)
                   : launch_rate<false>(ka, half, dim3(grid), lds, s, targets, T, codebook,
                                        out_states, out_cost, scratch, emax);
  return static_cast<int>(err);
}

#else
#error "build with -DTRELLIS_CPU (plain C++) or as HIP (-x hip)"
#endif
