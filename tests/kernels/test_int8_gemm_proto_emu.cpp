// tests/kernels/test_int8_gemm_proto_emu.cpp -- pure CPU (no HIP call, no device). Runs the int8 prefill GEMM's
// KERNEL SOURCE (libr4d's r4d_trellis_i8.h, the very source of the production unit r4d_gemm_trellis_nt_i8.hip, through
// the bench layer int8_gemm_proto_kernels.h, compiled as plain C++ with I8G_EMU: one OS thread per GPU
// thread, workgroup and wave barriers, shared memory, a software iu8 WMMA, a stub trellis decode; see
// i8g_host_emu.h) against exact references:
//   * i8g_quant_act   vs i8p::QuantizeActRef + PackA8, byte for byte;
//   * i8g_wscale / i8g_dump_w (KB 4 and 5, over the stub decode) vs the CPU table and int8 matrix, byte for byte;
//   * i8g_kernel, every instantiation (dense with the FWHT epilogue and plain, trellis KB 4 and 5; RESC 0..3;
//     SKW 2, 4, 8) for every legal (skw, skg) on K = 1024, N = 128 (one FWHT group, 4 x SKG workgroups racing for
//     the ticket), against the exact-integer fp64 reference (scale mask by RESC) with the epilogue applied in fp64,
//     and the trellis kernel against the dense kernel on the same int8 weights, byte for byte;
//   * two A parts: a launch with a8_0 / a8_1 and an n_split (trellis, KB 4 and 5, several (skw, skg)) against the exact
//     reference per part, byte-identical to two single-part launches, and a part-1 quantizer launch (blockIdx.y = 1).
//   * the fused producers' quantizer (r4d_trellis_i8_fused.h's i8g_quant_wave_block, which the transform kernels of
//     src/kernels/src/trellis_transform.hip call after their own f16 rounding, R4DX_PREFILL_INT8_FUSEDQ): on wide-range f16
//     inputs, one and two parts, K = 1024 and 17408, byte for byte i8g_quant_act's A8 and SA and the CPU quantizer's.
//   * the COARSE scales (R4DX_PREFILL_INT8_SCALES=coarse, RESC 4): i8g_quant_act_row and the producers' shape of it
//     (i8g_quant_row_wg on a row staged in LDS) against the CPU row quantizer; i8g_wscale_col and i8g_dump_w<COARSE> against the
//     CPU column table / int8 matrix; i8g_kernel RESC 4 (trellis KB 4 and 5, every legal (skw, skg)) against the exact coarse
//     reference, byte for byte against its dense twin and against RESC 1 on the same operands replicated per 128, one-hot
//     rows byte-exact, two A parts, and the int32 range at K = 17408 with every operand at +127.
// What this does not cover: the hardware (the iu8 WMMA layout, v_pk_fma_f16, v_perm, DPP: the device bench's selftest)
// and the real trellis decode (shipped, tested elsewhere).
#define I8G_EMU 1
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "int8_gemm_proto_kernels.h"
#include "r4d_trellis_i8_fused.h"   // the fused producers' quantizer: the device function src/kernels/src/trellis_transform.hip calls

namespace {

bool g_ok = true;
int g_checks = 0;
void Check(bool cond, const char* what) {
  ++g_checks;
  if (!cond || std::getenv("I8G_EMU_VERBOSE")) std::printf("%-110s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

float Bf16ToF(unsigned short b) { return __builtin_bit_cast(float, (unsigned)b << 16); }
constexpr float kOutScale = 0.0883883f;
int MaskOfResc(int resc) { static const int m[5] = {0, 3, 1, 2, 3}; return m[resc]; }

// the emulated launch of i8g_kernel for a configuration (the dispatch of I8gRun, over emu::launch)
// (a1 / sa1 / n_split: the second A part; a single part passes the first twice and n_split = N)
template <bool TR, int KB, bool FW, int RE, int SKW>
void LaunchT(const I8gCfg& c, const signed char* a8, const float* sa, const signed char* a8b, const float* sab, int n_split, const void* w,
             const float* sw, const float* svh, unsigned short* C, float* ws, unsigned* tk, int K, int N, float os) {
  emu::launch((unsigned)(N / 32), (unsigned)c.skg, 128u * SKW, i8g_kernel<TR, KB, FW, RE, SKW>, a8, sa, a8b, sab, n_split,
              (const unsigned char*)w, sw, svh, C, ws, tk, K, N, c.skg, os);
}
template <bool TR, int KB, bool FW, int RE>
void LaunchSkw(const I8gCfg& c, const signed char* a8, const float* sa, const signed char* a8b, const float* sab, int n_split, const void* w,
               const float* sw, const float* svh, unsigned short* C, float* ws, unsigned* tk, int K, int N, float os) {
  switch (c.skw) {
    case 2: LaunchT<TR, KB, FW, RE, 2>(c, a8, sa, a8b, sab, n_split, w, sw, svh, C, ws, tk, K, N, os); break;
    case 4: LaunchT<TR, KB, FW, RE, 4>(c, a8, sa, a8b, sab, n_split, w, sw, svh, C, ws, tk, K, N, os); break;
    default: LaunchT<TR, KB, FW, RE, 8>(c, a8, sa, a8b, sab, n_split, w, sw, svh, C, ws, tk, K, N, os); break;
  }
}
void RunParts(const I8gCfg& c, const signed char* a8, const float* sa, const signed char* a8b, const float* sab, int n_split, const void* w,
              const float* sw, const float* svh, unsigned short* C, float* ws, unsigned* tk, int K, int N, float os) {
#define CASE(TR, KB, FW, RE) LaunchSkw<TR, KB, FW, RE>(c, a8, sa, a8b, sab, n_split, w, sw, svh, C, ws, tk, K, N, os)
#define RESC4(TR, KB, FW) \
  switch (c.resc) { case 0: CASE(TR, KB, FW, 0); break; case 1: CASE(TR, KB, FW, 1); break; case 2: CASE(TR, KB, FW, 2); break; \
                    case 3: CASE(TR, KB, FW, 3); break; default: CASE(TR, KB, FW, 4); break; }
  if (!c.trellis) {
    if (c.fwht) { if (c.resc == 0) CASE(false, 4, true, 0); else if (c.resc == 1) CASE(false, 4, true, 1); else CASE(false, 4, true, 4); }
    else { if (c.resc == 0) CASE(false, 4, false, 0); else if (c.resc == 1) CASE(false, 4, false, 1); else CASE(false, 4, false, 4); }
  } else if (c.kb == 4) {
    RESC4(true, 4, true);
  } else {
    RESC4(true, 5, true);
  }
#undef RESC4
#undef CASE
}
void Run(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw, const float* svh, unsigned short* C,
         float* ws, unsigned* tk, int K, int N, float os) {
  RunParts(c, a8, sa, a8, sa, N, w, sw, svh, C, ws, tk, K, N, os);
}

struct Problem {
  int K, N, KB;
  std::vector<uint16_t> xh;            // f16 activations [256][K]
  std::vector<float> X;
  std::vector<unsigned> grid;          // trellis words (the stub decode reads them), (N/32) * (K/16) blocks of 16 KB words
  std::vector<int8_t> Ap, A8, Wp, W8;
  std::vector<float> sa, sw, svh;
  std::vector<double> raw[4];          // exact reference by scale mask
  // the COARSE operands (BuildCoarse): one scale per row / per column over the whole K
  std::vector<int8_t> Apc, A8c, Wpc, W8c;
  std::vector<float> sac, swc;
  std::vector<double> rawc;            // the exact reference of the coarse math
};

// the decoded weights Q [K][N] (f16 bits) of the problem's words, through the kernel's own decode helper (the stub here)
std::vector<uint16_t> DecodeQ(const Problem& p) {
  const int K = p.K, N = p.N, KB = p.KB, KT = K / 16;
  std::vector<uint16_t> Q((size_t)K * N);
  for (int pair = 0; pair < N / 32; ++pair)
    for (int kt = 0; kt < KT; ++kt)
      for (int lane = 0; lane < 32; ++lane) {
        v8h f0, f1;
        if (KB == 4) i8g_decode_block<4>(&p.grid[((size_t)pair * KT + kt) * 16 * KB], lane, f0, f1);
        else i8g_decode_block<5>(&p.grid[((size_t)pair * KT + kt) * 16 * KB], lane, f0, f1);
        // whole-vector casts: clang reads element 0 for __builtin_bit_cast(unsigned short, f[e]) (see r4d_trellis_dq.h)
        typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
        const u16x8 q0 = __builtin_bit_cast(u16x8, f0), q1 = __builtin_bit_cast(u16x8, f1);
        for (int e = 0; e < 8; ++e) {
          const size_t k = (size_t)kt * 16 + i8p::FragK16(lane, e);
          Q[k * N + i8p::FragCol(pair, lane, 0)] = q0[e];
          Q[k * N + i8p::FragCol(pair, lane, 1)] = q1[e];
        }
      }
  return Q;
}

// the coarse CPU operands of a problem (its activations p.X as they are now, its words) and the exact reference
void BuildCoarse(Problem& p) {
  i8p::QuantizeActRowRef(p.X.data(), p.K, p.sac, p.Apc);
  p.A8c = i8p::PackA8(p.Apc, p.K);
  const std::vector<uint16_t> Q = DecodeQ(p);
  i8p::QuantizeWeightsColRef(Q.data(), p.K, p.N, p.swc, p.Wpc);
  p.W8c = i8p::PackW8(p.Wpc, p.K, p.N);
  p.rawc.resize((size_t)256 * p.N);
  i8p::RefGemmInt8(p.Apc.data(), p.sac.data(), p.Wpc.data(), p.swc.data(), 256, p.N, p.K, 3, p.rawc.data());
}

void BuildProblem(Problem& p, int K, int N, int KB, uint32_t seed) {
  p.K = K; p.N = N; p.KB = KB;
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 0.8f);
  p.xh.resize((size_t)256 * K);
  p.X.resize(p.xh.size());
  for (size_t i = 0; i < p.xh.size(); ++i) {
    p.xh[i] = i8p::F32ToF16(nd(rng));
    p.X[i] = i8p::F16ToF32(p.xh[i]);
  }
  for (int k = 0; k < 128; ++k) { p.xh[(size_t)7 * K + 128 + k] = 0; p.X[(size_t)7 * K + 128 + k] = 0.f; }   // an all-zero block
  p.grid.resize((size_t)(N / 32) * (K / 16) * 16 * KB);
  for (unsigned& w : p.grid) w = rng();
  p.svh.resize(N);
  for (int n = 0; n < N; ++n) p.svh[n] = (rng() & 1 ? -1.f : 1.f) * (0.5f + (rng() % 1000) / 1000.f);
  // CPU quantized operands: the stub decode through the kernel's own decode helper, then the CPU quantizers
  i8p::QuantizeActRef(p.X.data(), K, p.sa, p.Ap);
  p.A8 = i8p::PackA8(p.Ap, K);
  const std::vector<uint16_t> Q = DecodeQ(p);
  i8p::QuantizeWeightsRef(Q.data(), K, N, p.sw, p.Wp);
  p.W8 = i8p::PackW8(p.Wp, K, N);
  for (int mask = 0; mask < 4; ++mask) {
    p.raw[mask].resize((size_t)256 * N);
    i8p::RefGemmInt8(p.Ap.data(), p.sa.data(), p.Wp.data(), p.sw.data(), 256, N, K, mask, p.raw[mask].data());
  }
}

// the 128-column groups g0 .. g1 - 1 (default: all) of C against the reference
long long CountViolations(const Problem& p, const std::vector<unsigned short>& C, bool fwht, int resc, float os, int g0 = 0,
                          int g1 = -1) {
  long long viol = 0;
  if (g1 < 0) g1 = p.N / 128;
  for (int r = 0; r < 256; ++r)
    for (int g = g0; g < g1; ++g) {
      double v[128], e[128], rms = 0;
      const std::vector<double>& rawv = resc == 4 ? p.rawc : p.raw[MaskOfResc(resc)];   // RESC 4: the coarse operands' reference
      for (int j = 0; j < 128; ++j) v[j] = rawv[(size_t)r * p.N + g * 128 + j];
      if (fwht) i8p::Fwht128(v);
      for (int j = 0; j < 128; ++j) {
        e[j] = fwht ? v[j] * (double)p.svh[g * 128 + j] * (double)os : v[j] * (double)os;
        rms += e[j] * e[j];
      }
      rms = std::sqrt(rms / 128);
      for (int j = 0; j < 128; ++j) {
        const double got = Bf16ToF(C[(size_t)r * p.N + g * 128 + j]);
        viol += std::fabs(got - e[j]) > 0.0085 * std::fabs(e[j]) + 1e-4 * rms;
      }
    }
  return viol;
}

template <class T>
bool Same(const std::vector<T>& a, const T* b, size_t n) { return a.size() == n && std::memcmp(a.data(), b, n * sizeof(T)) == 0; }

// the same weights with other activations (a second A part): re-quantize and re-derive the exact references
void ReplaceActivations(Problem& p, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.3f);
  for (size_t i = 0; i < p.xh.size(); ++i) {
    p.xh[i] = i8p::F32ToF16(nd(rng));
    p.X[i] = i8p::F16ToF32(p.xh[i]);
  }
  i8p::QuantizeActRef(p.X.data(), p.K, p.sa, p.Ap);
  p.A8 = i8p::PackA8(p.Ap, p.K);
  for (int mask = 0; mask < 4; ++mask)
    i8p::RefGemmInt8(p.Ap.data(), p.sa.data(), p.Wp.data(), p.sw.data(), 256, p.N, p.K, mask, p.raw[mask].data());
}

// fp32 FWHT-128 in the kernel's order (stage lg = 0..6, pair (i, i + 2^lg) -> (a + b, a - b)); trellis_ref::FwhtLdsF32's
// twin, which the production GPU test (test_trellis_i8_gemm.cpp, check C) predicts the kernel's bytes with
void Fwht128F32(float* v) {
#pragma clang fp contract(off)
  for (int lg = 0; lg < 7; ++lg) {
    const int h = 1 << lg;
    for (int pp = 0; pp < 64; ++pp) {
      const int i = ((pp >> lg) << (lg + 1)) | (pp & (h - 1));
      const float a = v[i], b = v[i + h];
      v[i] = a + b;
      v[i + h] = a - b;
    }
  }
}

// The fused producers' quantizer (src/kernels/src/trellis_transform.hip, R4DX_PREFILL_INT8_FUSEDQ) as a kernel of its
// own, the shape of a producer after its transform: grid (K / 128, 256), a wave32 per (row, 128-block), the lane's
// f16 values the block's elements lane + 32 r. The production kernels call the same i8g_quant_wave_block.
void FusedQuantKernel(const unsigned short* X, signed char* A8, float* SA, int K) {
  const int lane = (int)(threadIdx.x & 31u);
  const int kb = (int)blockIdx.x, row = (int)blockIdx.y;
  unsigned short hb[4];
  for (int r = 0; r < 4; ++r) hb[r] = X[(size_t)row * K + kb * 128 + lane + 32 * r];
  i8g_quant_wave_block(hb, lane, row, kb, K, A8, SA);
}

// The coarse producers' shape (src/kernels/src/trellis_transform.hip's r4dx_*_i8r after their transform): the row's f16 bits
// staged in the workgroup's shared memory by the waves, a barrier, then i8g_quant_row_wg. Grid (256 rows), 256 threads; the
// stand-alone kernel i8g_quant_act_row reads the row from global memory and must give the same bytes.
void RowLdsKernel(const unsigned short* X, signed char* A8, float* SA, int K) {
  I8G_LDS_DECL
  unsigned short* xl = (unsigned short*)lds;
  float* red = (float*)(lds + (size_t)K * 2);
  const int row = (int)blockIdx.x;
  for (int i = (int)threadIdx.x; i < K; i += 256) xl[i] = X[(size_t)row * K + i];
  __syncthreads();
  i8g_quant_row_wg(xl, K, row, A8, SA, red);
}

// random f16 values with a spread of magnitudes per 128-block (the transform's output spans the f16 range: subnormals
// to near the maximum), some blocks all zero, some with a single nonzero element, some with ties at the rounding midpoint
std::vector<unsigned short> WideRangeF16(int K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<unsigned short> x((size_t)256 * K);
  for (int r = 0; r < 256; ++r)
    for (int kb = 0; kb < K / 128; ++kb) {
      const int kind = (int)((r * 7 + kb * 13 + seed) % 11);
      const float mag = std::ldexp(1.0f, (int)(rng() % 31) - 20);   // 2^-20 .. 2^10
      for (int k = 0; k < 128; ++k) {
        float v = nd(rng) * mag;
        if (kind == 0) v = 0.f;
        if (kind == 1) v = k == 77 ? mag * 3.0f : 0.f;
        if (kind == 2) v = (float)((int)(rng() % 255) - 127) * 0.5f * mag + 0.5f * mag / 127.0f * (float)(k & 1);   // near midpoints
        if (kind == 3 && k == 5) v = 60000.f;
        x[(size_t)r * K + kb * 128 + k] = i8p::F32ToF16(v);
      }
    }
  return x;
}

}  // namespace

int main() {
  const auto t0 = std::chrono::steady_clock::now();
  const int K = 1024, N = 128, NKB = K / 128;
  std::vector<unsigned short> C((size_t)256 * N), Cd((size_t)256 * N);
  std::vector<float> ws((size_t)8 * 256 * N);
  std::vector<unsigned> tk(N / 128, 0u);
  std::vector<std::vector<unsigned short>> dense_out;   // keyed by (skw, skg, fwht, resc) below via a map of strings
  for (int KB : {4, 5}) {
    Problem p;
    BuildProblem(p, K, N, KB, 100 + KB);
    char what[200];
    // producers
    std::vector<signed char> a8(p.A8.size(), 0), ap(p.Ap.size(), 0);
    std::vector<float> sa(p.sa.size(), 0.f);
    emu::launch((unsigned)((256 * NKB + 7) / 8), 1, 256, i8g_quant_act, p.xh.data(), a8.data(), ap.data(), sa.data(), K, 0LL);
    std::snprintf(what, sizeof what, "KB %d: i8g_quant_act (emulated kernel) == CPU quantizer: A8 layout, plain A, scales (incl. an all-zero block)", KB);
    Check(std::memcmp(a8.data(), p.A8.data(), a8.size()) == 0 && std::memcmp(ap.data(), p.Ap.data(), ap.size()) == 0 &&
              std::memcmp(sa.data(), p.sa.data(), sa.size() * 4) == 0, what);
    {
      // the fused producers' device function (r4d_trellis_i8_fused.h) on the same f16 A: byte for byte i8g_quant_act's
      std::vector<signed char> a8f(p.A8.size(), 0x55);
      std::vector<float> saf(p.sa.size(), -1.f);
      emu::launch((unsigned)NKB, 256, 32, FusedQuantKernel, p.xh.data(), a8f.data(), saf.data(), K);
      std::snprintf(what, sizeof what, "KB %d: the fused producers' i8g_quant_wave_block == i8g_quant_act byte for byte (A8 layout, scales, all-zero block)", KB);
      Check(std::memcmp(a8f.data(), a8.data(), a8.size()) == 0 && std::memcmp(saf.data(), sa.data(), sa.size() * 4) == 0, what);
    }
    std::vector<float> sw(p.sw.size(), 0.f);
    std::vector<signed char> w8(p.W8.size(), 0), wp(p.Wp.size(), 0);
    const unsigned groups = (unsigned)((N / 32) * NKB), blocks = (unsigned)((N / 32) * (K / 16));
    if (KB == 4) {
      emu::launch((groups + 7) / 8, 1, 256, i8g_wscale<4>, (const unsigned*)p.grid.data(), sw.data(), K, N);
      emu::launch((blocks + 7) / 8, 1, 256, i8g_dump_w<4>, (const unsigned*)p.grid.data(), (const float*)sw.data(), w8.data(), wp.data(), K, N);
    } else {
      emu::launch((groups + 7) / 8, 1, 256, i8g_wscale<5>, (const unsigned*)p.grid.data(), sw.data(), K, N);
      emu::launch((blocks + 7) / 8, 1, 256, i8g_dump_w<5>, (const unsigned*)p.grid.data(), (const float*)sw.data(), w8.data(), wp.data(), K, N);
    }
    std::snprintf(what, sizeof what, "KB %d: i8g_wscale + i8g_dump_w (emulated, stub decode) == CPU table and int8 matrix (W8 layout and plain)", KB);
    {
      const bool s_ok = std::memcmp(sw.data(), p.sw.data(), sw.size() * 4) == 0, w_ok = std::memcmp(w8.data(), p.W8.data(), w8.size()) == 0,
                 p_ok = std::memcmp(wp.data(), p.Wp.data(), wp.size()) == 0;
      if (!(s_ok && w_ok && p_ok)) {
        size_t ns = 0, nw = 0, np = 0, first_s = (size_t)-1;
        for (size_t i = 0; i < sw.size(); ++i) if (sw[i] != p.sw[i]) { if (first_s == (size_t)-1) first_s = i; ++ns; }
        for (size_t i = 0; i < w8.size(); ++i) nw += w8[i] != p.W8[i];
        for (size_t i = 0; i < wp.size(); ++i) np += wp[i] != p.Wp[i];
        std::printf("    table differs in %zu of %zu entries (first %zu: device-side %.8g, CPU %.8g); W8 %zu of %zu bytes; plain %zu of %zu bytes\n", ns,
                    sw.size(), first_s, first_s == (size_t)-1 ? 0.0 : (double)sw[first_s], first_s == (size_t)-1 ? 0.0 : (double)p.sw[first_s], nw, w8.size(), np, wp.size());
      }
      Check(s_ok && w_ok && p_ok, what);
    }

    // the GEMMs
    int ran = 0, bad = 0, bitdiff = 0;
    for (int kind = 0; kind < 3; ++kind) {   // 0 dense + fwht, 1 dense plain, 2 trellis
      if (KB == 5 && kind < 2) continue;     // the dense kernel does not depend on KB: once
      for (int resc = 0; resc < (kind == 2 ? 4 : 2); ++resc)
        for (int skw : {2, 4, 8})
          for (int skg : {1, 2, 4, 8}) {
            I8gCfg c;
            c.trellis = kind == 2; c.kb = KB; c.fwht = kind != 1; c.resc = resc; c.skw = skw; c.skg = skg;
            if (I8gCheck(c, K, N)) continue;
            const float os = c.fwht ? kOutScale : 1.0f;
            std::fill(C.begin(), C.end(), (unsigned short)0xFFFF);
            std::fill(tk.begin(), tk.end(), 0u);
            Run(c, a8.data(), sa.data(), c.trellis ? (const void*)p.grid.data() : (const void*)w8.data(), sw.data(), p.svh.data(), C.data(),
                ws.data(), tk.data(), K, N, os);
            ++ran;
            const long long viol = CountViolations(p, C, c.fwht, resc, os);
            if (viol) {
              ++bad;
              std::printf("    FAIL %s kb%d skw%d skg%d %s resc%d: %lld of %d outside tolerance\n", c.trellis ? "trellis" : "dense", KB, skw, skg,
                          c.fwht ? "fwht" : "plain", resc, viol, 256 * N);
            }
            bool tk_reset = true;
            for (unsigned x : tk) tk_reset = tk_reset && x == 0;
            if (!tk_reset) { ++bad; std::printf("    FAIL tickets not reset: kb%d skw%d skg%d resc%d\n", KB, skw, skg, resc); }
            if (kind == 2 && resc <= 1) {
              I8gCfg d = c;
              d.trellis = false;
              std::fill(Cd.begin(), Cd.end(), (unsigned short)0xFFFF);
              Run(d, a8.data(), sa.data(), w8.data(), sw.data(), p.svh.data(), Cd.data(), ws.data(), tk.data(), K, N, os);
              if (Cd != C) { ++bitdiff; std::printf("    trellis != dense: kb%d skw%d skg%d resc%d\n", KB, skw, skg, resc); }
            }
          }
    }
    std::snprintf(what, sizeof what, "KB %d: %d emulated kernel runs (skw 2/4/8 x skg, RESC 0..3, FWHT / plain) match the exact reference, tickets reset", KB, ran);
    Check(bad == 0 && ran > 0, what);
    std::snprintf(what, sizeof what, "KB %d: the trellis kernel (stub decode + quantizer) == the dense kernel on the same int8 weights, byte for byte", KB);
    Check(bitdiff == 0, what);
  }
  // ---- two A parts (n_split): the production kernel's second activation pair ----
  for (int KB : {4, 5}) {
    const int N2 = 256;
    Problem p0, p1;
    BuildProblem(p0, K, N2, KB, 300 + KB);
    p1 = p0;
    ReplaceActivations(p1, 777 + KB);
    char what[240];
    // the part-1 quantizer launch: both parts in one launch (x at x0 and x0 + 256 K elements, outputs one after the other)
    std::vector<uint16_t> xcat(p0.xh);
    xcat.insert(xcat.end(), p1.xh.begin(), p1.xh.end());
    std::vector<signed char> a8cat(2 * p0.A8.size(), 0), apcat(2 * p0.Ap.size(), 0);
    std::vector<float> sacat(2 * p0.sa.size(), 0.f);
    emu::launch((unsigned)((256 * NKB + 7) / 8), 2, 256, i8g_quant_act, xcat.data(), a8cat.data(), apcat.data(), sacat.data(), K,
                (long long)256 * K);
    std::snprintf(what, sizeof what, "KB %d: i8g_quant_act with two parts (blockIdx.y) == the CPU quantizer of each part", KB);
    Check(std::memcmp(a8cat.data(), p0.A8.data(), p0.A8.size()) == 0 &&
              std::memcmp(a8cat.data() + p0.A8.size(), p1.A8.data(), p1.A8.size()) == 0 &&
              std::memcmp(apcat.data() + p0.Ap.size(), p1.Ap.data(), p1.Ap.size()) == 0 &&
              std::memcmp(sacat.data(), p0.sa.data(), p0.sa.size() * 4) == 0 &&
              std::memcmp(sacat.data() + p0.sa.size(), p1.sa.data(), p1.sa.size() * 4) == 0,
          what);
    const signed char *a0 = a8cat.data(), *a1 = a8cat.data() + p0.A8.size();
    const float *s0 = sacat.data(), *s1 = sacat.data() + p0.sa.size();
    std::vector<unsigned short> Cp((size_t)256 * N2), Ca((size_t)256 * N2), Cb((size_t)256 * N2);
    std::vector<float> ws2((size_t)8 * 256 * N2);
    std::vector<unsigned> tk2(N2 / 128, 0u);
    int ran = 0, bad = 0;
    for (int skw : {2, 4, 8})
      for (int skg : {1, 2, 4}) {
        I8gCfg c;
        c.trellis = true; c.kb = KB; c.fwht = true; c.resc = 0; c.skw = skw; c.skg = skg;
        if (I8gCheck(c, K, N2)) continue;
        std::fill(Cp.begin(), Cp.end(), (unsigned short)0xFFFF);
        RunParts(c, a0, s0, a1, s1, 128, p0.grid.data(), p0.sw.data(), p0.svh.data(), Cp.data(), ws2.data(), tk2.data(), K, N2,
                 kOutScale);
        // each group against the exact reference of its own part's activations
        const long long v0 = CountViolations(p0, Cp, true, 0, kOutScale, 0, 1);
        const long long v1 = CountViolations(p1, Cp, true, 0, kOutScale, 1, 2);
        // and byte for byte against the two single-part launches
        RunParts(c, a0, s0, a0, s0, N2, p0.grid.data(), p0.sw.data(), p0.svh.data(), Ca.data(), ws2.data(), tk2.data(), K, N2,
                 kOutScale);
        RunParts(c, a1, s1, a1, s1, N2, p0.grid.data(), p0.sw.data(), p0.svh.data(), Cb.data(), ws2.data(), tk2.data(), K, N2,
                 kOutScale);
        bool same = true;
        for (int r = 0; r < 256; ++r)
          for (int n = 0; n < N2; ++n) same = same && Cp[(size_t)r * N2 + n] == (n < 128 ? Ca : Cb)[(size_t)r * N2 + n];
        bool tk_reset = true;
        for (unsigned x : tk2) tk_reset = tk_reset && x == 0;
        ++ran;
        if (v0 || v1 || !same || !tk_reset) {
          ++bad;
          std::printf("    FAIL two parts kb%d skw%d skg%d: part 0 %lld, part 1 %lld outside tolerance, bytes %s single-part launches, "
                      "tickets %s\n", KB, skw, skg, v0, v1, same ? "==" : "!=", tk_reset ? "reset" : "NOT reset");
        }
      }
    std::snprintf(what, sizeof what,
                  "KB %d: %d two-part launches (n_split 128 of 256) match each part's exact reference and the two single-part "
                  "launches byte for byte", KB, ran);
    Check(bad == 0 && ran > 0, what);
  }
  // ---- one-hot rows: the whole chain byte-exact (the production GPU test's check C, validated here on the source) ----
  // Every row has a single nonzero activation, so each int32 partial sum is one product and the fp32 chain has no
  // rounding-order freedom: the output must equal the CPU's fp32 emulation of the epilogue byte for byte.
  for (int KB : {4, 5}) {
    Problem p;
    BuildProblem(p, K, N, KB, 500 + KB);
    std::mt19937 r2(900 + KB);
    std::vector<int> kk(256);
    for (int r = 0; r < 256; ++r) {
      std::fill(p.xh.begin() + (size_t)r * K, p.xh.begin() + (size_t)(r + 1) * K, (uint16_t)0);
      kk[r] = r * (K / 256) + (int)(r2() % (unsigned)(K / 256));
      const float m = 0.25f + (float)(r2() % 4000u) / 1000.f;
      p.xh[(size_t)r * K + kk[r]] = i8p::F32ToF16((r2() & 1u) ? -m : m);
    }
    for (size_t i = 0; i < p.xh.size(); ++i) p.X[i] = i8p::F16ToF32(p.xh[i]);
    i8p::QuantizeActRef(p.X.data(), K, p.sa, p.Ap);
    p.A8 = i8p::PackA8(p.Ap, K);
    // expected bytes
    std::vector<unsigned short> expect((size_t)256 * N);
    for (int r = 0; r < 256; ++r) {
      float v[128];
      const int k = kk[r], kb = k / 128;
      const float sa = p.sa[(size_t)kb * 256 + r];
      for (int j = 0; j < 128; ++j) {
        const int t = (int)p.Ap[(size_t)r * K + k] * (int)p.Wp[(size_t)j * K + k];
        const float prod = sa * p.sw[(size_t)kb * N + j];
        v[j] = (float)t * prod;
      }
      Fwht128F32(v);
      for (int j = 0; j < 128; ++j) expect[(size_t)r * N + j] = i8g_bf16_rn((v[j] * p.svh[j]) * kOutScale);
    }
    size_t bad = 0, runs = 0;
    for (int skw : {2, 4, 8})
      for (int skg : {1, 2, 4, 8}) {
        I8gCfg c;
        c.trellis = true; c.kb = KB; c.fwht = true; c.resc = 0; c.skw = skw; c.skg = skg;
        if (I8gCheck(c, K, N)) continue;
        std::fill(C.begin(), C.end(), (unsigned short)0xFFFF);
        Run(c, p.A8.data(), p.sa.data(), p.grid.data(), p.sw.data(), p.svh.data(), C.data(), ws.data(), tk.data(), K, N, kOutScale);
        ++runs;
        for (size_t i = 0; i < C.size(); ++i) bad += C[i] != expect[i];
      }
    char what[200];
    std::snprintf(what, sizeof what, "KB %d: %zu one-hot launches byte-exact vs the fp32 emulation of the epilogue (%zu bytes differ)", KB, runs, bad);
    Check(bad == 0 && runs > 0, what);
  }
  // ---- the fused producers' quantizer against the separate one: wide-range inputs, two parts, K = 17408 ----
  // R4DX_PREFILL_INT8_FUSEDQ promises the GEMM's operands are bit for bit the unfused chain's. A producer writes part p
  // at a8 + p * 256 K and sa + p * (K / 128) * 256 (a launch per output, as r4dx_trellis_input_i8 does it); the
  // separate quantizer is one launch over both parts (blockIdx.y). Also against the CPU quantizer.
  for (int Kq : {1024, 17408}) {
    const int parts = Kq == 1024 ? 2 : 1, nkb = Kq / 128;
    char what[240];
    std::vector<unsigned short> x;
    for (int p = 0; p < parts; ++p) {
      const std::vector<unsigned short> xp = WideRangeF16(Kq, 4242 + 17 * Kq + p);
      x.insert(x.end(), xp.begin(), xp.end());
    }
    const size_t a8n = (size_t)256 * Kq, san = (size_t)nkb * 256;
    std::vector<signed char> a8u(parts * a8n, 0x55), a8f(parts * a8n, 0x2A), apu(parts * a8n, 0);
    std::vector<float> sau(parts * san, -1.f), saf(parts * san, -2.f);
    emu::launch((unsigned)((256 * nkb + 7) / 8), (unsigned)parts, 256, i8g_quant_act, x.data(), a8u.data(), apu.data(), sau.data(), Kq,
                (long long)(parts > 1 ? a8n : 0));
    for (int p = 0; p < parts; ++p)
      emu::launch((unsigned)nkb, 256, 32, FusedQuantKernel, x.data() + p * a8n, a8f.data() + p * a8n, saf.data() + p * san, Kq);
    std::snprintf(what, sizeof what, "K %d, %d part(s), wide range: the fused producers' A8 and SA == i8g_quant_act's, byte for byte", Kq, parts);
    Check(std::memcmp(a8u.data(), a8f.data(), a8u.size()) == 0 && std::memcmp(sau.data(), saf.data(), sau.size() * 4) == 0, what);
    bool cpu_same = true;
    for (int p = 0; p < parts; ++p) {
      std::vector<float> X(a8n);
      for (size_t i = 0; i < a8n; ++i) X[i] = i8p::F16ToF32(x[p * a8n + i]);
      std::vector<float> sac;
      std::vector<signed char> apc;
      i8p::QuantizeActRef(X.data(), Kq, sac, apc);
      const std::vector<signed char> a8c = i8p::PackA8(apc, Kq);
      cpu_same = cpu_same && std::memcmp(a8c.data(), a8f.data() + p * a8n, a8n) == 0 && std::memcmp(sac.data(), saf.data() + p * san, san * 4) == 0;
    }
    std::snprintf(what, sizeof what, "K %d, %d part(s), wide range: the fused producers' output == the CPU quantizer (QuantizeActRef + PackA8)", Kq, parts);
    Check(cpu_same, what);
  }
  // ---- the COARSE scales (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse scales"): RESC 4 ----
  // One activation scale per row and one weight scale per column over the whole K. The kernel source is the very file the
  // production unit builds; the producers of src/kernels/src/trellis_transform.hip call the same i8g_quant_row_wg.
  {
    char what[260];
    // (1) the activation quantizer: the stand-alone kernel and the producer-shaped one (row staged in LDS), wide-range f16 with
    // zero rows and a one-nonzero row, one and two parts, K = 1024 / 17408, against the CPU quantizer byte for byte
    for (int Kq : {1024, 17408}) {
      const int parts = Kq == 1024 ? 2 : 1;
      std::vector<unsigned short> x;
      for (int p = 0; p < parts; ++p) {
        std::vector<unsigned short> xp = WideRangeF16(Kq, 6161 + 13 * Kq + p);
        std::fill(xp.begin() + (size_t)3 * Kq, xp.begin() + (size_t)4 * Kq, (unsigned short)0);         // an all-zero row
        std::fill(xp.begin() + (size_t)200 * Kq, xp.begin() + (size_t)201 * Kq, (unsigned short)0);
        xp[(size_t)200 * Kq + Kq / 2 + 3] = i8p::F32ToF16(-0.75f);                                       // a one-nonzero row
        x.insert(x.end(), xp.begin(), xp.end());
      }
      const size_t a8n = (size_t)256 * Kq;
      std::vector<signed char> a8u(parts * a8n, 0x55), a8l(parts * a8n, 0x2A);
      std::vector<float> sau(parts * 256, -1.f), sal(parts * 256, -2.f);
      emu::launch(256, (unsigned)parts, 256, i8g_quant_act_row, x.data(), a8u.data(), sau.data(), Kq, (long long)(parts > 1 ? a8n : 0));
      for (int p = 0; p < parts; ++p)
        emu::launch(256, 1, 256, RowLdsKernel, x.data() + p * a8n, a8l.data() + p * a8n, sal.data() + p * 256, Kq);
      bool cpu_same = true;
      for (int p = 0; p < parts; ++p) {
        std::vector<float> X(a8n);
        for (size_t i = 0; i < a8n; ++i) X[i] = i8p::F16ToF32(x[p * a8n + i]);
        std::vector<float> sac;
        std::vector<signed char> apc;
        i8p::QuantizeActRowRef(X.data(), Kq, sac, apc);
        const std::vector<signed char> a8c = i8p::PackA8(apc, Kq);
        cpu_same = cpu_same && std::memcmp(a8c.data(), a8u.data() + p * a8n, a8n) == 0 && std::memcmp(sac.data(), sau.data() + p * 256, 256 * 4) == 0;
      }
      std::snprintf(what, sizeof what, "COARSE K %d, %d part(s), wide range + zero rows: i8g_quant_act_row == the CPU row quantizer (QuantizeActRowRef + PackA8), A8 and SA", Kq, parts);
      Check(cpu_same, what);
      std::snprintf(what, sizeof what, "COARSE K %d: the producers' form (row staged in LDS, i8g_quant_row_wg) == the stand-alone kernel byte for byte", Kq);
      Check(std::memcmp(a8u.data(), a8l.data(), a8u.size()) == 0 && std::memcmp(sau.data(), sal.data(), sau.size() * 4) == 0, what);
    }
    // (2..5) per KB
    for (int KB : {4, 5}) {
      Problem p;
      BuildProblem(p, K, N, KB, 700 + KB);
      BuildCoarse(p);
      // (2) the column table and the int8 matrix from the words
      std::vector<float> swc(p.swc.size(), 0.f);
      std::vector<signed char> w8c(p.W8c.size(), 0), wpc(p.Wpc.size(), 0);
      const unsigned blocks = (unsigned)((N / 32) * (K / 16));
      if (KB == 4) {
        emu::launch((unsigned)(N / 32), 1, 256, i8g_wscale_col<4>, (const unsigned*)p.grid.data(), swc.data(), K, N);
        emu::launch((blocks + 7) / 8, 1, 256, i8g_dump_w<4, true>, (const unsigned*)p.grid.data(), (const float*)swc.data(), w8c.data(), wpc.data(), K, N);
      } else {
        emu::launch((unsigned)(N / 32), 1, 256, i8g_wscale_col<5>, (const unsigned*)p.grid.data(), swc.data(), K, N);
        emu::launch((blocks + 7) / 8, 1, 256, i8g_dump_w<5, true>, (const unsigned*)p.grid.data(), (const float*)swc.data(), w8c.data(), wpc.data(), K, N);
      }
      std::snprintf(what, sizeof what, "COARSE KB %d: i8g_wscale_col + i8g_dump_w<COARSE> (emulated, stub decode) == CPU column table and int8 matrix (W8 and plain)", KB);
      Check(std::memcmp(swc.data(), p.swc.data(), swc.size() * 4) == 0 && std::memcmp(w8c.data(), p.W8c.data(), w8c.size()) == 0 &&
                std::memcmp(wpc.data(), p.Wpc.data(), wpc.size()) == 0, what);
      // the activation operand through the emulated row quantizer
      std::vector<signed char> a8c(p.A8c.size(), 0);
      std::vector<float> sac(256, 0.f);
      emu::launch(256, 1, 256, i8g_quant_act_row, p.xh.data(), a8c.data(), sac.data(), K, 0LL);
      std::snprintf(what, sizeof what, "COARSE KB %d: i8g_quant_act_row (emulated) == the CPU row quantizer on the standard problem", KB);
      Check(std::memcmp(a8c.data(), p.A8c.data(), a8c.size()) == 0 && std::memcmp(sac.data(), p.sac.data(), 256 * 4) == 0, what);

      // (3) the GEMM: every legal (skw, skg), trellis RESC 4 vs the exact coarse reference, tickets back at zero, the dense twin
      // byte for byte, and RESC 1 on the same operands replicated per 128 (RESC 4 is RESC 1's math, so the bytes are equal)
      const int NKB = K / 128;
      std::vector<float> sw_rep((size_t)NKB * N), sa_rep((size_t)NKB * 256);
      for (int kb = 0; kb < NKB; ++kb) {
        for (int n = 0; n < N; ++n) sw_rep[(size_t)kb * N + n] = swc[n];
        for (int r = 0; r < 256; ++r) sa_rep[(size_t)kb * 256 + r] = sac[r];
      }
      int ran = 0, bad = 0, bitdiff = 0, rescdiff = 0;
      for (int skw : {2, 4, 8})
        for (int skg : {1, 2, 4, 8}) {
          I8gCfg c;
          c.trellis = true; c.kb = KB; c.fwht = true; c.resc = 4; c.skw = skw; c.skg = skg;
          if (I8gCheck(c, K, N)) continue;
          std::fill(C.begin(), C.end(), (unsigned short)0xFFFF);
          std::fill(tk.begin(), tk.end(), 0u);
          Run(c, a8c.data(), sac.data(), p.grid.data(), swc.data(), p.svh.data(), C.data(), ws.data(), tk.data(), K, N, kOutScale);
          ++ran;
          const long long viol = CountViolations(p, C, true, 4, kOutScale);
          bool tk_reset = true;
          for (unsigned x : tk) tk_reset = tk_reset && x == 0;
          if (viol || !tk_reset) { ++bad; std::printf("    FAIL coarse kb%d skw%d skg%d: %lld of %d outside tolerance, tickets %s\n", KB, skw, skg, viol, 256 * N, tk_reset ? "reset" : "NOT reset"); }
          I8gCfg d = c;
          d.trellis = false;
          std::fill(Cd.begin(), Cd.end(), (unsigned short)0xFFFF);
          Run(d, a8c.data(), sac.data(), p.W8c.data(), swc.data(), p.svh.data(), Cd.data(), ws.data(), tk.data(), K, N, kOutScale);
          if (Cd != C) { ++bitdiff; std::printf("    coarse trellis != dense twin: kb%d skw%d skg%d\n", KB, skw, skg); }
          I8gCfg r1 = c;
          r1.resc = 1;
          std::fill(Cd.begin(), Cd.end(), (unsigned short)0xFFFF);
          Run(r1, a8c.data(), sa_rep.data(), p.grid.data(), sw_rep.data(), p.svh.data(), Cd.data(), ws.data(), tk.data(), K, N, kOutScale);
          if (Cd != C) { ++rescdiff; std::printf("    RESC 4 != RESC 1 on replicated tables: kb%d skw%d skg%d\n", KB, skw, skg); }
        }
      std::snprintf(what, sizeof what, "COARSE KB %d: %d RESC 4 runs (skw 2/4/8 x skg) match the exact coarse reference, tickets reset", KB, ran);
      Check(bad == 0 && ran > 0, what);
      std::snprintf(what, sizeof what, "COARSE KB %d: RESC 4 trellis == its dense twin (same int8 weights) byte for byte, and == RESC 1 on the replicated tables", KB);
      Check(bitdiff == 0 && rescdiff == 0, what);

      // (4) one-hot rows: byte-exact against the fp32 emulation of the epilogue (coarse: v = f32(t * f32(sa[r] * sw[n])))
      {
        Problem q = p;
        std::mt19937 r2(1900 + KB);
        std::vector<int> kk(256);
        for (int r = 0; r < 256; ++r) {
          std::fill(q.xh.begin() + (size_t)r * K, q.xh.begin() + (size_t)(r + 1) * K, (uint16_t)0);
          kk[r] = r * (K / 256) + (int)(r2() % (unsigned)(K / 256));
          const float m = 0.25f + (float)(r2() % 4000u) / 1000.f;
          q.xh[(size_t)r * K + kk[r]] = i8p::F32ToF16((r2() & 1u) ? -m : m);
        }
        for (size_t i = 0; i < q.xh.size(); ++i) q.X[i] = i8p::F16ToF32(q.xh[i]);
        BuildCoarse(q);
        std::vector<unsigned short> expect((size_t)256 * N);
        for (int r = 0; r < 256; ++r) {
          float v[128];
          const int k = kk[r];
          for (int j = 0; j < 128; ++j) {
            const int t = (int)q.Apc[(size_t)r * K + k] * (int)q.Wpc[(size_t)j * K + k];
            v[j] = (float)t * (q.sac[r] * q.swc[j]);
          }
          Fwht128F32(v);
          for (int j = 0; j < 128; ++j) expect[(size_t)r * N + j] = i8g_bf16_rn((v[j] * q.svh[j]) * kOutScale);
        }
        size_t nbad = 0, runs = 0;
        for (int skw : {2, 4, 8})
          for (int skg : {1, 2, 4, 8}) {
            I8gCfg c;
            c.trellis = true; c.kb = KB; c.fwht = true; c.resc = 4; c.skw = skw; c.skg = skg;
            if (I8gCheck(c, K, N)) continue;
            std::fill(C.begin(), C.end(), (unsigned short)0xFFFF);
            Run(c, q.A8c.data(), q.sac.data(), q.grid.data(), q.swc.data(), q.svh.data(), C.data(), ws.data(), tk.data(), K, N, kOutScale);
            ++runs;
            for (size_t i = 0; i < C.size(); ++i) nbad += C[i] != expect[i];
          }
        std::snprintf(what, sizeof what, "COARSE KB %d: %zu one-hot launches byte-exact vs the fp32 emulation of the epilogue (%zu bytes differ)", KB, runs, nbad);
        Check(nbad == 0 && runs > 0, what);
      }
    }
    // (5) two A parts with coarse scales: the part-1 quantizer launch, n_split, against each part's reference and two
    // single-part launches, byte for byte
    for (int KB : {4, 5}) {
      const int N2 = 256;
      Problem p0, p1;
      BuildProblem(p0, K, N2, KB, 1300 + KB);
      p1 = p0;
      ReplaceActivations(p1, 1777 + KB);
      BuildCoarse(p0);
      BuildCoarse(p1);
      std::vector<uint16_t> xcat(p0.xh);
      xcat.insert(xcat.end(), p1.xh.begin(), p1.xh.end());
      std::vector<signed char> a8cat(2 * p0.A8c.size(), 0);
      std::vector<float> sacat(2 * 256, 0.f);
      emu::launch(256, 2, 256, i8g_quant_act_row, xcat.data(), a8cat.data(), sacat.data(), K, (long long)256 * K);
      std::snprintf(what, sizeof what, "COARSE KB %d: i8g_quant_act_row with two parts (blockIdx.y) == the CPU row quantizer of each part", KB);
      Check(std::memcmp(a8cat.data(), p0.A8c.data(), p0.A8c.size()) == 0 && std::memcmp(a8cat.data() + p0.A8c.size(), p1.A8c.data(), p1.A8c.size()) == 0 &&
                std::memcmp(sacat.data(), p0.sac.data(), 256 * 4) == 0 && std::memcmp(sacat.data() + 256, p1.sac.data(), 256 * 4) == 0, what);
      const signed char *a0 = a8cat.data(), *a1 = a8cat.data() + p0.A8c.size();
      const float *s0 = sacat.data(), *s1 = sacat.data() + 256;
      std::vector<unsigned short> Cp((size_t)256 * N2), Ca((size_t)256 * N2), Cb((size_t)256 * N2);
      std::vector<float> ws2((size_t)8 * 256 * N2);
      std::vector<unsigned> tk2(N2 / 128, 0u);
      int ran = 0, bad = 0;
      for (int skw : {2, 4, 8})
        for (int skg : {1, 2, 4}) {
          I8gCfg c;
          c.trellis = true; c.kb = KB; c.fwht = true; c.resc = 4; c.skw = skw; c.skg = skg;
          if (I8gCheck(c, K, N2)) continue;
          std::fill(Cp.begin(), Cp.end(), (unsigned short)0xFFFF);
          RunParts(c, a0, s0, a1, s1, 128, p0.grid.data(), p0.swc.data(), p0.svh.data(), Cp.data(), ws2.data(), tk2.data(), K, N2, kOutScale);
          const long long v0 = CountViolations(p0, Cp, true, 4, kOutScale, 0, 1);
          const long long v1 = CountViolations(p1, Cp, true, 4, kOutScale, 1, 2);
          RunParts(c, a0, s0, a0, s0, N2, p0.grid.data(), p0.swc.data(), p0.svh.data(), Ca.data(), ws2.data(), tk2.data(), K, N2, kOutScale);
          RunParts(c, a1, s1, a1, s1, N2, p0.grid.data(), p0.swc.data(), p0.svh.data(), Cb.data(), ws2.data(), tk2.data(), K, N2, kOutScale);
          bool same = true;
          for (int r = 0; r < 256; ++r)
            for (int n = 0; n < N2; ++n) same = same && Cp[(size_t)r * N2 + n] == (n < 128 ? Ca : Cb)[(size_t)r * N2 + n];
          bool tk_reset = true;
          for (unsigned x : tk2) tk_reset = tk_reset && x == 0;
          ++ran;
          if (v0 || v1 || !same || !tk_reset) {
            ++bad;
            std::printf("    FAIL coarse two parts kb%d skw%d skg%d: part 0 %lld, part 1 %lld outside tolerance, bytes %s single-part launches, tickets %s\n",
                        KB, skw, skg, v0, v1, same ? "==" : "!=", tk_reset ? "reset" : "NOT reset");
          }
        }
      std::snprintf(what, sizeof what, "COARSE KB %d: %d two-part RESC 4 launches (n_split 128 of 256) match each part's exact reference and the two single-part launches byte for byte", KB, ran);
      Check(bad == 0 && ran > 0, what);
    }
    // (6) the int32 range: K = 17408 with every activation and weight at +127 (the dense kernel, RESC 4, FWHT epilogue: the
    // trellis kernel's loop and slicing without the decode): each slice's int32 sum is K / (skw skg) x 127 x 127 (1.4e8 at
    // skw 2, skg 1, the longest slice any legal configuration has), below 2^31; the output must still match the exact reference
    {
      Problem q;
      q.K = 17408; q.N = 128; q.KB = 4;
      q.Ap.assign((size_t)256 * q.K, 127);
      q.Wp.assign((size_t)q.N * q.K, 127);
      q.A8 = i8p::PackA8(q.Ap, q.K);
      q.W8 = i8p::PackW8(q.Wp, q.K, q.N);
      q.sac.assign(256, 1.0f);
      q.swc.assign((size_t)q.N, 1.0f);
      std::mt19937 r3(2468);
      q.svh.resize(q.N);
      for (int n = 0; n < q.N; ++n) q.svh[n] = (r3() & 1 ? -1.f : 1.f) * (0.5f + (r3() % 1000) / 1000.f);
      q.rawc.resize((size_t)256 * q.N);
      i8p::RefGemmInt8(q.Ap.data(), q.sac.data(), q.Wp.data(), q.swc.data(), 256, q.N, q.K, 3, q.rawc.data());
      std::vector<unsigned short> Cq((size_t)256 * q.N);
      std::vector<float> wsq((size_t)8 * 256 * q.N);
      std::vector<unsigned> tkq(q.N / 128, 0u);
      long long viol = 0;
      int ranq = 0;
      for (const auto sk : {std::make_pair(2, 1), std::make_pair(4, 2), std::make_pair(8, 1)}) {
        I8gCfg c;
        c.trellis = false; c.fwht = true; c.resc = 4; c.skw = sk.first; c.skg = sk.second;
        if (I8gCheck(c, q.K, q.N)) continue;
        std::fill(Cq.begin(), Cq.end(), (unsigned short)0xFFFF);
        Run(c, q.A8.data(), q.sac.data(), q.W8.data(), q.swc.data(), q.svh.data(), Cq.data(), wsq.data(), tkq.data(), q.K, q.N, kOutScale);
        viol += CountViolations(q, Cq, true, 4, kOutScale);
        ++ranq;
      }
      std::snprintf(what, sizeof what, "COARSE K 17408, all operands +127 (int32 slice sums up to 1.4e8): %d RESC 4 runs match the exact reference (%lld outside tolerance)", ranq, viol);
      Check(viol == 0 && ranq > 0, what);
    }
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s (%d checks, %.1f s)\n", g_ok ? "ALL OK" : "FAILED", g_checks, s);
  return g_ok ? 0 : 1;
}
