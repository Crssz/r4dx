// tests/kernels/test_int8_gemm_proto_emu.cpp -- pure CPU (no HIP call, no device). Runs the int8 prefill GEMM
// prototype's KERNEL SOURCE (int8_gemm_proto_kernels.h, compiled as plain C++ with I8G_EMU: one OS thread per GPU
// thread, workgroup and wave barriers, shared memory, a software iu8 WMMA, a stub trellis decode; see
// i8g_host_emu.h) against exact references:
//   * i8g_quant_act   vs i8p::QuantizeActRef + PackA8, byte for byte;
//   * i8g_wscale / i8g_dump_w (KB 4 and 5, over the stub decode) vs the CPU table and int8 matrix, byte for byte;
//   * i8g_kernel, every instantiation (dense with the FWHT epilogue and plain, trellis KB 4 and 5; RESC 0..3;
//     SKW 2, 4, 8) for every legal (skw, skg) on K = 1024, N = 128 (one FWHT group, 4 x SKG workgroups racing for
//     the ticket), against the exact-integer fp64 reference (scale mask by RESC) with the epilogue applied in fp64,
//     and the trellis kernel against the dense kernel on the same int8 weights, byte for byte.
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
int MaskOfResc(int resc) { static const int m[4] = {0, 3, 1, 2}; return m[resc & 3]; }

// the emulated launch of i8g_kernel for a configuration (the dispatch of I8gRun, over emu::launch)
template <bool TR, int KB, bool FW, int RE, int SKW>
void LaunchT(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw, const float* svh,
             unsigned short* C, float* ws, unsigned* tk, int K, int N, float os) {
  emu::launch((unsigned)(N / 32), (unsigned)c.skg, 128u * SKW, i8g_kernel<TR, KB, FW, RE, SKW>, a8, sa, (const unsigned char*)w, sw, svh, C, ws,
              tk, K, N, c.skg, os);
}
template <bool TR, int KB, bool FW, int RE>
void LaunchSkw(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw, const float* svh,
               unsigned short* C, float* ws, unsigned* tk, int K, int N, float os) {
  switch (c.skw) {
    case 2: LaunchT<TR, KB, FW, RE, 2>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os); break;
    case 4: LaunchT<TR, KB, FW, RE, 4>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os); break;
    default: LaunchT<TR, KB, FW, RE, 8>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os); break;
  }
}
void Run(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw, const float* svh, unsigned short* C,
         float* ws, unsigned* tk, int K, int N, float os) {
#define CASE(TR, KB, FW, RE) LaunchSkw<TR, KB, FW, RE>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os)
#define RESC4(TR, KB, FW) \
  switch (c.resc) { case 0: CASE(TR, KB, FW, 0); break; case 1: CASE(TR, KB, FW, 1); break; case 2: CASE(TR, KB, FW, 2); break; default: CASE(TR, KB, FW, 3); break; }
  if (!c.trellis) {
    if (c.fwht) { if (c.resc == 0) CASE(false, 4, true, 0); else CASE(false, 4, true, 1); }
    else { if (c.resc == 0) CASE(false, 4, false, 0); else CASE(false, 4, false, 1); }
  } else if (c.kb == 4) {
    RESC4(true, 4, true);
  } else {
    RESC4(true, 5, true);
  }
#undef RESC4
#undef CASE
}

struct Problem {
  int K, N, KB;
  std::vector<uint16_t> xh;            // f16 activations [256][K]
  std::vector<float> X;
  std::vector<unsigned> grid;          // trellis words (the stub decode reads them), (N/32) * (K/16) blocks of 16 KB words
  std::vector<int8_t> Ap, A8, Wp, W8;
  std::vector<float> sa, sw, svh;
  std::vector<double> raw[4];          // exact reference by scale mask
};

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
  std::vector<uint16_t> Q((size_t)K * N);
  const int KT = K / 16;
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
  i8p::QuantizeWeightsRef(Q.data(), K, N, p.sw, p.Wp);
  p.W8 = i8p::PackW8(p.Wp, K, N);
  for (int mask = 0; mask < 4; ++mask) {
    p.raw[mask].resize((size_t)256 * N);
    i8p::RefGemmInt8(p.Ap.data(), p.sa.data(), p.Wp.data(), p.sw.data(), 256, N, K, mask, p.raw[mask].data());
  }
}

long long CountViolations(const Problem& p, const std::vector<unsigned short>& C, bool fwht, int resc, float os) {
  long long viol = 0;
  for (int r = 0; r < 256; ++r)
    for (int g = 0; g < p.N / 128; ++g) {
      double v[128], e[128], rms = 0;
      for (int j = 0; j < 128; ++j) v[j] = p.raw[MaskOfResc(resc)][(size_t)r * p.N + g * 128 + j];
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
    emu::launch((unsigned)((256 * NKB + 7) / 8), 1, 256, i8g_quant_act, p.xh.data(), a8.data(), ap.data(), sa.data(), K);
    std::snprintf(what, sizeof what, "KB %d: i8g_quant_act (emulated kernel) == CPU quantizer: A8 layout, plain A, scales (incl. an all-zero block)", KB);
    Check(std::memcmp(a8.data(), p.A8.data(), a8.size()) == 0 && std::memcmp(ap.data(), p.Ap.data(), ap.size()) == 0 &&
              std::memcmp(sa.data(), p.sa.data(), sa.size() * 4) == 0, what);
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
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s (%d checks, %.1f s)\n", g_ok ? "ALL OK" : "FAILED", g_checks, s);
  return g_ok ? 0 : 1;
}
