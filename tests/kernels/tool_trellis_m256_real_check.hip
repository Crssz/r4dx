// tests/kernels/tool_trellis_m256_real_check.hip -- bit identity of the PROTOTYPE M = 256 trellis GEMM
// (third_party/libr4d/r4d_gemm_trellis_nt_m256.hip) against the SHIPPED M = 64 tuning-table row, on REAL
// container weights: one launch of M = 256 must equal, byte for byte, four launches of the shipped kernel at
// the row src/model/gemm_tuning_table_trellis.inc gives (N, K, M = 64, KB), for every (SKW, SK / SKW) the
// M = 256 kernel has for that row's (SK, SKG), one activation matrix and the two-part (n_split) form, several
// seeds of N(0, 1) f16 activations, each M = 256 launch repeated (a race in the running-sum handoff would show
// as a run-to-run difference). Negative control per file: a different SK must differ.
//
// The tensors are raw slices of the container (per class and KB: <name>_kb<KB>.w = the N x K trellis words,
// <name>_kb<KB>.svh = N fp32), extracted once by the verifier; they are data, never committed.
//   tool_trellis_m256_real_check.exe --dir <folder> [--seeds 3] [--repeat 12] [--only mlp_down]
// Build: build_m256_bench.ps1 -Tool real_check.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "r4d.h"
#include "trellis_tuning_rows.hpp"

void r4d_gemm_trellis_nt_m256(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh, int64_t c,
                              int64_t ws, int64_t tickets, int M, int K, int N, int KB, int SK, int NP,
                              int SKG, int U, float out_scale, int64_t stream, int skw);

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void FillGauss(unsigned short* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    uint32_t g = (uint32_t)(i >> 32) * 0x9E3779B9u + h * 0x85EBCA6Bu + seed; g ^= g >> 13; g *= 0xC2B2AE35u; g ^= g >> 16;
    const float u1 = ((h >> 8) + 1) / 16777217.f, u2 = (g >> 8) / 16777216.f;
    const float z = sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
    p[i] = __half_as_ushort(__float2half(z));
  }
}

static bool ReadAll(const std::string& f, std::vector<char>& v) {
  std::ifstream s(f, std::ios::binary | std::ios::ate);
  if (!s) return false;
  const size_t n = (size_t)s.tellg();
  s.seekg(0);
  v.resize(n);
  s.read(v.data(), (std::streamsize)n);
  return true;
}

struct Cls { const char* name; int N, K; };
static const Cls kCls[] = {
    {"mlp_gate_up", 34816, 5120}, {"mlp_down", 5120, 17408},  {"gdn_in_proj_qkv", 10240, 5120},
    {"gdn_in_proj_z", 6144, 5120}, {"gdn_out_proj", 5120, 6144}, {"attn_o", 5120, 6144},
    {"attn_qg", 12288, 5120},      {"attn_k", 1024, 5120},       {"attn_v", 1024, 5120},
};

int main(int argc, char** argv) {
  std::string dir, only;
  int seeds = 3, repeat = 12;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dir" && i + 1 < argc) dir = argv[++i];
    else if (a == "--seeds" && i + 1 < argc) seeds = std::atoi(argv[++i]);
    else if (a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (dir.empty()) { std::fprintf(stderr, "--dir required\n"); return 2; }
  const float os = 0.0883883f;
  int fails = 0, checks = 0, files = 0;
  for (const Cls& c : kCls) {
    if (!only.empty() && only != c.name) continue;
    for (int KB : {4, 5}) {
      const int N = c.N, K = c.K;
      const trellis_rows::GemmTuningRow* row = nullptr;
      for (const auto& r : trellis_rows::kGemmTuningTable)
        if (r.layout == trellis_rows::Layout::kTrellis && r.N == N && r.K == K && r.M == 64 && r.rate == KB) row = &r;
      if (!row) { std::printf("%s KB%d: no shipped row\n", c.name, KB); ++fails; continue; }
      const auto T = row->tuning;
      std::vector<char> wv, sv;
      const std::string base = dir + "\\" + c.name + "_kb" + std::to_string(KB);
      if (!ReadAll(base + ".w", wv) || !ReadAll(base + ".svh", sv)) { std::printf("%s KB%d: file missing\n", c.name, KB); continue; }
      if ((size_t)N * K * KB / 8 != wv.size() || sv.size() != (size_t)N * 4) { std::printf("%s KB%d: size mismatch\n", c.name, KB); ++fails; continue; }
      ++files;
      unsigned char* dw; float* dsvh; unsigned short *dA, *dC0, *dC1; float* ws; unsigned* tk;
      CK(hipMalloc(&dw, wv.size())); CK(hipMalloc(&dsvh, sv.size()));
      CK(hipMalloc(&dA, (size_t)2 * 256 * K * 2)); CK(hipMalloc(&dC0, (size_t)256 * N * 2)); CK(hipMalloc(&dC1, (size_t)256 * N * 2));
      CK(hipMalloc(&ws, (size_t)8 * 256 * N * 4)); CK(hipMalloc(&tk, (size_t)(N / 128) * 4));
      CK(hipMemset(tk, 0, (size_t)(N / 128) * 4));
      CK(hipMemcpy(dw, wv.data(), wv.size(), hipMemcpyHostToDevice));
      CK(hipMemcpy(dsvh, sv.data(), sv.size(), hipMemcpyHostToDevice));
      std::printf("%-16s KB%d shipped M=64 row: WV%d SK%d MT%d NP%d NT%d SKG%d U%d\n", c.name, KB, T.WV, T.SK, T.MB, T.NPW, T.NT, T.SKG, T.U);
      std::vector<unsigned short> h0((size_t)256 * N), h1((size_t)256 * N), hr((size_t)256 * N);
      for (int two = 0; two < 2; ++two) {
        const int n_split = two ? N / 2 : N;
        auto stock = [&](const unsigned short* A0, const unsigned short* A1, unsigned short* C, int SKo) {
          for (int r = 0; r < 256; r += 64)
            r4d_gemm_trellis_nt_m64((int64_t)(A0 + (size_t)r * K), (int64_t)(A1 ? A1 + (size_t)r * K : nullptr), n_split, (int64_t)dw,
                                    (int64_t)dsvh, (int64_t)(C + (size_t)r * N), (int64_t)ws, (int64_t)tk, 64, K, N, KB,
                                    SKo ? 1 : T.WV, SKo ? SKo : T.SK, SKo ? 4 : T.MB, SKo ? 1 : T.NPW, T.SKG, SKo ? 1 : T.U, SKo ? 0 : T.NT, os, 0);
        };
        for (int seed = 101; seed < 101 + seeds; ++seed) {
          hipLaunchKernelGGL(FillGauss, dim3(2048), dim3(256), 0, 0, dA, (int64_t)2 * 256 * K, (uint32_t)seed);
          CK(hipDeviceSynchronize());
          const unsigned short* A1 = two ? dA + (size_t)256 * K : nullptr;
          CK(hipMemset(dC0, 0xFF, (size_t)256 * N * 2));
          stock(dA, A1, dC0, 0);
          CK(hipDeviceSynchronize());
          CK(hipMemcpy(h0.data(), dC0, h0.size() * 2, hipMemcpyDeviceToHost));
          size_t nz = 0, nn = 0;
          for (size_t i = 0; i < h0.size(); ++i) { nz += (h0[i] & 0x7fff) == 0; nn += (h0[i] & 0x7f80) == 0x7f80; }
          for (int skw : {2, 4, 8}) {
            if (T.SK % skw || T.SK / skw > 4 || 4 * skw > 32) continue;
            ++checks;
            CK(hipMemset(dC1, 0xFF, (size_t)256 * N * 2));
            r4d_gemm_trellis_nt_m256((int64_t)dA, (int64_t)A1, n_split, (int64_t)dw, (int64_t)dsvh, (int64_t)dC1, (int64_t)ws, (int64_t)tk,
                                     256, K, N, KB, T.SK, 1, T.SKG, 4, os, 0, skw);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(h1.data(), dC1, h1.size() * 2, hipMemcpyDeviceToHost));
            size_t nd = 0;
            for (size_t i = 0; i < h0.size(); ++i) nd += h0[i] != h1[i];
            size_t nrep = 0;
            for (int rep = 0; rep < repeat; ++rep) {
              r4d_gemm_trellis_nt_m256((int64_t)dA, (int64_t)A1, n_split, (int64_t)dw, (int64_t)dsvh, (int64_t)dC1, (int64_t)ws, (int64_t)tk,
                                       256, K, N, KB, T.SK, 1, T.SKG, 4, os, 0, skw);
              CK(hipDeviceSynchronize());
              CK(hipMemcpy(hr.data(), dC1, hr.size() * 2, hipMemcpyDeviceToHost));
              for (size_t i = 0; i < hr.size(); ++i) nrep += hr[i] != h1[i];
            }
            std::vector<unsigned> ht(N / 128);
            CK(hipMemcpy(ht.data(), tk, ht.size() * 4, hipMemcpyDeviceToHost));
            int tkb = 0;
            for (unsigned x : ht) tkb += x != 0;
            const bool ok = nd == 0 && nrep == 0 && tkb == 0 && nn == 0;
            std::printf("  %-8s seed %d SKW%d x%d: %zu of %zu bf16 differ from 4x shipped row, %zu run-to-run diffs over %d repeats, tickets %s, zeros %zu, nonfinite %zu  %s\n",
                        two ? "two-part" : "one-A", seed, skw, T.SK / skw, nd, h0.size(), nrep, repeat, tkb ? "NOT reset" : "reset", nz, nn, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
          }
          if (seed == 101) {   // negative control: a different SK must differ (the accumulation order is part of the bits)
            const int skc = T.SK == 4 ? 2 : 4;
            CK(hipMemset(dC0, 0xFF, (size_t)256 * N * 2));
            stock(dA, A1, dC0, skc);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(h0.data(), dC0, h0.size() * 2, hipMemcpyDeviceToHost));
            size_t nd = 0;
            for (size_t i = 0; i < h0.size(); ++i) nd += h0[i] != h1[i];
            std::printf("  %-8s control: M256 at SK%d vs 4x stock SK%d: %zu bf16 differ (must be > 0)  %s\n", two ? "two-part" : "one-A",
                        T.SK, skc, nd, nd ? "OK" : "FAIL");
            if (!nd) ++fails;
          }
        }
      }
      CK(hipFree(dw)); CK(hipFree(dsvh)); CK(hipFree(dA)); CK(hipFree(dC0)); CK(hipFree(dC1)); CK(hipFree(ws)); CK(hipFree(tk));
    }
  }
  std::printf("%d files, %d M256 launches compared: %s (%d failures)\n", files, checks, fails ? "FAILED" : "ALL IDENTICAL", fails);
  return fails ? 1 : 0;
}
