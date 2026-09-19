// tests/kernels/test_mxfp4_gemm.cpp -- end-to-end correctness for r4d_gemm_mxfp4a8_nt_m64,
// run on the ACTUAL output of r4dx_quant_act_fp8e4m3_row (not a synthetic fp8 buffer), against a
// CPU fp32 reference that dequantizes both operands: the real quantized activation the GPU
// produced, and an OCP-MXFP4 weight built and permuted exactly the way
// third_party/libr4d/mxfp4_layout.py's permute_w does (mirrored in C++ below -- libr4d's own
// correctness check, test_mxfp4_gemm.py, is Python/torch-only, so nothing in this repo
// previously exercised r4dx's fp8-activation quantizer feeding this GEMM kernel).
//
// Without this test, r4dx_quant_act_fp8e4m3_row's claim of feeding r4d_gemm_mxfp4a8_nt_m64
// correctly (kernels.h's doc comment) was validated only by test_quant_act_fp8.cpp, which is
// tautological at the value level (its CPU reference calls the SAME FloatToFp8E4M3 the kernel
// calls, so it can only catch a GPU/CPU disagreement on identical code, not a wrong contract with
// the GEMM kernel) -- see the Opus review that flagged this gap. mxfp4 is one of the three GEMM
// weight layouts the project is built around (docs/architecture.md), so it needs an actual
// quantize-then-GEMM, GPU-vs-CPU check.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

// checkpoint-order [N, K/2] uint8 (two 4-bit codes packed per byte, low nibble = even column) ->
// the fragment order r4d_gemm_mxfp4a8_nt_m64 reads. Mirrors third_party/libr4d/mxfp4_layout.py's
// permute_w exactly: slot l of tile (nt, ks) is w[nt][l&15][ks][4*(l>>4) ..+4], i.e. lane l takes
// row 16*nt + (l&15), k starting at 16*ks + 8*(l>>4).
std::vector<uint8_t> PermuteW(const std::vector<uint8_t>& packed, int N, int K) {
  const int nt = N / 16, ks = K / 16, half_bytes = K / 2;
  std::vector<uint8_t> out(static_cast<size_t>(nt) * ks * 32 * 4);
  for (int t = 0; t < nt; ++t) {
    for (int kb = 0; kb < ks; ++kb) {
      for (int l = 0; l < 32; ++l) {
        const int r = l & 15, h = l >> 4;
        const int n = t * 16 + r;
        for (int j = 0; j < 4; ++j) {
          const int kbyte = kb * 8 + 4 * h + j;
          const size_t out_idx = ((static_cast<size_t>(t) * ks + kb) * 32 + l) * 4 + j;
          out[out_idx] = packed[static_cast<size_t>(n) * half_bytes + kbyte];
        }
      }
    }
  }
  return out;
}

constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  const int M = 8, N = 512, K = 1024;
  const int WV = 4, SK = 4, MB = 1, NPW = 1;  // N/16=32 divisible by WV*NPW=4: no tail-block case
  const int group = r4d_gemm_mxfp4a8_nt_m64_group();  // OCP-MXFP4: one exponent per `group` of K
  if (K % 16 != 0 || K % group != 0) {
    std::fprintf(stderr, "test precondition violated: K=%d must be a multiple of 16 and of group=%d\n",
                 K, group);
    return 1;
  }
  const int nb = K / group;

  std::mt19937 rng(41);
  std::uniform_int_distribution<int> code_dist(0, 15);
  std::uniform_int_distribution<int> ref_dist(120, 135);
  std::uniform_int_distribution<int> drop_dist(0, 8);  // max_d=8, matches libr4d's test default
  std::normal_distribution<float> randn(0.0f, 1.0f);

  // ---- weight: per-row reference exponent, per-block dropped exponent, random e2m1 codes ------
  std::vector<int> row_ref_seed(N);
  for (auto& v : row_ref_seed) v = ref_dist(rng);
  std::vector<uint8_t> e8m0(static_cast<size_t>(nb) * N);  // [nb, N], what the kernel reads as `ws`
  std::vector<uint8_t> wref(N);                             // [N], per-row true max exponent
  for (int n = 0; n < N; ++n) {
    int row_max = 0;
    for (int b = 0; b < nb; ++b) {
      int drop = drop_dist(rng);
      int e = std::max(0, std::min(254, row_ref_seed[n] - drop));
      e8m0[static_cast<size_t>(b) * N + n] = static_cast<uint8_t>(e);
      row_max = std::max(row_max, e);
    }
    wref[n] = static_cast<uint8_t>(row_max);
  }

  std::vector<uint8_t> codes(static_cast<size_t>(N) * K);
  for (auto& c : codes) c = static_cast<uint8_t>(code_dist(rng));

  std::vector<uint8_t> packed(static_cast<size_t>(N) * (K / 2));
  for (int n = 0; n < N; ++n) {
    for (int b = 0; b < K / 2; ++b) {
      uint8_t lo = codes[static_cast<size_t>(n) * K + 2 * b];
      uint8_t hi = codes[static_cast<size_t>(n) * K + 2 * b + 1];
      packed[static_cast<size_t>(n) * (K / 2) + b] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  std::vector<uint8_t> wq = PermuteW(packed, N, K);

  std::vector<float> Wf(static_cast<size_t>(N) * K);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      uint8_t code = codes[static_cast<size_t>(n) * K + k];
      float mag = kE2M1[code & 0x7];
      float sign = (code & 0x8) ? -1.0f : 1.0f;
      int block = k / group;
      float scale = std::exp2(static_cast<float>(e8m0[static_cast<size_t>(block) * N + n]) - 127.0f);
      Wf[static_cast<size_t>(n) * K + k] = sign * mag * scale;
    }
  }

  // ---- activation: bf16 -> fp8e4m3 via r4dx's OWN quantizer kernel (not a synthetic buffer),
  // so this test exercises the real quantize-then-GEMM pipeline the kernels.h doc claims works.
  std::vector<uint16_t> x_h(static_cast<size_t>(M) * K);
  for (auto& v : x_h) v = FloatToBf16(randn(rng) * 0.4f);

  DeviceBuffer<uint16_t> x_d(x_h.size());
  DeviceBuffer<uint8_t> af8_d(x_h.size());
  DeviceBuffer<float> ascale_d(M);
  x_d.CopyFromHost(x_h);
  r4dx_quant_act_fp8e4m3_row(reinterpret_cast<int64_t>(x_d.data()), reinterpret_cast<int64_t>(af8_d.data()),
                              reinterpret_cast<int64_t>(ascale_d.data()), M, K, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint8_t> af8_h = af8_d.CopyToHost();
  std::vector<float> ascale_h = ascale_d.CopyToHost();

  std::vector<float> ref_out(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        float av = Fp8E4M3ToFloat(af8_h[static_cast<size_t>(m) * K + k]) * ascale_h[m];
        acc += static_cast<double>(av) * Wf[static_cast<size_t>(n) * K + k];
      }
      ref_out[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
    }
  }

  DeviceBuffer<uint8_t> wq_d(wq.size()), e8m0_d(e8m0.size()), wref_d(wref.size());
  DeviceBuffer<uint16_t> c_d(static_cast<size_t>(M) * N);
  wq_d.CopyFromHost(wq);
  e8m0_d.CopyFromHost(e8m0);
  wref_d.CopyFromHost(wref);

  r4d::GemmMxfp4a8NtM64(af8_d.data(), ascale_d.data(), wq_d.data(), e8m0_d.data(), wref_d.data(),
                         c_d.data(), M, K, N, WV, SK, MB, NPW, nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> c_h = c_d.CopyToHost();

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref_out.size(); ++i) {
    double got = Bf16ToFloat(c_h[i]);
    double diff = got - ref_out[i];
    num += diff * diff;
    den += static_cast<double>(ref_out[i]) * ref_out[i];
  }
  double rel = std::sqrt(num) / std::max(1e-9, std::sqrt(den));
  std::printf("mxfp4a8 gemm (M=%d,N=%d,K=%d,group=%d) norm rel err=%.4e\n", M, N, K, group, rel);
  bool ok = rel < 2e-2;  // same threshold libr4d's own test_mxfp4_gemm.py uses
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
