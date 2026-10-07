// tests/kernels/test_trellis_input_i8.cpp -- the fused activation quantizer of the int8 prefill GEMM
// (R4DX_PREFILL_INT8_FUSEDQ, docs/int8-prefill.md "The fused quantizer"), byte for byte against the chain it replaces.
//
// Today a 256-row int8 call gets its transformed f16 A from a trellis producer and then runs libr4d's separate
// r4d_trellis_i8_quant_act over it. The fused entries (kernels.h: r4dx_trellis_input_i8, r4dx_silu_mul_trellis_i8,
// r4dx_attn_gate_mul_trellis_i8) transform, round to f16 and quantize in one launch and write A8 + SA straight away.
// Their output must be the unfused chain's, bit for bit, on every byte of both buffers:
//   A. r4dx_trellis_input_i8 against r4dx_trellis_input_bf16 + TrellisI8QuantAct: nout 1..3 (a linear's two parts, the
//      shared qg / k / v), K in {3072, 5120, 6144, 8704, 17408}, prescale 0 / 4 / -3, activations spanning the f16 range of
//      the transform's output (rows of exact zeros and blocks of exact zeros included: the scale-1 case);
//   B. r4dx_silu_mul_trellis_i8 against r4dx_silu_mul_trellis_bf16 + the quantizer, the TP = 1 and TP = 2 rank widths
//      (17408, 8704), a tight and a padded input row stride;
//   C. r4dx_attn_gate_mul_trellis_i8 against r4dx_attn_gate_mul_trellis_bf16 + the quantizer, K = 6144 and 3072;
//   D. nothing outside the outputs is written (canary words around A8 and SA), every output byte IS written (the two
//      sides start from different poison), and the comparison can fail (a flipped byte is seen);
//   E. the preconditions: a row count other than 256 and a null a8 / sa throw.
//   Everything above runs twice: with the per-128 entries (r4dx_*_i8) and, as [coarse], with the per-ROW ones (r4dx_*_i8r,
//   R4DX_PREFILL_INT8_SCALES=coarse: one scale per row, SA [256] per output, a workgroup of 8 waves per row staging the f16 values
//   in LDS) against the f16 producers + TrellisI8QuantActRow (r4d_trellis_i8_quant_act_row).
//
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt); skips (77) without a device.
// Needs no golden data. The comparison against the CPU rules of the quantizer is on the CPU side:
// test_int8_gemm_proto_emu runs the same device function (r4d_trellis_i8_fused.h) against the CPU quantizer.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

constexpr int kM = 256;
int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    if (g_failures < 40) std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

template <typename T>
int64_t P(T* p) {
  return reinterpret_cast<int64_t>(p);
}

template <typename Fn>
bool Throws(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

// bf16 activations as test_trellis_input's RandomX: row scales log-uniform in [2^-8, 2^3] and two 20-60x outliers
// per row; a few rows of exact zeros and a row whose first 128-block is zero (a scale-1 block of the quantizer).
std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-8.f, 3.f), od(20.f, 60.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  for (int64_t m = 0; m < M; ++m) {
    const float s = std::exp2(ud(rng));
    std::vector<float> row(static_cast<size_t>(K));
    for (auto& v : row) v = nd(rng) * s;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    if (m == 3 || m == 200) std::fill(row.begin(), row.end(), 0.f);
    if (m == 100) std::fill(row.begin(), row.begin() + 128, 0.f);
    for (int64_t k = 0; k < K; ++k) x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
  }
  return x;
}

std::vector<float> RandomScales(std::mt19937_64& rng, int64_t n) {
  std::uniform_real_distribution<double> ud(std::log(0.02), std::log(2.0));
  std::uniform_int_distribution<int> sd(0, 1);
  std::vector<float> s(static_cast<size_t>(n));
  for (auto& v : s) v = F16ToFloat(FloatToF16(static_cast<float>((sd(rng) ? -1.0 : 1.0) * std::exp(ud(rng)))));
  return s;
}

// The int8 operand of `nout` outputs of K columns: A8 (256 K bytes each) and SA (K / 128 x 256 floats each), with
// canary words after them and a poison fill.
struct Operand {
  int nout, K;
  bool coarse;
  DeviceBuffer<int8_t> a8;
  DeviceBuffer<float> sa;
  size_t a8_n, sa_n;
  static constexpr size_t kCanary = 256;
  static constexpr uint8_t kCanaryByte = 0xC3;

  Operand(int nout_, int K_, uint8_t poison, bool coarse_ = false) : nout(nout_), K(K_), coarse(coarse_) {
    a8_n = static_cast<size_t>(nout) * kM * K;
    sa_n = static_cast<size_t>(nout) * (coarse ? 1 : K / 128) * kM;   // coarse: SA is [256] per output
    a8 = DeviceBuffer<int8_t>(a8_n + kCanary);
    sa = DeviceBuffer<float>(sa_n + kCanary);
    R4DX_HIP_CHECK(hipMemset(a8.data(), poison, a8_n));
    R4DX_HIP_CHECK(hipMemset(a8.data() + a8_n, kCanaryByte, kCanary));
    R4DX_HIP_CHECK(hipMemset(sa.data(), poison, sa_n * sizeof(float)));
    R4DX_HIP_CHECK(hipMemset(sa.data() + sa_n, kCanaryByte, kCanary * sizeof(float)));
  }
  int64_t A8(int o) { return P(a8.data() + static_cast<size_t>(o) * kM * K); }
  int64_t SA(int o) { return P(sa.data() + static_cast<size_t>(o) * (coarse ? 1 : K / 128) * kM); }
  bool CanariesIntact() const {
    std::vector<int8_t> a = a8.CopyToHost();
    std::vector<float> s = sa.CopyToHost();
    for (size_t i = a8_n; i < a.size(); ++i)
      if (static_cast<uint8_t>(a[i]) != kCanaryByte) return false;
    for (size_t i = sa_n; i < s.size(); ++i) {
      uint32_t w;
      std::memcpy(&w, &s[i], 4);
      if (w != 0xC3C3C3C3u) return false;
    }
    return true;
  }
};

// Whether two operands hold the same bytes over their payload (not the canaries); on a mismatch, how many bytes of each.
bool SameBytes(const Operand& a, const Operand& b, size_t* bad_a8 = nullptr, size_t* bad_sa = nullptr) {
  const std::vector<int8_t> a1 = a.a8.CopyToHost(), a2 = b.a8.CopyToHost();
  const std::vector<float> s1 = a.sa.CopyToHost(), s2 = b.sa.CopyToHost();
  size_t d8 = 0, ds = 0;
  for (size_t i = 0; i < a.a8_n; ++i) d8 += a1[i] != a2[i];
  for (size_t i = 0; i < a.sa_n; ++i) {
    uint32_t x, y;
    std::memcpy(&x, &s1[i], 4);
    std::memcpy(&y, &s2[i], 4);
    ds += x != y;
  }
  if (bad_a8 != nullptr) *bad_a8 = d8;
  if (bad_sa != nullptr) *bad_sa = ds;
  return d8 == 0 && ds == 0;
}

// the separate quantizer over `nout` f16 outputs of one buffer (each output is its own linear's A: one launch each,
// the layout a multi-part linear's single two-part launch also produces)
void QuantizeUnfused(const DeviceBuffer<uint16_t>& a16, Operand& op) {
  for (int o = 0; o < op.nout; ++o) {
    (op.coarse ? r4d::TrellisI8QuantActRow : r4d::TrellisI8QuantAct)(a16.data() + static_cast<size_t>(o) * kM * op.K, reinterpret_cast<int8_t*>(op.A8(o)),
                                                                     reinterpret_cast<float*>(op.SA(o)), 1, 0, op.K, nullptr);
  }
  R4DX_HIP_CHECK(hipDeviceSynchronize());
}

std::string Cfg(const char* what, int K, int nout, int prescale, bool coarse = false) {
  return std::string(what) + " K " + std::to_string(K) + " nout " + std::to_string(nout) + " prescale " + std::to_string(prescale) + (coarse ? " [coarse]" : "");
}

// ---- A: the input transform -----------------------------------------------------------------------------------
void TestTransform(std::mt19937_64& rng, bool coarse) {
  for (int K : {3072, 5120, 6144, 8704, 17408}) {
    DeviceBuffer<uint16_t> d_x(static_cast<size_t>(kM) * K);
    d_x.CopyFromHost(RandomX(rng, kM, K));
    for (int nout : {1, 2, 3}) {
      DeviceBuffer<float> d_suh(static_cast<size_t>(nout) * K);
      d_suh.CopyFromHost(RandomScales(rng, static_cast<int64_t>(nout) * K));
      for (int prescale : {0, 4, -3}) {
        int64_t suh[3] = {}, out[3] = {};
        DeviceBuffer<uint16_t> d_a16(static_cast<size_t>(nout) * kM * K);
        Operand ref(nout, K, 0x5A, coarse), got(nout, K, 0xA5, coarse);
        int64_t g8[3] = {}, gsa[3] = {};
        for (int o = 0; o < nout; ++o) {
          suh[o] = P(d_suh.data() + static_cast<size_t>(o) * K);
          out[o] = P(d_a16.data() + static_cast<size_t>(o) * kM * K);
          g8[o] = got.A8(o);
          gsa[o] = got.SA(o);
        }
        r4dx_trellis_input_bf16(P(d_x.data()), kM, K, nout, suh, out, prescale, 0);
        QuantizeUnfused(d_a16, ref);
        (coarse ? r4dx_trellis_input_i8r : r4dx_trellis_input_i8)(P(d_x.data()), kM, K, nout, suh, g8, gsa, prescale, 0);
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        size_t b8 = 0, bs = 0;
        const bool same = SameBytes(ref, got, &b8, &bs);
        Check(same, Cfg("transform: fused == transform + separate quantizer", K, nout, prescale, coarse) + " (" + std::to_string(b8) +
                        " A8 bytes, " + std::to_string(bs) + " scales differ)");
        Check(got.CanariesIntact(), Cfg("transform: nothing written past the outputs", K, nout, prescale, coarse));
      }
    }
  }
}

// ---- B: silu_mul -> mlp.down's A ------------------------------------------------------------------------------------
void TestSilu(std::mt19937_64& rng, bool coarse) {
  for (int inter : {17408, 8704})
    for (int pad : {0, 128}) {
      const int stride = 2 * inter + pad;
      std::vector<uint16_t> gu = RandomX(rng, kM, stride);
      DeviceBuffer<uint16_t> d_gu(gu.size());
      d_gu.CopyFromHost(gu);
      DeviceBuffer<float> d_suh(static_cast<size_t>(inter));
      d_suh.CopyFromHost(RandomScales(rng, inter));
      for (int prescale : {0, 4}) {
        DeviceBuffer<uint16_t> d_a16(static_cast<size_t>(kM) * inter);
        Operand ref(1, inter, 0x5A, coarse), got(1, inter, 0xA5, coarse);
        r4dx_silu_mul_trellis_bf16(P(d_gu.data()), kM, inter, stride, P(d_suh.data()), P(d_a16.data()), prescale, 0);
        QuantizeUnfused(d_a16, ref);
        (coarse ? r4dx_silu_mul_trellis_i8r : r4dx_silu_mul_trellis_i8)(P(d_gu.data()), kM, inter, stride, P(d_suh.data()), got.A8(0), got.SA(0), prescale, 0);
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        size_t b8 = 0, bs = 0;
        const bool same = SameBytes(ref, got, &b8, &bs);
        Check(same, "silu_mul: fused == silu_mul_trellis + separate quantizer, intermediate " + std::to_string(inter) + " stride " +
                        std::to_string(stride) + " prescale " + std::to_string(prescale) + (coarse ? " [coarse]" : "") + " (" + std::to_string(b8) + " A8 bytes, " +
                        std::to_string(bs) + " scales differ)");
        Check(got.CanariesIntact(), "silu_mul: nothing written past the outputs, intermediate " + std::to_string(inter) + (coarse ? " [coarse]" : ""));
      }
    }
}

// ---- C: attention gate-mul -> attn.o's A ----------------------------------------------------------------------------
void TestGateMul(std::mt19937_64& rng, bool coarse) {
  for (int K : {6144, 3072}) {
    DeviceBuffer<uint16_t> d_a(static_cast<size_t>(kM) * K), d_g(static_cast<size_t>(kM) * K);
    d_a.CopyFromHost(RandomX(rng, kM, K));
    d_g.CopyFromHost(RandomX(rng, kM, K));
    DeviceBuffer<float> d_suh(static_cast<size_t>(K));
    d_suh.CopyFromHost(RandomScales(rng, K));
    for (int prescale : {0, 4, -3}) {
      DeviceBuffer<uint16_t> d_a16(static_cast<size_t>(kM) * K);
      Operand ref(1, K, 0x5A, coarse), got(1, K, 0xA5, coarse);
      r4dx_attn_gate_mul_trellis_bf16(P(d_a.data()), P(d_g.data()), kM, K, P(d_suh.data()), P(d_a16.data()), prescale, 0);
      QuantizeUnfused(d_a16, ref);
      (coarse ? r4dx_attn_gate_mul_trellis_i8r : r4dx_attn_gate_mul_trellis_i8)(P(d_a.data()), P(d_g.data()), kM, K, P(d_suh.data()), got.A8(0), got.SA(0), prescale, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      size_t b8 = 0, bs = 0;
      const bool same = SameBytes(ref, got, &b8, &bs);
      Check(same, Cfg("gate_mul: fused == gate_mul_trellis + separate quantizer", K, 1, prescale, coarse) + " (" + std::to_string(b8) +
                      " A8 bytes, " + std::to_string(bs) + " scales differ)");
      Check(got.CanariesIntact(), Cfg("gate_mul: nothing written past the outputs", K, 1, prescale, coarse));
    }
  }
}

// ---- D: the comparison can fail; E: the preconditions ----------------------------------------------------------------
void TestNegativeAndPreconditions(std::mt19937_64& rng, bool coarse) {
  const char* tag = coarse ? " [coarse]" : "";
  const auto Run8 = [coarse](auto... a) { (coarse ? r4dx_trellis_input_i8r : r4dx_trellis_input_i8)(a...); };
  const auto RunSilu = [coarse](auto... a) { (coarse ? r4dx_silu_mul_trellis_i8r : r4dx_silu_mul_trellis_i8)(a...); };
  const auto RunGate = [coarse](auto... a) { (coarse ? r4dx_attn_gate_mul_trellis_i8r : r4dx_attn_gate_mul_trellis_i8)(a...); };
  const int K = 5120;
  DeviceBuffer<uint16_t> d_x(static_cast<size_t>(kM) * K);
  d_x.CopyFromHost(RandomX(rng, kM, K));
  DeviceBuffer<float> d_suh(static_cast<size_t>(K));
  d_suh.CopyFromHost(RandomScales(rng, K));
  DeviceBuffer<uint16_t> d_a16(static_cast<size_t>(kM) * K);
  Operand ref(1, K, 0x5A, coarse), got(1, K, 0xA5, coarse);
  int64_t suh[1] = {P(d_suh.data())}, out[1] = {P(d_a16.data())}, g8[1] = {got.A8(0)}, gsa[1] = {got.SA(0)};
  r4dx_trellis_input_bf16(P(d_x.data()), kM, K, 1, suh, out, 0, 0);
  QuantizeUnfused(d_a16, ref);
  Run8(P(d_x.data()), kM, K, 1, suh, g8, gsa, 0, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(SameBytes(ref, got), std::string("negative control setup: the fused and unfused operands start equal") + tag);
  int8_t flipped = 0;
  R4DX_HIP_CHECK(hipMemcpy(&flipped, got.a8.data() + 12345, 1, hipMemcpyDeviceToHost));
  flipped = static_cast<int8_t>(flipped ^ 1);
  R4DX_HIP_CHECK(hipMemcpy(got.a8.data() + 12345, &flipped, 1, hipMemcpyHostToDevice));
  Check(!SameBytes(ref, got), std::string("negative control: one flipped A8 byte is seen by the comparison") + tag);
  Check(Throws([&] { Run8(P(d_x.data()), 255, K, 1, suh, g8, gsa, 0, 0); }), std::string("255 rows throws (the A8 layout is 256)") + tag);
  Check(Throws([&] { Run8(P(d_x.data()), 64, K, 1, suh, g8, gsa, 0, 0); }), std::string("64 rows throws") + tag);
  const int64_t null8[1] = {0};
  Check(Throws([&] { Run8(P(d_x.data()), kM, K, 1, suh, null8, gsa, 0, 0); }), std::string("a null a8 throws") + tag);
  Check(Throws([&] { Run8(P(d_x.data()), kM, K, 1, suh, g8, null8, 0, 0); }), std::string("a null sa throws") + tag);
  Check(Throws([&] { Run8(P(d_x.data()), kM, K + 64, 1, suh, g8, gsa, 0, 0); }), std::string("K not a multiple of 128 throws") + tag);
  Check(Throws([&] { RunSilu(P(d_x.data()), 128, 4096, 8192, P(d_suh.data()), g8[0], gsa[0], 0, 0); }),
        std::string("silu_mul: 128 rows throws") + tag);
  Check(Throws([&] { RunSilu(P(d_x.data()), kM, 4096, 8191, P(d_suh.data()), g8[0], gsa[0], 0, 0); }),
        std::string("silu_mul: a row stride below 2 * intermediate throws") + tag);
  Check(Throws([&] { RunGate(P(d_x.data()), P(d_x.data()), 100, 4096, P(d_suh.data()), g8[0], gsa[0], 0, 0); }),
        std::string("gate_mul: 100 rows throws") + tag);
  Check(Throws([&] { RunGate(P(d_x.data()), 0, kM, 4096, P(d_suh.data()), g8[0], gsa[0], 0, 0); }),
        std::string("gate_mul: a null gate throws") + tag);
  if (coarse) {
    // the row buffer must fit in LDS: nout * K * 2 + 64 <= 64 KiB (3 x 17408 does not)
    DeviceBuffer<float> d_suh3(static_cast<size_t>(3) * 17408);
    DeviceBuffer<uint16_t> d_big(static_cast<size_t>(kM) * 17408);
    Operand big(3, 17408, 0x11, true);
    int64_t s3[3] = {P(d_suh3.data()), P(d_suh3.data() + 17408), P(d_suh3.data() + 2 * 17408)};
    int64_t b3[3] = {big.A8(0), big.A8(1), big.A8(2)}, sa3[3] = {big.SA(0), big.SA(1), big.SA(2)};
    Check(Throws([&] { r4dx_trellis_input_i8r(P(d_big.data()), kM, 17408, 3, s3, b3, sa3, 0, 0); }), "i8r: three outputs of K 17408 do not fit the row buffer in LDS and throw");
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  if (hipSetDevice(0) != hipSuccess) {
    std::printf("SKIP: hipSetDevice(0) failed\n");
    return 77;
  }
  try {
    std::mt19937_64 rng(20261007);
    for (const bool coarse : {false, true}) {   // the per-128 entries, then the per-row ones (R4DX_PREFILL_INT8_SCALES=coarse)
      TestTransform(rng, coarse);
      TestSilu(rng, coarse);
      TestGateMul(rng, coarse);
      TestNegativeAndPreconditions(rng, coarse);
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::printf("test_trellis_input_i8: %d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_trellis_input_i8: PASS (%d checks)\n", g_checks);
  return 0;
}
