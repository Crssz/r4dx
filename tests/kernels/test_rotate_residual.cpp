// tests/kernels/test_rotate_residual.cpp -- quant2's r4dx_kernels rotation ops against the fp64
// CPU reference in rotation_ref.hpp (itself pinned to the contract by test_rotation_ref_cpu):
//
//   r4dx_rotate_residual_bf16      x <- x Q and x <- x Q^T, rows 1 / 3 / 7, plus the round trip
//   r4dx_hadamard_inplace_bf16     gdn.out_proj's input: K 6144 (TP=1) and 3072 (TP=2 rank), B 128
//   r4dx_silu_mul_hadamard_bf16    mlp.down's input: intermediate 17408 (TP=1) and 8704 (TP=2
//                                  rank), B 512, both epilogues (none, f16)
//
// and the properties the model relies on beyond the values themselves, all BIT-exact:
//   - row independence: a row's output is identical whether it is launched alone or with others
//     (decode vs prefill vs verify windows must agree; see hadamard_device.h);
//   - TP slicing: a TP=2 rank running its own K-slice with its own sign slice reproduces exactly
//     its slice of the full-K output (no Hadamard block straddles the rank boundary);
//   - epilogues: silu_mul_hadamard's bf16 output does not change when an epilogue is requested,
//     and the f16 epilogue's bytes equal the standalone cast of that bf16 output (the
//     r4dx_epilogue contract, kernels.h -- as tests/kernels/test_fused_quant.cpp does for silu_mul);
//   - launch accounting: each call adds exactly one to r4dx_kernel_launch_counter_get();
//   - precondition throws (hence /EHc- in CMakeLists.txt).
// Tolerance for values: rotation_ref::CountOutOfTolerance (one bf16 rounding + fp32 noise).
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt), always on.
#include <hip/hip_runtime.h>

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
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/rotate_residual.h"
#include "rotation_ref.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

template <typename T>
int64_t P(T* p) {
  return reinterpret_cast<int64_t>(p);
}

// Residual-like bf16 values: N(0, 1) with a few large "outlier channel" entries (every 97th column
// x 24), the shape rotation exists to spread out.
std::vector<uint16_t> RandomResidualBf16(std::mt19937_64& rng, int64_t rows, int64_t K) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> v(static_cast<size_t>(rows * K));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t k = 0; k < K; ++k) {
      float x = nd(rng);
      if (k % 97 == 13) x *= 24.0f;
      v[r * K + k] = FloatToBf16(x);
    }
  }
  return v;
}

std::vector<uint16_t> RandomUniformBf16(std::mt19937_64& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(dist(rng));
  return v;
}

std::vector<double> RowToDouble(const std::vector<uint16_t>& v, int64_t row, int64_t K) {
  std::vector<double> out(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) out[k] = Bf16ToFloat(v[row * K + k]);
  return out;
}

std::vector<float> RowToFloat(const std::vector<uint16_t>& v, int64_t row, int64_t K) {
  std::vector<float> out(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) out[k] = Bf16ToFloat(v[row * K + k]);
  return out;
}

void CheckRowAgainstRef(const std::vector<uint16_t>& got, int64_t row, int64_t K,
                        const std::vector<double>& ref, const std::string& label) {
  double worst = 0.0;
  const int64_t bad = rotation_ref::CountOutOfTolerance(RowToFloat(got, row, K), ref, &worst);
  if (row == 0) std::printf("  %-52s worst err/bound %.3f\n", label.c_str(), worst);
  Check(bad == 0, label + " row " + std::to_string(row) + ": " + std::to_string(bad) +
                      " elements out of tolerance (worst err/bound " + std::to_string(worst) + ")");
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

// ---- r4dx_rotate_residual_bf16 -------------------------------------------------------------------
void TestRotateResidual(std::mt19937_64& rng) {
  using rotation_ref::kHidden;
  const std::vector<float> d = rotation_ref::RandomSigns(rng, kHidden);
  const std::vector<float> R = rotation_ref::RandomOrthogonal5(rng);
  DeviceBuffer<float> d_d(kHidden), R_d(25);
  d_d.CopyFromHost(d);
  R_d.CopyFromHost(R);

  for (int64_t rows : {1, 3, 7}) {
    const std::vector<uint16_t> x = RandomResidualBf16(rng, rows, kHidden);
    for (int inverse = 0; inverse < 2; ++inverse) {
      const std::string name = std::string(inverse ? "rotate Q^T" : "rotate Q") +
                               " rows=" + std::to_string(rows);
      DeviceBuffer<uint16_t> x_d(x.size());
      x_d.CopyFromHost(x);
      const int64_t before = r4dx_kernel_launch_counter_get();
      r4dx_rotate_residual_bf16(P(x_d.data()), rows, kHidden, P(d_d.data()), P(R_d.data()),
                                 inverse, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      Check(r4dx_kernel_launch_counter_get() - before == 1, name + ": launch count != 1");
      const std::vector<uint16_t> got = x_d.CopyToHost();
      for (int64_t r = 0; r < rows; ++r) {
        CheckRowAgainstRef(got, r, kHidden,
                           rotation_ref::ApplyQ(RowToDouble(x, r, kHidden), d.data(), R.data(),
                                                inverse != 0),
                           name);
      }

      // Row independence + determinism: each row again, alone (rows = 1 at its own offset).
      DeviceBuffer<uint16_t> y_d(x.size());
      y_d.CopyFromHost(x);
      for (int64_t r = 0; r < rows; ++r) {
        r4dx_rotate_residual_bf16(P(y_d.data() + r * kHidden), 1, kHidden, P(d_d.data()),
                                   P(R_d.data()), inverse, 0);
      }
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      Check(y_d.CopyToHost() == got, name + ": rows launched one at a time differ from the batch");
    }

    // Round trip: Q then Q^T on the device returns x up to two bf16 roundings.
    DeviceBuffer<uint16_t> x_d(x.size());
    x_d.CopyFromHost(x);
    r4dx_rotate_residual_bf16(P(x_d.data()), rows, kHidden, P(d_d.data()), P(R_d.data()), 0, 0);
    r4dx_rotate_residual_bf16(P(x_d.data()), rows, kHidden, P(d_d.data()), P(R_d.data()), 1, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint16_t> back = x_d.CopyToHost();
    for (int64_t r = 0; r < rows; ++r) {
      double num = 0.0, den = 0.0;
      for (int64_t k = 0; k < kHidden; ++k) {
        const double a = Bf16ToFloat(back[r * kHidden + k]);
        const double b = Bf16ToFloat(x[r * kHidden + k]);
        num += (a - b) * (a - b);
        den += b * b;
      }
      const double rel = std::sqrt(num / den);
      if (r == 0) std::printf("  %-52s rel L2 %.3e\n", ("round trip Q Q^T rows=" + std::to_string(rows)).c_str(), rel);
      // Each bf16 rounding contributes ~2^-9 / sqrt(3) relative RMS; two of them stay well
      // below 4e-3 unless something is structurally wrong (a wrong transform is O(1)).
      Check(rel < 4e-3, "round trip Q Q^T rows=" + std::to_string(rows) + " rel L2 " +
                            std::to_string(rel));
    }
  }

  Check(Throws([&] {
          r4dx_rotate_residual_bf16(0, 1, 4096, P(d_d.data()), P(R_d.data()), 0, 0);
        }),
        "rotate: hidden != 5120 must throw");
  Check(Throws([&] { r4dx_rotate_residual_bf16(1, 1, kHidden, 0, P(R_d.data()), 0, 0); }),
        "rotate: null signs must throw");
  // rows = 0 is a no-op, not a launch.
  const int64_t before = r4dx_kernel_launch_counter_get();
  r4dx_rotate_residual_bf16(0, 0, kHidden, P(d_d.data()), P(R_d.data()), 0, 0);
  Check(r4dx_kernel_launch_counter_get() == before, "rotate: rows=0 counted a launch");
}

// ---- r4dx_hadamard_inplace_bf16 ------------------------------------------------------------------
void TestHadamardInplace(std::mt19937_64& rng) {
  struct Case {
    int64_t K;
    int block;
    const char* what;
  };
  // gdn.out_proj at TP=1 (48 heads x 128) and one TP=2 rank (24 x 128); plus the other two block
  // sizes the contract uses, since the entry is generic.
  for (const Case& c : {Case{6144, 128, "gdn_out TP=1"}, Case{3072, 128, "gdn_out TP=2 rank"},
                        Case{6144, 256, "B=256"}, Case{8704, 512, "B=512"}}) {
    const std::vector<float> s = rotation_ref::RandomSigns(rng, c.K);
    DeviceBuffer<float> s_d(s.size());
    s_d.CopyFromHost(s);
    for (int64_t rows : {1, 5}) {
      const std::string name = std::string("hadamard_inplace ") + c.what + " rows=" +
                               std::to_string(rows);
      const std::vector<uint16_t> x = RandomResidualBf16(rng, rows, c.K);
      DeviceBuffer<uint16_t> x_d(x.size());
      x_d.CopyFromHost(x);
      const int64_t before = r4dx_kernel_launch_counter_get();
      r4dx_hadamard_inplace_bf16(P(x_d.data()), rows, c.K, P(s_d.data()), c.block, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      Check(r4dx_kernel_launch_counter_get() - before == 1, name + ": launch count != 1");
      const std::vector<uint16_t> got = x_d.CopyToHost();
      for (int64_t r = 0; r < rows; ++r) {
        CheckRowAgainstRef(got, r, c.K, rotation_ref::ApplyHb(RowToDouble(x, r, c.K), s.data(), c.block),
                           name);
      }

      // TP=2 slicing: rank 1's half (its own K, its own sign slice) == the full result's half.
      const int64_t half = c.K / 2;
      if (half % c.block == 0) {
        std::vector<uint16_t> xr(static_cast<size_t>(rows * half));
        for (int64_t r = 0; r < rows; ++r) {
          std::memcpy(&xr[r * half], &x[r * c.K + half], sizeof(uint16_t) * half);
        }
        DeviceBuffer<uint16_t> xr_d(xr.size());
        xr_d.CopyFromHost(xr);
        r4dx_hadamard_inplace_bf16(P(xr_d.data()), rows, half, P(s_d.data() + half), c.block, 0);
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        const std::vector<uint16_t> part = xr_d.CopyToHost();
        bool same = true;
        for (int64_t r = 0; r < rows && same; ++r) {
          same = std::memcmp(&part[r * half], &got[r * c.K + half], sizeof(uint16_t) * half) == 0;
        }
        Check(same, name + ": TP rank-1 K-slice differs from the full-K result's slice");
      }
    }
  }

  DeviceBuffer<float> s_d(6144);
  Check(Throws([&] { r4dx_hadamard_inplace_bf16(1, 1, 6144, P(s_d.data()), 384, 0); }),
        "hadamard_inplace: non-power-of-two block must throw");
  Check(Throws([&] { r4dx_hadamard_inplace_bf16(1, 1, 6144, P(s_d.data()), 2048, 0); }),
        "hadamard_inplace: block > 1024 must throw");
  Check(Throws([&] { r4dx_hadamard_inplace_bf16(1, 1, 6144 + 64, P(s_d.data()), 128, 0); }),
        "hadamard_inplace: K % block != 0 must throw");
}

// ---- r4dx_silu_mul_hadamard_bf16 -----------------------------------------------------------------
std::vector<double> SiluMulRow(const std::vector<uint16_t>& gate_up, int64_t row, int64_t I,
                               int64_t stride) {
  std::vector<double> out(static_cast<size_t>(I));
  for (int64_t k = 0; k < I; ++k) {
    const double g = Bf16ToFloat(gate_up[row * stride + k]);
    const double u = Bf16ToFloat(gate_up[row * stride + I + k]);
    out[k] = g / (1.0 + std::exp(-g)) * u;
  }
  return out;
}

// Runs the kernel; returns the bf16 out and (for epilogue != none) the epilogue bytes.
std::vector<uint16_t> RunSiluHad(const DeviceBuffer<uint16_t>& gu_d, int64_t rows, int64_t I,
                                 int64_t stride, const float* signs_dev, int block, int epilogue,
                                 std::vector<uint8_t>* epi_bytes) {
  DeviceBuffer<uint16_t> out_d(static_cast<size_t>(rows * I));
  const int elem = 2;  // f16, the only epilogue with an output
  DeviceBuffer<uint8_t> eo_d(epilogue != r4dx_epilogue_none ? static_cast<size_t>(rows * I * elem) : 0);
  const int64_t before = r4dx_kernel_launch_counter_get();
  r4dx_silu_mul_hadamard_bf16(P(const_cast<uint16_t*>(gu_d.data())), P(out_d.data()), rows, I,
                               stride, 0, epilogue, P(eo_d.data()),
                               P(const_cast<float*>(signs_dev)), block);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(r4dx_kernel_launch_counter_get() - before == 1, "silu_mul_hadamard: launch count != 1");
  if (epi_bytes != nullptr && epilogue != r4dx_epilogue_none) *epi_bytes = eo_d.CopyToHost();
  return out_d.CopyToHost();
}

void TestSiluMulHadamard(std::mt19937_64& rng) {
  constexpr int kBlockDown = 512;
  struct Case {
    int64_t I;
    int64_t pad;  // extra columns per gate_up row (in_row_stride = 2*I + pad)
    const char* what;
  };
  for (const Case& c : {Case{17408, 0, "down TP=1"}, Case{8704, 0, "down TP=2 rank"},
                        Case{17408, 64, "down TP=1, padded stride"}}) {
    const std::vector<float> s = rotation_ref::RandomSigns(rng, c.I);
    DeviceBuffer<float> s_d(s.size());
    s_d.CopyFromHost(s);
    const int64_t stride = 2 * c.I + c.pad;
    for (int64_t rows : {1, 4}) {
      const std::string name = std::string("silu_mul_hadamard ") + c.what + " rows=" +
                               std::to_string(rows);
      const std::vector<uint16_t> gu = RandomUniformBf16(rng, static_cast<size_t>(rows * stride), -4.0f, 4.0f);
      DeviceBuffer<uint16_t> gu_d(gu.size());
      gu_d.CopyFromHost(gu);

      const std::vector<uint16_t> base =
          RunSiluHad(gu_d, rows, c.I, stride, s_d.data(), kBlockDown, r4dx_epilogue_none, nullptr);
      for (int64_t r = 0; r < rows; ++r) {
        CheckRowAgainstRef(base, r, c.I,
                           rotation_ref::ApplyHb(SiluMulRow(gu, r, c.I, stride), s.data(), kBlockDown),
                           name);
      }

      // Row independence: row r alone == row r of the batch.
      for (int64_t r = 0; r < rows && rows > 1; ++r) {
        std::vector<uint16_t> one(gu.begin() + r * stride, gu.begin() + (r + 1) * stride);
        DeviceBuffer<uint16_t> one_d(one.size());
        one_d.CopyFromHost(one);
        const std::vector<uint16_t> o =
            RunSiluHad(one_d, 1, c.I, stride, s_d.data(), kBlockDown, r4dx_epilogue_none, nullptr);
        Check(std::memcmp(o.data(), &base[r * c.I], sizeof(uint16_t) * c.I) == 0,
              name + ": row " + std::to_string(r) + " launched alone differs from the batch");
      }

      // Epilogue: bf16 out unchanged; bytes == the standalone f16 cast of that bf16 out. (The fp8 and
      // int8 epilogues went with the mxfp4 and w4a8 layouts.)
      for (int epi : {r4dx_epilogue_f16}) {
        const std::string en = name + " epilogue=" + std::to_string(epi);
        std::vector<uint8_t> bytes;
        const std::vector<uint16_t> out =
            RunSiluHad(gu_d, rows, c.I, stride, s_d.data(), kBlockDown, epi, &bytes);
        Check(out == base, en + ": bf16 output changed when an epilogue was requested");
        std::vector<uint8_t> want(base.size() * 2);
        for (size_t i = 0; i < base.size(); ++i) {
          const uint16_t h = FloatToF16(Bf16ToFloat(base[i]));
          std::memcpy(&want[2 * i], &h, 2);
        }
        Check(bytes == want, en + ": epilogue bytes differ from the standalone cast of the bf16 out");
      }

      // TP=2 slicing: rank 1 holds gate[I/2:] | up[I/2:] per row and the sign slice [I/2, I).
      const int64_t half = c.I / 2;
      if (half % kBlockDown == 0) {
        std::vector<uint16_t> gr(static_cast<size_t>(rows * 2 * half));
        for (int64_t r = 0; r < rows; ++r) {
          std::memcpy(&gr[r * 2 * half], &gu[r * stride + half], sizeof(uint16_t) * half);
          std::memcpy(&gr[r * 2 * half + half], &gu[r * stride + c.I + half], sizeof(uint16_t) * half);
        }
        DeviceBuffer<uint16_t> gr_d(gr.size());
        gr_d.CopyFromHost(gr);
        const std::vector<uint16_t> part = RunSiluHad(gr_d, rows, half, 2 * half, s_d.data() + half,
                                                      kBlockDown, r4dx_epilogue_none, nullptr);
        bool same = true;
        for (int64_t r = 0; r < rows && same; ++r) {
          same = std::memcmp(&part[r * half], &base[r * c.I + half], sizeof(uint16_t) * half) == 0;
        }
        Check(same, name + ": TP rank-1 K-slice differs from the full-K result's slice");
      }
    }
  }

  DeviceBuffer<float> s_d(17408);
  Check(Throws([&] {
          r4dx_silu_mul_hadamard_bf16(1, 1, 1, 17408, 2 * 17408, 0, r4dx_epilogue_none, 0,
                                       P(s_d.data()), 384);
        }),
        "silu_mul_hadamard: non-power-of-two block must throw");
  Check(Throws([&] {
          r4dx_silu_mul_hadamard_bf16(1, 1, 1, 17408 + 256, 2 * (17408 + 256), 0,
                                       r4dx_epilogue_none, 0, P(s_d.data()), 512);
        }),
        "silu_mul_hadamard: intermediate % block != 0 must throw");
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937_64 rng(20260925);

  std::printf("r4dx_rotate_residual_bf16\n");
  TestRotateResidual(rng);
  std::printf("r4dx_hadamard_inplace_bf16\n");
  TestHadamardInplace(rng);
  std::printf("r4dx_silu_mul_hadamard_bf16\n");
  TestSiluMulHadamard(rng);

  if (g_failures == 0) {
    std::printf("PASS (%d checks)\n", g_checks);
    return 0;
  }
  std::printf("FAIL (%d of %d checks)\n", g_failures, g_checks);
  return 1;
}
