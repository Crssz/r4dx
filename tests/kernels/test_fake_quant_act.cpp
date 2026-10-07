// tests/kernels/test_fake_quant_act.cpp -- r4dx_fake_quant_act_f16 (R4DX_FAKEQ_ACT, docs/int8-prefill.md), bit
// for bit against src/model/fake_quant_act.h's fp32 reference (FakeQuantRowRef) followed by the f16 rounding the
// kernel does (FloatToF16, round to nearest even):
//   - modes row / blk128 / blk32, parts 1 and 2 (the second part part_stride elements after the first, with a
//     gap the kernel must leave alone), rows 1 / 7 / 64 / 256, K 128 / 5120 / 17408;
//   - activations as the rotated A looks: gaussian rows over a wide scale range with outliers, plus a
//     zero row, a row whose only live element is an f16 subnormal, a row with a non-finite element (left
//     untouched), and a row of one huge element among small ones;
//   - the precondition throws (hence /EHc-).
// GPU test, tiny (about a second): HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt); skips
// (77) without a HIP device.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fake_quant_act.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;
using namespace r4dx::model;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    if (g_failures < 40) std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
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

// One f16 row: gaussian at a log-uniform scale in [2^-6, 2^4] with two 10-40x outliers, as a rotated A has.
void FillRow(std::mt19937_64& rng, uint16_t* row, int K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> sd(-6.f, 4.f), od(10.f, 40.f);
  std::uniform_int_distribution<int> kd(0, K - 1);
  const float s = std::exp2(sd(rng));
  std::vector<float> v(static_cast<size_t>(K));
  for (auto& e : v) e = nd(rng) * s;
  for (int i = 0; i < 2; ++i) v[static_cast<size_t>(kd(rng))] *= od(rng);
  for (int k = 0; k < K; ++k) row[k] = FloatToF16(v[static_cast<size_t>(k)]);
}

// The reference for one row of f16 bits.
void RefRow(const uint16_t* in, int K, int mode, uint16_t* out) {
  std::vector<float> x(static_cast<size_t>(K)), v(static_cast<size_t>(K));
  for (int k = 0; k < K; ++k) x[static_cast<size_t>(k)] = F16ToFloat(in[k]);
  FakeQuantRowRef(x.data(), K, mode, v.data());
  for (int k = 0; k < K; ++k) {
    // A group that is passed through keeps its bits (a NaN's payload included); everything else is the f16
    // rounding of the fp32 value.
    out[k] = FloatToF16(v[static_cast<size_t>(k)]);
  }
}

void RunCase(std::mt19937_64& rng, int rows, int K, int parts, int mode) {
  const int64_t part_stride = static_cast<int64_t>(rows) * K + 64;  // a gap after part 0
  const size_t total = static_cast<size_t>(part_stride) * parts + 64;
  const uint16_t kSentinel = 0x5A5A;
  std::vector<uint16_t> h(total, kSentinel);
  for (int p = 0; p < parts; ++p) {
    for (int r = 0; r < rows; ++r) {
      uint16_t* row = h.data() + p * part_stride + static_cast<int64_t>(r) * K;
      FillRow(rng, row, K);
      // Special rows (only where there is room for them to be distinct).
      if (rows >= 7 && r == 2) {
        for (int k = 0; k < K; ++k) row[k] = (k & 1) ? 0x8000 : 0;  // zeros and -0
      } else if (rows >= 7 && r == 3) {
        for (int k = 0; k < K; ++k) row[k] = 0;
        row[K / 3] = 0x0001;  // the smallest f16 subnormal, alone in its group
      } else if (rows >= 7 && r == 4) {
        row[K - 1] = 0x7C00;  // +inf in the last block: its group is untouched, the others are quantized
      } else if (rows >= 7 && r == 5) {
        for (int k = 0; k < K; ++k) row[k] = FloatToF16(0.01f * static_cast<float>((k % 7) - 3));
        row[1] = FloatToF16(60000.f);  // one huge element among small ones: the rest flush to zero
      }
    }
  }
  DeviceBuffer<uint16_t> d(total);
  d.CopyFromHost(h);
  r4dx_fake_quant_act_f16(reinterpret_cast<int64_t>(d.data()), rows, K, parts, parts > 1 ? part_stride : 0,
                          mode, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> got = d.CopyToHost();

  std::vector<uint16_t> want(static_cast<size_t>(K));
  size_t bad = 0, bad_gap = 0;
  int first_bad_row = -1;
  for (int p = 0; p < parts; ++p) {
    for (int r = 0; r < rows; ++r) {
      const uint16_t* in = h.data() + p * part_stride + static_cast<int64_t>(r) * K;
      const uint16_t* g = got.data() + p * part_stride + static_cast<int64_t>(r) * K;
      RefRow(in, K, mode, want.data());
      for (int k = 0; k < K; ++k) {
        // An all-zero group is +0, also where the input was -0 (FloatToF16(+0) is 0x0000, as the kernel's).
        if (g[k] != want[static_cast<size_t>(k)]) {
          ++bad;
          if (first_bad_row < 0) first_bad_row = r;
        }
      }
    }
  }
  // Everything outside the parts' rows is untouched.
  for (size_t i = 0; i < total; ++i) {
    const int64_t off = static_cast<int64_t>(i);
    bool inside = false;
    for (int p = 0; p < parts; ++p) {
      if (off >= p * part_stride && off < p * part_stride + static_cast<int64_t>(rows) * K) inside = true;
    }
    if (!inside && got[i] != h[i]) ++bad_gap;
  }
  const std::string tag = "rows " + std::to_string(rows) + " K " + std::to_string(K) + " parts " +
                          std::to_string(parts) + " mode " + FakeQuantActName(mode);
  Check(bad == 0, tag + ": " + std::to_string(bad) + " elements differ from the reference (first row " +
                      std::to_string(first_bad_row) + ")");
  Check(bad_gap == 0, tag + ": " + std::to_string(bad_gap) + " elements outside the rows were written");
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
  std::mt19937_64 rng(20261007);
  for (int mode : {kFakeQuantRow, kFakeQuantBlk128, kFakeQuantBlk32}) {
    for (int rows : {1, 7, 64, 256}) {
      for (int K : {128, 5120, 17408}) {
        for (int parts : {1, 2}) RunCase(rng, rows, K, parts, mode);
      }
    }
  }

  // Preconditions.
  DeviceBuffer<uint16_t> d(128 * 4);
  const int64_t p = reinterpret_cast<int64_t>(d.data());
  Check(Throws([&] { r4dx_fake_quant_act_f16(p, 1, 128, 1, 0, 0, 0); }), "mode 0 throws");
  Check(Throws([&] { r4dx_fake_quant_act_f16(p, 1, 128, 1, 0, 4, 0); }), "mode 4 throws");
  Check(Throws([&] { r4dx_fake_quant_act_f16(p, 1, 100, 1, 0, 2, 0); }), "K not a multiple of 128 throws");
  Check(Throws([&] { r4dx_fake_quant_act_f16(p, 1, 128, 3, 0, 2, 0); }), "parts 3 throws");
  Check(Throws([&] { r4dx_fake_quant_act_f16(p, 2, 128, 2, 128, 2, 0); }), "part_stride < rows * K throws");
  Check(Throws([&] { r4dx_fake_quant_act_f16(0, 1, 128, 1, 0, 2, 0); }), "null a throws");
  Check(!Throws([&] { r4dx_fake_quant_act_f16(p, 0, 128, 1, 0, 2, 0); }), "rows 0 is a no-op");

  if (g_failures != 0) {
    std::printf("test_fake_quant_act: %d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_fake_quant_act: PASS (%d checks)\n", g_checks);
  return 0;
}
