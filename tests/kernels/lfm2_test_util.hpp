// tests/kernels/lfm2_test_util.hpp -- small shared helpers of the GPU tests test_lfm2_* (host + HIP runtime;
// the CPU references are lfm2_ref.hpp).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "lfm2_ref.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"

namespace lfm2_test {

inline bool g_ok = true;
inline void Check(bool cond, const char* what) {
  std::printf("%-100s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
inline void CheckF(bool cond, const std::string& what) { Check(cond, what.c_str()); }

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

// bf16-representable floats <-> the device's uint16 bits.
inline std::vector<uint16_t> ToBits(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = r4dx::core::FloatToBf16(v[i]);
  return o;
}
inline std::vector<float> FromBits(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = r4dx::core::Bf16ToFloat(v[i]);
  return o;
}
inline std::vector<float> RandBf16(std::mt19937& rng, size_t n, float mag) {
  std::uniform_real_distribution<float> d(-mag, mag);
  std::vector<float> v(n);
  for (auto& x : v) x = lfm2_ref::Bf16Round(d(rng));
  return v;
}

// |a - b| <= 1 bf16 ulp of b (relative 2^-7), with a tiny floor.
inline bool WithinUlp(float a, float b, float ulps = 1.0f) {
  return std::fabs(a - b) <= std::fabs(b) * (ulps / 128.0f) + 1e-30f;
}

// Number of elements that differ at all / by more than `ulps` bf16 ulps.
struct Diff {
  size_t n = 0, differ = 0, over = 0;
};
inline Diff Compare(const std::vector<float>& got, const std::vector<float>& want, float ulps) {
  Diff d;
  d.n = got.size();
  for (size_t i = 0; i < got.size(); ++i) {
    d.differ += got[i] != want[i];
    d.over += !WithinUlp(got[i], want[i], ulps);
  }
  return d;
}

}  // namespace lfm2_test
