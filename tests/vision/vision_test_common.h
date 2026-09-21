// tests/vision/vision_test_common.h -- shared plumbing for the CPU-only src/vision golden tests.
// Same shape as tests/model/test_common.h (reuse r4dx_convert::SafetensorsReader rather than write
// a third .safetensors parser, exit 77 for "the real golden data is not on this machine"), minus
// everything device-related: nothing under tests/vision touches HIP, so these tests always run.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx_vision_test {

inline int g_failures = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                                    \
      ++r4dx_vision_test::g_failures;                                         \
    }                                                                         \
  } while (0)

constexpr int kSkipReturnCode = 77;

inline bool FileExists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

inline int SkipMissing(const std::string& what) {
  std::fprintf(stderr,
               "[SKIP] %s not found; this test needs golden data produced from the real checkpoint "
               "by tools/reference/vision_golden.py (see docs/vision.md / docs/validation.md).\n",
               what.c_str());
  return kSkipReturnCode;
}

inline std::vector<float> ReadFloat(const r4dx_convert::SafetensorsReader& r,
                                     const std::string& name) {
  const auto& m = r.Meta(name);
  const int64_t n = m.ElemCount();
  std::vector<float> out(static_cast<size_t>(n));
  if (m.dtype == "F32") {
    std::memcpy(out.data(), r.Data(name), static_cast<size_t>(n) * 4);
  } else if (m.dtype == "BF16") {
    const auto* src = reinterpret_cast<const uint16_t*>(r.Data(name));
    for (int64_t i = 0; i < n; ++i) {
      out[static_cast<size_t>(i)] = r4dx::core::Bf16ToFloat(src[static_cast<size_t>(i)]);
    }
  } else {
    throw std::runtime_error("ReadFloat: unsupported dtype " + m.dtype + " for " + name);
  }
  return out;
}

// I64 and I32 golden tensors (index/position/cu_seqlens tables) both widen to int64 here so a
// caller never has to care which the reference happened to emit.
inline std::vector<int64_t> ReadInt(const r4dx_convert::SafetensorsReader& r,
                                     const std::string& name) {
  const auto& m = r.Meta(name);
  const int64_t n = m.ElemCount();
  std::vector<int64_t> out(static_cast<size_t>(n));
  if (m.dtype == "I64") {
    const auto* src = reinterpret_cast<const int64_t*>(r.Data(name));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = src[static_cast<size_t>(i)];
  } else if (m.dtype == "I32") {
    const auto* src = reinterpret_cast<const int32_t*>(r.Data(name));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = src[static_cast<size_t>(i)];
  } else {
    throw std::runtime_error("ReadInt: unsupported dtype " + m.dtype + " for " + name);
  }
  return out;
}

inline std::vector<uint8_t> ReadU8(const r4dx_convert::SafetensorsReader& r,
                                    const std::string& name) {
  const auto& m = r.Meta(name);
  if (m.dtype != "U8") throw std::runtime_error("ReadU8: '" + name + "' is " + m.dtype + ", not U8");
  const uint8_t* src = r.Data(name);
  return std::vector<uint8_t>(src, src + m.ElemCount());
}

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  int64_t num_differing = 0;
  int64_t count = 0;
  int64_t argmax = -1;
};

inline ErrorStats CompareFloat(const std::vector<float>& got, const std::vector<float>& ref) {
  if (got.size() != ref.size()) throw std::runtime_error("CompareFloat: size mismatch");
  ErrorStats s;
  s.count = static_cast<int64_t>(ref.size());
  double sum = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    sum += d;
    if (d > 0.0) ++s.num_differing;
    if (d > s.max_abs) {
      s.max_abs = d;
      s.argmax = static_cast<int64_t>(i);
    }
  }
  s.mean_abs = s.count > 0 ? sum / static_cast<double>(s.count) : 0.0;
  return s;
}

}  // namespace r4dx_vision_test
