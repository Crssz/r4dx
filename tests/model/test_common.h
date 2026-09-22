// tests/model/test_common.h -- shared golden-file plumbing for tests/model/*.cpp. Both golden
// tests need the same three things: a way to read one tensor out of a plain .safetensors file
// (tools/reference/layer_golden.py's output -- BF16 or F32, never the r4dx container's own U8
// packing), a device upload of that tensor, and a relative-L2 comparison against a device result.
// Reuses r4dx_convert::SafetensorsReader (the same mmap header parser src/model/container.cpp
// uses for the r4dx container itself -- a plain .safetensors file is header-shape-compatible)
// rather than writing a third parser.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_container_path.h"  // r4dx_test::ContainerPath

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx_test {

inline bool FileExists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// r4dx_test::ContainerPath (test_container_path.h) is available to every includer of this header.

// SKIP_RETURN_CODE 77 -- same convention tests/tokenizer/CMakeLists.txt's tokenizer_golden test
// uses for "the real model data this test needs isn't on this machine", mapped to CTest's SKIPPED
// (not FAILED) result.
constexpr int kSkipReturnCode = 77;

inline int SkipMissing(const std::string& what) {
  std::fprintf(stderr, "[SKIP] %s not found; this test needs real checkpoint/golden data that is "
                        "not vendored into the repo -- see docs/validation.md.\n",
               what.c_str());
  return kSkipReturnCode;
}

inline std::vector<float> ReadGoldenAsFloat(const r4dx_convert::SafetensorsReader& r,
                                             const std::string& name) {
  const auto& m = r.Meta(name);
  const uint8_t* data = r.Data(name);
  const int64_t n = m.ElemCount();
  std::vector<float> out(static_cast<size_t>(n));
  if (m.dtype == "BF16") {
    const auto* src = reinterpret_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = r4dx::core::Bf16ToFloat(src[static_cast<size_t>(i)]);
  } else if (m.dtype == "F32") {
    std::memcpy(out.data(), data, static_cast<size_t>(n) * 4);
  } else {
    throw std::runtime_error("ReadGoldenAsFloat: unsupported dtype " + m.dtype + " for " + name);
  }
  return out;
}

// Requires the golden tensor to be stored BF16 (every hidden-state / layer-output tensor
// layer_golden.py dumps is, since the reference model runs at torch_dtype=bfloat16 on GPU).
inline std::vector<uint16_t> ReadGoldenRawBf16(const r4dx_convert::SafetensorsReader& r,
                                                const std::string& name) {
  const auto& m = r.Meta(name);
  if (m.dtype != "BF16") {
    throw std::runtime_error("ReadGoldenRawBf16: '" + name + "' is " + m.dtype + ", not BF16");
  }
  const int64_t n = m.ElemCount();
  const auto* src = reinterpret_cast<const uint16_t*>(r.Data(name));
  return std::vector<uint16_t>(src, src + n);
}

inline r4dx::core::DeviceBuffer<uint16_t> UploadBf16(const std::vector<uint16_t>& host) {
  r4dx::core::DeviceBuffer<uint16_t> buf(host.size());
  buf.CopyFromHost(host);
  return buf;
}

// sqrt(sum((got-ref)^2)) / sqrt(sum(ref^2)) over the flattened tensor -- the same metric
// tests/kernels/test_mxfp4_gemm.cpp and tools/reference/layer_golden.py's own TOLERANCES use.
inline double RelL2(const std::vector<float>& got, const std::vector<float>& ref) {
  if (got.size() != ref.size()) throw std::runtime_error("RelL2: size mismatch");
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return std::sqrt(num) / std::max(1e-9, std::sqrt(den));
}

inline std::vector<float> WidenBf16(const std::vector<uint16_t>& x) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = r4dx::core::Bf16ToFloat(x[i]);
  return out;
}

}  // namespace r4dx_test
