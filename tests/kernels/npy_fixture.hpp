// tests/kernels/npy_fixture.hpp -- minimal reader for the numpy `.npy` v1/v2 arrays
// tools/reference/dflash2_ref.py writes into tools/reference/golden_out/dflash2/fixture_{a,b,c}/,
// plus the shared helpers the four DFlash2 kernel tests use to drive a kernel from those arrays.
//
// Scope, deliberately: little-endian C-order '<f4' (float32) and '<i8' (int64) arrays -- every dtype
// those fixtures contain -- plus '<u4' (uint32 ring words) and '<f2' (fp16, returned as raw bits),
// which the trellis goldens (tools/reference/trellis_golden.py -> tests/kernels/golden/trellis/)
// add. Anything else throws rather than silently mis-reading.
// The fixture directory is located through the R4DX_SOURCE_DIR compile definition
// (tests/CMakeLists.txt), never a hardcoded absolute path.
//
// The fixtures are gitignored and exactly reproducible (docs/dflash2.md section 9), so every test
// that needs them calls FixtureAvailable() first and returns kSkipReturnCode (77 -- CTest SKIPPED)
// when they are absent, the same convention tests/model and test_kernel_bandwidth.cpp use.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"

namespace r4dx_test {

// CTest SKIPPED, not FAILED. Mirrors tests/model/test_common.h's kSkipReturnCode.
constexpr int kSkipReturnCode = 77;

inline std::string FixtureDir(const std::string& name = "fixture_a") {
  return std::string(R4DX_SOURCE_DIR) + "/tools/reference/golden_out/dflash2/" + name;
}

inline bool FixtureAvailable(const std::string& name = "fixture_a") {
  std::ifstream f(FixtureDir(name) + "/manifest.json", std::ios::binary);
  return static_cast<bool>(f);
}

struct NpyHeader {
  std::string descr;
  std::vector<int64_t> shape;
  size_t data_offset = 0;
  int64_t Count() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

inline NpyHeader ParseNpyHeader(const std::vector<char>& raw, const std::string& path) {
  if (raw.size() < 12 || std::memcmp(raw.data(), "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("npy_fixture: not a .npy file: " + path);
  }
  const unsigned major = static_cast<unsigned char>(raw[6]);
  size_t hdr_start = 0, hdr_len = 0;
  if (major == 1) {
    hdr_len = static_cast<size_t>(static_cast<unsigned char>(raw[8])) |
              (static_cast<size_t>(static_cast<unsigned char>(raw[9])) << 8);
    hdr_start = 10;
  } else {
    hdr_len = 0;
    for (int b = 0; b < 4; ++b) {
      hdr_len |= static_cast<size_t>(static_cast<unsigned char>(raw[8 + b])) << (8 * b);
    }
    hdr_start = 12;
  }
  if (hdr_start + hdr_len > raw.size()) {
    throw std::runtime_error("npy_fixture: truncated header in " + path);
  }
  const std::string hdr(raw.data() + hdr_start, hdr_len);

  NpyHeader out;
  out.data_offset = hdr_start + hdr_len;

  const std::string dkey = "'descr':";
  size_t p = hdr.find(dkey);
  if (p == std::string::npos) throw std::runtime_error("npy_fixture: no descr in " + path);
  p = hdr.find('\'', p + dkey.size());
  size_t q = hdr.find('\'', p + 1);
  out.descr = hdr.substr(p + 1, q - p - 1);

  if (hdr.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("npy_fixture: only C-order arrays are supported: " + path);
  }

  const std::string skey = "'shape':";
  p = hdr.find(skey);
  if (p == std::string::npos) throw std::runtime_error("npy_fixture: no shape in " + path);
  p = hdr.find('(', p);
  q = hdr.find(')', p);
  const std::string dims = hdr.substr(p + 1, q - p - 1);
  size_t i = 0;
  while (i < dims.size()) {
    while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) ++i;
    size_t j = i;
    while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') ++j;
    if (j > i) out.shape.push_back(std::stoll(dims.substr(i, j - i)));
    i = j + 1;
  }
  return out;
}

inline std::vector<char> ReadWhole(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("npy_fixture: cannot open " + path);
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> raw(static_cast<size_t>(n));
  f.read(raw.data(), n);
  return raw;
}

// Loads an array whose descr must be `descr` into elements of T (sizeof(T) = the descr's width).
// `expect` (if non-empty) is checked against the stored shape, so a fixture regenerated with
// different dimensions fails loudly here instead of producing a garbage compare.
template <typename T>
inline std::vector<T> LoadNpyAs(const std::string& path, const char* descr,
                                const std::vector<int64_t>& expect) {
  const std::vector<char> raw = ReadWhole(path);
  const NpyHeader h = ParseNpyHeader(raw, path);
  if (h.descr != descr) {
    throw std::runtime_error(std::string("npy_fixture: expected '") + descr + "' in " + path +
                             ", got '" + h.descr + "'");
  }
  if (!expect.empty() && h.shape != expect) {
    std::string got;
    for (int64_t d : h.shape) got += std::to_string(d) + ",";
    throw std::runtime_error("npy_fixture: unexpected shape [" + got + "] in " + path);
  }
  std::vector<T> out(static_cast<size_t>(h.Count()));
  if (h.data_offset + out.size() * sizeof(T) > raw.size()) {
    throw std::runtime_error("npy_fixture: truncated data in " + path);
  }
  std::memcpy(out.data(), raw.data() + h.data_offset, out.size() * sizeof(T));
  return out;
}

inline std::vector<float> LoadNpyF32(const std::string& path,
                                      const std::vector<int64_t>& expect = {}) {
  return LoadNpyAs<float>(path, "<f4", expect);
}

inline std::vector<int64_t> LoadNpyI64(const std::string& path,
                                        const std::vector<int64_t>& expect = {}) {
  return LoadNpyAs<int64_t>(path, "<i8", expect);
}

// '<u4': the trellis goldens' ring words (pair grid or oracle layout), uploaded as-is.
inline std::vector<uint32_t> LoadNpyU32(const std::string& path,
                                         const std::vector<int64_t>& expect = {}) {
  return LoadNpyAs<uint32_t>(path, "<u4", expect);
}

// '<f2' as raw fp16 bit patterns, so a kernel's f16 output compares bit for bit.
inline std::vector<uint16_t> LoadNpyF16Bits(const std::string& path,
                                             const std::vector<int64_t>& expect = {}) {
  return LoadNpyAs<uint16_t>(path, "<f2", expect);
}

inline std::vector<int64_t> NpyShape(const std::string& path) {
  return ParseNpyHeader(ReadWhole(path), path).shape;
}

// ---- comparison helpers ---------------------------------------------------------------------
// The fixtures are fp32 (the Python reference accumulates in float64 and stores fp32); the kernels
// under test are bf16-in/bf16-out. So every fixture comparison is inherently limited by bf16's own
// ~2^-8 relative resolution on the INPUTS, not just the output. Both metrics below are reported by
// every test; thresholds are stated per call site.
struct ErrStats {
  double max_abs = 0.0;
  double max_rel = 0.0;   // per element, floored denominator (see `floor`)
  double norm_rel = 0.0;  // ||got - ref||_2 / ||ref||_2 over the whole tensor
};

inline ErrStats CompareToRef(const std::vector<uint16_t>& got_bf16, const std::vector<float>& ref,
                              double floor = 1e-2) {
  ErrStats s;
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = r4dx::core::Bf16ToFloat(got_bf16[i]);
    const double r = ref[i];
    const double d = std::abs(g - r);
    s.max_abs = std::max(s.max_abs, d);
    s.max_rel = std::max(s.max_rel, d / std::max(floor, std::abs(r)));
    num += (g - r) * (g - r);
    den += r * r;
  }
  s.norm_rel = std::sqrt(num) / std::max(1e-30, std::sqrt(den));
  return s;
}

inline ErrStats CompareF32ToRef(const std::vector<float>& got, const std::vector<float>& ref,
                                 double floor = 1e-2) {
  ErrStats s;
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = got[i], r = ref[i];
    const double d = std::abs(g - r);
    s.max_abs = std::max(s.max_abs, d);
    s.max_rel = std::max(s.max_rel, d / std::max(floor, std::abs(r)));
    num += (g - r) * (g - r);
    den += r * r;
  }
  s.norm_rel = std::sqrt(num) / std::max(1e-30, std::sqrt(den));
  return s;
}

inline std::vector<uint16_t> ToBf16(const std::vector<float>& x) {
  std::vector<uint16_t> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = r4dx::core::FloatToBf16(x[i]);
  return out;
}

}  // namespace r4dx_test
