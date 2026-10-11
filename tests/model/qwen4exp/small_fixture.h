// Reader for the fixtures tools/reference/qwen4exp/small_fixtures.py writes (index.txt + params.txt + one raw .bin per
// tensor), plus the device upload / error-report helpers the test_gpu_{hc,moe,ple,gdn_sigmoid}*.cpp tests share.
// Fixture directory: env Q4EXP_SMALL_FIXTURES, default E:\models\r4dx\q4exp-fixtures\SmallKernels. Tests exit 77 (skip)
// when index.txt is missing.
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"

namespace q4test {

constexpr int kSkip = 77;

struct Tensor {
  std::string dtype;  // bf16 | f32 | i32
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  int64_t count() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
  const uint16_t* bf16() const { return Typed<uint16_t>("bf16"); }
  const float* f32() const { return Typed<float>("f32"); }
  const int32_t* i32() const { return Typed<int32_t>("i32"); }

 private:
  template <typename T>
  const T* Typed(const char* want) const {
    if (dtype != want) throw std::runtime_error("fixture tensor is " + dtype + ", asked for " + want);
    return reinterpret_cast<const T*>(bytes.data());
  }
};

class Fixtures {
 public:
  static std::string Dir() {
    const char* e = std::getenv("Q4EXP_SMALL_FIXTURES");
    return e != nullptr && e[0] != 0 ? e : "E:\\models\\r4dx\\q4exp-fixtures\\SmallKernels";
  }

  // Returns false when the fixture directory is not there.
  bool Open() {
    dir_ = Dir();
    std::ifstream idx(dir_ + "\\index.txt");
    if (!idx) return false;
    std::string line;
    while (std::getline(idx, line)) {
      std::istringstream is(line);
      std::string key, dtype;
      int rank = 0;
      if (!(is >> key >> dtype >> rank)) continue;
      Tensor t;
      t.dtype = dtype;
      for (int i = 0; i < rank; ++i) {
        int64_t d = 0;
        is >> d;
        t.shape.push_back(d);
      }
      index_[key] = t;
    }
    std::ifstream par(dir_ + "\\params.txt");
    while (std::getline(par, line)) {
      std::istringstream is(line);
      std::string key;
      double v = 0;
      if (is >> key >> v) params_[key] = v;
    }
    return true;
  }

  const Tensor& Get(const std::string& key) {
    auto it = loaded_.find(key);
    if (it != loaded_.end()) return it->second;
    auto meta = index_.find(key);
    if (meta == index_.end()) throw std::runtime_error("fixture has no tensor '" + key + "'");
    Tensor t = meta->second;
    const int64_t elem = t.dtype == "bf16" ? 2 : 4;
    t.bytes.resize(static_cast<size_t>(t.count() * elem));
    std::ifstream f(dir_ + "\\" + key + ".bin", std::ios::binary);
    if (!f) throw std::runtime_error("fixture file missing for '" + key + "'");
    f.read(reinterpret_cast<char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
    if (static_cast<size_t>(f.gcount()) != t.bytes.size()) {
      throw std::runtime_error("fixture file for '" + key + "' is short");
    }
    return loaded_.emplace(key, std::move(t)).first->second;
  }

  double Param(const std::string& key) const {
    auto it = params_.find(key);
    if (it == params_.end()) throw std::runtime_error("fixture has no param '" + key + "'");
    return it->second;
  }

 private:
  std::string dir_;
  std::map<std::string, Tensor> index_;
  std::map<std::string, Tensor> loaded_;
  std::map<std::string, double> params_;
};

inline r4dx::core::DeviceBuffer<uint16_t> UploadBf16(const uint16_t* p, size_t n) {
  r4dx::core::DeviceBuffer<uint16_t> b(n);
  b.CopyFromHost(p, n);
  return b;
}
inline r4dx::core::DeviceBuffer<uint16_t> Upload(const Tensor& t) { return UploadBf16(t.bf16(), static_cast<size_t>(t.count())); }
inline r4dx::core::DeviceBuffer<float> UploadF32(const float* p, size_t n) {
  r4dx::core::DeviceBuffer<float> b(n);
  b.CopyFromHost(p, n);
  return b;
}
inline r4dx::core::DeviceBuffer<int32_t> UploadI32(const int32_t* p, size_t n) {
  r4dx::core::DeviceBuffer<int32_t> b(n);
  b.CopyFromHost(p, n);
  return b;
}

inline float F(uint16_t b) { return r4dx::core::Bf16ToFloat(b); }

struct ErrorStats {
  double max_abs = 0, mean_abs = 0, rms_ref = 0, max_rel_to_rms = 0;
  int64_t n = 0, exact = 0, bad = 0;
  int64_t worst = -1;
};

// Compares `got` against `ref` (bf16 bit patterns). An element is `bad` when |got - ref| > atol + rtol * |ref|.
inline ErrorStats CompareBf16(const uint16_t* got, const uint16_t* ref, int64_t n, double atol, double rtol) {
  ErrorStats s;
  s.n = n;
  double sum = 0, sumsq = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double g = F(got[i]), r = F(ref[i]);
    const double d = std::fabs(g - r);
    if (std::isnan(g) || std::isnan(r)) {
      if (std::isnan(g) != std::isnan(r)) s.bad++;
      continue;
    }
    if (got[i] == ref[i]) s.exact++;
    if (d > s.max_abs) {
      s.max_abs = d;
      s.worst = i;
    }
    sum += d;
    sumsq += r * r;
    if (d > atol + rtol * std::fabs(r)) s.bad++;
  }
  s.mean_abs = n ? sum / static_cast<double>(n) : 0;
  s.rms_ref = n ? std::sqrt(sumsq / static_cast<double>(n)) : 0;
  s.max_rel_to_rms = s.rms_ref > 0 ? s.max_abs / s.rms_ref : 0;
  return s;
}

// Prints one line and returns true when no element is bad.
inline bool Report(const std::string& name, const ErrorStats& s) {
  std::printf("  %-34s n=%-8lld max_abs=%.4g mean_abs=%.4g rms_ref=%.4g max/rms=%.4g exact=%.2f%% bad=%lld %s\n",
              name.c_str(), static_cast<long long>(s.n), s.max_abs, s.mean_abs, s.rms_ref, s.max_rel_to_rms,
              s.n ? 100.0 * static_cast<double>(s.exact) / static_cast<double>(s.n) : 100.0,
              static_cast<long long>(s.bad), s.bad == 0 ? "OK" : "FAIL");
  return s.bad == 0;
}

template <typename T>
std::vector<T> Download(const r4dx::core::DeviceBuffer<T>& b) {
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return b.CopyToHost();
}

inline void Sync() { R4DX_HIP_CHECK(hipDeviceSynchronize()); }

}  // namespace q4test
