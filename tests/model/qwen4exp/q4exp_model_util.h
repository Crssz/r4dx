// Shared helpers of the test_model_q4exp_*.cpp tests (Qwen4ExpModel on the tiny container): fixture paths, the HF golden
// reader, a capture store for Qwen4ExpModel::SetCapture, bf16 error statistics and the model loader. Header only; the
// glob of tests/model/qwen4exp/CMakeLists.txt matches test_*.cpp only.
//
// Fixtures (exit code 77 = skip when one is missing):
//   R4DX_Q4EXP_TINY_DIR   default E:\models\r4dx\tiny-q4exp: container-bf16.r4dx, container-w4a16.r4dx, sidecar.r4dx-ngram,
//                         golden\T{1,7,70,300}.recent.safetensors (tools/reference/qwen4exp/tiny_golden.py documents the
//                         tensor names; `recent` = the QSA tie rule r4dx implements, plan Q23).
// The test device is 0 when HIP_VISIBLE_DEVICES is set (ctest sets it to 1), else device 1 when there are two.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen4exp_model.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace q4m {

using r4dx::model::Qwen4ExpCapture;
using r4dx::model::Qwen4ExpModel;
using r4dx::model::Qwen4ExpModelOptions;

// stdout line-by-line even when ctest / a log redirect captures it, so a hang or a crash still leaves its progress.
inline void Unbuffer() { std::setvbuf(stdout, nullptr, _IONBF, 0); }

constexpr int kSkip = 77;

inline std::string TinyDir() {
  const char* e = std::getenv("R4DX_Q4EXP_TINY_DIR");
  return e != nullptr && e[0] != 0 ? e : "E:\\models\\r4dx\\tiny-q4exp";
}
inline std::string ContainerPath(const std::string& layout) { return TinyDir() + "\\container-" + layout + ".r4dx"; }
inline std::string GoldenPath(int T) { return TinyDir() + "\\golden\\T" + std::to_string(T) + ".recent.safetensors"; }
inline bool Exists(const std::string& p) { return std::filesystem::exists(p); }

inline int VisibleDevices() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 0;
  return n;
}
// The single-device default of the tests.
inline int TestDevice() {
  const char* hv = std::getenv("HIP_VISIBLE_DEVICES");
  if (hv != nullptr && hv[0] != 0) return 0;
  return VisibleDevices() >= 2 ? 1 : 0;
}

inline float Bf(uint16_t b) { return r4dx::core::Bf16ToFloat(b); }

// ---- the HF goldens ---------------------------------------------------------------------------------------------------
class Golden {
 public:
  Golden(int T, const std::string& path)
      : T_(T), reader_(std::make_unique<r4dx_convert::SafetensorsReader>(r4dx_convert::Utf8ToWide(path))) {
    const auto& m = reader_->Meta("input_ids");
    if (m.dtype != "I64" || m.ElemCount() != T) throw std::runtime_error("golden input_ids: unexpected dtype / length");
    const int64_t* p = reinterpret_cast<const int64_t*>(reader_->Data("input_ids"));
    for (int i = 0; i < T; ++i) ids_.push_back(static_cast<int32_t>(p[i]));
  }
  int T() const { return T_; }
  const std::vector<int32_t>& ids() const { return ids_; }
  bool Has(const std::string& name) const { return reader_->Has(name); }
  // bf16 tensor; `count` receives its element count.
  const uint16_t* Bf16(const std::string& name, int64_t* count = nullptr) const {
    const auto& m = reader_->Meta(name);
    if (m.dtype != "BF16") throw std::runtime_error("golden " + name + " is " + m.dtype + ", not BF16");
    if (count != nullptr) *count = m.ElemCount();
    return reinterpret_cast<const uint16_t*>(reader_->Data(name));
  }
  const int64_t* I64(const std::string& name, int64_t* count = nullptr) const {
    const auto& m = reader_->Meta(name);
    if (m.dtype != "I64") throw std::runtime_error("golden " + name + " is " + m.dtype + ", not I64");
    if (count != nullptr) *count = m.ElemCount();
    return reinterpret_cast<const int64_t*>(reader_->Data(name));
  }
  const uint8_t* U8(const std::string& name, int64_t* count = nullptr) const {
    const auto& m = reader_->Meta(name);
    if (m.dtype != "U8") throw std::runtime_error("golden " + name + " is " + m.dtype + ", not U8");
    if (count != nullptr) *count = m.ElemCount();
    return reinterpret_cast<const uint8_t*>(reader_->Data(name));
  }

 private:
  int T_;
  std::unique_ptr<r4dx_convert::SafetensorsReader> reader_;
  std::vector<int32_t> ids_;
};

// ---- captures ---------------------------------------------------------------------------------------------------------
struct Captured {
  Qwen4ExpCapture::Dtype dtype = Qwen4ExpCapture::Dtype::kBf16;
  int64_t cols = 0;
  int64_t rows = 0;  // rows filled so far (max over chunks of first_pos + rows)
  std::vector<uint16_t> bf16;
  std::vector<int32_t> i32;
};

class CaptureStore {
 public:
  r4dx::model::Qwen4ExpCaptureFn Fn() {
    return [this](const Qwen4ExpCapture& c) { Put(c); };
  }
  void Clear() { m_.clear(); }
  bool Has(const std::string& n) const { return m_.count(n) != 0; }
  const Captured& Get(const std::string& n) const { return m_.at(n); }
  const std::map<std::string, Captured>& all() const { return m_; }

 private:
  void Put(const Qwen4ExpCapture& c) {
    Captured& t = m_[c.name];
    t.dtype = c.dtype;
    t.cols = c.cols;
    t.rows = std::max(t.rows, c.first_pos + c.rows);
    const size_t n = static_cast<size_t>(c.rows * c.cols);
    const size_t off = static_cast<size_t>(c.first_pos * c.cols);
    if (c.dtype == Qwen4ExpCapture::Dtype::kBf16) {
      if (t.bf16.size() < off + n) t.bf16.resize(off + n);
      if (hipMemcpyAsync(t.bf16.data() + off, c.dev, n * 2, hipMemcpyDeviceToHost, c.stream) != hipSuccess) {
        throw std::runtime_error("capture copy failed: " + c.name);
      }
      if (hipStreamSynchronize(c.stream) != hipSuccess) throw std::runtime_error("capture sync failed: " + c.name);
    } else if (c.dtype == Qwen4ExpCapture::Dtype::kI32) {
      if (t.i32.size() < off + n) t.i32.resize(off + n);
      if (c.on_host) {
        std::memcpy(t.i32.data() + off, c.dev, n * 4);
      } else {
        if (hipMemcpyAsync(t.i32.data() + off, c.dev, n * 4, hipMemcpyDeviceToHost, c.stream) != hipSuccess ||
            hipStreamSynchronize(c.stream) != hipSuccess) {
          throw std::runtime_error("capture copy failed: " + c.name);
        }
      }
    } else {
      throw std::runtime_error("capture dtype not supported by the test store: " + c.name);
    }
  }
  std::map<std::string, Captured> m_;
};

// ---- error statistics -------------------------------------------------------------------------------------------------
struct Stats {
  int64_t n = 0;
  double max_abs = 0, mean_abs = 0, rms_ref = 0;
  double max_norm = 0;   // max_abs / rms_ref
  double mean_norm = 0;  // mean_abs / rms_ref
  int64_t max_ulp = 0;   // bf16 steps between the two values, over elements with |ref| >= 0.25 * rms_ref
  double frac_ulp_ge1 = 0, frac_ulp_ge2 = 0;  // of those elements
  int64_t big = 0;       // how many elements that is
  int64_t worst = -1;    // index of max_abs
  bool nan = false;
};

inline int64_t OrderedKey(uint16_t b) {  // monotone integer key of a bf16 value (+0 and -0 share key 0)
  const int64_t mag = b & 0x7FFF;
  return (b & 0x8000) ? -mag : mag;
}

inline Stats CompareBf16(const uint16_t* got, const uint16_t* ref, int64_t n) {
  Stats s;
  s.n = n;
  double sq = 0;
  for (int64_t i = 0; i < n; ++i) sq += static_cast<double>(Bf(ref[i])) * Bf(ref[i]);
  s.rms_ref = n > 0 ? std::sqrt(sq / static_cast<double>(n)) : 0.0;
  double sum = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double g = Bf(got[i]), r = Bf(ref[i]);
    if (std::isnan(g) || std::isinf(g)) s.nan = true;
    const double d = std::fabs(g - r);
    sum += d;
    if (d > s.max_abs || std::isnan(d)) {
      s.max_abs = d;
      s.worst = i;
    }
    if (std::fabs(r) >= 0.25 * s.rms_ref) {
      ++s.big;
      const int64_t u = std::llabs(OrderedKey(got[i]) - OrderedKey(ref[i]));
      s.max_ulp = std::max(s.max_ulp, u);
      s.frac_ulp_ge1 += u >= 1 ? 1 : 0;
      s.frac_ulp_ge2 += u >= 2 ? 1 : 0;
    }
  }
  s.mean_abs = n > 0 ? sum / static_cast<double>(n) : 0.0;
  if (s.big > 0) {
    s.frac_ulp_ge1 /= static_cast<double>(s.big);
    s.frac_ulp_ge2 /= static_cast<double>(s.big);
  }
  s.max_norm = s.rms_ref > 0 ? s.max_abs / s.rms_ref : 0.0;
  s.mean_norm = s.rms_ref > 0 ? s.mean_abs / s.rms_ref : 0.0;
  return s;
}

inline void PrintStatsHeader() {
  std::printf("  %-26s %9s %10s %10s %10s %9s %9s %6s %7s %7s\n", "tensor", "n", "max|err|", "mean|err|", "rms(ref)",
              "max/rms", "mean/rms", "maxulp", "ulp>=1", "ulp>=2");
}
inline void PrintStats(const std::string& name, const Stats& s) {
  std::printf("  %-26s %9lld %10.4g %10.4g %10.4g %9.4f %9.5f %6lld %6.1f%% %6.1f%%%s\n", name.c_str(),
              static_cast<long long>(s.n), s.max_abs, s.mean_abs, s.rms_ref, s.max_norm, s.mean_norm,
              static_cast<long long>(s.max_ulp), 100.0 * s.frac_ulp_ge1, 100.0 * s.frac_ulp_ge2, s.nan ? "  NaN/Inf!" : "");
}

// fp32 vs fp32 (logits rows): max / mean |diff|, relative to the rms of `ref`.
struct FStats {
  double max_abs = 0, mean_abs = 0, rms_ref = 0, max_norm = 0, mean_norm = 0;
  bool nan = false;
};
inline FStats CompareF32(const float* got, const float* ref, int64_t n) {
  FStats s;
  double sq = 0, sum = 0;
  for (int64_t i = 0; i < n; ++i) sq += static_cast<double>(ref[i]) * ref[i];
  s.rms_ref = n > 0 ? std::sqrt(sq / static_cast<double>(n)) : 0.0;
  for (int64_t i = 0; i < n; ++i) {
    if (std::isnan(got[i]) || std::isinf(got[i])) s.nan = true;
    const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
    s.max_abs = std::max(s.max_abs, d);
    sum += d;
  }
  s.mean_abs = n > 0 ? sum / static_cast<double>(n) : 0.0;
  s.max_norm = s.rms_ref > 0 ? s.max_abs / s.rms_ref : 0.0;
  s.mean_norm = s.rms_ref > 0 ? s.mean_abs / s.rms_ref : 0.0;
  return s;
}

inline int64_t ArgMax(const float* p, int64_t n) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; ++i) {
    if (p[i] > p[best]) best = i;
  }
  return best;
}

inline std::vector<float> WidenBf16(const uint16_t* p, int64_t n) {
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = Bf(p[i]);
  return out;
}

// ---- loading ------------------------------------------------------------------------------------------------------------
inline r4dx::model::Layout LayoutOf(const std::string& name) { return r4dx::model::LayoutFromName(name); }

// One tiny model. `devices` empty = both stages on TestDevice() (one stage, no split). `split` > 0 = two stages.
// `gdn_recurrent`: GDN prefill through the recurrent decode kernels (the slow reference path, see Qwen4ExpGdnCall::prefill_recurrent);
// false = libr4d's chunked scan (the production path).
inline std::unique_ptr<Qwen4ExpModel> LoadTiny(const std::string& layout, bool checkpoint = false,
                                               std::vector<int> devices = {}, int64_t split = 0,
                                               int64_t max_ctx = 4096, bool gdn_recurrent = false) {
  Qwen4ExpModelOptions o;
  o.container_path = ContainerPath(layout);
  o.layout = LayoutOf(layout);
  o.max_ctx = max_ctx;
  o.prompt_checkpoint = checkpoint;
  if (devices.empty()) devices = {TestDevice(), TestDevice()};
  o.devices = devices;
  o.split = split;
  o.gdn_prefill_recurrent = gdn_recurrent;
  return Qwen4ExpModel::Load(o);
}

}  // namespace q4m
