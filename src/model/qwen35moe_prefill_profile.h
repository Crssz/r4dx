// PrefillProfile -- R4DX_Q35_PROFILE=3: where the GPU time of a prefill chunk goes (docs/qwen35moe-plan.md section 8).
//
// A phase is the stream time between two consecutive Mark() calls and is credited to the NAME of the later mark, so GPU idle gaps (the
// host-routed MoE waits for the router ids on the host, then enqueues the expert GEMMs) show up under the phase that follows them. The marks
// are hipEvents recorded on the stage's stream (no synchronization); End() reads them back after the chunk's stream synchronize. Header only.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "r4dx/core/error.hpp"

namespace r4dx::model {

class PrefillProfile {
 public:
  PrefillProfile() {
    const char* e = std::getenv("R4DX_Q35_PROFILE");
    on_ = e != nullptr && std::strcmp(e, "3") == 0;
  }
  ~PrefillProfile() {
    for (hipEvent_t ev : pool_) static_cast<void>(hipEventDestroy(ev));
  }
  PrefillProfile(const PrefillProfile&) = delete;
  PrefillProfile& operator=(const PrefillProfile&) = delete;

  bool on() const { return on_; }

  // Starts a chunk: the first event is the origin of the first phase.
  void Begin(hipStream_t s) {
    if (!on_) return;
    marks_.clear();
    Mark(s, "begin");
  }
  void Mark(hipStream_t s, const char* name) {
    if (!on_) return;
    if (marks_.size() == pool_.size()) {
      hipEvent_t ev = nullptr;
      R4DX_HIP_CHECK(hipEventCreate(&ev));
      pool_.push_back(ev);
    }
    hipEvent_t ev = pool_[marks_.size()];
    R4DX_HIP_CHECK(hipEventRecord(ev, s));
    marks_.emplace_back(name, ev);
  }
  // After the stream has been synchronized: folds this chunk's phases into the totals.
  void End(int64_t rows, double wall_ms, double enqueue_ms = 0) {
    if (!on_ || marks_.size() < 2) return;
    for (size_t i = 1; i < marks_.size(); ++i) {
      float ms = 0;
      R4DX_HIP_CHECK(hipEventElapsedTime(&ms, marks_[i - 1].second, marks_[i].second));
      Entry& e = phases_[marks_[i].first];
      e.ms += ms;
      ++e.calls;
      total_gpu_ms_ += ms;
    }
    ++chunks_;
    rows_ += rows;
    wall_ms_ += wall_ms;
    enqueue_ms_ += enqueue_ms;
  }
  void Print(int device, int64_t first, int64_t last) const {
    if (!on_ || chunks_ == 0) return;
    std::vector<std::pair<std::string, Entry>> v(phases_.begin(), phases_.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    std::fprintf(stderr, "[Qwen35MoeStage dev %d, layers %lld-%lld] prefill profile: %lld chunk(s), %lld rows, GPU span %.1f ms, host wall %.1f ms of which %.1f ms enqueueing (%.0f rows/s)\n",
                 device, static_cast<long long>(first), static_cast<long long>(last), static_cast<long long>(chunks_),
                 static_cast<long long>(rows_), total_gpu_ms_, wall_ms_, enqueue_ms_, wall_ms_ > 0 ? 1000.0 * static_cast<double>(rows_) / wall_ms_ : 0.0);
    for (const auto& p : v) {
      std::fprintf(stderr, "    %-18s %9.1f ms %5.1f%%  (%lld calls, %.3f ms each)\n", p.first.c_str(), p.second.ms,
                   total_gpu_ms_ > 0 ? 100.0 * p.second.ms / total_gpu_ms_ : 0.0, static_cast<long long>(p.second.calls),
                   p.second.calls > 0 ? p.second.ms / static_cast<double>(p.second.calls) : 0.0);
    }
  }

 private:
  struct Entry {
    double ms = 0;
    int64_t calls = 0;
  };
  bool on_ = false;
  std::vector<hipEvent_t> pool_;
  std::vector<std::pair<std::string, hipEvent_t>> marks_;
  std::map<std::string, Entry> phases_;
  int64_t chunks_ = 0, rows_ = 0;
  double total_gpu_ms_ = 0, wall_ms_ = 0, enqueue_ms_ = 0;
};

}  // namespace r4dx::model
