// r4dx::core::Event: RAII hipEvent_t for timing and cross-stream synchronisation.
#pragma once

#include <hip/hip_runtime.h>

#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::core {

class Event {
 public:
  Event() { R4DX_HIP_CHECK(hipEventCreate(&event_)); }
  explicit Event(unsigned int flags) { R4DX_HIP_CHECK(hipEventCreateWithFlags(&event_, flags)); }
  ~Event() {
    if (event_ != nullptr) static_cast<void>(hipEventDestroy(event_));
  }

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) static_cast<void>(hipEventDestroy(event_));
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  hipEvent_t get() const noexcept { return event_; }
  operator hipEvent_t() const noexcept { return event_; }

  void Record(const Stream& stream) { R4DX_HIP_CHECK(hipEventRecord(event_, stream.get())); }
  void Synchronize() const { R4DX_HIP_CHECK(hipEventSynchronize(event_)); }

  // Milliseconds between this (start) event and `end`. Both must have been recorded and
  // synchronised (or the enclosing stream synchronised) first.
  float ElapsedMs(const Event& end) const {
    float ms = 0.0f;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, event_, end.event_));
    return ms;
  }

 private:
  hipEvent_t event_ = nullptr;
};

}  // namespace r4dx::core
