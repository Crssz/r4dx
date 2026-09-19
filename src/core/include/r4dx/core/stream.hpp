// r4dx::core::Stream: RAII hipStream_t. Move-only, non-blocking (hipStreamNonBlocking) by
// default so it never serialises against the legacy default stream.
#pragma once

#include <hip/hip_runtime.h>

#include <utility>

#include "r4dx/core/error.hpp"

namespace r4dx::core {

class Stream {
 public:
  Stream() { R4DX_HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)); }
  explicit Stream(unsigned int flags) { R4DX_HIP_CHECK(hipStreamCreateWithFlags(&stream_, flags)); }

  // Wraps the legacy default stream (handle 0) without creating a new one; Destroy() is then a
  // no-op. Useful for call sites that want "the null stream" through the same RAII type.
  static Stream Default() { return Stream(nullptr); }

  ~Stream() { Destroy(); }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept : stream_(other.stream_), owns_(other.owns_) {
    other.stream_ = nullptr;
    other.owns_ = false;
  }
  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      Destroy();
      stream_ = other.stream_;
      owns_ = other.owns_;
      other.stream_ = nullptr;
      other.owns_ = false;
    }
    return *this;
  }

  hipStream_t get() const noexcept { return stream_; }
  operator hipStream_t() const noexcept { return stream_; }

  void Synchronize() const { R4DX_HIP_CHECK(hipStreamSynchronize(stream_)); }

 private:
  explicit Stream(std::nullptr_t) : stream_(nullptr), owns_(false) {}

  void Destroy() noexcept {
    if (owns_ && stream_ != nullptr) {
      static_cast<void>(hipStreamDestroy(stream_));
    }
    stream_ = nullptr;
    owns_ = false;
  }

  hipStream_t stream_ = nullptr;
  bool owns_ = true;
};

}  // namespace r4dx::core
