// r4dx::core::PinnedBuffer<T>: RAII hipHostMalloc'd (page-locked) host array, for the
// embedding-lookup and sampling host<->device staging paths where an async copy needs pinned
// memory to actually overlap with compute.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <utility>

#include "r4dx/core/error.hpp"

namespace r4dx::core {

template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t count, unsigned int flags = hipHostMallocDefault) : count_(count) {
    if (count_ > 0) {
      R4DX_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T), flags));
    }
  }

  ~PinnedBuffer() { Free(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  size_t size() const noexcept { return count_; }
  size_t bytes() const noexcept { return count_ * sizeof(T); }

  T& operator[](size_t i) noexcept { return ptr_[i]; }
  const T& operator[](size_t i) const noexcept { return ptr_[i]; }

  T* begin() noexcept { return ptr_; }
  T* end() noexcept { return ptr_ + count_; }

 private:
  void Free() noexcept {
    if (ptr_ != nullptr) static_cast<void>(hipHostFree(ptr_));
    ptr_ = nullptr;
    count_ = 0;
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

}  // namespace r4dx::core
