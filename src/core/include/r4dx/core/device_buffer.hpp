// r4dx::core::DeviceBuffer<T>: RAII hipMalloc'd device array. Move-only; copy must be explicit
// (HostToDevice/DeviceToHost/DeviceToDevice) since a silent deep copy of GPU memory is never
// what a call site wants implicitly.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"  // docs/tp.md 2.7: counts allocs inside TP collectives

// Bounds-checks a requested element count against the buffer's actual capacity before any
// hipMemcpy* call -- a caller-supplied `count` larger than the allocation overflows the device
// buffer with no HIP-level signal (hipMemcpy itself has no idea how big `ptr_` is).
#define R4DX_CORE_CHECK_BOUNDS(requested, capacity, where)                    \
  do {                                                                       \
    if ((requested) > (capacity)) {                                          \
      throw std::out_of_range(where ": requested count exceeds buffer size"); \
    }                                                                        \
  } while (0)

namespace r4dx::core {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      static std::atomic<bool> logged{false};
      TpNoteDeviceAlloc("allocation (DeviceBuffer constructor)", logged);
      R4DX_HIP_CHECK(hipMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() { Free(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
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
  bool empty() const noexcept { return count_ == 0; }

  // Reallocates (dropping any previous contents) if the requested size differs.
  void Resize(size_t count) {
    if (count == count_) return;
    Free();
    count_ = count;
    if (count_ > 0) {
      static std::atomic<bool> logged{false};
      TpNoteDeviceAlloc("allocation (DeviceBuffer::Resize)", logged);
      R4DX_HIP_CHECK(hipMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  void ZeroAsync(const Stream& stream) const {
    if (count_ > 0) R4DX_HIP_CHECK(hipMemsetAsync(ptr_, 0, bytes(), stream.get()));
  }
  void Zero() const {
    if (count_ > 0) R4DX_HIP_CHECK(hipMemset(ptr_, 0, bytes()));
  }

  void CopyFromHost(const T* host, size_t count) {
    R4DX_CORE_CHECK_BOUNDS(count, count_, "DeviceBuffer::CopyFromHost");
    R4DX_HIP_CHECK(hipMemcpy(ptr_, host, count * sizeof(T), hipMemcpyHostToDevice));
  }
  void CopyFromHost(const std::vector<T>& host) { CopyFromHost(host.data(), host.size()); }
  void CopyFromHostAsync(const T* host, size_t count, const Stream& stream) {
    R4DX_CORE_CHECK_BOUNDS(count, count_, "DeviceBuffer::CopyFromHostAsync");
    R4DX_HIP_CHECK(
        hipMemcpyAsync(ptr_, host, count * sizeof(T), hipMemcpyHostToDevice, stream.get()));
  }

  void CopyToHost(T* host, size_t count) const {
    R4DX_CORE_CHECK_BOUNDS(count, count_, "DeviceBuffer::CopyToHost");
    R4DX_HIP_CHECK(hipMemcpy(host, ptr_, count * sizeof(T), hipMemcpyDeviceToHost));
  }
  std::vector<T> CopyToHost() const {
    std::vector<T> out(count_);
    if (count_ > 0) CopyToHost(out.data(), count_);
    return out;
  }
  void CopyToHostAsync(T* host, size_t count, const Stream& stream) const {
    R4DX_CORE_CHECK_BOUNDS(count, count_, "DeviceBuffer::CopyToHostAsync");
    R4DX_HIP_CHECK(
        hipMemcpyAsync(host, ptr_, count * sizeof(T), hipMemcpyDeviceToHost, stream.get()));
  }

  void CopyFromDevice(const DeviceBuffer<T>& other, size_t count) {
    R4DX_CORE_CHECK_BOUNDS(count, count_, "DeviceBuffer::CopyFromDevice");
    R4DX_HIP_CHECK(hipMemcpy(ptr_, other.ptr_, count * sizeof(T), hipMemcpyDeviceToDevice));
  }

 private:
  void Free() noexcept {
    if (ptr_ != nullptr) {
      static std::atomic<bool> logged{false};
      TpNoteDeviceAlloc("free (DeviceBuffer)", logged);
      static_cast<void>(hipFree(ptr_));
    }
    ptr_ = nullptr;
    count_ = 0;
  }

  T* ptr_ = nullptr;
  size_t count_ = 0;
};

}  // namespace r4dx::core
