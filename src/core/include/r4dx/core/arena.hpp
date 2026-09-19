// r4dx::core::Arena: a bump allocator over one DeviceBuffer<uint8_t>, for per-step activation
// scratch (rmsnorm output, gate_up, attention scratch, ...) that is entirely re-derivable from
// the current token/layer and so never needs individual frees -- only Reset() between steps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::core {

class Arena {
 public:
  Arena() = default;
  explicit Arena(size_t capacity_bytes) : storage_(capacity_bytes) {}

  void Reserve(size_t capacity_bytes) {
    storage_.Resize(capacity_bytes);
    offset_ = 0;
  }

  // Bump-allocates `count` T's, aligned to alignof(T) (or `align_bytes` if given). Never frees
  // individually; call Reset() to reclaim everything at once. Throws std::bad_alloc if the arena
  // is exhausted -- a caller that hits this should Reserve() a bigger arena, not catch and retry.
  template <typename T>
  T* Alloc(size_t count, size_t align_bytes = alignof(T)) {
    if (align_bytes == 0 || (align_bytes & (align_bytes - 1)) != 0) {
      throw std::invalid_argument("Arena::Alloc: align_bytes must be a nonzero power of two");
    }
    // Overflow-check every step in size_t arithmetic before it can wrap: a huge `count` (e.g. a
    // corrupt shape) must throw std::bad_alloc, not silently wrap into a tiny in-bounds `bytes`
    // that then passes the base+bytes > capacity check and hands back a bogus pointer.
    size_t base = AlignUp(offset_, align_bytes);
    if (base < offset_) throw std::bad_alloc();  // AlignUp overflowed
    size_t bytes;
    if (__builtin_mul_overflow(count, sizeof(T), &bytes)) throw std::bad_alloc();
    size_t end;
    if (__builtin_add_overflow(base, bytes, &end)) throw std::bad_alloc();
    if (end > storage_.bytes()) {
      throw std::bad_alloc();
    }
    offset_ = end;
    return reinterpret_cast<T*>(storage_.data() + base);
  }

  void Reset() noexcept { offset_ = 0; }

  size_t used_bytes() const noexcept { return offset_; }
  size_t capacity_bytes() const noexcept { return storage_.bytes(); }

 private:
  static size_t AlignUp(size_t v, size_t align) { return (v + align - 1) & ~(align - 1); }

  DeviceBuffer<uint8_t> storage_;
  size_t offset_ = 0;
};

}  // namespace r4dx::core
