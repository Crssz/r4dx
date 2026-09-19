// r4dx::core::TensorView: a non-owning view over a device (or host) buffer -- pointer, dtype,
// shape and strides in ELEMENTS (matching the convention every r4d.h struct/comment uses:
// R4DArgs.kv_block_stride, r4d_quant_act_i8's row stride, etc. are all elements, never bytes).
// Owns nothing; the DeviceBuffer/PinnedBuffer it points into must outlive it.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include "r4dx/core/dtype.hpp"

namespace r4dx::core {

inline constexpr int kMaxTensorRank = 4;

struct TensorView {
  void* data = nullptr;
  Dtype dtype = Dtype::kF32;
  int ndim = 0;
  std::array<int64_t, kMaxTensorRank> shape{};    // elements per dimension
  std::array<int64_t, kMaxTensorRank> strides{};  // elements, not bytes

  TensorView() = default;

  // Builds a view with row-major (C-contiguous) strides for `shape`.
  static TensorView Contiguous(void* ptr, Dtype dt, std::initializer_list<int64_t> shape_list) {
    if (shape_list.size() > static_cast<size_t>(kMaxTensorRank)) {
      throw std::invalid_argument("TensorView::Contiguous: shape rank exceeds kMaxTensorRank");
    }
    TensorView t;
    t.data = ptr;
    t.dtype = dt;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (int64_t d : shape_list) t.shape[i++] = d;
    int64_t stride = 1;
    for (int d = t.ndim - 1; d >= 0; --d) {
      t.strides[d] = stride;
      stride *= t.shape[d];
    }
    return t;
  }

  int64_t ElemCount() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return n;
  }
  int64_t ByteSize() const { return ElemCount() * DtypeSize(dtype); }

  bool IsContiguous() const {
    int64_t stride = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      if (shape[d] != 1 && strides[d] != stride) return false;
      stride *= shape[d];
    }
    return true;
  }

  template <typename T>
  T* as() const {
    return static_cast<T*>(data);
  }
};

}  // namespace r4dx::core
