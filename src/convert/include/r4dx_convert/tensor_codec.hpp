// r4dx_convert::tensor_codec -- read a source safetensors tensor into float32 (dequantizing bf16
// on the way in), and encode float32/raw bf16 back out to the container's on-disk byte forms.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx_convert {

// Reads `name` out of `model` and widens it to float32, row-major, C-contiguous. Only BF16 and
// F32 source dtypes are handled -- every weight in this checkpoint's config.json ("dtype":
// "bfloat16") is one of the two, and a third dtype showing up is a checkpoint-format change this
// converter should fail loudly on rather than silently misinterpret.
inline std::vector<float> ReadTensorAsFloat(ShardedModel& model, const std::string& name) {
  const TensorMeta& meta = model.Meta(name);
  const uint8_t* data = model.Data(name);
  const int64_t n = meta.ElemCount();
  std::vector<float> out(static_cast<size_t>(n));
  if (meta.dtype == "BF16") {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n; ++i) out[i] = r4dx::core::Bf16ToFloat(src[i]);
  } else if (meta.dtype == "F32") {
    std::memcpy(out.data(), data, static_cast<size_t>(n) * 4);
  } else {
    throw std::runtime_error("ReadTensorAsFloat: unsupported source dtype " + meta.dtype + " for " + name);
  }
  return out;
}

// float32 -> bf16 bytes, N*2 bytes, row-major.
inline std::vector<uint8_t> EncodeBf16(const std::vector<float>& w) {
  std::vector<uint8_t> out(w.size() * 2);
  uint16_t* p = reinterpret_cast<uint16_t*>(out.data());
  for (size_t i = 0; i < w.size(); ++i) p[i] = r4dx::core::FloatToBf16(w[i]);
  return out;
}

// float32 -> raw fp32 bytes, N*4 bytes.
inline std::vector<uint8_t> EncodeFp32(const std::vector<float>& w) {
  std::vector<uint8_t> out(w.size() * 4);
  std::memcpy(out.data(), w.data(), out.size());
  return out;
}

// Copies a BF16 source tensor's raw bytes through unchanged (no float round-trip) -- used for the
// bulk bf16-passthrough tensors (vision.*, norms, embeddings) where the source dtype is already
// exactly the container's on-disk form.
inline std::vector<uint8_t> CopyRawBf16(ShardedModel& model, const std::string& name) {
  const TensorMeta& meta = model.Meta(name);
  if (meta.dtype != "BF16")
    throw std::runtime_error("CopyRawBf16: source dtype " + meta.dtype + " for " + name + " is not BF16");
  const uint8_t* data = model.Data(name);
  const int64_t nbytes = meta.ElemCount() * 2;
  return std::vector<uint8_t>(data, data + nbytes);
}

}  // namespace r4dx_convert
