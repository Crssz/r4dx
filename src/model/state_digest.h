// r4dx::model::DigestDeviceBytes -- FNV-1a 64 of a device byte range, for the R4DX_TP_TESTING hooks that
// digest a Model's state (Model::DebugStateDigest, MtpHead::DebugKvDigest). A test tool: one blocking
// hipMemcpy of the whole range, so callers synchronize their stream first.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "r4dx/core/error.hpp"

namespace r4dx::model {

inline uint64_t Fnv1a64Bytes(const uint8_t* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

inline uint64_t DigestDeviceBytes(const void* dev, size_t bytes) {
  if (bytes == 0) return Fnv1a64Bytes(nullptr, 0);
  std::vector<uint8_t> host(bytes);
  R4DX_HIP_CHECK(hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost));
  return Fnv1a64Bytes(host.data(), host.size());
}

}  // namespace r4dx::model
