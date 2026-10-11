// Error / shape-check / padding helpers shared by the qwen4exp host code. `ctx` is the caller's error-message prefix
// (e.g. "qwen4exp::HyperConn"); every message reads "<ctx>: ...".
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "quant_linear.h"

namespace r4dx::model::qwen4exp {

[[noreturn]] inline void Fail(const char* ctx, const std::string& msg) {
  throw std::runtime_error(std::string(ctx) + ": " + msg);
}

// A HIP / q4x_* launcher return code (0 = hipSuccess); throws "<ctx>: <what> failed: <hip error string>".
inline void Check(const char* ctx, int code, const char* what) {
  if (code != 0) {
    throw std::runtime_error(std::string(ctx) + ": " + what + " failed: " +
                             hipGetErrorString(static_cast<hipError_t>(code)));
  }
}

inline void CheckLinear(const char* ctx, const QuantLinear* l, const char* name, int64_t n, int64_t k) {
  if (l == nullptr) Fail(ctx, std::string("missing weight ") + name);
  if (l->N != n || l->K != k) {
    Fail(ctx, std::string(name) + " is [" + std::to_string(l->N) + ", " + std::to_string(l->K) + "], expected [" +
                  std::to_string(n) + ", " + std::to_string(k) + "]");
  }
}

// Arena::Alloc rounds every allocation's end up to 16 bytes; ArenaBytes() sums add that slack explicitly.
inline constexpr size_t Padded(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

}  // namespace r4dx::model::qwen4exp
