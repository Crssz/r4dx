// r4dx::core error handling: HIP return codes and r4d C ABI return codes both become C++
// exceptions carrying the failing call's name, file:line, and (for HIP) hipGetErrorString.
#pragma once

#include <hip/hip_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace r4dx::core {

// Thrown by R4DX_HIP_CHECK for any non-success hipError_t.
class HipError : public std::runtime_error {
 public:
  HipError(hipError_t code, const char* expr, const char* file, int line)
      : std::runtime_error(Format(code, expr, file, line)), code_(code) {}

  hipError_t code() const noexcept { return code_; }

 private:
  static std::string Format(hipError_t code, const char* expr, const char* file, int line) {
    std::ostringstream oss;
    oss << "HIP error " << static_cast<int>(code) << " (" << hipGetErrorString(code) << ") from `"
        << expr << "` at " << file << ":" << line;
    return oss.str();
  }

  hipError_t code_;
};

// Thrown by R4DX_R4D_CHECK for a negative return code from an r4d.h entry point (attn/gdn family:
// "return 0 on success, negative on a shape this instantiation does not serve", per r4d.h).
class R4dError : public std::runtime_error {
 public:
  R4dError(int code, const char* kernel_name, const char* file, int line)
      : std::runtime_error(Format(code, kernel_name, file, line)), code_(code) {}

  int code() const noexcept { return code_; }

 private:
  static std::string Format(int code, const char* kernel_name, const char* file, int line) {
    std::ostringstream oss;
    oss << "r4d kernel `" << kernel_name << "` rejected this shape (return code " << code << ") at "
        << file << ":" << line;
    return oss.str();
  }

  int code_;
};

}  // namespace r4dx::core

#define R4DX_HIP_CHECK(expr)                                                     \
  do {                                                                           \
    hipError_t r4dx_hip_check_err__ = (expr);                                    \
    if (r4dx_hip_check_err__ != hipSuccess) {                                    \
      throw ::r4dx::core::HipError(r4dx_hip_check_err__, #expr, __FILE__, __LINE__); \
    }                                                                            \
  } while (0)

// Wrap a call to an r4d.h entry point that returns `int` (0 = success, negative = shape not
// served). `name` is a string literal used in the exception message.
#define R4DX_R4D_CHECK(name, expr)                                       \
  do {                                                                   \
    int r4dx_r4d_check_rc__ = (expr);                                    \
    if (r4dx_r4d_check_rc__ < 0) {                                       \
      throw ::r4dx::core::R4dError(r4dx_r4d_check_rc__, name, __FILE__, __LINE__); \
    }                                                                    \
  } while (0)
