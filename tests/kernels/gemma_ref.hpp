// tests/kernels/gemma_ref.hpp -- CPU references for the Gemma 4 kernels (docs/gemma4-plan.md 3.4,
// 3.5; task M1-15). Header-only, host code only, no HIP call: test_rope_proportional_cpu runs
// without a device, the GPU tests compare their kernel against the same functions.
//
// Each reference is written from the HF module's own op sequence rather than from the kernel's loop
// structure, so a pairing / rounding-point / identity-dims mistake in a kernel is not copied here.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cstring>

namespace gemma_ref {

// Round-to-nearest-even to bf16 and back (finite inputs), self-contained so the CPU-only test needs
// neither HIP headers nor a link library.
inline float Bf16Round(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

// HF Gemma4 rope, as the modeling code builds it: inv_freq has head_dim/2 entries, the first
// `rope_angles` equal to 1 / theta^(2i / freq_dim) and the rest ZERO ("proportional" NoPE tail;
// rope_angles == head_dim/2 and freq_dim == head_dim is the plain default rope); emb = cat(freqs,
// freqs); out = x * cos(emb) + rotate_half(x) * sin(emb), rotate_half(x) = cat(-x2, x1). Double
// precision on one head vector (x.size() == head_dim); returns the rotated vector. The angle is
// formed in fp32 (pos * inv_freq as floats, like the device) when `fp32_angle`.
inline std::vector<double> HfRope(const std::vector<double>& x, int rope_angles, int freq_dim,
                                  double theta, int pos, bool fp32_angle) {
  const int head_dim = static_cast<int>(x.size());
  const int half = head_dim / 2;
  std::vector<double> cs(head_dim), sn(head_dim);
  for (int j = 0; j < head_dim; ++j) {
    const int i = j % half;
    double inv_freq = 0.0;
    if (i < rope_angles) inv_freq = 1.0 / std::pow(theta, 2.0 * i / freq_dim);
    double angle = static_cast<double>(pos) * inv_freq;
    if (fp32_angle) angle = static_cast<double>(static_cast<float>(pos) * static_cast<float>(inv_freq));
    cs[j] = std::cos(angle);
    sn[j] = std::sin(angle);
  }
  std::vector<double> out(head_dim);
  for (int j = 0; j < head_dim; ++j) {
    const double rot = j < half ? -x[j + half] : x[j - half];  // rotate_half
    out[j] = x[j] * cs[j] + rot * sn[j];
  }
  return out;
}

// out = x * rsqrt(mean(x^2) + eps) [* w], one row, fp32 accumulation order irrelevant at test sizes
// (double here), result rounded to bf16 once.
inline std::vector<float> RmsNormRow(const std::vector<float>& x, const std::vector<float>* w,
                                     double eps) {
  double ss = 0.0;
  for (float v : x) ss += static_cast<double>(v) * v;
  const double rstd = 1.0 / std::sqrt(ss / x.size() + eps);
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    out[i] = Bf16Round(static_cast<float>(x[i] * rstd * (w ? static_cast<double>((*w)[i]) : 1.0)));
  }
  return out;
}

inline double GeluTanh(double x) {
  const double k = 0.7978845608028654;
  return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
}

}  // namespace gemma_ref
