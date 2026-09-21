// tests/kernels/test_rope_mrope3.cpp -- r4dx_rope_partial_mrope3_bf16 (the 3-axis multimodal
// mrope, docs/vision.md "Text-side splicing") against a CPU fp32 reference.
//
// The reference here is deliberately NOT a copy of the kernel's own bin->stream predicate. It
// replays `Qwen3_5TextRotaryEmbedding.recomposition_frequencies` LITERALLY, as the two python
// slice assignments that function performs:
//
//     freqs_thw            = freq[0]                       # all-temporal to start
//     freqs_thw[1:3*sec_h:3] = freq[1][1:3*sec_h:3]        # height
//     freqs_thw[2:3*sec_w:3] = freq[2][2:3*sec_w:3]        # width
//
// so a kernel that got the section BOUNDS wrong (rather than just the mod-3 phase) still fails
// here. The third case below exists for exactly that: with [11,11,10] both slices happen to reach
// their last in-range bin, making the assignment coincide with plain `bin % 3`, while [16,10,6]
// leaves bins 30 and 31 temporal -- so a `% 3`-only implementation passes case 1 and fails case 3.
//
// Arithmetic tolerance follows tests/kernels/test_rope.cpp's own reasoning: inv_freq and the angle
// multiply are computed in float32 through the kernel's exact expression order, and only the final
// sincos is evaluated in double, so this isolates the stream selection rather than float32's
// inherent angle quantization.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

// The reference's own two slice assignments, verbatim -- see the file comment.
std::vector<int> AxisTable(int half, int sec_h, int sec_w) {
  std::vector<int> axis(static_cast<size_t>(half), 0);
  for (int i = 1; i < 3 * sec_h && i < half; i += 3) axis[static_cast<size_t>(i)] = 1;
  for (int i = 2; i < 3 * sec_w && i < half; i += 3) axis[static_cast<size_t>(i)] = 2;
  return axis;
}

void Rope3Ref(std::vector<uint16_t>* x, int tokens, int heads, int head_dim, int rotary_dim,
              float theta, const std::vector<int32_t>& pos3, int sec_h, int sec_w) {
  const int half = rotary_dim / 2;
  const std::vector<int> axis = AxisTable(half, sec_h, sec_w);
  for (int t = 0; t < tokens; ++t) {
    for (int hd = 0; hd < half; ++hd) {
      const float pos =
          static_cast<float>(pos3[static_cast<size_t>(axis[static_cast<size_t>(hd)] * tokens + t)]);
      const float inv_freq =
          std::exp2(-(2.0f * hd) / static_cast<float>(rotary_dim) * std::log2(theta));
      const double angle = static_cast<double>(pos * inv_freq);
      const double c = std::cos(angle), s = std::sin(angle);
      for (int h = 0; h < heads; ++h) {
        const size_t base = (static_cast<size_t>(t) * heads + h) * head_dim;
        const double x0 = Bf16ToFloat((*x)[base + hd]);
        const double x1 = Bf16ToFloat((*x)[base + hd + half]);
        (*x)[base + hd] = FloatToBf16(static_cast<float>(x0 * c - x1 * s));
        (*x)[base + hd + half] = FloatToBf16(static_cast<float>(x1 * c + x0 * s));
      }
    }
  }
}

double MaxRelErr(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double max_rel = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
    max_rel = std::max(max_rel, std::abs(g - r) / std::max(1e-2, std::abs(r)));
  }
  return max_rel;
}

// Whole-tensor norm-relative error, for the long-context case -- same reasoning
// tests/kernels/test_rope.cpp's own NormRelErr records: at positions near 262144 the rotation's
// x0*c - x1*s lands near zero for some (token, head, bin) by chance, and a per-element relative
// metric spikes on that entry regardless of whether the stream selection is right.
double NormRelErr(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
    num += (g - r) * (g - r);
    den += r * r;
  }
  return std::sqrt(num) / std::max(1e-9, std::sqrt(den));
}

size_t CountDiffering(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) ++n;
  }
  return n;
}

constexpr int kHeadDim = 256, kRotaryDim = 64, kHalf = kRotaryDim / 2;
constexpr float kTheta = 1.0e7f;

// One (sections, positions) case: rotate q and k on device, compare against the CPU reference.
// `long_context` picks the error metric: a norm-relative one bounded at 5e-2 (test_rope.cpp's own
// long-context bound and reasoning) instead of the per-element 2e-2, because at positions near
// 262143 sincosf's own accuracy, not the stream selection, dominates a per-element maximum.
bool RunCase(const char* name, int sec_t, int sec_h, int sec_w, const std::vector<int32_t>& pos3,
             int tokens, std::mt19937* rng, bool long_context = false) {
  const int heads_q = 24, heads_k = 4;
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<uint16_t> q_h(static_cast<size_t>(tokens) * heads_q * kHeadDim);
  std::vector<uint16_t> k_h(static_cast<size_t>(tokens) * heads_k * kHeadDim);
  for (auto& v : q_h) v = FloatToBf16(dist(*rng));
  for (auto& v : k_h) v = FloatToBf16(dist(*rng));

  std::vector<uint16_t> q_ref = q_h, k_ref = k_h;
  Rope3Ref(&q_ref, tokens, heads_q, kHeadDim, kRotaryDim, kTheta, pos3, sec_h, sec_w);
  Rope3Ref(&k_ref, tokens, heads_k, kHeadDim, kRotaryDim, kTheta, pos3, sec_h, sec_w);

  DeviceBuffer<uint16_t> q_d(q_h.size()), k_d(k_h.size());
  DeviceBuffer<int32_t> pos_d(pos3.size());
  q_d.CopyFromHost(q_h);
  k_d.CopyFromHost(k_h);
  pos_d.CopyFromHost(pos3);
  r4dx_rope_partial_mrope3_bf16(reinterpret_cast<int64_t>(q_d.data()),
                                 reinterpret_cast<int64_t>(k_d.data()),
                                 reinterpret_cast<int64_t>(pos_d.data()), tokens, heads_q, heads_k,
                                 kHeadDim, kRotaryDim, kTheta, sec_t, sec_h, sec_w, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  const std::vector<uint16_t> q_got = q_d.CopyToHost();
  const std::vector<uint16_t> k_got = k_d.CopyToHost();
  const double rel_q = long_context ? NormRelErr(q_got, q_ref) : MaxRelErr(q_got, q_ref);
  const double rel_k = long_context ? NormRelErr(k_got, k_ref) : MaxRelErr(k_got, k_ref);
  const double tol = long_context ? 5e-2 : 2e-2;

  // The pass-through tail [rotary_dim, head_dim) must be untouched, exactly as in the single-row
  // kernel's own test.
  bool tail_ok = true;
  for (int t = 0; t < tokens && tail_ok; ++t) {
    for (int h = 0; h < heads_q && tail_ok; ++h) {
      const size_t base = (static_cast<size_t>(t) * heads_q + h) * kHeadDim;
      for (int d = kRotaryDim; d < kHeadDim; ++d) {
        if (q_got[base + d] != q_h[base + d]) tail_ok = false;
      }
    }
  }

  const bool ok = rel_q < tol && rel_k < tol && tail_ok;
  std::printf("  %-34s sections=[%d,%d,%d] %s q=%.4e k=%.4e tail=%s -> %s\n", name, sec_t, sec_h,
              sec_w, long_context ? "norm_rel" : "max_rel ", rel_q, rel_k,
              tail_ok ? "ok" : "CHANGED", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(20260922);
  bool ok = true;

  // ---- case 1: this model's real sections, a real image-shaped position assignment -------------
  // 12 text tokens, then a 4x5 merged image grid (20 tokens, t constant, h/w meshgrid offset by
  // the run's start position), then 6 more text tokens continuing after the image's SPATIAL
  // extent -- i.e. exactly the shape Qwen3_5Model.get_rope_index emits (docs/vision.md).
  {
    const int tokens = 38;
    std::vector<int32_t> pos3(static_cast<size_t>(3 * tokens));
    int64_t cur = 0;
    int i = 0;
    for (; i < 12; ++i, ++cur) {
      pos3[static_cast<size_t>(i)] = static_cast<int32_t>(cur);
      pos3[static_cast<size_t>(tokens + i)] = static_cast<int32_t>(cur);
      pos3[static_cast<size_t>(2 * tokens + i)] = static_cast<int32_t>(cur);
    }
    for (int h = 0; h < 4; ++h) {
      for (int w = 0; w < 5; ++w, ++i) {
        pos3[static_cast<size_t>(i)] = static_cast<int32_t>(cur);
        pos3[static_cast<size_t>(tokens + i)] = static_cast<int32_t>(cur + h);
        pos3[static_cast<size_t>(2 * tokens + i)] = static_cast<int32_t>(cur + w);
      }
    }
    cur += 5;  // max(h_patches, w_patches) / merge_size
    for (; i < tokens; ++i, ++cur) {
      pos3[static_cast<size_t>(i)] = static_cast<int32_t>(cur);
      pos3[static_cast<size_t>(tokens + i)] = static_cast<int32_t>(cur);
      pos3[static_cast<size_t>(2 * tokens + i)] = static_cast<int32_t>(cur);
    }
    ok &= RunCase("text + 4x5 image + text", 11, 11, 10, pos3, tokens, &rng);
  }

  // ---- case 2: long-context positions, all three axes far apart --------------------------------
  // Both the accurate-sincos regime the single-row kernel's own test covers AND a case where a
  // swapped h/w stream cannot hide behind similar magnitudes.
  {
    const int tokens = 5;
    std::vector<int32_t> pos3 = {200000, 220000, 240000, 261000, 262143,  // t
                                  11,     1234,   60000,  200,    99999,   // h
                                  7,      4321,   31,     180000, 5};      // w
    ok &= RunCase("long-context, axes far apart", 11, 11, 10, pos3, tokens, &rng,
                  /*long_context=*/true);
  }

  // ---- case 3: a section split where the bounds matter ------------------------------------------
  // [16,10,6]: the height slice stops at bin 29 and the width slice at bin 17, so bins 30/31 (and
  // 20,23,26,29 for width) stay TEMPORAL. A kernel that assigned streams by `bin % 3` alone would
  // pass case 1 and fail here.
  {
    const int tokens = 9;
    std::vector<int32_t> pos3(static_cast<size_t>(3 * tokens));
    for (int t = 0; t < tokens; ++t) {
      pos3[static_cast<size_t>(t)] = 1000 + t;
      pos3[static_cast<size_t>(tokens + t)] = 30000 + 7 * t;
      pos3[static_cast<size_t>(2 * tokens + t)] = 90000 - 13 * t;
    }
    ok &= RunCase("non-default sections [16,10,6]", 16, 10, 6, pos3, tokens, &rng);

    // Prove the bound is load-bearing rather than incidental: the axis table for [16,10,6] really
    // does differ from plain bin%3, on 5 of the 32 bins -- bin 31 (which %3 would give to height,
    // but the height slice stops at 29) and bins 20/23/26/29 (which %3 would give to width, but
    // the width slice stops at 17).
    const std::vector<int> a_real = AxisTable(kHalf, 10, 6);
    int differing = 0;
    for (int i = 0; i < kHalf; ++i) {
      if (a_real[static_cast<size_t>(i)] != i % 3) ++differing;
    }
    std::printf("  [16,10,6] axis table differs from bin%%3 on %d/%d bins\n", differing, kHalf);
    ok &= (differing == 5);

    // ...and that [11,11,10] (this model's own) does NOT, which is why case 1 alone would be a
    // weaker test than it looks.
    const std::vector<int> a_model = AxisTable(kHalf, 11, 10);
    int model_differing = 0;
    for (int i = 0; i < kHalf; ++i) {
      if (a_model[static_cast<size_t>(i)] != i % 3) ++model_differing;
    }
    std::printf("  [11,11,10] axis table differs from bin%%3 on %d/%d bins\n", model_differing,
                kHalf);
    ok &= (model_differing == 0);
  }

  // ---- case 4: three identical rows == the single-row kernel, BIT for BIT -----------------------
  // This is the property the whole text-only-is-unchanged argument rests on (kernels.h): a caller
  // that has no image can keep taking r4dx_rope_partial_mrope_bf16 and get the identical tensor.
  {
    const int tokens = 29, heads_q = 24, heads_k = 4;
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<uint16_t> q_h(static_cast<size_t>(tokens) * heads_q * kHeadDim);
    std::vector<uint16_t> k_h(static_cast<size_t>(tokens) * heads_k * kHeadDim);
    for (auto& v : q_h) v = FloatToBf16(dist(rng));
    for (auto& v : k_h) v = FloatToBf16(dist(rng));

    std::vector<int32_t> pos1(static_cast<size_t>(tokens));
    std::vector<int32_t> pos3(static_cast<size_t>(3 * tokens));
    for (int t = 0; t < tokens; ++t) {
      pos1[static_cast<size_t>(t)] = 4096 + t;
      for (int axis = 0; axis < 3; ++axis) {
        pos3[static_cast<size_t>(axis * tokens + t)] = 4096 + t;
      }
    }

    DeviceBuffer<uint16_t> q1(q_h.size()), k1(k_h.size()), q3(q_h.size()), k3(k_h.size());
    DeviceBuffer<int32_t> p1(pos1.size()), p3(pos3.size());
    q1.CopyFromHost(q_h);
    k1.CopyFromHost(k_h);
    q3.CopyFromHost(q_h);
    k3.CopyFromHost(k_h);
    p1.CopyFromHost(pos1);
    p3.CopyFromHost(pos3);

    r4dx_rope_partial_mrope_bf16(reinterpret_cast<int64_t>(q1.data()),
                                  reinterpret_cast<int64_t>(k1.data()),
                                  reinterpret_cast<int64_t>(p1.data()), tokens, heads_q, heads_k,
                                  kHeadDim, kRotaryDim, kTheta, 0);
    r4dx_rope_partial_mrope3_bf16(reinterpret_cast<int64_t>(q3.data()),
                                   reinterpret_cast<int64_t>(k3.data()),
                                   reinterpret_cast<int64_t>(p3.data()), tokens, heads_q, heads_k,
                                   kHeadDim, kRotaryDim, kTheta, 11, 11, 10, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());

    const size_t dq = CountDiffering(q1.CopyToHost(), q3.CopyToHost());
    const size_t dk = CountDiffering(k1.CopyToHost(), k3.CopyToHost());
    std::printf("  identical-rows vs single-row kernel: q %zu/%zu differ, k %zu/%zu differ\n", dq,
                q_h.size(), dk, k_h.size());
    ok &= (dq == 0 && dk == 0);
  }

  // ---- case 5: bad sections are rejected, not silently miscomputed -------------------------------
  {
    bool threw = false;
    try {
      DeviceBuffer<uint16_t> q(static_cast<size_t>(1) * 24 * kHeadDim);
      DeviceBuffer<uint16_t> k(static_cast<size_t>(1) * 4 * kHeadDim);
      DeviceBuffer<int32_t> p(3);
      r4dx_rope_partial_mrope3_bf16(reinterpret_cast<int64_t>(q.data()),
                                     reinterpret_cast<int64_t>(k.data()),
                                     reinterpret_cast<int64_t>(p.data()), 1, 24, 4, kHeadDim,
                                     kRotaryDim, kTheta, 11, 11, 11, 0);  // sums to 33, not 32
    } catch (const std::exception&) {
      threw = true;
    }
    std::printf("  sections that do not sum to rotary_dim/2 throw: %s\n", threw ? "yes" : "no");
    ok &= threw;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
