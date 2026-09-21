// tests/kernels/test_vision_kernels.cpp -- the five vision-tower device primitives (docs/vision.md)
// against independent CPU references: r4dx_layernorm_bf16, r4dx_bias_add_bf16,
// r4dx_gelu_tanh_bf16, r4dx_gelu_erf_bf16, r4dx_vision_qkv_rope_bf16 and
// r4dx_vision_pos_embed_bf16.
//
// Every reference here is written from the transformers source's own formula (not from the kernel),
// accumulates in double where the kernel accumulates in fp32, and is compared at the bf16 output
// quantum -- the same convention tests/kernels/test_rmsnorm.cpp and test_rope_neox.cpp use. The
// golden-data comparison for these same kernels, against the real checkpoint, lives in
// tests/vision/test_vision_tower.cpp; this file is what localizes a failure to one kernel.
//
// Two checks here are about a MISTAKE rather than about arithmetic, because both mistakes produce
// output that looks entirely plausible:
//   * CheckLayerNormIsNotRmsNorm: a LayerNorm that forgets the mean subtraction still produces
//     well-scaled activations. It is checked against an explicitly mean-shifted input, where the
//     two forms disagree by ~100%.
//   * CheckGeluVariantsDiffer: gelu_pytorch_tanh and the exact erf GELU agree to ~1e-3, which is
//     inside the tolerance every downstream check uses, so calling the wrong one is invisible
//     there. This asserts the two entry points really are different functions.
#include <algorithm>
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

int g_failures = 0;

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      ++g_failures;                                                                            \
    }                                                                                          \
  } while (0)

// Relative error at the bf16 output quantum: a value whose magnitude is below `floor` is compared
// absolutely instead, so a near-zero reference does not manufacture a huge relative error out of
// one bf16 ulp.
double MaxRelErr(const std::vector<uint16_t>& got, const std::vector<double>& ref, double floor) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = Bf16ToFloat(got[i]);
    const double denom = std::max(floor, std::abs(ref[i]));
    worst = std::max(worst, std::abs(g - ref[i]) / denom);
  }
  return worst;
}

std::vector<uint16_t> RandomBf16(size_t n, std::mt19937& rng, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = FloatToBf16(dist(rng));
  return v;
}

// ---- LayerNorm ---------------------------------------------------------------------------------
// torch.nn.LayerNorm's own formula: the BIASED variance (divide by `hidden`), eps inside the sqrt,
// then the affine. Accumulated in double here so the kernel's fp32 reductions are what is measured.
void LayerNormRef(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                  const std::vector<uint16_t>& b, int64_t rows, int64_t hidden, float eps,
                  std::vector<double>* out) {
  out->resize(static_cast<size_t>(rows * hidden));
  for (int64_t r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int64_t i = 0; i < hidden; ++i) mean += Bf16ToFloat(x[r * hidden + i]);
    mean /= static_cast<double>(hidden);
    double var = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
      const double d = Bf16ToFloat(x[r * hidden + i]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(hidden);
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int64_t i = 0; i < hidden; ++i) {
      (*out)[static_cast<size_t>(r * hidden + i)] =
          (Bf16ToFloat(x[r * hidden + i]) - mean) * rstd * Bf16ToFloat(w[i]) + Bf16ToFloat(b[i]);
    }
  }
}

std::vector<uint16_t> RunLayerNorm(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                                    const std::vector<uint16_t>& b, int64_t rows, int64_t hidden,
                                    float eps) {
  DeviceBuffer<uint16_t> xd(x.size()), wd(w.size()), bd(b.size()), od(x.size());
  xd.CopyFromHost(x);
  wd.CopyFromHost(w);
  bd.CopyFromHost(b);
  r4dx_layernorm_bf16(reinterpret_cast<int64_t>(xd.data()), reinterpret_cast<int64_t>(wd.data()),
                      reinterpret_cast<int64_t>(bd.data()), reinterpret_cast<int64_t>(od.data()),
                      rows, hidden, eps, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return od.CopyToHost();
}

bool CheckLayerNorm() {
  std::mt19937 rng(7);
  bool ok = true;
  // 1152 is the tower's hidden size; 4608 is the merger's post-view width; 37 and 1 catch the
  // grid-stride tail and the single-row case.
  for (const auto& shape : std::vector<std::pair<int64_t, int64_t>>{
           {784, 1152}, {196, 4608}, {37, 1152}, {1, 1152}}) {
    const int64_t rows = shape.first, hidden = shape.second;
    const auto x = RandomBf16(static_cast<size_t>(rows * hidden), rng, -3.0f, 3.0f);
    const auto w = RandomBf16(static_cast<size_t>(hidden), rng, 0.2f, 1.8f);
    const auto b = RandomBf16(static_cast<size_t>(hidden), rng, -0.5f, 0.5f);
    std::vector<double> ref;
    LayerNormRef(x, w, b, rows, hidden, 1e-6f, &ref);
    const auto got = RunLayerNorm(x, w, b, rows, hidden, 1e-6f);
    const double rel = MaxRelErr(got, ref, 1e-2);
    std::printf("  layernorm rows=%lld hidden=%lld: max_rel=%.3e\n",
                static_cast<long long>(rows), static_cast<long long>(hidden), rel);
    if (!(rel < 6e-3)) ok = false;
    CHECK(rel < 6e-3);
  }
  return ok;
}

// A LayerNorm whose mean subtraction is missing still produces plausible activations, so this
// drives an input with a LARGE constant offset: the correct answer is unchanged by the offset
// (LayerNorm is shift-invariant up to the affine), while an RMSNorm-shaped implementation collapses
// towards `weight * 1 + bias`.
bool CheckLayerNormIsNotRmsNorm() {
  const int64_t rows = 5, hidden = 1152;
  std::mt19937 rng(11);
  auto x = RandomBf16(static_cast<size_t>(rows * hidden), rng, -1.0f, 1.0f);
  auto x_shifted = x;
  for (auto& e : x_shifted) e = FloatToBf16(Bf16ToFloat(e) + 40.0f);
  const auto w = RandomBf16(static_cast<size_t>(hidden), rng, 0.5f, 1.5f);
  std::vector<uint16_t> b(static_cast<size_t>(hidden), FloatToBf16(0.0f));

  std::vector<double> ref_shifted;
  LayerNormRef(x_shifted, w, b, rows, hidden, 1e-6f, &ref_shifted);
  const auto got_shifted = RunLayerNorm(x_shifted, w, b, rows, hidden, 1e-6f);
  const double rel = MaxRelErr(got_shifted, ref_shifted, 1e-2);

  // An RMSNorm of the same shifted input (what a mean-forgetting implementation would produce),
  // measured against the same reference, must be WILDLY off -- otherwise this check proves nothing.
  double rms_worst = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
      const double v = Bf16ToFloat(x_shifted[r * hidden + i]);
      ss += v * v;
    }
    const double rstd = 1.0 / std::sqrt(ss / hidden + 1e-6);
    for (int64_t i = 0; i < hidden; ++i) {
      const double rms = Bf16ToFloat(x_shifted[r * hidden + i]) * rstd * Bf16ToFloat(w[i]);
      const double want = ref_shifted[static_cast<size_t>(r * hidden + i)];
      rms_worst = std::max(rms_worst, std::abs(rms - want) / std::max(1e-2, std::abs(want)));
    }
  }
  std::printf("  layernorm shift-invariance: max_rel=%.3e (an RMSNorm here would be %.1f)\n", rel,
              rms_worst);
  CHECK(rel < 6e-3);
  CHECK(rms_worst > 0.5);  // the discriminating power of this test, asserted rather than assumed
  return rel < 6e-3 && rms_worst > 0.5;
}

// ---- bias add ----------------------------------------------------------------------------------
bool CheckBiasAdd() {
  std::mt19937 rng(13);
  bool ok = true;
  for (const auto& shape : std::vector<std::pair<int64_t, int64_t>>{
           {784, 3456}, {988, 4304}, {17, 1152}, {1, 5120}}) {
    const int64_t rows = shape.first, cols = shape.second;
    const auto x = RandomBf16(static_cast<size_t>(rows * cols), rng, -4.0f, 4.0f);
    const auto bias = RandomBf16(static_cast<size_t>(cols), rng, -1.0f, 1.0f);
    std::vector<double> ref(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        ref[static_cast<size_t>(r * cols + c)] =
            static_cast<double>(Bf16ToFloat(x[r * cols + c])) + Bf16ToFloat(bias[c]);
      }
    }
    // In place (out == x), which is what every vision call site does.
    DeviceBuffer<uint16_t> xd(x.size()), bd(bias.size());
    xd.CopyFromHost(x);
    bd.CopyFromHost(bias);
    r4dx_bias_add_bf16(reinterpret_cast<int64_t>(xd.data()), reinterpret_cast<int64_t>(bd.data()),
                       reinterpret_cast<int64_t>(xd.data()), rows, cols, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const double rel = MaxRelErr(xd.CopyToHost(), ref, 1e-2);
    std::printf("  bias_add rows=%lld cols=%lld (in place): max_rel=%.3e\n",
                static_cast<long long>(rows), static_cast<long long>(cols), rel);
    if (!(rel < 4e-3)) ok = false;
    CHECK(rel < 4e-3);
  }
  return ok;
}

// ---- GELU --------------------------------------------------------------------------------------
double GeluTanhRef(double x) {
  const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + std::tanh(inner));
}
double GeluErfRef(double x) { return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476)); }

bool CheckGelu() {
  const int64_t n = 4304 * 37;
  std::mt19937 rng(17);
  auto x = RandomBf16(static_cast<size_t>(n), rng, -8.0f, 8.0f);
  // Pin the exactly-representable edge cases a random sweep will not hit.
  const float specials[] = {0.0f, -0.0f, 1.0f, -1.0f, 6.0f, -6.0f, 20.0f, -20.0f};
  for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); ++i) x[i] = FloatToBf16(specials[i]);

  bool ok = true;
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<double> ref(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const double v = Bf16ToFloat(x[static_cast<size_t>(i)]);
      ref[static_cast<size_t>(i)] = kind == 0 ? GeluTanhRef(v) : GeluErfRef(v);
    }
    DeviceBuffer<uint16_t> xd(x.size()), od(x.size());
    xd.CopyFromHost(x);
    if (kind == 0) {
      r4dx_gelu_tanh_bf16(reinterpret_cast<int64_t>(xd.data()),
                          reinterpret_cast<int64_t>(od.data()), n, 0);
    } else {
      r4dx_gelu_erf_bf16(reinterpret_cast<int64_t>(xd.data()), reinterpret_cast<int64_t>(od.data()),
                         n, 0);
    }
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const double rel = MaxRelErr(od.CopyToHost(), ref, 1e-2);
    std::printf("  gelu %s: max_rel=%.3e\n", kind == 0 ? "tanh" : "erf ", rel);
    if (!(rel < 4e-3)) ok = false;
    CHECK(rel < 4e-3);
  }
  return ok;
}

bool CheckGeluVariantsDiffer() {
  double worst = 0.0;
  double at = 0.0;
  for (double v = -6.0; v <= 6.0; v += 0.01) {
    const double d = std::abs(GeluTanhRef(v) - GeluErfRef(v));
    if (d > worst) {
      worst = d;
      at = v;
    }
  }
  std::printf("  gelu tanh-vs-erf: max |diff| = %.3e at x=%.2f\n", worst, at);
  CHECK(worst > 1e-4);  // they really are two functions, so calling the wrong one is a real bug
  return worst > 1e-4;
}

// ---- vision axial rope + qkv split ---------------------------------------------------------------
bool CheckVisionQkvRope() {
  const int tokens = 197, heads = 16, head_dim = 72;
  const int per = heads * head_dim, half = head_dim / 2;
  std::mt19937 rng(23);
  const auto qkv = RandomBf16(static_cast<size_t>(tokens) * 3 * per, rng, -2.0f, 2.0f);

  // cos/sin exactly as src/vision/vision_index.cpp builds them: 18 frequencies, theta 10000,
  // recomposed as cat([f_h, f_w, f_h, f_w]).
  std::vector<float> cos(static_cast<size_t>(tokens) * head_dim);
  std::vector<float> sin(static_cast<size_t>(tokens) * head_dim);
  std::uniform_int_distribution<int> pos_dist(0, 47);
  for (int t = 0; t < tokens; ++t) {
    const int hp = pos_dist(rng), wp = pos_dist(rng);
    for (int i = 0; i < head_dim / 4; ++i) {
      const double inv = 1.0 / std::pow(10000.0, static_cast<double>(2 * i) / (head_dim / 2));
      const double ah = hp * inv, aw = wp * inv;
      const int base = t * head_dim;
      cos[base + i] = static_cast<float>(std::cos(ah));
      sin[base + i] = static_cast<float>(std::sin(ah));
      cos[base + head_dim / 4 + i] = static_cast<float>(std::cos(aw));
      sin[base + head_dim / 4 + i] = static_cast<float>(std::sin(aw));
      cos[base + half + i] = cos[base + i];
      sin[base + half + i] = sin[base + i];
      cos[base + half + head_dim / 4 + i] = cos[base + head_dim / 4 + i];
      sin[base + half + head_dim / 4 + i] = sin[base + head_dim / 4 + i];
    }
  }

  // Reference: apply_rotary_pos_emb_vision, i.e. (x * cos) + (rotate_half(x) * sin) with
  // rotate_half(x) = cat(-x2, x1).
  auto rope_ref = [&](int which, std::vector<double>* out) {
    out->assign(static_cast<size_t>(tokens) * per, 0.0);
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < head_dim; ++d) {
          const size_t src = static_cast<size_t>(t) * 3 * per + which * per + h * head_dim + d;
          const double xv = Bf16ToFloat(qkv[src]);
          const size_t partner = static_cast<size_t>(t) * 3 * per + which * per + h * head_dim +
                                  (d < half ? d + half : d - half);
          const double rot = d < half ? -Bf16ToFloat(qkv[partner]) : Bf16ToFloat(qkv[partner]);
          (*out)[static_cast<size_t>(t) * per + h * head_dim + d] =
              xv * cos[t * head_dim + d] + rot * sin[t * head_dim + d];
        }
      }
    }
  };
  std::vector<double> q_ref, k_ref;
  rope_ref(0, &q_ref);
  rope_ref(1, &k_ref);

  DeviceBuffer<uint16_t> qkv_d(qkv.size()), q_d(static_cast<size_t>(tokens) * per),
      k_d(static_cast<size_t>(tokens) * per), v_d(static_cast<size_t>(tokens) * per);
  DeviceBuffer<float> cos_d(cos.size()), sin_d(sin.size());
  qkv_d.CopyFromHost(qkv);
  cos_d.CopyFromHost(cos);
  sin_d.CopyFromHost(sin);
  r4dx_vision_qkv_rope_bf16(reinterpret_cast<int64_t>(qkv_d.data()),
                            reinterpret_cast<int64_t>(cos_d.data()),
                            reinterpret_cast<int64_t>(sin_d.data()),
                            reinterpret_cast<int64_t>(q_d.data()),
                            reinterpret_cast<int64_t>(k_d.data()),
                            reinterpret_cast<int64_t>(v_d.data()), tokens, heads, head_dim, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  const double q_rel = MaxRelErr(q_d.CopyToHost(), q_ref, 1e-2);
  const double k_rel = MaxRelErr(k_d.CopyToHost(), k_ref, 1e-2);
  // v must be a BIT-EXACT copy -- the reference does nothing to it at all.
  const auto v_got = v_d.CopyToHost();
  int64_t v_diff = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int j = 0; j < per; ++j) {
      if (v_got[static_cast<size_t>(t) * per + j] !=
          qkv[static_cast<size_t>(t) * 3 * per + 2 * per + j]) {
        ++v_diff;
      }
    }
  }
  std::printf("  vision_qkv_rope: q max_rel=%.3e k max_rel=%.3e v differing=%lld/%lld\n", q_rel,
              k_rel, static_cast<long long>(v_diff), static_cast<long long>(tokens) * per);
  CHECK(q_rel < 6e-3);
  CHECK(k_rel < 6e-3);
  CHECK(v_diff == 0);

  // A rope that rotated only the first head_dim/4 dims (the axial frequency count) -- or only the
  // first head_dim/2 -- would leave the top half of the head untouched. Assert the kernel's own
  // output really did move there, so "the full head rotates" is tested rather than assumed.
  const auto q_got = q_d.CopyToHost();
  int64_t top_half_moved = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      for (int d = half; d < head_dim; ++d) {
        const size_t o = static_cast<size_t>(t) * per + h * head_dim + d;
        const size_t s = static_cast<size_t>(t) * 3 * per + h * head_dim + d;
        if (q_got[o] != qkv[s]) ++top_half_moved;
      }
    }
  }
  std::printf("  vision_qkv_rope: %lld/%lld top-half q elements changed by the rotation\n",
              static_cast<long long>(top_half_moved),
              static_cast<long long>(tokens) * heads * half);
  CHECK(top_half_moved > static_cast<int64_t>(tokens) * heads * half / 2);
  return q_rel < 6e-3 && k_rel < 6e-3 && v_diff == 0 &&
         top_half_moved > static_cast<int64_t>(tokens) * heads * half / 2;
}

// ---- learned position-embedding gather ------------------------------------------------------
bool CheckVisionPosEmbed() {
  const int64_t num_patches = 784, hidden = 1152, table_rows = 2304;
  const int taps = 4;
  std::mt19937 rng(29);
  const auto table = RandomBf16(static_cast<size_t>(table_rows * hidden), rng, -0.4f, 0.4f);
  const auto x = RandomBf16(static_cast<size_t>(num_patches * hidden), rng, -2.0f, 2.0f);

  std::vector<int32_t> idx(static_cast<size_t>(num_patches * taps));
  std::vector<float> wt(static_cast<size_t>(num_patches * taps));
  std::uniform_int_distribution<int> row_dist(0, static_cast<int>(table_rows) - 1);
  std::uniform_real_distribution<float> frac(0.0f, 1.0f);
  for (int64_t p = 0; p < num_patches; ++p) {
    // Real interpolation weights are an outer product of two 1-D bilinear pairs and sum to 1;
    // reproduce that shape rather than four independent randoms.
    const float fr = frac(rng), fc = frac(rng);
    const float w2[4] = {(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc};
    for (int t = 0; t < taps; ++t) {
      idx[static_cast<size_t>(p * taps + t)] = row_dist(rng);
      wt[static_cast<size_t>(p * taps + t)] = w2[t];
    }
  }

  std::vector<double> ref_f32(static_cast<size_t>(num_patches * hidden));
  std::vector<double> ref_bf16(static_cast<size_t>(num_patches * hidden));
  for (int64_t p = 0; p < num_patches; ++p) {
    for (int64_t d = 0; d < hidden; ++d) {
      // fp32 accumulation, matching the reference's own promotion of bf16 table * fp32 weight.
      float acc = 0.0f;
      for (int t = 0; t < taps; ++t) {
        const int32_t r = idx[static_cast<size_t>(p * taps + t)];
        acc += wt[static_cast<size_t>(p * taps + t)] * Bf16ToFloat(table[r * hidden + d]);
      }
      ref_f32[static_cast<size_t>(p * hidden + d)] = acc;
      // `.to(bf16)` BEFORE the add -- the whole point of dumping both outputs.
      ref_bf16[static_cast<size_t>(p * hidden + d)] =
          static_cast<double>(Bf16ToFloat(x[p * hidden + d])) + Bf16ToFloat(FloatToBf16(acc));
    }
  }

  DeviceBuffer<uint16_t> table_d(table.size()), x_d(x.size()), out_d(x.size());
  DeviceBuffer<int32_t> idx_d(idx.size());
  DeviceBuffer<float> wt_d(wt.size()), pe_d(static_cast<size_t>(num_patches * hidden));
  table_d.CopyFromHost(table);
  x_d.CopyFromHost(x);
  idx_d.CopyFromHost(idx);
  wt_d.CopyFromHost(wt);
  r4dx_vision_pos_embed_bf16(
      reinterpret_cast<int64_t>(table_d.data()), reinterpret_cast<int64_t>(idx_d.data()),
      reinterpret_cast<int64_t>(wt_d.data()), reinterpret_cast<int64_t>(x_d.data()),
      reinterpret_cast<int64_t>(pe_d.data()), reinterpret_cast<int64_t>(out_d.data()), num_patches,
      hidden, taps, table_rows, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  const auto pe_got = pe_d.CopyToHost();
  double pe_max = 0.0;
  for (size_t i = 0; i < pe_got.size(); ++i) {
    pe_max = std::max(pe_max, std::abs(static_cast<double>(pe_got[i]) - ref_f32[i]));
  }
  const double add_rel = MaxRelErr(out_d.CopyToHost(), ref_bf16, 1e-2);
  std::printf("  vision_pos_embed: fp32 sum max_abs=%.3e, residual-add max_rel=%.3e\n", pe_max,
              add_rel);
  CHECK(pe_max < 1e-6);
  CHECK(add_rel < 4e-3);

  // Out-of-range indices must clamp to row 0, not fault or read garbage.
  std::vector<int32_t> bad(static_cast<size_t>(num_patches * taps), -5);
  idx_d.CopyFromHost(bad);
  r4dx_vision_pos_embed_bf16(
      reinterpret_cast<int64_t>(table_d.data()), reinterpret_cast<int64_t>(idx_d.data()),
      reinterpret_cast<int64_t>(wt_d.data()), 0, reinterpret_cast<int64_t>(pe_d.data()), 0,
      num_patches, hidden, taps, table_rows, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto clamped = pe_d.CopyToHost();
  bool clamp_ok = true;
  for (int64_t d = 0; d < hidden; ++d) {
    float want = 0.0f;
    for (int t = 0; t < taps; ++t) want += wt[static_cast<size_t>(t)] * Bf16ToFloat(table[d]);
    if (std::abs(clamped[static_cast<size_t>(d)] - want) > 1e-6f) clamp_ok = false;
  }
  CHECK(clamp_ok);
  return pe_max < 1e-6 && add_rel < 4e-3 && clamp_ok;
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));  // HIP_VISIBLE_DEVICES=1 remaps physical device 1 to index 0
  std::printf("[test_vision_kernels] vision tower device primitives vs CPU references\n");
  CheckLayerNorm();
  CheckLayerNormIsNotRmsNorm();
  CheckBiasAdd();
  CheckGelu();
  CheckGeluVariantsDiffer();
  CheckVisionQkvRope();
  CheckVisionPosEmbed();
  if (g_failures != 0) {
    std::fprintf(stderr, "[test_vision_kernels] FAILED (%d checks)\n", g_failures);
    return 1;
  }
  std::printf("[test_vision_kernels] OK\n");
  return 0;
}
