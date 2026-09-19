// tests/kernels/test_rope.cpp -- r4dx_rope_partial_mrope_bf16 (text-only) vs a CPU fp32
// reference implementing modeling_qwen3_5.py's apply_rotary_pos_emb: NeoX/half-split pairing
// (rotate_half), rotary_dim = head_dim * partial_rotary_factor = 256*0.25 = 64, theta 1e7. For
// text-only input the three mrope position streams (t,h,w) all equal the token position, so this
// reduces to ordinary 1D rope over `pos_ids` -- see the TODO in kernels.h for the multimodal case.
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

// inv_freq AND the angle multiply are computed in float32 via the exact same expression order the
// kernel itself uses (exp2f(log2f(...)) for inv_freq, then a plain float `pos * inv_freq`) rather
// than in double -- at pos up to 262143 (needing 18 of float32's 24 mantissa bits), that plain
// float32 multiply alone has an inherent ~2^-5 rad (~0.03 rad) quantization step, which would
// swamp any difference this reference is actually trying to isolate if computed in double
// instead. Only the FINAL sin/cos evaluation of that (float32-quantized) angle is done at double
// precision, so this reference isolates exactly what's under test here -- sincosf's own accuracy
// given an angle, not the angle computation's own float32 rounding (a separate, inherent
// characteristic of doing rope in float32 at all, not something this change touches).
void RopeRef(std::vector<uint16_t>* x, int tokens, int heads, int head_dim, int rotary_dim,
             float theta, const std::vector<int32_t>& pos_ids) {
  const int half = rotary_dim / 2;
  for (int t = 0; t < tokens; ++t) {
    float pos = static_cast<float>(pos_ids[t]);
    for (int hd = 0; hd < half; ++hd) {
      float inv_freq = std::exp2(-(2.0f * hd) / static_cast<float>(rotary_dim) * std::log2(theta));
      float angle_f = pos * inv_freq;
      double angle = static_cast<double>(angle_f);
      double c = std::cos(angle), s = std::sin(angle);
      for (int h = 0; h < heads; ++h) {
        size_t base = (static_cast<size_t>(t) * heads + h) * head_dim;
        double x0 = Bf16ToFloat((*x)[base + hd]);
        double x1 = Bf16ToFloat((*x)[base + hd + half]);
        (*x)[base + hd] = FloatToBf16(static_cast<float>(x0 * c - x1 * s));
        (*x)[base + hd + half] = FloatToBf16(static_cast<float>(x1 * c + x0 * s));
      }
    }
  }
}

double MaxRelErr(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double max_rel = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
    max_rel = std::max(max_rel, std::abs(g - r) / std::max(1e-2, std::abs(r)));
  }
  return max_rel;
}

// Whole-tensor norm-relative error, the same style test_kv_write.cpp/test_gdn_chunk_scan.cpp use
// for regimes where a handful of near-zero reference elements (here: rotate_half's x0*c - x1*s
// landing near zero for a particular (token, head, freq-bin) by chance) would otherwise make a
// per-element max-relative metric spike on an entry the 1e-2 floor was never meant to cover,
// without that spike reflecting a real error in the rotation as a whole.
double NormRelErr(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
    num += (g - r) * (g - r);
    den += r * r;
  }
  return std::sqrt(num) / std::max(1e-9, std::sqrt(den));
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  const int tokens = 37, heads_q = 24, heads_k = 4, head_dim = 256, rotary_dim = 64;
  const float theta = 1.0e7f;

  std::mt19937 rng(19);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<uint16_t> q_h(static_cast<size_t>(tokens) * heads_q * head_dim);
  std::vector<uint16_t> k_h(static_cast<size_t>(tokens) * heads_k * head_dim);
  for (auto& v : q_h) v = FloatToBf16(dist(rng));
  for (auto& v : k_h) v = FloatToBf16(dist(rng));
  std::vector<int32_t> pos_ids(tokens);
  for (int t = 0; t < tokens; ++t) pos_ids[t] = t + 5;  // arbitrary non-zero start position

  std::vector<uint16_t> q_ref = q_h, k_ref = k_h;
  RopeRef(&q_ref, tokens, heads_q, head_dim, rotary_dim, theta, pos_ids);
  RopeRef(&k_ref, tokens, heads_k, head_dim, rotary_dim, theta, pos_ids);

  DeviceBuffer<uint16_t> q_d(q_h.size()), k_d(k_h.size());
  DeviceBuffer<int32_t> pos_d(pos_ids.size());
  q_d.CopyFromHost(q_h);
  k_d.CopyFromHost(k_h);
  pos_d.CopyFromHost(pos_ids);

  r4dx_rope_partial_mrope_bf16(reinterpret_cast<int64_t>(q_d.data()),
                                reinterpret_cast<int64_t>(k_d.data()),
                                reinterpret_cast<int64_t>(pos_d.data()), tokens, heads_q, heads_k,
                                head_dim, rotary_dim, theta, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint16_t> q_got = q_d.CopyToHost();
  std::vector<uint16_t> k_got = k_d.CopyToHost();

  double rel_q = MaxRelErr(q_got, q_ref);
  double rel_k = MaxRelErr(k_got, k_ref);
  std::printf("rope max rel err: q=%.4e k=%.4e\n", rel_q, rel_k);

  // The untouched tail [rotary_dim, head_dim) must be bit-for-bit unchanged.
  bool tail_ok = true;
  for (int t = 0; t < tokens && tail_ok; ++t) {
    for (int h = 0; h < heads_q && tail_ok; ++h) {
      size_t base = (static_cast<size_t>(t) * heads_q + h) * head_dim;
      for (int d = rotary_dim; d < head_dim; ++d) {
        if (q_got[base + d] != q_h[base + d]) tail_ok = false;
      }
    }
  }
  std::printf("rope pass-through tail unchanged: %s\n", tail_ok ? "yes" : "no");

  bool ok = rel_q < 2e-2 && rel_k < 2e-2 && tail_ok;

  // Long-context regime: pos_ids near max_position_embeddings (262144), where the kernel's
  // __sincosf fast-math intrinsic (fixed in this change to accurate sincosf) previously drifted
  // several times bf16's own ~4e-3 resolution. This is the only regime that distinguishes
  // accurate from approximate sincos, so it must be exercised explicitly rather than relying on
  // the small-position case above. Uses a norm-relative metric (not MaxRelErr) because at these
  // positions rotate_half's x0*c - x1*s can land near zero for a particular (token, head,
  // freq-bin) by chance, and a per-element max-relative check would spike on that one entry
  // regardless of whether sincos is accurate.
  {
    const int tokens_lc = 5, heads_lc = 4;
    std::vector<uint16_t> q_lc(static_cast<size_t>(tokens_lc) * heads_lc * head_dim);
    for (auto& v : q_lc) v = FloatToBf16(dist(rng));
    std::vector<int32_t> pos_lc = {200000, 220000, 240000, 261000, 262143};

    std::vector<uint16_t> q_lc_ref = q_lc;
    RopeRef(&q_lc_ref, tokens_lc, heads_lc, head_dim, rotary_dim, theta, pos_lc);

    DeviceBuffer<uint16_t> q_lc_d(q_lc.size()), k_dummy_d(q_lc.size());
    DeviceBuffer<int32_t> pos_lc_d(pos_lc.size());
    q_lc_d.CopyFromHost(q_lc);
    k_dummy_d.CopyFromHost(q_lc);  // k buffer, unused by this check but required by the call
    pos_lc_d.CopyFromHost(pos_lc);

    r4dx_rope_partial_mrope_bf16(reinterpret_cast<int64_t>(q_lc_d.data()),
                                  reinterpret_cast<int64_t>(k_dummy_d.data()),
                                  reinterpret_cast<int64_t>(pos_lc_d.data()), tokens_lc, heads_lc,
                                  heads_lc, head_dim, rotary_dim, theta, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> q_lc_got = q_lc_d.CopyToHost();

    double rel_lc = NormRelErr(q_lc_got, q_lc_ref);
    std::printf("rope long-context (pos up to 262143) norm rel err=%.4e\n", rel_lc);
    ok = ok && rel_lc < 5e-2;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
