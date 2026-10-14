// tests/kernels/lfm2_ref.hpp -- CPU references for the LFM2 kernels (docs/lfm2-plan.md 3.3 / 3.5; task L1-6;
// include/r4dx/kernels/lfm2_kernels.h). Header-only, host code only, no HIP call: test_lfm2_ref_cpu runs
// without a device, the GPU tests (test_lfm2_*) compare their kernels against the same functions.
//
// Tensors are std::vector<float> holding bf16-representable values (rounded with Bf16Round) unless a
// function says double. Each reference is written from the HF op sequence (Lfm2ShortConv, Lfm2MoeSparse-
// MoeBlock) rather than from the kernels' loop structure.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace lfm2_ref {

// Round-to-nearest-even to bf16 and back (finite inputs); self-contained so the CPU test needs no HIP.
inline float Bf16Round(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

// ---- ShortConv ---------------------------------------------------------------------------------------------
// The kernel's contract: bcx [T, stride] (B | C | x per row, stride >= 3H), state [2][H] in/out
// (state[0..H) = h at t-2, state[H..2H) = h at t-1), weight [H][3] (oldest..newest). Returns out [T, H].
//   h_t = bf16(B_t * x_t); acc = (w0*h_{t-2} + w1*h_{t-1}) + w2*h_t in fp32; out = bf16(C_t * bf16(acc)).
inline std::vector<float> ShortConv(const std::vector<float>& bcx, std::vector<float>* state,
                                    const std::vector<float>& weight, int T, int H, int64_t stride) {
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (int c = 0; c < H; ++c) {
    float h2 = (*state)[c], h1 = (*state)[H + c];
    const float w0 = weight[c * 3 + 0], w1 = weight[c * 3 + 1], w2 = weight[c * 3 + 2];
    for (int t = 0; t < T; ++t) {
      const float* row = &bcx[static_cast<size_t>(t) * stride];
      const float h0 = Bf16Round(row[c] * row[2 * H + c]);
      const float acc = Bf16Round((w0 * h2 + w1 * h1) + w2 * h0);
      out[static_cast<size_t>(t) * H + c] = Bf16Round(row[H + c] * acc);
      h2 = h1;
      h1 = h0;
    }
    (*state)[c] = h2;
    (*state)[H + c] = h1;
  }
  return out;
}

// The same op written as HF does, on the WHOLE sequence at once: Bx = B * x (bf16 elementwise), then a
// depthwise causal conv1d (left zero padding of kernel-1 = 2 columns, i.e. a zero initial state; here the
// padding is an explicit prefix `init_state` so a nonzero start can be tested), the conv accumulated in
// double and rounded to bf16 once, then C * conv_out. No chunking, no state machine. out [T, H].
inline std::vector<float> ShortConvWholeSequence(const std::vector<float>& bcx, const std::vector<float>& init_state,
                                                 const std::vector<float>& weight, int T, int H,
                                                 int64_t stride) {
  const int L = 3;
  // padded[t'][c], t' = 0..T+1: two history columns (init_state rows 0,1) then Bx.
  std::vector<float> padded(static_cast<size_t>(T + L - 1) * H);
  for (int c = 0; c < H; ++c) {
    padded[0 * H + c] = init_state[c];
    padded[1 * H + c] = init_state[H + c];
  }
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < H; ++c) {
      const float* row = &bcx[static_cast<size_t>(t) * stride];
      padded[static_cast<size_t>(t + L - 1) * H + c] = Bf16Round(row[c] * row[2 * H + c]);
    }
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < H; ++c) {
      double acc = 0.0;
      for (int j = 0; j < L; ++j)  // tap j multiplies padded[t + j] (j = 0 oldest)
        acc += static_cast<double>(weight[c * 3 + j]) * padded[static_cast<size_t>(t + j) * H + c];
      const float conv = Bf16Round(static_cast<float>(acc));
      out[static_cast<size_t>(t) * H + c] = Bf16Round(bcx[static_cast<size_t>(t) * stride + H + c] * conv);
    }
  return out;
}

// ---- router ----------------------------------------------------------------------------------------------
struct RouterOut {
  std::vector<int32_t> ids;  // [T, k]
  std::vector<float> w;      // [T, k]
  std::vector<float> score;  // [T, E] (s + bias, fp32), for near-tie diagnostics
};

// Top-k of `score` descending, ties -> lower index (the kernel's rule, written as a stable sort).
inline std::vector<int> TopKStable(const float* score, int E, int k) {
  std::vector<int> idx(E);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return score[a] > score[b]; });
  idx.resize(k);
  return idx;
}

// Selection + weights given the per-expert unbiased sigmoid s[T][E] (fp32) and the bias (fp32):
//   score = s + bias (fp32); sel = top-k (ties -> lower index); w = s[sel] / (sum_slots s[sel] + 1e-6) * scale.
inline RouterOut RouterFromSigmoid(const std::vector<float>& s, const std::vector<float>& bias, int T, int E,
                                   int k, float scale) {
  RouterOut o;
  o.ids.resize(static_cast<size_t>(T) * k);
  o.w.resize(static_cast<size_t>(T) * k);
  o.score.resize(static_cast<size_t>(T) * E);
  for (int t = 0; t < T; ++t) {
    float* sc = &o.score[static_cast<size_t>(t) * E];
    for (int e = 0; e < E; ++e) sc[e] = s[static_cast<size_t>(t) * E + e] + bias[e];
    const std::vector<int> sel = TopKStable(sc, E, k);
    float sum = 0.0f;
    for (int r = 0; r < k; ++r) sum = sum + s[static_cast<size_t>(t) * E + sel[r]];
    const float denom = sum + 1e-6f;
    for (int r = 0; r < k; ++r) {
      o.ids[static_cast<size_t>(t) * k + r] = sel[r];
      o.w[static_cast<size_t>(t) * k + r] = (s[static_cast<size_t>(t) * E + sel[r]] / denom) * scale;
    }
  }
  return o;
}

// Full router: logits in DOUBLE (the exact dot product; the kernel is fp32, within ~1e-6 relative),
// sigmoid in double, rounded to fp32. x [T, hidden] (row stride x_stride), gate_w [E, hidden].
inline RouterOut Router(const std::vector<float>& x, const std::vector<float>& gate_w,
                        const std::vector<float>& bias, int T, int hidden, int E, int k, float scale,
                        int64_t x_stride) {
  std::vector<float> s(static_cast<size_t>(T) * E);
  for (int t = 0; t < T; ++t)
    for (int e = 0; e < E; ++e) {
      double l = 0.0;
      for (int j = 0; j < hidden; ++j)
        l += static_cast<double>(x[static_cast<size_t>(t) * x_stride + j]) * gate_w[static_cast<size_t>(e) * hidden + j];
      s[static_cast<size_t>(t) * E + e] = static_cast<float>(1.0 / (1.0 + std::exp(-l)));
    }
  return RouterFromSigmoid(s, bias, T, E, k, scale);
}

// ---- expert counting sort ---------------------------------------------------------------------------------
struct SortOut {
  std::vector<int32_t> counts, offsets, sorted_pairs, pos_of_pair;
};

// Two-pass counting sort, the kernel's algorithm: ascending pair index inside each expert.
inline SortOut MoeSort(const std::vector<int32_t>& ids, int n_experts) {
  const int P = static_cast<int>(ids.size());
  SortOut o;
  o.counts.assign(n_experts, 0);
  o.offsets.assign(n_experts + 1, 0);
  o.sorted_pairs.assign(P, 0);
  o.pos_of_pair.assign(P, -1);
  for (int p = 0; p < P; ++p)
    if (ids[p] >= 0 && ids[p] < n_experts) ++o.counts[ids[p]];
  for (int e = 0; e < n_experts; ++e) o.offsets[e + 1] = o.offsets[e] + o.counts[e];
  std::vector<int32_t> cur(o.offsets.begin(), o.offsets.end() - 1);
  for (int p = 0; p < P; ++p) {
    if (ids[p] < 0 || ids[p] >= n_experts) continue;
    const int pos = cur[ids[p]]++;
    o.sorted_pairs[pos] = p;
    o.pos_of_pair[p] = pos;
  }
  return o;
}

// ---- combine ------------------------------------------------------------------------------------------------
// out[t, :] = bf16(sum_{slot in order} w * y_row), fp32 accumulation from 0. `row_of_pair` maps the flat pair
// index to the row of y (nullptr = identity, the [T, k, hidden] slot layout); negative rows are skipped.
inline std::vector<float> Combine(const std::vector<float>& y, const std::vector<int32_t>* row_of_pair,
                                  const std::vector<float>& w, int T, int k, int hidden) {
  std::vector<float> out(static_cast<size_t>(T) * hidden);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < hidden; ++i) {
      float acc = 0.0f;
      for (int s = 0; s < k; ++s) {
        const int64_t p = static_cast<int64_t>(t) * k + s;
        const int64_t row = row_of_pair ? (*row_of_pair)[p] : p;
        if (row < 0) continue;
        acc = acc + w[p] * y[static_cast<size_t>(row) * hidden + i];
      }
      out[static_cast<size_t>(t) * hidden + i] = Bf16Round(acc);
    }
  return out;
}

// ---- rmsnorm (head_dim 64 per-head qk norm) ----------------------------------------------------------------
// r4dx_rmsnorm_plain_bf16's contract: out = bf16(x * rsqrt(mean(x^2) + eps) * w), fp32 math, ONE rounding
// (double accumulation here; the kernel's fp32 tree differs by ~1e-7 relative).
inline std::vector<float> RmsNormPlain(const std::vector<float>& x, const std::vector<float>& w, int rows,
                                       int hidden, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int i = 0; i < hidden; ++i) ss += static_cast<double>(x[static_cast<size_t>(r) * hidden + i]) * x[static_cast<size_t>(r) * hidden + i];
    const float rstd = static_cast<float>(1.0 / std::sqrt(ss / hidden + eps));
    for (int i = 0; i < hidden; ++i)
      out[static_cast<size_t>(r) * hidden + i] = Bf16Round(x[static_cast<size_t>(r) * hidden + i] * rstd * w[i]);
  }
  return out;
}

// HF Lfm2MoeRMSNorm: weight * (x * rsqrt(mean(x^2) + eps)).to(bf16) -- the normalized value is rounded to bf16
// BEFORE the weight multiply (two roundings). Used to report how far the one-rounding kernel is from HF.
inline std::vector<float> RmsNormHf(const std::vector<float>& x, const std::vector<float>& w, int rows,
                                    int hidden, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int i = 0; i < hidden; ++i) ss += static_cast<double>(x[static_cast<size_t>(r) * hidden + i]) * x[static_cast<size_t>(r) * hidden + i];
    const float rstd = static_cast<float>(1.0 / std::sqrt(ss / hidden + eps));
    for (int i = 0; i < hidden; ++i) {
      const float n = Bf16Round(x[static_cast<size_t>(r) * hidden + i] * rstd);
      out[static_cast<size_t>(r) * hidden + i] = Bf16Round(w[i] * n);
    }
  }
  return out;
}

// HF NeoX rope (rotate_half) on one head, double precision: out = x*cos + rotate_half(x)*sin with
// inv_freq[i] = theta^(-2i/head_dim), rotate_half(x) = cat(-x2, x1). head_dim = x.size().
inline std::vector<double> RopeNeox(const std::vector<double>& x, double theta, int pos) {
  const int hd = static_cast<int>(x.size()), half = hd / 2;
  std::vector<double> o(hd);
  for (int i = 0; i < half; ++i) {
    const double ang = pos * std::pow(theta, -2.0 * i / hd);
    const double c = std::cos(ang), s = std::sin(ang);
    o[i] = x[i] * c - x[i + half] * s;
    o[i + half] = x[i + half] * c + x[i] * s;
  }
  return o;
}

}  // namespace lfm2_ref
