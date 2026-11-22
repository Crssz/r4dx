// r4dx::inkling2_ref -- the CPU reference (the SPEC) of the Inkling2 (inkling2_loop) kernels in
// inkling2_kernels.h (docs/inkling2-plan.md 3.3, 5). Header-only, host-only, dependency-free: fp32 math in
// a fixed, written-out order, HF transformers semantics (models/inkling2_loop/modeling_inkling2_loop.py,
// merged 2026-11-21). Every function names the HF module it mirrors. The GPU kernels compute the same
// expressions; their sums run in a different (block-reduction) order, so the GPU tests compare with a
// tolerance, while tests/kernels/test_inkling2_ref_cpu.cpp pins THIS file against goldens produced by the HF
// modules themselves (tools/inkling2_kern/gen_ref_goldens.py).
//
// Conventions: row-major arrays, `float` in and out (a bf16 tensor is passed as its float values), no
// allocation in the hot functions except small scratch vectors.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace r4dx::inkling2_ref {

inline constexpr float kRmsEps = 1e-6f;           // config.rms_norm_eps and Inkling2LoopUnweightedRMSNorm's default
inline constexpr float kClipEps = 1e-6f;          // Inkling2LoopL2Clip: threshold / (norm + 1e-6)
inline constexpr float kRouterEps = 1e-20f;       // Inkling2LoopTopKRouter: weights / (sum + 1e-20)
inline constexpr float kQkClip = 22.627416997969522f;
inline constexpr float kRelClip = 8.0f;
inline constexpr float kLogAlpha = 0.1f;
inline constexpr float kLogFloor = 8192.0f;
inline constexpr float kGateSoftcap = 4.0f;
inline constexpr float kUpSoftcap = 25.0f;
inline constexpr float kLogitSoftcap = 30.0f;
inline constexpr int kConvTaps = 4;  // config.conv_kernel_size: 3 previous inputs of state

// ---- norms -----------------------------------------------------------------------------------------------------
// Inkling2LoopUnweightedRMSNorm: out = x * (1 / sqrt(mean(x^2) + eps)); the mean is sum (index order) / n.
inline float RmsInv(const float* x, int n, float eps = kRmsEps) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  return 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
}
inline void RmsUnweighted(const float* x, int n, float* out, float eps = kRmsEps) {
  const float inv = RmsInv(x, n, eps);
  for (int i = 0; i < n; ++i) out[i] = x[i] * inv;
}
// Inkling2LoopRMSNorm (plain weight, not 1 + w): out = w * (x * inv).
inline void RmsWeighted(const float* x, const float* w, int n, float* out, float eps = kRmsEps) {
  const float inv = RmsInv(x, n, eps);
  for (int i = 0; i < n; ++i) out[i] = w[i] * (x[i] * inv);
}

// ---- attention residual (Inkling2LoopBlockAttentionResidual, Inkling2LoopMessageAttentionResidual) --------------
// states[i]: [T, H] (the stack [B..., h], at most 6 in the model); queries: [nq, H]; out: [nq, T, H].
//   r_i = rms(S_i[t]); logit_qi = (q . r_i) * (1 / sqrt(H)); p = softmax_i(logit); out[q][t] = sum_i p_i r_i.
// The mix is of the NORMALIZED states.
inline void AttnRes(const float* const* states, int n_states, const float* queries, int nq, int T, int H, float* out,
                    float eps = kRmsEps) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(H));
  std::vector<float> r(static_cast<size_t>(n_states) * H), p(n_states);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < n_states; ++i) RmsUnweighted(states[i] + static_cast<size_t>(t) * H, H, &r[i * static_cast<size_t>(H)], eps);
    for (int q = 0; q < nq; ++q) {
      const float* qv = queries + static_cast<size_t>(q) * H;
      float m = -INFINITY;
      for (int i = 0; i < n_states; ++i) {
        float d = 0.0f;
        for (int h = 0; h < H; ++h) d += qv[h] * r[i * static_cast<size_t>(H) + h];
        p[i] = d * scale;
        m = std::max(m, p[i]);
      }
      float l = 0.0f;
      for (int i = 0; i < n_states; ++i) {
        p[i] = std::exp(p[i] - m);
        l += p[i];
      }
      float* o = out + (static_cast<size_t>(q) * T + t) * H;
      for (int h = 0; h < H; ++h) {
        float a = 0.0f;
        for (int i = 0; i < n_states; ++i) a += (p[i] / l) * r[i * static_cast<size_t>(H) + h];
        o[h] = a;
      }
    }
  }
}

// ---- short convolutions ------------------------------------------------------------------------------------------
// Inkling2LoopShortConvolution (k/v in the sliding layers, post-attention and post-MLP everywhere): depthwise causal
// conv, 4 taps, weight [C][4] with tap 3 = the current token (PyTorch cross-correlation with left padding 3):
//   conv_t = w[c][0] * x_{t-3} + w[c][1] * x_{t-2} + w[c][2] * x_{t-1} + w[c][3] * x_t      (this order)
//   out_t  = (conv_t + x_t) / sqrt(5)       (+ residual_t when residual != nullptr: out_t = residual_t + that)
// state [3][C]: x_{t-3}, x_{t-2}, x_{t-1} before the chunk (zeros for a fresh sequence), updated in place to the
// last three inputs after it. Chunked prefill == decode: the math is a function of the input sequence only.
inline void SconvStatic(const float* x, int T, int C, const float* w, float* state, const float* residual,
                        float* out) {
  const float inv = std::sqrt(5.0f);
  for (int c = 0; c < C; ++c) {
    float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    const float* wc = w + static_cast<size_t>(c) * 4;
    for (int t = 0; t < T; ++t) {
      const float xt = x[static_cast<size_t>(t) * C + c];
      const float conv = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * xt;
      float y = (conv + xt) / inv;
      if (residual != nullptr) y = residual[static_cast<size_t>(t) * C + c] + y;
      out[static_cast<size_t>(t) * C + c] = y;
      h0 = h1;
      h1 = h2;
      h2 = xt;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
  }
}

// Inkling2LoopDynamicShortConvolution (k/v in the full layers): per-token taps shared within each of `heads` kv heads,
//   tap[t][c][lag] = static_w[lag][c] + dyn[t][head(c) * 4 + lag]       head(c) = c / (C / heads), lag 0 = current
//   sum_t = x_t * tap[0] + x_{t-1} * tap[1] + x_{t-2} * tap[2] + x_{t-3} * tap[3]           (this order)
//   out_t = (sum_t + x_t) / sqrt(9)
// static_w [4][C] (the checkpoint's [kernel, C] layout); dyn [T][heads * 4] = z @ weight_proj^T (a GEMM).
// state [3][C] as in SconvStatic (oldest first).
inline void SconvDynamic(const float* x, int T, int C, int heads, const float* static_w, const float* dyn,
                         float* state, float* out) {
  const int per_head = C / heads;
  for (int c = 0; c < C; ++c) {
    float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];  // x_{t-3}, x_{t-2}, x_{t-1}
    const int hd = c / per_head;
    for (int t = 0; t < T; ++t) {
      const float* d = dyn + static_cast<size_t>(t) * heads * 4 + hd * 4;
      const float xt = x[static_cast<size_t>(t) * C + c];
      const float t0 = static_w[c] + d[0], t1 = static_w[C + c] + d[1];
      const float t2 = static_w[2 * C + c] + d[2], t3 = static_w[3 * C + c] + d[3];
      const float sum = xt * t0 + h2 * t1 + h1 * t2 + h0 * t3;
      out[static_cast<size_t>(t) * C + c] = (sum + xt) / 3.0f;
      h0 = h1;
      h1 = h2;
      h2 = xt;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
  }
}

// ---- per-head ops --------------------------------------------------------------------------------------------------
// Inkling2LoopLogScale: tau(pos) = 1 + alpha * ln(max((pos + 1) / floor, 1)).
inline float LogScale(int64_t pos, float alpha = kLogAlpha, float floor = kLogFloor) {
  const float r = static_cast<float>(pos + 1) / floor;
  return 1.0f + alpha * std::log(std::max(r, 1.0f));
}
// Inkling2LoopScaledL2Clip (scale != nullptr) / Inkling2LoopL2Clip (scale == nullptr), then Inkling2LoopLogScale when
// log_alpha != 0 (Inkling2LoopLogScaledL2Clip: q and the relative states of the full layers) at position pos0 + t:
//   v = x * scale[h];  f = min(1, threshold / (||v|| + 1e-6));  out = v * f  (* tau(pos0 + t))
// x, out: [T][heads][d]; scale [heads][d].
inline void HeadClip(const float* x, int T, int heads, int d, const float* scale, float threshold, int64_t pos0,
                     float log_alpha, float log_floor, float* out) {
  std::vector<float> v(d);
  for (int t = 0; t < T; ++t) {
    const float tau = log_alpha != 0.0f ? LogScale(pos0 + t, log_alpha, log_floor) : 1.0f;
    for (int h = 0; h < heads; ++h) {
      const size_t o = (static_cast<size_t>(t) * heads + h) * d;
      float ss = 0.0f;
      for (int i = 0; i < d; ++i) {
        v[i] = scale != nullptr ? x[o + i] * scale[static_cast<size_t>(h) * d + i] : x[o + i];
        ss += v[i] * v[i];
      }
      const float f = std::min(1.0f, threshold / (std::sqrt(ss) + kClipEps));
      for (int i = 0; i < d; ++i) {
        float y = v[i] * f;
        if (log_alpha != 0.0f) y = y * tau;
        out[o + i] = y;
      }
    }
  }
}
// The value norm (Inkling2LoopKeyValueProjection, is_key false): unweighted RMS per head vector of d.
inline void HeadRms(const float* x, int T, int heads, int d, float* out, float eps = kRmsEps) {
  for (int64_t r = 0; r < static_cast<int64_t>(T) * heads; ++r) RmsUnweighted(x + r * d, d, out + r * d, eps);
}

// Inkling2LoopRelativePositionBias, the bank mix: rel[t][h][e] = sum_j r[t][h][j] * proj[j][e] (j ascending).
// r [T][heads][d_rel] (after HeadClip), proj [d_rel][extent], rel [T][heads][extent]. The attention adds
// rel[t][h][t - j] for 0 <= t - j < extent and 0 beyond (r4dx_inkling2_attn_ref_bf16kv / Attention below).
inline void RelLogits(const float* r, int T, int heads, int d_rel, const float* proj, int extent, float* rel) {
  for (int64_t row = 0; row < static_cast<int64_t>(T) * heads; ++row) {
    for (int e = 0; e < extent; ++e) {
      float a = 0.0f;
      for (int j = 0; j < d_rel; ++j) a += r[row * d_rel + j] * proj[static_cast<size_t>(j) * extent + e];
      rel[row * extent + e] = a;
    }
  }
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Inkling2LoopGatedOutputProjection's gate: out = o * sigmoid(g) (o the attention output, g = x @ Wgate^T).
inline void SigmoidGateMul(const float* o, const float* g, int64_t n, float* out) {
  for (int64_t i = 0; i < n; ++i) out[i] = o[i] * Sigmoid(g[i]);
}

// ---- message gate (Inkling2LoopMessageGate) -----------------------------------------------------------------------
// prep: out [T][2H] = [rms(x) | rms(m)] (the input of down_proj, a GEMM to g [T][R]).
inline void MsgGatePrep(const float* x, const float* m, int T, int H, float* out, float eps = kRmsEps) {
  for (int t = 0; t < T; ++t) {
    RmsUnweighted(x + static_cast<size_t>(t) * H, H, out + static_cast<size_t>(t) * 2 * H, eps);
    RmsUnweighted(m + static_cast<size_t>(t) * H, H, out + static_cast<size_t>(t) * 2 * H + H, eps);
  }
}
// apply: f = tanh(g @ forget^T), a = tanh(g @ gain^T) (sums over r ascending); out = (1 - f) * x + a * m.
// g [T][R], forget / gain [H][R].
inline void MsgGateApply(const float* x, const float* m, const float* g, const float* forget, const float* gain, int T,
                         int H, int R, float* out) {
  for (int t = 0; t < T; ++t) {
    const float* gt = g + static_cast<size_t>(t) * R;
    for (int h = 0; h < H; ++h) {
      float f = 0.0f, a = 0.0f;
      for (int r = 0; r < R; ++r) f += gt[r] * forget[static_cast<size_t>(h) * R + r];
      for (int r = 0; r < R; ++r) a += gt[r] * gain[static_cast<size_t>(h) * R + r];
      f = std::tanh(f);
      a = std::tanh(a);
      const size_t i = static_cast<size_t>(t) * H + h;
      out[i] = (1.0f - f) * x[i] + a * m[i];
    }
  }
}
// The composite, all fp32: g = down @ [rms x; rms m] (down [R][2H]).
inline void MsgGate(const float* x, const float* m, const float* down, const float* forget, const float* gain, int T,
                    int H, int R, float* out, float eps = kRmsEps) {
  std::vector<float> cat(static_cast<size_t>(T) * 2 * H), g(static_cast<size_t>(T) * R);
  MsgGatePrep(x, m, T, H, cat.data(), eps);
  for (int t = 0; t < T; ++t) {
    for (int r = 0; r < R; ++r) {
      float a = 0.0f;
      for (int k = 0; k < 2 * H; ++k) a += cat[static_cast<size_t>(t) * 2 * H + k] * down[static_cast<size_t>(r) * 2 * H + k];
      g[static_cast<size_t>(t) * R + r] = a;
    }
  }
  MsgGateApply(x, m, g.data(), forget, gain, T, H, R, out);
}

// ---- router (Inkling2LoopTopKRouter, scoring sigmoid) ---------------------------------------------------------------
// logits[e] = x . w[e] (k ascending); s = sigmoid(logits); sel = top-k of s + bias, DESCENDING, ties -> the lower
// expert index; weights[slot] = s[sel] / (sum of the k selected s, slot order, + 1e-20) * route_scale.
// x [T][H], w [E][H], bias [E]; ids / weights [T][k]; logits_out [T][E] optional.
inline void Router(const float* x, const float* w, const float* bias, int T, int H, int E, int k, float route_scale,
                   int32_t* ids, float* weights, float* logits_out = nullptr) {
  std::vector<float> s(E), score(E);
  std::vector<int> order(E);
  for (int t = 0; t < T; ++t) {
    for (int e = 0; e < E; ++e) {
      float a = 0.0f;
      for (int h = 0; h < H; ++h) a += x[static_cast<size_t>(t) * H + h] * w[static_cast<size_t>(e) * H + h];
      if (logits_out != nullptr) logits_out[static_cast<size_t>(t) * E + e] = a;
      s[e] = Sigmoid(a);
      score[e] = s[e] + bias[e];
      order[e] = e;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) sum += s[order[j]];
    for (int j = 0; j < k; ++j) {
      ids[static_cast<size_t>(t) * k + j] = order[j];
      weights[static_cast<size_t>(t) * k + j] = s[order[j]] / (sum + kRouterEps) * route_scale;
    }
  }
}

// ---- activations ---------------------------------------------------------------------------------------------------
// Inkling2LoopSoftcap: tanh(x / c) * c.
inline float Softcap(float x, float c) { return std::tanh(x / c) * c; }
// Inkling2LoopSoftcappedSwiGLU: (softcap(g, 4) * sigmoid(g)) * softcap(u, 25).
inline float SwigluSoftcap(float g, float u) { return Softcap(g, kGateSoftcap) * Sigmoid(g) * Softcap(u, kUpSoftcap); }

// ---- attention (eager_attention_forward with position_bias and sinks) ---------------------------------------------
// q [q_len][q_heads][d]; k, v [ctx][kv_heads][d] (key j at row j); the queries are the last q_len positions
// (qpos = ctx - q_len + t). Visible keys: klow..qpos, klow = window > 0 ? max(0, qpos - window + 1) : 0.
//   s_j = (q . k_j) * scale + (0 <= qpos - j < extent ? rel[t][h][qpos - j] : 0)    (rel may be null: no bias)
//   softmax over {s_j} and, when sink != null, one extra logit sink[t][h] whose value row is zero
//   out = sum_j p_j v_j.
// rel [q_len][q_heads][extent], sink [q_len][q_heads]; GQA: kv head = h / (q_heads / kv_heads).
inline void Attention(const float* q, const float* k, const float* v, const float* rel, const float* sink, int q_len,
                      int q_heads, int kv_heads, int d, int ctx, int window, int extent, float scale, float* out) {
  const int group = q_heads / kv_heads;
  std::vector<float> s(ctx);
  for (int t = 0; t < q_len; ++t) {
    const int qpos = ctx - q_len + t;
    const int klow = window > 0 ? std::max(0, qpos - window + 1) : 0;
    for (int h = 0; h < q_heads; ++h) {
      const int kvh = h / group;
      const float* qv = q + (static_cast<size_t>(t) * q_heads + h) * d;
      float m = sink != nullptr ? sink[static_cast<size_t>(t) * q_heads + h] : -INFINITY;
      for (int j = klow; j <= qpos; ++j) {
        const float* kv = k + (static_cast<size_t>(j) * kv_heads + kvh) * d;
        float a = 0.0f;
        for (int i = 0; i < d; ++i) a += qv[i] * kv[i];
        a = a * scale;
        const int dist = qpos - j;
        if (rel != nullptr && dist < extent) a = a + rel[(static_cast<size_t>(t) * q_heads + h) * extent + dist];
        s[j] = a;
        m = std::max(m, a);
      }
      float l = sink != nullptr ? std::exp(sink[static_cast<size_t>(t) * q_heads + h] - m) : 0.0f;
      for (int j = klow; j <= qpos; ++j) {
        s[j] = std::exp(s[j] - m);
        l += s[j];
      }
      float* o = out + (static_cast<size_t>(t) * q_heads + h) * d;
      for (int i = 0; i < d; ++i) {
        float a = 0.0f;
        for (int j = klow; j <= qpos; ++j) a += s[j] * v[(static_cast<size_t>(j) * kv_heads + kvh) * d + i];
        o[i] = a / l;
      }
    }
  }
}

}  // namespace r4dx::inkling2_ref
