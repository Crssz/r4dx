// tests/model/inkling2_cpu_graph.h -- the Inkling2Model graph written out on the CPU (header-only, no HIP), shared by
// test_inkling2_cpu_graph (CPU-only) and test_inkling2_tiny_e2e (the GPU engine's twin). docs/inkling2-plan.md 3.3 / 5.
//
// Engine::Chunk is Inkling2Model::RunChunk's state machine line for line -- the visit table, the fp32 block-state pool
// (h, B, h_pre, the previous pass's block / output, the restart of every pass >= 2), the messages, the per-slot conv
// states and K/V -- over the kernels' CPU spec (r4dx/kernels/inkling2_ref.hpp). Two numerics:
//   * default: the GPU engine's rounding points (bf16 GEMM inputs and outputs, bf16 K/V, bf16 attention / MoE / gate
//     outputs) on the container's weights (bf16, or mxfp8 decoded exactly);
//   * g_exact = true: fp32 throughout and the experts decoded from the ORIGINAL checkpoint in f32 (Weights::ckpt) -- HF's
//     fp32 forward, which test_inkling2_cpu_graph pins to float noise (the semantics check).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "inkling2_config.h"
#include "r4dx/kernels/inkling2_ref.hpp"
#include "r4dx_convert/inkling2_expert_codec.hpp"
#include "r4dx_convert/mxfp8_format.hpp"
#include "r4dx_convert/quant_mxfp8.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace inkling2_cpu {

using r4dx::model::Inkling2Config;
using r4dx::model::Inkling2Visit;
namespace ref = r4dx::inkling2_ref;
using Vec = std::vector<float>;
// bf16 round-to-nearest-even (finite values; the test never makes a NaN) -- host-only, no HIP header.
inline float Bf16Bits(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline bool g_exact = false;  // exact mode: no bf16 rounding anywhere, experts decoded from the checkpoint in f32 (HF fp32)
inline float Bf(float x) {
  if (g_exact) return x;
  uint32_t u;
  std::memcpy(&u, &x, 4);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline Vec RoundBf(Vec v) {
  for (auto& x : v) x = Bf(x);
  return v;
}

struct Weights {
  const r4dx_convert::SafetensorsReader* r = nullptr;
  std::map<std::string, Vec> cache;
  const Vec& Bf16(const std::string& name) {
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    if (!r->Has(name)) throw std::runtime_error("missing tensor " + name);
    const auto& m = r->Meta(name);
    const size_t n = static_cast<size_t>(m.end - m.begin) / 2;
    Vec v(n);
    const uint8_t* d = r->Data(name);
    for (size_t i = 0; i < n; ++i) {
      uint16_t b;
      std::memcpy(&b, d + 2 * i, 2);
      v[i] = Bf16Bits(b);
    }
    return cache[name] = std::move(v);
  }
  const Vec& F32(const std::string& name) {
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    if (!r->Has(name)) throw std::runtime_error("missing tensor " + name);
    const auto& m = r->Meta(name);
    Vec v(static_cast<size_t>(m.end - m.begin) / 4);
    std::memcpy(v.data(), r->Data(name), v.size() * 4);
    return cache[name] = std::move(v);
  }
  // A linear [N, K] of the container: its .bf16.w, or its .mxfp8.wq / .ws decoded exactly (r4dx_convert::DequantMxfp8).
  const Vec& Lin(const std::string& base, int64_t K) {
    if (r->Has(base + ".bf16.w")) return Bf16(base + ".bf16.w");
    const std::string key = base + ".mxfp8";
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const std::string wq = base + r4dx_convert::mxfp8::kWqSuffix, ws = base + r4dx_convert::mxfp8::kWsSuffix;
    if (!r->Has(wq) || !r->Has(ws)) throw std::runtime_error("missing linear " + base);
    const auto& m = r->Meta(wq);
    const int64_t N = static_cast<int64_t>(m.end - m.begin) / K;
    return cache[key] = r4dx_convert::DequantMxfp8(r->Data(wq), r->Data(ws), static_cast<int>(N), static_cast<int>(K));
  }
  // exact mode: expert e of `layer` (which: "gate_up" | "down") decoded in f32 from the original checkpoint
  const r4dx_convert::SafetensorsReader* ckpt = nullptr;
  const Vec& Expert(int64_t layer, int64_t e, const std::string& which, int64_t E, int64_t rows, int64_t cols) {
    const std::string key = "ckpt." + std::to_string(layer) + "." + std::to_string(e) + "." + which;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    namespace ik = r4dx_convert::inkling2;
    const std::string p = "model.llm.layers." + std::to_string(layer) + ".mlp.experts." + (which == "gate_up" ? "w13" : "w2") + "_weight";
    ik::ExpertTensors t;
    t.format = ckpt->Has(p + "_amax") ? ik::ExpertFormat::kNvfp4 : ik::ExpertFormat::kMxfp8;
    t.experts = E;
    t.rows = rows;
    t.cols = cols;
    t.weight = ckpt->Data(p);
    t.scale = ckpt->Data(p + "_scale");
    if (t.format == ik::ExpertFormat::kNvfp4) {
      t.amax = ckpt->Data(p + "_amax");
      t.denominator = ik::ReadF32(ckpt->Data(p + "_outer_scale_denominator"));
    }
    Vec raw(static_cast<size_t>(rows * cols)), out(raw.size());
    ik::DecodeExpert(t, e, raw.data());
    if (which == "gate_up") {
      ik::DeinterleaveGateUp(raw.data(), out.data(), rows, cols);
    } else {
      out = raw;
    }
    return cache[key] = std::move(out);
  }
};

// y[T, N] = bf16(bf16(x)[T, K] @ W[N, K]^T)   (the GEMM: bf16 operands, fp32 sums, one bf16 rounding)
inline Vec Gemm(const Vec& x, int64_t T, int64_t K, const Vec& w, int64_t N, bool round_out = true) {
  Vec y(static_cast<size_t>(T * N));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t n = 0; n < N; ++n) {
      float a = 0;
      for (int64_t k = 0; k < K; ++k) a += Bf(x[static_cast<size_t>(t * K + k)]) * w[static_cast<size_t>(n * K + k)];
      y[static_cast<size_t>(t * N + n)] = round_out ? Bf(a) : a;
    }
  return y;
}
inline Vec RmsW(const Vec& x, int64_t T, int64_t H, const Vec& w) {  // -> bf16 values
  Vec o(x.size());
  for (int64_t t = 0; t < T; ++t) ref::RmsWeighted(x.data() + t * H, w.data(), static_cast<int>(H), o.data() + t * H);
  return RoundBf(o);
}

struct Engine {
  Inkling2Config cfg;
  Weights w;
  std::vector<Inkling2Visit> visits;
  std::vector<Vec> conv;               // [slot * 4 + which] [3, C]
  std::vector<Vec> kcache, vcache;     // [slot] [pos][kv][hd] (bf16 values), grown
  int64_t pos = 0;

  void Reset() {
    const int64_t S = cfg.NumCacheSlots(), C = cfg.KvDim(), H = cfg.hidden_size;
    conv.assign(static_cast<size_t>(S * 4), Vec());
    for (int64_t s = 0; s < S; ++s)
      for (int k = 0; k < 4; ++k) conv[static_cast<size_t>(s * 4 + k)].assign(static_cast<size_t>(3 * (k < 2 ? C : H)), 0.0f);
    kcache.assign(static_cast<size_t>(S), Vec());
    vcache.assign(static_cast<size_t>(S), Vec());
    pos = 0;
  }

  Vec AttnRes(const std::vector<const Vec*>& st, const float* q, int nq, int64_t T) {
    std::vector<const float*> p;
    for (const Vec* s : st) p.push_back(s->data());
    Vec o(static_cast<size_t>(nq * T * cfg.hidden_size));
    ref::AttnRes(p.data(), static_cast<int>(p.size()), q, nq, static_cast<int>(T), static_cast<int>(cfg.hidden_size), o.data());
    return o;
  }

  Vec MsgGate(const std::string& b, const float* x, const float* m, int64_t T) {
    const int64_t H = cfg.hidden_size, R = cfg.gate_rank;
    Vec cat(static_cast<size_t>(T * 2 * H));
    ref::MsgGatePrep(x, m, static_cast<int>(T), static_cast<int>(H), cat.data());
    const Vec g = Gemm(cat, T, 2 * H, w.Bf16(b + "down"), R, /*round_out=*/false);
    Vec o(static_cast<size_t>(T * H));
    ref::MsgGateApply(x, m, g.data(), w.Bf16(b + "forget").data(), w.Bf16(b + "gain").data(), static_cast<int>(T),
                      static_cast<int>(H), static_cast<int>(R), o.data());
    return o;
  }

  Vec Attention(const Inkling2Visit& v, const Vec& xq, const Vec& xk, const Vec& xv, int64_t T) {
    const std::string b = "text.layers." + std::to_string(v.layer) + ".";
    const int64_t H = cfg.hidden_size, hd = cfg.head_dim, hq = cfg.num_attention_heads, kv = cfg.num_key_value_heads;
    const int64_t C = kv * hd, dr = cfg.d_rel, ext = cfg.RelExtent(v.layer);
    const bool full = cfg.IsFull(v.layer);
    const float la = full ? static_cast<float>(cfg.log_alpha) : 0.0f, lf = static_cast<float>(cfg.log_floor);
    Vec q(static_cast<size_t>(T * hq * hd));
    ref::HeadClip(Gemm(xq, T, H, w.Lin(b + "attn.q", H), hq * hd).data(), static_cast<int>(T), static_cast<int>(hq),
                  static_cast<int>(hd), w.Bf16(b + "attn.q_scale").data(), static_cast<float>(cfg.qk_clip), pos, la, lf, q.data());
    q = RoundBf(q);
    const Vec kl = Gemm(xk, T, H, w.Lin(b + "attn.k", H), C), vl = Gemm(xv, T, H, w.Lin(b + "attn.v", H), C);
    Vec kc(static_cast<size_t>(T * C)), vc(static_cast<size_t>(T * C));
    if (full) {
      const Vec dk = Gemm(xk, T, H, w.Bf16(b + "attn.k_conv_proj"), kv * 4, false);
      const Vec dv = Gemm(xv, T, H, w.Bf16(b + "attn.v_conv_proj"), kv * 4, false);
      ref::SconvDynamic(kl.data(), static_cast<int>(T), static_cast<int>(C), static_cast<int>(kv),
                        w.Bf16(b + "attn.k_conv_static").data(), dk.data(), conv[static_cast<size_t>(v.cache_slot * 4 + 0)].data(), kc.data());
      ref::SconvDynamic(vl.data(), static_cast<int>(T), static_cast<int>(C), static_cast<int>(kv),
                        w.Bf16(b + "attn.v_conv_static").data(), dv.data(), conv[static_cast<size_t>(v.cache_slot * 4 + 1)].data(), vc.data());
    } else {
      ref::SconvStatic(kl.data(), static_cast<int>(T), static_cast<int>(C), w.Bf16(b + "attn.k_conv").data(),
                       conv[static_cast<size_t>(v.cache_slot * 4 + 0)].data(), nullptr, kc.data());
      ref::SconvStatic(vl.data(), static_cast<int>(T), static_cast<int>(C), w.Bf16(b + "attn.v_conv").data(),
                       conv[static_cast<size_t>(v.cache_slot * 4 + 1)].data(), nullptr, vc.data());
    }
    Vec kk(kc.size()), vv(vc.size());
    ref::HeadClip(kc.data(), static_cast<int>(T), static_cast<int>(kv), static_cast<int>(hd), nullptr,
                  static_cast<float>(cfg.qk_clip), pos, 0.0f, 1.0f, kk.data());
    ref::HeadRms(vc.data(), static_cast<int>(T), static_cast<int>(kv), static_cast<int>(hd), vv.data());
    Vec& K = kcache[static_cast<size_t>(v.cache_slot)];
    Vec& V = vcache[static_cast<size_t>(v.cache_slot)];
    for (float x : RoundBf(kk)) K.push_back(x);
    for (float x : RoundBf(vv)) V.push_back(x);
    Vec r(static_cast<size_t>(T * hq * dr));
    ref::HeadClip(Gemm(xq, T, H, w.Lin(b + "attn.rel", H), hq * dr).data(), static_cast<int>(T), static_cast<int>(hq),
                  static_cast<int>(dr), w.Bf16(b + "attn.rel_scale").data(), static_cast<float>(cfg.rel_clip), pos, la, lf, r.data());
    Vec rel(static_cast<size_t>(T * hq * ext));
    ref::RelLogits(r.data(), static_cast<int>(T), static_cast<int>(hq), static_cast<int>(dr), w.Bf16(b + "attn.rel_proj").data(),
                   static_cast<int>(ext), rel.data());
    const Vec sink = Gemm(xq, T, H, w.Bf16(b + "attn.sink"), hq, false);
    Vec ao(static_cast<size_t>(T * hq * hd));
    ref::Attention(q.data(), K.data(), V.data(), rel.data(), sink.data(), static_cast<int>(T), static_cast<int>(hq),
                   static_cast<int>(kv), static_cast<int>(hd), static_cast<int>(pos + T), static_cast<int>(cfg.Window(v.layer)),
                   static_cast<int>(ext), 1.0f / std::sqrt(static_cast<float>(hd)), ao.data());
    ao = RoundBf(ao);
    const Vec g = Gemm(xq, T, H, w.Lin(b + "attn.gate", H), hq * hd);
    Vec gated(ao.size());
    ref::SigmoidGateMul(ao.data(), g.data(), static_cast<int64_t>(ao.size()), gated.data());
    return Gemm(RoundBf(gated), T, hq * hd, w.Lin(b + "attn.o", hq * hd), H);
  }

  Vec Moe(int64_t layer, const Vec& x, int64_t T) {
    const std::string b = "text.layers." + std::to_string(layer) + ".";
    const int64_t H = cfg.hidden_size, E = cfg.num_experts, K = cfg.num_experts_per_tok, I = cfg.moe_intermediate_size;
    std::vector<int32_t> ids(static_cast<size_t>(T * K));
    Vec wts(static_cast<size_t>(T * K));
    ref::Router(x.data(), w.Bf16(b + "moe.router").data(), w.F32(b + "moe.bias").data(), static_cast<int>(T),
                static_cast<int>(H), static_cast<int>(E), static_cast<int>(K), static_cast<float>(cfg.route_scale), ids.data(),
                wts.data());
    Vec out(static_cast<size_t>(T * H));
    for (int64_t t = 0; t < T; ++t) {
      Vec acc(static_cast<size_t>(H), 0.0f);
      const Vec xt(x.begin() + t * H, x.begin() + (t + 1) * H);
      for (int64_t j = 0; j < K; ++j) {
        const std::string eb = b + "moe.experts." + std::to_string(ids[static_cast<size_t>(t * K + j)]) + ".";
        const int64_t eid = ids[static_cast<size_t>(t * K + j)];
        const Vec gu = Gemm(xt, 1, H, g_exact ? w.Expert(layer, eid, "gate_up", E, 2 * I, H) : w.Lin(eb + "gate_up", H), 2 * I);
        Vec act(static_cast<size_t>(I));
        for (int64_t i = 0; i < I; ++i) act[static_cast<size_t>(i)] = Bf(ref::SwigluSoftcap(gu[static_cast<size_t>(i)], gu[static_cast<size_t>(I + i)]));
        const Vec y = Gemm(act, 1, I, g_exact ? w.Expert(layer, eid, "down", E, H, I) : w.Lin(eb + "down", I), H);
        for (int64_t h = 0; h < H; ++h) acc[static_cast<size_t>(h)] += wts[static_cast<size_t>(t * K + j)] * y[static_cast<size_t>(h)];
      }
      for (int64_t h = 0; h < H; ++h) out[static_cast<size_t>(t * H + h)] = Bf(acc[static_cast<size_t>(h)]);
    }
    return out;
  }

  // Inkling2Model::RunChunk's state machine; the pool indices are plain vectors here.
  Vec* dump = nullptr;  // debugging: every visit's output h appended
  Vec Chunk(const std::vector<int32_t>& ids) {
    const int64_t T = static_cast<int64_t>(ids.size()), H = cfg.hidden_size;
    std::vector<Vec> pool;
    const auto acquire = [&]() {
      pool.emplace_back(static_cast<size_t>(T * H), 0.0f);
      return static_cast<int>(pool.size() - 1);
    };
    int h = acquire(), h_pre = -1, prev_h = -1;
    std::vector<int> B, B_pre, prev_B;
    size_t n_pre = 0;
    {
      const Vec& emb = w.Bf16("text.embed_tokens");
      const Vec& en = w.Bf16("text.embed_norm");
      for (int64_t t = 0; t < T; ++t)
        ref::RmsWeighted(emb.data() + static_cast<int64_t>(ids[static_cast<size_t>(t)]) * H, en.data(), static_cast<int>(H),
                         pool[static_cast<size_t>(h)].data() + t * H);
    }
    const Vec& msgq = w.F32("text.msg_queries");
    for (const Inkling2Visit& v : visits) {
      if (std::getenv("R4DX_INKLING2_CPU_TRACE") != nullptr) std::printf("  visit %d layer %d pass %d\n", v.index, v.layer, v.pass);
      if (v.phase == Inkling2Visit::Phase::kLoop && v.pass >= 1 && v.loop_j == 0) {
        prev_B.assign(B.begin() + static_cast<std::ptrdiff_t>(n_pre + 1), B.end());
        prev_h = h;
        B = B_pre;
        h = h_pre;
      }
      const std::string b = "text.layers." + std::to_string(v.layer) + ".";
      const bool full = cfg.IsFull(v.layer), msgs = v.HasMessages();
      const auto stack = [&]() {
        std::vector<const Vec*> s;
        for (int i : B) s.push_back(&pool[static_cast<size_t>(i)]);
        s.push_back(&pool[static_cast<size_t>(h)]);
        return s;
      };
      Vec m;
      if (msgs) {
        std::vector<const Vec*> s;
        for (int i : prev_B) s.push_back(&pool[static_cast<size_t>(i)]);
        s.push_back(&pool[static_cast<size_t>(prev_h)]);
        m = AttnRes(s, msgq.data() + v.loop_j * 4 * H, 4, T);
      }
      const int nq = v.layer == 0 ? 0 : (full ? 3 : 1);
      Vec a = nq > 0 ? AttnRes(stack(), w.F32(b + "attn_query").data(), nq, T) : pool[static_cast<size_t>(h)];
      Vec x[3];
      for (int k = 0; k < 3; ++k) {
        const std::string norm = full ? (k == 0 ? "q_norm" : k == 1 ? "k_norm" : "v_norm") : "attn_norm";
        Vec src(a.begin() + (full && nq > 0 ? k * T * H : 0), a.begin() + (full && nq > 0 ? k * T * H : 0) + T * H);
        if (msgs) src = MsgGate(b + "msg." + (k == 0 ? "q" : k == 1 ? "k" : "v") + ".", src.data(), m.data() + k * T * H, T);
        x[k] = RmsW(src, T, H, w.Bf16(b + norm));
      }
      const Vec o = Attention(v, x[0], x[1], x[2], T);
      if (cfg.IsBoundary(v.layer)) {
        B.push_back(h);
        const int hn = acquire();
        ref::SconvStatic(o.data(), static_cast<int>(T), static_cast<int>(H), w.Bf16(b + "attn_conv").data(),
                         conv[static_cast<size_t>(v.cache_slot * 4 + 2)].data(), nullptr, pool[static_cast<size_t>(hn)].data());
        h = hn;
      } else {
        Vec& hv = pool[static_cast<size_t>(h)];
        ref::SconvStatic(o.data(), static_cast<int>(T), static_cast<int>(H), w.Bf16(b + "attn_conv").data(),
                         conv[static_cast<size_t>(v.cache_slot * 4 + 2)].data(), hv.data(), hv.data());
      }
      Vec y = AttnRes(stack(), w.F32(b + "mlp_query").data(), 1, T);
      if (msgs) y = MsgGate(b + "msg.mlp.", y.data(), m.data() + 3 * T * H, T);
      const Vec f = Moe(v.layer, RmsW(y, T, H, w.Bf16(b + "mlp_norm")), T);
      Vec& hv = pool[static_cast<size_t>(h)];
      ref::SconvStatic(f.data(), static_cast<int>(T), static_cast<int>(H), w.Bf16(b + "mlp_conv").data(),
                       conv[static_cast<size_t>(v.cache_slot * 4 + 3)].data(), hv.data(), hv.data());
      if (dump != nullptr) dump->insert(dump->end(), pool[static_cast<size_t>(h)].begin(), pool[static_cast<size_t>(h)].end());
      if (v.phase == Inkling2Visit::Phase::kPre && v.layer == cfg.loop_start - 1) {
        h_pre = h;
        B_pre = B;
        n_pre = B.size();
      }
    }
    // the head, last row
    std::vector<Vec> last;
    std::vector<const Vec*> st;
    for (int i : B) last.emplace_back(pool[static_cast<size_t>(i)].begin() + (T - 1) * H, pool[static_cast<size_t>(i)].end());
    last.emplace_back(pool[static_cast<size_t>(h)].begin() + (T - 1) * H, pool[static_cast<size_t>(h)].end());
    for (const Vec& l : last) st.push_back(&l);
    const Vec fr = AttnRes(st, w.F32("text.final_query").data(), 1, 1);
    const Vec xn = RmsW(fr, 1, H, w.Bf16("text.final_norm"));
    const Vec lg = Gemm(xn, 1, H, w.Lin("lm_head", H), cfg.HeadRowsStored());
    Vec out(static_cast<size_t>(cfg.unpadded_vocab_size));
    for (int64_t i = 0; i < cfg.unpadded_vocab_size; ++i)
      out[static_cast<size_t>(i)] = ref::Softcap(lg[static_cast<size_t>(i)], static_cast<float>(cfg.final_logit_softcap));
    pos += T;
    return out;
  }
};

inline double Kl(const float* p, const float* q, int64_t V) {
  double mp = -1e300, mq = -1e300;
  for (int64_t i = 0; i < V; ++i) {
    mp = std::max(mp, static_cast<double>(p[i]));
    mq = std::max(mq, static_cast<double>(q[i]));
  }
  double sp = 0, sq = 0;
  for (int64_t i = 0; i < V; ++i) {
    sp += std::exp(p[i] - mp);
    sq += std::exp(q[i] - mq);
  }
  double kl = 0;
  for (int64_t i = 0; i < V; ++i) {
    const double lp = p[i] - mp - std::log(sp), lq = q[i] - mq - std::log(sq);
    kl += std::exp(lp) * (lp - lq);
  }
  return kl;
}

inline Vec ReadNpy(const std::filesystem::path& p, int64_t* rows, int64_t* cols) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.u8string());
  std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (all.size() < 10 || all.compare(1, 5, "NUMPY") != 0) throw std::runtime_error("bad npy " + p.string());
  const size_t hl = static_cast<unsigned char>(all[8]) | (static_cast<size_t>(static_cast<unsigned char>(all[9])) << 8);
  const std::string hdr = all.substr(10, hl);
  const size_t s = hdr.find('('), e = hdr.find(')');
  const std::string dims = hdr.substr(s + 1, e - s - 1);
  *rows = std::stoll(dims.substr(0, dims.find(',')));
  *cols = std::stoll(dims.substr(dims.find(',') + 1));
  Vec v(static_cast<size_t>(*rows * *cols));
  std::memcpy(v.data(), all.data() + 10 + hl, v.size() * 4);
  return v;
}

}  // namespace inkling2_cpu
