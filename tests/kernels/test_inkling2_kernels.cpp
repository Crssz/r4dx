// tests/kernels/test_inkling2_kernels.cpp -- every entry point of r4dx/kernels/inkling2_kernels.h against its CPU spec
// r4dx/kernels/inkling2_ref.hpp (docs/inkling2-plan.md 5). GPU test (HIP_VISIBLE_DEVICES=1, label inkling2_gpu); WRITTEN,
// NOT YET RUN by its author. The kernels sum in block-reduction order, so fp32 results are compared with a relative
// tolerance and bf16 results to within 2 bf16 ulps of the reference's value.
//
// Coverage beyond the plain shapes: the short convolutions (static and dynamic) fed one sequence of 600 tokens in chunks
// 256 + 256 + 88 and as single chunks of T = 1, 2, 3, 63, 64, 255, 256 (the state carried across calls must make chunked ==
// whole, and a chunk shorter than the 3-tap state must shift it); the paged head writes (K and V halves of attn_ref.h's
// layout, through a ring block table); the reference attention with the relative bias, the sinks and a sliding window
// (prefill chunks that cross the window, decode rows, a ring table); the router at the real 128 experts / top-8 with
// ties; the dense small GEMM at the shapes the model uses it for.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/inkling2_kernels.h"
#include "r4dx/kernels/inkling2_ref.hpp"
#include "r4dx_convert/quant_mxfp8.hpp"

using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::FloatToBf16;
namespace ref = r4dx::inkling2_ref;

namespace {

int g_fail = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

std::mt19937 g_rng(20261122);
std::vector<float> Rand(size_t n, float scale = 1.0f) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g_rng);
  return v;
}
std::vector<float> RoundBf16(std::vector<float> v) {
  for (auto& x : v) x = Bf16ToFloat(FloatToBf16(x));
  return v;
}
std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = FloatToBf16(v[i]);
  return o;
}
template <class T>
DeviceBuffer<T> Dev(const std::vector<T>& v) {
  DeviceBuffer<T> b(std::max<size_t>(v.size(), 1));
  if (!v.empty()) b.CopyFromHost(v.data(), v.size());
  return b;
}
template <class T>
std::vector<T> Host(const DeviceBuffer<T>& b, size_t n) {
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<T> v(n);
  b.CopyToHost(v.data(), n);
  return v;
}
std::vector<float> HostBf16(const DeviceBuffer<uint16_t>& b, size_t n) {
  const std::vector<uint16_t> h = Host(b, n);
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) f[i] = Bf16ToFloat(h[i]);
  return f;
}
inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

// max |got - want| / max(|want|, floor) over the arrays.
double RelErr(const std::vector<float>& got, const std::vector<float>& want, double floor = 1e-3) {
  if (got.size() != want.size()) return 1e30;
  double e = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) && !(std::isinf(got[i]) && got[i] == want[i])) return 1e30;
    e = std::max(e, std::fabs(static_cast<double>(got[i]) - want[i]) / std::max<double>(std::fabs(want[i]), floor));
  }
  return e;
}
// bf16 outputs: within `ulps` bf16 ulps of the fp32 reference value.
bool Bf16Close(const std::vector<float>& got, const std::vector<float>& want, int ulps = 2) {
  for (size_t i = 0; i < got.size(); ++i) {
    const float w = want[i];
    const float ulp = std::ldexp(1.0f, std::max(-133, static_cast<int>(std::floor(std::log2(std::max(std::fabs(w), 1e-30f)))) - 7));
    if (!(std::fabs(got[i] - w) <= ulps * ulp + 1e-6f)) {
      std::printf("    bf16 mismatch at %zu: got %.8g want %.8g\n", i, got[i], w);
      return false;
    }
  }
  return true;
}

// ---- attnres ----------------------------------------------------------------------------------------------------
void TestAttnRes() {
  const int H = 2304;
  for (int n_states : {1, 2, 5, 6}) {
    for (int nq : {1, 3, 4}) {
      const int T = 3;
      std::vector<std::vector<float>> st(static_cast<size_t>(n_states));
      std::vector<DeviceBuffer<float>> sd;
      std::vector<int64_t> ptrs;
      std::vector<const float*> hp;
      for (int i = 0; i < n_states; ++i) {
        st[static_cast<size_t>(i)] = Rand(static_cast<size_t>(T * H), 0.5f + i);
        sd.push_back(Dev(st[static_cast<size_t>(i)]));
      }
      for (int i = 0; i < n_states; ++i) {
        ptrs.push_back(P(sd[static_cast<size_t>(i)].data()));
        hp.push_back(st[static_cast<size_t>(i)].data());
      }
      const std::vector<float> q = Rand(static_cast<size_t>(nq * H), 0.05f);
      DeviceBuffer<float> qd = Dev(q), od(static_cast<size_t>(nq * T * H));
      r4dx_inkling2_attnres_f32(ptrs.data(), n_states, P(qd.data()), nq, P(od.data()), T, H, 1e-6f, 0);
      std::vector<float> want(static_cast<size_t>(nq * T * H));
      ref::AttnRes(hp.data(), n_states, q.data(), nq, T, H, want.data());
      // The outputs mix RMS-normalized states (scale ~1), so the error is measured in absolute terms (floor 1):
      // fp32 rounding over the 2304-wide dots and norms lands around 1e-6.
      const double e = RelErr(Host(od, want.size()), want, 1.0);
      Check(e < 1e-5, "attnres n_states " + std::to_string(n_states) + " nq " + std::to_string(nq) + " (rel " + std::to_string(e) + ")");

      // Verify pre-computed inv_rms path is bit-exact with on-the-fly path
      std::vector<DeviceBuffer<float>> inv_d;
      std::vector<int64_t> inv_ptrs;
      for (int i = 0; i < n_states; ++i) {
        inv_d.emplace_back(static_cast<size_t>(T));
        r4dx_inkling2_row_inv_rms_f32(ptrs[static_cast<size_t>(i)], P(inv_d.back().data()), T, H, 1e-6f, 0);
      }
      for (int i = 0; i < n_states; ++i) inv_ptrs.push_back(P(inv_d[static_cast<size_t>(i)].data()));
      DeviceBuffer<float> od_pre(static_cast<size_t>(nq * T * H));
      r4dx_inkling2_attnres_prenorm_f32(ptrs.data(), inv_ptrs.data(), n_states, P(qd.data()), nq, P(od_pre.data()), 0, 0,
                                        T, H, 1e-6f, 0);
      Check(Host(od_pre, want.size()) == Host(od, want.size()),
            "attnres_prenorm bit-exact n_states " + std::to_string(n_states) + " nq " + std::to_string(nq));

      if (nq == 1) {
        const std::vector<float> nw = RoundBf16(Rand(static_cast<size_t>(H), 1.0f));
        DeviceBuffer<uint16_t> nwd = Dev(ToBf16(nw)), obf(static_cast<size_t>(T * H));
        r4dx_inkling2_attnres_prenorm_f32(ptrs.data(), inv_ptrs.data(), n_states, P(qd.data()), 1, 0, P(nwd.data()),
                                          P(obf.data()), T, H, 1e-6f, 0);
        std::vector<float> want_bf(static_cast<size_t>(T * H));
        for (int t = 0; t < T; ++t) {
          ref::RmsWeighted(want.data() + static_cast<size_t>(t) * H, nw.data(), H,
                           want_bf.data() + static_cast<size_t>(t) * H);
        }
        Check(Bf16Close(HostBf16(obf, want_bf.size()), want_bf, 2),
              "attnres_prenorm + fused rmsnorm_bf16 n_states " + std::to_string(n_states));
      }
    }
  }
}

// ---- short convolutions over chunk boundaries -------------------------------------------------------------------
void TestSconv(bool dynamic, bool x_bf16, bool residual) {
  const int C = 1024, heads = 8, Ttot = 600;
  std::vector<float> x = Rand(static_cast<size_t>(Ttot * C));
  if (x_bf16) x = RoundBf16(x);
  const std::vector<float> w = Rand(static_cast<size_t>(C * 4), 0.4f);       // static: [C][4]; dynamic static_w: [4][C]
  std::vector<float> dyn = RoundBf16(Rand(static_cast<size_t>(Ttot * heads * 4), 0.3f));
  const std::vector<float> res = Rand(static_cast<size_t>(Ttot * C));
  // the whole-sequence reference
  std::vector<float> want(static_cast<size_t>(Ttot * C)), st(static_cast<size_t>(3 * C), 0.0f);
  if (dynamic) {
    ref::SconvDynamic(x.data(), Ttot, C, heads, w.data(), dyn.data(), st.data(), want.data());
  } else {
    ref::SconvStatic(x.data(), Ttot, C, w.data(), st.data(), residual ? res.data() : nullptr, want.data());
  }
  const auto run = [&](const std::vector<int>& chunks, const std::string& tag) {
    DeviceBuffer<float> wd = Dev(w), state(static_cast<size_t>(3 * C)), out(static_cast<size_t>(Ttot * C));
    state.Zero();
    DeviceBuffer<uint16_t> xb = Dev(ToBf16(x));
    DeviceBuffer<float> xf = Dev(x), rd = Dev(res), dd = Dev(dyn);
    int t0 = 0;
    for (int T : chunks) {
      const int64_t xp = x_bf16 ? P(xb.data() + static_cast<int64_t>(t0) * C) : P(xf.data() + static_cast<int64_t>(t0) * C);
      if (dynamic) {
        r4dx_inkling2_sconv_dynamic_f32(xp, x_bf16 ? 1 : 0, P(dd.data() + static_cast<int64_t>(t0) * heads * 4), 0, P(wd.data()),
                                        P(state.data()), P(out.data() + static_cast<int64_t>(t0) * C), T, C, heads, 0);
      } else {
        // residual: out aliases the residual (the model's in-place h += conv(o))
        float* o = residual ? rd.data() + static_cast<int64_t>(t0) * C : out.data() + static_cast<int64_t>(t0) * C;
        r4dx_inkling2_sconv_static_f32(xp, x_bf16 ? 1 : 0, P(wd.data()), P(state.data()), residual ? P(o) : 0, P(o), T, C, 0);
      }
      t0 += T;
    }
    const std::vector<float> got = Host((!dynamic && residual) ? rd : out, static_cast<size_t>(t0) * C);
    const std::vector<float> w0(want.begin(), want.begin() + static_cast<std::ptrdiff_t>(t0) * C);
    const double e = RelErr(got, w0, 1e-3);
    Check(e < 1e-5, std::string(dynamic ? "sconv_dynamic" : "sconv_static") + (x_bf16 ? " bf16-x" : " f32-x") +
                        (residual ? " +residual" : "") + " " + tag + " (rel " + std::to_string(e) + ")");
  };
  run({256, 256, 88}, "chunks 256+256+88");
  run({1, 2, 3, 63, 64, 255, 212}, "chunks 1,2,3,63,64,255,212");
  for (int T : {1, 2, 3, 63, 64, 255, 256}) run({T}, "single chunk T=" + std::to_string(T));
  {
    std::vector<int> ones(40, 1);
    run(ones, "40 decode steps");
  }
}

// ---- per-head clip / rms, dense and paged -----------------------------------------------------------------------
void TestHead() {
  const int T = 5, heads = 8, d = 128;
  const std::vector<float> x = RoundBf16(Rand(static_cast<size_t>(T * heads * d), 3.0f));
  const std::vector<float> scale = Rand(static_cast<size_t>(heads * d), 1.0f);
  DeviceBuffer<uint16_t> xd = Dev(ToBf16(x));
  DeviceBuffer<float> sd = Dev(scale);
  for (int log : {0, 1}) {
    const int64_t pos0 = 9000;
    std::vector<float> want(x.size());
    ref::HeadClip(x.data(), T, heads, d, scale.data(), ref::kQkClip, pos0, log ? 0.1f : 0.0f, 8192.0f, want.data());
    DeviceBuffer<float> of(x.size());
    R4dxInkling2HeadOut o{};
    o.kind = 0;
    o.out_bf16 = 0;
    o.out = P(of.data());
    o.out_row_stride = heads * d;
    r4dx_inkling2_head_clip(P(xd.data()), 1, heads * d, P(sd.data()), ref::kQkClip, pos0, log ? 0.1f : 0.0f, 8192.0f, &o, T,
                            heads, d, 0);
    const double e = RelErr(Host(of, x.size()), want, 1e-3);
    Check(e < 1e-5, std::string("head_clip dense fp32") + (log ? " +logscale" : "") + " (rel " + std::to_string(e) + ")");
  }
  // paged K (clip) and V (rms) through a ring table: block 16, 3 physical blocks, positions 40..44
  const int bs = 16, nblk = 3, kvh = heads;
  std::vector<int32_t> table(8);
  for (int b = 0; b < 8; ++b) table[static_cast<size_t>(b)] = b % nblk;
  DeviceBuffer<int32_t> td = Dev(table);
  const int64_t blk_stride = static_cast<int64_t>(kvh) * bs * 2 * d, head_stride = static_cast<int64_t>(bs) * 2 * d;
  DeviceBuffer<uint16_t> kv(static_cast<size_t>(nblk * blk_stride));
  kv.Zero();
  const int64_t pos0 = 40;
  R4dxInkling2HeadOut o{};
  o.kind = 1;
  o.out = P(kv.data());
  o.block_table = P(td.data());
  o.block_size = bs;
  o.kv_block_stride = blk_stride;
  o.kv_head_stride = head_stride;
  o.col_offset = 0;
  r4dx_inkling2_head_clip(P(xd.data()), 1, heads * d, 0, ref::kQkClip, pos0, 0.0f, 1.0f, &o, T, heads, d, 0);
  o.col_offset = d;
  r4dx_inkling2_head_rms(P(xd.data()), 1, heads * d, 1e-6f, pos0, &o, T, heads, d, 0);
  std::vector<float> wk(x.size()), wv(x.size());
  ref::HeadClip(x.data(), T, heads, d, nullptr, ref::kQkClip, pos0, 0.0f, 1.0f, wk.data());
  ref::HeadRms(x.data(), T, heads, d, wv.data());
  const std::vector<float> cache = HostBf16(kv, kv.size());
  std::vector<float> gk, gv, ek, ev;
  for (int t = 0; t < T; ++t) {
    const int64_t p = pos0 + t;
    for (int h = 0; h < heads; ++h) {
      const int64_t base = table[static_cast<size_t>(p / bs)] * blk_stride + h * head_stride + (p % bs) * 2 * d;
      for (int i = 0; i < d; ++i) {
        gk.push_back(cache[static_cast<size_t>(base + i)]);
        gv.push_back(cache[static_cast<size_t>(base + d + i)]);
        ek.push_back(wk[(static_cast<size_t>(t) * heads + h) * d + i]);
        ev.push_back(wv[(static_cast<size_t>(t) * heads + h) * d + i]);
      }
    }
  }
  Check(Bf16Close(gk, ek), "head_clip paged K through a ring table (bf16)");
  Check(Bf16Close(gv, ev), "head_rms paged V through a ring table (bf16)");
}

void TestRelGateMsg() {
  {  // rel logits
    const int rows = 7 * 24, drel = 16, ext = 1024;
    const std::vector<float> r = Rand(static_cast<size_t>(rows * drel)), proj = Rand(static_cast<size_t>(drel * ext), 0.2f);
    std::vector<float> want(static_cast<size_t>(rows * ext));
    ref::RelLogits(r.data(), 7, 24, drel, proj.data(), ext, want.data());
    DeviceBuffer<float> rd = Dev(r), pd = Dev(proj), od(want.size());
    r4dx_inkling2_rel_logits_f32(P(rd.data()), P(pd.data()), P(od.data()), rows, drel, ext, 0);
    const double e = RelErr(Host(od, want.size()), want, 1e-2);
    Check(e < 1e-4, "rel_logits [168 x 16] x [16 x 1024] (rel " + std::to_string(e) + ")");
  }
  {  // sigmoid gate
    const int64_t n = 3 * 3072;
    const std::vector<float> o = RoundBf16(Rand(static_cast<size_t>(n))), g = RoundBf16(Rand(static_cast<size_t>(n), 3.0f));
    std::vector<float> want(static_cast<size_t>(n));
    ref::SigmoidGateMul(o.data(), g.data(), n, want.data());
    DeviceBuffer<uint16_t> od = Dev(ToBf16(o)), gd = Dev(ToBf16(g));
    r4dx_inkling2_sigmoid_gate_mul_bf16(P(od.data()), P(gd.data()), P(od.data()), n, 0);
    Check(Bf16Close(HostBf16(od, static_cast<size_t>(n)), want, 1), "sigmoid_gate_mul in place");
  }
  {  // message gate
    const int T = 4, H = 2304, R = 64;
    const std::vector<float> x = Rand(static_cast<size_t>(T * H)), m = Rand(static_cast<size_t>(T * H), 2.0f);
    const std::vector<float> down = RoundBf16(Rand(static_cast<size_t>(R * 2 * H), 0.02f));
    const std::vector<float> forget = RoundBf16(Rand(static_cast<size_t>(H * R), 0.1f)), gain = RoundBf16(Rand(static_cast<size_t>(H * R), 0.1f));
    std::vector<float> prep(static_cast<size_t>(T * 2 * H));
    ref::MsgGatePrep(x.data(), m.data(), T, H, prep.data());
    DeviceBuffer<float> xd = Dev(x), md = Dev(m), fd = Dev(forget), gnd = Dev(gain), od(static_cast<size_t>(T * H));
    DeviceBuffer<uint16_t> pd(prep.size());
    r4dx_inkling2_msg_gate_prep_bf16(P(xd.data()), P(md.data()), P(pd.data()), T, H, 1e-6f, 0);
    Check(Bf16Close(HostBf16(pd, prep.size()), prep, 1), "msg_gate_prep (bf16 [rms x | rms m])");
    // g from the bf16 prep (the GEMM input), through the dense kernel
    const std::vector<float> prep_b = RoundBf16(prep);
    std::vector<float> g(static_cast<size_t>(T * R));
    for (int t = 0; t < T; ++t)
      for (int r = 0; r < R; ++r) {
        double a = 0;
        for (int k = 0; k < 2 * H; ++k) a += static_cast<double>(prep_b[static_cast<size_t>(t) * 2 * H + k]) * down[static_cast<size_t>(r) * 2 * H + k];
        g[static_cast<size_t>(t) * R + r] = static_cast<float>(a);
      }
    DeviceBuffer<uint16_t> dd = Dev(ToBf16(down));
    DeviceBuffer<float> gd(g.size());
    DeviceBuffer<uint16_t> pb = Dev(ToBf16(prep_b));
    r4dx_inkling2_dense_bf16(P(pb.data()), 2 * H, P(dd.data()), P(gd.data()), 0, R, T, R, 2 * H, 0);
    const std::vector<float> gg = Host(gd, g.size());
    const double eg = RelErr(gg, g, 1e-2);
    Check(eg < 1e-4, "dense_bf16 fp32-out [4 x 4608] x [64 x 4608] (msg-gate down; rel " + std::to_string(eg) + ")");
    std::vector<float> want(static_cast<size_t>(T * H));
    ref::MsgGateApply(x.data(), m.data(), gg.data(), forget.data(), gain.data(), T, H, R, want.data());
    r4dx_inkling2_msg_gate_apply_f32(P(xd.data()), P(md.data()), P(gd.data()), 0, P(fd.data()), P(gnd.data()), P(od.data()), T,
                                     H, R, 0);
    const double e = RelErr(Host(od, want.size()), want, 1e-2);
    Check(e < 1e-4, "msg_gate_apply (rel " + std::to_string(e) + ")");
  }
  {  // dedicated Inkling2 MoE sort + gather + combine
    const int T = 64, K = 8, E = 128, H = 2304, Pn = T * K;
    std::vector<int32_t> ids(static_cast<size_t>(Pn));
    for (int p = 0; p < Pn; ++p) ids[static_cast<size_t>(p)] = (p * 37 + 13) % E;
    std::vector<int32_t> ref_cnt(static_cast<size_t>(E), 0), ref_off(static_cast<size_t>(E + 1), 0),
        ref_sorted(static_cast<size_t>(Pn), 0), ref_pos(static_cast<size_t>(Pn), 0);
    for (int e = 0; e < E; ++e) {
      for (int p = 0; p < Pn; ++p)
        if (ids[static_cast<size_t>(p)] == e) ++ref_cnt[static_cast<size_t>(e)];
      ref_off[static_cast<size_t>(e + 1)] = ref_off[static_cast<size_t>(e)] + ref_cnt[static_cast<size_t>(e)];
    }
    for (int e = 0; e < E; ++e) {
      int pos = ref_off[static_cast<size_t>(e)];
      for (int p = 0; p < Pn; ++p) {
        if (ids[static_cast<size_t>(p)] == e) {
          ref_sorted[static_cast<size_t>(pos)] = p;
          ref_pos[static_cast<size_t>(p)] = pos;
          ++pos;
        }
      }
    }
    DeviceBuffer<int32_t> id_d = Dev(ids), cnt_d(E), off_d(E + 1), sort_d(Pn), pos_d(Pn);
    r4dx_inkling2_moe_sort(P(id_d.data()), T, K, E, P(cnt_d.data()), P(off_d.data()), P(sort_d.data()), P(pos_d.data()), 0);
    Check(Host(cnt_d, E) == ref_cnt && Host(off_d, E + 1) == ref_off && Host(sort_d, Pn) == ref_sorted &&
              Host(pos_d, Pn) == ref_pos,
          "inkling2_moe_sort bit-exact (T=64, K=8, E=128)");
  }
}

void TestRouterSwigluSoftcapDense() {
  {  // router: 128 experts, top-8, with a forced tie
    const int T = 3, H = 2304, E = 128, K = 8;
    const std::vector<float> x = RoundBf16(Rand(static_cast<size_t>(T * H)));
    std::vector<float> w = RoundBf16(Rand(static_cast<size_t>(E * H), 0.02f));
    for (int h = 0; h < H; ++h) w[static_cast<size_t>(77) * H + h] = w[static_cast<size_t>(5) * H + h];  // experts 5 and 77 tie
    std::vector<float> bias = Rand(static_cast<size_t>(E), 0.01f);
    bias[77] = bias[5];
    std::vector<int32_t> ids(static_cast<size_t>(T * K));
    std::vector<float> wts(static_cast<size_t>(T * K)), lg(static_cast<size_t>(T * E));
    ref::Router(x.data(), w.data(), bias.data(), T, H, E, K, 8.0f, ids.data(), wts.data(), lg.data());
    DeviceBuffer<uint16_t> xd = Dev(ToBf16(x)), wd = Dev(ToBf16(w));
    DeviceBuffer<float> bd = Dev(bias), wo(wts.size()), lo(lg.size());
    DeviceBuffer<int32_t> io(ids.size());
    r4dx_inkling2_router_f32(P(xd.data()), P(wd.data()), P(bd.data()), P(io.data()), P(wo.data()), T, H, E, K, 8.0f, H,
                             P(lo.data()), 0);
    const std::vector<int32_t> gi = Host(io, ids.size());
    // the ids may differ only where the reference scores are within float noise of each other
    Check(gi == ids, "router ids (128 experts, top-8, ties to the lower index)");
    Check(RelErr(Host(wo, wts.size()), wts, 1e-3) < 1e-4, "router weights");
    Check(RelErr(Host(lo, lg.size()), lg, 1e-2) < 1e-4, "router logits_out");
  }
  {  // softcapped SwiGLU
    const int64_t rows = 5, I = 1152;
    const std::vector<float> gu = RoundBf16(Rand(static_cast<size_t>(rows * 2 * I), 6.0f));
    std::vector<float> want(static_cast<size_t>(rows * I));
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t i = 0; i < I; ++i)
        want[static_cast<size_t>(r * I + i)] = ref::SwigluSoftcap(gu[static_cast<size_t>(r * 2 * I + i)], gu[static_cast<size_t>(r * 2 * I + I + i)]);
    DeviceBuffer<uint16_t> gd = Dev(ToBf16(gu)), od(want.size());
    r4dx_inkling2_swiglu_softcap_bf16(P(gd.data()), P(od.data()), rows, I, 2 * I, 0);
    Check(Bf16Close(HostBf16(od, want.size()), want, 1), "swiglu_softcap [5 x 1152]");
  }
  {  // final logit softcap with a padded row stride
    const int rows = 2, n = 1000, xs = 1008;
    const std::vector<float> x = RoundBf16(Rand(static_cast<size_t>(rows * xs), 40.0f));
    std::vector<float> want(static_cast<size_t>(rows * n));
    for (int r = 0; r < rows; ++r)
      for (int i = 0; i < n; ++i) want[static_cast<size_t>(r * n + i)] = ref::Softcap(x[static_cast<size_t>(r * xs + i)], 30.0f);
    DeviceBuffer<uint16_t> xd = Dev(ToBf16(x));
    DeviceBuffer<float> od(want.size());
    r4dx_inkling2_logit_softcap_f32(P(xd.data()), 1, xs, P(od.data()), n, rows, n, 30.0f, 0);
    Check(RelErr(Host(od, want.size()), want, 1e-3) < 1e-5, "logit_softcap bf16 in, row stride 1008");
  }
  {  // dense bf16-out at an odd shape (the tiny fixture's expert down: K 32, N 64) and the sink shape (N 24)
    const std::vector<std::array<int, 3>> shapes = {{3, 64, 32}, {7, 24, 2304}, {1, 32, 2304}};
    for (const std::array<int, 3>& s : shapes) {
      const int M = s[0], N = s[1], K = s[2];
      const std::vector<float> x = RoundBf16(Rand(static_cast<size_t>(M * K))), w = RoundBf16(Rand(static_cast<size_t>(N * K), 0.1f));
      std::vector<float> want(static_cast<size_t>(M * N));
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double a = 0;
          for (int k = 0; k < K; ++k) a += static_cast<double>(x[static_cast<size_t>(m) * K + k]) * w[static_cast<size_t>(n) * K + k];
          want[static_cast<size_t>(m * N + n)] = static_cast<float>(a);
        }
      DeviceBuffer<uint16_t> xd = Dev(ToBf16(x)), wd = Dev(ToBf16(w)), ob(want.size());
      r4dx_inkling2_dense_bf16(P(xd.data()), K, P(wd.data()), P(ob.data()), 1, N, M, N, K, 0);
      Check(Bf16Close(HostBf16(ob, want.size()), want, 2),
            "dense_bf16 bf16-out [" + std::to_string(M) + " x " + std::to_string(K) + "] x [" + std::to_string(N) + " x " + std::to_string(K) + "]");
    }
  }
}

// ---- the reference attention with bias + sink + window ------------------------------------------------------------
void TestAttention() {
  const int qh = 24, kvh = 8, d = 128, bs = 16;
  struct Case {
    int ctx, q_len, window, extent;
    bool ring, sink, bias;
  };
  const Case cases[] = {
      {37, 37, 0, 1024, false, true, true},   // full prefill from 0
      {300, 44, 0, 64, false, true, true},    // full, a later chunk; bias extent shorter than the context
      {300, 1, 0, 64, false, true, true},     // full decode
      {300, 44, 128, 128, true, true, true},  // sliding chunk crossing the window, ring table
      {300, 1, 128, 128, true, true, true},   // sliding decode
      {300, 5, 128, 128, true, false, false}, // no bias, no sink
  };
  for (const Case& c : cases) {
    const int ring_blocks = (c.window + c.q_len - 1 + bs - 1) / bs + 1;
    const int lblocks = (c.ctx + bs - 1) / bs;
    const int nblk = c.ring ? ring_blocks : lblocks;
    std::vector<int32_t> table(static_cast<size_t>(lblocks));
    for (int b = 0; b < lblocks; ++b) table[static_cast<size_t>(b)] = c.ring ? b % ring_blocks : b;
    const std::vector<float> q = RoundBf16(Rand(static_cast<size_t>(c.q_len * qh * d), 0.3f));
    const std::vector<float> k = RoundBf16(Rand(static_cast<size_t>(c.ctx * kvh * d), 0.3f));
    const std::vector<float> v = RoundBf16(Rand(static_cast<size_t>(c.ctx * kvh * d)));
    const std::vector<float> rel = Rand(static_cast<size_t>(c.q_len * qh * c.extent), 0.5f);
    const std::vector<float> sink = Rand(static_cast<size_t>(c.q_len * qh), 1.0f);
    // the paged cache: only positions a ring still holds are written (the newest of each slot), which is what the model does
    const int64_t blk_stride = static_cast<int64_t>(kvh) * bs * 2 * d, head_stride = static_cast<int64_t>(bs) * 2 * d;
    std::vector<uint16_t> cache(static_cast<size_t>(nblk * blk_stride), 0);
    for (int p = 0; p < c.ctx; ++p)
      for (int h = 0; h < kvh; ++h)
        for (int i = 0; i < d; ++i) {
          const int64_t base = table[static_cast<size_t>(p / bs)] * blk_stride + h * head_stride + (p % bs) * 2 * d;
          cache[static_cast<size_t>(base + i)] = FloatToBf16(k[(static_cast<size_t>(p) * kvh + h) * d + i]);
          cache[static_cast<size_t>(base + d + i)] = FloatToBf16(v[(static_cast<size_t>(p) * kvh + h) * d + i]);
        }
    std::vector<float> want(static_cast<size_t>(c.q_len * qh * d));
    ref::Attention(q.data(), k.data(), v.data(), c.bias ? rel.data() : nullptr, c.sink ? sink.data() : nullptr, c.q_len, qh, kvh,
                   d, c.ctx, c.window, c.extent, 1.0f / std::sqrt(static_cast<float>(d)), want.data());
    DeviceBuffer<uint16_t> qd = Dev(ToBf16(q)), kvd = Dev(cache), od(want.size());
    DeviceBuffer<int32_t> td = Dev(table);
    DeviceBuffer<float> rd = Dev(rel), sd = Dev(sink);
    R4dxInkling2AttnRefArgs a{};
    a.q = P(qd.data());
    a.kv = P(kvd.data());
    a.block_table = P(td.data());
    a.out = P(od.data());
    a.rel = c.bias ? P(rd.data()) : 0;
    a.sink = c.sink ? P(sd.data()) : 0;
    a.q_len = c.q_len;
    a.q_heads = qh;
    a.kv_heads = kvh;
    a.head_dim = d;
    a.block_size = bs;
    a.ctx = c.ctx;
    a.window = c.window;
    a.rel_extent = c.extent;
    a.scale = 1.0f / std::sqrt(static_cast<float>(d));
    a.kv_block_stride = blk_stride;
    a.kv_head_stride = head_stride;
    const int rc = r4dx_inkling2_attn_ref_bf16kv(&a, 0);
    const std::string tag = "attn ctx " + std::to_string(c.ctx) + " q_len " + std::to_string(c.q_len) + " window " +
                            std::to_string(c.window) + (c.ring ? " ring" : "") + (c.bias ? " +bias" : "") + (c.sink ? " +sink" : "");
    Check(rc == 0, tag + ": launch accepted");
    Check(rc == 0 && Bf16Close(HostBf16(od, want.size()), want, 2), tag + ": matches inkling2_ref::Attention");
  }
  // refusals
  R4dxInkling2AttnRefArgs bad{};
  Check(r4dx_inkling2_attn_ref_bf16kv(nullptr, 0) < 0, "attn: null args refused");
  Check(r4dx_inkling2_attn_ref_bf16kv(&bad, 0) < 0, "attn: zero shapes refused");
}

// ---- S3 fused h128 attention vs reference kernel & CPU spec (contexts 1..9000, window/sink/bias on/off) ------------
void TestFusedAttention() {
  const int qh = 24, kvh = 8, d = 128, bs = 16, drel = 16, ext = 1024;
  struct FCase {
    int ctx, q_len, window;
    bool ring, sink, bias;
  };
  const FCase cases[] = {
      {1, 1, 0, false, true, true},
      {17, 1, 0, false, true, true},
      {17, 17, 0, false, true, true},
      {1023, 1, 0, false, true, true},
      {1024, 1, 1024, true, true, true},
      {1025, 1, 1024, true, true, true},
      {1536, 1, 1024, true, true, true},
      {1536, 17, 1024, true, true, true},
      {4096, 1, 0, false, true, true},
      {4096, 1, 0, false, false, false},
      {9000, 1, 0, false, true, true},
      {300, 64, 0, false, true, true},
      {300, 64, 128, true, true, true},
  };
  const std::vector<float> proj = Rand(static_cast<size_t>(drel * ext), 0.2f);
  DeviceBuffer<float> proj_d = Dev(proj);
  DeviceBuffer<float> ws_d(static_cast<size_t>(qh * 32 * (d + 2)));

  for (const FCase& c : cases) {
    const int ring_blocks = c.window > 0 ? (c.window + c.q_len - 1 + bs - 1) / bs + 1 : 0;
    const int lblocks = (c.ctx + bs - 1) / bs;
    const int nblk = c.ring ? std::min(lblocks, ring_blocks) : lblocks;
    std::vector<int32_t> table(static_cast<size_t>(lblocks));
    for (int b = 0; b < lblocks; ++b) table[static_cast<size_t>(b)] = c.ring ? b % nblk : b;
    const std::vector<float> q = RoundBf16(Rand(static_cast<size_t>(c.q_len * qh * d), 0.3f));
    const std::vector<float> r = Rand(static_cast<size_t>(c.q_len * qh * drel), 0.5f);
    const std::vector<float> sink = Rand(static_cast<size_t>(c.q_len * qh), 1.0f);
    const int64_t blk_stride = static_cast<int64_t>(kvh) * bs * 2 * d, head_stride = static_cast<int64_t>(bs) * 2 * d;
    std::vector<uint16_t> cache(static_cast<size_t>(nblk * blk_stride), 0);
    // Fill only the active blocks (the last nblk blocks of ctx) so 9000-ctx test stays fast on CPU setup.
    const int p_start = c.ring ? std::max(0, c.ctx - nblk * bs) : 0;
    const std::vector<float> kv_rand = RoundBf16(Rand(static_cast<size_t>((c.ctx - p_start) * kvh * 2 * d), 0.3f));
    for (int p = p_start; p < c.ctx; ++p) {
      const int64_t src_p = static_cast<int64_t>(p - p_start) * kvh * 2 * d;
      for (int h = 0; h < kvh; ++h) {
        const int64_t base = table[static_cast<size_t>(p / bs)] * blk_stride + h * head_stride + (p % bs) * 2 * d;
        for (int i = 0; i < 2 * d; ++i)
          cache[static_cast<size_t>(base + i)] = FloatToBf16(kv_rand[static_cast<size_t>(src_p + h * 2 * d + i)]);
      }
    }
    DeviceBuffer<uint16_t> qd = Dev(ToBf16(q)), kvd = Dev(cache), o_ref(static_cast<size_t>(c.q_len * qh * d)),
                           o_fus(static_cast<size_t>(c.q_len * qh * d));
    DeviceBuffer<int32_t> td = Dev(table);
    DeviceBuffer<float> rd = Dev(r), sd = Dev(sink);
    DeviceBuffer<float> rel_d(static_cast<size_t>(c.q_len * qh * ext));
    if (c.bias) {
      r4dx_inkling2_rel_logits_f32(P(rd.data()), P(proj_d.data()), P(rel_d.data()), c.q_len * qh, drel, ext, 0);
    }

    R4dxInkling2AttnRefArgs ar{};
    ar.q = P(qd.data());
    ar.kv = P(kvd.data());
    ar.block_table = P(td.data());
    ar.out = P(o_ref.data());
    ar.rel = c.bias ? P(rel_d.data()) : 0;
    ar.sink = c.sink ? P(sd.data()) : 0;
    ar.q_len = c.q_len;
    ar.q_heads = qh;
    ar.kv_heads = kvh;
    ar.head_dim = d;
    ar.block_size = bs;
    ar.ctx = c.ctx;
    ar.window = c.window;
    ar.rel_extent = ext;
    ar.scale = 1.0f / std::sqrt(static_cast<float>(d));
    ar.kv_block_stride = blk_stride;
    ar.kv_head_stride = head_stride;
    Check(r4dx_inkling2_attn_ref_bf16kv(&ar, 0) == 0, "fused_attn ref launch ctx=" + std::to_string(c.ctx));

    R4dxInkling2AttnFusedArgs af{};
    af.q = P(qd.data());
    af.kv = P(kvd.data());
    af.block_table = P(td.data());
    af.out = P(o_fus.data());
    af.r = c.bias ? P(rd.data()) : 0;
    af.rel_proj = c.bias ? P(proj_d.data()) : 0;
    af.sink = c.sink ? P(sd.data()) : 0;
    af.workspace = P(ws_d.data());
    af.q_len = c.q_len;
    af.q_heads = qh;
    af.kv_heads = kvh;
    af.head_dim = d;
    af.block_size = bs;
    af.ctx = c.ctx;
    af.window = c.window;
    af.d_rel = drel;
    af.rel_extent = ext;
    af.max_splits = 32;
    af.scale = 1.0f / std::sqrt(static_cast<float>(d));
    af.kv_block_stride = blk_stride;
    af.kv_head_stride = head_stride;
    const int rc = r4dx_inkling2_attn_fused_bf16kv(&af, 0);
    const std::string tag = "fused_attn ctx " + std::to_string(c.ctx) + " q_len " + std::to_string(c.q_len) +
                            " win " + std::to_string(c.window) + (c.bias ? " +bias" : "") + (c.sink ? " +sink" : "");
    Check(rc == 0, tag + ": launch accepted");
    const std::vector<float> fus_host = HostBf16(o_fus, static_cast<size_t>(c.q_len * qh * d));
    Check(rc == 0 && Bf16Close(fus_host, HostBf16(o_ref, static_cast<size_t>(c.q_len * qh * d)), 2),
          tag + ": matches ref kernel within 2 bf16 ULP");
    if (c.q_len == 1 || c.q_len <= 8) {
      DeviceBuffer<int32_t> pos_d = Dev(std::vector<int32_t>{c.ctx - c.q_len});
      DeviceBuffer<uint16_t> o_pd(static_cast<size_t>(c.q_len * qh * d));
      DeviceBuffer<float> ws_q(static_cast<size_t>(c.q_len * qh * 32 * (d + 2)));
      R4dxInkling2AttnFusedArgs ap = af;
      ap.out = P(o_pd.data());
      ap.workspace = P(ws_q.data());
      ap.ctx = 0;
      ap.pos_dev = P(pos_d.data());
      const int rcp = r4dx_inkling2_attn_fused_bf16kv(&ap, 0);
      Check(rcp == 0 && HostBf16(o_pd, static_cast<size_t>(c.q_len * qh * d)) == fus_host,
            tag + ": pos_dev mode is bit-identical to host ctx");
    }
  }
  R4dxInkling2AttnFusedArgs bad{};
  Check(r4dx_inkling2_attn_fused_bf16kv(nullptr, 0) < 0, "fused_attn: null args refused");
  Check(r4dx_inkling2_attn_fused_bf16kv(&bad, 0) < 0, "fused_attn: zero shapes refused");
}

// ---- DSpark / MTP drafter kernels + short-conv rollback + qgrkv_split ----------------------------------------------
void TestDsparkKernels() {
  const float eps = 1e-6f;
  // 1. mix_rmsnorm_bf16
  {
    const int T = 4, H = 512;
    const std::vector<float> t0 = RoundBf16(Rand(T * H)), t1 = RoundBf16(Rand(T * H)), t2 = RoundBf16(Rand(T * H)),
                             t3 = RoundBf16(Rand(T * H)), w = RoundBf16(Rand(H, 1.0f));
    const float a0 = 0.1f, a1 = 0.2f, a2 = 0.3f, a3 = 0.4f;
    std::vector<float> want(T * H);
    for (int t = 0; t < T; ++t) {
      double ss = 0.0;
      std::vector<float> m(H);
      for (int h = 0; h < H; ++h) {
        const size_t i = static_cast<size_t>(t * H + h);
        m[h] = a0 * t0[i] + a1 * t1[i] + a2 * t2[i] + a3 * t3[i];
        ss += static_cast<double>(m[h]) * m[h];
      }
      const float inv = 1.0f / std::sqrt(static_cast<float>(ss / H) + eps);
      for (int h = 0; h < H; ++h) want[static_cast<size_t>(t * H + h)] = m[h] * inv * w[h];
    }
    DeviceBuffer<uint16_t> d0 = Dev(ToBf16(t0)), d1 = Dev(ToBf16(t1)), d2 = Dev(ToBf16(t2)), d3 = Dev(ToBf16(t3)),
                           dw = Dev(ToBf16(w)), dout(T * H);
    r4dx_inkling2_dspark_mix_rmsnorm_bf16(P(d0.data()), P(d1.data()), P(d2.data()), P(d3.data()), a0, a1, a2, a3,
                                          P(dw.data()), P(dout.data()), T, H, eps, 0);
    Check(Bf16Close(HostBf16(dout, T * H), want, 2), "dspark_mix_rmsnorm_bf16");
  }

  // 2. head_norm_rope_bf16
  {
    const int T = 4, heads = 6, hd = 128;
    const int64_t pos0 = 37;
    const float theta = 1000000.0f;
    const std::vector<float> x = RoundBf16(Rand(T * heads * hd)), w = RoundBf16(Rand(hd, 1.0f));
    std::vector<float> want(T * heads * hd);
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < heads; ++h) {
        double ss = 0.0;
        const size_t base = static_cast<size_t>((t * heads + h) * hd);
        for (int i = 0; i < hd; ++i) ss += static_cast<double>(x[base + i]) * x[base + i];
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / hd) + eps);
        const int half = hd / 2;
        for (int i = 0; i < half; ++i) {
          const float x1 = Bf16ToFloat(FloatToBf16(x[base + i] * inv * w[i]));
          const float x2 = Bf16ToFloat(FloatToBf16(x[base + half + i] * inv * w[half + i]));
          const float angle = static_cast<float>(pos0 + t) * std::pow(theta, -2.0f * i / hd);
          const float c = std::cos(angle), s = std::sin(angle);
          want[base + i] = x1 * c - x2 * s;
          want[base + half + i] = x2 * c + x1 * s;
        }
      }
    }
    DeviceBuffer<uint16_t> dx = Dev(ToBf16(x)), dw = Dev(ToBf16(w));
    r4dx_inkling2_dspark_head_norm_rope_bf16(P(dx.data()), heads * hd, P(dw.data()), T, heads, hd, pos0, theta, eps, 0);
    Check(Bf16Close(HostBf16(dx, want.size()), want, 2), "dspark_head_norm_rope_bf16");
  }

  // 3. markov_gate_bf16 + add_softcap_f32 + transpose_bf16
  {
    const int R = 256;
    const std::vector<float> raw = RoundBf16(Rand(3 * R)), bias = RoundBf16(Rand(3 * R)), st = RoundBf16(Rand(R));
    std::vector<float> want_st(R), want_lat(R);
    auto rbf = [](float v) { return Bf16ToFloat(FloatToBf16(v)); };
    for (int c = 0; c < R; ++c) {
      const float g_in = rbf(raw[c] + bias[c]);
      const float cand_in = rbf(raw[R + c] + bias[R + c]);
      const float out_in = rbf(raw[2 * R + c] + bias[2 * R + c]);
      const float gate = rbf(1.0f / (1.0f + std::exp(-g_in)));
      const float cand = rbf(std::tanh(cand_in));
      want_st[c] = rbf(rbf(gate * st[c]) + rbf(rbf(1.0f - gate) * cand));
      want_lat[c] = rbf(std::tanh(out_in));
    }
    DeviceBuffer<uint16_t> dr = Dev(ToBf16(raw)), db = Dev(ToBf16(bias)), ds = Dev(ToBf16(st)), dso(R), dlat(R);
    r4dx_inkling2_dspark_markov_gate_bf16(P(dr.data()), P(db.data()), P(ds.data()), P(dso.data()), P(dlat.data()), R, 0);
    Check(Bf16Close(HostBf16(dso, R), want_st, 1), "dspark_markov_gate_bf16 state");
    Check(Bf16Close(HostBf16(dlat, R), want_lat, 1), "dspark_markov_gate_bf16 latent");

    const std::vector<float> a = RoundBf16(Rand(R)), b = RoundBf16(Rand(R)), c = RoundBf16(Rand(R));
    std::vector<float> want_sc(R);
    for (int i = 0; i < R; ++i) want_sc[i] = 30.0f * std::tanh((a[i] + b[i] + c[i]) / 30.0f);
    DeviceBuffer<uint16_t> da = Dev(ToBf16(a)), dbb = Dev(ToBf16(b)), dc = Dev(ToBf16(c));
    DeviceBuffer<float> dsc(R);
    r4dx_inkling2_dspark_add_softcap_f32(P(da.data()), P(dbb.data()), P(dc.data()), P(dsc.data()), R, 30.0f, 0);
    Check(RelErr(Host(dsc, R), want_sc) < 1e-4, "dspark_add_softcap_f32");

    const int rows = 16, cols = 48;
    const std::vector<uint16_t> m_in = ToBf16(Rand(rows * cols));
    DeviceBuffer<uint16_t> dm_in = Dev(m_in), dm_out(rows * cols);
    r4dx_inkling2_dspark_transpose_bf16(P(dm_in.data()), P(dm_out.data()), rows, cols, 0);
    std::vector<uint16_t> m_got(rows * cols);
    dm_out.CopyToHost(m_got.data(), m_got.size());
    bool tr_ok = true;
    for (int r = 0; r < rows; ++r)
      for (int cl = 0; cl < cols; ++cl)
        if (m_got[static_cast<size_t>(cl * rows + r)] != m_in[static_cast<size_t>(r * cols + cl)]) tr_ok = false;
    Check(tr_ok, "dspark_transpose_bf16");
  }

  // 4. sconv_static_hist + sconv_rollback exactness across n_keep = 0..T
  {
    const int T = 6, C = 512;
    const std::vector<float> x = RoundBf16(Rand(T * C)), w = Rand(C * 4, 0.5f), init = Rand(3 * C, 0.5f);
    DeviceBuffer<uint16_t> dx = Dev(ToBf16(x));
    DeviceBuffer<float> dw = Dev(w), dbak = Dev(init), dhist(T * C), dout(T * C), dst_full = Dev(init);
    r4dx_inkling2_sconv_static_hist_f32(P(dx.data()), 1, P(dw.data()), P(dst_full.data()), 0, P(dout.data()),
                                        P(dhist.data()), T, C, 0);
    for (int n_keep = 0; n_keep <= T; ++n_keep) {
      DeviceBuffer<float> dst_rb(3 * C);
      r4dx_inkling2_sconv_rollback_f32(P(dbak.data()), P(dhist.data()), P(dst_rb.data()), n_keep, C, 0);
      DeviceBuffer<float> dst_ref = Dev(init), dout_ref(std::max(1, n_keep) * C);
      if (n_keep > 0) {
        r4dx_inkling2_sconv_static_f32(P(dx.data()), 1, P(dw.data()), P(dst_ref.data()), 0, P(dout_ref.data()),
                                       n_keep, C, 0);
      }
      Check(Host(dst_rb, 3 * C) == Host(dst_ref, 3 * C),
            "sconv_rollback bit-exact n_keep=" + std::to_string(n_keep));
    }
  }

  // 5. qgrkv_split_bf16
  {
    const int T = 4, qn = 64, gn = 64, rn = 32, kn = 32, vn = 32, total = qn + gn + rn + kn + vn;
    const std::vector<uint16_t> src = ToBf16(Rand(T * total));
    DeviceBuffer<uint16_t> ds = Dev(src), dq(T * qn), dg(T * gn), dr(T * rn), dk(T * kn), dv(T * vn);
    r4dx_inkling2_qgrkv_split_bf16(P(ds.data()), total, P(dq.data()), 0, qn, P(dg.data()), qn, gn, P(dr.data()),
                                   qn + gn, rn, P(dk.data()), qn + gn + rn, kn, P(dv.data()), qn + gn + rn + kn, vn, T,
                                   0);
    std::vector<uint16_t> hq(T * qn), hv(T * vn);
    dq.CopyToHost(hq.data(), hq.size());
    dv.CopyToHost(hv.data(), hv.size());
    bool sp_ok = true;
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < qn; ++c)
        if (hq[t * qn + c] != src[t * total + c]) sp_ok = false;
      for (int c = 0; c < vn; ++c)
        if (hv[t * vn + c] != src[t * total + qn + gn + rn + kn + c]) sp_ok = false;
    }
    Check(sp_ok, "qgrkv_split_bf16");
  }
}

void TestSpeedupKernels() {
  // 1. softcap_argmax_rows_bf16 (write_pair=1, write_pair=0, ties, and n=0 sentinel)
  {
    const int rows = 3, stride = 300, n = 275;
    const float softcap = 30.0f;
    std::vector<float> f(rows * stride, -5.0f);
    // Row 0: tie between idx 5 and 200 -> lower index 5 must win.
    f[0 * stride + 5] = 40.0f;
    f[0 * stride + 200] = 40.0f;
    // Row 1: clear winner at 274 (last element).
    f[1 * stride + 274] = 12.5f;
    // Row 2: negative winner at 10.
    f[2 * stride + 10] = -0.25f;
    const std::vector<uint16_t> bf = ToBf16(f);
    DeviceBuffer<uint16_t> d_in = Dev(bf);
    DeviceBuffer<uint8_t> d_ws(static_cast<size_t>(rows) * 1024);
    DeviceBuffer<int32_t> d_pairs(rows * 2), d_idx(rows);
    r4dx_inkling2_softcap_argmax_rows_bf16(P(d_in.data()), stride, rows, n, softcap, P(d_ws.data()),
                                           P(d_pairs.data()), 1, 0);
    r4dx_inkling2_softcap_argmax_rows_bf16(P(d_in.data()), stride, rows, n, softcap, P(d_ws.data()),
                                           P(d_idx.data()), 0, 0);
    const std::vector<int32_t> pairs = Host(d_pairs, rows * 2);
    const std::vector<int32_t> idx = Host(d_idx, rows);
    auto to_f = [](int32_t bits) {
      float v = 0.0f;
      std::memcpy(&v, &bits, sizeof(v));
      return v;
    };
    const bool pair_ok =
        pairs[0] == 5 && std::fabs(to_f(pairs[1]) - softcap * std::tanh(Bf16ToFloat(bf[5]) / softcap)) < 1e-4f &&
        pairs[2] == 274 &&
        std::fabs(to_f(pairs[3]) - softcap * std::tanh(Bf16ToFloat(bf[stride + 274]) / softcap)) < 1e-4f &&
        pairs[4] == 10 &&
        std::fabs(to_f(pairs[5]) - softcap * std::tanh(Bf16ToFloat(bf[2 * stride + 10]) / softcap)) < 1e-4f;
    Check(pair_ok, "softcap_argmax_rows_bf16 write_pair=1");
    Check(idx[0] == 5 && idx[1] == 274 && idx[2] == 10, "softcap_argmax_rows_bf16 write_pair=0");

    DeviceBuffer<int32_t> d_empty(4);
    r4dx_inkling2_softcap_argmax_rows_bf16(0, 0, 2, 0, softcap, 0, P(d_empty.data()), 1, 0);
    const std::vector<int32_t> emp = Host(d_empty, 4);
    Check(emp[0] == 0x7fffffff && emp[2] == 0x7fffffff && std::isinf(to_f(emp[1])) && to_f(emp[1]) < 0.0f,
          "softcap_argmax_rows_bf16 n=0 sentinel");
  }

  // 2. dspark_markov_step_bf16 vs separate markov_prep_dev + W@x + add_f32 + markov_gate
  {
    const int R = 64, V = 8;
    const std::vector<uint16_t> w1 = ToBf16(Rand(V * R, 0.3f));
    const std::vector<uint16_t> st_in = ToBf16(Rand(R, 0.4f));
    const std::vector<uint16_t> w_rnn2r = ToBf16(Rand(3 * R * 2 * R, 0.2f));
    const std::vector<uint16_t> bias = ToBf16(Rand(3 * R, 0.1f));
    const std::vector<float> head_proj = Rand(3 * R, 0.2f);
    const std::vector<float> frnn_proj = Rand(3 * R, 0.2f);
    const std::vector<int32_t> prev_tok = {3};
    DeviceBuffer<uint16_t> dw1 = Dev(w1), dst_in = Dev(st_in), dwrnn = Dev(w_rnn2r), dbias = Dev(bias);
    DeviceBuffer<float> dhp = Dev(head_proj), dfp = Dev(frnn_proj);
    DeviceBuffer<int32_t> dtok = Dev(prev_tok);
    DeviceBuffer<uint16_t> d_st_got(R), d_lat_got(R);
    r4dx_inkling2_dspark_markov_step_bf16(P(dw1.data()), P(dtok.data()), V, P(dst_in.data()), P(dhp.data()),
                                          P(dfp.data()), P(dwrnn.data()), P(dbias.data()), P(d_st_got.data()),
                                          P(d_lat_got.data()), R, 0);

    std::vector<uint16_t> x2r(2 * R);
    for (int i = 0; i < R; ++i) {
      x2r[i] = st_in[i];
      x2r[R + i] = w1[3 * R + i];
    }
    std::vector<uint16_t> raw_ref(3 * R);
    for (int r = 0; r < 3 * R; ++r) {
      float acc = 0.0f;
      for (int c = 0; c < 2 * R; ++c) {
        acc += Bf16ToFloat(w_rnn2r[r * (2 * R) + c]) * Bf16ToFloat(x2r[c]);
      }
      acc += head_proj[r] + frnn_proj[r];
      raw_ref[r] = FloatToBf16(acc);
    }
    DeviceBuffer<uint16_t> draw = Dev(raw_ref), d_st_ref(R), d_lat_ref(R);
    r4dx_inkling2_dspark_markov_gate_bf16(P(draw.data()), P(dbias.data()), P(dst_in.data()), P(d_st_ref.data()),
                                          P(d_lat_ref.data()), R, 0);
    Check(Bf16Close(HostBf16(d_st_got, R), HostBf16(d_st_ref, R), 2) &&
              Bf16Close(HostBf16(d_lat_got, R), HostBf16(d_lat_ref, R), 2),
          "dspark_markov_step_bf16");
  }

  // 3. dspark_mix4_rmsnorm_bf16 vs 4x dspark_mix_rmsnorm_bf16
  {
    const int T = 3, H = 128;
    const float eps = 1e-6f;
    std::array<std::vector<uint16_t>, 4> taps, weights;
    std::array<DeviceBuffer<uint16_t>, 4> dtaps, dweights, douts_got, douts_ref;
    for (int i = 0; i < 4; ++i) {
      taps[i] = ToBf16(Rand(T * H, 0.5f));
      weights[i] = ToBf16(Rand(H, 0.8f));
      dtaps[i] = Dev(taps[i]);
      dweights[i] = Dev(weights[i]);
      douts_got[i] = DeviceBuffer<uint16_t>(T * H);
      douts_ref[i] = DeviceBuffer<uint16_t>(T * H);
    }
    const std::vector<float> coeffs = Rand(16, 0.4f);
    std::array<int64_t, 4> w_ptrs{P(dweights[0].data()), P(dweights[1].data()), P(dweights[2].data()),
                                  P(dweights[3].data())};
    std::array<int64_t, 4> o_ptrs{P(douts_got[0].data()), P(douts_got[1].data()), P(douts_got[2].data()),
                                  P(douts_got[3].data())};
    r4dx_inkling2_dspark_mix4_rmsnorm_bf16(P(dtaps[0].data()), P(dtaps[1].data()), P(dtaps[2].data()),
                                           P(dtaps[3].data()), coeffs.data(), w_ptrs.data(), o_ptrs.data(), T, H, eps,
                                           0);
    bool mix_ok = true;
    for (int l = 0; l < 4; ++l) {
      r4dx_inkling2_dspark_mix_rmsnorm_bf16(P(dtaps[0].data()), P(dtaps[1].data()), P(dtaps[2].data()),
                                            P(dtaps[3].data()), coeffs[l * 4 + 0], coeffs[l * 4 + 1],
                                            coeffs[l * 4 + 2], coeffs[l * 4 + 3], w_ptrs[l], P(douts_ref[l].data()), T,
                                            H, eps, 0);
      if (Host(douts_got[l], T * H) != Host(douts_ref[l], T * H)) mix_ok = false;
    }
    Check(mix_ok, "dspark_mix4_rmsnorm_bf16 bit-exact");
  }

  // 4. dspark_ctx_kv_write (layer 0 bf16 + layer 1 fp8) vs head_norm_rope + kv_write_bf16/fp8
  {
    const int T = 3, kv_heads = 2, hd = 128, slots0 = 8, slots1 = 16;
    const int64_t pos0 = 5;
    const float rope_theta = 10000.0f, eps = 1e-6f;
    const int kv_stride = kv_heads * hd;
    const std::vector<uint16_t> kv0 = ToBf16(Rand(T * 2 * kv_stride, 0.7f));
    const std::vector<uint16_t> kv1 = ToBf16(Rand(T * 2 * kv_stride, 0.7f));
    const std::vector<uint16_t> kn0 = ToBf16(Rand(hd, 0.8f)), vn0 = ToBf16(Rand(hd, 0.8f));
    DeviceBuffer<uint16_t> dkv0 = Dev(kv0), dkv1 = Dev(kv1), dkn0 = Dev(kn0), dvn0 = Dev(vn0);
    DeviceBuffer<uint16_t> ks0_got(slots0 * kv_stride), vs0_got(slots0 * kv_stride);
    DeviceBuffer<uint8_t> ks1_got(slots1 * kv_stride), vs1_got(slots1 * kv_stride);
    ks0_got.Zero();
    vs0_got.Zero();
    ks1_got.Zero();
    vs1_got.Zero();

    R4dxInkling2DsparkCtxKvArgs args{};
    args.n_layers = 2;
    args.T = T;
    args.kv_heads = kv_heads;
    args.head_dim = hd;
    args.slots[0] = slots0;
    args.slots[1] = slots1;
    args.kv_fp8[0] = 0;
    args.kv_fp8[1] = 1;
    args.pos0 = pos0;
    args.rope_theta = rope_theta;
    args.eps = eps;
    args.kv_in[0] = P(dkv0.data());
    args.kv_in[1] = P(dkv1.data());
    args.k_norm[0] = P(dkn0.data());
    args.v_norm[0] = P(dvn0.data());
    args.k_store[0] = P(ks0_got.data());
    args.v_store[0] = P(vs0_got.data());
    args.k_store[1] = P(ks1_got.data());
    args.v_store[1] = P(vs1_got.data());
    r4dx_inkling2_dspark_ctx_kv_write(&args, 0);

    // Reference: separate split + head_norm_rope + kv_write
    DeviceBuffer<uint16_t> ks0_ref(slots0 * kv_stride), vs0_ref(slots0 * kv_stride);
    DeviceBuffer<uint8_t> ks1_ref(slots1 * kv_stride), vs1_ref(slots1 * kv_stride);
    ks0_ref.Zero();
    vs0_ref.Zero();
    ks1_ref.Zero();
    vs1_ref.Zero();
    std::vector<uint16_t> k0_sep(T * kv_stride), v0_sep(T * kv_stride), k1_sep(T * kv_stride), v1_sep(T * kv_stride);
    for (int t = 0; t < T; ++t) {
      std::memcpy(k0_sep.data() + t * kv_stride, kv0.data() + t * 2 * kv_stride, kv_stride * 2);
      std::memcpy(v0_sep.data() + t * kv_stride, kv0.data() + (t * 2 + 1) * kv_stride, kv_stride * 2);
      std::memcpy(k1_sep.data() + t * kv_stride, kv1.data() + t * 2 * kv_stride, kv_stride * 2);
      std::memcpy(v1_sep.data() + t * kv_stride, kv1.data() + (t * 2 + 1) * kv_stride, kv_stride * 2);
    }
    DeviceBuffer<uint16_t> dk0 = Dev(k0_sep), dv0 = Dev(v0_sep), dv0_norm(T * kv_stride), dk1 = Dev(k1_sep),
                           dv1 = Dev(v1_sep);
    r4dx_inkling2_dspark_head_norm_rope_bf16(P(dk0.data()), kv_stride, P(dkn0.data()), T, kv_heads, hd, pos0,
                                             rope_theta, eps, 0);
    // Reference V norm using rmsnorm per head [T*kv_heads, hd]
    r4dx_inkling2_dspark_mix_rmsnorm_bf16(P(dv0.data()), P(dv0.data()), P(dv0.data()), P(dv0.data()), 1.0f, 0.0f, 0.0f,
                                          0.0f, P(dvn0.data()), P(dv0_norm.data()), T * kv_heads, hd, eps, 0);
    r4dx_inkling2_dspark_kv_write_bf16(P(dk0.data()), P(dv0_norm.data()), P(ks0_ref.data()), P(vs0_ref.data()), T,
                                       kv_heads, hd, pos0, slots0, 0);
    r4dx_inkling2_dspark_head_norm_rope_bf16(P(dk1.data()), kv_stride, 0, T, kv_heads, hd, pos0, rope_theta, eps, 0);
    r4dx_inkling2_dspark_kv_write_fp8(P(dk1.data()), P(dv1.data()), P(ks1_ref.data()), P(vs1_ref.data()), T, kv_heads,
                                      hd, pos0, slots1, 0);
    const bool ks0_ok = Host(ks0_got, slots0 * kv_stride) == Host(ks0_ref, slots0 * kv_stride);
    const bool vs0_ok = Bf16Close(HostBf16(vs0_got, slots0 * kv_stride), HostBf16(vs0_ref, slots0 * kv_stride), 2);
    const bool ks1_ok = Host(ks1_got, slots1 * kv_stride) == Host(ks1_ref, slots1 * kv_stride);
    const bool vs1_ok = Host(vs1_got, slots1 * kv_stride) == Host(vs1_ref, slots1 * kv_stride);
    if (!ks0_ok || !vs0_ok || !ks1_ok || !vs1_ok) {
      std::printf("    dspark_ctx_kv_write detail: ks0=%d vs0=%d ks1=%d vs1=%d\n",
                  ks0_ok, vs0_ok, ks1_ok, vs1_ok);
    }
    Check(ks0_ok && vs0_ok && ks1_ok && vs1_ok, "dspark_ctx_kv_write bf16+fp8");
  }

  // 5. strided attn_fused + sigmoid_gate_mul_strided + sconv_*_strided_hist
  {
    const int T = 3, qh = 4, kvh = 2, hd = 128, row_n = qh * hd, big_stride = row_n + 640;
    const int block_size = 16, n_blocks = 4, ctx = 12;
    const std::vector<uint16_t> big = ToBf16(Rand(T * big_stride, 0.5f));
    std::vector<uint16_t> q_contig(T * row_n), g_contig(T * row_n);
    for (int t = 0; t < T; ++t) {
      std::memcpy(q_contig.data() + t * row_n, big.data() + t * big_stride, row_n * 2);
      std::memcpy(g_contig.data() + t * row_n, big.data() + t * big_stride + row_n + 128, row_n * 2);
    }
    const std::vector<uint16_t> kv_pool = ToBf16(Rand(n_blocks * kvh * block_size * 2 * hd, 0.3f));
    const std::vector<int32_t> btbl = {2, 0};
    DeviceBuffer<uint16_t> dbig = Dev(big), dq = Dev(q_contig), dg = Dev(g_contig), dkv = Dev(kv_pool);
    DeviceBuffer<int32_t> dbt = Dev(btbl);
    DeviceBuffer<uint16_t> dout_strided(T * row_n), dout_contig(T * row_n);
    DeviceBuffer<float> dws(T * qh * 32 * (hd + 2));

    for (int qlen : {1, T}) {
      R4dxInkling2AttnFusedArgs a{};
      a.q = P(dbig.data());
      a.gate = P(dbig.data() + row_n + 128);
      a.q_row_stride = big_stride;
      a.gate_row_stride = big_stride;
      a.kv = P(dkv.data());
      a.block_table = P(dbt.data());
      a.out = P(dout_strided.data());
      a.workspace = P(dws.data());
      a.q_len = qlen;
      a.q_heads = qh;
      a.kv_heads = kvh;
      a.head_dim = hd;
      a.block_size = block_size;
      a.ctx = ctx;
      a.max_splits = 32;
      a.scale = 1.0f / std::sqrt(static_cast<float>(hd));
      a.kv_block_stride = static_cast<int64_t>(kvh) * block_size * (2 * hd);
      a.kv_head_stride = static_cast<int64_t>(block_size) * (2 * hd);
      r4dx_inkling2_attn_fused_bf16kv(&a, 0);

      R4dxInkling2AttnFusedArgs b = a;
      b.q = P(dq.data());
      b.gate = P(dg.data());
      b.q_row_stride = 0;
      b.gate_row_stride = 0;
      b.out = P(dout_contig.data());
      r4dx_inkling2_attn_fused_bf16kv(&b, 0);
      Check(Host(dout_strided, qlen * row_n) == Host(dout_contig, qlen * row_n),
            "attn_fused strided bit-exact q_len=" + std::to_string(qlen));
    }

    // sigmoid_gate_mul_strided_bf16
    DeviceBuffer<uint16_t> dsg_got(T * row_n), dsg_ref(T * row_n);
    r4dx_inkling2_sigmoid_gate_mul_strided_bf16(P(dq.data()), P(dbig.data() + row_n + 128), big_stride,
                                                P(dsg_got.data()), T, row_n, 0);
    r4dx_inkling2_sigmoid_gate_mul_bf16(P(dq.data()), P(dg.data()), P(dsg_ref.data()), T * row_n, 0);
    Check(Host(dsg_got, T * row_n) == Host(dsg_ref, T * row_n), "sigmoid_gate_mul_strided_bf16 bit-exact");

    // sconv_static_pair_strided_hist_f32 & sconv_dynamic_strided_hist_f32
    const int C = 128, sc_stride = C + 256;
    const std::vector<uint16_t> sc_big = ToBf16(Rand(T * sc_stride, 0.5f));
    std::vector<uint16_t> xk_sep(T * C), xv_sep(T * C);
    for (int t = 0; t < T; ++t) {
      std::memcpy(xk_sep.data() + t * C, sc_big.data() + t * sc_stride, C * 2);
      std::memcpy(xv_sep.data() + t * C, sc_big.data() + t * sc_stride + C, C * 2);
    }
    const std::vector<float> wk = Rand(4 * C, 0.3f), wv = Rand(4 * C, 0.3f), st0 = Rand(3 * C, 0.2f);
    DeviceBuffer<uint16_t> dsc_big = Dev(sc_big), dxk = Dev(xk_sep), dxv = Dev(xv_sep);
    DeviceBuffer<float> dwk = Dev(wk), dwv = Dev(wv);
    DeviceBuffer<float> dsk_got = Dev(st0), dsv_got = Dev(st0), dsk_ref = Dev(st0), dsv_ref = Dev(st0);
    DeviceBuffer<float> dok_got(T * C), dov_got(T * C), dok_ref(T * C), dov_ref(T * C);
    DeviceBuffer<float> dhk_got(T * C), dhv_got(T * C), dhk_ref(T * C), dhv_ref(T * C);
    r4dx_inkling2_sconv_static_pair_strided_hist_f32(
        P(dsc_big.data()), sc_stride, P(dsc_big.data() + C), sc_stride, 1, P(dwk.data()), P(dwv.data()),
        P(dsk_got.data()), P(dsv_got.data()), P(dok_got.data()), P(dov_got.data()), P(dhk_got.data()),
        P(dhv_got.data()), T, C, 0);
    r4dx_inkling2_sconv_static_hist_f32(P(dxk.data()), 1, P(dwk.data()), P(dsk_ref.data()), 0, P(dok_ref.data()),
                                        P(dhk_ref.data()), T, C, 0);
    r4dx_inkling2_sconv_static_hist_f32(P(dxv.data()), 1, P(dwv.data()), P(dsv_ref.data()), 0, P(dov_ref.data()),
                                        P(dhv_ref.data()), T, C, 0);
    Check(Host(dok_got, T * C) == Host(dok_ref, T * C) && Host(dov_got, T * C) == Host(dov_ref, T * C) &&
              Host(dhk_got, T * C) == Host(dhk_ref, T * C) && Host(dhv_got, T * C) == Host(dhv_ref, T * C),
          "sconv_static_pair_strided_hist_f32 bit-exact");

    const int heads = 4;
    const std::vector<float> dyn = Rand(T * heads * 4, 0.2f);
    DeviceBuffer<float> ddyn = Dev(dyn), dsd_got = Dev(st0), dsd_ref = Dev(st0);
    DeviceBuffer<float> dod_got(T * C), dod_ref(T * C), dhd_got(T * C), dhd_ref(T * C);
    r4dx_inkling2_sconv_dynamic_strided_hist_f32(P(dsc_big.data()), sc_stride, 1, P(ddyn.data()), 0, P(dwk.data()),
                                                 P(dsd_got.data()), P(dod_got.data()), P(dhd_got.data()), T, C, heads,
                                                 0);
    r4dx_inkling2_sconv_dynamic_hist_f32(P(dxk.data()), 1, P(ddyn.data()), 0, P(dwk.data()), P(dsd_ref.data()),
                                         P(dod_ref.data()), P(dhd_ref.data()), T, C, heads, 0);
    Check(Host(dod_got, T * C) == Host(dod_ref, T * C) && Host(dhd_got, T * C) == Host(dhd_ref, T * C),
          "sconv_dynamic_strided_hist_f32 bit-exact");
  }

  // 6. msg_gate_apply_bf16w_f32 & msg_gate_apply3_bf16w_f32
  {
    const int T = 5, H = 128, R = 64, x_stride = (T + 2) * H;
    std::vector<float> x_strided(3 * x_stride, 0.0f);
    std::array<std::vector<float>, 3> xg_f;
    std::vector<float> m_all(3 * T * H), g_all(3 * T * R);
    m_all = Rand(3 * T * H, 0.5f);
    g_all = Rand(3 * T * R, 0.5f);
    std::array<std::vector<uint16_t>, 3> fg_bf, gn_bf;
    std::array<std::vector<float>, 3> fg_f, gn_f;
    for (int gate = 0; gate < 3; ++gate) {
      xg_f[gate] = Rand(T * H, 0.5f);
      std::memcpy(x_strided.data() + gate * x_stride, xg_f[gate].data(), T * H * sizeof(float));
      fg_bf[gate] = ToBf16(Rand(H * R, 0.2f));
      gn_bf[gate] = ToBf16(Rand(H * R, 0.2f));
      fg_f[gate].resize(H * R);
      gn_f[gate].resize(H * R);
      for (int i = 0; i < H * R; ++i) {
        fg_f[gate][i] = Bf16ToFloat(fg_bf[gate][i]);
        gn_f[gate][i] = Bf16ToFloat(gn_bf[gate][i]);
      }
    }
    DeviceBuffer<float> dx_s = Dev(x_strided), dm = Dev(m_all), dg = Dev(g_all), dout3(3 * T * H);
    std::array<DeviceBuffer<uint16_t>, 3> dfg_bf{Dev(fg_bf[0]), Dev(fg_bf[1]), Dev(fg_bf[2])};
    std::array<DeviceBuffer<uint16_t>, 3> dgn_bf{Dev(gn_bf[0]), Dev(gn_bf[1]), Dev(gn_bf[2])};
    r4dx_inkling2_msg_gate_apply3_bf16w_f32(
        P(dx_s.data()), x_stride, P(dm.data()), P(dg.data()), P(dfg_bf[0].data()), P(dfg_bf[1].data()),
        P(dfg_bf[2].data()), P(dgn_bf[0].data()), P(dgn_bf[1].data()), P(dgn_bf[2].data()), P(dout3.data()), T, H, R,
        0);
    const std::vector<float> got3 = Host(dout3, 3 * T * H);

    // Compare single apply_bf16w and apply3_bf16w against existing fp32 msg_gate_apply_f32
    bool msg_ok = true;
    for (int gate = 0; gate < 3; ++gate) {
      DeviceBuffer<float> dxg = Dev(xg_f[gate]);
      DeviceBuffer<float> dfg = Dev(fg_f[gate]), dgn = Dev(gn_f[gate]);
      DeviceBuffer<float> dout1(T * H), dout_ref(T * H);
      r4dx_inkling2_msg_gate_apply_bf16w_f32(P(dxg.data()), P(dm.data() + gate * T * H), P(dg.data() + gate * T * R),
                                             0, P(dfg_bf[gate].data()), P(dgn_bf[gate].data()), P(dout1.data()), T, H,
                                             R, 0);
      r4dx_inkling2_msg_gate_apply_f32(P(dxg.data()), P(dm.data() + gate * T * H), P(dg.data() + gate * T * R), 0,
                                       P(dfg.data()), P(dgn.data()), P(dout_ref.data()), T, H, R, 0);
      const std::vector<float> ref_v = Host(dout_ref, T * H);
      const std::vector<float> got1 = Host(dout1, T * H);
      const std::vector<float> g3_slice(got3.begin() + gate * T * H, got3.begin() + (gate + 1) * T * H);
      const double rel = RelErr(got1, ref_v, 1e-2);
      const bool eq13 = (got1 == g3_slice);
      if (rel >= 1e-4 || !eq13) {
        std::printf("    msg_gate_apply gate=%d rel=%.6g eq13=%d\n", gate, rel, eq13);
        msg_ok = false;
      }
    }
    Check(msg_ok, "msg_gate_apply{_bf16w,3_bf16w}_f32");
  }

  // 7. router_fused_f32 vs router_f32
  {
    const int T = 4, H = 256, E = 128, top_k = 8, stride = 320;
    const float route_scale = 1.75f;
    const std::vector<uint16_t> x = ToBf16(Rand(T * stride, 0.5f));
    const std::vector<uint16_t> gw = ToBf16(Rand(E * H, 0.3f));
    const std::vector<float> bias = Rand(E, 0.1f);
    DeviceBuffer<uint16_t> dx = Dev(x), dgw = Dev(gw);
    DeviceBuffer<float> dbias = Dev(bias);
    DeviceBuffer<int32_t> dids_got(T * top_k), dids_ref(T * top_k);
    DeviceBuffer<float> dw_got(T * top_k), dw_ref(T * top_k), dlog_got(T * E), dlog_ref(T * E);
    r4dx_inkling2_router_fused_f32(P(dx.data()), P(dgw.data()), P(dbias.data()), P(dids_got.data()), P(dw_got.data()),
                                   T, H, E, top_k, route_scale, stride, P(dlog_got.data()), 0);
    r4dx_inkling2_router_f32(P(dx.data()), P(dgw.data()), P(dbias.data()), P(dids_ref.data()), P(dw_ref.data()), T, H, E,
                             top_k, route_scale, stride, P(dlog_ref.data()), 0);
    Check(Host(dids_got, T * top_k) == Host(dids_ref, T * top_k) &&
              RelErr(Host(dw_got, T * top_k), Host(dw_ref, T * top_k), 1e-4) < 1e-4 &&
              RelErr(Host(dlog_got, T * E), Host(dlog_ref, T * E), 1e-4) < 1e-4,
          "router_fused_f32");
  }

  // 8. dense_sink_kvconv_bf16
  {
    const int M = 4, N_sink = 32, N_kv = 16, K = 256;
    const std::vector<uint16_t> xq = ToBf16(Rand(M * K, 0.4f)), xk = ToBf16(Rand(M * K, 0.4f)),
                                xv = ToBf16(Rand(M * K, 0.4f));
    const std::vector<uint16_t> ws = ToBf16(Rand(N_sink * K, 0.3f)), wk = ToBf16(Rand(N_kv * K, 0.3f)),
                                wv = ToBf16(Rand(N_kv * K, 0.3f));
    DeviceBuffer<uint16_t> dxq = Dev(xq), dxk = Dev(xk), dxv = Dev(xv), dws = Dev(ws), dwk = Dev(wk), dwv = Dev(wv);
    DeviceBuffer<float> dos(M * N_sink), dok(M * N_kv), dov(M * N_kv);
    r4dx_inkling2_dense_sink_kvconv_bf16(P(dxq.data()), P(dxk.data()), P(dxv.data()), P(dws.data()), P(dwk.data()),
                                         P(dwv.data()), P(dos.data()), P(dok.data()), P(dov.data()), M, N_sink, N_kv,
                                         K, 0);
    auto cpu_mm = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int N) {
      std::vector<float> out(M * N, 0.0f);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          float s = 0.0f;
          for (int k = 0; k < K; ++k) s += Bf16ToFloat(a[m * K + k]) * Bf16ToFloat(b[n * K + k]);
          out[m * N + n] = s;
        }
      return out;
    };
    Check(RelErr(Host(dos, M * N_sink), cpu_mm(xq, ws, N_sink), 1e-3) < 1e-4 &&
              RelErr(Host(dok, M * N_kv), cpu_mm(xk, wk, N_kv), 1e-3) < 1e-4 &&
              RelErr(Host(dov, M * N_kv), cpu_mm(xv, wv, N_kv), 1e-3) < 1e-4,
          "dense_sink_kvconv_bf16");
  }

  // 9. quantize_pack_mxfp8_bf16 bit-exact vs r4dx_convert::QuantizeMxfp8 + PackMxfp8*
  {
    const int N = 64, K = 256;
    const std::vector<uint16_t> src_bf = ToBf16(Rand(N * K, 2.0f));
    std::vector<float> src_f(N * K);
    for (int i = 0; i < N * K; ++i) src_f[i] = Bf16ToFloat(src_bf[i]);
    std::vector<uint8_t> codes, scales;
    r4dx_convert::QuantizeMxfp8(src_f.data(), N, K, 1, codes, scales, "test_mxfp8");
    const std::vector<uint8_t> wq_ref = r4dx_convert::PackMxfp8Codes(codes, N, K, 1);
    const std::vector<uint8_t> ws_ref = r4dx_convert::PackMxfp8Scales(scales, N, K);

    DeviceBuffer<uint16_t> d_src = Dev(src_bf);
    DeviceBuffer<uint8_t> d_wq(N * K), d_ws(N * K / 32);
    r4dx_inkling2_quantize_pack_mxfp8_bf16(P(d_src.data()), P(d_wq.data()), P(d_ws.data()), N, K, 0);
    Check(Host(d_wq, N * K) == wq_ref && Host(d_ws, N * K / 32) == ws_ref,
          "quantize_pack_mxfp8_bf16 bit-exact");
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  TestAttnRes();
  for (bool dyn : {false, true})
    for (bool xb : {true, false})
      for (bool res : {false, true}) {
        if (dyn && res) continue;
        TestSconv(dyn, xb, res);
      }
  TestHead();
  TestRelGateMsg();
  TestRouterSwigluSoftcapDense();
  TestAttention();
  TestFusedAttention();
  TestDsparkKernels();
  TestSpeedupKernels();
  if (g_fail != 0) {
    std::printf("%d check(s) FAILED\n", g_fail);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
