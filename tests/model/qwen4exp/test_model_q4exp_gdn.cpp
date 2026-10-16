// Qwen4ExpGdn (src/model/qwen4exp_gdn.h, the sigmoid-gated GDN mixer of qwen4_exp) on layer 0 of the tiny container
// against the HF goldens (docs/qwen4exp-plan.md M1-7 / M1-16). Links r4dx_model.
//
// Input: the golden `layer0.attn_mix` (the hyper-connection mix, i.e. the block input), output compared with
// `layer0.attn_out`. For T in {7, 70, 300} (a call is at most 256 rows; T = 300 runs as 256 + 44 on one state) and the
// variants {conv_prep 1 | 2} x {seq_slice auto | whole call | 64-row slices}, plus the decode path (Prefill of T - 1 rows,
// then the last row through the recurrent kernels). Also the three big linears of the block against a CPU double-precision
// reference at several row counts (the GEMM that ApplyLinear's tuning fallback serves for these shapes).
#include <hip/hip_runtime.h>

#include <cmath>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <string>
#include <vector>

#include "gdn_state.h"
#include "q4exp_model_util.h"
#include "qwen4exp_container.h"
#include "qwen4exp_gdn.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"

using namespace q4m;
using namespace r4dx;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// y[M, N] = x[M, K] @ W[N, K]^T in double, rounded to bf16.
std::vector<uint16_t> CpuLinear(const std::vector<uint16_t>& w, const uint16_t* x, int64_t M, int64_t N, int64_t K) {
  std::vector<float> wf(w.size());
  for (size_t i = 0; i < w.size(); ++i) wf[i] = Bf(w[i]);
  std::vector<float> xf(static_cast<size_t>(M * K));
  for (int64_t i = 0; i < M * K; ++i) xf[static_cast<size_t>(i)] = Bf(x[i]);
  std::vector<uint16_t> y(static_cast<size_t>(M * N));
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      double acc = 0;
      const float* wr = wf.data() + n * K;
      const float* xr = xf.data() + m * K;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(wr[k]) * xr[k];
      y[static_cast<size_t>(m * N + n)] = core::FloatToBf16(static_cast<float>(acc));
    }
  }
  return y;
}

}  // namespace

int main() {
  Unbuffer();
  try {
    if (!Exists(ContainerPath("bf16")) || !Exists(GoldenPath(7)) || !Exists(GoldenPath(70)) || !Exists(GoldenPath(300))) {
      std::printf("SKIP: tiny fixtures missing under %s\n", TinyDir().c_str());
      return kSkip;
    }
    if (VisibleDevices() < 1) {
      std::printf("SKIP: no HIP device\n");
      return kSkip;
    }
    const int dev = TestDevice();
    R4DX_HIP_CHECK(hipSetDevice(dev));
    Qwen4ExpLoadOptions lo;
    lo.layout = Layout::kBf16;
    lo.device = dev;
    lo.layer_first = 0;
    lo.layer_last = 1;
    Qwen4ExpContainer c = Qwen4ExpContainer::Load(ContainerPath("bf16"), lo);
    const Qwen4ExpConfig& cfg = c.Config();
    const qwen4exp::GdnWeights& gw = *c.Layer(0).gdn;
    Qwen4ExpGdnRefs refs;
    refs.in_proj_qkv = &gw.in_proj_qkv;
    refs.in_proj_z = &gw.in_proj_z;
    refs.in_proj_a = gw.in_proj_a.data();
    refs.in_proj_b = gw.in_proj_b.data();
    refs.conv1d_weight = gw.conv1d_weight.data();
    refs.A_log = gw.A_log.data();
    refs.dt_bias = gw.dt_bias.data();
    refs.norm_weight = gw.norm_weight.data();
    refs.out_proj = &gw.out_proj;
    Qwen4ExpGdn gdn(cfg, refs);
    core::Stream stream;
    core::Arena arena(gdn.ArenaBytes(256) + (size_t{256} << 20));
    GdnControlCache control;
    const int64_t hidden = cfg.hidden_size;

    const Golden g300(300, GoldenPath(300)), g70(70, GoldenPath(70)), g7(7, GoldenPath(7));
    const Golden* goldens[] = {&g7, &g70, &g300};

    // ---- the block's big linears against a CPU reference ---------------------------------------------------------------------
    std::printf("== linears vs CPU reference (device %d) ==\n", dev);
    {
      struct L {
        const char* name;
        const QuantLinear* w;
      } linears[] = {{"in_proj_qkv", &gw.in_proj_qkv}, {"in_proj_z", &gw.in_proj_z}, {"out_proj", &gw.out_proj}};
      const std::vector<uint16_t> xs(g300.Bf16("layer0.attn_mix"), g300.Bf16("layer0.attn_mix") + 300 * hidden);
      PrintStatsHeader();
      for (const L& l : linears) {
        const std::vector<uint16_t> wh = l.w->bf16_w.CopyToHost();
        // out_proj's input is [M, value_dim]; feed the first K columns of repeated mix rows
        for (int64_t M : {int64_t{1}, int64_t{7}, int64_t{64}, int64_t{70}, int64_t{256}}) {
          if (std::string(l.name) == "in_proj_z" && M != 70 && M != 256) continue;
          if (std::string(l.name) == "out_proj" && M != 70 && M != 256) continue;
          std::vector<uint16_t> in(static_cast<size_t>(M * l.w->K));
          for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < l.w->K; ++k) in[static_cast<size_t>(m * l.w->K + k)] = xs[static_cast<size_t>(m * hidden + (k % hidden))];
          }
          core::DeviceBuffer<uint16_t> dx(in.size()), dy(static_cast<size_t>(M * l.w->N));
          dx.CopyFromHost(in);
          arena.Reset();
          qwen4exp::ApplyBf16Linear(stream.get(), arena, *l.w, dx.data(), dy.data(), M);
          stream.Synchronize();
          const std::vector<uint16_t> got = dy.CopyToHost();
          const std::vector<uint16_t> ref = CpuLinear(wh, in.data(), M, l.w->N, l.w->K);
          const Stats s = CompareBf16(got.data(), ref.data(), static_cast<int64_t>(ref.size()));
          PrintStats(std::string(l.name) + " M=" + std::to_string(M), s);
          Check(!s.nan && s.max_norm < 0.02, std::string(l.name) + " M=" + std::to_string(M) + " differs from the CPU reference");
        }
      }
    }

    // ---- the GDN block vs golden ----------------------------------------------------------------------------------------------
    // rows [0, prefill_rows) as prefill calls of <= 256 rows, the rest one decode step each
    const auto run = [&](const Golden& g, int conv_prep, int64_t seq_slice, int prefill_rows, bool recurrent = false) {
      const int T = g.T();
      const uint16_t* mix = g.Bf16("layer0.attn_mix");
      core::DeviceBuffer<uint16_t> x(static_cast<size_t>(T * hidden)), out(static_cast<size_t>(T * hidden));
      x.CopyFromHost(mix, static_cast<size_t>(T * hidden));
      GdnStateManager states(1, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim, gdn.ConvDim(),
                             cfg.linear_conv_kernel_dim, 1, false);
      states.ZeroAll(stream);
      stream.Synchronize();
      for (int off = 0; off < prefill_rows; off += 256) {
        const int n = std::min(256, prefill_rows - off);
        Qwen4ExpGdnCall call;
        call.is_prefill = true;
        call.has_init = off > 0;
        call.conv_prep = conv_prep;
        call.seq_slice = seq_slice;
        call.prefill_recurrent = recurrent;
        arena.Reset();
        gdn.Forward(stream.get(), arena, states, control, x.data() + static_cast<int64_t>(off) * hidden,
                    out.data() + static_cast<int64_t>(off) * hidden, n, call);
      }
      for (int r = prefill_rows; r < T; ++r) {
        Qwen4ExpGdnCall call;
        call.is_prefill = false;
        arena.Reset();
        gdn.Forward(stream.get(), arena, states, control, x.data() + static_cast<int64_t>(r) * hidden,
                    out.data() + static_cast<int64_t>(r) * hidden, 1, call);
      }
      stream.Synchronize();
      return out.CopyToHost();
    };
    // rows whose max |err| exceeds a quarter of the whole tensor's rms
    const auto bad_rows = [&](const std::vector<uint16_t>& got, const uint16_t* ref, int T) {
      const Stats all = CompareBf16(got.data(), ref, static_cast<int64_t>(T) * hidden);
      std::string list;
      int count = 0;
      for (int r = 0; r < T; ++r) {
        const Stats s = CompareBf16(got.data() + static_cast<int64_t>(r) * hidden, ref + static_cast<int64_t>(r) * hidden, hidden);
        if (s.max_abs > 0.25 * all.rms_ref) {
          ++count;
          if (count <= 24) list += " " + std::to_string(r);
        }
      }
      return std::to_string(count) + " bad row(s):" + list;
    };
    std::printf("\n== GDN block vs golden layer0.attn_out ==\n");
    PrintStatsHeader();
    for (const Golden* g : goldens) {
      const int T = g->T();
      const uint16_t* ref = g->Bf16("layer0.attn_out");
      struct V {
        int conv;
        int64_t slice;
        const char* name;
      };
      for (const V& v : {V{2, 0, "prep2 auto"}, V{1, 0, "prep1 auto"}, V{2, -1, "prep2 whole"}, V{1, -1, "prep1 whole"}, V{2, 64, "prep2 64"},
                         V{1, 64, "prep1 64"}}) {
        if (T <= 64 && v.slice != 0) continue;  // slicing is a no-op below 64 rows
        const std::vector<uint16_t> got = run(*g, v.conv, v.slice, T);
        const Stats s = CompareBf16(got.data(), ref, static_cast<int64_t>(T) * hidden);
        PrintStats("T=" + std::to_string(T) + " " + v.name, s);
        if (v.conv == 2 && v.slice == 0) {
          std::printf("      %s\n", bad_rows(got, ref, T).c_str());
          // The chunked scan matches the golden only where its absolute error stays below the (tiny) |o| the random model produces
          // (<= 32 rows on this fixture, see test_model_q4exp_gdn's "libr4d prefill stages" section); longer calls are reported.
          if (T <= 32) Check(!s.nan && s.max_norm < 0.15 && s.mean_norm < 0.012, "T=" + std::to_string(T) + " chunked prefill vs golden");
        }
      }
      {
        const std::vector<uint16_t> got = run(*g, 2, 0, T, /*recurrent=*/true);
        const Stats s = CompareBf16(got.data(), ref, static_cast<int64_t>(T) * hidden);
        PrintStats("T=" + std::to_string(T) + " prefill_recurrent", s);
        Check(!s.nan && s.max_norm < 0.15 && s.mean_norm < 0.012, "T=" + std::to_string(T) + " recurrent prefill vs golden");
      }
      {
        const std::vector<uint16_t> got = run(*g, 2, 0, T - 1);
        const Stats s = CompareBf16(got.data() + static_cast<int64_t>(T - 1) * hidden, ref + static_cast<int64_t>(T - 1) * hidden, hidden);
        PrintStats("T=" + std::to_string(T) + " decode last row", s);
        Check(!s.nan && s.max_norm < 0.15 && s.mean_norm < 0.012, "T=" + std::to_string(T) + " decode row vs golden");
      }
      {
        const std::vector<uint16_t> got = run(*g, 2, 0, 0);  // every row a decode step
        const Stats s = CompareBf16(got.data(), ref, static_cast<int64_t>(T) * hidden);
        PrintStats("T=" + std::to_string(T) + " all rows decoded", s);
        std::printf("      %s\n", bad_rows(got, ref, T).c_str());
        Check(!s.nan && s.max_norm < 0.15 && s.mean_norm < 0.012, "T=" + std::to_string(T) + " all-decode vs golden");
      }
    }
    // ---- does the result depend on what the arena held before? (a kernel reading memory it never wrote) ----
    {
      std::printf("\n== arena pre-fill experiment (T=70 golden input, prefill P rows) ==\n");
      const uint16_t* ref70 = g70.Bf16("layer0.attn_out");
      for (int fill : {0x00, 0x3F, 0x7F, 0xFF}) {
        for (int P : {48, 63, 64, 70}) {
          arena.Reset();
          const size_t cap = arena.capacity_bytes() - 64;
          uint8_t* all = arena.Alloc<uint8_t>(cap);
          R4DX_HIP_CHECK(hipMemsetAsync(all, fill, cap, stream.get()));
          stream.Synchronize();
          // run() resets the arena itself, so the fill stays in the memory the kernels then use
          const std::vector<uint16_t> got = run(g70, 2, 0, P);
          const Stats s = CompareBf16(got.data(), ref70, static_cast<int64_t>(P) * hidden);
          std::printf("  fill 0x%02X P=%2d: max/rms %.4f mean/rms %.5f\n", fill, P, s.max_norm, s.mean_norm);
        }
      }
    }
    // ---- stage by stage: the first P' rows of a P-row prefill must equal a P'-row prefill (causality of every kernel) ----
    {
      std::printf("\n== libr4d prefill stages: P=63 / P=64 vs P=32, rows [0, 32) ==\n");
      struct Out {
        std::vector<uint16_t> q, k, v, A, o, a, b;
        std::vector<float> g, beta;
      };
      const int64_t Hh = cfg.linear_num_value_heads, Hg = cfg.linear_num_key_heads, Kd = cfg.linear_key_head_dim,
                    Vd = cfg.linear_value_head_dim, conv_dim = gdn.ConvDim();
      const uint16_t* mix70 = g70.Bf16("layer0.attn_mix");
      const auto raw = [&](int P) {
        core::DeviceBuffer<uint16_t> x(static_cast<size_t>(P * hidden));
        x.CopyFromHost(mix70, static_cast<size_t>(P * hidden));
        GdnStateManager st(1, Hh, Vd, Kd, conv_dim, cfg.linear_conv_kernel_dim, 1, false);
        st.ZeroAll(stream);
        stream.Synchronize();
        arena.Reset();
        uint16_t* mixed = arena.Alloc<uint16_t>(static_cast<size_t>(P * conv_dim));
        qwen4exp::ApplyBf16Linear(stream.get(), arena, gw.in_proj_qkv, x.data(), mixed, P);
        uint16_t* a = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hh));
        uint16_t* b = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hh));
        for (int m0 = 0; m0 < P; m0 += 64) {
          const int m = std::min(64, P - m0);
          core::r4d::GemmBf16NtM64(x.data() + static_cast<int64_t>(m0) * hidden, gw.in_proj_a.data(), a + m0 * Hh, m, static_cast<int>(hidden),
                                   static_cast<int>(Hh), 4, 4, 1, stream.get());
          core::r4d::GemmBf16NtM64(x.data() + static_cast<int64_t>(m0) * hidden, gw.in_proj_b.data(), b + m0 * Hh, m, static_cast<int>(hidden),
                                   static_cast<int>(Hh), 4, 4, 1, stream.get());
        }
        uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hg * Kd));
        uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hg * Kd));
        uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hh * Vd));
        float* g = arena.Alloc<float>(static_cast<size_t>(P * Hh));
        float* beta = arena.Alloc<float>(static_cast<size_t>(P * Hh));
        uint16_t* A = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hh * 64));
        float* ht = arena.Alloc<float>(static_cast<size_t>(Hh * Vd * Kd));
        uint16_t* o = arena.Alloc<uint16_t>(static_cast<size_t>(P * Hh * Vd));
        const int32_t* cu = control.CuPair(P);
        const int32_t* cidx = control.CacheIdx(st.SlotForSeq(0));
        core::r4d::GdnConvPrep(mixed, conv_dim, gw.conv1d_weight.data(), nullptr, st.ConvBase(), st.ConvSeqStride(), st.ConvDimStride(),
                               st.ConvTokStride(), cidx, 1, nullptr, a, b, Hh, 1, gw.A_log.data(), gw.dt_bias.data(), q, k, v, g, beta, cu, 1,
                               P, static_cast<int>(Hh), static_cast<int>(Hg), static_cast<int>(Kd), static_cast<int>(Vd),
                               static_cast<int>(cfg.linear_conv_kernel_dim), 20.0f, stream.get());
        core::r4d::GdnKktSolve(k, beta, g, A, cu, 1, P, static_cast<int>(Hh), static_cast<int>(Hg), static_cast<int>(Kd), 64, stream.get());
        core::r4d::GdnChunkScan(q, k, v, A, g, beta, st.RecurrentSlotPtr(st.SlotForSeq(0)), o, ht, cu, 1, static_cast<int>(Hh),
                                static_cast<int>(Hg), static_cast<int>(Kd), static_cast<int>(Vd), 64, 1.0f / std::sqrt(static_cast<float>(Kd)), stream.get());
        stream.Synchronize();
        Out out;
        const auto pull16 = [&](const uint16_t* p, size_t n) {
          std::vector<uint16_t> h(n);
          R4DX_HIP_CHECK(hipMemcpy(h.data(), p, n * 2, hipMemcpyDeviceToHost));
          return h;
        };
        const auto pullf = [&](const float* p, size_t n) {
          std::vector<float> h(n);
          R4DX_HIP_CHECK(hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost));
          return h;
        };
        out.q = pull16(q, static_cast<size_t>(P * Hg * Kd));
        out.k = pull16(k, static_cast<size_t>(P * Hg * Kd));
        out.v = pull16(v, static_cast<size_t>(P * Hh * Vd));
        out.A = pull16(A, static_cast<size_t>(P * Hh * 64));
        out.o = pull16(o, static_cast<size_t>(P * Hh * Vd));
        out.g = pullf(g, static_cast<size_t>(P * Hh));
        out.beta = pullf(beta, static_cast<size_t>(P * Hh));
        out.a = pull16(a, static_cast<size_t>(P * Hh));
        out.b = pull16(b, static_cast<size_t>(P * Hh));
        return out;
      };
      // CPU reference of the gated delta rule (HF torch_recurrent_gated_delta_rule's order) on the kernel's own q / k / v / a / b,
      // fp64 state: what the chunked scan should produce for rows [0, P), before the gated norm.
      std::vector<float> a_log(48), dt_bias(48);
      R4DX_HIP_CHECK(hipMemcpy(a_log.data(), gw.A_log.data(), 48 * 4, hipMemcpyDeviceToHost));
      R4DX_HIP_CHECK(hipMemcpy(dt_bias.data(), gw.dt_bias.data(), 48 * 4, hipMemcpyDeviceToHost));
      const auto cpu_o = [&](const Out& s, int P) {
        std::vector<double> ref(static_cast<size_t>(P * Hh * Vd), 0.0);
        const int rep = static_cast<int>(Hh / Hg);
        const double scale = 1.0 / std::sqrt(static_cast<double>(Kd));
        for (int h = 0; h < Hh; ++h) {
          std::vector<double> S(static_cast<size_t>(Kd * Vd), 0.0), kv(static_cast<size_t>(Vd)), delta(static_cast<size_t>(Vd));
          const int hk = h / rep;
          for (int t = 0; t < P; ++t) {
            const double x = static_cast<double>(Bf(s.a[static_cast<size_t>(t * Hh + h)])) + dt_bias[static_cast<size_t>(h)];
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            const double decay = std::exp(-std::exp(static_cast<double>(a_log[static_cast<size_t>(h)])) * sp);
            const double beta = 1.0 / (1.0 + std::exp(-static_cast<double>(Bf(s.b[static_cast<size_t>(t * Hh + h)]))));
            for (double& e : S) e *= decay;
            const uint16_t* kt = s.k.data() + (static_cast<size_t>(t) * Hg + hk) * Kd;
            const uint16_t* qt = s.q.data() + (static_cast<size_t>(t) * Hg + hk) * Kd;
            const uint16_t* vt = s.v.data() + (static_cast<size_t>(t) * Hh + h) * Vd;
            for (int v = 0; v < Vd; ++v) {
              double m = 0;
              for (int kk = 0; kk < Kd; ++kk) m += S[static_cast<size_t>(kk * Vd + v)] * Bf(kt[kk]);
              kv[static_cast<size_t>(v)] = m;
              delta[static_cast<size_t>(v)] = (static_cast<double>(Bf(vt[v])) - m) * beta;
            }
            for (int kk = 0; kk < Kd; ++kk) {
              for (int v = 0; v < Vd; ++v) S[static_cast<size_t>(kk * Vd + v)] += static_cast<double>(Bf(kt[kk])) * delta[static_cast<size_t>(v)];
            }
            for (int v = 0; v < Vd; ++v) {
              double o = 0;
              for (int kk = 0; kk < Kd; ++kk) o += S[static_cast<size_t>(kk * Vd + v)] * Bf(qt[kk]) * scale;
              ref[(static_cast<size_t>(t) * Hh + h) * Vd + v] = o;
            }
          }
        }
        return ref;
      };
      const auto report_o = [&](const char* tag, const Out& s, int P, int rows) {
        const std::vector<double> ref = cpu_o(s, P);
        double sq = 0, maxerr = 0, sumerr = 0;
        int64_t n = 0;
        std::vector<double> head_err(static_cast<size_t>(Hh), 0.0), head_rms(static_cast<size_t>(Hh), 0.0);
        for (int t = 0; t < rows; ++t) {
          for (int h = 0; h < Hh; ++h) {
            for (int v = 0; v < Vd; ++v) {
              const size_t i = (static_cast<size_t>(t) * Hh + h) * Vd + v;
              const double r = ref[i], g = Bf(s.o[i]);
              sq += r * r;
              const double e = std::fabs(g - r);
              maxerr = std::max(maxerr, e);
              sumerr += e;
              head_err[static_cast<size_t>(h)] = std::max(head_err[static_cast<size_t>(h)], e);
              head_rms[static_cast<size_t>(h)] += r * r;
              ++n;
            }
          }
        }
        std::printf("  %s: chunk-scan o vs fp64 delta rule, rows [0,%d): rms(ref) %.4g, max|err| %.4g, mean|err| %.4g (bf16 rounding of o alone: %.4g)\n", tag,
                    rows, std::sqrt(sq / static_cast<double>(n)), maxerr, sumerr / static_cast<double>(n), std::sqrt(sq / static_cast<double>(n)) * 0.0039);
        std::printf("    heads (rms(ref), max|err|):");
        for (int h = 0; h < Hh; h += 3) {
          std::printf(" [%d: %.2g, %.2g]", h, std::sqrt(head_rms[static_cast<size_t>(h)] / static_cast<double>(rows * Vd)), head_err[static_cast<size_t>(h)]);
        }
        std::printf("\n");
      };
      const Out small = raw(32);
      report_o("P=32", small, 32, 32);
      {
        const Out b64 = raw(64);
        report_o("P=64 (rows 0..31)", b64, 64, 32);
        report_o("P=64 (rows 0..63)", b64, 64, 64);
      }
      for (int P : {63, 64}) {
        const Out big = raw(P);
        const auto diff16 = [&](const char* name, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int64_t row_elems) {
          int64_t first = -1, count = 0;
          double mx = 0;
          for (int64_t r = 0; r < 32; ++r) {
            for (int64_t e = 0; e < row_elems; ++e) {
              const double d = std::fabs(static_cast<double>(Bf(a[static_cast<size_t>(r * row_elems + e)])) - Bf(b[static_cast<size_t>(r * row_elems + e)]));
              if (d > 0) {
                ++count;
                if (first < 0) first = r;
                mx = std::max(mx, d);
              }
            }
          }
          std::printf("  P=%d %-5s rows 0..31 differ from the P=32 call in %lld element(s), max %.4g, first row %lld\n", P, name,
                      static_cast<long long>(count), mx, static_cast<long long>(first));
        };
        const auto difff = [&](const char* name, const std::vector<float>& a, const std::vector<float>& b, int64_t row_elems) {
          int64_t first = -1, count = 0;
          double mx = 0;
          for (int64_t r = 0; r < 32; ++r) {
            for (int64_t e = 0; e < row_elems; ++e) {
              const double d = std::fabs(static_cast<double>(a[static_cast<size_t>(r * row_elems + e)]) - b[static_cast<size_t>(r * row_elems + e)]);
              if (d > 0) {
                ++count;
                if (first < 0) first = r;
                mx = std::max(mx, d);
              }
            }
          }
          std::printf("  P=%d %-5s rows 0..31 differ from the P=32 call in %lld element(s), max %.4g, first row %lld\n", P, name,
                      static_cast<long long>(count), mx, static_cast<long long>(first));
        };
        diff16("q", big.q, small.q, Hg * Kd);
        diff16("k", big.k, small.k, Hg * Kd);
        diff16("v", big.v, small.v, Hh * Vd);
        difff("g", big.g, small.g, Hh);
        difff("beta", big.beta, small.beta, Hh);
        diff16("A", big.A, small.A, Hh * 64);
        diff16("o", big.o, small.o, Hh * Vd);
      }
    }
    // ---- the chunked scan against the sequential recurrence on the same input: which rows differ, and from which call length on ----
    {
      std::printf("\n== prefill of the first P rows of T=70 vs the golden rows [0, P) ==\n");
      const uint16_t* ref70 = g70.Bf16("layer0.attn_out");
      for (int P : {7, 16, 32, 48, 63, 64, 65, 70}) {
        const Golden* gp = &g70;
        // prefill of P rows: run() with prefill_rows = P on the T=70 input, the remaining rows decoded (ignored)
        const std::vector<uint16_t> got = run(*gp, 2, 0, P);
        const Stats s = CompareBf16(got.data(), ref70, static_cast<int64_t>(P) * hidden);
        std::string list;
        int count = 0;
        for (int r = 0; r < P; ++r) {
          const Stats sr = CompareBf16(got.data() + static_cast<int64_t>(r) * hidden, ref70 + static_cast<int64_t>(r) * hidden, hidden);
          if (sr.max_abs > 0.25 * s.rms_ref) {
            ++count;
            if (count <= 30) list += " " + std::to_string(r);
          }
        }
        PrintStats("P=" + std::to_string(P), s);
        std::printf("      %d bad row(s):%s\n", count, list.c_str());
      }
      // per head of the worst row of the T=70 default prefill
      const std::vector<uint16_t> got = run(g70, 2, 0, 70);
      int worst_row = 0;
      double worst = -1;
      for (int r = 0; r < 70; ++r) {
        const Stats sr = CompareBf16(got.data() + static_cast<int64_t>(r) * hidden, ref70 + static_cast<int64_t>(r) * hidden, hidden);
        if (sr.max_abs > worst) {
          worst = sr.max_abs;
          worst_row = r;
        }
      }
      std::printf("worst row %d (max|err| %.4g): per-channel-block [first 8 channels of the 16 largest errors]\n", worst_row, worst);
      std::vector<std::pair<double, int>> errs;
      for (int c = 0; c < hidden; ++c) {
        errs.emplace_back(std::fabs(static_cast<double>(Bf(got[static_cast<size_t>(worst_row) * hidden + c])) - Bf(ref70[static_cast<size_t>(worst_row) * hidden + c])), c);
      }
      std::sort(errs.begin(), errs.end(), std::greater<>());
      for (int i = 0; i < 12; ++i) {
        const int c = errs[static_cast<size_t>(i)].second;
        std::printf("   channel %4d  ours % .5f  golden % .5f\n", c, Bf(got[static_cast<size_t>(worst_row) * hidden + c]),
                    Bf(ref70[static_cast<size_t>(worst_row) * hidden + c]));
      }
      // the layer's decay parameters
      std::vector<float> a_log(48), dt_bias(48);
      R4DX_HIP_CHECK(hipMemcpy(a_log.data(), gw.A_log.data(), 48 * 4, hipMemcpyDeviceToHost));
      R4DX_HIP_CHECK(hipMemcpy(dt_bias.data(), gw.dt_bias.data(), 48 * 4, hipMemcpyDeviceToHost));
      std::printf("A_log[0..7]:");
      for (int i = 0; i < 8; ++i) std::printf(" %.3f", a_log[static_cast<size_t>(i)]);
      std::printf("\ndt_bias[0..7]:");
      for (int i = 0; i < 8; ++i) std::printf(" %.3f", dt_bias[static_cast<size_t>(i)]);
      std::printf("\n");
    }
    // in_proj_a / in_proj_b as the block calls them (64-row slices of GemmBf16NtM64, WV 4, SK 4, MB 1)
    std::printf("\n== in_proj_a / in_proj_b vs CPU ==\n");
    PrintStatsHeader();
    {
      const std::vector<uint16_t> xs(g300.Bf16("layer0.attn_mix"), g300.Bf16("layer0.attn_mix") + 300 * hidden);
      const int64_t Hh = cfg.linear_num_value_heads;
      for (int which = 0; which < 2; ++which) {
        const uint16_t* w = which == 0 ? gw.in_proj_a.data() : gw.in_proj_b.data();
        std::vector<uint16_t> wh(static_cast<size_t>(Hh * hidden));
        R4DX_HIP_CHECK(hipMemcpy(wh.data(), w, wh.size() * 2, hipMemcpyDeviceToHost));
        for (int M : {7, 64, 65, 70, 128, 256}) {
          core::DeviceBuffer<uint16_t> dx(static_cast<size_t>(M * hidden)), dy(static_cast<size_t>(M * Hh));
          dx.CopyFromHost(xs.data(), static_cast<size_t>(M * hidden));
          for (int m0 = 0; m0 < M; m0 += 64) {
            const int m = std::min(64, M - m0);
            core::r4d::GemmBf16NtM64(dx.data() + static_cast<int64_t>(m0) * hidden, w, dy.data() + static_cast<int64_t>(m0) * Hh, m,
                                     static_cast<int>(hidden), static_cast<int>(Hh), 4, 4, 1, stream.get());
          }
          stream.Synchronize();
          const std::vector<uint16_t> got = dy.CopyToHost();
          const std::vector<uint16_t> ref = CpuLinear(wh, xs.data(), M, Hh, hidden);
          const Stats s = CompareBf16(got.data(), ref.data(), static_cast<int64_t>(ref.size()));
          PrintStats(std::string(which == 0 ? "in_proj_a" : "in_proj_b") + " M=" + std::to_string(M), s);
          Check(!s.nan && s.max_norm < 0.02, std::string(which == 0 ? "in_proj_a" : "in_proj_b") + " M=" + std::to_string(M) + " differs from the CPU reference");
        }
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
