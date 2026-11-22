// test_inkling2_ref_cpu -- pins src/kernels/include/r4dx/kernels/inkling2_ref.hpp (the CPU spec of the Inkling2
// kernels) against goldens made by the HF transformers Inkling2Loop modules (tools/inkling2_kern/gen_ref_goldens.py
// -> tests/kernels/fixtures/inkling2/inkling2_ref_golden.bin), plus hand-checked properties the goldens do not
// cover: chunked == one-shot convolutions, router ties, the clip threshold, the bias extent and the sink column.
// Pure CPU, no HIP call. Exit 0 = pass, 1 = fail, 77 = golden file missing.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "r4dx/kernels/inkling2_ref.hpp"

namespace ref = r4dx::inkling2_ref;

namespace {

int g_fail = 0, g_checks = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_fail;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

using Golden = std::map<std::string, std::vector<float>>;

bool LoadGolden(const std::string& path, Golden* g) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "INK2GLD1", 8) != 0) return false;
  uint32_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 4);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t len = 0;
    f.read(reinterpret_cast<char*>(&len), 4);
    std::string name(len, '\0');
    f.read(name.data(), len);
    uint64_t cnt = 0;
    f.read(reinterpret_cast<char*>(&cnt), 8);
    std::vector<float> v(static_cast<size_t>(cnt));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(cnt * 4));
    if (!f) return false;
    (*g)[name] = std::move(v);
  }
  return true;
}

const std::vector<float>& Get(const Golden& g, const std::string& k) {
  static const std::vector<float> empty;
  auto it = g.find(k);
  if (it == g.end()) {
    Check(false, "golden record '" + k + "' missing");
    return empty;
  }
  return it->second;
}

// |a - b| <= atol + rtol * |b| for every element.
void Near(const std::vector<float>& got, const std::vector<float>& want, const std::string& what, float atol = 2e-5f,
          float rtol = 2e-5f) {
  if (got.size() != want.size()) {
    Check(false, what + ": size " + std::to_string(got.size()) + " vs " + std::to_string(want.size()));
    return;
  }
  double worst = 0.0;
  size_t wi = 0;
  bool ok = true;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - want[i]);
    const double tol = atol + rtol * std::fabs(want[i]);
    if (!(err <= tol)) ok = false;
    if (err / tol > worst) {
      worst = err / tol;
      wi = i;
    }
  }
  char buf[256];
  std::snprintf(buf, sizeof buf, "%s (%zu elems, worst err/tol %.3f at %zu: got %.8g want %.8g)", what.c_str(),
                got.size(), worst, wi, got.empty() ? 0.0 : got[wi], want.empty() ? 0.0 : want[wi]);
  Check(ok, buf);
  if (ok) std::printf("ok   %s\n", buf);
}

void TestAttnRes(const Golden& g) {
  struct Case {
    const char* name;
    int n_states, nq;
  };
  for (const Case c : {Case{"attnres3", 4, 3}, Case{"attnres1", 2, 1}, Case{"attnres4", 3, 4}}) {
    const std::string p = c.name;
    std::vector<const float*> st;
    for (int i = 0; i < c.n_states; ++i) st.push_back(Get(g, p + ".s" + std::to_string(i)).data());
    const auto& q = Get(g, p + ".q");
    const int H = static_cast<int>(q.size()) / c.nq;
    const int T = static_cast<int>(Get(g, p + ".s0").size()) / H;
    std::vector<float> out(static_cast<size_t>(c.nq) * T * H);
    ref::AttnRes(st.data(), c.n_states, q.data(), c.nq, T, H, out.data());
    Near(out, Get(g, p + ".out"), p + " vs HF");
  }
}

void TestConv(const Golden& g) {
  {
    const auto& x = Get(g, "sconv.x");
    const auto& w = Get(g, "sconv.w");
    const int C = static_cast<int>(w.size()) / 4, T = static_cast<int>(x.size()) / C;
    std::vector<float> st(3 * C, 0.0f), out(x.size());
    ref::SconvStatic(x.data(), T, C, w.data(), st.data(), nullptr, out.data());
    Near(out, Get(g, "sconv.out"), "sconv_static vs HF Inkling2LoopShortConvolution");
    // chunked (1 + 2 + 4) and token by token give the same bytes as one shot, state included
    for (const std::vector<int> split : {std::vector<int>{1, 2, 4}, std::vector<int>{1, 1, 1, 1, 1, 1, 1}}) {
      std::vector<float> st2(3 * C, 0.0f), out2(x.size());
      int t0 = 0;
      for (int n : split) {
        ref::SconvStatic(x.data() + static_cast<size_t>(t0) * C, n, C, w.data(), st2.data(), nullptr,
                         out2.data() + static_cast<size_t>(t0) * C);
        t0 += n;
      }
      Check(out2 == out && st2 == st, "sconv_static chunked == one shot (bit-exact, state too)");
    }
    // residual: out = residual + y
    std::vector<float> res(x.size()), st3(3 * C, 0.0f), out3(x.size());
    for (size_t i = 0; i < res.size(); ++i) res[i] = 0.25f * static_cast<float>(i % 7) - 0.5f;
    ref::SconvStatic(x.data(), T, C, w.data(), st3.data(), res.data(), out3.data());
    bool ok = true;
    for (size_t i = 0; i < res.size(); ++i) ok = ok && out3[i] == res[i] + out[i];
    Check(ok, "sconv_static residual: out = residual + y");
    // state = last three inputs
    bool sok = true;
    for (int c = 0; c < C; ++c)
      for (int j = 0; j < 3; ++j) sok = sok && st[j * C + c] == x[static_cast<size_t>(T - 3 + j) * C + c];
    Check(sok, "sconv_static state = the last three inputs (oldest first)");
  }
  {
    const auto& x = Get(g, "dconv.x");
    const auto& sw = Get(g, "dconv.static_w");
    const auto& dyn = Get(g, "dconv.dyn");
    const int C = static_cast<int>(sw.size()) / 4, T = static_cast<int>(x.size()) / C;
    const int heads = static_cast<int>(dyn.size()) / T / 4;
    std::vector<float> st(3 * C, 0.0f), out(x.size());
    ref::SconvDynamic(x.data(), T, C, heads, sw.data(), dyn.data(), st.data(), out.data());
    Near(out, Get(g, "dconv.out"), "sconv_dynamic vs HF Inkling2LoopDynamicShortConvolution");
    std::vector<float> st2(3 * C, 0.0f), out2(x.size());
    int t0 = 0;
    for (int n : {3, 1, 3}) {
      ref::SconvDynamic(x.data() + static_cast<size_t>(t0) * C, n, C, heads, sw.data(),
                        dyn.data() + static_cast<size_t>(t0) * heads * 4, st2.data(),
                        out2.data() + static_cast<size_t>(t0) * C);
      t0 += n;
    }
    Check(out2 == out && st2 == st, "sconv_dynamic chunked (3+1+3) == one shot (bit-exact, state too)");
  }
}

void TestHeads(const Golden& g) {
  const auto& x = Get(g, "clip.x");
  const auto& sc = Get(g, "clip.scale");
  const int heads = 3, d = 8, T = static_cast<int>(x.size()) / (heads * d);
  std::vector<float> out(x.size());
  ref::HeadClip(x.data(), T, heads, d, sc.data(), 2.0f, 0, 0.0f, ref::kLogFloor, out.data());
  Near(out, Get(g, "clip.out"), "head clip vs HF Inkling2LoopScaledL2Clip");
  // head 1 is under the threshold: out == x * scale exactly
  bool same = true;
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < d; ++i) {
      const size_t o = (static_cast<size_t>(t) * heads + 1) * d + i;
      same = same && out[o] == x[o] * sc[d + i];
    }
  Check(same, "head clip: a vector under the threshold is unchanged (factor exactly 1)");
  ref::HeadClip(x.data(), T, heads, d, sc.data(), 2.0f, /*pos0=*/1, 0.1f, 4.0f, out.data());
  Near(out, Get(g, "lclip.out"), "head clip + logscale (floor 4, positions 1..6) vs HF Inkling2LoopLogScaledL2Clip");
  ref::HeadRms(x.data(), T, heads, d, out.data());
  Near(out, Get(g, "hrms.out"), "head rms vs HF Inkling2LoopUnweightedRMSNorm");
  Check(ref::LogScale(8190) == 1.0f && ref::LogScale(8191) == 1.0f && ref::LogScale(16383) > 1.0f,
        "logscale: 1 up to position 8191, > 1 after");

  const auto& lin = Get(g, "rel.lin");
  const auto& rsc = Get(g, "rel.scale");
  const int DR = 4, E = 6, Tr = static_cast<int>(lin.size()) / (3 * DR);
  std::vector<float> rs(lin.size()), rel(static_cast<size_t>(Tr) * 3 * E);
  ref::HeadClip(lin.data(), Tr, 3, DR, rsc.data(), 1.5f, /*pos0=*/2, 0.1f, 4.0f, rs.data());
  Near(rs, Get(g, "rel.rs"), "relative states (clip 1.5 + logscale) vs HF Inkling2LoopRelativePositionBias.norm");
  ref::RelLogits(Get(g, "rel.rs").data(), Tr, 3, DR, Get(g, "rel.proj").data(), E, rel.data());
  Near(rel, Get(g, "rel.logits"), "rel_logits vs HF relative_states @ proj");
}

void TestAttention(const Golden& g) {
  for (const char* c : {"attn_full", "attn_swa"}) {
    const std::string p = c;
    const auto& sh = Get(g, p + ".shape");
    if (sh.size() != 7) continue;
    const int QL = static_cast<int>(sh[0]), QH = static_cast<int>(sh[1]), KVH = static_cast<int>(sh[2]);
    const int D = static_cast<int>(sh[3]), CTX = static_cast<int>(sh[4]), W = static_cast<int>(sh[5]);
    const int E = static_cast<int>(sh[6]);
    std::vector<float> out(static_cast<size_t>(QL) * QH * D);
    ref::Attention(Get(g, p + ".q").data(), Get(g, p + ".k").data(), Get(g, p + ".v").data(), Get(g, p + ".rel").data(),
                   Get(g, p + ".sink").data(), QL, QH, KVH, D, CTX, W, E, 1.0f / std::sqrt(static_cast<float>(D)),
                   out.data());
    Near(out, Get(g, p + ".out"), p + " (bias + sink) vs HF eager_attention_forward");
  }
  // Hand checks: one key, no bias, a sink: out = v * e^s / (e^s + e^sink).
  {
    const float q[2] = {1.0f, 0.0f}, k[2] = {2.0f, 0.0f}, v[2] = {3.0f, -1.0f}, sink[1] = {1.0f};
    float out[2];
    ref::Attention(q, k, v, nullptr, sink, 1, 1, 1, 2, 1, 0, 0, 1.0f, out);
    const float p = std::exp(2.0f) / (std::exp(2.0f) + std::exp(1.0f));
    Check(std::fabs(out[0] - 3.0f * p) < 1e-6f && std::fabs(out[1] + p) < 1e-6f, "attention: the sink absorbs mass");
    ref::Attention(q, k, v, nullptr, nullptr, 1, 1, 1, 2, 1, 0, 0, 1.0f, out);
    Check(out[0] == 3.0f && out[1] == -1.0f, "attention: no sink, one key -> its value exactly");
  }
  // bias beyond the extent is zero: 3 keys, extent 1 -> only the current key gets rel[0]
  {
    const float q[1] = {0.0f}, k[3] = {0, 0, 0}, v[3] = {1.0f, 2.0f, 4.0f};
    const float rel[1] = {std::log(2.0f)};
    float out[1];
    ref::Attention(q, k, v, rel, nullptr, 1, 1, 1, 1, 3, 0, 1, 1.0f, out);
    // weights 1, 1, 2 over values 1, 2, 4 -> (1 + 2 + 8) / 4
    Check(std::fabs(out[0] - 11.0f / 4.0f) < 1e-6f, "attention: rel applies at distance < extent only");
    ref::Attention(q, k, v, rel, nullptr, 1, 1, 1, 1, 3, 2, 1, 1.0f, out);  // window 2: keys 1, 2
    Check(std::fabs(out[0] - 10.0f / 3.0f) < 1e-6f, "attention: window 2 sees the last two keys");
  }
}

void TestMsgRouterAct(const Golden& g) {
  const auto& x = Get(g, "msg.x");
  const auto& m = Get(g, "msg.m");
  const auto& down = Get(g, "msg.down");
  const int H = 8, R = 4, T = static_cast<int>(x.size()) / H;
  std::vector<float> out(x.size()), prep(static_cast<size_t>(T) * 2 * H), gg(static_cast<size_t>(T) * R);
  ref::MsgGate(x.data(), m.data(), down.data(), Get(g, "msg.forget").data(), Get(g, "msg.gain").data(), T, H, R,
               out.data());
  Near(out, Get(g, "msg.out"), "msg gate vs HF Inkling2LoopMessageGate");
  ref::MsgGatePrep(x.data(), m.data(), T, H, prep.data());
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < R; ++r) {
      float a = 0.0f;
      for (int k = 0; k < 2 * H; ++k) a += prep[static_cast<size_t>(t) * 2 * H + k] * down[static_cast<size_t>(r) * 2 * H + k];
      gg[static_cast<size_t>(t) * R + r] = a;
    }
  Near(gg, Get(g, "msg.g"), "msg gate prep @ down vs HF down_proj(cat(rms x, rms m))");

  const auto& xr = Get(g, "router.x");
  const int Hr = 16, E = 8, K = 3, Tr = static_cast<int>(xr.size()) / Hr;
  std::vector<int32_t> ids(static_cast<size_t>(Tr) * K);
  std::vector<float> w(static_cast<size_t>(Tr) * K), logits(static_cast<size_t>(Tr) * E);
  ref::Router(xr.data(), Get(g, "router.w").data(), Get(g, "router.bias").data(), Tr, Hr, E, K, 8.0f, ids.data(),
              w.data(), logits.data());
  Near(logits, Get(g, "router.logits"), "router logits vs HF");
  std::vector<float> idf(ids.begin(), ids.end());
  Check(idf == Get(g, "router.ids"), "router ids (descending score) == HF topk set in score order");
  Near(w, Get(g, "router.weights"), "router weights vs HF");
  {  // ties: equal scores pick the lower index first
    const float xt[1] = {0.0f}, wt[4] = {0, 0, 0, 0}, bias[4] = {0.5f, 1.0f, 1.0f, 0.5f};
    int32_t id[3];
    float wt_out[3];
    ref::Router(xt, wt, bias, 1, 1, 4, 3, 2.0f, id, wt_out);
    Check(id[0] == 1 && id[1] == 2 && id[2] == 0, "router ties -> lower index (1, 2, then 0 before 3)");
    Check(std::fabs(wt_out[0] - 2.0f / 3.0f) < 1e-6f, "router weight = s / sum * scale");
  }

  const auto& gt = Get(g, "swiglu.g");
  const auto& up = Get(g, "swiglu.u");
  std::vector<float> sw(gt.size());
  for (size_t i = 0; i < gt.size(); ++i) sw[i] = ref::SwigluSoftcap(gt[i], up[i]);
  Near(sw, Get(g, "swiglu.out"), "softcapped swiglu vs HF");
  const auto& xs = Get(g, "softcap.x");
  std::vector<float> scv(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) scv[i] = ref::Softcap(xs[i], ref::kLogitSoftcap);
  Near(scv, Get(g, "softcap.out"), "logit softcap 30 vs HF");
  const auto& xw = Get(g, "rmsw.x");
  const auto& ww = Get(g, "rmsw.w");
  std::vector<float> rw(xw.size());
  for (size_t r = 0; r < xw.size() / 16; ++r) ref::RmsWeighted(xw.data() + r * 16, ww.data(), 16, rw.data() + r * 16);
  Near(rw, Get(g, "rmsw.out"), "weighted rmsnorm vs HF Inkling2LoopRMSNorm");
}

}  // namespace

int main() {
  const std::string path = std::string(R4DX_SOURCE_DIR) + "/tests/kernels/fixtures/inkling2/inkling2_ref_golden.bin";
  Golden g;
  if (!LoadGolden(path, &g)) {
    std::printf("SKIP: %s missing or unreadable (tools/inkling2_kern/gen_ref_goldens.py writes it)\n", path.c_str());
    return 77;
  }
  TestAttnRes(g);
  TestConv(g);
  TestHeads(g);
  TestAttention(g);
  TestMsgRouterAct(g);
  std::printf("test_inkling2_ref_cpu: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
