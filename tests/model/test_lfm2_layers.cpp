// tests/model/test_lfm2_layers.cpp -- docs/lfm2-plan.md L1-9, rung 3 (docs/validation.md): the C++ LFM2 decoder layers
// against the per-layer goldens of tools/reference/lfm2/layer_golden_lfm2.py + router_golden.py (L0-9) on the REAL
// LFM2.5-24B-A2B weights of the lfm2-l6.r4dx container (layers 0-5, bf16). GPU test, ONE device (HIP_VISIBLE_DEVICES=1).
//
// NOT RUN BY THE AUTHOR (no GPU work was allowed). Exits 77 (SKIPPED) while the manifest or the container is missing.
//   R4DX_LFM2_GOLDEN_DIR      default <source>/tools/reference/golden_out/lfm2 (layer_golden_lfm2.py --out-dir; the lane-A
//                             worktree's copy is C:\Users\pay20\dev\r4dx-lfm2-ref\tools\reference\golden_out\lfm2)
//   R4DX_LFM2_L6_CONTAINER    default <models root>/r4dx/lfm2/lfm2-l6.r4dx
//
// What is checked (tolerances: kFloorMultiple x the manifest's bf16-vs-fp32 twin floor of the tensor, never below kMinTol):
//   1. Each layer component (layer_000_conv_dense, layer_002_attn_moe, layer_003_conv_moe), Lfm2Model::DebugLayerForward:
//      a 64-row prefill chunk from position 0, then the 4 decode rows ONE AT A TIME (KV continuation, ShortConv state carry,
//      the M = 1 per-slot MoE path): operator_norm_out, mixer_out, layer_out (prefill rows and decode rows); MoE layers also
//      report the (token, slot) id agreement of the in-layer router with the golden fp32 ids (>= 95 %: the input differs).
//   2. The FFN alone on the golden ffn_norm_out (DebugFfnForward):
//      dense (layer 0)  vs ffn_out;
//      MoE (layers 2, 3) sorted path: router ids == router_ids_fp32 EXACTLY (as sets per row), weights vs
//      router_weights_fp32 rel 1e-3, output vs ffn_out_fp32_combine on the rows whose HF-bf16 ids equal the fp32 ids;
//      the per-slot (decode) path over the same rows equals the sorted path's ids and output (rel L2 < 1e-2; bytes
//      reported), and a single row through both paths is byte-identical (same M = 1 GEMMs).
//   3. shortconv_state (layer 0): the mixer of layer 0 over 300 rows fed 256 + 44, then 4 single rows: mixer_out vs
//      out_proj_y_full, the state after each step vs state_after_N (rel L2), and the 128 + 128 + 44 + 1x4 split equals
//      the 256 + 44 + 1x4 split byte for byte (the GEMMs slice M at 64 rows either way; the conv is exact).
//   4. Router goldens (router_real_layer_002/003, router_tie, router_bias_selects when present): the router kernel through
//      DebugFfnForward is covered by 2.; these are lane D's kernel tests (test_lfm2_router) and are not repeated.
//   5. qk_norm_rope: the container's layer-2 q/k norm weights equal the golden's; per-head RMSNorm + rope_neox (theta 1e7)
//      on the golden q_raw / k_raw at positions up to 131071 vs q_roped_f32 / k_roped_f32 (rel L2 < 1e-2).
//   6. embed: the device gather of the container table == embed_rows (bit exact); final_norm_head: the container's final
//      norm on x vs normed (rel < 5e-3) and the tied lm_head's logits on the golden's vocab slice vs logits_bf16 (< 1e-2).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "lfm2_model.h"
#include "linear.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"

#ifndef R4DX_LFM2_GOLDEN_DIR
#define R4DX_LFM2_GOLDEN_DIR "tools/reference/golden_out/lfm2"
#endif

using namespace r4dx;
using namespace r4dx::model;
using nlohmann::json;
using r4dx_test::RelL2;
using r4dx_test::WidenBf16;

namespace {

constexpr double kFloorMultiple = 4.0;
constexpr double kMinTol = 8e-3;  // bf16 output rounding + single- vs double-rounded norms

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Golden {
  json manifest;
  std::string dir;
  const json& Component(const std::string& name) const { return manifest.at("components").at(name); }
  bool Has(const std::string& name) const { return manifest.at("components").contains(name); }
  r4dx_convert::SafetensorsReader Open(const std::string& name) const {
    return r4dx_convert::SafetensorsReader(r4dx_convert::Utf8ToWide(dir + "/" + Component(name).at("file").get<std::string>()));
  }
};

double FloorTol(const json& meta, const char* tensor) {
  double floor = 0.0;
  if (meta.contains("fp32_floor") && meta.at("fp32_floor").contains(tensor)) {
    floor = meta.at("fp32_floor").at(tensor).at("rel_fro").get<double>();
  }
  return std::max(kMinTol, kFloorMultiple * floor);
}

template <class T>
std::vector<T> Rows(const std::vector<T>& all, int64_t width, int64_t r0, int64_t r1) {
  return std::vector<T>(all.begin() + r0 * width, all.begin() + r1 * width);
}

std::vector<int32_t> ReadI32(const r4dx_convert::SafetensorsReader& r, const std::string& name) {
  const auto& m = r.Meta(name);
  const int64_t n = m.ElemCount();
  std::vector<int32_t> out(static_cast<size_t>(n));
  if (m.dtype == "I32") {
    std::memcpy(out.data(), r.Data(name), out.size() * sizeof(int32_t));
  } else if (m.dtype == "I64") {
    const auto* src = reinterpret_cast<const int64_t*>(r.Data(name));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = static_cast<int32_t>(src[i]);
  } else {
    throw std::runtime_error("ReadI32: " + name + " is " + m.dtype);
  }
  return out;
}

std::vector<int32_t> SortedRow(const std::vector<int32_t>& ids, int64_t k, int64_t t) {
  std::vector<int32_t> v(ids.begin() + t * k, ids.begin() + (t + 1) * k);
  std::sort(v.begin(), v.end());
  return v;
}

// ---- 1. one layer, prefill chunk + decode rows -------------------------------------------------------------------
void TestLayerComponent(const Golden& g, Lfm2Model& m, const std::string& name) {
  if (!g.Has(name)) {
    std::printf("  (no %s in the manifest)\n", name.c_str());
    return;
  }
  const json& meta = g.Component(name);
  const int layer = meta.at("layer").get<int>();
  const int prefill = meta.at("prefill").get<int>(), decode = meta.at("decode").get<int>();
  const int64_t H = m.Config().hidden_size;
  const int64_t K = m.Config().num_experts_per_tok;
  const auto r = g.Open(name);
  const std::vector<uint16_t> x = r4dx_test::ReadGoldenRawBf16(r, "x");
  const std::vector<float> want_out = r4dx_test::ReadGoldenAsFloat(r, "layer_out");
  const std::vector<float> want_mixer = r4dx_test::ReadGoldenAsFloat(r, "mixer_out");
  const std::vector<float> want_opn = r4dx_test::ReadGoldenAsFloat(r, "operator_norm_out");
  const bool moe = meta.at("ffn").get<std::string>() == "moe";

  Lfm2LayerDebug dbg;
  const std::vector<uint16_t> a = m.DebugLayerForward(layer, Rows(x, H, 0, prefill), prefill, 0, &dbg);
  std::vector<uint16_t> b, b_mixer;
  std::vector<int32_t> b_ids;
  for (int t = 0; t < decode; ++t) {
    Lfm2LayerDebug d1;
    const std::vector<uint16_t> o = m.DebugLayerForward(layer, Rows(x, H, prefill + t, prefill + t + 1), 1, prefill + t, &d1);
    b.insert(b.end(), o.begin(), o.end());
    b_mixer.insert(b_mixer.end(), d1.mixer_out.begin(), d1.mixer_out.end());
    b_ids.insert(b_ids.end(), d1.moe.ids.begin(), d1.moe.ids.end());
  }
  const double e_opn = RelL2(WidenBf16(dbg.operator_norm_out), Rows(want_opn, H, 0, prefill));
  const double e_mix_a = RelL2(WidenBf16(dbg.mixer_out), Rows(want_mixer, H, 0, prefill));
  const double e_mix_b = RelL2(WidenBf16(b_mixer), Rows(want_mixer, H, prefill, prefill + decode));
  const double e_a = RelL2(WidenBf16(a), Rows(want_out, H, 0, prefill));
  const double e_b = RelL2(WidenBf16(b), Rows(want_out, H, prefill, prefill + decode));
  const double tol_mix = FloorTol(meta, "mixer_out"), tol_out = FloorTol(meta, "layer_out");
  std::printf("  %s: operator_norm %.3e | mixer prefill %.3e decode %.3e (tol %.3e) | layer_out prefill %.3e decode %.3e (tol %.3e)\n",
              name.c_str(), e_opn, e_mix_a, e_mix_b, tol_mix, e_a, e_b, tol_out);
  Check(e_opn <= kMinTol, name + ": operator_norm output");
  Check(e_mix_a <= tol_mix, name + ": mixer output, prefill chunk");
  Check(e_mix_b <= tol_mix, name + ": mixer output, decode rows (KV / ShortConv state carried from the chunk)");
  Check(e_a <= tol_out, name + ": layer_out, prefill chunk");
  Check(e_b <= tol_out, name + ": layer_out, decode rows");
  if (moe) {
    const std::vector<int32_t> want_ids = ReadI32(r, "router_ids_fp32");
    std::vector<int32_t> all_ids = dbg.moe.ids;
    all_ids.insert(all_ids.end(), b_ids.begin(), b_ids.end());
    int64_t agree = 0, total = 0;
    for (int64_t t = 0; t < prefill + decode; ++t) {
      const std::vector<int32_t> got = SortedRow(all_ids, K, t), want = SortedRow(want_ids, K, t);
      for (int32_t e : got) agree += std::binary_search(want.begin(), want.end(), e) ? 1 : 0;
      total += K;
    }
    const double frac = static_cast<double>(agree) / static_cast<double>(total);
    std::printf("  %s: in-layer router id agreement with the fp32 golden ids: %.4f (%lld / %lld pairs)\n", name.c_str(), frac,
                static_cast<long long>(agree), static_cast<long long>(total));
    Check(frac >= 0.95, name + ": in-layer router ids agree with the fp32 ids on >= 95 % of the (token, slot) pairs");
  }
}

// ---- 2. the FFN alone on the golden ffn_norm_out --------------------------------------------------------------------
void TestFfnAlone(const Golden& g, Lfm2Model& m, const std::string& name) {
  if (!g.Has(name)) return;
  const json& meta = g.Component(name);
  const int layer = meta.at("layer").get<int>();
  const int T = meta.at("prefill").get<int>();  // the first chunk's rows (<= prefill_chunk)
  const int64_t H = m.Config().hidden_size;
  const int64_t K = m.Config().num_experts_per_tok;
  const auto r = g.Open(name);
  const std::vector<uint16_t> xn = Rows(r4dx_test::ReadGoldenRawBf16(r, "ffn_norm_out"), H, 0, T);
  if (meta.at("ffn").get<std::string>() == "dense") {
    const std::vector<uint16_t> got = m.DebugFfnForward(layer, xn, T);
    const double e = RelL2(WidenBf16(got), Rows(r4dx_test::ReadGoldenAsFloat(r, "ffn_out"), H, 0, T));
    const double tol = FloorTol(meta, "ffn_out");
    std::printf("  %s dense FFN alone: rel L2 %.3e (tol %.3e)\n", name.c_str(), e, tol);
    Check(e <= tol, name + ": dense SwiGLU on the golden ffn_norm_out");
    return;
  }
  Lfm2MoeDebug ds, dp;
  const std::vector<uint16_t> sorted = m.DebugFfnForward(layer, xn, T, Lfm2MoePath::kSorted, &ds);
  const std::vector<uint16_t> per_slot = m.DebugFfnForward(layer, xn, T, Lfm2MoePath::kPerSlot, &dp);
  const std::vector<int32_t> ids32 = Rows(ReadI32(r, "router_ids_fp32"), K, 0, T);
  const std::vector<int32_t> ids16 = Rows(ReadI32(r, "router_ids"), K, 0, T);
  const std::vector<float> w32 = Rows(r4dx_test::ReadGoldenAsFloat(r, "router_weights_fp32"), K, 0, T);
  const std::vector<float> comb = Rows(r4dx_test::ReadGoldenAsFloat(r, "ffn_out_fp32_combine"), H, 0, T);
  bool ids_exact = true;
  double w_worst = 0;
  std::vector<float> got_rows, want_rows;
  int64_t clean_rows = 0;
  for (int64_t t = 0; t < T; ++t) {
    if (SortedRow(ds.ids, K, t) != SortedRow(ids32, K, t)) ids_exact = false;
    for (int64_t j = 0; j < K; ++j) {  // weight of each selected expert, matched by id
      const int32_t e = ds.ids[static_cast<size_t>(t * K + j)];
      for (int64_t jj = 0; jj < K; ++jj) {
        if (ids32[static_cast<size_t>(t * K + jj)] != e) continue;
        const double want = w32[static_cast<size_t>(t * K + jj)];
        w_worst = std::max(w_worst, std::fabs(ds.w[static_cast<size_t>(t * K + j)] - want) / std::max(1e-12, std::fabs(want)));
      }
    }
    if (SortedRow(ids16, K, t) == SortedRow(ids32, K, t)) {  // HF's bf16 router picked the same experts: comparable row
      ++clean_rows;
      for (int64_t c = 0; c < H; ++c) {
        got_rows.push_back(core::Bf16ToFloat(sorted[static_cast<size_t>(t * H + c)]));
        want_rows.push_back(comb[static_cast<size_t>(t * H + c)]);
      }
    }
  }
  const double e_out = RelL2(got_rows, want_rows);
  std::printf("  %s MoE alone (sorted): ids %s, worst weight rel err %.3e, output vs fp32 combine on %lld clean rows rel L2 %.3e\n",
              name.c_str(), ids_exact ? "exact" : "DIFFER", w_worst, static_cast<long long>(clean_rows), e_out);
  Check(ids_exact, name + ": router ids (fp32, r4dx) == router_ids_fp32 for every row");
  Check(w_worst <= 1e-3, name + ": router weights vs router_weights_fp32 (rel 1e-3)");
  Check(clean_rows > 0 && e_out <= 1e-2, name + ": MoE output vs the golden fp32 combine of the HF experts (rel L2 1e-2)");
  const double e_ps = RelL2(WidenBf16(per_slot), WidenBf16(sorted));
  std::printf("  %s per-slot (decode) path vs sorted path over %d rows: rel L2 %.3e, bytes %s\n", name.c_str(), T, e_ps,
              per_slot == sorted ? "identical" : "differ");
  Check(dp.ids == ds.ids, name + ": the per-slot and the sorted path route identically");
  Check(e_ps < 1e-2, name + ": per-slot path output ~ sorted path output");
  const std::vector<uint16_t> one = Rows(xn, H, 0, 1);
  const std::vector<uint16_t> o_auto = m.DebugFfnForward(layer, one, 1, Lfm2MoePath::kAuto);
  const std::vector<uint16_t> o_sort = m.DebugFfnForward(layer, one, 1, Lfm2MoePath::kSorted);
  Check(o_auto == o_sort, name + ": one row (M = 1 everywhere): the decode path == the sorted path byte for byte");
}

// ---- 3. ShortConv state carry ------------------------------------------------------------------------------------
std::vector<uint16_t> RunConvSplit(Lfm2Model& m, int layer, const std::vector<uint16_t>& x, int64_t H,
                                   const std::vector<int64_t>& bounds, std::vector<std::vector<uint16_t>>* states) {
  std::vector<uint16_t> mixer;
  int64_t pos = 0;
  for (int64_t end : bounds) {
    Lfm2LayerDebug d;
    m.DebugLayerForward(layer, Rows(x, H, pos, end), end - pos, pos, &d);
    mixer.insert(mixer.end(), d.mixer_out.begin(), d.mixer_out.end());
    if (states != nullptr) states->push_back(m.DebugConvState(layer));
    pos = end;
  }
  return mixer;
}

void TestShortConvState(const Golden& g, Lfm2Model& m) {
  if (!g.Has("shortconv_state")) return;
  const json& meta = g.Component("shortconv_state");
  const int layer = meta.at("layer").get<int>();
  const int64_t H = m.Config().hidden_size;
  const auto r = g.Open("shortconv_state");
  const std::vector<uint16_t> x = r4dx_test::ReadGoldenRawBf16(r, "x");
  const int64_t N = static_cast<int64_t>(x.size()) / H;  // 304
  const std::vector<float> want = r4dx_test::ReadGoldenAsFloat(r, "out_proj_y_full");
  std::vector<int64_t> bounds;
  for (const auto& b : meta.at("bounds")) bounds.push_back(b.at(1).get<int64_t>());  // [[0,256],[256,300],[300,301],...]
  std::vector<std::vector<uint16_t>> states;
  const std::vector<uint16_t> mixer = RunConvSplit(m, layer, x, H, bounds, &states);
  const double e = RelL2(WidenBf16(mixer), want);
  const double tol = g.Has("layer_000_conv_dense") ? FloorTol(g.Component("layer_000_conv_dense"), "mixer_out") : 2.5e-2;
  std::printf("  shortconv_state: mixer over %lld rows (256 + 44 + 4 x 1) vs out_proj(y_full): rel L2 %.3e (tol %.3e)\n",
              static_cast<long long>(N), e, tol);
  Check(e <= tol, "shortconv_state: the conv mixer across a chunk boundary and decode steps");
  double worst = 0;
  for (size_t i = 0; i < bounds.size(); ++i) {
    const std::string key = "state_after_" + std::to_string(bounds[i]);
    if (!r.Has(key)) continue;
    worst = std::max(worst, RelL2(WidenBf16(states[i]), r4dx_test::ReadGoldenAsFloat(r, key)));
  }
  std::printf("  shortconv_state: worst state rel L2 vs state_after_N %.3e\n", worst);
  Check(worst <= 2e-2, "shortconv_state: the [2, H] state after every step (h = B*x of r4dx's own in_proj GEMM)");
  std::vector<int64_t> split2 = {128, 256};
  split2.insert(split2.end(), bounds.begin() + 1, bounds.end());
  std::vector<std::vector<uint16_t>> states2;
  const std::vector<uint16_t> mixer2 = RunConvSplit(m, layer, x, H, split2, &states2);
  Check(mixer2 == mixer && states2.back() == states.back(),
        "shortconv_state: 128 + 128 + 44 + 1x4 == 256 + 44 + 1x4 byte for byte (mixer and final state)");
}

// ---- 5. q/k norm + rope at head_dim 64 ---------------------------------------------------------------------------
void TestQkNormRope(const Golden& g, Lfm2Model& m) {
  if (!g.Has("qk_norm_rope")) return;
  const json& meta = g.Component("qk_norm_rope");
  const int layer = meta.at("layer").get<int>();
  const auto r = g.Open("qk_norm_rope");
  const Lfm2Config& cfg = m.Config();
  const int D = static_cast<int>(cfg.head_dim), Hq = static_cast<int>(cfg.num_attention_heads),
            Hk = static_cast<int>(cfg.num_key_value_heads);
  const Lfm2LayerWeights& lw = m.GetContainer().Layer(layer);
  Check(lw.attn.q_norm.CopyToHost() == r4dx_test::ReadGoldenRawBf16(r, "q_norm_weight") &&
            lw.attn.k_norm.CopyToHost() == r4dx_test::ReadGoldenRawBf16(r, "k_norm_weight"),
        "qk_norm_rope: the container's q/k layernorm weights == the checkpoint's");
  const std::vector<int32_t> pos = ReadI32(r, "positions");
  const int n = static_cast<int>(pos.size());
  core::Stream stream;
  const int64_t s = P(stream.get());
  auto q = r4dx_test::UploadBf16(r4dx_test::ReadGoldenRawBf16(r, "q_raw"));
  auto k = r4dx_test::UploadBf16(r4dx_test::ReadGoldenRawBf16(r, "k_raw"));
  core::DeviceBuffer<int32_t> pd(pos.size());
  pd.CopyFromHost(pos);
  const float eps = static_cast<float>(cfg.norm_eps);
  r4dx_rmsnorm_plain_bf16(P(q.data()), P(lw.attn.q_norm.data()), P(q.data()), static_cast<int64_t>(n) * Hq, D, eps, 0, s);
  r4dx_rmsnorm_plain_bf16(P(k.data()), P(lw.attn.k_norm.data()), P(k.data()), static_cast<int64_t>(n) * Hk, D, eps, 0, s);
  r4dx_rope_neox_bf16(P(q.data()), P(k.data()), P(pd.data()), n, Hq, Hk, D, static_cast<float>(cfg.rope_theta), s);
  stream.Synchronize();
  const double eq = RelL2(WidenBf16(q.CopyToHost()), r4dx_test::ReadGoldenAsFloat(r, "q_roped_f32"));
  const double ek = RelL2(WidenBf16(k.CopyToHost()), r4dx_test::ReadGoldenAsFloat(r, "k_roped_f32"));
  std::printf("  qk_norm_rope (positions up to %d): q rel L2 %.3e, k %.3e vs the fp32-table reference\n", pos.back(), eq, ek);
  Check(eq < 1e-2 && ek < 1e-2, "qk_norm_rope: per-head q/k RMSNorm + rope_neox (theta 1e7, head_dim 64)");
}

// ---- 6. embedding gather, final norm, tied head ------------------------------------------------------------------
void TestEmbedFinalHead(const Golden& g, Lfm2Model& m) {
  const int64_t H = m.Config().hidden_size;
  core::Stream stream;
  const int64_t s = P(stream.get());
  if (g.Has("embed")) {
    const auto r = g.Open("embed");
    const std::vector<int32_t> ids = ReadI32(r, "token_ids");
    core::DeviceBuffer<int32_t> idd(ids.size());
    idd.CopyFromHost(ids);
    core::DeviceBuffer<uint16_t> out(ids.size() * static_cast<size_t>(H));
    r4dx_embedding_gather_bf16(P(m.GetContainer().EmbedTokensDevice().data()), P(idd.data()), P(out.data()),
                               static_cast<int64_t>(ids.size()), H, m.Config().vocab_size, s);
    stream.Synchronize();
    Check(out.CopyToHost() == r4dx_test::ReadGoldenRawBf16(r, "embed_rows"),
          "embed: the device gather of text.embed_tokens == the checkpoint rows (no scale), bit exact");
  }
  if (g.Has("final_norm_head")) {
    const json& meta = g.Component("final_norm_head");
    const auto r = g.Open("final_norm_head");
    const std::vector<uint16_t> x = r4dx_test::ReadGoldenRawBf16(r, "x");
    const int64_t n = static_cast<int64_t>(x.size()) / H;
    auto xd = r4dx_test::UploadBf16(x);
    core::DeviceBuffer<uint16_t> nd(x.size());
    r4dx_rmsnorm_plain_bf16(P(xd.data()), P(m.GetContainer().FinalNorm().data()), P(nd.data()), n, H,
                            static_cast<float>(m.Config().norm_eps), 0, s);
    const int64_t V = m.GetContainer().LmHead().N;
    core::DeviceBuffer<uint16_t> logits(static_cast<size_t>(n * V));
    core::Arena arena;
    arena.Reserve(64ull << 20);
    ApplyLinear(stream.get(), arena, m.GetContainer().LmHead(), nd.data(), logits.data(), n);
    stream.Synchronize();
    const double en = RelL2(WidenBf16(nd.CopyToHost()), r4dx_test::ReadGoldenAsFloat(r, "normed"));
    const int64_t c0 = meta.at("head_slice").at(0).get<int64_t>(), c1 = meta.at("head_slice").at(1).get<int64_t>();
    const std::vector<uint16_t> all = logits.CopyToHost();
    std::vector<float> got;
    for (int64_t t = 0; t < n; ++t) {
      for (int64_t c = c0; c < c1; ++c) got.push_back(core::Bf16ToFloat(all[static_cast<size_t>(t * V + c)]));
    }
    const double eh = RelL2(got, r4dx_test::ReadGoldenAsFloat(r, "logits_bf16"));
    std::printf("  final norm rel L2 %.3e; tied head on vocab [%lld, %lld) rel L2 %.3e\n", en, static_cast<long long>(c0),
                static_cast<long long>(c1), eh);
    Check(en < 5e-3, "final_norm_head: embedding_norm (the final norm, plain weight)");
    Check(eh < 1e-2, "final_norm_head: the tied lm_head rows straddling the real-vocab edge");
  }
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_lfm2_layers", []() -> int {
    const char* gd = std::getenv("R4DX_LFM2_GOLDEN_DIR");
    const std::string dir = gd != nullptr && *gd != '\0' ? gd : std::string(R4DX_LFM2_GOLDEN_DIR);
    const char* ce = std::getenv("R4DX_LFM2_L6_CONTAINER");
    const std::string container = ce != nullptr && *ce != '\0' ? ce : r4dx::ModelsPath("r4dx/lfm2/lfm2-l6.r4dx");
    if (!r4dx_test::FileExists(dir + "/manifest.json")) {
      return r4dx_test::SkipMissing(dir + "/manifest.json (layer_golden_lfm2.py, L0-9)");
    }
    if (!r4dx_test::FileExists(container)) return r4dx_test::SkipMissing(container);
    Golden g;
    g.dir = dir;
    {
      std::ifstream f(dir + "/manifest.json");
      g.manifest = json::parse(f);
    }
    if (g.manifest.value("tiny", false)) {
      std::printf("[SKIP] %s holds TINY-model goldens; this test needs the real-layer ones\n", dir.c_str());
      return r4dx_test::kSkipReturnCode;
    }
    Lfm2ModelOptions o;
    o.container_path = container;
    o.layer_limit = 4;  // layers 0..3 cover every component
    o.max_ctx = 4096;
    Lfm2Model m = Lfm2Model::Load(o);
    std::printf("---- layer components ----\n");
    for (const char* name : {"layer_000_conv_dense", "layer_002_attn_moe", "layer_003_conv_moe"}) TestLayerComponent(g, m, name);
    std::printf("---- FFN alone ----\n");
    for (const char* name : {"layer_000_conv_dense", "layer_002_attn_moe", "layer_003_conv_moe"}) TestFfnAlone(g, m, name);
    std::printf("---- ShortConv state ----\n");
    TestShortConvState(g, m);
    std::printf("---- q/k norm + rope ----\n");
    TestQkNormRope(g, m);
    std::printf("---- embedding / final norm / head ----\n");
    TestEmbedFinalHead(g, m);
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
