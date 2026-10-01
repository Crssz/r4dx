// tests/model/attention/test_gemma_attn_layer.cpp -- docs/gemma4-plan.md M1-21, rung 3 (docs/validation.md):
// the C++ Gemma 4 decoder layer against the per-layer bf16 goldens of tools/reference/gemma/layer_golden_gemma.py
// (M0-8) on the REAL Huihui weights (D:\models\r4dx\huihui-gemma\bf16.r4dx). GPU test, HIP device 1 only.
//
// NOT RUN BY THE AUTHOR (no GPU work was allowed): it compiles, and is the user's command to run once the M0-8
// goldens exist (tools/reference/golden_out/gemma/manifest.json). It exits 77 (SKIPPED) while the manifest or the
// container is missing.
//
// Components of the manifest it reads (the contract of layer_golden_gemma.py):
//   layer_000_sliding, layer_005_full  one decoder layer over T = prefill + decode rows from position 0
//       (tensors: x, input_ln, attn_out [= o_proj output], mid_residual, mlp_down, pre_scalar, layer_out, ...;
//       meta: layer, layer_type, prefill, decode, layer_scalar, fp32_floor[tensor].rel_fro).
//   sliding_ring_wrap                  the sliding layer over ring_prefill (1500) + decode (4) rows: the last rows
//       attend through a window that has wrapped the 1536-slot ring (tensors: x, layer_out, attn_out).
//   embed_scale                        gathered rows * bf16(sqrt(hidden)) (= 62.0): bit exact.
//   final_norm_softcap                 the final RMSNorm and the softcap probe.
//
// What is checked, per KV mode (bf16 first -- the KV-off-error rung -- then fp8 e4m3 with the container's static
// descales, which are the converter's 1.0 placeholders until M1-23 calibrates them, so the fp8 tolerance is looser):
//   1. GemmaAttnLayer alone: o_proj output of the layer for rows [0, prefill) from the golden `input_ln` input
//      against `attn_out` (the stage-0 reference kernel; for the sliding layer with fp8 KV also the libr4d
//      windowed kernel against the reference kernel's bytes).
//   2. The whole layer (GemmaModel::DebugLayerForward: attention + sandwich norms + GeGLU + layer_scalar) as a
//      prefill chunk and then the decode rows as a second chunk: `layer_out` rel L2 per part.
//   3. The ring wrap: the sliding layer fed 1504 rows in 256-row chunks; the last 4 rows vs the golden.
//   4. The scaled embedding gather (exact) and the final norm + softcap probe.
// Tolerances: a multiple (kFloorMultiple = 4, the manifest's own suggestion) of the manifest's bf16-vs-fp32 twin
// floor for the tensor, never below kMinTol.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gemma_model.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "kernels/model_kernels.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"

#ifndef R4DX_GEMMA_GOLDEN_DIR
#define R4DX_GEMMA_GOLDEN_DIR "tools/reference/golden_out/gemma"
#endif

using namespace r4dx;
using namespace r4dx::model;
using nlohmann::json;
using r4dx_test::RelL2;

namespace {

constexpr double kFloorMultiple = 4.0;
constexpr double kMinTol = 3e-3;     // bf16 rounding of the final cast alone
constexpr double kFp8KvTol = 0.12;   // fp8 e4m3 K/V with placeholder (1.0) descales: ~6% per element

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

std::vector<uint16_t> Rows(const std::vector<uint16_t>& all, int64_t hidden, int64_t r0, int64_t r1) {
  return std::vector<uint16_t>(all.begin() + r0 * hidden, all.begin() + r1 * hidden);
}

// ---- 1. GemmaAttnLayer alone ---------------------------------------------------------------------------------
std::vector<uint16_t> RunAttnAlone(GemmaModel& m, int layer, const std::vector<uint16_t>& normed_rows, int T,
                                   attention::GemmaKvDtype dt, attention::GemmaAttnBackend backend) {
  const GemmaConfig& cfg = m.Config();
  const GemmaLayerWeights& lw = m.GetContainer().Layer(layer);
  const int hidden = static_cast<int>(cfg.hidden_size);
  const bool full = cfg.IsFullLayer(layer);
  attention::GemmaAttnConfig ac;
  ac.hidden = hidden;
  ac.heads = static_cast<int>(cfg.num_attention_heads);
  ac.kv_heads = static_cast<int>(cfg.NumKvHeads(layer));
  ac.head_dim = static_cast<int>(cfg.HeadDim(layer));
  ac.rope_pairs = static_cast<int>(cfg.RotaryAngles(layer));
  ac.rope_freq_dim = ac.head_dim;
  ac.rope_theta = static_cast<float>(cfg.RopeTheta(layer));
  ac.rms_eps = static_cast<float>(cfg.rms_norm_eps);
  ac.window = full ? 0 : static_cast<int>(cfg.sliding_window);
  ac.k_eq_v = full;
  attention::GemmaAttnLayer layer_op(ac, backend);

  attention::SlidingRingGeometry geo(static_cast<int>(cfg.sliding_window), 288, 16, T + 16);
  attention::SlidingBlockTable table(geo);
  attention::GemmaKvCache kv = full ? attention::GemmaKvCache::Contiguous(ac.kv_heads, ac.head_dim, T + 16, dt)
                                    : attention::GemmaKvCache::Ring(ac.kv_heads, ac.head_dim, geo, table.Data(), table.Entries(), dt);
  attention::GemmaAttnWeights w;
  w.q = &lw.attn.q;
  w.k = &lw.attn.k;
  w.v = full ? nullptr : &lw.attn.v;
  w.o = &lw.attn.o;
  w.q_norm = lw.attn.q_norm.data();
  w.k_norm = lw.attn.k_norm.data();
  w.k_descale = lw.attn.k_descale.data();
  w.v_descale = lw.attn.v_descale.data();

  core::Stream stream;
  core::Arena arena;
  arena.Reserve(128ull << 20);
  core::DeviceBuffer<uint16_t> x(normed_rows.size()), out(static_cast<size_t>(T) * hidden);
  core::DeviceBuffer<int32_t> pos(T), slots(T), seqused(1);
  x.CopyFromHost(normed_rows);
  std::vector<int32_t> ph(T);
  for (int t = 0; t < T; ++t) ph[t] = t;
  pos.CopyFromHost(ph);
  const std::vector<int32_t> rs = geo.Slots(0, T);
  slots.CopyFromHost(full ? ph : rs);
  const int32_t su = T;
  seqused.CopyFromHost(&su, 1);
  layer_op.Forward(arena, x.data(), out.data(), w, kv, T, 0, pos.data(), slots.data(), seqused.data(), stream.get());
  stream.Synchronize();
  return out.CopyToHost();
}

void TestLayerComponent(const Golden& g, GemmaModel& m, const std::string& name, attention::GemmaKvDtype dt,
                        double extra_tol) {
  const json& meta = g.Component(name);
  const int layer = meta.at("layer").get<int>();
  const int prefill = meta.at("prefill").get<int>(), decode = meta.at("decode").get<int>();
  const int64_t hidden = m.Config().hidden_size;
  const auto r = g.Open(name);
  const std::vector<uint16_t> x = r4dx_test::ReadGoldenRawBf16(r, "x");
  const std::vector<float> want_out = r4dx_test::ReadGoldenAsFloat(r, "layer_out");
  const std::string tag = name + (dt == attention::GemmaKvDtype::kBf16 ? " [bf16 KV]" : " [fp8 KV]");

  // 1. the attention alone, prefill rows
  if (r.Has("input_ln") && r.Has("attn_out")) {
    const std::vector<uint16_t> normed = Rows(r4dx_test::ReadGoldenRawBf16(r, "input_ln"), hidden, 0, prefill);
    const std::vector<float> want_attn = r4dx_test::WidenBf16(Rows(r4dx_test::ReadGoldenRawBf16(r, "attn_out"), hidden, 0, prefill));
    const auto got = RunAttnAlone(m, layer, normed, prefill, dt, attention::GemmaAttnBackend::kReference);
    const double e = RelL2(r4dx_test::WidenBf16(got), want_attn);
    const double tol = std::max(FloorTol(meta, "attn_out"), 0.0) + extra_tol;
    std::printf("  %s attention alone (reference kernel): rel L2 %.3e (tol %.3e)\n", tag.c_str(), e, tol);
    Check(e <= tol, tag + ": GemmaAttnLayer o_proj output vs the golden attn_out");
    if (dt == attention::GemmaKvDtype::kFp8 && meta.at("layer_type").get<std::string>() == "sliding_attention") {
      const auto r4d = RunAttnAlone(m, layer, normed, prefill, dt, attention::GemmaAttnBackend::kLibr4d);
      const double d = RelL2(r4dx_test::WidenBf16(r4d), r4dx_test::WidenBf16(got));
      std::printf("  %s libr4d windowed kernel vs the reference kernel: rel L2 %.3e\n", tag.c_str(), d);
      Check(d < 1e-2, tag + ": the libr4d sliding kernel agrees with the stage-0 reference kernel");
    }
  }

  // 2. the whole layer: a prefill chunk, then the decode rows as a second chunk
  const std::vector<uint16_t> a = m.DebugLayerForward(layer, Rows(x, hidden, 0, prefill), prefill, 0);
  const std::vector<uint16_t> b = m.DebugLayerForward(layer, Rows(x, hidden, prefill, prefill + decode), decode, prefill);
  const double tol = FloorTol(meta, "layer_out") + extra_tol;
  const auto want_a = std::vector<float>(want_out.begin(), want_out.begin() + prefill * hidden);
  const auto want_b = std::vector<float>(want_out.begin() + prefill * hidden, want_out.begin() + (prefill + decode) * hidden);
  const double ea = RelL2(r4dx_test::WidenBf16(a), want_a), eb = RelL2(r4dx_test::WidenBf16(b), want_b);
  std::printf("  %s layer_out: prefill rows rel L2 %.3e, decode rows %.3e (tol %.3e)\n", tag.c_str(), ea, eb, tol);
  Check(ea <= tol, tag + ": layer_out of the prefill chunk");
  Check(eb <= tol, tag + ": layer_out of the decode rows (KV written by the first chunk)");
}

void TestRingWrap(const Golden& g, GemmaModel& m, attention::GemmaKvDtype dt, double extra_tol) {
  if (!g.Has("sliding_ring_wrap")) return;
  const json& meta = g.Component("sliding_ring_wrap");
  const int layer = meta.at("layer").get<int>(), T = meta.at("T").get<int>(), decode = meta.at("decode").get<int>();
  const int64_t hidden = m.Config().hidden_size;
  const auto r = g.Open("sliding_ring_wrap");
  const std::vector<uint16_t> x = r4dx_test::ReadGoldenRawBf16(r, "x");
  const std::vector<float> want = r4dx_test::ReadGoldenAsFloat(r, "layer_out");
  std::vector<uint16_t> last;
  const int64_t body = T - decode;
  for (int64_t pos = 0; pos < body;) {  // the prefill, in 256-row chunks: the ring wraps inside it
    const int64_t n = std::min<int64_t>(256, body - pos);
    m.DebugLayerForward(layer, Rows(x, hidden, pos, pos + n), n, pos);
    pos += n;
  }
  last = m.DebugLayerForward(layer, Rows(x, hidden, body, T), decode, body);  // the decode rows
  const std::vector<float> want_tail(want.begin() + static_cast<int64_t>(T - decode) * hidden, want.end());
  const double e = RelL2(r4dx_test::WidenBf16(last), want_tail);
  // The ring component carries no fp32_floor of its own; use the floor of the same layer's short golden.
  const json& floor_meta =
      meta.contains("fp32_floor") || !g.Has("layer_000_sliding") ? meta : g.Component("layer_000_sliding");
  const double tol = FloorTol(floor_meta, "layer_out") + extra_tol + 5e-3;
  std::printf("  ring wrap (%d positions, ring 1536): last %d rows rel L2 %.3e (tol %.3e)\n", T, decode, e, tol);
  Check(e <= tol, std::string("sliding_ring_wrap") + (dt == attention::GemmaKvDtype::kBf16 ? " [bf16 KV]" : " [fp8 KV]") +
                      ": the rows past the ring wrap match the golden");
}

void TestEmbedAndFinal(const Golden& g, GemmaModel& m) {
  const int64_t hidden = m.Config().hidden_size;
  core::Stream stream;
  if (g.Has("embed_scale")) {
    const auto r = g.Open("embed_scale");
    const std::vector<uint16_t> rows = r4dx_test::ReadGoldenRawBf16(r, "embed_rows");
    const std::vector<uint16_t> want = r4dx_test::ReadGoldenRawBf16(r, "embed_scaled");
    const int n = static_cast<int>(rows.size() / hidden);
    core::DeviceBuffer<uint16_t> table(rows.size()), out(rows.size());
    core::DeviceBuffer<int32_t> ids(n);
    table.CopyFromHost(rows);
    std::vector<int32_t> ih(n);
    for (int i = 0; i < n; ++i) ih[i] = i;
    ids.CopyFromHost(ih);
    r4dx_embedding_gather_scaled_bf16(P(table.data()), P(ids.data()), P(out.data()), n, hidden, n, m.Config().EmbedScale(),
                                       P(stream.get()));
    stream.Synchronize();
    Check(out.CopyToHost() == want, "embed_scale: the scaled gather is bit-exact against HF (rows * bf16(sqrt(hidden)) = 62.0)");
  }
  if (g.Has("final_norm_softcap")) {
    const auto r = g.Open("final_norm_softcap");
    const std::vector<uint16_t> x = r4dx_test::ReadGoldenRawBf16(r, "x");
    const std::vector<float> normed_want = r4dx_test::ReadGoldenAsFloat(r, "normed");
    const int n = static_cast<int>(x.size() / hidden);
    core::DeviceBuffer<uint16_t> xd(x.size()), nd(x.size());
    xd.CopyFromHost(x);
    r4dx_rmsnorm_plain_bf16(P(xd.data()), P(m.GetContainer().FinalNorm().data()), P(nd.data()), n, hidden,
                            static_cast<float>(m.Config().rms_norm_eps), 0, P(stream.get()));
    stream.Synchronize();
    const double e = RelL2(r4dx_test::WidenBf16(nd.CopyToHost()), normed_want);
    std::printf("  final norm rel L2 %.3e\n", e);
    Check(e < 5e-3, "final_norm_softcap: the final RMSNorm (plain weight) matches HF");
    // The softcap probe: cap * tanh(bf16(x) / cap), fp32 math.
    const std::vector<float> probe = r4dx_test::ReadGoldenAsFloat(r, "cap_probe_in");
    std::vector<uint16_t> pb(probe.size());
    for (size_t i = 0; i < probe.size(); ++i) pb[i] = core::FloatToBf16(probe[i]);
    core::DeviceBuffer<uint16_t> pin(pb.size());
    core::DeviceBuffer<float> pout(pb.size());
    pin.CopyFromHost(pb);
    const float cap = static_cast<float>(m.Config().final_logit_softcapping);
    r4dx_model_widen_softcap_bf16_to_f32(P(pin.data()), P(pout.data()), static_cast<int64_t>(pb.size()), cap, P(stream.get()));
    stream.Synchronize();
    const std::vector<float> got = pout.CopyToHost();
    double worst = 0;
    for (size_t i = 0; i < pb.size(); ++i) {
      const float want = cap * std::tanh(core::Bf16ToFloat(pb[i]) / cap);
      worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want)));
    }
    std::printf("  softcap probe worst abs err vs std::tanh: %.3e\n", worst);
    Check(worst < 1e-3, "final_norm_softcap: widen_softcap = cap * tanh(x / cap) over [-200, 200]");
  }
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_gemma_attn_layer", []() -> int {
    const std::string dir = std::getenv("R4DX_GEMMA_GOLDEN_DIR") != nullptr ? std::getenv("R4DX_GEMMA_GOLDEN_DIR")
                                                                              : std::string(R4DX_GEMMA_GOLDEN_DIR);
    const std::string container = std::getenv("R4DX_GEMMA_BF16_CONTAINER") != nullptr
                                      ? std::getenv("R4DX_GEMMA_BF16_CONTAINER")
                                      : std::string("D:\\models\\r4dx\\huihui-gemma\\bf16.r4dx");
    if (!r4dx_test::FileExists(dir + "/manifest.json")) return r4dx_test::SkipMissing(dir + "/manifest.json (M0-8 layer_golden_gemma.py)");
    if (!r4dx_test::FileExists(container)) return r4dx_test::SkipMissing(container);

    Golden g;
    g.dir = dir;
    {
      std::ifstream f(dir + "/manifest.json");
      g.manifest = json::parse(f);
    }
    int max_layer = 0;
    for (const char* name : {"layer_000_sliding", "layer_005_full", "sliding_ring_wrap"}) {
      if (g.Has(name)) max_layer = std::max(max_layer, g.Component(name).at("layer").get<int>());
    }
    for (const bool bf16_kv : {true, false}) {
      GemmaModelOptions o;
      o.container_path = container;
      o.layer_limit = max_layer + 1;
      o.kv = bf16_kv ? GemmaKvMode::kBf16 : GemmaKvMode::kFp8;
      o.max_ctx = 4096;
      GemmaModel m = GemmaModel::Load(o);
      const attention::GemmaKvDtype dt = bf16_kv ? attention::GemmaKvDtype::kBf16 : attention::GemmaKvDtype::kFp8;
      const double extra = bf16_kv ? 0.0 : kFp8KvTol;
      std::printf("---- KV %s ----\n", bf16_kv ? "bf16" : "fp8");
      for (const char* name : {"layer_000_sliding", "layer_005_full"}) {
        if (g.Has(name)) TestLayerComponent(g, m, name, dt, extra);
      }
      TestRingWrap(g, m, dt, extra);
      if (bf16_kv) TestEmbedAndFinal(g, m);
    }
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
