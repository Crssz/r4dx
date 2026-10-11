// qwen4_exp on REAL weights, first four layers (GDN, GDN+PLE, GDN, QSA) against the real-model layer goldens
// (E:\models\r4dx\q4exp\ref\layer_goldens\T70{,eos}.recent.safetensors, tools/reference/qwen4exp/real_ref.py; HF's own decoder layers
// on the AWQ checkpoint dequantized to bf16). Links r4dx_model. Exit 77 = a fixture is missing.
//
// Fixtures:
//   R4DX_Q4EXP_REAL_PREFIX    default D:\q4exp\real-prefix4-bf16.r4dx: the PREFIX container
//                             `r4dx-convert --layers 4 --layouts bf16 --lm-head w4a16 --w4a16-group-rule ^lm_head$=32
//                              --kv-calib E:\models\r4dx\q4exp\ref\kv_calib.json --ngram-sidecar <sidecar>` (__metadata__.layers_present 4;
//                             every body linear and every expert bf16, so the weights are bit-equal to what the reference used)
//   R4DX_Q4EXP_REAL_GOLDENS   default E:\models\r4dx\q4exp\ref\layer_goldens
//   the n-gram sidecar named by the container's metadata (95 GiB, read unbuffered, 16 rows per token)
//
// What runs (device = TestDevice(): physical device 1 under ctest's HIP_VISIBLE_DEVICES=1):
//   Phase A  TEACHER FORCED: each layer alone on the golden's residual_in (and, for the PLE layer, the golden's ple_embedding rows):
//            what one layer computes from exactly the reference's input.
//   Phase B  FREE RUNNING: Qwen4ExpModel over layers 0..3 from the token ids (embedding gather, n-gram path, chunking, all layers).
//   Both phases in BOTH GDN prefill modes -- libr4d's chunked scan (the production path) and the recurrent decode kernels (the
//   accurate reference path) -- and for both goldens: T70 (70 tokens) and T70eos (the same with EOS 248044 at positions 17 and 41,
//   which exercises the n-gram history segmentation of the PLE layer).
// Every capture the golden also has is compared (max|err| and mean|err| relative to the golden tensor's rms); the router's top-10 ids
// are compared as sets with the golden's own router logits deciding which differences are near-ties. T = 70 is below the dense limit
// of the QSA layer (70 < 2051 tokens), so layer 3's attention is exact against the golden.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "q4exp_model_util.h"
#include "qwen4exp_container.h"
#include "qwen4exp_stage.h"
#include "r4dx/core/pinned_buffer.hpp"

using namespace q4m;
using r4dx::model::Layout;
using r4dx::model::Qwen4ExpChunk;
using r4dx::model::Qwen4ExpContainer;
using r4dx::model::Qwen4ExpHcRefs;
using r4dx::model::Qwen4ExpLayerRefs;
using r4dx::model::Qwen4ExpLoadOptions;
using r4dx::model::Qwen4ExpPleRefs;
using r4dx::model::Qwen4ExpStage;
using r4dx::model::Qwen4ExpStageOptions;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

constexpr int kT = 70;
constexpr int kLayers = 4;

std::string PrefixPath() {
  const char* e = std::getenv("R4DX_Q4EXP_REAL_PREFIX");
  return e != nullptr && e[0] != 0 ? e : "D:\\q4exp\\real-prefix4-bf16.r4dx";
}
std::string GoldenDir() {
  const char* e = std::getenv("R4DX_Q4EXP_REAL_GOLDENS");
  return e != nullptr && e[0] != 0 ? e : "E:\\models\\r4dx\\q4exp\\ref\\layer_goldens";
}
std::string GoldenFile(const std::string& stem) { return GoldenDir() + "\\" + stem + ".recent.safetensors"; }

// ---- bounds -----------------------------------------------------------------------------------------------------------------------
// max|err| / rms(ref) and mean|err| / rms(ref) of a whole tensor (all 70 rows). The model computes in bf16 with accumulation orders that
// differ from torch's (a bf16 ulp is 0.8% of a value) and the residual / MoE outputs have a few outlier rows and elements, so these are
// noise bounds, set from what the runs on this container observed (worst over both goldens and both GDN modes; the observed worst values
// are in docs/qwen4exp.md "Real-weights prefix") with a margin of about 1.3-1.5x, not ulp bounds: a wrong layout, scale or gate reads as
// errors of several rms (the shipped GDN scan, on this checkpoint, read 5.4 / 0.34 at layer 0 attn_out).
struct Tol {
  double max_norm, mean_norm;
};

bool Has(const std::string& name, const char* s) { return name.find(s) != std::string::npos; }
bool IsGdnLayer(const std::string& name) { return std::atoi(name.c_str() + 5) % 4 != 3; }  // "layerN.": layers 3, 7, ... are QSA

// Teacher forced: one layer from the golden's own input.
Tol TfTol(const std::string& name) {
  if (Has(name, "inject")) return {0.05, 0.003};
  if (Has(name, "attn_pre_o")) return {1.3, 0.025};
  if (Has(name, "moe_out")) return {1.4, 0.06};
  if (Has(name, "attn_out")) return IsGdnLayer(name) ? Tol{0.3, 0.012} : Tol{0.7, 0.04};
  if (Has(name, "residual")) return {1.1, 0.007};
  return {0.6, 0.02};  // the hyper-connection mixes
}
// Free running: four layers of accumulated noise (and an expert swapped at a router near-tie moves a row by its routing weight).
Tol FrTol(const std::string& name) {
  if (Has(name, "inject")) return {0.15, 0.006};
  if (Has(name, "attn_pre_o")) return {1.3, 0.03};
  if (Has(name, "moe_out")) return {1.8, 0.08};
  if (Has(name, "attn_out")) return {0.8, 0.05};
  if (Has(name, "residual")) return {1.1, 0.015};
  return {0.7, 0.03};
}
// The two GDN prefill modes on IDENTICAL input (teacher forced, layers 0-2): the chunked scan must equal the recurrent kernels to noise.
bool IsCvrChecked(const std::string& name) { return IsGdnLayer(name) && Has(name, "attn_out"); }
constexpr Tol kCvrTol = {0.25, 0.01};

// ---- the router's expert sets ---------------------------------------------------------------------------------------------------------
// top-10 of 512 router logits that are bf16 numbers: a row whose 10th / 11th logits are within bf16 noise of each other may legitimately
// pick the other one. The golden's own router logits tell which rows those are (distance to the cut in units of the row's logit rms).
struct TopkReport {
  int rows = 0, order_rows = 0, set_rows = 0, unexplained = 0;
  double worst = 0;
};

TopkReport CompareTopk(const Captured& cap, const Golden& g, const std::string& name) {
  TopkReport r;
  int64_t n = 0, nl = 0;
  const int64_t* ref = g.I64(name, &n);
  const int rows = static_cast<int>(cap.rows), k = static_cast<int>(cap.cols);
  const std::string rl_name = name.substr(0, name.find('.')) + ".router_logits";
  const uint16_t* rl = g.Has(rl_name) ? g.Bf16(rl_name, &nl) : nullptr;
  const int E = rl != nullptr ? static_cast<int>(nl / rows) : 0;
  r.rows = rows;
  for (int row = 0; row < rows; ++row) {
    const int32_t* mine = cap.i32.data() + static_cast<size_t>(row) * k;
    const int64_t* hf = ref + static_cast<int64_t>(row) * k;
    bool same_order = true;
    std::vector<int> only_mine, only_hf;
    for (int j = 0; j < k; ++j) {
      same_order &= mine[j] == hf[j];
      bool in_hf = false, in_mine = false;
      for (int q = 0; q < k; ++q) {
        in_hf |= mine[j] == hf[q];
        in_mine |= hf[j] == mine[q];
      }
      if (!in_hf) only_mine.push_back(mine[j]);
      if (!in_mine) only_hf.push_back(static_cast<int>(hf[j]));
    }
    r.order_rows += same_order ? 0 : 1;
    if (only_mine.empty() && only_hf.empty()) continue;
    ++r.set_rows;
    bool explained = rl != nullptr;
    if (rl != nullptr) {
      std::vector<float> l(static_cast<size_t>(E));
      for (int e = 0; e < E; ++e) l[static_cast<size_t>(e)] = Bf(rl[static_cast<int64_t>(row) * E + e]);
      std::vector<float> sorted = l;
      std::sort(sorted.begin(), sorted.end(), std::greater<float>());
      const float cut = sorted[static_cast<size_t>(k - 1)];
      double sq = 0;
      for (float v : l) sq += static_cast<double>(v) * v;
      const double row_rms = std::sqrt(sq / static_cast<double>(E));
      for (const std::vector<int>* side : {&only_mine, &only_hf}) {
        for (int e : *side) {
          const double d = std::fabs(l[static_cast<size_t>(e)] - cut) / (row_rms > 0 ? row_rms : 1.0);
          r.worst = std::max(r.worst, d);
          explained &= d <= 0.03;
        }
      }
    }
    r.unexplained += explained ? 0 : 1;
  }
  return r;
}

// ---- one run's captures against one golden ----------------------------------------------------------------------------------------------
struct PointStats {
  std::string name;
  Stats s;
  bool have = false;
};

// Layer-major, capture-order-independent: every bf16 capture the golden also has, restricted to layers [0, kLayers).
std::vector<std::string> ComparableNames(const CaptureStore& store, const Golden& g) {
  std::vector<std::string> out;
  for (const auto& [name, cap] : store.all()) {
    if (name.rfind("layer", 0) != 0) continue;
    const int layer = std::atoi(name.c_str() + 5);
    if (layer < 0 || layer >= kLayers || !g.Has(name)) continue;
    out.push_back(name);
  }
  std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
    const int la = std::atoi(a.c_str() + 5), lb = std::atoi(b.c_str() + 5);
    return la != lb ? la < lb : a < b;
  });
  return out;
}

Stats StatsOf(const Captured& cap, const Golden& g, const std::string& name) {
  int64_t n = 0;
  const uint16_t* ref = g.Bf16(name, &n);
  if (static_cast<int64_t>(cap.bf16.size()) != n) {
    std::fprintf(stderr, "FAIL: %s has %lld elements, golden %lld\n", name.c_str(), static_cast<long long>(cap.bf16.size()),
                 static_cast<long long>(n));
    ++g_failures;
    Stats bad;
    bad.nan = true;
    return bad;
  }
  return CompareBf16(cap.bf16.data(), ref, n);
}

// The joint table of one (phase, golden): per point the chunked and the recurrent GDN prefill against the golden, and chunked against
// recurrent. Bounds apply to both modes.
void Report(const std::string& title, const CaptureStore& chunked, const CaptureStore& recurrent, const Golden& g,
            Tol (*tol)(const std::string&), bool check_cvr) {
  std::printf("\n==== %s ====\n", title.c_str());
  std::printf("  %-24s | %-27s | %-27s | %s\n", "point", "chunked  max/rms  mean/rms", "recurrent max/rms mean/rms",
              "chunked vs recurrent");
  for (const std::string& name : ComparableNames(recurrent, g)) {
    const Captured& rc = recurrent.Get(name);
    if (rc.dtype == Qwen4ExpCapture::Dtype::kI32) {
      const TopkReport tr = CompareTopk(rc, g, name);
      TopkReport tc;
      if (chunked.Has(name)) tc = CompareTopk(chunked.Get(name), g, name);
      std::printf("  %-24s | expert-set rows differing: %2d/%d (%d not near-ties) | %2d/%d (%d not near-ties) | worst gap to the cut %.4f / %.4f of the row logit rms\n",
                  name.c_str(), tc.set_rows, tc.rows, tc.unexplained, tr.set_rows, tr.rows, tr.unexplained, tc.worst, tr.worst);
      Check(tr.unexplained == 0, title + " recurrent " + name + ": expert sets differ from the golden beyond router near-ties");
      Check(tc.unexplained == 0, title + " chunked " + name + ": expert sets differ from the golden beyond router near-ties");
      continue;
    }
    const Stats sr = StatsOf(rc, g, name);
    Stats sc;
    sc.nan = true;
    double cvr_max = -1, cvr_mean = -1;
    if (chunked.Has(name)) {
      const Captured& cc = chunked.Get(name);
      sc = StatsOf(cc, g, name);
      if (cc.bf16.size() == rc.bf16.size()) {
        const Stats d = CompareBf16(cc.bf16.data(), rc.bf16.data(), static_cast<int64_t>(cc.bf16.size()));
        cvr_max = d.max_norm;
        cvr_mean = d.mean_norm;
      }
    }
    std::printf("  %-24s | %9.4f %9.5f%s | %9.4f %9.5f%s | %9.4f %9.5f   (rms ref %.4g)\n", name.c_str(), sc.max_norm,
                sc.mean_norm, sc.nan ? " NaN!" : "     ", sr.max_norm, sr.mean_norm, sr.nan ? " NaN!" : "     ", cvr_max, cvr_mean,
                sr.rms_ref);
    const Tol t = tol(name);
    Check(!sr.nan && sr.max_norm <= t.max_norm && sr.mean_norm <= t.mean_norm,
          title + " recurrent " + name + ": error above its bound (max/rms " + std::to_string(sr.max_norm) + ", mean/rms " +
              std::to_string(sr.mean_norm) + "; bound " + std::to_string(t.max_norm) + " / " + std::to_string(t.mean_norm) + ")");
    Check(!sc.nan && sc.max_norm <= t.max_norm && sc.mean_norm <= t.mean_norm,
          title + " chunked " + name + ": error above its bound (max/rms " + std::to_string(sc.max_norm) + ", mean/rms " +
              std::to_string(sc.mean_norm) + "; bound " + std::to_string(t.max_norm) + " / " + std::to_string(t.mean_norm) + ")");
    if (check_cvr && IsCvrChecked(name)) {
      Check(cvr_max >= 0 && cvr_max <= kCvrTol.max_norm && cvr_mean <= kCvrTol.mean_norm,
            title + " " + name + ": chunked vs recurrent GDN prefill differ (max/rms " + std::to_string(cvr_max) + ", mean/rms " +
                std::to_string(cvr_mean) + "; bound " + std::to_string(kCvrTol.max_norm) + " / " + std::to_string(kCvrTol.mean_norm) + ")");
    }
  }
}

// Where in the sequence a GDN layer's output goes wrong: the error of attn_out per block of rows (a 64-row scan chunk boundary
// sits between rows 63 and 64; the tiny model's chunked error showed up in rows 0..31 of a longer call).
void RowProfile(const std::string& title, const CaptureStore& chunked, const CaptureStore& recurrent, const Golden& g) {
  std::printf("\n-- %s: attn_out error per row block (max|err| / rms of the whole tensor)\n", title.c_str());
  std::printf("  %-18s %s\n", "", "rows 0-15 | 16-31 | 32-47 | 48-63 | 64-69");
  for (int layer : {0, 1, 2}) {
    const std::string name = "layer" + std::to_string(layer) + ".attn_out";
    if (!g.Has(name) || !recurrent.Has(name) || !chunked.Has(name)) continue;
    int64_t n = 0;
    const uint16_t* ref = g.Bf16(name, &n);
    const int64_t cols = n / kT;
    const Stats whole = CompareBf16(recurrent.Get(name).bf16.data(), ref, n);
    const std::array<std::pair<const char*, const CaptureStore*>, 2> modes = {
        std::make_pair("chunked", &chunked), std::make_pair("recurrent", &recurrent)};
    for (const auto& [label, st] : modes) {
      const Captured& c = st->Get(name);
      std::printf("  layer%d %-9s ", layer, label);
      const int edges[] = {0, 16, 32, 48, 64, kT};
      for (int b = 0; b < 5; ++b) {
        double mx = 0;
        for (int64_t i = edges[b] * cols; i < edges[b + 1] * cols; ++i) {
          mx = std::max(mx, std::fabs(static_cast<double>(Bf(c.bf16[static_cast<size_t>(i)])) - Bf(ref[i])));
        }
        std::printf(" %7.4f", whole.rms_ref > 0 ? mx / whole.rms_ref : 0.0);
      }
      std::printf("\n");
    }
  }
}

// ---- Phase A: one layer from the golden's input ----------------------------------------------------------------------------------
Qwen4ExpHcRefs HcRefsOf(const r4dx::model::qwen4exp::HcWeights& w) {
  Qwen4ExpHcRefs r;
  r.norm = w.norm.data();
  r.down = w.down.data();
  r.up = w.up.data();
  r.inject = w.inject.empty() ? nullptr : w.inject.data();
  return r;
}

// The borrowed pointers of one resident layer (a copy of qwen4exp_model.cpp's LayerRefs: the stage header deliberately does not see the
// weight-owning one, and the model keeps its helper private).
Qwen4ExpLayerRefs LayerRefsOf(const Qwen4ExpContainer& c, int64_t i) {
  const r4dx::model::qwen4exp::LayerWeights& lw = c.Layer(i);
  Qwen4ExpLayerRefs r;
  r.index = i;
  r.qsa = lw.qsa;
  r.hc_attn = HcRefsOf(lw.hc_attn);
  r.hc_mlp = HcRefsOf(lw.hc_mlp);
  if (!lw.qsa) {
    const auto& g = *lw.gdn;
    r.gdn.in_proj_qkv = &g.in_proj_qkv;
    r.gdn.in_proj_z = &g.in_proj_z;
    r.gdn.in_proj_a = g.in_proj_a.data();
    r.gdn.in_proj_b = g.in_proj_b.data();
    r.gdn.conv1d_weight = g.conv1d_weight.data();
    r.gdn.A_log = g.A_log.data();
    r.gdn.dt_bias = g.dt_bias.data();
    r.gdn.norm_weight = g.norm_weight.data();
    r.gdn.out_proj = &g.out_proj;
  } else {
    const auto& a = *lw.attn;
    r.attn.qg = &a.qg;
    r.attn.k = &a.k;
    r.attn.v = &a.v;
    r.attn.o = &a.o;
    r.attn.q_norm = a.q_norm.data();
    r.attn.k_norm = a.k_norm.data();
    r.attn.indexer_qk = a.indexer_qk.data();
    r.attn.indexer_q_norm = a.indexer_q_norm.data();
    r.attn.indexer_k_norm = a.indexer_k_norm.data();
    r.attn.k_descale = a.k_descale.data();
    r.attn.v_descale = a.v_descale.data();
  }
  const auto& mw = lw.moe;
  r.moe.router = mw.router.data();
  r.moe.shared_gate_proj = &mw.shared.gate;
  r.moe.shared_up_proj = &mw.shared.up;
  r.moe.shared_down_proj = &mw.shared.down;
  r.moe.shared_gate = mw.shared_gate.data();
  for (const auto& e : mw.experts) {
    r.moe.expert_gate_up.push_back(&e.gate_up);
    r.moe.expert_down.push_back(&e.down);
  }
  if (lw.ple) {
    const auto& p = *lw.ple;
    Qwen4ExpPleRefs pr;
    pr.key_proj = p.key_proj.data();
    pr.value_proj = p.value_proj.data();
    pr.norm_key = p.norm_key.data();
    pr.norm_query = p.norm_query.data();
    pr.norm_conv = p.norm_conv.data();
    pr.conv1d = p.conv1d.data();
    r.ple = pr;
  }
  return r;
}

struct GoldenSet {
  std::string stem;
  std::unique_ptr<Golden> g;
};

// results[golden][mode] (mode 0 chunked, 1 recurrent)
using Stores = std::map<std::string, std::array<CaptureStore, 2>>;

void RunTeacherForced(const std::string& path, std::vector<GoldenSet>& goldens, Stores& out) {
  Qwen4ExpLoadOptions lo;
  lo.layout = Layout::kBf16;
  lo.device = TestDevice();
  lo.layer_first = 0;
  lo.layer_last = kLayers;
  const auto t0 = std::chrono::steady_clock::now();
  Qwen4ExpContainer c = Qwen4ExpContainer::Load(path, lo);
  std::printf("[phase A] container loaded onto device %d: layers [%lld, %lld), %.2f GiB, %.1f s\n", c.Device(),
              static_cast<long long>(c.FirstLayer()), static_cast<long long>(c.EndLayer()),
              static_cast<double>(c.DeviceBytes()) / (1024.0 * 1024.0 * 1024.0),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  const auto& cfg = c.Config();
  const int64_t wide = cfg.hc_count * cfg.hidden_size;
  r4dx::core::PinnedBuffer<uint16_t> resid(static_cast<size_t>(kT * wide));
  r4dx::core::PinnedBuffer<uint16_t> ple_rows(static_cast<size_t>(kT * cfg.ple_embed_dim));

  for (int layer = 0; layer < kLayers; ++layer) {
    Qwen4ExpStageOptions so;
    so.device = c.Device();
    so.config = cfg;
    so.max_ctx = 4096;
    so.max_rows = 256;
    Qwen4ExpStage stage(so, {LayerRefsOf(c, layer)}, std::nullopt);
    for (GoldenSet& gs : goldens) {
      const std::string in_name = "layer" + std::to_string(layer) + ".residual_in";
      int64_t n = 0;
      const uint16_t* rin = gs.g->Bf16(in_name, &n);
      Check(n == kT * wide, gs.stem + " " + in_name + ": unexpected size");
      const bool is_ple = cfg.IsPleLayer(layer);
      if (is_ple) {
        int64_t m = 0;
        const uint16_t* pe = gs.g->Bf16("layer" + std::to_string(layer) + ".ple_embedding", &m);
        Check(m == kT * cfg.ple_embed_dim, gs.stem + " ple_embedding: unexpected size");
        std::memcpy(ple_rows.data(), pe, static_cast<size_t>(m) * 2);
      }
      for (int mode = 0; mode < 2; ++mode) {
        CaptureStore& store = out[gs.stem][static_cast<size_t>(mode)];
        stage.Reset();
        stage.SetCapture(store.Fn());
        std::memcpy(resid.data(), rin, static_cast<size_t>(n) * 2);
        stage.LoadResidual(resid.data(), kT);
        Qwen4ExpChunk chunk;
        chunk.rows = kT;
        chunk.first_pos = 0;
        chunk.is_prefill = true;
        chunk.gdn_prefill_recurrent = mode == 1;
        if (is_ple) chunk.ple_rows = [&]() -> const uint16_t* { return ple_rows.data(); };
        stage.RunLayers(chunk);
        stage.SetCapture(nullptr);
      }
    }
  }
}

// ---- Phase B: the model over layers 0..3 -------------------------------------------------------------------------------------------------
void RunFreeRunning(const std::string& path, std::vector<GoldenSet>& goldens, Stores& out) {
  for (int mode = 0; mode < 2; ++mode) {
    r4dx::model::Qwen4ExpModelOptions o;
    o.container_path = path;
    o.layout = Layout::kBf16;
    o.max_ctx = 4096;
    o.devices = {TestDevice(), TestDevice()};
    o.split = 0;
    o.gdn_prefill_recurrent = mode == 1;
    const auto t0 = std::chrono::steady_clock::now();
    auto model = Qwen4ExpModel::Load(o);
    std::printf("[phase B] model loaded (GDN prefill %s): %d stage(s), %.1f s\n", mode == 1 ? "recurrent" : "chunked", model->NumStages(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    Check(model->NumStages() == 1 && model->StageFirstLayer(0) == 0 && model->StageEndLayer(0) == kLayers,
          "the model over a prefix container is one stage over layers [0, 4)");
    for (GoldenSet& gs : goldens) {
      CaptureStore& store = out[gs.stem][static_cast<size_t>(mode)];
      model->Reset();
      model->SetCapture(store.Fn());
      (void)model->Prefill(gs.g->ids());
      model->SetCapture(nullptr);
      const auto& tm = model->LastTimings();
      std::printf("[phase B] %s prefill %.1f ms (stage0 %.1f, ple wait %.2f, head %.1f)\n", gs.stem.c_str(), tm.total_ms, tm.stage_ms[0],
                  tm.ple_wait_ms, tm.head_ms);
    }
  }
}

}  // namespace

int RunTest();
int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}

int RunTest() {
  Unbuffer();
  const std::string path = PrefixPath();
  if (!Exists(path)) {
    std::printf("SKIP: %s missing (r4dx-convert --layers 4 ..., see this file's header)\n", path.c_str());
    return kSkip;
  }
  std::vector<GoldenSet> goldens;
  for (const char* stem : {"T70", "T70eos"}) {
    if (!Exists(GoldenFile(stem))) {
      std::printf("SKIP: %s missing\n", GoldenFile(stem).c_str());
      return kSkip;
    }
    GoldenSet gs;
    gs.stem = stem;
    gs.g = std::make_unique<Golden>(kT, GoldenFile(stem));
    goldens.push_back(std::move(gs));
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  {
    const auto info = Qwen4ExpContainer::Inspect(path);
    std::printf("container %s: layers_present %lld of %lld, %.2f GiB of layer weights + %.2f GiB head\n", path.c_str(),
                static_cast<long long>(info.layers_present), static_cast<long long>(info.config.num_hidden_layers),
                [&] {
                  double t = 0;
                  for (int64_t b : info.PerLayerDeviceBytes()) t += static_cast<double>(b);
                  return t / (1024.0 * 1024.0 * 1024.0);
                }(),
                static_cast<double>(info.TailDeviceBytes()) / (1024.0 * 1024.0 * 1024.0));
    if (info.layers_present != kLayers) {
      std::printf("SKIP: the container holds %lld layers, this test wants a %d-layer prefix\n", static_cast<long long>(info.layers_present),
                  kLayers);
      return kSkip;
    }
    // The prefix-container contract (Qwen4ExpContainer): the per-layer tables cover the present layers only, and a layer range beyond them
    // throws before any upload.
    Check(info.metadata.value("layers_present", int64_t{-1}) == kLayers, "__metadata__.layers_present is 4");
    Check(info.layer_bytes.size() == kLayers && info.gate_up_kb.size() == kLayers && info.down_kb.size() == kLayers,
          "the per-layer tables of Inspect have one entry per PRESENT layer");
    Check(info.config.num_hidden_layers == 48, "the recorded config still says 48 layers");
    for (const std::pair<int64_t, int64_t>& range : {std::pair<int64_t, int64_t>{2, 5}, {4, 5}, {0, 48}}) {
      Qwen4ExpLoadOptions lo;
      lo.layout = Layout::kBf16;
      lo.layer_first = range.first;
      lo.layer_last = range.second;
      bool threw = false;
      std::string what;
      try {
        (void)Qwen4ExpContainer::Load(path, lo);
      } catch (const std::invalid_argument& e) {
        threw = true;
        what = e.what();
      }
      Check(threw && what.find("prefix container") != std::string::npos,
            "Load of layers [" + std::to_string(range.first) + ", " + std::to_string(range.second) + ") is refused naming the prefix: " + what);
    }
  }
  std::printf("device %d (HIP_VISIBLE_DEVICES=%s)\n", TestDevice(), std::getenv("HIP_VISIBLE_DEVICES") ? std::getenv("HIP_VISIBLE_DEVICES") : "unset");

  Stores tf, fr;
  RunTeacherForced(path, goldens, tf);
  RunFreeRunning(path, goldens, fr);

  for (GoldenSet& gs : goldens) {
    Report("teacher forced, " + gs.stem, tf[gs.stem][0], tf[gs.stem][1], *gs.g, TfTol, /*check_cvr=*/true);
    RowProfile("teacher forced, " + gs.stem, tf[gs.stem][0], tf[gs.stem][1], *gs.g);
  }
  for (GoldenSet& gs : goldens) {
    Report("free running, " + gs.stem, fr[gs.stem][0], fr[gs.stem][1], *gs.g, FrTol, /*check_cvr=*/false);
    RowProfile("free running, " + gs.stem, fr[gs.stem][0], fr[gs.stem][1], *gs.g);
  }
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
