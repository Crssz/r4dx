// Layer-isolated golden test of Qwen4ExpModel on the tiny container (docs/qwen4exp-plan.md 4.11, task M1-16): every layer L is run
// ALONE, over the HF golden's `layerL.residual_in` (Qwen4ExpModel::RunLayerIsolated), so its error is its own and not the
// accumulated drift of the layers before it, and compared with the golden's points of that layer (the hyper-connection mixes
// and injection weights, the mixer's output and its o_proj input, the residual after the mixer, the MoE's input, router
// choice and output, the residual after the layer; `residual_ple` of the PLE layer). Prompts T = 1, 7, 70, 300; both GDN prefill
// paths (libr4d's chunked scan = production, the recurrent decode kernels = reference).
//
// The QSA layer (index 3) also gets its SELECTION compared on EVERY row: the golden's `index_mask` [T, T] against the tokens
// QsaLayerCore selected (rows whose query saw <= budget / ratio complete blocks attend densely: the mask must be causal-full
// there). For every row whose selected set differs, the score margin at the cut is computed from the golden's own `index_q` and
// `index_block_keys` (fp32 recompute, the `recent` tie rule) to decide whether it is a legitimate near-tie. Outputs of that
// layer are bounded on the rows whose selection matches EXACTLY (and, for the MoE points, whose expert set also matches); the
// others are reported.
//
// Tolerances (kTol below) are set from the noise this test observes; they bound max|err| / rms(ref) and mean|err| / rms(ref) of
// each point over the checked rows. Fixtures as test_model_q4exp_tiny.cpp (R4DX_Q4EXP_TINY_DIR, exit 77 = skip).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <utility>
#include <string>
#include <vector>

#include "q4exp_model_util.h"

using namespace q4m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Tol {
  double max_norm, mean_norm;
};

// Bounds per capture point against the HF golden, max|err| / rms(ref) and mean|err| / rms(ref) over the checked rows, set to ~1.6x
// the worst value this test observed (layers 0-2 / the QSA layer, T = 1 / 7 / 70 / 300, recurrent GDN prefill; run 2026-10-10):
//   GDN layers (bf16 kernels vs torch's CPU matmuls; a bf16 ulp is 0.4-0.8% of a value):
//     attn_mix 0.030 / 5e-5   attn_inject 0 / 0 (bit-exact)   attn_out 0.043 / 0.0052   residual_attn 0.078 / 0.0048
//     mlp_mix 0.089 / 0.0055   mlp_inject 0.015 / 0.0023   moe_out 0.131 / 0.0125   residual_out 0.077 / 0.0059   residual_ple 0.009 / 2e-8
//   QSA layer: its KV cache is fp8 e4m3 (design, plan 4.6), the golden's is bf16, so the attention output carries the fp8 rounding
//     of K and V (3 mantissa bits, half-ulp 6%): attn_pre_o 0.78 / 0.030 (a CPU recompute with fp8-rounded K, V from the golden's
//     own q / k / v predicts 0.43 / 0.027 on the dense rows and 0.18 / 0.017 at T = 1; observed 0.46 / 0.027 and 0.16 / 0.017),
//     then attn_out 0.29 / 0.033, residual_attn 0.27 / 0.023, mlp_mix 0.24 / 0.025, mlp_inject 0.037 / 0.011, moe_out 0.30 / 0.042,
//     residual_out 0.29 / 0.024. The tight check of the attention itself is the fp8-aware reference (kFp8Ref below).
Tol PointTol(const std::string& point, bool qsa) {
  if (!qsa) {
    if (point == "residual_ple") return {0.02, 1e-5};
    if (point == "attn_mix") return {0.06, 5e-4};
    if (point == "attn_inject") return {1e-9, 1e-9};
    if (point == "attn_out") return {0.08, 0.010};
    if (point == "residual_attn") return {0.13, 0.010};
    if (point == "mlp_mix") return {0.15, 0.011};
    if (point == "mlp_inject") return {0.03, 0.005};
    if (point == "moe_out") return {0.22, 0.022};
    if (point == "residual_out") return {0.13, 0.012};
  } else {
    if (point == "attn_mix") return {0.06, 5e-4};
    if (point == "attn_inject") return {1e-9, 1e-9};
    if (point == "attn_pre_o") return {1.3, 0.05};
    if (point == "attn_out") return {0.5, 0.06};
    if (point == "residual_attn") return {0.45, 0.04};
    if (point == "mlp_mix") return {0.4, 0.04};
    if (point == "mlp_inject") return {0.08, 0.02};
    if (point == "moe_out") return {0.5, 0.07};
    if (point == "residual_out") return {0.45, 0.04};
  }
  throw std::runtime_error("PointTol: no bound for " + point);
}
// Router near-tie: an expert swapped against the golden's top-10 must sit within this fraction of the row's router-logit rms from the
// cut. Observed worst: 0.017 on the GDN layers, 0.083 on the QSA layer (its mlp_mix carries the fp8 noise above, 0.24 max/rms).
double RouterTieTol(bool qsa) { return qsa ? 0.15 : 0.03; }

// ---- row subsets ----------------------------------------------------------------------------------------------------
Stats StatsRows(const uint16_t* got, const uint16_t* ref, int64_t cols, const std::vector<int>& rows) {
  std::vector<uint16_t> g, r;
  g.reserve(rows.size() * static_cast<size_t>(cols));
  r.reserve(rows.size() * static_cast<size_t>(cols));
  for (int row : rows) {
    g.insert(g.end(), got + static_cast<int64_t>(row) * cols, got + static_cast<int64_t>(row + 1) * cols);
    r.insert(r.end(), ref + static_cast<int64_t>(row) * cols, ref + static_cast<int64_t>(row + 1) * cols);
  }
  return CompareBf16(g.data(), r.data(), static_cast<int64_t>(g.size()));
}

std::vector<int> RowsWhere(const std::vector<char>& flag) {
  std::vector<int> r;
  for (size_t i = 0; i < flag.size(); ++i) {
    if (flag[i]) r.push_back(static_cast<int>(i));
  }
  return r;
}
std::vector<char> Not(const std::vector<char>& a) {
  std::vector<char> o(a.size());
  for (size_t i = 0; i < a.size(); ++i) o[i] = !a[i];
  return o;
}
std::vector<char> And(const std::vector<char>& a, const std::vector<char>& b) {
  std::vector<char> o(a.size());
  for (size_t i = 0; i < a.size(); ++i) o[i] = a[i] && b[i];
  return o;
}

// ---- router -----------------------------------------------------------------------------------------------------------
struct RouterResult {
  std::vector<char> same_set;  // per row: the expert SET equals the golden's
  int order_rows = 0, set_rows = 0, unexplained = 0;
  double worst = 0;  // largest distance of a differing expert's logit from the cut, in units of the row's logit rms
};

// A top-10 of the golden's bf16 router logits may legitimately swap a near-tie. Rows in `consider` are checked for that.
RouterResult CheckRouter(const Captured& cap, const Golden& g, const std::string& layer, const std::vector<char>& consider,
                         double tie_tol) {
  RouterResult out;
  int64_t n = 0, nl = 0;
  const int64_t* ref = g.I64(layer + ".topk_ids", &n);
  const uint16_t* rl = g.Bf16(layer + ".router_logits", &nl);
  const int rows = static_cast<int>(cap.rows), k = static_cast<int>(cap.cols);
  const int E = static_cast<int>(nl / rows);
  out.same_set.assign(static_cast<size_t>(rows), 1);
  for (int r = 0; r < rows; ++r) {
    const int32_t* mine = cap.i32.data() + static_cast<size_t>(r) * k;
    const int64_t* hf = ref + static_cast<int64_t>(r) * k;
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
    if (consider[static_cast<size_t>(r)]) out.order_rows += same_order ? 0 : 1;
    if (only_mine.empty() && only_hf.empty()) continue;
    out.same_set[static_cast<size_t>(r)] = 0;
    if (!consider[static_cast<size_t>(r)]) continue;
    ++out.set_rows;
    std::vector<float> l(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) l[static_cast<size_t>(e)] = Bf(rl[static_cast<int64_t>(r) * E + e]);
    std::vector<float> sorted = l;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float cut = sorted[static_cast<size_t>(k - 1)];
    double sq = 0;
    for (float v : l) sq += static_cast<double>(v) * v;
    const double row_rms = std::sqrt(sq / static_cast<double>(E));
    bool explained = true;
    for (const std::vector<int>* side : {&only_mine, &only_hf}) {
      for (int e : *side) {
        const double d = std::fabs(l[static_cast<size_t>(e)] - cut) / (row_rms > 0 ? row_rms : 1.0);
        out.worst = std::max(out.worst, d);
        explained &= d <= tie_tol;
      }
    }
    out.unexplained += explained ? 0 : 1;
  }
  return out;
}

// ---- QSA selection ----------------------------------------------------------------------------------------------------
struct SelectionResult {
  std::vector<char> match;  // per row: the selected token set equals the golden mask's row
  int rows = 0, dense_rows = 0, sparse_rows = 0, differ = 0, differ_beyond_near_tie = 0;
  int golden_vs_recompute = 0, mine_vs_recompute = 0;  // sparse rows whose set differs from the fp32 `recent` recompute
  double worst_rel = 0;                                 // over the differing rows (see below)
  std::vector<double> rel_per_differing_row;
  std::vector<std::string> details;  // one line per differing row: the swapped blocks' scores, the cut gap
};

// Selection of the QSA layer `layer` on every row. `mine` = captured sel_tokens / sel_count; `gold` = the golden's index_mask.
SelectionResult CheckSelection(const Qwen4ExpModel& model, const Golden& g, const std::string& layer, const CaptureStore& store,
                               int T, double near_tie_rel) {
  const auto& cfg = model.Config();
  const int topk = static_cast<int>(cfg.BlockTopK());
  const int ratio = static_cast<int>(cfg.indexer_compress_ratio);
  const int D = static_cast<int>(cfg.indexer_head_dim), H = static_cast<int>(cfg.indexer_n_heads);
  const Captured& st = store.Get(layer + ".sel_tokens");
  const Captured& sc = store.Get(layer + ".sel_count");
  const int stride = static_cast<int>(st.cols);
  int64_t nm = 0, nq = 0, nk = 0;
  const uint8_t* mask = g.U8(layer + ".index_mask", &nm);
  const uint16_t* iq = g.Bf16(layer + ".index_q", &nq);
  const uint16_t* ik = g.Bf16(layer + ".index_block_keys", &nk);
  Check(nm == static_cast<int64_t>(T) * T, layer + ": index_mask is not [T, T]");
  SelectionResult out;
  out.rows = T;
  out.match.assign(static_cast<size_t>(T), 1);
  const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(D));
  int golden_anomalies = 0;
  for (int p = 0; p < T; ++p) {
    const int nb = (p + 1) / ratio;
    std::vector<char> gold(static_cast<size_t>(p + 1)), mine(static_cast<size_t>(p + 1), 0);
    for (int j = 0; j <= p; ++j) gold[static_cast<size_t>(j)] = mask[static_cast<int64_t>(p) * T + j] != 0;
    for (int j = p + 1; j < T; ++j) golden_anomalies += mask[static_cast<int64_t>(p) * T + j] != 0;  // above the diagonal
    if (nb <= topk) {
      // dense: attends to everything it sees; the kernel's list is unspecified, the golden's mask must be full
      ++out.dense_rows;
      for (int j = 0; j <= p; ++j) golden_anomalies += gold[static_cast<size_t>(j)] ? 0 : 1;
      continue;
    }
    ++out.sparse_rows;
    const int count = sc.i32[static_cast<size_t>(p)];
    Check(count > 0 && count <= stride, layer + ": selected count out of range at row " + std::to_string(p));
    for (int i = 0; i < count && i < stride; ++i) {
      const int j = st.i32[static_cast<size_t>(p) * stride + i];
      Check(j >= 0 && j <= p, layer + ": selected token outside [0, p] at row " + std::to_string(p));
      if (j >= 0 && j <= p) mine[static_cast<size_t>(j)] = 1;
    }
    const bool same = mine == gold;
    // fp32 recompute of the block scores from the golden's own indexer inputs, and the `recent` top-k
    std::vector<double> score(static_cast<size_t>(nb), 0.0);
    for (int b = 0; b < nb; ++b) {
      double s = 0;
      for (int h = 0; h < H; ++h) {
        double dot = 0;
        for (int d = 0; d < D; ++d) {
          dot += static_cast<double>(Bf(iq[(static_cast<int64_t>(p) * H + h) * D + d])) * Bf(ik[static_cast<int64_t>(b) * D + d]);
        }
        s += dot > 0 ? dot : 0.0;
      }
      score[static_cast<size_t>(b)] = s * inv_sqrt_d;
    }
    std::vector<int> order(static_cast<size_t>(nb));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (score[static_cast<size_t>(a)] != score[static_cast<size_t>(b)]) return score[static_cast<size_t>(a)] > score[static_cast<size_t>(b)];
      return a > b;  // the `recent` rule: the later block wins a tie
    });
    std::vector<char> recomputed(static_cast<size_t>(p + 1), 0);
    for (int i = 0; i < topk; ++i) {
      const int b = order[static_cast<size_t>(i)];
      for (int t = 0; t < ratio; ++t) recomputed[static_cast<size_t>(b * ratio + t)] = 1;
    }
    for (int j = nb * ratio; j <= p; ++j) recomputed[static_cast<size_t>(j)] = 1;  // the incomplete tail
    out.golden_vs_recompute += gold == recomputed ? 0 : 1;
    out.mine_vs_recompute += mine == recomputed ? 0 : 1;
    if (same) continue;
    out.match[static_cast<size_t>(p)] = 0;
    ++out.differ;
    // the distance of every block that only one side selected from the cut (midway between the topk-th and the next score),
    // relative to the rms of the row's scores
    const double cut = 0.5 * (score[static_cast<size_t>(order[static_cast<size_t>(topk - 1)])] +
                              score[static_cast<size_t>(order[static_cast<size_t>(topk)])]);
    double sq = 0;
    for (double s : score) sq += s * s;
    const double rms = std::sqrt(sq / nb);
    double worst = 0;
    std::string detail;
    for (int b = 0; b < nb; ++b) {
      const bool in_m = mine[static_cast<size_t>(b * ratio)] != 0, in_g = gold[static_cast<size_t>(b * ratio)] != 0;
      if (in_m == in_g) continue;
      worst = std::max(worst, std::fabs(score[static_cast<size_t>(b)] - cut) / (rms > 0 ? rms : 1.0));
      char buf[160];
      std::snprintf(buf, sizeof buf, " block %d (%s only) score %.6f;", b, in_m ? "ours" : "golden", score[static_cast<size_t>(b)]);
      detail += buf;
    }
    // a differing TAIL token (j >= nb * ratio) is never a near-tie
    for (int j = nb * ratio; j <= p; ++j) {
      if (mine[static_cast<size_t>(j)] != gold[static_cast<size_t>(j)]) worst = 1e9;
    }
    out.rel_per_differing_row.push_back(worst);
    {
      char buf[200];
      std::snprintf(buf, sizeof buf, "row %d (%d complete blocks): cut score %.6f, gap topk-th vs next %.3g, rms(scores) %.4f,", p, nb, cut,
                    score[static_cast<size_t>(order[static_cast<size_t>(topk - 1)])] - score[static_cast<size_t>(order[static_cast<size_t>(topk)])], rms);
      out.details.push_back(std::string(buf) + detail);
    }
    out.worst_rel = std::max(out.worst_rel, worst);
    out.differ_beyond_near_tie += worst > near_tie_rel ? 1 : 0;
  }
  Check(golden_anomalies == 0, layer + ": the golden's index_mask is not causal / not full on its dense rows (" +
                                   std::to_string(golden_anomalies) + " cells)");
  return out;
}

// The attention of the QSA layer is judged against a recompute from the golden's own q / k / v / gate (post norm and rope) with the
// K / V rounded through the fp8 e4m3 cache exactly as the kernel writes it (fp8(bf16 * 1 / descale), read back * descale), over the
// tokens QsaLayerCore selected (dense rows: every visible token), softmax scale head_dim^-0.5, GQA 12:1, output * sigmoid(gate);
// double accumulation, bf16 result. What is left is bf16 rounding and the kernel's accumulation order, not the fp8 cache noise.
// Bound: observed worst 0.478 / 0.0025 over T = 1 / 7 / 70 / 300 (median per-row max error 0.06 of the rms): the kernel's q / k are
// bf16 products of its own projections, not the golden's, and the tiny model's softmax inputs span ~10 (max - min of the scaled
// scores per row, printed by the test), so a 0.4% bf16 difference in q . k moves attention weights by several percent on the
// worst rows; the mean bound is the sensitive one (a wrong selection / cache read / gate would move it by an order of magnitude).
constexpr Tol kFp8RefTol = {0.8, 0.005};

std::vector<uint16_t> Fp8AttentionReference(const r4dx::model::Qwen4ExpConfig& cfg, const Golden& g, const std::string& layer,
                                            const CaptureStore& store, int T, const std::string& container_path,
                                            std::vector<double>* score_spread = nullptr) {
  const int Hq = static_cast<int>(cfg.num_attention_heads), Hkv = static_cast<int>(cfg.num_key_value_heads);
  const int D = static_cast<int>(cfg.head_dim);
  const int group = Hq / Hkv;
  const int topk = static_cast<int>(cfg.BlockTopK());
  const int ratio = static_cast<int>(cfg.indexer_compress_ratio);
  const r4dx_convert::SafetensorsReader cont(r4dx_convert::Utf8ToWide(container_path));
  float kd[8] = {}, vd[8] = {};
  for (const auto& [suffix, dst] : {std::pair<const char*, float*>{".attn.k_descale", kd}, {".attn.v_descale", vd}}) {
    const std::string name = "text.layers." + layer.substr(5) + suffix;
    const auto& m = cont.Meta(name);
    if (m.dtype != "U8" || m.ElemCount() != Hkv * 4) throw std::runtime_error(name + ": unexpected descale layout");
    std::memcpy(dst, cont.Data(name), static_cast<size_t>(Hkv) * sizeof(float));
  }
  const uint16_t* q = g.Bf16(layer + ".attn_q");
  const uint16_t* k = g.Bf16(layer + ".attn_k");
  const uint16_t* v = g.Bf16(layer + ".attn_v");
  const uint16_t* gate = g.Bf16(layer + ".attn_gate");
  const auto rt = [](uint16_t b, float descale) {
    return r4dx::core::Fp8E4M3ToFloat(r4dx::core::FloatToFp8E4M3(Bf(b) * (1.0f / descale))) * descale;
  };
  std::vector<float> kq(static_cast<size_t>(T) * Hkv * D), vq(kq.size());
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < Hkv; ++h) {
      for (int d = 0; d < D; ++d) {
        const size_t i = (static_cast<size_t>(t) * Hkv + h) * D + d;
        kq[i] = rt(k[i], kd[h]);
        vq[i] = rt(v[i], vd[h]);
      }
    }
  }
  const Captured& st = store.Get(layer + ".sel_tokens");
  const Captured& sc = store.Get(layer + ".sel_count");
  const int stride = static_cast<int>(st.cols);
  std::vector<uint16_t> out(static_cast<size_t>(T) * Hq * D);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  std::vector<int> toks;
  std::vector<double> w;
  for (int p = 0; p < T; ++p) {
    toks.clear();
    if ((p + 1) / ratio <= topk) {
      for (int j = 0; j <= p; ++j) toks.push_back(j);
    } else {
      const int count = sc.i32[static_cast<size_t>(p)];
      for (int i = 0; i < count; ++i) toks.push_back(st.i32[static_cast<size_t>(p) * stride + i]);
    }
    w.resize(toks.size());
    for (int hq = 0; hq < Hq; ++hq) {
      const int kv = hq / group;
      const uint16_t* qh = q + (static_cast<size_t>(p) * Hq + hq) * D;
      double mx = -1e300, mn = 1e300;
      for (size_t i = 0; i < toks.size(); ++i) {
        const float* kr = &kq[(static_cast<size_t>(toks[i]) * Hkv + kv) * D];
        double s = 0;
        for (int d = 0; d < D; ++d) s += static_cast<double>(Bf(qh[d])) * kr[d];
        w[i] = s * scale;
        mx = std::max(mx, w[i]);
        mn = std::min(mn, w[i]);
      }
      if (score_spread != nullptr) (*score_spread)[static_cast<size_t>(p)] = std::max((*score_spread)[static_cast<size_t>(p)], mx - mn);
      double den = 0;
      for (double& x : w) {
        x = std::exp(x - mx);
        den += x;
      }
      for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (size_t i = 0; i < toks.size(); ++i) acc += w[i] * vq[(static_cast<size_t>(toks[i]) * Hkv + kv) * D + d];
        const size_t oi = (static_cast<size_t>(p) * Hq + hq) * D + d;
        const double gt = 1.0 / (1.0 + std::exp(-static_cast<double>(Bf(gate[oi]))));
        out[oi] = r4dx::core::FloatToBf16(static_cast<float>(acc / den * gt));
      }
    }
  }
  return out;
}

// ---- one (mode, prompt) -----------------------------------------------------------------------------------------------
struct Worst {
  double max_norm = 0, mean_norm = 0;
  std::string where;
};

void Note(std::map<std::string, Worst>& table, const std::string& point, const std::string& where, const Stats& s) {
  Worst& w = table[point];
  if (s.max_norm > w.max_norm) w.where = where;
  w.max_norm = std::max(w.max_norm, s.max_norm);
  w.mean_norm = std::max(w.mean_norm, s.mean_norm);
}

void RunPrompt(Qwen4ExpModel& model, const Golden& g, const std::string& mode, bool strict, std::map<std::string, Worst>& worst) {
  const int T = g.T();
  const auto& cfg = model.Config();
  std::printf("\n-- [%s] prompt T=%d --\n", mode.c_str(), T);
  for (int64_t L = 0; L < cfg.num_hidden_layers; ++L) {
    const std::string ln = "layer" + std::to_string(L);
    const bool qsa = cfg.IsQsaLayer(L), ple = cfg.IsPleLayer(L);
    const uint16_t* rin = g.Bf16(ln + ".residual_in");
    CaptureStore store;
    model.SetCapture(store.Fn());
    model.RunLayerIsolated(g.ids(), L, rin);
    model.SetCapture(nullptr);
    const std::string tag = mode + " T" + std::to_string(T) + " " + ln;
    std::printf("\n %s (%s%s)  %.1f ms\n", ln.c_str(), qsa ? "QSA" : "GDN", ple ? " + PLE" : "", model.LastTimings().stage_ms[0] + model.LastTimings().stage_ms[1]);

    // the input really was the golden's
    {
      const Captured& c = store.Get(ln + ".residual_in");
      Check(c.bf16.size() == static_cast<size_t>(T) * 10240 && std::memcmp(c.bf16.data(), rin, c.bf16.size() * 2) == 0,
            tag + ": the isolated run did not start from the golden residual_in");
    }

    // row classes
    std::vector<char> all(static_cast<size_t>(T), 1), sel_ok(static_cast<size_t>(T), 1);
    std::vector<char> sel_diff(static_cast<size_t>(T), 0);
    if (qsa) {
      const SelectionResult sr = CheckSelection(model, g, ln, store, T, /*near_tie_rel=*/0.05);
      sel_ok = sr.match;
      sel_diff = Not(sr.match);
      std::printf("  selection: %d rows (%d dense, %d sparse); selected set differs from the golden mask in %d sparse row(s)\n", sr.rows,
                  sr.dense_rows, sr.sparse_rows, sr.differ);
      if (sr.sparse_rows > 0) {
        std::printf("    fp32 `recent` recompute from the golden's index_q / index_block_keys vs golden mask: %d row(s) differ; vs ours: %d row(s) differ\n",
                    sr.golden_vs_recompute, sr.mine_vs_recompute);
      }
      if (sr.differ > 0) {
        std::vector<double> d = sr.rel_per_differing_row;
        std::sort(d.begin(), d.end());
        std::printf("    differing rows: distance of the swapped blocks from the cut / rms(scores): median %.4g, p90 %.4g, max %.4g; %d row(s) > 0.05\n",
                    d[d.size() / 2], d[static_cast<size_t>(0.9 * static_cast<double>(d.size() - 1))], d.back(), sr.differ_beyond_near_tie);
        std::string first;
        int shown = 0;
        for (int p = 0; p < T && shown < 12; ++p) {
          if (!sr.match[static_cast<size_t>(p)]) first += (shown++ ? "," : "") + std::to_string(p);
        }
        std::printf("    first differing rows: %s%s\n", first.c_str(), sr.differ > 12 ? ",..." : "");
        for (size_t i = 0; i < sr.details.size() && i < 6; ++i) std::printf("      %s\n", sr.details[i].c_str());
      }
      Check(sr.differ_beyond_near_tie == 0, tag + ": " + std::to_string(sr.differ_beyond_near_tie) +
                                                " row(s) select a different set than the golden beyond a score near-tie");
    }

    // points
    struct Point {
      const char* name;
      int64_t cols;
      int rows_class;  // 0 all rows, 1 selection-matching rows, 2 selection-matching rows with the same expert set
    };
    std::vector<Point> points;
    if (ple) points.push_back({"residual_ple", 10240, 0});
    points.push_back({"attn_mix", 2560, 0});
    points.push_back({"attn_inject", 4, 0});
    if (qsa) points.push_back({"attn_pre_o", 6144, 1});
    points.push_back({"attn_out", 2560, 1});
    points.push_back({"residual_attn", 10240, 1});
    points.push_back({"mlp_mix", 2560, 1});
    points.push_back({"mlp_inject", 4, 1});

    // router first: its same-set rows bound the MoE points
    const RouterResult rr = CheckRouter(store.Get(ln + ".topk_ids"), g, ln, sel_ok, RouterTieTol(qsa));
    std::printf("  topk_ids (rows with a matching selection): order differs in %d, expert SET differs in %d (%d beyond a router near-tie of %.2f of the row's logit rms; worst %.4f)\n",
                rr.order_rows, rr.set_rows, rr.unexplained, RouterTieTol(qsa), rr.worst);
    Check(rr.unexplained == 0, tag + ": expert sets differ from the golden beyond router near-ties");
    points.push_back({"moe_out", 2560, 2});
    points.push_back({"residual_out", 10240, 2});

    const std::vector<char> ok2 = And(sel_ok, rr.same_set);
    PrintStatsHeader();
    for (const Point& pt : points) {
      const std::string name = ln + "." + pt.name;
      const Captured& c = store.Get(name);
      int64_t n = 0;
      const uint16_t* ref = g.Bf16(name, &n);
      Check(static_cast<int64_t>(c.bf16.size()) == n, tag + " " + pt.name + ": element count differs from the golden");
      const std::vector<char>& cls = pt.rows_class == 0 ? all : (pt.rows_class == 1 ? sel_ok : ok2);
      const std::vector<int> rows = RowsWhere(cls);
      if (rows.empty()) continue;
      const Stats s = StatsRows(c.bf16.data(), ref, pt.cols, rows);
      PrintStats(std::string(pt.name) + (rows.size() == static_cast<size_t>(T) ? "" : " [" + std::to_string(rows.size()) + " rows]"), s);
      const Tol t = PointTol(pt.name, qsa);
      Check(!s.nan, tag + " " + pt.name + ": NaN / Inf");
      const bool in_bound = s.max_norm <= t.max_norm && s.mean_norm <= t.mean_norm;
      if (strict) {
        Check(in_bound, tag + " " + pt.name + ": error above its bound (max/rms " + std::to_string(s.max_norm) + " > " +
                            std::to_string(t.max_norm) + " or mean/rms " + std::to_string(s.mean_norm) + " > " + std::to_string(t.mean_norm) + ")");
      }
      Note(worst, pt.name, tag, s);
      if (pt.rows_class != 0 && rows.size() != static_cast<size_t>(T)) {
        const std::vector<int> other = RowsWhere(Not(cls));
        const Stats so = StatsRows(c.bf16.data(), ref, pt.cols, other);
        PrintStats(std::string("  (the other ") + std::to_string(other.size()) + " rows, unbounded)", so);
      }
    }
    if (qsa) {
      // the attention itself against a CPU recompute that shares the kernel's fp8 K / V rounding and OUR selection (every row)
      std::vector<double> spread(static_cast<size_t>(T), 0.0);
      const std::vector<uint16_t> ref = Fp8AttentionReference(cfg, g, ln, store, T, ContainerPath("bf16"), &spread);
      const Captured& c = store.Get(ln + ".attn_pre_o");
      const std::vector<int> every = RowsWhere(all);
      const int64_t width = cfg.num_attention_heads * cfg.head_dim;
      const Stats s = StatsRows(c.bf16.data(), ref.data(), width, every);
      PrintStats("attn_pre_o vs fp8 ref [all]", s);
      {
        // per row: max|err| over the row's elements / rms of the whole reference; where do the outliers sit, and is the row's
        // softmax input (max - min of the scaled scores, which amplifies bf16 noise in OUR q / k) wide there?
        std::vector<double> row_err(static_cast<size_t>(T));
        for (int p = 0; p < T; ++p) {
          double m = 0;
          for (int64_t i = 0; i < width; ++i) {
            m = std::max(m, std::fabs(static_cast<double>(Bf(c.bf16[static_cast<size_t>(p * width + i)])) - Bf(ref[static_cast<size_t>(p * width + i)])));
          }
          row_err[static_cast<size_t>(p)] = s.rms_ref > 0 ? m / s.rms_ref : 0.0;
        }
        std::vector<double> sorted = row_err;
        std::sort(sorted.begin(), sorted.end());
        const int worst_row = static_cast<int>(std::max_element(row_err.begin(), row_err.end()) - row_err.begin());
        std::printf("    per-row max|err|/rms: median %.4f p95 %.4f max %.4f (row %d, score spread %.1f; median spread %.1f, max spread %.1f)\n",
                    sorted[sorted.size() / 2], sorted[static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1))], sorted.back(), worst_row,
                    spread[static_cast<size_t>(worst_row)], [&] { std::vector<double> sp = spread; std::sort(sp.begin(), sp.end()); return sp[sp.size() / 2]; }(),
                    *std::max_element(spread.begin(), spread.end()));
      }
      Check(!s.nan, tag + " attn_pre_o vs fp8 reference: NaN / Inf");
      Check(s.max_norm <= kFp8RefTol.max_norm && s.mean_norm <= kFp8RefTol.mean_norm,
            tag + " attn_pre_o differs from the fp8-aware reference (max/rms " + std::to_string(s.max_norm) + ", mean/rms " +
                std::to_string(s.mean_norm) + ")");
      Note(worst, "attn_pre_o~fp8ref", tag, s);
    }
    // GDN: where along the sequence does the mixer's error live (chunked-scan diagnosis)
    if (!qsa && T >= 64) {
      const Captured& c = store.Get(ln + ".attn_out");
      int64_t n = 0;
      const uint16_t* ref = g.Bf16(ln + ".attn_out", &n);
      std::string line;
      for (int b0 = 0; b0 < T; b0 += 32) {
        std::vector<int> rows;
        for (int r = b0; r < std::min(T, b0 + 32); ++r) rows.push_back(r);
        const Stats s = StatsRows(c.bf16.data(), ref, 2560, rows);
        char buf[32];
        std::snprintf(buf, sizeof buf, " %.3f", s.max_norm);
        line += buf;
      }
      std::printf("  attn_out max/rms per 32-row block:%s\n", line.c_str());
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
  if (!Exists(ContainerPath("bf16"))) {
    std::printf("SKIP: %s missing\n", ContainerPath("bf16").c_str());
    return kSkip;
  }
  const int prompts[] = {1, 7, 70, 300};
  for (int T : prompts) {
    if (!Exists(GoldenPath(T))) {
      std::printf("SKIP: %s missing\n", GoldenPath(T).c_str());
      return kSkip;
    }
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  struct Mode {
    const char* name;
    bool recurrent;
    bool strict;
  };
  const Mode modes[] = {
      {"recurrent GDN prefill (reference)", true, true},
      {"chunked GDN prefill (production)", false, true},
  };
  std::map<std::string, Worst> worst_all;
  for (const Mode& m : modes) {
    std::printf("\n==== %s, device %d ====\n", m.name, TestDevice());
    auto model = LoadTiny("bf16", false, {}, 0, 4096, m.recurrent);
    std::map<std::string, Worst> worst;
    for (int T : prompts) {
      const Golden g(T, GoldenPath(T));
      RunPrompt(*model, g, m.name, m.strict, worst);
    }
    std::printf("\n worst over every layer and prompt [%s]:\n  %-16s %9s %9s  at\n", m.name, "point", "max/rms", "mean/rms");
    for (const auto& [point, w] : worst) std::printf("  %-16s %9.4f %9.5f  %s\n", point.c_str(), w.max_norm, w.mean_norm, w.where.c_str());
  }
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
