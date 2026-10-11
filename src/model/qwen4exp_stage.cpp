#include "qwen4exp_stage.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>

#include "gdn_state.h"
#include "kernels/model_kernels.h"
#include "linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/hc_conn.h"
#include "r4dx/model/qwen4exp/moe_block.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"
#include "r4dx/model/qwen4exp/ple_layer.h"
#include "r4dx/model/qwen4exp/qsa.h"

namespace r4dx::model {

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

[[noreturn]] void Fail(const std::string& msg) { throw std::runtime_error("Qwen4ExpStage: " + msg); }

size_t Padded(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

// A non-owning bf16 linear over a container tensor (ApplyLinear reads .data() / .size() only).
QuantLinear BorrowedBf16Linear(const uint16_t* w, int64_t N, int64_t K, const char* what) {
  if (w == nullptr) Fail(std::string("missing weight ") + what);
  QuantLinear q;
  q.layout = Layout::kBf16;
  q.N = N;
  q.K = K;
  q.bf16_w = core::DeviceBuffer<uint16_t>::View(const_cast<uint16_t*>(w), static_cast<size_t>(N * K));
  return q;
}

}  // namespace

Qwen4ExpDeviceScope::Qwen4ExpDeviceScope(int device) {
  R4DX_HIP_CHECK(hipGetDevice(&previous_));
  if (previous_ != device) {
    R4DX_HIP_CHECK(hipSetDevice(device));
    changed_ = true;
  }
}

Qwen4ExpDeviceScope::~Qwen4ExpDeviceScope() {
  if (changed_) static_cast<void>(hipSetDevice(previous_));
}

// ---- one layer's runtime objects ------------------------------------------------------------------------------------
namespace {

struct LayerRt {
  int64_t index = 0;
  bool qsa = false;
  std::deque<QuantLinear> views;  // borrowed bf16 linears (stable addresses: HyperConn etc. keep pointers)
  std::unique_ptr<qwen4exp::HyperConn> hc_attn, hc_mlp;

  // GDN
  std::unique_ptr<GdnStateManager> gdn_states;
  std::unique_ptr<Qwen4ExpGdn> gdn;

  // QSA
  std::unique_ptr<qwen4exp::QsaLayerCore> qsa_core;
  std::unique_ptr<attention::PagedKvCache> kv;
  qwen4exp::QsaWeights qsa_w;
  const QuantLinear *qg = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *idx_qk = nullptr;

  // MoE
  qwen4exp::MoeWeights moe_w;
  const Qwen4ExpMoeRefs* moe_refs = nullptr;  // points into Impl::layer_refs
  // PLE
  std::unique_ptr<qwen4exp::PleLayer> ple;
  core::DeviceBuffer<uint16_t> ple_ckpt;

  // Steady-state decode HIP Graphs (M=1, zero CPU driver overhead)
  hipGraphExec_t decode_graph_exec = nullptr;
  hipGraphExec_t moe_graph_exec = nullptr;


  ~LayerRt() {
    if (decode_graph_exec) (void)hipGraphExecDestroy(decode_graph_exec);
    if (moe_graph_exec) (void)hipGraphExecDestroy(moe_graph_exec);
  }
};

}  // namespace

struct Qwen4ExpStage::Impl {
  Qwen4ExpStageOptions opt;
  Qwen4ExpConfig cfg;
  int device = 0;
  int64_t first = 0, end = 0;
  int64_t hidden = 0, wide = 0, hc = 0, vocab = 0;

  core::Stream stream;
  core::Arena arena;          // per-block scratch (reset between blocks)
  core::Arena expert_arena;   // ApplyLinear scratch of ONE expert GEMM (reset before each, see RunMoe)

  core::DeviceBuffer<uint16_t> h;        // residual [max_rows, hc * hidden]
  core::DeviceBuffer<uint16_t> x;        // block input [max_rows, hidden]
  core::DeviceBuffer<uint16_t> blk;      // block output [max_rows, hidden]
  core::DeviceBuffer<uint16_t> inj;      // injection weights [max_rows, hc]
  core::DeviceBuffer<uint16_t> emb_in;   // embedding rows [max_rows, hidden]
  core::DeviceBuffer<int32_t> iota;      // 0 .. max_rows - 1
  core::DeviceBuffer<uint16_t> ple_emb;  // n-gram rows [max_rows, ple_embed_dim]
  core::DeviceBuffer<int32_t> positions; // [max_rows]
  core::PinnedBuffer<int32_t> positions_host;

  GdnControlCache control;
  std::unique_ptr<qwen4exp::MoeBlockBase> moe;
  std::vector<Qwen4ExpLayerRefs> layer_refs;
  std::vector<std::unique_ptr<LayerRt>> layers;
  bool has_ple = false, has_qsa = false;
  int64_t decode_steps_seen = 0;
  // head
  std::optional<Qwen4ExpHeadRefs> head_refs;
  std::unique_ptr<qwen4exp::HyperConn> hc_final;
  std::deque<QuantLinear> head_views;          // the final mixer's borrowed bf16 linears
  core::DeviceBuffer<float> logits_dev;        // [vocab] fp32, the last head call's row
  core::DeviceBuffer<int32_t> argmax_dev;      // [1]
  core::PinnedBuffer<int32_t> greedy_host;     // [1]
  core::DeviceBuffer<int32_t> summary_ids_dev;   // [K]
  core::DeviceBuffer<float> summary_vals_dev;    // [K]
  core::DeviceBuffer<float> summary_lse_dev;     // [1]
  core::DeviceBuffer<uint8_t> topk_ws;           // r4dx_topk_lse_workspace_bytes()
  core::DeviceBuffer<uint16_t> logits_all;       // capture mode: [max_rows, vocab] bf16
  // Grouped Trellis decode MoE buffers (M=1, zero CPU-GPU sync)
  core::DeviceBuffer<uint16_t> dmoe_logits;
  core::DeviceBuffer<int32_t> dmoe_ids;
  core::DeviceBuffer<float> dmoe_weights;
  core::DeviceBuffer<int32_t> dmoe_ids_sorted;
  core::DeviceBuffer<float> dmoe_weights_sorted;
  core::DeviceBuffer<uint16_t> dmoe_a0_gu;
  core::DeviceBuffer<uint16_t> dmoe_a1_gu;
  core::DeviceBuffer<uint16_t> dmoe_gu;
  core::DeviceBuffer<float> dmoe_ws_gu;
  core::DeviceBuffer<uint16_t> dmoe_a0_down;
  core::DeviceBuffer<uint16_t> dmoe_yp;
  core::DeviceBuffer<float> dmoe_ws_down;
  core::DeviceBuffer<uint16_t> dmoe_shared_g;
  core::DeviceBuffer<uint16_t> dmoe_shared_u;
  core::DeviceBuffer<uint16_t> dmoe_shared_h;
  core::DeviceBuffer<uint16_t> dmoe_shared_y;
  core::DeviceBuffer<uint16_t> dmoe_shared_gate_logit;

  void RunMoeGroupedDecode(LayerRt& L);

  Qwen4ExpCaptureFn capture;
  double last_run_ms = 0, last_head_ms = 0, last_ple_wait_ms = 0;
  int64_t last_first_pos = 0;  // first position of the chunk RunLayers ran last (the head's capture rows start there)

  void Cap(int64_t layer, const char* suffix, const void* dev, Qwen4ExpCapture::Dtype dt, int64_t rows, int64_t cols,
           int64_t first_pos, bool on_host = false) {
    if (!capture) return;
    Qwen4ExpCapture c;
    c.name = layer >= 0 ? "layer" + std::to_string(layer) + "." + suffix : std::string(suffix);
    c.dev = dev;
    c.dtype = dt;
    c.rows = rows;
    c.cols = cols;
    c.first_pos = first_pos;
    c.stream = stream.get();
    c.on_host = on_host;
    capture(c);
  }
  void CapBf16(int64_t layer, const char* suffix, const void* dev, int64_t rows, int64_t cols, int64_t first_pos) {
    Cap(layer, suffix, dev, Qwen4ExpCapture::Dtype::kBf16, rows, cols, first_pos);
  }

  void RunGdn(LayerRt& L, const Qwen4ExpChunk& c);
  void RunQsa(LayerRt& L, const Qwen4ExpChunk& c);
  void RunMoe(LayerRt& L, const Qwen4ExpChunk& c);
};

// The stage's core::Stream is created by Impl's default constructor, so Impl itself must be constructed with the stage's
// device current (a stream belongs to the device that was current when it was created; kernels launched on it for another
// device's memory fault inside amdhip64 when more than one device is visible).
std::unique_ptr<Qwen4ExpStage::Impl> MakeStageImpl(int device) {
  Qwen4ExpDeviceScope dev(device);
  return std::make_unique<Qwen4ExpStage::Impl>();
}

// ---- construction ---------------------------------------------------------------------------------------------------

Qwen4ExpStage::Qwen4ExpStage(const Qwen4ExpStageOptions& options, std::vector<Qwen4ExpLayerRefs> layers,
                             std::optional<Qwen4ExpHeadRefs> head)
    : impl_(MakeStageImpl(options.device)) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(options.device);
  m.opt = options;
  m.cfg = options.config;
  m.device = options.device;
  const Qwen4ExpConfig& cfg = m.cfg;
  if (layers.empty()) Fail("no layers");
  if (options.max_rows < 1 || options.max_rows > qwen4exp::QsaLayerCore::kMaxChunk) {
    Fail("max_rows must be in [1, " + std::to_string(qwen4exp::QsaLayerCore::kMaxChunk) + "]");
  }
  if (options.max_ctx < 1) Fail("max_ctx must be >= 1");
  m.first = layers.front().index;
  m.end = m.first + static_cast<int64_t>(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    if (layers[i].index != m.first + static_cast<int64_t>(i)) Fail("layers are not consecutive");
    if (layers[i].qsa != cfg.IsQsaLayer(layers[i].index)) {
      Fail("layer " + std::to_string(layers[i].index) + " is a " + (layers[i].qsa ? "QSA" : "GDN") +
           " layer in the weights but a " + (cfg.IsQsaLayer(layers[i].index) ? "QSA" : "GDN") +
           " layer in the config");
    }
  }
  m.hidden = cfg.hidden_size;
  m.hc = cfg.hc_count;
  m.wide = m.hc * m.hidden;
  m.vocab = cfg.vocab_size;
  const int64_t R = options.max_rows;
  const int64_t ple_dim = cfg.ple_embed_dim;

  m.layer_refs = std::move(layers);
  m.head_refs = std::move(head);

  // ---- per-layer objects (the borrowed views first) -------------------------------------------------------------
  const qwen4exp::HyperConnConfig hcc{static_cast<int>(cfg.hc_count), static_cast<int>(m.hidden),
                                      static_cast<int>(cfg.hc_lowrank), static_cast<float>(cfg.rms_norm_eps)};
  const auto make_hc = [&](LayerRt& L, const Qwen4ExpHcRefs& r, const char* what) {
    qwen4exp::HyperConnWeights w;
    w.norm = r.norm;
    L.views.push_back(BorrowedBf16Linear(r.down, cfg.hc_lowrank, m.wide, what));
    w.down = &L.views.back();
    L.views.push_back(BorrowedBf16Linear(r.up, m.wide, cfg.hc_lowrank, what));
    w.up = &L.views.back();
    if (r.inject != nullptr) {
      L.views.push_back(BorrowedBf16Linear(r.inject, cfg.hc_count, m.wide, what));
      w.inject = &L.views.back();
    }
    return std::make_unique<qwen4exp::HyperConn>(hcc, w);
  };

  qwen4exp::MoeConfig moec;
  moec.hidden = static_cast<int>(m.hidden);
  moec.experts = static_cast<int>(cfg.num_experts);
  moec.top_k = static_cast<int>(cfg.num_experts_per_tok);
  moec.inter = static_cast<int>(cfg.moe_intermediate_size);
  moec.shared_inter = static_cast<int>(cfg.shared_expert_intermediate_size);
  moec.norm_topk = cfg.norm_topk_prob;
  m.moe = std::make_unique<qwen4exp::MoeBlockBase>(moec, R);

  size_t block_arena = 0;  // the largest single block's arena demand at max_rows
  const auto bump = [&](size_t b) { block_arena = std::max(block_arena, b); };
  bump(m.moe->ArenaBytes(R));

  qwen4exp::QsaDims qd;
  qd.heads_q = static_cast<int>(cfg.num_attention_heads);
  qd.heads_kv = static_cast<int>(cfg.num_key_value_heads);
  qd.head_dim = static_cast<int>(cfg.head_dim);
  qd.rotary_dim = static_cast<int>(cfg.RotaryDim());
  qd.rope_theta = cfg.rope_theta;
  qd.rms_eps = static_cast<float>(cfg.rms_norm_eps);
  qd.index_heads = static_cast<int>(cfg.indexer_n_heads);
  qd.index_dim = static_cast<int>(cfg.indexer_head_dim);
  qd.compress_ratio = static_cast<int>(cfg.indexer_compress_ratio);
  qd.budget = static_cast<int>(cfg.indexer_budget);
  qd.max_ctx = static_cast<int>(options.max_ctx);
  if (cfg.indexer_kv_heads != 1) Fail("the indexer has exactly one key head");

  const int64_t q_width = cfg.num_attention_heads * cfg.head_dim;
  const int64_t kv_width = cfg.num_key_value_heads * cfg.head_dim;
  const int64_t idx_width = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim;

  for (const Qwen4ExpLayerRefs& r : m.layer_refs) {
    auto L = std::make_unique<LayerRt>();
    L->index = r.index;
    L->qsa = r.qsa;
    L->hc_attn = make_hc(*L, r.hc_attn, "hc.attn");
    L->hc_mlp = make_hc(*L, r.hc_mlp, "hc.mlp");
    bump(L->hc_attn->ArenaBytes(R));

    if (!r.qsa) {
      L->gdn = std::make_unique<Qwen4ExpGdn>(cfg, r.gdn);
      L->gdn_states = std::make_unique<GdnStateManager>(
          /*max_seqs=*/1, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim,
          L->gdn->ConvDim(), cfg.linear_conv_kernel_dim, /*max_decode_window=*/1, /*write_once=*/false);
      L->gdn_states->ZeroAll(m.stream);
      if (options.prompt_checkpoint) L->gdn_states->AllocateCheckpoint();
      bump(L->gdn->ArenaBytes(R));
    } else {
      const Qwen4ExpQsaRefs& a = r.attn;
      for (const QuantLinear* q : {a.qg, a.k, a.v, a.o}) {
        if (q == nullptr) Fail("layer " + std::to_string(r.index) + ": missing attention linear");
      }
      if (a.qg->N != 2 * q_width || a.qg->K != m.hidden || a.k->N != kv_width || a.v->N != kv_width ||
          a.o->N != m.hidden || a.o->K != q_width) {
        Fail("layer " + std::to_string(r.index) + ": attention linear shapes do not match the config");
      }
      L->qg = a.qg;
      L->k = a.k;
      L->v = a.v;
      L->o = a.o;
      L->views.push_back(BorrowedBf16Linear(a.indexer_qk, idx_width, m.hidden, "attn.indexer.qk"));
      L->idx_qk = &L->views.back();
      if (a.q_norm == nullptr || a.k_norm == nullptr || a.indexer_q_norm == nullptr ||
          a.indexer_k_norm == nullptr || a.k_descale == nullptr || a.v_descale == nullptr) {
        Fail("layer " + std::to_string(r.index) + ": missing attention norm / descale");
      }
      L->qsa_w.q_norm = a.q_norm;
      L->qsa_w.k_norm = a.k_norm;
      L->qsa_w.idx_q_norm = a.indexer_q_norm;
      L->qsa_w.idx_k_norm = a.indexer_k_norm;
      L->qsa_w.k_descale = a.k_descale;
      L->qsa_w.v_descale = a.v_descale;
      L->qsa_core = std::make_unique<qwen4exp::QsaLayerCore>(qd);
      L->kv = std::make_unique<attention::PagedKvCache>(qd.heads_kv, qd.head_dim, /*block_size=*/16,
                                                        static_cast<int>(options.max_ctx));
      const size_t proj = Padded(static_cast<size_t>(R * (2 * q_width + 2 * kv_width + idx_width)) * 2);
      bump(proj + L->qsa_core->ArenaBytes(static_cast<int>(R)));
      m.has_qsa = true;
    }

    // MoE
    const Qwen4ExpMoeRefs& e = r.moe;
    if (static_cast<int64_t>(e.expert_gate_up.size()) != cfg.num_experts ||
        static_cast<int64_t>(e.expert_down.size()) != cfg.num_experts) {
      Fail("layer " + std::to_string(r.index) + ": expert count does not match the config");
    }
    for (const QuantLinear* q : e.expert_gate_up) {
      if (q == nullptr) Fail("layer " + std::to_string(r.index) + ": missing expert gate_up");
    }
    for (const QuantLinear* q : e.expert_down) {
      if (q == nullptr) Fail("layer " + std::to_string(r.index) + ": missing expert down");
    }
    L->views.push_back(BorrowedBf16Linear(e.router, cfg.num_experts, m.hidden, "moe.router"));
    L->moe_w.router = &L->views.back();
    L->views.push_back(BorrowedBf16Linear(e.shared_gate, 1, m.hidden, "moe.shared_gate"));
    L->moe_w.shared_gate = &L->views.back();
    if (e.shared_gate_proj == nullptr || e.shared_up_proj == nullptr || e.shared_down_proj == nullptr) {
      Fail("layer " + std::to_string(r.index) + ": missing shared expert linear");
    }
    L->moe_w.shared_gate_proj = e.shared_gate_proj;
    L->moe_w.shared_up_proj = e.shared_up_proj;
    L->moe_w.shared_down_proj = e.shared_down_proj;
    L->moe_refs = &e;

    // PLE
    if (r.ple.has_value() != cfg.IsPleLayer(r.index)) {
      Fail("layer " + std::to_string(r.index) + (r.ple ? " carries PLE weights but the config has no PLE here"
                                                       : " is a PLE layer in the config but has no PLE weights"));
    }
    if (r.ple) {
      const Qwen4ExpPleRefs& p = *r.ple;
      L->views.push_back(BorrowedBf16Linear(p.key_proj, m.wide, ple_dim, "ple.key_proj"));
      const QuantLinear* key = &L->views.back();
      L->views.push_back(BorrowedBf16Linear(p.value_proj, m.hidden, ple_dim, "ple.value_proj"));
      const QuantLinear* value = &L->views.back();
      qwen4exp::PleConfig pc;
      pc.hc = static_cast<int>(cfg.hc_count);
      pc.hidden = static_cast<int>(m.hidden);
      pc.embed_dim = static_cast<int>(ple_dim);
      pc.conv_kernel = static_cast<int>(cfg.ple_conv_kernel_size);
      pc.conv_dilation = static_cast<int>(cfg.ngram_size);
      pc.eps = static_cast<float>(cfg.rms_norm_eps);
      qwen4exp::PleWeights pw;
      pw.key_proj = key;
      pw.value_proj = value;
      pw.norm_key = p.norm_key;
      pw.norm_query = p.norm_query;
      pw.norm_conv = p.norm_conv;
      pw.conv1d = p.conv1d;
      L->ple = std::make_unique<qwen4exp::PleLayer>(pc, pw);
      if (options.prompt_checkpoint) L->ple_ckpt = core::DeviceBuffer<uint16_t>(L->ple->StateElems());
      bump(L->ple->ArenaBytes(R));
      m.has_ple = true;
    }
    m.layers.push_back(std::move(L));
  }

  // ---- head ---------------------------------------------------------------------------------------------------------
  if (m.head_refs) {
    if (m.head_refs->lm_head == nullptr) Fail("the head has no lm_head");
    if (m.head_refs->lm_head->N != m.vocab || m.head_refs->lm_head->K != m.hidden) {
      Fail("lm_head is [" + std::to_string(m.head_refs->lm_head->N) + ", " + std::to_string(m.head_refs->lm_head->K) +
           "], expected [" + std::to_string(m.vocab) + ", " + std::to_string(m.hidden) + "]");
    }
    qwen4exp::HyperConnWeights w;
    w.norm = m.head_refs->hc_final.norm;
    if (m.head_refs->hc_final.inject != nullptr) Fail("the final mixer has no inject weight");
    m.head_views.push_back(BorrowedBf16Linear(m.head_refs->hc_final.down, cfg.hc_lowrank, m.wide, "hc_final.down"));
    w.down = &m.head_views.back();
    m.head_views.push_back(BorrowedBf16Linear(m.head_refs->hc_final.up, m.wide, cfg.hc_lowrank, "hc_final.up"));
    w.up = &m.head_views.back();
    m.hc_final = std::make_unique<qwen4exp::HyperConn>(hcc, w);
    m.logits_dev = core::DeviceBuffer<float>(static_cast<size_t>(m.vocab));
    m.argmax_dev = core::DeviceBuffer<int32_t>(1);
    m.greedy_host = core::PinnedBuffer<int32_t>(1);
    m.summary_ids_dev = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
    m.summary_vals_dev = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
    m.summary_lse_dev = core::DeviceBuffer<float>(1);
    m.topk_ws = core::DeviceBuffer<uint8_t>(static_cast<size_t>(r4dx_topk_lse_workspace_bytes()));
    bump(m.hc_final->ArenaBytes(R));
    bump(Padded(static_cast<size_t>(m.vocab) * 2));
  }

  // ---- stage-wide buffers -----------------------------------------------------------------------------------------
  m.h = core::DeviceBuffer<uint16_t>(static_cast<size_t>(R * m.wide));
  m.x = core::DeviceBuffer<uint16_t>(static_cast<size_t>(R * m.hidden));
  m.blk = core::DeviceBuffer<uint16_t>(static_cast<size_t>(R * m.hidden));
  m.inj = core::DeviceBuffer<uint16_t>(static_cast<size_t>(R * m.hc));
  m.emb_in = core::DeviceBuffer<uint16_t>(static_cast<size_t>(R * m.hidden));
  m.iota = core::DeviceBuffer<int32_t>(static_cast<size_t>(R));
  {
    std::vector<int32_t> it(static_cast<size_t>(R));
    for (int64_t i = 0; i < R; ++i) it[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    m.iota.CopyFromHost(it);
  }
  if (m.has_ple) m.ple_emb = core::DeviceBuffer<uint16_t>(static_cast<size_t>(R * ple_dim));
  if (m.has_qsa) {
    m.positions = core::DeviceBuffer<int32_t>(static_cast<size_t>(R));
    m.positions_host = core::PinnedBuffer<int32_t>(static_cast<size_t>(R));
  }
  // Grouped decode MoE buffers (M=1, zero CPU-GPU sync)
  const size_t kTopK = static_cast<size_t>(cfg.num_experts_per_tok);
  m.dmoe_logits = core::DeviceBuffer<uint16_t>(static_cast<size_t>(cfg.num_experts));
  m.dmoe_ids = core::DeviceBuffer<int32_t>(kTopK);
  m.dmoe_weights = core::DeviceBuffer<float>(kTopK);
  m.dmoe_ids_sorted = core::DeviceBuffer<int32_t>(kTopK);
  m.dmoe_weights_sorted = core::DeviceBuffer<float>(kTopK);
  m.dmoe_a0_gu = core::DeviceBuffer<uint16_t>(kTopK * static_cast<size_t>(m.hidden));
  m.dmoe_a1_gu = core::DeviceBuffer<uint16_t>(kTopK * static_cast<size_t>(m.hidden));
  m.dmoe_gu = core::DeviceBuffer<uint16_t>(kTopK * 1280);
  m.dmoe_ws_gu = core::DeviceBuffer<float>(kTopK * 5120);
  m.dmoe_a0_down = core::DeviceBuffer<uint16_t>(kTopK * 640);
  m.dmoe_yp = core::DeviceBuffer<uint16_t>(kTopK * static_cast<size_t>(m.hidden));
  m.dmoe_ws_down = core::DeviceBuffer<float>(kTopK * 5120);
  m.dmoe_shared_g = core::DeviceBuffer<uint16_t>(640);
  m.dmoe_shared_u = core::DeviceBuffer<uint16_t>(640);
  m.dmoe_shared_h = core::DeviceBuffer<uint16_t>(640);
  m.dmoe_shared_y = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.hidden));
  m.dmoe_shared_gate_logit = core::DeviceBuffer<uint16_t>(1);
  // Prewarm GDN control cache so it never calls hipMalloc during stream capture
  m.control.Prewarm(std::min<int64_t>(m.opt.max_rows, 256), 0, 1);

  // The arena: the largest block's demand, plus room for the ApplyLinear scratch of its (up to 64-row) chunks that the
  // sizes above leave out (trellis: the transformed A and the split-group partials; a handful of GEMMs per block).
  constexpr size_t kLinearScratch = size_t{256} << 20;
  m.arena.Reserve(block_arena + kLinearScratch);
  m.expert_arena.Reserve(size_t{96} << 20);
  R4DX_HIP_CHECK(hipStreamSynchronize(nullptr));  // the null-stream memsets / uploads above
  m.stream.Synchronize();
}

Qwen4ExpStage::~Qwen4ExpStage() {
  if (!impl_) return;
  Qwen4ExpDeviceScope dev(impl_->device);
  static_cast<void>(hipStreamSynchronize(impl_->stream.get()));
  impl_.reset();  // frees on the stage's device
}

int Qwen4ExpStage::Device() const { return impl_->device; }
hipStream_t Qwen4ExpStage::Stream() const { return impl_->stream.get(); }
int64_t Qwen4ExpStage::FirstLayer() const { return impl_->first; }
int64_t Qwen4ExpStage::EndLayer() const { return impl_->end; }
bool Qwen4ExpStage::HasHead() const { return impl_->head_refs.has_value(); }
bool Qwen4ExpStage::HasPle() const { return impl_->has_ple; }
double Qwen4ExpStage::LastRunMs() const { return impl_->last_run_ms; }
double Qwen4ExpStage::LastHeadMs() const { return impl_->last_head_ms; }
double Qwen4ExpStage::LastPleWaitMs() const { return impl_->last_ple_wait_ms; }

void Qwen4ExpStage::SetCapture(Qwen4ExpCaptureFn fn) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  m.capture = std::move(fn);
  if (m.capture && m.head_refs && m.logits_all.size() == 0) {
    m.logits_all = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.opt.max_rows * m.vocab));
  }
}

// ---- chunk input ----------------------------------------------------------------------------------------------------

void Qwen4ExpStage::LoadEmbedding(const uint16_t* host_rows, int64_t rows) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (rows < 1 || rows > m.opt.max_rows) Fail("LoadEmbedding: rows out of range");
  R4DX_HIP_CHECK(hipMemcpyAsync(m.emb_in.data(), host_rows, static_cast<size_t>(rows * m.hidden) * 2,
                                hipMemcpyHostToDevice, m.stream.get()));
  // table = the staged rows themselves, ids = 0 .. rows - 1: the gather degenerates to the repeat
  qwen4exp::EmbedRepeat(m.stream.get(), m.emb_in.data(), rows, m.iota.data(), rows, static_cast<int>(m.hidden),
                        static_cast<int>(m.hc), m.h.data());
}

void Qwen4ExpStage::LoadResidual(const uint16_t* host_rows, int64_t rows) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (rows < 1 || rows > m.opt.max_rows) Fail("LoadResidual: rows out of range");
  R4DX_HIP_CHECK(hipMemcpyAsync(m.h.data(), host_rows, static_cast<size_t>(rows * m.wide) * 2,
                                hipMemcpyHostToDevice, m.stream.get()));
}

void Qwen4ExpStage::StoreResidual(uint16_t* host_rows, int64_t rows) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (rows < 1 || rows > m.opt.max_rows) Fail("StoreResidual: rows out of range");
  R4DX_HIP_CHECK(hipMemcpyAsync(host_rows, m.h.data(), static_cast<size_t>(rows * m.wide) * 2,
                                hipMemcpyDeviceToHost, m.stream.get()));
  m.stream.Synchronize();
}

// ---- the layers -----------------------------------------------------------------------------------------------------

void Qwen4ExpStage::Impl::RunGdn(LayerRt& L, const Qwen4ExpChunk& c) {
  Qwen4ExpGdnCall call;
  call.is_prefill = c.is_prefill;
  call.has_init = c.first_pos > 0;
  call.prefill_recurrent = c.gdn_prefill_recurrent;
  L.gdn->Forward(stream.get(), arena, *L.gdn_states, control, x.data(), blk.data(), c.rows, call);
}

void Qwen4ExpStage::Impl::RunQsa(LayerRt& L, const Qwen4ExpChunk& c) {
  const hipStream_t s = stream.get();
  const int64_t T = c.rows;
  const int64_t q_width = cfg.num_attention_heads * cfg.head_dim;
  const int64_t kv_width = cfg.num_key_value_heads * cfg.head_dim;
  const int64_t idx_width = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim;
  uint16_t* qg = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * q_width));
  uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(T * kv_width));
  uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(T * kv_width));
  uint16_t* idxqk = arena.Alloc<uint16_t>(static_cast<size_t>(T * idx_width));
  qwen4exp::ApplyBf16Linear(s, arena, *L.qg, x.data(), qg, T);
  qwen4exp::ApplyBf16Linear(s, arena, *L.k, x.data(), k, T);
  qwen4exp::ApplyBf16Linear(s, arena, *L.v, x.data(), v, T);
  qwen4exp::ApplyBf16Linear(s, arena, *L.idx_qk, x.data(), idxqk, T);
  qwen4exp::QsaInputs in;
  in.qg = qg;
  in.k = k;
  in.v = v;
  in.idxqk = idxqk;
  uint16_t* attn = L.qsa_core->Forward(s, arena, in, L.qsa_w, *L.kv, static_cast<int>(c.first_pos),
                                       static_cast<int>(T), positions.data());
  CapBf16(L.index, "attn_pre_o", attn, T, q_width, c.first_pos);
  if (capture) {
    // the selection of the chunk's rows: ascending token ids (stride = SelectStride()) and counts, valid for the rows whose
    // query saw more complete blocks than the budget keeps (the others attend densely to [0, p])
    Cap(L.index, "sel_tokens", L.qsa_core->SelectedTokens(), Qwen4ExpCapture::Dtype::kI32, T, L.qsa_core->SelectStride(),
        c.first_pos);
    Cap(L.index, "sel_count", L.qsa_core->SelectedCounts(), Qwen4ExpCapture::Dtype::kI32, T, 1, c.first_pos);
  }
  qwen4exp::ApplyBf16Linear(s, arena, *L.o, attn, blk.data(), T);
}

void Qwen4ExpStage::Impl::RunMoeGroupedDecode(LayerRt& L) {
  const hipStream_t s = stream.get();
  const Qwen4ExpMoeRefs& e = *L.moe_refs;
  const int top_k = static_cast<int>(cfg.num_experts_per_tok);
  const int hidden_dim = static_cast<int>(cfg.hidden_size);

  // 1. Router GEMM
  qwen4exp::ApplyBf16Linear(s, arena, *L.moe_w.router, x.data(), dmoe_logits.data(), 1);

  // 2. Route kernel
  q4x_moe_route_bf16(dmoe_logits.data(), dmoe_ids.data(), dmoe_weights.data(), nullptr, 1,
                     static_cast<int>(cfg.num_experts), top_k, cfg.norm_topk_prob ? 1 : 0, s);

  // 3. Shared expert GEMMs (asynchronous)
  qwen4exp::ApplyBf16Linear(s, arena, *L.moe_w.shared_gate_proj, x.data(), dmoe_shared_g.data(), 1);
  qwen4exp::ApplyBf16Linear(s, arena, *L.moe_w.shared_up_proj, x.data(), dmoe_shared_u.data(), 1);
  q4x_moe_silu_mul_bf16(dmoe_shared_g.data(), dmoe_shared_u.data(), dmoe_shared_h.data(), 1, 640, 640, 640, s);
  qwen4exp::ApplyBf16Linear(s, arena, *L.moe_w.shared_down_proj, dmoe_shared_h.data(), dmoe_shared_y.data(), 1);
  qwen4exp::ApplyBf16Linear(s, arena, *L.moe_w.shared_gate, x.data(), dmoe_shared_gate_logit.data(), 1);

  // 4. Sort selected expert IDs on device
  qwen4exp::GroupedMoeDecodeSort(dmoe_ids.data(), dmoe_weights.data(),
                                 dmoe_ids_sorted.data(), dmoe_weights_sorted.data(),
                                 top_k, s);

  // 5. Input transform gate_up
  qwen4exp::GroupedTrellisInputGateUp(x.data(), dmoe_ids_sorted.data(),
                                     e.trellis_ptrs.gate_up_suh,
                                     dmoe_a0_gu.data(), dmoe_a1_gu.data(),
                                     top_k, s);

  // 6. GEMV gate_up
  qwen4exp::GroupedTrellisGemvGateUp(e.trellis_ptrs.gate_up_bits,
                                    dmoe_a0_gu.data(), dmoe_a1_gu.data(),
                                    dmoe_ids_sorted.data(),
                                    e.trellis_ptrs.gate_up_w,
                                    e.trellis_ptrs.gate_up_svh,
                                    dmoe_gu.data(), dmoe_ws_gu.data(),
                                    e.expert_gate_up[0]->trellis_tickets,
                                    top_k, s);

  // 7. Fused silu_mul + input transform down
  qwen4exp::GroupedSiluMulTrellisInputDown(dmoe_gu.data(), dmoe_ids_sorted.data(),
                                          e.trellis_ptrs.down_suh,
                                          dmoe_a0_down.data(),
                                          top_k, s);

  // 8. GEMV down
  qwen4exp::GroupedTrellisGemvDown(e.trellis_ptrs.down_bits,
                                  dmoe_a0_down.data(),
                                  dmoe_ids_sorted.data(),
                                  e.trellis_ptrs.down_w,
                                  e.trellis_ptrs.down_svh,
                                  dmoe_yp.data(), dmoe_ws_down.data(),
                                  e.expert_down[0]->trellis_tickets,
                                  top_k, s);

  // 9. Combine
  qwen4exp::GroupedMoeDecodeCombine(dmoe_yp.data(), dmoe_weights_sorted.data(),
                                   dmoe_shared_y.data(), dmoe_shared_gate_logit.data(),
                                   blk.data(), top_k, hidden_dim, s);
}

void Qwen4ExpStage::Impl::RunMoe(LayerRt& L, const Qwen4ExpChunk& c) {
  const hipStream_t s = stream.get();
  const Qwen4ExpMoeRefs& e = *L.moe_refs;
  if (c.rows == 1 && e.trellis_ptrs.gate_up_w != nullptr && !capture) {
    RunMoeGroupedDecode(L);
    return;
  }
  // One expert GEMM's ApplyLinear scratch comes from its own small arena, reset before each call: the experts' calls
  // (up to 2 per expert, hundreds per prefill chunk) would otherwise pile their scratch up in the block arena. Stream
  // order makes the reuse safe (every GEMM of the layer runs on `stream`, one after another).
  const qwen4exp::ExpertLinearFn fn = [&](int expert, qwen4exp::ExpertLinear which, const uint16_t* xin, int64_t rows,
                                          uint16_t* y, core::Arena& /*block_arena*/) {
    expert_arena.Reset();
    const QuantLinear* w = which == qwen4exp::ExpertLinear::kGateUp ? e.expert_gate_up[static_cast<size_t>(expert)]
                                                                     : e.expert_down[static_cast<size_t>(expert)];
    qwen4exp::ApplyBf16Linear(s, expert_arena, *w, xin, y, rows);
  };
  moe->Forward(s, arena, L.moe_w, x.data(), blk.data(), c.rows, fn);
  if (capture) {
    Cap(L.index, "topk_ids", moe->last_ids(), Qwen4ExpCapture::Dtype::kI32, c.rows, cfg.num_experts_per_tok,
        c.first_pos, /*on_host=*/true);
  }
}

void Qwen4ExpStage::RunLayers(const Qwen4ExpChunk& c) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  const auto t0 = Clock::now();
  m.last_ple_wait_ms = 0;
  m.last_first_pos = c.first_pos;
  if (c.rows < 1 || c.rows > m.opt.max_rows) Fail("RunLayers: rows out of range");
  if (c.first_pos < 0 || c.first_pos + c.rows > m.opt.max_ctx) {
    Fail("RunLayers: positions [" + std::to_string(c.first_pos) + ", " + std::to_string(c.first_pos + c.rows) +
         ") exceed max_ctx " + std::to_string(m.opt.max_ctx));
  }
  const hipStream_t s = m.stream.get();
  const int64_t T = c.rows;
  const bool is_decode = (T == 1 && !c.is_prefill && !m.capture);
  if (is_decode) ++m.decode_steps_seen;
  m.arena.Reset();
  if (m.has_qsa) {
    for (int64_t t = 0; t < T; ++t) m.positions_host[static_cast<size_t>(t)] = static_cast<int32_t>(c.first_pos + t);
    R4DX_HIP_CHECK(hipMemcpyAsync(m.positions.data(), m.positions_host.data(), static_cast<size_t>(T) * 4,
                                  hipMemcpyHostToDevice, s));
  }

  for (auto& lp : m.layers) {
    LayerRt& L = *lp;
    if (c.only_layer >= 0 && L.index != c.only_layer) continue;
    m.CapBf16(L.index, "residual_in", m.h.data(), T, m.wide, c.first_pos);

    if (L.ple) {
      if (!c.ple_rows) Fail("layer " + std::to_string(L.index) + " is a PLE layer but the chunk carries no n-gram rows");
      const auto tw = Clock::now();
      const uint16_t* rows_host = c.ple_rows();
      m.last_ple_wait_ms += MsSince(tw);
      if (rows_host == nullptr) Fail("the n-gram rows callback returned null");
      R4DX_HIP_CHECK(hipMemcpyAsync(m.ple_emb.data(), rows_host,
                                    static_cast<size_t>(T * m.cfg.ple_embed_dim) * 2, hipMemcpyHostToDevice, s));
      L.ple->Forward(s, m.arena, m.h.data(), m.ple_emb.data(), T);
      m.arena.Reset();
      m.CapBf16(L.index, "residual_ple", m.h.data(), T, m.wide, c.first_pos);
    }

    const Qwen4ExpMoeRefs& e = *L.moe_refs;

    // Steady-state decode fast path for GDN layers: capture & launch full layer graph
    if (is_decode && !L.ple && !L.qsa && e.trellis_ptrs.gate_up_w != nullptr) {
      if (L.decode_graph_exec != nullptr) {
        R4DX_HIP_CHECK(hipGraphLaunch(L.decode_graph_exec, s));
        continue;
      }
      if (m.decode_steps_seen > 1) {
        hipGraph_t g = nullptr;
        if (hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal) == hipSuccess) {
          L.hc_attn->Prep(s, m.arena, m.h.data(), 1, m.x.data(), m.inj.data());
          m.arena.Reset();
          m.RunGdn(L, c);
          m.arena.Reset();
          L.hc_attn->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), 1, m.h.data());
          L.hc_mlp->Prep(s, m.arena, m.h.data(), 1, m.x.data(), m.inj.data());
          m.arena.Reset();
          m.RunMoeGroupedDecode(L);
          m.arena.Reset();
          L.hc_mlp->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), 1, m.h.data());
          if (hipStreamEndCapture(s, &g) == hipSuccess &&
              hipGraphInstantiate(&L.decode_graph_exec, g, nullptr, nullptr, 0) == hipSuccess) {
            (void)hipGraphDestroy(g);
            R4DX_HIP_CHECK(hipGraphLaunch(L.decode_graph_exec, s));
            continue;
          }
          if (g) (void)hipGraphDestroy(g);
        }
      }
    }

    // ---- mixer block ----
    L.hc_attn->Prep(s, m.arena, m.h.data(), T, m.x.data(), m.inj.data());
    m.arena.Reset();
    m.CapBf16(L.index, "attn_mix", m.x.data(), T, m.hidden, c.first_pos);
    m.CapBf16(L.index, "attn_inject", m.inj.data(), T, m.hc, c.first_pos);
    if (L.qsa) {
      m.RunQsa(L, c);
    } else {
      m.RunGdn(L, c);
    }
    m.arena.Reset();
    m.CapBf16(L.index, "attn_out", m.blk.data(), T, m.hidden, c.first_pos);
    L.hc_attn->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), T, m.h.data());
    m.CapBf16(L.index, "residual_attn", m.h.data(), T, m.wide, c.first_pos);

    // ---- MoE block ----
    if (is_decode && e.trellis_ptrs.gate_up_w != nullptr) {
      if (L.moe_graph_exec != nullptr) {
        R4DX_HIP_CHECK(hipGraphLaunch(L.moe_graph_exec, s));
      } else if (m.decode_steps_seen > 1) {
        hipGraph_t g = nullptr;
        if (hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal) == hipSuccess) {
          L.hc_mlp->Prep(s, m.arena, m.h.data(), 1, m.x.data(), m.inj.data());
          m.arena.Reset();
          m.RunMoeGroupedDecode(L);
          m.arena.Reset();
          L.hc_mlp->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), 1, m.h.data());
          if (hipStreamEndCapture(s, &g) == hipSuccess &&
              hipGraphInstantiate(&L.moe_graph_exec, g, nullptr, nullptr, 0) == hipSuccess) {
            (void)hipGraphDestroy(g);
            R4DX_HIP_CHECK(hipGraphLaunch(L.moe_graph_exec, s));
          } else {
            if (g) (void)hipGraphDestroy(g);
            L.hc_mlp->Prep(s, m.arena, m.h.data(), 1, m.x.data(), m.inj.data());
            m.arena.Reset();
            m.RunMoeGroupedDecode(L);
            m.arena.Reset();
            L.hc_mlp->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), 1, m.h.data());
          }
        } else {
          L.hc_mlp->Prep(s, m.arena, m.h.data(), 1, m.x.data(), m.inj.data());
          m.arena.Reset();
          m.RunMoeGroupedDecode(L);
          m.arena.Reset();
          L.hc_mlp->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), 1, m.h.data());
        }
      } else {
        L.hc_mlp->Prep(s, m.arena, m.h.data(), 1, m.x.data(), m.inj.data());
        m.arena.Reset();
        m.RunMoeGroupedDecode(L);
        m.arena.Reset();
        L.hc_mlp->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), 1, m.h.data());
      }
    } else {
      L.hc_mlp->Prep(s, m.arena, m.h.data(), T, m.x.data(), m.inj.data());
      m.arena.Reset();
      m.CapBf16(L.index, "mlp_mix", m.x.data(), T, m.hidden, c.first_pos);
      m.CapBf16(L.index, "mlp_inject", m.inj.data(), T, m.hc, c.first_pos);
      m.RunMoe(L, c);
      m.arena.Reset();
      m.CapBf16(L.index, "moe_out", m.blk.data(), T, m.hidden, c.first_pos);
      L.hc_mlp->Inject(s, m.h.data(), m.blk.data(), m.inj.data(), T, m.h.data());
    }
    m.CapBf16(L.index, "residual_out", m.h.data(), T, m.wide, c.first_pos);
  }
  if (!is_decode) {
    m.stream.Synchronize();
    m.last_run_ms = MsSince(t0);
  }
}

// ---- head -----------------------------------------------------------------------------------------------------------

Qwen4ExpHeadResult Qwen4ExpStage::Head(int64_t rows, const Qwen4ExpHeadRequest& req) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (!m.head_refs) Fail("Head: this stage has no head");
  if (rows < 1 || rows > m.opt.max_rows) Fail("Head: rows out of range");
  const auto t0 = Clock::now();
  const hipStream_t s = m.stream.get();
  const QuantLinear& lm = *m.head_refs->lm_head;
  m.arena.Reset();

  if (m.capture) {
    // every row of the chunk (tests): the mixer, then the lm_head over all of them
    m.hc_final->Mix(s, m.arena, m.h.data(), rows, m.x.data());
    m.arena.Reset();
    m.Cap(-1, "mixer_out", m.x.data(), Qwen4ExpCapture::Dtype::kBf16, rows, m.hidden, m.last_first_pos);
    qwen4exp::ApplyBf16Linear(s, m.arena, lm, m.x.data(), m.logits_all.data(), rows);
    m.arena.Reset();
    m.Cap(-1, "logits", m.logits_all.data(), Qwen4ExpCapture::Dtype::kBf16, rows, m.vocab, m.last_first_pos);
  }

  if (req.kind == Qwen4ExpHeadRequest::Kind::kAllArgmax) {
    m.hc_final->Mix(s, m.arena, m.h.data(), rows, m.x.data());
    uint16_t* logits_bf16 = m.arena.Alloc<uint16_t>(static_cast<size_t>(rows * m.vocab));
    qwen4exp::ApplyBf16Linear(s, m.arena, lm, m.x.data(), logits_bf16, rows);
    Qwen4ExpHeadResult out;
    out.tokens.resize(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) {
      r4dx_argmax_bf16(reinterpret_cast<int64_t>(logits_bf16 + r * m.vocab),
                       reinterpret_cast<int64_t>(m.argmax_dev.data()), m.vocab, reinterpret_cast<int64_t>(s));
      R4DX_HIP_CHECK(hipMemcpyAsync(m.greedy_host.data(), m.argmax_dev.data(), sizeof(int32_t),
                                    hipMemcpyDeviceToHost, s));
      m.stream.Synchronize();
      out.tokens[static_cast<size_t>(r)] = m.greedy_host[0];
    }
    out.token = out.tokens.back();
    m.last_head_ms = MsSince(t0);
    return out;
  }

  // the last row only: the mixer and the lm_head are row-independent
  const uint16_t* h_last = m.h.data() + (rows - 1) * m.wide;
  m.hc_final->Mix(s, m.arena, h_last, 1, m.x.data());
  uint16_t* logits_bf16 = m.arena.Alloc<uint16_t>(static_cast<size_t>(m.vocab));
  qwen4exp::ApplyBf16Linear(s, m.arena, lm, m.x.data(), logits_bf16, 1);

  Qwen4ExpHeadResult out;
  switch (req.kind) {
    case Qwen4ExpHeadRequest::Kind::kAllArgmax:
      break;
    case Qwen4ExpHeadRequest::Kind::kLogits: {
      r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_dev.data()),
                                   m.vocab, reinterpret_cast<int64_t>(s));
      m.stream.Synchronize();
      out.logits.resize(static_cast<size_t>(m.vocab));
      m.logits_dev.CopyToHost(out.logits.data(), static_cast<size_t>(m.vocab));
      break;
    }
    case Qwen4ExpHeadRequest::Kind::kArgmax: {
      r4dx_argmax_bf16(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.argmax_dev.data()), m.vocab,
                       reinterpret_cast<int64_t>(s));
      R4DX_HIP_CHECK(hipMemcpyAsync(m.greedy_host.data(), m.argmax_dev.data(), sizeof(int32_t), hipMemcpyDeviceToHost, s));
      m.stream.Synchronize();
      out.token = m.greedy_host[0];
      break;
    }
    case Qwen4ExpHeadRequest::Kind::kSummary: {
      if (!(req.inv_temperature > 0.0f)) Fail("Head: kSummary needs inv_temperature > 0");
      r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_dev.data()),
                                   m.vocab, reinterpret_cast<int64_t>(s));
      r4dx_topk_lse_f32_ws(reinterpret_cast<int64_t>(m.logits_dev.data()), reinterpret_cast<int64_t>(m.summary_ids_dev.data()),
                           reinterpret_cast<int64_t>(m.summary_vals_dev.data()),
                           reinterpret_cast<int64_t>(m.summary_lse_dev.data()), 1, m.vocab, req.inv_temperature,
                           reinterpret_cast<int64_t>(s), reinterpret_cast<int64_t>(m.topk_ws.data()));
      m.stream.Synchronize();
      std::vector<int32_t> ids(R4DX_TOPK_LSE_K);
      std::vector<float> vals(R4DX_TOPK_LSE_K);
      float lse = 0.0f;
      m.summary_ids_dev.CopyToHost(ids.data(), ids.size());
      m.summary_vals_dev.CopyToHost(vals.data(), vals.size());
      m.summary_lse_dev.CopyToHost(&lse, 1);
      kernels::RowSummary& sm = out.summary;
      sm.k = R4DX_TOPK_LSE_K;
      sm.vocab = m.vocab;
      sm.inv_temperature = req.inv_temperature;
      sm.lse = lse;
      for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
        sm.ids[j] = ids[static_cast<size_t>(j)];
        sm.vals[j] = vals[static_cast<size_t>(j)];
      }
      break;
    }
  }
  m.arena.Reset();
  m.last_head_ms = MsSince(t0);
  return out;
}

std::vector<float> Qwen4ExpStage::LastLogitsRow() {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (!m.head_refs) Fail("LastLogitsRow: this stage has no head");
  std::vector<float> row(static_cast<size_t>(m.vocab));
  m.logits_dev.CopyToHost(row.data(), row.size());
  return row;
}

// ---- state ----------------------------------------------------------------------------------------------------------

void Qwen4ExpStage::Reset() {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  const hipStream_t s = m.stream.get();
  for (auto& lp : m.layers) {
    LayerRt& L = *lp;
    if (L.gdn_states) L.gdn_states->ZeroAll(m.stream);
    if (L.ple) L.ple->Reset(s);
    if (L.qsa_core) L.qsa_core->Reset();
  }
  m.stream.Synchronize();
  m.arena.Reset();
  m.expert_arena.Reset();
}

void Qwen4ExpStage::SaveCheckpoint() {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (!m.opt.prompt_checkpoint) Fail("SaveCheckpoint: the stage was built without prompt_checkpoint");
  for (auto& lp : m.layers) {
    LayerRt& L = *lp;
    if (L.gdn_states) L.gdn_states->SaveCheckpoint(0, m.stream);
    if (L.ple) L.ple->SaveState(m.stream.get(), L.ple_ckpt.data());
  }
  m.stream.Synchronize();
}

void Qwen4ExpStage::RestoreCheckpoint(int64_t pos) {
  Impl& m = *impl_;
  Qwen4ExpDeviceScope dev(m.device);
  if (!m.opt.prompt_checkpoint) Fail("RestoreCheckpoint: the stage was built without prompt_checkpoint");
  for (auto& lp : m.layers) {
    LayerRt& L = *lp;
    if (L.gdn_states) L.gdn_states->RestoreCheckpoint(0, m.stream);
    if (L.ple) L.ple->LoadState(m.stream.get(), L.ple_ckpt.data());
    if (L.qsa_core) L.qsa_core->TruncateTo(static_cast<int>(pos));
  }
  m.stream.Synchronize();
  m.arena.Reset();
  m.expert_arena.Reset();
}

}  // namespace r4dx::model
