#include "qwen35moe_stage.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <type_traits>
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
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_block.h"
#include "tp/tp_vocab.h"

namespace r4dx::model {

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

constexpr const char* kCtx = "Qwen35MoeStage";

[[noreturn]] void Fail(const std::string& msg) { qwen4exp::Fail(kCtx, msg); }

using qwen4exp::Padded;

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

Qwen35MoeDeviceScope::Qwen35MoeDeviceScope(int device) {
  R4DX_HIP_CHECK(hipGetDevice(&previous_));
  if (previous_ != device) {
    R4DX_HIP_CHECK(hipSetDevice(device));
    changed_ = true;
  }
}

Qwen35MoeDeviceScope::~Qwen35MoeDeviceScope() {
  if (changed_) static_cast<void>(hipSetDevice(previous_));
}

// ---- one layer's runtime objects ------------------------------------------------------------------------------------
namespace {

struct LayerRt {
  int64_t index = 0;
  bool full = false;
  std::deque<QuantLinear> views;  // borrowed bf16 linears (stable addresses)
  const uint16_t* input_ln = nullptr;
  const uint16_t* post_ln = nullptr;

  // GDN
  std::unique_ptr<GdnStateManager> gdn_states;
  std::unique_ptr<Qwen4ExpGdn> gdn;

  // full attention
  std::unique_ptr<attention::AttentionLayer> attn;
  std::unique_ptr<attention::PagedKvCache> kv;
  attention::AttnWeights attn_w;

  // MoE
  qwen4exp::MoeWeights moe_w;
  Qwen35GroupedMoeWeights gmoe_w;              // the same router / shared-expert pointers for the grouped decode MoE
  const Qwen35MoeMoeRefs* moe_refs = nullptr;  // points into Impl::layer_refs
  bool grouped = false;                        // decode MoE takes Qwen35GroupedMoeDecode (geometry + trellis experts qualify)
};

}  // namespace

struct Qwen35MoeStage::Impl {
  Qwen35MoeStageOptions opt;
  Qwen35MoeConfig cfg;
  int device = 0;
  int64_t first = 0, end = 0;
  int64_t hidden = 0, vocab = 0;
  // --tp 2: this rank's endpoint and its slice of the vocabulary (vocab_local == vocab, vocab_offset == 0 at TP=1)
  core::TpComm* comm = nullptr;
  int64_t vocab_local = 0, vocab_offset = 0;
  core::DeviceBuffer<int32_t> argmax_pair_dev;  // [2]: {local index, value bits} of the local argmax (TP)
  std::vector<float> gather_shard_host;
  tp::SubmitBounder submit;  // inert unless TP with tp_submit_layers > 0
  int64_t submit_layers = 0;

  core::Stream stream;
  core::Arena arena;          // per-block scratch (reset between blocks)
  core::Arena expert_arena;   // ApplyLinear scratch of ONE expert GEMM (reset before each, see RunMoe)

  core::DeviceBuffer<uint16_t> h;        // residual [max_rows, hidden]
  core::DeviceBuffer<uint16_t> hmid;     // residual after the mixer [max_rows, hidden]
  core::DeviceBuffer<uint16_t> x;        // normed block input [max_rows, hidden]
  core::DeviceBuffer<uint16_t> blk;      // block output [max_rows, hidden]
  core::DeviceBuffer<int32_t> positions; // [max_rows]
  core::DeviceBuffer<int32_t> seqused_k; // [1]
  core::DeviceBuffer<int32_t> seqused_k_slices;  // [max_rows / 64]
  core::DeviceBuffer<int32_t> seqused_sub;       // [max_rows] verify windows: the causal context of each decode sub-launch
  int verify_sub = 4;                            // rows per decode-kernel launch of a verify window's attention (0: the prefill kernel)
  core::PinnedBuffer<int32_t> positions_host, seqused_host, slices_host, sub_host;

  GdnControlCache control;
  std::unique_ptr<qwen4exp::MoeBlockBase> moe;
  std::unique_ptr<Qwen35GroupedMoeDecode> gmoe;  // the M = 1 on-device MoE (null when no layer qualifies)
  bool grouped_on = false;                      // gmoe is used by the layers that qualify
  int gmoe_rows_cap = 1;                        // verify windows of 2..cap rows take Qwen35GroupedMoeDecode::RunRows
  bool rows_moe_on = true;                      // R4DX_Q35_ROWS=0: verify windows keep the host-routed MoE (diagnostics / A-B)
  bool grouped_capture = false;                 // ... also while a capture is installed (tests)
  std::vector<Qwen35MoeLayerRefs> layer_refs;
  std::vector<std::unique_ptr<LayerRt>> layers;
  // head
  std::optional<Qwen35MoeHeadRefs> head_refs;
  core::DeviceBuffer<float> logits_dev;        // [vocab] fp32, the last head call's row
  core::DeviceBuffer<int32_t> argmax_dev;      // [1]
  core::PinnedBuffer<int32_t> greedy_host;     // [1]
  core::DeviceBuffer<int32_t> summary_ids_dev; // [K]
  core::DeviceBuffer<float> summary_vals_dev;  // [K]
  core::DeviceBuffer<float> summary_lse_dev;   // [1]
  core::DeviceBuffer<uint8_t> topk_ws;         // r4dx_topk_lse_workspace_bytes()
  core::DeviceBuffer<uint16_t> logits_all;     // capture mode: [max_rows, vocab] bf16
  core::DeviceBuffer<uint16_t> norm_all;       // capture mode: [max_rows, hidden]

  // ---- speculative window (spec_window > 1): the 27B Model's bookkeeping (mtp_num_accepted_dev_, gdn_book_) per stage -------
  int64_t spec_window = 1;
  core::DeviceBuffer<int32_t> num_accepted_dev;  // [1] rows the last window committed (conv history offset / replay length)
  core::PinnedBuffer<int32_t> num_accepted_host; // [1] staging of the async write above
  bool num_accepted_valid = false;
  int64_t last_committed = 1;
  GdnPendingBook book;
  core::DeviceBuffer<float> logits_rows;         // [spec_window, vocab] fp32 (HeadRows)
  core::DeviceBuffer<int32_t> argmax_rows_dev;   // [spec_window]
  core::PinnedBuffer<int32_t> argmax_rows_host;  // [spec_window]

  // ---- R4DX_Q35_PROFILE=1: per-kind GPU time of the verify chunks (events around each layer's mixer and MoE), printed at destruction --
  bool prof_on = false;
  std::vector<hipEvent_t> prof_ev;       // [3 * layers + 1]
  double prof_ms[3] = {0, 0, 0};         // gdn, attention, moe
  int64_t prof_calls = 0, prof_rows = 0;
  double prof_enqueue_ms = 0, prof_wall_ms = 0;  // host time to enqueue a verify chunk's layers / until the stream is done
  bool prof_phase = false;               // R4DX_Q35_PROFILE=2: the GDN layers' phases
  hipEvent_t prof_phase_ev[4];
  double prof_phase_ms[3] = {0, 0, 0};   // proj, scan, out_proj
  int64_t prof_phase_calls = 0;

  // ---- DFlash2 feature capture ---------------------------------------------------------------------------------------
  std::vector<Qwen35MoeFeatureTap> taps;
  int64_t tap_pitch = 0;
  uint16_t* tap_dst = nullptr;
  // The taps of a stage sit in consecutive columns of the host buffer (the model assigns them in layer order), so the rows are gathered
  // on the device into tap_dev ([rows, ntaps * hidden], one small D2D copy per tap) and leave for the host in ONE 2D copy at the end of
  // the chunk: 8 separate strided D2H copies per step cost ~3 ms of stream bubbles on a 17 ms decode step.
  bool tap_contiguous = false;
  core::DeviceBuffer<uint16_t> tap_dev;

  Qwen35MoeCaptureFn capture;
  double last_run_ms = 0, last_head_ms = 0;
  // EnableStepProfile: hipEvents around the layers of a RunLayers call ([0], [1]) and around a Head call ([2], [3]); the GPU spans, not host time.
  bool step_prof = false;
  hipEvent_t sp_ev[4] = {nullptr, nullptr, nullptr, nullptr};
  void SpMark(int i) {
    if (step_prof) R4DX_HIP_CHECK(hipEventRecord(sp_ev[i], stream.get()));
  }
  int64_t last_first_pos = 0;  // first position of the chunk RunLayers ran last (the head's capture rows start there)

  void Cap(int64_t layer, const char* suffix, const void* dev, Qwen35MoeCapture::Dtype dt, int64_t rows, int64_t cols,
           int64_t first_pos, bool on_host = false) {
    if (!capture) return;
    Qwen35MoeCapture c;
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
    Cap(layer, suffix, dev, Qwen35MoeCapture::Dtype::kBf16, rows, cols, first_pos);
  }

  void Norm(const uint16_t* in, const uint16_t* weight, uint16_t* out, int64_t rows) {
    r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(in), reinterpret_cast<int64_t>(weight), reinterpret_cast<int64_t>(out), rows,
                      hidden, static_cast<float>(cfg.rms_norm_eps), reinterpret_cast<int64_t>(stream.get()));
  }
  void ResidualAdd(const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t rows) {
    r4dx_residual_add_bf16(reinterpret_cast<int64_t>(a), reinterpret_cast<int64_t>(b), reinterpret_cast<int64_t>(out),
                           rows * hidden, reinterpret_cast<int64_t>(stream.get()));
  }

  // Prefill after speculation: replays the committed prefix of the last log into every GDN state and moves the conv history back
  // to offset 0 (Model::CollapseSpeculativeWindow).
  void CollapseWindow() {
    if (spec_window <= 1) return;
    if (book.Pending()) {
      for (auto& lp : layers) {
        if (lp->gdn_states) lp->gdn_states->Materialize(0, book.Count(), stream);
      }
    }
    book.Clear();
    if (num_accepted_valid && last_committed > 1) {
      for (auto& lp : layers) {
        if (lp->gdn_states) lp->gdn_states->ShiftConv(0, last_committed, stream);
      }
    }
    num_accepted_valid = false;
    last_committed = 1;
  }
  void ResetSpec() {
    num_accepted_valid = false;
    last_committed = 1;
    book.Clear();
  }
  // Async write of `n` into num_accepted_dev. The value is staged in pinned memory: slot 0 for a commit (written only while the
  // device is idle -- the previous verify synchronized), slot 1 holds the constant 1 of a plain decode step for good, so the
  // host never rewrites a staging word an enqueued copy has yet to read.
  void WriteNumAccepted(int32_t n, bool plain_step = false) {
    int32_t* src = num_accepted_host.data();
    if (plain_step) {
      src += 1;
    } else {
      src[0] = n;
    }
    R4DX_HIP_CHECK(hipMemcpyAsync(num_accepted_dev.data(), src, sizeof(int32_t), hipMemcpyHostToDevice, stream.get()));
  }

  // --tp 2: sums blk [rows, hidden] over the ranks in place (bf16 RNE of the two fp32-added partials, bit-identical on both ranks).
  void AllReduceBlock(int64_t rows) {
    if (comm != nullptr) comm->AllReduceSumBf16Rows(blk.data(), rows, hidden, stream.get());
  }
  // --tp 2: the full fp32 logits row from the ranks' vocab slices (a collective: every rank calls it, in rank order).
  std::vector<float> GatherVocabRow() {
    gather_shard_host.resize(static_cast<size_t>(vocab_local));
    logits_dev.CopyToHost(gather_shard_host.data(), gather_shard_host.size());
    std::vector<float> full(static_cast<size_t>(vocab));
    comm->HostAllGather(gather_shard_host.data(), gather_shard_host.size() * sizeof(float), full.data());
    return full;
  }
  void RunGdn(LayerRt& L, const Qwen35MoeChunk& c);
  void RunAttention(LayerRt& L, const Qwen35MoeChunk& c);
  void RunMoe(LayerRt& L, const Qwen35MoeChunk& c);
};

// The stage's core::Stream is created by Impl's default constructor, so Impl itself must be constructed with the stage's
// device current (a stream belongs to the device that was current when it was created).
std::unique_ptr<Qwen35MoeStage::Impl> MakeQ35StageImpl(int device) {
  Qwen35MoeDeviceScope dev(device);
  return std::make_unique<Qwen35MoeStage::Impl>();
}

// ---- construction ---------------------------------------------------------------------------------------------------

Qwen35MoeStage::Qwen35MoeStage(const Qwen35MoeStageOptions& options, std::vector<Qwen35MoeLayerRefs> layers,
                               std::optional<Qwen35MoeHeadRefs> head)
    : impl_(MakeQ35StageImpl(options.device)) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(options.device);
  m.opt = options;
  m.cfg = options.config;
  m.device = options.device;
  const Qwen35MoeConfig& cfg = m.cfg;
  if (layers.empty()) Fail("no layers");
  if (options.max_rows < 1 || options.max_rows > 256 || (options.max_rows > 64 && options.max_rows % 64 != 0)) {
    Fail("max_rows must be in [1, 64] or a multiple of 64 up to 256");
  }
  if (options.max_ctx < 1) Fail("max_ctx must be >= 1");
  if (options.spec_window < 1 || options.spec_window > std::min<int64_t>(options.max_rows, 64)) {
    Fail("spec_window must be in [1, min(max_rows, 64)]");
  }
  m.spec_window = options.spec_window;
  m.first = layers.front().index;
  m.end = m.first + static_cast<int64_t>(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    if (layers[i].index != m.first + static_cast<int64_t>(i)) Fail("layers are not consecutive");
    if (layers[i].full_attention == cfg.IsGdnLayer(layers[i].index)) {
      Fail("layer " + std::to_string(layers[i].index) + " is a " + (layers[i].full_attention ? "full-attention" : "GDN") +
           " layer in the weights but a " + (cfg.IsGdnLayer(layers[i].index) ? "GDN" : "full-attention") +
           " layer in the config");
    }
  }
  m.hidden = cfg.hidden_size;
  m.vocab = cfg.vocab_size;
  m.comm = options.tp_comm;
  m.vocab_local = cfg.VocabShardSize();
  m.vocab_offset = cfg.VocabShardBegin();
  if (cfg.tp_world > 1) {
    if (m.comm == nullptr || m.comm->World() != cfg.tp_world || m.comm->Rank() != cfg.tp_rank) {
      Fail("a tensor-parallel stage (config.tp_world > 1) needs a TpComm endpoint of the same world and rank");
    }
    if (options.tp_submit_layers < 0 || options.tp_submit_layers > 64 || options.tp_max_inflight_units < 0 || options.tp_max_inflight_units > 64) {
      Fail("tp_submit_layers and tp_max_inflight_units must be in [0, 64]");
    }
    if (options.tp_submit_layers > 0) {
      m.submit = tp::SubmitBounder(options.tp_max_inflight_units);  // its events live on this (the stage's) device
      m.submit_layers = options.tp_submit_layers;
    }
  } else if (m.comm != nullptr) {
    Fail("tp_comm is set but config.tp_world is 1");
  }
  const int64_t R = options.max_rows;

  m.layer_refs = std::move(layers);
  m.head_refs = std::move(head);

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

  attention::AttnConfig ac;
  ac.hidden = static_cast<int>(cfg.hidden_size);
  ac.num_heads = static_cast<int>(cfg.num_attention_heads);
  ac.kv_heads = static_cast<int>(cfg.num_key_value_heads);
  ac.head_dim = static_cast<int>(cfg.head_dim);
  ac.rotary_dim = static_cast<int>(cfg.RotaryDim());
  ac.rope_theta = static_cast<float>(cfg.rope_theta);
  ac.rms_eps = static_cast<float>(cfg.rms_norm_eps);
  ac.mrope_section_t = static_cast<int>(cfg.mrope_section.at(0));
  ac.mrope_section_h = static_cast<int>(cfg.mrope_section.at(1));
  ac.mrope_section_w = static_cast<int>(cfg.mrope_section.at(2));
  ac.comm = m.comm;  // the o_proj all-reduce of site A2 (nullptr at TP=1)

  GdnGeometry geo;
  geo.hidden = cfg.hidden_size;
  geo.key_heads = cfg.linear_num_key_heads;
  geo.value_heads = cfg.linear_num_value_heads;
  geo.key_dim = cfg.linear_key_head_dim;
  geo.value_dim = cfg.linear_value_head_dim;
  geo.conv_kernel = cfg.linear_conv_kernel_dim;
  geo.rms_eps = cfg.rms_norm_eps;
  geo.sigmoid_gate = false;  // the Qwen3.5 gated norm gates with silu

  const int64_t q_width = cfg.num_attention_heads * cfg.head_dim;
  const int64_t kv_width = cfg.num_key_value_heads * cfg.head_dim;

  for (const Qwen35MoeLayerRefs& r : m.layer_refs) {
    auto L = std::make_unique<LayerRt>();
    L->index = r.index;
    L->full = r.full_attention;
    if (r.input_layernorm == nullptr || r.post_attention_layernorm == nullptr) {
      Fail("layer " + std::to_string(r.index) + ": missing layernorm weight");
    }
    L->input_ln = r.input_layernorm;
    L->post_ln = r.post_attention_layernorm;

    if (!r.full_attention) {
      L->gdn = std::make_unique<Qwen4ExpGdn>(geo, r.gdn);
      L->gdn_states = std::make_unique<GdnStateManager>(
          /*max_seqs=*/1, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim,
          L->gdn->ConvDim(), cfg.linear_conv_kernel_dim, /*max_decode_window=*/options.spec_window,
          /*write_once=*/options.spec_window > 1);
      L->gdn_states->ZeroAll(m.stream);
      if (options.prompt_checkpoint) L->gdn_states->AllocateCheckpoint();
      bump(L->gdn->ArenaBytes(R));
    } else {
      const Qwen35MoeAttnRefs& a = r.attn;
      for (const QuantLinear* q : {a.qg, a.k, a.v, a.o}) {
        if (q == nullptr) Fail("layer " + std::to_string(r.index) + ": missing attention linear");
      }
      if (a.qg->N != 2 * q_width || a.qg->K != m.hidden || a.k->N != kv_width || a.v->N != kv_width ||
          a.o->N != m.hidden || a.o->K != q_width) {
        Fail("layer " + std::to_string(r.index) + ": attention linear shapes do not match the config");
      }
      if (a.q_norm == nullptr || a.k_norm == nullptr || a.k_descale == nullptr || a.v_descale == nullptr) {
        Fail("layer " + std::to_string(r.index) + ": missing attention norm / descale");
      }
      L->attn_w.input_layernorm = r.input_layernorm;
      L->attn_w.qg = a.qg;
      L->attn_w.k = a.k;
      L->attn_w.v = a.v;
      L->attn_w.o = a.o;
      L->attn_w.q_norm = a.q_norm;
      L->attn_w.k_norm = a.k_norm;
      L->attn_w.k_descale = a.k_descale;
      L->attn_w.v_descale = a.v_descale;
      L->attn = std::make_unique<attention::AttentionLayer>(ac);
      L->kv = std::make_unique<attention::PagedKvCache>(ac.kv_heads, ac.head_dim, /*block_size=*/16,
                                                        static_cast<int>(options.max_ctx));
    }

    // MoE
    const Qwen35MoeMoeRefs& e = r.moe;
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
    L->gmoe_w = {L->moe_w.router, L->moe_w.shared_gate_proj, L->moe_w.shared_up_proj, L->moe_w.shared_down_proj,
                 L->moe_w.shared_gate};
    L->moe_refs = &e;
    // The grouped kernels fix the input / output scale at 1/sqrt(128): a container with a trellis prescale keeps the host-routed MoE.
    L->grouped = Qwen35GroupedMoeDecode::Supported(cfg.hidden_size, cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size,
                                                   cfg.num_experts, cfg.num_experts_per_tok, e.trellis) &&
                 e.expert_gate_up[0]->layout == Layout::kTrellis && e.expert_gate_up[0]->trellis_prescale_log2 == 0 &&
                 e.expert_down[0]->layout == Layout::kTrellis && e.expert_down[0]->trellis_prescale_log2 == 0;
    m.layers.push_back(std::move(L));
  }

  // The grouped decode MoE: one scratch set for the stage (its layers run one after another on the stream).
  if (std::any_of(m.layers.begin(), m.layers.end(), [](const auto& lp) { return lp->grouped; })) {
    const int rows_cap = options.spec_window > 1
                             ? static_cast<int>(std::min<int64_t>(options.spec_window, Qwen35GroupedMoeDecode::kMaxRows))
                             : 1;
    m.gmoe = std::make_unique<Qwen35GroupedMoeDecode>(static_cast<int>(cfg.num_experts), static_cast<int>(cfg.num_experts_per_tok),
                                                      cfg.norm_topk_prob, rows_cap, static_cast<int>(cfg.moe_intermediate_size));
    m.gmoe_rows_cap = rows_cap;
    if (const char* e = std::getenv("R4DX_Q35_ROWS"); e != nullptr && std::strcmp(e, "0") == 0) m.rows_moe_on = false;
    int mode = options.grouped_decode;
    if (mode < 0) {
      const char* env = std::getenv("R4DX_Q35_GROUPED");
      mode = (env != nullptr && std::strcmp(env, "0") == 0) ? 0 : 1;
    }
    m.grouped_on = mode != 0;
  }

  // ---- head ---------------------------------------------------------------------------------------------------------
  if (m.head_refs) {
    if (m.head_refs->lm_head == nullptr || m.head_refs->final_norm == nullptr) Fail("the head has no lm_head / final_norm");
    if (m.head_refs->lm_head->N != m.vocab_local || m.head_refs->lm_head->K != m.hidden) {
      Fail("lm_head is [" + std::to_string(m.head_refs->lm_head->N) + ", " + std::to_string(m.head_refs->lm_head->K) +
           "], expected [" + std::to_string(m.vocab_local) + ", " + std::to_string(m.hidden) + "]");
    }
    m.logits_dev = core::DeviceBuffer<float>(static_cast<size_t>(m.vocab_local));
    if (m.comm != nullptr) m.argmax_pair_dev = core::DeviceBuffer<int32_t>(2);
    m.argmax_dev = core::DeviceBuffer<int32_t>(1);
    m.greedy_host = core::PinnedBuffer<int32_t>(1);
    m.summary_ids_dev = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
    m.summary_vals_dev = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
    m.summary_lse_dev = core::DeviceBuffer<float>(1);
    m.topk_ws = core::DeviceBuffer<uint8_t>(static_cast<size_t>(r4dx_topk_lse_workspace_bytes()));
    bump(Padded(static_cast<size_t>(m.vocab_local) * 2));
    if (options.spec_window > 1) {
      m.logits_rows = core::DeviceBuffer<float>(static_cast<size_t>(options.spec_window * m.vocab_local));
      // [window] row indices; under TP the {index, value} pairs of the rank's vocab slice ([window, 2] ints)
      m.argmax_rows_dev = core::DeviceBuffer<int32_t>(static_cast<size_t>(options.spec_window) * (m.comm != nullptr ? 2 : 1));
      m.argmax_rows_host = core::PinnedBuffer<int32_t>(static_cast<size_t>(options.spec_window) * (m.comm != nullptr ? 2 : 1));
      bump(Padded(static_cast<size_t>(options.spec_window * m.vocab_local) * 2));
    }
  }
  if (options.spec_window > 1) {
    m.num_accepted_dev = core::DeviceBuffer<int32_t>(1);
    m.num_accepted_host = core::PinnedBuffer<int32_t>(2);
    m.num_accepted_host[0] = 0;
    m.num_accepted_host[1] = 1;  // the constant a plain decode step writes
    m.num_accepted_dev.Zero();
    // The control arrays of the window path, uploaded now (a first use is a blocking hipMalloc + copy in the middle of a step).
    const int32_t slot = m.layers.empty() || !m.layers.front()->gdn_states ? 1 : m.layers.front()->gdn_states->SlotForSeq(0);
    m.control.Prewarm(options.spec_window, slot, options.spec_window);
  }

  // ---- stage-wide buffers -----------------------------------------------------------------------------------------
  const size_t rh = static_cast<size_t>(R * m.hidden);
  m.h = core::DeviceBuffer<uint16_t>(rh);
  m.hmid = core::DeviceBuffer<uint16_t>(rh);
  m.x = core::DeviceBuffer<uint16_t>(rh);
  m.blk = core::DeviceBuffer<uint16_t>(rh);
  m.positions = core::DeviceBuffer<int32_t>(static_cast<size_t>(R));
  m.seqused_k = core::DeviceBuffer<int32_t>(1);
  m.seqused_k_slices = core::DeviceBuffer<int32_t>(static_cast<size_t>(std::max<int64_t>(1, R / 64)));
  m.seqused_sub = core::DeviceBuffer<int32_t>(static_cast<size_t>(R));
  m.sub_host = core::PinnedBuffer<int32_t>(static_cast<size_t>(R));
  // The decode kernel serves q_len * gqa <= 64 rows (4 here); its split law (decode_splits_g16) depends on the cache capacity and the kv heads,
  // not on q_len, so a q_len = 4 launch rounds every row exactly like a q_len = 1 launch (audit: windows 5, 8, 16 bit-identical to decode at 4.4k context).
  m.verify_sub = std::max<int>(1, static_cast<int>(64 / (cfg.num_attention_heads / cfg.num_key_value_heads)));
  if (const char* e = std::getenv("R4DX_Q35_VERIFY_SUB"); e != nullptr && e[0] != 0) m.verify_sub = std::atoi(e);
  m.positions_host = core::PinnedBuffer<int32_t>(static_cast<size_t>(R));
  m.seqused_host = core::PinnedBuffer<int32_t>(1);
  m.slices_host = core::PinnedBuffer<int32_t>(static_cast<size_t>(std::max<int64_t>(1, R / 64)));
  // Prewarm GDN control cache so it never calls hipMalloc mid-run
  m.control.Prewarm(std::min<int64_t>(m.opt.max_rows, 256), 0, 1);

  // The arena: the largest block's demand, plus room for the attention layer's activations and split-KV scratch (up to
  // 96 MiB) and the ApplyLinear scratch of its chunks (trellis: the transformed A and the split-group partials).
  constexpr size_t kLinearScratch = size_t{384} << 20;
  m.arena.Reserve(block_arena + kLinearScratch);
  m.expert_arena.Reserve(size_t{96} << 20);
  if (const char* e = std::getenv("R4DX_Q35_PROFILE"); e != nullptr && e[0] == '1' && options.spec_window > 1) {
    m.prof_on = true;
    m.prof_ev.resize(3 * m.layers.size() + 1);
    for (hipEvent_t& ev : m.prof_ev) R4DX_HIP_CHECK(hipEventCreate(&ev));
  }
  if (const char* e = std::getenv("R4DX_Q35_PROFILE"); e != nullptr && e[0] == '2' && options.spec_window > 1) {
    m.prof_on = true;
    m.prof_phase = true;
    m.prof_ev.resize(3 * m.layers.size() + 1);
    for (hipEvent_t& ev : m.prof_ev) R4DX_HIP_CHECK(hipEventCreate(&ev));
    for (hipEvent_t& ev : m.prof_phase_ev) R4DX_HIP_CHECK(hipEventCreate(&ev));
  }
  R4DX_HIP_CHECK(hipStreamSynchronize(nullptr));  // the null-stream memsets / uploads above
  m.stream.Synchronize();
}

Qwen35MoeStage::~Qwen35MoeStage() {
  if (!impl_) return;
  Qwen35MoeDeviceScope dev(impl_->device);
  static_cast<void>(hipStreamSynchronize(impl_->stream.get()));
  if (impl_->prof_on && impl_->prof_calls > 0) {
    const double n = static_cast<double>(impl_->prof_calls);
    std::fprintf(stderr,
                 "[Qwen35MoeStage dev %d, layers %lld-%lld] verify profile over %lld calls (avg %.1f rows): GDN %.2f ms, attention %.2f ms, MoE %.2f ms per call; host enqueue %.1f ms of %.1f ms wall per call\n",
                 impl_->device, static_cast<long long>(impl_->first), static_cast<long long>(impl_->end - 1),
                 static_cast<long long>(impl_->prof_calls), static_cast<double>(impl_->prof_rows) / n, impl_->prof_ms[0] / n,
                 impl_->prof_ms[1] / n, impl_->prof_ms[2] / n, impl_->prof_enqueue_ms / n, impl_->prof_wall_ms / n);
  }
  if (impl_->prof_phase && impl_->prof_phase_calls > 0) {
    const double n = static_cast<double>(impl_->prof_phase_calls);
    std::fprintf(stderr, "[Qwen35MoeStage dev %d] verify GDN layer phases over %lld layer-calls: projections %.3f ms, conv + scan %.3f ms, out_proj %.3f ms per layer\n",
                 impl_->device, static_cast<long long>(impl_->prof_phase_calls), impl_->prof_phase_ms[0] / n, impl_->prof_phase_ms[1] / n,
                 impl_->prof_phase_ms[2] / n);
    for (hipEvent_t ev : impl_->prof_phase_ev) static_cast<void>(hipEventDestroy(ev));
  }
  for (hipEvent_t ev : impl_->prof_ev) static_cast<void>(hipEventDestroy(ev));
  if (impl_->step_prof) EnableStepProfile(false);
  impl_.reset();  // frees on the stage's device
}

int Qwen35MoeStage::Device() const { return impl_->device; }
hipStream_t Qwen35MoeStage::Stream() const { return impl_->stream.get(); }
int64_t Qwen35MoeStage::FirstLayer() const { return impl_->first; }
int64_t Qwen35MoeStage::EndLayer() const { return impl_->end; }
bool Qwen35MoeStage::HasHead() const { return impl_->head_refs.has_value(); }
double Qwen35MoeStage::LastRunMs() const { return impl_->last_run_ms; }
double Qwen35MoeStage::LastHeadMs() const { return impl_->last_head_ms; }

void Qwen35MoeStage::SetGroupedDecode(bool on, bool with_capture) {
  Impl& m = *impl_;
  m.grouped_on = on && m.gmoe != nullptr;
  m.grouped_capture = with_capture;
}

int64_t Qwen35MoeStage::GroupedDecodeLayers() const {
  const Impl& m = *impl_;
  if (!m.grouped_on) return 0;
  return std::count_if(m.layers.begin(), m.layers.end(), [](const auto& lp) { return lp->grouped; });
}

void Qwen35MoeStage::SetCapture(Qwen35MoeCaptureFn fn) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  m.capture = std::move(fn);
  if (m.capture && m.head_refs && m.logits_all.size() == 0) {
    if (m.comm == nullptr) m.logits_all = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.opt.max_rows * m.vocab));
    m.norm_all = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.opt.max_rows * m.hidden));
  }
}

// ---- chunk input ----------------------------------------------------------------------------------------------------

void Qwen35MoeStage::LoadEmbedding(const uint16_t* host_rows, int64_t rows) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (rows < 1 || rows > m.opt.max_rows) Fail("LoadEmbedding: rows out of range");
  R4DX_HIP_CHECK(hipMemcpyAsync(m.h.data(), host_rows, static_cast<size_t>(rows * m.hidden) * 2, hipMemcpyHostToDevice,
                                m.stream.get()));
}

void Qwen35MoeStage::LoadResidual(const uint16_t* host_rows, int64_t rows) { LoadEmbedding(host_rows, rows); }

void Qwen35MoeStage::StoreResidual(uint16_t* host_rows, int64_t rows) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (rows < 1 || rows > m.opt.max_rows) Fail("StoreResidual: rows out of range");
  R4DX_HIP_CHECK(hipMemcpyAsync(host_rows, m.h.data(), static_cast<size_t>(rows * m.hidden) * 2, hipMemcpyDeviceToHost,
                                m.stream.get()));
  m.stream.Synchronize();
}

// ---- the layers -----------------------------------------------------------------------------------------------------

void Qwen35MoeStage::Impl::RunGdn(LayerRt& L, const Qwen35MoeChunk& c) {
  // x = rmsnorm(h); blk = gdn(x); hmid = h + blk
  Norm(h.data(), L.input_ln, x.data(), c.rows);
  CapBf16(L.index, "attn_in", x.data(), c.rows, hidden, c.first_pos);
  Qwen4ExpGdnCall call;
  call.is_prefill = c.is_prefill;
  call.has_init = c.first_pos > 0;
  call.prefill_recurrent = c.gdn_prefill_recurrent;
  if (spec_window > 1 && !c.is_prefill) {
    // The 27B's protocol (Model::RunChunk / VerifyWindow): the conv history offset is the last window's committed count, and a
    // committed prefix of the last log still pending is replayed in this call's seed load.
    call.verify = c.verify;
    call.num_accepted = num_accepted_valid ? num_accepted_dev.data() : nullptr;
    call.gdn_pending = book.Pending() ? num_accepted_dev.data() : nullptr;
  }
  // R4DX_Q35_PROFILE=2: split the verify GDN layers into projections / conv + scan / out_proj (a sync per layer: diagnostics only)
  std::vector<std::pair<const char*, hipEvent_t>> marks;
  std::function<void(const char*)> mark;
  if (prof_on && prof_phase && c.verify) {
    hipEvent_t e0 = prof_phase_ev[0];
    R4DX_HIP_CHECK(hipEventRecord(e0, stream.get()));
    marks.emplace_back("start", e0);
    mark = [&](const char* name) {
      hipEvent_t e = prof_phase_ev[marks.size()];
      static_cast<void>(hipEventRecord(e, stream.get()));
      marks.emplace_back(name, e);
    };
    call.mark = &mark;
  }
  L.gdn->Forward(stream.get(), arena, *L.gdn_states, control, x.data(), blk.data(), c.rows, call);
  if (!marks.empty()) {
    R4DX_HIP_CHECK(hipEventSynchronize(marks.back().second));
    for (size_t i = 1; i < marks.size(); ++i) {
      float ms = 0;
      R4DX_HIP_CHECK(hipEventElapsedTime(&ms, marks[i - 1].second, marks[i].second));
      prof_phase_ms[i - 1] += ms;
    }
    ++prof_phase_calls;
  }
  arena.Reset();
  AllReduceBlock(c.rows);  // --tp 2 site A1: out_proj is row-parallel, blk holds this rank's partial sum
  CapBf16(L.index, "attn_out", blk.data(), c.rows, hidden, c.first_pos);
  ResidualAdd(h.data(), blk.data(), hmid.data(), c.rows);
}

void Qwen35MoeStage::Impl::RunAttention(LayerRt& L, const Qwen35MoeChunk& c) {
  // AttentionLayer: hmid = h + o_proj(attention(rmsnorm(h))) -- the input norm and the residual add are its own.
  const int64_t T = c.rows;
  const hipStream_t s = stream.get();
  const bool sliced = T > 64;
  if (capture) {  // capture only: the layer re-derives this norm itself
    Norm(h.data(), L.input_ln, x.data(), T);
    CapBf16(L.index, "attn_in", x.data(), T, hidden, c.first_pos);
  }
  for (int64_t t = 0; t < T; ++t) positions_host[static_cast<size_t>(t)] = static_cast<int32_t>(c.first_pos + t);
  seqused_host[0] = static_cast<int32_t>(c.first_pos + T);
  R4DX_HIP_CHECK(hipMemcpyAsync(positions.data(), positions_host.data(), static_cast<size_t>(T) * 4, hipMemcpyHostToDevice, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(seqused_k.data(), seqused_host.data(), 4, hipMemcpyHostToDevice, s));
  if (sliced) {
    for (int64_t j = 0; j < T / 64; ++j) slices_host[static_cast<size_t>(j)] = static_cast<int32_t>(c.first_pos + 64 * (j + 1));
    R4DX_HIP_CHECK(hipMemcpyAsync(seqused_k_slices.data(), slices_host.data(), static_cast<size_t>(T / 64) * 4,
                                  hipMemcpyHostToDevice, s));
  }
  // A verify window runs the DECODE attention kernel `verify_sub` rows at a time (see AttentionLayer::Forward), so each row's result is
  // the plain decode step's bit for bit; the prefill kernel it would otherwise take for more than 4 rows rounds differently.
  int decode_sub = 0;
  if (c.verify && T > 1 && verify_sub > 0) {
    decode_sub = verify_sub;
    for (int64_t j = 0; j * decode_sub < T; ++j) {
      sub_host[static_cast<size_t>(j)] = static_cast<int32_t>(c.first_pos + std::min<int64_t>((j + 1) * decode_sub, T));
    }
    R4DX_HIP_CHECK(hipMemcpyAsync(seqused_sub.data(), sub_host.data(), static_cast<size_t>((T + decode_sub - 1) / decode_sub) * 4,
                                  hipMemcpyHostToDevice, s));
  }
  L.attn->Forward(arena, h.data(), hmid.data(), L.attn_w, *L.kv, static_cast<int>(T), static_cast<int>(c.first_pos),
                  positions.data(), seqused_k.data(), s, /*x_normed_in=*/nullptr, /*next_norm_weight=*/nullptr,
                  /*x_normed_out=*/nullptr, /*prof=*/nullptr, /*x_normed_pre_epilogue=*/0, /*x_normed_pre_data=*/nullptr,
                  /*next_epilogue=*/0, /*next_epilogue_out=*/nullptr, /*rope_pos3=*/nullptr,
                  /*prefill_split_kv=*/c.is_prefill && T > 1, /*attn_slice=*/sliced ? 64 : 0,
                  sliced ? seqused_k_slices.data() : nullptr, /*batch=*/nullptr, decode_sub, seqused_sub.data());
  arena.Reset();
  // The layer returns h + o_proj(..); the o_proj output alone is hmid - h (not captured separately).
}

void Qwen35MoeStage::Impl::RunMoe(LayerRt& L, const Qwen35MoeChunk& c) {
  const hipStream_t s = stream.get();
  const Qwen35MoeMoeRefs& e = *L.moe_refs;
  Norm(hmid.data(), L.post_ln, x.data(), c.rows);
  CapBf16(L.index, "mlp_in", x.data(), c.rows, hidden, c.first_pos);
  if (c.rows == 1 && grouped_on && L.grouped && (!capture || grouped_capture)) {
    gmoe->Run(s, arena, L.gmoe_w, e.trellis, x.data(), blk.data());
    arena.Reset();
    AllReduceBlock(1);
    CapBf16(L.index, "moe_out", blk.data(), 1, hidden, c.first_pos);
    ResidualAdd(hmid.data(), blk.data(), h.data(), 1);
    return;
  }
  // A speculative verify window (2..cap rows): all rows through the M-row grouped kernels, no host sync, each routed expert's weights
  // streamed once. (Prefill chunks, even short ones, keep the host-routed path: their numerics are the validated baseline.)
  if (c.verify && c.rows >= 2 && c.rows <= gmoe_rows_cap && grouped_on && L.grouped && (!capture || grouped_capture) && rows_moe_on) {
    gmoe->RunRows(s, arena, L.gmoe_w, e.trellis, x.data(), blk.data(), c.rows);
    arena.Reset();
    AllReduceBlock(c.rows);
    CapBf16(L.index, "moe_out", blk.data(), c.rows, hidden, c.first_pos);
    ResidualAdd(hmid.data(), blk.data(), h.data(), c.rows);
    return;
  }
  // One expert GEMM's ApplyLinear scratch comes from its own small arena, reset before each call: the experts' calls (up to
  // 2 per expert, hundreds per prefill chunk) would otherwise pile their scratch up in the block arena. Stream order makes
  // the reuse safe (every GEMM of the layer runs on `stream`, one after another).
  const qwen4exp::ExpertLinearFn fn = [&](int expert, qwen4exp::ExpertLinear which, const uint16_t* xin, int64_t rows,
                                          uint16_t* y, core::Arena& /*block_arena*/) {
    expert_arena.Reset();
    const QuantLinear* w = which == qwen4exp::ExpertLinear::kGateUp ? e.expert_gate_up[static_cast<size_t>(expert)]
                                                                     : e.expert_down[static_cast<size_t>(expert)];
    ApplyLinear(s, expert_arena, *w, xin, y, rows);
  };
  moe->Forward(s, arena, L.moe_w, x.data(), blk.data(), c.rows, fn);
  arena.Reset();
  AllReduceBlock(c.rows);  // --tp 2 site M: routed + gated shared partials of this rank
  if (capture) {
    Cap(L.index, "topk_ids", moe->last_ids(), Qwen35MoeCapture::Dtype::kI32, c.rows, cfg.num_experts_per_tok, c.first_pos,
        /*on_host=*/true);
  }
  CapBf16(L.index, "moe_out", blk.data(), c.rows, hidden, c.first_pos);
  ResidualAdd(hmid.data(), blk.data(), h.data(), c.rows);
}

void Qwen35MoeStage::RunLayers(const Qwen35MoeChunk& c) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  const auto t0 = Clock::now();
  m.last_first_pos = c.first_pos;
  if (c.rows < 1 || c.rows > m.opt.max_rows || (c.rows > 64 && c.rows % 64 != 0)) {
    Fail("RunLayers: rows " + std::to_string(c.rows) + " out of range (1..64, or a multiple of 64 up to " +
         std::to_string(m.opt.max_rows) + ")");
  }
  if (c.first_pos < 0 || c.first_pos + c.rows > m.opt.max_ctx) {
    Fail("RunLayers: positions [" + std::to_string(c.first_pos) + ", " + std::to_string(c.first_pos + c.rows) +
         ") exceed max_ctx " + std::to_string(m.opt.max_ctx));
  }
  const int64_t T = c.rows;
  if (c.verify && (c.is_prefill || T > m.spec_window || m.spec_window <= 1)) {
    Fail("RunLayers: a verify chunk needs spec_window > 1, !is_prefill and rows <= spec_window");
  }
  if (m.spec_window > 1 && !c.is_prefill && !c.verify && T != 1) Fail("RunLayers: a speculative stage decodes one row (or verifies)");
  const bool is_decode = (T == 1 && !c.is_prefill && !m.capture);
  // Bounded prefill submission under TP, cut the same way on both ranks (lockstep): a 256-row chunk is four 64-row chunks' worth of work per layer.
  const bool bounded = m.comm != nullptr && c.is_prefill && m.submit.Active();
  int64_t unit_layers = bounded ? tp::UnitLayersForContext(m.submit_layers, c.first_pos + T) : 0;
  if (bounded && T > 64) unit_layers = std::max<int64_t>(1, unit_layers * 64 / T);
  if (bounded) m.submit.Reset();
  m.arena.Reset();
  if (c.is_prefill) m.CollapseWindow();
  m.SpMark(0);

  for (auto& lp : m.layers) {
    LayerRt& L = *lp;
    if (c.only_layer >= 0 && L.index != c.only_layer) continue;
    if (bounded && L.index > m.first && (L.index - m.first) % unit_layers == 0) m.submit.EndUnit(m.stream.get());
    for (size_t ti = 0; ti < m.taps.size(); ++ti) {
      const Qwen35MoeFeatureTap& tap = m.taps[ti];
      if (tap.layer != L.index || m.tap_dst == nullptr) continue;
      if (m.tap_contiguous) {
        const size_t width = static_cast<size_t>(m.taps.size() * m.hidden) * 2;
        R4DX_HIP_CHECK(hipMemcpy2DAsync(m.tap_dev.data() + static_cast<int64_t>(ti) * m.hidden, width, m.h.data(),
                                        static_cast<size_t>(m.hidden) * 2, static_cast<size_t>(m.hidden) * 2, static_cast<size_t>(T),
                                        hipMemcpyDeviceToDevice, m.stream.get()));
      } else {
        R4DX_HIP_CHECK(hipMemcpy2DAsync(m.tap_dst + tap.column * m.hidden, static_cast<size_t>(m.tap_pitch) * 2, m.h.data(),
                                        static_cast<size_t>(m.hidden) * 2, static_cast<size_t>(m.hidden) * 2,
                                        static_cast<size_t>(T), hipMemcpyDeviceToHost, m.stream.get()));
      }
    }
    m.CapBf16(L.index, "residual_in", m.h.data(), T, m.hidden, c.first_pos);
    const bool prof = m.prof_on && c.verify;
    const size_t li = static_cast<size_t>(L.index - m.first);
    if (prof) R4DX_HIP_CHECK(hipEventRecord(m.prof_ev[3 * li], m.stream.get()));
    if (L.full) {
      m.RunAttention(L, c);
    } else {
      m.RunGdn(L, c);
    }
    if (prof) R4DX_HIP_CHECK(hipEventRecord(m.prof_ev[3 * li + 1], m.stream.get()));
    m.CapBf16(L.index, "residual_attn", m.hmid.data(), T, m.hidden, c.first_pos);
    m.RunMoe(L, c);
    if (prof) R4DX_HIP_CHECK(hipEventRecord(m.prof_ev[3 * li + 2], m.stream.get()));
    m.CapBf16(L.index, "residual_out", m.h.data(), T, m.hidden, c.first_pos);
  }
  if (m.spec_window > 1 && !c.is_prefill && c.only_layer < 0) {
    if (c.verify) {
      m.book.OnVerifyLaunched(T);
    } else {
      // A plain decode step "commits" its one row: the next call's conv history sits at offset 0, nothing is pending (this
      // step's kernel replayed any pending prefix and stored the final state into B).
      m.WriteNumAccepted(1, /*plain_step=*/true);
      m.num_accepted_valid = true;
      m.last_committed = 1;
      m.book.OnPlainDecode();
    }
  }
  if (m.tap_contiguous && m.tap_dst != nullptr && !m.taps.empty()) {
    const size_t width = static_cast<size_t>(m.taps.size() * m.hidden) * 2;
    R4DX_HIP_CHECK(hipMemcpy2DAsync(m.tap_dst + m.taps.front().column * m.hidden, static_cast<size_t>(m.tap_pitch) * 2, m.tap_dev.data(),
                                    width, width, static_cast<size_t>(T), hipMemcpyDeviceToHost, m.stream.get()));
  }
  m.SpMark(1);
  const double enqueue_ms = MsSince(t0);
  if (!is_decode) m.stream.Synchronize();
  if (m.prof_on && c.verify) {
    m.prof_enqueue_ms += enqueue_ms;
    m.prof_wall_ms += MsSince(t0);
    for (auto& lp : m.layers) {
      const size_t li = static_cast<size_t>(lp->index - m.first);
      float a = 0, b = 0;
      R4DX_HIP_CHECK(hipEventElapsedTime(&a, m.prof_ev[3 * li], m.prof_ev[3 * li + 1]));
      R4DX_HIP_CHECK(hipEventElapsedTime(&b, m.prof_ev[3 * li + 1], m.prof_ev[3 * li + 2]));
      m.prof_ms[lp->full ? 1 : 0] += a;
      m.prof_ms[2] += b;
    }
    ++m.prof_calls;
    m.prof_rows += T;
  }
  m.last_run_ms = MsSince(t0);
}

// ---- head -----------------------------------------------------------------------------------------------------------

Qwen35MoeHeadResult Qwen35MoeStage::Head(int64_t rows, const Qwen35MoeHeadRequest& req) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (!m.head_refs) Fail("Head: this stage has no head");
  if (rows < 1 || rows > m.opt.max_rows) Fail("Head: rows out of range");
  const auto t0 = Clock::now();
  const hipStream_t s = m.stream.get();
  const QuantLinear& lm = *m.head_refs->lm_head;
  const uint16_t* final_norm = m.head_refs->final_norm;
  m.arena.Reset();
  m.SpMark(2);

  if (m.capture) {
    // every row of the chunk (tests): the final norm, then the lm_head over all of them
    m.Norm(m.h.data(), final_norm, m.norm_all.data(), rows);
    m.Cap(-1, "norm_out", m.norm_all.data(), Qwen35MoeCapture::Dtype::kBf16, rows, m.hidden, m.last_first_pos);
    if (m.comm == nullptr) {  // (a rank holds a vocab slice of the logits: the all-rows logits capture is a TP=1 test hook)
      ApplyLinear(s, m.arena, lm, m.norm_all.data(), m.logits_all.data(), rows);
      m.arena.Reset();
      m.Cap(-1, "logits", m.logits_all.data(), Qwen35MoeCapture::Dtype::kBf16, rows, m.vocab, m.last_first_pos);
    }
  }

  // the last row only: the norm and the lm_head are row-independent
  const uint16_t* h_last = m.h.data() + (rows - 1) * m.hidden;
  m.Norm(h_last, final_norm, m.x.data(), 1);
  uint16_t* logits_bf16 = m.arena.Alloc<uint16_t>(static_cast<size_t>(m.vocab_local));  // the rank's vocab slice under TP
  ApplyLinear(s, m.arena, lm, m.x.data(), logits_bf16, 1);

  Qwen35MoeHeadResult out;
  switch (req.kind) {
    case Qwen35MoeHeadRequest::Kind::kLogits: {
      r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_dev.data()),
                                   m.vocab_local, reinterpret_cast<int64_t>(s));
      m.SpMark(3);
      m.stream.Synchronize();
      if (m.comm != nullptr) {
        out.logits = m.GatherVocabRow();
        break;
      }
      out.logits.resize(static_cast<size_t>(m.vocab));
      m.logits_dev.CopyToHost(out.logits.data(), static_cast<size_t>(m.vocab));
      break;
    }
    case Qwen35MoeHeadRequest::Kind::kArgmax: {
      if (m.comm != nullptr) {
        // the rank's argmax over its slice (lowest-index tie-break), merged over the ranks in rank order (tp_vocab.h: exact)
        r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_dev.data()),
                                     m.vocab_local, reinterpret_cast<int64_t>(s));
        r4dx_argmax_val_f32(reinterpret_cast<int64_t>(m.logits_dev.data()), reinterpret_cast<int64_t>(m.argmax_pair_dev.data()),
                            reinterpret_cast<int64_t>(m.argmax_pair_dev.data() + 1), m.vocab_local, reinterpret_cast<int64_t>(s));
        m.SpMark(3);
        m.stream.Synchronize();
        tp::ArgmaxPair mine{};
        static_assert(sizeof(tp::ArgmaxPair) == 2 * sizeof(int32_t), "{idx, value} pair is two 32-bit words");
        R4DX_HIP_CHECK(hipMemcpy(&mine, m.argmax_pair_dev.data(), sizeof(mine), hipMemcpyDeviceToHost));
        mine.idx += static_cast<int32_t>(m.vocab_offset);
        std::vector<tp::ArgmaxPair> all(static_cast<size_t>(m.comm->World()));
        m.comm->HostAllGather(&mine, sizeof(mine), all.data());
        out.token = tp::MergeArgmax(all.data(), m.comm->World());
        break;
      }
      r4dx_argmax_bf16(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.argmax_dev.data()), m.vocab,
                       reinterpret_cast<int64_t>(s));
      m.SpMark(3);
      R4DX_HIP_CHECK(hipMemcpyAsync(m.greedy_host.data(), m.argmax_dev.data(), sizeof(int32_t), hipMemcpyDeviceToHost, s));
      m.stream.Synchronize();
      out.token = m.greedy_host[0];
      break;
    }
    case Qwen35MoeHeadRequest::Kind::kSummary: {
      if (!(req.inv_temperature > 0.0f)) Fail("Head: kSummary needs inv_temperature > 0");
      r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_dev.data()),
                                   m.vocab_local, reinterpret_cast<int64_t>(s));
      r4dx_topk_lse_f32_ws(reinterpret_cast<int64_t>(m.logits_dev.data()), reinterpret_cast<int64_t>(m.summary_ids_dev.data()),
                           reinterpret_cast<int64_t>(m.summary_vals_dev.data()),
                           reinterpret_cast<int64_t>(m.summary_lse_dev.data()), 1, m.vocab_local, req.inv_temperature,
                           reinterpret_cast<int64_t>(s), reinterpret_cast<int64_t>(m.topk_ws.data()));
      m.SpMark(3);
      m.stream.Synchronize();
      std::vector<int32_t> ids(R4DX_TOPK_LSE_K);
      std::vector<float> vals(R4DX_TOPK_LSE_K);
      float lse = 0.0f;
      m.summary_ids_dev.CopyToHost(ids.data(), ids.size());
      m.summary_vals_dev.CopyToHost(vals.data(), vals.size());
      m.summary_lse_dev.CopyToHost(&lse, 1);
      kernels::RowSummary& sm = out.summary;
      sm.k = R4DX_TOPK_LSE_K;
      sm.vocab = m.vocab_local;
      sm.inv_temperature = req.inv_temperature;
      sm.lse = lse;
      for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
        sm.ids[j] = ids[static_cast<size_t>(j)] + static_cast<int32_t>(m.vocab_offset);
        sm.vals[j] = vals[static_cast<size_t>(j)];
      }
      if (m.comm != nullptr) {
        static_assert(std::is_trivially_copyable_v<kernels::RowSummary>, "row summaries cross the host exchange as raw bytes");
        std::vector<kernels::RowSummary> all(static_cast<size_t>(m.comm->World()));
        m.comm->HostAllGather(&sm, sizeof(kernels::RowSummary), all.data());
        sm = tp::MergeRowSummaries(all.data(), m.comm->World(), m.vocab);
      }
      break;
    }
  }
  m.arena.Reset();
  m.last_head_ms = MsSince(t0);
  return out;
}

std::vector<float> Qwen35MoeStage::LastLogitsRow() {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (!m.head_refs) Fail("LastLogitsRow: this stage has no head");
  if (m.comm != nullptr) return m.GatherVocabRow();
  std::vector<float> row(static_cast<size_t>(m.vocab));
  m.logits_dev.CopyToHost(row.data(), row.size());
  return row;
}

// ---- state ----------------------------------------------------------------------------------------------------------

void Qwen35MoeStage::Reset() {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  for (auto& lp : m.layers) {
    if (lp->gdn_states) lp->gdn_states->ZeroAll(m.stream);
  }
  m.stream.Synchronize();
  m.arena.Reset();
  m.expert_arena.Reset();
  m.ResetSpec();
  m.submit.Reset();
}

void Qwen35MoeStage::SaveCheckpoint() {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (!m.opt.prompt_checkpoint) Fail("SaveCheckpoint: the stage was built without prompt_checkpoint");
  for (auto& lp : m.layers) {
    if (lp->gdn_states) lp->gdn_states->SaveCheckpoint(0, m.stream);
  }
  m.stream.Synchronize();
}

void Qwen35MoeStage::RestoreCheckpoint(int64_t /*pos*/) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (!m.opt.prompt_checkpoint) Fail("RestoreCheckpoint: the stage was built without prompt_checkpoint");
  for (auto& lp : m.layers) {
    if (lp->gdn_states) lp->gdn_states->RestoreCheckpoint(0, m.stream);
  }
  m.stream.Synchronize();
  m.arena.Reset();
  m.expert_arena.Reset();
  m.ResetSpec();  // the checkpoint was taken right after a prefill, with nothing pending
}

// ---- speculative window ----------------------------------------------------------------------------------------------

void Qwen35MoeStage::EnableStepProfile(bool on) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (on && !m.step_prof) {
    for (hipEvent_t& ev : m.sp_ev) R4DX_HIP_CHECK(hipEventCreate(&ev));
  } else if (!on && m.step_prof) {
    for (hipEvent_t& ev : m.sp_ev) {
      static_cast<void>(hipEventDestroy(ev));
      ev = nullptr;
    }
  }
  m.step_prof = on;
}

tp::SubmitBounder::Stats Qwen35MoeStage::SubmitStats() const { return impl_->submit.GetStats(); }

double Qwen35MoeStage::LastLayersGpuMs() const {
  Impl& m = *impl_;
  if (!m.step_prof) Fail("LastLayersGpuMs: EnableStepProfile(true) was not called");
  Qwen35MoeDeviceScope dev(m.device);
  R4DX_HIP_CHECK(hipEventSynchronize(m.sp_ev[1]));
  float ms = 0;
  R4DX_HIP_CHECK(hipEventElapsedTime(&ms, m.sp_ev[0], m.sp_ev[1]));
  return ms;
}

double Qwen35MoeStage::LastHeadGpuMs() const {
  Impl& m = *impl_;
  if (!m.step_prof) Fail("LastHeadGpuMs: EnableStepProfile(true) was not called");
  Qwen35MoeDeviceScope dev(m.device);
  R4DX_HIP_CHECK(hipEventSynchronize(m.sp_ev[3]));
  float ms = 0;
  R4DX_HIP_CHECK(hipEventElapsedTime(&ms, m.sp_ev[2], m.sp_ev[3]));
  return ms;
}

int64_t Qwen35MoeStage::SpecWindow() const { return impl_->spec_window; }

void Qwen35MoeStage::CommitVerified(int64_t n) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (m.spec_window <= 1) Fail("CommitVerified: the stage was built without a speculative window");
  if (n < 1 || n > m.book.LastVerifyT()) {
    Fail("CommitVerified: n = " + std::to_string(n) + " is outside [1, " + std::to_string(m.book.LastVerifyT()) +
         "] (the rows of the last verify call)");
  }
  m.WriteNumAccepted(static_cast<int32_t>(n));
  m.num_accepted_valid = true;
  m.last_committed = n;
  static_cast<void>(m.book.OnCommit(n));  // pending iff the call logged (T > 1)
}

std::vector<int32_t> Qwen35MoeStage::HeadRows(int64_t rows) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (!m.head_refs) Fail("HeadRows: this stage has no head");
  if (m.spec_window <= 1 || rows < 1 || rows > m.spec_window) Fail("HeadRows: rows out of range / no speculative window");
  const auto t0 = Clock::now();
  const hipStream_t s = m.stream.get();
  m.arena.Reset();
  m.Norm(m.h.data(), m.head_refs->final_norm, m.x.data(), rows);
  uint16_t* logits_bf16 = m.arena.Alloc<uint16_t>(static_cast<size_t>(rows * m.vocab_local));
  ApplyLinear(s, m.arena, *m.head_refs->lm_head, m.x.data(), logits_bf16, rows);
  // The widen is exact, so the comparisons of the f32 row argmax are the bf16 ones (r4dx_argmax_bf16's, the plain decode step's).
  r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_rows.data()),
                               rows * m.vocab_local, reinterpret_cast<int64_t>(s));
  if (m.comm != nullptr) {
    // --tp 2: the argmax of each row over this rank's vocab slice with its value (pair t = {idx, val} at 2t, 2t + 1), merged over the ranks on the
    // host in rank order (tp_vocab.h MergeArgmaxRows: exact, ties to the lowest global id) -- every rank returns the same tokens.
    r4dx_argmax_rows_f32(reinterpret_cast<int64_t>(m.logits_rows.data()), m.vocab_local, reinterpret_cast<int64_t>(m.argmax_rows_dev.data()),
                         reinterpret_cast<int64_t>(m.argmax_rows_dev.data() + 1), /*out_stride=*/2, rows, m.vocab_local,
                         reinterpret_cast<int64_t>(s));
    R4DX_HIP_CHECK(hipMemcpyAsync(m.argmax_rows_host.data(), m.argmax_rows_dev.data(), static_cast<size_t>(rows) * sizeof(tp::ArgmaxPair),
                                  hipMemcpyDeviceToHost, s));
    m.stream.Synchronize();
    std::vector<tp::ArgmaxPair> mine(static_cast<size_t>(rows));
    std::memcpy(mine.data(), m.argmax_rows_host.data(), static_cast<size_t>(rows) * sizeof(tp::ArgmaxPair));
    for (tp::ArgmaxPair& p : mine) p.idx += static_cast<int32_t>(m.vocab_offset);
    const int world = m.comm->World();
    std::vector<tp::ArgmaxPair> all(static_cast<size_t>(world) * static_cast<size_t>(rows));
    m.comm->HostAllGather(mine.data(), mine.size() * sizeof(tp::ArgmaxPair), all.data());
    std::vector<int32_t> merged(static_cast<size_t>(rows));
    tp::MergeArgmaxRows(all.data(), world, static_cast<int>(rows), merged.data());
    m.arena.Reset();
    m.last_head_ms = MsSince(t0);
    return merged;
  }
  r4dx_argmax_rows_f32(reinterpret_cast<int64_t>(m.logits_rows.data()), m.vocab,
                       reinterpret_cast<int64_t>(m.argmax_rows_dev.data()), /*out_val=*/0, /*out_stride=*/1, rows, m.vocab,
                       reinterpret_cast<int64_t>(s));
  R4DX_HIP_CHECK(hipMemcpyAsync(m.argmax_rows_host.data(), m.argmax_rows_dev.data(), static_cast<size_t>(rows) * sizeof(int32_t),
                                hipMemcpyDeviceToHost, s));
  m.stream.Synchronize();
  m.arena.Reset();
  std::vector<int32_t> out(m.argmax_rows_host.data(), m.argmax_rows_host.data() + rows);
  m.last_head_ms = MsSince(t0);
  return out;
}

std::vector<float> Qwen35MoeStage::LastHeadRowsLogits(int64_t rows) {
  Impl& m = *impl_;
  if (m.comm != nullptr) {
    // --tp 2 (a collective: every rank calls it): the ranks' vocab slices of the last HeadRows call gathered into full rows [rows, vocab]
    if (m.logits_rows.size() == 0 || rows < 1 || rows > m.spec_window) Fail("LastHeadRowsLogits: no HeadRows result / rows out of range");
    std::vector<float> mine(static_cast<size_t>(rows * m.vocab_local));
    m.logits_rows.CopyToHost(mine.data(), mine.size());
    const int world = m.comm->World();
    std::vector<float> all(mine.size() * static_cast<size_t>(world));
    m.comm->HostAllGather(mine.data(), mine.size() * sizeof(float), all.data());
    std::vector<float> full(static_cast<size_t>(rows * m.vocab));
    for (int r = 0; r < world; ++r) {
      for (int64_t t = 0; t < rows; ++t) {
        std::memcpy(full.data() + static_cast<size_t>(t * m.vocab + r * m.vocab_local), all.data() + static_cast<size_t>(r) * mine.size() + static_cast<size_t>(t * m.vocab_local),
                    static_cast<size_t>(m.vocab_local) * sizeof(float));
      }
    }
    return full;
  }
  Qwen35MoeDeviceScope dev(m.device);
  if (m.logits_rows.size() == 0 || rows < 1 || rows > m.spec_window) Fail("LastHeadRowsLogits: no HeadRows result / rows out of range");
  std::vector<float> out(static_cast<size_t>(rows * m.vocab));
  m.logits_rows.CopyToHost(out.data(), out.size());
  return out;
}

void Qwen35MoeStage::SetFeatureCapture(std::vector<Qwen35MoeFeatureTap> taps, int64_t row_pitch) {
  Impl& m = *impl_;
  for (const Qwen35MoeFeatureTap& t : taps) {
    if (t.column < 0 || (t.column + 1) * m.hidden > row_pitch) Fail("SetFeatureCapture: a tap column is outside the row pitch");
  }
  m.taps = std::move(taps);
  m.tap_pitch = row_pitch;
  m.tap_contiguous = !m.taps.empty();
  for (size_t i = 0; i < m.taps.size(); ++i) m.tap_contiguous = m.tap_contiguous && m.taps[i].column == m.taps[0].column + static_cast<int64_t>(i);
  if (m.tap_contiguous) {
    Qwen35MoeDeviceScope dev(m.device);
    m.tap_dev = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.opt.max_rows * static_cast<int64_t>(m.taps.size()) * m.hidden));
  }
}

void Qwen35MoeStage::SetFeatureDest(uint16_t* host_dst) { impl_->tap_dst = host_dst; }

}  // namespace r4dx::model




