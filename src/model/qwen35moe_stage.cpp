#include "qwen35moe_stage.h"

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
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_block.h"

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
  const Qwen35MoeMoeRefs* moe_refs = nullptr;  // points into Impl::layer_refs
};

}  // namespace

struct Qwen35MoeStage::Impl {
  Qwen35MoeStageOptions opt;
  Qwen35MoeConfig cfg;
  int device = 0;
  int64_t first = 0, end = 0;
  int64_t hidden = 0, vocab = 0;

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
  core::PinnedBuffer<int32_t> positions_host, seqused_host, slices_host;

  GdnControlCache control;
  std::unique_ptr<qwen4exp::MoeBlockBase> moe;
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

  Qwen35MoeCaptureFn capture;
  double last_run_ms = 0, last_head_ms = 0;
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
          L->gdn->ConvDim(), cfg.linear_conv_kernel_dim, /*max_decode_window=*/1, /*write_once=*/false);
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
    L->moe_refs = &e;
    m.layers.push_back(std::move(L));
  }

  // ---- head ---------------------------------------------------------------------------------------------------------
  if (m.head_refs) {
    if (m.head_refs->lm_head == nullptr || m.head_refs->final_norm == nullptr) Fail("the head has no lm_head / final_norm");
    if (m.head_refs->lm_head->N != m.vocab || m.head_refs->lm_head->K != m.hidden) {
      Fail("lm_head is [" + std::to_string(m.head_refs->lm_head->N) + ", " + std::to_string(m.head_refs->lm_head->K) +
           "], expected [" + std::to_string(m.vocab) + ", " + std::to_string(m.hidden) + "]");
    }
    m.logits_dev = core::DeviceBuffer<float>(static_cast<size_t>(m.vocab));
    m.argmax_dev = core::DeviceBuffer<int32_t>(1);
    m.greedy_host = core::PinnedBuffer<int32_t>(1);
    m.summary_ids_dev = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
    m.summary_vals_dev = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
    m.summary_lse_dev = core::DeviceBuffer<float>(1);
    m.topk_ws = core::DeviceBuffer<uint8_t>(static_cast<size_t>(r4dx_topk_lse_workspace_bytes()));
    bump(Padded(static_cast<size_t>(m.vocab) * 2));
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
  R4DX_HIP_CHECK(hipStreamSynchronize(nullptr));  // the null-stream memsets / uploads above
  m.stream.Synchronize();
}

Qwen35MoeStage::~Qwen35MoeStage() {
  if (!impl_) return;
  Qwen35MoeDeviceScope dev(impl_->device);
  static_cast<void>(hipStreamSynchronize(impl_->stream.get()));
  impl_.reset();  // frees on the stage's device
}

int Qwen35MoeStage::Device() const { return impl_->device; }
hipStream_t Qwen35MoeStage::Stream() const { return impl_->stream.get(); }
int64_t Qwen35MoeStage::FirstLayer() const { return impl_->first; }
int64_t Qwen35MoeStage::EndLayer() const { return impl_->end; }
bool Qwen35MoeStage::HasHead() const { return impl_->head_refs.has_value(); }
double Qwen35MoeStage::LastRunMs() const { return impl_->last_run_ms; }
double Qwen35MoeStage::LastHeadMs() const { return impl_->last_head_ms; }

void Qwen35MoeStage::SetCapture(Qwen35MoeCaptureFn fn) {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  m.capture = std::move(fn);
  if (m.capture && m.head_refs && m.logits_all.size() == 0) {
    m.logits_all = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.opt.max_rows * m.vocab));
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
  L.gdn->Forward(stream.get(), arena, *L.gdn_states, control, x.data(), blk.data(), c.rows, call);
  arena.Reset();
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
  L.attn->Forward(arena, h.data(), hmid.data(), L.attn_w, *L.kv, static_cast<int>(T), static_cast<int>(c.first_pos),
                  positions.data(), seqused_k.data(), s, /*x_normed_in=*/nullptr, /*next_norm_weight=*/nullptr,
                  /*x_normed_out=*/nullptr, /*prof=*/nullptr, /*x_normed_pre_epilogue=*/0, /*x_normed_pre_data=*/nullptr,
                  /*next_epilogue=*/0, /*next_epilogue_out=*/nullptr, /*rope_pos3=*/nullptr,
                  /*prefill_split_kv=*/c.is_prefill && T > 1, /*attn_slice=*/sliced ? 64 : 0,
                  sliced ? seqused_k_slices.data() : nullptr, /*batch=*/nullptr);
  arena.Reset();
  // The layer returns h + o_proj(..); the o_proj output alone is hmid - h (not captured separately).
}

void Qwen35MoeStage::Impl::RunMoe(LayerRt& L, const Qwen35MoeChunk& c) {
  const hipStream_t s = stream.get();
  const Qwen35MoeMoeRefs& e = *L.moe_refs;
  Norm(hmid.data(), L.post_ln, x.data(), c.rows);
  CapBf16(L.index, "mlp_in", x.data(), c.rows, hidden, c.first_pos);
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
  const bool is_decode = (T == 1 && !c.is_prefill && !m.capture);
  m.arena.Reset();

  for (auto& lp : m.layers) {
    LayerRt& L = *lp;
    if (c.only_layer >= 0 && L.index != c.only_layer) continue;
    m.CapBf16(L.index, "residual_in", m.h.data(), T, m.hidden, c.first_pos);
    if (L.full) {
      m.RunAttention(L, c);
    } else {
      m.RunGdn(L, c);
    }
    m.CapBf16(L.index, "residual_attn", m.hmid.data(), T, m.hidden, c.first_pos);
    m.RunMoe(L, c);
    m.CapBf16(L.index, "residual_out", m.h.data(), T, m.hidden, c.first_pos);
  }
  if (!is_decode) m.stream.Synchronize();
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

  if (m.capture) {
    // every row of the chunk (tests): the final norm, then the lm_head over all of them
    m.Norm(m.h.data(), final_norm, m.norm_all.data(), rows);
    m.Cap(-1, "norm_out", m.norm_all.data(), Qwen35MoeCapture::Dtype::kBf16, rows, m.hidden, m.last_first_pos);
    ApplyLinear(s, m.arena, lm, m.norm_all.data(), m.logits_all.data(), rows);
    m.arena.Reset();
    m.Cap(-1, "logits", m.logits_all.data(), Qwen35MoeCapture::Dtype::kBf16, rows, m.vocab, m.last_first_pos);
  }

  // the last row only: the norm and the lm_head are row-independent
  const uint16_t* h_last = m.h.data() + (rows - 1) * m.hidden;
  m.Norm(h_last, final_norm, m.x.data(), 1);
  uint16_t* logits_bf16 = m.arena.Alloc<uint16_t>(static_cast<size_t>(m.vocab));
  ApplyLinear(s, m.arena, lm, m.x.data(), logits_bf16, 1);

  Qwen35MoeHeadResult out;
  switch (req.kind) {
    case Qwen35MoeHeadRequest::Kind::kLogits: {
      r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.logits_dev.data()),
                                   m.vocab, reinterpret_cast<int64_t>(s));
      m.stream.Synchronize();
      out.logits.resize(static_cast<size_t>(m.vocab));
      m.logits_dev.CopyToHost(out.logits.data(), static_cast<size_t>(m.vocab));
      break;
    }
    case Qwen35MoeHeadRequest::Kind::kArgmax: {
      r4dx_argmax_bf16(reinterpret_cast<int64_t>(logits_bf16), reinterpret_cast<int64_t>(m.argmax_dev.data()), m.vocab,
                       reinterpret_cast<int64_t>(s));
      R4DX_HIP_CHECK(hipMemcpyAsync(m.greedy_host.data(), m.argmax_dev.data(), sizeof(int32_t), hipMemcpyDeviceToHost, s));
      m.stream.Synchronize();
      out.token = m.greedy_host[0];
      break;
    }
    case Qwen35MoeHeadRequest::Kind::kSummary: {
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

std::vector<float> Qwen35MoeStage::LastLogitsRow() {
  Impl& m = *impl_;
  Qwen35MoeDeviceScope dev(m.device);
  if (!m.head_refs) Fail("LastLogitsRow: this stage has no head");
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
}

}  // namespace r4dx::model
