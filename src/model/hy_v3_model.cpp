#include "hy_v3_model.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/hy_v3_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/lfm2_kernels.h"
#include "r4dx/model/attention/lfm2_attention_layer.hpp"  // Lfm2PrefillSplits

namespace r4dx::model {

namespace {

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

constexpr float kMinSummaryTemperature = 0.01f;

float SummaryInvTemperature(const kernels::SampleParams& params) {
  if (!(params.temperature > 0.0f)) return 0.0f;
  if (params.temperature < kMinSummaryTemperature) return 0.0f;
  const float inv = 1.0f / params.temperature;
  if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity())) return 0.0f;
  return inv;
}

// R4DX_HYV3_KV=bf16 forces bf16 KV cache (default: fp8 e4m3, halving KV VRAM).
bool HyV3KvFp8Enabled() {
  static const bool v = [] {
    const char* e = std::getenv("R4DX_HYV3_KV");
    if (e == nullptr || *e == '\0' || std::strcmp(e, "fp8") == 0) return true;
    if (std::strcmp(e, "bf16") == 0) return false;
    throw std::invalid_argument(std::string("R4DX_HYV3_KV='") + e + "' (want fp8|bf16)");
  }();
  return v;
}

// R4DX_HYV3_GRAPH=0 disables HIP graph capture for T==1 decode (default: enabled).
bool HyV3DecodeGraphEnabled() {
  static const bool v = [] {
    const char* e = std::getenv("R4DX_HYV3_GRAPH");
    if (e == nullptr || *e == '\0' || std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) return true;
    if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return false;
    return true;
  }();
  return v;
}

template <class Record>
bool CaptureDecodeGraph(hipStream_t s, hipGraphExec_t* exec, Record&& record) {
  if (hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal) != hipSuccess) return false;
  struct Guard {
    hipStream_t stream;
    bool capturing = true;
    hipGraph_t graph = nullptr;
    ~Guard() {
      if (capturing) static_cast<void>(hipStreamEndCapture(stream, &graph));
      if (graph != nullptr) static_cast<void>(hipGraphDestroy(graph));
    }
  } guard{s};
  record();
  guard.capturing = false;
  if (hipStreamEndCapture(s, &guard.graph) != hipSuccess) return false;
  return hipGraphInstantiate(exec, guard.graph, nullptr, nullptr, 0) == hipSuccess;
}

void ApplyLinearFast(hipStream_t stream, core::Arena& arena, const QuantLinear& w,
                     const uint16_t* a_bf16, uint16_t* c_bf16, int64_t M) {
  if (M == 1 && w.layout == Layout::kW4a16) {
    const int group = w.w4a16_group > 0 ? w.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
    // Exact 64-CU alignment for o_proj (N=2048 -> 128 tiles / WV=2 = 64 blocks = 1.0/CU, SK=4 -> 8 waves/block, 512B LDS)
    if (w.N == 2048 && w.K == 4096 && group == 64) {
      r4dx_gemm_w4a16_bf16a_m1(P(a_bf16), P(w.wq.data()), P(w.w4a16_wsz.data()), P(c_bf16),
                               4096, 2048, 64, /*WV=*/2, /*SK=*/4, /*NPW=*/1, /*NT=*/1, P(stream));
      return;
    }
    const LinearTuning t = PickTuning(Layout::kW4a16, w.N, w.K, 1, group);
    r4dx_gemm_w4a16_bf16a_m1(P(a_bf16), P(w.wq.data()), P(w.w4a16_wsz.data()), P(c_bf16),
                             static_cast<int>(w.K), static_cast<int>(w.N), group, t.WV, t.SK,
                             t.NPW, t.NT, P(stream));
    return;
  }
  ApplyLinear(stream, arena, w, a_bf16, c_bf16, M);
}

}  // namespace

void HyV3Model::GraphExecDeleter::operator()(void* p) const noexcept {
  if (p != nullptr) static_cast<void>(hipGraphExecDestroy(static_cast<hipGraphExec_t>(p)));
}

void HyV3Model::InvalidateDecodeGraph() noexcept {
  decode_graph_exec_.reset();
  decode_graph_sb_ = -1;
  decode_graph_mode_ = -1;
  decode_graph_inv_temp_ = 0.0f;
}

HyV3Model HyV3Model::Load(const HyV3ModelOptions& opts) {
  if (opts.prefill_chunk < 1 || opts.prefill_chunk > 256) {
    throw std::invalid_argument("HyV3Model::Load: prefill_chunk must be in [1, 256]");
  }
  HyV3Model m;
  m.opts_ = opts;
  const HyV3ContainerInfo peek = HyV3Container::Inspect(opts.container_path, opts.layer_limit);
  HyV3LoadOptions lo;
  lo.layout = opts.layout;
  lo.lm_head_layout = (opts.layout == Layout::kW4a16 && peek.has_w8a16) ? Layout::kW8a16 : opts.layout;
  lo.layer_limit = opts.layer_limit;
  m.container_ = HyV3Container::Load(opts.container_path, lo);

  const HyV3Config& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t layers = m.container_.NumLoadedLayers();
  m.max_ctx_ = cfg.ResolveMaxCtx(opts.max_ctx);
  m.max_chunk_ = opts.prefill_chunk;
  m.kv_fp8_ = HyV3KvFp8Enabled();
  m.decode_graph_enabled_ = HyV3DecodeGraphEnabled() && (opts.layout == Layout::kW4a16);

  const int64_t rows = std::max<int64_t>(m.max_chunk_, 64);
  m.ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows));
  m.positions_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows));
  m.seqused_host_ = core::PinnedBuffer<int32_t>(1);
  m.ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.positions_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
  m.seqused_dev_ = core::DeviceBuffer<int32_t>(1);
  m.buf_h_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * hidden));
  m.buf_x_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * hidden));
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(cfg.vocab_size));
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  m.summary_ids_dev_ = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
  m.summary_vals_dev_ = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
  m.summary_lse_dev_ = core::DeviceBuffer<float>(1);
  m.moe_host_.Allocate(rows, cfg.num_experts_per_tok + 1, cfg.num_experts + 1);
  m.arena_.Reserve(128ull * 1024 * 1024);

  std::vector<float> ones(static_cast<size_t>(cfg.num_key_value_heads), 1.0f);
  m.kv_descale_ones_ = core::DeviceBuffer<float>(ones.size());
  m.kv_descale_ones_.CopyFromHost(ones.data(), ones.size());

  m.kv_.reserve(static_cast<size_t>(layers));
  const int kvh = static_cast<int>(cfg.num_key_value_heads);
  const int hd = static_cast<int>(cfg.head_dim);
  const auto kv_dtype = m.kv_fp8_ ? attention::GemmaKvDtype::kFp8 : attention::GemmaKvDtype::kBf16;
  for (int64_t i = 0; i < layers; ++i) {
    m.kv_.push_back(attention::GemmaKvCache::Contiguous(kvh, hd, m.max_ctx_, kv_dtype));
    m.kv_bytes_ += static_cast<int64_t>(m.kv_.back().Bytes());
  }
  // Warm up lazy internal scratch buffers in argmax/topk_lse before any HIP graph stream capture.
  if (cfg.vocab_size > 0) {
    const int64_t s = P(m.stream_.get());
    r4dx_argmax_f32(P(m.logits_dev_.data()), P(m.argmax_dev_.data()), cfg.vocab_size, s);
    r4dx_topk_lse_f32_ws(P(m.logits_dev_.data()), P(m.summary_ids_dev_.data()), P(m.summary_vals_dev_.data()),
                         P(m.summary_lse_dev_.data()), 1, cfg.vocab_size, 1.0f, s, 0);
  }
  m.stream_.Synchronize();
  std::cerr << "[r4dx::model::HyV3Model] " << layers << " layers, max_ctx " << m.max_ctx_
            << ", KV " << (m.kv_fp8_ ? "fp8" : "bf16") << " = "
            << (static_cast<double>(m.kv_bytes_) / (1024.0 * 1024.0 * 1024.0))
            << " GiB, attention r4d h128_gqa8, grouped w4a16 MoE\n";
  return m;
}

void HyV3Model::Reset() {
  stream_.Synchronize();
  pos_ = 0;
  started_ = false;
  arena_.Reset();
  ckpt_pos_ = -1;
}

void HyV3Model::SaveCheckpoint() {
  if (!opts_.prompt_checkpoint) throw std::runtime_error("HyV3Model::SaveCheckpoint: loaded without prompt_checkpoint");
  if (!started_) throw std::runtime_error("HyV3Model::SaveCheckpoint: nothing has been fed yet");
  ckpt_pos_ = pos_;
}

void HyV3Model::RestoreCheckpoint() {
  if (!opts_.prompt_checkpoint) throw std::runtime_error("HyV3Model::RestoreCheckpoint: loaded without prompt_checkpoint");
  if (ckpt_pos_ < 0) throw std::runtime_error("HyV3Model::RestoreCheckpoint: no checkpoint since Load()/Reset()");
  stream_.Synchronize();
  pos_ = ckpt_pos_;
  started_ = true;
  arena_.Reset();
}

void HyV3Model::UploadPositionsAsync(int64_t start_pos, int64_t T) {
  for (int64_t t = 0; t < T; ++t) positions_host_.data()[t] = static_cast<int32_t>(start_pos + t);
  positions_dev_.CopyFromHostAsync(positions_host_.data(), static_cast<size_t>(T), stream_);
  seqused_host_.data()[0] = static_cast<int32_t>(start_pos + T);
  seqused_dev_.CopyFromHostAsync(seqused_host_.data(), 1, stream_);
}

void HyV3Model::RunLayer(int64_t i, uint16_t* h, uint16_t* x_cur, const uint16_t* next_norm_w,
                         uint16_t* x_next, int64_t T, int64_t start_pos, SpanAccumulator* prof) {
  const HyV3Config& cfg = container_.Config();
  const int64_t H = cfg.hidden_size;
  const int Hq = static_cast<int>(cfg.num_attention_heads);
  const int Hk = static_cast<int>(cfg.num_key_value_heads);
  const int D = static_cast<int>(cfg.head_dim);
  const float eps = static_cast<float>(cfg.norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);
  const HyV3LayerWeights& lw = container_.Layer(i);
  const size_t n = static_cast<size_t>(T * H);

  uint16_t* sub = arena_.Alloc<uint16_t>(n, 16);

  // ---- Attention: Q/K/V -> fused qk_norm+rope+kv_write -> r4d h128_gqa8 -> O ----
  uint16_t* q = arena_.Alloc<uint16_t>(static_cast<size_t>(T * Hq * D), 16);
  uint16_t* k = arena_.Alloc<uint16_t>(static_cast<size_t>(T * Hk * D), 16);
  uint16_t* v = arena_.Alloc<uint16_t>(static_cast<size_t>(T * Hk * D), 16);
  uint16_t* ao = arena_.Alloc<uint16_t>(static_cast<size_t>(T * Hq * D), 16);

  if (T == 1 && lw.attn.q.layout == Layout::kW4a16 && lw.attn.k.layout == Layout::kW4a16 &&
      lw.attn.v.layout == Layout::kW4a16) {
    const int gq = lw.attn.q.w4a16_group > 0 ? lw.attn.q.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
    const int gk = lw.attn.k.w4a16_group > 0 ? lw.attn.k.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
    const int gv = lw.attn.v.w4a16_group > 0 ? lw.attn.v.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
    if (gq == gk && gq == gv) {
      // 320 tiles / WV=2 = 160 blocks x SK=4 (8 waves/block = 1280 waves = 20 waves/CU, 512B LDS, 8 groups/wave!)
      ProfiledCall(prof, s_raw, "gemm:attn.qkv_fused", [&] {
        r4dx_gemm_w4a16_qkv_bf16a_m1(P(x_cur),
                                     P(lw.attn.q.wq.data()), P(lw.attn.q.w4a16_wsz.data()), P(q), Hq * D,
                                     P(lw.attn.k.wq.data()), P(lw.attn.k.w4a16_wsz.data()), P(k), Hk * D,
                                     P(lw.attn.v.wq.data()), P(lw.attn.v.w4a16_wsz.data()), P(v), Hk * D,
                                     static_cast<int>(H), gq, /*WV=*/2, /*SK=*/4, /*NT=*/1, s);
      });
    } else {
      ProfiledCall(prof, s_raw, "gemm:attn.q_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.q, x_cur, q, T); });
      ProfiledCall(prof, s_raw, "gemm:attn.k_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.k, x_cur, k, T); });
      ProfiledCall(prof, s_raw, "gemm:attn.v_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.v, x_cur, v, T); });
    }
  } else {
    ProfiledCall(prof, s_raw, "gemm:attn.q_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.q, x_cur, q, T); });
    ProfiledCall(prof, s_raw, "gemm:attn.k_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.k, x_cur, k, T); });
    ProfiledCall(prof, s_raw, "gemm:attn.v_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.v, x_cur, v, T); });
  }

  attention::GemmaKvCache& kv = kv_[static_cast<size_t>(i)];
  ProfiledCall(prof, s_raw, "attn.precore", [&] {
    r4dx_hy_v3_attn_precore(P(q), P(k), P(v), P(lw.attn.q_norm.data()), P(lw.attn.k_norm.data()),
                            P(positions_dev_.data()), P(kv.Data()), kv_fp8_ ? 1 : 0,
                            kv_fp8_ ? P(kv_descale_ones_.data()) : 0,
                            kv_fp8_ ? P(kv_descale_ones_.data()) : 0,
                            static_cast<int>(T), Hq, Hk, eps, static_cast<float>(cfg.rope_theta),
                            attention::GemmaKvCache::kBlockSize, kv.KvBlockStride(), kv.KvHeadStride(), s);
  });

  {
    R4DArgs a{};
    a.q = q;
    a.kv = kv.Data();
    a.out = ao;
    a.block_table = kv.BlockTable();
    a.seqused_k = seqused_dev_.data();
    a.k_descale = kv_fp8_ ? kv_descale_ones_.data() : nullptr;
    a.v_descale = kv_fp8_ ? kv_descale_ones_.data() : nullptr;
    a.q_descale = nullptr;
    a.scratch = nullptr;
    a.num_seqs = 1;
    a.q_len = static_cast<int>(T);
    a.q_heads = Hq;
    a.kv_heads = Hk;
    a.head_dim = D;
    a.block_size = attention::GemmaKvCache::kBlockSize;
    a.max_blocks = kv.TableEntries();
    a.max_ctx = static_cast<int>(start_pos + T);
    a.scale = 1.0f / std::sqrt(static_cast<float>(D));
    a.kv_block_stride = kv.KvBlockStride();
    a.kv_head_stride = kv.KvHeadStride();
    a.splits = 0;

    if (T <= 8) {
      const int64_t sb = core::r4d::AttnDecodeH128Gqa8ScratchBytes(a);
      if (sb > 0) a.scratch = arena_.Alloc<uint8_t>(static_cast<size_t>(sb), 256);
      ProfiledCall(prof, s_raw, "attn.core_decode", [&] {
        if (kv_fp8_) core::r4d::AttnDecodeH128Gqa8Fp8Kv(a, s_raw);
        else core::r4d::AttnDecodeH128Gqa8Bf16Kv(a, s_raw);
      });
    } else {
      const int splits = attention::Lfm2PrefillSplits(a.max_ctx, static_cast<int>(T), Hk);
      a.splits = splits;
      if (splits > 1) {
        const int64_t sb = core::r4d::AttnPrefillSplitKvH128Gqa8ScratchBytes(a);
        if (sb > 0) a.scratch = arena_.Alloc<uint8_t>(static_cast<size_t>(sb), 256);
        ProfiledCall(prof, s_raw, "attn.core_prefill", [&] {
          if (kv_fp8_) core::r4d::AttnPrefillSplitKvH128Gqa8Fp8Kv(a, s_raw);
          else core::r4d::AttnPrefillSplitKvH128Gqa8Bf16Kv(a, s_raw);
        });
      } else {
        ProfiledCall(prof, s_raw, "attn.core_prefill", [&] {
          if (kv_fp8_) core::r4d::AttnPrefillH128Gqa8Fp8Kv(a, s_raw);
          else core::r4d::AttnPrefillH128Gqa8Bf16Kv(a, s_raw);
        });
      }
    }
  }

  ProfiledCall(prof, s_raw, "gemm:attn.o_proj", [&] { ApplyLinearFast(s_raw, arena_, lw.attn.o, ao, sub, T); });

  ProfiledCall(prof, s_raw, "layer.post_attn_norm", [&] {
    r4dx_hy_v3_residual_rmsnorm_bf16(P(h), P(sub), P(lw.post_attn_norm.data()), P(x_cur),
                                     static_cast<int>(T), static_cast<int>(H), eps, s);
  });

  if (lw.dense) {
    const int64_t I = cfg.intermediate_size;
    uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(T * 2 * I), 16);
    uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(T * I), 16);
    ProfiledCall(prof, s_raw, "gemm:mlp.gate_up", [&] { ApplyLinearFast(s_raw, arena_, lw.mlp.gate_up, x_cur, gu, T); });
    ProfiledCall(prof, s_raw, "mlp.silu_mul", [&] { r4dx_silu_mul_bf16(P(gu), P(act), T, I, 2 * I, s); });
    ProfiledCall(prof, s_raw, "gemm:mlp.down", [&] { ApplyLinearFast(s_raw, arena_, lw.mlp.down, act, sub, T); });
    if (next_norm_w != nullptr && x_next != nullptr) {
      ProfiledCall(prof, s_raw, "layer.res_next_norm", [&] {
        r4dx_hy_v3_residual_rmsnorm_bf16(P(h), P(sub), P(next_norm_w), P(x_next),
                                         static_cast<int>(T), static_cast<int>(H), eps, s);
      });
    } else {
      ProfiledCall(prof, s_raw, "layer.residual_add", [&] { r4dx_residual_add_bf16(P(h), P(sub), P(h), T * H, s); });
    }
  } else if (lw.HasW4MoePtrs()) {
    // ---- Fast path: zero-sync grouped w4a16 MoE (8 routed + 1 shared = 9 slots) + fused combine/res/next-norm ----
    const int64_t E = cfg.num_experts;
    const int64_t K = cfg.num_experts_per_tok;
    const int64_t S = K + 1;  // 8 routed + 1 shared expert at index E
    const int64_t Im = cfg.moe_intermediate_size;
    const int64_t Pn = T * S;

    int32_t* ids = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
    float* wts = arena_.Alloc<float>(static_cast<size_t>(Pn), 16);
    float* sig_scratch = (T == 1) ? arena_.Alloc<float>(static_cast<size_t>(E), 16) : nullptr;
    ProfiledCall(prof, s_raw, "moe.router", [&] {
      r4dx_hy_v3_router_f32(P(x_cur), P(lw.moe.router.data()), P(lw.moe.expert_bias.data()), P(ids), P(wts),
                            P(sig_scratch), static_cast<int>(T), static_cast<int>(H), static_cast<int>(E),
                            static_cast<int>(K), static_cast<float>(cfg.router_scaling_factor), /*append_shared=*/1, s);
    });

    uint16_t* ys = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);
    uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * Im), 16);
    const int g = lw.moe_w4_group;

    if (T == 1) {
      if (g == 64) {
        // Fused gate_up + silu_mul (WV=2, SK=4 -> 216 WGs x 8 waves = 1728 waves, 8 groups/wave, 1KB LDS!)
        ProfiledCall(prof, s_raw, "gemm:moe.gate_up_silu", [&] {
          r4dx_gemm_w4a16_moe_gate_up_silu_decode(P(x_cur), P(lw.gate_up_w4_wq_ptrs.data()),
                                                  P(lw.gate_up_w4_wsz_ptrs.data()), P(ids), P(act),
                                                  static_cast<int>(Pn), static_cast<int>(S),
                                                  static_cast<int>(E + 1), static_cast<int>(H),
                                                  static_cast<int>(Im), g, /*WV=*/2, /*SK=*/4, /*NT=*/1, s);
        });
      } else {
        uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * 2 * Im), 16);
        const LinearTuning t_gu = PickTuning(Layout::kW4a16, 2 * Im, H, 1, g);
        ProfiledCall(prof, s_raw, "gemm:moe.gate_up", [&] {
          r4dx_gemm_w4a16_moe_decode(P(x_cur), P(lw.gate_up_w4_wq_ptrs.data()), P(lw.gate_up_w4_wsz_ptrs.data()),
                                     P(ids), P(gu), static_cast<int>(Pn), static_cast<int>(S),
                                     static_cast<int>(E + 1), static_cast<int>(H), static_cast<int>(2 * Im),
                                     g, t_gu.WV, t_gu.SK, t_gu.NPW, t_gu.NT, s);
        });
        ProfiledCall(prof, s_raw, "moe.silu_mul", [&] { r4dx_silu_mul_bf16(P(gu), P(act), Pn, Im, 2 * Im, s); });
      }
      if (g == 64 && S <= 16) {
        // Fused down + weighted combine + residual add into h[H] (128 blocks = 2.0/CU, 9 waves/block, zero ys traffic!)
        ProfiledCall(prof, s_raw, "gemm:moe.down_comb_res", [&] {
          r4dx_gemm_w4a16_moe_down_combine_res_decode(P(act), P(lw.down_w4_wq_ptrs.data()),
                                                      P(lw.down_w4_wsz_ptrs.data()), P(ids), P(wts), P(h),
                                                      static_cast<int>(S), static_cast<int>(E + 1),
                                                      static_cast<int>(Im), static_cast<int>(H), g, /*NT=*/1, s);
        });
        if (next_norm_w != nullptr && x_next != nullptr) {
          ProfiledCall(prof, s_raw, "layer.next_norm", [&] {
            r4dx_hy_v3_rmsnorm_bf16(P(h), P(next_norm_w), P(x_next), 1, static_cast<int>(H), eps, s);
          });
        }
      } else {
        ProfiledCall(prof, s_raw, "gemm:moe.down", [&] {
          r4dx_gemm_w4a16_moe_decode(P(act), P(lw.down_w4_wq_ptrs.data()), P(lw.down_w4_wsz_ptrs.data()),
                                     P(ids), P(ys), static_cast<int>(Pn), /*a_row_group=*/1,
                                     static_cast<int>(E + 1), static_cast<int>(Im), static_cast<int>(H),
                                     g, /*WV=*/4, /*SK=*/1, /*NPW=*/1, /*NT=*/1, s);
        });
        ProfiledCall(prof, s_raw, "moe.combine_res_norm", [&] {
          r4dx_hy_v3_moe_combine_residual_rmsnorm_bf16(P(ys), 0, P(wts), P(h), P(next_norm_w), P(x_next),
                                                       1, static_cast<int>(S), static_cast<int>(H), eps, s);
        });
      }
    } else {
      uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * 2 * Im), 16);
      int32_t* counts = arena_.Alloc<int32_t>(static_cast<size_t>(E + 1), 16);
      int32_t* offsets = arena_.Alloc<int32_t>(static_cast<size_t>(E + 2), 16);
      int32_t* sorted = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
      int32_t* pos_of_pair = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
      uint16_t* xs = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);
      ProfiledCall(prof, s_raw, "moe.sort", [&] {
        r4dx_lfm2_moe_sort(P(ids), static_cast<int>(T), static_cast<int>(S), static_cast<int>(E + 1),
                           P(counts), P(offsets), P(sorted), P(pos_of_pair), s);
      });
      ProfiledCall(prof, s_raw, "moe.gather", [&] {
        r4dx_lfm2_moe_gather_bf16(P(x_cur), P(sorted), P(xs), static_cast<int>(Pn), static_cast<int>(S),
                                  static_cast<int>(H), H, P(offsets + E + 1), s);
      });
      // Prefill MoE: WV=2, SK=1, NPW=4 -> ZERO LDS, ZERO __syncthreads(), 4x fewer fragA_bf16 conversions!
      ProfiledCall(prof, s_raw, "gemm:moe.gate_up", [&] {
        r4dx_gemm_w4a16_moe_prefill(P(xs), P(lw.gate_up_w4_wq_ptrs.data()), P(lw.gate_up_w4_wsz_ptrs.data()),
                                    P(offsets), P(gu), static_cast<int>(Pn), static_cast<int>(E + 1),
                                    static_cast<int>(H), static_cast<int>(2 * Im), g, /*WV=*/2, /*SK=*/1,
                                    /*NPW=*/4, /*NT=*/1, s);
      });
      ProfiledCall(prof, s_raw, "moe.silu_mul", [&] { r4dx_silu_mul_bf16(P(gu), P(act), Pn, Im, 2 * Im, s); });
      ProfiledCall(prof, s_raw, "gemm:moe.down", [&] {
        r4dx_gemm_w4a16_moe_prefill(P(act), P(lw.down_w4_wq_ptrs.data()), P(lw.down_w4_wsz_ptrs.data()),
                                    P(offsets), P(ys), static_cast<int>(Pn), static_cast<int>(E + 1),
                                    static_cast<int>(Im), static_cast<int>(H), g, /*WV=*/2, /*SK=*/1,
                                    /*NPW=*/4, /*NT=*/1, s);
      });
      ProfiledCall(prof, s_raw, "moe.combine_res_norm", [&] {
        r4dx_hy_v3_moe_combine_residual_rmsnorm_bf16(P(ys), P(pos_of_pair), P(wts), P(h), P(next_norm_w), P(x_next),
                                                     static_cast<int>(T), static_cast<int>(S), static_cast<int>(H),
                                                     eps, s);
      });
    }
  } else {
    // Fallback staged path for non-w4a16 layouts (bf16 / w8a16)
    const int64_t E = cfg.num_experts;
    const int64_t K = cfg.num_experts_per_tok;
    const int64_t Im = cfg.moe_intermediate_size;
    const int64_t Pn = T * K;
    int32_t* ids = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
    float* wts = arena_.Alloc<float>(static_cast<size_t>(Pn), 16);
    r4dx_hy_v3_router_f32(P(x_cur), P(lw.moe.router.data()), P(lw.moe.expert_bias.data()), P(ids), P(wts),
                          0, static_cast<int>(T), static_cast<int>(H), static_cast<int>(E), static_cast<int>(K),
                          static_cast<float>(cfg.router_scaling_factor), /*append_shared=*/0, s);
    uint16_t* ys = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);
    uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(2 * Im), 16);
    uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(Im), 16);
    R4DX_HIP_CHECK(hipMemcpyAsync(moe_host_.ids.data(), ids, static_cast<size_t>(Pn) * sizeof(int32_t),
                                  hipMemcpyDeviceToHost, s_raw));
    R4DX_HIP_CHECK(hipStreamSynchronize(s_raw));
    const size_t mark = arena_.Mark();
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t j = 0; j < K; ++j) {
        arena_.Rewind(mark);
        const size_t ue = static_cast<size_t>(moe_host_.ids.data()[t * K + j]);
        ApplyLinear(s_raw, arena_, lw.moe.gate_up[ue], x_cur + t * H, gu, 1);
        r4dx_silu_mul_bf16(P(gu), P(act), 1, Im, 2 * Im, s);
        ApplyLinear(s_raw, arena_, lw.moe.down[ue], act, ys + (t * K + j) * H, 1);
      }
    }
    r4dx_lfm2_moe_combine_bf16(P(ys), 0, P(wts), P(sub), static_cast<int>(T), static_cast<int>(K), static_cast<int>(H), s);
    uint16_t* sgu = arena_.Alloc<uint16_t>(static_cast<size_t>(T * 2 * Im), 16);
    uint16_t* sact = arena_.Alloc<uint16_t>(static_cast<size_t>(T * Im), 16);
    uint16_t* sout = arena_.Alloc<uint16_t>(n, 16);
    ApplyLinear(s_raw, arena_, lw.shared_mlp.gate_up, x_cur, sgu, T);
    r4dx_silu_mul_bf16(P(sgu), P(sact), T, Im, 2 * Im, s);
    ApplyLinear(s_raw, arena_, lw.shared_mlp.down, sact, sout, T);
    r4dx_residual_add_bf16(P(sub), P(sout), P(sub), T * H, s);
    if (next_norm_w != nullptr && x_next != nullptr) {
      r4dx_lfm2_residual_rmsnorm_plain_bf16(P(h), P(sub), P(next_norm_w), P(x_next), T, H, eps, s);
    } else {
      r4dx_residual_add_bf16(P(h), P(sub), P(h), T * H, s);
    }
  }

  arena_.Reset();
}

void HyV3Model::RunDecodeBody(uint16_t* h, uint16_t* x, int64_t start_pos, bool want_logits,
                              bool greedy, const SummaryRequest* summary_out, SpanAccumulator* prof) {
  const HyV3Config& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const float eps = static_cast<float>(cfg.norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);

  ProfiledCall(prof, s_raw, "embed", [&] {
    r4dx_embedding_gather_bf16(P(container_.EmbedTokensDevice().data()), P(ids_dev_.data()), P(h), 1, hidden,
                               cfg.vocab_size, s);
  });
  if (num_layers > 0) {
    ProfiledCall(prof, s_raw, "layer.input_norm", [&] {
      r4dx_hy_v3_rmsnorm_bf16(P(h), P(container_.Layer(0).input_norm.data()), P(x), 1, static_cast<int>(hidden), eps, s);
    });
  }
  for (int64_t i = 0; i < num_layers; ++i) {
    const uint16_t* next_nw = (i + 1 < num_layers)
                                  ? container_.Layer(i + 1).input_norm.data()
                                  : (want_logits ? container_.FinalNorm().data() : nullptr);
    uint16_t* x_next = (next_nw != nullptr) ? x : nullptr;
    RunLayer(i, h, x, next_nw, x_next, 1, start_pos, prof);
  }
  if (want_logits) {
    const uint16_t* xn = x;
    uint16_t* logits_bf16 = arena_.Alloc<uint16_t>(static_cast<size_t>(cfg.vocab_size), 16);
    if (num_layers == 0) {
      uint16_t* xn0 = arena_.Alloc<uint16_t>(static_cast<size_t>(hidden), 16);
      r4dx_rmsnorm_plain_bf16(P(h), P(container_.FinalNorm().data()), P(xn0), 1, hidden, eps, 0, s);
      xn = xn0;
    }
    ProfiledCall(prof, s_raw, "gemm:lm_head", [&] { ApplyLinear(s_raw, arena_, container_.LmHead(), xn, logits_bf16, 1); });
    ProfiledCall(prof, s_raw, "lm_head.widen", [&] {
      r4dx_model_widen_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), cfg.vocab_size, s);
    });
    if (greedy) {
      r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), cfg.vocab_size, s);
    } else if (summary_out != nullptr) {
      r4dx_topk_lse_f32_ws(P(logits_dev_.data()), P(summary_ids_dev_.data()), P(summary_vals_dev_.data()),
                           P(summary_lse_dev_.data()), 1, cfg.vocab_size, summary_out->inv_temperature, s, 0);
    }
    arena_.Reset();
  }
}

std::vector<float> HyV3Model::RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                                       const SummaryRequest* summary_out, SpanAccumulator* prof) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("HyV3Model::RunChunk: token_ids.size() must be in [1, " + std::to_string(max_chunk_) + "]");
  }
  if (pos_ + T > max_ctx_) {
    throw std::runtime_error("HyV3Model::RunChunk: position " + std::to_string(pos_ + T) + " exceeds max_ctx " +
                             std::to_string(max_ctx_));
  }
  const HyV3Config& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const float eps = static_cast<float>(cfg.norm_eps);
  const hipStream_t s_raw = stream_.get();
  const int64_t s = P(s_raw);

  for (int32_t id : token_ids) {
    if (id < 0 || id >= cfg.vocab_size) {
      throw std::invalid_argument("HyV3Model::RunChunk: token id " + std::to_string(id) + " out of range");
    }
  }

  std::copy(token_ids.begin(), token_ids.end(), ids_host_.data());
  ids_dev_.CopyFromHostAsync(ids_host_.data(), static_cast<size_t>(T), stream_);
  UploadPositionsAsync(pos_, T);
  uint16_t* h = buf_h_.data();
  uint16_t* x = buf_x_.data();

  if (T == 1 && prof == nullptr && decode_graph_enabled_ && num_layers > 0) {
    R4DArgs probe{};
    probe.num_seqs = 1;
    probe.q_len = 1;
    probe.q_heads = static_cast<int>(cfg.num_attention_heads);
    probe.kv_heads = static_cast<int>(cfg.num_key_value_heads);
    probe.head_dim = static_cast<int>(cfg.head_dim);
    probe.max_ctx = static_cast<int>(pos_ + 1);
    probe.splits = 0;
    const int64_t sb = core::r4d::AttnDecodeH128Gqa8ScratchBytes(probe);
    const int mode = !want_logits ? 0 : (greedy_out != nullptr ? 1 : (summary_out != nullptr ? 2 : 3));
    const float inv_t = (mode == 2 && summary_out != nullptr) ? summary_out->inv_temperature : 0.0f;

    if (decode_graph_exec_ != nullptr && decode_graph_sb_ == sb && decode_graph_mode_ == mode &&
        decode_graph_inv_temp_ == inv_t) {
      R4DX_HIP_CHECK(hipGraphLaunch(static_cast<hipGraphExec_t>(decode_graph_exec_.get()), s_raw));
    } else {
      InvalidateDecodeGraph();
      arena_.Reset();
      hipGraphExec_t exec = nullptr;
      const bool captured = CaptureDecodeGraph(s_raw, &exec, [&] {
        RunDecodeBody(h, x, pos_, want_logits, greedy_out != nullptr, summary_out, nullptr);
      });
      if (captured && exec != nullptr) {
        decode_graph_exec_.reset(exec);
        decode_graph_sb_ = sb;
        decode_graph_mode_ = mode;
        decode_graph_inv_temp_ = inv_t;
        R4DX_HIP_CHECK(hipGraphLaunch(exec, s_raw));
      } else {
        decode_graph_enabled_ = false;
        RunDecodeBody(h, x, pos_, want_logits, greedy_out != nullptr, summary_out, nullptr);
      }
    }
  } else if (T == 1) {
    RunDecodeBody(h, x, pos_, want_logits, greedy_out != nullptr, summary_out, prof);
  } else {
    ProfiledCall(prof, s_raw, "embed", [&] {
      r4dx_embedding_gather_bf16(P(container_.EmbedTokensDevice().data()), P(ids_dev_.data()), P(h), T, hidden,
                                 cfg.vocab_size, s);
    });
    if (num_layers > 0) {
      ProfiledCall(prof, s_raw, "layer.input_norm", [&] {
        r4dx_rmsnorm_plain_bf16(P(h), P(container_.Layer(0).input_norm.data()), P(x), T, hidden, eps, 0, s);
      });
    }
    for (int64_t i = 0; i < num_layers; ++i) {
      const uint16_t* next_nw = (i + 1 < num_layers)
                                    ? container_.Layer(i + 1).input_norm.data()
                                    : (want_logits ? container_.FinalNorm().data() : nullptr);
      uint16_t* x_next = (next_nw != nullptr) ? x : nullptr;
      RunLayer(i, h, x, next_nw, x_next, T, pos_, prof);
    }
    if (want_logits) {
      const uint16_t* xn = x + (T - 1) * hidden;
      uint16_t* logits_bf16 = arena_.Alloc<uint16_t>(static_cast<size_t>(cfg.vocab_size), 16);
      if (num_layers == 0) {
        uint16_t* xn0 = arena_.Alloc<uint16_t>(static_cast<size_t>(hidden), 16);
        r4dx_rmsnorm_plain_bf16(P(h + (T - 1) * hidden), P(container_.FinalNorm().data()), P(xn0), 1, hidden, eps, 0, s);
        xn = xn0;
      }
      ProfiledCall(prof, s_raw, "gemm:lm_head", [&] { ApplyLinear(s_raw, arena_, container_.LmHead(), xn, logits_bf16, 1); });
      ProfiledCall(prof, s_raw, "lm_head.widen", [&] {
        r4dx_model_widen_bf16_to_f32(P(logits_bf16), P(logits_dev_.data()), cfg.vocab_size, s);
      });
      if (greedy_out != nullptr) {
        r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), cfg.vocab_size, s);
      } else if (summary_out != nullptr) {
        r4dx_topk_lse_f32_ws(P(logits_dev_.data()), P(summary_ids_dev_.data()), P(summary_vals_dev_.data()),
                             P(summary_lse_dev_.data()), 1, cfg.vocab_size, summary_out->inv_temperature, s, 0);
      }
      arena_.Reset();
    }
  }
  if (prof == nullptr) stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits && prof == nullptr) {
    if (greedy_out != nullptr) {
      argmax_dev_.CopyToHost(greedy_out, 1);
    } else if (summary_out != nullptr) {
      FetchRowSummary(summary_out->inv_temperature, summary_out->out);
    } else {
      logits.resize(static_cast<size_t>(cfg.vocab_size));
      logits_dev_.CopyToHost(logits.data(), logits.size());
    }
  }
  pos_ += T;
  started_ = true;
  return logits;
}

void HyV3Model::FetchRowSummary(float inv_temperature, kernels::RowSummary* out) {
  out->k = R4DX_TOPK_LSE_K;
  out->vocab = container_.Config().vocab_size;
  out->inv_temperature = inv_temperature;
  summary_ids_dev_.CopyToHost(out->ids, R4DX_TOPK_LSE_K);
  summary_vals_dev_.CopyToHost(out->vals, R4DX_TOPK_LSE_K);
  summary_lse_dev_.CopyToHost(&out->lse, 1);
}

std::vector<float> HyV3Model::Prefill(const std::vector<int32_t>& token_ids) {
  if (token_ids.empty()) throw std::invalid_argument("HyV3Model::Prefill: empty token_ids");
  const int64_t total = static_cast<int64_t>(token_ids.size());
  for (int64_t off = 0; off < total; off += max_chunk_) {
    const int64_t n = std::min<int64_t>(max_chunk_, total - off);
    const bool last = (off + n == total);
    std::vector<int32_t> chunk(token_ids.begin() + off, token_ids.begin() + off + n);
    auto out = RunChunk(chunk, last, nullptr, nullptr, nullptr);
    if (last) return out;
  }
  return {};
}

std::vector<float> HyV3Model::DecodeStep(int32_t token_id) {
  return RunChunk({token_id}, true, nullptr, nullptr, nullptr);
}

int32_t HyV3Model::DecodeStepGreedy(int32_t token_id) {
  int32_t out = -1;
  RunChunk({token_id}, true, &out, nullptr, nullptr);
  return out;
}

int32_t HyV3Model::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
  if (params.temperature <= 0.0f) return DecodeStepGreedy(token_id);
  const float inv_temp = SummaryInvTemperature(params);
  const int64_t vocab = container_.Config().vocab_size;
  if (inv_temp == 0.0f || vocab <= R4DX_TOPK_LSE_K) {
    const std::vector<float> logits = DecodeStep(token_id);
    return kernels::SampleCanonical(logits.data(), vocab, params, kernels::DrawUniform01(rng));
  }
  kernels::RowSummary sum;
  SummaryRequest req{inv_temp, &sum};
  RunChunk({token_id}, true, nullptr, &req, nullptr);
  const double u = kernels::DrawUniform01(rng);
  const kernels::SummarySampleResult res = kernels::SampleFromSummary(sum, params, u);
  if (res.resolved) return res.token;
  ++sampled_fallback_rows_;
  std::vector<float> logits(static_cast<size_t>(vocab));
  logits_dev_.CopyToHost(logits.data(), logits.size());
  return kernels::SampleCanonical(logits.data(), vocab, params, u);
}

StepProfile HyV3Model::DecodeStepProfiled(int32_t token_id) {
  SpanAccumulator prof;
  const auto t0 = std::chrono::steady_clock::now();
  RunChunk({token_id}, true, nullptr, nullptr, &prof);
  const auto t1 = std::chrono::steady_clock::now();
  stream_.Synchronize();
  const auto t2 = std::chrono::steady_clock::now();
  StepProfile p;
  for (const auto& e : prof.Finish()) {
    p.entries.push_back({e.name, e.ms, e.count});
    p.gpu_sum_ms += e.ms;
  }
  p.host_enqueue_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  p.finish_wait_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  p.wall_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
  return p;
}

StepProfile HyV3Model::PrefillProfiled(const std::vector<int32_t>& token_ids) {
  SpanAccumulator prof;
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t total = static_cast<int64_t>(token_ids.size());
  for (int64_t off = 0; off < total; off += max_chunk_) {
    const int64_t n = std::min<int64_t>(max_chunk_, total - off);
    const bool last = (off + n == total);
    std::vector<int32_t> chunk(token_ids.begin() + off, token_ids.begin() + off + n);
    RunChunk(chunk, last, nullptr, nullptr, &prof);
  }
  const auto t1 = std::chrono::steady_clock::now();
  stream_.Synchronize();
  const auto t2 = std::chrono::steady_clock::now();
  StepProfile p;
  for (const auto& e : prof.Finish()) {
    p.entries.push_back({e.name, e.ms, e.count});
    p.gpu_sum_ms += e.ms;
  }
  p.host_enqueue_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  p.finish_wait_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  p.wall_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
  return p;
}

}  // namespace r4dx::model
