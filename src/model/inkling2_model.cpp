#include "inkling2_model.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/inkling2_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/lfm2_kernels.h"
#include "r4dx/kernels/mxfp8_gemm.h"
#include "tp/tp_vocab.h"

namespace r4dx::model {

namespace {

inline int64_t P(const void* p) {
    return reinterpret_cast<int64_t>(p);
}
constexpr float kMinSummaryTemperature = 0.01f;
constexpr uint32_t kNegInfBits = 0xFF800000u;
constexpr int kPoolBuffers = 12;

bool InkEnvFlag(const char* name, bool default_val) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
        return default_val;
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 || std::strcmp(v, "false") == 0)
        return false;
    return true;
}

bool ParseNv4TuneEnv(const char* name, LinearTuning& out) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
        return false;
    int wv = 0, sk = 0, npw = 0, nt = 0;
    if (std::sscanf(v, "%d,%d,%d,%d", &wv, &sk, &npw, &nt) == 4 && wv >= 1 && sk >= 1 &&
        (npw == 1 || npw == 4)) {
        out.WV = wv;
        out.SK = sk;
        out.NPW = npw;
        out.NT = nt;
        return true;
    }
    return false;
}

LinearTuning PickNvfp4MoeTuning(int64_t N, int64_t K, int64_t M, bool is_down) {
    LinearTuning t = PickTuning(Layout::kMxfp8, N, K, M);
    static const bool kVerifyNt8 = InkEnvFlag("R4DX_INK_NV4_VERIFY_NT", false);
    const int nt_max_m = kVerifyNt8 ? 8 : 4;
    // Dedicated R9700 NVFP4 MoE rows (9 K-blocks/wave, NPW=4): measured +3.7 tok/s on TP=2 DSpark
    // (104.5 -> 108.2).
    if (N >= 64 && (N % 64) == 0) {
        if (K == 2304) {
            t.WV = 1;
            t.SK = 8;
            t.NPW = 4;
            t.NT = (M <= nt_max_m ? 1 : 0);
        } else if (K == 1152) {
            t.WV = 1;
            t.SK = 4;
            t.NPW = 4;
            t.NT = (M <= nt_max_m ? 1 : 0);
        } else if (K == 576) {
            t.WV = 1;
            t.SK = 2;
            t.NPW = 4;
            t.NT = (M <= nt_max_m ? 1 : 0);
        }
    }
    LinearTuning ov = t;
    if (ParseNv4TuneEnv(is_down ? "R4DX_INK_NV4_DN" : "R4DX_INK_NV4_GU", ov) &&
        (K % (ov.SK * 32) == 0)) {
        return ov;
    }
    return t;
}

int64_t MoeDecodeMaxT() {
    static const int64_t v = []() -> int64_t {
        const char* e = std::getenv("R4DX_INK_MOE_DECODE_MAX_T");
        if (e == nullptr || *e == '\0') return 4;
        char* end = nullptr;
        const long parsed = std::strtol(e, &end, 10);
        if (end == e || *end != '\0') return 4;
        return std::clamp<int64_t>(parsed, 1, 8);
    }();
    return v;
}

bool FastArgmaxEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_FAST_ARGMAX", true);
    return v;
}

bool AsyncPostVerifyEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_ASYNC_POST_VERIFY", true);
    return v;
}

bool FuseRouterEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_FUSE_ROUTER", false);
    return v;
}

bool BatchSinkKvEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_BATCH_SINK_KV", true);
    return v;
}

bool ArBf16ToF32Enabled() {
    static const bool v = InkEnvFlag("R4DX_INK_AR_BF16_TO_F32", true);
    return v;
}

bool StridedPrefillQkvEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_STRIDED_PREFILL_QKV", true);
    return v;
}

bool DsparkAdaptEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_DSPARK_ADAPT", true);
    return v;
}

bool DsparkSpecSampleEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_SPEC_SAMPLE", false);
    return v;
}

bool DsparkAsyncMarkovEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_DSPARK_ASYNC", true);
    return v;
}

bool TpFp32AllReduceEnabled() {
    static const bool v = InkEnvFlag("R4DX_INK_TP_FP32_AR", false);
    return v;
}

struct TopKDist {
    int n = 0;
    bool closed = false;
    double tail_mass = 0.0;
    int32_t ids[kernels::kRowSummaryMaxK]{};
    double probs[kernels::kRowSummaryMaxK]{};

    double Prob(int32_t id) const {
        for (int j = 0; j < n; ++j) {
            if (ids[j] == id)
                return probs[j];
        }
        return 0.0;
    }

    int32_t Sample(double u) const {
        if (n <= 0)
            return 0;
        double cum = 0.0;
        for (int j = 0; j < n; ++j) {
            cum += probs[j];
            if (u < cum)
                return ids[j];
        }
        return ids[n - 1];
    }
};

// Reconstruct normalized filtered probability distribution from a RowSummary.
// If force_close_topk is true (used for the proposal distribution Q), normalizes over the surviving
// top-K entries so Q is always a valid closed probability distribution (lossless for speculative
// sampling).
TopKDist BuildTopKDist(const kernels::RowSummary& s, const kernels::SampleParams& p,
                       bool force_close_topk) {
    TopKDist d;
    if (s.k <= 0 || s.vocab <= 0 || s.k > kernels::kRowSummaryMaxK)
        return d;
    const int K = s.k;
    const float max_s = s.vals[0] / p.temperature;
    double w[kernels::kRowSummaryMaxK];
    for (int j = 0; j < K; ++j) {
        w[j] = std::exp(static_cast<double>(s.vals[j] / p.temperature - max_s));
    }
    const bool need_topk = p.top_k > 0 && p.top_k < static_cast<int>(s.vocab);
    const bool need_topp = p.top_p < 1.0f;
    const bool need_minp = p.min_p > 0.0f;

    int limit = K;
    bool closed = (static_cast<int64_t>(K) >= s.vocab);
    if (need_topk && p.top_k <= K) {
        limit = p.top_k;
        closed = true;
    }
    if (need_minp) {
        int m = 0;
        while (m < limit && w[m] >= static_cast<double>(p.min_p)) ++m;
        if (m < limit) {
            limit = std::max(1, m);
            closed = true;
        }
    }
    const double s_full =
        std::exp(static_cast<double>(s.lse) -
                 static_cast<double>(s.vals[0]) * static_cast<double>(s.inv_temperature));
    if (need_topp) {
        double sum_lim = 0.0;
        for (int j = 0; j < limit; ++j) sum_lim += w[j];
        const double denom = closed ? sum_lim : std::max(sum_lim, s_full);
        const double thr = static_cast<double>(p.top_p) * denom;
        double cum = 0.0;
        int keep = limit;
        for (int j = 0; j < limit; ++j) {
            cum += w[j];
            if (cum >= thr) {
                keep = j + 1;
                closed = true;
                break;
            }
        }
        limit = keep;
    }
    if (limit <= 0)
        limit = 1;
    double sum_vis = 0.0;
    for (int j = 0; j < limit; ++j) sum_vis += w[j];
    const bool is_closed = closed || force_close_topk;
    const double norm = is_closed ? sum_vis : std::max(sum_vis, s_full);
    d.n = limit;
    d.closed = is_closed;
    for (int j = 0; j < limit; ++j) {
        d.ids[j] = s.ids[j];
        d.probs[j] = (norm > 0.0) ? (w[j] / norm) : (j == 0 ? 1.0 : 0.0);
    }
    d.tail_mass = is_closed ? 0.0 : std::max(0.0, 1.0 - (norm > 0.0 ? sum_vis / norm : 1.0));
    return d;
}

bool MoeGroupedDecodeEnabled() {
    static const bool kEnabled = InkEnvFlag("R4DX_INK_MOE_DECODE", true);
    return kEnabled;
}

bool MoeGroupedPrefillEnabled() {
    static const bool kEnabled = InkEnvFlag("R4DX_INK_MOE_PREFILL", true);
    return kEnabled;
}

bool MoeCheckOffsetsEnabled() {
    static const bool kEnabled = InkEnvFlag("R4DX_INK_MOE_CHECK_OFFSETS", false);
    return kEnabled;
}

bool AttnFusedEnabled() {
    static const bool kEnabled = InkEnvFlag("R4DX_INK_ATTN_FUSED", true);
    return kEnabled;
}

bool DecodeGraphEnabled() {
    static const bool kEnabled = InkEnvFlag("R4DX_INK_DECODE_GRAPH", true);
    return kEnabled;
}

template <class Record>
bool CaptureDecodeGraph(hipStream_t s, hipGraphExec_t* exec, Record&& record) {
    if (hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal) != hipSuccess)
        return false;
    struct Guard {
        hipStream_t stream;
        bool capturing = true;
        hipGraph_t graph = nullptr;
        ~Guard() {
            if (capturing)
                static_cast<void>(hipStreamEndCapture(stream, &graph));
            if (graph != nullptr)
                static_cast<void>(hipGraphDestroy(graph));
        }
    } guard{s};
    record();
    guard.capturing = false;
    if (hipStreamEndCapture(s, &guard.graph) != hipSuccess)
        return false;
    return hipGraphInstantiate(exec, guard.graph, nullptr, nullptr, 0) == hipSuccess;
}

float SummaryInvTemperature(const kernels::SampleParams& params) {
    if (!(params.temperature > 0.0f) || params.temperature < kMinSummaryTemperature)
        return 0.0f;
    const float inv = 1.0f / params.temperature;
    if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity()))
        return 0.0f;
    return inv;
}

}  // namespace

void Inkling2Model::GraphExecDeleter::operator()(void* p) const noexcept {
    if (p != nullptr)
        static_cast<void>(hipGraphExecDestroy(static_cast<hipGraphExec_t>(p)));
}

// The per-chunk fp32 state: indices into pool_ (see the header comment).
struct Inkling2Model::Chunk {
    std::vector<bool> used;
    int h = -1, h_pre = -1, prev_h = -1;
    std::vector<int> B, B_pre, prev_B;
    size_t n_pre = 0;
    int Acquire() {
        for (size_t i = 0; i < used.size(); ++i) {
            if (!used[i]) {
                used[i] = true;
                return static_cast<int>(i);
            }
        }
        throw std::logic_error(
            "Inkling2Model: the fp32 state pool is exhausted (a buffer was not released)");
    }
    void Release(int i) {
        if (i >= 0)
            used.at(static_cast<size_t>(i)) = false;
    }
};

// ---- load
// -----------------------------------------------------------------------------------------------------------

Inkling2Model Inkling2Model::Load(const Inkling2ModelOptions& opts) {
    if (opts.prefill_chunk < 1 || opts.prefill_chunk > 256)
        throw std::invalid_argument("Inkling2Model::Load: prefill_chunk must be in [1, 256]");
    const bool is_tp = opts.tp_world > 1;
    if (opts.tp_world < 1 || opts.tp_world > 2 || opts.tp_rank < 0 || opts.tp_rank >= opts.tp_world)
        throw std::invalid_argument(
            "Inkling2Model::Load: tp_world must be 1 or 2 and 0 <= tp_rank < tp_world");
    if (!is_tp && opts.tp_comm != nullptr)
        throw std::invalid_argument("Inkling2Model::Load: tp_comm needs tp_world > 1");
    if (is_tp && (opts.tp_comm == nullptr || opts.tp_comm->World() != opts.tp_world ||
                  opts.tp_comm->Rank() != opts.tp_rank))
        throw std::invalid_argument(
            "Inkling2Model::Load: tp_world > 1 needs a TpComm endpoint of the same world and rank");
    SetTp2TuningForThisThread(is_tp);
    struct Tp2FlagOnThrow {
        int uncaught = std::uncaught_exceptions();
        ~Tp2FlagOnThrow() {
            if (std::uncaught_exceptions() > uncaught)
                SetTp2TuningForThisThread(false);
        }
    } tp2_flag_on_throw;

    Inkling2Model m;
    m.opts_ = opts;
    m.comm_ = opts.tp_comm;
    Inkling2LoadOptions lo;
    lo.layout = opts.layout;
    lo.tp_world = opts.tp_world;
    lo.tp_rank = opts.tp_rank;
    m.container_ = Inkling2Container::Load(opts.container_path, lo);
    const Inkling2Config& cfg = m.container_.Config();
    const int64_t H = cfg.hidden_size;
    m.max_ctx_ = Inkling2Config::ResolveMaxCtx(opts.max_ctx);
    m.max_chunk_ = opts.prefill_chunk;
    m.visits_ = cfg.Visits();
    const int64_t rows = m.max_chunk_;

    m.pool_.reserve(kPoolBuffers);
    m.pool_inv_rms_.reserve(kPoolBuffers);
    for (int i = 0; i < kPoolBuffers; ++i) {
        m.pool_.emplace_back(static_cast<size_t>(rows * H));
        m.pool_inv_rms_.emplace_back(static_cast<size_t>(rows));
    }
    m.msg_ = core::DeviceBuffer<float>(static_cast<size_t>(4 * rows * H));

    // KV caches + block tables.
    const int64_t bs = m.kv_block_;
    m.full_blocks_ = (m.max_ctx_ + bs - 1) / bs;
    m.ring_blocks_ =
        std::min<int64_t>(m.full_blocks_, (cfg.sliding_window + rows - 1 + bs - 1) / bs + 1);
    {
        std::vector<int32_t> full(static_cast<size_t>(m.full_blocks_)),
            ring(static_cast<size_t>(m.full_blocks_));
        for (int64_t b = 0; b < m.full_blocks_; ++b) {
            full[static_cast<size_t>(b)] = static_cast<int32_t>(b);
            ring[static_cast<size_t>(b)] = static_cast<int32_t>(b % m.ring_blocks_);
        }
        m.table_full_ = core::DeviceBuffer<int32_t>(full.size());
        m.table_full_.CopyFromHost(full);
        m.table_ring_ = core::DeviceBuffer<int32_t>(ring.size());
        m.table_ring_.CopyFromHost(ring);
    }
    const int64_t slots = cfg.NumCacheSlots();
    std::vector<int> slot_layer(static_cast<size_t>(slots), -1);
    for (const Inkling2Visit& v : m.visits_)
        slot_layer.at(static_cast<size_t>(v.cache_slot)) = v.layer;
    const int64_t kvl = cfg.num_key_value_heads, hd = cfg.head_dim;
    const int64_t block_elems = kvl * bs * 2 * hd;
    const bool want_kv_fp8 =
        InkEnvFlag("R4DX_INK_KV_FP8", false) && AttnFusedEnabled() && hd == 128;
    m.kv_fp8_enabled_ = want_kv_fp8;
    m.kv_.resize(static_cast<size_t>(slots));
    m.kv_fp8_.resize(static_cast<size_t>(slots));
    m.kv_ckpt_.resize(static_cast<size_t>(slots));
    int64_t conv_total = 0;
    int64_t conv_hist_total = 0;
    constexpr int64_t kVerifyMaxT = 8;
    m.conv_off_.assign(static_cast<size_t>(slots * 4), 0);
    m.conv_hist_off_.assign(static_cast<size_t>(slots * 4), 0);
    m.conv_ch_.assign(static_cast<size_t>(slots * 4), 0);
    for (int64_t s = 0; s < slots; ++s) {
        const int layer = slot_layer[static_cast<size_t>(s)];
        if (layer < 0)
            throw std::logic_error("Inkling2Model::Load: cache slot without a visit");
        const bool full = cfg.IsFull(layer);
        const int64_t nblk = full ? m.full_blocks_ : m.ring_blocks_;
        if (full && want_kv_fp8) {
            m.kv_fp8_[static_cast<size_t>(s)] =
                core::DeviceBuffer<uint8_t>(static_cast<size_t>(nblk * block_elems));
            m.kv_bytes_ += static_cast<int64_t>(m.kv_fp8_[static_cast<size_t>(s)].bytes());
        } else {
            m.kv_[static_cast<size_t>(s)] =
                core::DeviceBuffer<uint16_t>(static_cast<size_t>(nblk * block_elems));
            m.kv_bytes_ += static_cast<int64_t>(m.kv_[static_cast<size_t>(s)].bytes());
        }
        if (opts.prompt_checkpoint && !full)
            m.kv_ckpt_[static_cast<size_t>(s)] =
                core::DeviceBuffer<uint16_t>(static_cast<size_t>(nblk * block_elems));
        const int64_t C[4] = {kvl * hd, kvl * hd, H, H};
        for (int w = 0; w < 4; ++w) {
            const size_t idx = static_cast<size_t>(s * 4 + w);
            m.conv_off_[idx] = conv_total;
            m.conv_hist_off_[idx] = conv_hist_total;
            m.conv_ch_[idx] = C[w];
            conv_total += 3 * C[w];
            conv_hist_total += kVerifyMaxT * C[w];
        }
    }
    m.conv_ = core::DeviceBuffer<float>(static_cast<size_t>(conv_total));
    m.conv_.Zero();
    if (opts.prompt_checkpoint) {
        m.conv_ckpt_ = core::DeviceBuffer<float>(static_cast<size_t>(conv_total));
        m.conv_ckpt_.Zero();
    }
    if (opts.enable_vision && opts.tp_rank == 0) {
        m.vision_ = Inkling2VisionTower::TryLoad(opts.container_path);
    }
    if (!opts.dspark_path.empty()) {
        const std::string dp = (opts.dspark_path == "auto" || opts.dspark_path == "self")
                                   ? opts.container_path
                                   : opts.dspark_path;
        m.dspark_ =
            std::make_unique<Inkling2DsparkDraft>(Inkling2DsparkDraft::Load(dp, m.max_ctx_));
        for (int j = 0; j < 4; ++j)
            m.dspark_taps_[static_cast<size_t>(j)] =
                core::DeviceBuffer<uint16_t>(static_cast<size_t>(rows * H));
        m.conv_verify_bak_ = core::DeviceBuffer<float>(static_cast<size_t>(conv_total));
        m.conv_verify_hist_ = core::DeviceBuffer<float>(static_cast<size_t>(conv_hist_total));
        m.conv_off_dev_ = core::DeviceBuffer<int64_t>(m.conv_off_.size());
        m.conv_off_dev_.CopyFromHost(m.conv_off_);
        m.conv_hist_off_dev_ = core::DeviceBuffer<int64_t>(m.conv_hist_off_.size());
        m.conv_hist_off_dev_.CopyFromHost(m.conv_hist_off_);
        std::vector<int32_t> conv_ch32(m.conv_ch_.size());
        m.conv_max_ch_ = 0;
        for (size_t i = 0; i < m.conv_ch_.size(); ++i) {
            conv_ch32[i] = static_cast<int32_t>(m.conv_ch_[i]);
            if (conv_ch32[i] > m.conv_max_ch_)
                m.conv_max_ch_ = conv_ch32[i];
        }
        m.conv_ch_dev_ = core::DeviceBuffer<int32_t>(conv_ch32.size());
        m.conv_ch_dev_.CopyFromHost(conv_ch32);
        m.dspark_argmax_ws_ = core::DeviceBuffer<uint8_t>(256 * (sizeof(float) + sizeof(int32_t)));
        m.verify_logits_dev_ =
            core::DeviceBuffer<float>(static_cast<size_t>(kVerifyMaxT * cfg.HeadRowsLocal()));
        m.verify_argmax_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(kVerifyMaxT));
        m.verify_argmax_pair_dev_ =
            core::DeviceBuffer<int32_t>(static_cast<size_t>(kVerifyMaxT * 2));
        m.verify_summary_ids_dev_ =
            core::DeviceBuffer<int32_t>(static_cast<size_t>(kVerifyMaxT * R4DX_TOPK_LSE_K));
        m.verify_summary_vals_dev_ =
            core::DeviceBuffer<float>(static_cast<size_t>(kVerifyMaxT * R4DX_TOPK_LSE_K));
        m.verify_summary_lse_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(kVerifyMaxT));
    }

    m.moe_ids_host_ =
        core::PinnedBuffer<int32_t>(static_cast<size_t>(rows * cfg.num_experts_per_tok));
    m.moe_off_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(cfg.num_experts + 1));
    m.ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows));
    m.ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(rows));
    m.pos_host_ = core::PinnedBuffer<int32_t>(1);
    m.pos_dev_ = core::DeviceBuffer<int32_t>(1);
    {
        const int64_t hl = cfg.num_attention_heads;
        const int64_t gqa = (kvl > 0) ? (hl / kvl) : 0;
        const bool tp_graph_ok =
            !is_tp || (opts.tp_comm != nullptr && opts.tp_comm->IsGraphCapturable());
        bool all_grouped_mx =
            tp_graph_ok && DecodeGraphEnabled() && MoeGroupedDecodeEnabled() &&
            AttnFusedEnabled() && hd == 128 && cfg.d_rel <= 32 && kvl > 0 && hl % kvl == 0 &&
            (gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 6 || gqa == 8);
        for (int64_t l = 0; all_grouped_mx && l < cfg.num_hidden_layers; ++l) {
            const Inkling2LayerWeights& lw = m.container_.Layer(l);
            const bool mx_ok = !lw.gate_up.empty() && !lw.down.empty() &&
                               lw.gate_up[0].layout == Layout::kMxfp8 &&
                               lw.down[0].layout == Layout::kMxfp8;
            if (lw.gate_up_ptrs0.empty() || lw.gate_up_ptrs1.empty() || lw.down_ptrs0.empty() ||
                lw.down_ptrs1.empty() || (!lw.nvfp4_experts && !mx_ok)) {
                all_grouped_mx = false;
            }
        }
        m.decode_graph_eligible_ = all_grouped_mx;
    }
    const int64_t vl = cfg.HeadRowsLocal();
    m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(vl));
    m.lg_bf16_dev_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(kVerifyMaxT * vl));
    m.softcap_argmax_ws_ = core::DeviceBuffer<uint8_t>(static_cast<size_t>(kVerifyMaxT) * 1024);
    m.vocab_local_valid_ =
        std::max<int64_t>(0, std::min<int64_t>(vl, cfg.unpadded_vocab_size - cfg.HeadRowBegin()));
    m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
    m.argmax_pair_dev_ = core::DeviceBuffer<int32_t>(2);
    m.argmax_host_ = core::PinnedBuffer<int32_t>(1);
    m.argmax_pair_host_ = core::PinnedBuffer<int32_t>(2);
    m.verify_argmax_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(kVerifyMaxT));
    m.verify_pairs_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(kVerifyMaxT * 2));
    m.summary_ids_dev_ = core::DeviceBuffer<int32_t>(R4DX_TOPK_LSE_K);
    m.summary_vals_dev_ = core::DeviceBuffer<float>(R4DX_TOPK_LSE_K);
    m.summary_lse_dev_ = core::DeviceBuffer<float>(1);
    m.summary_ids_host_ = core::PinnedBuffer<int32_t>(R4DX_TOPK_LSE_K);
    m.summary_vals_host_ = core::PinnedBuffer<float>(R4DX_TOPK_LSE_K);
    m.summary_lse_host_ = core::PinnedBuffer<float>(1);
    m.verify_summary_ids_host_ =
        core::PinnedBuffer<int32_t>(static_cast<size_t>(kVerifyMaxT * R4DX_TOPK_LSE_K));
    m.verify_summary_vals_host_ =
        core::PinnedBuffer<float>(static_cast<size_t>(kVerifyMaxT * R4DX_TOPK_LSE_K));
    m.verify_summary_lse_host_ = core::PinnedBuffer<float>(static_cast<size_t>(kVerifyMaxT));
    R4DX_HIP_CHECK(hipEventCreateWithFlags(&m.step_done_.e, hipEventDisableTiming));
    if (is_tp)
        m.topk_lse_ws_ =
            core::DeviceBuffer<uint8_t>(static_cast<size_t>(r4dx_topk_lse_workspace_bytes()));
    // The widest visit: the relative bias [T, heads, extent] fp32 (25 MiB at T 256), the MoE gather
    // / slot rows (2 x T * top_k * H bf16 = 19 MiB), the attention inputs and temporaries -- well
    // inside 256 MiB.
    m.arena_.Reserve(256ull * 1024 * 1024);
    if (m.dspark_) {
        m.dspark_->InitFoldedLmHead(vl, m.stream_, m.arena_,
                                    [&](const uint16_t* h_bf16, int64_t M, uint16_t* out_bf16) {
                                        m.Lin(m.container_.LmHead(), h_bf16, out_bf16, M);
                                    });
    }
    m.stream_.Synchronize();

    int nvfp4_layers = 0;
    for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
        if (m.container_.Layer(l).nvfp4_experts)
            ++nvfp4_layers;
    }
    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::cerr << "[r4dx::model::Inkling2Model] " << cfg.num_hidden_layers << " stored layers, "
              << m.visits_.size() << " visits, max_ctx " << m.max_ctx_ << ", KV "
              << (m.kv_fp8_enabled_ ? "fp8(full)/bf16(local)" : "bf16") << " = "
              << static_cast<double>(m.kv_bytes_) / gib << " GiB (sliding ring "
              << m.ring_blocks_ * bs << " positions), conv states "
              << static_cast<double>(m.conv_.bytes()) / (1024.0 * 1024.0) << " MiB, layout "
              << LayoutName(opts.layout) << ", residual fp32, attention "
              << (AttnFusedEnabled() ? "fused h128 (on-the-fly bias + sink)"
                                     : "reference (bias + sink)")
              << ", MoE "
              << (nvfp4_layers > 0 ? ("grouped decode (nvfp4 " + std::to_string(nvfp4_layers) +
                                      "/" + std::to_string(cfg.num_hidden_layers) + ")")
                                   : (MoeGroupedDecodeEnabled() ? "grouped decode (mxfp8)"
                                                                : "staged (host sync per visit)"))
              << (m.decode_graph_eligible_ ? ", decode graph on" : "");
    if (is_tp)
        std::cerr << ", TP rank " << opts.tp_rank << "/" << opts.tp_world << " (head rows " << vl
                  << ")";
    std::cerr << "\n";
    return m;
}

// ---- sequence state
// -------------------------------------------------------------------------------------------------

void Inkling2Model::Reset() {
    conv_.ZeroAsync(stream_);
    stream_.Synchronize();
    pos_ = 0;
    dspark_adapt_k_ = 3;
    started_ = false;
    arena_.Reset();
    ckpt_pos_ = -1;
    if (dspark_)
        dspark_->Reset();
}

void Inkling2Model::SaveCheckpoint() {
    if (!opts_.prompt_checkpoint)
        throw std::runtime_error("Inkling2Model::SaveCheckpoint: loaded without prompt_checkpoint");
    if (!started_)
        throw std::runtime_error("Inkling2Model::SaveCheckpoint: nothing has been fed yet");
    stream_.Synchronize();
    R4DX_HIP_CHECK(hipMemcpyAsync(conv_ckpt_.data(), conv_.data(), conv_.bytes(),
                                  hipMemcpyDeviceToDevice, stream_.get()));
    for (size_t s = 0; s < kv_.size(); ++s) {
        if (kv_ckpt_[s].empty())
            continue;
        R4DX_HIP_CHECK(hipMemcpyAsync(kv_ckpt_[s].data(), kv_[s].data(), kv_[s].bytes(),
                                      hipMemcpyDeviceToDevice, stream_.get()));
    }
    if (dspark_)
        dspark_->SaveCheckpoint(stream_);
    stream_.Synchronize();
    ckpt_pos_ = pos_;
}

void Inkling2Model::RestoreCheckpoint() {
    if (!opts_.prompt_checkpoint)
        throw std::runtime_error(
            "Inkling2Model::RestoreCheckpoint: loaded without prompt_checkpoint");
    if (ckpt_pos_ < 0)
        throw std::runtime_error(
            "Inkling2Model::RestoreCheckpoint: no checkpoint since Load()/Reset()");
    stream_.Synchronize();
    R4DX_HIP_CHECK(hipMemcpyAsync(conv_.data(), conv_ckpt_.data(), conv_.bytes(),
                                  hipMemcpyDeviceToDevice, stream_.get()));
    for (size_t s = 0; s < kv_.size(); ++s) {
        if (kv_ckpt_[s].empty())
            continue;
        R4DX_HIP_CHECK(hipMemcpyAsync(kv_[s].data(), kv_ckpt_[s].data(), kv_[s].bytes(),
                                      hipMemcpyDeviceToDevice, stream_.get()));
    }
    if (dspark_)
        dspark_->RestoreCheckpoint(stream_);
    stream_.Synchronize();
    pos_ = ckpt_pos_;
    started_ = true;
    arena_.Reset();
}

float* Inkling2Model::ConvState(int slot, int which) {
    return conv_.data() + conv_off_.at(static_cast<size_t>(slot) * 4 + static_cast<size_t>(which));
}

float* Inkling2Model::ConvHist(int slot, int which) {
    return conv_verify_hist_.data() +
           conv_hist_off_.at(static_cast<size_t>(slot) * 4 + static_cast<size_t>(which));
}

void Inkling2Model::SetRouterTrace(bool on) {
    RequireNotTp("SetRouterTrace");
    trace_.enabled = on;
    trace_.ids.clear();
}

// ---- building blocks
// ------------------------------------------------------------------------------------------------

void Inkling2Model::Lin(const QuantLinear& w, const uint16_t* x, uint16_t* y, int64_t M) {
    // The tiled bf16 GEMM takes K % 64 and N % 16; the tiny fixture's experts (K 32) and any odd
    // shape take the plain dense kernel instead (exact fp32 sums, one bf16 rounding, like the
    // GEMM).
    if (w.layout == Layout::kBf16 && (w.K % 64 != 0 || w.N % 16 != 0)) {
        r4dx_inkling2_dense_bf16(P(x), w.K, P(w.bf16_w.data()), P(y), 1, w.N, static_cast<int>(M),
                                 static_cast<int>(w.N), static_cast<int>(w.K), P(stream_.get()));
        return;
    }
    ApplyLinear(stream_.get(), arena_, w, x, y, M);
}

void Inkling2Model::MsgGate(const Inkling2MsgGateWeights& g, const float* x, const float* m,
                            float* out, int64_t T) {
    const Inkling2Config& cfg = container_.Config();
    const int64_t H = cfg.hidden_size, R = cfg.gate_rank;
    const int64_t s = P(stream_.get());
    uint16_t* cat = arena_.Alloc<uint16_t>(static_cast<size_t>(T * 2 * H), 16);
    float* gv = arena_.Alloc<float>(static_cast<size_t>(T * R), 16);
    const float eps = static_cast<float>(cfg.rms_norm_eps);
    r4dx_inkling2_msg_gate_prep_bf16(P(x), P(m), P(cat), static_cast<int>(T), static_cast<int>(H),
                                     eps, s);
    r4dx_inkling2_dense_bf16(P(cat), 2 * H, P(g.down.data()), P(gv), 0, R, static_cast<int>(T),
                             static_cast<int>(R), static_cast<int>(2 * H), s);
    r4dx_inkling2_msg_gate_apply_bf16w_f32(P(x), P(m), P(gv), 0, P(g.forget.data()), P(g.gain.data()),
                                           P(out), static_cast<int>(T), static_cast<int>(H),
                                           static_cast<int>(R), s);
}

void Inkling2Model::MsgGate3AttnNormBf16(const Inkling2LayerWeights& w, const float* a_src,
                                         const float* m_all, uint16_t* xq, uint16_t* xk,
                                         uint16_t* xv, int64_t T) {
    const Inkling2Config& cfg = container_.Config();
    const int64_t H = cfg.hidden_size, R = cfg.gate_rank;
    const int64_t s = P(stream_.get());
    const float eps = static_cast<float>(cfg.rms_norm_eps);
    const int64_t x_stride = (w.full && w.n_attn_queries == 3) ? (T * H) : 0;
    uint16_t* cat = arena_.Alloc<uint16_t>(static_cast<size_t>(3 * T * 2 * H), 16);
    float* gv = arena_.Alloc<float>(static_cast<size_t>(3 * T * R), 16);
    float* tmp = arena_.Alloc<float>(static_cast<size_t>(3 * T * H), 16);
    r4dx_inkling2_msg_gate_prep3_bf16(P(a_src), x_stride, P(m_all), P(cat), static_cast<int>(T),
                                      static_cast<int>(H), eps, s);
    r4dx_inkling2_dense3_bf16(P(cat), P(w.msg[0].down.data()), P(w.msg[1].down.data()),
                              P(w.msg[2].down.data()), P(gv), static_cast<int>(T),
                              static_cast<int>(R), static_cast<int>(2 * H), s);
    r4dx_inkling2_msg_gate_apply3_bf16w_f32(
        P(a_src), x_stride, P(m_all), P(gv), P(w.msg[0].forget.data()), P(w.msg[1].forget.data()),
        P(w.msg[2].forget.data()), P(w.msg[0].gain.data()), P(w.msg[1].gain.data()),
        P(w.msg[2].gain.data()), P(tmp), static_cast<int>(T), static_cast<int>(H),
        static_cast<int>(R), s);
    const uint16_t* nw0 = w.full ? w.q_norm.data() : w.attn_norm.data();
    const uint16_t* nw1 = w.full ? w.k_norm.data() : w.attn_norm.data();
    const uint16_t* nw2 = w.full ? w.v_norm.data() : w.attn_norm.data();
    r4dx_inkling2_rmsnorm3_f32in_bf16(P(tmp), P(nw0), P(nw1), P(nw2), P(xq), P(xk), P(xv),
                                      static_cast<int>(T), static_cast<int>(H), eps, s);
}

void Inkling2Model::Attention(const Inkling2Visit& v, const uint16_t* xq, const uint16_t* xk,
                              const uint16_t* xv, uint16_t* out, int64_t T, SpanAccumulator* prof,
                              bool use_pos_dev, bool graph_long_ctx, bool capture_conv_hist,
                              float* out_f32) {
    const Inkling2Config& cfg = container_.Config();
    const Inkling2Config& g = container_.GlobalConfig();
    const Inkling2LayerWeights& w = container_.Layer(v.layer);
    const Inkling2AttnWeights& a = w.attn;
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t H = cfg.hidden_size, hd = cfg.head_dim, hl = cfg.num_attention_heads,
                  kvl = cfg.num_key_value_heads;
    const int64_t C = kvl * hd, drel = cfg.d_rel, ext = cfg.RelExtent(v.layer);
    const bool full = w.full;
    const int64_t pos0 = pos_;
    const float log_alpha = full ? static_cast<float>(cfg.log_alpha) : 0.0f;
    const float log_floor = static_cast<float>(cfg.log_floor);
    const int slot = v.cache_slot;
    const int bs = kv_block_;
    const int32_t* table = full ? table_full_.data() : table_ring_.data();
    const bool slot_fp8 = full && kv_fp8_enabled_ && !kv_fp8_.at(static_cast<size_t>(slot)).empty();
    const int64_t kv_ptr = slot_fp8 ? P(kv_fp8_.at(static_cast<size_t>(slot)).data())
                                    : P(kv_.at(static_cast<size_t>(slot)).data());

    // Helper to run an MXFP8 linear from a contiguous 16-row-tile slice of a.qgrkv_{wq,ws}.
    const auto lin_slice = [&](int64_t row0, int64_t N_slice, const uint16_t* x_in,
                               uint16_t* y_out) {
        constexpr int64_t kMxfp8ChunkM = 256;
        const uint8_t* wq_p = a.qgrkv_wq.data() + static_cast<size_t>(row0 * H);
        const uint8_t* ws_p = a.qgrkv_ws.data() + static_cast<size_t>(row0 * (H / 32));
        for (int64_t m0 = 0; m0 < T; m0 += kMxfp8ChunkM) {
            const int m = static_cast<int>(std::min(kMxfp8ChunkM, T - m0));
            const LinearTuning t = PickTuning(Layout::kMxfp8, N_slice, H, std::min<int64_t>(m, 64));
            r4dx_gemm_mxfp8_nt(P(x_in + m0 * H), P(wq_p), P(ws_p), P(y_out + m0 * N_slice), m,
                               static_cast<int>(H), static_cast<int>(N_slice), t.WV, t.SK, t.MB,
                               t.NPW, t.NT, s);
        }
    };
    const bool has_concat = !a.qgrkv_wq.empty();
    const bool concat_all5 = has_concat && xk == xq && xv == xq;
    const bool concat_qgr = has_concat && !concat_all5;
    const int64_t gqa = hl / kvl;
    const bool use_fused_attn =
        AttnFusedEnabled() && hd == 128 && drel <= 32 && hl % kvl == 0 &&
        (gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 6 || gqa == 8);
    const bool strided_prefill =
        use_fused_attn && hd == 128 && T > 16 && StridedPrefillQkvEnabled();

    const int64_t rel_all = g.num_attention_heads * drel;
    uint16_t* qlin = nullptr;
    uint16_t* klin = nullptr;
    uint16_t* vlin = nullptr;
    uint16_t* rlin = nullptr;
    uint16_t* gt = nullptr;
    int64_t q_in_stride = hl * hd;
    int64_t g_in_stride = hl * hd;
    int64_t r_in_stride = rel_all;
    int64_t kv_in_stride = C;
    if (concat_all5) {
        uint16_t* all = arena_.Alloc<uint16_t>(static_cast<size_t>(T * a.qgrkv_n), 16);
        ProfiledCall(prof, sr, "gemm:attn.q", [&] { lin_slice(0, a.qgrkv_n, xq, all); });
        if (T == 1 || strided_prefill) {
            qlin = all + a.q_row0;
            gt = all + a.gate_row0;
            rlin = all + a.rel_row0;
            klin = all + a.k_row0;
            vlin = all + a.v_row0;
            if (T > 1) {
                q_in_stride = a.qgrkv_n;
                g_in_stride = a.qgrkv_n;
                r_in_stride = a.qgrkv_n;
                kv_in_stride = a.qgrkv_n;
            }
        } else {
            qlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
            gt = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
            rlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * rel_all), 16);
            klin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
            vlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
            r4dx_inkling2_qgrkv_split_bf16(P(all), a.qgrkv_n, P(qlin), a.q_row0, hl * hd, P(gt),
                                           a.gate_row0, hl * hd, P(rlin), a.rel_row0, rel_all,
                                           P(klin), a.k_row0, C, P(vlin), a.v_row0, C,
                                           static_cast<int>(T), s);
        }
    } else if (concat_qgr) {
        uint16_t* qgr = arena_.Alloc<uint16_t>(static_cast<size_t>(T * a.qgr_n), 16);
        ProfiledCall(prof, sr, "gemm:attn.q", [&] { lin_slice(0, a.qgr_n, xq, qgr); });
        if (T == 1 || strided_prefill) {
            qlin = qgr + a.q_row0;
            gt = qgr + a.gate_row0;
            rlin = qgr + a.rel_row0;
            if (T > 1) {
                q_in_stride = a.qgr_n;
                g_in_stride = a.qgr_n;
                r_in_stride = a.qgr_n;
            }
        } else {
            qlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
            gt = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
            rlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * rel_all), 16);
            r4dx_inkling2_qgrkv_split_bf16(P(qgr), a.qgr_n, P(qlin), a.q_row0, hl * hd, P(gt),
                                           a.gate_row0, hl * hd, P(rlin), a.rel_row0, rel_all, 0, 0,
                                           0, 0, 0, 0, static_cast<int>(T), s);
        }
        klin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
        vlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
        ProfiledCall(prof, sr, "gemm:attn.kv", [&] {
            lin_slice(a.k_row0, C, xk, klin);
            lin_slice(a.v_row0, C, xv, vlin);
        });
    } else {
        qlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
        klin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
        vlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
        rlin = arena_.Alloc<uint16_t>(static_cast<size_t>(T * rel_all), 16);
        gt = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
        ProfiledCall(prof, sr, "gemm:attn.q", [&] {
            if (has_concat)
                lin_slice(a.q_row0, hl * hd, xq, qlin);
            else
                Lin(a.q, xq, qlin, T);
        });
        ProfiledCall(prof, sr, "gemm:attn.kv", [&] {
            if (has_concat) {
                lin_slice(a.k_row0, C, xk, klin);
                lin_slice(a.v_row0, C, xv, vlin);
            } else {
                Lin(a.k, xk, klin, T);
                Lin(a.v, xv, vlin, T);
            }
        });
    }

    const int64_t pos_dev_ptr = use_pos_dev ? P(pos_dev_.data()) : 0;
    const bool batch_sink_kv = full && hd == 128 && T <= 16 && BatchSinkKvEnabled();
    uint16_t* q = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
    float* r = arena_.Alloc<float>(static_cast<size_t>(T * hl * drel), 16);
    float* rel =
        use_fused_attn ? nullptr : arena_.Alloc<float>(static_cast<size_t>(T * hl * ext), 16);
    float* sink = arena_.Alloc<float>(static_cast<size_t>(T * hl), 16);
    // q + r head_clip (fused into 1 launch when hd <= 128 && drel <= 128)
    ProfiledCall(prof, sr, "attn.rel_sink", [&] {
        if (!concat_all5 && !concat_qgr) {
            if (has_concat)
                lin_slice(a.rel_row0, rel_all, xq, rlin);
            else
                Lin(a.rel, xq, rlin, T);
        }
        if (hd <= 128 && drel <= 128) {
            r4dx_inkling2_head_clip_qr(P(qlin), q_in_stride, P(a.q_scale.data()),
                                       static_cast<float>(cfg.qk_clip), P(q), static_cast<int>(hd),
                                       P(rlin + cfg.tp_rank * hl * drel), r_in_stride,
                                       P(a.rel_scale.data()), static_cast<float>(cfg.rel_clip),
                                       P(r), static_cast<int>(drel), pos0, pos_dev_ptr, log_alpha,
                                       log_floor, static_cast<int>(T), static_cast<int>(hl), s);
        } else {
            R4dxInkling2HeadOut dense_q{};
            dense_q.kind = 0;
            dense_q.out_bf16 = 1;
            dense_q.out = P(q);
            dense_q.out_row_stride = hl * hd;
            dense_q.pos_dev = pos_dev_ptr;
            r4dx_inkling2_head_clip(P(qlin), 1, q_in_stride, P(a.q_scale.data()),
                                    static_cast<float>(cfg.qk_clip), pos0, log_alpha, log_floor,
                                    &dense_q, static_cast<int>(T), static_cast<int>(hl),
                                    static_cast<int>(hd), s);
            R4dxInkling2HeadOut dense_r{};
            dense_r.kind = 0;
            dense_r.out_bf16 = 0;
            dense_r.out = P(r);
            dense_r.out_row_stride = hl * drel;
            dense_r.pos_dev = pos_dev_ptr;
            r4dx_inkling2_head_clip(P(rlin + cfg.tp_rank * hl * drel), 1, r_in_stride,
                                    P(a.rel_scale.data()), static_cast<float>(cfg.rel_clip), pos0,
                                    log_alpha, log_floor, &dense_r, static_cast<int>(T),
                                    static_cast<int>(hl), static_cast<int>(drel), s);
        }
        if (!use_fused_attn) {
            r4dx_inkling2_rel_logits_f32(P(r), P(a.rel_proj.data()), P(rel),
                                         static_cast<int>(T * hl), static_cast<int>(drel),
                                         static_cast<int>(ext), s);
        }
        if (!batch_sink_kv) {
            r4dx_inkling2_dense_bf16(P(xq), H, P(a.sink.data()), P(sink), 0, hl, static_cast<int>(T),
                                     static_cast<int>(hl), static_cast<int>(H), s);
        }
    });
    // k, v: short conv -> clip / rms -> the paged cache
    ProfiledCall(prof, sr, "attn.kv_conv", [&] {
        const int64_t khist = capture_conv_hist ? P(ConvHist(slot, 0)) : 0;
        const int64_t vhist = capture_conv_hist ? P(ConvHist(slot, 1)) : 0;
        if (!full && !slot_fp8 && hd == 128 && T <= 16) {
            r4dx_inkling2_sconv_kv_paged_bf16(
                P(klin), P(vlin), P(a.k_conv.data()), P(a.v_conv.data()), P(ConvState(slot, 0)),
                P(ConvState(slot, 1)), khist, vhist, static_cast<float>(cfg.qk_clip),
                static_cast<float>(cfg.rms_norm_eps), pos0, pos_dev_ptr, kv_ptr, P(table), bs,
                kvl * bs * 2 * hd, static_cast<int64_t>(bs) * 2 * hd, static_cast<int>(T),
                static_cast<int>(kvl), s);
            return;
        }
        if (full && hd == 128 && T <= 16) {
            float* dk = arena_.Alloc<float>(static_cast<size_t>(T * kvl * 4), 16);
            float* dv = arena_.Alloc<float>(static_cast<size_t>(T * kvl * 4), 16);
            if (batch_sink_kv) {
                r4dx_inkling2_dense_sink_kvconv_bf16(
                    P(xq), P(xk), P(xv), P(a.sink.data()), P(a.k_conv_proj.data()),
                    P(a.v_conv_proj.data()), P(sink), P(dk), P(dv), static_cast<int>(T),
                    static_cast<int>(hl), static_cast<int>(kvl * 4), static_cast<int>(H), s);
            } else {
                r4dx_inkling2_dense2_bf16(P(xk), P(xv), P(a.k_conv_proj.data()),
                                          P(a.v_conv_proj.data()), P(dk), P(dv), static_cast<int>(T),
                                          static_cast<int>(kvl * 4), static_cast<int>(H), s);
            }
            r4dx_inkling2_sconv_dyn_kv_paged(
                P(klin), P(vlin), P(dk), P(dv), P(a.k_conv_static.data()),
                P(a.v_conv_static.data()), P(ConvState(slot, 0)), P(ConvState(slot, 1)), khist,
                vhist, static_cast<float>(cfg.qk_clip), static_cast<float>(cfg.rms_norm_eps), pos0,
                pos_dev_ptr, kv_ptr, slot_fp8 ? 1 : 0, P(table), bs, kvl * bs * 2 * hd,
                static_cast<int64_t>(bs) * 2 * hd, static_cast<int>(T), static_cast<int>(kvl), s);
            return;
        }
        float* kc = arena_.Alloc<float>(static_cast<size_t>(T * C), 16);
        float* vc = arena_.Alloc<float>(static_cast<size_t>(T * C), 16);
        if (full) {
            float* dk = arena_.Alloc<float>(static_cast<size_t>(T * kvl * 4), 16);
            float* dv = arena_.Alloc<float>(static_cast<size_t>(T * kvl * 4), 16);
            r4dx_inkling2_dense_bf16(P(xk), H, P(a.k_conv_proj.data()), P(dk), 0, kvl * 4,
                                     static_cast<int>(T), static_cast<int>(kvl * 4),
                                     static_cast<int>(H), s);
            r4dx_inkling2_dense_bf16(P(xv), H, P(a.v_conv_proj.data()), P(dv), 0, kvl * 4,
                                     static_cast<int>(T), static_cast<int>(kvl * 4),
                                     static_cast<int>(H), s);
            if (kv_in_stride != C) {
                r4dx_inkling2_sconv_dynamic_strided_hist_f32(
                    P(klin), kv_in_stride, 1, P(dk), 0, P(a.k_conv_static.data()),
                    P(ConvState(slot, 0)), P(kc), khist, static_cast<int>(T), static_cast<int>(C),
                    static_cast<int>(kvl), s);
                r4dx_inkling2_sconv_dynamic_strided_hist_f32(
                    P(vlin), kv_in_stride, 1, P(dv), 0, P(a.v_conv_static.data()),
                    P(ConvState(slot, 1)), P(vc), vhist, static_cast<int>(T), static_cast<int>(C),
                    static_cast<int>(kvl), s);
            } else {
                r4dx_inkling2_sconv_dynamic_hist_f32(
                    P(klin), 1, P(dk), 0, P(a.k_conv_static.data()), P(ConvState(slot, 0)), P(kc),
                    khist, static_cast<int>(T), static_cast<int>(C), static_cast<int>(kvl), s);
                r4dx_inkling2_sconv_dynamic_hist_f32(
                    P(vlin), 1, P(dv), 0, P(a.v_conv_static.data()), P(ConvState(slot, 1)), P(vc),
                    vhist, static_cast<int>(T), static_cast<int>(C), static_cast<int>(kvl), s);
            }
        } else if (kv_in_stride != C) {
            r4dx_inkling2_sconv_static_pair_strided_hist_f32(
                P(klin), kv_in_stride, P(vlin), kv_in_stride, 1, P(a.k_conv.data()),
                P(a.v_conv.data()), P(ConvState(slot, 0)), P(ConvState(slot, 1)), P(kc), P(vc),
                khist, vhist, static_cast<int>(T), static_cast<int>(C), s);
        } else {
            r4dx_inkling2_sconv_static_pair_hist_f32(
                P(klin), P(vlin), 1, P(a.k_conv.data()), P(a.v_conv.data()), P(ConvState(slot, 0)),
                P(ConvState(slot, 1)), P(kc), P(vc), khist, vhist, static_cast<int>(T),
                static_cast<int>(C), s);
        }
        R4dxInkling2HeadOut paged{};
        paged.kind = 1;
        paged.out = kv_ptr;
        paged.block_table = P(table);
        paged.block_size = bs;
        paged.kv_block_stride = kvl * bs * 2 * hd;
        paged.kv_head_stride = static_cast<int64_t>(bs) * 2 * hd;
        paged.pos_dev = pos_dev_ptr;
        paged.out_fp8 = slot_fp8 ? 1 : 0;
        paged.col_offset = 0;
        r4dx_inkling2_head_clip(P(kc), 0, C, 0, static_cast<float>(cfg.qk_clip), pos0, 0.0f, 1.0f,
                                &paged, static_cast<int>(T), static_cast<int>(kvl),
                                static_cast<int>(hd), s);
        paged.col_offset = static_cast<int>(hd);
        r4dx_inkling2_head_rms(P(vc), 0, C, static_cast<float>(cfg.rms_norm_eps), pos0, &paged,
                               static_cast<int>(T), static_cast<int>(kvl), static_cast<int>(hd), s);
    });
    // attention, gate, o
    const bool fused_gate = use_fused_attn && T <= 16 && (concat_all5 || concat_qgr);
    uint16_t* ao = arena_.Alloc<uint16_t>(static_cast<size_t>(T * hl * hd), 16);
    ProfiledCall(prof, sr, "attn.core", [&] {
        if (use_fused_attn) {
            constexpr int kMaxSplits = 32;
            float* attn_ws =
                (T <= 16)
                    ? arena_.Alloc<float>(static_cast<size_t>(T * hl * kMaxSplits * (hd + 2)), 16)
                    : nullptr;
            R4dxInkling2AttnFusedArgs af{};
            af.q = P(q);
            af.kv = kv_ptr;
            af.block_table = P(table);
            af.out = P(ao);
            af.r = P(r);
            af.rel_proj = P(a.rel_proj.data());
            af.sink = P(sink);
            af.workspace = P(attn_ws);
            af.q_len = static_cast<int>(T);
            af.q_heads = static_cast<int>(hl);
            af.kv_heads = static_cast<int>(kvl);
            af.head_dim = static_cast<int>(hd);
            af.block_size = bs;
            af.ctx = static_cast<int>(pos0 + T);
            af.window = static_cast<int>(cfg.Window(v.layer));
            af.d_rel = static_cast<int>(drel);
            af.rel_extent = static_cast<int>(ext);
            af.max_splits = (use_pos_dev && !graph_long_ctx) ? 1 : kMaxSplits;
            af.scale = 1.0f / std::sqrt(static_cast<float>(hd));
            af.kv_block_stride = kvl * bs * 2 * hd;
            af.kv_head_stride = static_cast<int64_t>(bs) * 2 * hd;
            af.pos_dev = pos_dev_ptr;
            af.kv_fp8 = slot_fp8 ? 1 : 0;
            af.gate = fused_gate ? P(gt) : 0;
            af.gate_row_stride = fused_gate ? g_in_stride : 0;
            const int rc = r4dx_inkling2_attn_fused_bf16kv(&af, s);
            if (rc != 0)
                throw std::runtime_error(
                    "Inkling2Model: r4dx_inkling2_attn_fused_bf16kv refused the launch (" +
                    std::to_string(rc) + ")");
        } else {
            R4dxInkling2AttnRefArgs ar{};
            ar.q = P(q);
            ar.kv = kv_ptr;
            ar.block_table = P(table);
            ar.out = P(ao);
            ar.rel = P(rel);
            ar.sink = P(sink);
            ar.q_len = static_cast<int>(T);
            ar.q_heads = static_cast<int>(hl);
            ar.kv_heads = static_cast<int>(kvl);
            ar.head_dim = static_cast<int>(hd);
            ar.block_size = bs;
            ar.ctx = static_cast<int>(pos0 + T);
            ar.window = static_cast<int>(cfg.Window(v.layer));
            ar.rel_extent = static_cast<int>(ext);
            ar.scale = 1.0f / std::sqrt(static_cast<float>(hd));
            ar.kv_block_stride = kvl * bs * 2 * hd;
            ar.kv_head_stride = static_cast<int64_t>(bs) * 2 * hd;
            const int rc = r4dx_inkling2_attn_ref_bf16kv(&ar, s);
            if (rc != 0)
                throw std::runtime_error(
                    "Inkling2Model: r4dx_inkling2_attn_ref_bf16kv refused the launch (" +
                    std::to_string(rc) + ")");
        }
    });
    ProfiledCall(prof, sr, "gemm:attn.gate_o", [&] {
        if (!concat_all5 && !concat_qgr) {
            if (has_concat)
                lin_slice(a.gate_row0, hl * hd, xq, gt);
            else
                Lin(a.gate, xq, gt, T);
        }
        if (!fused_gate) {
            if (g_in_stride != hl * hd) {
                r4dx_inkling2_sigmoid_gate_mul_strided_bf16(
                    P(ao), P(gt), g_in_stride, P(ao), static_cast<int>(T),
                    static_cast<int>(hl * hd), s);
            } else {
                r4dx_inkling2_sigmoid_gate_mul_bf16(P(ao), P(gt), P(ao), T * hl * hd, s);
            }
        }
        Lin(a.o, ao, out, T);
    });
    if (comm_ != nullptr) {
        ProfiledCall(prof, sr, "tp.allreduce", [&] {
            if (out_f32 != nullptr) {
                if (!(ArBf16ToF32Enabled() &&
                      comm_->AllReduceSumBf16ToF32Rows(out, out_f32, T, H, sr))) {
                    r4dx_inkling2_sconv_capture_x_f32(P(out), 1, P(out_f32), static_cast<int>(T),
                                                      static_cast<int>(H), s);
                    if (!comm_->AllReduceSumF32(out_f32, T * H, sr)) {
                        comm_->AllReduceSumBf16Rows(out, T, H, sr);
                        r4dx_inkling2_sconv_capture_x_f32(
                            P(out), 1, P(out_f32), static_cast<int>(T), static_cast<int>(H), s);
                    }
                }
            } else {
                comm_->AllReduceSumBf16Rows(out, T, H, sr);
            }
        });
    }
}

void Inkling2Model::Moe(int64_t layer, const uint16_t* x, uint16_t* out, int64_t T, int visit,
                        SpanAccumulator* prof, float* out_f32) {
    const Inkling2Config& cfg = container_.Config();
    const Inkling2LayerWeights& w = container_.Layer(layer);
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t H = cfg.hidden_size, E = cfg.num_experts, K = cfg.num_experts_per_tok,
                  I = cfg.moe_intermediate_size;
    const int64_t Pn = T * K;
    int32_t* ids = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
    float* wts = arena_.Alloc<float>(static_cast<size_t>(Pn), 16);
    uint16_t* ys = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);
    const bool use_fused_router = FuseRouterEnabled() && T <= 8;
    float* router_logits =
        (!use_fused_router && MoeGroupedDecodeEnabled()) || trace_.enabled
            ? arena_.Alloc<float>(static_cast<size_t>(T * E), 16)
            : nullptr;
    ProfiledCall(prof, sr, "moe.router", [&] {
        if (use_fused_router) {
            r4dx_inkling2_router_fused_f32(P(x), P(w.router.data()), P(w.bias.data()), P(ids), P(wts),
                                           static_cast<int>(T), static_cast<int>(H), static_cast<int>(E),
                                           static_cast<int>(K), static_cast<float>(cfg.route_scale), H,
                                           P(router_logits), s);
        } else {
            r4dx_inkling2_router_f32(P(x), P(w.router.data()), P(w.bias.data()), P(ids), P(wts),
                                     static_cast<int>(T), static_cast<int>(H), static_cast<int>(E),
                                     static_cast<int>(K), static_cast<float>(cfg.route_scale), H,
                                     P(router_logits), s);
        }
    });
    if (trace_.enabled) {
        std::vector<int32_t> h(static_cast<size_t>(Pn));
        R4DX_HIP_CHECK(
            hipMemcpyAsync(h.data(), ids, h.size() * sizeof(int32_t), hipMemcpyDeviceToHost, sr));
        R4DX_HIP_CHECK(hipStreamSynchronize(sr));
        if (static_cast<int>(trace_.ids.size()) <= visit)
            trace_.ids.resize(static_cast<size_t>(visit + 1));
        auto& dst = trace_.ids[static_cast<size_t>(visit)];
        dst.insert(dst.end(), h.begin(), h.end());
    }
    const bool grouped_nv4 =
        w.nvfp4_experts && !w.gate_up_ptrs0.empty() && !w.down_ptrs0.empty();
    const bool grouped_mx = !grouped_nv4 && MoeGroupedDecodeEnabled() &&
                            !w.gate_up_ptrs0.empty() && !w.gate_up_ptrs1.empty() &&
                            !w.down_ptrs0.empty() && !w.down_ptrs1.empty() &&
                            !w.gate_up.empty() && w.gate_up[0].layout == Layout::kMxfp8 &&
                            w.down[0].layout == Layout::kMxfp8;
    const bool use_moe_decode = (T == 1) || (T <= MoeDecodeMaxT() && (grouped_nv4 || grouped_mx));
    if (use_moe_decode) {
        if (grouped_nv4) {
            uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * 2 * I), 16);
            uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * I), 16);
            const LinearTuning t_gu = PickNvfp4MoeTuning(2 * I, H, 1, false);
            const LinearTuning t_dn = PickNvfp4MoeTuning(H, I, 1, true);
            ProfiledCall(prof, sr, "gemm:moe.gate_up", [&] {
                r4dx_gemm_nvfp4_moe_decode(
                    P(x), P(w.gate_up_ptrs0.data()), P(w.gate_up_ptrs1.data()),
                    P(w.gate_up_factors.data()), P(ids), P(gu), static_cast<int>(Pn),
                    static_cast<int>(K), static_cast<int>(E), static_cast<int>(H),
                    static_cast<int>(2 * I), t_gu.WV, t_gu.SK, t_gu.NPW, t_gu.NT, s);
            });
            ProfiledCall(prof, sr, "moe.act",
                         [&] { r4dx_inkling2_swiglu_softcap_bf16(P(gu), P(act), Pn, I, 2 * I, s); });
            ProfiledCall(prof, sr, "gemm:moe.down", [&] {
                r4dx_gemm_nvfp4_moe_decode(P(act), P(w.down_ptrs0.data()), P(w.down_ptrs1.data()),
                                           P(w.down_factors.data()), P(ids), P(ys),
                                           static_cast<int>(Pn), 1, static_cast<int>(E),
                                           static_cast<int>(I), static_cast<int>(H), t_dn.WV,
                                           t_dn.SK, t_dn.NPW, t_dn.NT, s);
            });
        } else if (grouped_mx) {
            uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * 2 * I), 16);
            uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * I), 16);
            const LinearTuning t_gu = PickTuning(Layout::kMxfp8, w.gate_up[0].N, w.gate_up[0].K, 1);
            const LinearTuning t_dn = PickTuning(Layout::kMxfp8, w.down[0].N, w.down[0].K, 1);
            ProfiledCall(prof, sr, "gemm:moe.gate_up", [&] {
                r4dx_gemm_mxfp8_moe_decode(
                    P(x), P(w.gate_up_ptrs0.data()), P(w.gate_up_ptrs1.data()), P(ids), P(gu),
                    static_cast<int>(Pn), static_cast<int>(K), static_cast<int>(E),
                    static_cast<int>(w.gate_up[0].K), static_cast<int>(w.gate_up[0].N), t_gu.WV,
                    t_gu.SK, t_gu.NPW, t_gu.NT, s);
            });
            ProfiledCall(prof, sr, "moe.act",
                         [&] { r4dx_inkling2_swiglu_softcap_bf16(P(gu), P(act), Pn, I, 2 * I, s); });
            ProfiledCall(prof, sr, "gemm:moe.down", [&] {
                r4dx_gemm_mxfp8_moe_decode(
                    P(act), P(w.down_ptrs0.data()), P(w.down_ptrs1.data()), P(ids), P(ys),
                    static_cast<int>(Pn), 1, static_cast<int>(E), static_cast<int>(w.down[0].K),
                    static_cast<int>(w.down[0].N), t_dn.WV, t_dn.SK, t_dn.NPW, t_dn.NT, s);
            });
        } else {
            uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(2 * I), 16);
            uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(I), 16);
            ProfiledCall(prof, sr, "moe.readback", [&] {
                R4DX_HIP_CHECK(hipMemcpyAsync(moe_ids_host_.data(), ids,
                                              static_cast<size_t>(Pn) * sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
                R4DX_HIP_CHECK(hipStreamSynchronize(sr));
            });
            const size_t mark = arena_.Mark();
            for (int64_t j = 0; j < K; ++j) {
                arena_.Rewind(mark);
                const int32_t e = moe_ids_host_.data()[j];
                if (e < 0 || e >= E)
                    throw std::runtime_error("Inkling2Model: router returned expert id " +
                                             std::to_string(e));
                ProfiledCall(prof, sr, "gemm:moe.gate_up",
                             [&] { Lin(w.gate_up[static_cast<size_t>(e)], x, gu, 1); });
                ProfiledCall(prof, sr, "moe.act", [&] {
                    r4dx_inkling2_swiglu_softcap_bf16(P(gu), P(act), 1, I, 2 * I, s);
                });
                ProfiledCall(prof, sr, "gemm:moe.down",
                             [&] { Lin(w.down[static_cast<size_t>(e)], act, ys + j * H, 1); });
            }
        }
        ProfiledCall(prof, sr, "moe.combine", [&] {
            r4dx_inkling2_moe_combine_bf16(P(ys), 0, P(wts), P(out), static_cast<int>(T),
                                           static_cast<int>(K), static_cast<int>(H), s);
        });
    } else {
        int32_t* counts = arena_.Alloc<int32_t>(static_cast<size_t>(E), 16);
        int32_t* offsets = arena_.Alloc<int32_t>(static_cast<size_t>(E + 1), 16);
        int32_t* sorted = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
        int32_t* pos_of_pair = arena_.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
        uint16_t* xs = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);
        ProfiledCall(prof, sr, "moe.sort", [&] {
            r4dx_inkling2_moe_sort(P(ids), static_cast<int>(T), static_cast<int>(K),
                                   static_cast<int>(E), P(counts), P(offsets), P(sorted),
                                   P(pos_of_pair), s);
            r4dx_inkling2_moe_gather_bf16(P(x), P(sorted), P(xs), static_cast<int>(Pn),
                                          static_cast<int>(K), static_cast<int>(H), H,
                                          P(offsets + E), s);
        });
        const bool grouped_nv4_pf =
            w.nvfp4_experts && !w.gate_up_ptrs0.empty() && !w.down_ptrs0.empty();
        const bool grouped_mx_pf = !grouped_nv4_pf && MoeGroupedPrefillEnabled() &&
                                   !w.gate_up_ptrs0.empty() && !w.gate_up_ptrs1.empty() &&
                                   !w.down_ptrs0.empty() && !w.down_ptrs1.empty() &&
                                   !w.gate_up.empty() && w.gate_up[0].layout == Layout::kMxfp8 &&
                                   w.down[0].layout == Layout::kMxfp8;
        if (grouped_nv4_pf) {
            uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * 2 * I), 16);
            uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * I), 16);
            const int64_t m_tune = std::min<int64_t>(T, 64);
            const LinearTuning t_gu = PickNvfp4MoeTuning(2 * I, H, m_tune, false);
            const LinearTuning t_dn = PickNvfp4MoeTuning(H, I, m_tune, true);
            ProfiledCall(prof, sr, "gemm:moe.gate_up", [&] {
                r4dx_gemm_nvfp4_moe_prefill(
                    P(xs), P(w.gate_up_ptrs0.data()), P(w.gate_up_ptrs1.data()),
                    P(w.gate_up_factors.data()), P(offsets), P(gu), static_cast<int>(Pn),
                    static_cast<int>(E), static_cast<int>(H), static_cast<int>(2 * I), t_gu.WV,
                    t_gu.SK, t_gu.NPW, t_gu.NT, s);
            });
            ProfiledCall(prof, sr, "moe.act", [&] {
                r4dx_inkling2_swiglu_softcap_bf16(P(gu), P(act), Pn, I, 2 * I, s);
            });
            ProfiledCall(prof, sr, "gemm:moe.down", [&] {
                r4dx_gemm_nvfp4_moe_prefill(P(act), P(w.down_ptrs0.data()), P(w.down_ptrs1.data()),
                                            P(w.down_factors.data()), P(offsets), P(ys),
                                            static_cast<int>(Pn), static_cast<int>(E),
                                            static_cast<int>(I), static_cast<int>(H), t_dn.WV,
                                            t_dn.SK, t_dn.NPW, t_dn.NT, s);
            });
        } else if (grouped_mx_pf) {
            if (MoeCheckOffsetsEnabled()) {
                R4DX_HIP_CHECK(hipMemcpyAsync(moe_off_host_.data(), offsets,
                                              static_cast<size_t>(E + 1) * sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
                R4DX_HIP_CHECK(hipStreamSynchronize(sr));
                const int32_t* off = moe_off_host_.data();
                if (off[0] != 0 || off[E] != Pn)
                    throw std::runtime_error("Inkling2Model: the expert sort covers " +
                                             std::to_string(off[E]) + " of " + std::to_string(Pn) +
                                             " pairs");
            }
            uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * 2 * I), 16);
            uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(Pn * I), 16);
            const int64_t m_tune = std::min<int64_t>(T, 64);
            const LinearTuning t_gu =
                PickTuning(Layout::kMxfp8, w.gate_up[0].N, w.gate_up[0].K, m_tune);
            const LinearTuning t_dn = PickTuning(Layout::kMxfp8, w.down[0].N, w.down[0].K, m_tune);
            ProfiledCall(prof, sr, "gemm:moe.gate_up", [&] {
                r4dx_gemm_mxfp8_moe_prefill(
                    P(xs), P(w.gate_up_ptrs0.data()), P(w.gate_up_ptrs1.data()), P(offsets), P(gu),
                    static_cast<int>(Pn), static_cast<int>(E), static_cast<int>(w.gate_up[0].K),
                    static_cast<int>(w.gate_up[0].N), t_gu.WV, t_gu.SK, t_gu.NPW, t_gu.NT, s);
            });
            ProfiledCall(prof, sr, "moe.act", [&] {
                r4dx_inkling2_swiglu_softcap_bf16(P(gu), P(act), Pn, I, 2 * I, s);
            });
            ProfiledCall(prof, sr, "gemm:moe.down", [&] {
                r4dx_gemm_mxfp8_moe_prefill(
                    P(act), P(w.down_ptrs0.data()), P(w.down_ptrs1.data()), P(offsets), P(ys),
                    static_cast<int>(Pn), static_cast<int>(E), static_cast<int>(w.down[0].K),
                    static_cast<int>(w.down[0].N), t_dn.WV, t_dn.SK, t_dn.NPW, t_dn.NT, s);
            });
        } else {
            uint16_t* gu = arena_.Alloc<uint16_t>(static_cast<size_t>(T * 2 * I), 16);
            uint16_t* act = arena_.Alloc<uint16_t>(static_cast<size_t>(T * I), 16);
            ProfiledCall(prof, sr, "moe.readback", [&] {
                R4DX_HIP_CHECK(hipMemcpyAsync(moe_off_host_.data(), offsets,
                                              static_cast<size_t>(E + 1) * sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
                R4DX_HIP_CHECK(hipStreamSynchronize(sr));
            });
            const int32_t* off = moe_off_host_.data();
            if (off[0] != 0 || off[E] != Pn)
                throw std::runtime_error("Inkling2Model: the expert sort covers " +
                                         std::to_string(off[E]) + " of " + std::to_string(Pn) +
                                         " pairs");
            const size_t mark = arena_.Mark();
            for (int64_t e = 0; e < E; ++e) {
                const int64_t r0 = off[e], c = off[e + 1] - off[e];
                if (c == 0)
                    continue;
                if (c < 0 || c > T)
                    throw std::runtime_error("Inkling2Model: bad expert row count " +
                                             std::to_string(c));
                arena_.Rewind(mark);
                ProfiledCall(prof, sr, "gemm:moe.gate_up",
                             [&] { Lin(w.gate_up[static_cast<size_t>(e)], xs + r0 * H, gu, c); });
                ProfiledCall(prof, sr, "moe.act", [&] {
                    r4dx_inkling2_swiglu_softcap_bf16(P(gu), P(act), c, I, 2 * I, s);
                });
                ProfiledCall(prof, sr, "gemm:moe.down",
                             [&] { Lin(w.down[static_cast<size_t>(e)], act, ys + r0 * H, c); });
            }
        }
        ProfiledCall(prof, sr, "moe.combine", [&] {
            r4dx_inkling2_moe_combine_bf16(P(ys), P(pos_of_pair), P(wts), P(out),
                                           static_cast<int>(T), static_cast<int>(K),
                                           static_cast<int>(H), s);
        });
    }
    if (comm_ != nullptr) {
        ProfiledCall(prof, sr, "tp.allreduce", [&] {
            if (out_f32 != nullptr) {
                if (!(ArBf16ToF32Enabled() &&
                      comm_->AllReduceSumBf16ToF32Rows(out, out_f32, T, H, sr))) {
                    r4dx_inkling2_sconv_capture_x_f32(P(out), 1, P(out_f32), static_cast<int>(T),
                                                      static_cast<int>(H), s);
                    if (!comm_->AllReduceSumF32(out_f32, T * H, sr)) {
                        comm_->AllReduceSumBf16Rows(out, T, H, sr);
                        r4dx_inkling2_sconv_capture_x_f32(
                            P(out), 1, P(out_f32), static_cast<int>(T), static_cast<int>(H), s);
                    }
                }
            } else {
                comm_->AllReduceSumBf16Rows(out, T, H, sr);
            }
        });
    }
}

// ---- one visit
// ------------------------------------------------------------------------------------------------------

void Inkling2Model::RunVisit(const Inkling2Visit& v, Chunk& c, int64_t T, SpanAccumulator* prof,
                             bool use_pos_dev, bool graph_long_ctx, bool capture_conv_hist) {
    const Inkling2Config& cfg = container_.Config();
    const Inkling2LayerWeights& w = container_.Layer(v.layer);
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t H = cfg.hidden_size;
    const float eps = static_cast<float>(cfg.rms_norm_eps);
    const size_t mark = arena_.Mark();
    const size_t n = static_cast<size_t>(T * H);
    int64_t st_ptrs[6]{}, st_inv[6]{};
    const auto fill_stack = [&](const std::vector<int>& blks, int h_idx) -> int {
        int k = 0;
        for (int b : blks) {
            st_ptrs[k] = P(pool_[static_cast<size_t>(b)].data());
            st_inv[k] = P(pool_inv_rms_[static_cast<size_t>(b)].data());
            ++k;
        }
        st_ptrs[k] = P(pool_[static_cast<size_t>(h_idx)].data());
        st_inv[k] = P(pool_inv_rms_[static_cast<size_t>(h_idx)].data());
        return k + 1;
    };
    float* h = pool_[static_cast<size_t>(c.h)].data();

    // messages (passes >= 2): [q, k, v, mlp] = AttnRes(msg_query[j], [prev blocks..., prev h])
    const bool msgs = v.HasMessages();
    if (msgs) {
        const int n_prev = fill_stack(c.prev_B, c.prev_h);
        ProfiledCall(prof, sr, "attnres.msg", [&] {
            r4dx_inkling2_attnres_prenorm_f32(
                st_ptrs, st_inv, n_prev,
                P(container_.MsgQueries().data() + static_cast<int64_t>(v.loop_j) * 4 * H), 4,
                P(msg_.data()), 0, 0, static_cast<int>(T), static_cast<int>(H), eps, s);
        });
    }
    const auto msg = [&](int m) { return msg_.data() + static_cast<int64_t>(m) * T * H; };

    // attention inputs
    uint16_t* xq = arena_.Alloc<uint16_t>(n, 16);
    uint16_t* xk = arena_.Alloc<uint16_t>(n, 16);
    uint16_t* xv = arena_.Alloc<uint16_t>(n, 16);
    ProfiledCall(prof, sr, "attnres.attn", [&] {
        if (!msgs && !w.full && w.n_attn_queries == 1) {
            // Sliding layer without messages (nq == 1): fuse AttnRes + attn_norm -> xq in a single
            // launch!
            const int n_st = fill_stack(c.B, c.h);
            r4dx_inkling2_attnres_prenorm_f32(st_ptrs, st_inv, n_st, P(w.attn_query.data()), 1, 0,
                                              P(w.attn_norm.data()), P(xq), static_cast<int>(T),
                                              static_cast<int>(H), eps, s);
            xk = xq;
            xv = xq;
            return;
        }
        float* a = nullptr;
        if (w.n_attn_queries > 0) {
            a = arena_.Alloc<float>(static_cast<size_t>(w.n_attn_queries) * n, 16);
            const int n_st = fill_stack(c.B, c.h);
            r4dx_inkling2_attnres_prenorm_f32(st_ptrs, st_inv, n_st, P(w.attn_query.data()),
                                              w.n_attn_queries, P(a), 0, 0, static_cast<int>(T),
                                              static_cast<int>(H), eps, s);
        }
        if (msgs) {
            MsgGate3AttnNormBf16(w, a != nullptr ? a : h, msg(0), xq, xk, xv, T);
            return;
        }
        const auto src = [&](int m) -> const float* {
            if (a == nullptr)
                return h;  // layer 0: no attention residual
            return w.full ? a + static_cast<int64_t>(m) * T * H : a;
        };
        uint16_t* xs[3] = {xq, xk, xv};
        for (int m = 0; m < 3; ++m) {
            const uint16_t* norm = w.full ? (m == 0   ? w.q_norm.data()
                                             : m == 1 ? w.k_norm.data()
                                                      : w.v_norm.data())
                                          : w.attn_norm.data();
            const float* in = src(m);
            if (!w.full && m > 0) {
                xs[m] = xq;  // one input for q, k, v
                continue;
            }
            r4dx_rmsnorm_plain_f32in_bf16(P(in), P(norm), P(xs[m]), T, H, eps, s);
        }
        xk = xs[1];
        xv = xs[2];
    });

    const bool use_fp32_ar = (comm_ != nullptr && T <= 16 && TpFp32AllReduceEnabled());
    uint16_t* o = arena_.Alloc<uint16_t>(n, 16);
    float* o_f32 = use_fp32_ar ? arena_.Alloc<float>(n, 16) : nullptr;
    Attention(v, xq, xk, xv, o, T, prof, use_pos_dev, graph_long_ctx, capture_conv_hist, o_f32);

    // post-attention convolution; a block boundary keeps the incoming partial as a block
    ProfiledCall(prof, sr, "conv.post_attn", [&] {
        const int64_t hist2 = capture_conv_hist ? P(ConvHist(v.cache_slot, 2)) : 0;
        const int64_t o_in = use_fp32_ar ? P(o_f32) : P(o);
        const int o_bf16 = use_fp32_ar ? 0 : 1;
        if (cfg.IsBoundary(v.layer)) {
            c.B.push_back(c.h);
            const int hn = c.Acquire();
            r4dx_inkling2_sconv_static_hist_norm_f32(
                o_in, o_bf16, P(w.attn_conv.data()), P(ConvState(v.cache_slot, 2)), 0,
                P(pool_[static_cast<size_t>(hn)].data()), hist2,
                P(pool_inv_rms_[static_cast<size_t>(hn)].data()), static_cast<int>(T),
                static_cast<int>(H), eps, s);
            c.h = hn;
        } else {
            r4dx_inkling2_sconv_static_hist_norm_f32(
                o_in, o_bf16, P(w.attn_conv.data()), P(ConvState(v.cache_slot, 2)), P(h), P(h),
                hist2, P(pool_inv_rms_[static_cast<size_t>(c.h)].data()), static_cast<int>(T),
                static_cast<int>(H), eps, s);
        }
    });
    h = pool_[static_cast<size_t>(c.h)].data();

    // MLP input
    uint16_t* xm = arena_.Alloc<uint16_t>(n, 16);
    ProfiledCall(prof, sr, "attnres.mlp", [&] {
        const int n_st = fill_stack(c.B, c.h);
        if (!msgs) {
            // No message gate: fuse AttnRes + mlp_norm -> xm in a single launch!
            r4dx_inkling2_attnres_prenorm_f32(st_ptrs, st_inv, n_st, P(w.mlp_query.data()), 1, 0,
                                              P(w.mlp_norm.data()), P(xm), static_cast<int>(T),
                                              static_cast<int>(H), eps, s);
            return;
        }
        float* y = arena_.Alloc<float>(n, 16);
        r4dx_inkling2_attnres_prenorm_f32(st_ptrs, st_inv, n_st, P(w.mlp_query.data()), 1, P(y), 0,
                                          0, static_cast<int>(T), static_cast<int>(H), eps, s);
        float* y2 = arena_.Alloc<float>(n, 16);
        MsgGate(w.msg[3], y, msg(3), y2, T);
        r4dx_rmsnorm_plain_f32in_bf16(P(y2), P(w.mlp_norm.data()), P(xm), T, H, eps, s);
    });
    uint16_t* f = arena_.Alloc<uint16_t>(n, 16);
    float* f_f32 = use_fp32_ar ? arena_.Alloc<float>(n, 16) : nullptr;
    Moe(v.layer, xm, f, T, v.index, prof, f_f32);
    ProfiledCall(prof, sr, "conv.post_mlp", [&] {
        const int64_t hist3 = capture_conv_hist ? P(ConvHist(v.cache_slot, 3)) : 0;
        const int64_t f_in = use_fp32_ar ? P(f_f32) : P(f);
        const int f_bf16 = use_fp32_ar ? 0 : 1;
        r4dx_inkling2_sconv_static_hist_norm_f32(f_in, f_bf16, P(w.mlp_conv.data()),
                                                 P(ConvState(v.cache_slot, 3)), P(h), P(h), hist3,
                                                 P(pool_inv_rms_[static_cast<size_t>(c.h)].data()),
                                                 static_cast<int>(T), static_cast<int>(H), eps, s);
    });
    arena_.Rewind(mark);
}

// ---- one chunk
// ------------------------------------------------------------------------------------------------------

void Inkling2Model::Head(const float* h_row, const std::vector<const float*>& blocks,
                         SpanAccumulator* prof, bool materialize_f32_logits) {
    const Inkling2Config& cfg = container_.Config();
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t H = cfg.hidden_size, vl = cfg.HeadRowsLocal();
    uint16_t* xn = arena_.Alloc<uint16_t>(static_cast<size_t>(H), 16);
    uint16_t* lg = lg_bf16_dev_.data();
    ProfiledCall(prof, sr, "final", [&] {
        int64_t st[6]{};
        int k = 0;
        for (const float* b : blocks) st[k++] = P(b);
        st[k++] = P(h_row);
        r4dx_inkling2_attnres_prenorm_f32(st, nullptr, k, P(container_.FinalQuery().data()), 1, 0,
                                          P(container_.FinalNorm().data()), P(xn), 1,
                                          static_cast<int>(H), static_cast<float>(cfg.rms_norm_eps),
                                          s);
    });
    ProfiledCall(prof, sr, "gemm:lm_head", [&] { Lin(container_.LmHead(), xn, lg, 1); });
    if (materialize_f32_logits) {
        ProfiledCall(prof, sr, "lm_head.softcap", [&] {
            r4dx_inkling2_logit_softcap_f32(P(lg), 1, vl, P(logits_dev_.data()), vl, 1,
                                            static_cast<int>(vl),
                                            static_cast<float>(cfg.final_logit_softcap), s);
            if (vocab_local_valid_ < vl) {
                R4DX_HIP_CHECK(hipMemsetD32Async(
                    reinterpret_cast<hipDeviceptr_t>(logits_dev_.data() + vocab_local_valid_),
                    static_cast<int>(kNegInfBits), static_cast<size_t>(vl - vocab_local_valid_),
                    sr));
            }
        });
    }
}

std::vector<float> Inkling2Model::RunChunk(const std::vector<int32_t>& token_ids, bool want_logits,
                                           int32_t* greedy_out, const SummaryRequest* summary,
                                           SpanAccumulator* prof, bool verify_conv_hist,
                                           std::vector<int32_t>* verify_greedy_out,
                                           float verify_inv_temp,
                                           std::vector<kernels::RowSummary>* verify_summaries_out,
                                           const std::vector<ImageSpan>* chunk_images,
                                           const std::function<void()>* while_busy) {
    const int64_t T = static_cast<int64_t>(token_ids.size());
    if (T < 1 || T > max_chunk_)
        throw std::runtime_error("Inkling2Model::RunChunk: token_ids.size() must be in [1, " +
                                 std::to_string(max_chunk_) + "]");
    if (pos_ + T > max_ctx_)
        throw std::runtime_error("Inkling2Model::RunChunk: position " + std::to_string(pos_ + T) +
                                 " exceeds max_ctx " + std::to_string(max_ctx_));
    const Inkling2Config& cfg = container_.Config();
    const int64_t H = cfg.hidden_size;
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    for (int32_t id : token_ids) {
        if (id < 0 || id >= cfg.vocab_size)
            throw std::invalid_argument("Inkling2Model::RunChunk: token id " + std::to_string(id) +
                                        " outside [0, " + std::to_string(cfg.vocab_size) + ")");
    }
    std::copy(token_ids.begin(), token_ids.end(), ids_host_.data());
    ids_dev_.CopyFromHostAsync(ids_host_.data(), static_cast<size_t>(T), stream_);

    const bool capture_taps = (dspark_ != nullptr);

    auto run_body = [&](bool use_pos_dev, bool graph_long_ctx) {
        arena_.Reset();
        Chunk c;
        c.used.assign(pool_.size(), false);
        c.h = c.Acquire();
        ProfiledCall(prof, sr, "embed", [&] {
            const bool vis_after_norm = InkEnvFlag("R4DX_INK_VIS_AFTER_NORM", true);
            uint16_t* e = arena_.Alloc<uint16_t>(static_cast<size_t>(T * H), 16);
            r4dx_embedding_gather_bf16(P(container_.EmbedTokens().data()), P(ids_dev_.data()), P(e),
                                       T, H, cfg.vocab_size, s);
            if (chunk_images != nullptr && !vis_after_norm) {
                for (const ImageSpan& sp : *chunk_images) {
                    if (sp.tokens <= 0 || sp.embeds == nullptr)
                        continue;
                    const hipMemcpyKind kind =
                        sp.embeds_on_host ? hipMemcpyHostToDevice : hipMemcpyDeviceToDevice;
                    R4DX_HIP_CHECK(hipMemcpyAsync(
                        e + static_cast<size_t>(sp.offset * H), sp.embeds,
                        static_cast<size_t>(sp.tokens * H) * sizeof(uint16_t), kind, sr));
                }
            }
            float* h0 = pool_[static_cast<size_t>(c.h)].data();
            r4dx_rmsnorm_plain_bf16(P(e), P(container_.EmbedNorm().data()), P(h0), T, H,
                                    static_cast<float>(cfg.rms_norm_eps), /*out_fp32=*/1, s);
            if (chunk_images != nullptr && vis_after_norm) {
                for (const ImageSpan& sp : *chunk_images) {
                    if (sp.tokens <= 0 || sp.embeds == nullptr)
                        continue;
                    const uint16_t* src_dev = sp.embeds;
                    if (sp.embeds_on_host) {
                        uint16_t* tmp =
                            arena_.Alloc<uint16_t>(static_cast<size_t>(sp.tokens * H), 16);
                        R4DX_HIP_CHECK(hipMemcpyAsync(
                            tmp, sp.embeds, static_cast<size_t>(sp.tokens * H) * sizeof(uint16_t),
                            hipMemcpyHostToDevice, sr));
                        src_dev = tmp;
                    }
                    float* dst = h0 + static_cast<size_t>(sp.offset * H);
                    R4DX_HIP_CHECK(hipMemsetAsync(
                        dst, 0, static_cast<size_t>(sp.tokens * H) * sizeof(float), sr));
                    r4dx_inkling2_add_f32_bf16(P(dst), P(src_dev), P(dst), sp.tokens * H, s);
                }
            }
            r4dx_inkling2_row_inv_rms_f32(P(h0), P(pool_inv_rms_[static_cast<size_t>(c.h)].data()),
                                          static_cast<int>(T), static_cast<int>(H),
                                          static_cast<float>(cfg.rms_norm_eps), s);
        });
        const size_t base_mark = arena_.Mark();

        for (const Inkling2Visit& v : visits_) {
            if (v.phase == Inkling2Visit::Phase::kLoop && v.pass >= 1 && v.loop_j == 0) {
                c.Release(c.prev_h);
                for (int b : c.prev_B) c.Release(b);
                c.prev_B.assign(c.B.begin() + static_cast<std::ptrdiff_t>(c.n_pre + 1), c.B.end());
                c.prev_h = c.h;
                c.B = c.B_pre;
                c.h = c.h_pre;
            }
            if (v.phase == Inkling2Visit::Phase::kPost && v.layer == cfg.loop_end) {
                c.Release(c.prev_h);
                for (int b : c.prev_B) c.Release(b);
                c.prev_h = -1;
                c.prev_B.clear();
            }
            if (capture_taps) {
                int tap_idx = -1;
                if (v.index == 5)
                    tap_idx = 0;
                else if (v.index == 23)
                    tap_idx = 1;
                else if (v.index == 41)
                    tap_idx = 2;
                if (tap_idx >= 0) {
                    r4dx_f32_to_bf16(P(pool_[static_cast<size_t>(c.h)].data()),
                                     P(dspark_taps_[static_cast<size_t>(tap_idx)].data()), T * H,
                                     s);
                }
            }
            RunVisit(v, c, T, prof, use_pos_dev, graph_long_ctx, verify_conv_hist);
            if (v.phase == Inkling2Visit::Phase::kPre && v.layer == cfg.loop_start - 1) {
                c.h_pre = c.h;
                c.B_pre = c.B;
                c.n_pre = c.B.size();
            }
        }
        arena_.Rewind(base_mark);

        if (capture_taps) {
            int64_t st[6]{}, inv[6]{};
            int k = 0;
            for (int b : c.B) {
                st[k] = P(pool_[static_cast<size_t>(b)].data());
                inv[k] = P(pool_inv_rms_[static_cast<size_t>(b)].data());
                ++k;
            }
            st[k] = P(pool_[static_cast<size_t>(c.h)].data());
            inv[k] = P(pool_inv_rms_[static_cast<size_t>(c.h)].data());
            ++k;
            r4dx_inkling2_attnres_prenorm_f32(
                st, inv, k, P(container_.FinalQuery().data()), 1, 0,
                P(container_.FinalNorm().data()), P(dspark_taps_[3].data()), static_cast<int>(T),
                static_cast<int>(H), static_cast<float>(cfg.rms_norm_eps), s);
        }

        const bool is_verify = (verify_greedy_out != nullptr || verify_summaries_out != nullptr);
        if (is_verify) {
            Lin(container_.LmHead(), dspark_taps_[3].data(), lg_bf16_dev_.data(), T);
        } else if (want_logits) {
            std::vector<const float*> blocks;
            for (int b : c.B) blocks.push_back(pool_[static_cast<size_t>(b)].data() + (T - 1) * H);
            Head(pool_[static_cast<size_t>(c.h)].data() + (T - 1) * H, blocks, prof,
                 /*materialize_f32_logits=*/prof != nullptr);
        }
        arena_.Reset();
    };

    const bool is_verify = (verify_greedy_out != nullptr || verify_summaries_out != nullptr);
    const bool try_graph = T == 1 && want_logits && prof == nullptr && !trace_.enabled &&
                           decode_graph_eligible_ && !decode_graph_disabled_ && !is_verify;
    const bool try_verify_graph =
        T > 1 && T <= 8 && is_verify && prof == nullptr && !trace_.enabled &&
        decode_graph_eligible_ && !decode_graph_disabled_ &&
        (T <= MoeDecodeMaxT() || MoeGroupedPrefillEnabled()) && !MoeCheckOffsetsEnabled();
    bool launched_graph = false;
    bool just_captured = false;
    if (try_graph) {
        if (decode_steps_seen_ >= 1) {
            pos_host_.data()[0] = static_cast<int32_t>(pos_);
            pos_dev_.CopyFromHostAsync(pos_host_.data(), 1, stream_);
            const bool long_ctx = (pos_ >= 128);
            GraphExecPtr& slot = long_ctx ? decode_graph_long_ : decode_graph_short_;
            if (!slot) {
                hipGraphExec_t exec = nullptr;
                if (CaptureDecodeGraph(sr, &exec, [&] {
                        run_body(/*use_pos_dev=*/true, /*graph_long_ctx=*/long_ctx);
                    })) {
                    slot.reset(exec);
                    just_captured = true;
                } else {
                    decode_graph_disabled_ = true;
                    std::cerr << "[r4dx::model::Inkling2Model] HIP graph capture failed; falling "
                                 "back to eager decode\n";
                }
            }
            if (slot) {
                R4DX_HIP_CHECK(hipGraphLaunch(static_cast<hipGraphExec_t>(slot.get()), sr));
                launched_graph = true;
            }
        }
        ++decode_steps_seen_;
    } else if (try_verify_graph) {
        pos_host_.data()[0] = static_cast<int32_t>(pos_);
        pos_dev_.CopyFromHostAsync(pos_host_.data(), 1, stream_);
        GraphExecPtr& vg = verify_graphs_[static_cast<size_t>(T)];
        if (!vg) {
            hipGraphExec_t exec = nullptr;
            if (CaptureDecodeGraph(
                    sr, &exec, [&] { run_body(/*use_pos_dev=*/true, /*graph_long_ctx=*/true); })) {
                vg.reset(exec);
                just_captured = true;
            } else {
                decode_graph_disabled_ = true;
                std::cerr << "[r4dx::model::Inkling2Model] HIP verify graph capture failed; "
                             "falling back to eager verify\n";
            }
        }
        if (vg) {
            R4DX_HIP_CHECK(hipGraphLaunch(static_cast<hipGraphExec_t>(vg.get()), sr));
            launched_graph = true;
        }
    }
    if (launched_graph && !just_captured && comm_ != nullptr) {
        comm_->NoteGraphReplay(static_cast<size_t>(T * H * 2),
                               static_cast<int64_t>(2 * visits_.size()));
    }
    if (!launched_graph) {
        run_body(/*use_pos_dev=*/false, /*graph_long_ctx=*/true);
    }

    const int64_t vl = cfg.HeadRowsLocal();
    const bool fast_argmax = FastArgmaxEnabled();
    bool verify_tp_fast = false;
    bool single_tp_fast = false;

    const auto materialize_verify_f32 = [&] {
        r4dx_inkling2_logit_softcap_f32(P(lg_bf16_dev_.data()), 1, vl, P(verify_logits_dev_.data()),
                                        vl, static_cast<int>(T), static_cast<int>(vl),
                                        static_cast<float>(cfg.final_logit_softcap), s);
        if (vocab_local_valid_ < vl) {
            for (int64_t r = 0; r < T; ++r) {
                R4DX_HIP_CHECK(hipMemsetD32Async(
                    reinterpret_cast<hipDeviceptr_t>(verify_logits_dev_.data() + r * vl +
                                                     vocab_local_valid_),
                    static_cast<int>(kNegInfBits), static_cast<size_t>(vl - vocab_local_valid_),
                    sr));
            }
        }
    };
    const auto materialize_single_f32 = [&] {
        r4dx_inkling2_logit_softcap_f32(P(lg_bf16_dev_.data()), 1, vl, P(logits_dev_.data()), vl, 1,
                                        static_cast<int>(vl),
                                        static_cast<float>(cfg.final_logit_softcap), s);
        if (vocab_local_valid_ < vl) {
            R4DX_HIP_CHECK(hipMemsetD32Async(
                reinterpret_cast<hipDeviceptr_t>(logits_dev_.data() + vocab_local_valid_),
                static_cast<int>(kNegInfBits), static_cast<size_t>(vl - vocab_local_valid_), sr));
        }
    };

    if (verify_greedy_out != nullptr) {
        if (fast_argmax) {
            r4dx_inkling2_softcap_argmax_rows_bf16(
                P(lg_bf16_dev_.data()), vl, static_cast<int>(T), static_cast<int>(vocab_local_valid_),
                static_cast<float>(cfg.final_logit_softcap), P(softcap_argmax_ws_.data()),
                comm_ != nullptr ? P(verify_argmax_pair_dev_.data()) : P(verify_argmax_dev_.data()),
                comm_ != nullptr ? 1 : 0, s);
        } else {
            materialize_verify_f32();
            if (comm_ != nullptr) {
                r4dx_argmax_rows_f32(P(verify_logits_dev_.data()), vl,
                                     P(verify_argmax_pair_dev_.data()),
                                     P(verify_argmax_pair_dev_.data() + 1), 2, T, vl, s);
            } else {
                r4dx_argmax_rows_f32(P(verify_logits_dev_.data()), vl,
                                     P(verify_argmax_dev_.data()), 0, 1, T, vocab_local_valid_, s);
            }
        }
        if (comm_ != nullptr) {
            verify_tp_fast = comm_->AllReduceArgmaxRows(
                verify_argmax_pair_dev_.data(), static_cast<int32_t>(cfg.HeadRowBegin()),
                static_cast<int>(T), verify_argmax_dev_.data(), sr);
            if (verify_tp_fast) {
                R4DX_HIP_CHECK(hipMemcpyAsync(verify_argmax_host_.data(), verify_argmax_dev_.data(),
                                              static_cast<size_t>(T) * sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
            } else {
                R4DX_HIP_CHECK(hipMemcpyAsync(
                    verify_pairs_host_.data(), verify_argmax_pair_dev_.data(),
                    static_cast<size_t>(2 * T) * sizeof(int32_t), hipMemcpyDeviceToHost, sr));
            }
        } else {
            R4DX_HIP_CHECK(hipMemcpyAsync(verify_argmax_host_.data(), verify_argmax_dev_.data(),
                                          static_cast<size_t>(T) * sizeof(int32_t),
                                          hipMemcpyDeviceToHost, sr));
        }
    } else if (verify_summaries_out != nullptr) {
        materialize_verify_f32();
        r4dx_topk_lse_f32_ws(P(verify_logits_dev_.data()), P(verify_summary_ids_dev_.data()),
                             P(verify_summary_vals_dev_.data()), P(verify_summary_lse_dev_.data()),
                             static_cast<int>(T), comm_ != nullptr ? vl : vocab_local_valid_,
                             verify_inv_temp, s, comm_ != nullptr ? P(topk_lse_ws_.data()) : 0);
        R4DX_HIP_CHECK(hipMemcpyAsync(verify_summary_ids_host_.data(),
                                      verify_summary_ids_dev_.data(),
                                      static_cast<size_t>(T * R4DX_TOPK_LSE_K) * sizeof(int32_t),
                                      hipMemcpyDeviceToHost, sr));
        R4DX_HIP_CHECK(hipMemcpyAsync(verify_summary_vals_host_.data(),
                                      verify_summary_vals_dev_.data(),
                                      static_cast<size_t>(T * R4DX_TOPK_LSE_K) * sizeof(float),
                                      hipMemcpyDeviceToHost, sr));
        R4DX_HIP_CHECK(hipMemcpyAsync(verify_summary_lse_host_.data(),
                                      verify_summary_lse_dev_.data(),
                                      static_cast<size_t>(T) * sizeof(float),
                                      hipMemcpyDeviceToHost, sr));
    }

    if (want_logits && !is_verify && prof == nullptr) {
        if (greedy_out != nullptr && fast_argmax) {
            r4dx_inkling2_softcap_argmax_rows_bf16(
                P(lg_bf16_dev_.data()), vl, 1, static_cast<int>(vocab_local_valid_),
                static_cast<float>(cfg.final_logit_softcap), P(softcap_argmax_ws_.data()),
                comm_ != nullptr ? P(argmax_pair_dev_.data()) : P(argmax_dev_.data()),
                comm_ != nullptr ? 1 : 0, s);
            if (comm_ != nullptr) {
                single_tp_fast = comm_->AllReduceArgmaxPair(
                    argmax_pair_dev_.data(), static_cast<int32_t>(cfg.HeadRowBegin()),
                    argmax_dev_.data(), sr);
                if (single_tp_fast) {
                    R4DX_HIP_CHECK(hipMemcpyAsync(argmax_host_.data(), argmax_dev_.data(),
                                                  sizeof(int32_t), hipMemcpyDeviceToHost, sr));
                } else {
                    R4DX_HIP_CHECK(hipMemcpyAsync(argmax_pair_host_.data(), argmax_pair_dev_.data(),
                                                  2 * sizeof(int32_t), hipMemcpyDeviceToHost, sr));
                }
            } else {
                R4DX_HIP_CHECK(hipMemcpyAsync(argmax_host_.data(), argmax_dev_.data(), sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
            }
        } else {
            materialize_single_f32();
            if (greedy_out != nullptr && comm_ != nullptr) {
                r4dx_argmax_val_f32(P(logits_dev_.data()), P(argmax_pair_dev_.data()),
                                    P(argmax_pair_dev_.data() + 1), vl, s);
                R4DX_HIP_CHECK(hipMemcpyAsync(argmax_pair_host_.data(), argmax_pair_dev_.data(),
                                              2 * sizeof(int32_t), hipMemcpyDeviceToHost, sr));
            } else if (greedy_out != nullptr) {
                r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), vocab_local_valid_, s);
                R4DX_HIP_CHECK(hipMemcpyAsync(argmax_host_.data(), argmax_dev_.data(), sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
            } else if (summary != nullptr) {
                r4dx_topk_lse_f32_ws(
                    P(logits_dev_.data()), P(summary_ids_dev_.data()), P(summary_vals_dev_.data()),
                    P(summary_lse_dev_.data()), 1, comm_ != nullptr ? vl : vocab_local_valid_,
                    summary->inv_temperature, s, comm_ != nullptr ? P(topk_lse_ws_.data()) : 0);
                R4DX_HIP_CHECK(hipMemcpyAsync(summary_ids_host_.data(), summary_ids_dev_.data(),
                                              R4DX_TOPK_LSE_K * sizeof(int32_t),
                                              hipMemcpyDeviceToHost, sr));
                R4DX_HIP_CHECK(hipMemcpyAsync(summary_vals_host_.data(), summary_vals_dev_.data(),
                                              R4DX_TOPK_LSE_K * sizeof(float),
                                              hipMemcpyDeviceToHost, sr));
                R4DX_HIP_CHECK(hipMemcpyAsync(summary_lse_host_.data(), summary_lse_dev_.data(),
                                              sizeof(float), hipMemcpyDeviceToHost, sr));
            }
        }
    }

    if (capture_taps && !is_verify) {
        dspark_->InjectTaps(dspark_taps_[0].data(), dspark_taps_[1].data(), dspark_taps_[2].data(),
                            dspark_taps_[3].data(), T, pos_, stream_, arena_);
    }

    if (while_busy != nullptr && prof == nullptr) {
        R4DX_HIP_CHECK(hipEventRecord(step_done_.e, sr));
        try {
            (*while_busy)();
        } catch (...) {
            (void)hipEventSynchronize(step_done_.e);
            throw;
        }
        R4DX_HIP_CHECK(hipEventSynchronize(step_done_.e));
    } else if (prof == nullptr) {
        stream_.Synchronize();
    }

    if (verify_greedy_out != nullptr) {
        verify_greedy_out->resize(static_cast<size_t>(T));
        if (comm_ != nullptr) {
            if (verify_tp_fast) {
                std::memcpy(verify_greedy_out->data(), verify_argmax_host_.data(),
                            static_cast<size_t>(T) * sizeof(int32_t));
            } else {
                MergeVerifyPairs(verify_pairs_host_.data(), T, verify_greedy_out);
            }
        } else {
            std::memcpy(verify_greedy_out->data(), verify_argmax_host_.data(),
                        static_cast<size_t>(T) * sizeof(int32_t));
        }
    } else if (verify_summaries_out != nullptr) {
        verify_summaries_out->resize(static_cast<size_t>(T));
        const int64_t row_vocab = comm_ != nullptr ? cfg.HeadRowsLocal() : vocab_local_valid_;
        for (int64_t r = 0; r < T; ++r) {
            kernels::RowSummary& rs = (*verify_summaries_out)[static_cast<size_t>(r)];
            rs.k = R4DX_TOPK_LSE_K;
            rs.vocab = row_vocab;
            rs.inv_temperature = verify_inv_temp;
            rs.lse = verify_summary_lse_host_.data()[r];
            for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
                rs.ids[j] = verify_summary_ids_host_.data()[r * R4DX_TOPK_LSE_K + j];
                rs.vals[j] = verify_summary_vals_host_.data()[r * R4DX_TOPK_LSE_K + j];
            }
        }
        if (comm_ != nullptr)
            MergeVerifySummaries(T, verify_summaries_out);
    }

    std::vector<float> logits;
    if (want_logits && prof == nullptr) {
        if (comm_ != nullptr) {
            if (greedy_out != nullptr) {
                if (single_tp_fast) {
                    *greedy_out = argmax_host_.data()[0];
                } else {
                    tp::ArgmaxPair mine{};
                    std::memcpy(&mine, argmax_pair_host_.data(), sizeof(mine));
                    if (mine.idx != 0x7fffffff)
                        mine.idx += static_cast<int32_t>(container_.Config().HeadRowBegin());
                    std::array<tp::ArgmaxPair, 2> all{};
                    comm_->HostAllGather(&mine, sizeof(mine), all.data());
                    *greedy_out = tp::MergeArgmax(all.data(), comm_->World());
                }
            } else if (summary != nullptr) {
                summary->out->k = R4DX_TOPK_LSE_K;
                summary->out->vocab = container_.Config().HeadRowsLocal();
                summary->out->inv_temperature = summary->inv_temperature;
                summary->out->lse = summary_lse_host_.data()[0];
                for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
                    summary->out->ids[j] = summary_ids_host_.data()[j];
                    summary->out->vals[j] = summary_vals_host_.data()[j];
                }
                MergeSummary(summary->out);
            } else {
                GatherVocabRow(&logits);
            }
        } else if (greedy_out != nullptr) {
            *greedy_out = argmax_host_.data()[0];
        } else if (summary != nullptr) {
            summary->out->k = R4DX_TOPK_LSE_K;
            summary->out->vocab = vocab_local_valid_;
            summary->out->inv_temperature = summary->inv_temperature;
            summary->out->lse = summary_lse_host_.data()[0];
            for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
                summary->out->ids[j] = summary_ids_host_.data()[j];
                summary->out->vals[j] = summary_vals_host_.data()[j];
            }
        } else {
            logits.resize(static_cast<size_t>(Vocab()));
            logits_dev_.CopyToHost(logits.data(), logits.size());
        }
    }
    pos_ += T;
    started_ = true;
    return logits;
}

void Inkling2Model::FetchRowSummary(float inv_temperature, kernels::RowSummary* out) {
    R4DX_HIP_CHECK(hipMemcpyAsync(summary_ids_host_.data(), summary_ids_dev_.data(),
                                  R4DX_TOPK_LSE_K * sizeof(int32_t), hipMemcpyDeviceToHost,
                                  stream_.get()));
    R4DX_HIP_CHECK(hipMemcpyAsync(summary_vals_host_.data(), summary_vals_dev_.data(),
                                  R4DX_TOPK_LSE_K * sizeof(float), hipMemcpyDeviceToHost,
                                  stream_.get()));
    R4DX_HIP_CHECK(hipMemcpyAsync(summary_lse_host_.data(), summary_lse_dev_.data(), sizeof(float),
                                  hipMemcpyDeviceToHost, stream_.get()));
    stream_.Synchronize();
    out->k = R4DX_TOPK_LSE_K;
    out->vocab = comm_ != nullptr ? container_.Config().HeadRowsLocal() : vocab_local_valid_;
    out->inv_temperature = inv_temperature;
    out->lse = summary_lse_host_.data()[0];
    for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
        out->ids[j] = summary_ids_host_.data()[j];
        out->vals[j] = summary_vals_host_.data()[j];
    }
}

// ---- public forward
// -------------------------------------------------------------------------------------------------

void Inkling2Model::EncodeImages(const float* pixel_values, int64_t total_patches,
                                 const std::vector<vision::GridThw>& grids,
                                 core::DeviceBuffer<uint16_t>* out_bf16) {
    vision_.Encode(pixel_values, total_patches, grids, out_bf16, stream_, arena_);
}

std::vector<float> Inkling2Model::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                    const std::vector<ImageSpan>& images) {
    if (images.empty())
        return Prefill(token_ids);
    if (token_ids.empty())
        throw std::runtime_error("Inkling2Model::PrefillMultimodal: token_ids is empty");
    const int64_t H = container_.Config().hidden_size;
    std::vector<float> logits;
    for (size_t off = 0; off < token_ids.size();) {
        const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
        const int64_t c0 = static_cast<int64_t>(off), c1 = static_cast<int64_t>(off + n);
        std::vector<ImageSpan> slice;
        for (const ImageSpan& sp : images) {
            const int64_t s0 = std::max(c0, sp.offset);
            const int64_t s1 = std::min(c1, sp.offset + sp.tokens);
            if (s1 > s0) {
                ImageSpan sub;
                sub.offset = s0 - c0;
                sub.tokens = s1 - s0;
                sub.embeds = sp.embeds + static_cast<size_t>((s0 - sp.offset) * H);
                sub.embeds_on_host = sp.embeds_on_host;
                slice.push_back(sub);
            }
        }
        const std::vector<int32_t> chunk(token_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                         token_ids.begin() + static_cast<std::ptrdiff_t>(off + n));
        const bool last = off + n == token_ids.size();
        off += n;
        std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr, false, nullptr,
                                        0.0f, nullptr, slice.empty() ? nullptr : &slice);
        if (last)
            logits = std::move(l);
    }
    return logits;
}

std::vector<float> Inkling2Model::Prefill(const std::vector<int32_t>& token_ids) {
    if (token_ids.empty())
        throw std::runtime_error("Inkling2Model::Prefill: token_ids is empty");
    std::vector<float> logits;
    for (size_t off = 0; off < token_ids.size();) {
        const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
        const std::vector<int32_t> chunk(token_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                         token_ids.begin() + static_cast<std::ptrdiff_t>(off + n));
        const bool last = off + n == token_ids.size();
        off += n;
        std::vector<float> l = RunChunk(chunk, last, nullptr, nullptr, nullptr);
        if (last)
            logits = std::move(l);
    }
    return logits;
}

std::vector<float> Inkling2Model::DecodeStep(int32_t token_id) {
    return RunChunk({token_id}, true, nullptr, nullptr, nullptr);
}

int32_t Inkling2Model::DecodeStepGreedy(int32_t token_id) {
    int32_t next = -1;
    RunChunk({token_id}, true, &next, nullptr, nullptr);
    return next;
}

int32_t Inkling2Model::DecodeStepGreedyOverlap(int32_t token_id,
                                               const std::function<void()>& while_busy) {
    if (comm_ != nullptr || !InkEnvFlag("R4DX_INK_HOST_OVERLAP", true)) {
        while_busy();
        return DecodeStepGreedy(token_id);
    }
    int32_t next = -1;
    RunChunk({token_id}, true, &next, nullptr, nullptr, false, nullptr, 0.0f, nullptr, nullptr,
             &while_busy);
    return next;
}

int32_t Inkling2Model::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                                         std::mt19937_64& rng) {
    if (params.temperature <= 0.0f)
        return DecodeStepGreedy(token_id);
    const float inv_t = SummaryInvTemperature(params);
    const int64_t vocab = Vocab();
    if (inv_t == 0.0f || vocab_local_valid_ <= R4DX_TOPK_LSE_K) {
        const std::vector<float> logits = DecodeStep(token_id);
        return kernels::SampleCanonical(logits.data(), vocab, params, kernels::DrawUniform01(rng));
    }
    kernels::RowSummary summary;
    const SummaryRequest req{inv_t, &summary};
    RunChunk({token_id}, true, nullptr, &req, nullptr);
    const double u = kernels::DrawUniform01(rng);
    const kernels::SummarySampleResult r = kernels::SampleFromSummary(summary, params, u);
    if (r.resolved)
        return r.token;
    ++sampled_fallback_rows_;
    if (comm_ != nullptr) {
        GatherVocabRow(&sampled_row_scratch_);
    } else {
        sampled_row_scratch_.resize(static_cast<size_t>(vocab));
        logits_dev_.CopyToHost(sampled_row_scratch_.data(), static_cast<size_t>(vocab));
    }
    return kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
}

// ---- tensor parallel
// ------------------------------------------------------------------------------------------------

void Inkling2Model::RequireNotTp(const char* what) const {
    if (comm_ != nullptr)
        throw core::TpUnsupportedError(std::string("Inkling2Model::") + what +
                                       ": not supported under TP");
}

int32_t Inkling2Model::MergeGreedyPairAt(const int32_t* dev_pair) {
    tp::ArgmaxPair mine{};
    static_assert(sizeof(tp::ArgmaxPair) == 2 * sizeof(int32_t),
                  "{idx, value} pair is two 32-bit words");
    R4DX_HIP_CHECK(hipMemcpy(&mine, dev_pair, sizeof(mine), hipMemcpyDeviceToHost));
    if (mine.idx != 0x7fffffff) mine.idx += static_cast<int32_t>(container_.Config().HeadRowBegin());
    std::array<tp::ArgmaxPair, 2> all{};
    comm_->HostAllGather(&mine, sizeof(mine), all.data());
    return tp::MergeArgmax(all.data(), comm_->World());
}

void Inkling2Model::MergeVerifyPairs(const int32_t* pairs_host, int64_t rows,
                                     std::vector<int32_t>* out) {
    const int32_t row_off = static_cast<int32_t>(container_.Config().HeadRowBegin());
    std::vector<tp::ArgmaxPair> mine(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) {
        mine[static_cast<size_t>(r)].idx =
            (pairs_host[2 * r] == 0x7fffffff) ? 0x7fffffff : (pairs_host[2 * r] + row_off);
        std::memcpy(&mine[static_cast<size_t>(r)].val, &pairs_host[2 * r + 1], sizeof(float));
    }
    std::vector<tp::ArgmaxPair> gathered(static_cast<size_t>(comm_->World() * rows));
    comm_->HostAllGather(mine.data(), mine.size() * sizeof(tp::ArgmaxPair), gathered.data());
    out->resize(static_cast<size_t>(rows));
    tp::MergeArgmaxRows(gathered.data(), comm_->World(), static_cast<int>(rows), out->data());
}

int32_t Inkling2Model::MergeGreedyPair() {
    return MergeGreedyPairAt(argmax_pair_dev_.data());
}

int32_t Inkling2Model::LmHeadArgmaxFromBf16(const uint16_t* h_bf16) {
    const Inkling2Config& cfg = container_.Config();
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t vl = cfg.HeadRowsLocal();
    const size_t mark = arena_.Mark();
    uint16_t* lg = arena_.Alloc<uint16_t>(static_cast<size_t>(vl), 16);
    Lin(container_.LmHead(), h_bf16, lg, 1);
    bool tp_fast = false;
    if (FastArgmaxEnabled()) {
        r4dx_inkling2_softcap_argmax_rows_bf16(
            P(lg), vl, 1, static_cast<int>(vocab_local_valid_),
            static_cast<float>(cfg.final_logit_softcap), P(softcap_argmax_ws_.data()),
            comm_ != nullptr ? P(argmax_pair_dev_.data()) : P(argmax_dev_.data()),
            comm_ != nullptr ? 1 : 0, s);
        if (comm_ != nullptr) {
            tp_fast = comm_->AllReduceArgmaxPair(
                argmax_pair_dev_.data(), static_cast<int32_t>(cfg.HeadRowBegin()),
                argmax_dev_.data(), sr);
        }
    } else {
        r4dx_inkling2_logit_softcap_f32(P(lg), 1, vl, P(logits_dev_.data()), vl, 1,
                                        static_cast<int>(vl),
                                        static_cast<float>(cfg.final_logit_softcap), s);
        if (vocab_local_valid_ < vl) {
            R4DX_HIP_CHECK(hipMemsetD32Async(
                reinterpret_cast<hipDeviceptr_t>(logits_dev_.data() + vocab_local_valid_),
                static_cast<int>(kNegInfBits), static_cast<size_t>(vl - vocab_local_valid_), sr));
        }
        if (comm_ != nullptr) {
            r4dx_argmax_val_f32(P(logits_dev_.data()), P(argmax_pair_dev_.data()),
                                P(argmax_pair_dev_.data() + 1), vl, s);
        } else {
            r4dx_argmax_f32(P(logits_dev_.data()), P(argmax_dev_.data()), vocab_local_valid_, s);
        }
    }
    if (comm_ == nullptr || tp_fast) {
        R4DX_HIP_CHECK(hipMemcpyAsync(argmax_host_.data(), argmax_dev_.data(), sizeof(int32_t),
                                      hipMemcpyDeviceToHost, sr));
    }
    arena_.Rewind(mark);
    stream_.Synchronize();
    if (comm_ != nullptr && !tp_fast)
        return MergeGreedyPair();
    return argmax_host_.data()[0];
}

int32_t Inkling2Model::DsparkLogitsArgmax(const uint16_t* base_bf16, const uint16_t* delta_bf16,
                                          const uint16_t* bias_bf16) {
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    bool tp_fast = false;
    if (comm_ != nullptr) {
        r4dx_inkling2_dspark_add_argmax_bf16(
            P(base_bf16), P(delta_bf16), P(bias_bf16), P(dspark_argmax_ws_.data()),
            P(argmax_pair_dev_.data()), P(argmax_pair_dev_.data() + 1), vocab_local_valid_, s);
        tp_fast = comm_->AllReduceArgmaxPair(
            argmax_pair_dev_.data(), static_cast<int32_t>(container_.Config().HeadRowBegin()),
            argmax_dev_.data(), sr);
        if (tp_fast) {
            R4DX_HIP_CHECK(hipMemcpyAsync(argmax_host_.data(), argmax_dev_.data(), sizeof(int32_t),
                                          hipMemcpyDeviceToHost, sr));
        }
    } else {
        r4dx_inkling2_dspark_add_argmax_bf16(P(base_bf16), P(delta_bf16), P(bias_bf16),
                                             P(dspark_argmax_ws_.data()), P(argmax_dev_.data()), 0,
                                             vocab_local_valid_, s);
        R4DX_HIP_CHECK(hipMemcpyAsync(argmax_host_.data(), argmax_dev_.data(), sizeof(int32_t),
                                      hipMemcpyDeviceToHost, sr));
    }
    stream_.Synchronize();
    if (comm_ != nullptr && !tp_fast)
        return MergeGreedyPair();
    return argmax_host_.data()[0];
}

void Inkling2Model::DsparkLogitsArgmaxAsync(const uint16_t* base_bf16, const uint16_t* delta_bf16,
                                            const uint16_t* bias_bf16, int32_t* out_tok_dev) {
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    if (comm_ != nullptr) {
        r4dx_inkling2_dspark_add_argmax_bf16(
            P(base_bf16), P(delta_bf16), P(bias_bf16), P(dspark_argmax_ws_.data()),
            P(argmax_pair_dev_.data()), P(argmax_pair_dev_.data() + 1), vocab_local_valid_, s);
        const bool ok = comm_->AllReduceArgmaxPair(
            argmax_pair_dev_.data(), static_cast<int32_t>(container_.Config().HeadRowBegin()),
            out_tok_dev, sr);
        if (!ok) {
            throw std::logic_error(
                "Inkling2Model::DsparkLogitsArgmaxAsync: TpComm does not support AllReduceArgmaxPair");
        }
    } else {
        r4dx_inkling2_dspark_add_argmax_bf16(P(base_bf16), P(delta_bf16), P(bias_bf16),
                                             P(dspark_argmax_ws_.data()), P(out_tok_dev), 0,
                                             vocab_local_valid_, s);
    }
}

void Inkling2Model::DsparkLogitsSummary(const uint16_t* base_bf16, const uint16_t* delta_bf16,
                                        const uint16_t* bias_bf16, float inv_temp,
                                        kernels::RowSummary* out) {
    const Inkling2Config& cfg = container_.Config();
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t vl = cfg.HeadRowsLocal();
    r4dx_inkling2_dspark_add_softcap_f32(P(base_bf16), P(delta_bf16), P(bias_bf16),
                                         P(logits_dev_.data()), vl,
                                         static_cast<float>(cfg.final_logit_softcap), s);
    if (vocab_local_valid_ < vl) {
        R4DX_HIP_CHECK(hipMemsetD32Async(
            reinterpret_cast<hipDeviceptr_t>(logits_dev_.data() + vocab_local_valid_),
            static_cast<int>(kNegInfBits), static_cast<size_t>(vl - vocab_local_valid_), sr));
    }
    r4dx_topk_lse_f32_ws(P(logits_dev_.data()), P(summary_ids_dev_.data()),
                         P(summary_vals_dev_.data()), P(summary_lse_dev_.data()), 1,
                         comm_ != nullptr ? vl : vocab_local_valid_, inv_temp, s,
                         comm_ != nullptr ? P(topk_lse_ws_.data()) : 0);
    stream_.Synchronize();
    FetchRowSummary(inv_temp, out);
    if (comm_ != nullptr)
        MergeSummary(out);
}

std::vector<int32_t> Inkling2Model::DecodeStepDflashGreedy(int32_t token_id, int64_t draft_k,
                                                           int64_t* walk_len_out) {
    if (!dspark_)
        throw std::runtime_error("Inkling2Model::DecodeStepDflashGreedy: no DSpark drafter loaded");
    const Inkling2Config& cfg = container_.Config();
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t H = cfg.hidden_size;
    const int64_t k_max = (draft_k > 0) ? std::min<int64_t>(draft_k, dspark_->NumSpecTokens())
                                        : dspark_->NumSpecTokens();
    const bool adapt = DsparkAdaptEnabled() && k_max >= 3;
    int64_t k = adapt ? std::clamp<int64_t>(dspark_adapt_k_, 2, k_max) : k_max;
    k = std::min<int64_t>(k, max_ctx_ - pos_ - 1);
    if (k <= 0) {
        if (walk_len_out != nullptr)
            *walk_len_out = 0;
        return {DecodeStepGreedy(token_id)};
    }

    // 1. Draft k speculative tokens using DSpark.
    const auto embed_anchor = [&](int32_t id, uint16_t* out_bf16) {
        ids_host_.data()[0] = id;
        ids_dev_.CopyFromHostAsync(ids_host_.data(), 1, stream_);
        const size_t emark = arena_.Mark();
        uint16_t* e = arena_.Alloc<uint16_t>(static_cast<size_t>(H), 16);
        r4dx_embedding_gather_bf16(P(container_.EmbedTokens().data()), P(ids_dev_.data()), P(e), 1,
                                   H, cfg.vocab_size, s);
        r4dx_rmsnorm_plain_bf16(P(e), P(container_.EmbedNorm().data()), P(out_bf16), 1, H,
                                static_cast<float>(cfg.rms_norm_eps), /*out_fp32=*/0, s);
        arena_.Rewind(emark);
    };
    const auto lm_argmax = [&](const uint16_t* h_bf16) { return LmHeadArgmaxFromBf16(h_bf16); };
    const auto lm_batch = [&](const uint16_t* h_bf16, int64_t M, uint16_t* out_bf16) {
        Lin(container_.LmHead(), h_bf16, out_bf16, M);
    };
    const auto lg_argmax = [&](const uint16_t* base_bf16, const uint16_t* delta_bf16,
                               const uint16_t* bias_bf16) {
        return DsparkLogitsArgmax(base_bf16, delta_bf16, bias_bf16);
    };
    const bool use_async_markov =
        DsparkAsyncMarkovEnabled() && (comm_ == nullptr || comm_->IsGraphCapturable());
    DsparkLogitsArgmaxAsyncFn lg_argmax_async;
    if (use_async_markov) {
        lg_argmax_async = [&](const uint16_t* base_bf16, const uint16_t* delta_bf16,
                              const uint16_t* bias_bf16, int32_t* out_tok_dev) {
            DsparkLogitsArgmaxAsync(base_bf16, delta_bf16, bias_bf16, out_tok_dev);
        };
    }
    bool replayed_full = false;
    std::vector<int32_t> drafted =
        dspark_->DraftGreedy(token_id, pos_, k, stream_, arena_, embed_anchor, lm_argmax, lm_batch,
                             lg_argmax, lg_argmax_async, &replayed_full);
    if (replayed_full && comm_ != nullptr) {
        comm_->NoteGraphReplay(16, static_cast<int64_t>(drafted.size()));
    }

    // 2. Snapshot short-conv states before running the [token_id, d_0..d_{k-1}] verify window.
    R4DX_HIP_CHECK(hipMemcpyAsync(conv_verify_bak_.data(), conv_.data(), conv_.bytes(),
                                  hipMemcpyDeviceToDevice, sr));
    std::vector<int32_t> window;
    window.reserve(static_cast<size_t>(1 + k));
    window.push_back(token_id);
    window.insert(window.end(), drafted.begin(), drafted.end());

    const int64_t pos_before = pos_;
    std::vector<int32_t> tgt;
    (void)RunChunk(window, /*want_logits=*/false, nullptr, nullptr, nullptr,
                   /*verify_conv_hist=*/true, &tgt);

    // 3. Accept prefix + bonus token.
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(1 + k));
    out.push_back(tgt[0]);
    int64_t n_keep = 1;
    for (int64_t i = 0; i < k; ++i) {
        if (tgt[static_cast<size_t>(i)] == drafted[static_cast<size_t>(i)]) {
            out.push_back(tgt[static_cast<size_t>(i + 1)]);
            ++n_keep;
        } else {
            break;
        }
    }

    // 4. Exact short-conv rollback if fewer than all 1 + k tokens were kept.
    if (n_keep < 1 + k) {
        r4dx_inkling2_sconv_rollback_batch_f32(
            P(conv_verify_bak_.data()), P(conv_verify_hist_.data()), P(conv_.data()),
            P(conv_off_dev_.data()), P(conv_hist_off_dev_.data()), P(conv_ch_dev_.data()),
            static_cast<int>(conv_off_.size()), conv_max_ch_, static_cast<int>(n_keep), s);
        pos_ = pos_before + n_keep;
    }

    // 5. Inject the n_keep committed taps into the DSpark context KV cache.
    dspark_->InjectTaps(dspark_taps_[0].data(), dspark_taps_[1].data(), dspark_taps_[2].data(),
                        dspark_taps_[3].data(), n_keep, pos_before, stream_, arena_);
    if (!AsyncPostVerifyEnabled()) stream_.Synchronize();
    if (adapt) {
        if (n_keep == 1 + k)
            dspark_adapt_k_ = std::min<int64_t>(k + 1, k_max);
        else if (n_keep == 1)
            dspark_adapt_k_ = std::max<int64_t>(2, k - 1);
    }
    if (walk_len_out != nullptr)
        *walk_len_out = k;
    return out;
}

std::vector<int32_t> Inkling2Model::DecodeStepDflashSampled(int32_t token_id, int64_t draft_k,
                                                            const kernels::SampleParams& params,
                                                            std::mt19937_64& rng,
                                                            int64_t* walk_len_out) {
    if (params.temperature <= 0.0f)
        return DecodeStepDflashGreedy(token_id, draft_k, walk_len_out);
    if (!dspark_)
        throw std::runtime_error(
            "Inkling2Model::DecodeStepDflashSampled: no DSpark drafter loaded");
    const float inv_t = SummaryInvTemperature(params);
    if (inv_t == 0.0f || vocab_local_valid_ <= R4DX_TOPK_LSE_K) {
        if (walk_len_out != nullptr)
            *walk_len_out = 0;
        return {DecodeStepSampled(token_id, params, rng)};
    }
    const Inkling2Config& cfg = container_.Config();
    const hipStream_t sr = stream_.get();
    const int64_t s = P(sr);
    const int64_t H = cfg.hidden_size;
    const int64_t k_max = (draft_k > 0) ? std::min<int64_t>(draft_k, dspark_->NumSpecTokens())
                                        : dspark_->NumSpecTokens();
    const bool adapt = DsparkAdaptEnabled() && k_max >= 3;
    int64_t k = adapt ? std::clamp<int64_t>(dspark_adapt_k_, 2, k_max) : k_max;
    k = std::min<int64_t>(k, max_ctx_ - pos_ - 1);
    if (k <= 0) {
        if (walk_len_out != nullptr)
            *walk_len_out = 0;
        return {DecodeStepSampled(token_id, params, rng)};
    }

    // 1. Draft k speculative tokens with DSpark (stochastic proposal when R4DX_INK_SPEC_SAMPLE=1,
    // else greedy).
    const bool spec_sample = DsparkSpecSampleEnabled();
    std::vector<TopKDist> draft_dists;
    if (spec_sample)
        draft_dists.reserve(static_cast<size_t>(k));

    const auto embed_anchor = [&](int32_t id, uint16_t* out_bf16) {
        ids_host_.data()[0] = id;
        ids_dev_.CopyFromHostAsync(ids_host_.data(), 1, stream_);
        const size_t emark = arena_.Mark();
        uint16_t* e = arena_.Alloc<uint16_t>(static_cast<size_t>(H), 16);
        r4dx_embedding_gather_bf16(P(container_.EmbedTokens().data()), P(ids_dev_.data()), P(e), 1,
                                   H, cfg.vocab_size, s);
        r4dx_rmsnorm_plain_bf16(P(e), P(container_.EmbedNorm().data()), P(out_bf16), 1, H,
                                static_cast<float>(cfg.rms_norm_eps), /*out_fp32=*/0, s);
        arena_.Rewind(emark);
    };
    const auto lm_argmax = [&](const uint16_t* h_bf16) { return LmHeadArgmaxFromBf16(h_bf16); };
    const auto lm_batch = [&](const uint16_t* h_bf16, int64_t M, uint16_t* out_bf16) {
        Lin(container_.LmHead(), h_bf16, out_bf16, M);
    };
    const auto lg_step = [&](const uint16_t* base_bf16, const uint16_t* delta_bf16,
                             const uint16_t* bias_bf16) -> int32_t {
        if (!spec_sample)
            return DsparkLogitsArgmax(base_bf16, delta_bf16, bias_bf16);
        kernels::RowSummary q_sum;
        DsparkLogitsSummary(base_bf16, delta_bf16, bias_bf16, inv_t, &q_sum);
        TopKDist qd = BuildTopKDist(q_sum, params, /*force_close_topk=*/true);
        const int32_t tok = qd.Sample(kernels::DrawUniform01(rng));
        draft_dists.push_back(qd);
        return tok;
    };
    const bool use_async_markov = !spec_sample && DsparkAsyncMarkovEnabled() &&
                                  (comm_ == nullptr || comm_->IsGraphCapturable());
    DsparkLogitsArgmaxAsyncFn lg_argmax_async;
    if (use_async_markov) {
        lg_argmax_async = [&](const uint16_t* base_bf16, const uint16_t* delta_bf16,
                              const uint16_t* bias_bf16, int32_t* out_tok_dev) {
            DsparkLogitsArgmaxAsync(base_bf16, delta_bf16, bias_bf16, out_tok_dev);
        };
    }
    bool replayed_full = false;
    std::vector<int32_t> drafted =
        dspark_->DraftGreedy(token_id, pos_, k, stream_, arena_, embed_anchor, lm_argmax, lm_batch,
                             lg_step, lg_argmax_async, &replayed_full);
    if (replayed_full && comm_ != nullptr) {
        comm_->NoteGraphReplay(16, static_cast<int64_t>(drafted.size()));
    }

    // 2. Snapshot short-conv states and verify [token_id, d_0..d_{k-1}] -> per-row top-K + LSE
    // summaries.
    R4DX_HIP_CHECK(hipMemcpyAsync(conv_verify_bak_.data(), conv_.data(), conv_.bytes(),
                                  hipMemcpyDeviceToDevice, sr));
    std::vector<int32_t> window;
    window.reserve(static_cast<size_t>(1 + k));
    window.push_back(token_id);
    window.insert(window.end(), drafted.begin(), drafted.end());

    const int64_t pos_before = pos_;
    std::vector<kernels::RowSummary> summaries;
    (void)RunChunk(window, /*want_logits=*/false, nullptr, nullptr, nullptr,
                   /*verify_conv_hist=*/true, nullptr, inv_t, &summaries);

    // 3. Accept/reject: lossless stochastic speculative sampling (when spec_sample) or seed-exact
    // match.
    const int64_t vocab = Vocab();
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(1 + k));
    int64_t n_keep = 0;
    if (spec_sample && static_cast<int64_t>(draft_dists.size()) == k) {
        bool all_accepted = true;
        for (int64_t i = 0; i < k; ++i) {
            const int32_t d_tok = drafted[static_cast<size_t>(i)];
            const TopKDist& qd = draft_dists[static_cast<size_t>(i)];
            const TopKDist pd = BuildTopKDist(summaries[static_cast<size_t>(i)], params,
                                              /*force_close_topk=*/false);
            const double q_p = qd.Prob(d_tok);
            double p_p = pd.Prob(d_tok);
            bool have_full = false;
            if (p_p == 0.0 && !pd.closed && pd.tail_mass > 0.0) {
                ++sampled_fallback_rows_;
                GatherVerifyVocabRow(i, &sampled_row_scratch_);
                have_full = true;
                // Compute exact full-vocab probability of d_tok under pure temperature.
                const float max_s = summaries[static_cast<size_t>(i)].vals[0] / params.temperature;
                const double s_full = std::exp(
                    static_cast<double>(summaries[static_cast<size_t>(i)].lse) -
                    static_cast<double>(summaries[static_cast<size_t>(i)].vals[0]) *
                        static_cast<double>(summaries[static_cast<size_t>(i)].inv_temperature));
                if (d_tok >= 0 && d_tok < vocab && s_full > 0.0) {
                    p_p =
                        std::exp(static_cast<double>(
                            sampled_row_scratch_[static_cast<size_t>(d_tok)] / params.temperature -
                            max_s)) /
                        s_full;
                }
            }
            const double u = kernels::DrawUniform01(rng);
            if (q_p <= 0.0 ? false : (u * q_p <= p_p)) {
                out.push_back(d_tok);
                ++n_keep;
                continue;
            }
            // Rejected: sample from residual norm(max(0, P - Q)).
            double res[kernels::kRowSummaryMaxK];
            double res_sum = 0.0;
            for (int j = 0; j < pd.n; ++j) {
                res[j] = std::max(0.0, pd.probs[j] - qd.Prob(pd.ids[j]));
                res_sum += res[j];
            }
            const double u2 = kernels::DrawUniform01(rng);
            const double total_mass = res_sum + pd.tail_mass;
            int32_t tok = -1;
            if (total_mass > 0.0 && (pd.closed || u2 * total_mass < res_sum)) {
                const double target = u2 * total_mass;
                double cum = 0.0;
                tok = pd.ids[pd.n - 1];
                for (int j = 0; j < pd.n; ++j) {
                    cum += res[j];
                    if (target < cum) {
                        tok = pd.ids[j];
                        break;
                    }
                }
            } else {
                // Rare: draw landed in unclosed tail or degenerate residual -> exact fallback.
                if (!have_full) {
                    ++sampled_fallback_rows_;
                    GatherVerifyVocabRow(i, &sampled_row_scratch_);
                }
                tok = kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u2);
            }
            out.push_back(tok);
            ++n_keep;
            all_accepted = false;
            break;
        }
        if (all_accepted) {
            const double u = kernels::DrawUniform01(rng);
            const kernels::SummarySampleResult r =
                kernels::SampleFromSummary(summaries[static_cast<size_t>(k)], params, u);
            int32_t tok = r.token;
            if (!r.resolved) {
                ++sampled_fallback_rows_;
                GatherVerifyVocabRow(k, &sampled_row_scratch_);
                tok = kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
            }
            out.push_back(tok);
            ++n_keep;
        }
    } else {
        for (int64_t i = 0; i <= k; ++i) {
            const double u = kernels::DrawUniform01(rng);
            const kernels::SummarySampleResult r =
                kernels::SampleFromSummary(summaries[static_cast<size_t>(i)], params, u);
            int32_t tok = r.token;
            if (!r.resolved) {
                ++sampled_fallback_rows_;
                GatherVerifyVocabRow(i, &sampled_row_scratch_);
                tok = kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
            }
            out.push_back(tok);
            ++n_keep;
            if (i < k && tok == drafted[static_cast<size_t>(i)])
                continue;
            break;
        }
    }

    // 4. Exact short-conv rollback if fewer than all 1 + k tokens were kept.
    if (n_keep < 1 + k) {
        r4dx_inkling2_sconv_rollback_batch_f32(
            P(conv_verify_bak_.data()), P(conv_verify_hist_.data()), P(conv_.data()),
            P(conv_off_dev_.data()), P(conv_hist_off_dev_.data()), P(conv_ch_dev_.data()),
            static_cast<int>(conv_off_.size()), conv_max_ch_, static_cast<int>(n_keep), s);
        pos_ = pos_before + n_keep;
    }

    // 5. Inject the n_keep committed taps into the DSpark context KV cache.
    dspark_->InjectTaps(dspark_taps_[0].data(), dspark_taps_[1].data(), dspark_taps_[2].data(),
                        dspark_taps_[3].data(), n_keep, pos_before, stream_, arena_);
    if (!AsyncPostVerifyEnabled()) stream_.Synchronize();
    if (adapt) {
        if (n_keep == 1 + k)
            dspark_adapt_k_ = std::min<int64_t>(k + 1, k_max);
        else if (n_keep == 1)
            dspark_adapt_k_ = std::max<int64_t>(2, k - 1);
    }
    if (walk_len_out != nullptr)
        *walk_len_out = k;
    return out;
}

void Inkling2Model::MergeSummary(kernels::RowSummary* out) {
    static_assert(std::is_trivially_copyable_v<kernels::RowSummary>,
                  "row summaries cross the host exchange as raw bytes");
    for (int j = 0; j < out->k; ++j)
        out->ids[j] += static_cast<int32_t>(container_.Config().HeadRowBegin());
    std::array<kernels::RowSummary, 2> all{};
    comm_->HostAllGather(out, sizeof(kernels::RowSummary), all.data());
    *out = tp::MergeRowSummaries(all.data(), comm_->World(), Vocab());
}

void Inkling2Model::MergeVerifySummaries(int64_t rows, std::vector<kernels::RowSummary>* out) {
    static_assert(std::is_trivially_copyable_v<kernels::RowSummary>,
                  "row summaries cross the host exchange as raw bytes");
    const int32_t row_off = static_cast<int32_t>(container_.Config().HeadRowBegin());
    for (int64_t r = 0; r < rows; ++r) {
        kernels::RowSummary& rs = (*out)[static_cast<size_t>(r)];
        for (int j = 0; j < rs.k; ++j) rs.ids[j] += row_off;
    }
    const int w = comm_->World();
    std::vector<kernels::RowSummary> all(static_cast<size_t>(w * rows));
    comm_->HostAllGather(out->data(), static_cast<size_t>(rows) * sizeof(kernels::RowSummary),
                         all.data());
    std::array<kernels::RowSummary, 2> per_rank{};
    for (int64_t r = 0; r < rows; ++r) {
        for (int rank = 0; rank < w; ++rank) {
            per_rank[static_cast<size_t>(rank)] =
                all[static_cast<size_t>(rank * rows + r)];
        }
        (*out)[static_cast<size_t>(r)] = tp::MergeRowSummaries(per_rank.data(), w, Vocab());
    }
}

void Inkling2Model::GatherVocabRow(std::vector<float>* full) {
    const int64_t vl = container_.Config().HeadRowsLocal();
    gather_host_.resize(static_cast<size_t>(vl));
    R4DX_HIP_CHECK(hipMemcpy(gather_host_.data(), logits_dev_.data(),
                             static_cast<size_t>(vl) * sizeof(float), hipMemcpyDeviceToHost));
    full->resize(static_cast<size_t>(vl * comm_->World()));
    comm_->HostAllGather(gather_host_.data(), static_cast<size_t>(vl) * sizeof(float),
                         full->data());
    full->resize(static_cast<size_t>(Vocab()));
}

void Inkling2Model::GatherVerifyVocabRow(int64_t row, std::vector<float>* full) {
    const int64_t vl = container_.Config().HeadRowsLocal();
    const float* src = verify_logits_dev_.data() + row * vl;
    if (comm_ == nullptr) {
        full->resize(static_cast<size_t>(Vocab()));
        R4DX_HIP_CHECK(hipMemcpy(full->data(), src, static_cast<size_t>(Vocab()) * sizeof(float),
                                 hipMemcpyDeviceToHost));
        return;
    }
    gather_host_.resize(static_cast<size_t>(vl));
    R4DX_HIP_CHECK(hipMemcpy(gather_host_.data(), src, static_cast<size_t>(vl) * sizeof(float),
                             hipMemcpyDeviceToHost));
    full->resize(static_cast<size_t>(vl * comm_->World()));
    comm_->HostAllGather(gather_host_.data(), static_cast<size_t>(vl) * sizeof(float),
                         full->data());
    full->resize(static_cast<size_t>(Vocab()));
}

void Inkling2Model::TpWarmup() {
    if (comm_ == nullptr)
        throw std::logic_error("Inkling2Model::TpWarmup: only a tensor-parallel rank warms up");
    const int64_t n = std::min<int64_t>(max_chunk_, 64);
    if (max_ctx_ < n + 3)
        throw std::invalid_argument("Inkling2Model::TpWarmup: max_ctx too small for the warm-up");
    std::vector<int32_t> ids(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % Vocab());
    (void)Prefill(ids);
    (void)DecodeStepGreedy(0);
    kernels::SampleParams sp;
    std::mt19937_64 rng(0x7e57);
    (void)DecodeStepSampled(1, sp, rng);
    (void)DecodeStep(2);
    Reset();
}

// ---- profiling
// ------------------------------------------------------------------------------------------------------

namespace {
std::vector<ProfileEntry> ToEntries(std::vector<SpanEntry> raw) {
    std::vector<ProfileEntry> out;
    out.reserve(raw.size());
    for (auto& e : raw) out.push_back({std::move(e.name), e.ms, e.count});
    return out;
}
}  // namespace

StepProfile Inkling2Model::DecodeStepProfiled(int32_t token_id) {
    RequireNotTp("DecodeStepProfiled");
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    r4dx_kernel_launch_counter_reset();
    SpanAccumulator acc;
    RunChunk({token_id}, true, nullptr, nullptr, &acc);
    const auto enq = Clock::now();
    StepProfile sp;
    sp.entries = ToEntries(acc.Finish());
    const auto fin = Clock::now();
    for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
    sp.host_enqueue_ms = std::chrono::duration<double, std::milli>(enq - t0).count();
    sp.finish_wait_ms = std::chrono::duration<double, std::milli>(fin - enq).count();
    sp.wall_ms = std::chrono::duration<double, std::milli>(fin - t0).count();
    sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();
    return sp;
}

StepProfile Inkling2Model::PrefillProfiled(const std::vector<int32_t>& token_ids) {
    RequireNotTp("PrefillProfiled");
    using Clock = std::chrono::steady_clock;
    if (token_ids.empty())
        throw std::runtime_error("Inkling2Model::PrefillProfiled: token_ids is empty");
    const auto t0 = Clock::now();
    r4dx_kernel_launch_counter_reset();
    SpanAccumulator acc;
    std::vector<SpanEntry> merged;
    for (size_t off = 0; off < token_ids.size();) {
        const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
        const std::vector<int32_t> chunk(token_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                         token_ids.begin() + static_cast<std::ptrdiff_t>(off + n));
        off += n;
        RunChunk(chunk, false, nullptr, nullptr, &acc);
        for (SpanEntry& e : acc.Finish()) {
            auto it = std::find_if(merged.begin(), merged.end(),
                                   [&](const SpanEntry& m) { return m.name == e.name; });
            if (it == merged.end()) {
                merged.push_back(std::move(e));
            } else {
                it->ms += e.ms;
                it->count += e.count;
            }
        }
    }
    StepProfile sp;
    sp.entries = ToEntries(std::move(merged));
    for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
    sp.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();
    return sp;
}

}  // namespace r4dx::model
