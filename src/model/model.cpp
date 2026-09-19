#include "model.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "embedding.h"
#include "final_lm_head.h"
#include "gdn_layer.h"
#include "mlp.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/types.hpp"

namespace r4dx::model {

namespace {

// Minimal hipEvent-pair span accumulator for Model::DecodeStepProfiled (tools/profile pass,
// 2026-09-19). Spans are recorded async (no per-span sync -- that would serialize the pipeline and
// skew exactly the numbers this is trying to measure); Finish() does ONE hipEventSynchronize at
// the very end of the step, then reads back every pair's elapsedTime and accumulates it into the
// named bucket (several calls under the same name, e.g. one per GDN layer, sum into one entry).
class SpanAccumulator {
 public:
  void Add(hipStream_t stream, const std::string& name,
            const std::function<void()>& body) {
    hipEvent_t start, end;
    R4DX_HIP_CHECK(hipEventCreate(&start));
    R4DX_HIP_CHECK(hipEventCreate(&end));
    R4DX_HIP_CHECK(hipEventRecord(start, stream));
    body();
    R4DX_HIP_CHECK(hipEventRecord(end, stream));
    pending_.push_back({name, start, end});
  }

  std::vector<Model::ProfileEntry> Finish() {
    if (!pending_.empty()) {
      R4DX_HIP_CHECK(hipEventSynchronize(pending_.back().end));
    }
    std::vector<Model::ProfileEntry> out;
    std::unordered_map<std::string, size_t> index;
    for (auto& p : pending_) {
      float ms = 0.0f;
      R4DX_HIP_CHECK(hipEventElapsedTime(&ms, p.start, p.end));
      static_cast<void>(hipEventDestroy(p.start));
      static_cast<void>(hipEventDestroy(p.end));
      auto it = index.find(p.name);
      if (it == index.end()) {
        index[p.name] = out.size();
        out.push_back({p.name, static_cast<double>(ms), 1});
      } else {
        out[it->second].ms += ms;
        out[it->second].count += 1;
      }
    }
    pending_.clear();
    return out;
  }

 private:
  struct Pending {
    std::string name;
    hipEvent_t start, end;
  };
  std::vector<Pending> pending_;
};

}  // namespace

Model Model::Load(const ModelOptions& opts) {
  Model m;
  m.container_ = Container::Load(opts.container_path, opts.layout, opts.layout, opts.layer_limit,
                                  opts.mtp_head_layout.value_or(opts.layout),
                                  opts.embed_device_resident);
  const ModelConfig& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = m.container_.NumLoadedLayers();

  if (opts.mtp_draft_k > 0 && !m.container_.HasMtp()) {
    throw std::runtime_error(
        "Model::Load: mtp_draft_k > 0 but the container has no mtp.* weights (convert with "
        "--mtp on)");
  }
  m.mtp_draft_k_ = opts.mtp_draft_k;
  // GDN's per-sequence window bank (gdn_state.h's file comment) must be sized for the largest
  // speculative-verify window this Model will ever run: mtp_draft_k drafts plus the seed token.
  // ==1 (window index 0 only) when MTP is disabled -- every decode call then degenerates exactly
  // to this class's pre-MTP behavior.
  const int64_t max_decode_window = 1 + opts.mtp_draft_k;

  m.max_chunk_ = 64;
  m.embed_staging_ = core::PinnedBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.embed_ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.embed_ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.buf_a_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.buf_b_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(cfg.vocab_size));
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  m.attn_positions_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.attn_seqused_k_ = core::DeviceBuffer<int32_t>(1);
  // Per-layer activation scratch (rmsnorm output, gate_up, GDN conv/kkt/chunk-scan buffers,
  // activation-quant scratch): a few MB at T<=64 (see linear.h/gdn_layer.cpp's own buffer sizes).
  // 96MB gives headroom without materially affecting the ~15-35GB the weights themselves occupy.
  m.arena_.Reserve(96ull * 1024 * 1024);

  const auto adims = core::r4d::GetAttnDims();  // head_dim=256, gqa=6, block_size=16

  m.gdn_states_.resize(static_cast<size_t>(num_layers));
  m.kv_caches_.resize(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    if (cfg.IsGdnLayer(i)) {
      m.gdn_states_[static_cast<size_t>(i)].emplace(
          /*max_seqs=*/1, /*H=*/cfg.linear_num_value_heads, /*V=*/cfg.linear_value_head_dim,
          /*K=*/cfg.linear_key_head_dim, cfg.ConvDim(), cfg.linear_conv_kernel_dim,
          max_decode_window);
      m.gdn_states_[static_cast<size_t>(i)]->ZeroAll(m.stream_);
    } else {
      m.kv_caches_[static_cast<size_t>(i)].emplace(
          static_cast<int>(cfg.num_key_value_heads), static_cast<int>(cfg.head_dim),
          adims.block_size, static_cast<int>(opts.max_ctx));
    }
  }

  if (opts.mtp_draft_k > 0) {
    m.mtp_.emplace(cfg, m.container_.Mtp(), opts.mtp_draft_k, opts.max_ctx);
    m.mtp_seed_hidden_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(hidden));
    m.mtp_num_accepted_dev_ = core::DeviceBuffer<int32_t>(1);
    m.mtp_logits_dev_ =
        core::DeviceBuffer<float>(static_cast<size_t>((opts.mtp_draft_k + 1) * cfg.vocab_size));
    m.mtp_argmax_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(opts.mtp_draft_k + 1));
  }

  m.stream_.Synchronize();
  return m;
}

std::vector<float> Model::RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                                    bool want_logits, int32_t* greedy_token_out) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Model::RunChunk: token_ids.size() must be in [1, " +
                              std::to_string(max_chunk_) + "]");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;

  // Device-resident draft loop (docs/mtp.md "device-resident draft loop"): when the container's
  // text.embed_tokens has a VRAM mirror (Container::EmbedTokensDeviceResident()), gather straight
  // from it via a device kernel instead of a host memcpy + async H2D of the whole [T,hidden]
  // staging buffer -- only a small [T] int32 id array needs to cross the H2D boundary. Falls back
  // to the original host-gather path when the container was loaded with embed_device_resident=
  // false or the free-VRAM heuristic decided the mirror would not fit (Container::Load's comment).
  if (container_.EmbedTokensDeviceResident()) {
    std::copy(token_ids.begin(), token_ids.end(), embed_ids_host_.begin());
    embed_ids_dev_.CopyFromHostAsync(embed_ids_host_.data(), static_cast<size_t>(T), stream_);
    EmbedTokensDeviceGather(stream_, container_.EmbedTokensDevice(), hidden, embed_ids_dev_.data(),
                             T, buf_a_.data());
  } else {
    EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, token_ids,
                embed_staging_, buf_a_);
  }

  // Full-attention layers' positions/slot_mapping (== pos_+t, see model.h) and seqused_k (==
  // pos_+T) are identical for every attention layer in this chunk -- upload them once here rather
  // than once per attention layer. Plain (blocking) hipMemcpy is safe at this specific point only:
  // the previous RunChunk call (if any) ended with stream_.Synchronize() before returning its
  // logits, so the device is guaranteed idle here and nothing can race this write -- see
  // AttentionLayer::Forward's doc comment, which requires exactly that (a non-arena, ordered
  // upload) for these two buffers.
  std::vector<int32_t> positions_h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(pos_ + t);
  attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
  const int32_t seqused_k_h = static_cast<int32_t>(pos_ + T);
  attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

  uint16_t* cur = buf_a_.data();
  uint16_t* other = buf_b_.data();

  for (int64_t i = 0; i < num_layers; ++i) {
    const LayerWeights& lw = container_.Layer(i);

    if (cfg.IsGdnLayer(i)) {
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
      GdnLayerParams p;
      p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
      p.is_prefill = is_prefill_path;
      p.has_init = has_init;
      // Plain-decode/MTP-desync guard (review finding, 2026-09-19): a decode step run through this
      // method (DecodeStep/DecodeStepGreedy, is_prefill_path==false) on an MTP-enabled Model must
      // seed from wherever the LAST verify round (DecodeStepMtpGreedy) actually left the window
      // state -- nullptr would silently reseed from window index 0 regardless of that round's own
      // num_accepted, desyncing GDN state from the real committed sequence. VerifyWindow/
      // DecodeStepMtpGreedy never call RunChunk at all, so this is the only place plain decode
      // needs to thread it. See the `mtp_` block below RunChunk's main loop, which sets
      // mtp_num_accepted_dev_=1/mtp_num_accepted_valid_=true right after this call commits, so the
      // NEXT plain decode step (or the next MTP round) seeds correctly regardless of which kind of
      // step follows.
      p.num_accepted =
          (!is_prefill_path && mtp_ && mtp_num_accepted_valid_) ? mtp_num_accepted_dev_.data()
                                                                  : nullptr;
      layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur, cur,
                    T, p);
    } else {
      attention::AttnConfig acfg;
      acfg.hidden = static_cast<int>(hidden);
      acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
      acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
      acfg.head_dim = static_cast<int>(cfg.head_dim);
      acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
      acfg.rope_theta = static_cast<float>(cfg.rope_theta);
      acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
      attention::AttentionLayer layer(acfg);

      attention::AttnWeights aw;
      aw.input_layernorm = lw.input_layernorm.data();
      aw.qg = &lw.attn->qg;
      aw.k_w = lw.attn->k.data();
      aw.v_w = lw.attn->v.data();
      aw.o = &lw.attn->o;
      aw.q_norm = lw.attn->q_norm.data();
      aw.k_norm = lw.attn->k_norm.data();
      aw.k_descale = lw.attn->k_descale.data();
      aw.v_descale = lw.attn->v_descale.data();

      layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)],
                    static_cast<int>(T), static_cast<int>(pos_), attn_positions_.data(),
                    attn_seqused_k_.data(), stream_.get());
      std::swap(cur, other);
    }

    Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
    mlp.Forward(stream_, arena_, cur, cur, T);

    arena_.Reset();
  }

  // ---- MTP lockstep KV priming (docs/mtp.md, mtp_head.h's PrimeKv comment) ----------------------
  // Extends MtpHead's own KV cache by exactly the real positions THIS call just made knowable --
  // unconditionally (not gated by want_logits: a prefill chunk that discards its own logits still
  // reveals real "next tokens" for MTP's boundary/within-chunk (h_i, t_{i+1}) pairs). No-op at
  // exactly zero extra cost when mtp_ is unset (the overwhelming common case).
  if (mtp_) {
    if (mtp_seed_valid_) {
      // The ONE position left dangling by the previous RunChunk/DecodeStepMtpGreedy call: h_i =
      // that call's own last-row hidden state (mtp_seed_hidden_), t_{i+1} = THIS call's own first
      // input token.
      mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_ - 1, mtp_seed_hidden_.data(),
                    {token_ids[0]}, container_.EmbedTokensHost(), cfg.vocab_size);
    }
    if (T > 1) {
      // Within-chunk pairs: h_i = this chunk's own rows 0..T-2 (`cur`, unmodified since the layer
      // loop above finished), t_{i+1} = this chunk's own token_ids[1..T-1]. Row T-1 is left
      // dangling for the NEXT call, exactly like the boundary case above.
      const std::vector<int32_t> next_toks(token_ids.begin() + 1, token_ids.end());
      mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_, cur, next_toks,
                    container_.EmbedTokensHost(), cfg.vocab_size);
    }
    R4DX_HIP_CHECK(hipMemcpyAsync(mtp_seed_hidden_.data(), cur + (T - 1) * hidden,
                                   static_cast<size_t>(hidden) * sizeof(uint16_t),
                                   hipMemcpyDeviceToDevice, stream_.get()));
    mtp_seed_valid_ = true;
    arena_.Reset();
  }

  // Prefill discards every chunk's logits except the last (Prefill() below only keeps the final
  // RunChunk's return value) -- skip final_norm + the full-vocab lm_head GEMM + the bf16->fp32
  // widen + the 1MB D2H copy for a chunk whose logits nobody reads. On a long prompt this is the
  // difference between one full-vocab GEMM (~2.5GB bf16 lm_head weight read) and one per 64-token
  // chunk. The stream sync below still runs unconditionally (not folded into `want_logits`): the
  // NEXT RunChunk call's top-of-function attn_positions_/attn_seqused_k_ upload is a plain
  // (blocking) hipMemcpy that is only race-free because it assumes the device is fully idle when
  // it runs (see that comment) -- skipping the sync here would silently break that invariant for
  // every chunk but the last.
  if (want_logits) {
    FinalLmHead head(cfg, container_.FinalNorm(), container_.LmHead());
    head.Forward(stream_, arena_, cur + (T - 1) * hidden, logits_dev_.data(), /*T=*/1);
    // Greedy path (host-overhead pass, 2026-09-19): argmax logits_dev_ ON DEVICE while it's still
    // hot, so the only D2H this call ever does is 4 bytes instead of vocab*4 -- see
    // r4dx_argmax_f32 (src/kernels) and DecodeStepGreedy's own comment (model.h).
    if (greedy_token_out != nullptr) {
      r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                       reinterpret_cast<int64_t>(argmax_dev_.data()), cfg.vocab_size,
                       reinterpret_cast<int64_t>(stream_.get()));
    }
    arena_.Reset();
  }

  // stream_ is created with hipStreamNonBlocking (r4dx::core::Stream's default), which by design
  // does NOT implicitly synchronize against the legacy/null stream a plain (no-stream-argument)
  // hipMemcpy uses. logits_dev_.CopyToHost()/argmax_dev_.CopyToHost() below (and the next call's
  // attn_positions_/attn_seqused_k_ upload) is exactly that plain, synchronous D2H/H2D hipMemcpy --
  // it must be preceded by an explicit stream_.Synchronize() (not followed by one) or it can start
  // copying before this chunk's kernels (queued on stream_ above) have actually finished, silently
  // reading/overwriting stale/in-flight data. Same class of bug as the one fixed in
  // gdn_layer.cpp's UploadArray (see that file's comment) -- any plain hipMemcpy/
  // DeviceBuffer::CopyToHost/CopyFromHost call against a hipStreamNonBlocking stream's output
  // needs an explicit wait first, never an implicit one.
  stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits) {
    if (greedy_token_out != nullptr) {
      argmax_dev_.CopyToHost(greedy_token_out, 1);
    } else {
      logits.resize(static_cast<size_t>(cfg.vocab_size));
      logits_dev_.CopyToHost(logits.data(), logits.size());
    }
  }

  // Plain-decode/MTP-desync guard (review finding, 2026-09-19; see the p.num_accepted comment
  // above in the GDN branch for the read side): a plain decode step through this method always
  // "accepts" exactly the one token it was given, so GDN's NEXT call (whether another plain decode
  // step or the next MTP verify round) must seed from window index 0 of THIS step's own commit --
  // num_accepted=1 encodes exactly that (gdn_state.h's file comment: sidx[naccept-1] with naccept=1
  // reads window index 0, the slot this step just wrote). Safe here (device is idle, the sync
  // above already waited for every kernel that read the PREVIOUS value of
  // mtp_num_accepted_dev_). Prefill never touches this: mtp_num_accepted_valid_ is reset by
  // Prefill() itself for the first verify round after a fresh prefill.
  if (!is_prefill_path && mtp_) {
    const int32_t one = 1;
    mtp_num_accepted_dev_.CopyFromHost(&one, 1);
    mtp_num_accepted_valid_ = true;
  }

  pos_ += T;
  started_ = true;
  return logits;
}

std::vector<float> Model::Prefill(const std::vector<int32_t>& token_ids) {
  if (token_ids.empty()) throw std::runtime_error("Model::Prefill: token_ids is empty");
  // Prefill's chunked-scan GDN path always lands its result at window index 0 (GdnLayerParams::slot
  // == GdnStateManager::SlotForSeq, never a windowed verify slot -- see that class's file comment),
  // exactly what a nullptr-seeded MTP verify call reads. Any num_accepted carried from a PRIOR
  // generation (e.g. --chat's previous turn) refers to THAT turn's now-stale window slots, not this
  // fresh prefill's -- drop it so the first verify round after this call seeds correctly.
  mtp_num_accepted_valid_ = false;
  std::vector<float> logits;
  for (size_t off = 0; off < token_ids.size(); off += static_cast<size_t>(max_chunk_)) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                      token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const bool is_last_chunk = (off + n) == token_ids.size();
    std::vector<float> chunk_logits = RunChunk(chunk, /*is_prefill_path=*/true,
                                                /*want_logits=*/is_last_chunk);
    if (is_last_chunk) logits = std::move(chunk_logits);
  }
  return logits;
}

std::vector<float> Model::DecodeStep(int32_t token_id) {
  return RunChunk({token_id}, /*is_prefill_path=*/false, /*want_logits=*/true);
}

int32_t Model::DecodeStepGreedy(int32_t token_id) {
  int32_t next = -1;
  RunChunk({token_id}, /*is_prefill_path=*/false, /*want_logits=*/true, &next);
  return next;
}

// Coarser granularity than the task's ideal ("each GEMM by (N,K)", separate GDN-kernel/norm/rope/
// quant entries): this pass's time budget only extends to instrumenting the per-BLOCK boundaries
// model.cpp already calls directly (GdnLayer::Forward, AttentionLayer::Forward, Mlp::Forward,
// FinalLmHead::Forward each run one or more GEMMs plus several small kernels internally, all
// folded into that block's one entry here) rather than also threading a profiler handle into
// gdn_layer.cpp/attention_layer.hpp/mlp.cpp/final_lm_head.cpp to split GEMM-by-shape from
// norm/rope/quant within each block -- see docs/perf.md's profile table and "Known gaps" note for
// what this does and does not break out, and tools/profile/README.md for how to extend it.
Model::StepProfile Model::DecodeStepProfiled(int32_t token_id) {
  using Clock = std::chrono::steady_clock;
  const auto wall_t0 = Clock::now();

  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;
  const hipStream_t s = stream_.get();

  SpanAccumulator acc;

  acc.Add(s, "embed", [&] {
    EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, {token_id},
                embed_staging_, buf_a_);
  });

  std::vector<int32_t> positions_h = {static_cast<int32_t>(pos_)};
  attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
  const int32_t seqused_k_h = static_cast<int32_t>(pos_ + 1);
  attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

  uint16_t* cur = buf_a_.data();
  uint16_t* other = buf_b_.data();

  for (int64_t i = 0; i < num_layers; ++i) {
    const LayerWeights& lw = container_.Layer(i);

    if (cfg.IsGdnLayer(i)) {
      acc.Add(s, "gdn_layers (GDN kernels + in/out_proj GEMMs)", [&] {
        GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
        GdnLayerParams p;
        p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
        p.is_prefill = false;
        p.has_init = has_init;
        layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur,
                      cur, 1, p);
      });
    } else {
      acc.Add(s, "attn_layers (attention kernels + qg/k/v/o GEMMs)", [&] {
        attention::AttnConfig acfg;
        acfg.hidden = static_cast<int>(hidden);
        acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
        acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
        acfg.head_dim = static_cast<int>(cfg.head_dim);
        acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
        acfg.rope_theta = static_cast<float>(cfg.rope_theta);
        acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
        attention::AttentionLayer layer(acfg);

        attention::AttnWeights aw;
        aw.input_layernorm = lw.input_layernorm.data();
        aw.qg = &lw.attn->qg;
        aw.k_w = lw.attn->k.data();
        aw.v_w = lw.attn->v.data();
        aw.o = &lw.attn->o;
        aw.q_norm = lw.attn->q_norm.data();
        aw.k_norm = lw.attn->k_norm.data();
        aw.k_descale = lw.attn->k_descale.data();
        aw.v_descale = lw.attn->v_descale.data();

        layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)], 1,
                      static_cast<int>(pos_), attn_positions_.data(), attn_seqused_k_.data(), s);
      });
      std::swap(cur, other);
    }

    acc.Add(s, "mlp (gate_up/down GEMMs + silu)", [&] {
      Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
      mlp.Forward(stream_, arena_, cur, cur, 1);
    });

    arena_.Reset();
  }

  acc.Add(s, "final_norm+lm_head (full-vocab GEMM + widen)", [&] {
    FinalLmHead head(cfg, container_.FinalNorm(), container_.LmHead());
    head.Forward(stream_, arena_, cur, logits_dev_.data(), /*T=*/1);
    r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                     reinterpret_cast<int64_t>(argmax_dev_.data()), cfg.vocab_size,
                     reinterpret_cast<int64_t>(s));
    arena_.Reset();
  });

  // Every kernel `acc` recorded above was already ENQUEUED (asynchronously) by this point --
  // `host_enqueue_ms` below is the pure host-side cost of issuing them (CPU work like the
  // embedding gather plus launch overhead), not GPU-blocked. acc.Finish() then does ONE
  // hipEventSynchronize (blocks until that already-queued GPU work actually completes) and reads
  // back every event pair's elapsed time; the subsequent 4-byte D2H is then safe with no further
  // wait (the GPU is already idle). Both are host-chrono-measured together as `finish_wait_ms` --
  // see StepProfile's own comment (model.h) for why this must NOT be summed with `gpu_sum_ms`
  // (they measure overlapping time from two different clocks, not sequential costs).
  const auto enqueue_done_t = Clock::now();
  std::vector<Model::ProfileEntry> entries = acc.Finish();  // blocks until the GPU is idle
  int32_t next = -1;
  argmax_dev_.CopyToHost(&next, 1);  // GPU already idle (Finish() above already waited) -- safe
  const auto finish_t = Clock::now();

  const auto wall_t1 = Clock::now();
  StepProfile sp;
  sp.entries = std::move(entries);
  for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
  sp.host_enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_done_t - wall_t0).count();
  sp.finish_wait_ms = std::chrono::duration<double, std::milli>(finish_t - enqueue_done_t).count();
  sp.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

  pos_ += 1;
  started_ = true;
  return sp;
}

// ---- MTP self-speculation (docs/mtp.md) --------------------------------------------------------

std::vector<int32_t> Model::VerifyWindow(const std::vector<int32_t>& candidates,
                                          std::vector<float>* logits_out) {
  if (!mtp_) {
    throw std::runtime_error(
        "Model::VerifyWindow: MTP is not enabled on this Model (Load() with "
        "ModelOptions::mtp_draft_k > 0 and a container that has mtp.* weights)");
  }
  const int64_t T = static_cast<int64_t>(candidates.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Model::VerifyWindow: candidates.size() must be in [1, " +
                              std::to_string(max_chunk_) + "]");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;  // always true: VerifyWindow only ever runs after a Prefill

  // Same device-resident gather as RunChunk (model.cpp's own comment above) -- VerifyWindow is the
  // other hot-path caller (once per MTP round, docs/mtp.md).
  if (container_.EmbedTokensDeviceResident()) {
    std::copy(candidates.begin(), candidates.end(), embed_ids_host_.begin());
    embed_ids_dev_.CopyFromHostAsync(embed_ids_host_.data(), static_cast<size_t>(T), stream_);
    EmbedTokensDeviceGather(stream_, container_.EmbedTokensDevice(), hidden, embed_ids_dev_.data(),
                             T, buf_a_.data());
  } else {
    EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, candidates,
                embed_staging_, buf_a_);
  }

  // Same reasoning/hazard as RunChunk's own upload (model.cpp's RunChunk comment): safe here
  // because the previous call (Prefill/DecodeStep*/VerifyWindow) always ends with
  // stream_.Synchronize() before returning.
  std::vector<int32_t> positions_h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(pos_ + t);
  attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
  const int32_t seqused_k_h = static_cast<int32_t>(pos_ + T);
  attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

  const int32_t* num_accepted_ptr = mtp_num_accepted_valid_ ? mtp_num_accepted_dev_.data() : nullptr;

  uint16_t* cur = buf_a_.data();
  uint16_t* other = buf_b_.data();

  for (int64_t i = 0; i < num_layers; ++i) {
    const LayerWeights& lw = container_.Layer(i);

    if (cfg.IsGdnLayer(i)) {
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
      GdnLayerParams p;
      p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
      p.is_prefill = false;
      p.has_init = has_init;
      p.num_accepted = num_accepted_ptr;
      layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur, cur,
                    T, p);
    } else {
      attention::AttnConfig acfg;
      acfg.hidden = static_cast<int>(hidden);
      acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
      acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
      acfg.head_dim = static_cast<int>(cfg.head_dim);
      acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
      acfg.rope_theta = static_cast<float>(cfg.rope_theta);
      acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
      attention::AttentionLayer layer(acfg);

      attention::AttnWeights aw;
      aw.input_layernorm = lw.input_layernorm.data();
      aw.qg = &lw.attn->qg;
      aw.k_w = lw.attn->k.data();
      aw.v_w = lw.attn->v.data();
      aw.o = &lw.attn->o;
      aw.q_norm = lw.attn->q_norm.data();
      aw.k_norm = lw.attn->k_norm.data();
      aw.k_descale = lw.attn->k_descale.data();
      aw.v_descale = lw.attn->v_descale.data();

      // Note: this call unconditionally writes K/V for every one of the T candidate positions into
      // the paged cache BEFORE the attention math runs (AttentionLayer::Forward's own doc comment),
      // including whichever candidates turn out to be rejected below -- harmless (docs/mtp.md,
      // this class's own file comment): pos_ only ever advances by however many candidates
      // DecodeStepMtpGreedy actually commits, so a rejected candidate's stale KV entry is
      // unconditionally overwritten the next time that same slot (== that same position) is
      // written, before anything could ever read it.
      layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)],
                    static_cast<int>(T), static_cast<int>(pos_), attn_positions_.data(),
                    attn_seqused_k_.data(), stream_.get());
      std::swap(cur, other);
    }

    Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
    mlp.Forward(stream_, arena_, cur, cur, T);

    arena_.Reset();
  }

  // Per-position logits + greedy argmax, for every one of the T candidate positions (not just the
  // last -- this is what distinguishes a verify window from Prefill/DecodeStep's own tail-only
  // want_logits path).
  FinalLmHead head(cfg, container_.FinalNorm(), container_.LmHead());
  head.Forward(stream_, arena_, cur, mtp_logits_dev_.data(), T);
  for (int64_t t = 0; t < T; ++t) {
    r4dx_argmax_f32(
        reinterpret_cast<int64_t>(mtp_logits_dev_.data() + t * cfg.vocab_size),
        reinterpret_cast<int64_t>(mtp_argmax_dev_.data() + t), cfg.vocab_size,
        reinterpret_cast<int64_t>(stream_.get()));
  }
  arena_.Reset();

  mtp_last_hidden_ = cur;  // valid until the next RunChunk/VerifyWindow's EmbedTokens overwrites it

  stream_.Synchronize();  // same hazard class as RunChunk's own D2H -- see that method's comment

  std::vector<int32_t> preds(static_cast<size_t>(T));
  mtp_argmax_dev_.CopyToHost(preds.data(), preds.size());
  if (logits_out != nullptr) {
    logits_out->resize(static_cast<size_t>(T * cfg.vocab_size));
    mtp_logits_dev_.CopyToHost(logits_out->data(), logits_out->size());
  }
  return preds;
}

std::vector<int32_t> Model::DecodeStepMtpGreedy(int32_t token_id, int64_t k) {
  if (!mtp_) {
    throw std::runtime_error(
        "Model::DecodeStepMtpGreedy: MTP is not enabled on this Model (Load() with "
        "ModelOptions::mtp_draft_k > 0 and a container that has mtp.* weights)");
  }
  if (k < 0 || k > mtp_draft_k_) {
    throw std::runtime_error("Model::DecodeStepMtpGreedy: k must be in [0, " +
                              std::to_string(mtp_draft_k_) + "]");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;

  std::vector<int32_t> drafts;
  if (k > 0) {
    // base_pos: the REAL sequence position step 0 writes into MTP's own KV cache -- pos_-1 is
    // exactly the ONE position PrimeKv would otherwise prime from (mtp_seed_hidden_, token_id) if
    // this call drafted nothing at all (see mtp_head.h's PrimeKv comment and model.cpp's RunChunk).
    drafts = mtp_->Draft(stream_, arena_, cfg, container_.Mtp(), mtp_seed_hidden_.data(), token_id,
                          container_.EmbedTokensHost(),
                          container_.EmbedTokensDeviceResident() ? container_.EmbedTokensDevice()
                                                                  : nullptr,
                          cfg.vocab_size, container_.LmHead(), k,
                          /*base_pos=*/pos_ - 1);
    arena_.Reset();
  } else if (mtp_seed_valid_) {
    // k==0 degenerate call (doc's own "single DecodeStepGreedy-equivalent" case): Draft() is
    // skipped entirely, so nothing above would otherwise prime MTP's KV at position pos_-1 --
    // without this, a later k>0 call on the same Model would compute a correct base_pos but find
    // that position's KV entry missing (never written), breaking lockstep continuity. Mirrors
    // exactly what Draft()'s own step 0 would have computed.
    mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_ - 1, mtp_seed_hidden_.data(),
                  {token_id}, container_.EmbedTokensHost(), cfg.vocab_size);
    arena_.Reset();
  }

  std::vector<int32_t> candidates;
  candidates.reserve(drafts.size() + 1);
  candidates.push_back(token_id);
  candidates.insert(candidates.end(), drafts.begin(), drafts.end());

  const std::vector<int32_t> preds = VerifyWindow(candidates);  // size == candidates.size()

  // Greedy acceptance: the longest prefix of drafts whose own predecessor's argmax matches it.
  // preds[i] is the real model's own next-token prediction after processing candidate i
  // (candidates[0]==token_id, candidates[1..]==drafts) -- so preds[i] should equal drafts[i] (the
  // draft that FOLLOWS candidate i) for the draft to be confirmed.
  int64_t num_accepted_drafts = 0;
  while (num_accepted_drafts < static_cast<int64_t>(drafts.size()) &&
         preds[static_cast<size_t>(num_accepted_drafts)] ==
             drafts[static_cast<size_t>(num_accepted_drafts)]) {
    ++num_accepted_drafts;
  }
  const int32_t corrected = preds[static_cast<size_t>(num_accepted_drafts)];
  const int64_t num_committed = num_accepted_drafts + 1;  // +1 for token_id itself

  // Thread this round's acceptance count into the NEXT GDN decode/verify call (gdn_state.h's file
  // comment) and advance pos_ by exactly what was committed -- NOT by candidates.size(), which
  // would silently accept every draft regardless of whether the real model agreed.
  const int32_t num_committed_i32 = static_cast<int32_t>(num_committed);
  mtp_num_accepted_dev_.CopyFromHost(&num_committed_i32, 1);
  mtp_num_accepted_valid_ = true;
  pos_ += num_committed;
  started_ = true;

  // Reseed MTP's h_seed from the row that produced `corrected` -- candidate index
  // num_accepted_drafts of THIS call's window, i.e. mtp_last_hidden_'s row num_accepted_drafts.
  R4DX_HIP_CHECK(hipMemcpyAsync(
      mtp_seed_hidden_.data(), mtp_last_hidden_ + num_accepted_drafts * hidden,
      static_cast<size_t>(hidden) * sizeof(uint16_t), hipMemcpyDeviceToDevice, stream_.get()));
  stream_.Synchronize();

  std::vector<int32_t> result(drafts.begin(), drafts.begin() + num_accepted_drafts);
  result.push_back(corrected);
  return result;
}

}  // namespace r4dx::model
