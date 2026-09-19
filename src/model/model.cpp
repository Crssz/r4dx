#include "model.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "embedding.h"
#include "final_lm_head.h"
#include "gdn_layer.h"
#include "mlp.h"
#include "r4dx/core/r4d.hpp"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/types.hpp"

namespace r4dx::model {

Model Model::Load(const ModelOptions& opts) {
  Model m;
  m.container_ = Container::Load(opts.container_path, opts.layout, opts.layout, opts.layer_limit);
  const ModelConfig& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = m.container_.NumLoadedLayers();

  m.max_chunk_ = 64;
  m.embed_staging_ = core::PinnedBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.buf_a_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.buf_b_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(cfg.vocab_size));
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
          /*max_decode_window=*/1);
      m.gdn_states_[static_cast<size_t>(i)]->ZeroAll(m.stream_);
    } else {
      m.kv_caches_[static_cast<size_t>(i)].emplace(
          static_cast<int>(cfg.num_key_value_heads), static_cast<int>(cfg.head_dim),
          adims.block_size, static_cast<int>(opts.max_ctx));
    }
  }
  m.stream_.Synchronize();
  return m;
}

std::vector<float> Model::RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                                    bool want_logits) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Model::RunChunk: token_ids.size() must be in [1, " +
                              std::to_string(max_chunk_) + "]");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;

  EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, token_ids,
              embed_staging_, buf_a_);

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
      aw.qg_w = lw.attn->qg.bf16_w.data();
      aw.k_w = lw.attn->k.data();
      aw.v_w = lw.attn->v.data();
      aw.o_w = lw.attn->o.bf16_w.data();
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
    arena_.Reset();
  }

  // stream_ is created with hipStreamNonBlocking (r4dx::core::Stream's default), which by design
  // does NOT implicitly synchronize against the legacy/null stream a plain (no-stream-argument)
  // hipMemcpy uses. logits_dev_.CopyToHost() below (and the next call's attn_positions_/
  // attn_seqused_k_ upload) is exactly that plain, synchronous D2H/H2D hipMemcpy -- it must be
  // preceded by an explicit stream_.Synchronize() (not followed by one) or it can start copying
  // before this chunk's kernels (queued on stream_ above) have actually finished, silently
  // reading/overwriting stale/in-flight data. Same class of bug as the one fixed in
  // gdn_layer.cpp's UploadArray (see that file's comment) -- any plain hipMemcpy/
  // DeviceBuffer::CopyToHost/CopyFromHost call against a hipStreamNonBlocking stream's output
  // needs an explicit wait first, never an implicit one.
  stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits) {
    logits.resize(static_cast<size_t>(cfg.vocab_size));
    logits_dev_.CopyToHost(logits.data(), logits.size());
  }

  pos_ += T;
  started_ = true;
  return logits;
}

std::vector<float> Model::Prefill(const std::vector<int32_t>& token_ids) {
  if (token_ids.empty()) throw std::runtime_error("Model::Prefill: token_ids is empty");
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

}  // namespace r4dx::model
