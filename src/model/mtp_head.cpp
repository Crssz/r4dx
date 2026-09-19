#include "mtp_head.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "embedding.h"
#include "final_lm_head.h"
#include "linear.h"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {
constexpr int kGemmWV = 4, kGemmSK = 4, kGemmMB = 1;  // gdn_layer.cpp's plain-bf16-linear constants
}  // namespace

MtpHead::MtpHead(const ModelConfig& cfg, const MtpWeights& w, int64_t max_draft, int64_t max_ctx)
    : attn_layer_([&] {
        attention::AttnConfig acfg;
        acfg.hidden = static_cast<int>(cfg.hidden_size);
        acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
        acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
        acfg.head_dim = static_cast<int>(cfg.head_dim);
        acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
        acfg.rope_theta = static_cast<float>(cfg.rope_theta);
        acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
        return acfg;
      }()),
      // Real, growing per-sequence cache sized like every backbone attention layer's own
      // PagedKvCache -- see mtp_head.h's file comment for why this is no longer a tiny scratch
      // block reset every Draft() call.
      kv_(static_cast<int>(cfg.num_key_value_heads), static_cast<int>(cfg.head_dim),
          core::r4d::GetAttnDims().block_size, /*max_context_tokens=*/static_cast<int>(max_ctx)),
      embed_staging_host_(static_cast<size_t>(cfg.hidden_size)),
      embed_staging_dev_(static_cast<size_t>(cfg.hidden_size)),
      positions_(1),
      seqused_k_(1),
      logits_dev_(static_cast<size_t>(cfg.vocab_size)),
      argmax_dev_(1),
      prime_positions_host_(static_cast<size_t>(kMaxPrime)),
      prime_positions_dev_(static_cast<size_t>(kMaxPrime)),
      prime_seqused_host_(1),
      prime_seqused_dev_(1),
      prime_embed_host_(static_cast<size_t>(kMaxPrime * cfg.hidden_size)),
      prime_embed_dev_(static_cast<size_t>(kMaxPrime * cfg.hidden_size)) {
  (void)w;      // cfg/w size this object's buffers above; neither is stored -- see mtp_head.h
  (void)max_draft;  // no longer sizes kv_ (kv_ is now sized to max_ctx) -- kept as a documented
                     // upper bound on Draft()'s own `k` argument, enforced by Model, not here.
}

std::vector<int32_t> MtpHead::Draft(core::Stream& stream, core::Arena& arena,
                                     const ModelConfig& cfg, const MtpWeights& w,
                                     const uint16_t* h_seed, int32_t seed_token,
                                     const uint16_t* embed_table, int64_t vocab,
                                     const QuantLinear& lm_head, int64_t k, int64_t base_pos) {
  std::vector<int32_t> drafts;
  if (k <= 0) return drafts;
  drafts.reserve(static_cast<size_t>(k));

  const int64_t hidden = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const hipStream_t s = stream.get();

  // cur_hidden: device bf16 [hidden], the (h_seed, cur_token) pair's "h_i" -- h_seed for the first
  // draft, this layer's own previous-step output for every draft after that (see file comment).
  const uint16_t* cur_hidden = h_seed;
  uint16_t* prev_out = nullptr;  // owned scratch from a prior iteration, kept alive across the loop
  int32_t cur_token = seed_token;

  for (int64_t step = 0; step < k; ++step) {
    // ---- concat(pre_fc_norm_embedding(embed(cur_token)), pre_fc_norm_hidden(cur_hidden)) -------
    // Embedding occupies fc's FIRST `hidden` input columns, hidden state the SECOND -- see
    // mtp_head.h's file comment for the reference this follows and the incident this corrects.
    uint16_t* concat_buf = arena.Alloc<uint16_t>(static_cast<size_t>(2 * hidden));
    EmbedTokens(stream, embed_table, vocab, hidden, {cur_token}, embed_staging_host_,
                embed_staging_dev_);
    r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(embed_staging_dev_.data()),
                       reinterpret_cast<int64_t>(w.pre_fc_norm_embedding.data()),
                       reinterpret_cast<int64_t>(concat_buf), /*rows=*/1, hidden, eps,
                       reinterpret_cast<int64_t>(s));

    r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(cur_hidden),
                       reinterpret_cast<int64_t>(w.pre_fc_norm_hidden.data()),
                       reinterpret_cast<int64_t>(concat_buf + hidden), /*rows=*/1, hidden, eps,
                       reinterpret_cast<int64_t>(s));

    // ---- fc: [2*hidden] -> [hidden], plain bf16 linear (container-format.md: no .{layout} suffix)
    uint16_t* fc_out = arena.Alloc<uint16_t>(static_cast<size_t>(hidden));
    core::r4d::GemmBf16NtM64(concat_buf, w.fc.data(), fc_out, /*M=*/1,
                              static_cast<int>(2 * hidden), static_cast<int>(hidden), kGemmWV,
                              kGemmSK, kGemmMB, s);

    // ---- one full-attention decoder layer (mtp.layer's own weights + this head's own REAL,
    // lockstep-primed KV cache -- see mtp_head.h's file comment) -- writes into REAL sequence
    // position base_pos+step, exactly the position Model::RunChunk's own PrimeKv would prime next
    // if this round drafted nothing at all (model.cpp's DecodeStepMtpGreedy derives base_pos).
    const int32_t pos_h = static_cast<int32_t>(base_pos + step);
    positions_.CopyFromHost(&pos_h, 1);
    const int32_t seqused_h = pos_h + 1;
    seqused_k_.CopyFromHost(&seqused_h, 1);

    attention::AttnWeights aw;
    aw.input_layernorm = w.layer.input_layernorm.data();
    aw.qg = &w.layer.attn->qg;
    aw.k_w = w.layer.attn->k.data();
    aw.v_w = w.layer.attn->v.data();
    aw.o = &w.layer.attn->o;
    aw.q_norm = w.layer.attn->q_norm.data();
    aw.k_norm = w.layer.attn->k_norm.data();
    aw.k_descale = w.layer.attn->k_descale.data();
    aw.v_descale = w.layer.attn->v_descale.data();

    uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(hidden));
    attn_layer_.Forward(arena, fc_out, attn_out, aw, kv_, /*T=*/1, /*start_pos=*/pos_h,
                         positions_.data(), seqused_k_.data(), s);

    uint16_t* h_out = arena.Alloc<uint16_t>(static_cast<size_t>(hidden));
    Mlp mlp(cfg, w.layer.post_attention_layernorm, w.layer.mlp);
    mlp.Forward(stream, arena, attn_out, h_out, /*T=*/1);

    // ---- mtp.norm -> shared lm_head -> greedy argmax (on device, one 4-byte D2H per draft step) --
    FinalLmHead head(cfg, w.norm, lm_head);
    head.Forward(stream, arena, h_out, logits_dev_.data(), /*T=*/1);
    r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                     reinterpret_cast<int64_t>(argmax_dev_.data()), vocab,
                     reinterpret_cast<int64_t>(s));
    // Every buffer above (positions_/seqused_k_ H2D upload for the NEXT iteration, and reading
    // argmax_dev_ back) is a plain synchronous hipMemcpy against this hipStreamNonBlocking stream --
    // must be preceded by an explicit wait, same class of hazard as model.cpp's RunChunk (see that
    // file's comment).
    stream.Synchronize();
    int32_t next_token = -1;
    argmax_dev_.CopyToHost(&next_token, 1);

    drafts.push_back(next_token);
    cur_token = next_token;
    cur_hidden = h_out;
    prev_out = h_out;
    (void)prev_out;  // kept alive by the arena until Reset(); no separate ownership needed
  }
  return drafts;
}

void MtpHead::PrimeKv(core::Stream& stream, core::Arena& arena, const ModelConfig& cfg,
                      const MtpWeights& w, int64_t base_pos, const uint16_t* h_rows,
                      const std::vector<int32_t>& next_tokens, const uint16_t* embed_table,
                      int64_t vocab) {
  const int64_t n = static_cast<int64_t>(next_tokens.size());
  if (n <= 0) return;
  if (n > kMaxPrime) {
    throw std::runtime_error("MtpHead::PrimeKv: n must be <= " + std::to_string(kMaxPrime));
  }
  const int64_t hidden = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const hipStream_t s = stream.get();

  // pre_fc_norm_embedding/pre_fc_norm_hidden each need a CONTIGUOUS [n,hidden] output (no
  // strided-row GEMM/rmsnorm entry point exists in this codebase -- see linear.h's own comment),
  // so build the two halves into separate contiguous buffers, then splice them into concat_buf via
  // two strided D2D copies (hipMemcpy2DAsync) rather than a bespoke interleave kernel.
  uint16_t* embed_normed = arena.Alloc<uint16_t>(static_cast<size_t>(n * hidden));
  EmbedTokens(stream, embed_table, vocab, hidden, next_tokens, prime_embed_host_, prime_embed_dev_);
  r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(prime_embed_dev_.data()),
                     reinterpret_cast<int64_t>(w.pre_fc_norm_embedding.data()),
                     reinterpret_cast<int64_t>(embed_normed), n, hidden, eps,
                     reinterpret_cast<int64_t>(s));

  uint16_t* hidden_normed = arena.Alloc<uint16_t>(static_cast<size_t>(n * hidden));
  r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(h_rows),
                     reinterpret_cast<int64_t>(w.pre_fc_norm_hidden.data()),
                     reinterpret_cast<int64_t>(hidden_normed), n, hidden, eps,
                     reinterpret_cast<int64_t>(s));

  uint16_t* concat_buf = arena.Alloc<uint16_t>(static_cast<size_t>(n * 2 * hidden));
  const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(uint16_t);
  const size_t concat_pitch = 2 * row_bytes;
  R4DX_HIP_CHECK(hipMemcpy2DAsync(concat_buf, concat_pitch, embed_normed, row_bytes, row_bytes,
                                   static_cast<size_t>(n), hipMemcpyDeviceToDevice, s));
  R4DX_HIP_CHECK(hipMemcpy2DAsync(reinterpret_cast<uint8_t*>(concat_buf) + row_bytes, concat_pitch,
                                   hidden_normed, row_bytes, row_bytes, static_cast<size_t>(n),
                                   hipMemcpyDeviceToDevice, s));

  uint16_t* fc_out = arena.Alloc<uint16_t>(static_cast<size_t>(n * hidden));
  core::r4d::GemmBf16NtM64(concat_buf, w.fc.data(), fc_out, static_cast<int>(n),
                            static_cast<int>(2 * hidden), static_cast<int>(hidden), kGemmWV,
                            kGemmSK, kGemmMB, s);

  std::vector<int32_t> positions_h(static_cast<size_t>(n));
  for (int64_t t = 0; t < n; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(base_pos + t);
  std::copy(positions_h.begin(), positions_h.end(), prime_positions_host_.begin());
  prime_positions_dev_.CopyFromHostAsync(prime_positions_host_.data(), static_cast<size_t>(n),
                                          stream);
  prime_seqused_host_[0] = static_cast<int32_t>(base_pos + n);
  prime_seqused_dev_.CopyFromHostAsync(prime_seqused_host_.data(), 1, stream);

  attention::AttnWeights aw;
  aw.input_layernorm = w.layer.input_layernorm.data();
  aw.qg = &w.layer.attn->qg;
  aw.k_w = w.layer.attn->k.data();
  aw.v_w = w.layer.attn->v.data();
  aw.o = &w.layer.attn->o;
  aw.q_norm = w.layer.attn->q_norm.data();
  aw.k_norm = w.layer.attn->k_norm.data();
  aw.k_descale = w.layer.attn->k_descale.data();
  aw.v_descale = w.layer.attn->v_descale.data();

  // Output (h', the post-attention-sublayer activation) is discarded -- priming only needs the
  // K/V write (see this method's .h comment for why no mlp/norm/lm_head/chaining is needed here).
  uint16_t* discard_out = arena.Alloc<uint16_t>(static_cast<size_t>(n * hidden));
  attn_layer_.Forward(arena, fc_out, discard_out, aw, kv_, static_cast<int>(n),
                       static_cast<int>(base_pos), prime_positions_dev_.data(),
                       prime_seqused_dev_.data(), s);
}

}  // namespace r4dx::model
