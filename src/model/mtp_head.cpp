#include "mtp_head.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "attn_config.h"
#include "embedding.h"
#include "final_lm_head.h"
#include "linear.h"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/embedding.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {
constexpr int kGemmWV = 4, kGemmSK = 4, kGemmMB = 1;  // gdn_layer.cpp's plain-bf16-linear constants
}  // namespace

MtpHead::MtpHead(const ModelConfig& cfg, const MtpWeights& w, int64_t max_draft, int64_t max_ctx)
    : attn_layer_(MakeAttnConfig(cfg)),
      // Real, growing per-sequence cache sized like every backbone attention layer's own
      // PagedKvCache -- see mtp_head.h's file comment for why this is no longer a tiny scratch
      // block reset every Draft() call.
      kv_(static_cast<int>(cfg.num_key_value_heads), static_cast<int>(cfg.head_dim),
          core::r4d::GetAttnDims().block_size, /*max_context_tokens=*/static_cast<int>(max_ctx)),
      embed_staging_host_(static_cast<size_t>(cfg.hidden_size)),
      embed_staging_dev_(static_cast<size_t>(cfg.hidden_size)),
      logits_dev_(static_cast<size_t>(cfg.vocab_size)),
      argmax_dev_(1),
      subset_argmax_dev_(1),
      positions_dev_(static_cast<size_t>(max_draft)),
      positions_host_(static_cast<size_t>(max_draft)),
      seqused_dev_(static_cast<size_t>(max_draft)),
      seqused_host_(static_cast<size_t>(max_draft)),
      seed_token_dev_(1),
      draft_ids_dev_(static_cast<size_t>(max_draft)),
      draft_ids_host_(static_cast<size_t>(max_draft)),
      rope3_dev_(static_cast<size_t>(3 * max_draft)),
      rope3_host_(static_cast<size_t>(3 * max_draft)),
      max_draft_(max_draft),
      prime_positions_host_(static_cast<size_t>(kMaxPrime)),
      prime_positions_dev_(static_cast<size_t>(kMaxPrime)),
      // Sized to kMaxPrime (not 1) so host_staging_offset can address a disjoint scalar slot per
      // back-to-back call (RunChunk's boundary + within-chunk pair) -- see PrimeKv's .h doc
      // comment. prime_seqused_dev_ stays size 1: its destination is always offset 0 (device-side
      // reuse across calls is safe via HIP's own stream-issue-order guarantee).
      prime_seqused_host_(static_cast<size_t>(kMaxPrime)),
      prime_seqused_dev_(1),
      prime_embed_host_(static_cast<size_t>(kMaxPrime * cfg.hidden_size)),
      prime_embed_dev_(static_cast<size_t>(kMaxPrime * cfg.hidden_size)),
      prime_rope3_host_(static_cast<size_t>(3 * kMaxPrime)),
      prime_rope3_dev_(static_cast<size_t>(3 * kMaxPrime)) {
  (void)w;  // cfg/w size this object's buffers above; neither is stored -- see mtp_head.h
}

std::vector<int32_t> MtpHead::Draft(core::Stream& stream, core::Arena& arena,
                                     const ModelConfig& cfg, const MtpWeights& w,
                                     const uint16_t* h_seed, int32_t seed_token,
                                     const uint16_t* embed_table, const uint16_t* embed_table_dev,
                                     int64_t vocab, const QuantLinear& lm_head, int64_t k,
                                     int64_t base_pos, bool use_reduced_vocab, bool mrope_active,
                                     int64_t mrope_delta) {
  std::vector<int32_t> drafts;
  if (k <= 0) return drafts;
  if (k > max_draft_) {
    throw std::runtime_error(
        "MtpHead::Draft: k exceeds max_draft this instance was constructed for");
  }
  drafts.reserve(static_cast<size_t>(k));

  const int64_t hidden = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const hipStream_t s = stream.get();
  const bool device_resident = (embed_table_dev != nullptr);
  // Reduced-vocab draft head (docs/r9700.md R9, this method's own .h doc comment): active only when
  // the caller asked for it AND the container actually has one -- an old container (or a run
  // converted without --draft-vocab-ids) simply has draft_lm_head.N==0, so this unconditionally
  // degrades to the pre-R9 full-vocab path with no caller-visible difference beyond speed.
  const bool reduced_vocab = use_reduced_vocab && w.HasDraftHead();
  const QuantLinear& draft_head = reduced_vocab ? w.draft_lm_head : lm_head;
  const int64_t draft_head_vocab = reduced_vocab ? w.draft_vocab_ids.size() : vocab;

  // Preload this WHOLE draft window's positions/seqused_k in one H2D upload each -- was one
  // blocking CopyFromHost per step (device-resident draft loop, docs/mtp.md). Done unconditionally
  // (even on the host-gather fallback path): this part of the fix is independent of embedding
  // residency and benefits both.
  for (int64_t step = 0; step < k; ++step) {
    positions_host_[static_cast<size_t>(step)] = static_cast<int32_t>(base_pos + step);
    seqused_host_[static_cast<size_t>(step)] = static_cast<int32_t>(base_pos + step + 1);
  }
  positions_dev_.CopyFromHostAsync(positions_host_.data(), static_cast<size_t>(k), stream);
  seqused_dev_.CopyFromHostAsync(seqused_host_.data(), static_cast<size_t>(k), stream);
  // 3-axis rope window (docs/vision.md): every drafted position is past the prompt, hence text, so
  // all three streams hold the same `sequence index + delta` -- built and uploaded here for the
  // same reason positions_/seqused_ are, one H2D for the whole window rather than one per step.
  if (mrope_active) {
    for (int64_t step = 0; step < k; ++step) {
      const int32_t p = static_cast<int32_t>(base_pos + step + mrope_delta);
      rope3_host_[static_cast<size_t>(3 * step + 0)] = p;
      rope3_host_[static_cast<size_t>(3 * step + 1)] = p;
      rope3_host_[static_cast<size_t>(3 * step + 2)] = p;
    }
    rope3_dev_.CopyFromHostAsync(rope3_host_.data(), static_cast<size_t>(3 * k), stream);
  }

  // Device-resident path's one remaining small H2D: the seed token (step 0's embedding gather
  // input) is not yet known on-device -- every step AFTER 0 instead feeds straight from the
  // PREVIOUS step's own argmax_dev_ output (r4dx_argmax_f32), no H2D at all.
  const int32_t* cur_token_dev = nullptr;
  if (device_resident) {
    seed_token_dev_.CopyFromHostAsync(&seed_token, 1, stream);
    cur_token_dev = seed_token_dev_.data();
  }

  // cur_hidden: device bf16 [hidden], the (h_seed, cur_token) pair's "h_i" -- h_seed for the first
  // draft, this layer's own previous-step output for every draft after that (see file comment).
  const uint16_t* cur_hidden = h_seed;
  int32_t cur_token = seed_token;  // host-gather fallback path only

  for (int64_t step = 0; step < k; ++step) {
    // ---- concat(pre_fc_norm_embedding(embed(cur_token)), pre_fc_norm_hidden(cur_hidden)) -------
    // Embedding occupies fc's FIRST `hidden` input columns, hidden state the SECOND -- see
    // mtp_head.h's file comment for the reference this follows and the incident this corrects.
    uint16_t* concat_buf = arena.Alloc<uint16_t>(static_cast<size_t>(2 * hidden));
    if (device_resident) {
      // Entirely on-device: cur_token_dev is either seed_token_dev_ (step 0) or the PREVIOUS
      // step's own argmax_dev_ (step>0, written by that step's own r4dx_argmax_f32 call below) --
      // no host sync, D2H, or H2D between chained draft steps.
      EmbedTokensDeviceGather(stream, embed_table_dev, hidden, cur_token_dev, /*n=*/1, vocab,
                               embed_staging_dev_.data());
    } else {
      EmbedTokens(stream, embed_table, vocab, hidden, {cur_token}, embed_staging_host_,
                  embed_staging_dev_);
    }
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
    // positions_dev_/seqused_dev_ were preloaded for the whole window above -- index into them by
    // pointer arithmetic instead of a per-step upload.
    const int32_t pos_h = static_cast<int32_t>(base_pos + step);

    attention::AttnWeights aw;
    aw.input_layernorm = w.layer.input_layernorm.data();
    aw.qg = &w.layer.attn->qg;
    aw.k = &w.layer.attn->k;
    aw.v = &w.layer.attn->v;
    aw.o = &w.layer.attn->o;
    aw.q_norm = w.layer.attn->q_norm.data();
    aw.k_norm = w.layer.attn->k_norm.data();
    aw.k_descale = w.layer.attn->k_descale.data();
    aw.v_descale = w.layer.attn->v_descale.data();

    uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(hidden));
    attn_layer_.Forward(arena, fc_out, attn_out, aw, kv_, /*T=*/1, /*start_pos=*/pos_h,
                         positions_dev_.data() + step, seqused_dev_.data() + step, s,
                         /*x_normed_in=*/nullptr, /*next_norm_weight=*/nullptr,
                         /*x_normed_out=*/nullptr, /*prof=*/nullptr, /*x_normed_pre_epilogue=*/0,
                         /*x_normed_pre_data=*/nullptr, /*x_normed_pre_scale=*/nullptr,
                         /*next_epilogue=*/0, /*next_epilogue_out=*/nullptr,
                         /*next_epilogue_scale=*/nullptr,
                         mrope_active ? rope3_dev_.data() + 3 * step : nullptr);

    uint16_t* h_out = arena.Alloc<uint16_t>(static_cast<size_t>(hidden));
    Mlp mlp(cfg, w.layer.post_attention_layernorm, w.layer.mlp);
    mlp.Forward(stream, arena, attn_out, h_out, /*T=*/1);

    // ---- mtp.norm -> draft lm_head (reduced-vocab if available/requested, else the shared
    // full-vocab one) -> greedy argmax (device) -- see this method's .h doc comment for why using
    // the reduced head here can only ever LOWER acceptance, never produce a wrong ACCEPTED token
    // (verification always re-checks against the real model's own full-vocab head, never this one).
    // FinalLmHead is fully generic in its own lm_head's row count (final_lm_head.cpp's Forward
    // reads `lm_head_.N` as the vocab it produces), so the reduced head is just a smaller QuantLinear
    // fed through the exact same GEMM path -- no separate kernel needed for the GEMM itself.
    // logits_dev_ is sized for the FULL vocab (ctor) and the reduced head only ever needs its first
    // draft_head_vocab elements, so reusing it here is always in-bounds either way.
    FinalLmHead head(cfg, w.norm, draft_head);
    head.Forward(stream, arena, h_out, logits_dev_.data(), /*T=*/1);
    if (reduced_vocab) {
      // logits_dev_[0..draft_head_vocab) holds this step's SUBSET logits -- argmax over exactly
      // that many elements gives a subset-LOCAL index, not a real vocab id yet.
      r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                       reinterpret_cast<int64_t>(subset_argmax_dev_.data()), draft_head_vocab,
                       reinterpret_cast<int64_t>(s));
      // One device-side hop (no host sync) maps that subset-local index back to a real vocab id via
      // w.draft_vocab_ids -- from here on, argmax_dev_ holds a real vocab id exactly like the
      // full-vocab path always has, so every line below this point (device-resident chaining,
      // draft_ids_dev_ accumulation, the host-gather fallback's own D2H) is completely unchanged.
      r4dx_gather_i32(reinterpret_cast<int64_t>(w.draft_vocab_ids.data()),
                       reinterpret_cast<int64_t>(subset_argmax_dev_.data()),
                       reinterpret_cast<int64_t>(argmax_dev_.data()), /*n=*/1, draft_head_vocab,
                       reinterpret_cast<int64_t>(s));
    } else {
      r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                       reinterpret_cast<int64_t>(argmax_dev_.data()), draft_head_vocab,
                       reinterpret_cast<int64_t>(s));
    }

    if (device_resident) {
      // D2D, not D2H -- stays entirely on-device, ordered after this step's own argmax write by
      // HIP's own same-stream issue-order guarantee (no separate sync needed). Accumulates into
      // draft_ids_dev_[step] for ONE batched D2H readback after the loop, instead of one per step.
      R4DX_HIP_CHECK(hipMemcpyAsync(draft_ids_dev_.data() + step, argmax_dev_.data(),
                                     sizeof(int32_t), hipMemcpyDeviceToDevice, s));
      cur_token_dev = argmax_dev_.data();  // next step's gather input, zero syncs
      cur_hidden = h_out;
    } else {
      // Host-gather fallback: unavoidable per-step sync (embed_table has no device mirror to
      // gather from), same as the original implementation.
      stream.Synchronize();
      int32_t next_token = -1;
      argmax_dev_.CopyToHost(&next_token, 1);
      drafts.push_back(next_token);
      cur_token = next_token;
      cur_hidden = h_out;
    }
  }

  if (device_resident) {
    // The ONE sync + D2H for the whole k-step window, replacing what was previously k of each.
    stream.Synchronize();
    draft_ids_dev_.CopyToHost(draft_ids_host_.data(), static_cast<size_t>(k));
    drafts.assign(draft_ids_host_.begin(), draft_ids_host_.begin() + k);
  }
  return drafts;
}

void MtpHead::PrimeKv(core::Stream& stream, core::Arena& arena, const ModelConfig& cfg,
                      const MtpWeights& w, int64_t base_pos, const uint16_t* h_rows,
                      const std::vector<int32_t>& next_tokens, const uint16_t* embed_table,
                      int64_t vocab, int64_t host_staging_offset, const int32_t* rope3_host) {
  const int64_t n = static_cast<int64_t>(next_tokens.size());
  if (n <= 0) return;
  if (n > kMaxPrime) {
    throw std::runtime_error("MtpHead::PrimeKv: n must be <= " + std::to_string(kMaxPrime));
  }
  if (host_staging_offset < 0 || host_staging_offset + n > kMaxPrime) {
    throw std::runtime_error(
        "MtpHead::PrimeKv: host_staging_offset + n must be <= " + std::to_string(kMaxPrime) +
        " (see PrimeKv's own .h doc comment: back-to-back calls must use disjoint offsets)");
  }
  const int64_t hidden = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const hipStream_t s = stream.get();

  // pre_fc_norm_embedding/pre_fc_norm_hidden each need a CONTIGUOUS [n,hidden] output (no
  // strided-row GEMM/rmsnorm entry point exists in this codebase -- see linear.h's own comment),
  // so build the two halves into separate contiguous buffers, then splice them into concat_buf via
  // two strided D2D copies (hipMemcpy2DAsync) rather than a bespoke interleave kernel.
  //
  // Gather straight into this call's OWN slice of prime_embed_host_ (offset by
  // host_staging_offset*hidden elements, not the buffer's start) and upload from that same slice --
  // NOT r4dx::model::EmbedTokens' own buffer-start-relative helper -- so a back-to-back PrimeKv
  // call from RunChunk cannot overwrite this call's still-in-flight H2D source before the GPU has
  // actually read it (see this method's .h doc comment for the race this closes). The device
  // destination (prime_embed_dev_.data(), unconditionally offset 0) is unaffected -- device-side
  // reuse across calls is already safe via HIP's own stream-issue-order guarantee.
  uint16_t* embed_normed = arena.Alloc<uint16_t>(static_cast<size_t>(n * hidden));
  uint16_t* const embed_host_slice = prime_embed_host_.data() + host_staging_offset * hidden;
  kernels::EmbeddingGatherHost(embed_table, vocab, hidden, next_tokens, embed_host_slice);
  prime_embed_dev_.CopyFromHostAsync(embed_host_slice, static_cast<size_t>(n * hidden), stream);
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

  // Same host_staging_offset slicing as the embedding gather above, for the identical reason:
  // prime_positions_host_/prime_seqused_host_ are this object's own pinned scratch, and a
  // back-to-back call must not overwrite a slice a previous call's still-in-flight H2D hasn't read
  // yet. Device destinations (prime_positions_dev_/prime_seqused_dev_) stay offset-0/fixed-size.
  std::vector<int32_t> positions_h(static_cast<size_t>(n));
  for (int64_t t = 0; t < n; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(base_pos + t);
  std::copy(positions_h.begin(), positions_h.end(), prime_positions_host_.begin() + host_staging_offset);
  prime_positions_dev_.CopyFromHostAsync(prime_positions_host_.data() + host_staging_offset,
                                          static_cast<size_t>(n), stream);
  prime_seqused_host_[static_cast<size_t>(host_staging_offset)] = static_cast<int32_t>(base_pos + n);
  prime_seqused_dev_.CopyFromHostAsync(prime_seqused_host_.data() + host_staging_offset, 1, stream);

  // 3-axis rope rows (docs/vision.md), same disjoint-host-slice discipline as the two uploads
  // above -- 3 elements per position, so this call's slice starts at 3*host_staging_offset and is
  // 3*n long, which still fits 3*kMaxPrime for RunChunk's (0, 1) offset pair.
  if (rope3_host != nullptr) {
    std::copy(rope3_host, rope3_host + 3 * n, prime_rope3_host_.begin() + 3 * host_staging_offset);
    prime_rope3_dev_.CopyFromHostAsync(prime_rope3_host_.data() + 3 * host_staging_offset,
                                        static_cast<size_t>(3 * n), stream);
  }

  attention::AttnWeights aw;
  aw.input_layernorm = w.layer.input_layernorm.data();
  aw.qg = &w.layer.attn->qg;
  aw.k = &w.layer.attn->k;
  aw.v = &w.layer.attn->v;
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
                       prime_seqused_dev_.data(), s, /*x_normed_in=*/nullptr,
                       /*next_norm_weight=*/nullptr, /*x_normed_out=*/nullptr, /*prof=*/nullptr,
                       /*x_normed_pre_epilogue=*/0, /*x_normed_pre_data=*/nullptr,
                       /*x_normed_pre_scale=*/nullptr, /*next_epilogue=*/0,
                       /*next_epilogue_out=*/nullptr, /*next_epilogue_scale=*/nullptr,
                       rope3_host != nullptr ? prime_rope3_dev_.data() : nullptr);
}

}  // namespace r4dx::model
