#include "dflash_draft.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include <memory>

#include "embedding.h"
#include "kernels/model_kernels.h"
#include "linear.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/embedding.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

DflashEmbeddingProvider MakeTargetEmbeddingProvider(const Container& container) {
  const int64_t hidden = container.Config().hidden_size;
  const int64_t vocab = container.Config().vocab_size;
  if (container.EmbedTokensDeviceResident()) {
    const uint16_t* table_dev = container.EmbedTokensDevice();
    return [table_dev, hidden, vocab](core::Stream& stream, const int32_t*, const int32_t* ids_dev,
                                      int64_t n, uint16_t* out_dev) {
      r4dx_embedding_gather_bf16(reinterpret_cast<int64_t>(table_dev),
                                 reinterpret_cast<int64_t>(ids_dev),
                                 reinterpret_cast<int64_t>(out_dev), n, hidden, vocab,
                                 reinterpret_cast<int64_t>(stream.get()));
    };
  }
  const uint16_t* table_host = container.EmbedTokensHost();
  // One staging buffer per provider instance, kept alive by the closure (a block is only 8 rows, so
  // this is ~80 KB and is reused every round rather than re-allocated).
  auto staging = std::make_shared<std::vector<uint16_t>>();
  return [table_host, hidden, vocab, staging](core::Stream& stream, const int32_t* ids_host,
                                              const int32_t*, int64_t n, uint16_t* out_dev) {
    staging->resize(static_cast<size_t>(n * hidden));
    const std::vector<int32_t> ids(ids_host, ids_host + n);
    kernels::EmbeddingGatherHost(table_host, vocab, hidden, ids, staging->data());
    R4DX_HIP_CHECK(hipMemcpyAsync(out_dev, staging->data(),
                                  staging->size() * sizeof(uint16_t), hipMemcpyHostToDevice,
                                  stream.get()));
  };
}

DflashLmHeadProvider MakeTargetLmHeadProvider(const Container& container) {
  const QuantLinear* lm_head = &container.LmHead();
  return [lm_head](core::Stream& stream, core::Arena& arena, const uint16_t* x_dev,
                   float* logits_out, int64_t T) {
    const int64_t vocab = lm_head->N;
    uint16_t* logits_bf16 = arena.Alloc<uint16_t>(static_cast<size_t>(T * vocab));
    ApplyLinear(stream, arena, *lm_head, x_dev, logits_bf16, T);
    r4dx_model_widen_bf16_to_f32(reinterpret_cast<int64_t>(logits_bf16),
                                 reinterpret_cast<int64_t>(logits_out), T * vocab,
                                 reinterpret_cast<int64_t>(stream.get()));
  };
}

namespace {

// Byte-span-derived element count, exactly src/model/container.cpp's own ElemCountBySize: every
// tensor in an r4dx container is declared `dtype: "U8"` with a trailing element-width dimension
// (docs/container-format.md), so the real element count is the span divided by the element size
// this loader knows the tensor must have -- never `TensorMeta::ElemCount()`.
int64_t ElemCountBySize(const DflashDraftWeights& w, const std::string& name, int64_t elem_bytes) {
  const auto& m = w.TensorMeta(name);
  const uint64_t span = m.end - m.begin;
  if (span % static_cast<uint64_t>(elem_bytes) != 0) {
    throw std::runtime_error("DflashDraft: tensor '" + name + "' byte span is not a multiple of " +
                             std::to_string(elem_bytes));
  }
  return static_cast<int64_t>(span / static_cast<uint64_t>(elem_bytes));
}

core::DeviceBuffer<uint16_t> UploadRawU16(const DflashDraftWeights& w, const std::string& name) {
  const int64_t n = ElemCountBySize(w, name, 2);
  core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint16_t*>(w.TensorData(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<uint8_t> UploadRawU8(const DflashDraftWeights& w, const std::string& name) {
  const int64_t n = ElemCountBySize(w, name, 1);
  core::DeviceBuffer<uint8_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(w.TensorData(name), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<uint32_t> UploadRawU32(const DflashDraftWeights& w, const std::string& name) {
  const int64_t n = ElemCountBySize(w, name, 4);
  core::DeviceBuffer<uint32_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint32_t*>(w.TensorData(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<int8_t> UploadRawI8(const DflashDraftWeights& w, const std::string& name) {
  const int64_t n = ElemCountBySize(w, name, 1);
  core::DeviceBuffer<int8_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const int8_t*>(w.TensorData(name)), static_cast<size_t>(n));
  return buf;
}

// Every DFlash2 norm weight is stored FP32 on disk (docs/container-format.md's tensor table: the
// GGUF source keeps `*_norm.weight` in f32 and the converter passes it through unchanged), while
// `r4dx_rmsnorm_plain_bf16` takes a bf16 weight like every other norm in this repo. Round once
// here rather than adding an fp32-weight variant of that kernel: the product is consumed as bf16
// immediately afterwards anyway, so the rounding adds at most one bf16 step of relative error to a
// value that is about to be rounded to bf16 regardless -- measured end to end in
// tests/model/test_dflash_draft.cpp, which reports the resulting per-intermediate error against the
// fp32 Python reference.
core::DeviceBuffer<uint16_t> UploadF32AsBf16(const DflashDraftWeights& w, const std::string& name) {
  const int64_t n = ElemCountBySize(w, name, 4);
  const auto* src = reinterpret_cast<const float*>(w.TensorData(name));
  std::vector<uint16_t> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = core::FloatToBf16(src[static_cast<size_t>(i)]);
  core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(host);
  return buf;
}

std::vector<uint16_t> ReadHostBf16(const DflashDraftWeights& w, const std::string& name) {
  const int64_t n = ElemCountBySize(w, name, 2);
  std::vector<uint16_t> host(static_cast<size_t>(n));
  std::memcpy(host.data(), w.TensorData(name), static_cast<size_t>(n) * 2);
  return host;
}

// Mirrors src/model/container.cpp's LoadQuantLinear against a DFlash2 draft container's own
// `<base>.{layout}.*` tensor names (docs/container-format.md "DFlash2 draft container").
QuantLinear LoadQuantLinear(const DflashDraftWeights& w, const std::string& base, Layout layout,
                            int64_t N, int64_t K) {
  QuantLinear q;
  q.layout = layout;
  q.N = N;
  q.K = K;
  switch (layout) {
    case Layout::kBf16:
      q.bf16_w = UploadRawU16(w, base + ".bf16.w");
      break;
    case Layout::kW4a16:
      q.wq = UploadRawU8(w, base + ".w4a16.wq");
      q.w4a16_wsz = UploadRawU32(w, base + ".w4a16.wsz");
      break;
    case Layout::kW4a8:
      q.wq = UploadRawU8(w, base + ".w4a8.wq");
      q.w4a8_ws = UploadRawU32(w, base + ".w4a8.ws");
      break;
    case Layout::kMxfp4:
      q.mxfp4_wq = UploadRawU8(w, base + ".mxfp4.wq");
      q.mxfp4_ws = UploadRawU8(w, base + ".mxfp4.ws");
      q.mxfp4_wref = UploadRawI8(w, base + ".mxfp4.wref");
      break;
  }
  return q;
}

int64_t AlignUp(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

}  // namespace

DflashDraft DflashDraft::Load(const DflashDraftOptions& opts) {
  DflashDraft d;
  DflashDraftWeights w = DflashDraftWeights::Open(opts.container_path);
  d.cfg_ = w.Config();
  d.layout_ = opts.layout;
  d.max_inject_rows_ = opts.max_inject_rows;

  const Dflash2Config& c = d.cfg_;
  const int64_t hidden = c.hidden_size;
  const int64_t n_layer = c.block_count;
  const int64_t heads_q = c.attention.head_count;
  const int64_t heads_kv = c.attention.head_count_kv;
  const int64_t head_dim = c.attention.key_length;
  const int64_t ffn = c.feed_forward_length;
  const int64_t rank = c.selector_rank;
  const int64_t B = c.block_size;
  const int64_t n_groups = hidden / c.conv_group_size;
  const int64_t dyn_n = 2 * c.conv_kernel_size * n_groups;
  const int64_t feat = static_cast<int64_t>(c.target_layers.size()) * hidden;

  if (c.attention.value_length != head_dim) {
    throw std::runtime_error("DflashDraft: key_length != value_length is not supported");
  }
  if (c.conv_kernel_size != 2 || c.conv_group_size != 16) {
    // r4dx_dflash_conv_bf16 wraps libr4d's r4d_dflash_conv_t2_g16_bf16, the only geometry libr4d
    // compiles (docs/dflash2.md section 6b) -- fail loudly rather than compute the wrong conv.
    throw std::runtime_error("DflashDraft: only conv_kernel_size=2 / conv_group_size=16 is built");
  }
  if (c.selector_top_k != 16) {
    throw std::runtime_error("DflashDraft: only selector_top_k=16 is built (r4dx_topk16_f32)");
  }
  if (opts.max_inject_rows < 1 || opts.max_inject_rows > 64) {
    throw std::runtime_error("DflashDraft: max_inject_rows must be in [1, 64]");
  }

  d.lm_head_vocab_ = (opts.lm_head_vocab > 0) ? opts.lm_head_vocab : c.vocab_size;
  d.mask_token_id_ =
      (opts.mask_token_id_override >= 0) ? opts.mask_token_id_override : c.mask_token_id;
  if (d.mask_token_id_ < 0 || d.mask_token_id_ >= d.lm_head_vocab_) {
    throw std::runtime_error("DflashDraft: mask_token_id " + std::to_string(d.mask_token_id_) +
                             " is outside the lm_head vocab " + std::to_string(d.lm_head_vocab_));
  }

  // ---- weights ---------------------------------------------------------------------------------
  d.fc_ = LoadQuantLinear(w, "dflash.fc", opts.layout, hidden, feat);
  d.enc_output_norm_ = UploadF32AsBf16(w, "dflash.enc_output_norm");
  d.output_norm_ = UploadF32AsBf16(w, "dflash.output_norm");
  d.selector_hidden_ = LoadQuantLinear(w, "dflash.selector.hidden", opts.layout, rank, hidden);
  d.selector_predecessor_ = ReadHostBf16(w, "dflash.selector.predecessor");
  d.selector_successor_ = ReadHostBf16(w, "dflash.selector.successor");
  if (static_cast<int64_t>(d.selector_predecessor_.size()) < d.lm_head_vocab_ * rank ||
      static_cast<int64_t>(d.selector_successor_.size()) < d.lm_head_vocab_ * rank) {
    throw std::runtime_error("DflashDraft: selector codebooks are smaller than lm_head_vocab*rank");
  }

  d.layers_.resize(static_cast<size_t>(n_layer));
  for (int64_t i = 0; i < n_layer; ++i) {
    LayerWeights& lw = d.layers_[static_cast<size_t>(i)];
    const std::string base = "dflash.layers." + std::to_string(i) + ".";
    lw.input_layernorm = UploadF32AsBf16(w, base + "input_layernorm");
    lw.post_attention_layernorm = UploadF32AsBf16(w, base + "post_attention_layernorm");
    lw.q_norm = UploadF32AsBf16(w, base + "self_attn.q_norm");
    lw.k_norm = UploadF32AsBf16(w, base + "self_attn.k_norm");
    lw.attn_conv_base = UploadRawU16(w, base + "self_attn.conv.base");
    lw.mlp_conv_base = UploadRawU16(w, base + "mlp.conv.base");
    lw.q_proj = LoadQuantLinear(w, base + "self_attn.q_proj", opts.layout, heads_q * head_dim, hidden);
    lw.k_proj = LoadQuantLinear(w, base + "self_attn.k_proj", opts.layout, heads_kv * head_dim, hidden);
    lw.v_proj = LoadQuantLinear(w, base + "self_attn.v_proj", opts.layout, heads_kv * head_dim, hidden);
    lw.o_proj = LoadQuantLinear(w, base + "self_attn.o_proj", opts.layout, hidden, heads_q * head_dim);
    lw.attn_conv_proj = LoadQuantLinear(w, base + "self_attn.conv.proj", opts.layout, dyn_n, hidden);
    lw.gate_proj = LoadQuantLinear(w, base + "mlp.gate_proj", opts.layout, ffn, hidden);
    lw.up_proj = LoadQuantLinear(w, base + "mlp.up_proj", opts.layout, ffn, hidden);
    lw.down_proj = LoadQuantLinear(w, base + "mlp.down_proj", opts.layout, hidden, ffn);
    lw.mlp_conv_proj = LoadQuantLinear(w, base + "mlp.conv.proj", opts.layout, dyn_n, hidden);
  }

  // ---- KV ring ---------------------------------------------------------------------------------
  // slots == sliding_window: a slot is reused only once its previous occupant has aged out of every
  // possible query's visible range, which is exactly r4dx_dflash_attn_bf16's `window <= slots`
  // precondition (docs/dflash2.md section 6b).
  d.slots_ = c.attention.sliding_window;
  const size_t store_elems =
      static_cast<size_t>(n_layer) * static_cast<size_t>(d.slots_) *
      static_cast<size_t>(heads_kv * head_dim);
  d.k_store_ = core::DeviceBuffer<uint16_t>(store_elems);
  d.v_store_ = core::DeviceBuffer<uint16_t>(store_elems);
  // Zeroed once at load: a slot is never READ before it is written (the attention kernel only ever
  // visits positions < n_injected), but leaving uninitialised VRAM in a buffer this class hands to
  // a kernel would make any future off-by-one read nondeterministic instead of reproducible.
  d.k_store_.Zero();
  d.v_store_.Zero();

  // ---- scratch ----------------------------------------------------------------------------------
  const size_t R = static_cast<size_t>(opts.max_inject_rows);
  d.g_dev_ = core::DeviceBuffer<uint16_t>(R * static_cast<size_t>(hidden));
  d.ik_dev_ = core::DeviceBuffer<uint16_t>(R * static_cast<size_t>(heads_kv * head_dim));
  d.iv_dev_ = core::DeviceBuffer<uint16_t>(R * static_cast<size_t>(heads_kv * head_dim));
  d.ipos_dev_ = core::DeviceBuffer<int32_t>(R);
  d.ipos_host_ = core::PinnedBuffer<int32_t>(R);

  const size_t Bs = static_cast<size_t>(B);
  d.x_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(hidden));
  d.h_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(hidden));
  d.conv_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(hidden));
  d.proj_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(hidden));
  d.xf_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(hidden));
  d.dyn_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(dyn_n));
  d.q_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(heads_q * head_dim));
  d.k_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(heads_kv * head_dim));
  d.v_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(heads_kv * head_dim));
  d.attn_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(heads_q * head_dim));
  d.gate_tmp_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(ffn));
  d.up_tmp_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(ffn));
  d.gate_up_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(2 * ffn));
  d.act_ = core::DeviceBuffer<uint16_t>(Bs * static_cast<size_t>(ffn));
  d.logits_dev_ = core::DeviceBuffer<float>(Bs * static_cast<size_t>(d.lm_head_vocab_));
  d.block_ids_dev_ = core::DeviceBuffer<int32_t>(Bs);
  d.block_ids_host_ = core::PinnedBuffer<int32_t>(Bs);
  d.pos_dev_ = core::DeviceBuffer<int32_t>(Bs);
  d.pos_host_ = core::PinnedBuffer<int32_t>(Bs);

  // The single readback blob. Offsets are 16-byte aligned so each slice satisfies the alignment its
  // producing kernel's own vector stores want, and so the fp32/int32 slices stay naturally aligned.
  const int64_t topk = c.selector_top_k;
  d.sel_off_unary_ = 0;
  d.sel_off_cand_ = AlignUp(d.sel_off_unary_ + B * topk * 4, 16);
  d.sel_off_gate_ = AlignUp(d.sel_off_cand_ + B * topk * 4, 16);
  d.sel_stage_bytes_ = AlignUp(d.sel_off_gate_ + B * rank * 2, 16);
  d.sel_stage_dev_ = core::DeviceBuffer<uint8_t>(static_cast<size_t>(d.sel_stage_bytes_));
  d.sel_stage_host_ = core::PinnedBuffer<uint8_t>(static_cast<size_t>(d.sel_stage_bytes_));

  return d;
}

void DflashDraft::InjectFeatures(core::Stream& stream, core::Arena& arena,
                                 const uint16_t* features_dev, int64_t rows, int64_t start_pos) {
  if (rows <= 0) return;
  if (rows > max_inject_rows_) {
    throw std::runtime_error("DflashDraft::InjectFeatures: rows exceeds max_inject_rows");
  }
  if (start_pos < n_injected_) {
    // MONOTONIC. This is not a convenience check: docs/dflash2.md section 5's "no rollback needed"
    // argument holds only because every write goes to a position at or above anything already
    // stored, so a stale rejected row is always physically overwritten before it can be read as
    // committed data. Writing BELOW n_injected_ would overwrite a position this drafter may already
    // have attended to -- i.e. exactly the rollback that argument rules out.
    throw std::runtime_error("DflashDraft::InjectFeatures: start_pos (" +
                             std::to_string(start_pos) + ") < InjectedCount() (" +
                             std::to_string(n_injected_) + "); injection is monotonic");
  }
  if (start_pos > n_injected_) {
    // GAP (see the .h doc comment): the caller skipped some positions entirely, so every ring slot
    // below `start_pos` is now stale. Rather than clear those bytes (up to a full 2048-slot ring x
    // 5 layers x 2 tensors), move the validity lower bound up and let the attention kernel's own
    // `store_begin` clamp keep them unread -- they will be physically overwritten by a later
    // injection at their own position long before `valid_from_` could ever come back down (it
    // never does; only Reset() lowers it).
    valid_from_ = start_pos;
    n_injected_ = start_pos;
  }

  const int64_t hidden = cfg_.hidden_size;
  const int64_t heads_kv = cfg_.attention.head_count_kv;
  const int64_t head_dim = cfg_.attention.key_length;
  const int64_t kv_row = heads_kv * head_dim;
  const float eps = static_cast<float>(cfg_.attention.rms_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());

  // g = rmsnorm_plain(fc(features)) -- shared by all five layers (docs/dflash2.md section 4.1).
  ApplyLinear(stream, arena, fc_, features_dev, g_dev_.data(), rows);
  r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(g_dev_.data()),
                          reinterpret_cast<int64_t>(enc_output_norm_.data()),
                          reinterpret_cast<int64_t>(g_dev_.data()), rows, hidden, eps,
                          /*out_fp32=*/0, s);

  for (int64_t t = 0; t < rows; ++t) {
    ipos_host_[static_cast<size_t>(t)] = static_cast<int32_t>(start_pos + t);
  }
  ipos_dev_.CopyFromHostAsync(ipos_host_.data(), static_cast<size_t>(rows), stream);

  const int64_t n_layer = cfg_.block_count;
  for (int64_t il = 0; il < n_layer; ++il) {
    const LayerWeights& lw = layers_[static_cast<size_t>(il)];
    ApplyLinear(stream, arena, lw.k_proj, g_dev_.data(), ik_dev_.data(), rows);
    ApplyLinear(stream, arena, lw.v_proj, g_dev_.data(), iv_dev_.data(), rows);
    // k_norm is per-head over head_dim, so the [rows, kv_heads, head_dim] slab normalises as
    // rows*kv_heads independent head_dim-wide rows. V is raw -- no norm, no rope.
    r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(ik_dev_.data()),
                            reinterpret_cast<int64_t>(lw.k_norm.data()),
                            reinterpret_cast<int64_t>(ik_dev_.data()), rows * heads_kv, head_dim,
                            eps, /*out_fp32=*/0, s);
    r4dx_rope_neox_bf16(/*q=*/0, reinterpret_cast<int64_t>(ik_dev_.data()),
                        reinterpret_cast<int64_t>(ipos_dev_.data()), static_cast<int>(rows),
                        /*heads_q=*/0, static_cast<int>(heads_kv), static_cast<int>(head_dim),
                        static_cast<float>(cfg_.rope.freq_base), s);

    // Ring write: positions are contiguous, so the destination slots are contiguous too except for
    // at most one wrap -- one or two plain D2D copies per layer, no kernel.
    uint16_t* kdst = k_store_.data() + il * slots_ * kv_row;
    uint16_t* vdst = v_store_.data() + il * slots_ * kv_row;
    const int64_t slot0 = start_pos % slots_;
    const int64_t n1 = std::min(rows, slots_ - slot0);
    const int64_t n2 = rows - n1;
    R4DX_HIP_CHECK(hipMemcpyAsync(kdst + slot0 * kv_row, ik_dev_.data(),
                                  static_cast<size_t>(n1 * kv_row) * sizeof(uint16_t),
                                  hipMemcpyDeviceToDevice, stream.get()));
    R4DX_HIP_CHECK(hipMemcpyAsync(vdst + slot0 * kv_row, iv_dev_.data(),
                                  static_cast<size_t>(n1 * kv_row) * sizeof(uint16_t),
                                  hipMemcpyDeviceToDevice, stream.get()));
    if (n2 > 0) {
      R4DX_HIP_CHECK(hipMemcpyAsync(kdst, ik_dev_.data() + n1 * kv_row,
                                    static_cast<size_t>(n2 * kv_row) * sizeof(uint16_t),
                                    hipMemcpyDeviceToDevice, stream.get()));
      R4DX_HIP_CHECK(hipMemcpyAsync(vdst, iv_dev_.data() + n1 * kv_row,
                                    static_cast<size_t>(n2 * kv_row) * sizeof(uint16_t),
                                    hipMemcpyDeviceToDevice, stream.get()));
    }
  }

  n_injected_ = start_pos + rows;
  last_inject_rows_ = rows;
}

void DflashDraft::ForwardLayer(core::Stream& stream, core::Arena& arena, int64_t il,
                               DflashRoundTrace* trace) {
  const LayerWeights& lw = layers_[static_cast<size_t>(il)];
  const int64_t B = cfg_.block_size;
  const int64_t hidden = cfg_.hidden_size;
  const int64_t ffn = cfg_.feed_forward_length;
  const int64_t heads_q = cfg_.attention.head_count;
  const int64_t heads_kv = cfg_.attention.head_count_kv;
  const int64_t head_dim = cfg_.attention.key_length;
  const int64_t kv_row = heads_kv * head_dim;
  const float eps = static_cast<float>(cfg_.attention.rms_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());
  const int block_pow2 = static_cast<int>(B);  // block_size is 8, already a power of two

  // ---- attention sub-block ----------------------------------------------------------------------
  r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_.data()),
                          reinterpret_cast<int64_t>(lw.input_layernorm.data()),
                          reinterpret_cast<int64_t>(h_.data()), B, hidden, eps, 0, s);
  // dyn is computed ONCE per sub-block, from the PRE-conv normed input, and is reused by BOTH conv
  // sides (docs/dflash2.md section 4.2) -- not recomputed from the post-attention tensor.
  ApplyLinear(stream, arena, lw.attn_conv_proj, h_.data(), dyn_.data(), B);
  r4dx_dflash_conv_bf16(reinterpret_cast<int64_t>(h_.data()), reinterpret_cast<int64_t>(dyn_.data()),
                        reinterpret_cast<int64_t>(lw.attn_conv_base.data()),
                        reinterpret_cast<int64_t>(conv_.data()), static_cast<int>(B),
                        static_cast<int>(hidden), /*side=*/0, block_pow2, s);

  ApplyLinear(stream, arena, lw.q_proj, conv_.data(), q_.data(), B);
  ApplyLinear(stream, arena, lw.k_proj, conv_.data(), k_.data(), B);
  ApplyLinear(stream, arena, lw.v_proj, conv_.data(), v_.data(), B);
  r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(q_.data()),
                          reinterpret_cast<int64_t>(lw.q_norm.data()),
                          reinterpret_cast<int64_t>(q_.data()), B * heads_q, head_dim, eps, 0, s);
  r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(k_.data()),
                          reinterpret_cast<int64_t>(lw.k_norm.data()),
                          reinterpret_cast<int64_t>(k_.data()), B * heads_kv, head_dim, eps, 0, s);
  r4dx_rope_neox_bf16(reinterpret_cast<int64_t>(q_.data()), reinterpret_cast<int64_t>(k_.data()),
                      reinterpret_cast<int64_t>(pos_dev_.data()), static_cast<int>(B),
                      static_cast<int>(heads_q), static_cast<int>(heads_kv),
                      static_cast<int>(head_dim), static_cast<float>(cfg_.rope.freq_base), s);

  // INVARIANT (docs/dflash2.md section 5): k_/v_ here are the block's OWN keys and values. They are
  // handed to the attention kernel as scratch operands and are NEVER written into k_store_/v_store_
  // -- that is the whole reason a partially-rejected verify round needs no ring rollback. Only
  // InjectFeatures() ever writes the ring, and only for positions the target has already committed.
  // `valid_from_` is passed as the kernel's `store_begin`: the visible store is the contiguous run
  // [valid_from_, n_injected_) intersected with the window, so an injection gap's stale bytes are
  // never read (docs/dflash2.md section 5, InjectFeatures' own doc comment).
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  r4dx_dflash_attn_bf16(
      reinterpret_cast<int64_t>(q_.data()), reinterpret_cast<int64_t>(k_.data()),
      reinterpret_cast<int64_t>(v_.data()),
      reinterpret_cast<int64_t>(k_store_.data() + il * slots_ * kv_row),
      reinterpret_cast<int64_t>(v_store_.data() + il * slots_ * kv_row),
      reinterpret_cast<int64_t>(attn_.data()), static_cast<int>(B), static_cast<int>(heads_q),
      static_cast<int>(heads_kv), static_cast<int>(head_dim), static_cast<int>(n_injected_),
      static_cast<int>(valid_from_), static_cast<int>(cfg_.attention.sliding_window),
      static_cast<int>(slots_), scale, s);

  ApplyLinear(stream, arena, lw.o_proj, attn_.data(), proj_.data(), B);
  r4dx_dflash_conv_bf16(reinterpret_cast<int64_t>(proj_.data()),
                        reinterpret_cast<int64_t>(dyn_.data()),
                        reinterpret_cast<int64_t>(lw.attn_conv_base.data()),
                        reinterpret_cast<int64_t>(conv_.data()), static_cast<int>(B),
                        static_cast<int>(hidden), /*side=*/1, block_pow2, s);
  r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x_.data()),
                         reinterpret_cast<int64_t>(conv_.data()),
                         reinterpret_cast<int64_t>(x_.data()), B * hidden, s);
  if (trace != nullptr) {
    stream.Synchronize();
    trace->x_post_attn[static_cast<size_t>(il)] = x_.CopyToHost();
  }

  // ---- FFN sub-block -----------------------------------------------------------------------------
  r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_.data()),
                          reinterpret_cast<int64_t>(lw.post_attention_layernorm.data()),
                          reinterpret_cast<int64_t>(h_.data()), B, hidden, eps, 0, s);
  ApplyLinear(stream, arena, lw.mlp_conv_proj, h_.data(), dyn_.data(), B);
  r4dx_dflash_conv_bf16(reinterpret_cast<int64_t>(h_.data()), reinterpret_cast<int64_t>(dyn_.data()),
                        reinterpret_cast<int64_t>(lw.mlp_conv_base.data()),
                        reinterpret_cast<int64_t>(conv_.data()), static_cast<int>(B),
                        static_cast<int>(hidden), /*side=*/0, block_pow2, s);

  ApplyLinear(stream, arena, lw.gate_proj, conv_.data(), gate_tmp_.data(), B);
  ApplyLinear(stream, arena, lw.up_proj, conv_.data(), up_tmp_.data(), B);
  // r4dx_silu_mul_bf16 consumes the main container's FUSED [rows, 2*intermediate] gate_up layout,
  // but a DFlash2 container keeps ffn_gate/ffn_up as two separate tensors (the GGUF source never
  // fuses them -- docs/container-format.md). Interleave the two [B, ffn] results into that layout
  // with two strided D2D copies (~280 KB each at B=8) rather than adding a second silu_mul entry
  // point or repacking the weights at load time.
  R4DX_HIP_CHECK(hipMemcpy2DAsync(gate_up_.data(), static_cast<size_t>(2 * ffn) * sizeof(uint16_t),
                                  gate_tmp_.data(), static_cast<size_t>(ffn) * sizeof(uint16_t),
                                  static_cast<size_t>(ffn) * sizeof(uint16_t),
                                  static_cast<size_t>(B), hipMemcpyDeviceToDevice, stream.get()));
  R4DX_HIP_CHECK(hipMemcpy2DAsync(gate_up_.data() + ffn,
                                  static_cast<size_t>(2 * ffn) * sizeof(uint16_t), up_tmp_.data(),
                                  static_cast<size_t>(ffn) * sizeof(uint16_t),
                                  static_cast<size_t>(ffn) * sizeof(uint16_t),
                                  static_cast<size_t>(B), hipMemcpyDeviceToDevice, stream.get()));
  r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up_.data()),
                     reinterpret_cast<int64_t>(act_.data()), B, ffn, 2 * ffn, s);
  ApplyLinear(stream, arena, lw.down_proj, act_.data(), proj_.data(), B);
  r4dx_dflash_conv_bf16(reinterpret_cast<int64_t>(proj_.data()),
                        reinterpret_cast<int64_t>(dyn_.data()),
                        reinterpret_cast<int64_t>(lw.mlp_conv_base.data()),
                        reinterpret_cast<int64_t>(conv_.data()), static_cast<int>(B),
                        static_cast<int>(hidden), /*side=*/1, block_pow2, s);
  r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x_.data()),
                         reinterpret_cast<int64_t>(conv_.data()),
                         reinterpret_cast<int64_t>(x_.data()), B * hidden, s);
  if (trace != nullptr) {
    stream.Synchronize();
    trace->x_post_ffn[static_cast<size_t>(il)] = x_.CopyToHost();
  }
}

DflashDraftResult DflashDraft::DraftRound(core::Stream& stream, core::Arena& arena,
                                          int32_t anchor_id, int64_t k, float p_min, int64_t n_min,
                                          const DflashEmbeddingProvider& embed,
                                          const DflashLmHeadProvider& lm_head,
                                          DflashRoundTrace* trace, double* device_ms) {
  const int64_t B = cfg_.block_size;
  const int64_t hidden = cfg_.hidden_size;
  const int64_t rank = cfg_.selector_rank;
  const int64_t topk = cfg_.selector_top_k;
  const float eps = static_cast<float>(cfg_.attention.rms_eps);
  const int64_t s = reinterpret_cast<int64_t>(stream.get());

  if (!embed || !lm_head) {
    throw std::runtime_error("DflashDraft::DraftRound: embedding/lm_head provider is empty");
  }
  if (k < 0) k = 0;
  k = std::min<int64_t>(k, B - 1);
  if (anchor_id < 0 || anchor_id >= lm_head_vocab_) {
    throw std::runtime_error("DflashDraft::DraftRound: anchor_id out of vocab range");
  }
  if (n_injected_ + B > cfg_.context_length) {
    throw std::runtime_error("DflashDraft::DraftRound: block would run past context_length");
  }

  // Device-time instrumentation (see the .h doc comment): both records must sit on the SAME side of
  // this method's own stream synchronize, or the elapsed time is not the round's device time.
  hipEvent_t ev0 = nullptr, ev1 = nullptr;
  if (device_ms != nullptr) {
    R4DX_HIP_CHECK(hipEventCreate(&ev0));
    R4DX_HIP_CHECK(hipEventCreate(&ev1));
    R4DX_HIP_CHECK(hipEventRecord(ev0, stream.get()));
  }

  // [anchor, <mask> x (block_size-1)] at absolute positions n .. n+block_size-1.
  block_ids_host_[0] = anchor_id;
  for (int64_t t = 1; t < B; ++t) block_ids_host_[static_cast<size_t>(t)] =
      static_cast<int32_t>(mask_token_id_);
  for (int64_t t = 0; t < B; ++t) pos_host_[static_cast<size_t>(t)] =
      static_cast<int32_t>(n_injected_ + t);
  block_ids_dev_.CopyFromHostAsync(block_ids_host_.data(), static_cast<size_t>(B), stream);
  pos_dev_.CopyFromHostAsync(pos_host_.data(), static_cast<size_t>(B), stream);

  embed(stream, block_ids_host_.data(), block_ids_dev_.data(), B, x_.data());

  if (trace != nullptr) {
    trace->block_ids.assign(block_ids_host_.data(), block_ids_host_.data() + B);
    trace->x_post_attn.assign(static_cast<size_t>(cfg_.block_count), {});
    trace->x_post_ffn.assign(static_cast<size_t>(cfg_.block_count), {});
  }
  for (int64_t il = 0; il < cfg_.block_count; ++il) ForwardLayer(stream, arena, il, trace);

  r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_.data()),
                          reinterpret_cast<int64_t>(output_norm_.data()),
                          reinterpret_cast<int64_t>(xf_.data()), B, hidden, eps, 0, s);
  // The SAME x_final feeds both the lm_head (via the target's own head) and the selector gate.
  lm_head(stream, arena, xf_.data(), logits_dev_.data(), B);

  uint8_t* stage = sel_stage_dev_.data();
  r4dx_topk16_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                  reinterpret_cast<int64_t>(stage + sel_off_cand_),
                  reinterpret_cast<int64_t>(stage + sel_off_unary_), static_cast<int>(B),
                  lm_head_vocab_, s);
  ApplyLinear(stream, arena, selector_hidden_, xf_.data(),
              reinterpret_cast<uint16_t*>(stage + sel_off_gate_), B);

  // THE one device->host copy of the round, followed by THE one synchronize.
  // ev1 goes BEFORE the readback, not after: a small async D2H into PINNED host memory can be
  // serviced by the SDMA/blit engine rather than the compute queue, and an event recorded behind it
  // then carries that queue's timestamp instead -- which is what made this instrumentation report
  // 0.03-0.27 ms for a round whose wall clock is 6-10 ms (with the occasional correct sample).
  // The copy itself is 5 KB and is excluded from the reported figure by exactly that amount.
  if (device_ms != nullptr) R4DX_HIP_CHECK(hipEventRecord(ev1, stream.get()));
  sel_stage_dev_.CopyToHostAsync(sel_stage_host_.data(), static_cast<size_t>(sel_stage_bytes_),
                                 stream);
  stream.Synchronize();
  if (device_ms != nullptr) {
    float ms = 0.0f;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    *device_ms = static_cast<double>(ms);
    R4DX_HIP_CHECK(hipEventDestroy(ev0));
    R4DX_HIP_CHECK(hipEventDestroy(ev1));
  }

  const auto* unary = reinterpret_cast<const float*>(sel_stage_host_.data() + sel_off_unary_);
  const auto* cand = reinterpret_cast<const int32_t*>(sel_stage_host_.data() + sel_off_cand_);
  const auto* gate = reinterpret_cast<const uint16_t*>(sel_stage_host_.data() + sel_off_gate_);

  if (trace != nullptr) {
    trace->x_final_normed = xf_.CopyToHost();
    trace->logits.resize(static_cast<size_t>(B * lm_head_vocab_));
    logits_dev_.CopyToHost(trace->logits.data(), trace->logits.size());
    trace->cand.assign(cand, cand + B * topk);
    trace->unary.assign(unary, unary + B * topk);
    trace->gate.resize(static_cast<size_t>(B * rank));
    for (int64_t i = 0; i < B * rank; ++i) {
      trace->gate[static_cast<size_t>(i)] = core::Bf16ToFloat(gate[i]);
    }
  }

  return SelectorWalk(anchor_id, k, p_min, n_min, cand, unary, gate, trace);
}

DflashDraftResult DflashDraft::SelectorWalk(int32_t anchor_id, int64_t k, float p_min,
                                            int64_t n_min, const int32_t* cand, const float* unary,
                                            const uint16_t* gate, DflashRoundTrace* trace) const {
  const int64_t B = cfg_.block_size;
  const int64_t rank = cfg_.selector_rank;
  const int64_t topk = cfg_.selector_top_k;

  DflashDraftResult out;
  std::vector<int32_t> P{anchor_id};
  int64_t pred_idx = 0;
  std::vector<float> cond(static_cast<size_t>(rank));
  std::vector<float> row(static_cast<size_t>(topk));
  if (trace != nullptr) {
    trace->score.assign(static_cast<size_t>(B - 1), {});
    trace->walk_prob.assign(static_cast<size_t>(B - 1), 0.0f);
    trace->walk_b.assign(static_cast<size_t>(B - 1), -1);
  }

  for (int64_t t = 1; t < B; ++t) {
    // The `k` cap is checked BEFORE emitting, not after: checked after, `k == 0` would still emit
    // position 1's token (size() >= 0 is true only once a token is already in the vector). k == 0
    // means "draft nothing", which a driver uses to fall back to a plain verify of the anchor alone.
    if (static_cast<int64_t>(out.tokens.size()) >= k) break;
    const int32_t* cand_t = cand + t * topk;
    const float* unary_t = unary + t * topk;
    const uint16_t* gate_t = gate + t * rank;

    // score[a][b] = sum_r successor[cand_t[b]][r] * predecessor[P[a]][r] * gate_t[r] + unary_t[b]
    // (docs/dflash2.md section 4.3). The walk only ever reads row `pred_idx`, so only that row is
    // computed on the hot path; the FULL matrix is materialised only for a trace (tests compare it
    // against the reference's own score_t{1..7} fixtures).
    if (trace != nullptr) {
      std::vector<float>& mat = trace->score[static_cast<size_t>(t - 1)];
      mat.assign(static_cast<size_t>(P.size()) * static_cast<size_t>(topk), 0.0f);
      for (size_t a = 0; a < P.size(); ++a) {
        const uint16_t* pred_a = SelectorRow(selector_predecessor_, P[a]);
        for (int64_t r = 0; r < rank; ++r) {
          cond[static_cast<size_t>(r)] = core::Bf16ToFloat(pred_a[r]) * core::Bf16ToFloat(gate_t[r]);
        }
        for (int64_t b = 0; b < topk; ++b) {
          const uint16_t* succ_b = SelectorRow(selector_successor_, cand_t[b]);
          float acc = 0.0f;
          for (int64_t r = 0; r < rank; ++r) {
            acc += cond[static_cast<size_t>(r)] * core::Bf16ToFloat(succ_b[r]);
          }
          mat[a * static_cast<size_t>(topk) + static_cast<size_t>(b)] = acc + unary_t[b];
        }
      }
      for (int64_t b = 0; b < topk; ++b) {
        row[static_cast<size_t>(b)] =
            mat[static_cast<size_t>(pred_idx) * static_cast<size_t>(topk) + static_cast<size_t>(b)];
      }
    } else {
      const uint16_t* pred_a = SelectorRow(selector_predecessor_, P[static_cast<size_t>(pred_idx)]);
      for (int64_t r = 0; r < rank; ++r) {
        cond[static_cast<size_t>(r)] = core::Bf16ToFloat(pred_a[r]) * core::Bf16ToFloat(gate_t[r]);
      }
      for (int64_t b = 0; b < topk; ++b) {
        const uint16_t* succ_b = SelectorRow(selector_successor_, cand_t[b]);
        float acc = 0.0f;
        for (int64_t r = 0; r < rank; ++r) {
          acc += cond[static_cast<size_t>(r)] * core::Bf16ToFloat(succ_b[r]);
        }
        row[static_cast<size_t>(b)] = acc + unary_t[b];
      }
    }

    int64_t b_best = 0;
    for (int64_t b = 1; b < topk; ++b) {
      if (row[static_cast<size_t>(b)] > row[static_cast<size_t>(b_best)]) b_best = b;
    }
    // p_min: the softmax probability of the argmax within this row, i.e.
    // 1 / sum_b exp(row[b] - row_max) evaluated AT the argmax (speculative.cpp:1241-1249, quoted in
    // docs/dflash2.md's "Early stop" row). The position that trips the gate is NOT emitted, and the
    // walk stops there.
    const float smax = row[static_cast<size_t>(b_best)];
    double denom = 0.0;
    for (int64_t b = 0; b < topk; ++b) {
      denom += std::exp(static_cast<double>(row[static_cast<size_t>(b)]) - smax);
    }
    const float prob = static_cast<float>(1.0 / denom);
    if (trace != nullptr) {
      trace->walk_prob[static_cast<size_t>(t - 1)] = prob;
      trace->walk_b[static_cast<size_t>(t - 1)] = static_cast<int32_t>(b_best);
    }
    if (p_min > 0.0f && prob < p_min) {
      out.stopped_by_p_min = true;
      break;
    }

    out.tokens.push_back(cand_t[b_best]);
    pred_idx = b_best;
    P.assign(cand_t, cand_t + topk);
  }

  out.walk_len = static_cast<int64_t>(out.tokens.size());
  // n_min: the reference driver discards the ENTIRE draft when the walk produced fewer than n_min
  // tokens (speculative.cpp:1254-1256 / docs/dflash2.md sections 4.3 and 5 step 3) rather than
  // verifying a too-short block.
  if (n_min > 0 && out.walk_len < n_min) {
    out.tokens.clear();
    out.discarded_by_n_min = true;
  }
  return out;
}

std::vector<uint16_t> DflashDraft::DebugEncodedG(core::Stream& stream, int64_t rows) const {
  stream.Synchronize();
  const int64_t n = rows * cfg_.hidden_size;
  std::vector<uint16_t> out(static_cast<size_t>(n));
  if (n > 0) g_dev_.CopyToHost(out.data(), out.size());
  return out;
}

std::vector<uint16_t> DflashDraft::DebugStoreK(core::Stream& stream, int64_t layer,
                                               int64_t pos_begin, int64_t count) const {
  stream.Synchronize();
  const int64_t kv_row = cfg_.attention.head_count_kv * cfg_.attention.key_length;
  std::vector<uint16_t> out(static_cast<size_t>(count * kv_row));
  for (int64_t i = 0; i < count; ++i) {
    const int64_t slot = (pos_begin + i) % slots_;
    R4DX_HIP_CHECK(hipMemcpy(out.data() + i * kv_row,
                             k_store_.data() + (layer * slots_ + slot) * kv_row,
                             static_cast<size_t>(kv_row) * sizeof(uint16_t),
                             hipMemcpyDeviceToHost));
  }
  return out;
}

std::vector<uint16_t> DflashDraft::DebugStoreV(core::Stream& stream, int64_t layer,
                                               int64_t pos_begin, int64_t count) const {
  stream.Synchronize();
  const int64_t kv_row = cfg_.attention.head_count_kv * cfg_.attention.key_length;
  std::vector<uint16_t> out(static_cast<size_t>(count * kv_row));
  for (int64_t i = 0; i < count; ++i) {
    const int64_t slot = (pos_begin + i) % slots_;
    R4DX_HIP_CHECK(hipMemcpy(out.data() + i * kv_row,
                             v_store_.data() + (layer * slots_ + slot) * kv_row,
                             static_cast<size_t>(kv_row) * sizeof(uint16_t),
                             hipMemcpyDeviceToHost));
  }
  return out;
}

}  // namespace r4dx::model
