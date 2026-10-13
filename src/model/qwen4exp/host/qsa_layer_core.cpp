// QsaLayerCore: see include/r4dx/model/qwen4exp/qsa.h.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/error.hpp"
#include "r4dx/model/qwen4exp/fuse_flags.h"
#include "r4dx/model/qwen4exp/qsa.h"
#include "r4dx/model/qwen4exp/qsa_kernels.h"

namespace r4dx::model::qwen4exp {

namespace {

constexpr int kGroup = 12;
constexpr int kTileTokens = 256;
constexpr size_t kScoreScratchBytes = size_t{16} << 20;  // target size of the score scratch

int CeilDiv(int a, int b) { return (a + b - 1) / b; }

// Arena::Alloc rounds every allocation's end up to 16 bytes; the sizes below add that slack explicitly.
size_t Padded(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

// Split schedule of the attention kernel. A decode-sized chunk has too few (query, kv head) pairs to fill the GPU,
// so its token tiles are spread over workgroups (flash-decoding splits); a prefill-sized chunk has plenty, so one
// workgroup per (query, kv head) walks all its tiles and the partial buffers stay small.
int TilesPerSplit(int T, int ntiles) {
  if (T <= 16) return 1;
  if (T <= 64) return std::min(ntiles, 3);
  return ntiles;
}

}  // namespace

QsaLayerCore::QsaLayerCore(const QsaDims& dims) : dims_(dims) {
  const auto bad = [&](const char* what) {
    throw std::invalid_argument(std::string("QsaLayerCore: ") + what);
  };
  if (dims_.heads_kv <= 0 || dims_.heads_q != dims_.heads_kv * kGroup) bad("heads_q must be 12 * heads_kv");
  if (dims_.head_dim != 256) bad("head_dim must be 256");
  if (dims_.index_dim != 128 || dims_.index_heads <= 0 || dims_.index_heads > 8) bad("index_dim must be 128, index_heads in 1..8");
  if (dims_.index_heads != 4) bad("the score kernel is written for index_heads == 4");
  if (dims_.rotary_dim <= 0 || dims_.rotary_dim % 2 != 0 || dims_.rotary_dim > 64) bad("rotary_dim must be even and in 2..64");
  if (dims_.compress_ratio <= 0 || dims_.compress_ratio > 64) bad("compress_ratio must be in 1..64");
  if (dims_.budget <= 0 || dims_.budget % dims_.compress_ratio != 0) bad("budget must be a positive multiple of compress_ratio");
  if (dims_.max_ctx <= 0) bad("max_ctx must be positive");
  if (!(dims_.rope_theta > 1.0)) bad("rope_theta must be > 1");

  topk_ = dims_.budget / dims_.compress_ratio;
  max_blocks_ = CeilDiv(dims_.max_ctx, dims_.compress_ratio);
  sel_stride_ = dims_.budget + dims_.compress_ratio - 1;
  ntiles_max_ = CeilDiv(sel_stride_, kTileTokens);
  const size_t row_bytes = static_cast<size_t>(max_blocks_) * sizeof(float);
  score_rows_ = static_cast<int>(std::clamp<size_t>(kScoreScratchBytes / row_bytes, 4, kMaxChunk));

  // HF Qwen4ExpTextRotaryEmbedding.compute_default_rope_parameters: 1 / (theta ** (arange(0, dim, 2) / dim)) in
  // fp32 (the base is exactly representable: 1e7).
  const int half = dims_.rotary_dim / 2;
  const float theta = static_cast<float>(dims_.rope_theta);
  for (int i = 0; i < half; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(dims_.rotary_dim);
    inv_freq_host_[i] = 1.0f / std::pow(theta, e);
  }
  inv_freq_.Resize(static_cast<size_t>(half));
  inv_freq_.CopyFromHost(std::vector<float>(inv_freq_host_, inv_freq_host_ + half));

  raw_keys_.Resize(static_cast<size_t>(dims_.max_ctx) * dims_.index_dim);
  block_keys_.Resize(static_cast<size_t>(max_blocks_) * dims_.index_dim);
  scores_.Resize(static_cast<size_t>(score_rows_) * max_blocks_);
  sel_tokens_.Resize(static_cast<size_t>(kMaxChunk) * sel_stride_);
  sel_count_.Resize(kMaxChunk);
  raw_keys_.Zero();
  block_keys_.Zero();
}

void QsaLayerCore::TruncateTo(int pos) {
  if (pos < 0) throw std::invalid_argument("QsaLayerCore::TruncateTo: pos must be >= 0");
  valid_pos_ = std::min(valid_pos_, pos);
}

size_t QsaLayerCore::ArenaBytes(int T) const {
  const size_t hq = static_cast<size_t>(dims_.heads_q), hkv = static_cast<size_t>(dims_.heads_kv);
  const size_t t = static_cast<size_t>(T);
  // nsplit at its largest for this T (a short context launches fewer)
  const int nsplit = CeilDiv(ntiles_max_, TilesPerSplit(T, ntiles_max_));
  size_t bytes = 0;
  bytes += Padded(t * hq * dims_.head_dim * sizeof(uint16_t));                                     // q
  bytes += Padded(t * dims_.index_heads * dims_.index_dim * sizeof(uint16_t));                      // indexer q
  bytes += Padded(t * hq * dims_.head_dim * sizeof(uint16_t));                                     // attn_out
  bytes += Padded(t * hkv * nsplit * kGroup * dims_.head_dim * sizeof(float));                      // part_acc
  bytes += Padded(t * hkv * nsplit * kGroup * 2 * sizeof(float));                                   // part_ml
  return bytes;
}

uint16_t* QsaLayerCore::Forward(hipStream_t stream, core::Arena& arena, const QsaInputs& in, const QsaWeights& w,
                                attention::PagedKvCache& kv, int first_pos, int T, const int32_t* positions) {
  if (T < 1 || T > kMaxChunk) throw std::invalid_argument("QsaLayerCore::Forward: T must be in 1..256");
  if (first_pos < 0 || first_pos > valid_pos_) {
    throw std::runtime_error("QsaLayerCore::Forward: first_pos " + std::to_string(first_pos) +
                             " is beyond the valid positions (" + std::to_string(valid_pos_) +
                             "): the layer state would have a hole");
  }
  if (first_pos + T > dims_.max_ctx) {
    throw std::out_of_range("QsaLayerCore::Forward: first_pos + T exceeds max_ctx");
  }
  if (kv.KvHeads() != dims_.heads_kv || kv.HeadDim() != dims_.head_dim || kv.BlockSize() != 16) {
    throw std::invalid_argument("QsaLayerCore::Forward: the KV cache must be (kv_heads, head_dim 256, block 16)");
  }
  kv.CheckCapacity(first_pos, T);
  // r4dx_q4_qsa_prep writes the KV rows at pos / block_size without the block table while r4dx_q4_qsa_attn reads them
  // through it: they agree only for the identity table PagedKvCache builds (BuildContiguousBlockTable). Checked once
  // per table (one small D2H), never in the steady state.
  if (kv.BlockTable() != verified_table_) {
    std::vector<int32_t> table(static_cast<size_t>(kv.MaxBlocks()));
    R4DX_HIP_CHECK(hipMemcpy(table.data(), kv.BlockTable(), table.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < table.size(); ++i) {
      if (table[i] != static_cast<int32_t>(i)) {
        throw std::runtime_error("QsaLayerCore::Forward: the KV cache block table is not the identity (entry " +
                                 std::to_string(i) + " is " + std::to_string(table[i]) +
                                 "); the KV write kernel bypasses the table");
      }
    }
    verified_table_ = kv.BlockTable();
  }
  if (!in.qg || !in.k || !in.v || !in.idxqk || !w.q_norm || !w.k_norm || !w.idx_q_norm || !w.idx_k_norm ||
      !w.k_descale || !w.v_descale || !positions) {
    throw std::invalid_argument("QsaLayerCore::Forward: null input / weight pointer");
  }

  const int hq = dims_.heads_q, hkv = dims_.heads_kv, ratio = dims_.compress_ratio;
  uint16_t* q_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * hq * dims_.head_dim);
  uint16_t* idx_q = arena.Alloc<uint16_t>(static_cast<size_t>(T) * dims_.index_heads * dims_.index_dim);
  uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * hq * dims_.head_dim);
  const int nsplit_cap = CeilDiv(ntiles_max_, TilesPerSplit(T, ntiles_max_));  // what ArenaBytes(T) reserves
  float* part_acc = arena.Alloc<float>(static_cast<size_t>(T) * hkv * nsplit_cap * kGroup * dims_.head_dim);
  float* part_ml = arena.Alloc<float>(static_cast<size_t>(T) * hkv * nsplit_cap * kGroup * 2);

  // ---- 1. norm + rope + KV write + raw index keys ----
  Q4QsaPrepArgs prep{};
  prep.qg = in.qg;
  prep.k = in.k;
  prep.v = in.v;
  prep.idxqk = in.idxqk;
  prep.q_norm = w.q_norm;
  prep.k_norm = w.k_norm;
  prep.idx_q_norm = w.idx_q_norm;
  prep.k_descale = w.k_descale;
  prep.v_descale = w.v_descale;
  prep.inv_freq = inv_freq_.data();
  prep.positions = positions;
  prep.q_out = q_out;
  prep.idx_q_out = idx_q;
  prep.raw_keys = raw_keys_.data();
  prep.kv_cache = kv.Data();
  prep.T = T;
  prep.heads_q = hq;
  prep.heads_kv = hkv;
  prep.head_dim = dims_.head_dim;
  prep.rotary_dim = dims_.rotary_dim;
  prep.index_heads = dims_.index_heads;
  prep.index_dim = dims_.index_dim;
  prep.block_size = kv.BlockSize();
  prep.eps = dims_.rms_eps;
  prep.kv_block_stride = kv.KvBlockStride();
  prep.kv_head_stride = kv.KvHeadStride();
  R4DX_HIP_CHECK(r4dx_q4_qsa_prep(&prep, stream));

  // ---- 2. block keys of the blocks completing in this chunk ----
  Q4QsaBlockKeyArgs bk{};
  bk.raw_keys = raw_keys_.data();
  bk.idx_k_norm = w.idx_k_norm;
  bk.inv_freq = inv_freq_.data();
  bk.positions = positions;
  bk.block_keys = block_keys_.data();
  bk.T = T;
  bk.ratio = ratio;
  bk.index_dim = dims_.index_dim;
  bk.rotary_dim = dims_.rotary_dim;
  bk.eps = dims_.rms_eps;
  R4DX_HIP_CHECK(r4dx_q4_qsa_block_keys(&bk, stream));

  // ---- 3. selection for the rows that see more than topk complete blocks ----
  // Query p is sparse iff (p + 1) / ratio > topk, i.e. p >= ratio * (topk + 1) - 1; p grows with the row.
  const int first_sparse_pos = ratio * (topk_ + 1) - 1;
  const int first_sparse_row = std::max(0, first_sparse_pos - first_pos);
  for (int r0 = first_sparse_row; r0 < T; r0 += score_rows_) {
    const int rows = std::min(score_rows_, T - r0);
    Q4QsaScoreArgs sc{};
    sc.idx_q = idx_q;
    sc.block_keys = block_keys_.data();
    sc.positions = positions;
    sc.scores = scores_.data();
    sc.row0 = r0;
    sc.rows = rows;
    sc.blocks_max = (first_pos + r0 + rows) / ratio;  // complete blocks of the last row of the sub-batch
    sc.score_stride = max_blocks_;
    sc.ratio = ratio;
    sc.topk = topk_;
    R4DX_HIP_CHECK(r4dx_q4_qsa_scores(&sc, stream));
    Q4QsaSelectArgs sel{};
    sel.scores = scores_.data();
    sel.positions = positions;
    sel.sel_tokens = sel_tokens_.data();
    sel.sel_count = sel_count_.data();
    sel.row0 = r0;
    sel.rows = rows;
    sel.score_stride = max_blocks_;
    sel.sel_stride = sel_stride_;
    sel.ratio = ratio;
    sel.topk = topk_;
    R4DX_HIP_CHECK(r4dx_q4_qsa_select(&sel, stream));
  }

  // ---- 4. attention over the selected tokens + output gate ----
  const int max_tokens = std::min(sel_stride_, first_pos + T);  // longest token list of any row of the chunk
  const int ntiles = CeilDiv(max_tokens, kTileTokens);
  const int tps = TilesPerSplit(T, ntiles);
  const int nsplit = CeilDiv(ntiles, tps);
  if (nsplit > nsplit_cap) throw std::logic_error("QsaLayerCore::Forward: split count above the reserved scratch");

  Q4QsaAttnArgs at{};
  at.q = q_out;
  at.qg = in.qg;
  at.kv_cache = kv.Data();
  at.block_table = kv.BlockTable();
  at.positions = positions;
  at.sel_tokens = sel_tokens_.data();
  at.sel_count = sel_count_.data();
  at.k_descale = w.k_descale;
  at.v_descale = w.v_descale;
  at.part_acc = part_acc;
  at.part_ml = part_ml;
  at.out = attn_out;
  at.T = T;
  at.heads_q = hq;
  at.heads_kv = hkv;
  at.sel_stride = sel_stride_;
  at.ratio = ratio;
  at.topk = topk_;
  at.tiles_per_split = tps;
  at.nsplit = nsplit;
  // Decode (one query) over at most a few tiles starts only heads_kv workgroups, i.e. two compute units: split the twelve query
  // heads of each kv head over four workgroups (three heads each; bit-identical, see qsa_attn.hip; 4 measured best of
  // 1 / 2 / 3 / 4 / 6 / 12 at 243 and 2000 tokens). R4DX_Q4_FUSE_QSA=0 keeps one workgroup per kv head.
  at.head_split = (T == 1 && (FuseMask() & kFuseQsa) != 0) ? 4 : 1;
  at.kv_block_stride = kv.KvBlockStride();
  at.kv_head_stride = kv.KvHeadStride();
  R4DX_HIP_CHECK(r4dx_q4_qsa_attn(&at, stream));

  valid_pos_ = first_pos + T;
  last_idx_q_ = idx_q;
  return attn_out;
}

}  // namespace r4dx::model::qwen4exp
