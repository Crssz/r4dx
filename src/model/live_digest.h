// HIP-free canonical digest of a Model's LIVE state (docs/pp-tp2-hybrid.md 4, gate G-H1): the bytes the next token would be computed
// from and nothing else, hashed in an order that does not depend on where the bytes live. A device Model (Model::
// DebugLiveStateDigest) and a host image (hybrid::LiveImage, what ReshardRef and the dump files hold) therefore hash identically
// when they hold the same state, whatever their layouts: a KV cache is [block][head][token][2*head_dim] on the device, the
// digest walks (layer, head, row) records; a speculating rank's conv line is longer than the three live entries, the digest sees
// the three; the rows of the last KV block past the position, the recurrent slots a window leaves behind, the stale tail of a
// long conv line and the MTP head's one unprimed row are not state and are not hashed (Model::DebugStateDigest hashes the whole
// buffers and is valid only for cold-after-Reset scenarios; the design's critique item 9).
//
// One 64-bit FNV-1a per record group:
//   "pos"                 the sequence position
//   "kv.<layer>"          rows [0, pos) of each head of an attention layer, head-major
//   "mtp.kv"              the MTP head's KV, rows [0, MtpLiveRows(pos)) -- row pos-1 is primed by the NEXT call, not live
//   "gdn.rec.<layer>"     the live recurrent slot, [v_heads][V][K] fp32
//   "gdn.conv.<layer>"    the live conv entries, compact [channels][conv_live] bf16, channels ascending
//   "dflash.injected" / "dflash.lo" / "dflash.k.<l>" / "dflash.v.<l>"   the drafter's visible window [lo, injected)
// A device Model streams its KV through KvDigester in block groups (so a 128k-token cache never sits on the host whole); the host
// image goes through the same class, which is what makes "identical state => identical digest" a property of one piece of code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "reshard_plan.h"

namespace r4dx::model::hybrid {

constexpr uint64_t kFnvInit = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

inline uint64_t FnvUpdate(uint64_t h, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= kFnvPrime;
  }
  return h;
}
inline uint64_t FnvU64(uint64_t h, uint64_t v) { return FnvUpdate(h, &v, sizeof(v)); }

// A byte range with its length mixed in (so ranges of different lengths cannot collide on a common prefix).
inline uint64_t DigestBytes(const void* data, size_t n) { return FnvUpdate(FnvU64(kFnvInit, static_cast<uint64_t>(n)), data, n); }

// ---- the KV records ---------------------------------------------------------------------------------------------------------
// Canonical records of one attention layer: for head in [0, kv_heads): for row in [0, rows): the 2 * head_dim fp8 bytes of K then V.
// Fed whole blocks in ascending order ([block][head][token][row_bytes], the cache layout); rows past `rows` are never hashed.
class KvDigester {
 public:
  KvDigester(int64_t kv_heads, int64_t block_tokens, int64_t row_bytes, int64_t rows)
      : heads_(kv_heads), block_tokens_(block_tokens), row_bytes_(row_bytes), rows_(rows < 0 ? 0 : rows),
        h_(static_cast<size_t>(kv_heads > 0 ? kv_heads : 0), kFnvInit) {
    if (kv_heads < 1 || block_tokens < 1 || row_bytes < 1) throw std::invalid_argument("KvDigester: heads, block_tokens and row_bytes must be positive");
  }
  int64_t BlocksNeeded() const { return (rows_ + block_tokens_ - 1) / block_tokens_; }
  int64_t BlockBytes() const { return heads_ * block_tokens_ * row_bytes_; }
  int64_t NextBlock() const { return next_block_; }
  // `n_blocks` whole blocks starting at block `first_block` (which must be the next one wanted). Blocks past BlocksNeeded() are ignored.
  void Feed(const uint8_t* blocks, int64_t first_block, int64_t n_blocks) {
    if (first_block != next_block_) throw std::invalid_argument("KvDigester::Feed: blocks must arrive in order without gaps");
    const int64_t last = std::min(first_block + n_blocks, BlocksNeeded());
    const int64_t head_block = block_tokens_ * row_bytes_;
    for (int64_t b = first_block; b < last; ++b) {
      const int64_t rows_here = std::min(block_tokens_, rows_ - b * block_tokens_);
      const uint8_t* block = blocks + static_cast<size_t>((b - first_block) * BlockBytes());
      for (int64_t h = 0; h < heads_; ++h) {
        h_[static_cast<size_t>(h)] = FnvUpdate(h_[static_cast<size_t>(h)], block + static_cast<size_t>(h * head_block), static_cast<size_t>(rows_here * row_bytes_));
      }
    }
    next_block_ = std::max(next_block_, last);
  }
  uint64_t Finish() const {
    if (next_block_ != BlocksNeeded()) throw std::logic_error("KvDigester::Finish: not every block was fed");
    uint64_t f = FnvU64(FnvU64(kFnvInit, static_cast<uint64_t>(heads_)), static_cast<uint64_t>(rows_));
    for (const uint64_t h : h_) f = FnvU64(f, h);
    return f;
  }

 private:
  int64_t heads_, block_tokens_, row_bytes_, rows_;
  int64_t next_block_ = 0;
  std::vector<uint64_t> h_;
};

// One-shot over a host KV image that holds at least BlocksNeeded() whole blocks.
inline uint64_t DigestKvImage(const std::vector<uint8_t>& image, int64_t kv_heads, int64_t block_tokens, int64_t row_bytes, int64_t rows) {
  KvDigester d(kv_heads, block_tokens, row_bytes, rows);
  const int64_t block_bytes = d.BlockBytes();
  if (static_cast<int64_t>(image.size()) % block_bytes != 0 || static_cast<int64_t>(image.size()) / block_bytes < d.BlocksNeeded()) {
    throw std::invalid_argument("DigestKvImage: the image holds fewer than the " + std::to_string(d.BlocksNeeded()) + " whole blocks that " +
                                std::to_string(rows) + " rows need");
  }
  d.Feed(image.data(), 0, d.BlocksNeeded());
  return d.Finish();
}

// The MTP head's KV is primed one position behind the backbone: after a call that ended at `pos`, rows [0, pos - 1) are written
// (row pos - 1 needs the NEXT call's first token), so only those are state.
inline int64_t MtpLiveRows(int64_t pos) { return pos > 0 ? pos - 1 : 0; }

// ---- the DFlash window ------------------------------------------------------------------------------------------------------
// The drafter's ring holds the last `slots` injected positions; the store a draft block can read is [valid_from, injected)
// intersected with the last `slots` of them. [lo, hi) of it, empty when nothing was injected. Equal for a drafter that was fed every
// row and one that was fed only a tail of >= slots rows (valid_from <= injected - slots), which is the equivalence the hybrid's
// tail injection relies on.
inline void DflashWindow(int64_t injected, int64_t valid_from, int64_t slots, int64_t* lo, int64_t* hi) {
  *hi = injected < 0 ? 0 : injected;
  *lo = std::max<int64_t>(std::max<int64_t>(valid_from, 0), *hi - slots);
  if (*lo > *hi) *lo = *hi;
}
// `count` rows of `row_elems` bf16 (kv_heads * head_dim), positions lo, lo + 1, ... in ring-slot-independent order.
inline uint64_t DigestDflashRows(const uint16_t* rows, int64_t count, int64_t row_elems, int64_t lo) {
  uint64_t h = FnvU64(FnvU64(kFnvInit, static_cast<uint64_t>(lo)), static_cast<uint64_t>(count));
  return FnvUpdate(h, rows, static_cast<size_t>(count * row_elems) * sizeof(uint16_t));
}

// ---- the list ---------------------------------------------------------------------------------------------------------------
using DigestList = std::vector<std::pair<std::string, uint64_t>>;
inline std::string DigestName(const char* kind, int64_t layer) { return std::string(kind) + "." + std::to_string(layer); }

// What one holder's per-layer buffers are made of (a full Model: all KV heads; a rank: its share).
struct DigestShape {
  int64_t block_tokens = 16;
  int64_t kv_heads = 0;
  int64_t kv_row_bytes = 0;  // 2 * head_dim (K then V, fp8)
  int64_t conv_live = 3;
};

// The digest of a host image at position `pos`: the same records, names and values a device Model with that live state reports.
inline DigestList DigestLiveImage(const LiveImage& img, const DigestShape& sh, int64_t pos) {
  DigestList out;
  out.emplace_back("pos", static_cast<uint64_t>(pos));
  for (const auto& [layer, bytes] : img.kv) out.emplace_back(DigestName("kv", layer), DigestKvImage(bytes, sh.kv_heads, sh.block_tokens, sh.kv_row_bytes, pos));
  if (!img.mtp_kv.empty()) out.emplace_back("mtp.kv", DigestKvImage(img.mtp_kv, sh.kv_heads, sh.block_tokens, sh.kv_row_bytes, MtpLiveRows(pos)));
  for (const auto& [layer, bytes] : img.gdn_rec) out.emplace_back(DigestName("gdn.rec", layer), DigestBytes(bytes.data(), bytes.size()));
  for (const auto& [layer, conv] : img.gdn_conv) out.emplace_back(DigestName("gdn.conv", layer), DigestBytes(conv.data(), conv.size() * sizeof(uint16_t)));
  return out;
}

// "" when the two lists name the same records with the same values (order is irrelevant), else the first difference.
inline std::string DiffDigests(const DigestList& a, const DigestList& b) {
  std::map<std::string, uint64_t> ma, mb;
  for (const auto& [k, v] : a) ma[k] = v;
  for (const auto& [k, v] : b) mb[k] = v;
  for (const auto& [k, v] : ma) {
    const auto it = mb.find(k);
    if (it == mb.end()) return "record '" + k + "' is missing on the right";
    if (it->second != v) return "record '" + k + "' differs";
  }
  for (const auto& [k, v] : mb) {
    (void)v;
    if (ma.find(k) == ma.end()) return "record '" + k + "' is missing on the left";
  }
  return "";
}

}  // namespace r4dx::model::hybrid
