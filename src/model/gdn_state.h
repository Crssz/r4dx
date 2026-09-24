// r4dx::model::GdnStateManager -- the two pieces of per-sequence GDN state (docs/architecture.md
// "GDN state"), sized for up to `max_seqs` concurrent sequences and shared by every GDN layer's
// own instance (one GdnStateManager per layer -- state does not cross layers).
//
// Slot 0 is permanently reserved/unused: r4d_gdn_conv_update_w4_h128_bf16 and
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state both treat a slot index <= 0 as
// "NULL_BLOCK_ID, nothing scheduled here" and silently no-op (see their .hip source comments), so
// a caller that assigned sequence 0 to physical slot 0 would get silently-dropped decode calls.
// Sequence i (0-based) therefore lives at physical slot i+1; SlotForSeq()/RecurrentSlotPtr() below
// apply that offset once so callers never have to remember it.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

class GdnStateManager {
 public:
  // H,V,K: r4d_gdn_dims() (48/128/128 -- value heads, head_v, head_k in this model).
  // conv_dim: 2*key_dim + value_dim (10240 in this model). conv_width: linear_conv_kernel_dim
  // (4). max_decode_window: the largest number of candidate tokens ApplyDecode will ever be asked
  // to process in one call (a plain single-token decode needs 1; an MTP speculative-verify window
  // needs draft_k+1) -- the conv state's rolling buffer depth is conv_width-2 + max_decode_window
  // (r4d_gdn_conv_w4_h128_bf16.hip's state_len_max: the kernel's decode-side rewrite loop is only
  // self-consistent when `slen_eff == slen + width - 2`, i.e. `state_len_max == max_query_len +
  // width - 2`, NOT width-1 -- verified against r4d_gdn_conv_update_kernel's slen_eff/VAL
  // derivation, which needs a depth of exactly CP_ST==width-1 for a plain single-token decode
  // (max_decode_window==1), not width). The recurrent state (MTP pass, 2026-09-19) additionally
  // reserves `max_decode_window` PHYSICAL SLOTS per sequence rather than one: per
  // r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip's kernel body, `sidx[t]` is the slot
  // candidate token `t` of a call WRITES its post-token state snapshot into (never overwriting the
  // slot a still-in-flight verification might need), and a later call's `num_accepted` device
  // pointer (`naccept[n]`) selects which of THOSE slots (`sidx[naccept[n]-1]`) to seed the next
  // call's initial state from -- see WindowSlot()/GdnControlCache::SidxBase() below for how this
  // model threads that: `sidx` is always the SAME ascending {WindowSlot(seq,0)..
  // WindowSlot(seq,max_decode_window-1)} array every call (indices beyond a short call's own T are
  // simply never read/written that call), so a `num_accepted` carried from one call to the next
  // indexes consistently across calls without any reallocation or renumbering.
  GdnStateManager(int64_t max_seqs, int64_t H, int64_t V, int64_t K, int64_t conv_dim,
                  int64_t conv_width, int64_t max_decode_window)
      : H_(H),
        V_(V),
        K_(K),
        conv_dim_(conv_dim),
        state_len_max_(conv_width - 2 + max_decode_window),
        max_decode_window_(max_decode_window < 1 ? 1 : max_decode_window),
        recurrent_(static_cast<size_t>((max_seqs * max_decode_window_ + 1) * H * V * K)),
        conv_(static_cast<size_t>((max_seqs + 1) * conv_dim * state_len_max_)) {}

  // seq_id is 0-based. SlotForSeq is WindowSlot(seq_id, 0) -- the physical slot a plain
  // (non-speculative, window index 0) decode step always reads and writes in place, unchanged from
  // this class's pre-MTP behavior when max_decode_window==1.
  int32_t SlotForSeq(int32_t seq_id) const { return 1 + seq_id * static_cast<int32_t>(max_decode_window_); }
  int32_t WindowSlot(int32_t seq_id, int32_t window_idx) const { return SlotForSeq(seq_id) + window_idx; }
  int64_t MaxDecodeWindow() const { return max_decode_window_; }

  int64_t RecurrentSlotStride() const { return H_ * V_ * K_; }
  int64_t RecurrentHeadStride() const { return V_ * K_; }
  float* RecurrentBase() { return recurrent_.data(); }
  float* RecurrentSlotPtr(int32_t slot) { return recurrent_.data() + static_cast<int64_t>(slot) * RecurrentSlotStride(); }

  int64_t ConvSeqStride() const { return conv_dim_ * state_len_max_; }  // cs_seq
  int64_t ConvDimStride() const { return state_len_max_; }              // cs_dim
  int64_t ConvTokStride() const { return 1; }                           // cs_tok
  int64_t StateLenMax() const { return state_len_max_; }
  uint16_t* ConvBase() { return conv_.data(); }

  void ZeroAll(core::Stream& stream) {
    recurrent_.ZeroAsync(stream);
    conv_.ZeroAsync(stream);
  }

 private:
  int64_t H_, V_, K_, conv_dim_, state_len_max_, max_decode_window_;
  core::DeviceBuffer<float> recurrent_;
  core::DeviceBuffer<uint16_t> conv_;
};

// r4dx::model::GdnControlCache -- caches the tiny per-call control arrays every GdnLayer::Forward
// call uploads to the device (cu={bos,bos+T}, cache_idx={slot}, sidx={slot,...} T times,
// has_init={1}), keyed by the (T, slot) values that determine their contents.
//
// In this model's single-sequence scope (Model's own doc comment: SCOPE num_seqs==1) these are
// pure functions of T and the (always-constant, ==1) slot, so every distinct value only needs
// uploading ONCE, ever: the first Get() for a given key does one hipMemcpy into a freshly
// hipMalloc'd buffer (safe without any stream synchronization -- a brand-new allocation is never
// concurrently read/written by another in-flight kernel, unlike the arena's reused bytes), then
// every later call for the same key reuses the cached device pointer with zero uploads and zero
// host-blocking syncs. Decode's hot path (T==1, slot constant across the whole session) therefore
// uploads exactly once total across the whole generation, not once per token per layer -- see
// gdn_layer.cpp's UploadArray, which this replaces (that helper's hipStreamSynchronize per call
// was 3 host-blocking pipeline drains x 48 GDN layers = 144 syncs per decode token).
class GdnControlCache {
 public:
  const int32_t* CuPair(int64_t T) {
    return Get(cu_, T, [T] { return std::vector<int32_t>{0, static_cast<int32_t>(T)}; });
  }
  const int32_t* CacheIdx(int32_t slot) {
    return Get(cache_idx_, static_cast<int64_t>(slot), [slot] { return std::vector<int32_t>{slot}; });
  }
  const int32_t* Sidx(int64_t T, int32_t slot) {
    const int64_t key = (T << 20) ^ static_cast<int64_t>(slot);  // T, slot are both tiny (<=64)
    return Get(sidx_, key,
               [T, slot] { return std::vector<int32_t>(static_cast<size_t>(T), slot); });
  }

  // MTP verify pass (2026-09-19): the stable, ASCENDING {base_slot, base_slot+1, ...,
  // base_slot+width-1} array recurrent_update's `sidx`/`indices_stride` and conv_update's
  // `num_accepted`-driven seed read both need (see GdnStateManager's file comment) -- unlike
  // Sidx() above (every entry the same physical slot, right for a plain non-speculative decode's
  // T==1 call, wrong for a speculative window where each candidate token must land in its OWN
  // slot). `width` must be >= the caller's GdnStateManager::MaxDecodeWindow() (the number of
  // physical slots actually reserved for this base_slot's sequence -- gdn_layer.cpp always passes
  // `states.MaxDecodeWindow()` here, deriving it from the SAME GdnStateManager instance the call's
  // T is bounded by, rather than a separately-configured global that a caller could forget to set
  // and silently under-size (review finding: an earlier version of this class stored its own
  // Model-wide max_window_ set once via a since-removed SetMaxWindow(), which every NON-Model
  // caller -- e.g. tests/model/test_gdn_layer.cpp's direct GdnLayer usage at T=4 -- had no reason
  // to know it needed to call; the resulting 1-element array was read/written out of bounds at
  // t=1..3, an illegal device memory access that manifested as a hang, not a clean crash). Cached
  // by (base_slot, width).
  const int32_t* SidxBase(int32_t base_slot, int64_t width) {
    const int64_t key = (static_cast<int64_t>(base_slot) << 32) ^ width;
    return Get(sidx_base_, key, [base_slot, width] {
      std::vector<int32_t> v(static_cast<size_t>(width));
      for (int64_t i = 0; i < width; ++i) v[static_cast<size_t>(i)] = base_slot + static_cast<int32_t>(i);
      return v;
    });
  }
  const uint8_t* HasInitTrue() {
    if (has_init_true_.empty()) {
      if (frozen_) Miss("HasInitTrue", 0);
      has_init_true_ = core::DeviceBuffer<uint8_t>(1);
      const uint8_t v = 1;
      has_init_true_.CopyFromHost(&v, 1);
    }
    return has_init_true_.data();
  }

  // ---- tensor parallel (docs/tp.md 2.7, 2.9 step 9) -------------------------------------------
  // A first use of a key costs a hipMalloc and a blocking hipMemcpy -- both implicit device syncs,
  // which must never happen inside a tensor-parallel collective command (docs/tp.md 6.3.7 L4). The
  // TP warm-up (Model::TpWarmup) therefore uploads every key a single sequence can ever ask for
  // BEFORE it runs, and freezes the cache after it: CuPair(1..max_T), CacheIdx(slot),
  // SidxBase(slot, window) and HasInitTrue(). After Freeze() a miss throws std::logic_error instead
  // of allocating. Never called at TP=1, where the cache stays lazy exactly as before.
  void Prewarm(int64_t max_T, int32_t slot, int64_t window) {
    for (int64_t T = 1; T <= max_T; ++T) (void)CuPair(T);
    (void)CacheIdx(slot);
    (void)SidxBase(slot, window);
    (void)HasInitTrue();
  }
  void Freeze() { frozen_ = true; }
  bool Frozen() const { return frozen_; }

 private:
  [[noreturn]] static void Miss(const char* what, int64_t key) {
    throw std::logic_error(std::string("GdnControlCache::") + what + ": key " + std::to_string(key) +
                           " was not prewarmed before Freeze() (docs/tp.md 2.7: a lazy hipMalloc "
                           "inside a tensor-parallel collective command)");
  }

  template <typename MakeFn>
  const int32_t* Get(std::unordered_map<int64_t, core::DeviceBuffer<int32_t>>& map, int64_t key,
                      MakeFn make) {
    auto it = map.find(key);
    if (it != map.end()) return it->second.data();
    if (frozen_) Miss("Get", key);
    std::vector<int32_t> host = make();
    core::DeviceBuffer<int32_t> buf(host.size());
    buf.CopyFromHost(host);
    auto res = map.emplace(key, std::move(buf));
    return res.first->second.data();
  }

  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> cu_;
  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> cache_idx_;
  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> sidx_;
  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> sidx_base_;
  core::DeviceBuffer<uint8_t> has_init_true_;
  bool frozen_ = false;
};

}  // namespace r4dx::model
