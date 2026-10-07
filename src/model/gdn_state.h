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

#include "gdn_write_once.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/r4d.hpp"
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
  //
  // `write_once` (docs/gdn-write-once.md, gdn_write_once.h): keep ONE recurrent state B per sequence plus
  // two ping-pong logs of the last verify window's per-row updates instead of `max_decode_window` state
  // slots per sequence. A verify call (GdnLayer::Forward, T > 1) stores its rows into the log and applies
  // the PREVIOUS call's accepted prefix to B inside its own seed load; Materialize() applies the pending
  // prefix to B on its own (a prefill, a digest). The window then only sizes the logs and the conv history;
  // WindowSlot() does not exist. Requires a window > 1 (otherwise there is nothing to save and the manager
  // is the window-slot one) and a conv_dim that matches the key-head geometry (the log's key rows are
  // sized from it).
  GdnStateManager(int64_t max_seqs, int64_t H, int64_t V, int64_t K, int64_t conv_dim,
                  int64_t conv_width, int64_t max_decode_window, bool write_once = false)
      : H_(H),
        V_(V),
        K_(K),
        conv_dim_(conv_dim),
        conv_width_(conv_width),
        state_len_max_(conv_width - 2 + max_decode_window),
        max_decode_window_(max_decode_window < 1 ? 1 : max_decode_window),
        plan_{max_seqs, max_decode_window_, write_once && max_decode_window_ > 1},
        recurrent_(static_cast<size_t>(plan_.RecurrentSlots() * H * V * K)),
        conv_(static_cast<size_t>((max_seqs + 1) * conv_dim * state_len_max_)) {
    if (plan_.write_once) {
      // conv_dim = 2 * Hg * K + H * V: the key-head count of THIS rank's geometry
      const int64_t key_part = conv_dim - H * V;
      if (key_part <= 0 || key_part % (2 * K) != 0 || max_seqs != 1) {
        throw std::logic_error(
            "GdnStateManager: write_once needs max_seqs == 1 and conv_dim == 2 * Hg * K + H * V");
      }
      Hg_ = key_part / (2 * K);
      const size_t floats = static_cast<size_t>(GdnLogFloats(H, Hg_, V, K, max_decode_window_));
      log_[0] = core::DeviceBuffer<float>(floats);
      log_[1] = core::DeviceBuffer<float>(floats);
    }
  }

  // seq_id is 0-based. SlotForSeq is WindowSlot(seq_id, 0) -- the physical slot a plain
  // (non-speculative, window index 0) decode step always reads and writes in place, unchanged from
  // this class's pre-MTP behavior when max_decode_window==1. Write-once: the sequence's one slot (B).
  int32_t SlotForSeq(int32_t seq_id) const { return plan_.SlotForSeq(seq_id); }
  int32_t WindowSlot(int32_t seq_id, int32_t window_idx) const {
    if (plan_.write_once) throw std::logic_error("GdnStateManager::WindowSlot: a write-once manager has no window slots");
    return SlotForSeq(seq_id) + window_idx;
  }
  int64_t MaxDecodeWindow() const { return max_decode_window_; }
  bool WriteOnce() const { return plan_.write_once; }
  int64_t KeyHeads() const { return Hg_; }  // write-once only

  int64_t RecurrentSlotStride() const { return H_ * V_ * K_; }
  int64_t RecurrentHeadStride() const { return V_ * K_; }
  float* RecurrentBase() { return recurrent_.data(); }
  float* RecurrentSlotPtr(int32_t slot) { return recurrent_.data() + static_cast<int64_t>(slot) * RecurrentSlotStride(); }

  int64_t ConvSeqStride() const { return conv_dim_ * state_len_max_; }  // cs_seq
  int64_t ConvDimStride() const { return state_len_max_; }              // cs_dim
  int64_t ConvTokStride() const { return 1; }                           // cs_tok
  int64_t StateLenMax() const { return state_len_max_; }
  uint16_t* ConvBase() { return conv_.data(); }
  // The two state buffers whole (every physical slot), for a test that digests them (Model::
  // DebugStateDigest, R4DX_TP_TESTING).
  size_t RecurrentElems() const { return recurrent_.size(); }
  size_t ConvElems() const { return conv_.size(); }

  void ZeroAll(core::Stream& stream) {
    recurrent_.ZeroAsync(stream);
    conv_.ZeroAsync(stream);
  }

  // After a speculative verify call that committed n > 1 of its candidates, sequence `seq_id`'s live
  // state is in window slot n-1 and its conv history at offset n-1 of the rolling buffer (the next
  // decode/verify call is pointed there through num_accepted). A chunked-scan PREFILL reads neither:
  // it seeds from window 0 and from the history at offset 0 (gdn_layer.cpp's is_prefill branch,
  // r4d_gdn_conv_prep's cache read). Moves the live state there -- where a plain decode step leaves
  // it. Enqueued on `stream`. n == 1 is already in place.
  void CollapseWindow(int32_t seq_id, int64_t n, core::Stream& stream) {
    if (n <= 1) return;
    if (n > max_decode_window_) throw std::logic_error("GdnStateManager::CollapseWindow: n exceeds the window");
    if (plan_.write_once) throw std::logic_error("GdnStateManager::CollapseWindow: write-once state -- Materialize + ShiftConv");
    R4DX_HIP_CHECK(hipMemcpyAsync(RecurrentSlotPtr(SlotForSeq(seq_id)),
                                  RecurrentSlotPtr(WindowSlot(seq_id, static_cast<int32_t>(n - 1))),
                                  static_cast<size_t>(RecurrentSlotStride()) * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream.get()));
    ShiftConv(seq_id, n, stream);
  }
  // The conv half of CollapseWindow (unchanged by the write-once state): the history a step reads moves from
  // offset n-1 to offset 0. n <= 1 is already in place.
  void ShiftConv(int32_t seq_id, int64_t n, core::Stream& stream) {
    if (n <= 1) return;
    if (n > max_decode_window_) throw std::logic_error("GdnStateManager::ShiftConv: n exceeds the window");
    // The conv line is [conv_dim][state_len_max] with each channel's history contiguous; the history a
    // step reads is conv_width-1 entries. One strided column per entry, in ascending order: column j
    // reads entry n-1+j, which only a LATER column (j' = n-1+j > j) overwrites.
    uint16_t* line = ConvSeqPtr(seq_id);
    const size_t pitch = static_cast<size_t>(state_len_max_) * sizeof(uint16_t);
    for (int64_t j = 0; j < conv_width_ - 1; ++j) {
      R4DX_HIP_CHECK(hipMemcpy2DAsync(line + j, pitch, line + (n - 1) + j, pitch, sizeof(uint16_t),
                                      static_cast<size_t>(conv_dim_), hipMemcpyDeviceToDevice, stream.get()));
    }
  }

  // ---- write-once state (docs/gdn-write-once.md) ---------------------------------------------------------
  // The log buffers (libr4d's r4d_gdn_wo_log_bytes layout: u, kk, eg). LogIn is the log of the last launched
  // verify call with T > 1, which a pending commit refers to; LogOut the one the next such call writes;
  // FlipParity() once per such launch (GdnLayer::Forward).
  const float* LogIn() { return log_[static_cast<size_t>(ring_.In())].data(); }
  float* LogOut() { return log_[static_cast<size_t>(ring_.Out())].data(); }
  void FlipParity() { ring_.Flip(); }
  size_t LogBytes() const { return log_[0].bytes() + log_[1].bytes(); }
  // Applies the first n rows of LogIn() to sequence `seq_id`'s state, in place (n = the committed count of
  // the last verify call, >= 1). Enqueued on `stream`; nothing else is touched. The logical state does not
  // change: replaying here and continuing with no pending performs the same fp32 operations the next call's
  // seed load would have, so the caller may flush at any point.
  void Materialize(int32_t seq_id, int64_t n, core::Stream& stream) {
    if (!plan_.write_once) throw std::logic_error("GdnStateManager::Materialize: not a write-once manager");
    if (n < 1 || n > max_decode_window_) throw std::logic_error("GdnStateManager::Materialize: n outside [1, window]");
    float* b = RecurrentSlotPtr(SlotForSeq(seq_id));
    core::r4d::GdnStateReplay(b, b, RecurrentHeadStride(), LogIn(), /*pending=*/nullptr, static_cast<int>(n),
                              static_cast<int>(max_decode_window_), static_cast<int>(H_),
                              static_cast<int>(Hg_), static_cast<int>(K_), static_cast<int>(V_), stream.get());
  }
  // The same replay into a scratch slot (a test comparing it with a window slot of the legacy path).
  void MaterializeTo(int32_t seq_id, int64_t n, float* dst, core::Stream& stream) {
    if (!plan_.write_once) throw std::logic_error("GdnStateManager::MaterializeTo: not a write-once manager");
    if (n < 0 || n > max_decode_window_) throw std::logic_error("GdnStateManager::MaterializeTo: n outside [0, window]");
    core::r4d::GdnStateReplay(RecurrentSlotPtr(SlotForSeq(seq_id)), dst, RecurrentHeadStride(), LogIn(),
                              /*pending=*/nullptr, static_cast<int>(n), static_cast<int>(max_decode_window_),
                              static_cast<int>(H_), static_cast<int>(Hg_), static_cast<int>(K_),
                              static_cast<int>(V_), stream.get());
  }

  // ---- prompt checkpoint (Model::SaveCheckpoint, docs/server.md "Prefix cache") ------------------
  // One spare copy of a sequence's window-0 recurrent slot and its conv state: the per-sequence state
  // that moving the position back cannot rewind (the next call READS it, where the KV caches are
  // overwritten before they are read). Allocated only when a Model is loaded to checkpoint.
  void AllocateCheckpoint() {
    ckpt_recurrent_ = core::DeviceBuffer<float>(static_cast<size_t>(RecurrentSlotStride()));
    ckpt_conv_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(ConvSeqStride()));
  }
  size_t CheckpointBytes() const { return ckpt_recurrent_.bytes() + ckpt_conv_.bytes(); }
  // Both enqueue on `stream` and do not synchronize. The copy is of window 0 and of the conv cache
  // line the layer addresses with cache_idx == SlotForSeq(seq_id) (gdn_layer.cpp): the live state
  // only while no speculative round has moved it to another window slot / history offset since the
  // last prefill, which is Model::SaveCheckpoint's precondition.
  void SaveCheckpoint(int32_t seq_id, core::Stream& stream) {
    CopyAsync(ckpt_recurrent_.data(), RecurrentSlotPtr(SlotForSeq(seq_id)), ckpt_recurrent_.bytes(), stream);
    CopyAsync(ckpt_conv_.data(), ConvSeqPtr(seq_id), ckpt_conv_.bytes(), stream);
  }
  void RestoreCheckpoint(int32_t seq_id, core::Stream& stream) {
    CopyAsync(RecurrentSlotPtr(SlotForSeq(seq_id)), ckpt_recurrent_.data(), ckpt_recurrent_.bytes(), stream);
    CopyAsync(ConvSeqPtr(seq_id), ckpt_conv_.data(), ckpt_conv_.bytes(), stream);
  }

 private:
  uint16_t* ConvSeqPtr(int32_t seq_id) { return conv_.data() + static_cast<int64_t>(SlotForSeq(seq_id)) * ConvSeqStride(); }
  static void CopyAsync(void* dst, const void* src, size_t bytes, core::Stream& stream) {
    if (bytes == 0) throw std::logic_error("GdnStateManager: checkpoint storage was never allocated");
    R4DX_HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream.get()));
  }

  int64_t H_, V_, K_, conv_dim_, conv_width_, state_len_max_, max_decode_window_;
  GdnSlotPlan plan_;
  int64_t Hg_ = 0;                      // write-once only
  core::DeviceBuffer<float> recurrent_;
  core::DeviceBuffer<uint16_t> conv_;
  GdnLogRing ring_;                     // write-once only
  core::DeviceBuffer<float> log_[2];    // write-once only: the ping-pong logs
  core::DeviceBuffer<float> ckpt_recurrent_;  // [H*V*K], empty unless AllocateCheckpoint()
  core::DeviceBuffer<uint16_t> ckpt_conv_;    // [conv_dim * state_len_max]
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
  // SidxBase(slot, window) and HasInitTrue() -- plus, on a wide Model, CuPair(256) for a super-chunk's
  // one-call GDN sequence ops (TpWarmup adds it). After Freeze() a miss throws std::logic_error instead
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
