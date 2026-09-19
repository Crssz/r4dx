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
  // to process in one call (a plain single-token decode needs 1; a speculative window needs
  // however many draft tokens it verifies at once) -- the conv state's rolling buffer depth is
  // conv_width-2 + max_decode_window (r4d_gdn_conv_w4_h128_bf16.hip's state_len_max: the kernel's
  // decode-side rewrite loop is only self-consistent when `slen_eff == slen + width - 2`, i.e.
  // `state_len_max == max_query_len + width - 2`, NOT width-1 -- verified against
  // r4d_gdn_conv_update_kernel's slen_eff/VAL derivation, which needs a depth of exactly
  // CP_ST==width-1 for a plain single-token decode (max_decode_window==1), not width).
  GdnStateManager(int64_t max_seqs, int64_t H, int64_t V, int64_t K, int64_t conv_dim,
                  int64_t conv_width, int64_t max_decode_window)
      : H_(H),
        V_(V),
        K_(K),
        conv_dim_(conv_dim),
        state_len_max_(conv_width - 2 + max_decode_window),
        recurrent_(static_cast<size_t>((max_seqs + 1) * H * V * K)),
        conv_(static_cast<size_t>((max_seqs + 1) * conv_dim * state_len_max_)) {}

  int32_t SlotForSeq(int32_t seq_id) const { return seq_id + 1; }  // seq_id is 0-based

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
  int64_t H_, V_, K_, conv_dim_, state_len_max_;
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
  const uint8_t* HasInitTrue() {
    if (has_init_true_.empty()) {
      has_init_true_ = core::DeviceBuffer<uint8_t>(1);
      const uint8_t v = 1;
      has_init_true_.CopyFromHost(&v, 1);
    }
    return has_init_true_.data();
  }

 private:
  template <typename MakeFn>
  const int32_t* Get(std::unordered_map<int64_t, core::DeviceBuffer<int32_t>>& map, int64_t key,
                      MakeFn make) {
    auto it = map.find(key);
    if (it != map.end()) return it->second.data();
    std::vector<int32_t> host = make();
    core::DeviceBuffer<int32_t> buf(host.size());
    buf.CopyFromHost(host);
    auto res = map.emplace(key, std::move(buf));
    return res.first->second.data();
  }

  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> cu_;
  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> cache_idx_;
  std::unordered_map<int64_t, core::DeviceBuffer<int32_t>> sidx_;
  core::DeviceBuffer<uint8_t> has_init_true_;
};

}  // namespace r4dx::model
