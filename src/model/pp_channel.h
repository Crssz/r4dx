// r4dx::model::pp::StageChannel -- the hand-over between the two stages of the pipeline-parallel prefill
// (docs/pp-prefill.md section 5): a ring of pinned-host slots, one producer thread (stage A, device 0) and one
// consumer thread (stage B, device 1), FIFO, with back-pressure, bounded waits and a poison flag. No P2P copy and no
// device-side wait anywhere: a slot is plain host memory that stage A fills with hipMemcpyAsync D2H (and waits for)
// before it publishes, and stage B drains with hipMemcpyAsync H2D (and finishes with) before it releases it.
//
// This header knows nothing about HIP or Model -- the slot memory is handed in by the owner (PpModel allocates it
// pinned) -- so the whole protocol, including a million-item two-thread stress run, is CPU-tested
// (tests/model/test_pp_channel_cpu.cpp). It also holds the HIP-free arithmetic of one slot's payload (SlotLayout), which
// stage A's export and stage B's import both derive from the same (position, rows) so they cannot disagree.
//
// Protocol (all under one mutex; the payload is plain memory published by that mutex hand-off):
//   producer:  slot = AcquireFree(); ...fill slot->data, set slot->hdr...; Publish(slot)
//   consumer:  slot = AcquireFull(); ...drain slot->data...;               Release(slot)
//   once per call, after the producer's last chunk: PublishBulk(call_id); the consumer's WaitBulk(call_id) /
//   BulkReady(call_id) (the GDN state is a separate pinned buffer the owner holds; the channel only carries the flag)
// FIFO order is enforced by counters (the n-th AcquireFree returns slot n % slots, the n-th AcquireFull too). Any wait
// that finds the channel poisoned throws ChannelPoisoned; one that runs out of time throws ChannelTimeout (it does not
// poison: the caller decides). BeginCall() needs a quiescent channel (every slot released); Reset() clears a poisoned
// one once both sides are known to be out of it.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::model::pp {

class ChannelError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class ChannelPoisoned : public ChannelError {
 public:
  using ChannelError::ChannelError;
};
class ChannelTimeout : public ChannelError {
 public:
  using ChannelError::ChannelError;
};

// ---- one slot's payload ------------------------------------------------------------------------------------------
// What stage A hands stage B for one chunk (docs/pp-prefill.md 1.3, 3.1, 4): the residual stream `cur` and the
// fused-norm pair (buf_normed_, buf_normed_pre_) -- three buffers of rows x hidden bf16-sized elements --, the DFlash
// feature columns of the target layers stage A owns (rows x dfl_cols x hidden bf16, row-major, packed), and the KV
// blocks of every attention layer in [0, split) that the chunk wrote (whole blocks, one contiguous byte range per
// layer). Offsets are 256-byte aligned.
struct KvPiece {
  int64_t layer = 0;
  int64_t block0 = 0, blocks = 0;
  size_t dev_off = 0;   // byte offset into the layer's KV cache (block0 * block stride)
  size_t bytes = 0;
  size_t host_off = 0;  // byte offset into the slot
};
struct SlotLayout {
  size_t carry_bytes = 0;
  size_t off_cur = 0, off_norm = 0, off_pre = 0, off_dfl = 0;
  size_t dfl_bytes = 0;  // rows x dfl_cols x hidden x 2
  std::vector<KvPiece> kv;
  size_t kv_bytes = 0;
  size_t total = 0;
};
constexpr size_t kSlotAlign = 256;
inline size_t AlignUp(size_t v, size_t a = kSlotAlign) { return (v + a - 1) & ~(a - 1); }

// The KV blocks [pos, pos + rows) touch in a layer: whole blocks of `block_size` rows (the same arithmetic as
// pp_plan.h's KvBlocksTouched; restated here so this header stands alone).
inline void KvBlockSpan(int64_t pos, int64_t rows, int64_t block_size, int64_t* block0, int64_t* blocks) {
  const int64_t b0 = pos / block_size;
  const int64_t b1 = (pos + rows + block_size - 1) / block_size;
  *block0 = b0;
  *blocks = b1 - b0;
}

inline SlotLayout MakeSlotLayout(int64_t pos, int64_t rows, int64_t hidden, int64_t dfl_cols,
                                 const std::vector<int64_t>& attn_layers, int64_t block_size,
                                 size_t block_stride_bytes) {
  // A chunk always carries something: rows >= 1 and hidden >= 1 (hidden 0 once shipped an empty carry while both stages'
  // layouts, headers and payload sizes still agreed -- the 2026-10-08 identity failure, docs/pp-prefill.md 4.6).
  if (rows < 1 || hidden < 1 || dfl_cols < 0 || pos < 0) {
    throw std::invalid_argument("pp::MakeSlotLayout: rows " + std::to_string(rows) + ", hidden " + std::to_string(hidden) +
                                ", dfl_cols " + std::to_string(dfl_cols) + ", pos " + std::to_string(pos) +
                                " (rows and hidden must be >= 1, the others >= 0)");
  }
  SlotLayout l;
  l.carry_bytes = static_cast<size_t>(rows * hidden) * 2;
  size_t off = 0;
  const auto take = [&off](size_t bytes) {
    const size_t at = off;
    off += AlignUp(bytes);
    return at;
  };
  l.off_cur = take(l.carry_bytes);
  l.off_norm = take(l.carry_bytes);
  l.off_pre = take(l.carry_bytes);
  l.dfl_bytes = static_cast<size_t>(rows * dfl_cols * hidden) * 2;
  l.off_dfl = take(l.dfl_bytes);
  for (const int64_t layer : attn_layers) {
    KvPiece p;
    p.layer = layer;
    KvBlockSpan(pos, rows, block_size, &p.block0, &p.blocks);
    p.dev_off = static_cast<size_t>(p.block0) * block_stride_bytes;
    p.bytes = static_cast<size_t>(p.blocks) * block_stride_bytes;
    p.host_off = take(p.bytes);
    l.kv_bytes += p.bytes;
    l.kv.push_back(p);
  }
  l.total = off;
  return l;
}

// The largest payload a chunk of at most `max_rows` rows can have: the worst alignment of the chunk's start (a
// chunk that starts one row before a block boundary touches one block more).
inline size_t MaxSlotBytes(int64_t max_rows, int64_t hidden, int64_t dfl_cols, int64_t attn_layer_count,
                           int64_t block_size, size_t block_stride_bytes) {
  const int64_t blocks = (max_rows + block_size - 1) / block_size + 1;
  const size_t carry = AlignUp(static_cast<size_t>(max_rows * hidden) * 2);
  return 3 * carry + AlignUp(static_cast<size_t>(max_rows * dfl_cols * hidden) * 2) +
         static_cast<size_t>(attn_layer_count) * AlignUp(static_cast<size_t>(blocks) * block_stride_bytes);
}

// ---- the GDN live state, compact (docs/pp-prefill.md 3.2) --------------------------------------------------------
// Per GDN layer, in layer order: the live recurrent slot (H x V x K fp32) then the conv history as [conv_dim][hist]
// bf16, hist = conv_width - 1 (the two stages' conv lines have different row pitches: state_len_max = conv_width - 2 +
// max_decode_window, 3 on stage A and larger on a speculating stage B, so the host form is the compact one).
struct GdnWire {
  size_t recurrent_bytes = 0;
  size_t conv_bytes = 0;
  size_t PerLayer() const { return AlignUp(recurrent_bytes + conv_bytes); }
};
inline GdnWire MakeGdnWire(int64_t H, int64_t V, int64_t K, int64_t conv_dim, int64_t conv_width) {
  GdnWire w;
  w.recurrent_bytes = static_cast<size_t>(H * V * K) * sizeof(float);
  w.conv_bytes = static_cast<size_t>(conv_dim * (conv_width - 1)) * 2;
  return w;
}

// ---- the channel ---------------------------------------------------------------------------------------------------
struct SlotHeader {
  int64_t call_id = 0;
  int64_t seq = 0;       // chunk index within the call (set by Publish)
  int64_t pos = 0;       // absolute position of the chunk's first row
  int64_t rows = 0;
  int64_t dfl_cols = 0;  // DFlash columns carried
  bool last = false;     // the call's last chunk
  size_t bytes = 0;      // payload bytes used
};

class StageChannel {
 public:
  struct Slot {
    uint8_t* data = nullptr;
    size_t capacity = 0;
    int index = 0;
    SlotHeader hdr;
  };
  struct Stats {
    int64_t published = 0, taken = 0;
    double producer_wait_ms = 0;  // AcquireFree blocked (back-pressure: stage B is the slower stage)
    double consumer_wait_ms = 0;  // AcquireFull blocked (starvation: the fill, or stage A is slower)
  };

  // `memory[i]` is slot i's payload (capacity bytes each), owned by the caller and outliving the channel.
  StageChannel(const std::vector<uint8_t*>& memory, size_t capacity) {
    if (memory.empty()) throw std::invalid_argument("StageChannel: at least one slot");
    slots_.resize(memory.size());
    for (size_t i = 0; i < memory.size(); ++i) {
      slots_[i].data = memory[i];
      slots_[i].capacity = capacity;
      slots_[i].index = static_cast<int>(i);
    }
  }
  StageChannel(const StageChannel&) = delete;
  StageChannel& operator=(const StageChannel&) = delete;

  int Slots() const { return static_cast<int>(slots_.size()); }
  size_t Capacity() const { return slots_[0].capacity; }

  // ---- lifecycle (owner thread, quiescent channel) ----
  // Starts a call: every slot must have been released. Resets the FIFO counters and the bulk flag.
  void BeginCall(int64_t call_id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (poisoned_) throw ChannelPoisoned("StageChannel::BeginCall: poisoned (" + reason_ + ")");
    if (!Quiescent()) throw std::logic_error("StageChannel::BeginCall: the previous call left a slot in flight");
    call_id_ = call_id;
    acquired_ = published_ = taken_ = released_ = 0;
    bulk_call_ = -1;
  }
  // Clears the poison and every counter. Both sides must be out of the channel (the owner joined stage A).
  void Reset() {
    std::lock_guard<std::mutex> lk(mu_);
    poisoned_ = false;
    reason_.clear();
    acquired_ = published_ = taken_ = released_ = 0;
    bulk_call_ = -1;
    call_id_ = -1;
  }
  // Idempotent; the first reason sticks. Wakes every waiter.
  void Poison(const std::string& why) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!poisoned_) {
        poisoned_ = true;
        reason_ = why;
      }
    }
    cv_.notify_all();
  }
  bool Poisoned() const {
    std::lock_guard<std::mutex> lk(mu_);
    return poisoned_;
  }
  std::string PoisonReason() const {
    std::lock_guard<std::mutex> lk(mu_);
    return reason_;
  }
  Stats GetStats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = stats_;
    return s;
  }
  void ResetStats() {
    std::lock_guard<std::mutex> lk(mu_);
    stats_ = Stats{};
  }

  // ---- producer (stage A) ----
  // Waits for the next free slot in FIFO order (the ring is not full and the previous one was published).
  Slot* AcquireFree(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto ready = [&] { return poisoned_ || (acquired_ == published_ && acquired_ - released_ < Slots64()); };
    WaitLocked(lk, ready, timeout, "AcquireFree", &stats_.producer_wait_ms);
    Slot* s = &slots_[static_cast<size_t>(acquired_ % Slots64())];
    ++acquired_;
    s->hdr = SlotHeader{};
    s->hdr.call_id = call_id_;
    return s;
  }
  // `s->hdr` (pos, rows, dfl_cols, last, bytes) is filled in by the caller; seq is set here.
  void Publish(Slot* s) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (poisoned_) throw ChannelPoisoned("StageChannel::Publish: poisoned (" + reason_ + ")");
      if (acquired_ != published_ + 1 || s != &slots_[static_cast<size_t>(published_ % Slots64())]) {
        throw std::logic_error("StageChannel::Publish: not the slot AcquireFree returned last");
      }
      if (s->hdr.bytes > s->capacity) throw std::logic_error("StageChannel::Publish: payload exceeds the slot");
      s->hdr.seq = published_;
      ++published_;
      ++stats_.published;
    }
    cv_.notify_all();
  }
  // The GDN state of the call is complete in the owner's bulk buffer.
  void PublishBulk(int64_t call_id) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (poisoned_) throw ChannelPoisoned("StageChannel::PublishBulk: poisoned (" + reason_ + ")");
      bulk_call_ = call_id;
    }
    cv_.notify_all();
  }

  // ---- consumer (stage B) ----
  Slot* AcquireFull(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto ready = [&] { return poisoned_ || published_ > taken_; };
    WaitLocked(lk, ready, timeout, "AcquireFull", &stats_.consumer_wait_ms);
    Slot* s = &slots_[static_cast<size_t>(taken_ % Slots64())];
    ++taken_;
    ++stats_.taken;
    return s;
  }
  void Release(Slot* s) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (released_ >= taken_ || s != &slots_[static_cast<size_t>(released_ % Slots64())]) {
        throw std::logic_error("StageChannel::Release: not the oldest slot AcquireFull returned");
      }
      ++released_;
    }
    cv_.notify_all();
  }
  bool BulkReady(int64_t call_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (poisoned_) throw ChannelPoisoned("StageChannel::BulkReady: poisoned (" + reason_ + ")");
    return bulk_call_ == call_id;
  }
  void WaitBulk(int64_t call_id, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    double ignored = 0;
    WaitLocked(lk, [&] { return poisoned_ || bulk_call_ == call_id; }, timeout, "WaitBulk", &ignored);
  }

  // Chunks published but not yet released (diagnostics, tests).
  int64_t InFlight() const {
    std::lock_guard<std::mutex> lk(mu_);
    return published_ - released_;
  }

 private:
  int64_t Slots64() const { return static_cast<int64_t>(slots_.size()); }
  bool Quiescent() const { return acquired_ == published_ && published_ == taken_ && taken_ == released_; }

  template <class Pred>
  void WaitLocked(std::unique_lock<std::mutex>& lk, Pred ready, std::chrono::milliseconds timeout, const char* what,
                  double* waited_ms) {
    if (!ready()) {
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = cv_.wait_for(lk, timeout, ready);
      *waited_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (!ok) {
        throw ChannelTimeout(std::string("StageChannel::") + what + ": no progress for " +
                             std::to_string(timeout.count()) + " ms");
      }
    }
    if (poisoned_) throw ChannelPoisoned(std::string("StageChannel::") + what + ": poisoned (" + reason_ + ")");
  }

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Slot> slots_;
  int64_t acquired_ = 0, published_ = 0, taken_ = 0, released_ = 0;
  int64_t call_id_ = -1;
  int64_t bulk_call_ = -1;
  bool poisoned_ = false;
  std::string reason_;
  Stats stats_;
};

}  // namespace r4dx::model::pp
