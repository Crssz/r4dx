// r4dx::model::pp::StageBuffers -- the pinned-host buffers of the two-stage prefill, sized from one spec and allocated BY THE THREAD THAT
// OWNS THE PRODUCING DEVICE (docs/pp-tp2-hybrid.md 7 "Who owns what", critique item 3). Header-only and HIP-free (the allocator is injected;
// pp_stage_runner.h's PinnedAllocator() is the real one), so the sizing rules and the ownership protocol are CPU-tested
// (tests/model/test_pp_runner_cpu.cpp).
//
// Who produces what:
//   stage-A-produced   the channel slots (A's D2H of the carry), the GDN hand-off buffer (A's live GDN state at the end of a call);
//   stage-B-produced   the GDN sync buffer and the KV sync buffer (B's state for A's next call: the sync-back).
// A buffer is written by the device of its producer and read by the other stage's device. The owner allocates each group from a
// thread bound to the producer's device -- AllocateAProduced() from stage A's thread, AllocateBProduced() from stage B's -- and nothing
// is allocated lazily afterwards: the KV sync buffer is sized at its maximum up front (KvSyncCapacity). PpModel calls the first from
// stage A's worker and the second from the facade (which is stage B's thread); the hybrid will call both from the two rank workers.
// The class itself assumes nothing about which thread that is. Allocation is not thread-safe: the owner publishes the result (the
// worker join) before another thread reads a pointer.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pp_channel.h"

namespace r4dx::model::pp {

// A sync-back larger than this (KV rows stage A is missing x the attention layers' block bytes) is not worth a pinned buffer of its
// own: the call runs on the decode Model alone. 512 MiB = 32k rows of 8 attention layers.
constexpr size_t kMaxKvSyncBytes = 512ull * 1024 * 1024;
constexpr size_t kMinKvSyncBytes = 16ull << 20;

// The KV sync buffer's size: the wire bytes of the whole stage-KV capacity (rows [0, S) of the stage's attention layers -- the most a
// sync-back can ever ask for), clamped to [kMinKvSyncBytes, kMaxKvSyncBytes]. A larger ask is refused by KvSyncFits (the call runs on B alone).
inline size_t KvSyncCapacity(size_t wire_bytes_full_ctx) {
  return std::min(kMaxKvSyncBytes, std::max(kMinKvSyncBytes, wire_bytes_full_ctx));
}
inline bool KvSyncFits(size_t need, size_t capacity) { return need <= capacity; }

// One channel slot's capacity. `carry_kv` false is the hybrid's slot (docs/pp-tp2-hybrid.md 1): the carry and the DFlash columns only --
// stage Y has no layers below the split, so stage X's KV blocks stay on X until the reshard and never enter a slot.
inline size_t StageSlotBytes(bool carry_kv, int64_t chunk_rows, int64_t hidden, int64_t dfl_cols, int64_t attn_layers, int64_t block_size,
                             size_t block_stride_bytes) {
  return MaxSlotBytes(chunk_rows, hidden, dfl_cols, carry_kv ? attn_layers : 0, block_size, block_stride_bytes);
}

struct StageBufferSpec {
  int slots = 3;
  size_t slot_bytes = 0;     // per slot
  size_t gdn_bytes = 0;      // the GDN hand-off / sync buffers (each)
  size_t kv_sync_bytes = 0;  // the KV sync buffer
};

// The geometry the spec derives from (all of it known without touching a device).
struct StageBufferGeometry {
  int slots = 3;
  int64_t chunk_rows = 0;
  int64_t hidden = 0;
  int64_t dfl_cols_max = 0;     // DFlash columns stage A can ship (the target layers below the largest split, at least the test-capture room)
  int64_t attn_layers_max = 0;  // attention layers below the largest split
  int64_t block_size = 0;
  size_t block_stride_bytes = 0;
  size_t gdn_bytes = 0;         // Model::PpGdnWireBytes(largest split)
  size_t kv_wire_full_ctx = 0;  // Model::PpKvWireBytes(largest split, 0, stage-KV capacity)
};
inline StageBufferSpec MakeStageBufferSpec(const StageBufferGeometry& g, bool carry_kv) {
  if (g.slots < 1) throw std::invalid_argument("pp::MakeStageBufferSpec: at least one slot");
  StageBufferSpec s;
  s.slots = g.slots;
  s.slot_bytes = StageSlotBytes(carry_kv, g.chunk_rows, g.hidden, g.dfl_cols_max, g.attn_layers_max, g.block_size, g.block_stride_bytes);
  s.gdn_bytes = g.gdn_bytes;
  s.kv_sync_bytes = KvSyncCapacity(g.kv_wire_full_ctx);
  return s;
}

// How the memory is obtained (pinned host memory in production; a fake in the CPU test). `alloc` returns nullptr never (it throws);
// it is not called for a zero-byte block.
struct HostAllocator {
  std::function<uint8_t*(size_t)> alloc;
  std::function<void(uint8_t*)> release;
};

class StageBuffers {
 public:
  StageBuffers(const StageBufferSpec& spec, HostAllocator allocator) : spec_(spec), mem_(std::move(allocator)) {
    if (spec_.slots < 1) throw std::invalid_argument("StageBuffers: at least one slot");
    if (!mem_.alloc || !mem_.release) throw std::invalid_argument("StageBuffers: an allocator is required");
  }
  ~StageBuffers() {
    ReleaseAProduced();
    ReleaseBProduced();
  }
  StageBuffers(const StageBuffers&) = delete;
  StageBuffers& operator=(const StageBuffers&) = delete;

  const StageBufferSpec& Spec() const { return spec_; }

  // ---- allocation: from the thread that owns the producing device. Throws std::logic_error when the group is already allocated;
  // a failed allocation frees what it had taken (the group stays unallocated).
  void AllocateAProduced() {
    if (a_allocated_) throw std::logic_error("StageBuffers::AllocateAProduced: already allocated");
    std::vector<uint8_t*> slots;
    uint8_t* hand = nullptr;
    try {
      for (int i = 0; i < spec_.slots; ++i) slots.push_back(Take(spec_.slot_bytes));
      hand = Take(spec_.gdn_bytes);
    } catch (...) {
      for (uint8_t* p : slots) Give(p);
      throw;
    }
    slots_ = std::move(slots);
    gdn_hand_ = hand;
    a_allocated_ = true;
    a_thread_ = std::this_thread::get_id();
  }
  void AllocateBProduced() {
    if (b_allocated_) throw std::logic_error("StageBuffers::AllocateBProduced: already allocated");
    uint8_t* sync = nullptr;
    uint8_t* kv = nullptr;
    try {
      sync = Take(spec_.gdn_bytes);
      kv = Take(spec_.kv_sync_bytes);
    } catch (...) {
      Give(sync);
      throw;
    }
    gdn_sync_ = sync;
    kv_sync_ = kv;
    b_allocated_ = true;
    b_thread_ = std::this_thread::get_id();
  }
  // Frees a group (idempotent). The owner calls it from the same thread that allocated when it can (tearing a rank worker down); any
  // thread may, once nothing reads the memory any more.
  void ReleaseAProduced() noexcept {
    for (uint8_t* p : slots_) Give(p);
    slots_.clear();
    Give(gdn_hand_);
    gdn_hand_ = nullptr;
    a_allocated_ = false;
  }
  void ReleaseBProduced() noexcept {
    Give(gdn_sync_);
    Give(kv_sync_);
    gdn_sync_ = kv_sync_ = nullptr;
    b_allocated_ = false;
  }

  bool AProducedAllocated() const { return a_allocated_; }
  bool BProducedAllocated() const { return b_allocated_; }
  // The threads that ran the allocations (diagnostics, tests); a default id before the allocation.
  std::thread::id AProducedThread() const { return a_thread_; }
  std::thread::id BProducedThread() const { return b_thread_; }

  // ---- the memory (valid between Allocate and Release)
  // The pointers for pp::StageChannel's constructor (stage-A-produced, so allocated first).
  const std::vector<uint8_t*>& SlotPointers() const {
    if (!a_allocated_) throw std::logic_error("StageBuffers::SlotPointers: the stage-A-produced buffers are not allocated");
    return slots_;
  }
  uint8_t* GdnHand() const { return gdn_hand_; }   // A -> B: the GDN live state at the end of a call
  uint8_t* GdnSync() const { return gdn_sync_; }   // B -> A: the GDN live state before a call
  uint8_t* KvSync() const { return kv_sync_; }     // B -> A: the KV rows A is missing before a call
  size_t KvSyncBytes() const { return b_allocated_ ? spec_.kv_sync_bytes : 0; }

 private:
  uint8_t* Take(size_t bytes) { return bytes == 0 ? nullptr : mem_.alloc(bytes); }
  void Give(uint8_t* p) noexcept {
    if (p == nullptr) return;
    try {
      mem_.release(p);
    } catch (...) {
    }
  }

  StageBufferSpec spec_;
  HostAllocator mem_;
  std::vector<uint8_t*> slots_;
  uint8_t* gdn_hand_ = nullptr;
  uint8_t* gdn_sync_ = nullptr;
  uint8_t* kv_sync_ = nullptr;
  bool a_allocated_ = false, b_allocated_ = false;
  std::thread::id a_thread_, b_thread_;
};

}  // namespace r4dx::model::pp
