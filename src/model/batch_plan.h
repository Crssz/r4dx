// r4dx::model::batch -- the host-side bookkeeping of batched decode (docs/batch-decode.md): which of a
// Model's batch slots are live and how far each one has got, and the per-step metadata of one
// Model::DecodeBatch call (token ids, rope positions, KV slot mapping, attention seqused_k, the GDN
// kernels' cu / cache_idx / sidx arrays) laid out so ONE async H2D copy stages all of it, the way
// step_meta.h does for the single-sequence decode step.
//
// Header-only and HIP-free, so tests/model/test_batch_plan_cpu.cpp can pin every offset and every rule on
// the host.
//
// A batch slot owns, in every full-attention layer, `slot_ctx` tokens of one shared fp8 PagedKvCache
// (slot s holds cache positions [s * slot_ctx, (s + 1) * slot_ctx), i.e. blocks [s * blocks_per_slot, ...)
// of the cache's identity block table) and, in every GDN layer, physical slot s + 1 of a separate
// GdnStateManager (slot 0 of a manager is reserved, gdn_state.h). A step feeds ONE token per row; the
// rows of a step are distinct slots, in any order, each already holding the sequence's prompt.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx::model::batch {

// Upper bound of ModelOptions::batch_slots. The metadata sections below are 64 ints wide (3 axes of mrope
// rows need 3 * slots <= 64), the skinny GEMMs take at most 64 rows, and a row beyond ~16 stops paying: the
// weight stream is amortised long before that, and the per-row attention launches (docs/batch-decode.md 3.2)
// grow linearly.
inline constexpr int kMaxSlots = 16;

// int32 element offsets into one step's metadata array (and its device mirror, Model::batch_meta_dev_).
// Every section starts on a 256-byte boundary of the (hipMalloc'd, so 256-byte aligned) device array.
inline constexpr int64_t kMetaIds = 0;       // [rows]      token id fed by row r
inline constexpr int64_t kMetaRopePos = 64;  // [rows]      text rope position of row r (== the sequence index)
inline constexpr int64_t kMetaSlotMap = 128; // [rows]      KV slot mapping: slot * slot_ctx + position
inline constexpr int64_t kMetaSeqUsed = 192; // [rows]      attention seqused_k: position + 1
inline constexpr int64_t kMetaCacheIdx = 256;// [rows]      GDN conv-state line / recurrent slot of row r
inline constexpr int64_t kMetaCu = 320;      // [rows + 1]  GDN cu_seqlens: 0, 1, ..., rows
inline constexpr int64_t kMetaSidx = 384;    // [rows]      GDN recurrent_update's state-write slot (window 1: == cache_idx)
inline constexpr int64_t kMetaRope3 = 448;   // [3, rows]   mrope rows (t, then h, then w); written only when a row is mrope
inline constexpr int64_t kMetaInts = 512;
inline constexpr int64_t kMetaRows = 64;     // section width; the real bound is kMaxSlots

// A batch slot's GDN physical slot. GdnStateManager reserves physical slot 0 and puts sequence i at
// physical slot i * window + 1; a batch Model has no speculative window (window 1), so slot s -> s + 1.
// Model::Load checks this against GdnStateManager::SlotForSeq.
inline constexpr int32_t GdnSlotOf(int slot) { return slot + 1; }

struct SlotState {
  bool active = false;
  int64_t pos = 0;           // tokens committed to the slot (the next token is fed at position `pos`)
  bool mrope = false;        // an image sits in this sequence: rope positions are pos + mrope_delta
  int64_t mrope_delta = 0;
};

class SlotTable {
 public:
  SlotTable() = default;
  SlotTable(int slots, int64_t slot_ctx) : slots_(static_cast<size_t>(slots)), slot_ctx_(slot_ctx) {
    if (slots < 1 || slots > kMaxSlots) {
      throw std::invalid_argument("batch::SlotTable: slots must be in [1, " + std::to_string(kMaxSlots) + "]");
    }
    if (slot_ctx < 1) throw std::invalid_argument("batch::SlotTable: slot_ctx must be positive");
  }

  int Slots() const { return static_cast<int>(slots_.size()); }
  int64_t SlotCtx() const { return slot_ctx_; }

  // Marks `slot` live at position `pos` (the prompt length the import copied in). `pos` must leave room for
  // at least one more token (pos < slot_ctx): a slot that is already full has nothing to decode.
  void Activate(int slot, int64_t pos, bool mrope = false, int64_t mrope_delta = 0) {
    Check(slot);
    if (pos < 1) throw std::invalid_argument("batch::SlotTable::Activate: a slot needs a prompt (pos >= 1)");
    if (pos >= slot_ctx_) {
      throw std::invalid_argument("batch::SlotTable::Activate: position " + std::to_string(pos) +
                                  " leaves no room in a " + std::to_string(slot_ctx_) + "-token slot");
    }
    SlotState& s = slots_[static_cast<size_t>(slot)];
    s.active = true;
    s.pos = pos;
    s.mrope = mrope;
    s.mrope_delta = mrope_delta;
  }
  void Release(int slot) {
    Check(slot);
    slots_[static_cast<size_t>(slot)] = SlotState{};
  }
  void Clear() {
    for (SlotState& s : slots_) s = SlotState{};
  }
  // After a step fed one token into `slot`.
  void Advance(int slot) {
    Check(slot);
    SlotState& s = slots_[static_cast<size_t>(slot)];
    if (!s.active) throw std::logic_error("batch::SlotTable::Advance: slot is not active");
    ++s.pos;
  }

  const SlotState& State(int slot) const {
    Check(slot);
    return slots_[static_cast<size_t>(slot)];
  }
  bool Active(int slot) const { return State(slot).active; }
  int64_t Pos(int slot) const { return State(slot).pos; }
  int ActiveCount() const {
    int n = 0;
    for (const SlotState& s : slots_) n += s.active ? 1 : 0;
    return n;
  }

 private:
  void Check(int slot) const {
    if (slot < 0 || slot >= static_cast<int>(slots_.size())) {
      throw std::invalid_argument("batch::SlotTable: slot " + std::to_string(slot) + " outside [0, " +
                                  std::to_string(slots_.size()) + ")");
    }
  }
  std::vector<SlotState> slots_;
  int64_t slot_ctx_ = 0;
};

// "" when a step over `slots` is legal, else the reason: 1..Slots() rows, every slot in range, active and
// distinct, and room for the token this step feeds (pos < slot_ctx -- Activate keeps pos below it, Advance
// can reach it).
inline std::string CheckStep(const SlotTable& t, const int* slots, int rows) {
  if (rows < 1 || rows > t.Slots()) {
    return "a batch step needs 1.." + std::to_string(t.Slots()) + " rows, got " + std::to_string(rows);
  }
  uint32_t seen = 0;
  for (int r = 0; r < rows; ++r) {
    const int s = slots[r];
    if (s < 0 || s >= t.Slots()) return "slot " + std::to_string(s) + " is outside [0, " + std::to_string(t.Slots()) + ")";
    if (seen & (1u << s)) return "slot " + std::to_string(s) + " appears twice in one step";
    seen |= 1u << s;
    if (!t.Active(s)) return "slot " + std::to_string(s) + " holds no sequence (BatchImport it first)";
    if (t.Pos(s) >= t.SlotCtx()) {
      return "slot " + std::to_string(s) + " is full (" + std::to_string(t.Pos(s)) + " of " +
             std::to_string(t.SlotCtx()) + " tokens)";
    }
  }
  return "";
}

// Fills `h` (kMetaInts elements; the unused rows and padding are left as they are) for one step: row r feeds
// tokens[r] into slots[r] at that slot's position. Returns true when any row is mrope-active, i.e. when the
// rope3 section is valid and the attention layers must take the 3-axis rope entry point (every row of the
// step then gets a rope3 column, text-only rows included: pos on all three axes).
inline bool FillStepMeta(const SlotTable& t, const int* slots, const int32_t* tokens, int rows, int32_t* h) {
  const std::string why = CheckStep(t, slots, rows);
  if (!why.empty()) throw std::invalid_argument("batch::FillStepMeta: " + why);
  bool any_mrope = false;
  for (int r = 0; r < rows; ++r) any_mrope = any_mrope || t.State(slots[r]).mrope;
  for (int r = 0; r < rows; ++r) {
    const int s = slots[r];
    const SlotState& st = t.State(s);
    h[kMetaIds + r] = tokens[r];
    h[kMetaRopePos + r] = static_cast<int32_t>(st.pos);
    h[kMetaSlotMap + r] = static_cast<int32_t>(static_cast<int64_t>(s) * t.SlotCtx() + st.pos);
    h[kMetaSeqUsed + r] = static_cast<int32_t>(st.pos + 1);
    h[kMetaCacheIdx + r] = GdnSlotOf(s);
    h[kMetaCu + r] = r;
    h[kMetaSidx + r] = GdnSlotOf(s);
    if (any_mrope) {
      const int32_t p = static_cast<int32_t>(st.pos + (st.mrope ? st.mrope_delta : 0));
      h[kMetaRope3 + 0 * rows + r] = p;
      h[kMetaRope3 + 1 * rows + r] = p;
      h[kMetaRope3 + 2 * rows + r] = p;
    }
  }
  h[kMetaCu + rows] = rows;
  return any_mrope;
}

// FNV-1a over what a tensor-parallel group must agree on before a batch step's collectives are enqueued
// (Model::DecodeBatch's CheckLockstep): the row count, then each row's slot, token and position.
inline uint64_t StepFingerprint(const SlotTable& t, const int* slots, const int32_t* tokens, int rows) {
  uint64_t x = 0xcbf29ce484222325ull;
  const auto mix = [&x](uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      x ^= (v >> (8 * i)) & 0xffu;
      x *= 0x100000001b3ull;
    }
  };
  mix(static_cast<uint64_t>(rows));
  for (int r = 0; r < rows; ++r) {
    mix(static_cast<uint64_t>(slots[r]));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(tokens[r])));
    mix(static_cast<uint64_t>(t.Pos(slots[r])));
  }
  return x;
}

// ---- memory (docs/batch-decode.md 4) -----------------------------------------------------------------
// fp8 KV bytes of one token across `attn_layers` layers on a rank holding `kv_heads` heads: K and V, head_dim
// bytes each (the cache's (num_blocks, kv_heads, block, 2 * head_dim) layout).
inline int64_t KvBytesPerToken(int64_t attn_layers, int64_t kv_heads, int64_t head_dim) {
  return attn_layers * kv_heads * 2 * head_dim;
}
// What `slots` batch slots of `slot_ctx` tokens cost in KV.
inline int64_t BatchKvBytes(int slots, int64_t slot_ctx, int64_t attn_layers, int64_t kv_heads, int64_t head_dim) {
  return static_cast<int64_t>(slots) * slot_ctx * KvBytesPerToken(attn_layers, kv_heads, head_dim);
}
// What `slots` batch slots cost in GDN state: per layer, an fp32 recurrent state (H * V * K) plus a conv line
// (conv_dim * state_len_max bf16) for each of slots + 1 physical slots (slot 0 is reserved and allocated).
inline int64_t BatchGdnBytes(int slots, int64_t gdn_layers, int64_t H, int64_t V, int64_t K, int64_t conv_dim,
                             int64_t state_len_max) {
  const int64_t per_slot = H * V * K * 4 + conv_dim * state_len_max * 2;
  return gdn_layers * (static_cast<int64_t>(slots) + 1) * per_slot;
}

}  // namespace r4dx::model::batch
