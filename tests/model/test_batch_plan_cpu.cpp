// tests/model/test_batch_plan_cpu.cpp -- CPU-only checks of src/model/batch_plan.h (docs/batch-decode.md): the slot table's
// rules, the step metadata a Model::DecodeBatch call stages (every offset, the KV slot mapping, the GDN cu / cache_idx /
// sidx arrays, the mrope columns), the step validation, the lockstep fingerprint, and the memory figures against the
// numbers the doc quotes. No HIP call; always runs.
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_plan.h"

using namespace r4dx::model::batch;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

template <class Fn>
bool Throws(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void Layout() {
  // Sections are 256-byte aligned (64 ints) and do not overlap; the rope3 section holds 3 * kMaxSlots ints.
  Check(kMetaIds == 0 && kMetaRopePos == 64 && kMetaSlotMap == 128 && kMetaSeqUsed == 192, "first four sections");
  Check(kMetaCacheIdx == 256 && kMetaCu == 320 && kMetaSidx == 384 && kMetaRope3 == 448, "last four sections");
  Check(kMetaInts == 512, "one 2 KiB copy");
  for (int64_t off : {kMetaIds, kMetaRopePos, kMetaSlotMap, kMetaSeqUsed, kMetaCacheIdx, kMetaCu, kMetaSidx, kMetaRope3}) {
    Check((off * 4) % 256 == 0, "section on a 256-byte boundary");
  }
  Check(kMetaRope3 + 3 * kMaxSlots <= kMetaInts, "rope3 fits");
  Check(kMetaCu + kMaxSlots + 1 <= kMetaSidx, "cu (rows + 1) fits before sidx");
  Check(GdnSlotOf(0) == 1 && GdnSlotOf(5) == 6, "GDN physical slot 0 is reserved: slot s -> s + 1");
}

void Table() {
  SlotTable t(4, 1000);
  Check(t.Slots() == 4 && t.SlotCtx() == 1000 && t.ActiveCount() == 0, "fresh table");
  t.Activate(2, 10);
  Check(t.Active(2) && t.Pos(2) == 10 && t.ActiveCount() == 1 && !t.Active(0), "activate");
  t.Advance(2);
  Check(t.Pos(2) == 11, "advance");
  t.Release(2);
  Check(!t.Active(2) && t.Pos(2) == 0 && t.ActiveCount() == 0, "release clears the slot");
  Check(Throws([&] { t.Advance(2); }), "advancing an inactive slot throws");
  Check(Throws([&] { t.Activate(4, 10); }), "slot out of range");
  Check(Throws([&] { t.Activate(-1, 10); }), "negative slot");
  Check(Throws([&] { t.Activate(0, 0); }), "a slot needs a prompt");
  Check(Throws([&] { t.Activate(0, 1000); }), "a full slot cannot be activated");
  t.Activate(0, 999);
  Check(t.Pos(0) == 999, "the last position with room for one token");
  t.Activate(1, 5);
  t.Clear();
  Check(t.ActiveCount() == 0, "clear");
  Check(Throws([] { SlotTable bad(0, 10); }) && Throws([] { SlotTable bad(kMaxSlots + 1, 10); }) &&
            Throws([] { SlotTable bad(2, 0); }),
        "constructor bounds");
}

void Validation() {
  SlotTable t(4, 100);
  t.Activate(0, 10);
  t.Activate(1, 20);
  t.Activate(3, 99);
  const int ok[] = {3, 0, 1};
  Check(CheckStep(t, ok, 3).empty(), "a legal step: any order, a subset of the slots");
  const int dup[] = {0, 1, 0};
  Check(!CheckStep(t, dup, 3).empty(), "a repeated slot");
  const int idle[] = {0, 2};
  Check(!CheckStep(t, idle, 2).empty(), "an inactive slot");
  const int range[] = {0, 4};
  Check(!CheckStep(t, range, 2).empty(), "a slot out of range");
  const int neg[] = {-1};
  Check(!CheckStep(t, neg, 1).empty(), "a negative slot");
  Check(!CheckStep(t, ok, 0).empty() && !CheckStep(t, ok, 5).empty(), "row count outside [1, slots]");
  t.Advance(3);  // pos 100 == slot_ctx: the slot is full
  const int full[] = {3};
  Check(!CheckStep(t, full, 1).empty(), "a full slot cannot decode");
  int32_t h[kMetaInts] = {};
  const int32_t tok[] = {1, 2, 3};
  Check(Throws([&] { FillStepMeta(t, dup, tok, 3, h); }), "FillStepMeta refuses an illegal step");
}

void Meta() {
  SlotTable t(4, 1000);
  t.Activate(3, 7);
  t.Activate(0, 100);
  t.Activate(2, 1);
  const int slots[] = {3, 0, 2};
  const int32_t tokens[] = {111, 222, 333};
  std::vector<int32_t> h(kMetaInts, -7);
  const bool mrope = FillStepMeta(t, slots, tokens, 3, h.data());
  Check(!mrope, "text-only step");
  // row 0: slot 3 at 7; row 1: slot 0 at 100; row 2: slot 2 at 1
  const int32_t pos[] = {7, 100, 1};
  for (int r = 0; r < 3; ++r) {
    Check(h[kMetaIds + r] == tokens[r], "token ids in row order");
    Check(h[kMetaRopePos + r] == pos[r], "rope position == sequence index");
    Check(h[kMetaSlotMap + r] == slots[r] * 1000 + pos[r], "slot mapping = slot * slot_ctx + position");
    Check(h[kMetaSeqUsed + r] == pos[r] + 1, "seqused_k = position + 1");
    Check(h[kMetaCacheIdx + r] == slots[r] + 1, "cache_idx = GDN physical slot");
    Check(h[kMetaSidx + r] == slots[r] + 1, "sidx = the same slot (window 1)");
    Check(h[kMetaCu + r] == r, "cu = 0, 1, 2");
  }
  Check(h[kMetaCu + 3] == 3, "cu ends at the row count");
  Check(h[kMetaIds + 3] == -7 && h[kMetaRope3] == -7, "unused rows and the rope3 section are left alone");
  // slot 0 is the first slot of the cache: its slot mapping is the plain position (the single-sequence layout)
  Check(h[kMetaSlotMap + 1] == 100, "slot 0's mapping equals the sequence index");
}

void Mrope() {
  SlotTable t(3, 500);
  t.Activate(0, 10, /*mrope=*/true, /*delta=*/-4);
  t.Activate(1, 20);  // a text-only sequence batched with an image one
  const int slots[] = {1, 0};
  const int32_t tokens[] = {5, 6};
  std::vector<int32_t> h(kMetaInts, 0);
  const bool mrope = FillStepMeta(t, slots, tokens, 2, h.data());
  Check(mrope, "one mrope row switches the step to the 3-axis rope");
  // rope3 is [3, rows]: axis-major. Row 0 is slot 1 (text: pos), row 1 is slot 0 (pos + delta = 6).
  for (int axis = 0; axis < 3; ++axis) {
    Check(h[kMetaRope3 + axis * 2 + 0] == 20, "a text row ropes at its position on all three axes");
    Check(h[kMetaRope3 + axis * 2 + 1] == 6, "an image row ropes at pos + delta on all three axes");
  }
  // the KV slot mapping stays the plain sequence index whatever the rope does
  Check(h[kMetaSlotMap + 1] == 0 * 500 + 10 && h[kMetaRopePos + 1] == 10, "slot mapping and the text rope position keep the index");
}

void Fingerprint() {
  SlotTable t(4, 1000);
  t.Activate(0, 10);
  t.Activate(1, 20);
  const int a[] = {0, 1};
  const int32_t ta[] = {1, 2};
  const uint64_t base = StepFingerprint(t, a, ta, 2);
  Check(base == StepFingerprint(t, a, ta, 2), "deterministic");
  const int32_t tb[] = {1, 3};
  Check(base != StepFingerprint(t, a, tb, 2), "a token differs");
  const int b[] = {1, 0};
  const int32_t tc[] = {2, 1};
  Check(base != StepFingerprint(t, b, tc, 2), "the row order differs");
  Check(base != StepFingerprint(t, a, ta, 1), "the row count differs");
  t.Advance(0);
  Check(base != StepFingerprint(t, a, ta, 2), "a position differs");
}

void Memory() {
  // The 27B on one card: 16 full-attention layers x 4 KV heads x head_dim 256 -> 16 KiB per token, K and V.
  Check(KvBytesPerToken(16, 4, 256) == 32768, "TP=1: 32 KiB of fp8 K+V per token across the 16 attention layers");
  // Per TP rank (2 KV heads): 16 KiB per token, the figure docs/pp-tp2-hybrid.md uses.
  Check(KvBytesPerToken(16, 2, 256) == 16384, "TP=2: 16 KiB per token per rank");
  const int64_t gib = int64_t{1} << 30;
  Check(BatchKvBytes(4, 32768, 16, 4, 256) == 4 * gib, "4 slots x 32k tokens at TP=1 = 4 GiB");
  Check(BatchKvBytes(4, 32768, 16, 2, 256) == 2 * gib, "... 2 GiB per rank at TP=2");
  // GDN: 48 layers; H=48, V=K=128 -> 3 MiB fp32 recurrent state per slot; conv_dim 10240, state_len_max 3 -> 60 KiB.
  const int64_t per_slot = int64_t{48} * 128 * 128 * 4 + int64_t{10240} * 3 * 2;
  Check(BatchGdnBytes(4, 48, 48, 128, 128, 10240, 3) == 48 * 5 * per_slot, "GDN: (slots + 1) physical slots per layer");
  Check(BatchGdnBytes(4, 48, 48, 128, 128, 10240, 3) < gib, "4 slots of GDN state stay under 1 GiB");
}

}  // namespace

int main() {
  Layout();
  Table();
  Validation();
  Meta();
  Mrope();
  Fingerprint();
  Memory();
  if (g_fails != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_fails);
    return 1;
  }
  std::printf("test_batch_plan_cpu: all checks passed\n");
  return 0;
}
