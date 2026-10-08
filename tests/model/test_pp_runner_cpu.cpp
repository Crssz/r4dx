// tests/model/test_pp_runner_cpu.cpp -- CPU-only checks of src/model/pp_stage_buffers.h, the buffer half of the thread-agnostic stage runners
// (docs/pp-tp2-hybrid.md 7 "Who owns what", critique item 3): the sizing rules (the hybrid's carry_kv = false slot, the KV sync buffer at its
// maximum up front), the allocation protocol (nothing allocated until the owning thread asks; each group on the thread that asks; a failed
// allocation leaves nothing behind; release frees exactly once), and a fake two-stage call whose stages swap threads between calls -- the
// channel and the buffers have no thread affinity. The HIP-side runners (pp_stage_runner.cpp) are guarded by test_pp_real_identity /
// test_pp_emulate_identity / test_hybrid_emulate_identity on a GPU. No HIP call, no container, always runs.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pp_stage_buffers.h"

using namespace r4dx::model::pp;
using std::chrono::milliseconds;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

template <class Ex>
bool Throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const Ex&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// A host allocator that records who allocated what, on which thread, and what is still live.
struct Recorder {
  std::map<uint8_t*, size_t> live;
  std::vector<std::pair<size_t, std::thread::id>> allocs;
  int frees = 0;
  int fail_on_alloc = -1;  // the n-th alloc call (0-based) throws
  int alloc_calls = 0;

  HostAllocator Make() {
    HostAllocator a;
    a.alloc = [this](size_t bytes) -> uint8_t* {
      if (alloc_calls++ == fail_on_alloc) throw std::runtime_error("out of pinned memory");
      uint8_t* p = static_cast<uint8_t*>(std::malloc(bytes));
      live[p] = bytes;
      allocs.emplace_back(bytes, std::this_thread::get_id());
      return p;
    };
    a.release = [this](uint8_t* p) {
      if (live.erase(p) != 1) throw std::logic_error("double free");
      ++frees;
      std::free(p);
    };
    return a;
  }
  ~Recorder() {
    for (auto& kv : live) std::free(kv.first);
  }
};

// The production geometry (Qwen3.8-27B): hidden 5120, 256-row chunks, 16-row blocks of 4 kv heads x 2 x 256 fp8 = 32 KiB, 7 attention layers
// below the largest split (k = 29), 3 DFlash columns.
StageBufferGeometry Production() {
  StageBufferGeometry g;
  g.slots = 3;
  g.chunk_rows = 256;
  g.hidden = 5120;
  g.dfl_cols_max = 4;
  g.attn_layers_max = 7;
  g.block_size = 16;
  g.block_stride_bytes = 32768;
  g.gdn_bytes = 80ull << 20;
  g.kv_wire_full_ctx = 7ull * (131072 / 16) * 32768;  // 1.75 GiB: more than the cap
  return g;
}

void Sizing() {
  const StageBufferGeometry g = Production();
  const StageBufferSpec pp2 = MakeStageBufferSpec(g, /*carry_kv=*/true);
  const StageBufferSpec hyb = MakeStageBufferSpec(g, /*carry_kv=*/false);
  Check(pp2.slot_bytes == MaxSlotBytes(256, 5120, 4, 7, 16, 32768), "carry_kv = true is the --pp 2 slot (unchanged formula)");
  Check(hyb.slot_bytes == MaxSlotBytes(256, 5120, 4, 0, 16, 32768), "carry_kv = false is the carry-only slot (the P2 harness's)");
  const size_t kv_per_layer = AlignUp(static_cast<size_t>((256 + 15) / 16 + 1) * 32768);
  Check(pp2.slot_bytes - hyb.slot_bytes == 7 * kv_per_layer, "the KV payload is the whole difference: 7 layers x the worst-case blocks");
  Check(hyb.slot_bytes < pp2.slot_bytes, "the hybrid slot is smaller");
  Check(hyb.slots == 3 && hyb.gdn_bytes == g.gdn_bytes, "slot count and GDN size pass through");

  // Whatever a chunk can be (any start, 1..256 rows), the layout of its mode fits the slot of that mode.
  bool fits = true, hyb_lean = true;
  const std::vector<int64_t> attn = {3, 7, 11, 15, 19, 23, 27};
  for (int64_t pos : {int64_t{0}, int64_t{1}, int64_t{15}, int64_t{16}, int64_t{4095}, int64_t{131071 - 255}}) {
    for (int64_t rows : {int64_t{1}, int64_t{17}, int64_t{255}, int64_t{256}}) {
      const SlotLayout full = MakeSlotLayout(pos, rows, 5120, 4, attn, 16, 32768);
      const SlotLayout lean = MakeSlotLayout(pos, rows, 5120, 4, {}, 16, 32768);
      if (full.total > pp2.slot_bytes || lean.total > hyb.slot_bytes) fits = false;
      if (!lean.kv.empty() || lean.kv_bytes != 0 || lean.total != lean.off_dfl + AlignUp(lean.dfl_bytes)) hyb_lean = false;
    }
  }
  Check(fits, "every chunk's payload fits the slot of its mode");
  Check(hyb_lean, "without attention layers the layout is the carry and the DFlash columns only");

  // The KV sync buffer: the stage-KV capacity's wire bytes, clamped; allocated at that size once.
  Check(KvSyncCapacity(0) == kMinKvSyncBytes, "no attention layers: the 16 MiB floor (as the lazy buffer's minimum was)");
  Check(KvSyncCapacity(100ull << 20) == (100ull << 20), "below the cap: exactly the whole-capacity wire size");
  Check(KvSyncCapacity(kMaxKvSyncBytes) == kMaxKvSyncBytes && KvSyncCapacity(8ull << 30) == kMaxKvSyncBytes,
        "above the cap: 512 MiB");
  Check(pp2.kv_sync_bytes == kMaxKvSyncBytes, "the 128k geometry gets the capped buffer");
  Check(KvSyncFits(kMaxKvSyncBytes, kMaxKvSyncBytes) && !KvSyncFits(kMaxKvSyncBytes + 1, kMaxKvSyncBytes) && KvSyncFits(0, 0),
        "KvSyncFits: need <= capacity (the B-only fallback above it, as before)");
  // The largest sync-back a stage-KV of S tokens can ask for is rows [0, S): never more than the buffer.
  StageBufferGeometry small = g;
  small.kv_wire_full_ctx = 7ull * (32768 / 16) * 32768;
  const StageBufferSpec s32k = MakeStageBufferSpec(small, true);
  Check(s32k.kv_sync_bytes == small.kv_wire_full_ctx && s32k.kv_sync_bytes < kMaxKvSyncBytes, "32k stage-KV: 448 MiB, below the cap");
  Check(Throws<std::invalid_argument>([&] {
          StageBufferGeometry bad = g;
          bad.slots = 0;
          (void)MakeStageBufferSpec(bad, true);
        }),
        "zero slots refused");
}

void Allocation() {
  Recorder rec;
  StageBufferSpec spec = MakeStageBufferSpec(Production(), false);
  spec.gdn_bytes = 1ull << 20;        // (the real sizes are checked in Sizing; keep the fake allocations small)
  spec.kv_sync_bytes = 32ull << 20;
  {
    StageBuffers bufs(spec, rec.Make());
    Check(rec.allocs.empty(), "construction allocates nothing");
    Check(!bufs.AProducedAllocated() && !bufs.BProducedAllocated(), "nothing allocated yet");
    Check(Throws<std::logic_error>([&] { (void)bufs.SlotPointers(); }), "SlotPointers before the stage-A allocation is refused");
    Check(bufs.KvSyncBytes() == 0 && bufs.GdnHand() == nullptr && bufs.KvSync() == nullptr, "no memory, no size");

    // Each group is allocated by the thread that asks, and only then.
    std::thread::id a_id, b_id;
    std::thread ta([&] {
      a_id = std::this_thread::get_id();
      bufs.AllocateAProduced();
    });
    ta.join();
    Check(bufs.AProducedAllocated() && !bufs.BProducedAllocated(), "stage-A-produced allocated, stage-B-produced not");
    Check(rec.allocs.size() == 4, "3 slots + the GDN hand-off");
    bool all_on_a = true;
    for (const auto& al : rec.allocs) all_on_a = all_on_a && al.second == a_id;
    Check(all_on_a && bufs.AProducedThread() == a_id, "every stage-A-produced block was allocated on the asking thread");
    Check(bufs.SlotPointers().size() == 3 && bufs.GdnHand() != nullptr && bufs.GdnSync() == nullptr, "slots + hand-off exist, the sync buffers do not");

    std::thread tb([&] {
      b_id = std::this_thread::get_id();
      bufs.AllocateBProduced();
    });
    tb.join();
    Check(rec.allocs.size() == 6, "+ the GDN sync buffer and the KV sync buffer");
    bool b_ok = true;
    for (size_t i = 4; i < rec.allocs.size(); ++i) b_ok = b_ok && rec.allocs[i].second == b_id;
    Check(b_ok && bufs.BProducedThread() == b_id && a_id != b_id, "the sync-back buffers were allocated on the other thread");
    Check(bufs.KvSyncBytes() == spec.kv_sync_bytes && rec.allocs[5].first == spec.kv_sync_bytes, "the KV sync buffer exists at its maximum from the start");
    Check(bufs.GdnSync() != nullptr && bufs.KvSync() != nullptr && bufs.GdnSync() != bufs.GdnHand(), "distinct blocks");
    Check(Throws<std::logic_error>([&] { bufs.AllocateAProduced(); }) && Throws<std::logic_error>([&] { bufs.AllocateBProduced(); }),
          "a second allocation of a group is refused (nothing grows lazily)");
    Check(rec.allocs.size() == 6, "... and allocated nothing");

    bufs.ReleaseAProduced();
    Check(!bufs.AProducedAllocated() && bufs.BProducedAllocated() && rec.frees == 4 && rec.live.size() == 2, "release A frees its 4 blocks only");
    bufs.ReleaseAProduced();
    Check(rec.frees == 4, "release is idempotent");
    bufs.AllocateAProduced();  // a released group can be allocated again (the owner's reload)
    Check(rec.live.size() == 6, "re-allocation after release");
  }
  Check(rec.live.empty() && rec.frees == 10, "the destructor frees whatever is left, exactly once");
}

void ZeroSizeAndFailure() {
  {
    // No GDN layers below the split (a tiny split): zero-byte blocks are never requested and read as null.
    Recorder rec;
    StageBufferSpec spec;
    spec.slots = 2;
    spec.slot_bytes = 1024;
    spec.gdn_bytes = 0;
    spec.kv_sync_bytes = kMinKvSyncBytes;
    StageBuffers bufs(spec, rec.Make());
    bufs.AllocateAProduced();
    bufs.AllocateBProduced();
    Check(rec.allocs.size() == 3 && bufs.GdnHand() == nullptr && bufs.GdnSync() == nullptr, "gdn_bytes = 0: no allocation, null pointers");
  }
  for (int fail_at = 0; fail_at < 4; ++fail_at) {
    Recorder rec;
    rec.fail_on_alloc = fail_at;
    StageBufferSpec spec;
    spec.slots = 3;
    spec.slot_bytes = 4096;
    spec.gdn_bytes = 1 << 20;
    spec.kv_sync_bytes = kMinKvSyncBytes;
    StageBuffers bufs(spec, rec.Make());
    Check(Throws<std::runtime_error>([&] { bufs.AllocateAProduced(); }), "an allocation failure propagates");
    Check(rec.live.empty() && !bufs.AProducedAllocated(), "... and the half-built group is freed");
    rec.fail_on_alloc = -1;
    bufs.AllocateAProduced();
    Check(bufs.AProducedAllocated() && rec.live.size() == 4, "the group can be allocated again afterwards");
  }
  {
    Recorder rec;
    rec.fail_on_alloc = 1;  // the KV sync buffer (the second block of the B group)
    StageBufferSpec spec;
    spec.slots = 1;
    spec.slot_bytes = 64;
    spec.gdn_bytes = 128;
    spec.kv_sync_bytes = kMinKvSyncBytes;
    StageBuffers bufs(spec, rec.Make());
    Check(Throws<std::runtime_error>([&] { bufs.AllocateBProduced(); }) && rec.live.empty() && !bufs.BProducedAllocated(),
          "a failing KV sync allocation frees the GDN sync block taken before it");
  }
  Check(Throws<std::invalid_argument>([] { StageBuffers b(StageBufferSpec{}, HostAllocator{}); }), "an allocator is required");
}

// A fake two-stage call: stage A fills the slots (payload = f(call, chunk, byte)), publishes the GDN hand-off flag; stage B drains and checks.
// Neither stage knows which thread it runs on: the threads swap roles between the calls (A on thread 1, then on the main thread, ...).
void ThreadAgnosticCalls() {
  Recorder rec;
  StageBufferSpec spec;
  spec.slots = 3;
  spec.slot_bytes = 4096;
  spec.gdn_bytes = 4096;
  spec.kv_sync_bytes = kMinKvSyncBytes;
  StageBuffers bufs(spec, rec.Make());
  std::thread alloc_a([&] { bufs.AllocateAProduced(); });  // "stage A's worker"
  alloc_a.join();
  bufs.AllocateBProduced();                                  // "the facade"
  StageChannel ch(bufs.SlotPointers(), spec.slot_bytes);

  constexpr int kChunks = 40;
  const auto stage_a = [&](int64_t call) -> std::exception_ptr {
    try {
      for (int i = 0; i < kChunks; ++i) {
        StageChannel::Slot* s = ch.AcquireFree(milliseconds(5000));
        for (size_t b = 0; b < 64; ++b) s->data[b] = static_cast<uint8_t>(call * 31 + i + b);
        s->hdr.rows = i;
        s->hdr.bytes = 64;
        s->hdr.last = i == kChunks - 1;
        ch.Publish(s);
      }
      std::memset(bufs.GdnHand(), static_cast<int>(call), 16);
      ch.PublishBulk(call);
      return nullptr;
    } catch (...) {
      ch.Poison("stage A failed");
      return std::current_exception();
    }
  };
  const auto stage_b = [&](int64_t call, bool* ok) -> std::exception_ptr {
    try {
      for (int i = 0; i < kChunks; ++i) {
        StageChannel::Slot* s = ch.AcquireFull(milliseconds(5000));
        for (size_t b = 0; b < 64; ++b) {
          if (s->data[b] != static_cast<uint8_t>(call * 31 + i + b)) *ok = false;
        }
        ch.Release(s);
      }
      ch.WaitBulk(call, milliseconds(5000));
      if (bufs.GdnHand()[0] != static_cast<uint8_t>(call)) *ok = false;
      return nullptr;
    } catch (...) {
      ch.Poison("stage B failed");
      return std::current_exception();
    }
  };

  bool ok = true;
  for (int64_t call = 1; call <= 4; ++call) {
    ch.BeginCall(call);
    std::exception_ptr ea, eb;
    if (call % 2 == 1) {  // A on a worker, B here
      std::thread t([&] { ea = stage_a(call); });
      eb = stage_b(call, &ok);
      t.join();
    } else {  // B on a worker, A here
      std::thread t([&] { eb = stage_b(call, &ok); });
      ea = stage_a(call);
      t.join();
    }
    if (ea || eb) ok = false;
  }
  Check(ok, "four calls, the stages swapping threads, move every chunk and the hand-off intact");
  Check(ch.InFlight() == 0 && !ch.Poisoned(), "the channel ends quiescent");
}

}  // namespace

int main() {
  Sizing();
  Allocation();
  ZeroSizeAndFailure();
  ThreadAgnosticCalls();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_pp_runner_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_pp_runner_cpu: PASS\n");
  return 0;
}
