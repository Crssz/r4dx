// tests/model/test_pp_channel_cpu.cpp -- CPU-only checks of src/model/pp_channel.h, the hand-over of the real
// pipeline-parallel prefill (docs/pp-prefill.md Phase 2): FIFO slot order, back-pressure, bounded waits, the poison
// flag, the per-call reset, the bulk (GDN state) flag, the slot payload arithmetic and a million-item two-thread stress
// run with payload integrity checks. No HIP call, no container, always runs.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "pp_channel.h"

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

struct Ring {
  std::vector<std::vector<uint8_t>> mem;
  std::vector<uint8_t*> ptrs;
  explicit Ring(int slots, size_t cap) : mem(static_cast<size_t>(slots), std::vector<uint8_t>(cap)) {
    for (auto& m : mem) ptrs.push_back(m.data());
  }
};

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

void Fifo() {
  Ring r(3, 64);
  StageChannel ch(r.ptrs, 64);
  ch.BeginCall(7);
  // Fill the ring: three slots acquire and publish without a consumer.
  StageChannel::Slot* s[3];
  for (int i = 0; i < 3; ++i) {
    s[i] = ch.AcquireFree(milliseconds(50));
    Check(s[i]->index == i, "the n-th AcquireFree returns slot n % slots");
    Check(s[i]->hdr.call_id == 7, "a slot carries the call id");
    s[i]->hdr.pos = 100 * i;
    s[i]->hdr.rows = 256;
    s[i]->hdr.bytes = 10;
    ch.Publish(s[i]);
    Check(s[i]->hdr.seq == i, "Publish numbers the chunks from 0");
  }
  Check(ch.InFlight() == 3, "three chunks in flight");
  // The ring is full: a fourth acquire times out (back-pressure), and does not poison.
  Check(Throws<ChannelTimeout>([&] { (void)ch.AcquireFree(milliseconds(20)); }), "a full ring times out the producer");
  Check(!ch.Poisoned(), "a timeout does not poison");
  // The consumer drains in order.
  for (int i = 0; i < 3; ++i) {
    StageChannel::Slot* t = ch.AcquireFull(milliseconds(50));
    Check(t == s[i] && t->hdr.pos == 100 * i, "the consumer sees the slots in publish order");
    ch.Release(t);
  }
  Check(ch.InFlight() == 0, "all released");
  // Nothing published: the consumer times out (starvation), then a release frees the producer.
  Check(Throws<ChannelTimeout>([&] { (void)ch.AcquireFull(milliseconds(20)); }), "an empty ring times out the consumer");
  StageChannel::Slot* a = ch.AcquireFree(milliseconds(50));
  Check(a->index == 0 || a->index == 3 % 3, "the ring wraps");
  a->hdr.rows = 1;
  ch.Publish(a);
  StageChannel::Slot* b = ch.AcquireFull(milliseconds(50));
  ch.Release(b);
  Check(ch.GetStats().published == 4 && ch.GetStats().taken == 4, "stats count the chunks");
}

void Misuse() {
  Ring r(2, 32);
  StageChannel ch(r.ptrs, 32);
  ch.BeginCall(1);
  StageChannel::Slot* a = ch.AcquireFree(milliseconds(50));
  Check(Throws<ChannelTimeout>([&] { (void)ch.AcquireFree(milliseconds(10)); }),
        "a second acquire before Publish does not hand out another slot");
  a->hdr.bytes = 33;
  Check(Throws<std::logic_error>([&] { ch.Publish(a); }), "a payload larger than the slot is refused");
  a->hdr.bytes = 32;
  ch.Publish(a);
  Check(Throws<std::logic_error>([&] { ch.Publish(a); }), "publishing twice is refused");
  Check(Throws<std::logic_error>([&] { ch.BeginCall(2); }), "BeginCall needs a quiescent channel");
  StageChannel::Slot* t = ch.AcquireFull(milliseconds(50));
  ch.Release(t);
  Check(Throws<std::logic_error>([&] { ch.Release(t); }), "releasing twice is refused");
  ch.BeginCall(2);  // quiescent again
  Check(Throws<std::invalid_argument>([&] { StageChannel bad({}, 8); }), "a channel needs at least one slot");
}

void PoisonAndReset() {
  Ring r(3, 16);
  StageChannel ch(r.ptrs, 16);
  ch.BeginCall(1);
  // A consumer blocked in AcquireFull wakes with ChannelPoisoned when the producer fails.
  std::atomic<int> got{0};
  std::thread consumer([&] {
    try {
      (void)ch.AcquireFull(milliseconds(5000));
      got = 1;
    } catch (const ChannelPoisoned&) {
      got = 2;
    } catch (...) {
      got = 3;
    }
  });
  std::this_thread::sleep_for(milliseconds(30));
  ch.Poison("stage A threw");
  consumer.join();
  Check(got == 2, "poison wakes a blocked consumer");
  Check(ch.Poisoned() && ch.PoisonReason() == "stage A threw", "the first reason sticks");
  ch.Poison("another");
  Check(ch.PoisonReason() == "stage A threw", "poison is idempotent");
  Check(Throws<ChannelPoisoned>([&] { (void)ch.AcquireFree(milliseconds(10)); }), "a poisoned channel refuses the producer");
  Check(Throws<ChannelPoisoned>([&] { ch.BeginCall(2); }), "and a new call");
  Check(Throws<ChannelPoisoned>([&] { ch.PublishBulk(1); }), "and the bulk flag");
  // A producer blocked on a full ring wakes too.
  ch.Reset();
  Check(!ch.Poisoned(), "Reset clears the poison");
  ch.BeginCall(3);
  for (int i = 0; i < 3; ++i) ch.Publish(ch.AcquireFree(milliseconds(50)));
  std::atomic<int> pgot{0};
  std::thread producer([&] {
    try {
      (void)ch.AcquireFree(milliseconds(5000));
      pgot = 1;
    } catch (const ChannelPoisoned&) {
      pgot = 2;
    } catch (...) {
      pgot = 3;
    }
  });
  std::this_thread::sleep_for(milliseconds(30));
  ch.Poison("stage B threw");
  producer.join();
  Check(pgot == 2, "poison wakes a producer blocked on a full ring");
  // Reset works from a state with chunks in flight (an aborted call), and the next call starts clean.
  ch.Reset();
  ch.BeginCall(4);
  StageChannel::Slot* s = ch.AcquireFree(milliseconds(50));
  Check(s->index == 0 && s->hdr.call_id == 4, "after Reset the ring starts again at slot 0");
}

void Bulk() {
  Ring r(3, 16);
  StageChannel ch(r.ptrs, 16);
  ch.BeginCall(9);
  Check(!ch.BulkReady(9), "no bulk yet");
  Check(Throws<ChannelTimeout>([&] { ch.WaitBulk(9, milliseconds(20)); }), "WaitBulk times out without a publisher");
  std::thread t([&] {
    std::this_thread::sleep_for(milliseconds(20));
    ch.PublishBulk(9);
  });
  ch.WaitBulk(9, milliseconds(2000));
  t.join();
  Check(ch.BulkReady(9) && !ch.BulkReady(10), "bulk is per call");
  ch.BeginCall(10);
  Check(!ch.BulkReady(9) && !ch.BulkReady(10), "BeginCall clears the bulk flag");
}

void Layout() {
  // 256 rows from an aligned position, 8 attention layers, 16-row blocks, 32 KiB block stride (4 heads x 16 x 512 B).
  const size_t stride = 32768;
  std::vector<int64_t> layers = {3, 7, 11, 15, 19, 23, 27, 31};
  const SlotLayout a = MakeSlotLayout(256, 256, 5120, /*dfl_cols=*/2, layers, 16, stride);
  Check(a.carry_bytes == 256u * 5120u * 2u, "a carry buffer is rows x hidden x 2 bytes (2.5 MiB)");
  Check(a.kv.size() == 8 && a.kv[0].blocks == 16 && a.kv[0].block0 == 16, "256 aligned rows = 16 blocks of each layer");
  Check(a.kv[0].bytes == 16u * stride && a.kv_bytes == 8u * 16u * stride, "4 MiB of KV for eight layers");
  Check(a.dfl_bytes == 256u * 2u * 5120u * 2u, "two DFlash columns");
  const SlotLayout m = MakeSlotLayout(8, 256, 5120, 2, layers, 16, stride);
  Check(m.kv[0].blocks == 17, "a mid-block start touches one more block");
  Check(m.total <= MaxSlotBytes(256, 5120, 2, 8, 16, stride), "MaxSlotBytes bounds the worst alignment");
  // Non-overlap and alignment of every region, for a spread of positions and row counts.
  bool ok = true;
  for (int64_t pos : {int64_t{0}, int64_t{1}, int64_t{15}, int64_t{16}, int64_t{300}, int64_t{8191}}) {
    for (int64_t rows : {int64_t{1}, int64_t{44}, int64_t{64}, int64_t{256}}) {
      for (int64_t dfl : {int64_t{0}, int64_t{1}, int64_t{3}}) {
        const SlotLayout l = MakeSlotLayout(pos, rows, 5120, dfl, layers, 16, stride);
        std::vector<std::pair<size_t, size_t>> regions = {{l.off_cur, l.carry_bytes}, {l.off_norm, l.carry_bytes},
                                                          {l.off_pre, l.carry_bytes}, {l.off_dfl, l.dfl_bytes}};
        for (const KvPiece& p : l.kv) regions.push_back({p.host_off, p.bytes});
        std::sort(regions.begin(), regions.end());
        for (size_t i = 0; i < regions.size(); ++i) {
          if (regions[i].first % kSlotAlign != 0) ok = false;
          if (i > 0 && regions[i - 1].first + regions[i - 1].second > regions[i].first) ok = false;
          if (regions[i].first + regions[i].second > l.total) ok = false;
        }
        if (l.total > MaxSlotBytes(rows, 5120, dfl, 8, 16, stride)) ok = false;
      }
    }
  }
  Check(ok, "slot regions are 256-byte aligned, disjoint and inside the total, which MaxSlotBytes bounds");
  // The GDN wire: 3 MiB fp32 recurrent state + [10240][3] bf16 history per layer.
  const GdnWire w = MakeGdnWire(48, 128, 128, 10240, 4);
  Check(w.recurrent_bytes == 3u * 1024u * 1024u && w.conv_bytes == 10240u * 3u * 2u, "GDN wire sizes");
  Check(w.PerLayer() % kSlotAlign == 0 && w.PerLayer() >= w.recurrent_bytes + w.conv_bytes, "GDN wire is aligned");
}

// A million chunks through a 3-slot ring, producer and consumer on their own threads, the consumer slower in bursts:
// every payload arrives intact and in order, no slot is touched by both sides at once (a checksum written by the
// producer is verified by the consumer, and the consumer scribbles over the slot before releasing it).
void Stress() {
  constexpr int kItems = 1000000;
  Ring r(3, 64);
  StageChannel ch(r.ptrs, 64);
  ch.BeginCall(1);
  std::atomic<bool> bad{false};
  std::thread producer([&] {
    try {
      for (int i = 0; i < kItems; ++i) {
        StageChannel::Slot* s = ch.AcquireFree(milliseconds(10000));
        for (int b = 0; b < 64; ++b) s->data[b] = static_cast<uint8_t>(i + b);
        s->hdr.pos = i;
        s->hdr.rows = i % 256 + 1;
        s->hdr.bytes = 64;
        s->hdr.last = i == kItems - 1;
        ch.Publish(s);
      }
      ch.PublishBulk(1);
    } catch (...) {
      bad = true;
      ch.Poison("producer");
    }
  });
  std::thread consumer([&] {
    try {
      for (int i = 0; i < kItems; ++i) {
        StageChannel::Slot* s = ch.AcquireFull(milliseconds(10000));
        if (s->hdr.seq != i || s->hdr.pos != i || s->hdr.rows != i % 256 + 1 || s->hdr.bytes != 64 ||
            s->hdr.last != (i == kItems - 1)) {
          bad = true;
        }
        for (int b = 0; b < 64; ++b) {
          if (s->data[b] != static_cast<uint8_t>(i + b)) bad = true;
          s->data[b] = 0xEE;
        }
        if ((i & 0xFFFF) == 0) std::this_thread::sleep_for(milliseconds(1));  // a burst of back-pressure
        ch.Release(s);
      }
      ch.WaitBulk(1, milliseconds(10000));
    } catch (...) {
      bad = true;
      ch.Poison("consumer");
    }
  });
  producer.join();
  consumer.join();
  Check(!bad, "1M chunks through the ring arrive intact, in order, and exclusively owned");
  Check(ch.InFlight() == 0 && ch.GetStats().published == kItems && ch.GetStats().taken == kItems, "all drained");
}

}  // namespace

int main() {
  Fifo();
  Misuse();
  PoisonAndReset();
  Bulk();
  Layout();
  Stress();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_pp_channel_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_pp_channel_cpu: PASS\n");
  return 0;
}

