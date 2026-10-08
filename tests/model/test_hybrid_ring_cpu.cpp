// tests/model/test_hybrid_ring_cpu.cpp -- CPU-only checks of src/model/hybrid_ring.h, the host handshake of the hybrid mode's two-card
// reshard (docs/pp-tp2-hybrid.md 3, 7): RingPipe (one direction of the pinned ring: publish / consume, FIFO, back-pressure, bounded
// waits, poison), HostGate, and RunLockstep, the per-card loop. Two threads play the two cards with fake copies into plain memory;
// the stress run checks that no schedule deadlocks, that no slot is refilled before its batch was consumed, that every byte arrives
// in order, and that a failure on one side frees the other at once. No HIP call; always runs.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hybrid_ring.h"

using namespace r4dx::model::hybrid;
using std::chrono::milliseconds;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
template <class E = std::exception, class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

void PipeBasics() {
  RingPipe p(3);
  Check(p.Slots() == 3, "slots");
  p.WaitSlotFree(0, milliseconds(10));
  p.WaitSlotFree(2, milliseconds(10));
  Check(Throws<RingTimeout>([&] { p.WaitSlotFree(3, milliseconds(20)); }), "batch 3 needs batch 0 consumed: the wait times out");
  Check(Throws<RingTimeout>([&] { p.WaitPublished(0, milliseconds(20)); }), "nothing published yet");
  Check(Throws<std::logic_error>([&] { p.Publish(1); }), "batches are published in order");
  p.Publish(0);
  p.Publish(1);
  p.WaitPublished(1, milliseconds(10));
  Check(Throws<std::logic_error>([&] { p.Consume(1); }), "consumed in order");
  p.Consume(0);
  p.WaitSlotFree(3, milliseconds(10));
  Check(p.Published() == 2 && p.Consumed() == 1, "counters");
  p.Consume(1);
  Check(Throws<std::logic_error>([&] { p.Consume(2); }), "never consumed ahead of the publish");
  p.Poison("first");
  p.Poison("second");
  Check(Throws<RingPoisoned>([&] { p.WaitPublished(5, milliseconds(10)); }), "poisoned: a wait throws");
  Check(Throws<RingPoisoned>([&] { p.WaitSlotFree(0, milliseconds(10)); }), "poisoned even where the condition holds");
  Check(Throws<RingPoisoned>([&] { p.Publish(2); }), "poisoned: publish throws");
  Check(p.Poisoned(), "poisoned flag");
  p.Reset();
  Check(!p.Poisoned() && p.Published() == 0 && p.Consumed() == 0, "reset clears the poison and the counters");
  p.Publish(0);
}

void PipeWakeups() {
  RingPipe p(2);
  std::atomic<int> state{0};
  std::thread waiter([&] {
    try {
      p.WaitPublished(0, milliseconds(5000));
      state = 1;
    } catch (const RingPoisoned&) {
      state = 2;
    } catch (...) {
      state = 3;
    }
  });
  std::this_thread::sleep_for(milliseconds(30));
  Check(state == 0, "the waiter is blocked");
  p.Publish(0);
  waiter.join();
  Check(state == 1, "a publish wakes the waiter");

  RingPipe q(2);
  std::thread w2([&] {
    try {
      q.WaitPublished(0, milliseconds(5000));
      state = 11;
    } catch (const RingPoisoned&) {
      state = 12;
    } catch (...) {
      state = 13;
    }
  });
  std::this_thread::sleep_for(milliseconds(30));
  q.Poison("peer failed");
  w2.join();
  Check(state == 12, "a poison wakes the waiter with RingPoisoned");
}

void Gate() {
  HostGate g;
  Check(!g.IsOpen(), "closed");
  Check(Throws<RingTimeout>([&] { g.Wait(milliseconds(20)); }), "a closed gate times out");
  std::atomic<int> state{0};
  std::thread t([&] {
    try {
      g.Wait(milliseconds(5000));
      state = 1;
    } catch (...) {
      state = 2;
    }
  });
  std::this_thread::sleep_for(milliseconds(20));
  g.Open();
  t.join();
  Check(state == 1 && g.IsOpen(), "Open releases the waiter");
  g.Wait(milliseconds(1));
  g.Reset();
  Check(!g.IsOpen(), "reset closes it");
  g.Poison("x");
  Check(Throws<RingPoisoned>([&] { g.Wait(milliseconds(10)); }), "a poisoned gate throws");
  g.Reset();
  g.Open();
  g.Wait(milliseconds(1));
}

void Classify() {
  const auto cap = [](auto f) {
    try {
      f();
    } catch (...) {
      return std::current_exception();
    }
    return std::exception_ptr();
  };
  Check(IsPoisonConsequence(cap([] { throw RingPoisoned("x"); })), "RingPoisoned is a consequence");
  Check(IsPoisonConsequence(cap([] { throw r4dx::model::pp::ChannelPoisoned("x"); })), "ChannelPoisoned is a consequence");
  Check(!IsPoisonConsequence(cap([] { throw RingTimeout("x"); })), "a timeout is a cause");
  Check(!IsPoisonConsequence(cap([] { throw std::runtime_error("x"); })), "anything else is a cause");
  Check(!IsPoisonConsequence(cap([] { throw r4dx::model::pp::ChannelTimeout("x"); })), "a channel timeout is a cause");
}

void LockstepSteps_() {
  Check(LockstepSteps(0, 0) == 0, "no batches, no steps");
  Check(LockstepSteps(3, 0) == 3, "exports only");
  Check(LockstepSteps(0, 3) == 4, "imports only: import i runs one step after its publish");
  Check(LockstepSteps(5, 3) == 5 && LockstepSteps(2, 6) == 7, "the longer side decides");
}

// ---- two cards ------------------------------------------------------------------------------------------------------------------------
// A fake card: `ring` is its outgoing ring (slots x piece bytes), `out_data[g]` the bytes of its g-th outgoing batch (numbered across ALL
// phases, as the pipe numbers them). The peer's imports read this ring and compare with the expected pattern. `consumed_by_peer[g]` is set
// by the peer right after it finished reading batch g; an export that refills slot g % slots before batch g - slots was consumed is a bug
// the test catches. Phases run back to back on each thread with NO barrier between them, as in TpModel (a card with nothing to gather
// starts its scatter while the peer still drains the gather).
struct Card {
  int id = 0;
  int slots = 3;
  size_t piece = 0;
  std::vector<uint8_t> ring;
  std::vector<std::vector<uint8_t>> out_data;
  std::unique_ptr<std::atomic<bool>[]> consumed_by_peer;
  std::atomic<int> bad{0};
  std::atomic<int64_t> exported{0}, imported{0};
  int64_t fail_export_at = -1;   // throw in the enqueue of this (global) exported batch
  int64_t fail_import_at = -1;   // ... of this (global) imported batch of the peer
  int max_delay_us = 0;
  std::mt19937 rng;
};

// A spin, not a sleep: Windows sleeps in ~15 ms quanta, which would turn a microsecond of jitter into minutes of test.
void Delay(Card& c) {
  if (c.max_delay_us <= 0) return;
  const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(c.rng() % static_cast<unsigned>(c.max_delay_us + 1));
  while (std::chrono::steady_clock::now() < until) std::this_thread::yield();
}

void Prepare(Card& c, int id, int64_t n_total_out, size_t piece, uint32_t seed, int max_delay_us) {
  c.id = id;
  c.piece = piece;
  c.rng.seed(seed);
  c.max_delay_us = max_delay_us;
  c.ring.assign(static_cast<size_t>(c.slots) * piece, 0xEE);
  c.out_data.clear();
  for (int64_t i = 0; i < n_total_out; ++i) {
    std::vector<uint8_t> d(1 + c.rng() % piece);
    for (size_t k = 0; k < d.size(); ++k) d[k] = static_cast<uint8_t>((i * 131 + k * 7 + id * 29) & 0xFF);
    c.out_data.push_back(std::move(d));
  }
  c.consumed_by_peer.reset(new std::atomic<bool>[static_cast<size_t>(n_total_out) + 1]);
  for (int64_t i = 0; i <= n_total_out; ++i) c.consumed_by_peer[static_cast<size_t>(i)] = false;
  c.bad = 0;
  c.exported = c.imported = 0;
}

struct Phase {
  int64_t n0 = 0, n1 = 0;  // batches card 0 sends / card 1 sends
};

// One card's loop over all phases; `peer` is the other card. Returns the exception (null on success).
std::exception_ptr RunCard(Card& me, Card& peer, RingPipe& out, RingPipe& in, const std::vector<Phase>& phases, milliseconds timeout,
                           const std::function<void(const std::string&)>& poison) {
  try {
    int64_t out_base = 0, in_base = 0;
    for (const Phase& ph : phases) {
      const int64_t n_out = me.id == 0 ? ph.n0 : ph.n1, n_in = me.id == 0 ? ph.n1 : ph.n0;
      LockstepCallbacks cb;
      cb.export_enqueue = [&](int64_t i) {
        const int64_t g = out_base + i;
        if (g == me.fail_export_at) throw std::runtime_error("injected export failure");
        // the slot must be free: the peer consumed batch g - slots
        if (g - me.slots >= 0 && !me.consumed_by_peer[static_cast<size_t>(g - me.slots)].load()) ++me.bad;
        Delay(me);
        const size_t slot = static_cast<size_t>(g % me.slots);
        const std::vector<uint8_t>& d = me.out_data[static_cast<size_t>(g)];
        std::memcpy(me.ring.data() + slot * me.piece, d.data(), d.size());
      };
      cb.export_sync = [&](int64_t) {
        Delay(me);
        ++me.exported;
      };
      cb.import_enqueue = [&](int64_t i) {
        const int64_t g = in_base + i;
        if (g == me.fail_import_at) throw std::runtime_error("injected import failure");
        Delay(me);
        const size_t slot = static_cast<size_t>(g % peer.slots);
        const std::vector<uint8_t>& want = peer.out_data[static_cast<size_t>(g)];
        if (std::memcmp(peer.ring.data() + slot * peer.piece, want.data(), want.size()) != 0) ++me.bad;
      };
      cb.import_sync = [&](int64_t i) {
        Delay(me);
        peer.consumed_by_peer[static_cast<size_t>(in_base + i)] = true;
        ++me.imported;
      };
      RunLockstep(out, in, n_out, n_in, out_base, in_base, cb, timeout);
      out_base += n_out;
      in_base += n_in;
    }
    return nullptr;
  } catch (...) {
    const std::exception_ptr e = std::current_exception();
    try {
      std::rethrow_exception(e);
    } catch (const std::exception& x) {
      poison(x.what());
    }
    return e;
  }
}

struct Outcome {
  std::exception_ptr err[2];
  int bad = 0;
  int64_t exported[2] = {0, 0}, imported[2] = {0, 0};
};

Outcome TwoCards(const std::vector<Phase>& phases, size_t piece, uint32_t seed, int delay_us, int64_t fail_export0 = -1, int64_t fail_import1 = -1) {
  int64_t t0 = 0, t1 = 0;
  for (const Phase& p : phases) {
    t0 += p.n0;
    t1 += p.n1;
  }
  Card a, b;
  Prepare(a, 0, t0, piece, seed * 2 + 1, delay_us);
  Prepare(b, 1, t1, piece, seed * 2 + 2, delay_us);
  a.fail_export_at = fail_export0;
  b.fail_import_at = fail_import1;
  RingPipe pipe_a(a.slots), pipe_b(b.slots);  // pipe_a: card a's outgoing = card b's incoming
  const auto poison = [&](const std::string& why) {
    pipe_a.Poison(why);
    pipe_b.Poison(why);
  };
  Outcome o;
  std::thread ta([&] { o.err[0] = RunCard(a, b, pipe_a, pipe_b, phases, milliseconds(4000), poison); });
  std::thread tb([&] { o.err[1] = RunCard(b, a, pipe_b, pipe_a, phases, milliseconds(4000), poison); });
  ta.join();
  tb.join();
  o.bad = a.bad + b.bad;
  o.exported[0] = a.exported;
  o.exported[1] = b.exported;
  o.imported[0] = a.imported;
  o.imported[1] = b.imported;
  return o;
}
Outcome TwoCards(int64_t n0, int64_t n1, size_t piece, uint32_t seed, int delay_us, int64_t fail_export0 = -1, int64_t fail_import1 = -1) {
  return TwoCards(std::vector<Phase>{{n0, n1}}, piece, seed, delay_us, fail_export0, fail_import1);
}

void Lockstep() {
  {  // symmetric, small ring traffic
    const Outcome o = TwoCards(7, 7, 4096, 1, 0);
    Check(!o.err[0] && !o.err[1] && o.bad == 0, "7 + 7 batches: clean, every byte as published");
    Check(o.exported[0] == 7 && o.exported[1] == 7 && o.imported[0] == 7 && o.imported[1] == 7, "every batch crossed in both directions");
  }
  {  // one direction only (a scatter where one card has nothing to send)
    const Outcome o = TwoCards(9, 0, 1024, 2, 0);
    Check(!o.err[0] && !o.err[1] && o.bad == 0 && o.exported[0] == 9 && o.imported[1] == 9, "9 + 0 batches");
    const Outcome p = TwoCards(0, 5, 1024, 3, 0);
    Check(!p.err[0] && !p.err[1] && p.bad == 0 && p.exported[1] == 5 && p.imported[0] == 5, "0 + 5 batches");
    const Outcome z = TwoCards(0, 0, 1024, 4, 0);
    Check(!z.err[0] && !z.err[1], "0 + 0 batches: nothing to do");
  }
  {  // far more batches than slots, with jitter on every copy
    const Outcome o = TwoCards(40, 25, 512, 5, 150);
    Check(!o.err[0] && !o.err[1] && o.bad == 0 && o.exported[0] == 40 && o.imported[1] == 40 && o.exported[1] == 25 && o.imported[0] == 25,
          "40 + 25 batches through 3 slots with jitter: no slot refilled early, no byte lost");
  }
  {  // two phases back to back (gather, then scatter), no barrier between them; the second reuses slots the peer may still be draining
    const Outcome o = TwoCards({{1, 0}, {6, 6}}, 512, 6, 100);
    Check(!o.err[0] && !o.err[1] && o.bad == 0 && o.exported[0] == 7 && o.imported[1] == 7 && o.exported[1] == 6 && o.imported[0] == 6,
          "gather with one batch from card 0 then a scatter: the second phase never overwrites the first phase's last slot early");
    const Outcome p = TwoCards({{0, 0}, {5, 4}}, 512, 7, 50);
    Check(!p.err[0] && !p.err[1] && p.bad == 0, "an empty gather then a scatter");
  }
  {  // the stress: random shapes, random phases, random jitter
    std::mt19937 rng(77);
    int clean = 0;
    for (int it = 0; it < 300; ++it) {
      std::vector<Phase> phases(1 + rng() % 3);
      int64_t t0 = 0, t1 = 0;
      for (Phase& p : phases) {
        p.n0 = static_cast<int64_t>(rng() % 9);
        p.n1 = static_cast<int64_t>(rng() % 9);
        t0 += p.n0;
        t1 += p.n1;
      }
      const Outcome o = TwoCards(phases, 256 + rng() % 1024, 1000 + static_cast<uint32_t>(it), static_cast<int>(rng() % 3) * 40);
      const bool ok = !o.err[0] && !o.err[1] && o.bad == 0 && o.exported[0] == t0 && o.exported[1] == t1 && o.imported[0] == t1 && o.imported[1] == t0;
      if (!ok) {
        std::fprintf(stderr, "  stress %d (%zu phases, totals %lld / %lld) failed (bad %d)\n", it, phases.size(), static_cast<long long>(t0),
                     static_cast<long long>(t1), o.bad);
        Check(false, "stress: random shapes");
      } else {
        ++clean;
      }
    }
    Check(clean == 300, "300 random two-card runs: no deadlock, no corruption");
  }
}

void Failures() {
  {  // card a fails exporting batch 4 of 10: card b must not wait for the rest
    const auto t0 = std::chrono::steady_clock::now();
    const Outcome o = TwoCards(10, 10, 1024, 21, 20, /*fail_export0=*/4);
    const auto ms = std::chrono::duration_cast<milliseconds>(std::chrono::steady_clock::now() - t0).count();
    Check(o.err[0] && o.err[1], "both sides end with an error");
    Check(o.err[0] && Throws<std::runtime_error>([&] { std::rethrow_exception(o.err[0]); }) && !IsPoisonConsequence(o.err[0]), "the failing side reports its own error (a cause)");
    Check(o.err[1] && IsPoisonConsequence(o.err[1]), "the other side sees only the poison (a consequence)");
    Check(ms < 3000, "...and is freed at once, not after its timeout");
  }
  {  // card b fails importing
    const Outcome o = TwoCards(10, 10, 1024, 22, 20, -1, /*fail_import1=*/2);
    Check(o.err[0] && o.err[1], "an import failure ends both sides");
    Check(o.err[1] && !IsPoisonConsequence(o.err[1]) && o.err[0] && IsPoisonConsequence(o.err[0]), "cause on the failing side, consequence on the other");
  }
  {  // a peer that never comes: a bounded wait, not a hang
    RingPipe out(3), in(3);
    LockstepCallbacks cb;
    cb.export_enqueue = [](int64_t) {};
    cb.export_sync = [](int64_t) {};
    cb.import_enqueue = [](int64_t) {};
    cb.import_sync = [](int64_t) {};
    const auto t0 = std::chrono::steady_clock::now();
    Check(Throws<RingTimeout>([&] { RunLockstep(out, in, 0, 2, cb, milliseconds(60)); }), "waiting for a peer that never publishes: RingTimeout");
    Check(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3), "... within the bound");
    // exporting more than the ring holds with nobody consuming: back-pressure times out too
    RingPipe out2(3), in2(3);
    Check(Throws<RingTimeout>([&] { RunLockstep(out2, in2, 5, 0, cb, milliseconds(60)); }), "a ring nobody drains: RingTimeout at batch 3");
  }
}

}  // namespace

int main() {
  PipeBasics();
  PipeWakeups();
  Gate();
  Classify();
  LockstepSteps_();
  Lockstep();
  Failures();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_ring_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_ring_cpu: PASS\n");
  return 0;
}
