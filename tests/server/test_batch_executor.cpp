// tests/server/test_batch_executor.cpp -- pure CPU unit test for r4dx::server::BatchExecutor (src/server/batch_executor.h,
// docs/batch-decode.md 7): rows posted by several threads are decoded together on ONE executor thread, each caller gets its own row's
// token, jobs run on that same thread between steps, a failing step fails exactly its rows, Leave() stops the gather window waiting for a
// finished sequence, Stop() fails what is queued. A fake decode function stands in for Model::DecodeBatch; the threads are real.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "batch_executor.h"

using r4dx::server::BatchExecutor;
using Row = BatchExecutor::Row;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                 \
    }                                                                               \
  } while (0)

// The fake model: the token after `t` fed into slot `s` is f(s, t), the same whatever else is in the batch.
int32_t Next(int slot, int32_t token) { return (token * 7 + slot * 13 + 1) % 1000; }

Row MakeRow(int slot, int32_t token) {
  Row r;
  r.slot = slot;
  r.token = token;
  r.params.temperature = 0.0f;
  return r;
}

struct Recorder {
  std::mutex mu;
  std::vector<std::vector<int>> batches;     // the slots of each decode call, in row order
  std::set<std::thread::id> decode_threads;
  std::chrono::milliseconds delay{0};
  std::atomic<int32_t> fail_on_token{-1};    // a row with this token makes the whole call throw

  BatchExecutor::DecodeFn Fn() {
    return [this](const std::vector<Row>& rows) {
      if (delay.count() > 0) std::this_thread::sleep_for(delay);
      std::vector<int32_t> out;
      std::vector<int> slots;
      for (const Row& r : rows) {
        if (r.token == fail_on_token.load()) throw std::runtime_error("injected decode failure");
        out.push_back(Next(r.slot, r.token));
        slots.push_back(r.slot);
      }
      std::lock_guard<std::mutex> lock(mu);
      batches.push_back(slots);
      decode_threads.insert(std::this_thread::get_id());
      return out;
    };
  }
};

void TestSingleRow() {
  Recorder rec;
  BatchExecutor ex({4, std::chrono::microseconds(100)}, rec.Fn());
  CHECK(ex.Step(MakeRow(2, 5)) == Next(2, 5));
  CHECK(ex.Step(MakeRow(0, 9)) == Next(0, 9));
  const auto st = ex.GetStats();
  CHECK(st.steps == 2 && st.rows == 2 && st.rows_hist[1] == 2);
}

// K sequences decoding in lockstep from K threads: every step carries all K rows (the gather window waits for the stragglers), and every
// thread's token chain equals what it would have produced alone.
void TestLockstepBatching() {
  constexpr int kThreads = 4, kSteps = 25;
  Recorder rec;
  BatchExecutor ex({kThreads, std::chrono::milliseconds(500)}, rec.Fn());
  std::vector<std::vector<int32_t>> chains(kThreads);
  std::vector<std::thread> threads;
  for (int s = 0; s < kThreads; ++s) ex.Join(s);  // all four are about to decode: the first step waits for every row too
  for (int s = 0; s < kThreads; ++s) {
    threads.emplace_back([&, s] {
      int32_t tok = s + 1;
      for (int i = 0; i < kSteps; ++i) {
        tok = ex.Step(MakeRow(s, tok));
        chains[static_cast<size_t>(s)].push_back(tok);
        // a little host work between tokens (stop-string scan, streaming), different per sequence
        std::this_thread::sleep_for(std::chrono::microseconds(100 * (s + 1)));
      }
      ex.Leave(s);
    });
  }
  for (auto& t : threads) t.join();
  for (int s = 0; s < kThreads; ++s) {
    int32_t tok = s + 1;
    for (int i = 0; i < kSteps; ++i) {
      tok = Next(s, tok);
      CHECK(chains[static_cast<size_t>(s)][static_cast<size_t>(i)] == tok);
    }
  }
  const auto st = ex.GetStats();
  CHECK(st.rows == kThreads * kSteps);
  CHECK(st.steps == kSteps);                  // one call per token round: the gather window held every step for all four rows
  CHECK(st.rows_hist[kThreads] == kSteps);
  CHECK(st.failed_steps == 0);
  CHECK(rec.decode_threads.size() == 1);      // one thread owns the model
  CHECK(rec.decode_threads.count(std::this_thread::get_id()) == 0);
}

// A sequence that finished Leave()s: the rest are no longer held for it (with a 5 s window the test would take minutes if they were).
void TestLeaveStopsWaiting() {
  Recorder rec;
  BatchExecutor ex({2, std::chrono::seconds(5)}, rec.Fn());
  std::thread other([&] {
    // posts one row and leaves
    (void)ex.Step(MakeRow(1, 3));
    ex.Leave(1);
  });
  const auto t0 = std::chrono::steady_clock::now();
  int32_t tok = 1;
  for (int i = 0; i < 6; ++i) tok = ex.Step(MakeRow(0, tok));
  other.join();
  const auto waited = std::chrono::steady_clock::now() - t0;
  CHECK(waited < std::chrono::seconds(8));   // at most the first step waited for slot 1; none after it left
  ex.Leave(0);
}

// A step that throws fails exactly its rows; the executor survives and the next step is fine.
void TestFailureIsolated() {
  Recorder rec;
  BatchExecutor ex({2, std::chrono::milliseconds(300)}, rec.Fn());
  rec.fail_on_token = 666;
  ex.Join(0);
  ex.Join(1);
  std::future<int32_t> a = ex.StepAsync(MakeRow(0, 666));
  std::future<int32_t> b = ex.StepAsync(MakeRow(1, 4));  // same step: fails with it
  bool a_threw = false, b_threw = false;
  try { (void)a.get(); } catch (const std::runtime_error&) { a_threw = true; }
  try { (void)b.get(); } catch (const std::runtime_error&) { b_threw = true; }
  CHECK(a_threw && b_threw);
  rec.fail_on_token = -1;
  CHECK(ex.Step(MakeRow(1, 4)) == Next(1, 4));
  const auto st = ex.GetStats();
  CHECK(st.failed_steps == 1 && st.steps == 2);
}

// Jobs run on the executor thread -- the same one that decodes -- and a job waiting while steps flow gets its turn between two of them.
void TestJobsInterleave() {
  Recorder rec;
  rec.delay = std::chrono::milliseconds(2);
  BatchExecutor ex({2, std::chrono::microseconds(200)}, rec.Fn());
  std::atomic<bool> stop{false};
  std::vector<std::thread> decoders;
  for (int s = 0; s < 2; ++s) {
    decoders.emplace_back([&, s] {
      int32_t tok = s;
      while (!stop.load()) tok = ex.Step(MakeRow(s, tok));
      ex.Leave(s);
    });
  }
  std::thread::id job_thread;
  const int answer = ex.Run([&] {
    job_thread = std::this_thread::get_id();
    return 42;
  });
  CHECK(answer == 42);
  bool threw = false;
  try {
    ex.Run([]() -> int { throw std::logic_error("job failed"); });
  } catch (const std::logic_error& e) {
    threw = std::string(e.what()) == "job failed";
  }
  CHECK(threw);
  // jobs from several threads at once all run, one at a time, on the one thread
  std::mutex mu;
  std::set<std::thread::id> ids;
  int running = 0, max_running = 0;
  std::vector<std::thread> clients;
  for (int c = 0; c < 4; ++c) {
    clients.emplace_back([&] {
      ex.Run([&] {
        std::lock_guard<std::mutex> lock(mu);
        ++running;
        max_running = std::max(max_running, running);
        ids.insert(std::this_thread::get_id());
        --running;
      });
    });
  }
  for (auto& t : clients) t.join();
  stop = true;
  for (auto& t : decoders) t.join();
  CHECK(ids.size() == 1 && max_running == 1);
  CHECK(*ids.begin() == job_thread);
  CHECK(rec.decode_threads.size() == 1 && *rec.decode_threads.begin() == job_thread);
  CHECK(ex.GetStats().jobs >= 6);
}

void TestMisuse() {
  Recorder rec;
  rec.delay = std::chrono::milliseconds(150);
  BatchExecutor ex({2, std::chrono::microseconds(100)}, rec.Fn());
  bool threw = false;
  try { (void)ex.StepAsync(MakeRow(2, 1)); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);  // slot out of range
  threw = false;
  try { (void)ex.StepAsync(MakeRow(-1, 1)); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
  // two rows of one slot waiting at once: the second post is a bug in the caller. The first call is in the (slow) decode, so post a
  // second row for slot 0 behind it, then a third.
  std::future<int32_t> first = ex.StepAsync(MakeRow(0, 1));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // the executor took it and sits in the decode
  std::future<int32_t> queued = ex.StepAsync(MakeRow(0, 2));
  threw = false;
  try { (void)ex.StepAsync(MakeRow(0, 3)); } catch (const std::logic_error&) { threw = true; }
  CHECK(threw);
  CHECK(first.get() == Next(0, 1));
  CHECK(queued.get() == Next(0, 2));
}

void TestStop() {
  Recorder rec;
  rec.delay = std::chrono::milliseconds(150);
  auto ex = std::make_unique<BatchExecutor>(BatchExecutor::Options{2, std::chrono::microseconds(100)}, rec.Fn());
  std::future<int32_t> running = ex->StepAsync(MakeRow(0, 1));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));  // in the decode now
  std::future<int32_t> queued = ex->StepAsync(MakeRow(1, 2));  // waits behind it
  ex->Stop();
  CHECK(running.get() == Next(0, 1));  // the step in flight finished
  bool failed = false;
  try { (void)queued.get(); } catch (const std::runtime_error&) { failed = true; }
  CHECK(failed);                       // the queued one was failed, not lost
  bool refused = false;
  try { (void)ex->Step(MakeRow(0, 1)); } catch (const std::runtime_error&) { refused = true; }
  CHECK(refused);
  refused = false;
  try { ex->Run([] { return 1; }); } catch (const std::runtime_error&) { refused = true; }
  CHECK(refused);
  ex->Stop();  // idempotent
  ex.reset();
}

void TestBadConstruction() {
  bool threw = false;
  try { BatchExecutor ex({0, std::chrono::microseconds(1)}, [](const std::vector<Row>&) { return std::vector<int32_t>(); }); }
  catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { BatchExecutor ex({2, std::chrono::microseconds(1)}, nullptr); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

}  // namespace

int main() {
  TestSingleRow();
  TestLockstepBatching();
  TestLeaveStopsWaiting();
  TestFailureIsolated();
  TestJobsInterleave();
  TestMisuse();
  TestStop();
  TestBadConstruction();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all batch executor checks passed\n");
  return 0;
}
