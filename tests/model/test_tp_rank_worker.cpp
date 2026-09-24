// tests/model/test_tp_rank_worker.cpp -- CPU-only (threads, no GPU). docs/tp.md 2.2, 2.6, 10.1.
//
// The facade <-> rank-thread hand-off of src/model/tp/tp_rank_worker.h:
//   * 2 workers x 1,000,000 empty commands with WorkerTiming{0 us, 0 us}, so EVERY hand-off in both
//     directions goes through a condvar (the lost-wakeup hazard the publish-under-mutex protocol
//     exists for), in order, with no hang -- guarded by this test's own 120 s alarm;
//   * 10,000 commands with random 0-2 ms of work on one rank, default (spinning) timing;
//   * thread_init runs first, on the worker thread, and every command runs on that same thread;
//   * an exception in a command comes back through TakeError(), once; a thread_init failure fails
//     every command without running it;
//   * the progress watchdog fires only when no heartbeat moves (fake clock), both as a unit and
//     through WaitAllIdle;
//   * a post to a busy worker is refused; the destructor with a busy worker waits for it and never
//     touches the command it is running.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tp/tp_rank_worker.h"

using namespace r4dx::model::tp;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

// The test's own hang alarm: a hand-off bug shows up as a hang, which must fail, not stall ctest.
class Alarm {
 public:
  explicit Alarm(seconds limit)
      : th_([this, limit] {
          std::unique_lock<std::mutex> lk(mu_);
          if (!cv_.wait_for(lk, limit, [&] { return done_; })) {
            std::fprintf(stderr, "FAIL: test_tp_rank_worker did not finish in %lld s (hang)\n",
                         static_cast<long long>(limit.count()));
            std::fflush(stderr);
            std::_Exit(1);
          }
        }) {}
  ~Alarm() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
    }
    cv_.notify_all();
    th_.join();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread th_;  // last: starts after the members it uses exist
};

void BusyFor(microseconds d) {
  const auto end = std::chrono::steady_clock::now() + d;
  while (std::chrono::steady_clock::now() < end) {
  }
}

void TestMillionHandoffs() {
  CompletionGroup group;
  const WorkerTiming timing{microseconds(0), microseconds(0)};  // no spinning anywhere
  std::thread::id init_id[2], run_id[2];
  uint64_t count[2] = {0, 0};   // written only by rank r's thread
  bool in_order[2] = {true, true};
  std::vector<std::unique_ptr<RankWorker>> w;
  for (int r = 0; r < 2; ++r) {
    w.push_back(std::make_unique<RankWorker>(
        r, [&init_id, r] { init_id[r] = std::this_thread::get_id(); }, &group, timing));
  }
  const std::vector<RankWorker*> all = {w[0].get(), w[1].get()};
  ProgressWatchdog wd;
  // No 1 s wake slice: a single lost completion wakeup must hang into the alarm, not cost 1 s.
  const milliseconds wake = std::chrono::minutes(10);

  const uint64_t n = 1000000;
  bool waits_ok = true;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < n && waits_ok; ++i) {
    for (int r = 0; r < 2; ++r) {
      w[r]->Post([&count, &in_order, r, i] {
        if (count[r] != i) in_order[r] = false;
        count[r] = i + 1;
      });
    }
    waits_ok = WaitAllIdle(all, group, timing, wd, wake) == WaitResult::kDone;
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  1,000,000 lockstep commands x 2 workers, condvar only: %.2f s (%.2f us each)\n", s,
              s * 1e6 / static_cast<double>(n));
  Check(waits_ok && count[0] == n && count[1] == n && in_order[0] && in_order[1],
        "2 workers x 1,000,000 empty commands through both condvars: all ran, in order, no hang");
  Check(w[0]->Posted() == n && w[0]->Done() == n && w[1]->Posted() == n && w[1]->Done() == n &&
            w[0]->Idle() && w[1]->Idle(),
        "Posted() == Done() == 1,000,000 on both workers");

  for (int r = 0; r < 2; ++r) {
    w[r]->Post([&run_id, r] { run_id[r] = std::this_thread::get_id(); });
  }
  WaitAllIdle(all, group, timing, wd, wake);
  Check(init_id[0] == run_id[0] && init_id[1] == run_id[1] && run_id[0] != run_id[1] &&
            run_id[0] != std::this_thread::get_id(),
        "thread_init and every command run on the worker's own thread, one thread per rank");
}

void TestRandomWork() {
  CompletionGroup group;
  const WorkerTiming timing;  // production spins: 20 ms rank, 200 us facade
  uint64_t count[2] = {0, 0};
  std::vector<std::unique_ptr<RankWorker>> w;
  for (int r = 0; r < 2; ++r) w.push_back(std::make_unique<RankWorker>(r, nullptr, &group, timing));
  const std::vector<RankWorker*> all = {w[0].get(), w[1].get()};
  ProgressWatchdog wd;
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> work_us(0, 2000);
  bool waits_ok = true;
  const uint64_t n = 10000;
  for (uint64_t i = 0; i < n && waits_ok; ++i) {
    const microseconds d(work_us(rng));
    w[0]->Post([&count, d] {
      BusyFor(d);
      ++count[0];
    });
    w[1]->Post([&count] { ++count[1]; });
    waits_ok = WaitAllIdle(all, group, timing, wd) == WaitResult::kDone;
  }
  Check(waits_ok && count[0] == n && count[1] == n,
        "10,000 commands with random 0-2 ms work on rank 0 (spin-then-block on both sides)");
}

void TestErrors() {
  CompletionGroup group;
  std::vector<std::unique_ptr<RankWorker>> w;
  for (int r = 0; r < 2; ++r) w.push_back(std::make_unique<RankWorker>(r, nullptr, &group));
  const std::vector<RankWorker*> all = {w[0].get(), w[1].get()};
  ProgressWatchdog wd;
  const WorkerTiming timing;

  bool ran1 = false;
  w[0]->Post([] { throw std::runtime_error("boom on rank 0"); });
  w[1]->Post([&ran1] { ran1 = true; });
  WaitAllIdle(all, group, timing, wd);
  std::string what;
  if (std::exception_ptr e = w[0]->TakeError()) {
    try {
      std::rethrow_exception(e);
    } catch (const std::runtime_error& ex) {
      what = ex.what();
    }
  }
  Check(what == "boom on rank 0", "a command's exception comes back through TakeError()");
  Check(!w[0]->TakeError() && !w[1]->TakeError() && ran1,
        "TakeError() hands the error out once; the other rank's command ran and has none");
  bool ok_after = false;
  w[0]->Post([&ok_after] { ok_after = true; });
  WaitAllIdle({w[0].get()}, group, timing, wd);
  Check(ok_after && !w[0]->TakeError(), "the worker keeps running commands after an exception");

  // Post to a busy worker: refused, and the running command is untouched.
  std::atomic<int> runs{0};
  w[1]->Post([&runs] {
    BusyFor(milliseconds(100));
    ++runs;
  });
  bool refused = false;
  try {
    w[1]->Post([&runs] { runs += 100; });
  } catch (const std::logic_error&) {
    refused = true;
  }
  WaitAllIdle({w[1].get()}, group, timing, wd);
  Check(refused && runs == 1, "Post() to a busy worker throws std::logic_error; nothing is lost");

  // A thread_init failure (a failed hipSetDevice in production) fails every command, unrun.
  RankWorker bad(7, [] { throw std::runtime_error("init failed"); }, &group);
  bool ran = false;
  bad.Post([&ran] { ran = true; });
  WaitAllIdle({&bad}, group, timing, wd);
  std::string init_what;
  if (std::exception_ptr e = bad.TakeError()) {
    try {
      std::rethrow_exception(e);
    } catch (const std::runtime_error& ex) {
      init_what = ex.what();
    }
  }
  Check(!ran && init_what == "init failed",
        "a thread_init exception fails the first command (unrun) through TakeError()");
}

void TestWatchdogUnit() {
  using C = ProgressWatchdog::Clock;
  const C::time_point t0{};
  ProgressWatchdog wd(seconds(60));
  wd.Arm({0, 0}, t0);
  Check(!wd.Stalled({0, 0}, t0 + seconds(59)), "watchdog: quiet for 59 s -> no fire");
  Check(wd.Stalled({0, 0}, t0 + seconds(60)), "watchdog: quiet for 60 s -> fires");
  wd.Arm({0, 0}, t0);
  Check(!wd.Stalled({1, 0}, t0 + seconds(30)), "watchdog: a heartbeat moved -> no fire");
  Check(!wd.Stalled({1, 0}, t0 + seconds(89)), "watchdog: the 60 s restart from the last move");
  Check(wd.Stalled({1, 0}, t0 + seconds(90)), "watchdog: 60 s after the last move -> fires");
  wd.Arm({100, 50}, t0);  // a lagging rank catching up while the maximum stands still
  Check(!wd.Stalled({100, 60}, t0 + seconds(59)) && !wd.Stalled({100, 60}, t0 + seconds(118)),
        "watchdog: rank 1 moving 50 -> 60 under rank 0's 100 counts as progress");
  Check(wd.Stalled({100, 60}, t0 + seconds(119)), "watchdog: then 60 s quiet -> fires");

  // Limits past what the clock's nanoseconds can hold (~292 years) must not overflow into "fire".
  const auto years = [](int64_t y) { return std::chrono::hours(24 * 365 * y); };
  ProgressWatchdog never(ProgressWatchdog::kNoStallLimit);
  never.Arm({0, 0}, t0);
  Check(!never.Stalled({0, 0}, t0 + seconds(1)) && !never.Stalled({0, 0}, t0 + years(100)),
        "watchdog: kNoStallLimit never fires (1 s, 100 years quiet)");
  ProgressWatchdog millennium(std::chrono::duration_cast<milliseconds>(years(1000)));
  millennium.Arm({0, 0}, t0);
  Check(!millennium.Stalled({0, 0}, t0 + seconds(1)),
        "watchdog: a 1000-year limit does not fire after 1 s");
}

// WaitAllIdle with a fake clock (10 s per reading): while the command's heartbeat keeps moving, no
// amount of fake time fires the watchdog; once it stops, the watchdog fires within 60 fake
// seconds and leaves the (still busy) worker alone.
void TestWatchdogThroughWait() {
  CompletionGroup group;
  RankWorker w(0, nullptr, &group);
  std::atomic<bool> release{false};
  w.Post([&release] {
    while (!release.load()) std::this_thread::yield();
  });
  ProgressWatchdog wd(seconds(60));
  ProgressWatchdog::Clock::time_point fake{};
  int readings = 0;
  bool bump = true;
  const auto now = [&] {
    fake += seconds(10);
    if (bump) w.Heartbeat().fetch_add(1, std::memory_order_relaxed);  // "progress" between wakes
    if (++readings == 100) bump = false;
    return fake;
  };
  const WaitResult r = WaitAllIdle({&w}, group, WorkerTiming{}, wd, milliseconds(1), now);
  Check(r == WaitResult::kStalled && readings >= 100 && readings <= 110,
        "WaitAllIdle: no fire through 1000 fake seconds of moving heartbeats, fires within 60 "
        "fake seconds of them stopping (" + std::to_string(readings) + " clock readings)");
  Check(!w.Idle(), "a fired watchdog leaves the busy worker untouched");
  release = true;
  ProgressWatchdog wd2;
  Check(WaitAllIdle({&w}, group, WorkerTiming{}, wd2) == WaitResult::kDone && w.Idle(),
        "the worker finishes once released");
}

void TestDestructorWaitsForBusyWorker() {
  CompletionGroup group;
  std::atomic<int> runs{0};
  std::atomic<bool> finished{false};
  auto w = std::make_unique<RankWorker>(0, nullptr, &group);
  // Measured from before the Post: the command starts after it and busies 300 ms on the same
  // clock, so the destructor cannot return earlier however late the threads get scheduled.
  const auto t0 = std::chrono::steady_clock::now();
  w->Post([&] {
    ++runs;
    BusyFor(milliseconds(300));
    finished = true;
  });
  std::this_thread::sleep_for(milliseconds(20));  // the command is (normally) running now
  w.reset();
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Check(finished && runs == 1 && ms >= 300.0,
        "~RankWorker with a busy worker waits for the running command (" + std::to_string(ms) +
            " ms since the post), which completes exactly once");

  auto idle = std::make_unique<RankWorker>(1, nullptr, &group);
  const auto t1 = std::chrono::steady_clock::now();
  idle.reset();
  const double idle_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
  Check(idle_ms < 1000.0, "~RankWorker of an idle (blocked) worker returns promptly (" +
                              std::to_string(idle_ms) + " ms)");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Alarm alarm(seconds(120));
  TestWatchdogUnit();
  TestErrors();
  TestWatchdogThroughWait();
  TestDestructorWaitsForBusyWorker();
  TestRandomWork();
  TestMillionHandoffs();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
