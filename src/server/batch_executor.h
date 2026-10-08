// r4dx::server::BatchExecutor -- the one thread that owns the model in batched serving (docs/batch-decode.md 7) and the rendezvous
// where the decode steps of several requests meet.
//
// `--batch N` runs N request threads at once (Engine::BatchWorkerLoop), each executing the ordinary Engine::RunRequest. None of them
// touches the model: every model call goes through a per-request port (batch_port.h) to this executor, which runs on ONE thread -- the
// facade thread TextModel's contract wants (TpModel and PpModel are single-caller; HIP device binding is per thread). Two kinds of work:
//
//   * jobs (Run): any closure, run on the executor thread, result (or exception) handed back to the caller -- a prompt's Reset +
//     Prefill + BatchImport, an image encode, a slot release.
//   * decode steps (StepAsync / Step): one row (slot, the token to feed, the sampling parameters). The rows that are waiting when the
//     executor looks are decoded TOGETHER, by one `decode` call (Model::DecodeBatch), and every caller gets its own row's token back.
//
// Scheduling. After a step the executor gives one waiting job a turn (so a new request's prefill starts between two tokens of the others,
// not after they finish), then goes back to steps; with no steps waiting it runs jobs back to back. A prefill is therefore a stall for
// the decoding requests: serialized prefill, the cost docs/batch-decode.md 7.3 quantifies. Before a step the executor also waits up to
// `Options::gather` for the sequences that are still between tokens (they decoded, scanned stop strings and streamed the last token and
// are about to post the next row): a step starts as soon as every decoding sequence has posted its row, or when the window closes.
// "Decoding" means: Join()ed (or has posted a row) and not yet Leave()d.
//
// Failure. A throwing `decode` fails every row of that step with the exception; the executor itself keeps running. Stop() fails whatever
// is still queued and joins the thread.
//
// Header-only and HIP-free (the model enters through the `decode` function), so tests/server/test_batch_executor.cpp runs it against a
// fake with real threads.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "batch_row.h"

namespace r4dx::server {

class BatchExecutor {
 public:
  using Row = r4dx::model::BatchDecodeRow;
  using DecodeFn = std::function<std::vector<int32_t>(const std::vector<Row>&)>;

  struct Options {
    int slots = 2;                                // the most rows one step carries, and the slot numbers a row may name: [0, slots)
    std::chrono::microseconds gather{2000};       // how long a step waits for the decoding sequences that have not posted a row yet
  };
  struct Stats {
    int64_t steps = 0;                            // batched decode calls
    int64_t rows = 0;                             // rows decoded, summed
    int64_t jobs = 0;                             // Run() closures executed
    int64_t failed_steps = 0;
    std::vector<int64_t> rows_hist;               // rows_hist[n] = steps that carried n rows (n in [1, slots])
  };

  BatchExecutor(Options options, DecodeFn decode)
      : opts_(options), decode_(std::move(decode)), decoding_(static_cast<size_t>(options.slots), false) {
    if (opts_.slots < 1) throw std::invalid_argument("BatchExecutor: slots must be >= 1");
    if (!decode_) throw std::invalid_argument("BatchExecutor: a decode function is required");
    stats_.rows_hist.assign(static_cast<size_t>(opts_.slots) + 1, 0);
    thread_ = std::thread([this] { Loop(); });
  }
  ~BatchExecutor() { Stop(); }
  BatchExecutor(const BatchExecutor&) = delete;
  BatchExecutor& operator=(const BatchExecutor&) = delete;

  // Runs `f` on the executor thread and returns its result; an exception it throws is rethrown here. Throws std::runtime_error when the
  // executor has been stopped. Must not be called from the executor thread itself (it would wait for itself).
  template <class F>
  auto Run(F&& f) -> std::invoke_result_t<F&> {
    using R = std::invoke_result_t<F&>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> result = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_) throw std::runtime_error("batch executor stopped");
      jobs_.push_back([task] { (*task)(); });
    }
    cv_.notify_all();
    return result.get();
  }

  // Posts `row` to the next step and returns the future of its token. The row's slot must not already have a row waiting (a sequence
  // posts one row, then waits for it). Marks the slot decoding until Leave(slot).
  std::future<int32_t> StepAsync(const Row& row) {
    if (row.slot < 0 || row.slot >= opts_.slots) throw std::invalid_argument("BatchExecutor::StepAsync: slot out of range");
    auto req = std::make_unique<StepReq>();
    req->row = row;
    std::future<int32_t> token = req->token.get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_) throw std::runtime_error("batch executor stopped");
      for (const auto& w : steps_) {
        if (w->row.slot == row.slot) throw std::logic_error("BatchExecutor::StepAsync: this slot already has a row waiting");
      }
      decoding_[static_cast<size_t>(row.slot)] = true;
      steps_.push_back(std::move(req));
    }
    cv_.notify_all();
    return token;
  }
  int32_t Step(const Row& row) { return StepAsync(row).get(); }

  // The sequence in `slot` is about to decode (its prompt is in the slot, its first row is a token away): steps wait for its row from now
  // on. StepAsync does this too; calling it before the first row is what lets the very first step of a new sequence be batched with the
  // others instead of racing them. A joined sequence that never posts (the request ended on its first token) must Leave().
  void Join(int slot) {
    if (slot < 0 || slot >= opts_.slots) throw std::invalid_argument("BatchExecutor::Join: slot out of range");
    std::lock_guard<std::mutex> lock(mu_);
    decoding_[static_cast<size_t>(slot)] = true;
  }

  // The sequence in `slot` will post no more rows (its request finished or failed): a step stops waiting for it.
  void Leave(int slot) {
    if (slot < 0 || slot >= opts_.slots) return;
    {
      std::lock_guard<std::mutex> lock(mu_);
      decoding_[static_cast<size_t>(slot)] = false;
    }
    cv_.notify_all();
  }

  // Fails everything still queued with "batch executor stopped" and joins the thread. Idempotent.
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_ && !thread_.joinable()) return;
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    std::deque<std::function<void()>> jobs;
    std::deque<std::unique_ptr<StepReq>> steps;
    {
      std::lock_guard<std::mutex> lock(mu_);
      jobs.swap(jobs_);
      steps.swap(steps_);
    }
    jobs.clear();  // a queued packaged_task destroyed unrun makes its future throw broken_promise
    for (auto& s : steps) s->token.set_exception(std::make_exception_ptr(std::runtime_error("batch executor stopped")));
  }

  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
  }
  // Rows waiting for the next step / sequences that count as decoding (tests, diagnostics).
  int PendingSteps() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(steps_.size());
  }
  int Decoding() const {
    std::lock_guard<std::mutex> lock(mu_);
    return DecodingLocked();
  }
  int Slots() const { return opts_.slots; }

 private:
  struct StepReq {
    Row row;
    std::promise<int32_t> token;
  };
  using Clock = std::chrono::steady_clock;

  int DecodingLocked() const {
    int n = 0;
    for (bool d : decoding_) n += d ? 1 : 0;
    return n;
  }

  void Loop() {
    std::unique_lock<std::mutex> lock(mu_);
    bool job_turn = false;  // after a step, one waiting job runs before the next step
    for (;;) {
      cv_.wait(lock, [&] { return stop_ || !jobs_.empty() || !steps_.empty(); });
      if (stop_) return;
      if (!jobs_.empty() && (steps_.empty() || job_turn)) {
        std::function<void()> job = std::move(jobs_.front());
        jobs_.pop_front();
        job_turn = false;
        lock.unlock();
        job();  // a packaged_task: its own exception goes to the caller's future
        lock.lock();
        ++stats_.jobs;
        continue;
      }
      // A step: wait (bounded) for the decoding sequences that are still between tokens, unless a job is waiting or the step is full.
      const auto deadline = Clock::now() + opts_.gather;
      while (!stop_ && jobs_.empty() &&
             static_cast<int>(steps_.size()) < std::min(opts_.slots, std::max(1, DecodingLocked()))) {
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) break;
      }
      if (stop_) return;
      if (steps_.empty()) continue;
      std::vector<std::unique_ptr<StepReq>> taken;
      while (!steps_.empty() && static_cast<int>(taken.size()) < opts_.slots) {
        taken.push_back(std::move(steps_.front()));
        steps_.pop_front();
      }
      lock.unlock();
      std::vector<Row> rows;
      rows.reserve(taken.size());
      for (const auto& t : taken) rows.push_back(t->row);
      std::exception_ptr error;
      std::vector<int32_t> out;
      try {
        out = decode_(rows);
        if (out.size() != rows.size()) throw std::logic_error("BatchExecutor: the decode function returned the wrong number of tokens");
      } catch (...) {
        error = std::current_exception();
      }
      for (size_t i = 0; i < taken.size(); ++i) {
        if (error) {
          taken[i]->token.set_exception(error);
        } else {
          taken[i]->token.set_value(out[i]);
        }
      }
      lock.lock();
      ++stats_.steps;
      stats_.rows += static_cast<int64_t>(rows.size());
      if (error) ++stats_.failed_steps;
      ++stats_.rows_hist[rows.size()];
      job_turn = true;
    }
  }

  const Options opts_;
  const DecodeFn decode_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> jobs_;
  std::deque<std::unique_ptr<StepReq>> steps_;
  std::vector<bool> decoding_;
  Stats stats_;
  bool stop_ = false;
  std::thread thread_;  // last: started by the constructor once everything above exists
};

}  // namespace r4dx::server
