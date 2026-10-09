// tests/server/test_batch_port.cpp -- r4dx::server::BatchPort (src/server/batch_port.h) with a real BatchExecutor and a CPU fake TextModel
// that has batch slots (fake_batch_model.h). No GPU work and no tokenizer: each "request" is a thread that makes the calls Engine::RunRequest
// makes, in its order -- Reset, Prefill, then DecodeStepGreedyOverlap / DecodeStepSampled per token -- through its own port.
//
// Checked: a batched run gives every request the token chain a one-at-a-time run gives it; requests really decode together; the model is only
// ever called from the executor thread; the primary lock keeps one request's Reset..BatchImport from interleaving with another's; slots are
// released; the overlap callback's exception is rethrown after the step; sampled rows carry their params and rng; unsupported calls and a
// decode before any prefill are refused; a failing import releases the primary lock (the next request is not blocked).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "batch_executor.h"
#include "batch_port.h"
#include "fake_batch_model.h"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                                                 \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                                 \
      std::fprintf(stderr, "\n");                                                        \
      ++g_failures;                                                                      \
    }                                                                                    \
  } while (0)

using fake_batch::FakeBatchModel;
using r4dx::server::BatchExecutor;
using r4dx::server::BatchPort;

int32_t Argmax(const std::vector<float>& row) {
  return static_cast<int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}

struct Rig {
  explicit Rig(int slots, int64_t slot_ctx = 4096, std::chrono::microseconds gather = std::chrono::milliseconds(200))
      : model(slots, slot_ctx),
        exec(BatchExecutor::Options{slots, gather},
             [this](const std::vector<r4dx::model::BatchDecodeRow>& rows) { return model.DecodeBatch(rows); }) {}
  FakeBatchModel model;
  std::mutex primary;
  BatchExecutor exec;  // after the model: it stops (joins its thread) first
};

// What Engine::RunRequest does for a plain greedy request: prefill, then feed each token and take the next.
std::vector<int32_t> Request(r4dx::model::TextModel& m, const std::vector<int32_t>& prompt, int steps) {
  m.Reset();
  int32_t tok = Argmax(m.Prefill(prompt));
  std::vector<int32_t> out;
  for (int i = 0; i < steps; ++i) {
    out.push_back(tok);
    tok = m.DecodeStepGreedyOverlap(tok, [] { /* decode / stop-string scan / stream the token */ });
  }
  return out;
}

void TestBatchedEqualsAlone() {
  constexpr int kReq = 6, kSlots = 3, kSteps = 20;
  std::vector<std::vector<int32_t>> prompts;
  for (int i = 0; i < kReq; ++i) prompts.push_back({10 + i, 20 + 2 * i, 30 + 3 * i, 40 + i});

  // one at a time, on the plain single-sequence state
  std::vector<std::vector<int32_t>> want;
  {
    FakeBatchModel alone(1, 4096);
    for (const auto& p : prompts) {
      alone.Reset();
      int32_t tok = Argmax(alone.Prefill(p));
      std::vector<int32_t> out;
      for (int i = 0; i < kSteps; ++i) {
        out.push_back(tok);
        tok = alone.DecodeStepGreedy(tok);
      }
      want.push_back(out);
    }
  }

  Rig rig(kSlots);
  std::vector<std::vector<int32_t>> got(kReq);
  // kReq requests over kSlots request threads, each thread serving its share (the Engine's BatchWorkerLoop)
  std::atomic<int> next{0};
  std::vector<std::thread> workers;
  for (int slot = 0; slot < kSlots; ++slot) {
    workers.emplace_back([&, slot] {
      for (;;) {
        const int i = next.fetch_add(1);
        if (i >= kReq) return;
        BatchPort port(rig.model, rig.exec, rig.primary, slot);
        got[static_cast<size_t>(i)] = Request(port, prompts[static_cast<size_t>(i)], kSteps);
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int i = 0; i < kReq; ++i) {
    CHECK(got[static_cast<size_t>(i)] == want[static_cast<size_t>(i)], "request %d: the batched chain differs from the one-at-a-time chain", i);
  }
  CHECK(rig.model.max_rows >= 2 && rig.model.max_rows <= kSlots, "widest step %d rows (want 2..%d): the requests did not decode together",
        rig.model.max_rows, kSlots);
  CHECK(rig.model.batch_rows == static_cast<int64_t>(kReq) * kSteps, "rows decoded: %lld, want %d", static_cast<long long>(rig.model.batch_rows),
        kReq * kSteps);
  CHECK(rig.model.batch_calls < rig.model.batch_rows, "no step carried two rows (%lld calls for %lld rows)",
        static_cast<long long>(rig.model.batch_calls), static_cast<long long>(rig.model.batch_rows));
  CHECK(rig.model.threads.size() == 1 && rig.model.threads.count(std::this_thread::get_id()) == 0,
        "the model was called from %zu threads (one executor thread, none of the test's)", rig.model.threads.size());
  CHECK(rig.model.interleavings == 0, "%d Reset()/Prefill() calls interleaved with another request's prefill..import", rig.model.interleavings);
  CHECK(rig.model.imports == kReq && rig.model.releases == kReq && rig.model.ActiveSlots() == 0,
        "imports %d releases %d active %d", rig.model.imports, rig.model.releases, rig.model.ActiveSlots());
}

// The overlap form: the callback runs while the step is in flight, and its exception is rethrown only after the step finished.
void TestOverlap() {
  Rig rig(2, 4096, std::chrono::microseconds(100));
  rig.model.decode_delay_ms = 60;  // the step takes visible time
  BatchPort port(rig.model, rig.exec, rig.primary, 0);
  port.Reset();
  int32_t tok = Argmax(port.Prefill({1, 2, 3}));
  const auto t0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::duration callback_at{};
  tok = port.DecodeStepGreedyOverlap(tok, [&] { callback_at = std::chrono::steady_clock::now() - t0; });
  const auto total = std::chrono::steady_clock::now() - t0;
  CHECK(callback_at < std::chrono::milliseconds(40), "the callback ran %lld ms in: not while the step was in flight",
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(callback_at).count()));
  CHECK(total >= std::chrono::milliseconds(55), "the call returned before the step finished (%lld ms)",
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(total).count()));

  // a throwing callback: the step is still waited for, the slot has advanced, then the exception arrives
  const int64_t rows_before = rig.model.batch_rows;
  bool threw = false;
  const auto t1 = std::chrono::steady_clock::now();
  try {
    (void)port.DecodeStepGreedyOverlap(tok, [] { throw std::runtime_error("sink failed"); });
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()) == "sink failed";
  }
  const auto waited = std::chrono::steady_clock::now() - t1;
  CHECK(threw, "the callback's exception was not rethrown");
  CHECK(waited >= std::chrono::milliseconds(55), "the exception skipped the wait for the step (%lld ms)",
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(waited).count()));
  CHECK(rig.model.batch_rows == rows_before + 1, "the step did not run");
}

// Sampled rows carry their params and generator; a greedy row has none.
void TestSampledRow() {
  Rig rig(2, 4096, std::chrono::microseconds(100));
  BatchPort port(rig.model, rig.exec, rig.primary, 1);
  port.Reset();
  int32_t tok = Argmax(port.Prefill({5, 6}));
  r4dx::kernels::SampleParams sp;
  sp.temperature = 0.7f;
  sp.top_k = 40;
  std::mt19937_64 rng(7);
  tok = port.DecodeStepSampled(tok, sp, rng);
  tok = port.DecodeStepGreedy(tok);
  (void)tok;
  CHECK(rig.model.calls.size() == 2, "calls: %zu", rig.model.calls.size());
  if (rig.model.calls.size() == 2) {
    const auto& a = rig.model.calls[0][0];
    const auto& b = rig.model.calls[1][0];
    CHECK(a.slot == 1 && a.params.temperature == 0.7f && a.params.top_k == 40 && a.rng == &rng, "the sampled row lost its params or rng");
    CHECK(b.slot == 1 && b.params.temperature <= 0.0f && b.rng == nullptr, "the greedy row is not greedy");
  }
}

void TestRefusals() {
  Rig rig(2);
  BatchPort port(rig.model, rig.exec, rig.primary, 0);
  bool a = false, b = false, c = false, d = false, e = false;
  try { (void)port.DecodeStepGreedy(1); } catch (const std::logic_error&) { a = true; }   // no prefill yet: the slot holds nothing
  CHECK(a, "a decode step before any prefill was accepted");
  port.Reset();
  (void)port.Prefill({1, 2});
  try { (void)port.DecodeStep(1); } catch (const std::logic_error&) { b = true; }
  try { (void)port.DecodeStepMtpGreedy(1, 3); } catch (const std::logic_error&) { c = true; }
  try { port.SaveCheckpoint(); } catch (const std::logic_error&) { d = true; }
  try { (void)port.DecodeStepDflashGreedy(1, 3, 0.0f, 0); } catch (const std::logic_error&) { e = true; }
  CHECK(b && c && d && e, "unsupported calls were not refused (%d %d %d %d)", b, c, d, e);
  CHECK(!port.MtpEnabled() && !port.DflashEnabled(), "a port reports speculation");
  CHECK(port.BatchSlots() == 2 && port.BatchSlotCtx() == 4096, "slots/ctx pass-through");
}

// A prefill whose import fails: the exception reaches the caller, the primary lock is NOT leaked, the slot is not left joined, the next
// request (on another thread) goes through.
void TestImportFailureFreesPrimary() {
  Rig rig(2, 4096, std::chrono::microseconds(100));
  rig.model.fail_import = true;
  {
    BatchPort bad(rig.model, rig.exec, rig.primary, 0);
    bad.Reset();
    bool threw = false;
    try { (void)bad.Prefill({1, 2, 3}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw, "the import failure did not reach the caller");
    // `bad` still holds the primary lock here (its request is failing); destroying it must give it back
  }
  std::vector<int32_t> out;
  std::thread other([&] {
    BatchPort ok(rig.model, rig.exec, rig.primary, 1);
    out = Request(ok, {9, 8, 7}, 4);
  });
  other.join();  // would hang if the lock leaked
  CHECK(out.size() == 4, "the request after a failed import did not run");
  CHECK(rig.model.ActiveSlots() == 0, "slots still active: %d", rig.model.ActiveSlots());
}

// A failing step reaches every row's request, and the ports still release their slots.
void TestStepFailure() {
  Rig rig(2, 4096, std::chrono::milliseconds(300));
  std::atomic<int> failed{0};
  std::vector<std::thread> ts;
  rig.model.fail_batch_after = 1;  // the second step fails
  for (int s = 0; s < 2; ++s) {
    ts.emplace_back([&, s] {
      BatchPort port(rig.model, rig.exec, rig.primary, s);
      try {
        (void)Request(port, {1, 2, 3, s}, 6);
      } catch (const std::runtime_error&) {
        ++failed;
      }
    });
  }
  for (auto& t : ts) t.join();
  CHECK(failed.load() >= 1, "no request saw the failing step");
  CHECK(rig.model.releases == 2 && rig.model.ActiveSlots() == 0, "releases %d active %d", rig.model.releases, rig.model.ActiveSlots());
}

}  // namespace

int main() {
  TestBatchedEqualsAlone();
  TestOverlap();
  TestSampledRow();
  TestRefusals();
  TestImportFailureFreesPrimary();
  TestStepFailure();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all batch port checks passed\n");
  return 0;
}
