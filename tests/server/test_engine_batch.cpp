// tests/server/test_engine_batch.cpp -- `--batch N` serving (docs/batch-decode.md 7) end to end through engine.cpp itself: Engine::
// BatchWorkerLoop's request threads, BatchPort, BatchExecutor, against a CPU fake TextModel that has batch slots. No GPU work (the fake never
// calls HIP; the binary links what r4dx-server links only because engine.cpp does). Needs the real tokenizer directory
// (R4DX_TOKENIZER_MODEL_DIR) and skips (77) without it, like test_engine_recovery.
//
// The fake's "logits" are a pure function of every token a sequence has been fed, in the single-sequence state and in a batch slot alike,
// so a request's text depends on its own tokens only. Checked:
//   1. K concurrent requests (more than the slots) answer exactly the text a one-request-at-a-time engine gives each of them;
//   2. batching really happened (a DecodeBatch call carried more than one row) and the model was only ever called from ONE thread;
//   3. the single-sequence state was never shared: no request's Reset() came between another's Prefill and its BatchImport;
//   4. every slot is released when its request ends; a request longer than --batch-ctx answers 400 without touching a slot;
//   5. a DecodeBatch that fails fails exactly the requests in that step (500), and the engine serves the next ones.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dialect.h"
#include "fake_batch_model.h"  // FakeBatchModel: a CPU TextModel with batch slots (tests/server)
#include "engine.h"
#include "model_config.h"
#include "model_types.h"
#include "openai_types.h"
#include "response_sink.h"
#include "text_model.h"
#include "tokenizer.h"

#include "r4dx/models_root.h"

#ifndef R4DX_TOKENIZER_MODEL_DIR
#define R4DX_TOKENIZER_MODEL_DIR (r4dx::ModelsPath("Huihui-Qwen3.8-27B-abliterated"))
#endif

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                                                    \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s -- ", __FILE__, __LINE__, #cond);    \
      std::fprintf(stderr, __VA_ARGS__);                                                    \
      std::fprintf(stderr, "\n");                                                           \
      ++g_failures;                                                                         \
    }                                                                                       \
  } while (0)

using r4dx::model::BatchDecodeRow;

using fake_batch::FakeBatchModel;

struct Harness {
  std::unique_ptr<r4dx::server::Engine> engine;
  FakeBatchModel* fake = nullptr;
};

Harness MakeEngine(const std::string& tokenizer_dir, int batch_slots, int64_t batch_ctx = 4096) {
  Harness e;
  r4dx::server::EngineOptions opts;
  opts.tokenizer_dir = tokenizer_dir;
  opts.model_opts.max_ctx = 8192;
  opts.model_opts.batch_slots = batch_slots;
  opts.model_opts.batch_ctx = batch_ctx;
  opts.model_opts.prompt_checkpoint = false;
  opts.log_level = "warn";
  opts.max_queue = 32;
  opts.batch_gather = std::chrono::milliseconds(50);  // generous: a step waits for the others' rows, so batches form reliably
  FakeBatchModel** slot = &e.fake;
  opts.model_loader = [slot, batch_slots, batch_ctx](const r4dx::model::ModelOptions&, const r4dx::model::TpOptions&) {
    auto m = std::make_unique<FakeBatchModel>(std::max(batch_slots, 1), batch_ctx);
    *slot = m.get();
    return std::unique_ptr<r4dx::model::TextModel>(std::move(m));
  };
  e.engine = std::make_unique<r4dx::server::Engine>(std::move(opts));
  e.engine->LoadAndStart();
  return e;
}

r4dx::server::ChatMessage Msg(const std::string& role, const std::string& content) {
  r4dx::server::ChatMessage m;
  m.role = role;
  m.content = content;
  return m;
}

std::shared_ptr<r4dx::server::BufferingSink> Submit(Harness& e, const std::string& user, int64_t max_tokens) {
  auto req = std::make_shared<r4dx::server::PendingRequest>();
  req->kind = r4dx::server::RequestKind::kChat;
  req->request_id = r4dx::server::GenerateRequestId("chatcmpl-");
  req->messages = {Msg("user", user)};
  req->sampling.temperature = 0.0f;
  req->max_tokens = max_tokens;
  auto sink = std::make_shared<r4dx::server::BufferingSink>(false, true, &e.engine->Dialect());
  req->sink = sink;
  if (!e.engine->Submit(req)) throw std::runtime_error("queue full");
  return sink;
}

void Scenario(const std::string& tokenizer_dir) {
  const int failures_before = g_failures;
  const std::vector<std::string> prompts = {
      "What is 2 plus 2?", "Name a color.", "Say hello in one short sentence.", "Count to three.",
      "What is the capital of France?", "Write one word.", "Describe the sea.", "Tell me a number."};
  constexpr int64_t kTokens = 12;

  // The reference: one request at a time, no batching.
  std::vector<std::string> want;
  {
    Harness ref = MakeEngine(tokenizer_dir, /*batch_slots=*/0);
    for (const std::string& p : prompts) {
      const auto s = Submit(ref, p, kTokens);
      s->Wait();
      CHECK(!s->errored && s->completion_tokens == kTokens, "reference request failed: %s", s->error_message.c_str());
      want.push_back(s->text);
    }
    ref.engine->Shutdown();
  }

  // 1-4. Eight requests at once through three slots.
  Harness e = MakeEngine(tokenizer_dir, /*batch_slots=*/3);
  std::vector<std::shared_ptr<r4dx::server::BufferingSink>> sinks;
  for (const std::string& p : prompts) sinks.push_back(Submit(e, p, kTokens));
  for (auto& s : sinks) s->Wait();
  for (size_t i = 0; i < prompts.size(); ++i) {
    CHECK(!sinks[i]->errored, "batched request %zu failed: %s", i, sinks[i]->error_message.c_str());
    CHECK(sinks[i]->text == want[i] && sinks[i]->completion_tokens == kTokens,
          "batched request %zu text '%s' differs from the one-at-a-time engine's '%s'", i, sinks[i]->text.c_str(), want[i].c_str());
  }
  CHECK(e.fake->max_rows >= 2, "no DecodeBatch call carried more than one row (max %d): the requests never decoded together", e.fake->max_rows);
  CHECK(e.fake->max_rows <= 3, "a step carried %d rows with 3 slots", e.fake->max_rows);
  CHECK(e.fake->threads.size() == 1, "the model was called from %zu threads", e.fake->threads.size());
  CHECK(e.fake->interleavings == 0, "%d Reset()/Prefill() calls came between another request's Prefill and its BatchImport",
        e.fake->interleavings);
  CHECK(e.fake->imports == static_cast<int>(prompts.size()), "imports: %d, want %zu", e.fake->imports, prompts.size());
  CHECK(e.fake->ActiveSlots() == 0 && e.fake->releases == static_cast<int>(prompts.size()),
        "slots still active after every request ended: %d (releases %d)", e.fake->ActiveSlots(), e.fake->releases);

  // 4. A prompt that does not fit a slot is refused up front, and no slot is touched.
  const int imports_before = e.fake->imports;
  const auto big = Submit(e, std::string(30000, 'a') + " b c d", 4);
  big->Wait();
  CHECK(big->errored && big->error_status == 400, "a prompt longer than --batch-ctx answered errored=%d status=%d",
        static_cast<int>(big->errored), big->error_status);
  CHECK(e.fake->imports == imports_before, "the refused prompt was imported");

  // 5. A failing batched step fails the requests in it; the engine keeps serving.
  e.fake->fail_batch_after = 2;  // the third DecodeBatch call from now
  std::vector<std::shared_ptr<r4dx::server::BufferingSink>> hit;
  for (int i = 0; i < 3; ++i) hit.push_back(Submit(e, prompts[static_cast<size_t>(i)], kTokens));
  int failed = 0;
  for (auto& s : hit) {
    s->Wait();
    if (s->errored) {
      ++failed;
      CHECK(s->error_status == 500, "a failed step answered status %d", s->error_status);
    }
  }
  CHECK(failed >= 1, "the injected DecodeBatch failure failed no request");
  const auto after = Submit(e, prompts[0], kTokens);
  after->Wait();
  CHECK(!after->errored && after->text == want[0], "the request after a failed step: errored=%d '%s'", static_cast<int>(after->errored),
        after->text.c_str());
  CHECK(e.fake->ActiveSlots() == 0, "slots still active after the failure scenario: %d", e.fake->ActiveSlots());

  if (g_failures == failures_before) {
    std::printf("8 requests through 3 slots: texts equal the one-at-a-time engine's, widest step %d rows, %lld steps for %lld rows, one model "
                "thread, no interleaved prefill, slots released; an oversized prompt -> 400; a failed step -> 500 for its rows only\n",
                e.fake->max_rows, static_cast<long long>(e.fake->batch_calls), static_cast<long long>(e.fake->batch_rows));
  }
  e.engine->Shutdown();
}

}  // namespace

int main() {
  const std::string tokenizer_dir = R4DX_TOKENIZER_MODEL_DIR;
  if (!std::filesystem::exists(std::filesystem::path(tokenizer_dir) / "tokenizer.json")) {
    std::fprintf(stderr, "SKIP: %s/tokenizer.json not found (set -DR4DX_TOKENIZER_MODEL_DIR=...)\n", tokenizer_dir.c_str());
    return 77;
  }
  try {
    Scenario(tokenizer_dir);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all engine batch checks passed\n");
  return 0;
}
