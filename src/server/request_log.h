// r4dx::server::RequestLog -- the optional per-request JSON Lines log (`r4dx-server --request-log
// <path>`, docs/server.md "Request log"). One JSON object per request that reached a verdict, appended
// to a file, so a few days of real client traffic (an editor agent resending a growing conversation)
// can be analysed afterwards: how long each prompt was, how much of it the prefix cache saved, and
// what the prefill and decode cost.
//
// PRIVACY, by construction: RequestLogRecord has no string field that could carry caller text. The
// only strings are the request id (server generated), the endpoint name, the finish reason (one of a
// fixed set) and the speculation mode. Message text, tool definitions and arguments, file contents,
// prompts and completions are never stored here, so they cannot reach the file. The one exception is
// the opt-in RequestLogTokens (`--request-log-tokens`): token ids of the prompt and the completion,
// which are the text in another encoding.
//
// Off (the default) costs nothing: Engine/HttpServer hold a null pointer and every log site is one
// pointer test. On, Write() is one mutex-guarded fwrite + fflush per request and can never throw,
// fail or block a request for longer than that write (a failed write is counted and warned about once
// on stderr). CPU-only and header-light (no json, no HIP) so tests/server/test_request_log.cpp runs
// without a GPU.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace r4dx::server {

// `--request-log-tokens` (docs/server.md "Request log" > "Token capture"): the extra, OPT-IN content an
// offline speculation simulator needs. Unlike the rest of the record this IS derived from caller text
// (token ids detokenize back to it), so it exists only when the flag was given; the file is then as
// private as the prompts themselves.
struct RequestLogTokens {
  // The whole prompt the model saw (chat template applied, image tokens expanded), not only the
  // non-reused suffix. Held in full here; the writer emits only the part past `prompt_shared`.
  std::vector<int32_t> prompt_ids;
  // Leading tokens identical to the previous token-carrying line's prompt_ids + generated_ids.
  // Filled by RequestLog::Write (it owns the previous line); FormatRequestLogLine emits
  // prompt_ids[prompt_shared:]. 0 = the line carries the whole prompt.
  size_t prompt_shared = 0;
  std::vector<int32_t> generated_ids;  // the tokens shown to the client (an EOS is not among them)
  // One entry per speculative verify round (empty at speculative "none"): draft tokens proposed and
  // accepted. A round emits accepted + 1 tokens; the first generated token comes from the prefill's
  // logits and belongs to no round.
  std::vector<int32_t> round_drafted;
  std::vector<int32_t> round_accepted;
  std::optional<int64_t> draft_k;  // the per-round draft cap the server ran with (--dflash-k / --mtp)
  std::optional<double> top_p;
  std::optional<double> min_p;
  std::optional<int64_t> top_k;
  std::optional<uint64_t> seed;  // the request's own seed; unset = a fresh random one
};

// Every field is optional where a request can end before it is known (a 429 never reaches the
// engine; a 400 can fail before the prompt is tokenized): an unset optional is written as JSON null,
// never as a fabricated 0.
struct RequestLogRecord {
  // Local ISO-8601 time the line was written (a request's completion). Empty = RequestLog::Write
  // stamps the current time.
  std::string timestamp;
  std::string request_id;  // the response's `id` ("chatcmpl-..." / "cmpl-..."); empty -> null
  std::string endpoint;    // "chat/completions" | "completions"
  std::optional<bool> stream;

  // The status the HTTP response carried. A streamed request the engine accepted is always 200: its
  // headers are committed before the engine runs, so a failure inside it travels as an SSE error
  // event (see error_status).
  int http_status = 0;
  // The status the engine assigned a request it failed (400 bad prompt, 500 internal). Unset on
  // success and on a request rejected before the engine (those carry their own http_status).
  std::optional<int> error_status;

  std::optional<bool> thinking;       // resolved enable_thinking
  std::optional<int64_t> max_tokens;  // the request's max_tokens (the server default when omitted)
  std::optional<double> temperature;
  std::optional<int64_t> tools_count;  // tool definitions the request offered
  std::optional<int64_t> image_count;  // image content parts in the conversation

  // usage.prompt_tokens: the FULL rendered prompt, image tokens included.
  std::optional<int64_t> prompt_tokens;
  // timings.prompt_n: the tokens actually fed to prefill this request = prompt_tokens - cached_tokens.
  std::optional<int64_t> prompt_n;
  // The leading prompt tokens NOT prefilled again because the model state already held them (the
  // previous request's tokens, or its prompt checkpoint).
  std::optional<int64_t> cached_tokens;
  std::optional<int64_t> completion_tokens;  // usage.completion_tokens
  // usage.completion_tokens_details.reasoning_tokens; unset when thinking was off.
  std::optional<int64_t> reasoning_tokens;
  std::optional<std::string> finish_reason;  // stop | length | tool_calls | cancelled; unset on error

  std::optional<double> queue_wait_ms;  // Submit() until the worker picked the request up
  std::optional<double> prompt_ms;      // timings.prompt_ms: time inside prefill
  std::optional<double> predicted_ms;   // timings.predicted_ms: decode time

  // timings.draft_n / draft_n_accepted: set only when a speculative round ran.
  std::optional<int64_t> draft_n;
  std::optional<int64_t> draft_n_accepted;
  // "none" | "mtp" | "dflash": the server's configured mode. Unset for a request that never reached
  // the engine.
  std::optional<std::string> speculative;

  // The client went away (streamed request: its connection dropped) before generation finished.
  bool cancelled = false;

  // Prefix reuse: true when the new prompt did not extend the model state and Model::Reset() plus a
  // full prefill was taken; checkpoint_restore when the prompt-checkpoint path was taken instead.
  std::optional<bool> full_reset;
  std::optional<bool> checkpoint_restore;
  std::optional<double> reset_ms;
  std::optional<double> restore_ms;

  // timings.image_n / image_ms: images this request encoded itself (reused ones are not counted).
  std::optional<int64_t> image_n;
  std::optional<double> image_ms;

  // Set only under --request-log-tokens, only for a request that got as far as a tokenized prompt.
  // Its keys are appended after the fixed ones above, so a line without it is unchanged.
  std::optional<RequestLogTokens> tokens;
};

// One JSON object on one line, no trailing newline. Field order is fixed (see docs/server.md).
// `*_per_second` are derived exactly like the `timings` object's (0 when the matching ms is 0).
// Never throws for any record.
std::string FormatRequestLogLine(const RequestLogRecord& rec);

// "2026-09-30T14:03:22.123+07:00": the local time with its UTC offset. The pure form below takes the
// broken-down local time, the offset and the milliseconds, so tests can pin it.
std::string FormatIso8601(const std::tm& local, int utc_offset_minutes, int millis);
std::string LocalIso8601Now();

class RequestLog {
 public:
  // Opens `path` for append (created when missing, never truncated, no BOM: the file starts with the
  // first JSON line, every line ends "\n" alone). nullptr and `*error` set when it cannot be opened.
  static std::unique_ptr<RequestLog> Open(const std::string& path, std::string* error);
  // Adopts an already-open stream (owned, closed in the destructor). For tests that need a stream
  // whose writes fail.
  static std::unique_ptr<RequestLog> FromStream(std::FILE* file, std::string path);

  ~RequestLog();
  RequestLog(const RequestLog&) = delete;
  RequestLog& operator=(const RequestLog&) = delete;

  // Appends one line and flushes it. Thread-safe. Never throws; on an I/O failure it counts the
  // failure and warns once on stderr, and the request carries on.
  void Write(RequestLogRecord rec) noexcept;

  const std::string& path() const { return path_; }
  int64_t lines_written() const { return lines_written_.load(); }
  int64_t write_failures() const { return write_failures_.load(); }

 private:
  RequestLog(std::FILE* file, std::string path) : file_(file), path_(std::move(path)) {}

  std::mutex mu_;
  std::FILE* file_;
  std::string path_;
  std::atomic<int64_t> lines_written_{0};
  std::atomic<int64_t> write_failures_{0};
  // prompt_ids + generated_ids of the last token-carrying line written OK (empty = none, or its write
  // failed): what the next token-carrying line's prompt_shared is measured against. Guarded by mu_.
  std::vector<int32_t> last_tokens_;
};

}  // namespace r4dx::server
