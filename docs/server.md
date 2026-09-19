# r4dx-server: OpenAI-compatible chat API

`src/server/` builds `r4dx-server`, an HTTP server exposing a subset of the OpenAI chat/completions
API over `r4dx::model::Model` (single model, single GPU, HIP device 1). See `docs/architecture.md`
for where this sits in the overall module map and `docs/status.md` for milestone history.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status":"ok","model":"<model id>"}` |
| GET | `/v1/models` | OpenAI models-list shape, one entry (the loaded container's `model_id`, from its `__metadata__.model_id`, `docs/container-format.md`) |
| POST | `/v1/chat/completions` | Chat messages through the real `chat_template.jinja`; non-streaming JSON or SSE (`"stream": true`) |
| POST | `/v1/completions` | A raw prompt string, tokenized directly (no chat template) |

### Request fields

`model` (optional, informational only -- the server always answers with whatever container it
loaded), `messages` (chat) / `prompt` (completions), `temperature`, `top_p`, `top_k`, `min_p`,
`max_tokens` (also accepts the newer `max_completion_tokens`, which takes precedence if both are
given), `seed`, `stop` (a string or array of strings), `stream`, `chat_template_kwargs` (an object
passed straight through to `ChatTemplate::render()`'s `extra_context` -- `chat_template.h`
documents `enable_thinking`/`reasoning_effort`/`preserve_thinking`/`add_vision_id` as the fields
Qwen3.8-27B's own template understands), `tools` (passed straight through to the template so it
still renders tool definitions into the prompt text -- see "Tool calls" below).

Message `content` may be a plain string or an OpenAI-style array of parts
(`[{"type":"text","text":"..."}]`); any non-`"text"` part (`image_url`, ...) is rejected with a
`400 invalid_request_error` -- there is no vision tower forward pass yet (`docs/status.md`'s "Known
gaps"). Message `role` must be `system`, `user`, or `assistant`; `tool`/`function` roles are
rejected the same way (see "Tool calls" below).

### Response shapes

Standard OpenAI `chat.completion` / `chat.completion.chunk` / `text_completion` objects, including
`usage.{prompt_tokens,completion_tokens,total_tokens}` and `finish_reason` (`"stop"` -- EOS or a
`stop` string matched; `"length"` -- `max_tokens` reached; `"cancelled"` -- client disconnected
mid-stream). Streaming responses are `Content-Type: text/event-stream`, one `data: <json>\n\n`
event per chunk, terminated by the literal line `data: [DONE]\n\n`.

## Deferred / known gaps

- **Tool calls**: `tools`/`chat_template_kwargs` are accepted and rendered into the prompt (so a
  template-aware client can still get tool definitions in front of the model), but a model-emitted
  tool call comes back as plain `message.content` text -- it is not parsed into a structured
  `tool_calls` response field, and request messages with `role: "tool"` are rejected. Parsing the
  model's tool-call surface syntax back into JSON is future work.
- **Vision**: image content parts are rejected with `400` (see above) -- the vision tower forward
  pass is a separate milestone (`docs/status.md`).
- **Sampling defaults vs. explicit values**: `--default-temperature`/`--default-top-p`/
  `--default-top-k`/`--default-min-p` seed every sampling field a request does not itself set
  (`openai_types.cpp`'s `ParseSampling` starts from the server's defaults and only overwrites a
  field the JSON body actually contains) -- a request that explicitly sends `"temperature": 1.0` is
  correctly distinguished from one that omits `temperature` entirely.
- **`--stop` trimming across token boundaries**: `engine.cpp`'s stop-string check searches a
  bounded lookback window (63 chars) into already-emitted text plus the newest decoded piece, which
  covers stop strings that straddle a token boundary but is not an unbounded search over the whole
  generation -- adequate for the short stop strings (`"</s>"`, `"\n\n"`, ...) OpenAI clients
  typically send.
- **Multi-sequence / batching**: `r4dx::model::Model` is single-sequence (`model.h`'s own SCOPE
  comment); this server serializes all requests through one worker thread that owns the one `Model`
  instance (task design point 2) rather than batching -- see "Concurrency model" below.
- **`stream_options.include_usage`**: accepted in the request body but not acted on -- streaming
  (SSE) responses never carry a final `usage` chunk regardless of this flag (only the non-streaming
  response includes `usage`). Flagged here (review finding, 2026-09-19) rather than left silently
  unsupported.

## Concurrency model

One `Engine` worker thread owns the `Model`/`Tokenizer`/`ChatTemplate` and processes exactly one
request at a time (`src/server/engine.h`/`.cpp`). HTTP handler threads (cpp-httplib's own thread
pool) never touch the model directly -- they build a `PendingRequest`, `Engine::Submit()` it onto a
`BoundedQueue` (`--max-queue`, default 16), and either:

- **non-streaming**: block on a `BufferingSink::Wait()` until the worker finishes, then serialize
  the buffered text as one JSON response;
- **streaming**: hand cpp-httplib's chunked-content-provider callback a `StreamingSink`, which the
  worker pushes formatted SSE events into as tokens are generated (`response_sink.h`/`.cpp`).

`Submit()` returns `false` (the caller answers `429`) once the queue is at `--max-queue` --
concurrent connections beyond that are rejected outright rather than queued indefinitely.

**Cancellation**: a client disconnecting mid-stream is detected by cpp-httplib's `DataSink::write`
failing; the content-provider callback then calls `StreamingSink::Cancel()`, which the worker's
generation loop polls (`ResponseSink::IsCancelled()`) between decode steps and stops on, reporting
`finish_reason: "cancelled"` in its log line. Non-streaming requests have no analogous early-exit
signal (httplib's blocking POST handler has no way to observe a disconnect until it returns).

**Prefix reuse**: mirrors `src/cli/main.cpp`'s `--chat` REPL exactly (task design point 2). The
engine keeps `fed_tokens_`, the token sequence already committed to the model's KV/GDN state. Each
request's prompt (rendered through the chat template for `/v1/chat/completions`, or the raw prompt
string for `/v1/completions`) is re-tokenized in full; if it extends `fed_tokens_` as a prefix, only
the new tail tokens are fed via `Model::Prefill`; otherwise the model is reloaded fresh
(`Model::Load`) and the whole prompt is re-prefilled from scratch. This is exactly what a normal
OpenAI chat client does (resend the whole growing `messages` array each turn), so a multi-turn
conversation against this server reuses the KV cache the same way `--chat` does -- there is no
separate "session id" concept.

**Reset cost**: a mismatched-prefix reset calls `model_.reset()` before constructing the
replacement `Model` (review finding, 2026-09-19 -- an earlier version assigned the new `Model`
directly, which briefly held two complete containers' worth of VRAM at once: the new `Load()` fully
constructs every weight `DeviceBuffer` before the assignment destroys the old `unique_ptr`, ~2x
peak VRAM for the duration of the reload). This still pays a full container reload's latency on
every non-extending request (measured ~18.6s against the real 64-layer w4a16 container) -- it
fires on the first request of every new conversation once any request has been served. A cheaper
fix (a `Model::Reset()` that re-zeroes state without re-reading weights from disk) is future work;
`model_.reset()` only fixes the VRAM half.
An exception mid-request (`Model::Prefill`/`DecodeStep` throwing partway through) also clears
`fed_tokens_` in the `catch` block, forcing the next request down this same full-reset path rather
than risking a stale prefix match against a model whose real state has silently diverged.

### `Model` adapter note (for the Integrate stage)

Per this task's own instructions, `src/server` does not modify `src/model`. "Reset" (the mismatched
-prefix fallback above) is implemented as a full `Model::Load(opts_.model_opts)` call, exactly like
`src/cli/main.cpp`'s own fallback -- no new `Model` method was needed. No other adapter was
required: `Model::Prefill`/`DecodeStep`/`Config()`/`GetContainer()` (for `ModelId()`) were
sufficient as-is.

## CLI flags

```
r4dx-server --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16}
    [--tokenizer-dir <dir>] [--host <addr>] [--port N] [--max-ctx N]
    [--max-tokens-default N] [--max-queue N] [--think {on|off}] [--layers N]
    [--default-temperature F] [--default-top-p F] [--default-top-k N]
    [--default-min-p F] [--log-level {debug|info|warn|error}]
```

`--tokenizer-dir` defaults to `C:\AI\models\Qwen3.8-27B`, same as `r4dx-cli`. `--think` sets the
server-wide default for the chat template's `enable_thinking` when a request's
`chat_template_kwargs` does not itself set it. `--layers N` loads only the first `N` layers
(`r4dx::model::ModelOptions::layer_limit`) -- required for a test container that physically carries
fewer layers than its (verbatim-copied) `config.json` declares, e.g.
`D:\models\r4dx\qwen38-27b-l4-bf16.r4dx`; omit it (or pass `-1`) for a real, full-size container.
`--log-level` gates the one-line-per-request log (`debug` also gets extra detail; `info` -- the
default -- prints exactly the required prompt/generated/prefill/decode-tok/s line; `warn`/`error`
quiet it down).

## Testing

`tests/server/**` (CPU-only, no HIP device, registered in `ctest`): `test_server_args` (CLI
parsing), `test_openai_types` (request validation + response JSON shapes), `test_sse` (SSE chunk
formatting), `test_response_sink` (`BufferingSink`/`StreamingSink`), `test_request_queue`
(`BoundedQueue` capacity/FIFO/close/threaded producer-consumer). All pass as part of the normal
`.\tests\run_tests.ps1` run (29/29 total as of this stage, the 24 pre-existing plus these 5).

`tools/server/smoke.ps1` is the GPU integration test: starts `r4dx-server` on HIP device 1 against
the 4-layer test container (`--layout w4a16 --layers 4`, since that container's config.json still
declares 64 layers -- see `--layers` above), hits `/v1/models`, a non-streaming and a streaming
`/v1/chat/completions`, and a rejected-image-part request, checking JSON/SSE shapes and status
codes (the 4-layer model's text is nonsense, so only shapes/counts are checked, never the text
itself). Run it with `.\tools\server\smoke.ps1`; pass `-Model`/`-Layout`/`-Layers -1` to point it at
a real container instead.

### Real-answer smoke run (once, against the full 64-layer container)

Run 2026-09-19 against `D:\models\r4dx\qwen38-27b.r4dx` (`--layout w4a16`, the same prompt/settings
`docs/perf.md`'s own CLI table uses, `temperature: 0`, `max_tokens: 128`, thinking off): the smoke
script's shape checks all passed (`tools/server/smoke.ps1 -Model D:\models\r4dx\qwen38-27b.r4dx
-Layout w4a16 -Layers -1`), and a follow-up manual streaming request against the same server
produced this verbatim streamed answer (`POST /v1/chat/completions`, `"stream": true`, content
deltas concatenated):

```
Silicon threads weave,
Parallel light in the dark,
Pixels bloom anew.

A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer for output to a display. Unlike a CPU, which is optimized for complex, sequential tasks, a GPU is built to handle thousands of simple calculations simultaneously, making it ideal for rendering graphics and modern machine learning workloads.
```

The response's `finish_reason` was `"stop"` (real EOS, not a length cutoff), `usage.prompt_tokens =
29`, `usage.completion_tokens = 91`. The server's one-line request log:

```
[r4dx-server][info] request chatcmpl-a1beaf9674b27afd: prompt=29 new=29 generated=91 finish=stop prefill=400.94 tok/s decode=32.30 tok/s
```

-- prefill/decode throughput matches `docs/perf.md`'s CLI w4a16 numbers for the same prompt within
normal run-to-run noise (CLI: 416.74 prefill / 29.04 decode tok/s), confirming the server drives the
same `Model::Prefill`/`DecodeStep` path with the same per-token cost. The haiku and the first
sentence of the explanation are byte-identical to `docs/perf.md`'s own w4a16 CLI output; the tail of
the second sentence differs slightly token-for-token from that (separately captured, days-earlier)
CLI run despite both being `temperature: 0` greedy decodes -- consistent with ordinary GPU
kernel-level nondeterminism (reduction/atomic order) at quantized precision, not a server bug (the
non-streaming `/v1/chat/completions` response for the identical request in this same run was itself
byte-identical to the streamed reconstruction above, confirming the server's own streaming and
non-streaming paths agree).

An equivalent non-streaming request in the same run:

```json
{
  "id": "chatcmpl-5f77e9bb54cc14ef",
  "object": "chat.completion",
  "created": 1789827360,
  "model": "Qwen/Qwen3.8-27B",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Silicon threads weave,\nParallel light in the dark,\nPixels bloom anew.\n\nA GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer for output to a display. Unlike a CPU, which is optimized for complex, sequential tasks, a GPU is built to handle thousands of simple calculations simultaneously, making it ideal for rendering graphics and modern machine learning workloads." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 29, "completion_tokens": 91, "total_tokens": 120 }
}
```

## A bug found and fixed while building this

**cpp-httplib's chunked-content-provider "end of stream" contract**: returning `false` from a
`set_chunked_content_provider` callback is `detail::write_content_chunked`'s *cancellation* signal
(`Error::Canceled`) -- it skips the terminating `"0\r\n\r\n"` chunk entirely, so every HTTP/1.1
client (curl, browsers, PowerShell's `Invoke-WebRequest`) correctly reports the connection as
closed/truncated mid-transfer. The first version of `ServeStream` (`http_server.cpp`) returned
`false` once `StreamingSink::Next()` ran dry, exactly this bug -- streaming responses truncated
before `[DONE]` on every client tested. Fixed by calling `httplib::DataSink::done()` (writes the
terminator) and returning `true` instead. Caught by `tools/server/smoke.ps1` itself (the streaming
checks failed against a real running server even though every CPU-only unit test passed, since none
of them exercise real cpp-httplib chunked-transfer wire behavior) -- see `docs/validation.md`'s
rung 5 ("generation sanity ... catches integration bugs that per-tensor/per-component diffs can
miss") for why this class of bug needs exactly this kind of end-to-end run.
