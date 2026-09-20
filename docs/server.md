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
Qwen3.8-27B's own template understands), `tools` (rendered into the prompt AND parsed back out of
the generation into a structured `message.tool_calls` response field -- see "Tool calls" below),
`tool_choice` (`"none"`/`"auto"`/`"required"`/`{"type":"function","function":{"name":...}}`, see
"Tool calls").

Message `content` may be a plain string or an OpenAI-style array of parts
(`[{"type":"text","text":"..."}]`); any non-`"text"` part (`image_url`, ...) is rejected with a
`400 invalid_request_error` -- there is no vision tower forward pass yet (`docs/status.md`'s "Known
gaps"). Message `role` must be `system`, `user`, `assistant`, `tool`, or `function` -- the latter two
carry a tool result back to the model (see "Tool calls" below).

### Response shapes

Standard OpenAI `chat.completion` / `chat.completion.chunk` / `text_completion` objects, including
`usage.{prompt_tokens,completion_tokens,total_tokens}` and `finish_reason` (`"stop"` -- EOS or a
`stop` string matched; `"length"` -- `max_tokens` reached; `"cancelled"` -- client disconnected
mid-stream). Streaming responses are `Content-Type: text/event-stream`, one `data: <json>\n\n`
event per chunk, terminated by the literal line `data: [DONE]\n\n`.

## Tool calls

`tools`/`chat_template_kwargs` are rendered into the prompt by `chat_template.jinja`
(`src/server/engine.cpp`), and a model-emitted call is parsed back out of the generation into a
structured OpenAI `message.tool_calls` field by `src/server/tool_call_parser.h`/`.cpp`. Full
call/result/answer multi-turn round trips are supported (`role: "tool"`/`"function"` request
messages), and this composes with prefix reuse (below) -- a multi-turn tool conversation does not
desync the KV prefix, since the tool-carrying turns are just more messages re-rendered through the
same `PrefixState::Extend` prefix-match logic every other request uses.

**`role: "function"` correction (review finding, 2026-09-20):** `C:\AI\models\Qwen3.8-27B\
chat_template.jinja` has no `"function"` branch at all -- only `system`/`user`/`assistant`/`tool`
-- so `Engine::RunRequest` remaps a `role: "function"` message to `"tool"` before rendering (the
template's `tool` branch only ever reads `content`, never `tool_call_id`, so the remap is exact;
`name` carries no template effect either way). Previously an unmapped `"function"` role hit the
template's own `raise_exception('Unexpected message role.')` and the request 500'd. Separately, a
`messages` array containing no `role: "user"` turn anywhere (e.g. only `role: "tool"`/`"function"`
turns) hits the template's own `raise_exception('No user query found in messages.')`; this and any
other `ChatTemplate::render()` failure are now reported as `400 invalid_request_error` (a
caller-shape problem) rather than the engine's generic `500`.

**Confirmed surface syntax** (learned from `C:\AI\models\Qwen3.8-27B\chat_template.jinja` and
verified by driving the real 64-layer container through `r4dx-server` with a real tool definition,
both streaming and non-streaming -- not guessed; `tool_call_parser.h`'s file comment has the full
derivation). This is NOT the generic `<tool_call>{"name":...,"arguments":{...}}</tool_call>`
JSON-body shape some other Qwen checkpoints use -- this checkpoint's real shape is XML-ish and
per-parameter, no JSON envelope at the outer level:

```
<tool_call>
<function=NAME>
<parameter=PARAM_NAME>
VALUE
</parameter>
...
</function>
</tool_call>
```

Multiple calls in one turn are simply concatenated `<tool_call>...</tool_call>` blocks back to
back (confirmed with a real two-city weather prompt). A `<think>...</think>` reasoning block, when
present, precedes the first `<tool_call>` and is left untouched in `content` (no
`reasoning_content` extraction -- out of scope, flagged as a follow-up). A parameter `VALUE` is
plain text, not JSON-tagged; the parser tries to `JSON::parse` the trimmed value and falls back to
a JSON string when that fails (the same convention the template's own reverse-rendering direction
uses), so `42`/`true`/`[1,2]`/`{"a":1}`/`null` become their real JSON types and anything else
(`"Boston, MA"`) becomes a JSON string.

**Detokenization**: verified safe on both the streaming and non-streaming paths before this parser
was written -- `<tool_call>`/`</tool_call>` are added-but-not-special tokens
(`src/tokenizer/tokenizer.h`), so `skip_special_tokens=true` (this server's default, `engine.cpp`)
never strips them; `<function=...>`/`<parameter=...>` are not tokens at all (dynamic names), so
they arrive as ordinary detokenized text. The parser only ever runs on the fully-detokenized
string, never assumes a tag aligns to a token or decoder "piece".

**Response shape**: `message.tool_calls = [{id, type:"function", function:{name, arguments}}]`,
where `arguments` is a JSON-ENCODED STRING (OpenAI's real wire shape, not a JSON object --
`tool_call_parser.h`'s own "this trips people up" note). `id` is server-generated
(`GenerateRequestId("call_")`) since the model's own syntax carries no id. `message.content` is
`null` when the turn was a pure call with no prose, or the leading/trailing prose otherwise.
`finish_reason` is `"tool_calls"` instead of `"stop"` whenever at least one call was parsed
(unless the client disconnected mid-stream, in which case `"cancelled"` still wins).

**tool_choice**: `"none"` clears `tools` before rendering (the model is never told about any tool,
so it structurally cannot call one); `"auto"`/`"required"` leave `tools` untouched (`"required"` is
accepted but NOT enforced by constrained decoding -- a `required` request whose generation happens
not to contain a call is not itself flagged as an error); a named
`{"type":"function","function":{"name":...}}` choice filters `tools` down to just that one entry
before rendering. Any other `tool_choice` shape (an unrecognized string, a malformed object, a name
not present in `tools`) is rejected with a clear `400 invalid_request_error` rather than silently
ignored (`openai_types.cpp`'s `ParseToolChoice`).

**Streaming decision**: OpenAI's own server streams tool-call arguments as incremental
`delta.tool_calls[].function.arguments` byte deltas. This server does NOT do that: whenever a
request offers any `tools`, the ENTIRE generation is buffered (not streamed token-by-token) and,
once complete, delivered as ordinary `content` deltas' worth of prose (if any) followed by one
single complete `delta.tool_calls` chunk carrying every call at once (`response_sink.cpp`'s
`StreamingSink::OnToolCalls`, `index`-tagged so a client expecting real per-delta streaming still
assembles the array correctly). This is a deliberate choice, not an oversight: it guarantees a
client can never observe a half-formed `<tool_call>`/`<function=...>` tag leak into a `content`
delta (the failure mode a real per-token streaming parser would risk), at the honest cost of
"fake" (all-at-once) streaming latency for any request that offers tools, whether or not a call
actually happens. A request with no `tools` is completely unaffected -- real per-token streaming,
unchanged (`engine.cpp`'s `tool_mode` gate).

**Robustness**: `ParseToolCalls` never throws. A parameter value that fails to parse as JSON just
becomes a JSON string (not an error). A structurally malformed `<tool_call>` span (missing/
unclosed `<function=...>`/`</function>`/`<parameter=...>`, stray prose where only tag structure is
expected, an unclosed `<tool_call>` itself) degrades to literal content -- its own raw source text,
tags included -- rather than being dropped or crashing the worker thread; `had_malformed_call` is
still set so a caller can log it, but nothing is surfaced to the client as an error. A call naming
a tool the request never defined is dropped from `tool_calls` and replaced with a short synthetic
note in `content` naming the undefined function and its arguments (`DropUnknownToolCalls`) rather
than silently vanishing. `tests/server/test_tool_call_parser.cpp` covers every case above, including
three verbatim captures from the real container (one call, two calls in one turn, and a call
preceded by a `<think>` block) plus a dozen synthetic malformed-input cases.

**Testing**: `tests/server/test_tool_call_parser.cpp` (parser unit cases), the tool-call-specific
cases in `tests/server/test_openai_types.cpp` (request-side `tool_calls`/`tool_choice`
parsing/validation, `role: "tool"`/`"function"` messages) and `tests/server/test_response_sink.cpp`
(the streaming `OnToolCalls` SSE chunk shape), and `tools/server/smoke.ps1`'s real end-to-end tool
round trip against a container (a tool-offering request, feeding the parsed call's result back as a
`role: "tool"` message, checking the final answer references the tool result).

## Milestone 4 integration (2026-09-20)

`tools/server/smoke.ps1` re-run three ways against a clean `build.ps1 -Clean` rebuild, HIP device 1:
default 4-layer container **28/28**, 4-layer MTP container (`-Mtp 3`) **29/29** (MTP path
confirmed taken), real 64-layer container (`-Layers -1 -ToolRoundTrip`) **34/34**, including a real
tool call/result/answer round trip and the REVIEW/FIX pass's regression checks (`role:"function"`
now renders instead of 500, a no-user-turn conversation now 400s instead of 500, `tools`+`stop`
no longer leaks the stop string into `message.content`). See `docs/status.md`'s "Milestone 4: done"
section for the full integration accounting.

## Deferred / known gaps

- **Vision**: image content parts are rejected with `400` (see above) -- the vision tower forward
  pass is a separate milestone (`docs/status.md`); its architecture and preprocessing are now fully
  documented (`docs/vision.md`) with real-hardware validation goldens, but no `src/model` C++ exists
  yet, so this `400` remains accurate.
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

**Reset cost (server-catches-up-with-engine stage)**: a mismatched-prefix reset now calls
`model_->Reset()` (`r4dx::model::Model::Reset()`, `src/model/model.h`/`.cpp`) instead of a full
`Model::Load()`. `Reset()` re-zeroes every piece of per-sequence state (GDN recurrent/conv state
via `GdnStateManager::ZeroAll`, `pos_`/`started_`, MTP head bookkeeping) WITHOUT touching any
weight `DeviceBuffer` or re-reading the container from disk -- see `model.h`'s `Reset()` doc
comment for why the KV caches (backbone and MTP's own) need no explicit clearing (contiguous
slot==position addressing, self-correcting via overwrite-before-read, same property
`docs/mtp.md`'s rejected-candidate handling already relies on). Measured (this stage, real 64-layer
w4a16 container, `HIP_VISIBLE_DEVICES=1`): **Reset() cost is sub-millisecond to a few
milliseconds**, vs. the ~18.6s full-reload `model_.reset()`/fresh-`Load()` path this replaces --
see the per-request log line's new `reset=X.XXms` field (`src/server/engine.cpp`) for the number
from a live run. This also removes the earlier 2x-peak-VRAM-during-reload hazard entirely (no new
`Model`/`Container` is ever constructed after startup, so there is nothing for the old
`model_.reset()`-before-`Load()` ordering fix to protect against).

The prefix-reuse bookkeeping itself moved into its own header, `src/server/prefix_state.h`'s
`PrefixState` (pure `std::vector<int32_t>` arithmetic, no HIP/`r4dx::model` dependency -- unit-
tested in `tests/server/test_prefix_state.cpp`, since `Engine` itself cannot be CPU-tested). Its
`Commit()` tracks **committed** tokens, not merely **displayed** ones -- see the MTP section below
and `prefix_state.h`'s own file comment for the mid-round-stop edge case this distinction exists to
handle. An exception mid-request (`Model::Prefill`/`DecodeStep*`/`Reset` throwing partway through)
calls `prefix_.Invalidate()` in the `catch` block, forcing the next request down the full-reset
path rather than risking a stale prefix match against a model whose real state has silently
diverged (unchanged in spirit from the earlier `fed_tokens_.clear()`, just renamed).

### `Model` adapter note

The previous stage's note ("`src/server` does not modify `src/model`") no longer applies: this
stage's task item 1 explicitly asked for a real `Model::Reset()` method (the "cheap reset" above),
so `src/model/model.h`/`.cpp` gained one. `Model::Prefill`/`DecodeStep`/`DecodeStepMtpGreedy`/
`MtpEnabled`/`Config()`/`GetContainer()` needed no further changes for the server to drive MTP --
see "MTP" below.

## MTP (server-catches-up-with-engine stage)

`--mtp N` (`server_args.h`, default 0, same semantics as `src/cli/cli_args.h`'s own `--mtp`) sets
`ModelOptions::mtp_draft_k` for the one `Model` this server process loads at startup -- `N>0`
requires `--model` to point at an MTP-converted container (`Container::HasMtp()`, `docs/mtp.md`),
exactly like `r4dx-cli`.

Per-request routing (`Engine::RunRequest`, `engine.cpp`): a request takes the MTP path
(`Model::DecodeStepMtpGreedy`) iff `model_->MtpEnabled()` (the server was started with `--mtp N>0`
against an MTP container) AND that specific request's own `sampling.temperature <= 0` (greedy) --
every other request (non-greedy, or MTP disabled server-wide) takes the pre-existing plain-decode
loop (`Model::DecodeStep`/`r4dx::kernels::Sample`), unchanged. This mirrors `r4dx-cli`'s own
`greedy && args.mtp > 0` gate exactly, just evaluated per-request instead of once at process
start-up -- MTP has no notion of probabilistic (non-greedy) acceptance yet (`docs/mtp.md`'s "Only
greedy acceptance is implemented"), so a non-greedy request simply never sees it.

**Streaming**: every token an MTP round actually confirms is pushed to the client as soon as it is
committed -- the round's own per-candidate loop calls the same stop-string-aware
`Engine::EmitToken` helper the plain-decode loop uses, once per accepted token, in order, exactly
like the plain path (task's own "streaming must emit every accepted token of a round as it is
committed"). Cancellation (`StreamingSink::Cancel()`/`IsCancelled()`) is checked once per ROUND
(between `DecodeStepMtpGreedy` calls), not per token within a round -- a round is one atomic GPU
call, so there is no earlier point at which the loop could observe cancellation mid-round.

**Mid-round `fed`/`committed` bookkeeping fix** (`docs/mtp.md`'s "mid-round" gap, closed by this
stage in both binaries): `Model::DecodeStepMtpGreedy` commits every element of its returned round
except the LAST one atomically, regardless of where a caller's own display loop decides to stop
(`--max-tokens` reached, or an EOS candidate that isn't the round's own last element). Both
`src/cli/main.cpp`'s `TurnResult::committed_tokens` and `PrefixState::Commit()`'s
`committed_tokens` argument now track this correctly -- see either file's own comment for the full
derivation, and `tests/server/test_prefix_state.cpp`'s
`TestCommitTracksCommittedNotDisplayedTokens` for a CPU-testable regression check of the
bookkeeping contract (not the real MTP round mechanics themselves, which `tests/model/test_mtp.cpp`
already covers against a real container).

**Correction (Milestone 3 integration pass, 2026-09-20)**: this section originally said
`--mtp-head-layout` "does not correspond to any existing concept in this codebase" -- that was true
only against the pre-merge "server catches up with the engine" tree in isolation. The parallel
"MTP quality" stage (merged in the same milestone) added exactly that knob at the `src/model`
level: `ModelOptions::mtp_head_layout` (`std::optional<Layout>`, default `std::nullopt` = track the
body `--layout`) and `Container::Load`'s own `mtp_head_layout` parameter, which the measured
head-layout sweep (`docs/mtp.md`'s "MTP head layout") showed wins on speed in 23/24 configurations
with no acceptance cost -- see `docs/status.md`'s "Milestone 3 merge note" for the full correction.
`src/cli/cli_args.h` exposes it as `--mtp-head-layout {bf16,layout}`.

**RESOLVED (Milestone 4 follow-up, 2026-09-20)**: `src/server/server_args.h` now exposes the same
`--mtp-head-layout {bf16,layout}` flag (default `layout`, matching the CLI and the measured-winning
configuration), wired to `ModelOptions::mtp_head_layout` in `src/server/main.cpp` identically to the
CLI's own conversion. Covered by `tests/server/test_server_args.cpp`'s `TestMtpHeadLayoutFlag`
(defaults, `bf16` override, and rejection of an unrecognized value).

**New (docs/r9700.md R9, "reduced-vocab draft head")**: `--mtp-draft-head {reduced,full}` (default
`reduced`), mirroring `src/cli/cli_args.h`'s own flag, wired to `ModelOptions::
mtp_draft_reduced_vocab` in `src/server/main.cpp` identically to the CLI. `reduced` uses the
container's OPTIONAL `mtp.draft_head.*` tensors (docs/container-format.md) to speed up drafting when
present, degrading to the exact pre-R9 full-vocab behavior automatically when absent; `full` forces
the full-vocab head unconditionally. Verification always stays full-vocab regardless of this flag,
so it can only affect drafting speed/acceptance, never generated output (see docs/mtp.md's
"Reduced-vocab draft head" section for the full argument and the measured K-sweep). Covered by
`tests/server/test_server_args.cpp`'s `TestMtpDraftHeadFlag`.

## CLI flags

```
r4dx-server --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16}
    [--tokenizer-dir <dir>] [--host <addr>] [--port N] [--max-ctx N]
    [--max-tokens-default N] [--max-queue N] [--think {on|off}] [--layers N]
    [--default-temperature F] [--default-top-p F] [--default-top-k N]
    [--default-min-p F] [--log-level {debug|info|warn|error}] [--mtp N]
    [--mtp-head-layout {bf16|layout}] [--mtp-draft-head {reduced|full}]
    [--embed-device-resident {on|off}]
```

`--mtp N` (default 0): see "MTP" above -- requires an MTP-converted `--model` container when N>0.

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
parsing), `test_openai_types` (request validation + response JSON shapes, including `tools`/
`tool_choice`/`role: "tool"`/`"function"` parsing), `test_sse` (SSE chunk formatting),
`test_response_sink` (`BufferingSink`/`StreamingSink`, including the tool-calls streaming chunk
shape), `test_request_queue` (`BoundedQueue` capacity/FIFO/close/threaded producer-consumer),
`test_prefix_state` (`PrefixState`'s prefix-match / invalidate / MTP-aware commit bookkeeping, see
"MTP" above), `test_tool_call_parser` (see "Tool calls" above -- real-capture and malformed-input
cases for the model's surface syntax). All pass as part of the normal `.\tests\run_tests.ps1` run.

`tools/server/smoke.ps1` is the GPU integration test: starts `r4dx-server` on HIP device 1 against
the 4-layer test container (`--layout w4a16 --layers 4`, since that container's config.json still
declares 64 layers -- see `--layers` above), hits `/v1/models`, a non-streaming and a streaming
`/v1/chat/completions`, two consecutive different-prompt requests (checking the server's own stderr
log to confirm no container reload happened -- see "Reset cost" above), a rejected-image-part
request, and (a `-ToolRoundTrip` switch) a real tool call/result/answer multi-turn round trip
(request offers a tool definition, the server's parsed `message.tool_calls` is fed back as a
`role: "tool"` follow-up message, checking the server accepts it and answers), checking JSON/SSE
shapes and status codes (the 4-layer model's text is nonsense, so only shapes/counts are checked,
never the text itself, except the tool-round-trip check, which needs a real container to exercise
meaningfully -- see below). Run it with `.\tools\server\smoke.ps1`; pass `-Model`/`-Layout`/
`-Layers -1` to point it at a real container instead, `-Mtp N` to exercise the MTP path against an
MTP-converted container (`.\tools\server\smoke.ps1 -Model D:\models\r4dx\qwen38-27b-l4-mtp.r4dx
-Layout w4a16 -Mtp 3`, or `-Model D:\models\r4dx\qwen38-27b.r4dx -Layers -1 -Mtp 3` against the real
container) -- with `-Mtp N>0` an extra check confirms at least one request's log line shows the MTP
path was taken -- and `-ToolRoundTrip` for the tool round-trip check (`.\tools\server\smoke.ps1
-Model D:\models\r4dx\qwen38-27b.r4dx -Layers -1 -ToolRoundTrip`; skipped by default against the
4-layer container, whose nonsense output cannot reliably be coaxed into emitting a well-formed
`<tool_call>` block).

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
