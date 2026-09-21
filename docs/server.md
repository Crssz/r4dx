# r4dx-server: OpenAI-compatible chat API

`src/server/` builds `r4dx-server`, an HTTP server exposing a subset of the OpenAI chat/completions
API over `r4dx::model::Model` (single model, single GPU, HIP device 1). See `docs/architecture.md`
for where this sits in the overall module map and `docs/status.md` for milestone history.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status":"ok","model":"<model id>"}` |
| GET | `/v1/models` | OpenAI models-list shape, one entry (the loaded container's `model_id`, from its `__metadata__.model_id`, `docs/container-format.md`) plus r4dx extension fields -- see "Model metadata" below |
| GET | `/v1/models/{id}` | The single model object (not list-wrapped) for `{id}`; `404` with the standard error JSON if `{id}` is not the loaded container's own `model_id` |
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
carry a tool result back to the model (see "Tool calls" below). A message may also carry an optional
`reasoning_content` (string) -- see "`reasoning_content`" below for the multi-turn replay contract.

### Response shapes

Standard OpenAI `chat.completion` / `chat.completion.chunk` / `text_completion` objects, including
`usage.{prompt_tokens,completion_tokens,total_tokens}` and `finish_reason` (`"stop"` -- EOS or a
`stop` string matched; `"length"` -- `max_tokens` reached; `"cancelled"` -- client disconnected
mid-stream). Streaming responses are `Content-Type: text/event-stream`, one `data: <json>\n\n`
event per chunk, terminated by the literal line `data: [DONE]\n\n`.

## Model metadata

`GET /v1/models` and `GET /v1/models/{id}` keep `id`/`object`/`created`/`owned_by` exactly as
before, plus these r4dx extension fields (different client libraries look for different ones, so
the common set is emitted on every entry -- `BuildModelEntryJson`, `openai_types.h`/`.cpp`):

| Field | Meaning |
|---|---|
| `context_length` | This server's own `--max-ctx` (`Engine::MaxCtx()`). |
| `max_model_len` | Same value, under vLLM's own OpenAI-compatible-server field name. |
| `max_completion_tokens` | Same value again, mirroring the request-side field name. |
| `meta.n_ctx` | Same value, under llama.cpp's `/v1/models` field name. |
| `meta.n_ctx_train` | The checkpoint's own native context length (`kModelNativeContextLength` = 262144, Qwen3.8-27B's `config.json` `max_position_embeddings`) -- a named constant, not read from the loaded container: `r4dx::model::ModelConfig` does not carry this field (nothing in the layer graph needs it). |
| `capabilities` | `["completion", "chat", "tool_use", "reasoning"]` -- a fixed list reflecting what this server actually does (plain completion, the chat template, `tools`/`tool_choice`, and `chat_template_kwargs.enable_thinking`). |
| `supported_parameters` | Exactly the request fields this server's parsers actually honour: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens`, `max_completion_tokens`, `stop`, `stream`, `stream_options`, `tools`, `tool_choice`, `chat_template_kwargs`. Anything not in this list (e.g. `logprobs`, `presence_penalty`) is silently ignored today, so it is deliberately left off rather than falsely advertised. |
| `architecture.input_modalities` / `.output_modalities` | `["text"]` / `["text"]` -- no vision tower yet ("Deferred / known gaps" below). |

`max_ctx` is plumbed from `ServerArgs::max_ctx` (`server_args.h`) through `EngineOptions::
model_opts.max_ctx` into `Engine::MaxCtx()`, which both the `/v1/models` list route and the
`/v1/models/{id}` single-object route read directly -- there is no separate copy to keep in sync.
`GET /v1/models/{id}` 404s (standard `ErrorBody` shape) for any `{id}` other than the loaded
container's own `ModelId()`, since this server only ever loads exactly one model.

**Testing**: `tests/server/test_openai_types.cpp`'s `TestBuildModelEntryJson*`/
`TestBuildModelsResponse*` cases (every field, plus "only lists parameters actually parsed"), and
`tools/server/smoke.ps1`'s `/v1/models` checks (`context_length` matches the `--max-ctx` the script
launched the server with, `"reasoning"` present in `capabilities`).

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
present, precedes the first `<tool_call>` in the raw generation -- but by the time `ParseToolCalls`
ever sees the text, `Engine::RunRequest`'s `tool_mode` block has already stripped it out and
delivered it separately as `message.reasoning_content` (the "`reasoning_content`" section below has
the full writeup; this parser itself never special-cases the tag at all, same as always). A
parameter `VALUE` is plain text, not JSON-tagged; the parser tries to `JSON::parse` the trimmed value and falls back to
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

## `reasoning_content`

DeepSeek/vLLM's `reasoning_content` convention: when a request's RESOLVED `enable_thinking` is true
(`chat_template_kwargs.enable_thinking` if present and a JSON boolean, else the server's `--think`
default -- `ResolveEnableThinking`, `openai_types.h`/`.cpp`; computed independently but identically
by `http_server.cpp`, before the request even reaches the worker thread, and by `Engine::
RunRequest`, so the two calls can never drift apart), the generation is split at the FIRST
`"</think>"` into a reasoning span and an answer span. The splitting itself lives in exactly one
place, `src/server/reasoning_splitter.h`/`.cpp`'s `ReasoningSplitter` (a small pure, HIP-free state
machine, `tests/server/test_reasoning_splitter.cpp`), used both by `BufferingSink`/`StreamingSink`
(the per-piece streaming split) and by `Engine::RunRequest`'s `tool_mode` block (a one-shot
whole-buffer split, same class, just fed the entire buffered string in one `Push()` call). The
model's own opening `"<think>\n"` is part of the PROMPT's generation preamble, never generated text
(`tool_call_parser.h`'s file comment has the full derivation) -- so this only ever scans for the
CLOSING tag.

**Non-streaming** (`/v1/chat/completions`, `"stream"` false/absent): `message.reasoning_content` is
the reasoning span, trimmed of leading/trailing whitespace/newlines; `message.content` is the
answer span, with any leading blank lines right after the tag skipped (never leaked into either
field). If `"</think>"` never appears before generation ends (EOS, `--stop`, cancellation, or
`max_tokens`) -- `reasoning_content` is everything generated, `content` is `""` (empty string, not
null), and `finish_reason` stays whatever it already was (`"length"` for the common max_tokens
case). With thinking off, there is no `reasoning_content` key anywhere and no tag scanning happens
at all -- a literal `"</think>"` in a normal answer passes through completely untouched, byte-for-
byte identical to before this feature existed.

**Streaming**: while inside the thinking span, deltas are `{"reasoning_content": "..."}` (no
`"content"` key in that delta); once the close tag is seen, deltas switch to ordinary
`{"content": "..."}` -- never both keys in one delta. The tag itself and the blank lines right
after it never reach the client in either field. The tag can split across token/piece boundaries
(`"</thi"` + `"nk>"`) or land at the edge of a piece; `ReasoningSplitter` holds back only the
minimal suffix that could still be a prefix of `"</think>"` (at most 7 bytes) and flushes it the
instant it cannot be -- nothing is ever lost or duplicated. Since `"</think>"` is pure ASCII and
every UTF-8 continuation/lead byte has its high bit set, this holdback can never split a multi-byte
character. If the close tag never appears, `OnDone` flushes whatever is still held back as a final
`reasoning_content` delta before the `finish_reason` chunk (the never-closed contract above, applied
per-delta). With thinking off, no chunk ever gains a `reasoning_content` key -- byte-for-byte
unchanged.

**Composes with `--stop`**: stop-string matching applies to the ANSWER part only. `Engine::
RunRequest` tracks, in `accumulated`'s own character-offset space, whether/where `"</think>"` has
appeared; `EmitToken`'s stop search is given a `min_stop_search_from` floor that is `std::string::
npos` (skip stop matching entirely) for as long as the tag has not yet been seen, then the tag's own
end position once it has -- so a stop string that happens to appear inside the model's own
chain-of-thought can never truncate generation before the answer even starts. A thinking-off
request always passes floor `0` (no restriction, today's behavior, unaffected).

**Tool-call mode** (`tools` present, whole-generation buffering, "Tool calls" above): the thinking
span is stripped out of the buffered generation BEFORE `ParseToolCalls` even runs (`Engine::
RunRequest`'s `tool_mode` block, using the same `ReasoningSplitter` fed the whole buffer at once)
and delivered via `ResponseSink::OnReasoningContent` -- the one-shot equivalent of `OnToken`'s
per-piece split, since tool_mode never streams per-piece deltas at all. `tool_calls`/`content`/
`finish_reason` semantics are otherwise unchanged; `ParseToolCalls` itself never sees the tag.

**Multi-turn**: a client resending an earlier assistant turn may include the `reasoning_content`
this server returned for it (`ChatMessage::reasoning_content`, `openai_types.h`, accepted on any
message role -- validated as a string when present, a non-string is a `400`). It is passed straight
through into the rendered message JSON, unconditionally -- `C:\AI\models\Qwen3.8-27B\
chat_template.jinja` already reads `message.reasoning_content` on an assistant turn and decides
whether to keep it wrapped in `<think>...</think>` based on `chat_template_kwargs.preserve_thinking`
(default: keep it) and how far back the turn is relative to the latest user query, so no extra
server-side gating logic is needed -- the field is passed through and the template decides.
**Prefix reuse**: since the re-rendered assistant turn (its own `<think>...</think>` block rebuilt
from JSON `reasoning_content`/`content`) is not guaranteed to be byte-identical to the model's own
originally-generated tokens for that turn (trimming/whitespace differences), a second turn of a
thinking conversation may legitimately take `PrefixState`'s `Reset()`+reprefill path rather than the
prefix-extend fast path -- both are correct, just different costs; see this stage's own real
two-turn verification run for which one actually happened and why.

**`usage.completion_tokens_details`**: `usage.completion_tokens` keeps counting ALL generated
tokens (reasoning + answer, unchanged). `completion_tokens_details.reasoning_tokens` is added
(`BuildUsageJson`, `openai_types.cpp`, shared by every usage-object-producing builder) with the
number of generated tokens up to and including the close tag -- computed by `Engine::RunRequest`
itself (`generated_tokens.size()` at the moment `"</think>"` first appears in `accumulated`; the
sink-level `ReasoningSplitter` only ever sees decoded TEXT, so it cannot count tokens). The whole
`completion_tokens_details` object is OMITTED (not merely zeroed) when thinking is off, so that
path's `usage` object stays byte-for-byte identical to before this field existed.

**`/v1/completions` is NOT split**: the raw-text completions endpoint has no chat template and
therefore no `enable_thinking` concept at all -- `CompletionRequest` carries no
`chat_template_kwargs`, so `usage.completion_tokens_details` is never attached and no
`reasoning_content` key or delta ever appears on this endpoint, regardless of what the raw prompt
text itself contains.

**Testing**: `tests/server/test_reasoning_splitter.cpp` (the splitter itself: every split position
of `"</think>"` across piece boundaries, byte-by-byte fragmentation, the never-closed case, UTF-8
multi-byte content on both sides, composition with a stop search scoped to the answer side only),
`tests/server/test_response_sink.cpp` (`BufferingSink`/`StreamingSink` routing, `OnReasoningContent`
tool_mode bypass, thinking-off byte-identical passthrough), `tests/server/test_openai_types.cpp`
(`reasoning_content` request-side parsing/validation, `ResolveEnableThinking`, the response
builders' `reasoning_content`/`completion_tokens_details` fields), and `tools/server/smoke.ps1`'s
real-container checks (non-streaming and streaming `enable_thinking: true` requests, and an
`enable_thinking: false` request confirming no `reasoning_content` key appears anywhere).

## `timings`

An r4dx extension beyond the OpenAI spec: a `timings` object attached as a top-level sibling of
`usage` on every non-streaming `/v1/chat/completions` response (both `BuildChatCompletionResponse`
overloads, including a tool-call response), every non-streaming `/v1/completions` response, and --
for a streaming request -- on the SSE finish_reason chunk (the same chunk that carries the
non-null `finish_reason`), for both endpoints. Field names deliberately match llama.cpp's own
`/completion` `timings` object, so existing bench tooling written against a llama.cpp server keeps
working unmodified against this one:

```json
"timings": {
  "prompt_n": 12,
  "prompt_ms": 45.2,
  "prompt_per_second": 265.5,
  "predicted_n": 8,
  "predicted_ms": 96.4,
  "predicted_per_second": 83.0,
  "draft_n": 32,
  "draft_n_accepted": 24
}
```

- **`prompt_n`/`prompt_ms`**: tokens actually fed to `Model::Prefill` THIS request and how long
  that took. `prompt_n` is deliberately NOT `usage.prompt_tokens` (the whole conversation-so-far
  token count) -- it is `Engine::RunRequest`'s own `new_tokens_i32.size()`, i.e. it excludes
  whatever prefix reuse (`PrefixState::Extend`, this file's "Concurrency model" section) skipped
  re-prefilling. A client resending a growing `messages` array (the normal chat-client pattern)
  will see `prompt_n` stay small turn over turn while `usage.prompt_tokens` keeps growing with the
  conversation -- that gap IS prefix reuse, made visible instead of just implied by a fast
  response. A prefix mismatch (`Model::Reset()` + full re-prefill) instead shows `prompt_n ==
  usage.prompt_tokens` for that request, same as if prefix reuse never existed.
- **`predicted_n`/`predicted_ms`**: generated token count (always equal to
  `usage.completion_tokens`) and wall-clock decode time for this request, whichever loop produced
  it (plain sampling, greedy MTP, or greedy DFlash2).
- **`*_per_second`**: `n / (ms / 1000)`, computed as `0.0` (never left to produce `NaN`/`inf`, both
  of which nlohmann::json's own `dump()` refuses to serialize) whenever the corresponding `*_ms` is
  `0` -- e.g. a zero-token generation (immediate EOS).
- **`draft_n`/`draft_n_accepted`**: total drafted/accepted token counts, summed across every
  speculative round this request ran (`mtp_drafted`/`mtp_accepted` for `--mtp`,
  `dflash_drafted`/`dflash_accepted` for `--dflash` -- mutually exclusive, `server_args.h` rejects
  both together). Present ONLY when a speculative path actually ran at least one round this
  request (greedy MTP or DFlash2, `use_mtp`/`use_dflash` in `engine.cpp`, gated the same way the
  existing `mtp:`/`dflash:` stderr log-line suffixes are) -- absent for a plain-decode or sampled
  (`temperature > 0`) request, and absent even on a speculative-capable server if the request
  finished (e.g. immediate EOS) before any round ran.

Implementation: `TimingStats` (`src/server/openai_types.h`, next to `UsageStats`) is filled in by
`Engine::RunRequest` from values it already measures for its own stderr log line (`prefill_seconds`,
`decode_seconds`, `new_tokens_i32.size()`, `generated_tokens.size()`, the mtp_*/dflash_* counters)
and handed to `ResponseSink::OnDone`; `BuildTimingsJson` serializes it. That stderr log line itself
is unchanged by this feature.

## `stream_options`

`stream_options: {"include_usage": bool}` (OpenAI's own streaming-usage opt-in), accepted on both
`/v1/chat/completions` and `/v1/completions`. Validated unconditionally: `stream_options` must be
an object if present, `include_usage` must be a boolean if present -- either violation is a clear
`400 invalid_request_error` (`openai_types.cpp`'s `ParseStreamOptions`). **Tolerance choice**:
`stream_options` on a non-streaming request (`"stream"` absent or `false`) is accepted and simply
ignored, rather than rejected the way OpenAI's own API does -- a caller that always sets
`stream_options` regardless of `stream` (a common client-library pattern) should not have to special
-case this server, and the field genuinely has nothing to do once there is no stream to attach usage
chunks to.

When `stream: true` and `stream_options.include_usage: true`, the exact SSE chunk sequence is:

1. The role-preamble chunk (chat only) and every content/tool_calls delta chunk each additionally
   carry a top-level `"usage": null`.
2. The finish_reason chunk (unchanged shape otherwise, still carries `timings` per the section
   above) ALSO carries `"usage": null`.
3. One more chunk follows, `"choices": []`, with the real `"usage"` object
   (`{prompt_tokens, completion_tokens, total_tokens}`) and the same `"timings"` object as the
   finish_reason chunk.
4. `data: [DONE]\n\n`.

When `include_usage` is absent or `false`, streaming output is byte-for-byte identical to before
this feature except for the `timings` key added to the finish_reason chunk (item 1 above) -- no
chunk ever gains a `usage` key. Tool-call mode (`engine.cpp`'s whole-generation buffering, "Tool
calls" above) behaves identically either way: the buffered `OnToolCalls` chunk is just another
"normal chunk" for the purposes of the `usage: null` rule.

Implementation: `StreamingSink` (`response_sink.h`/`.cpp`) takes an `include_usage` constructor
flag (`http_server.cpp` passes `ChatCompletionRequest`/`CompletionRequest::
stream_options_include_usage` straight through) and stores the request's `prompt_tokens` from
`OnStart` so `OnDone` can build the dedicated usage chunk's `usage` object without engine.cpp having
to pass it separately.

**Testing**: `tests/server/test_openai_types.cpp` (`stream_options` parsing/validation,
`BuildTimingsJson` arithmetic including the zero-`ms` and draft-fields-present/absent cases, every
builder's `timings` attachment) and `tests/server/test_response_sink.cpp` (chunk ordering and
`usage`/`timings` placement with and without `include_usage`), plus `tools/server/smoke.ps1`'s
`timings`/`stream_options` checks against a real container (prompt_n/predicted_n arithmetic, prefix
reuse made visible through `prompt_n`, the include_usage chunk sequence, and `draft_n` with
`-Dflash`/`-Mtp`).

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
`full`, flipped from `reduced` 2026-09-20 per Milestone 5 B2 item 8 -- matched-K=3 real-hardware
re-measurement: `full` 68.20/68.64 tok/s (46.3% acceptance) vs `reduced` 53.59/54.17 tok/s (20.9%
acceptance), byte-identical generated text either way, see docs/mtp.md's "Flag default flipped"
correction), mirroring `src/cli/cli_args.h`'s
own flag, wired to `ModelOptions:: mtp_draft_reduced_vocab` in `src/server/main.cpp` identically to
the CLI. `reduced` uses the
container's OPTIONAL `mtp.draft_head.*` tensors (docs/container-format.md) to speed up drafting when
present, degrading to the exact pre-R9 full-vocab behavior automatically when absent; `full` forces
the full-vocab head unconditionally. Verification always stays full-vocab regardless of this flag,
so it can only affect drafting speed/acceptance, never generated output (see docs/mtp.md's
"Reduced-vocab draft head" section for the full argument and the measured K-sweep). Covered by
`tests/server/test_server_args.cpp`'s `TestMtpDraftHeadFlag`.

**New (Milestone 5 stage S3, docs/dflash2.md)**: `--dflash <draft.r4dx>` enables server-side DFlash2
block-diffusion self-speculative decode, mirroring `src/cli/cli_args.h`'s own `--dflash` flag
exactly -- same `--dflash-k`/`--dflash-p-min`/`--dflash-n-min` companions, same mutual exclusion
with `--mtp > 0` (rejected at arg-parse time, both `ServerUsageError` and `Model::Load`'s own
re-check), same greedy-only gate (`Engine::RunRequest`'s `use_dflash` mirrors `use_mtp` exactly).
The DFlash2 draft container is independent of `--layout` (the TARGET body's layout) -- it carries
its own packed layout in its own metadata. `tools/server/smoke.ps1` gained `-Dflash <path>` and a
`" dflash: "` request-log check; run for real against the real 64-layer container + the real w4a16
draft container, all 28 checks passed (streaming and tool-call mode both unaffected). Covered by
`tests/server/test_server_args.cpp`'s `TestDflashFlags`.

**Sampled (`temperature>0`) traffic no longer pays for the drafter (Milestone 5 follow-up,
2026-09-21).** This section previously documented a ~3% "known tax" as unfixable; it is fixed. The
mechanism it described was real: `--dflash` is a `Model`-wide, load-time flag, `Model::Load`
attaches the drafter, and `RunChunk` used to auto-feed it (per-layer target feature capture +
`DflashDraft::InjectFeatures`) on *every* prefill chunk and decode step for the life of the process,
while `Engine::RunRequest`'s `use_dflash` gate only decided whether a request's decode loop *reads*
drafted tokens (greedy-only, mirroring `use_mtp`). The reason it was recorded rather than fixed was
that `InjectFeatures` demanded `start_pos == InjectedCount()` (strictly append-only), so skipping
injection on a sampled turn would have made the very next greedy turn on the same session throw.

**What changed.** `DflashDraft` now tolerates a gap in its own ring: injection is still monotonic,
but `start_pos > InjectedCount()` is accepted and moves a new validity lower bound,
`DflashDraft::ValidFrom()`, up to the resume position. That bound is handed to
`r4dx_dflash_attn_bf16` as its new `store_begin` parameter, which clamps the visible key range to
`[max(store_begin, q_pos - window + 1), n_injected)` -- so the skipped positions' stale ring bytes
are never read, and never need clearing (docs/dflash2.md section 5). On top of that,
`Model::SetDflashInjectionEnabled(bool)` turns the capture *and* the injection off together, and
`Engine::RunRequest` calls it with `use_dflash` (i.e. `temperature <= 0`) before the request's
prefill, on both the prefix-reuse and the `Reset()`+re-prefill path. A sampled request therefore
captures nothing and injects nothing; the next greedy request resumes injection at the current
position, which is exactly the cold-ring gap the drafter now tolerates.

**Measured, real 64-layer `qwen38-27b-v3.r4dx` at `--layout w4a16` + the w4a16 draft container,
HIP device 1, one server process at a time.** Identical request each time (`temperature=0.7`,
`top_p=0.95`, `seed=12345`, 42-token prompt, 128 generated tokens), two runs per configuration,
after a discarded warm-up request:

| Configuration | sampled decode (run 1 / run 2) | vs plain server |
|---|---|---|
| plain server (no `--dflash`) | 28.52 / 28.54 tok/s | -- |
| `--dflash`, BEFORE this change | 28.03 / 28.04 tok/s | **-1.75%** |
| `--dflash`, AFTER this change | 28.30 / 28.32 tok/s | **-0.77%** |

**The remaining 0.77% is not the injection, and cannot be removed per-request.** Loading with
`dflash_draft_k > 0` at all sets `Model::draft_window_ = 8`, which sizes the GDN window state bank
and KV scratch for a verify window (`kv+gdn_state` 2.35 GiB vs the plain server's 0.34 GiB) and
makes every plain decode step thread a `num_accepted` pointer through all the GDN layers plus one
small blocking H2D. That is structural to running a speculative server, not to DFlash2: the control
run, an `--mtp 7` server (no DFlash2 code involved at all) on the identical request, measured
**27.96 / 27.98 tok/s** -- a *larger* sampled-traffic cost than the fixed `--dflash` server's. The
drafter's ~1.0 GiB of VRAM is likewise still held for the life of the process regardless of whether
any request uses it.

**The honest downside: the first greedy request after sampled traffic drafts from a cold ring.**
Its drafter has no injected context below the resume position, so its early rounds have less to
condition on and accept fewer tokens; acceptance recovers as the ring refills behind the frontier.
Measured on the same server, a greedy multi-turn continuation immediately after two 128-token
sampled requests (so `PrefixState` extended rather than reset, and the ring really was cold):
68.86 tok/s, 21 rounds, 21.8% accept, 2.48 tok/round -- against the 33.0% accept / 3.28 tok/round
a greedy request gets on a fully warm ring. It is still ~1.8x the plain server's 38.59 tok/s on the
same request, so the trade is a modest acceptance dip on one turn in exchange for sampled requests
paying nothing at all. `r4dx-cli` is unaffected: it still clears `args.dflash` outright at
`temperature>0` before `Model::Load` (`src/cli/main.cpp`), which a one-request-per-invocation
process can do for free.

Covered by `tests/kernels/test_dflash_attn.cpp` (the `store_begin` sweep with the invisible slots
filled with junk), `tests/model/test_dflash_draft.cpp` Part 5 (a gapped ring drafts bit-identically
to a ring that only ever saw the post-gap rows) and `tests/model/test_dflash_e2e.cpp`'s
`CheckInjectionToggleGap` (the full greedy -> injection-off -> greedy-again sequence on the real
target, whose post-gap tokens must equal an independently loaded non-dflash reference exactly).

## CLI flags

```
r4dx-server --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16}
    [--tokenizer-dir <dir>] [--host <addr>] [--port N] [--max-ctx N]
    [--max-tokens-default N] [--max-queue N] [--think {on|off}] [--layers N]
    [--default-temperature F] [--default-top-p F] [--default-top-k N]
    [--default-min-p F] [--log-level {debug|info|warn|error}] [--mtp N]
    [--mtp-head-layout {bf16|layout}] [--mtp-draft-head {reduced|full}]
    [--embed-device-resident {on|off}] [--dflash <draft.r4dx>] [--dflash-k N]
    [--dflash-p-min F] [--dflash-n-min N]
```

`--mtp N` (default 0): see "MTP" above -- requires an MTP-converted `--model` container when N>0.

`--dflash <draft.r4dx>` (default empty, disabled): see the "New (Milestone 5 stage S3...)" note
above -- mutually exclusive with `--mtp N>0`.

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
