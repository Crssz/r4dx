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
Qwen3.8-27B's own template understands), the thinking controls `enable_thinking` / `reasoning_effort`
/ `reasoning` / `thinking` / `include_reasoning` (see "Thinking controls" below), `tools` (rendered
into the prompt AND parsed back out of
the generation into a structured `message.tool_calls` response field -- see "Tool calls" below),
`tool_choice` (`"none"`/`"auto"`/`"required"`/`{"type":"function","function":{"name":...}}`, see
"Tool calls").

Message `content` may be a plain string or an OpenAI-style array of parts
(`[{"type":"text","text":"..."}]`, `[{"type":"image_url","image_url":{"url":"data:..."}}]`, or a mix
of both -- see "Images" below for the full contract, formats, limits and error cases). Message
`role` must be `system`, `user`, `assistant`, `tool`, or `function` -- the latter two
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
| `capabilities` | `["completion", "chat", "tool_use", "reasoning"]` -- what this server actually does (plain completion, the chat template, `tools`/`tool_choice`, and `chat_template_kwargs.enable_thinking`), plus a fifth entry `"image"` whenever the loaded container's vision tower is resident (`Model::HasVision()`, "Images" below). |
| `supported_parameters` | Exactly the request fields this server's parsers actually honour: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens`, `max_completion_tokens`, `stop`, `stream`, `stream_options`, `tools`, `tool_choice`, `chat_template_kwargs`, `reasoning`, `reasoning_effort`, `include_reasoning`, `enable_thinking`, `thinking`. Anything not in this list (e.g. `logprobs`, `presence_penalty`) is silently ignored today, so it is deliberately left off rather than falsely advertised. |
| `top_provider` | `{context_length, max_completion_tokens, is_moderated: false}` -- the OpenRouter-shaped repeat of the same `--max-ctx` numbers above, under the field names an OpenRouter-shaped client reads. `is_moderated` is false: nothing in this process filters or classifies a generation. |
| `reasoning` | `{supported_efforts, default_enabled, mandatory: false}` -- the OpenRouter-shaped thinking-capability block. `supported_efforts` is `["none","minimal","low","medium","high","xhigh","max"]`, exactly the set `ParseThinkingControls` accepts (`"none"` = off, each other level maps onto one of the template's own three -- see "Thinking controls" below). `default_enabled` is the server's own `--think` flag, so a client can pre-set its toggle the way this server will really behave. `mandatory` is false: thinking can always be turned off. |
| `architecture.input_modalities` / `.output_modalities` | `["text","image"]` / `["text"]` when the loaded container's vision tower is resident (`Model::HasVision()`), `["text"]` / `["text"]` when it is not (`--vision off`, or a container with no `vision.*` tensors). The input list has exactly ONE writer, `ModelInputModalities()` (`openai_types.cpp`); the plainer `modalities` spelling and the `capabilities` row above are driven off the same `has_vision` flag. See "Images" below. |

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
`delta.tool_calls[].function.arguments` byte deltas. This server does NOT do that -- the CALLS
themselves are always delivered as one single complete `delta.tool_calls` chunk carrying every call
at once (`response_sink.cpp`'s `StreamingSink::OnToolCalls`, `index`-tagged so a client expecting
real per-delta streaming still assembles the array correctly), at the end, after the whole span has
been parsed. The ANSWER TEXT around them, however, does stream live:

* **`tools` + `"stream": true`** -- content streams token by token as ordinary `delta.content`,
  exactly like a request with no tools, until a literal `<tool_call>` opener appears in the ANSWER
  part of the generation. At that instant the content stream shuts off for good; everything from
  there on is buffered and classified once, at the end, by `ParseToolCalls`. The gate is
  `src/server/tool_stream_gate.h`/`.cpp`'s `ToolStreamGate` (a small pure, HIP-free state machine,
  `tests/server/test_tool_stream_gate.cpp`), driven by `Engine::RunRequest`'s own per-request
  forwarding router. It gates on the literal `<tool_call>` -- byte-for-byte the same constant
  `ParseToolCalls` scans for -- and holds back any trailing text that is still a proper prefix of it
  (at most 10 bytes, the same minimal-holdback rule `ReasoningSplitter` uses for `</think>`), so a
  client never sees even a partial `<tool_` leak into a content delta. Because the opener is pure
  ASCII, that holdback can never split a multi-byte UTF-8 character.
* **`tools` + `"stream": false`** -- nothing is delivered live (there is nothing live about a
  non-streaming request); the whole response is re-derived from the accumulated text afterwards,
  byte-for-byte as it always was.
* **no `tools`** -- completely unaffected, real per-token streaming, unchanged.

**Streamed content equals non-streamed content.** The concatenation of every `delta.content` a
streaming request receives is byte-identical to the `message.content` the non-streaming path returns
for the same generation. Two things make that hold: the gate does NOT trim whitespace before an
opener (neither does `ParseToolCalls` -- its `content` is the text outside every span, verbatim), and
whatever the gate held back is delivered after the fact as one final content delta computed by
`ToolStreamRemainder(streamed, parsed.content)` -- the same function
`tests/server/test_tool_stream_gate.cpp` asserts the invariant with, over every 1-byte and every
2-piece chunking of a dozen representative generations. So a malformed `<tool_call>` span that
degrades to literal content, prose that follows a call, and a `DropUnknownToolCalls` note all still
reach a streaming client, just at the end rather than live. The one thing streaming cannot undo is a
`--stop` string that only completes ACROSS a token boundary: up to `len-1` of its bytes may already
have gone out, exactly as on a request with no `tools` at all (the non-streaming path re-derives from
the accumulated text and is exact).

This replaces the previous whole-generation buffering, which applied to every request that merely
OFFERED tools. That was safe but produced "fake" (all-at-once) streaming for clients that attach a
`tools` array to every turn (Unsloth Studio does), so a plain prose answer that never called anything
showed nothing at all until generation finished. The per-request stderr log line reports `tools=<N>`
(definitions offered) next to `stream=`, which is exactly what decides which of the three cases above
a request took.

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

**Testing**: `tests/server/test_tool_call_parser.cpp` (parser unit cases),
`tests/server/test_tool_stream_gate.cpp` (the live gate: holdback across every opener split point,
`<`/`<tool_box>`/`<tool_calls>` lookalikes, malformed and unterminated spans, unicode, and the
property test above over every chunking), the tool-call-specific cases in
`tests/server/test_openai_types.cpp` (request-side `tool_calls`/`tool_choice` parsing/validation,
`role: "tool"`/`"function"` messages) and `tests/server/test_response_sink.cpp` (the streaming
`OnToolCalls` SSE chunk shape), and `tools/server/smoke.ps1`'s real end-to-end checks against a
container: the tool call/result/answer round trip (a tool-offering request, feeding the parsed
call's result back as a `role: "tool"` message, checking the final answer references the tool
result), plus the live-streaming checks -- a tool-offering request asking a plain prose question
must produce many content deltas with the first arriving well before the last, the concatenated
streamed content must equal the non-streaming content for the same greedy request, and no content
delta may contain `<tool_call`/`</tool_call`.

## Thinking controls

This checkpoint thinks when its chat template's `enable_thinking` variable is true. Clients,
however, spell "turn thinking on" five different ways, and before this section existed this server
accepted exactly one of them (`chat_template_kwargs.enable_thinking`), which is why a client with a
thinking toggle -- Unsloth Studio among them -- could not drive it. All five are now accepted and
reduced to one answer by `ParseThinkingControls` (`openai_types.h`/`.cpp`), whose result rides on
`ChatCompletionRequest::thinking` / `PendingRequest::thinking` and is resolved by the single
`ResolveEnableThinking(ThinkingControls, bool)` that both `http_server.cpp` (which must pick the
sink's splitting behavior before the request reaches the worker thread) and `Engine::RunRequest`
(which renders the template and does the generation-time splitting) call.

### Accepted fields, in precedence order

The FIRST source present wins; an explicit request field always beats the `--think` server default.

| # | Field | Shape | Meaning |
|---|---|---|---|
| 1 | `chat_template_kwargs.enable_thinking` | bool | Highest, because it addresses THIS checkpoint's own template directly and is the only spelling this server accepted before -- so no existing caller's behavior moved. A non-boolean here is tolerated (not a 400) and falls through to the next source, exactly as `ResolveEnableThinking(json, bool)` always tolerated it: `chat_template_kwargs` is an opaque passthrough to the template. |
| 2 | `reasoning` | `{"enabled": bool, "effort": str, "exclude": bool, "max_tokens": int}` | OpenRouter's unified field. `enabled` wins over `effort` inside the object. `exclude` is orthogonal (see below). `max_tokens` is validated and then ignored -- there is no reasoning-token budget here -- and is therefore NOT in `supported_parameters`. |
| 3 | `thinking` | `{"type": "enabled"\|"disabled"}` | Anthropic's shape, which several OpenAI-compatible gateways mirror verbatim. An optional `budget_tokens` is validated and ignored, same as `reasoning.max_tokens`. |
| 4 | `enable_thinking` | bool | Top level. vLLM/SGLang and several local servers accept it directly. |
| 5 | `reasoning_effort` | string | OpenAI's own field name. `"none"` means thinking OFF; every other non-empty string means ON. |
| 6 | -- | -- | Nothing named: the server's `--think` flag decides, byte-for-byte as before. |

Anything malformed is a `400 invalid_request_error` with the standard error body, never a silently
ignored field: a non-object `reasoning`/`thinking`, a non-boolean `enabled`/`exclude`/
`enable_thinking`/`include_reasoning`, a non-string or blank `effort`/`reasoning_effort`, a
`thinking.type` that is neither `"enabled"` nor `"disabled"`, or a negative/non-integer
`reasoning.max_tokens`/`thinking.budget_tokens`. Every one of those, and every precedence rule
above, is covered by `tests/server/test_openai_types.cpp`'s `TestThinking*` cases and re-checked
end to end by `tools/server/smoke.ps1`.

### Effort levels really do something here

`C:\AI\models\Qwen3.8-27B\chat_template.jinja` reads `reasoning_effort` itself and accepts exactly
three values -- `xhigh` (its default), `medium` and `low` -- raising
`Unexpected reasoning effort ...` on anything else (which this server would surface as a `400`,
since a template render failure is a caller-shape problem). `xhigh` and `low` each inject a
different `reasoning_instructions` sentence into the system prompt; `medium` injects none. So an
effort is not cosmetic, and passing a client's level through verbatim would break ordinary
requests. `ParseThinkingControls` maps the de-facto scale onto the template's three:

| Client effort | Template level |
|---|---|
| `none` | (thinking off -- no effort is passed at all) |
| `minimal`, `low` | `low` |
| `medium` | `medium` |
| `high`, `xhigh`, `max` | `xhigh` |
| anything else | (none -- thinking on, template's own default `xhigh`) |

Matching is case- and whitespace-insensitive. The mapped level is written into the template's
`extra_context` as `reasoning_effort` by `Engine::RunRequest`, but only when the request did not
send an explicit `chat_template_kwargs.reasoning_effort` -- that passthrough still wins, same rule
`enable_thinking` follows. `tools/server/smoke.ps1` proves the level really reaches the template by
checking that `low`/`medium`/`high` render prompts of different token counts.

### `reasoning.exclude` / `include_reasoning`: think, but withhold the thought

`reasoning: {"exclude": true}` (OpenRouter) and `include_reasoning: false` (its legacy boolean
spelling) are orthogonal to the on/off question: thinking still happens, the generation is still
split so `content` is the ANSWER alone with no `</think>` leaking through, and
`usage.completion_tokens_details.reasoning_tokens` is still reported (the tokens were really
spent) -- only the `reasoning_content` field and its streaming deltas are suppressed. Either one
asking for exclusion wins over the other asking for inclusion. Implemented as one `emit_reasoning`
flag on both sinks (`response_sink.h`), defaulted to true so every existing call site is unaffected.

### What the RESOLVED value changes

`enable_thinking` in the template's `extra_context` (only when the request sent none itself -- the
passthrough rule above), whether the sinks split at all, the `min_stop_search_from` floor that
confines `--stop` matching to the answer, `usage.completion_tokens_details.reasoning_tokens`, and
the `thinking=yes|no` field of the per-request stderr log line (which is what
`tools/server/smoke.ps1` reads to check every spelling above against BOTH containers, since it does
not depend on the model producing a well-formed `</think>` span).

**Text-only behaviour is unchanged.** A request naming none of these fields resolves to `--think`
and renders exactly the `extra_context` it did before this section existed, so its output is
byte-identical. `tools/server/smoke.ps1` asserts that directly: every thinking-OFF spelling must
produce `message.content` byte-identical to a `chat_template_kwargs.enable_thinking: false`
baseline, with no `reasoning_content` and no `completion_tokens_details` key anywhere.

**`/v1/completions` is unaffected**: it has no chat template and therefore no thinking concept.
These fields are accepted in the body (they simply have nowhere to go) rather than rejected.

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
whole-buffer split, same class, just fed the entire accumulated string in one `Push()` call). The
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

**Tool-call mode** (`tools` present, "Tool calls" above): the thinking span is always stripped out
of the accumulated generation BEFORE `ParseToolCalls` even runs (`Engine::RunRequest`'s `tool_mode`
block, using the same `ReasoningSplitter` fed the whole string at once), so `ParseToolCalls` never
sees the tag and a `<tool_call>` the model writes INSIDE its own chain-of-thought is not a call --
and, for the same reason, never closes the live stream gate either. How the reasoning reaches the
client depends on `stream`: a NON-streaming request gets it one-shot via
`ResponseSink::OnReasoningContent` (already trimmed), while a STREAMING one gets ordinary live
`reasoning_content` deltas, because the engine forwards raw generated bytes -- `</think>` included --
to `OnToken` and the sink's own splitter does the split, exactly as on a request with no `tools`.
The whitespace rules therefore follow the path they are on, unchanged: the non-streaming span is
whole-span trimmed, the streamed deltas are not (only the tag and the blank lines right after it are
dropped). `tool_calls`/`content`/`finish_reason` semantics are otherwise unchanged.

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
real-container checks (non-streaming and streaming `enable_thinking: true` requests, an
`enable_thinking: false` request confirming no `reasoning_content` key appears anywhere, and a
thinking + `tools` + streaming request confirming the reasoning span really does arrive as many
live deltas rather than one post-generation lump, with its streamed `content` still equal to the
same request's non-streaming `message.content`).

## Client compatibility: Unsloth Studio

Read from the installed copy's own source (READ ONLY, never modified): the backend Python under
`%USERPROFILE%\.unsloth\studio\unsloth_studio\Lib\site-packages\studio\backend` and the bundled,
minified frontend under `...\studio\frontend\dist\assets`. Identifiers below are quoted from those
files so every claim is checkable; anything that would need the GUI driven to confirm is called out
as unverified.

### How a connection is made, and which provider types allow a custom base URL

`core/inference/providers.py`'s `PROVIDER_REGISTRY` is the whole list. Four entries are
`"hidden": True` and surfaced by the frontend as "Custom" connection presets rather than the
provider dropdown -- `providers-api-*.js`'s own preset table is
`[{providerType: "llama_cpp", baseUrlPlaceholder: "http://localhost:8080/v1"}, {providerType:
"vllm", ...}, {providerType: "ollama", ...}]` plus a generic `custom`. `base_url_editable` defaults
to `True`, and only `openai_codex` sets it to `False`, so every other type (including `openrouter`)
accepts a custom base URL. A loopback URL is fine: `providers.py`'s SSRF guard blocks only cloud
metadata hosts, and non-public addresses only when the operator opts in via
`UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS` ("loopback and LAN endpoints are the normal case").

### (a) Where the model card's values come from -- NOT from this server

The card is built in `chat-*.js` (labels `Model ID`, `Connection`, `Endpoint`, `Accepts`,
`Generates`, `Context window`, `Max output`, `Reasoning`). Its inputs:

* **Context window** -- `Oe(providerType, modelId)?.contextLength`, i.e. the models.dev catalogue
  (`routes/providers.py`'s `/model-catalog`, trimmed by `trim_models_dev_catalog`, keyed by
  provider type + model id) or the OpenRouter live capability map. Neither has an entry for a local
  server, so it renders **"Not published"**.
* **Max output** -- `_n(providerType, modelId)`: OpenRouter reads its live capabilities, everything
  else a hardcoded per-provider prefix table (`gpt-4`, `claude-*`, `gemini-*`, ...). No match for a
  local model id, so it renders **"Follows the connection"** (the per-connection `max_output_tokens`
  the user may type in the connection form).
* **Reasoning** -- `_r(providerType, modelId, {isReasoningProvider, baseUrl})`, see (b).
* **Accepts** -- `Oe(...)?.inputModalities`, else `rt(providerType, modelId)` for a plain
  `Text`/`Text · Images` answer. Both are catalogue-driven, so a local connection renders **"Text"**.

**This server's `/v1/models` is never consulted for any of it.** `core/inference/
provider_model_capabilities.py` has `MODEL_CAPABILITY_PROVIDERS = frozenset({"openrouter"})` and
`provider_model_capabilities()` returns `[]` for every other provider type -- and even for
`openrouter`, the frontend gates the fetch to the official URL:

```js
var Vm = {openrouter: `https://openrouter.ai/api/v1`};
function Hm(e){ let t = Vm[e.providerType]; if(!t) return false;
                let n = (e.baseUrl ?? ``).trim().replace(/\/+$/,``).toLowerCase();
                return n === `` || n === t; }
// ... for (let n of e) { ... if(!Hm(n)) continue;  <- skips /api/providers/model-capabilities
```

So an OpenRouter-type connection pointed at a local base URL never has its capabilities fetched at
all. **Conclusion, stated plainly: the OpenRouter-shaped `top_provider`/`reasoning`/
`supported_parameters`/`architecture` fields this server now publishes on `/v1/models` do NOT change
anything in this Unsloth Studio build.** They were still added (they are the one machine-readable
capability shape several other OpenAI-compatible clients read, and the vision milestone needs
`architecture.input_modalities` to exist), but nobody should expect Studio's card to start filling
in because of them.

### (b) Where a thinking toggle comes from, and the one setup that gets one

`providers-api-*.js`'s `_r(providerType, modelId, opts)` decides `supportsReasoning` /
`reasoningStyle` / `supportsReasoningOff` / `reasoningEffortLevels`. For the four local presets:

| Connection type | Result |
|---|---|
| `custom` (generic "Custom") | falls to `default: return Z()` -- `supportsReasoning: false`. **No toggle.** |
| `llama_cpp` | `hr("llama_cpp", modelId, true)` -- a models.dev lookup by NORMALIZED MODEL NAME. This container's `model_id` (`Qwen/Qwen3.8-27B`) normalizes to `qwen3.8`, which is in no catalogue, so `null` -> `Z()`. **No toggle.** |
| `vllm` | `dr(providerType, opts)`: `providerType === "vllm" && opts?.isReasoningProvider` returns `{supportsReasoning: true, supportsReasoningOff: true, reasoningStyle: "enable_thinking"}` -- **unconditionally, no catalogue lookup, no model-id match needed.** |
| `openrouter` (base URL overridden) | no catalogue match, so `case "openrouter": return gr`, and `gr = {supportsReasoning: true, reasoningStyle: "enable_thinking", supportsReasoningOff: true}`. **Toggle appears**, but the body it sends is the OpenRouter shape (see (c)). |

`isReasoningProvider` is the connection's own saved `isReasoningModel` flag. `connections-tab-*.js`
renders it as a switch with `htmlFor: "provider-is-reasoning"`, label **"Reasoning model"** and the
caption **"This server runs a reasoning model"**, and gates that switch on `We(providerType)`, whose
set is `new Set(["vllm"])` -- it exists for the vLLM preset and no other.

### (c) Exactly what Unsloth then sends to this endpoint

`core/inference/external_provider.py`'s `stream_chat_completion` builds the body:

* **`vllm` and `llama_cpp`** take the `provider_info.get("supports_chat_template_kwargs")` branch
  (both registry entries set it; `custom` deliberately does not). It sets
  `body["chat_template_kwargs"]["enable_thinking"] = bool(thinking)` whenever the resolved thinking
  is not `None`, and additionally `body["reasoning_effort"] = effort` when the effort is one of
  `low|medium|high` (aliased first through `_LOCAL_SERVER_EFFORT_ALIASES = {"minimal": "low",
  "xhigh": "high", "max": "high"}`). Note the asymmetry it documents: `"none"` rides on the kwarg
  alone and is never sent as a top-level `reasoning_effort`, because "vLLM through 0.16 types the
  top-level reasoning_effort as low | medium | high and 400s on `none`". With
  `reasoningStyle: "enable_thinking"` (what the vLLM switch produces) the frontend sends only
  `enableThinking`, so in practice the wire carries **`chat_template_kwargs.enable_thinking`** --
  which this server already honoured before this milestone.
* **`openrouter`** takes its own branch: `body["reasoning"] = {"effort": ...}` when an effort was
  picked, `{"enabled": False}` for off, `{"enabled": True}` for on.
* **`custom`** takes none of them and sends no thinking field at all. It also carries
  `"body_omit": ("top_k", "min_p", "repetition_penalty")`.

Everything in that list is now among the supported forms ("Thinking controls" above), so either
setup works.

Other request-shape facts worth knowing, from the same file: `vllm` is in
`_USAGE_STREAM_OPTION_PROVIDERS`, so it sends `stream_options: {"include_usage": true}` on every
streaming request (supported here); `vllm`/`llama_cpp` are in `_CONTINUATION_FLAG_PROVIDERS`, so a
"continue this assistant turn" action adds `continue_final_message` / `add_generation_prompt`, which
this server does not implement and silently ignores; `vllm`, `llama_cpp`, `ollama` and `custom` are
in `_TEMPLATE_APPLYING_PROVIDERS`, so Studio neutralizes control markup (`</think>` and turn
markers) inside the messages it sends.

### (d) What it renders as thinking

`delta.reasoning_content` -- the canonical field. `chat-*.js` accumulates
`(delta.reasoning_content ?? "") + delta.reasoning_details[].text`; it never reads `delta.reasoning`
itself. The backend normalizes for it first: `core/inference/sse_control_frames.py`'s
`_normalize_reasoning_deltas` renames `delta.reasoning` to `delta.reasoning_content`, but **only
when `reasoning_content` is absent or blank** ("The `delta.reasoning` alias Ollama and newer vLLM
send is renamed to the canonical `reasoning_content`, streamed deltas only").

**Decision on the OpenRouter-style `reasoning` alias: NOT added.** The evidence says Unsloth cannot
benefit from it -- this server already sends `reasoning_content`, so the rename is skipped and the
alias would be dead weight on every delta; and since the client concatenates its reasoning sources,
a future client that summed `reasoning` and `reasoning_content` would double the thinking text. No
client was found that needs it, and the deliverable was explicitly "add it if and only if the
investigation shows a client needs it". If one ever turns up, it is one line in
`StreamingSink::PushSplitDelta` and the two response builders.

Request-side, Studio replays a previous turn's thought as `message.reasoning_content` on the
assistant message (`pje` in `chat-*.js` sets `v.reasoning_content = g`), which this server already
accepts -- see "`reasoning_content`"'s Multi-turn paragraph.

### (e) How image attachments are sent, and what gates the attach button

Shape (`chat-*.js`'s `hZ`): OpenAI content parts, with the image as a **`data:` URI**, never a
remote URL --

```js
t.push({type: `image_url`,
        image_url: {url: n.startsWith(`data:`) ? n : `data:image/png;base64,${n}`}});
```

alongside the ordinary `{type: "text", text: ...}` parts. The uploader accepts
`image/jpeg,image/png,image/webp,image/gif`.

The gate is `pY({isExternalModel, externalSupportsVision, ...})`, and for an external model its
whole body is `n === false || ... ? "<model> cannot accept images." : null`. `externalSupportsVision`
is `rt(providerType, modelId)`, which returns the user's per-model override if one exists, else the
catalogue answer, else `qe(providerType)` (`true` for `openai`/`anthropic`/`gemini`/`openrouter`,
`false` for `cohere`/`deepseek`/`mistral`, **`null` otherwise**). For `vllm`/`llama_cpp`/`custom`
there is no catalogue entry and no provider-wide answer, so it is `null` -- and `pY` only blocks on
an explicit `false`. **So the attach button is already open for a local connection**: Unsloth will
happily POST `image_url` parts at this server today -- see "Images" for the exact part shape this
server now accepts and how it renders. Nothing has to change in Studio for it: the shape it sends
(`data:` URI, `image_url` object) is exactly the confirmed shape "Images" documents.

### (3) The click-path to get a thinking toggle

As far as the source proves, and no further:

1. Connections -> add a connection -> choose the **Custom** option, then the **vLLM** preset
   (`providerType: "vllm"`; `connections-tab-*.js`'s preset table).
2. Base URL: `http://127.0.0.1:8080/v1` (whatever `--host`/`--port` this server was started with).
3. API key: this server does not authenticate, and the registry marks the local presets' key
   optional -- leave it blank, or type any placeholder if the form insists.
4. Model IDs: type the loaded container's own `model_id` (this checkpoint's is `Qwen/Qwen3.8-27B`,
   what `GET /v1/models` returns). `/v1/models` is served, so a "reload models" action should
   discover it; the manual field exists either way.
5. Turn ON the **"Reasoning model"** switch ("This server runs a reasoning model"). This is the
   step that produces the toggle -- it is the only input to `dr()`, and it only exists for the vLLM
   preset.

**Not verified without driving the GUI**: the exact menu wording and order of steps 1-4, whether the
form rejects an empty API key, whether the model list auto-populates from `/v1/models` for this
provider type, and where the resulting thinking toggle is rendered in the composer. Everything about
*what the toggle does to the request body*, and about the model card's values, is from the source
and is stated above.

An OpenRouter-type connection with its base URL overridden is the other setup that yields a toggle
(it sends `reasoning: {"enabled": bool}`, also supported here), but it is the worse choice: its
capability fetch is skipped (see (a)), its model picker is `model_list_mode: "curated"`, and it
sends OpenRouter attribution headers. The vLLM preset is the recommended one.

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
chunk ever gains a `usage` key. Tool-call mode ("Tool calls" above) behaves identically either way:
the end-of-generation `OnToolCalls` chunk is just another "normal chunk" for the purposes of the
`usage: null` rule.

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

## Images

Milestone 8 stage 5 (2026-09-22): `image_url`/`input_image` message content parts are no longer a
deferred-feature `400` -- see [vision.md](vision.md) for the tower/splicing design this section
wires into the OpenAI surface.

**Accepted shapes** (Stage 1's Unsloth Studio investigation, plus the two alternate shapes several
other OpenAI-compatible clients send): `{"type":"image_url","image_url":{"url":"data:..."}}` (the
confirmed real-client shape), `image_url` as a bare string in place of the `{"url":...}` object, and
the Responses-API-style `{"type":"input_image","image_url":"data:..."}` (`"image"` also accepted as
the field name there). Any position in the content array, any number of images across a single
message or a whole conversation, in any of `image/png`, `image/jpeg`, `image/gif`, `image/bmp`.

**Decoded and preprocessed at PARSE time**, not in the engine: base64-decode, format sniff,
`smart_resize` -- all pure host C++ with no HIP/`r4dx::model` dependency
(`src/vision/CMakeLists.txt`'s own "no HIP, no r4dx_model" claim), so `openai_types.cpp` does it
directly (`r4dx::server::ImagePart`/`ContentPart`, `ParseChatCompletionRequest`'s new `image_cfg`
parameter) and it stays exercisable by `tests/server/test_openai_types.cpp` with no container and
no GPU, exactly like every other request-validation rule in that file. Only the vision TOWER
forward pass (does this container even have one? run `Model::EncodeImages`) needs the loaded
`Model`, so that part happens one layer up, in `Engine::RunRequest` (`engine.cpp`).

**Remote URLs are never fetched**: an `http://`/`https://` `image_url` is a clean `400` --
*"remote image URLs are not fetched by this server (a local single-user server should not make
outbound requests on a client's behalf) -- send the image as a data: URI instead"* -- regardless of
whether the loaded container even has a vision tower (this check runs before any Model is
consulted). This is a deliberate scope decision, not a missing feature: an r4dx server has no
sandboxing, rate limiting, or SSRF protection around an outbound HTTP client, and none of the real
clients this project has actually investigated (Unsloth Studio, docs/server.md's own client-
compatibility section) send anything but a `data:` URI in the first place.

**Every other bad-input case is also a clean `400`, never a crash**: a container with no vision
tower loaded (`"this model/container has no vision tower (loaded without vision.* tensors, or
started with --vision off)"`), an unsupported/unrecognized image format (WebP included -- stb_image,
and therefore `r4dx::vision::DecodeImageBytes`, does not decode it), a malformed `data:` URI, base64
that decodes to corrupt/undecodable image bytes, oversize base64
(`kMaxImagesPerRequest`/`kMaxImageBase64Chars`, `openai_types.h` -- 8 images and ~32 MiB of base64
text per request, generous local-server ceilings rather than hardware limits), and a prompt whose
image tokens (after placeholder expansion) push the total past `--max-ctx`.

**Chat template rendering**: a message with at least one image content part renders as a real
content ARRAY (`{"type":"image"}`/`{"type":"text",...}` entries, order preserved) so
`chat_template.jinja`'s own image handling fires (`vision_start`/`image_pad`/`vision_end`,
docs/vision.md) -- every other message (the overwhelming common case) still renders as a plain
string or JSON `null`, byte-for-byte as before this stage. The template emits exactly ONE
`<|image_pad|>` per image; `r4dx::vision::ExpandImagePlaceholders`
(`src/vision/image_prompt.h` -- shared by the CLI's own `--image` path and by
`tests/vision/tool_vision_chat`) expands each single placeholder into that image's real
merged-token-count run BEFORE the prefix-reuse decision, exactly like the HF processor does; this
is also what makes `usage.prompt_tokens` count the real (post-expansion) image tokens with no
separate accounting needed.

**Prefix cache, image-aware** (`src/server/prefix_state.h`'s `PrefixState::ImageKey`, built in
Milestone 8 stage 4 and now fed from real request data): every image placeholder is the SAME token
id, so two requests carrying two DIFFERENT pictures at the same position produce byte-identical
token sequences -- token equality alone stopped being sufficient the moment images existed.
`Engine::RunRequest` fingerprints every image content part in the conversation (a 64-bit hash over
the raw, post-base64-decode bytes the client sent, `Fnv1a64` in `openai_types.cpp`) and passes the
whole list to `PrefixState::Extend`; a request whose images do not exactly extend what was already
fed falls to the `Model::Reset()` + full-reprefill path, same as a text mismatch. When the prefix
DOES extend, only images whose placeholder run lands in the NEW tail are handed to
`Model::EncodeImages` -- an already-fed image's rows are already resident in the model's real
KV/GDN state and are never re-encoded, which is the measurable point of `timings.image_n`/
`image_ms` below (absent entirely on a turn that encoded nothing new).

**The precondition on that reuse, and one thing it is NOT** (review finding re-measured,
2026-09-22): reuse holds only while re-rendering the conversation and re-tokenizing it reproduces
the exact token ids that were committed. The earlier turns come back as *client-supplied text* (an
`assistant` message whose content is the string the client received), so the server re-tokenizes
them; anything that changes those bytes on the way back -- including the client's own encoding of
the request body -- correctly costs the prefix.

The review pass reported that **any** non-ASCII character in the replayed answer breaks reuse. Two
separate things were going on, and only one of them is this server's. The first is the measuring
client: same conversation, same server, same real container, a turn-1 answer carrying 3 en-dashes,
three transports:

| how the turn-2 body was sent | turn 2 `timings.prompt_n` | reused? |
|---|---|---|
| raw UTF-8 bytes | 22 of 297 | yes -- no `image_n`, no re-encode |
| PowerShell `Invoke-WebRequest -Body <string>`, `Content-Type: application/json` | 297 of 297 | no -- full re-prefill + `image_n=1` |
| the same cmdlet with `application/json; charset=utf-8` | 22 of 297 | yes -- no `image_n` |

PowerShell 5.1 encodes a string body with the content type's charset and falls back to Latin-1 when
none is given, so every non-ASCII character arrives mangled and the re-rendered prompt genuinely
differs from what was committed. The server is right to decline; the bug is client-side. (Every
`Invoke-WebRequest` in `tools/server/smoke.ps1` now sends the charset for this reason.)

The second thing is this server's, and it is real: **the decode->encode round trip is not the
identity for every string**, so some replayed answers genuinely do not match what was committed.
It is string-specific, not "any non-ASCII" -- sweeping a correctly-encoded client over the same
image and the same turn-2 request:

| turn-1 answer (same image, same question shape) | non-ASCII chars | reused? |
|---|---|---|
| English one word / long plain ASCII / markdown | 0 | yes |
| `circle — square — triangle` (em dashes) | 2 | yes |
| a free-form English description (en dashes) | 3 | yes |
| emoji | 6 | yes |
| Korean, one sentence | 25 | yes |
| Japanese, one short sentence | 25 | yes |
| Japanese, cut off mid-generation (`finish=length`) | 137 | yes |
| **Japanese, one longer sentence (a different image)** | 84 | **no** |
| **Thai, one sentence** | 104 | **no** |

The two failures carry no U+FFFD and are not truncated (`finish=stop`, and the Thai one repeats at
both `max_tokens` 80 and 250), so they are genuine tokenizer round-trip asymmetries -- the same
class docs/status.md records on the CLI's `--chat` side (a leading-space BPE variant). En-dashes
and em-dashes, the characters the review pointed at, are NOT the trigger; longer non-Latin text
frequently is. Treat reuse across a turn as best-effort, not guaranteed.

When it does miss, the failure is graceful -- `Model::Reset()` + full re-prefill, correct output --
but with an image in the prompt it costs a full re-encode (~30-150 ms depending on image size) on
top of the re-prefill, so a client that gets long CJK/Thai answers pays roughly double on every
turn. The real fix is to dedupe re-tokenization against the raw committed token ids; it is not
vision-specific and is not done. `smoke.ps1` pins both ends: `vision multi-turn` keeps turn 1 to
one ASCII word so a failure there points at the image mechanism, and `vision multi-turn
(free-form)` replays a real Japanese answer (asserted to be non-ASCII) and asserts the two outcomes
stay CONSISTENT -- either the prefix was reused and nothing was re-encoded, or it was not and the
image was re-encoded and the whole prompt re-prefilled. The combination it exists to catch is the
third one: a reused prefix whose image rows were silently dropped.

**Response extensions** (`src/server/openai_types.h`):
- `GET /v1/models`: `architecture.input_modalities` and `modalities` gain `"image"`, and
  `capabilities` gains `"image"`, whenever the loaded container's vision tower is actually resident
  (`Model::HasVision()`) -- the one-line switch Stage 1 prepared (`ModelInputModalities`), now
  flipped from the real Model instead of a hardcoded `["text"]`.
- `timings.image_n`/`timings.image_ms`: how many images THIS request's own `EncodeImages` calls
  covered and how long they took, summed across the request. Present only when at least one image
  was actually encoded this request (the same "omit, don't zero" convention `draft_n` already
  uses) -- absent on a text-only request, and absent on a turn whose only image was already fed on
  an earlier turn (prefix reuse, above).
- `usage.prompt_tokens` already counts image tokens with no code change needed: they are part of
  the expanded token sequence by the time `usage` is computed.

**CLI parity** (`src/cli/main.cpp`): `--image <path>` (repeatable) attaches one or more images to
the next turn -- the one-shot `--prompt` itself, or (in `--chat`) whichever line is typed first;
every later `--chat` turn attaches images with one or more leading `"/image <path>"` input lines.
`--stats` prints the image encode time and the number of image tokens spliced into that turn's
prefill. The CLI keeps one `ImageBatch` (grid list + device embeddings) per turn that ever attached
an image, alive for the whole process, for the same reason the server keeps `PrefixState::
ImageKey`: a later turn's chat-template re-render still carries every earlier turn's own image
content part and has to re-expand its placeholder, but must not re-encode it.

**Testing**: `tests/server/test_openai_types.cpp` (every parse-time rule above -- accepted shapes,
remote-URL/format/corrupt-data/oversize/too-many-images rejection, content-hash determinism, the
`/v1/models` modality flip, `timings.image_n`/`image_ms` JSON shape), `tests/cli/test_args.cpp`
(`--image` parsing), and `tools/server/smoke.ps1`'s `-Vision` switch (real container only --
describing a synthetic image, reading a rendered string back off an OCR image, two images in one
request, image + tools, image + thinking, streaming, multi-turn prefix reuse with no re-encode,
the reuse-or-graceful-degradation consistency check with a real non-ASCII (Japanese) answer
replayed verbatim,
different-image-same-text NOT reusing the prefix, and the full bad-input battery; the DEFAULT run
against the 4-layer container, which has no vision tensors, instead asserts the one thing that must
ALWAYS hold -- a well-formed local image against a non-vision container is a clean 400 naming the
real reason).

## Deferred / known gaps
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

**Per-request routing (`Engine::RunRequest`, `engine.cpp`), current as of Milestone 6 stage S3**: a
request takes the MTP path iff `model_->MtpEnabled()` (the server was started with `--mtp N>0`
against an MTP container) -- this no longer depends on the request's own temperature. Within that
path, `sampling.temperature <= 0` picks `Model::DecodeStepMtpGreedy` (byte-identical to the
pre-S3 behavior: accept iff the draft equals the target's argmax) and `temperature > 0` picks
`Model::DecodeStepMtpSampled` (sample-and-match rejection sampling against the request's own
distribution, [sampling.md](sampling.md) section 9). Every request still uses one seeded
`std::mt19937_64`, unchanged either way. See the "SUPERSEDED (Milestone 6 stage S3)" note below for
the full routing description shared with DFlash2 and plain decode.

Before stage S3, acceptance meant "the draft equals the target's argmax", so there was nothing for
a sampling request to accept, and the gate above was `... AND sampling.temperature <= 0`, with every
other request falling back to plain decode. That is what stage S3 removed, once
`Model::DecodeStepMtpSampled` / `DecodeStepDflashSampled` (added in stage S2) made rejection
sampling against a sampling target possible, provably without changing a single emitted token
([sampling.md](sampling.md) section 9) -- measured 2.3-6.4 tokens per round on real requests.

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

**SUPERSEDED (Milestone 6 stage S3): speculation now runs at any temperature, not just greedy.**
Every "greedy-only gate" statement above (the MTP section's per-request routing, the DFlash2
section's `use_dflash`/`use_mtp` gate, and the sampled-traffic-pays-nothing section's premise that a
sampled request never reads a drafted token at all) described `temperature <= 0`-gated routing that
no longer exists. What is true now, in both `Engine::RunRequest` (`engine.cpp`) and `r4dx-cli`'s
`RunTurn` (`main.cpp`):

* `use_mtp` is `model_->MtpEnabled()` and `use_dflash` is "a drafter was `Model::Load`'d" -- neither
  depends on the request's `temperature` any more. A request takes the DFlash2 path when a drafter
  is loaded, else the MTP path when MTP is enabled, else plain decode -- exactly as before, just
  without the greedy restriction.
* Within a speculative path, `temperature <= 0` still takes the exact pre-S3 GREEDY method
  (`DecodeStepMtpGreedy` / `DecodeStepDflashGreedy`, byte-identical) and `temperature > 0` takes the
  new SAMPLED method (`DecodeStepMtpSampled` / `DecodeStepDflashSampled`, [sampling.md](sampling.md)
  section 9's lossless sample-and-match rejection sampling) -- one seeded `std::mt19937_64` per
  request, exactly one draw per emitted token, unchanged either way.
* Plain (non-speculative) decode also switched its `temperature > 0` path from a full-vocab
  `r4dx::kernels::Sample` + `Model::DecodeStep` loop to `Model::DecodeStepSampled` (the device-row-
  summary fast path, [sampling.md](sampling.md) section 8) -- this milestone's other half of the
  fix, since a full-vocab CPU pass every token was the other reason sampled traffic was slow. Greedy
  plain decode is untouched.
* `Model::SetDflashInjectionEnabled` is no longer toggled per-request on `temperature`: a sampled
  request now uses the drafter exactly like a greedy one does, so injection is simply on whenever a
  drafter is loaded (`use_dflash`, now temperature-independent) -- the ring-gap tolerance mechanism
  this section describes is otherwise unchanged and no longer needed for this reason (a `temperature
  <= 0`/`> 0` alternation on the same session no longer toggles injection at all), though a
  `--mtp`/`--dflash` server can still see a cold ring after a restart or the very first request.
* `r4dx-cli` no longer clears `args.mtp`/`args.dflash` at `--temperature > 0` -- the paragraph above
  saying it does ("a one-request-per-invocation process can do for free") described the old
  behavior; both flags now work at any temperature there too, identically to the server.
* The per-request stderr log line gained three fields: `temperature=`, `stream=yes/no`,
  `thinking=yes/no` -- so which decode path a request actually took (and whether it streamed) is
  visible without cross-referencing the request body.
* **Losslessness is the new gate**, replacing "acceptance is provably correct for a sampling target"
  as an argument: for a fixed seed, a sampled speculative request emits the SAME token sequence a
  plain sampled request would, modulo the pre-existing batched-verify numeric mechanism
  ([sampling.md](sampling.md) section 9.3/11). The AUTHORITATIVE check for this is the ctest
  exact-verify-row classifier (`tests/model/test_mtp.cpp`'s `CheckSampledRoundsMatchPlain`,
  `tests/model/test_dflash_e2e.cpp`'s `CheckSampledDflashMatchesPlain`): it captures the real verify
  row a mismatch used and proves the emitted token is a legitimate canonical sample of it, so it can
  tell a genuine bookkeeping bug apart from the known numeric mechanism even when nearly every
  trajectory diverges somewhere. `tools/validate_spec_sampling.ps1` (modeled on
  `tools/validate_dflash.ps1`) is a real-hardware, black-box SHA/text-diff smoke check on top of
  that -- useful for a quick real-container sanity pass and for catching a regression outside the
  ctest matrix, but at this container's high base divergence rate its cross-family control alone has
  limited power to distinguish "known mechanism" from "new bug" (see its own file comment and
  section 12.4 of [sampling.md](sampling.md)); it is not a substitute for the ctest classifier. See
  that script's own file comment and its results below.
* `tools/server/smoke.ps1` gained a seeded-`temperature=0.7`-request check (against a
  `-Mtp`/`-Dflash` server): `timings.draft_n > 0`, the same seeded request twice returns identical
  text, and -- real container only -- its text matches a same-seeded plain sampled `r4dx-cli` run.

See [sampling.md](sampling.md) section 12 for the measured cost/acceptance table and
`tools/validate_spec_sampling.ps1`'s own results table.

## CLI flags

```
r4dx-server --model <container.r4dx> --layout {mxfp4|w4a16|w4a8|bf16}
    [--tokenizer-dir <dir>] [--host <addr>] [--port N] [--max-ctx N]
    [--max-tokens-default N] [--max-queue N] [--think {on|off}] [--layers N]
    [--default-temperature F] [--default-top-p F] [--default-top-k N]
    [--default-min-p F] [--log-level {debug|info|warn|error}] [--mtp N]
    [--mtp-head-layout {bf16|layout}] [--mtp-draft-head {reduced|full}]
    [--embed-device-resident {on|off}] [--dflash <draft.r4dx>] [--dflash-k N]
    [--dflash-p-min F] [--dflash-n-min N] [--vision {auto|on|off}]
    [--image-max-pixels N]
```

`--mtp N` (default 0): see "MTP" above -- requires an MTP-converted `--model` container when N>0.

`--vision {auto|on|off}` (default `auto`) and `--image-max-pixels N` (default 1048576, i.e.
1024x1024): the vision tower, docs/vision.md. `auto` loads the container's 333 `vision.*` tensors
(0.9154 GiB measured) iff it has them, so a text-only container is unchanged; `on` fails the load
when the container has none; `off` reclaims the 0.9 GiB on a vision-capable container.
`--image-max-pixels` caps pixels PER IMAGE and **downsizes** an image above it through the same
`smart_resize` rule the reference processor uses -- it never rejects one. Both are validated and
reported at startup, and `--image-max-pixels` is exactly the `image_cfg` every `image_url` content
part is now decoded/preprocessed with (`ParseChatCompletionRequest`, see "Images" above) -- a
request against a server started with `--vision off`, or against a container with no `vision.*`
tensors at all, still gets a clean `400` naming that reason.

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
`test_prefix_state` (`PrefixState`'s prefix-match / invalidate / MTP-aware commit bookkeeping,
including `TestImageAwarePrefixReuse`, see "MTP" above and "Images"), `test_tool_call_parser` (see
"Tool calls" above -- real-capture and malformed-input cases for the model's surface syntax),
`test_tool_stream_gate` (the live tool-call stream gate and its "streamed content == non-streamed
content" property over every chunking of a dozen representative generations),
`test_reasoning_splitter` (the `</think>` split). `test_openai_types` also covers every image
content-part rule from "Images" above (accepted shapes, remote-URL/format/corrupt-data/oversize/
too-many-images rejection, content-hash determinism, the `/v1/models` modality flip,
`timings.image_n`/`image_ms`). All pass as part of the normal `.\tests\run_tests.ps1` run.

`tools/server/smoke.ps1` is the GPU integration test: starts `r4dx-server` on HIP device 1 against
the 4-layer test container (`--layout w4a16 --layers 4`, since that container's config.json still
declares 64 layers -- see `--layers` above), hits `/v1/models`, a non-streaming and a streaming
`/v1/chat/completions`, two consecutive different-prompt requests (checking the server's own stderr
log to confirm no container reload happened -- see "Reset cost" above), a remote-image-url-rejected
request and (against this default container) a local-image-rejected-with-"no vision tower" request,
and (a `-ToolRoundTrip` switch) a real tool call/result/answer multi-turn round trip
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
`<tool_call>` block), and `-Vision` for the full image suite against a real vision-capable
container (`.\tools\server\smoke.ps1 -Model D:\models\r4dx\qwen38-27b-v3.r4dx -Layout w4a16
-Layers -1 -Vision`; see "Images" above for the checks it runs). The live tool-call streaming checks ("Tool calls" above) run unconditionally:
a tool-offering streaming request asking a plain prose question must yield many `delta.content`
events whose concatenation equals the same greedy request's non-streaming `message.content`, and no
content delta may carry `<tool_call`/`</tool_call`; the per-delta arrival-time assertion (first
delta well before the end of the stream) is real-container-only, since the 4-layer container's
decode is fast enough for the whole stream to arrive in a single socket read.

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

**A malformed body could answer `500` with an EMPTY body instead of `400`** (found 2026-09-22 while
re-measuring a review finding, pre-existing and not vision-specific). `ParseJsonBody` maps a JSON
parse failure to `ApiError{400, ...}` carrying nlohmann's own message -- and for an ill-formed
UTF-8 byte that message *quotes the offending bytes*. `RespondError` then `dump()`ed that message
into the response JSON, and nlohmann's default handler throws `json::type_error` rather than
serialize invalid UTF-8. The throw happened inside the handler's own `catch` block, so it escaped
the handler entirely and cpp-httplib answered `500` with no body at all. Reproducible with any
request whose body carries a stray high byte -- which is what PowerShell 5.1 produces from a
`-Body <string>` containing non-ASCII when the content type names no charset, i.e. the exact
condition the prefix-reuse repro above hits. Fixed by dumping the error body with
`json::error_handler_t::replace`: the bad bytes become U+FFFD and the caller gets the clean
`400 invalid_request_error` with the parser's reason, like every other bad input. Nothing this
server generates itself is ever invalid UTF-8, so no well-formed response changed.
