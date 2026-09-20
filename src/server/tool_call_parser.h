// r4dx::server tool-call parsing -- turns this checkpoint's own generated tool-call surface
// syntax back into OpenAI's structured `tool_calls` response shape (docs/server.md's "Tool
// calls" section; task: docs/server.md "Deferred / known gaps" -> "Tool calls").
//
// CONFIRMED SURFACE SYNTAX, verified two ways before this file was written (not guessed -- see
// docs/server.md for the full write-up):
//   1. Read C:\AI\models\Qwen3.8-27B\chat_template.jinja directly: the "if tools" system-prompt
//      block it renders tells the model to reply in exactly this shape, and the template's own
//      assistant-message-with-tool_calls rendering path (the "render an earlier turn's tool call
//      back into the prompt" direction) round-trips through the identical shape.
//   2. Drove the real 64-layer container through r4dx-server on HIP device 1 with a real tool
//      definition and captured the verbatim output (both non-streaming and streaming; see
//      docs/server.md).
//
// This is NOT the generic "<tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>" JSON-body
// format some other Qwen checkpoints/templates use -- guessing that shape for THIS checkpoint
// would have produced a parser that never matches real output. The real shape is an XML-ish,
// per-parameter format with no JSON envelope at the outer level:
//
//   <tool_call>
//   <function=NAME>
//   <parameter=PARAM_NAME>
//   VALUE
//   </parameter>
//   ...
//   </function>
//   </tool_call>
//
// Confirmed against the real model: multiple calls in one turn are simply concatenated blocks
// (verified with a two-city weather prompt -- "...</tool_call>\n<tool_call>\n..."), and a
// `<think>...</think>` reasoning block (only emitted when the request turns thinking on) may
// precede the first `<tool_call>` -- the model's own generated text does NOT include the
// opening "<think>\n" (that is part of the prompt's generation preamble, not generated), but DOES
// include the closing "</think>\n\n" -- this parser does not special-case it: it is literal text
// outside any <tool_call>...</tool_call> span, so it lands in `content` unchanged, exactly like
// today's un-parsed behavior for a plain answer (no regression; reasoning_content extraction is
// out of this task's scope, flagged in docs/server.md as a follow-up).
//
// DETOKENIZATION HAZARD, verified (not assumed) before writing this parser: `<tool_call>`/
// `</tool_call>` are added tokens in tokenizer.json but flagged `"special": false`
// (src/tokenizer/tokenizer.h's own doc comment already said so; confirmed by reading
// src/tokenizer/bpe_tokenizer.cpp -- `Tokenizer::decode`/`StreamDecoder::push` only ever drop a
// token when `id_is_special[id]` is true, which is exactly `"special": true`). So
// `skip_special_tokens=true` (the server's default on both the streaming and non-streaming
// paths, engine.cpp) never strips these markers -- confirmed empirically too, both call shapes
// above captured them intact. `<function=...>`/`<parameter=...>` are NOT added tokens at all
// (their names are dynamic, so they can't be single tokens); they arrive as ordinary BPE'd text
// spread across several decoded pieces (confirmed in the captured SSE stream, docs/server.md) --
// this parser only ever runs on the fully-detokenized string, never assumes any tag aligns to a
// single token or decoder "piece".
//
// A parameter VALUE's type is not tagged in this wire format (it is text, not JSON) -- this
// parser uses the same convention the template's own reverse-rendering direction uses
// (chat_template.jinja: `args_value | string if args_value is string else args_value | tojson`):
// try to JSON-parse the trimmed value; a value that parses as JSON (a number, bool, null, array,
// or object) is taken as that value, anything else (including plain prose like "Boston, MA") is
// kept as a JSON string. This matches the common "Qwen3-Coder-style" tool-parser convention other
// inference servers use for the same wire format, for the same reason: OpenAI's
// `function.arguments` field must be unambiguous JSON, and the wire format alone can't
// distinguish the string "42" from the number 42 any other way.
#pragma once

#include <string>
#include <vector>

namespace r4dx::server {

struct ParsedToolCall {
  std::string name;
  // Always a syntactically valid JSON *object* string (e.g. "{}" for a call with no parameters)
  // -- OpenAI's own `function.arguments` field is a JSON-ENCODED STRING, not a JSON object (the
  // task's own explicit "this trips people up" note).
  std::string arguments_json;
};

struct ToolCallParseResult {
  // Prose outside every <tool_call>...</tool_call> span, concatenated in original order. A
  // <think>...</think> block, if present, ends up here too (see file comment). Empty ("", not
  // absent) when there is none -- callers decide null-vs-empty-string for the OpenAI wire shape.
  std::string content;
  std::vector<ParsedToolCall> tool_calls;
  // True iff at least one <tool_call> span existed but could not be parsed as this checkpoint's
  // surface syntax (missing/malformed <function=...>, no </function>, an unclosed <tool_call>,
  // prose mixed inside the span where a structural tag was expected, ...). Every such span
  // degrades to literal content (its own raw source text, tags included) rather than being
  // thrown away or crashing the caller -- this task's own robustness requirement. Callers may
  // still want to log this even though it is never surfaced to the client as an error.
  bool had_malformed_call = false;
};

// Parses `text` (a fully-detokenized generation; a leading/trailing thinking block, if any, is
// left in `content` verbatim) for zero or more <tool_call> blocks in this checkpoint's real
// surface syntax. Never throws -- every malformed block degrades to content text instead of
// raising (see ToolCallParseResult::had_malformed_call). A `text` with no "<tool_call>" substring
// at all returns `{content == text, tool_calls == {}, had_malformed_call == false}` unchanged --
// the overwhelmingly common case (no tool was called), so this is cheap to call unconditionally.
ToolCallParseResult ParseToolCalls(const std::string& text);

// Drops any parsed tool call whose name is not in `known_names` (case-sensitive, matching OpenAI
// function-name semantics). tool_choice: "none" already removes every tool definition from the
// rendered prompt before generation even starts (openai_types.h), so this only ever matters for
// tool_choice "auto"/"required"/named, where the model still hallucinated a function it was never
// given this turn -- "a call for an undefined tool" from this task's own robustness list. Each
// dropped call is NOT silently discarded: a short, clearly-synthetic note (naming the undefined
// function and its would-be arguments) is appended to `result.content` instead, so a caller
// inspecting only `content` never sees content quietly vanish. Returns the number of calls
// dropped (0 if `known_names` is empty, i.e. the caller has no tool list to check against).
int DropUnknownToolCalls(ToolCallParseResult& result, const std::vector<std::string>& known_names);

}  // namespace r4dx::server
