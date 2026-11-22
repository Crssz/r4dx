// Inkling-2-Mini (arch inkling2_loop) server dialect: the typed-part output stream parser and the request-side
// normalisation (docs/inkling2-plan.md section 6). Pure CPU, no HIP: unit-tested alone (tests/server/test_inkling2_dialect.cpp).
//
// The model's generation is a sequence of PARTS over special tokens (chat_template.jinja renders history the same way):
//
//   <|message_model|> [NAME] <|content_thinking|>        reasoning text  <|end_message|>
//   <|message_model|> [NAME] <|content_text|>            answer text     <|end_message|>
//   <|message_model|> [NAME] <|content_invoke_tool_json|> {"name":"f","args":{...}} <|end_message|>
//   ... then <|content_model_end_sampling|> (200006, the EOS: the engine stops on it, the parser never sees it).
//
// The template has no generation prompt, so the first generated token is the model's own <|message_model|>.
// Inkling2StreamParser consumes the DECODED text of the stream (Tokenizer::Options::keep_special_on_decode must list the
// five markers, see Inkling2KeepSpecial()), in pieces of any size: a marker may be split across pieces at any byte (at most
// 30 bytes are ever held back, only while they are a proper prefix of a marker). Output events: reasoning text, answer
// text, and whole tool calls (tool calls are buffered until their <|end_message|>; arguments are re-serialised to the
// compact JSON string OpenAI's `function.arguments` carries).
//
// Malformed or truncated parts degrade to content, as the other dialects' parsers do: a tool-call part whose body is not
// {"name": string, "args": object} (or that never ends) is emitted as its raw source text (markers included) and
// had_malformed() is set; text outside any part (a model that skips the header) is content; a part cut off by the end of
// generation keeps what was streamed (reasoning/text) or degrades (tool call).
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nlohmann/json.hpp"
#include "tool_call_parser.h"

namespace r4dx::server {

// The special-token texts the parser needs to see in the decoded stream (ModelDialect::keep_special_on_decode).
const std::vector<std::string>& Inkling2KeepSpecial();

class Inkling2StreamParser {
 public:
  struct Event {
    enum class Kind { kReasoning, kContent, kToolCall };
    Kind kind = Kind::kContent;
    std::string text;      // kReasoning / kContent
    ParsedToolCall call;   // kToolCall
  };

  Inkling2StreamParser() = default;

  // Feeds the next piece of decoded text; returns zero or more events (concatenating all kReasoning texts and all kContent
  // texts over the whole stream gives the reasoning and the answer).
  std::vector<Event> Push(std::string_view piece);

  // Call once after the last Push (EOS, length, stop string): flushes held bytes and degrades an unfinished tool-call part.
  std::vector<Event> Finish();

  // True once at least one tool call was parsed (the server's finish_reason becomes "tool_calls").
  bool tool_call_seen() const { return tool_calls_ > 0; }
  size_t tool_call_count() const { return tool_calls_; }
  // True iff some part (a tool call, or a header with no content kind) was malformed/truncated and degraded to content.
  bool had_malformed() const { return malformed_; }
  // True while inside a <|content_thinking|> part (the engine counts usage.reasoning_tokens from it, token by token).
  bool in_thinking() const { return state_ == State::kThinking; }

 private:
  enum class State { kBetween, kHeader, kThinking, kText, kTool };
  void Drain(bool final, std::vector<Event>& out);
  void EmitText(State kind, std::string text, std::vector<Event>& out);
  void Degrade(std::string raw, std::vector<Event>& out);
  void FinishTool(std::vector<Event>& out);

  State state_ = State::kBetween;
  std::string buf_;      // unconsumed input (may end in a partial marker)
  std::string header_;   // NAME between <|message_model|> and the content-kind token
  std::string tool_body_;
  std::string raw_open_; // source text of the tool part so far (header markers), for degradation
  size_t tool_calls_ = 0;
  bool malformed_ = false;
};

// Whole-text form (non-streaming responses): runs the parser over `text` and concatenates.
struct Inkling2Parsed {
  std::string reasoning;
  std::string content;
  std::vector<ParsedToolCall> tool_calls;
  bool had_malformed = false;
};
Inkling2Parsed ParseInkling2Generation(std::string_view text);

// ---- request side ---------------------------------------------------------------------------------------------------

// The template raises on a JSON-string `tool_calls[].function.arguments`; parse every such string into a mapping (key order
// kept). "" / null become {}. Throws std::invalid_argument naming the message index when a string is not a JSON object.
void CanonicalizeInkling2Messages(nlohmann::ordered_json& messages);

// The `reasoning_effort` value to hand the template, or nullopt to leave it unset (the template then uses 0.9).
//   * chat_template_kwargs.reasoning_effort (non-null): passed through untouched (an explicit template variable wins).
//   * enable_thinking == false: "none" (renders "Thinking effort level: 0"), whatever effort was named.
//   * request_effort (top-level `reasoning_effort` or `reasoning.effort`, string or number): a string is trimmed and
//     lower-cased and must be one of none/minimal/low/medium/high/xhigh/max; a number must be in [0, 0.99]. Anything else
//     throws std::invalid_argument (an HTTP 400 for the caller).
// `chat_template_kwargs` and `request_effort` may be null/absent json.
std::optional<nlohmann::ordered_json> ResolveInkling2ReasoningEffort(const nlohmann::json& chat_template_kwargs,
                                                                      std::optional<bool> enable_thinking,
                                                                      const nlohmann::json& request_effort);

// Reads the effort field of a request body the way ResolveInkling2ReasoningEffort expects it: `reasoning.effort` wins over
// the top-level `reasoning_effort`; null json when neither is present.
nlohmann::json Inkling2RequestEffort(const nlohmann::json& body);

}  // namespace r4dx::server
