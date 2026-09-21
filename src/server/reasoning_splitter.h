// r4dx::server::ReasoningSplitter -- splits a model's raw generated text at the first "</think>"
// close tag into a reasoning span and an answer span (the DeepSeek/vLLM `reasoning_content`
// convention, docs/server.md's "reasoning_content" section). Pure text state machine, no HIP/
// r4dx::model dependency, so it is unit-testable on CPU alone
// (tests/server/test_reasoning_splitter.cpp), exactly like response_sink.h/openai_types.h.
//
// The model's own opening "<think>\n" is part of the PROMPT's generation preamble, never generated
// text (engine.cpp only ever turns thinking on/off via `enable_thinking`, which controls whether
// the prompt ends with an open or a pre-closed think block -- see chat_template.h) -- so this class
// never scans for an opening tag, only the closing one, and starts in `Mode::kReasoning`
// unconditionally.
//
// One instance is fed EITHER piece-by-piece as generation proceeds (Tokenizer::StreamDecoder::
// push's return value, response_sink.h's OnToken contract) OR the entire buffered generation in a
// single Push() call (engine.cpp's tool-call-mode whole-buffer split, RunRequest's `tool_mode`
// block) -- the same state machine serves both, so the splitting logic lives in exactly one place
// rather than being smeared across BufferingSink/StreamingSink/Engine::RunRequest.
//
// Byte-exact holdback: at most `strlen("</think>") - 1 == 7` bytes are ever held back between
// Push() calls (the longest possible proper prefix of "</think>"), flushed the instant they can no
// longer extend into a match. Since "</think>" is pure ASCII and every UTF-8 continuation/lead byte
// of a multi-byte character has its high bit set (>= 0x80), no substring of "</think>" can ever be
// confused with part of a multi-byte character -- so this holdback boundary always falls on a
// UTF-8 character boundary as long as every `piece` passed in already does (Tokenizer::
// StreamDecoder's own contract). Multi-byte content on either side of the tag is therefore never
// split mid-character.
//
// Trimming policy (deliberately asymmetric -- see docs/server.md): this class only ever (a) strips
// the "</think>" tag itself and (b) skips blank lines (a run of '\n'/'\r') immediately following
// it, so streaming reasoning_content/content deltas concatenate back to the untrimmed source text
// exactly except for that one boundary. It does NOT trim leading/trailing whitespace off the
// reasoning span itself -- callers that want the non-streaming "trim the whole span" behavior
// (openai_types.h's message.reasoning_content) do that once, themselves, on the concatenation of
// every kReasoning event (see BufferingSink::OnDone / Engine::RunRequest's tool_mode block).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace r4dx::server {

class ReasoningSplitter {
 public:
  // One classified fragment of a Push()/Finish() call's input. `is_reasoning` tags which half of
  // the split this text belongs to; a single Push() call can return 0, 1, or 2 events (2 only when
  // one piece itself straddles the close tag: the reasoning tail, then the answer head).
  struct Event {
    bool is_reasoning;
    std::string text;
  };

  enum class Mode { kReasoning, kAnswer };

  ReasoningSplitter() = default;

  Mode mode() const { return mode_; }

  // Feeds the next piece of raw generated text. Returns zero or more classified events, already
  // stripped of the "</think>" tag itself and (answer side only) any blank lines immediately
  // following it. Once `mode()` has become kAnswer, every subsequent Push() is a pure passthrough
  // (no tag re-scanning) -- a literal "</think>" appearing in the real answer text is not
  // special-cased, matching tool_call_parser.h's own note that this checkpoint's generated text
  // carries at most one such tag.
  std::vector<Event> Push(const std::string& piece);

  // Call once after the last Push(): flushes whatever is still held back internally. If the close
  // tag was never seen (`mode() == kReasoning`), the held bytes are genuine trailing reasoning text
  // (they could not have matched "</think>", or Push would already have consumed them) -- this is
  // the "never-closed" case (docs/server.md: content stays empty, everything generated is
  // reasoning). If the tag WAS seen, any held bytes can only be from the leading-blank-line skip
  // (generation ended entirely inside the blank lines right after the tag) and are correctly
  // dropped, never surfaced as content.
  std::vector<Event> Finish();

 private:
  void ProcessAnswer(const std::string& buf, std::vector<Event>& events);

  Mode mode_ = Mode::kReasoning;
  std::string tag_hold_;         // reasoning-side: bytes that could still extend into "</think>"
  std::string nl_hold_;          // answer-side: leading blank-line bytes not yet ruled final
  bool answer_started_ = false;  // true once a non-newline answer byte has been emitted
};

// Trims leading/trailing ASCII whitespace (space, tab, '\n', '\r') -- the non-streaming
// `message.reasoning_content` trim rule (docs/server.md's "reasoning_content" section, task item
// 5a). Exposed here, rather than folded into BufferingSink, because Engine::RunRequest's
// tool-call-mode one-shot split (RunRequest's `tool_mode` block) needs the exact same trim applied
// to the exact same concatenation-of-kReasoning-events text that BufferingSink::OnDone produces.
std::string TrimReasoningWhitespace(const std::string& s);

}  // namespace r4dx::server
