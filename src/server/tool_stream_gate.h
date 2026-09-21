// r4dx::server::ToolStreamGate -- decides, live as generation streams, how much of a tool-offering
// request's ANSWER text may still be handed to the client as `delta.content` (docs/server.md's
// "Tool calls" streaming decision). Pure text state machine, no HIP/httplib dependency, so it is
// unit-testable on CPU alone (tests/server/test_tool_stream_gate.cpp), exactly like
// reasoning_splitter.h / response_sink.h / openai_types.h.
//
// WHY this exists: a request that OFFERS tools is not a request that USES one -- a real client
// (Unsloth Studio) sends a `tools` array on literally every turn -- so buffering the whole
// generation whenever `tools` is present (this server's own previous behavior) turned every such
// turn into "fake" streaming: the client saw nothing at all until the model was completely done,
// even for a plain prose answer that never called anything. This class keeps ordinary prose
// streaming token by token and shuts the content stream off the instant a real "<tool_call>" opener
// appears; from there on Engine::RunRequest buffers and parses exactly as it always did
// (tool_call_parser.h), delivering the calls as one complete `tool_calls` batch at the end.
//
// The opener is the literal "<tool_call>" -- byte-for-byte the same constant ParseToolCalls itself
// scans for (tool_call_parser.cpp's `kOpen`), so the gate can never close on something the parser
// would not treat as the start of a call, nor stay open through something it would. Two properties
// of this checkpoint's real surface syntax (tool_call_parser.h's file comment, both VERIFIED there
// rather than assumed) make that opener the only thing worth gating on: "<tool_call>" is an added
// token flagged `"special": false`, so `skip_special_tokens` never strips it and it always reaches
// this class as ordinary detokenized text; and it is not guaranteed to arrive inside a single
// decoded piece, so the gate must assume it can be split at any byte boundary.
//
// Byte-exact holdback: at most `strlen("<tool_call>") - 1 == 10` bytes are ever held back between
// Push() calls (the longest possible proper prefix of the opener), released the instant they can no
// longer extend into a match -- the same rule, for the same reason, as ReasoningSplitter's
// "</think>" holdback. The opener is pure ASCII and every UTF-8 continuation/lead byte of a
// multi-byte character has its high bit set (>= 0x80), so the holdback boundary always falls on a
// UTF-8 character boundary as long as every `piece` passed in already does (Tokenizer::
// StreamDecoder's own contract) -- "<tool" can never be confused with part of a character.
//
// NOT trimmed here: whitespace before an opener is streamed like any other prose, because
// ParseToolCalls does not trim it either -- its `content` is the text outside every span, verbatim
// (tool_call_parser.cpp's `result.content += text.substr(pos, tc_start - pos)`). Holding those
// blank lines back would make the streamed concatenation SHORTER than the non-streaming
// `message.content` for the same generation, which is exactly the invariant this class exists to
// preserve. The one place blank lines really do disappear (the ones right after "</think>") is
// ReasoningSplitter's job, upstream of this class and identical on both paths.
//
// This class only ever sees the ANSWER half of a generation: Engine::RunRequest strips the thinking
// span first (ReasoningSplitter), exactly as it already did before tool-call parsing, so a
// "<tool_call>" the model writes INSIDE its own chain-of-thought never closes the gate -- matching
// the non-streaming path, where ParseToolCalls never sees that text either.
#pragma once

#include <cstddef>
#include <string>

namespace r4dx::server {

class ToolStreamGate {
 public:
  ToolStreamGate() = default;

  // True once a literal "<tool_call>" has been seen. Terminal: a closed gate never streams another
  // byte, because everything from the opener onward belongs to a span only the end-of-generation
  // ParseToolCalls pass can classify (a well-formed call -> `tool_calls`; a malformed one ->
  // literal content, delivered after the fact by ToolStreamRemainder below).
  bool closed() const { return closed_; }

  // Total bytes released by Push()/Finish() so far -- i.e. the length of the prefix of the answer
  // text the client has already received as content deltas. Engine::RunRequest uses this to slice
  // the same prefix back off ParseToolCalls' own `content` (see ToolStreamRemainder).
  size_t streamed_bytes() const { return streamed_; }

  // Feeds the next piece of ANSWER text; returns the text that is safe to stream to the client
  // right now (possibly empty: a piece that is entirely a potential opener prefix, or any piece at
  // all once the gate has closed). Never returns text containing -- or ending in a proper prefix of
  // -- "<tool_call>".
  std::string Push(const std::string& piece);

  // Call once after the last Push(): releases whatever is still held back. A holdback surviving to
  // this point is a proper prefix of "<tool_call>" that generation ended before completing, so the
  // span it might have started can never be a well-formed call -- ParseToolCalls keeps that text as
  // literal content, and so must the stream. Returns "" on a gate that already closed (nothing is
  // held back after that).
  std::string Finish();

 private:
  bool closed_ = false;
  std::string hold_;    // bytes that could still extend into "<tool_call>"
  size_t streamed_ = 0;
};

// The content the caller must still deliver, once generation is over, on top of everything the gate
// already streamed: the tail of ParseToolCalls' own `content` that never went out live. Factored
// out as a free function so Engine::RunRequest and tests/server/test_tool_stream_gate.cpp compute
// it the exact same way -- the test's whole point is the invariant
// `streamed_concat + ToolStreamRemainder(streamed_concat, parsed.content) == parsed.content`,
// which is what makes a streamed reply byte-identical to the non-streaming one.
//
// `parsed_content` normally starts with `streamed` (the gate only ever releases prose that lies
// outside every span, which is exactly what the parser puts at the front of `content`). If it
// somehow does not, this returns `parsed_content` whole rather than a guessed tail: a client seeing
// a duplicated prefix is a visible, debuggable glitch, whereas silently dropping content is not.
std::string ToolStreamRemainder(const std::string& streamed, const std::string& parsed_content);

}  // namespace r4dx::server
