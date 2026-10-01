// r4dx::server::ModelDialect -- the per-model-family text surface the server needs beyond what the
// chat template renders: where reasoning opens/closes, how tool calls are delimited, which special
// tokens must survive detokenization, and a few prompt/sampling conventions (docs/gemma4-plan.md
// section 5.3's table). Pure data plus a few CPU-only helpers: no HIP, no r4dx::model dependency, so
// it is unit-testable alone (tests/server/test_dialect.cpp) like reasoning_splitter.h.
//
// Selected once (Engine::LoadAndStart, task M1-14) from `--dialect {auto|qwen35|gemma4}`. Qwen3.5 is
// the default and its entries reproduce today's hard-coded server behavior byte for byte.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "reasoning_splitter.h"

namespace r4dx::server {

enum class DialectKind { kQwen35, kGemma4 };

struct ModelDialect {
  DialectKind kind = DialectKind::kQwen35;
  const char* name = "qwen35";

  // Reasoning markers. Qwen: opener is part of the prompt ("<think>\n"), only the close tag is ever
  // scanned. Gemma: the model generates "<|channel>thought\n" ... "<channel|>" itself.
  ReasoningSyntax reasoning;
  // The text a rendered prompt ends with when thinking is ON and the reasoning span is already open
  // in the prompt: Qwen "<think>\n", Gemma "<|channel>thought\n" (only after a tool response).
  std::string prompt_reasoning_open;
  // Gemma, thinking OFF: the template appends this empty, pre-closed span to the generation prompt.
  std::string prompt_reasoning_closed;

  // Tool-call delimiters (the gate's opener and the parser's span markers).
  std::string tool_open;
  std::string tool_close;

  // Special tokens whose text must reach the splitter/parser (decode keep-list; empty for Qwen,
  // whose markers are non-special added tokens).
  std::vector<std::string> keep_special_on_decode;

  bool chat_template_polyfills = true;   // minja polyfills (must be off for Gemma)
  bool bos_on_raw_prompt = false;        // /v1/completions prepends bos_id()
  bool has_reasoning_effort = true;      // template understands `reasoning_effort`
  // Default EOS ids (generation_config.json); empty means "use the tokenizer's own list" (Qwen).
  std::vector<int32_t> eos_ids;
  // Default tokenizer directory (empty: the server's historical hard-coded default).
  std::string default_tokenizer_dir;

  // Where generated text starts relative to reasoning, given the resolved `enable_thinking` and the
  // rendered prompt. Qwen thinking on: kInReasoning (prompt ends "<think>\n"); off: kAnswer.
  // Gemma thinking on: kInReasoning when the prompt ends with `prompt_reasoning_open` (after a tool
  // response), else kExpectOpener; thinking off: kExpectOpener as well (the pre-closed span is in the
  // prompt, so the model normally answers directly, but a stray opener is still split cleanly).
  ReasoningSplitter::StartState ReasoningStart(bool enable_thinking, std::string_view rendered_prompt) const;

  // True iff the rendered prompt ends with an open reasoning span (OnStart's
  // `reasoning_open_in_prompt`).
  bool ReasoningOpenInPrompt(bool enable_thinking, std::string_view rendered_prompt) const;

  // Text suffix of the rendered prompt that the engine's checkpoint logic must stay clear of (the
  // text whose token count is `ckpt_back`, task M1-14): Qwen thinking on "<think>\n" (1 token),
  // Gemma thinking off "<|channel>thought\n<channel|>" (expected 4 tokens). Empty when none applies;
  // the caller encodes it to count tokens.
  std::string CheckpointSuffix(bool enable_thinking, std::string_view rendered_prompt) const;
};

const ModelDialect& Qwen35Dialect();
const ModelDialect& Gemma4Dialect();
const ModelDialect& DialectFor(DialectKind kind);

// "qwen35" / "gemma4" (case-sensitive). Returns false for anything else (including "auto").
bool ParseDialectName(std::string_view s, DialectKind* out);

// `--dialect auto`: Gemma iff tokenizer_config.json says `tokenizer_class == "GemmaTokenizer"` or
// `has_turn_token` (the vocab contains "<|turn>"); otherwise Qwen. Malformed JSON counts as "no
// information" (falls back on `has_turn_token`).
DialectKind DetectDialect(std::string_view tokenizer_config_json, bool has_turn_token);

// Cross-check a chosen dialect against the loaded model's architecture string (TextModel::Config(),
// task M1-24; empty means unknown and always passes). Returns "" when consistent, otherwise a
// fail-fast error message.
std::string CheckDialectAgainstArch(const ModelDialect& d, std::string_view model_arch);

}  // namespace r4dx::server
