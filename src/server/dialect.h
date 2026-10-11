// r4dx::server::ModelDialect -- the per-model-family text surface the server needs beyond what the
// chat template renders: where reasoning opens/closes, how tool calls are delimited, which special
// tokens must survive detokenization, and a few prompt/sampling conventions (docs/gemma4-plan.md
// section 5.3's table). Pure data plus a few CPU-only helpers: no HIP, no r4dx::model dependency, so
// it is unit-testable alone (tests/server/test_dialect.cpp) like reasoning_splitter.h.
//
// Selected once (Engine::LoadAndStart, task M1-14) from `--dialect {auto|qwen35|gemma4|qwen4_exp}`. Qwen3.5 is
// the default and its entries reproduce today's hard-coded server behavior byte for byte. qwen4_exp
// (Qwen3.8-Flash-Next, docs/qwen4exp.md) has the Qwen3.5 text surface -- its chat_template.jinja is byte-identical
// to the 27B's -- and a dialect of its own only so a log line, `--dialect` and the arch cross-check can name it.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "r4dx/models_root.h"
#include "reasoning_splitter.h"

namespace r4dx::server {

enum class DialectKind { kQwen35, kGemma4, kQwen4Exp };

// The historical tokenizer directory (server_args.h / src/cli/cli_args.h defaults), <R4DX_MODELS_ROOT>\Huihui-Qwen3.8-27B-abliterated (root default E:\models): what an empty
// --tokenizer-dir resolves to for the Qwen dialect, whose ModelDialect::default_tokenizer_dir is empty.
inline std::string QwenDefaultTokenizerDir() { return r4dx::ModelsPath("Huihui-Qwen3.8-27B-abliterated"); }

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
// Same data as Qwen35Dialect() except kind/name: reasoning "<think>\n" ... "</think>", "<tool_call>" with
// <function=..><parameter=..> bodies, reasoning_effort, EOS = the tokenizer directory's generation_config.json
// (the Swift checkpoint's is [248046, 248044], identical to the 27B's), default tokenizer directory = the 27B's.
const ModelDialect& Qwen4ExpDialect();
const ModelDialect& DialectFor(DialectKind kind);

// "qwen35" / "gemma4" / "qwen4_exp" (case-sensitive). Returns false for anything else (including "auto").
bool ParseDialectName(std::string_view s, DialectKind* out);

// `--dialect auto`: Gemma iff tokenizer_config.json says `tokenizer_class == "GemmaTokenizer"` or
// `has_turn_token` (the vocab contains "<|turn>"); otherwise Qwen. Malformed JSON counts as "no
// information" (falls back on `has_turn_token`).
DialectKind DetectDialect(std::string_view tokenizer_config_json, bool has_turn_token);

// The dialect a run uses (shared by r4dx-server's Engine and r4dx-cli, task M1-14): `requested` (an
// explicit --dialect) wins; otherwise, when `tokenizer_dir` is non-empty, DetectDialect over its
// tokenizer_config.json and a scan of tokenizer.json for the "<|turn>" token; otherwise (no tokenizer
// dir given) `container_is_gemma` -- the container header's architecture.
DialectKind ResolveDialectKind(std::optional<DialectKind> requested, const std::string& tokenizer_dir,
                               bool container_is_gemma);

// ResolveDialectKind with the container's architecture string (ArchName(DetectArch(container)): "qwen3_5",
// "gemma4_unified", "qwen4_exp"; empty/unknown = Qwen3.5) instead of the bool. Same rules, plus: a "qwen4_exp"
// container resolves to kQwen4Exp whenever the tokenizer (or its absence) does not say Gemma. A Gemma
// tokenizer on a qwen4_exp container still resolves to kGemma4, which CheckDialectAgainstArch then refuses.
DialectKind ResolveDialectKindForArch(std::optional<DialectKind> requested, const std::string& tokenizer_dir,
                                      std::string_view container_arch);

// The fail-fast message for the qwen4_exp dialect when no --tokenizer-dir was given and the default (the 27B's)
// tokenizer directory has no tokenizer.json: `resolved_dir` is ResolveTokenizerDir's answer, `user_given` whether
// --tokenizer-dir named it. "" in every other case (other dialects, an explicit directory -- the tokenizer
// loader's own error names that one -- or a directory that is there), so Qwen3.5 and Gemma behave as before.
std::string CheckDefaultTokenizerDir(const ModelDialect& d, const std::string& resolved_dir, bool user_given);

// What r4dx-cli / r4dx-server were asked for that a qwen4_exp container cannot do (docs/qwen4exp-plan.md 4.11 item 4:
// the model owns its two-card split, so no --tp/--pp; milestone 1 has no MTP head, DFlash drafter or vision tower).
struct Qwen4ExpFlags {
  int64_t tp = 1;           // --tp
  int64_t pp = -1;          // --pp (-1: not given, follows R4DX_PP, which the loader refuses on its own)
  int64_t mtp = 0;          // --mtp K
  bool dflash = false;      // --dflash <container> given
  bool vision_on = false;   // --vision on
  bool images = false;      // --image given (CLI only)
};
// "" when every flag is acceptable, otherwise one fail-fast message naming every refused flag. Run before the model
// load (the loader throws too, later and less precisely).
std::string CheckQwen4ExpFlags(const Qwen4ExpFlags& f);

// `requested` when non-empty; else the dialect's default_tokenizer_dir, else (Qwen) the historical
// QwenDefaultTokenizerDir().
std::string ResolveTokenizerDir(const ModelDialect& d, const std::string& requested);

// Cross-check a chosen dialect against the loaded model's architecture string (TextModel::Config(),
// task M1-24; empty means unknown and always passes). Returns "" when consistent, otherwise a
// fail-fast error message.
std::string CheckDialectAgainstArch(const ModelDialect& d, std::string_view model_arch);

}  // namespace r4dx::server
