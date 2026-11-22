#include "dialect.h"

#include <fstream>
#include <iterator>

#include "inkling2_stream_parser.h"
#include "nlohmann/json.hpp"

namespace r4dx::server {

namespace {

bool EndsWith(std::string_view s, std::string_view suffix) {
  return !suffix.empty() && s.size() >= suffix.size() &&
         s.substr(s.size() - suffix.size()) == suffix;
}

ModelDialect MakeQwen35() {
  ModelDialect d;
  d.kind = DialectKind::kQwen35;
  d.name = "qwen35";
  d.reasoning.open = "";
  d.reasoning.close = "</think>";
  d.reasoning.skip_blank_after_close = true;
  d.prompt_reasoning_open = "<think>\n";
  d.prompt_reasoning_closed = "";
  d.tool_open = "<tool_call>";
  d.tool_close = "</tool_call>";
  d.chat_template_polyfills = true;
  d.bos_on_raw_prompt = false;
  d.has_reasoning_effort = true;
  return d;
}

ModelDialect MakeQwen4Exp() {
  // The 27B's text surface, byte for byte: Swift-1.5-Qwen3.8-Flash-Next's chat_template.jinja has the same
  // SHA-256 as Huihui-Qwen3.8-27B-abliterated's (tests/server/test_q4exp_template.cpp pins it when both exist),
  // and so does its generation_config.json (eos [248046, 248044]), which the tokenizer directory supplies.
  ModelDialect d = MakeQwen35();
  d.kind = DialectKind::kQwen4Exp;
  d.name = "qwen4_exp";
  return d;
}

ModelDialect MakeGemma4() {
  ModelDialect d;
  d.kind = DialectKind::kGemma4;
  d.name = "gemma4";
  d.reasoning.open = "<|channel>thought\n";
  d.reasoning.close = "<channel|>";
  d.reasoning.skip_blank_after_close = true;
  d.prompt_reasoning_open = "<|channel>thought\n";
  d.prompt_reasoning_closed = "<|channel>thought\n<channel|>";
  d.tool_open = "<|tool_call>";
  d.tool_close = "<tool_call|>";
  d.keep_special_on_decode = {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|\"|>"};
  d.chat_template_polyfills = false;
  d.bos_on_raw_prompt = true;
  d.has_reasoning_effort = false;
  d.eos_ids = {1, 106, 50};  // generation_config.json: <eos>, <turn|>, <|tool_response>
  d.default_tokenizer_dir = r4dx::ModelsPath("Huihui-gemma-4-12B-it-abliterated-tok");
  return d;
}

ModelDialect MakeLfm2() {
  ModelDialect d;
  d.kind = DialectKind::kLfm2;
  d.name = "lfm2";
  // The generation prompt ends with an open "<think>" (no newline, a non-special added token): the model
  // generates "...reasoning</think>answer" (docs/lfm2-plan.md 5.3). Only the close tag is ever scanned.
  d.reasoning.open = "";
  d.reasoning.close = "</think>";
  d.reasoning.skip_blank_after_close = true;
  d.prompt_reasoning_open = "<think>";
  d.prompt_reasoning_closed = "";
  // Both are non-special added tokens (always rendered by decode), so keep_special_on_decode stays empty.
  d.tool_open = "<|tool_call_start|>";
  d.tool_close = "<|tool_call_end|>";
  d.chat_template_polyfills = false;  // minja's caps probe misreads the template; polyfills would rewrite tool_calls
  d.bos_on_raw_prompt = true;         // trained with a leading <|startoftext|>; the tokenizer never adds it
  d.has_reasoning_effort = false;
  d.forces_thinking = true;
  // generation_config.json / the model card: temperature 0.1, top_k 50, repetition_penalty 1.05.
  d.sampling.temperature = 0.1f;
  d.sampling.top_k = 50;
  d.sampling.repetition_penalty = 1.05f;
  d.eos_ids = {124900};  // generation_config.json: <|im_end|>
  // The checkpoint directory is the tokenizer directory (tokenizer.json, tokenizer_config.json,
  // chat_template.jinja, generation_config.json all sit next to the safetensors).
  d.default_tokenizer_dir = r4dx::ModelsPath("LiquidAI/LFM2.5-24B-A2B");
  return d;
}

ModelDialect MakeNanbeige() {
  // Qwen's text surface (docs/nanbeige-plan.md 5.3; read from the template, N0-6): the generation prompt ends
  // "<think>\n" (thinking on) or "<think>\n\n</think>\n\n" (off), tool calls are "<tool_call>" XML.
  ModelDialect d = MakeQwen35();
  d.kind = DialectKind::kNanbeige;
  d.name = "nanbeige";
  d.chat_template_polyfills = false;  // tests/tokenizer/golden_nanbeige.json renders exactly without them
  d.bos_on_raw_prompt = true;         // the tokenizer's post-processor prepends <|im_start|> to a raw prompt
  d.has_reasoning_effort = false;     // the template reads enable_thinking / preserve_thinking / tool_call_format
  d.eos_ids = {166101};               // generation_config.json: <|im_end|>
  d.default_tokenizer_dir = r4dx::ModelsPath("Nanbeige/Nanbeige4.3-33B-tok");
  return d;
}

ModelDialect MakeInkling2() {
  // docs/inkling2-plan.md section 6; every fact below is read from the checkpoint's chat_template.jinja / generation_config.json.
  ModelDialect d;
  d.kind = DialectKind::kInkling2;
  d.name = "inkling2";
  d.structured_parts = true;
  d.numeric_reasoning_effort = true;
  // Not used (the typed-part parser replaces the splitter/gate); kept consistent for code that prints them.
  d.reasoning.open = "<|message_model|><|content_thinking|>";
  d.reasoning.close = "<|end_message|>";
  d.reasoning.skip_blank_after_close = false;
  d.prompt_reasoning_open = "";
  d.prompt_reasoning_closed = "";
  d.tool_open = "<|content_invoke_tool_json|>";
  d.tool_close = "<|end_message|>";
  d.keep_special_on_decode = Inkling2KeepSpecial();
  d.chat_template_polyfills = false;  // tests/tokenizer/chat_inkling2_golden.json renders exactly without them
  d.bos_on_raw_prompt = false;        // tokenizer.json has no post-processor BOS; the template never emits <|begin_of_text|>
  d.has_reasoning_effort = true;      // `reasoning_effort`: level name or number (ResolveInkling2ReasoningEffort), not Qwen's 3 levels
  d.sampling.temperature = 1.0f;      // generation_config.json: do_sample true, temperature 1.0 (no top_k/top_p/penalty)
  d.eos_ids = {200006};               // generation_config.json: <|content_model_end_sampling|>
  d.default_tokenizer_dir = r4dx::ModelsPath("OpenUncensored/Inkling-2-Mini-Uncensored");
  return d;
}

}  // namespace

const ModelDialect& Qwen35Dialect() {
  static const ModelDialect d = MakeQwen35();
  return d;
}

const ModelDialect& Gemma4Dialect() {
  static const ModelDialect d = MakeGemma4();
  return d;
}

const ModelDialect& Lfm2Dialect() {
  static const ModelDialect d = MakeLfm2();
  return d;
}

const ModelDialect& Qwen4ExpDialect() {
  static const ModelDialect d = MakeQwen4Exp();
  return d;
}

const ModelDialect& NanbeigeDialect() {
  static const ModelDialect d = MakeNanbeige();
  return d;
}

const ModelDialect& Inkling2Dialect() {
  static const ModelDialect d = MakeInkling2();
  return d;
}

const ModelDialect& DialectFor(DialectKind kind) {
  switch (kind) {
    case DialectKind::kGemma4:
      return Gemma4Dialect();
    case DialectKind::kLfm2:
      return Lfm2Dialect();
    case DialectKind::kQwen4Exp:
      return Qwen4ExpDialect();
    case DialectKind::kNanbeige:
      return NanbeigeDialect();
    case DialectKind::kInkling2:
      return Inkling2Dialect();
    case DialectKind::kQwen35:
      break;
  }
  return Qwen35Dialect();
}

bool ParseDialectName(std::string_view s, DialectKind* out) {
  if (s == "qwen35") {
    if (out) *out = DialectKind::kQwen35;
    return true;
  }
  if (s == "gemma4") {
    if (out) *out = DialectKind::kGemma4;
    return true;
  }
  if (s == "lfm2") {
    if (out) *out = DialectKind::kLfm2;
    return true;
  }
  if (s == "qwen4_exp") {
    if (out) *out = DialectKind::kQwen4Exp;
    return true;
  }
  if (s == "nanbeige") {
    if (out) *out = DialectKind::kNanbeige;
    return true;
  }
  if (s == "inkling2") {
    if (out) *out = DialectKind::kInkling2;
    return true;
  }
  return false;
}

DialectKind DetectDialect(std::string_view tokenizer_config_json, bool has_turn_token, bool has_tool_call_token,
                          bool has_inkling2_tokens) {
  bool gemma = has_turn_token;
  bool lfm2_bos = false, nanbeige = false;
  const nlohmann::json j = nlohmann::json::parse(tokenizer_config_json.begin(), tokenizer_config_json.end(),
                                                 nullptr, /*allow_exceptions=*/false);
  if (j.is_object()) {
    auto it = j.find("tokenizer_class");
    if (it != j.end() && it->is_string() && it->get<std::string>() == "GemmaTokenizer") gemma = true;
    auto bos = j.find("bos_token");
    if (bos != j.end() && bos->is_string() && bos->get<std::string>() == "<|startoftext|>") lfm2_bos = true;
    nanbeige = it != j.end() && it->is_string() && it->get<std::string>() == "LlamaTokenizer" && bos != j.end() &&
               bos->is_string() && bos->get<std::string>() == "<|im_start|>";
  }
  if (gemma) return DialectKind::kGemma4;
  if (lfm2_bos && has_tool_call_token) return DialectKind::kLfm2;
  if (has_inkling2_tokens) return DialectKind::kInkling2;
  if (nanbeige) return DialectKind::kNanbeige;
  return DialectKind::kQwen35;
}

namespace {

std::string ReadFileIfPresent(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::string JoinPath(const std::string& dir, const char* name) {
  if (dir.empty()) return name;
  const char last = dir.back();
  return (last == '/' || last == '\\') ? dir + name : dir + "/" + name;
}

}  // namespace

DialectKind ResolveDialectKind(std::optional<DialectKind> requested, const std::string& tokenizer_dir,
                               bool container_is_gemma, bool container_is_lfm2) {
  return ResolveDialectKindForArch(requested, tokenizer_dir,
                                   container_is_gemma ? "gemma4_unified" : container_is_lfm2 ? "lfm2_moe" : "");
}

DialectKind ResolveDialectKindForArch(std::optional<DialectKind> requested, const std::string& tokenizer_dir,
                                      std::string_view container_arch) {
  if (requested) return *requested;
  const bool container_is_gemma = container_arch.substr(0, 5) == "gemma";
  const bool container_is_lfm2 = ArchNameIsLfm2(container_arch);
  const DialectKind qwen_kind = container_arch == "qwen4_exp" ? DialectKind::kQwen4Exp : DialectKind::kQwen35;
  if (tokenizer_dir.empty()) {
    if (container_arch == "llama") return DialectKind::kNanbeige;
    if (ArchNameIsInkling2(container_arch)) return DialectKind::kInkling2;
    return container_is_gemma ? DialectKind::kGemma4 : container_is_lfm2 ? DialectKind::kLfm2 : qwen_kind;
  }
  const std::string config = ReadFileIfPresent(JoinPath(tokenizer_dir, "tokenizer_config.json"));
  // The (tens of MB) vocab is only scanned when the config alone did not say Gemma.
  if (DetectDialect(config, /*has_turn_token=*/false) == DialectKind::kGemma4) return DialectKind::kGemma4;
  const std::string vocab = ReadFileIfPresent(JoinPath(tokenizer_dir, "tokenizer.json"));
  const bool has_turn_token = vocab.find("\"<|turn>\"") != std::string::npos;
  const bool has_tool_call_token = vocab.find("\"<|tool_call_start|>\"") != std::string::npos;
  const bool has_inkling2_tokens = vocab.find("\"<|message_model|>\"") != std::string::npos &&
                                   vocab.find("\"<|content_model_end_sampling|>\"") != std::string::npos;
  const DialectKind detected = DetectDialect(config, has_turn_token, has_tool_call_token, has_inkling2_tokens);
  // The tokenizer says Gemma or LFM2: that wins. Otherwise it is a Qwen tokenizer, and the container picks which Qwen dialect.
  return detected == DialectKind::kQwen35 ? qwen_kind : detected;
}

std::string ResolveTokenizerDir(const ModelDialect& d, const std::string& requested) {
  if (!requested.empty()) return requested;
  return d.default_tokenizer_dir.empty() ? QwenDefaultTokenizerDir() : d.default_tokenizer_dir;
}

bool ArchNameIsLfm2(std::string_view model_arch) { return model_arch.substr(0, 4) == "lfm2"; }

bool ArchNameIsInkling2(std::string_view model_arch) { return model_arch == "inkling2_loop"; }

std::string CheckDefaultTokenizerDir(const ModelDialect& d, const std::string& resolved_dir, bool user_given) {
  if (d.kind != DialectKind::kQwen4Exp || user_given) return "";
  if (std::ifstream(JoinPath(resolved_dir, "tokenizer.json"), std::ios::binary)) return "";
  return "qwen4_exp has no tokenizer of its own to default to: it uses the Qwen3.5 tokenizer, whose default directory '" +
         resolved_dir + "' has no tokenizer.json (set R4DX_MODELS_ROOT or pass --tokenizer-dir <dir> with the "
         "Huihui-Qwen3.8-27B-abliterated or base Qwen3.8-27B tokenizer files; the Swift checkpoint's own "
         "tokenizer.json is not loadable by r4dx)";
}

Qwen4ExpFlags MakeQwen4ExpFlags(int64_t tp, int64_t pp, int64_t mtp, bool dflash_given, bool vision_on, bool images) {
  Qwen4ExpFlags f;
  f.tp = tp;
  f.pp = pp;
  f.mtp = mtp;
  f.dflash = dflash_given;
  f.vision_on = vision_on;
  f.images = images;
  return f;
}

std::string CheckQwen4ExpFlags(const Qwen4ExpFlags& f) {
  std::string refused;
  const auto add = [&](const std::string& what) { refused += (refused.empty() ? "" : "; ") + what; };
  if (f.tp != 1) {
    add("--tp " + std::to_string(f.tp) + " (qwen4_exp splits its own layers over the two cards; there is no tensor parallelism)");
  }
  if (f.pp == 2) add("--pp 2 (qwen4_exp already runs as a two-stage pipeline of its own; there is no --pp)");
  if (f.mtp > 0) add("--mtp " + std::to_string(f.mtp) + " (qwen4_exp has no MTP head)");
  if (f.dflash) add("--dflash (there is no DFlash drafter for qwen4_exp)");
  if (f.vision_on) add("--vision on (qwen4_exp is text only)");
  if (f.images) add("--image (qwen4_exp is text only)");
  if (refused.empty()) return "";
  return "qwen4_exp does not support: " + refused + " (docs/qwen4exp.md, \"Known limits\")";
}

std::string CheckDialectAgainstArch(const ModelDialect& d, std::string_view model_arch) {
  if (model_arch.empty()) return "";
  const bool arch_is_gemma = model_arch.substr(0, 5) == "gemma";
  const bool arch_is_lfm2 = ArchNameIsLfm2(model_arch);
  if (d.kind == DialectKind::kGemma4 && !arch_is_gemma) {
    return std::string("dialect gemma4 does not match model architecture '") + std::string(model_arch) + "'";
  }
  if (d.kind == DialectKind::kLfm2 && !arch_is_lfm2) {
    return std::string("dialect lfm2 does not match model architecture '") + std::string(model_arch) + "'";
  }
  // The two Qwen dialects share one text surface, so either serves any Qwen architecture (qwen3_5 or qwen4_exp);
  // a Gemma or LFM2 container is refused (and told which dialect it needs).
  if (d.kind != DialectKind::kGemma4 && arch_is_gemma) {
    return std::string("dialect ") + d.name + " does not match model architecture '" + std::string(model_arch) +
           "' (pass --dialect gemma4)";
  }
  if ((d.kind == DialectKind::kQwen35 || d.kind == DialectKind::kQwen4Exp) && arch_is_lfm2) {
    return std::string("dialect ") + d.name + " does not match model architecture '" + std::string(model_arch) +
           "' (pass --dialect lfm2)";
  }
  // Nanbeige is the dense llama arch's dialect and nothing else's: its EOS ids and BOS are this tokenizer's.
  const bool arch_is_llama = model_arch == "llama";
  if (d.kind == DialectKind::kNanbeige && !arch_is_llama) {
    return std::string("dialect nanbeige does not match model architecture '") + std::string(model_arch) + "'";
  }
  if (d.kind != DialectKind::kNanbeige && arch_is_llama) {
    return std::string("dialect ") + d.name + " does not match model architecture 'llama' (pass --dialect nanbeige)";
  }
  // Inkling-2 likewise: the typed-part output, EOS 200006 and the effort-message template are this checkpoint's alone.
  const bool arch_is_inkling2 = ArchNameIsInkling2(model_arch);
  if (d.kind == DialectKind::kInkling2 && !arch_is_inkling2) {
    return std::string("dialect inkling2 does not match model architecture '") + std::string(model_arch) + "'";
  }
  if (d.kind != DialectKind::kInkling2 && arch_is_inkling2) {
    return std::string("dialect ") + d.name + " does not match model architecture 'inkling2_loop' (pass --dialect inkling2)";
  }
  return "";
}

bool ModelDialect::ReasoningOpenInPrompt(bool enable_thinking, std::string_view rendered_prompt) const {
  return enable_thinking && EndsWith(rendered_prompt, prompt_reasoning_open);
}

ReasoningSplitter::StartState ModelDialect::ReasoningStart(bool enable_thinking,
                                                           std::string_view rendered_prompt) const {
  using S = ReasoningSplitter::StartState;
  // Inkling-2 has no ReasoningSplitter flow (structured_parts): everything is answer text as far as it is concerned.
  if (kind == DialectKind::kInkling2) return S::kAnswer;
  if (kind == DialectKind::kGemma4) {
    return ReasoningOpenInPrompt(enable_thinking, rendered_prompt) ? S::kInReasoning : S::kExpectOpener;
  }
  // Qwen and LFM2: the opener is in the prompt, generated text starts inside the reasoning span (LFM2
  // always thinks: forces_thinking makes enable_thinking true upstream).
  return enable_thinking ? S::kInReasoning : S::kAnswer;
}

// Inkling-2: always empty. The template has no generation prompt, so a rendered prompt ends with the last message's
// <|end_message|> and the model's reply (its own <|message_model|> ... parts, then <|content_model_end_sampling|>) starts
// strictly AFTER it. A later turn replays the conversation with that reply appended (reasoning dropped or kept, tool-call
// JSON re-sorted by the template), so every difference sits after the previous prompt's last token: the previous prompt is
// always a token prefix of the next one, and a checkpoint at the prompt end is always reusable (ckpt_back 0). Qwen and LFM2
// differ because their prompts END inside the assistant turn ("<think>\n" / "<think>"), which a replay rewrites. The
// "Thinking effort level" system message sits before the FIRST non-system message (not at the tail), so it only breaks the
// prefix when the effort itself changes between turns.
std::string ModelDialect::CheckpointSuffix(bool /*enable_thinking*/, std::string_view rendered_prompt) const {
  if (kind == DialectKind::kInkling2) return "";
  if (kind == DialectKind::kGemma4) {
    // Thinking on ends "<|turn>model\n" (no reasoning suffix) or, after a tool response,
    // "<|channel>thought\n" (replays identically, so no holdback either): 0 tokens.
    if (EndsWith(rendered_prompt, prompt_reasoning_closed)) return prompt_reasoning_closed;
    return "";
  }
  return EndsWith(rendered_prompt, prompt_reasoning_open) ? prompt_reasoning_open : "";
}

}  // namespace r4dx::server
