#include "dialect.h"

#include <fstream>
#include <iterator>

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
  d.default_tokenizer_dir = "D:\\models\\Huihui-gemma-4-12B-it-abliterated-tok";
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

const ModelDialect& DialectFor(DialectKind kind) {
  return kind == DialectKind::kGemma4 ? Gemma4Dialect() : Qwen35Dialect();
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
  return false;
}

DialectKind DetectDialect(std::string_view tokenizer_config_json, bool has_turn_token) {
  bool gemma = has_turn_token;
  const nlohmann::json j = nlohmann::json::parse(tokenizer_config_json.begin(), tokenizer_config_json.end(),
                                                 nullptr, /*allow_exceptions=*/false);
  if (j.is_object()) {
    auto it = j.find("tokenizer_class");
    if (it != j.end() && it->is_string() && it->get<std::string>() == "GemmaTokenizer") gemma = true;
  }
  return gemma ? DialectKind::kGemma4 : DialectKind::kQwen35;
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
                               bool container_is_gemma) {
  if (requested) return *requested;
  if (tokenizer_dir.empty()) return container_is_gemma ? DialectKind::kGemma4 : DialectKind::kQwen35;
  const std::string config = ReadFileIfPresent(JoinPath(tokenizer_dir, "tokenizer_config.json"));
  // The (tens of MB) vocab is only scanned when the config alone did not say Gemma.
  if (DetectDialect(config, /*has_turn_token=*/false) == DialectKind::kGemma4) return DialectKind::kGemma4;
  const bool has_turn_token =
      ReadFileIfPresent(JoinPath(tokenizer_dir, "tokenizer.json")).find("\"<|turn>\"") != std::string::npos;
  return DetectDialect(config, has_turn_token);
}

std::string ResolveTokenizerDir(const ModelDialect& d, const std::string& requested) {
  if (!requested.empty()) return requested;
  return d.default_tokenizer_dir.empty() ? std::string(kQwenDefaultTokenizerDir) : d.default_tokenizer_dir;
}

std::string CheckDialectAgainstArch(const ModelDialect& d, std::string_view model_arch) {
  if (model_arch.empty()) return "";
  const bool arch_is_gemma = model_arch.substr(0, 5) == "gemma";
  if (d.kind == DialectKind::kGemma4 && !arch_is_gemma) {
    return std::string("dialect gemma4 does not match model architecture '") + std::string(model_arch) + "'";
  }
  if (d.kind == DialectKind::kQwen35 && arch_is_gemma) {
    return std::string("dialect qwen35 does not match model architecture '") + std::string(model_arch) +
           "' (pass --dialect gemma4)";
  }
  return "";
}

bool ModelDialect::ReasoningOpenInPrompt(bool enable_thinking, std::string_view rendered_prompt) const {
  return enable_thinking && EndsWith(rendered_prompt, prompt_reasoning_open);
}

ReasoningSplitter::StartState ModelDialect::ReasoningStart(bool enable_thinking,
                                                           std::string_view rendered_prompt) const {
  using S = ReasoningSplitter::StartState;
  if (kind == DialectKind::kGemma4) {
    return ReasoningOpenInPrompt(enable_thinking, rendered_prompt) ? S::kInReasoning : S::kExpectOpener;
  }
  return enable_thinking ? S::kInReasoning : S::kAnswer;
}

std::string ModelDialect::CheckpointSuffix(bool /*enable_thinking*/, std::string_view rendered_prompt) const {
  if (kind == DialectKind::kGemma4) {
    // Thinking on ends "<|turn>model\n" (no reasoning suffix) or, after a tool response,
    // "<|channel>thought\n" (replays identically, so no holdback either): 0 tokens.
    if (EndsWith(rendered_prompt, prompt_reasoning_closed)) return prompt_reasoning_closed;
    return "";
  }
  return EndsWith(rendered_prompt, prompt_reasoning_open) ? prompt_reasoning_open : "";
}

}  // namespace r4dx::server
