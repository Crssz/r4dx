// tests/server/test_dialect.cpp -- ModelDialect table, selection, arch cross-check and the
// prompt-state helpers the engine wiring (M1-14) will call.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "dialect.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

using namespace r4dx::server;
using Start = ReasoningSplitter::StartState;

void TestTables() {
  const auto& q = Qwen35Dialect();
  CHECK(q.kind == DialectKind::kQwen35);
  CHECK(q.reasoning.close == "</think>" && q.reasoning.open.empty());
  CHECK(q.tool_open == "<tool_call>" && q.tool_close == "</tool_call>");
  CHECK(q.chat_template_polyfills && !q.bos_on_raw_prompt && q.has_reasoning_effort);
  CHECK(q.keep_special_on_decode.empty() && q.eos_ids.empty());

  const auto& g = Gemma4Dialect();
  CHECK(g.kind == DialectKind::kGemma4);
  CHECK(g.reasoning.open == "<|channel>thought\n" && g.reasoning.close == "<channel|>");
  CHECK(g.tool_open == "<|tool_call>" && g.tool_close == "<tool_call|>");
  CHECK(g.tool_open.size() - 1 == 11);
  CHECK(!g.chat_template_polyfills && g.bos_on_raw_prompt && !g.has_reasoning_effort);
  CHECK(g.keep_special_on_decode.size() == 5);
  CHECK((g.eos_ids == std::vector<int32_t>{1, 106, 50}));
  CHECK(&DialectFor(DialectKind::kGemma4) == &g && &DialectFor(DialectKind::kQwen35) == &q);
}

void TestParseName() {
  DialectKind k;
  CHECK(ParseDialectName("qwen35", &k) && k == DialectKind::kQwen35);
  CHECK(ParseDialectName("gemma4", &k) && k == DialectKind::kGemma4);
  CHECK(!ParseDialectName("auto", &k));
  CHECK(!ParseDialectName("Gemma4", &k));
  CHECK(!ParseDialectName("", &k));
}

void TestDetect() {
  CHECK(DetectDialect("{\"tokenizer_class\":\"GemmaTokenizer\"}", false) == DialectKind::kGemma4);
  CHECK(DetectDialect("{\"tokenizer_class\":\"Qwen2Tokenizer\"}", false) == DialectKind::kQwen35);
  CHECK(DetectDialect("{\"tokenizer_class\":\"Qwen2Tokenizer\"}", true) == DialectKind::kGemma4);
  CHECK(DetectDialect("not json", false) == DialectKind::kQwen35);
  CHECK(DetectDialect("not json", true) == DialectKind::kGemma4);
  CHECK(DetectDialect("", false) == DialectKind::kQwen35);
  CHECK(DetectDialect("[1,2]", false) == DialectKind::kQwen35);
  CHECK(DetectDialect("{\"tokenizer_class\":5}", false) == DialectKind::kQwen35);
}

void TestArchCheck() {
  CHECK(CheckDialectAgainstArch(Gemma4Dialect(), "gemma4_unified").empty());
  CHECK(CheckDialectAgainstArch(Gemma4Dialect(), "").empty());
  CHECK(!CheckDialectAgainstArch(Gemma4Dialect(), "qwen3_5").empty());
  CHECK(CheckDialectAgainstArch(Qwen35Dialect(), "qwen3_5").empty());
  CHECK(!CheckDialectAgainstArch(Qwen35Dialect(), "gemma4_unified").empty());
}

void TestPromptState() {
  const auto& q = Qwen35Dialect();
  const std::string qthink = "<|im_start|>assistant\n<think>\n";
  CHECK(q.ReasoningStart(true, qthink) == Start::kInReasoning);
  CHECK(q.ReasoningOpenInPrompt(true, qthink));
  CHECK(q.ReasoningStart(false, "<|im_start|>assistant\n<think>\n\n</think>\n\n") == Start::kAnswer);
  CHECK(!q.ReasoningOpenInPrompt(false, "x"));
  CHECK(q.CheckpointSuffix(true, qthink) == "<think>\n");
  CHECK(q.CheckpointSuffix(false, "<think>\n\n</think>\n\n").empty());

  const auto& g = Gemma4Dialect();
  const std::string on = "<bos><|turn>user\nhi<turn|>\n<|turn>model\n";
  const std::string off = on + "<|channel>thought\n<channel|>";
  const std::string after_tool = "<bos>...<tool_response|><|channel>thought\n";
  CHECK(g.ReasoningStart(true, on) == Start::kExpectOpener);
  CHECK(!g.ReasoningOpenInPrompt(true, on));
  CHECK(g.ReasoningStart(true, after_tool) == Start::kInReasoning);
  CHECK(g.ReasoningOpenInPrompt(true, after_tool));
  CHECK(g.ReasoningStart(false, off) == Start::kExpectOpener);
  CHECK(!g.ReasoningOpenInPrompt(false, off));
  CHECK(g.CheckpointSuffix(false, off) == "<|channel>thought\n<channel|>");
  CHECK(g.CheckpointSuffix(true, on).empty());
  CHECK(g.CheckpointSuffix(true, after_tool).empty());
}

// ResolveDialectKind / ResolveTokenizerDir (task M1-14), over throwaway tokenizer directories.
void TestResolve() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "r4dx_test_dialect_resolve";
  fs::remove_all(root);
  auto make_dir = [&](const std::string& name, const std::string& config, const std::string& tokenizer_json) {
    const fs::path d = root / name;
    fs::create_directories(d);
    if (!config.empty()) std::ofstream(d / "tokenizer_config.json") << config;
    if (!tokenizer_json.empty()) std::ofstream(d / "tokenizer.json") << tokenizer_json;
    return d.string();
  };
  const std::string gemma_cfg = make_dir("gemma_cfg", "{\"tokenizer_class\":\"GemmaTokenizer\"}", "");
  const std::string gemma_vocab = make_dir("gemma_vocab", "{\"tokenizer_class\":\"TokenizersBackend\"}",
                                           "{\"added_tokens\":[{\"id\":105,\"content\":\"<|turn>\"}]}");
  const std::string qwen = make_dir("qwen", "{\"tokenizer_class\":\"Qwen2Tokenizer\"}",
                                    "{\"added_tokens\":[{\"id\":1,\"content\":\"<|im_start|>\"}]}");
  const std::string empty_dir = make_dir("empty", "", "");

  // an explicit request always wins, whatever the directory says
  CHECK(ResolveDialectKind(DialectKind::kQwen35, gemma_cfg, true) == DialectKind::kQwen35);
  CHECK(ResolveDialectKind(DialectKind::kGemma4, qwen, false) == DialectKind::kGemma4);
  // auto with a directory: tokenizer_config.json, then the vocab
  CHECK(ResolveDialectKind(std::nullopt, gemma_cfg, false) == DialectKind::kGemma4);
  CHECK(ResolveDialectKind(std::nullopt, gemma_vocab, false) == DialectKind::kGemma4);
  CHECK(ResolveDialectKind(std::nullopt, qwen, true) == DialectKind::kQwen35);  // the dir outranks the arch
  CHECK(ResolveDialectKind(std::nullopt, empty_dir, false) == DialectKind::kQwen35);
  // auto without a directory: the container's architecture
  CHECK(ResolveDialectKind(std::nullopt, "", true) == DialectKind::kGemma4);
  CHECK(ResolveDialectKind(std::nullopt, "", false) == DialectKind::kQwen35);

  CHECK(ResolveTokenizerDir(Qwen35Dialect(), "") == QwenDefaultTokenizerDir());
  CHECK(ResolveTokenizerDir(Gemma4Dialect(), "") == Gemma4Dialect().default_tokenizer_dir);
  CHECK(!Gemma4Dialect().default_tokenizer_dir.empty());
  CHECK(ResolveTokenizerDir(Gemma4Dialect(), "X:\\tok") == "X:\\tok");
  fs::remove_all(root);
}

}  // namespace

int main() {
  TestResolve();
  TestTables();
  TestParseName();
  TestDetect();
  TestArchCheck();
  TestPromptState();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_dialect: OK");
  return 0;
}
