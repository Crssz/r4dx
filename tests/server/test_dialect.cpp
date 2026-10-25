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

// ---- LFM2.5 (docs/lfm2-plan.md 5.3, L2-3) ------------------------------------------------------------
void TestLfm2Table() {
  const auto& l = Lfm2Dialect();
  CHECK(l.kind == DialectKind::kLfm2 && std::string(l.name) == "lfm2");
  CHECK(&DialectFor(DialectKind::kLfm2) == &l);
  CHECK(l.reasoning.open.empty() && l.reasoning.close == "</think>");
  CHECK(l.prompt_reasoning_open == "<think>" && l.prompt_reasoning_closed.empty());
  CHECK(l.tool_open == "<|tool_call_start|>" && l.tool_close == "<|tool_call_end|>");
  CHECK(l.tool_open.size() == 19 && l.tool_open.size() - 1 == 18);  // gate holdback 18
  CHECK(!l.chat_template_polyfills && l.bos_on_raw_prompt && !l.has_reasoning_effort && l.forces_thinking);
  CHECK(l.keep_special_on_decode.empty());  // both tool markers and <think> are NON-special added tokens
  CHECK((l.eos_ids == std::vector<int32_t>{124900}));
  CHECK(!l.default_tokenizer_dir.empty());
  // generation_config.json defaults, LFM2 only
  CHECK(l.sampling.temperature && *l.sampling.temperature == 0.1f);
  CHECK(l.sampling.top_k && *l.sampling.top_k == 50);
  CHECK(l.sampling.repetition_penalty && *l.sampling.repetition_penalty == 1.05f);
  CHECK(!l.sampling.top_p && !l.sampling.min_p);
  // Qwen and Gemma carry no defaults and never force thinking: their requests are unchanged.
  for (const ModelDialect* d : {&Qwen35Dialect(), &Gemma4Dialect()}) {
    CHECK(!d->forces_thinking);
    CHECK(!d->sampling.temperature && !d->sampling.top_k && !d->sampling.top_p && !d->sampling.min_p &&
          !d->sampling.repetition_penalty);
  }
  DialectKind k;
  CHECK(ParseDialectName("lfm2", &k) && k == DialectKind::kLfm2);
  CHECK(!ParseDialectName("LFM2", &k));
}

void TestLfm2PromptState() {
  const auto& l = Lfm2Dialect();
  // The template ends the generation prompt with an open "<think>" (no newline). The model always thinks.
  const std::string prompt = "<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>";
  CHECK(l.ReasoningStart(true, prompt) == Start::kInReasoning);
  CHECK(l.ReasoningOpenInPrompt(true, prompt));
  CHECK(l.CheckpointSuffix(true, prompt) == "<think>");
  // After a tool turn the prompt still ends with the same opener.
  CHECK(l.CheckpointSuffix(true, "...<|im_start|>tool\nr<|im_end|>\n<|im_start|>assistant\n<think>") == "<think>");
  CHECK(l.CheckpointSuffix(true, "no opener at the end").empty());
}

void TestLfm2Detect() {
  const std::string lfm_cfg = "{\"bos_token\":\"<|startoftext|>\",\"tokenizer_class\":\"TokenizersBackend\"}";
  CHECK(DetectDialect(lfm_cfg, false, true) == DialectKind::kLfm2);
  CHECK(DetectDialect(lfm_cfg, false, false) == DialectKind::kQwen35);  // bos alone is not enough
  CHECK(DetectDialect("{\"bos_token\":null}", false, true) == DialectKind::kQwen35);
  CHECK(DetectDialect("{\"bos_token\":\"<bos>\"}", false, true) == DialectKind::kQwen35);
  CHECK(DetectDialect("{\"tokenizer_class\":\"GemmaTokenizer\",\"bos_token\":\"<|startoftext|>\"}", false, true) ==
        DialectKind::kGemma4);  // Gemma outranks
  CHECK(DetectDialect(lfm_cfg, true, true) == DialectKind::kGemma4);
  CHECK(DetectDialect("not json", false, true) == DialectKind::kQwen35);
  // The two-argument form every existing caller uses is unchanged.
  CHECK(DetectDialect(lfm_cfg, false) == DialectKind::kQwen35);

  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "r4dx_test_dialect_lfm2";
  fs::remove_all(root);
  fs::create_directories(root / "lfm");
  fs::create_directories(root / "noconfig_tool");
  std::ofstream(root / "lfm" / "tokenizer_config.json") << lfm_cfg;
  std::ofstream(root / "lfm" / "tokenizer.json")
      << "{\"added_tokens\":[{\"id\":124905,\"content\":\"<|tool_call_start|>\"}]}";
  std::ofstream(root / "noconfig_tool" / "tokenizer.json")
      << "{\"added_tokens\":[{\"id\":124905,\"content\":\"<|tool_call_start|>\"}]}";
  const std::string lfm = (root / "lfm").string();
  CHECK(ResolveDialectKind(std::nullopt, lfm, false) == DialectKind::kLfm2);
  CHECK(ResolveDialectKind(std::nullopt, (root / "noconfig_tool").string(), false) == DialectKind::kQwen35);
  CHECK(ResolveDialectKind(DialectKind::kQwen35, lfm, false) == DialectKind::kQwen35);  // explicit wins
  CHECK(ResolveDialectKind(DialectKind::kLfm2, "", false) == DialectKind::kLfm2);
  // no tokenizer dir: the container architecture decides
  CHECK(ResolveDialectKind(std::nullopt, "", false, true) == DialectKind::kLfm2);
  CHECK(ResolveDialectKind(std::nullopt, "", true, false) == DialectKind::kGemma4);
  CHECK(ResolveDialectKind(std::nullopt, "", false, false) == DialectKind::kQwen35);
  fs::remove_all(root);

  CHECK(ResolveTokenizerDir(Lfm2Dialect(), "") == Lfm2Dialect().default_tokenizer_dir);
  CHECK(ResolveTokenizerDir(Lfm2Dialect(), "X:\\lfm") == "X:\\lfm");
}

void TestLfm2ArchCheck() {
  CHECK(ArchNameIsLfm2("lfm2_moe") && ArchNameIsLfm2("lfm2") && !ArchNameIsLfm2("qwen3_5") && !ArchNameIsLfm2(""));
  CHECK(CheckDialectAgainstArch(Lfm2Dialect(), "lfm2_moe").empty());
  CHECK(CheckDialectAgainstArch(Lfm2Dialect(), "").empty());
  CHECK(!CheckDialectAgainstArch(Lfm2Dialect(), "qwen3_5").empty());
  CHECK(!CheckDialectAgainstArch(Lfm2Dialect(), "gemma4_unified").empty());
  CHECK(!CheckDialectAgainstArch(Qwen35Dialect(), "lfm2_moe").empty());
  CHECK(!CheckDialectAgainstArch(Gemma4Dialect(), "lfm2_moe").empty());
  CHECK(CheckDialectAgainstArch(Qwen35Dialect(), "qwen3_5").empty());  // unchanged
  // next to qwen4_exp (merge of origin/main): the LFM2 and qwen4_exp dialects refuse each other's containers
  CHECK(!CheckDialectAgainstArch(Lfm2Dialect(), "qwen4_exp").empty());
  CHECK(CheckDialectAgainstArch(Qwen4ExpDialect(), "lfm2_moe").find("--dialect lfm2") != std::string::npos);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "lfm2_moe") == DialectKind::kLfm2);
  CHECK(ResolveDialectKindForArch(DialectKind::kQwen35, "", "lfm2_moe") == DialectKind::kQwen35);  // explicit wins
}

// qwen4_exp (docs/qwen4exp.md): the Qwen3.5 text surface under its own name. The container-header half (arch string,
// synthetic safetensors shell) is tests/server/test_q4exp_dialect.cpp, which links the arch library.
void TestQwen4Exp() {
  const auto& q = Qwen35Dialect();
  const auto& e = Qwen4ExpDialect();
  CHECK(e.kind == DialectKind::kQwen4Exp);
  CHECK(std::string(e.name) == "qwen4_exp");
  CHECK(&DialectFor(DialectKind::kQwen4Exp) == &e && &e != &q);
  // everything except kind/name is the Qwen3.5 table
  CHECK(e.reasoning.open == q.reasoning.open && e.reasoning.close == q.reasoning.close &&
        e.reasoning.skip_blank_after_close == q.reasoning.skip_blank_after_close);
  CHECK(e.prompt_reasoning_open == "<think>\n" && e.prompt_reasoning_closed.empty());
  CHECK(e.tool_open == "<tool_call>" && e.tool_close == "</tool_call>");
  CHECK(e.keep_special_on_decode.empty());
  CHECK(e.chat_template_polyfills && !e.bos_on_raw_prompt && e.has_reasoning_effort);
  // EOS and tokenizer directory come from the tokenizer directory / the Qwen default, as for qwen35
  CHECK(e.eos_ids.empty() && e.default_tokenizer_dir.empty());
  CHECK(ResolveTokenizerDir(e, "") == QwenDefaultTokenizerDir());
  CHECK(ResolveTokenizerDir(e, "X:/tok") == "X:/tok");

  const std::string qthink = "<|im_start|>assistant\n<think>\n";
  CHECK(e.ReasoningStart(true, qthink) == Start::kInReasoning);
  CHECK(e.ReasoningStart(false, "<|im_start|>assistant\n<think>\n\n</think>\n\n") == Start::kAnswer);
  CHECK(e.ReasoningOpenInPrompt(true, qthink) && !e.ReasoningOpenInPrompt(false, qthink));
  CHECK(e.CheckpointSuffix(true, qthink) == q.CheckpointSuffix(true, qthink));

  DialectKind k;
  CHECK(ParseDialectName("qwen4_exp", &k) && k == DialectKind::kQwen4Exp);
  CHECK(!ParseDialectName("Qwen4_exp", &k) && !ParseDialectName("qwen4exp", &k));

  // arch cross-check: a gemma container is refused (and told what to pass), everything else passes
  CHECK(CheckDialectAgainstArch(e, "qwen4_exp").empty());
  CHECK(CheckDialectAgainstArch(e, "").empty());
  CHECK(CheckDialectAgainstArch(e, "qwen3_5").empty());
  CHECK(CheckDialectAgainstArch(q, "qwen4_exp").empty());
  const std::string err = CheckDialectAgainstArch(e, "gemma4_unified");
  CHECK(err.find("qwen4_exp") != std::string::npos && err.find("--dialect gemma4") != std::string::npos);
  const std::string err2 = CheckDialectAgainstArch(Gemma4Dialect(), "qwen4_exp");
  CHECK(err2.find("gemma4") != std::string::npos && err2.find("qwen4_exp") != std::string::npos);

  // ResolveDialectKindForArch: no tokenizer dir -> the container's arch decides; an explicit request wins
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "qwen4_exp") == DialectKind::kQwen4Exp);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "qwen3_5") == DialectKind::kQwen35);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "") == DialectKind::kQwen35);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "gemma4_unified") == DialectKind::kGemma4);
  CHECK(ResolveDialectKindForArch(DialectKind::kQwen35, "", "qwen4_exp") == DialectKind::kQwen35);
  CHECK(ResolveDialectKindForArch(DialectKind::kQwen4Exp, "", "gemma4_unified") == DialectKind::kQwen4Exp);
  // the bool overload is unchanged
  CHECK(ResolveDialectKind(std::nullopt, "", false) == DialectKind::kQwen35);
  CHECK(ResolveDialectKind(std::nullopt, "", true) == DialectKind::kGemma4);

  // flags a qwen4_exp container refuses
  Qwen4ExpFlags f;
  CHECK(CheckQwen4ExpFlags(f).empty());
  f.tp = 2;
  CHECK(CheckQwen4ExpFlags(f).find("--tp 2") != std::string::npos);
  f = Qwen4ExpFlags{};
  f.pp = 2;
  CHECK(CheckQwen4ExpFlags(f).find("--pp 2") != std::string::npos);
  f.pp = 1;
  CHECK(CheckQwen4ExpFlags(f).empty());
  f.pp = -1;
  CHECK(CheckQwen4ExpFlags(f).empty());
  f = Qwen4ExpFlags{};
  f.mtp = 3;
  CHECK(CheckQwen4ExpFlags(f).find("--mtp 3") != std::string::npos);
  f = Qwen4ExpFlags{};
  f.dflash = true;
  CHECK(CheckQwen4ExpFlags(f).find("--dflash") != std::string::npos);
  f = Qwen4ExpFlags{};
  f.images = true;
  CHECK(CheckQwen4ExpFlags(f).find("--image") != std::string::npos);
  f = Qwen4ExpFlags{};
  f.vision_on = true;
  CHECK(CheckQwen4ExpFlags(f).find("--vision on") != std::string::npos);
  // every refused flag is named in the one message
  f.tp = 2;
  f.pp = 2;
  f.mtp = 1;
  f.dflash = true;
  f.images = true;
  const std::string all = CheckQwen4ExpFlags(f);
  for (const char* want : {"--tp 2", "--pp 2", "--mtp 1", "--dflash", "--vision on", "--image"}) {
    CHECK(all.find(want) != std::string::npos);
  }

  // the default tokenizer directory must exist for qwen4_exp, with an actionable message when it does not
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "r4dx_test_dialect_q4exp_tok";
  fs::remove_all(root);
  fs::create_directories(root / "has");
  std::ofstream(root / "has" / "tokenizer.json") << "{}";
  const std::string has = (root / "has").string();
  const std::string missing = (root / "missing").string();
  const std::string miss_err = CheckDefaultTokenizerDir(e, missing, /*user_given=*/false);
  CHECK(miss_err.find("--tokenizer-dir") != std::string::npos && miss_err.find(missing) != std::string::npos);
  CHECK(CheckDefaultTokenizerDir(e, has, false).empty());
  CHECK(CheckDefaultTokenizerDir(e, missing, /*user_given=*/true).empty());  // the loader names that one itself
  CHECK(CheckDefaultTokenizerDir(q, missing, false).empty());                // other dialects: unchanged
  CHECK(CheckDefaultTokenizerDir(Gemma4Dialect(), missing, false).empty());
  fs::remove_all(root);
}

// Nanbeige4.3-33B (docs/nanbeige-plan.md 5.3): Qwen's text surface under its own name, its own EOS / BOS / tokenizer
// directory, and the dense llama arch's only dialect.
void TestNanbeige() {
  const auto& n = NanbeigeDialect();
  const auto& q = Qwen35Dialect();
  CHECK(n.kind == DialectKind::kNanbeige && std::string(n.name) == "nanbeige");
  CHECK(n.reasoning.close == q.reasoning.close && n.reasoning.open == q.reasoning.open &&
        n.prompt_reasoning_open == "<think>\n" && n.tool_open == "<tool_call>" && n.tool_close == "</tool_call>");
  CHECK(!n.chat_template_polyfills && n.bos_on_raw_prompt && !n.has_reasoning_effort && !n.forces_thinking);
  CHECK(n.eos_ids == std::vector<int32_t>{166101});
  CHECK(n.default_tokenizer_dir.find("Nanbeige4.3-33B-tok") != std::string::npos);
  CHECK(&DialectFor(DialectKind::kNanbeige) == &n);
  DialectKind k = DialectKind::kQwen35;
  CHECK(ParseDialectName("nanbeige", &k) && k == DialectKind::kNanbeige);
  // Detection: LlamaTokenizer + <|im_start|> BOS; either alone is not Nanbeige.
  CHECK(DetectDialect(R"({"tokenizer_class": "LlamaTokenizer", "bos_token": "<|im_start|>"})", false) == DialectKind::kNanbeige);
  CHECK(DetectDialect(R"({"tokenizer_class": "LlamaTokenizer", "bos_token": "<s>"})", false) == DialectKind::kQwen35);
  CHECK(DetectDialect(R"({"tokenizer_class": "Qwen2Tokenizer", "bos_token": "<|im_start|>"})", false) == DialectKind::kQwen35);
  CHECK(DetectDialect(R"({"tokenizer_class": "GemmaTokenizer", "bos_token": "<|im_start|>"})", false) == DialectKind::kGemma4);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "llama") == DialectKind::kNanbeige);
  CHECK(ResolveDialectKindForArch(std::nullopt, "", "qwen3_5") == DialectKind::kQwen35);
  // Prompt state: the Qwen rules.
  CHECK(n.ReasoningStart(true, "...<|im_start|>assistant\n<think>\n") == Start::kInReasoning);
  CHECK(n.ReasoningStart(false, "...<|im_start|>assistant\n<think>\n\n</think>\n\n") == Start::kAnswer);
  CHECK(n.CheckpointSuffix(true, "x<think>\n") == "<think>\n");
  // Arch cross-check: nanbeige <-> llama only.
  CHECK(CheckDialectAgainstArch(n, "llama").empty());
  CHECK(!CheckDialectAgainstArch(n, "qwen3_5").empty() && !CheckDialectAgainstArch(n, "lfm2_moe").empty());
  const std::string err = CheckDialectAgainstArch(q, "llama");
  CHECK(err.find("--dialect nanbeige") != std::string::npos);
  CHECK(!CheckDialectAgainstArch(Gemma4Dialect(), "llama").empty() && !CheckDialectAgainstArch(Lfm2Dialect(), "llama").empty());
  // The other dialects are unchanged on their own archs.
  CHECK(CheckDialectAgainstArch(q, "qwen3_5").empty() && CheckDialectAgainstArch(Lfm2Dialect(), "lfm2_moe").empty());
}

}  // namespace

int main() {
  TestLfm2Table();
  TestLfm2PromptState();
  TestLfm2Detect();
  TestLfm2ArchCheck();
  TestResolve();
  TestTables();
  TestParseName();
  TestDetect();
  TestArchCheck();
  TestPromptState();
  TestQwen4Exp();
  TestNanbeige();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_dialect: OK");
  return 0;
}
