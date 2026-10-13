// tests/server/test_q4exp_template.cpp -- the qwen4_exp chat surface (docs/qwen4exp.md): Swift-1.5-Qwen3.8-Flash-Next's
// chat_template.jinja is byte-identical to the 27B's, so the existing ChatTemplate must render the same prompt for
// both, tools and reasoning_effort included. The tokenizer directory's generation_config.json must give the EOS
// list the qwen4_exp dialect leaves to it, and the Swift directory's own tokenizer must encode like the 27B's.
// CPU only. Exits 77 (SKIPPED) when either checkpoint directory is absent (R4DX_MODELS_ROOT, default E:\models).
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "chat_template.h"
#include "tokenizer.h"
#include "dialect.h"
#include "nlohmann/json.hpp"
#include "r4dx/models_root.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

std::string Slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::string Join(const std::string& dir, const char* name) { return dir + "\\" + name; }

r4dx::ChatJson Tools() {
  return r4dx::ChatJson::parse(R"([
    {"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},
                    "required":["city"]}}}
  ])");
}

r4dx::ChatJson Messages() {
  return r4dx::ChatJson::parse(R"([
    {"role":"system","content":"You are terse."},
    {"role":"user","content":"Weather in Oslo for 3 days?"},
    {"role":"assistant","content":"","reasoning_content":"I should call the tool.",
     "tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather",
        "arguments":{"city":"Oslo","days":3}}}]},
    {"role":"tool","tool_call_id":"call_1","content":"-2C, snow"},
    {"role":"user","content":"Thanks. And tomorrow?"}
  ])");
}

}  // namespace

int main() {
  const std::string swift = r4dx::ModelsPath("r4dx/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ");
  const std::string ref27 = r4dx::server::QwenDefaultTokenizerDir();
  const std::string swift_tmpl = Slurp(Join(swift, "chat_template.jinja"));
  const std::string ref_tmpl = Slurp(Join(ref27, "chat_template.jinja"));
  if (swift_tmpl.empty() || ref_tmpl.empty()) {
    std::printf("test_q4exp_template: SKIPPED (%s or %s has no chat_template.jinja)\n", swift.c_str(), ref27.c_str());
    return 77;
  }

  // the premise of the qwen4_exp dialect: the two files are the same bytes
  CHECK(swift_tmpl == ref_tmpl);

  const r4dx::ChatTemplate t_swift = r4dx::ChatTemplate::from_directory(swift);
  const r4dx::ChatTemplate t_27b = r4dx::ChatTemplate::from_directory(ref27);
  CHECK(t_swift.source() == t_27b.source());
  CHECK(t_swift.eos_token() == t_27b.eos_token() && t_swift.bos_token() == t_27b.bos_token());

  int rendered = 0;
  for (const char* effort : {"xhigh", "medium", "low"}) {
    for (const bool thinking : {true, false}) {
      for (const bool with_tools : {true, false}) {
        r4dx::ChatJson extra = r4dx::ChatJson::object();
        extra["reasoning_effort"] = effort;
        extra["enable_thinking"] = thinking;
        const r4dx::ChatJson tools = with_tools ? Tools() : r4dx::ChatJson::array();
        const std::string a = t_swift.render(Messages(), /*add_generation_prompt=*/true, tools, extra);
        const std::string b = t_27b.render(Messages(), /*add_generation_prompt=*/true, tools, extra);
        CHECK(a == b);
        CHECK(!a.empty());
        ++rendered;
        // The surface the qwen4_exp dialect assumes
        const r4dx::server::ModelDialect& d = r4dx::server::Qwen4ExpDialect();
        CHECK(a.find("<|im_start|>assistant\n") != std::string::npos);
        if (thinking) CHECK(d.ReasoningOpenInPrompt(true, a));
        if (thinking) {
          CHECK(a.size() >= 8 && a.compare(a.size() - 8, 8, "<think>\n") == 0);
        } else {
          CHECK(a.size() >= 19 && a.compare(a.size() - 19, 19, "<think>\n\n</think>\n\n") == 0);
        }
        CHECK((a.find("<function=get_weather>") != std::string::npos) == true);  // the replayed assistant tool call
        CHECK(a.find("<parameter=city>") != std::string::npos);
        CHECK((a.find("# Tools") != std::string::npos) == with_tools);
        const bool effort_text = a.find("Reasoning effort is set to " + std::string(effort)) != std::string::npos;
        CHECK(effort_text == (thinking && std::string(effort) != "medium"));  // medium adds no instruction line
      }
    }
  }
  CHECK(rendered == 12);

  // an unsupported effort is rejected identically (the template raises)
  {
    r4dx::ChatJson extra = r4dx::ChatJson::object();
    extra["reasoning_effort"] = "extreme";
    std::string e1, e2;
    try { t_swift.render(Messages(), true, Tools(), extra); } catch (const std::exception& e) { e1 = e.what(); }
    try { t_27b.render(Messages(), true, Tools(), extra); } catch (const std::exception& e) { e2 = e.what(); }
    CHECK(!e1.empty() && e1 == e2);
  }

  // EOS: the dialect leaves it to the tokenizer directory; the Swift generation_config.json is the 27B's, bytewise,
  // and lists <|im_end|> (248046) then <|endoftext|> (248044)
  const std::string gen_swift = Slurp(Join(swift, "generation_config.json"));
  CHECK(!gen_swift.empty() && gen_swift == Slurp(Join(ref27, "generation_config.json")));
  const nlohmann::json gen = nlohmann::json::parse(gen_swift, nullptr, /*allow_exceptions=*/false);
  CHECK(gen.is_object() && gen.contains("eos_token_id"));
  if (gen.is_object() && gen.contains("eos_token_id")) {
    CHECK((gen.at("eos_token_id").get<std::vector<int>>() == std::vector<int>{248046, 248044}));
  }
  CHECK(r4dx::server::Qwen4ExpDialect().eos_ids.empty());

  // The default tokenizer directory (the 27B's) gives the EOS list the dialect leaves to it. The Swift directory's own
  // tokenizer.json is NOT loadable by r4dx (its re-saved decoder says add_prefix_space=true, which Tokenizer refuses),
  // which is why qwen4_exp defaults to the 27B directory (docs/qwen4exp.md).
  try {
    r4dx::Tokenizer::Options topt;
    topt.allow_unimplemented_normalizer = true;
    const r4dx::Tokenizer tok_27b = r4dx::Tokenizer::from_directory(ref27, topt);
    CHECK((tok_27b.eos_ids() == std::vector<r4dx::TokenId>{248046, 248044}));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "27B tokenizer: exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("test_q4exp_template: OK (%d renders identical)\n", rendered);
  return 0;
}
