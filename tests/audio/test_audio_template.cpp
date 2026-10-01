// CPU-only: the engine's audio content-part JSON, rendered through the Gemma chat template, must yield exactly one
// <|audio|> (258881) per clip. Regression for the review finding that the engine emitted type "input_audio" which
// the real chat_template.jinja ignores (it only matches 'audio'), so every audio request got a 400.
// Part 1 (always): a minimal template with the real template's matching rule. Part 2: the real template + tokenizer
// from R4DX_GEMMA_TOKENIZER_DIR (default D:/models/Huihui-gemma-4-12B-it-abliterated); skipped if absent.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "audio_frames.h"
#include "chat_template.h"
#include "tokenizer.h"

static int g_fail = 0;
#define CHECK(cond, msg)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      ++g_fail;                                                 \
    }                                                           \
  } while (0)

// Mirrors engine.cpp's message building for a user turn with text + `clips` audio parts.
static r4dx::ChatJson Messages(int clips) {
  r4dx::ChatJson content = r4dx::ChatJson::array();
  content.push_back(r4dx::ChatJson{{"type", "text"}, {"text", "Transcribe:"}});
  for (int i = 0; i < clips; ++i)
    content.push_back(r4dx::ChatJson{{"type", r4dx::audio::kAudioTemplatePartType}});
  return r4dx::ChatJson::array({r4dx::ChatJson{{"role", "user"}, {"content", content}}});
}

static int Count(const std::string& s, const std::string& needle) {
  int n = 0;
  for (size_t p = s.find(needle); p != std::string::npos; p = s.find(needle, p + 1)) ++n;
  return n;
}

int main() {
  const std::string mini =
      "{% for m in messages %}{% for item in m['content'] %}"
      "{% if item['type'] == 'text' %}{{ item['text'] }}"
      "{% elif item['type'] == 'audio' %}<|audio|>{% endif %}{% endfor %}{% endfor %}";
  auto t = r4dx::ChatTemplate::from_source(mini, "", "");
  CHECK(Count(t.render(Messages(2), false), "<|audio|>") == 2, "mini template: one placeholder per clip");
  // The old, wrong type is silently dropped by such a template (documents why the constant is "audio").
  r4dx::ChatJson bad = Messages(1);
  bad[0]["content"][1]["type"] = "input_audio";
  CHECK(Count(t.render(bad, false), "<|audio|>") == 0, "input_audio is dropped by the template");

  std::string dir = "D:/models/Huihui-gemma-4-12B-it-abliterated";
  if (const char* e = std::getenv("R4DX_GEMMA_TOKENIZER_DIR")) dir = e;
  std::ifstream probe(dir + "/chat_template.jinja");
  if (!probe) {
    std::printf("SKIP real-template part: %s/chat_template.jinja not found\n", dir.c_str());
    return g_fail ? 1 : 0;
  }
  r4dx::ChatTemplateOptions opts;
  opts.apply_polyfills = false;  // Gemma dialect (engine.cpp: dialect_->chat_template_polyfills)
  auto real = r4dx::ChatTemplate::from_directory(dir, opts);
  r4dx::TokenizerOptions topts;
  auto tok = r4dx::Tokenizer::from_directory(dir, topts);
  for (int clips : {1, 2, 3}) {
    const std::string text = real.render(Messages(clips), true, r4dx::ChatJson::array(),
                                         r4dx::ChatJson{{"enable_thinking", false}});
    CHECK(Count(text, "<|audio|>") == clips, "real template: one <|audio|> per clip");
    int ids = 0;
    for (auto id : tok.encode(text, true)) ids += (id == r4dx::audio::kAudioTokenId);
    CHECK(ids == clips, "real template + tokenizer: exactly one 258881 per clip");
    // And the engine's expansion accepts it.
    std::vector<int32_t> raw;
    for (auto id : tok.encode(text, true)) raw.push_back(static_cast<int32_t>(id));
    CHECK(r4dx::audio::ExpandAudioPlaceholders(raw, std::vector<int64_t>(clips, 5)).spans.size() == size_t(clips),
          "ExpandAudioPlaceholders accepts the rendered prompt");
  }
  std::printf(g_fail ? "FAILED\n" : "OK\n");
  return g_fail ? 1 : 0;
}
