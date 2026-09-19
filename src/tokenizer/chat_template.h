// r4dx chat template rendering -- public API for src/cli and src/server.
//
// Thin wrapper around the vendored minja Jinja2 engine (third_party/minja) applying
// Qwen3.8-27B's `chat_template.jinja` (C:\AI\models\Qwen3.8-27B\chat_template.jinja): OpenAI
// chat-message array -> the `<|im_start|>role\n...<|im_end|>\n` prompt string, including tool
// definitions, tool-call/tool-response rendering, and the `<think>` reasoning block.
#pragma once

#include <memory>
#include <string>

#include "nlohmann/json.hpp"

namespace r4dx {

// Messages/tools are OpenAI-chat-style JSON, e.g.:
//   messages = [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]
//   tools    = [{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}]
// `nlohmann::ordered_json` (not plain `json`) is used throughout because the template relies on
// object key insertion order (e.g. `tool.arguments | items` iterates a tool call's parameters in
// the order the caller supplied them, not alphabetically).
using ChatJson = nlohmann::ordered_json;

class ChatTemplate {
public:
    // Reads `<model_dir>/chat_template.jinja` if present, else falls back to the
    // `chat_template` field of `<model_dir>/tokenizer_config.json`. bos/eos token text come from
    // tokenizer_config.json (`bos_token`/`eos_token`), used only for the template's own
    // `bos_token`/`eos_token` globals (Qwen3.8-27B's template does not reference them directly,
    // but generic Jinja chat templates may). Throws std::runtime_error if no template source can
    // be found or it fails to parse.
    static ChatTemplate from_directory(const std::string& model_dir);

    static ChatTemplate from_source(const std::string& jinja_source, const std::string& bos_token,
                                     const std::string& eos_token);

    ChatTemplate();
    ~ChatTemplate();
    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;
    ChatTemplate(const ChatTemplate&);
    ChatTemplate& operator=(const ChatTemplate&);

    // Renders `messages` (a JSON array) through the template. `tools` is a JSON array of tool
    // schemas, or `nullptr`/empty/absent for none. `extra_context` supplies additional Jinja
    // globals the template reads -- Qwen3.8-27B's template understands at least:
    //   enable_thinking   (bool, default true)   -- omit/true renders an open "<think>\n" prompt
    //                                                suffix when add_generation_prompt is set;
    //                                                false renders a pre-closed empty think block
    //   reasoning_effort   ("xhigh"|"medium"|"low", default "xhigh")
    //   preserve_thinking  (bool, default true)   -- keep <think>...</think> in prior assistant
    //                                                turns rather than only the latest one
    //   add_vision_id      (bool)                 -- prefix image/video content with "Picture N:"
    // The returned string is exactly what should be passed to Tokenizer::encode(...,
    // /*parse_special=*/true) -- it legitimately contains literal <|im_start|>/<|im_end|>/<think>
    // control sequences that must be recognized as their token ids. CAUTION: message bodies are
    // spliced into the output verbatim, so parse_special=true on the composed result also
    // recognizes any special-token surface form a caller-supplied message body happens to contain
    // -- see the CAUTION note on Tokenizer::encode()'s parse_special parameter.
    std::string render(const ChatJson& messages, bool add_generation_prompt,
                        const ChatJson& tools = ChatJson::array(),
                        const ChatJson& extra_context = ChatJson::object()) const;

    const std::string& source() const;
    const std::string& bos_token() const;
    const std::string& eos_token() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    explicit ChatTemplate(std::shared_ptr<Impl> impl);
};

}  // namespace r4dx
