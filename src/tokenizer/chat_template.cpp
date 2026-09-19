#include "chat_template.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "minja/chat-template.hpp"
#include "nlohmann/json.hpp"

namespace r4dx {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

struct ChatTemplate::Impl {
    minja::chat_template tpl;

    Impl(const std::string& source, const std::string& bos, const std::string& eos)
        : tpl(source, bos, eos) {}
};

ChatTemplate::ChatTemplate() = default;
ChatTemplate::~ChatTemplate() = default;
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;
ChatTemplate::ChatTemplate(const ChatTemplate&) = default;
ChatTemplate& ChatTemplate::operator=(const ChatTemplate&) = default;

ChatTemplate::ChatTemplate(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

ChatTemplate ChatTemplate::from_source(const std::string& jinja_source, const std::string& bos_token,
                                        const std::string& eos_token) {
    if (jinja_source.empty()) {
        throw std::runtime_error("ChatTemplate::from_source: empty jinja source");
    }
    try {
        return ChatTemplate(std::make_shared<Impl>(jinja_source, bos_token, eos_token));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ChatTemplate: failed to parse jinja template: ") + e.what());
    }
}

ChatTemplate ChatTemplate::from_directory(const std::string& model_dir) {
    const std::string sep = (!model_dir.empty() && (model_dir.back() == '/' || model_dir.back() == '\\'))
                                 ? ""
                                 : "/";
    std::string source = read_file(model_dir + sep + "chat_template.jinja");

    std::string bos_token;
    std::string eos_token;

    const std::string tok_cfg_path = model_dir + sep + "tokenizer_config.json";
    std::string tok_cfg_raw = read_file(tok_cfg_path);
    if (!tok_cfg_raw.empty()) {
        nlohmann::json cfg = nlohmann::json::parse(tok_cfg_raw, /*cb=*/nullptr, /*allow_exceptions=*/true);
        auto read_tok = [&](const char* key) -> std::string {
            if (!cfg.contains(key) || cfg[key].is_null()) return {};
            if (cfg[key].is_string()) return cfg[key].get<std::string>();
            if (cfg[key].is_object() && cfg[key].contains("content")) return cfg[key]["content"].get<std::string>();
            return {};
        };
        bos_token = read_tok("bos_token");
        eos_token = read_tok("eos_token");

        if (source.empty() && cfg.contains("chat_template")) {
            if (cfg["chat_template"].is_string()) {
                source = cfg["chat_template"].get<std::string>();
            } else if (cfg["chat_template"].is_array()) {
                // Some configs store [{"name":"default","template":"..."}, ...]; prefer "default".
                for (const auto& entry : cfg["chat_template"]) {
                    if (entry.contains("name") && entry["name"] == "default" && entry.contains("template")) {
                        source = entry["template"].get<std::string>();
                        break;
                    }
                }
            }
        }
    }

    if (source.empty()) {
        throw std::runtime_error("ChatTemplate::from_directory: no chat_template.jinja and no "
                                  "tokenizer_config.json[\"chat_template\"] found under " +
                                  model_dir);
    }

    return from_source(source, bos_token, eos_token);
}

std::string ChatTemplate::render(const ChatJson& messages, bool add_generation_prompt, const ChatJson& tools,
                                  const ChatJson& extra_context) const {
    if (!impl_) {
        throw std::runtime_error("ChatTemplate::render: template not loaded");
    }
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.tools = tools.is_null() ? ChatJson::array() : tools;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.extra_context = extra_context.is_null() ? ChatJson::object() : extra_context;

    minja::chat_template_options opts;  // defaults: polyfills on, but Qwen3.8-27B's template
                                         // natively supports system role + tools, so caps
                                         // detection should make every polyfill a no-op.
    try {
        return impl_->tpl.apply(inputs, opts);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ChatTemplate::render: ") + e.what());
    }
}

const std::string& ChatTemplate::source() const {
    if (!impl_) throw std::runtime_error("ChatTemplate::source: template not loaded");
    return impl_->tpl.source();
}
const std::string& ChatTemplate::bos_token() const {
    if (!impl_) throw std::runtime_error("ChatTemplate::bos_token: template not loaded");
    return impl_->tpl.bos_token();
}
const std::string& ChatTemplate::eos_token() const {
    if (!impl_) throw std::runtime_error("ChatTemplate::eos_token: template not loaded");
    return impl_->tpl.eos_token();
}

}  // namespace r4dx
