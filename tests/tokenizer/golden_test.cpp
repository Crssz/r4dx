// CTest 'tokenizer_golden': replays tests/tokenizer/golden.json (produced by
// tools/tok_ref/gen_golden.py against the real HF tokenizer for
// C:\AI\models\Qwen3.8-27B) through r4dx's own Tokenizer + ChatTemplate and asserts exact
// agreement. Exits non-zero (and prints every mismatch, not just the first) on any divergence.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "chat_template.h"
#include "nlohmann/json.hpp"
#include "tokenizer.h"

// nlohmann::ordered_json (not plain json, which reorders object keys alphabetically): golden.json
// cases carry chat "tools"/"messages" objects whose key order matters (Qwen3.8-27B's template
// renders each tool via `tojson`, which serializes an object in its own iteration order), so the
// parsed document must preserve the exact insertion order gen_golden.py wrote.
using json = nlohmann::ordered_json;
using r4dx::ChatJson;
using r4dx::TokenId;
using r4dx::Tokenizer;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "golden_test: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<TokenId> ids_from_json(const json& arr) {
    std::vector<TokenId> out;
    out.reserve(arr.size());
    for (const auto& v : arr) out.push_back(v.get<TokenId>());
    return out;
}

bool ids_equal(const std::vector<TokenId>& a, const std::vector<TokenId>& b) { return a == b; }

std::string ids_to_string(const std::vector<TokenId>& ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

int g_failures = 0;

void fail(const std::string& case_name, const std::string& what) {
    std::fprintf(stderr, "FAIL [%s]: %s\n", case_name.c_str(), what.c_str());
    g_failures++;
}

void check_encode_case(const Tokenizer& tok, const json& c) {
    const std::string name = c.at("name").get<std::string>();
    const std::string text = c.at("text").get<std::string>();
    const bool parse_special = c.at("parse_special").get<bool>();
    const std::vector<TokenId> expected_ids = ids_from_json(c.at("ids"));

    const std::vector<TokenId> got_ids = tok.encode(text, parse_special);
    if (!ids_equal(got_ids, expected_ids)) {
        fail(name, "encode mismatch\n  text     : " + text + "\n  expected : " + ids_to_string(expected_ids) +
                        "\n  got      : " + ids_to_string(got_ids));
        return;
    }

    if (c.contains("decoded")) {
        const std::string expected_decoded = c.at("decoded").get<std::string>();
        const std::string got_decoded = tok.decode(got_ids, /*skip_special_tokens=*/true);
        if (got_decoded != expected_decoded) {
            fail(name, "decode mismatch\n  expected : " + expected_decoded + "\n  got      : " + got_decoded);
        }
    }
}

void check_chat_case(const Tokenizer& tok, const r4dx::ChatTemplate& tmpl, const json& c) {
    const std::string name = c.at("name").get<std::string>();
    const ChatJson messages = c.at("messages");
    const ChatJson tools = c.contains("tools") && !c.at("tools").is_null() ? ChatJson(c.at("tools")) : ChatJson::array();
    const bool add_gen = c.at("add_generation_prompt").get<bool>();
    ChatJson extra = ChatJson::object();
    if (c.contains("extra_context")) extra = c.at("extra_context");
    const std::string expected_prompt = c.at("prompt").get<std::string>();
    const std::vector<TokenId> expected_ids = ids_from_json(c.at("ids"));

    std::string got_prompt;
    try {
        got_prompt = tmpl.render(messages, add_gen, tools, extra);
    } catch (const std::exception& e) {
        fail(name, std::string("render threw: ") + e.what());
        return;
    }
    if (got_prompt != expected_prompt) {
        fail(name, "chat render mismatch\n  expected : " + expected_prompt + "\n  got      : " + got_prompt);
        return;
    }

    const std::vector<TokenId> got_ids = tok.encode(got_prompt, /*parse_special=*/true);
    if (!ids_equal(got_ids, expected_ids)) {
        fail(name, "chat prompt encode mismatch\n  expected : " + ids_to_string(expected_ids) +
                        "\n  got      : " + ids_to_string(got_ids));
    }
}

// Exercises StreamDecoder against a handful of the multi-byte-heavy golden cases (emoji, CJK,
// Thai) to check it reproduces the one-shot decode() when fed one token at a time, including a
// mid-codepoint flush() at the very end.
void check_stream_decoder(const Tokenizer& tok, const json& doc) {
    for (const auto& c : doc.at("cases")) {
        if (c.at("kind") != "encode") continue;
        const std::string name = c.at("name").get<std::string>();
        if (name.rfind("emoji_", 0) != 0 && name.rfind("thai_", 0) != 0 && name.rfind("chinese_", 0) != 0 &&
            name.rfind("japanese_", 0) != 0) {
            continue;
        }
        const std::vector<TokenId> ids = ids_from_json(c.at("ids"));
        const std::string expected = c.at("decoded").get<std::string>();

        auto dec = tok.make_stream_decoder(/*skip_special_tokens=*/true);
        std::string got;
        for (const TokenId id : ids) got += dec.push(id);
        got += dec.flush();

        if (got != expected) {
            fail(name + " (stream)", "streaming decode mismatch\n  expected : " + expected + "\n  got      : " + got);
        }
    }
}

// CTest SKIP_RETURN_CODE (see tests/tokenizer/CMakeLists.txt): this test needs the real,
// un-vendored model directory on disk; report an explicit SKIP rather than a pass-with-no-coverage
// or a hard failure when it is absent.
constexpr int kSkipReturnCode = 77;

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

// Compares an accessor's id against golden.json's recorded id, where the golden field may be JSON
// null (meaning "no such token", i.e. r4dx::kNoToken).
void check_id_field(const char* field_name, const json& doc, TokenId got) {
    if (!doc.contains(field_name)) return;
    const json& v = doc.at(field_name);
    const TokenId expected = v.is_null() ? r4dx::kNoToken : v.get<TokenId>();
    if (got != expected) {
        fail(std::string("header.") + field_name,
             "expected " + std::to_string(expected) + ", got " + std::to_string(got));
    }
}

// Replays golden.json's "compat_cases" (generated via AutoTokenizer, i.e. transformers 5.17.0's
// Qwen2Tokenizer pre-tokenizer bug -- see the block comment above kQwen2SplitTrigger in
// bpe_tokenizer.cpp) against a Tokenizer loaded with Options{.hf_transformers_compat=true}, so the
// opt-in compat path itself is under test, not just documented.
void check_compat_cases(const std::string& model_dir, const json& doc) {
    if (!doc.contains("compat_cases")) return;
    Tokenizer::Options options;
    options.hf_transformers_compat = true;
    options.allow_unimplemented_normalizer = true;  // same known NFC gap as the default tokenizer
    Tokenizer compat_tok;
    try {
        compat_tok = Tokenizer::from_directory(model_dir, options);
    } catch (const std::exception& e) {
        fail("<compat_cases>", std::string("failed to load hf_transformers_compat tokenizer: ") + e.what());
        return;
    }
    for (const auto& c : doc.at("compat_cases")) {
        check_encode_case(compat_tok, c);
    }
}

}  // namespace

int main() {
    const std::string model_dir = R4DX_TOKENIZER_MODEL_DIR;
    const std::string golden_path = std::string(TOKENIZER_TEST_DIR) + "/golden.json";

    if (!file_exists(model_dir + "/tokenizer.json")) {
        std::fprintf(stderr,
                      "tokenizer_golden: SKIPPED -- %s/tokenizer.json not found (set "
                      "-DR4DX_TOKENIZER_MODEL_DIR=... to point at the Qwen3.8-27B checkout)\n",
                      model_dir.c_str());
        return kSkipReturnCode;
    }

    // Qwen3.8-27B declares normalizer.type=NFC, which this implementation does not apply (see the
    // KNOWN GAP comment in tokenizer.h); the golden corpus is NFC-normalized by construction (see
    // tools/tok_ref/gen_golden.py), so this known, already-accounted-for gap is explicitly
    // acknowledged here rather than silently ignored.
    Tokenizer::Options default_options;
    default_options.allow_unimplemented_normalizer = true;

    Tokenizer tok;
    r4dx::ChatTemplate tmpl;
    try {
        tok = Tokenizer::from_directory(model_dir, default_options);
        tmpl = r4dx::ChatTemplate::from_directory(model_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "golden_test: failed to load tokenizer/chat template from %s: %s\n", model_dir.c_str(),
                     e.what());
        return 1;
    }

    json doc = json::parse(read_file(golden_path), /*cb=*/nullptr, /*allow_exceptions=*/true);

    check_id_field("vocab_size", doc, static_cast<TokenId>(tok.vocab_size()));
    check_id_field("bos_token_id", doc, tok.bos_id());
    check_id_field("eos_token_id", doc, tok.eos_id());
    check_id_field("pad_token_id", doc, tok.pad_id());

    size_t n_encode = 0, n_chat = 0;
    for (const auto& c : doc.at("cases")) {
        const std::string kind = c.at("kind").get<std::string>();
        if (kind == "encode") {
            check_encode_case(tok, c);
            n_encode++;
        } else if (kind == "chat") {
            check_chat_case(tok, tmpl, c);
            n_chat++;
        } else {
            fail("<unknown>", "unrecognized case kind: " + kind);
        }
    }
    check_stream_decoder(tok, doc);
    check_compat_cases(model_dir, doc);

    std::printf("tokenizer_golden: %zu encode cases, %zu chat cases, %d failure(s)\n", n_encode, n_chat, g_failures);
    return g_failures == 0 ? 0 : 1;
}
