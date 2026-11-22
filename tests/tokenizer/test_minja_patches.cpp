// CTest 'tokenizer_minja_patches': self-contained checks (no model files) of the local patches to
// third_party/minja/minja.hpp and of ChatTemplateOptions, all of which Gemma 4's chat template
// depends on (docs/gemma4-plan.md section 5.1 item 4; third_party/VERSIONS.md "Local patches").
#include <cstdio>
#include <string>

#include "chat_template.h"

using r4dx::ChatJson;
using r4dx::ChatTemplate;
using r4dx::ChatTemplateOptions;

namespace {

int g_failures = 0;

void expect_eq(const char* what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL [%s]\n  expected: %s\n  got     : %s\n", what, want.c_str(), got.c_str());
        g_failures++;
    }
}

std::string render(const std::string& tmpl, const ChatJson& messages = ChatJson::array(),
                   const ChatJson& extra = ChatJson::object(), const ChatTemplateOptions& opts = {}) {
    return ChatTemplate::from_source(tmpl, "", "", opts).render(messages, /*add_generation_prompt=*/false,
                                                                 ChatJson::array(), extra);
}

}  // namespace

int main() {
    // 1. Adjacent string literals concatenate (Gemma's raise_exception message spans 3 literals).
    expect_eq("adjacent literals", render("{{ 'a' \"b\"\n  'c' }}"), "abc");
    expect_eq("adjacent literals in call", render("{{ ('x' 'y') | upper }}"), "XY");
    expect_eq("single literal unchanged", render("{{ 'a' }}{{ \"b\" }}"), "ab");
    {
        bool threw = false;
        try {
            render("{{ raise_exception(\n \"one \"\n \"two\"\n) }}");
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("one two") != std::string::npos;
        }
        if (!threw) {
            std::fprintf(stderr, "FAIL [raise_exception message]: expected a throw containing \"one two\"\n");
            g_failures++;
        }
    }

    // 2. dictsort is case-insensitive (Jinja default case_sensitive=False), stable on ties.
    {
        const ChatJson extra = ChatJson::parse(R"({"d": {"b": 1, "Z": 2, "a": 3, "B": 4, "A": 5, "c": 6}})");
        expect_eq("dictsort case-insensitive",
                  render("{% for k, v in d | dictsort %}{{ k }}{{ v }} {% endfor %}", ChatJson::array(), extra),
                  "a3 A5 b1 B4 c6 Z2 ");
    }

    // 3. An empty dict is falsy; a non-empty one is truthy.
    {
        const ChatJson extra = ChatJson::parse(R"({"e": {}, "f": {"k": 1}, "g": [], "h": ""})");
        expect_eq("empty dict falsy",
                  render("{% if e %}T{% else %}F{% endif %}{% if f %}T{% else %}F{% endif %}"
                         "{% if g %}T{% else %}F{% endif %}{% if h %}T{% else %}F{% endif %}",
                         ChatJson::array(), extra),
                  "FTFF");
    }

    // 4. ChatTemplateOptions::apply_polyfills. This template never mentions tool_calls, so minja's
    // caps probe says it does not support them and (default) rewrites the message; with
    // apply_polyfills=false the message reaches the template untouched.
    {
        const std::string tmpl = "{% for m in messages %}{{ m.role }}:{% if m.tool_calls %}TC{% else %}-{% endif %};{% endfor %}";
        const ChatJson messages = ChatJson::parse(
            R"([{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":{"a":1}}}]}])");
        ChatTemplateOptions off;
        off.apply_polyfills = false;
        expect_eq("polyfills off keeps tool_calls", render(tmpl, messages, ChatJson::object(), off), "assistant:TC;");
        expect_eq("polyfills on (default) rewrites tool_calls", render(tmpl, messages), "assistant:-;");
    }

    // 5. Inkling-2 patches: trailing commas in dict/array literals, dictsort(case_sensitive=true),
    // tojson(sort_keys=..., separators=...), and the `float` filter.
    {
        ChatTemplateOptions off;
        off.apply_polyfills = false;
        expect_eq("dict trailing comma", render("{% set d = {'a': 1, 'b': 2,} %}{{ d | length }}", ChatJson::array(), ChatJson::object(), off), "2");
        expect_eq("array trailing comma", render("{% set l = [1, 2,] %}{{ l | length }}", ChatJson::array(), ChatJson::object(), off), "2");
        const ChatJson extra = ChatJson::parse(R"({"d": {"b": 1, "Z": 2, "a": 3, "B": 4}})");
        expect_eq("dictsort case_sensitive=true",
                  render("{% for k, v in d | dictsort(case_sensitive=true) %}{{ k }}{{ v }} {% endfor %}", ChatJson::array(), extra, off),
                  "B4 Z2 a3 b1 ");
        expect_eq("dictsort case_sensitive=false",
                  render("{% for k, v in d | dictsort(case_sensitive=false) %}{{ k }}{{ v }} {% endfor %}", ChatJson::array(), extra, off),
                  "a3 b1 B4 Z2 ");
        const ChatJson nested = ChatJson::parse(R"({"v": {"b": [1, {"z": null, "a": "é\n"}], "a": true}})");
        expect_eq("tojson sort_keys + compact separators",
                  render("{{ v | tojson(sort_keys=true, separators=(',', ':')) }}", ChatJson::array(), nested, off),
                  "{\"a\":true,\"b\":[1,{\"a\":\"\xC3\xA9\\n\",\"z\":null}]}");
        expect_eq("tojson separators only keeps insertion order",
                  render("{{ v | tojson(separators=(',', ':')) }}", ChatJson::array(), nested, off),
                  "{\"b\":[1,{\"z\":null,\"a\":\"\xC3\xA9\\n\"}],\"a\":true}");
        expect_eq("tojson sort_keys default separators",
                  render("{{ v | tojson(sort_keys=true) }}", ChatJson::array(), nested, off),
                  "{\"a\": true, \"b\": [1, {\"a\": \"\xC3\xA9\\n\", \"z\": null}]}");
        expect_eq("tojson plain unchanged", render("{{ v | tojson }}", ChatJson::array(), nested, off),
                  "{\"b\": [1, {\"z\": null, \"a\": \"\xC3\xA9\\n\"}], \"a\": true}");
        expect_eq("float filter",
                  render("{{ 3 | float }}|{{ '0.25' | float }}|{{ 'x' | float }}|{{ 1.5 | float }}", ChatJson::array(), ChatJson::object(), off),
                  "3.0|0.25|0.0|1.5");
    }

    std::printf("tokenizer_minja_patches: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
