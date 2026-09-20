#include "tool_call_parser.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string_view>

#include "nlohmann/json.hpp"

namespace r4dx::server {

namespace {

constexpr const char* kOpen = "<tool_call>";
constexpr const char* kClose = "</tool_call>";
constexpr size_t kOpenLen = 11;   // strlen("<tool_call>")
constexpr size_t kCloseLen = 12;  // strlen("</tool_call>")

bool IsBlank(std::string_view s) {
  for (char c : s) {
    if (!std::isspace(static_cast<unsigned char>(c))) return false;
  }
  return true;
}

// Strips exactly one leading '\n' and one trailing '\n' if present (the template wraps every
// parameter value with a single '\n' on each side -- see chat_template.jinja's own
// '<parameter=' + name + '>\n' / '\n</parameter>' rendering). A multi-line value's OWN internal
// newlines are untouched, matching the task's explicit "can span multiple lines" example.
std::string StripOneSurroundingNewline(std::string_view s) {
  size_t begin = 0, end = s.size();
  if (begin < end && s[begin] == '\n') ++begin;
  if (end > begin && s[end - 1] == '\n') --end;
  return std::string(s.substr(begin, end - begin));
}

// Parses one <parameter=NAME>\nVALUE\n</parameter> block sequence (the text strictly between a
// call's `<function=NAME>` and its `</function>`) into a JSON object, in encounter order.
// Returns std::nullopt if the structure itself is broken (a missing '>' or '</parameter>', or
// non-whitespace text where only tag structure is expected) -- a parameter VALUE that merely
// fails to parse as JSON is NOT an error here (see file header: it just becomes a JSON string).
std::optional<nlohmann::ordered_json> ParseParameters(std::string_view params_text) {
  nlohmann::ordered_json args = nlohmann::ordered_json::object();
  size_t pos = 0;
  while (pos < params_text.size()) {
    const size_t p_start = params_text.find("<parameter=", pos);
    if (p_start == std::string_view::npos) {
      if (!IsBlank(params_text.substr(pos))) return std::nullopt;  // stray prose, not whitespace
      break;
    }
    if (!IsBlank(params_text.substr(pos, p_start - pos))) return std::nullopt;

    const size_t name_start = p_start + 11;  // strlen("<parameter=")
    const size_t name_end = params_text.find('>', name_start);
    if (name_end == std::string_view::npos) return std::nullopt;
    const std::string_view raw_name = params_text.substr(name_start, name_end - name_start);
    if (IsBlank(raw_name)) return std::nullopt;
    const std::string name(raw_name);

    const size_t val_start = name_end + 1;
    const size_t p_close = params_text.find("</parameter>", val_start);
    if (p_close == std::string_view::npos) return std::nullopt;
    const std::string value = StripOneSurroundingNewline(params_text.substr(val_start, p_close - val_start));

    // Type inference convention -- see tool_call_parser.h's file comment.
    try {
      args[name] = nlohmann::ordered_json::parse(value);
    } catch (const nlohmann::json::exception&) {
      // Catches parse_error (syntactically not JSON) AND out_of_range/type_error (syntactically
      // JSON but not representable, e.g. a numeric literal like "1e999" that overflows double --
      // nlohmann::json::out_of_range is a SIBLING of parse_error, not a subclass, so a narrower
      // catch here previously let it escape uncaught; see tool_call_parser.h's "never throws"
      // contract). Either way, degrade to a plain JSON string, same as any other unparseable value.
      args[name] = value;
    }

    pos = p_close + 12;  // strlen("</parameter>")
  }
  return args;
}

// Parses the text strictly between one call's `<tool_call>` and `</tool_call>` markers. Returns
// std::nullopt if the block's structure doesn't match this checkpoint's confirmed surface syntax
// (see tool_call_parser.h) -- the caller falls back to treating the whole raw span as content.
std::optional<ParsedToolCall> ParseOneCall(std::string_view inner) {
  const size_t fn_tag = inner.find("<function=");
  if (fn_tag == std::string_view::npos) return std::nullopt;
  if (!IsBlank(inner.substr(0, fn_tag))) return std::nullopt;  // prose before <function=...>

  const size_t name_start = fn_tag + 10;  // strlen("<function=")
  const size_t name_end = inner.find('>', name_start);
  if (name_end == std::string_view::npos) return std::nullopt;
  const std::string_view raw_name = inner.substr(name_start, name_end - name_start);
  if (IsBlank(raw_name)) return std::nullopt;

  const size_t fn_close = inner.find("</function>", name_end + 1);
  if (fn_close == std::string_view::npos) return std::nullopt;
  if (!IsBlank(inner.substr(fn_close + 11))) return std::nullopt;  // prose after </function>

  const std::optional<nlohmann::ordered_json> args =
      ParseParameters(inner.substr(name_end + 1, fn_close - (name_end + 1)));
  if (!args) return std::nullopt;

  ParsedToolCall call;
  call.name = std::string(raw_name);
  call.arguments_json = args->dump();
  return call;
}

}  // namespace

ToolCallParseResult ParseToolCalls(const std::string& text) {
  ToolCallParseResult result;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t tc_start = text.find(kOpen, pos);
    if (tc_start == std::string::npos) {
      result.content += text.substr(pos);
      break;
    }
    result.content += text.substr(pos, tc_start - pos);

    const size_t inner_start = tc_start + kOpenLen;
    const size_t tc_end = text.find(kClose, inner_start);
    if (tc_end == std::string::npos) {
      // Unclosed <tool_call> -- everything from here to end of string is unparseable; there is
      // nothing sensible left to scan after it.
      result.content += text.substr(tc_start);
      result.had_malformed_call = true;
      break;
    }

    const std::string_view inner(text.data() + inner_start, tc_end - inner_start);
    std::optional<ParsedToolCall> call = ParseOneCall(inner);
    if (call) {
      result.tool_calls.push_back(std::move(*call));
    } else {
      result.content += text.substr(tc_start, (tc_end + kCloseLen) - tc_start);
      result.had_malformed_call = true;
    }
    pos = tc_end + kCloseLen;
  }
  return result;
}

int DropUnknownToolCalls(ToolCallParseResult& result, const std::vector<std::string>& known_names) {
  if (known_names.empty() || result.tool_calls.empty()) return 0;
  std::vector<ParsedToolCall> kept;
  kept.reserve(result.tool_calls.size());
  int dropped = 0;
  for (auto& call : result.tool_calls) {
    const bool known = std::find(known_names.begin(), known_names.end(), call.name) != known_names.end();
    if (known) {
      kept.push_back(std::move(call));
    } else {
      result.content += "[dropped call to undefined tool '" + call.name + "': " + call.arguments_json + "]";
      ++dropped;
    }
  }
  result.tool_calls = std::move(kept);
  return dropped;
}

}  // namespace r4dx::server
