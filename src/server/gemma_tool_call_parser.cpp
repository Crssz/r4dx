// Gemma 4 tool-call parser (docs/gemma4-plan.md section 5.3). Surface syntax, from google/gemma-4-12B-it's
// chat_template.jinja (render direction) and tokenizer_config.json's response_schema (parse direction,
// x-regex `call\:(?P<name>\w+)(?P<arguments>\{.*\})`, string_delims `<|"|>`, unquoted_keys):
//
//   <|tool_call>call:NAME{key:value,key2:<|"|>text<|"|>,obj:{a:1},list:[1,<|"|>x<|"|>]}<tool_call|>
//
// Strings are wrapped in `<|"|>...<|"|>` (raw bytes in between, no escaping); keys are bare (a quoted
// key `<|"|>k<|"|>` is also accepted, the template emits those inside tool DECLARATIONS); numbers,
// true/false/null are bare; a bare word that is none of those is kept as a string. Recursive descent
// over the span rather than "find the next <tool_call|>", so a string argument that happens to contain
// the close marker does not end the call early. Multiple calls are simply concatenated spans. A span
// that does not parse degrades to literal content (its raw text, markers included) with
// had_malformed_call = true; never throws.
//
// UNVERIFIED against HF `tok.parse_response` (plan section 5.3: bare-word and number edge rules): the
// vectors recorded by tools/tok_ref/gen_golden_gemma.py (M1-9) should be replayed against this parser.
#include <cctype>
#include <optional>
#include <string_view>

#include "nlohmann/json.hpp"
#include "tool_call_parser.h"

namespace r4dx::server {

namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view kEsc = "<|\"|>";
constexpr int kMaxDepth = 64;

class Cursor {
 public:
  explicit Cursor(std::string_view s) : s_(s) {}

  size_t pos() const { return p_; }
  bool eof() const { return p_ >= s_.size(); }
  void SkipWs() {
    while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
  }
  bool Peek(char c) const { return p_ < s_.size() && s_[p_] == c; }
  bool Eat(char c) {
    if (!Peek(c)) return false;
    ++p_;
    return true;
  }
  bool StartsWith(std::string_view t) const { return s_.compare(p_, t.size(), t) == 0 && p_ + t.size() <= s_.size(); }
  bool Eat(std::string_view t) {
    if (!StartsWith(t)) return false;
    p_ += t.size();
    return true;
  }

  // `<|"|>...<|"|>` -> the raw bytes between.
  std::optional<std::string> EatEscapedString() {
    if (!Eat(kEsc)) return std::nullopt;
    const size_t end = s_.find(kEsc, p_);
    if (end == std::string_view::npos) return std::nullopt;
    std::string out(s_.substr(p_, end - p_));
    p_ = end + kEsc.size();
    return out;
  }

  // Bare token up to (not including) any byte in `stops`, trimmed of surrounding whitespace.
  std::string EatBare(std::string_view stops) {
    const size_t start = p_;
    while (p_ < s_.size() && stops.find(s_[p_]) == std::string_view::npos) ++p_;
    size_t b = start, e = p_;
    while (b < e && std::isspace(static_cast<unsigned char>(s_[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s_[e - 1]))) --e;
    return std::string(s_.substr(b, e - b));
  }

 private:
  std::string_view s_;
  size_t p_ = 0;
};

bool LooksLikeNumber(const std::string& t) {
  size_t i = 0;
  if (i < t.size() && t[i] == '-') ++i;
  const size_t int_start = i;
  while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
  if (i == int_start) return false;
  if (i < t.size() && t[i] == '.') {
    ++i;
    const size_t f = i;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
    if (i == f) return false;
  }
  if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
    ++i;
    if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
    const size_t f = i;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
    if (i == f) return false;
  }
  return i == t.size();
}

bool ParseValue(Cursor& c, Json& out, int depth);

bool ParseObject(Cursor& c, Json& out, int depth) {
  if (depth > kMaxDepth || !c.Eat('{')) return false;
  out = Json::object();
  c.SkipWs();
  if (c.Eat('}')) return true;
  for (;;) {
    c.SkipWs();
    std::string key;
    if (c.StartsWith(kEsc)) {
      auto k = c.EatEscapedString();
      if (!k) return false;
      key = std::move(*k);
    } else {
      key = c.EatBare(":,{}[]");
      if (key.empty() || key.find('<') != std::string::npos) return false;
    }
    c.SkipWs();
    if (!c.Eat(':')) return false;
    Json v;
    if (!ParseValue(c, v, depth + 1)) return false;
    out[key] = std::move(v);
    c.SkipWs();
    if (c.Eat(',')) {
      c.SkipWs();
      if (c.Eat('}')) return true;  // tolerate a trailing comma
      continue;
    }
    return c.Eat('}');
  }
}

bool ParseArray(Cursor& c, Json& out, int depth) {
  if (depth > kMaxDepth || !c.Eat('[')) return false;
  out = Json::array();
  c.SkipWs();
  if (c.Eat(']')) return true;
  for (;;) {
    Json v;
    if (!ParseValue(c, v, depth + 1)) return false;
    out.push_back(std::move(v));
    c.SkipWs();
    if (c.Eat(',')) {
      c.SkipWs();
      if (c.Eat(']')) return true;
      continue;
    }
    return c.Eat(']');
  }
}

bool ParseValue(Cursor& c, Json& out, int depth) {
  c.SkipWs();
  if (c.StartsWith(kEsc)) {
    auto s = c.EatEscapedString();
    if (!s) return false;
    out = std::move(*s);
    return true;
  }
  if (c.Peek('{')) return ParseObject(c, out, depth);
  if (c.Peek('[')) return ParseArray(c, out, depth);
  const std::string tok = c.EatBare(",}]");
  if (tok.empty() || tok.find('<') != std::string::npos) return false;
  if (tok == "true") { out = true; return true; }
  if (tok == "false") { out = false; return true; }
  if (tok == "null") { out = nullptr; return true; }
  if (LooksLikeNumber(tok)) {
    try {
      out = Json::parse(tok);
      return true;
    } catch (const nlohmann::json::exception&) {
      // leading zeros / overflow: fall through to a string
    }
  }
  out = tok;
  return true;
}

// `call:NAME{...}` immediately followed (modulo whitespace) by the close marker. On success returns the
// call and sets `*end` to the offset just past the close marker; `begin` is the offset just past the opener.
std::optional<ParsedToolCall> ParseOneGemmaCall(std::string_view text, size_t begin,
                                                std::string_view close, size_t* end) {
  Cursor c(text.substr(begin));
  c.SkipWs();
  if (!c.Eat(std::string_view("call:"))) return std::nullopt;
  const std::string name = c.EatBare("{<");
  if (name.empty()) return std::nullopt;
  for (char ch : name) {
    if (std::isspace(static_cast<unsigned char>(ch))) return std::nullopt;
  }
  Json args = Json::object();
  if (c.Peek('{')) {
    if (!ParseObject(c, args, 0)) return std::nullopt;
  }
  c.SkipWs();
  if (!c.Eat(close)) return std::nullopt;
  *end = begin + c.pos();
  ParsedToolCall call;
  call.name = name;
  call.arguments_json = args.dump(-1, ' ', false, Json::error_handler_t::replace);
  return call;
}

}  // namespace

ToolCallParseResult ParseGemmaToolCalls(const std::string& text, const ModelDialect& d) {
  ToolCallParseResult result;
  const std::string_view open = d.tool_open;
  const std::string_view close = d.tool_close;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t tc_start = text.find(open, pos);
    if (tc_start == std::string::npos) {
      result.content += text.substr(pos);
      break;
    }
    result.content += text.substr(pos, tc_start - pos);

    const size_t inner_start = tc_start + open.size();
    size_t end = 0;
    std::optional<ParsedToolCall> call = ParseOneGemmaCall(text, inner_start, close, &end);
    if (call) {
      result.tool_calls.push_back(std::move(*call));
      pos = end;
      continue;
    }
    result.had_malformed_call = true;
    const size_t tc_end = text.find(close, inner_start);
    if (tc_end == std::string::npos) {
      result.content += text.substr(tc_start);
      break;
    }
    result.content += text.substr(tc_start, tc_end + close.size() - tc_start);
    pos = tc_end + close.size();
  }
  return result;
}

}  // namespace r4dx::server
