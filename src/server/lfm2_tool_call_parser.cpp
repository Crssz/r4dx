// LFM2.5 tool-call parser (docs/lfm2-plan.md section 5.3). Surface syntax, from LiquidAI/LFM2.5-24B-A2B's
// chat_template.jinja (render direction, `render_tool_calls` / `format_arg_value`):
//
//   <|tool_call_start|>[name(arg='text', n=3, flag=True, nested=[1, "two", true, null, {"k": "v"}]), other()]
//   <|tool_call_end|>
//
// i.e. a Python list of keyword-only calls. Values are Python literals: strings in single or double quotes
// (the template renders a string argument as '...' with \\ \' \n \r escaped, and a nested list/dict with
// `tojson`, so JSON double-quoted strings, true/false/null and JSON escapes such as \uXXXX also occur),
// ints, floats, True/False/None, lists and dicts with string keys. Recursive descent over the span, not
// "find the next <|tool_call_end|>", so a string argument that contains the close marker does not end the
// call early. Multiple calls are one list; several spans are simply concatenated.
//
// Strict where Python is strict: positional arguments, bare words that are not True/False/None/true/
// false/null, an unbalanced bracket, a missing close marker (a generation truncated by max_tokens) and an
// empty list make the WHOLE span malformed: it degrades to literal content (raw text, markers included)
// with had_malformed_call = true; never throws. Lenient where Python is lenient: whitespace/newlines
// between tokens, trailing commas, and any run of non-delimiter characters as a call or argument name
// (the template renders OpenAI tool names verbatim, e.g. "mcp__srv__tool-1", which is not a Python
// identifier). A JSON tool-call list (`[{"name": ..., "arguments": ...}]`, what the model emits if the
// system prompt asks for JSON) is NOT parsed here (plan 5.3): it is malformed -> content.
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "nlohmann/json.hpp"
#include "tool_call_parser.h"

namespace r4dx::server {

namespace {

using Json = nlohmann::ordered_json;

constexpr int kMaxDepth = 64;

void AppendUtf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

int HexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

class Cursor {
 public:
  explicit Cursor(std::string_view s) : s_(s) {}

  size_t pos() const { return p_; }
  bool eof() const { return p_ >= s_.size(); }
  char cur() const { return s_[p_]; }
  void SkipWs() {
    while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
  }
  bool Peek(char c) const { return p_ < s_.size() && s_[p_] == c; }
  bool Eat(char c) {
    if (!Peek(c)) return false;
    ++p_;
    return true;
  }
  bool StartsWith(std::string_view t) const {
    return p_ + t.size() <= s_.size() && s_.compare(p_, t.size(), t) == 0;
  }
  bool Eat(std::string_view t) {
    if (!StartsWith(t)) return false;
    p_ += t.size();
    return true;
  }

  // Run of bytes that are not whitespace and not one of `( ) [ ] { } , = : ' "`: a call or argument name,
  // a bare number, or a bare literal word.
  std::string_view EatWord() {
    const size_t start = p_;
    while (p_ < s_.size()) {
      const unsigned char c = static_cast<unsigned char>(s_[p_]);
      if (std::isspace(c) || std::string_view("()[]{},=:'\"").find(static_cast<char>(c)) != std::string_view::npos)
        break;
      ++p_;
    }
    return s_.substr(start, p_ - start);
  }

  // A quoted string at the cursor (either quote), unescaped Python-style. Fails on an unterminated string.
  bool EatString(std::string& out) {
    if (eof() || (cur() != '\'' && cur() != '"')) return false;
    const char q = s_[p_++];
    out.clear();
    while (p_ < s_.size()) {
      const char c = s_[p_++];
      if (c == q) return true;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p_ >= s_.size()) return false;
      const char e = s_[p_++];
      switch (e) {
        case '\\': out += '\\'; break;
        case '\'': out += '\''; break;
        case '"': out += '"'; break;
        case '/': out += '/'; break;  // JSON
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'v': out += '\v'; break;
        case 'a': out += '\a'; break;
        case '0': out += '\0'; break;
        case '\n': break;  // Python line continuation
        case 'x':
        case 'u':
        case 'U': {
          const size_t n = e == 'x' ? 2 : (e == 'u' ? 4 : 8);
          uint32_t cp = 0;
          bool ok = p_ + n <= s_.size();
          for (size_t i = 0; ok && i < n; ++i) {
            const int h = HexVal(s_[p_ + i]);
            if (h < 0) ok = false;
            else cp = cp * 16 + static_cast<uint32_t>(h);
          }
          if (!ok || cp > 0x10FFFF) {  // Python: a SyntaxError; keep the text verbatim instead
            out += '\\';
            out += e;
            break;
          }
          p_ += n;
          // JSON surrogate pair 😀
          if (e == 'u' && cp >= 0xD800 && cp <= 0xDBFF && p_ + 6 <= s_.size() && s_[p_] == '\\' &&
              s_[p_ + 1] == 'u') {
            uint32_t lo = 0;
            bool lo_ok = true;
            for (size_t i = 0; i < 4; ++i) {
              const int h = HexVal(s_[p_ + 2 + i]);
              if (h < 0) lo_ok = false;
              else lo = lo * 16 + static_cast<uint32_t>(h);
            }
            if (lo_ok && lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              p_ += 6;
            }
          }
          if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;  // lone surrogate
          AppendUtf8(out, cp);
          break;
        }
        default:  // Python keeps the backslash of an unknown escape
          out += '\\';
          out += e;
          break;
      }
    }
    return false;  // ran off the end: unterminated
  }

 private:
  std::string_view s_;
  size_t p_ = 0;
};

bool IsNumber(std::string_view t) {
  size_t i = 0;
  if (i < t.size() && (t[i] == '-' || t[i] == '+')) ++i;
  size_t digits = 0;
  while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i, ++digits;
  if (i < t.size() && t[i] == '.') {
    ++i;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i, ++digits;
  }
  if (digits == 0) return false;
  if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
    ++i;
    if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
    const size_t f = i;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
    if (i == f) return false;
  }
  return i == t.size();
}

// Python number text -> Json. Accepts what JSON does plus a leading '+', a leading '.', a trailing '.'
// and underscores-free forms; ints too large for int64/uint64 become doubles (nlohmann's own rule).
std::optional<Json> NumberFromText(std::string_view t) {
  std::string s(t);
  if (!s.empty() && s[0] == '+') s.erase(0, 1);
  const bool neg = !s.empty() && s[0] == '-';
  if (neg) s.erase(0, 1);
  if (!s.empty() && s[0] == '.') s.insert(0, "0");
  const size_t dot = s.find('.');
  if (dot != std::string::npos && (dot + 1 == s.size() || !std::isdigit(static_cast<unsigned char>(s[dot + 1])))) {
    s.insert(dot + 1, "0");  // "5." / "5.e3"
  }
  // JSON forbids leading zeros ("007"); Python forbids them too for non-zero ints.
  if (s.size() > 1 && s[0] == '0' && std::isdigit(static_cast<unsigned char>(s[1]))) return std::nullopt;
  if (neg) s.insert(0, "-");
  try {
    return Json::parse(s);
  } catch (const nlohmann::json::exception&) {
    return std::nullopt;
  }
}

bool ParseValue(Cursor& c, Json& out, int depth);

bool ParseList(Cursor& c, Json& out, int depth) {
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
      if (c.Eat(']')) return true;  // trailing comma
      continue;
    }
    return c.Eat(']');
  }
}

bool ParseDict(Cursor& c, Json& out, int depth) {
  if (depth > kMaxDepth || !c.Eat('{')) return false;
  out = Json::object();
  c.SkipWs();
  if (c.Eat('}')) return true;
  for (;;) {
    c.SkipWs();
    std::string key;
    if (!c.EatString(key)) return false;  // string keys only (JSON-compatible)
    c.SkipWs();
    if (!c.Eat(':')) return false;
    Json v;
    if (!ParseValue(c, v, depth + 1)) return false;
    out[key] = std::move(v);
    c.SkipWs();
    if (c.Eat(',')) {
      c.SkipWs();
      if (c.Eat('}')) return true;
      continue;
    }
    return c.Eat('}');
  }
}

bool ParseValue(Cursor& c, Json& out, int depth) {
  c.SkipWs();
  if (c.eof()) return false;
  if (c.cur() == '\'' || c.cur() == '"') {
    std::string s;
    if (!c.EatString(s)) return false;
    out = std::move(s);
    return true;
  }
  if (c.cur() == '[') return ParseList(c, out, depth);
  if (c.cur() == '{') return ParseDict(c, out, depth);
  const std::string_view w = c.EatWord();
  if (w.empty()) return false;
  if (w == "True" || w == "true") { out = true; return true; }
  if (w == "False" || w == "false") { out = false; return true; }
  if (w == "None" || w == "null") { out = nullptr; return true; }
  if (IsNumber(w)) {
    if (auto n = NumberFromText(w)) {
      out = std::move(*n);
      return true;
    }
  }
  return false;  // bare identifier / unsupported literal
}

// NAME "(" [ IDENT "=" value ("," IDENT "=" value)* [","] ] ")"
std::optional<ParsedToolCall> ParseOneCall(Cursor& c) {
  c.SkipWs();
  const std::string_view name = c.EatWord();
  if (name.empty()) return std::nullopt;
  c.SkipWs();
  if (!c.Eat('(')) return std::nullopt;
  Json args = Json::object();
  c.SkipWs();
  if (!c.Eat(')')) {
    for (;;) {
      c.SkipWs();
      const std::string_view key = c.EatWord();
      if (key.empty()) return std::nullopt;  // also rejects a positional string/number argument
      c.SkipWs();
      if (!c.Eat('=')) return std::nullopt;  // positional bare word
      Json v;
      if (!ParseValue(c, v, 0)) return std::nullopt;
      args[std::string(key)] = std::move(v);
      c.SkipWs();
      if (c.Eat(',')) {
        c.SkipWs();
        if (c.Eat(')')) break;  // trailing comma
        continue;
      }
      if (!c.Eat(')')) return std::nullopt;
      break;
    }
  }
  ParsedToolCall call;
  call.name = std::string(name);
  call.arguments_json = args.dump(-1, ' ', false, Json::error_handler_t::replace);
  return call;
}

// `[call, call, ...]` followed (modulo whitespace) by the close marker. On success appends the calls and
// sets *end to the offset just past the close marker; `begin` is the offset just past the opener.
bool ParseSpan(std::string_view text, size_t begin, std::string_view close, std::vector<ParsedToolCall>* calls,
               size_t* end) {
  Cursor c(text.substr(begin));
  c.SkipWs();
  if (!c.Eat('[')) return false;
  std::vector<ParsedToolCall> parsed;
  for (;;) {
    c.SkipWs();
    std::optional<ParsedToolCall> call = ParseOneCall(c);
    if (!call) return false;
    parsed.push_back(std::move(*call));
    c.SkipWs();
    if (c.Eat(',')) {
      c.SkipWs();
      if (c.Peek(']')) {  // trailing comma
        c.Eat(']');
        break;
      }
      continue;
    }
    if (!c.Eat(']')) return false;
    break;
  }
  c.SkipWs();
  if (!c.Eat(close)) return false;
  *end = begin + c.pos();
  for (auto& p : parsed) calls->push_back(std::move(p));
  return true;
}

}  // namespace

ToolCallParseResult ParseLfm2ToolCalls(const std::string& text, const ModelDialect& d) {
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
    if (ParseSpan(text, inner_start, close, &result.tool_calls, &end)) {
      pos = end;
      continue;
    }
    result.had_malformed_call = true;
    // Resynchronize on the next close marker; a span that never closes swallows the rest as content.
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
