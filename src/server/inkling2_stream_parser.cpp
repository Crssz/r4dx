#include "inkling2_stream_parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace r4dx::server {

namespace {

constexpr std::string_view kMsg = "<|message_model|>";
constexpr std::string_view kThink = "<|content_thinking|>";
constexpr std::string_view kText = "<|content_text|>";
constexpr std::string_view kTool = "<|content_invoke_tool_json|>";
constexpr std::string_view kEnd = "<|end_message|>";
constexpr std::string_view kMarkers[] = {kMsg, kThink, kText, kTool, kEnd};
constexpr size_t kMaxHold = 28;  // the longest marker (kTool) is 28 bytes: at most 27 are ever held back
constexpr size_t kMaxHeader = 1024;

// Position of the earliest full marker in `s` and which one; npos when none.
size_t FindMarker(const std::string& s, std::string_view* which) {
  size_t best = std::string::npos;
  for (const std::string_view m : kMarkers) {
    const size_t p = s.find(m);
    if (p != std::string::npos && p < best) {
      best = p;
      *which = m;
    }
  }
  return best;
}

// Start of the trailing bytes of `s` that are a proper prefix of some marker (so they may still become one); s.size()
// when there are none.
size_t HoldStart(const std::string& s) {
  const size_t n = s.size();
  const size_t lo = n > kMaxHold + 1 ? n - (kMaxHold + 1) : 0;
  for (size_t p = lo; p < n; ++p) {
    if (s[p] != '<') continue;
    const std::string_view tail(s.data() + p, n - p);
    for (const std::string_view m : kMarkers) {
      if (tail.size() < m.size() && m.substr(0, tail.size()) == tail) return p;
    }
  }
  return n;
}

void Append(std::vector<Inkling2StreamParser::Event>& out, Inkling2StreamParser::Event::Kind kind, std::string text) {
  if (text.empty()) return;
  if (!out.empty() && out.back().kind == kind && kind != Inkling2StreamParser::Event::Kind::kToolCall) {
    out.back().text += text;
    return;
  }
  Inkling2StreamParser::Event e;
  e.kind = kind;
  e.text = std::move(text);
  out.push_back(std::move(e));
}

bool IsBlank(const std::string& s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; });
}

std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

}  // namespace

const std::vector<std::string>& Inkling2KeepSpecial() {
  static const std::vector<std::string> v = {std::string(kMsg), std::string(kThink), std::string(kText), std::string(kTool),
                                             std::string(kEnd)};
  return v;
}

void Inkling2StreamParser::EmitText(State kind, std::string text, std::vector<Event>& out) {
  if (text.empty()) return;
  switch (kind) {
    case State::kThinking:
      Append(out, Event::Kind::kReasoning, std::move(text));
      break;
    case State::kText:
    case State::kBetween:
      Append(out, Event::Kind::kContent, std::move(text));
      break;
    case State::kHeader:
      header_ += text;
      break;
    case State::kTool:
      tool_body_ += text;
      break;
  }
}

void Inkling2StreamParser::Degrade(std::string raw, std::vector<Event>& out) {
  malformed_ = true;
  Append(out, Event::Kind::kContent, std::move(raw));
}

void Inkling2StreamParser::FinishTool(std::vector<Event>& out) {
  const std::string raw = raw_open_ + tool_body_ + std::string(kEnd);
  const nlohmann::ordered_json j = nlohmann::ordered_json::parse(tool_body_, nullptr, /*allow_exceptions=*/false);
  std::string name = Trim(header_);
  nlohmann::ordered_json args = nlohmann::ordered_json::object();
  bool ok = j.is_object();
  if (ok) {
    auto n = j.find("name");
    if (n != j.end()) {
      if (n->is_string() && !n->get<std::string>().empty()) {
        name = n->get<std::string>();
      } else {
        ok = false;
      }
    }
    auto a = j.find("args");
    if (ok && a != j.end() && !a->is_null()) {
      if (a->is_object()) {
        args = *a;
      } else if (a->is_string()) {  // a model that double-encodes its arguments
        nlohmann::ordered_json inner = nlohmann::ordered_json::parse(a->get<std::string>(), nullptr, false);
        if (inner.is_object()) args = std::move(inner); else ok = false;
      } else {
        ok = false;
      }
    }
  }
  if (!ok || name.empty()) {
    Degrade(raw, out);
    return;
  }
  Event e;
  e.kind = Event::Kind::kToolCall;
  e.call.name = std::move(name);
  e.call.arguments_json = args.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
  out.push_back(std::move(e));
  ++tool_calls_;
}

void Inkling2StreamParser::Drain(bool final, std::vector<Event>& out) {
  while (true) {
    std::string_view which;
    const size_t mpos = FindMarker(buf_, &which);
    const size_t limit = mpos != std::string::npos ? mpos : (final ? buf_.size() : HoldStart(buf_));
    if (limit > 0) {
      std::string payload = buf_.substr(0, limit);
      buf_.erase(0, limit);
      EmitText(state_, std::move(payload), out);
      if (state_ == State::kHeader && header_.size() > kMaxHeader) {  // not a header at all: give up on it
        Degrade(std::string(kMsg) + header_, out);
        header_.clear();
        state_ = State::kBetween;
      }
    }
    if (mpos == std::string::npos) return;  // nothing complete left (the rest is held back or flushed above)

    buf_.erase(0, which.size());
    if (which == kMsg) {
      if (state_ == State::kHeader) Degrade(std::string(kMsg) + header_, out);
      else if (state_ == State::kTool) Degrade(raw_open_ + tool_body_, out);
      header_.clear();
      tool_body_.clear();
      state_ = State::kHeader;
    } else if (which == kEnd) {
      if (state_ == State::kTool) {
        FinishTool(out);
      } else if (state_ == State::kHeader) {
        Degrade(std::string(kMsg) + header_ + std::string(kEnd), out);
      }
      header_.clear();
      tool_body_.clear();
      state_ = State::kBetween;
    } else {  // a content-kind marker
      if (state_ == State::kBetween || state_ == State::kHeader) {
        const State next = which == kThink ? State::kThinking : which == kText ? State::kText : State::kTool;
        if (next == State::kTool) {
          raw_open_ = (state_ == State::kHeader ? std::string(kMsg) + header_ : std::string()) + std::string(kTool);
          tool_body_.clear();
          // header_ keeps the NAME (the fallback for a body without "name")
        } else {
          header_.clear();
        }
        state_ = next;
      } else {
        // a content-kind marker inside a part: literal text of that part
        EmitText(state_, std::string(which), out);
      }
    }
  }
}

std::vector<Inkling2StreamParser::Event> Inkling2StreamParser::Push(std::string_view piece) {
  std::vector<Event> out;
  buf_.append(piece);
  Drain(/*final=*/false, out);
  return out;
}

std::vector<Inkling2StreamParser::Event> Inkling2StreamParser::Finish() {
  std::vector<Event> out;
  Drain(/*final=*/true, out);
  if (state_ == State::kTool) {
    Degrade(raw_open_ + tool_body_, out);
  } else if (state_ == State::kHeader) {
    Degrade(std::string(kMsg) + header_, out);
  }
  header_.clear();
  tool_body_.clear();
  state_ = State::kBetween;
  return out;
}

Inkling2Parsed ParseInkling2Generation(std::string_view text) {
  Inkling2StreamParser p;
  Inkling2Parsed r;
  auto take = [&](std::vector<Inkling2StreamParser::Event> events) {
    for (auto& e : events) {
      switch (e.kind) {
        case Inkling2StreamParser::Event::Kind::kReasoning: r.reasoning += e.text; break;
        case Inkling2StreamParser::Event::Kind::kContent: r.content += e.text; break;
        case Inkling2StreamParser::Event::Kind::kToolCall: r.tool_calls.push_back(std::move(e.call)); break;
      }
    }
  };
  take(p.Push(text));
  take(p.Finish());
  r.had_malformed = p.had_malformed();
  return r;
}

void CanonicalizeInkling2Messages(nlohmann::ordered_json& messages) {
  if (!messages.is_array()) return;
  for (size_t i = 0; i < messages.size(); ++i) {
    auto& m = messages[i];
    if (!m.is_object() || !m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
    auto& calls = m["tool_calls"];
    for (size_t k = 0; k < calls.size(); ++k) {
      if (!calls[k].is_object() || !calls[k].contains("function") || !calls[k]["function"].is_object()) continue;
      auto& fn = calls[k]["function"];
      auto a = fn.find("arguments");
      if (a == fn.end() || !a->is_string()) continue;
      const std::string s = a->get<std::string>();
      if (IsBlank(s)) {
        *a = nlohmann::ordered_json::object();
        continue;
      }
      nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(s, nullptr, /*allow_exceptions=*/false);
      if (!parsed.is_object()) {
        throw std::invalid_argument("messages[" + std::to_string(i) + "].tool_calls[" + std::to_string(k) +
                                    "].function.arguments is not a JSON object");
      }
      *a = std::move(parsed);
    }
  }
}

std::optional<nlohmann::ordered_json> ResolveInkling2ReasoningEffort(const nlohmann::json& chat_template_kwargs,
                                                                      std::optional<bool> enable_thinking,
                                                                      const nlohmann::json& request_effort) {
  if (chat_template_kwargs.is_object()) {
    auto it = chat_template_kwargs.find("reasoning_effort");
    if (it != chat_template_kwargs.end() && !it->is_null()) return nlohmann::ordered_json::parse(it->dump());
  }
  if (enable_thinking.has_value() && !*enable_thinking) return nlohmann::ordered_json("none");
  if (request_effort.is_null() || request_effort.is_discarded()) return std::nullopt;
  if (request_effort.is_number()) {
    const double v = request_effort.get<double>();
    if (!(v >= 0.0 && v <= 0.99)) {
      throw std::invalid_argument("'reasoning_effort' must be in [0.0, 0.99] for this model");
    }
    return nlohmann::ordered_json(v);
  }
  if (request_effort.is_string()) {
    std::string s = Trim(request_effort.get<std::string>());
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    static const char* const kLevels[] = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};
    for (const char* level : kLevels) {
      if (s == level) return nlohmann::ordered_json(s);
    }
    throw std::invalid_argument("'reasoning_effort' must be one of none, minimal, low, medium, high, xhigh, max or a number "
                                "in [0.0, 0.99]");
  }
  throw std::invalid_argument("'reasoning_effort' must be a string or a number");
}

nlohmann::json Inkling2RequestEffort(const nlohmann::json& body) {
  if (!body.is_object()) return nullptr;
  auto r = body.find("reasoning");
  if (r != body.end() && r->is_object()) {
    auto e = r->find("effort");
    if (e != r->end() && !e->is_null()) return *e;
  }
  auto e = body.find("reasoning_effort");
  if (e != body.end() && !e->is_null()) return *e;
  return nullptr;
}

}  // namespace r4dx::server
