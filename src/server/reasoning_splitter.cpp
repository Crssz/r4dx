#include "reasoning_splitter.h"

#include <algorithm>
#include <cstdint>

namespace r4dx::server {

namespace {

constexpr char kTag[] = "</think>";
constexpr size_t kTagLen = 8;  // strlen(kTag), fixed at compile time -- avoid a runtime strlen call

// Longest k (0 <= k <= kTagLen-1) such that the last k bytes of `buf` equal the first k bytes of
// kTag -- i.e. the longest suffix of `buf` that could still grow into a full tag match given more
// input. 0 means no suffix of `buf` could possibly be a tag prefix.
size_t LongestTagPrefixSuffix(const std::string& buf) {
  const size_t max_k = std::min(kTagLen - 1, buf.size());
  // Signed loop variable: `k` counts down to 0 and must not wrap (size_t 0 - 1 is huge) the way an
  // unsigned `for (size_t k = max_k; k >= 1; --k)` would on its final iteration.
  for (int64_t k = static_cast<int64_t>(max_k); k >= 1; --k) {
    const size_t uk = static_cast<size_t>(k);
    if (buf.compare(buf.size() - uk, uk, kTag, 0, uk) == 0) return uk;
  }
  return 0;
}

}  // namespace

std::vector<ReasoningSplitter::Event> ReasoningSplitter::Push(const std::string& piece) {
  std::vector<Event> events;
  if (piece.empty()) return events;

  if (mode_ == Mode::kAnswer) {
    ProcessAnswer(piece, events);
    return events;
  }

  std::string buf = tag_hold_ + piece;
  tag_hold_.clear();

  const size_t p = buf.find(kTag);
  if (p == std::string::npos) {
    const size_t hold_len = LongestTagPrefixSuffix(buf);
    if (buf.size() > hold_len) events.push_back({true, buf.substr(0, buf.size() - hold_len)});
    tag_hold_ = buf.substr(buf.size() - hold_len);
    return events;
  }

  if (p > 0) events.push_back({true, buf.substr(0, p)});
  mode_ = Mode::kAnswer;
  ProcessAnswer(buf.substr(p + kTagLen), events);
  return events;
}

void ReasoningSplitter::ProcessAnswer(const std::string& buf, std::vector<Event>& events) {
  if (answer_started_) {
    if (!buf.empty()) events.push_back({false, buf});
    return;
  }

  std::string combined = nl_hold_ + buf;
  nl_hold_.clear();
  size_t i = 0;
  while (i < combined.size() && (combined[i] == '\n' || combined[i] == '\r')) ++i;
  if (i == combined.size()) {
    // Nothing but blank lines so far -- keep holding (could still all turn out to be trailing
    // blank lines with nothing after, per Finish()'s own contract, or more text may still arrive).
    nl_hold_ = combined;
    return;
  }

  answer_started_ = true;
  const std::string content = combined.substr(i);
  if (!content.empty()) events.push_back({false, content});
}

std::vector<ReasoningSplitter::Event> ReasoningSplitter::Finish() {
  std::vector<Event> events;
  if (mode_ == Mode::kReasoning) {
    if (!tag_hold_.empty()) events.push_back({true, tag_hold_});
    tag_hold_.clear();
  } else {
    nl_hold_.clear();
  }
  return events;
}

std::string TrimReasoningWhitespace(const std::string& s) {
  auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  size_t start = 0;
  while (start < s.size() && is_ws(static_cast<unsigned char>(s[start]))) ++start;
  size_t end = s.size();
  while (end > start && is_ws(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(start, end - start);
}

}  // namespace r4dx::server
