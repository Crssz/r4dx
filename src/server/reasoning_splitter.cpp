#include "reasoning_splitter.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace r4dx::server {

namespace {

// Longest k (0 <= k <= tag.size()-1) such that the last k bytes of `buf` equal the first k bytes of
// `tag` -- i.e. the longest suffix of `buf` that could still grow into a full tag match given more
// input. 0 means no suffix of `buf` could possibly be a tag prefix.
size_t LongestTagPrefixSuffix(const std::string& buf, const std::string& tag) {
  const size_t max_k = std::min(tag.size() - 1, buf.size());
  // Signed loop variable: `k` counts down to 0 and must not wrap (size_t 0 - 1 is huge) the way an
  // unsigned `for (size_t k = max_k; k >= 1; --k)` would on its final iteration.
  for (int64_t k = static_cast<int64_t>(max_k); k >= 1; --k) {
    const size_t uk = static_cast<size_t>(k);
    if (buf.compare(buf.size() - uk, uk, tag, 0, uk) == 0) return uk;
  }
  return 0;
}

}  // namespace

ReasoningSplitter::ReasoningSplitter(ReasoningSyntax syntax, StartState start)
    : syntax_(std::move(syntax)) {
  switch (start) {
    case StartState::kInReasoning:
      mode_ = Mode::kReasoning;
      break;
    case StartState::kExpectOpener:
      // An empty opener could never be matched; degrade to "already in reasoning".
      mode_ = syntax_.open.empty() ? Mode::kReasoning : Mode::kExpectOpener;
      break;
    case StartState::kAnswer:
      mode_ = Mode::kAnswer;
      answer_started_ = true;
      break;
  }
}

std::vector<ReasoningSplitter::Event> ReasoningSplitter::Push(const std::string& piece) {
  std::vector<Event> events;
  if (piece.empty()) return events;

  if (mode_ == Mode::kAnswer) {
    ProcessAnswer(piece, events);
    return events;
  }

  std::string buf = tag_hold_ + piece;
  tag_hold_.clear();

  if (mode_ == Mode::kExpectOpener) {
    const std::string& open = syntax_.open;
    const size_t n = std::min(buf.size(), open.size());
    if (buf.compare(0, n, open, 0, n) != 0) {
      // Diverged: the held bytes were ordinary answer text and so is everything after them.
      mode_ = Mode::kAnswer;
      answer_started_ = true;
      events.push_back({false, std::move(buf)});
      return events;
    }
    if (buf.size() < open.size()) {
      tag_hold_ = std::move(buf);  // still a proper prefix of the opener
      return events;
    }
    mode_ = Mode::kReasoning;
    buf.erase(0, open.size());
    if (buf.empty()) return events;
  }

  const std::string& close = syntax_.close;
  const size_t p = buf.find(close);
  if (p == std::string::npos) {
    const size_t hold_len = LongestTagPrefixSuffix(buf, close);
    if (buf.size() > hold_len) events.push_back({true, buf.substr(0, buf.size() - hold_len)});
    tag_hold_ = buf.substr(buf.size() - hold_len);
    return events;
  }

  if (p > 0) events.push_back({true, buf.substr(0, p)});
  mode_ = Mode::kAnswer;
  skip_blank_ = syntax_.skip_blank_after_close;
  ProcessAnswer(buf.substr(p + close.size()), events);
  return events;
}

void ReasoningSplitter::ProcessAnswer(const std::string& buf, std::vector<Event>& events) {
  if (answer_started_ || !skip_blank_) {
    answer_started_ = true;
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
  } else if (mode_ == Mode::kExpectOpener) {
    // Generation ended inside a proper prefix of the opener: it was plain answer text.
    if (!tag_hold_.empty()) events.push_back({false, tag_hold_});
    tag_hold_.clear();
  } else {
    nl_hold_.clear();
  }
  return events;
}

ReasoningLocator::ReasoningLocator(ReasoningSyntax syntax, ReasoningSplitter::StartState start)
    : syntax_(std::move(syntax)), state_(State::kReasoning), end_(std::string::npos) {
  switch (start) {
    case ReasoningSplitter::StartState::kInReasoning:
      break;
    case ReasoningSplitter::StartState::kExpectOpener:
      if (!syntax_.open.empty()) state_ = State::kPending;
      break;
    case ReasoningSplitter::StartState::kAnswer:
      state_ = State::kAnswer;
      end_ = 0;
      break;
  }
}

void ReasoningLocator::Update(const std::string& accumulated) {
  if (state_ == State::kAnswer) return;
  size_t from = 0;
  if (state_ == State::kPending) {
    const std::string& open = syntax_.open;
    const size_t n = std::min(accumulated.size(), open.size());
    if (accumulated.compare(0, n, open, 0, n) != 0) {
      state_ = State::kAnswer;
      end_ = 0;
      return;
    }
    if (accumulated.size() < open.size()) return;
    state_ = State::kReasoning;
    from = open.size();
  }
  const size_t p = accumulated.find(syntax_.close, from);
  if (p == std::string::npos) return;
  state_ = State::kAnswer;
  closed_by_tag_ = true;
  end_ = p + syntax_.close.size();
}

size_t ReasoningLocator::AnswerBegin(const std::string& accumulated, size_t limit) const {
  if (state_ != State::kAnswer) return std::string::npos;
  limit = std::min(limit, accumulated.size());
  size_t i = end_;
  if (closed_by_tag_ && syntax_.skip_blank_after_close) {
    while (i < limit && (accumulated[i] == '\n' || accumulated[i] == '\r')) ++i;
  }
  return i < limit ? i : std::string::npos;
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
