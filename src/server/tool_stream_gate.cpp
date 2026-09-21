#include "tool_stream_gate.h"

#include <algorithm>
#include <cstdint>

namespace r4dx::server {

namespace {

// Byte-for-byte tool_call_parser.cpp's own `kOpen`/`kOpenLen` -- see this file's header for why the
// gate and the parser must agree on this literal exactly.
constexpr char kOpen[] = "<tool_call>";
constexpr size_t kOpenLen = 11;  // strlen(kOpen), fixed at compile time -- no runtime strlen call

// Longest k (0 <= k <= kOpenLen-1) such that the last k bytes of `buf` equal the first k bytes of
// kOpen -- i.e. the longest suffix of `buf` that could still grow into a full opener given more
// input. 0 means no suffix of `buf` could possibly be an opener prefix. Same helper, same shape, as
// reasoning_splitter.cpp's LongestTagPrefixSuffix (including the signed loop variable: `k` counts
// down to 0 and must not wrap the way an unsigned `k >= 1` loop would on its final iteration).
size_t LongestOpenPrefixSuffix(const std::string& buf) {
  const size_t max_k = std::min(kOpenLen - 1, buf.size());
  for (int64_t k = static_cast<int64_t>(max_k); k >= 1; --k) {
    const size_t uk = static_cast<size_t>(k);
    if (buf.compare(buf.size() - uk, uk, kOpen, 0, uk) == 0) return uk;
  }
  return 0;
}

}  // namespace

std::string ToolStreamGate::Push(const std::string& piece) {
  if (closed_ || piece.empty()) return std::string();

  std::string buf = hold_ + piece;
  hold_.clear();

  const size_t p = buf.find(kOpen);
  std::string out;
  if (p != std::string::npos) {
    // From the opener on, nothing else this generation is streamable -- the rest is buffered by the
    // caller and classified once, at the end, by ParseToolCalls.
    closed_ = true;
    out = buf.substr(0, p);
  } else {
    const size_t hold_len = LongestOpenPrefixSuffix(buf);
    out = buf.substr(0, buf.size() - hold_len);
    hold_ = buf.substr(buf.size() - hold_len);
  }
  streamed_ += out.size();
  return out;
}

std::string ToolStreamGate::Finish() {
  if (closed_ || hold_.empty()) return std::string();
  std::string out = std::move(hold_);
  hold_.clear();
  streamed_ += out.size();
  return out;
}

std::string ToolStreamRemainder(const std::string& streamed, const std::string& parsed_content) {
  if (parsed_content.size() >= streamed.size() &&
      parsed_content.compare(0, streamed.size(), streamed) == 0) {
    return parsed_content.substr(streamed.size());
  }
  return parsed_content;  // see this function's doc comment: duplicate rather than lose
}

}  // namespace r4dx::server
