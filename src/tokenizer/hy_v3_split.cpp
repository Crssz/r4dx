#include "hy_v3_split.h"

#include <cstdint>
#include <string>
#include <vector>

#include "vendor/llamacpp_unicode/unicode.h"

namespace r4dx {

const char* const kHyV3SplitRegex0 = "\\p{N}{1,3}";
const char* const kHyV3SplitRegex1 = "[\xE4\xB8\x80-\xE9\xBE\xA5\xE3\x81\x80-\xE3\x82\x9F\xE3\x82\xA0-\xE3\x83\xBF]+";
const char* const kHyV3SplitRegex2 =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
    "[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
    " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

namespace {

bool is_cjk_kana(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
}

bool is_ascii_punct(uint32_t cp) {
  if (cp >= 0x21 && cp <= 0x2F) return true;
  if (cp >= 0x3A && cp <= 0x40) return true;
  if (cp >= 0x5B && cp <= 0x60) return true;
  if (cp >= 0x7B && cp <= 0x7E) return true;
  return false;
}

bool is_ascii_alpha(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

struct Piece {
  size_t begin = 0;
  size_t end = 0;
  bool isolated = false;
};

// Applies Split(pattern, Isolated) over non-isolated spans.
template <typename Pred>
void split_isolated(const std::vector<uint32_t>& cpts, std::vector<Piece>& pieces, Pred&& match_fn) {
  std::vector<Piece> next;
  next.reserve(pieces.size() * 2);
  for (const Piece& p : pieces) {
    if (p.isolated) {
      next.push_back(p);
      continue;
    }
    size_t gap = p.begin;
    size_t i = p.begin;
    while (i < p.end) {
      const size_t m_end = match_fn(cpts, i, p.end);
      if (m_end > i) {
        if (i > gap) next.push_back({gap, i, false});
        next.push_back({i, m_end, true});
        i = m_end;
        gap = m_end;
      } else {
        ++i;
      }
    }
    if (gap < p.end) next.push_back({gap, p.end, false});
  }
  pieces.swap(next);
}

size_t match_regex2(const std::vector<uint32_t>& cpts,
                    const std::vector<unicode_cpt_flags>& fl,
                    size_t p, size_t end) {
  // 1. [ascii_punct][A-Za-z]+
  if (p + 1 < end && is_ascii_punct(cpts[p]) && is_ascii_alpha(cpts[p + 1])) {
    size_t e = p + 2;
    while (e < end && is_ascii_alpha(cpts[e])) ++e;
    return e;
  }
  // 2. [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
  {
    auto is_lm = [&](size_t idx) { return fl[idx].is_letter || fl[idx].is_accent_mark; };
    auto is_prefix = [&](size_t idx) {
      const uint32_t cp = cpts[idx];
      return cp != '\r' && cp != '\n' && !fl[idx].is_letter && !fl[idx].is_punctuation && !fl[idx].is_symbol;
    };
    if (is_prefix(p) && p + 1 < end && is_lm(p + 1)) {
      size_t e = p + 2;
      while (e < end && is_lm(e)) ++e;
      return e;
    }
    if (is_lm(p)) {
      size_t e = p + 1;
      while (e < end && is_lm(e)) ++e;
      return e;
    }
  }
  // 3. " ?[\p{P}\p{S}]+[\r\n]*"
  {
    auto is_ps = [&](size_t idx) { return fl[idx].is_punctuation || fl[idx].is_symbol; };
    size_t q = p;
    if (cpts[q] == ' ' && q + 1 < end && is_ps(q + 1)) ++q;
    if (is_ps(q)) {
      while (q < end && is_ps(q)) ++q;
      while (q < end && (cpts[q] == '\r' || cpts[q] == '\n')) ++q;
      return q;
    }
  }
  // 4. \s*[\r\n]+ | 5. \s+(?!\S) | 6. \s+
  if (fl[p].is_whitespace) {
    size_t r = p;
    size_t last_rn_end = 0;
    while (r < end && fl[r].is_whitespace) {
      if (cpts[r] == '\r' || cpts[r] == '\n') last_rn_end = r + 1;
      ++r;
    }
    if (last_rn_end > 0) return last_rn_end;
    if (r == end) return r;
    if (r - p >= 2) return r - 1;
    return r;
  }
  return p;
}

}  // namespace

std::vector<std::string> hy_v3_split_byte_encoded(const std::string& text) {
  if (text.empty()) return {};
  const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
  std::vector<unicode_cpt_flags> fl(cpts.size());
  for (size_t i = 0; i < cpts.size(); ++i) fl[i] = unicode_cpt_flags_from_cpt(cpts[i]);

  std::vector<Piece> pieces = {{0, cpts.size(), false}};
  // Pass 1: \p{N}{1,3}
  split_isolated(cpts, pieces, [&](const std::vector<uint32_t>&, size_t i, size_t end) {
    size_t e = i;
    while (e < end && e - i < 3 && fl[e].is_number) ++e;
    return e;
  });
  // Pass 2: [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+
  split_isolated(cpts, pieces, [&](const std::vector<uint32_t>& c, size_t i, size_t end) {
    size_t e = i;
    while (e < end && is_cjk_kana(c[e])) ++e;
    return e;
  });
  // Pass 3: main regex
  split_isolated(cpts, pieces, [&](const std::vector<uint32_t>& c, size_t i, size_t end) {
    return match_regex2(c, fl, i, end);
  });

  std::vector<std::string> out;
  out.reserve(pieces.size());
  for (const Piece& p : pieces) {
    std::string raw;
    for (size_t i = p.begin; i < p.end; ++i) raw += unicode_cpt_to_utf8(cpts[i]);
    std::string enc;
    enc.reserve(raw.size() * 2);
    for (unsigned char b : raw) enc += unicode_byte_to_utf8(b);
    out.push_back(std::move(enc));
  }
  return out;
}

}  // namespace r4dx
