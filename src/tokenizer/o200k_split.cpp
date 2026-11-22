#include "o200k_split.h"

#include <array>
#include <utility>

#include "vendor/llamacpp_unicode/unicode.h"

namespace r4dx {

const char* const kO200kInkling2Regex =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

namespace {

constexpr uint8_t kLu = 1, kLl = 2, kLt = 4, kLm = 8, kLo = 16, kM = 32, kN = 64, kWs = 128;
constexpr uint8_t kLetter = kLu | kLl | kLt | kLm | kLo;
constexpr uint8_t kUpperish = kLu | kLt | kLm | kLo | kM;  // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
constexpr uint8_t kLowerish = kLl | kLm | kLo | kM;        // [\p{Ll}\p{Lm}\p{Lo}\p{M}]

struct CatRun {
    uint32_t first;
    uint8_t bits;
};

constexpr CatRun kRuns[] = {
#include "o200k_cats.inc"
};

// One byte per code point (1.1 MB), built once.
const std::vector<uint8_t>& cat_table() {
    static const std::vector<uint8_t> table = [] {
        std::vector<uint8_t> t(0x110000, 0);
        const size_t n = sizeof(kRuns) / sizeof(kRuns[0]);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t end = (i + 1 < n) ? kRuns[i + 1].first : 0x110000;
            for (uint32_t cp = kRuns[i].first; cp < end; ++cp) t[cp] = kRuns[i].bits;
        }
        return t;
    }();
    return table;
}

class Matcher {
public:
    explicit Matcher(const std::vector<uint32_t>& c) : c_(c), n_(c.size()), cat_(cat_table()) {}

    uint8_t f(size_t i) const { return i < n_ ? (c_[i] < 0x110000 ? cat_[c_[i]] : 0) : 0; }
    bool is_upperish(size_t i) const { return (f(i) & kUpperish) != 0; }
    bool is_lowerish(size_t i) const { return (f(i) & kLowerish) != 0; }
    bool is_letter(size_t i) const { return (f(i) & kLetter) != 0; }
    bool is_number(size_t i) const { return (f(i) & kN) != 0; }
    bool is_ws(size_t i) const { return (f(i) & kWs) != 0; }
    bool is_rn(size_t i) const { return i < n_ && (c_[i] == '\r' || c_[i] == '\n'); }
    // [^\r\n\p{L}\p{N}] (a mark, a space, a symbol, ... -- but not a CR/LF, letter or digit)
    bool is_prefix_char(size_t i) const { return i < n_ && !is_rn(i) && !is_letter(i) && !is_number(i); }
    // [^\s\p{L}\p{N}]
    bool is_symbol(size_t i) const { return i < n_ && !is_ws(i) && !is_letter(i) && !is_number(i); }

    // End of the match starting at p, or p when no alternative matches (cannot happen for n_ > p).
    size_t match(size_t p) const {
        size_t e;
        // Alternation 1: P? U* L+ C?   (optional prefix tried first, then without)
        if (is_prefix_char(p) && (e = word_a(p + 1)) != kNone) return e;
        if ((e = word_a(p)) != kNone) return e;
        // Alternation 2: P? U+ L* C?
        if (is_prefix_char(p) && (e = word_b(p + 1)) != kNone) return e;
        if ((e = word_b(p)) != kNone) return e;
        // \p{N}{1,3}
        if (is_number(p)) {
            size_t k = p;
            while (k < n_ && k - p < 3 && is_number(k)) ++k;
            return k;
        }
        // " ?[^\s\p{L}\p{N}]+[\r\n/]*"
        {
            size_t q = p;
            if (q < n_ && c_[q] == ' ' && is_symbol(q + 1)) ++q;
            if (is_symbol(q)) {
                while (is_symbol(q)) ++q;
                while (q < n_ && (is_rn(q) || c_[q] == '/')) ++q;
                return q;
            }
        }
        if (is_ws(p)) {
            size_t r = p;
            size_t last_rn_end = 0;
            while (is_ws(r)) {
                if (is_rn(r)) last_rn_end = r + 1;
                ++r;
            }
            if (last_rn_end) return last_rn_end;       // \s*[\r\n]+
            if (r == n_) return r;                       // \s+(?!\S) at the end of the text
            if (r - p >= 2) return r - 1;                // \s+(?!\S): leave the last blank for the next token
            return r;                                    // \s+
        }
        return p + 1;
    }

private:
    static constexpr size_t kNone = static_cast<size_t>(-1);

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)? at i; returns the end (i when absent). Oniguruma's (?i) also folds U+017F
    // (LATIN SMALL LETTER LONG S) onto 's'.
    size_t contraction(size_t i) const {
        if (i + 1 < n_ && c_[i] == '\'') {
            const uint32_t a = lower(c_[i + 1]);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
            if (i + 2 < n_) {
                const uint32_t b = lower(c_[i + 2]);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
            }
        }
        return i;
    }
    static uint32_t lower(uint32_t cp) {
        if (cp >= 'A' && cp <= 'Z') return cp + 32;
        if (cp == 0x17F) return 's';
        return cp;
    }

    // U* L+ C? starting at q (U greedy with backtracking: the longest U* run after which a lowerish code
    // point follows). Returns kNone when no such split exists.
    size_t word_a(size_t q) const {
        size_t u = 0;
        while (is_upperish(q + u)) ++u;
        for (size_t j = u + 1; j-- > 0;) {
            if (is_lowerish(q + j)) {
                size_t e = q + j;
                while (is_lowerish(e)) ++e;
                return contraction(e);
            }
        }
        return kNone;
    }

    // U+ L* C? starting at q.
    size_t word_b(size_t q) const {
        size_t e = q;
        while (is_upperish(e)) ++e;
        if (e == q) return kNone;
        while (is_lowerish(e)) ++e;
        return contraction(e);
    }

    const std::vector<uint32_t>& c_;
    size_t n_;
    const std::vector<uint8_t>& cat_;
};

// GPT-2 byte -> printable UTF-8 string table, built once.
const std::array<std::string, 256>& byte_table() {
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> t;
        for (int b = 0; b < 256; ++b) t[b] = unicode_byte_to_utf8(static_cast<uint8_t>(b));
        return t;
    }();
    return table;
}

}  // namespace

std::vector<size_t> o200k_split_lengths(const std::vector<uint32_t>& cpts) {
    std::vector<size_t> out;
    const Matcher m(cpts);
    size_t p = 0;
    while (p < cpts.size()) {
        const size_t e = m.match(p);
        out.push_back(e - p);
        p = e;
    }
    return out;
}

std::vector<std::string> o200k_split_byte_encoded(const std::string& text) {
    const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
    const std::vector<size_t> lens = o200k_split_lengths(cpts);
    const auto& bt = byte_table();
    std::vector<std::string> words;
    words.reserve(lens.size());
    size_t pos = 0;
    for (const size_t len : lens) {
        std::string utf8;
        for (size_t i = pos; i < pos + len; ++i) utf8 += unicode_cpt_to_utf8(cpts[i]);
        std::string enc;
        enc.reserve(utf8.size() * 2);
        for (const unsigned char b : utf8) enc += bt[b];
        words.push_back(std::move(enc));
        pos += len;
    }
    return words;
}

}  // namespace r4dx
