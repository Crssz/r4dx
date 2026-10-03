// Tokenizer implementation.
//
// Pre-tokenizer regex splitting delegates to `unicode_regex_split()` (vendored verbatim from
// llama.cpp, vendor/llamacpp_unicode/unicode.cpp -- see NOTICE in that directory), specifically
// its hand-written `unicode_regex_split_custom_qwen2()` / `..._qwen35()` fast paths, which
// implement (respectively) the older Qwen2 pattern and the newer Qwen3.5 pattern:
//
//   Qwen2:   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   Qwen3.5: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Qwen3.8-27B declares the Qwen3.5 pattern in TWO independent on-disk places --
// tokenizer.json's `pre_tokenizer.pretokenizers[0].pattern.Regex` AND tokenizer_config.json's own
// `pretokenize_regex` field. transformers 5.17.0 has no dedicated Qwen3.5 tokenizer class and
// loads this checkpoint as `Qwen2Tokenizer` (tokenizer_config.json's `tokenizer_class`), whose
// `__init__` reconstructs the pre_tokenizer from its own hardcoded (older Qwen2) pattern,
// discarding BOTH on-disk fields -- confirmed empirically against the live reference tokenizer,
// see the comment above kQwen2SplitTrigger below. This is a transformers bug for this checkpoint,
// not the intended tokenization: the merge vocab contains tokens the Qwen2 pattern can never
// produce (e.g. the Thai word "กรุงเทพมหานคร" is the single token 220258 under the on-disk
// declaration but 3 tokens via transformers, because the Qwen2 pattern never absorbs combining
// marks into a letter run). r4dx therefore trusts the on-disk declaration by default (preferring
// tokenizer_config.json's `pretokenize_regex` when present, else tokenizer.json's own
// `pre_tokenizer` field -- both agree for this checkpoint) and only reproduces transformers
// 5.17.0's override when the caller explicitly opts in via
// `Tokenizer::Options{.hf_transformers_compat = true}`, for A/B testing against that (buggy)
// reference only. See load_impl() below for the selection logic.
//
// std::regex (ECMAScript grammar) cannot express the `(?i:...)` inline-flag group these patterns
// use, which is why llama.cpp -- and this port -- hand-code the split instead of using std::regex
// for it.
//
// The BPE merge loop (add_new_bigram / the symbol-chain priority-queue merge) and the
// added/special-token fragment splitter are r4dx's own port of the equivalent functions in
// llama.cpp's src/llama-vocab.cpp (`llm_tokenizer_bpe_session::tokenize()`,
// `tokenizer_st_partition()`), rewritten against this file's Impl data structures rather than
// llama_vocab's.
#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "nlohmann/json.hpp"
#include "vendor/llamacpp_unicode/unicode.h"

namespace r4dx {
namespace {

using nlohmann::json;

// Trigger strings `unicode_regex_split()`'s internal dispatcher (`unicode_regex_split_custom()`
// in unicode.cpp) matches against to select one of its hand-written splitters. These are NOT the
// same strings as tokenizer.json's own `(?i:...)`-spelled regex (std::regex/ECMAScript cannot
// express inline flag groups); they are llama.cpp's manually case-expanded, dispatcher-recognized
// form of the identical patterns -- see the block comment above.
//
// tokenizer.json's pre_tokenizer.pretokenizers[0].pattern.Regex for Qwen3.8-27B, AND
// tokenizer_config.json's own `pretokenize_regex` field, both independently declare the
// *Qwen3.5* pattern (letter runs include combining marks: `[\p{L}\p{M}]+`). But
// tokenizer_config.json's `tokenizer_class` is `"Qwen2Tokenizer"` -- transformers 5.17.0 has no
// dedicated Qwen3.5 tokenizer class yet, so `AutoTokenizer.from_pretrained` instantiates
// `Qwen2Tokenizer`, whose `__init__` (transformers/models/qwen2/tokenization_qwen2.py)
// *reconstructs* the backend Rust tokenizer's pre_tokenizer from its own hardcoded
// `PRETOKENIZE_REGEX` constant -- the *older* Qwen2 pattern (`\p{L}+`, no `\p{M}`) -- discarding
// BOTH on-disk fields entirely. Confirmed empirically: for this checkpoint,
// `tok.backend_tokenizer.to_str()`'s actual runtime `pre_tokenizer.pretokenizers[0]` differs from
// tokenizer.json's on-disk field and matches the Qwen2 pattern -- e.g. it splits the
// mark-containing Thai word "กรุงเทพมหานคร" as ["กร", "ุงเทพมหานคร"] (a letter run ending
// where marks are NOT absorbed, then a second run whose optional leading char is the mark itself,
// per `[^\r\n\p{L}\p{N}]?\p{L}+`), not as one single `[\p{L}\p{M}]+` run -- and this is a
// transformers bug for this checkpoint, not the intended tokenization (see the top-of-file
// comment). r4dx therefore trusts the on-disk declaration by default and only reproduces the
// runtime (tokenizer_class-selected) Qwen2 pattern when `Tokenizer::Options.hf_transformers_compat`
// is explicitly set (see load_impl() below).
const std::string kQwen2SplitTrigger =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?"
    "[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const std::string kQwen35SplitTrigger =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"
    "\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// tokenizer.json's actual pre_tokenizer.pretokenizers[0].pattern.Regex, one of these two known
// forms. Checked against the loaded file so an unexpected tokenizer.json shape fails loudly at
// load time instead of silently mis-tokenizing.
const std::string kQwen2SplitExpected =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*"
    "[\\r\\n]+|\\s+(?!\\S)|\\s+";
const std::string kQwen35SplitExpected =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+"
    "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::string read_file_or_throw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Tokenizer: failed to open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string read_file_optional(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string join_dir(const std::string& dir, const std::string& name) {
    const std::string sep = (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) ? "" : "/";
    return dir + sep + name;
}

// Longest-token-first UTF-8 byte length lookup table, ported alongside unicode_len_utf8; used
// only by the streaming decoder to find the boundary of a trailing incomplete codepoint.
size_t valid_utf8_prefix_length(const std::string& s) {
    const size_t n = s.size();
    const size_t max_back = std::min<size_t>(4, n);
    for (size_t back = 1; back <= max_back; ++back) {
        const unsigned char b = static_cast<unsigned char>(s[n - back]);
        if ((b & 0xC0) != 0x80) {
            // Lead byte (or single-byte ASCII) found `back` bytes from the end.
            const size_t expect = unicode_len_utf8(static_cast<char>(b));
            if (back < expect) {
                // Incomplete trailing sequence: keep it buffered.
                return n - back;
            }
            return n;  // complete (back == expect), or an unexpected/over-long tail we just flush
        }
    }
    // Four-plus trailing continuation bytes with no lead byte in the window: not valid UTF-8
    // (should not happen for well-formed model output); flush everything rather than buffer
    // forever.
    return n;
}

}  // namespace

struct AddedTokenInfo {
    TokenId id;
    std::string text;
    bool special;
};

struct Tokenizer::Impl {
    // id -> stored surface text: byte-encoded (GPT-2 style) for merge-vocab ids, literal UTF-8
    // text for added-token ids.
    std::vector<std::string> id_to_token;
    std::vector<bool> id_is_added;
    std::vector<bool> id_is_special;

    std::unordered_map<std::string, TokenId> token_to_id;  // covers both merge-vocab and added
    std::unordered_map<std::string, int32_t> merge_rank;   // key: "<left> <right>" (byte-encoded)
    bool ignore_merges = false;

    // Dispatcher trigger string passed to unicode_regex_split() -- kQwen2SplitTrigger or
    // kQwen35SplitTrigger, chosen in load_impl() by tokenizer_config.json's tokenizer_class (see
    // the block comment above those constants).
    std::string split_trigger;

    // Added tokens for fragment splitting, sorted longest-surface-text-first (ties broken by
    // ascending id) so a longer added token is never shadowed by a shorter one that happens to be
    // one of its prefixes.
    std::vector<AddedTokenInfo> added_tokens_by_len;

    TokenId bos = kNoToken;
    TokenId eos = kNoToken;
    TokenId pad = kNoToken;
    std::vector<TokenId> eos_list;

    TokenizerKind kind = TokenizerKind::kByteLevelRegex;

    // id -> true for special added tokens decode()/StreamDecoder keep despite skip_special_tokens
    // (Options::keep_special_on_decode); empty when none were requested.
    std::vector<bool> keep_special;

    // --- kSpmByteFallback only -------------------------------------------------------------
    // id -> byte value for a `<0xNN>` byte-fallback token, else -1.
    std::vector<int16_t> byte_of;
    // byte value -> id of its `<0xNN>` token.
    std::array<TokenId, 256> byte_tok{};
    // id -> bytes the token decodes to (U+2581 replaced by ' '; the single raw byte for `<0xNN>`).
    std::vector<std::string> decoded_piece;
    // id of a one-codepoint ("single character") vocab token for ASCII bytes, kNoToken if absent.
    std::array<TokenId, 128> ascii_tok{};
    // id of the U+2581 token (what a space normalizes to), kNoToken if absent.
    TokenId space_tok = kNoToken;
    // k newlines (k = 1..31) -> id of that one vocab token; empty unless all 31 exist.
    std::vector<TokenId> newline_run_tok;
    // (left_id, right_id) -> (rank, merged_id), open addressing, power-of-two capacity.
    struct PairEntry {
        uint64_t key;
        int32_t rank;
        TokenId merged;
    };
    static constexpr uint64_t kEmptyPair = ~0ull;
    std::vector<PairEntry> pair_table;
    uint64_t pair_mask = 0;

    const PairEntry* find_pair(TokenId left, TokenId right) const {
        const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
        uint64_t h = (key * 0x9E3779B97F4A7C15ull) >> 20;
        for (;; ++h) {
            const PairEntry& e = pair_table[h & pair_mask];
            if (e.key == key) return &e;
            if (e.key == kEmptyPair) return nullptr;
        }
    }

    void insert_pair(TokenId left, TokenId right, int32_t rank, TokenId merged) {
        const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
        uint64_t h = (key * 0x9E3779B97F4A7C15ull) >> 20;
        for (;; ++h) {
            PairEntry& e = pair_table[h & pair_mask];
            if (e.key == key || e.key == kEmptyPair) {
                e = PairEntry{key, rank, merged};  // a repeated pair overwrites: last wins, as in HF
                return;
            }
        }
    }

    int find_bpe_rank(const std::string& left, const std::string& right) const {
        auto it = merge_rank.find(left + " " + right);
        return it == merge_rank.end() ? -1 : it->second;
    }

    TokenId lookup(const std::string& s) const {
        auto it = token_to_id.find(s);
        return it == token_to_id.end() ? kNoToken : it->second;
    }
};

namespace {

// --- BPE merge loop, ported from llama.cpp's llm_tokenizer_bpe_session::tokenize() -----------

struct Symbol {
    int prev = -1;
    int next = -1;
    size_t offset = 0;
    size_t n = 0;
};

struct Bigram {
    int left;
    int right;
    std::string text;
    int rank;
};

struct BigramCmp {
    // std::priority_queue is a max-heap w.r.t. this comparator; we want the lowest rank (and,
    // among equal ranks, the leftmost pair) to pop first, so this returns true when `l` has LOWER
    // priority than `r` (mirrors llm_bigram_bpe::comparator).
    bool operator()(const Bigram& l, const Bigram& r) const {
        return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
};

// Runs byte-level BPE merges over one pre-tokenized word (already GPT-2 byte-encoded) and appends
// the resulting token ids to `out`.
void bpe_merge_word(const Tokenizer::Impl& impl, const std::string& word, std::vector<TokenId>& out) {
    if (word.empty()) return;

    if (impl.ignore_merges) {
        TokenId whole = impl.lookup(word);
        if (whole != kNoToken) {
            out.push_back(whole);
            return;
        }
    }

    std::vector<Symbol> symbols;
    symbols.reserve(word.size());
    {
        size_t offset = 0;
        int index = 0;
        while (offset < word.size()) {
            const size_t char_len = std::min(word.size() - offset, unicode_len_utf8(word[offset]));
            Symbol sym;
            sym.offset = offset;
            sym.n = char_len;
            sym.prev = index - 1;
            offset += char_len;
            sym.next = (offset == word.size()) ? -1 : index + 1;
            symbols.push_back(sym);
            index++;
        }
    }

    std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp> work_queue;
    auto add_new_bigram = [&](int left, int right) {
        if (left == -1 || right == -1) return;
        const std::string left_token = word.substr(symbols[left].offset, symbols[left].n);
        const std::string right_token = word.substr(symbols[right].offset, symbols[right].n);
        const int rank = impl.find_bpe_rank(left_token, right_token);
        if (rank < 0) return;
        work_queue.push(Bigram{left, right, left_token + right_token, rank});
    };

    for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
        add_new_bigram(i - 1, i);
    }

    while (!work_queue.empty()) {
        Bigram bigram = work_queue.top();
        work_queue.pop();

        Symbol& left_symbol = symbols[bigram.left];
        Symbol& right_symbol = symbols[bigram.right];
        if (left_symbol.n == 0 || right_symbol.n == 0) continue;

        const std::string left_token = word.substr(left_symbol.offset, left_symbol.n);
        const std::string right_token = word.substr(right_symbol.offset, right_symbol.n);
        if (left_token + right_token != bigram.text) continue;  // stale entry

        left_symbol.n += right_symbol.n;
        right_symbol.n = 0;
        left_symbol.next = right_symbol.next;
        if (right_symbol.next >= 0) {
            symbols[right_symbol.next].prev = bigram.left;
        }

        add_new_bigram(left_symbol.prev, bigram.left);
        add_new_bigram(bigram.left, left_symbol.next);
    }

    for (const Symbol& sym : symbols) {
        if (sym.n == 0) continue;
        const std::string piece = word.substr(sym.offset, sym.n);
        const TokenId tok = impl.lookup(piece);
        if (tok != kNoToken) {
            out.push_back(tok);
            continue;
        }
        // Fallback: the GPT-2 byte vocab guarantees every single encoded byte-character has its
        // own base-vocab token, so this only fires if `piece` failed to fully reduce -- split it
        // one encoded "byte" (UTF-8 codepoint of the byte-encoding) at a time.
        size_t off = 0;
        while (off < piece.size()) {
            const size_t clen = std::min(piece.size() - off, unicode_len_utf8(piece[off]));
            const TokenId b = impl.lookup(piece.substr(off, clen));
            if (b != kNoToken) out.push_back(b);
            off += clen;
        }
    }
}

// --- added/special-token fragment splitting, ported from tokenizer_st_partition() -------------

struct Fragment {
    bool is_token;
    TokenId token;
    std::string_view text;
};

std::vector<Fragment> split_added_tokens(const Tokenizer::Impl& impl, std::string_view text,
                                          bool parse_special) {
    std::vector<Fragment> fragments{Fragment{false, kNoToken, text}};
    for (const AddedTokenInfo& tok : impl.added_tokens_by_len) {
        if (tok.special && !parse_special) continue;  // gated: see Tokenizer::encode doc comment
        std::vector<Fragment> next;
        next.reserve(fragments.size());
        for (const Fragment& frag : fragments) {
            if (frag.is_token) {
                next.push_back(frag);
                continue;
            }
            std::string_view s = frag.text;
            size_t pos = 0;
            while (true) {
                const size_t match = s.find(tok.text, pos);
                if (match == std::string_view::npos) {
                    if (pos < s.size()) next.push_back(Fragment{false, kNoToken, s.substr(pos)});
                    break;
                }
                if (match > pos) next.push_back(Fragment{false, kNoToken, s.substr(pos, match - pos)});
                next.push_back(Fragment{true, tok.id, {}});
                pos = match + tok.text.size();
            }
        }
        fragments = std::move(next);
    }
    return fragments;
}

// --- decode helpers -----------------------------------------------------------------------

// Raw bytes a single merge-vocab token decodes to: each codepoint of the stored (byte-encoded)
// surface form maps back to exactly one original byte via the GPT-2 byte<->unicode table.
std::string decode_merge_vocab_piece(const std::string& stored) {
    std::string out;
    out.reserve(stored.size());
    for (const uint32_t cpt : unicode_cpts_from_utf8(stored)) {
        out.push_back(static_cast<char>(unicode_utf8_to_byte(unicode_cpt_to_utf8(cpt))));
    }
    return out;
}

// --- kSpmByteFallback (Gemma) backend -----------------------------------------------------------
//
// tokenizer.json shape (checked by validate_spm below): normalizer Replace(" " -> U+2581); no regex
// pre-tokenizer (its Split on " " is a no-op once spaces are gone); BPE over the whole text with
// byte_fallback; decoder Replace(U+2581 -> " ") + ByteFallback + Fuse. Encoding is therefore
//   split added tokens (not normalized) -> per fragment: chunk on ([^\n]+|\n+), BPE each chunk.
// Splitting on newlines is exact for this vocab: no vocab entry or merge mixes '\n' with another
// character, so a merge can never cross a newline. A '\n' run is emitted as greedy tokens of 31
// ('\n' x 31 is the longest run token), which is what BPE produces for it (checked against HF for
// runs of 1..300 by tools/tok_ref/check_hf_equiv.py).

constexpr char kSpaceMark[] = "\xE2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK
constexpr size_t kMaxNewlineRun = 31;

enum class Utf8Status { kOk, kIncomplete, kInvalid };

// Classifies the UTF-8 sequence starting at p[0] with n >= 1 bytes available (strict: no overlongs,
// surrogates or > U+10FFFF, same as Rust's from_utf8). *len is the sequence length when kOk.
Utf8Status utf8_status(const unsigned char* p, size_t n, size_t* len) {
    const unsigned char b0 = p[0];
    size_t need = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (b0 < 0x80) {
        *len = 1;
        return Utf8Status::kOk;
    } else if (b0 >= 0xC2 && b0 <= 0xDF) {
        need = 2;
    } else if (b0 == 0xE0) {
        need = 3;
        lo = 0xA0;
    } else if (b0 == 0xED) {
        need = 3;
        hi = 0x9F;
    } else if (b0 >= 0xE1 && b0 <= 0xEF) {
        need = 3;
    } else if (b0 == 0xF0) {
        need = 4;
        lo = 0x90;
    } else if (b0 >= 0xF1 && b0 <= 0xF3) {
        need = 4;
    } else if (b0 == 0xF4) {
        need = 4;
        hi = 0x8F;
    } else {
        return Utf8Status::kInvalid;
    }
    for (size_t i = 1; i < need; ++i) {
        if (i >= n) return Utf8Status::kIncomplete;
        if (p[i] < lo || p[i] > hi) return Utf8Status::kInvalid;
        lo = 0x80;
        hi = 0xBF;
    }
    *len = need;
    return Utf8Status::kOk;
}

bool is_valid_utf8(const std::string& s) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
    size_t i = 0;
    while (i < s.size()) {
        size_t len = 0;
        if (utf8_status(p + i, s.size() - i, &len) != Utf8Status::kOk) return false;
        i += len;
    }
    return true;
}

constexpr char kReplacementChar[] = "\xEF\xBF\xBD";

struct SpmSym {
    TokenId id;
    int prev;
    int next;
};

struct SpmBigram {
    int32_t rank;
    int left;
    TokenId left_id;
    TokenId right_id;
};

struct SpmBigramCmp {
    // Max-heap comparator: lowest rank first, then leftmost pair (HF tokenizers' merge order).
    bool operator()(const SpmBigram& l, const SpmBigram& r) const {
        return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
};

// BPE over one chunk of raw (un-normalized) text that contains no '\n'. A ' ' counts as U+2581.
void spm_encode_chunk(const Tokenizer::Impl& impl, std::string_view chunk, std::vector<TokenId>& out) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(chunk.data());
    const size_t n = chunk.size();

    std::vector<SpmSym> syms;
    syms.reserve(n);
    auto push_sym = [&](TokenId id) {
        const int idx = static_cast<int>(syms.size());
        syms.push_back(SpmSym{id, idx - 1, -1});
        if (idx > 0) syms[idx - 1].next = idx;
    };
    auto push_bytes = [&](const unsigned char* s, size_t len) {
        for (size_t k = 0; k < len; ++k) push_sym(impl.byte_tok[s[k]]);
    };

    for (size_t i = 0; i < n;) {
        const unsigned char b = p[i];
        if (b == 0x20) {
            if (impl.space_tok != kNoToken) {
                push_sym(impl.space_tok);
            } else {
                push_bytes(reinterpret_cast<const unsigned char*>(kSpaceMark), 3);
            }
            ++i;
        } else if (b < 0x80) {
            const TokenId id = impl.ascii_tok[b];
            if (id != kNoToken) push_sym(id); else push_sym(impl.byte_tok[b]);
            ++i;
        } else {
            size_t len = 0;
            if (utf8_status(p + i, n - i, &len) == Utf8Status::kOk) {
                const TokenId id = impl.lookup(std::string(chunk.substr(i, len)));
                if (id != kNoToken) push_sym(id); else push_bytes(p + i, len);
                i += len;
            } else {
                push_sym(impl.byte_tok[b]);  // invalid byte: keep it, as its own <0xNN>
                ++i;
            }
        }
    }
    if (syms.empty()) return;

    std::priority_queue<SpmBigram, std::vector<SpmBigram>, SpmBigramCmp> queue;
    auto add_bigram = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        const Tokenizer::Impl::PairEntry* e = impl.find_pair(syms[left].id, syms[right].id);
        if (e) queue.push(SpmBigram{e->rank, left, syms[left].id, syms[right].id});
    };
    for (int i = 1; i < static_cast<int>(syms.size()); ++i) add_bigram(i - 1, i);

    while (!queue.empty()) {
        const SpmBigram bg = queue.top();
        queue.pop();
        SpmSym& left = syms[bg.left];
        if (left.id != bg.left_id || left.next < 0) continue;  // stale: left changed or dead
        const int right_idx = left.next;
        SpmSym& right = syms[right_idx];
        if (right.id != bg.right_id) continue;                  // stale: right changed
        const Tokenizer::Impl::PairEntry* e = impl.find_pair(left.id, right.id);
        if (!e) continue;

        left.id = e->merged;
        left.next = right.next;
        if (right.next >= 0) syms[right.next].prev = bg.left;
        right.id = -2;  // dead; never equals a queued id
        right.prev = right.next = -1;

        add_bigram(left.prev, bg.left);
        add_bigram(bg.left, left.next);
    }

    for (int i = 0; i >= 0; i = syms[i].next) out.push_back(syms[i].id);
}

void spm_encode_text(const Tokenizer::Impl& impl, std::string_view text, std::vector<TokenId>& out) {
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\n') {
            size_t j = i;
            while (j < text.size() && text[j] == '\n') ++j;
            size_t run = j - i;
            if (!impl.newline_run_tok.empty()) {
                while (run > 0) {
                    const size_t k = std::min(run, kMaxNewlineRun);
                    out.push_back(impl.newline_run_tok[k - 1]);
                    run -= k;
                }
            } else {
                // Vocab without the 1..31 run tokens: fall back to plain BPE on the run (can only
                // produce byte tokens / whatever merges exist).
                const std::string nl(run, '\n');
                spm_encode_chunk(impl, nl, out);
            }
            i = j;
        } else {
            size_t j = text.find('\n', i);
            if (j == std::string_view::npos) j = text.size();
            spm_encode_chunk(impl, text.substr(i, j - i), out);
            i = j;
        }
    }
}

inline bool decode_skips(const Tokenizer::Impl& impl, TokenId id, bool skip_special_tokens) {
    return skip_special_tokens && impl.id_is_special[id] && !(id < static_cast<TokenId>(impl.keep_special.size()) && impl.keep_special[id]);
}

// One-shot decode, matching HF's Sequence[Replace, ByteFallback, Fuse]: a run of consecutive byte
// tokens is validated as a whole (strict UTF-8); an invalid run becomes one U+FFFD per byte token.
std::string spm_decode(const Tokenizer::Impl& impl, const std::vector<TokenId>& ids, bool skip_special_tokens) {
    std::string out;
    std::string run;
    auto flush_run = [&]() {
        if (run.empty()) return;
        if (is_valid_utf8(run)) {
            out += run;
        } else {
            for (size_t k = 0; k < run.size(); ++k) out += kReplacementChar;
        }
        run.clear();
    };
    for (const TokenId id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= impl.id_to_token.size()) continue;  // ignore invalid ids
        if (decode_skips(impl, id, skip_special_tokens)) continue;
        if (impl.byte_of[id] >= 0) {
            run.push_back(static_cast<char>(impl.byte_of[id]));
            continue;
        }
        flush_run();
        out += impl.decoded_piece[id];
    }
    flush_run();
    return out;
}

// Incremental form of the byte-run handling: emits each complete valid codepoint as soon as it is
// available, replaces an invalid byte by U+FFFD, and holds only a still-incomplete trailing
// sequence in `run`. `final` (token boundary, flush) resolves a held incomplete tail to U+FFFD.
// Differs from spm_decode only for malformed runs (HF replaces a whole invalid run byte by byte).
std::string spm_drain_run(std::string& run, bool final) {
    std::string out;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(run.data());
    size_t i = 0;
    while (i < run.size()) {
        size_t len = 0;
        const Utf8Status st = utf8_status(p + i, run.size() - i, &len);
        if (st == Utf8Status::kOk) {
            out.append(run, i, len);
            i += len;
        } else if (st == Utf8Status::kIncomplete && !final) {
            break;
        } else {
            out += kReplacementChar;
            ++i;
        }
    }
    run.erase(0, i);
    return out;
}

// --- tokenizer.json loading -----------------------------------------------------------------

void validate_pretokenizer(const json& root, const Tokenizer::Options& options) {
    if (!root.contains("model") || root["model"].value("type", "") != "BPE") {
        throw std::runtime_error("Tokenizer: model.type is not \"BPE\"");
    }
    const json& model = root["model"];
    if (model.value("byte_fallback", false)) {
        throw std::runtime_error("Tokenizer: model.byte_fallback=true is not supported");
    }
    if (!model.value("continuing_subword_prefix", std::string()).empty() ||
        !model.value("end_of_word_suffix", std::string()).empty()) {
        throw std::runtime_error("Tokenizer: non-empty continuing_subword_prefix/end_of_word_suffix "
                                  "is not supported");
    }

    // tokenizer.json's `normalizer` stage (e.g. NFC canonical composition) is NOT applied by this
    // implementation (see the KNOWN GAP comment in tokenizer.h) -- it is the one pre-tokenization
    // stage that can silently mis-tokenize already-decomposed Unicode text. Refuse to load unless
    // the caller explicitly acknowledges the gap, so a future checkpoint with a normalizer this
    // implementation truly cannot ignore (e.g. NFKC, lowercasing) fails loudly instead of silently
    // mis-tokenizing.
    if (root.contains("normalizer") && !root["normalizer"].is_null()) {
        const std::string norm_type = root["normalizer"].value("type", "");
        if (!options.allow_unimplemented_normalizer) {
            throw std::runtime_error(
                "Tokenizer: tokenizer.json declares normalizer.type=\"" + norm_type +
                "\", which this implementation does not apply (see tokenizer.h's NFC KNOWN GAP "
                "comment); pass Tokenizer::Options{.allow_unimplemented_normalizer=true} to load "
                "anyway and accept the resulting divergence on non-NFC-normalized input");
        }
    }

    if (!root.contains("pre_tokenizer") || root["pre_tokenizer"].value("type", "") != "Sequence") {
        throw std::runtime_error("Tokenizer: expected pre_tokenizer.type == \"Sequence\"");
    }
    const json& pretoks = root["pre_tokenizer"]["pretokenizers"];
    if (!pretoks.is_array() || pretoks.size() < 2) {
        throw std::runtime_error("Tokenizer: expected >=2 entries in pre_tokenizer.pretokenizers");
    }
    const json& split = pretoks[0];
    if (split.value("type", "") != "Split" || !split.contains("pattern") ||
        !split["pattern"].contains("Regex")) {
        throw std::runtime_error("Tokenizer: pre_tokenizer.pretokenizers[0] is not a Regex Split");
    }
    const std::string regex = split["pattern"]["Regex"].get<std::string>();
    if (regex != kQwen2SplitExpected && regex != kQwen35SplitExpected) {
        throw std::runtime_error(
            "Tokenizer: pre_tokenizer regex does not match either pattern this tokenizer "
            "implements (Qwen2 or Qwen3.5; got: " +
            regex + ")");
    }
    const json& byte_level = pretoks[1];
    if (byte_level.value("type", "") != "ByteLevel") {
        throw std::runtime_error("Tokenizer: pre_tokenizer.pretokenizers[1] is not ByteLevel");
    }
    if (byte_level.value("add_prefix_space", false)) {
        throw std::runtime_error("Tokenizer: pre_tokenizer.pretokenizers[1].add_prefix_space=true is "
                                  "not supported (would silently shift every first token)");
    }
    if (byte_level.value("use_regex", true)) {
        throw std::runtime_error("Tokenizer: pre_tokenizer.pretokenizers[1].use_regex=true is not "
                                  "supported (the ByteLevel stage is expected to run over "
                                  "already-Split pieces only)");
    }

    if (!root.contains("decoder") || root["decoder"].value("type", "") != "ByteLevel") {
        throw std::runtime_error("Tokenizer: expected decoder.type == \"ByteLevel\"");
    }
    const json& decoder = root["decoder"];
    if (decoder.value("add_prefix_space", false)) {
        throw std::runtime_error("Tokenizer: decoder.add_prefix_space=true is not supported");
    }
}

// Throws unless tokenizer.json is exactly the Gemma-style shape documented above spm_encode_chunk.
void validate_spm(const json& root) {
    auto fail = [](const std::string& what) {
        throw std::runtime_error("Tokenizer: byte_fallback tokenizer.json is not the supported SPM-style shape: " + what);
    };
    const json& model = root["model"];
    if (model.value("type", "") != "BPE") fail("model.type is not BPE");
    if (model.value("ignore_merges", false)) fail("model.ignore_merges=true");
    // (these keys are present but null in Gemma's tokenizer.json; json::value() throws on null)
    for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        if (model.contains(key) && !model[key].is_null() && !model[key].get<std::string>().empty()) {
            fail(std::string("non-empty ") + key);
        }
    }
    if (model.contains("dropout") && !model["dropout"].is_null()) fail("model.dropout is set");

    if (!root.contains("normalizer") || root["normalizer"].is_null() ||
        root["normalizer"].value("type", "") != "Replace" ||
        root["normalizer"].value("content", "") != kSpaceMark ||
        !root["normalizer"].contains("pattern") || root["normalizer"]["pattern"].value("String", "") != " ") {
        fail("normalizer is not Replace(\" \" -> U+2581)");
    }
    if (root.contains("pre_tokenizer") && !root["pre_tokenizer"].is_null()) {
        const json& pt = root["pre_tokenizer"];
        if (pt.value("type", "") != "Split" || !pt.contains("pattern") || pt["pattern"].value("String", "x") != " " ||
            pt.value("behavior", "") != "MergedWithPrevious" || pt.value("invert", false)) {
            fail("pre_tokenizer is not Split(\" \", MergedWithPrevious)");
        }
    }
    if (!root.contains("decoder") || root["decoder"].value("type", "") != "Sequence" ||
        !root["decoder"].contains("decoders") || root["decoder"]["decoders"].size() != 3) {
        fail("decoder is not Sequence[Replace, ByteFallback, Fuse]");
    }
    {
        const json& d = root["decoder"]["decoders"];
        if (d[0].value("type", "") != "Replace" || d[0].value("content", "") != " " ||
            !d[0].contains("pattern") || d[0]["pattern"].value("String", "") != kSpaceMark ||
            d[1].value("type", "") != "ByteFallback" || d[2].value("type", "") != "Fuse") {
            fail("decoder is not Sequence[Replace(U+2581 -> \" \"), ByteFallback, Fuse]");
        }
    }
    // A post-processor that inserts special tokens (BOS/EOS) would change every encode; this
    // backend adds none.
    if (root.contains("post_processor") && !root["post_processor"].is_null()) {
        const json& pp = root["post_processor"];
        if (pp.value("type", "") != "TemplateProcessing") fail("post_processor is not TemplateProcessing");
        for (const char* key : {"single", "pair"}) {
            if (!pp.contains(key)) continue;
            for (const auto& piece : pp[key]) {
                if (piece.contains("SpecialToken")) fail("post_processor inserts special tokens");
            }
        }
    }
    for (const auto& entry : root.value("added_tokens", json::array())) {
        if (entry.value("normalized", false) || entry.value("lstrip", false) || entry.value("rstrip", false) ||
            entry.value("single_word", false)) {
            fail("added token " + entry.value("content", std::string()) + " has normalized/lstrip/rstrip/single_word set");
        }
    }
}

// Builds the kSpmByteFallback lookup tables from the already-populated vocab / merge JSON.
void build_spm_tables(Tokenizer::Impl& impl, const json& root) {
    const size_t vocab_n = impl.id_to_token.size();

    impl.byte_of.assign(vocab_n, -1);
    impl.byte_tok.fill(kNoToken);
    static const char* kHex = "0123456789ABCDEF";
    for (int b = 0; b < 256; ++b) {
        std::string t = "<0x";
        t += kHex[b >> 4];
        t += kHex[b & 15];
        t += ">";
        const TokenId id = impl.lookup(t);
        if (id == kNoToken || impl.id_is_added[id]) {
            throw std::runtime_error("Tokenizer: byte_fallback vocab is missing token " + t);
        }
        impl.byte_tok[b] = id;
        impl.byte_of[id] = static_cast<int16_t>(b);
    }

    impl.decoded_piece.assign(vocab_n, std::string());
    for (size_t id = 0; id < vocab_n; ++id) {
        if (impl.byte_of[id] >= 0) {
            impl.decoded_piece[id] = std::string(1, static_cast<char>(impl.byte_of[id]));
            continue;
        }
        const std::string& s = impl.id_to_token[id];
        std::string& d = impl.decoded_piece[id];
        d.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            if (s.compare(i, 3, kSpaceMark) == 0) {
                d.push_back(' ');
                i += 3;
            } else {
                d.push_back(s[i++]);
            }
        }
    }

    impl.ascii_tok.fill(kNoToken);
    for (int c = 0; c < 128; ++c) {
        if (c == 0x20) continue;
        const TokenId id = impl.lookup(std::string(1, static_cast<char>(c)));
        if (id != kNoToken && !impl.id_is_added[id]) impl.ascii_tok[c] = id;
    }
    {
        const TokenId id = impl.lookup(kSpaceMark);
        if (id != kNoToken && !impl.id_is_added[id]) impl.space_tok = id;
    }
    {
        std::vector<TokenId> runs;
        for (size_t k = 1; k <= kMaxNewlineRun; ++k) {
            const TokenId id = impl.lookup(std::string(k, '\n'));
            if (id == kNoToken || impl.id_is_added[id]) {
                runs.clear();
                break;
            }
            runs.push_back(id);
        }
        impl.newline_run_tok = std::move(runs);
    }

    const json& merges = root["model"]["merges"];
    size_t cap = 1024;
    while (cap < merges.size() * 2) cap <<= 1;
    impl.pair_table.assign(cap, Tokenizer::Impl::PairEntry{Tokenizer::Impl::kEmptyPair, 0, 0});
    impl.pair_mask = cap - 1;
    for (size_t i = 0; i < merges.size(); ++i) {
        std::string a, b;
        if (merges[i].is_array() && merges[i].size() == 2) {
            a = merges[i][0].get<std::string>();
            b = merges[i][1].get<std::string>();
        } else if (merges[i].is_string()) {
            const std::string s = merges[i].get<std::string>();
            const size_t sp = s.find(' ', 1);  // token texts contain no ' ' (normalized to U+2581)
            if (sp == std::string::npos) throw std::runtime_error("Tokenizer: bad merge string at " + std::to_string(i));
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        } else {
            throw std::runtime_error("Tokenizer: unrecognized model.merges[" + std::to_string(i) + "] shape");
        }
        const TokenId ia = impl.lookup(a), ib = impl.lookup(b), im = impl.lookup(a + b);
        if (ia == kNoToken || ib == kNoToken || im == kNoToken) {
            throw std::runtime_error("Tokenizer: merge " + std::to_string(i) + " references a token not in the vocab");
        }
        impl.insert_pair(ia, ib, static_cast<int32_t>(i), im);
    }
}

// Empirically confirms `unicode_regex_split()` actually dispatched to the intended hand-written
// fast path (unicode_regex_split_custom_qwen2()/_qwen35(), see unicode.cpp's
// unicode_regex_split_custom() dispatcher) rather than silently falling back to the generic
// std::regex/collapsed-text path, which would change tokenization with no error (see the trigger
// constants' block comment above). Probes with a documented, previously-verified fact about this
// checkpoint's merge vocab: the Thai word for Bangkok mixes a base-letter run with a combining
// mark and splits differently under the two fast paths -- one single run under the Qwen3.5 path
// (marks absorbed), two+ runs under the Qwen2 path (marks not absorbed). If a future re-vendor of
// unicode.cpp changes either dispatcher literal so it no longer matches kQwen2SplitTrigger /
// kQwen35SplitTrigger below, this throws at load time instead of silently mis-tokenizing.
void verify_split_dispatch(const std::string& trigger) {
    const std::string probe = "กรุงเทพมหานคร";
    const std::vector<std::string> parts = unicode_regex_split(probe, {trigger}, /*byte_encode=*/false);
    if (trigger == kQwen35SplitTrigger) {
        if (parts.size() != 1 || parts[0] != probe) {
            throw std::runtime_error(
                "Tokenizer: split-dispatch self-check failed for the Qwen3.5 trigger -- "
                "unicode_regex_split_custom_qwen35() likely did not fire (fell back to the generic "
                "std::regex path). Check that kQwen35SplitTrigger still matches "
                "vendor/llamacpp_unicode/unicode.cpp's dispatcher literal after any re-vendor.");
        }
    } else if (trigger == kQwen2SplitTrigger) {
        if (parts.size() < 2) {
            throw std::runtime_error(
                "Tokenizer: split-dispatch self-check failed for the Qwen2 trigger -- "
                "unicode_regex_split_custom_qwen2() likely did not fire (fell back to the generic "
                "std::regex path). Check that kQwen2SplitTrigger still matches "
                "vendor/llamacpp_unicode/unicode.cpp's dispatcher literal after any re-vendor.");
        }
    }
}

std::string added_token_text(const json& entry) {
    if (entry.contains("content")) return entry["content"].get<std::string>();
    return {};
}

Tokenizer::Impl load_impl(const std::string& tokenizer_json_path, const std::string& tokenizer_config_json_path,
                           const std::string& generation_config_json_path, const Tokenizer::Options& options) {
    Tokenizer::Impl impl;

    json root = json::parse(read_file_or_throw(tokenizer_json_path), /*cb=*/nullptr, /*allow_exceptions=*/true);
    if (!root.contains("model") || !root["model"].is_object()) {
        throw std::runtime_error("Tokenizer: tokenizer.json has no model object");
    }
    // Kind is selected from the file: byte_fallback BPE is the SPM-style (Gemma) backend, anything
    // else must be the Qwen-style byte-level regex shape.
    const bool spm = root["model"].value("byte_fallback", false);
    if (spm) {
        validate_spm(root);
        impl.kind = TokenizerKind::kSpmByteFallback;
    } else {
        validate_pretokenizer(root, options);
    }

    const json& model = root["model"];
    impl.ignore_merges = model.value("ignore_merges", false);

    // Default the split trigger to tokenizer.json's own declared pre_tokenizer field
    // (validate_pretokenizer() above already checked it is one of the two known patterns);
    // possibly overridden just below by tokenizer_config.json's own `pretokenize_regex` field if
    // that disagrees (it doesn't for this checkpoint -- both on-disk sources agree), and finally
    // by options.hf_transformers_compat if the caller explicitly opted into reproducing
    // transformers 5.17.0's Qwen2Tokenizer bug -- see the block comment on
    // kQwen2SplitTrigger/kQwen35SplitTrigger above for why that override exists at all.
    if (!spm) {
        const std::string regex =
            root["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].get<std::string>();
        impl.split_trigger = (regex == kQwen2SplitExpected) ? kQwen2SplitTrigger : kQwen35SplitTrigger;
    }

    // Merge vocab: model.vocab is {token_text: id}; size the id_to_token table to the max id + 1
    // across BOTH the merge vocab and the added tokens appended after it.
    size_t max_id = 0;
    for (const auto& [text, id_json] : model["vocab"].items()) {
        max_id = std::max(max_id, static_cast<size_t>(id_json.get<int64_t>()));
    }
    const json& added = root.value("added_tokens", json::array());
    for (const auto& entry : added) {
        max_id = std::max(max_id, static_cast<size_t>(entry["id"].get<int64_t>()));
    }

    impl.id_to_token.assign(max_id + 1, std::string());
    impl.id_is_added.assign(max_id + 1, false);
    impl.id_is_special.assign(max_id + 1, false);

    for (const auto& [text, id_json] : model["vocab"].items()) {
        const TokenId id = static_cast<TokenId>(id_json.get<int64_t>());
        impl.id_to_token[id] = text;
        impl.token_to_id[text] = id;
    }

    for (const auto& entry : added) {
        const TokenId id = static_cast<TokenId>(entry["id"].get<int64_t>());
        const std::string text = added_token_text(entry);
        const bool special = entry.value("special", false);
        impl.id_to_token[id] = text;
        impl.id_is_added[id] = true;
        impl.id_is_special[id] = special;
        impl.token_to_id[text] = id;  // added tokens override merge-vocab collisions, matching HF
        impl.added_tokens_by_len.push_back(AddedTokenInfo{id, text, special});
    }
    std::sort(impl.added_tokens_by_len.begin(), impl.added_tokens_by_len.end(),
              [](const AddedTokenInfo& a, const AddedTokenInfo& b) {
                  if (a.text.size() != b.text.size()) return a.text.size() > b.text.size();
                  return a.id < b.id;
              });

    // Merges: array of "left right" strings (this tokenizer.json's format) or [left, right]
    // 2-element arrays (older tokenizers.js format); support both. The SPM backend keeps its own
    // id-keyed table instead (build_spm_tables).
    const json& merges = model["merges"];
    if (spm) {
        build_spm_tables(impl, root);
    } else {
        impl.merge_rank.reserve(merges.size() * 2);
    }
    for (size_t i = 0; i < (spm ? 0 : merges.size()); ++i) {
        std::string key;
        if (merges[i].is_string()) {
            key = merges[i].get<std::string>();
        } else if (merges[i].is_array() && merges[i].size() == 2) {
            key = merges[i][0].get<std::string>() + " " + merges[i][1].get<std::string>();
        } else {
            throw std::runtime_error("Tokenizer: unrecognized model.merges[" + std::to_string(i) + "] shape");
        }
        impl.merge_rank.emplace(std::move(key), static_cast<int32_t>(i));
    }

    // bos/eos/pad: generation_config.json's ids are authoritative when present (task-specified
    // source); fall back to tokenizer_config.json's token *text*, looked up in the vocab we just
    // built, when generation_config.json is absent or missing a field. EXCEPT bos: Qwen's
    // generation_config.json reuses <|endoftext|> as `bos_token_id`, but that field is actually
    // the *decoder* start id, not a prompt BOS -- tokenizer_config.json's own
    // `add_bos_token=false`/`bos_token=null` is this checkpoint's authoritative statement that it
    // has no BOS at all, and takes priority over generation_config.json's field when it says so
    // (see the bos_id() doc comment in tokenizer.h).
    std::string tc_bos_text, tc_eos_text, tc_pad_text;
    bool tc_says_no_bos = false;
    if (!tokenizer_config_json_path.empty()) {
        const std::string raw = read_file_optional(tokenizer_config_json_path);
        if (!raw.empty()) {
            json cfg = json::parse(raw, nullptr, true);
            auto text_of = [&](const char* key) -> std::string {
                if (!cfg.contains(key) || cfg[key].is_null()) return {};
                if (cfg[key].is_string()) return cfg[key].get<std::string>();
                if (cfg[key].is_object()) return added_token_text(cfg[key]);
                return {};
            };
            tc_bos_text = text_of("bos_token");
            tc_eos_text = text_of("eos_token");
            tc_pad_text = text_of("pad_token");

            if ((cfg.contains("bos_token") && cfg["bos_token"].is_null()) ||
                cfg.value("add_bos_token", true) == false) {
                tc_says_no_bos = true;
            }

            // On-disk pre-tokenizer declaration, cross-checked against tokenizer.json's own field
            // (both should agree for a well-formed checkpoint; they do for this one).
            if (!spm) {
                const std::string tc_pretokenize_regex = cfg.value("pretokenize_regex", std::string());
                if (tc_pretokenize_regex == kQwen2SplitExpected) {
                    impl.split_trigger = kQwen2SplitTrigger;
                } else if (tc_pretokenize_regex == kQwen35SplitExpected) {
                    impl.split_trigger = kQwen35SplitTrigger;
                }

                // Explicit opt-in only: reproduce transformers 5.17.0's Qwen2Tokenizer bug of
                // discarding both on-disk pre-tokenizer declarations above.
                if (options.hf_transformers_compat && cfg.value("tokenizer_class", std::string()) == "Qwen2Tokenizer") {
                    impl.split_trigger = kQwen2SplitTrigger;
                }
            }
        }
    }
    if (!spm) verify_split_dispatch(impl.split_trigger);

    if (!tc_bos_text.empty()) impl.bos = impl.lookup(tc_bos_text);
    if (!tc_eos_text.empty()) impl.eos = impl.lookup(tc_eos_text);
    if (!tc_pad_text.empty()) impl.pad = impl.lookup(tc_pad_text);

    if (!generation_config_json_path.empty()) {
        const std::string raw = read_file_optional(generation_config_json_path);
        if (!raw.empty()) {
            json cfg = json::parse(raw, nullptr, true);
            if (!tc_says_no_bos && cfg.contains("bos_token_id") && cfg["bos_token_id"].is_number_integer()) {
                impl.bos = cfg["bos_token_id"].get<TokenId>();
            }
            if (cfg.contains("pad_token_id") && cfg["pad_token_id"].is_number_integer()) {
                impl.pad = cfg["pad_token_id"].get<TokenId>();
            }
            if (cfg.contains("eos_token_id")) {
                if (cfg["eos_token_id"].is_number_integer()) {
                    impl.eos_list = {cfg["eos_token_id"].get<TokenId>()};
                } else if (cfg["eos_token_id"].is_array()) {
                    for (const auto& v : cfg["eos_token_id"]) impl.eos_list.push_back(v.get<TokenId>());
                }
            }
        }
    }
    if (tc_says_no_bos) impl.bos = kNoToken;
    if (impl.eos_list.empty() && impl.eos != kNoToken) impl.eos_list = {impl.eos};
    if (!impl.eos_list.empty()) impl.eos = impl.eos_list.front();

    if (!options.keep_special_on_decode.empty()) {
        impl.keep_special.assign(impl.id_to_token.size(), false);
        for (const std::string& text : options.keep_special_on_decode) {
            const TokenId id = impl.lookup(text);
            if (id == kNoToken || !impl.id_is_added[id] || !impl.id_is_special[id]) {
                throw std::runtime_error("Tokenizer: keep_special_on_decode entry \"" + text +
                                          "\" is not a special added token");
            }
            impl.keep_special[id] = true;
        }
    }

    return impl;
}

}  // namespace

// --- Tokenizer public API -----------------------------------------------------------------

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Tokenizer Tokenizer::from_files(const std::string& tokenizer_json_path,
                                 const std::string& tokenizer_config_json_path,
                                 const std::string& generation_config_json_path, const Options& options) {
    try {
        return Tokenizer(std::make_unique<Impl>(load_impl(tokenizer_json_path, tokenizer_config_json_path,
                                                            generation_config_json_path, options)));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Tokenizer::from_files: ") + e.what());
    }
}

Tokenizer Tokenizer::from_directory(const std::string& model_dir, const Options& options) {
    return from_files(join_dir(model_dir, "tokenizer.json"), join_dir(model_dir, "tokenizer_config.json"),
                       join_dir(model_dir, "generation_config.json"), options);
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool parse_special) const {
    if (!impl_) throw std::runtime_error("Tokenizer::encode: tokenizer not loaded");
    std::vector<TokenId> out;
    const bool spm = impl_->kind == TokenizerKind::kSpmByteFallback;

    for (const Fragment& frag : split_added_tokens(*impl_, text, parse_special)) {
        if (frag.is_token) {
            out.push_back(frag.token);
            continue;
        }
        if (frag.text.empty()) continue;
        if (spm) {
            spm_encode_text(*impl_, frag.text, out);
            continue;
        }
        const std::string chunk(frag.text);
        for (const std::string& word : unicode_regex_split(chunk, {impl_->split_trigger}, /*byte_encode=*/true)) {
            bpe_merge_word(*impl_, word, out);
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids, bool skip_special_tokens) const {
    if (!impl_) throw std::runtime_error("Tokenizer::decode: tokenizer not loaded");
    if (impl_->kind == TokenizerKind::kSpmByteFallback) return spm_decode(*impl_, ids, skip_special_tokens);
    std::string out;
    for (const TokenId id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= impl_->id_to_token.size()) continue;  // ignore invalid ids
        if (decode_skips(*impl_, id, skip_special_tokens)) continue;
        if (impl_->id_is_added[id]) {
            out += impl_->id_to_token[id];
        } else {
            out += decode_merge_vocab_piece(impl_->id_to_token[id]);
        }
    }
    return out;
}

struct Tokenizer::StreamDecoder::Impl {
    const Tokenizer::Impl* tok = nullptr;
    bool skip_special = true;
    std::string buffer;  // Qwen: undecoded UTF-8 tail; SPM: pending `<0xNN>` byte run
};

Tokenizer::StreamDecoder::StreamDecoder() : impl_(std::make_unique<Impl>()) {}
Tokenizer::StreamDecoder::~StreamDecoder() = default;
Tokenizer::StreamDecoder::StreamDecoder(StreamDecoder&&) noexcept = default;
Tokenizer::StreamDecoder& Tokenizer::StreamDecoder::operator=(StreamDecoder&&) noexcept = default;

std::string Tokenizer::StreamDecoder::push(TokenId id) {
    Impl& s = *impl_;
    if (!s.tok || id < 0 || static_cast<size_t>(id) >= s.tok->id_to_token.size()) return {};
    if (decode_skips(*s.tok, id, s.skip_special)) return {};

    if (s.tok->kind == TokenizerKind::kSpmByteFallback) {
        if (s.tok->byte_of[id] >= 0) {
            s.buffer.push_back(static_cast<char>(s.tok->byte_of[id]));
            return spm_drain_run(s.buffer, /*final=*/false);
        }
        std::string emitted = spm_drain_run(s.buffer, /*final=*/true);  // a held tail can never complete now
        emitted += s.tok->decoded_piece[id];
        return emitted;
    }

    s.buffer += s.tok->id_is_added[id] ? s.tok->id_to_token[id] : decode_merge_vocab_piece(s.tok->id_to_token[id]);

    const size_t safe = valid_utf8_prefix_length(s.buffer);
    if (safe == 0) return {};
    std::string emitted = s.buffer.substr(0, safe);
    s.buffer.erase(0, safe);
    return emitted;
}

std::string Tokenizer::StreamDecoder::flush() {
    Impl& s = *impl_;
    if (s.tok && s.tok->kind == TokenizerKind::kSpmByteFallback) return spm_drain_run(s.buffer, /*final=*/true);
    std::string rest = std::move(s.buffer);
    s.buffer.clear();
    return rest;
}

Tokenizer::StreamDecoder Tokenizer::make_stream_decoder(bool skip_special_tokens) const {
    if (!impl_) throw std::runtime_error("Tokenizer::make_stream_decoder: tokenizer not loaded");
    StreamDecoder dec;
    dec.impl_->tok = impl_.get();
    dec.impl_->skip_special = skip_special_tokens;
    return dec;
}

Tokenizer::Kind Tokenizer::kind() const { return impl_ ? impl_->kind : TokenizerKind::kByteLevelRegex; }

size_t Tokenizer::vocab_size() const { return impl_ ? impl_->id_to_token.size() : 0; }

bool Tokenizer::is_special(TokenId id) const {
    return impl_ && id >= 0 && static_cast<size_t>(id) < impl_->id_is_special.size() && impl_->id_is_special[id];
}

bool Tokenizer::is_added_token(TokenId id) const {
    return impl_ && id >= 0 && static_cast<size_t>(id) < impl_->id_is_added.size() && impl_->id_is_added[id];
}

const std::string& Tokenizer::piece(TokenId id) const {
    if (!impl_) throw std::runtime_error("Tokenizer::piece: tokenizer not loaded");
    return impl_->id_to_token.at(id);
}

TokenId Tokenizer::bos_id() const { return impl_ ? impl_->bos : kNoToken; }
TokenId Tokenizer::eos_id() const { return impl_ ? impl_->eos : kNoToken; }
TokenId Tokenizer::pad_id() const { return impl_ ? impl_->pad : kNoToken; }

const std::vector<TokenId>& Tokenizer::eos_ids() const {
    static const std::vector<TokenId> empty;
    return impl_ ? impl_->eos_list : empty;
}

}  // namespace r4dx
