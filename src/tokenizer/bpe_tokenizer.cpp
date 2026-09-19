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
    validate_pretokenizer(root, options);

    const json& model = root["model"];
    impl.ignore_merges = model.value("ignore_merges", false);

    // Default the split trigger to tokenizer.json's own declared pre_tokenizer field
    // (validate_pretokenizer() above already checked it is one of the two known patterns);
    // possibly overridden just below by tokenizer_config.json's own `pretokenize_regex` field if
    // that disagrees (it doesn't for this checkpoint -- both on-disk sources agree), and finally
    // by options.hf_transformers_compat if the caller explicitly opted into reproducing
    // transformers 5.17.0's Qwen2Tokenizer bug -- see the block comment on
    // kQwen2SplitTrigger/kQwen35SplitTrigger above for why that override exists at all.
    {
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
    // 2-element arrays (older tokenizers.js format); support both.
    const json& merges = model["merges"];
    impl.merge_rank.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); ++i) {
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
    verify_split_dispatch(impl.split_trigger);

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

    for (const Fragment& frag : split_added_tokens(*impl_, text, parse_special)) {
        if (frag.is_token) {
            out.push_back(frag.token);
            continue;
        }
        if (frag.text.empty()) continue;
        const std::string chunk(frag.text);
        for (const std::string& word : unicode_regex_split(chunk, {impl_->split_trigger}, /*byte_encode=*/true)) {
            bpe_merge_word(*impl_, word, out);
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids, bool skip_special_tokens) const {
    if (!impl_) throw std::runtime_error("Tokenizer::decode: tokenizer not loaded");
    std::string out;
    for (const TokenId id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= impl_->id_to_token.size()) continue;  // ignore invalid ids
        if (skip_special_tokens && impl_->id_is_special[id]) continue;
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
    std::string buffer;
};

Tokenizer::StreamDecoder::StreamDecoder() : impl_(std::make_unique<Impl>()) {}
Tokenizer::StreamDecoder::~StreamDecoder() = default;
Tokenizer::StreamDecoder::StreamDecoder(StreamDecoder&&) noexcept = default;
Tokenizer::StreamDecoder& Tokenizer::StreamDecoder::operator=(StreamDecoder&&) noexcept = default;

std::string Tokenizer::StreamDecoder::push(TokenId id) {
    Impl& s = *impl_;
    if (!s.tok || id < 0 || static_cast<size_t>(id) >= s.tok->id_to_token.size()) return {};
    if (s.skip_special && s.tok->id_is_special[id]) return {};

    s.buffer += s.tok->id_is_added[id] ? s.tok->id_to_token[id] : decode_merge_vocab_piece(s.tok->id_to_token[id]);

    const size_t safe = valid_utf8_prefix_length(s.buffer);
    if (safe == 0) return {};
    std::string emitted = s.buffer.substr(0, safe);
    s.buffer.erase(0, safe);
    return emitted;
}

std::string Tokenizer::StreamDecoder::flush() {
    Impl& s = *impl_;
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
