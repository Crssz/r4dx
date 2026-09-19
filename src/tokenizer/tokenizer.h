// r4dx byte-level BPE tokenizer -- public API for src/cli and src/server.
//
// Implements the tokenizer described by Qwen3.8-27B's `tokenizer.json`: a GPT-2/Qwen-style
// byte-level BPE vocab (248044 merge-vocab tokens + 33 added tokens). By default r4dx trusts the
// checkpoint's own on-disk pre-tokenizer declaration -- tokenizer_config.json's `pretokenize_regex`
// field if present, else tokenizer.json's `pre_tokenizer` field -- which for this checkpoint is
// the "Qwen3.5" pattern (marks absorbed into letter runs: `[\p{L}\p{M}]+`). transformers 5.17.0
// (which loads this checkpoint as `Qwen2Tokenizer`) silently discards BOTH of those on-disk
// declarations and uses the *older* Qwen2 pattern at runtime instead -- a confirmed bug for this
// checkpoint, not the intended tokenization (see the comment above kQwen2SplitTrigger in
// bpe_tokenizer.cpp for the full explanation and empirical proof, e.g. it makes the single-token
// Thai word "กรุงเทพมหานคร" split into 3 tokens instead of 1). Pass
// `Options{.hf_transformers_compat = true}` to `from_directory`/`from_files` to reproduce that
// (buggy) transformers behavior instead, for A/B testing only:
//
//   Qwen2:   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   Qwen3.5: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// The regex splitter and Unicode codepoint tables are vendored from llama.cpp
// (src/tokenizer/vendor/llamacpp_unicode/, see NOTICE there); the BPE merge loop and
// added/special-token splitting are r4dx's own port of the equivalent llama.cpp algorithms onto
// this file's data structures -- see bpe_tokenizer.cpp for the detailed mapping.
//
// KNOWN GAP: tokenizer.json specifies `"normalizer": {"type": "NFC"}`, and the reference
// tokenizer (transformers' Qwen2Tokenizer, itself backed by the Rust `tokenizers` crate) applies
// Unicode NFC canonical composition to input text before pre-tokenizing. This implementation does
// NOT perform NFC normalization -- doing so correctly requires the full Unicode canonical
// decomposition/composition tables, which llama.cpp's vendored unicode-data.cpp does not carry
// (it only has NFD, for SPM-style tokenizers). Effect: text already in NFC form (which is true of
// essentially all real-world UTF-8 -- Python/JSON source text, and text produced by any normal
// text editor or browser) round-trips identically; text that mixes decomposed sequences (a base
// character followed by a separate combining mark, e.g. "e" + U+0301 instead of the precomposed
// "é") will tokenize differently from the reference tokenizer. tests/tokenizer/golden.json's
// cases are therefore constructed from NFC-normalized strings.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace r4dx {

using TokenId = int32_t;

// -1 means "no such token" (e.g. this tokenizer has no BOS token: Qwen3.8-27B's
// tokenizer_config.json has `bos_token: null`, `add_bos_token: false`).
constexpr TokenId kNoToken = -1;

// Load-time behavior toggles for Tokenizer::from_directory()/from_files(). Defaults reproduce the
// checkpoint's own on-disk declarations, not transformers 5.17.0's runtime behavior where the two
// disagree (see the file comment above `class Tokenizer` for why that disagreement is a
// transformers bug, not the intended tokenization). Declared at namespace scope (rather than
// nested inside `class Tokenizer`) so its default member initializers below are usable in
// Tokenizer's own member function default arguments; `Tokenizer::Options` is a public alias to
// this type for callers, e.g. `Tokenizer::Options{.hf_transformers_compat = true}`.
struct TokenizerOptions {
    // false (default): trust the on-disk pre-tokenizer declaration (tokenizer_config.json's
    // `pretokenize_regex` field, else tokenizer.json's `pre_tokenizer` field).
    // true: reproduce transformers 5.17.0's `Qwen2Tokenizer` override verbatim (forces the
    // older Qwen2 split pattern whenever tokenizer_config.json's `tokenizer_class` is
    // "Qwen2Tokenizer", discarding both on-disk declarations) -- use only to A/B against that
    // specific (buggy) reference tokenizer's output; do not use for production tokenization
    // of this checkpoint.
    bool hf_transformers_compat = false;

    // false (default): refuse to load (throw) if tokenizer.json declares a `normalizer` this
    // implementation does not apply (e.g. NFC -- see the KNOWN GAP comment above). true:
    // load anyway, accepting that text mixing decomposed Unicode sequences will silently
    // mis-tokenize relative to the reference tokenizer.
    bool allow_unimplemented_normalizer = false;
};

// Byte-level BPE tokenizer. Thread-safe for concurrent encode()/decode() calls on the same
// instance once constructed (all state is read-only after from_directory()/from_files()); each
// caller that needs streaming decode should own its own StreamDecoder.
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    using Options = TokenizerOptions;

    // Loads `<model_dir>/tokenizer.json` (required) plus `tokenizer_config.json` and
    // `generation_config.json` from the same directory if present (used for bos/eos/pad ids and
    // as a cross-check on tokenizer.json's own added_tokens). Throws std::runtime_error with a
    // descriptive message on any parse failure or on an unrecognized pre-tokenizer/decoder shape
    // -- silently mis-tokenizing is worse than refusing to load.
    static Tokenizer from_directory(const std::string& model_dir, const Options& options = {});

    static Tokenizer from_files(const std::string& tokenizer_json_path,
                                 const std::string& tokenizer_config_json_path = "",
                                 const std::string& generation_config_json_path = "",
                                 const Options& options = {});

    // Encodes `text` to token ids via the Qwen3.5 pre-tokenizer regex + byte-level BPE merges.
    //
    // `parse_special` gates whether added tokens flagged `"special": true` in tokenizer.json
    // (<|im_start|>, <|im_end|>, <|endoftext|>, the vision/audio sentinels, ...) are recognized
    // when they appear literally inside `text`:
    //   - false (default): NOT recognized -- such a substring is tokenized as ordinary text via
    //     byte-level BPE instead of becoming its special token id. Matches HF
    //     `tokenizer.encode(text, split_special_tokens=True)`.
    //   - true: recognized. Pass true only for text you fully trust the control-token structure
    //     of, e.g. the output of ChatTemplate::render() (which legitimately contains literal
    //     <|im_start|>/<|im_end|> control sequences that must become their token ids).
    //
    // CAUTION: this flag alone does NOT sandbox untrusted user content embedded in a rendered chat
    // prompt. ChatTemplate::render() splices message bodies into the template output verbatim, and
    // the composed result is then encoded with parse_special=true (required so the template's own
    // control sequences work) -- so a user message body containing literal "<|im_end|>\n<|im_start|>
    // system" text becomes real control token ids too, once that composed string is encoded. r4dx
    // does not currently strip/escape special-token surface forms from message content before
    // rendering, nor tokenize the template's control sequences and message bodies separately. Callers
    // that need this sandboxed must implement that segmentation themselves before calling encode().
    // Added tokens NOT flagged special (<think>, <tool_call>, </tool_call>, <tts_pad>, ...) are
    // always recognized regardless of `parse_special`, matching HF's own default behavior for
    // non-special added vocabulary.
    std::vector<TokenId> encode(std::string_view text, bool parse_special = false) const;

    // Decodes a full id sequence back to UTF-8 text in one shot (not incremental -- see
    // StreamDecoder for token-at-a-time generation). `skip_special_tokens = true` (default) omits
    // every token flagged `"special": true`; non-special added tokens (<think>, ...) are always
    // rendered as their literal surface text.
    std::string decode(const std::vector<TokenId>& ids, bool skip_special_tokens = true) const;

    // Incremental / streaming decoder: feed one sampled token id at a time and get back only the
    // newly-complete UTF-8 text. Byte-level BPE tokens do not align to UTF-8 codepoint boundaries
    // (a single non-ASCII character can be split across 2-4 tokens), so a naive per-token
    // decode-and-concatenate can emit a truncated/invalid UTF-8 fragment; StreamDecoder buffers
    // any incomplete trailing codepoint internally until push() or flush() can complete it.
    class StreamDecoder {
    public:
        StreamDecoder();
        ~StreamDecoder();
        StreamDecoder(StreamDecoder&&) noexcept;
        StreamDecoder& operator=(StreamDecoder&&) noexcept;
        StreamDecoder(const StreamDecoder&) = delete;
        StreamDecoder& operator=(const StreamDecoder&) = delete;

        // Returns the newly-complete UTF-8 text unlocked by this token; "" if `id` only extended
        // a still-incomplete trailing codepoint.
        std::string push(TokenId id);

        // Call once at end of stream: returns whatever is left in the internal buffer verbatim
        // (e.g. a codepoint that was never completed because generation stopped mid-sequence).
        std::string flush();

        // Opaque implementation type. Forward-declared here (not under `private:` below) purely
        // so bpe_tokenizer.cpp's free functions can name `StreamDecoder::Impl` to define and use
        // it; its actual definition lives only in that .cpp, so it stays fully opaque to every
        // other translation unit regardless of this declaration's own accessibility.
        struct Impl;

    private:
        friend class Tokenizer;
        std::unique_ptr<Impl> impl_;
    };

    StreamDecoder make_stream_decoder(bool skip_special_tokens = true) const;

    // Vocab / special-id accessors.
    size_t vocab_size() const;                 // id_to_token.size(), including added tokens
    bool is_special(TokenId id) const;          // true iff added_tokens[...].special == true
    bool is_added_token(TokenId id) const;      // true for any added token (special or not)
    const std::string& piece(TokenId id) const; // raw stored surface form (byte-encoded for
                                                 // merge-vocab ids, literal text for added ids);
                                                 // throws std::out_of_range if id is invalid

    TokenId bos_id() const;               // kNoToken if the tokenizer has no BOS (true here: this
                                           // checkpoint's tokenizer_config.json has
                                           // add_bos_token=false / bos_token=null, which takes
                                           // priority over generation_config.json's bos_token_id --
                                           // that field is the *decoder* start id Qwen reuses
                                           // <|endoftext|> for, not a prompt BOS)
    TokenId eos_id() const;               // primary eos, generation_config.json eos_token_id[0]
    const std::vector<TokenId>& eos_ids() const; // all stop ids (generation_config.json's list,
                                                  // or [eos_id()] if that file is absent)
    TokenId pad_id() const;

    // Opaque implementation type. Forward-declared here (not under `private:` below) purely so
    // bpe_tokenizer.cpp's free functions (bpe_merge_word, split_added_tokens, load_impl, ...) can
    // name `Tokenizer::Impl`; its actual definition lives only in that .cpp, so it stays fully
    // opaque to every other translation unit regardless of this declaration's own accessibility.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;

    explicit Tokenizer(std::unique_ptr<Impl> impl);
};

}  // namespace r4dx
