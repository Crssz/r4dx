// r4dx_convert::KeepBf16Selector -- the `--keep-bf16 <regex>` escape hatch (src/convert/main.cpp).
//
// Every linear whose CONTAINER BASE NAME the regex matches is written as `<base>.bf16.w` ONLY --
// none of the quantized layouts the run otherwise asked for. src/model/container.cpp's
// LoadQuantLinearWithFallback already recognizes that on-disk shape and falls exactly those linears
// back to bf16 at load time, so the result is a container that is quantized everywhere EXCEPT the
// matched tensor class. That is the measurement instrument docs/validation.md "Milestone 11 /
// sensitivity" is built on: convert one class at a time in bf16, re-run the KL harness, and the
// KL that disappears is the KL that class was responsible for.
//
// MATCHING is std::regex_search with ECMAScript syntax over the container base name (e.g.
// "text.layers.17.attn.o", "lm_head") -- a SEARCH, not a full match, so the natural spellings work
// unanchored: "attn\.o$" selects every layer's attention output projection, "^text\.layers\.[0-7]\."
// selects the first eight layers' linears. Anchor both ends (`^...$`) when a substring would
// over-select. An invalid regex is a hard error before any byte is written; a VALID regex that
// matches nothing is a warning, not an error -- a sensitivity sweep scripted over a list of class
// regexes should report "that class has no linears in this container" and still produce a container,
// rather than dying halfway through a batch.
//
// The accounting: for each matched linear this records what it actually costs (bf16) against what
// it WOULD have cost in the layouts the run asked for, so the log states the price of the
// experiment in bytes without anyone re-deriving it from two `ls -l` outputs.
#pragma once

#include <cstdint>
#include <ostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/linear_layouts.hpp"

namespace r4dx_convert {

class KeepBf16Selector {
 public:
  // `pattern` empty => disabled (Matches() is always false, nothing is logged, no warning).
  explicit KeepBf16Selector(const std::string& pattern) : pattern_(pattern) {
    if (pattern_.empty()) return;
    try {
      re_ = std::regex(pattern_, std::regex::ECMAScript);
    } catch (const std::regex_error& e) {
      throw std::runtime_error("--keep-bf16: invalid ECMAScript regex '" + pattern_ +
                               "': " + e.what());
    }
    enabled_ = true;
  }

  bool Enabled() const { return enabled_; }
  const std::string& Pattern() const { return pattern_; }

  bool Matches(const std::string& container_base) const {
    return enabled_ && std::regex_search(container_base, re_);
  }

  // Records one matched linear and returns the LayoutSet it must actually be written in.
  // `would_have_been` is the LayoutSet this linear would have gotten without --keep-bf16 -- the
  // baseline the reported byte delta is against. Called from the (single-threaded) planning pass,
  // in container order, so the per-linear lines below read top to bottom like the model does.
  void Record(const std::string& container_base, int N, int K, const LayoutSet& would_have_been,
              std::ostream& log) {
    const uint64_t kept = LinearLayoutBytes(N, K, KeptBf16LayoutSet());
    const uint64_t original = LinearLayoutBytes(N, K, would_have_been);
    matched_.push_back(container_base);
    kept_bytes_ += kept;
    original_bytes_ += original;
    log << "[r4dx-convert] keep-bf16: " << container_base << " [" << N << "," << K << "] bf16 "
        << kept << " B (would have been " << original << " B, " << Signed(kept, original) << ")\n";
  }

  // One summary line after the planning pass. Goes to `warn` (stderr) rather than `log` when the
  // regex matched nothing: a sweep driver that mistypes a class regex would otherwise get a
  // perfectly ordinary-looking conversion whose KL is, mysteriously, exactly the baseline's.
  void Report(std::ostream& log, std::ostream& warn) const {
    if (!enabled_) return;
    if (matched_.empty()) {
      warn << "[r4dx-convert] WARNING: --keep-bf16 '" << pattern_
           << "' matched no linear -- every linear was quantized as usual\n";
      return;
    }
    log << "[r4dx-convert] keep-bf16 '" << pattern_ << "': " << matched_.size()
        << " linear(s) kept as bf16, " << kept_bytes_ << " B instead of " << original_bytes_
        << " B (" << Signed(kept_bytes_, original_bytes_) << ", "
        << (static_cast<double>(static_cast<int64_t>(kept_bytes_) -
                                static_cast<int64_t>(original_bytes_)) /
            (1024.0 * 1024.0 * 1024.0))
        << " GiB)\n";
  }

  size_t MatchedCount() const { return matched_.size(); }
  const std::vector<std::string>& Matched() const { return matched_; }
  // Signed byte delta of the whole selection (positive = the container grew).
  int64_t ExtraBytes() const {
    return static_cast<int64_t>(kept_bytes_) - static_cast<int64_t>(original_bytes_);
  }

 private:
  static std::string Signed(uint64_t kept, uint64_t original) {
    const int64_t d = static_cast<int64_t>(kept) - static_cast<int64_t>(original);
    return (d >= 0 ? "+" : "") + std::to_string(d) + " B";
  }

  std::string pattern_;
  bool enabled_ = false;
  std::regex re_;
  std::vector<std::string> matched_;
  uint64_t kept_bytes_ = 0, original_bytes_ = 0;
};

}  // namespace r4dx_convert
