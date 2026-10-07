// r4dx::core::DecodeLegacy: the per-item kill switches of the decode-step efficiency pass
// (docs/perf.md "Decode step: bit-exact launch and host-edge cuts"). Every item is bit-exact against
// the code it replaced, so none of them is a user-facing flag; this exists to A/B one item at a time
// on the same binary:
//
//   R4DX_DECODE_LEGACY=argmax,ab,attn,host,gdnwo  (comma / semicolon / space separated; "all" = every item)
//
//   argmax  the one-workgroup argmax kernel (and the widen launch ahead of it) instead of the
//           multi-workgroup one, in every caller (plain decode, MTP/DFlash draft and verify, TP=2 pairs)
//   ab      gdn.in_proj_a and gdn.in_proj_b as two launches instead of one merged launch
//   attn    split_qg + qk_norm + rope + kv_write as five launches instead of one fused kernel
//   host    the blocking per-step uploads, stream synchronize and 4-byte D2H, and the CLI/server
//           output ahead of the next step, instead of pinned async uploads, an event poll and the
//           output overlapped with the step
//   gdnwo   the GDN recurrent state with one slot per candidate row of a speculative window (every
//           row stores its own state) instead of the write-once state (docs/gdn-write-once.md): the
//           kill switch of ModelOptions::gdn_write_once / R4DX_GDN_WRITE_ONCE (gdn_write_once.h)
//
// Read once per process (the env is never re-read), like the other R4DX_* selectors. An unknown token
// is reported once on stderr and ignored.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace r4dx::core {

enum class DecodeItem : unsigned {
  kArgmax = 1u << 0,
  kAb = 1u << 1,
  kAttn = 1u << 2,
  kHost = 1u << 3,
  kGdnWo = 1u << 4,
};
inline constexpr unsigned kDecodeItemAll = 0x1Fu;

// Pure parse of the variable's value (null/empty = no item): the mask of legacy items.
inline unsigned ParseDecodeLegacy(const char* value, bool warn = true) {
  if (value == nullptr) return 0;
  unsigned mask = 0;
  std::string tok;
  const std::string text(value);
  for (size_t i = 0; i <= text.size(); ++i) {
    const char c = i < text.size() ? text[i] : ',';
    if (c == ',' || c == ';' || c == ' ' || c == '\t') {
      if (!tok.empty()) {
        if (tok == "all") {
          mask |= kDecodeItemAll;
        } else if (tok == "argmax") {
          mask |= static_cast<unsigned>(DecodeItem::kArgmax);
        } else if (tok == "ab") {
          mask |= static_cast<unsigned>(DecodeItem::kAb);
        } else if (tok == "attn") {
          mask |= static_cast<unsigned>(DecodeItem::kAttn);
        } else if (tok == "host") {
          mask |= static_cast<unsigned>(DecodeItem::kHost);
        } else if (tok == "gdnwo") {
          mask |= static_cast<unsigned>(DecodeItem::kGdnWo);
        } else if (warn) {
          std::fprintf(stderr,
                       "r4dx: R4DX_DECODE_LEGACY token '%s' not recognized (argmax|ab|attn|host|gdnwo|all); "
                       "ignored\n",
                       tok.c_str());
        }
        tok.clear();
      }
    } else {
      tok.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c);
    }
  }
  return mask;
}

inline unsigned DecodeLegacyMask() {
  static const unsigned mask = ParseDecodeLegacy(std::getenv("R4DX_DECODE_LEGACY"));
  return mask;
}

inline bool DecodeLegacy(DecodeItem item) {
  return (DecodeLegacyMask() & static_cast<unsigned>(item)) != 0;
}

}  // namespace r4dx::core
