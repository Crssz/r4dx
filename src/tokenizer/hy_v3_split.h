// Hand-written pre-tokenizer splitter for hy_v3 (Hy-MT3-30B-A3B) tokenizer.json:
// Sequence of 3 Isolated Regex Splits:
//   1. \p{N}{1,3}
//   2. [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+
//   3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// followed by ByteLevel(add_prefix_space=false, use_regex=false).
#pragma once

#include <string>
#include <vector>

namespace r4dx {

extern const char* const kHyV3SplitRegex0;
extern const char* const kHyV3SplitRegex1;
extern const char* const kHyV3SplitRegex2;

std::vector<std::string> hy_v3_split_byte_encoded(const std::string& text);

}  // namespace r4dx
