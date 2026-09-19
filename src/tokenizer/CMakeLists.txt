# Tokenizer: byte-level BPE (tokenizer.json) + minja chat-template rendering.
#
# vendor/llamacpp_unicode/{unicode.h,unicode.cpp,unicode-data.h,unicode-data.cpp} are vendored
# verbatim from llama.cpp (MIT) -- see vendor/llamacpp_unicode/NOTICE. Everything else here is
# r4dx's own code (bpe_tokenizer.cpp, chat_template.cpp), including a port -- not a vendored copy
# -- of llama.cpp's BPE merge loop and special-token splitter onto this file's data structures.
add_library(r4dx_tokenizer STATIC
  bpe_tokenizer.cpp
  chat_template.cpp
  vendor/llamacpp_unicode/unicode.cpp
  vendor/llamacpp_unicode/unicode-data.cpp
)

target_include_directories(r4dx_tokenizer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
# nlohmann/json.hpp and minja/{minja,chat-template}.hpp are vendored under third_party/<name>/,
# so third_party/ itself is the include root that makes `#include "nlohmann/json.hpp"` and
# `#include "minja/chat-template.hpp"` resolve (minja's own internal `#include "minja.hpp"` is a
# same-directory "" include and needs no extra path). PUBLIC: chat_template.h (a public header of
# this target) itself includes <nlohmann/json.hpp>, so consumers linking r4dx_tokenizer need this
# path too.
target_include_directories(r4dx_tokenizer PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party
)

target_compile_features(r4dx_tokenizer PUBLIC cxx_std_17)
