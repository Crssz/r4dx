// r4dx/models_root.h -- the ONE place the on-disk models root is decided for C++ code.
//
// Every default model/container/tokenizer/corpus path in src/ and tests/ is <root>\<rest>, where
// <root> is the environment variable R4DX_MODELS_ROOT, defaulting to E:\models. Nothing else in the
// C++ tree spells a literal models-root path. (CMake derives the same root for its cache variables
// from $ENV{R4DX_MODELS_ROOT} or E:/models -- see the root CMakeLists.txt.)
//
// Header-only and dependency-free; reachable from every target through the global include directory
// src/core/include set in the root CMakeLists.txt:  #include "r4dx/models_root.h"
#pragma once

#include <cstdlib>
#include <string>

namespace r4dx {

// R4DX_MODELS_ROOT when set and non-empty (trailing separators stripped), else E:\models.
inline std::string ModelsRoot() {
  std::string root;
#ifdef _MSC_VER
  // getenv is deprecated under the MSVC CRT headers clang-cl uses; _dupenv_s is the sanctioned spelling.
  char* env = nullptr;
  size_t env_len = 0;
  if (_dupenv_s(&env, &env_len, "R4DX_MODELS_ROOT") == 0 && env != nullptr) root = env;
  std::free(env);
#else
  if (const char* env = std::getenv("R4DX_MODELS_ROOT")) root = env;
#endif
  if (root.empty()) root = "E:\\models";
  while (root.size() > 1 && (root.back() == '/' || root.back() == '\\')) root.pop_back();
  return root;
}

// <ModelsRoot()><sep><rel>; rel uses '/' or '\\' separators and is normalised to the platform's.
inline std::string ModelsPath(const std::string& rel) {
#ifdef _WIN32
  const char sep = '\\', other = '/';
#else
  const char sep = '/', other = '\\';
#endif
  std::string out = ModelsRoot();
  if (!rel.empty()) {
    out.push_back(sep);
    for (char c : rel) out.push_back(c == other ? sep : c);
  }
  return out;
}

}  // namespace r4dx
