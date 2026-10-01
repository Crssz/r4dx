// tests/vision/gemma_test_util.h -- plumbing for the Gemma 4 vision CPU tests. Deliberately dependency-free
// (no r4dx_core / r4dx_convert, so they link only r4dx_vision and run on a machine with no GPU and no
// ROCm runtime): CHECK, a golden-dir lookup and raw binary readers for the files
// tools/reference/gemma/vision_*_golden.py write.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace gemma_test {

inline int g_failures = 0;

#define GCHECK(cond)                                                              \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++gemma_test::g_failures;                                                   \
    }                                                                             \
  } while (0)

constexpr int kSkip = 77;

// $R4DX_GEMMA_GOLDEN_DIR, else the compile-time default (tools/reference/golden_out/gemma).
inline std::string GoldenDir() {
  if (const char* e = std::getenv("R4DX_GEMMA_GOLDEN_DIR"); e != nullptr && *e != '\0') return e;
#ifdef R4DX_GEMMA_GOLDEN_DIR
  return R4DX_GEMMA_GOLDEN_DIR;
#else
  return "tools/reference/golden_out/gemma";
#endif
}

inline bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

inline int Skip(const std::string& what, const std::string& hint) {
  std::fprintf(stderr, "[SKIP] %s not found; generate it with: %s\n", what.c_str(), hint.c_str());
  return kSkip;
}

}  // namespace gemma_test
