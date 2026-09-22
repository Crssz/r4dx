// tests/model/test_container_path.h -- one helper, in its own header so the tests that cannot
// include test_common.h (test_dflash_draft.cpp: both it and tests/kernels/npy_fixture.hpp define
// r4dx_test::kSkipReturnCode) can still use it.
//
// Redirects a hard-coded test-container path to another directory when R4DX_TEST_CONTAINER_DIR is
// set (same basename, new directory); returns `default_path` unchanged otherwise, which is the
// normal case.
//
// Why this exists: the containers these tests read live at fixed D:\ paths and were packed at the
// w4a16 group of 128, which was the default until Milestone 11. A build configured with a
// different R4DX_W4A16_GROUP (see docs/build-windows.md "w4a16 group size") correctly REFUSES
// them at Container::Load -- a container and the binaries that read it are a matched pair -- so
// without an override there is no way to run these tests on such a build at all.
//
// Since Milestone 11 the DEFAULT build is group 64, so on this machine it is the default build
// that needs the override, and `win-hip-g128` that wants it unset. Group-matched copies (same
// basenames) live in D:\models\r4dx\g64\ -- docs/build-windows.md has the conversion recipe:
//
//   $env:R4DX_TEST_CONTAINER_DIR = 'D:\models\r4dx\g64'
//   & $ctest --preset win-hip
//
// Returns const char* (backed by a std::deque, so the strings live for the process and never move)
// so the call sites stay `const char* kContainerPath = ...` / `#define ...` exactly as they were.
#pragma once

#include <cstdlib>
#include <deque>
#include <string>

namespace r4dx_test {

inline const char* ContainerPath(const char* default_path) {
#ifdef _MSC_VER
  // getenv is deprecated under the MSVC CRT headers clang-cl uses; _dupenv_s is the sanctioned
  // spelling and is what avoids a -Wdeprecated-declarations warning in every TU that includes this.
  char* dir = nullptr;
  size_t dir_len = 0;
  const bool have_dir = (_dupenv_s(&dir, &dir_len, "R4DX_TEST_CONTAINER_DIR") == 0 &&
                         dir != nullptr && *dir != '\0');
  const std::string dir_str = have_dir ? std::string(dir) : std::string();
  std::free(dir);
  if (!have_dir) return default_path;
#else
  const char* dir = std::getenv("R4DX_TEST_CONTAINER_DIR");
  if (dir == nullptr || *dir == '\0') return default_path;
  const std::string dir_str(dir);
#endif
  std::string base(default_path);
  const size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);
  std::string d = dir_str;
  if (!d.empty() && d.back() != '/' && d.back() != '\\') d.push_back('/');
  static std::deque<std::string> owned;  // stable element addresses, unlike std::vector
  owned.push_back(d + base);
  return owned.back().c_str();
}

}  // namespace r4dx_test
