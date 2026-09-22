// tests/model/test_container_path.h -- where the tests (and the tests/**/tool_* diagnostics) find
// the containers they open, and what they do when one cannot be opened. In its own header, not
// test_common.h, so the tests that cannot include test_common.h (test_dflash_draft.cpp: both it and
// tests/kernels/npy_fixture.hpp define r4dx_test::kSkipReturnCode) and tests/vision (its own
// vision_test_common.h) can still use it.
//
// WHY THIS IS GROUP-AWARE. A container and the binaries that read it are a matched pair: the w4a16
// group a container was PACKED with (__metadata__.quant.w4a16.group) must equal the group this
// build's r4d_gemm_w4a16_nt_m64 was COMPILED with (the R4DX_W4A16_GROUP CMake option), or
// Container::Load / DflashDraftWeights::Open refuse it (CheckW4a16Group, src/model/quant_linear.h).
// The fixed test containers therefore exist once per group -- the historical group-128 copies at
// D:\models\r4dx\<name>, group-64 copies under the same basenames in D:\models\r4dx\g64\ -- and a
// test has to open the copy that matches the binary it is running in.
//
// The group is read at RUN time from r4d_gemm_w4a16_nt_m64_group(), not from a compile
// definition: that is the exact number CheckW4a16Group compares every container against, so the
// directory chosen here cannot disagree with the loader that will judge it (a -D on the test target
// could drift from the kernel's own -DR4D_GEMM_W4_GROUP if a target were ever built without it). It
// is a host function with no device requirement, and every target including this header already
// links r4d_core.
//
// RESOLUTION ORDER -- ContainerPath(default_path), for a fixed test container:
//   1. R4DX_TEST_CONTAINER_DIR, if set and non-empty: <that dir>/<basename of default_path>.
//   2. otherwise the build group's own directory, GroupContainerDir():
//        group 128 -> default_path unchanged (D:/models/r4dx/<name>, the historical copies);
//        any other -> D:/models/r4dx/g<group>/<name> (group 64 -> D:/models/r4dx/g64/<name>).
// So `ctest` needs NO environment variable on either the default (group-64) build or the
// win-hip-g128 build; R4DX_TEST_CONTAINER_DIR is only for pointing the fixed test containers
// somewhere else (it wins over the group, so pointing it at copies of the wrong group makes the
// affected tests FAIL on the loader's group guard -- deliberately, not silently).
//
// ProductionTargetPath() / ProductionDrafterPath() -- the real 64-layer container and its w4a16
// DFlash2 drafter, for test_dflash_e2e, test_vision_tower and the tool_* diagnostics' defaults. The
// production pair is a different FILE per group, not a copy in another directory:
//   group 64  -> qwen38-27b-v6.r4dx + qwen38-27b-dflash2-w4a16-g64.r4dx
//   group 128 -> qwen38-27b-v3.r4dx + qwen38-27b-dflash2-w4a16.r4dx
// both in D:/models/r4dx. R4DX_TEST_CONTAINER_DIR does NOT apply to them: its job is to point the
// FIXED test containers at a set of same-basename copies, and the production pair is not copied
// anywhere -- it is already a different file per group. (Honouring it here would turn the old
// manual recipe, R4DX_TEST_CONTAINER_DIR=D:\models\r4dx\g64, into a silent SKIP of every
// real-container test, since no 64-layer container lives in g64\.) Tools take --model instead.
//
// WHEN A CONTAINER CANNOT BE OPENED. A MISSING file is a SKIP: every caller checks
// FileExists(path) first and returns SkipMissing(path) (exit 77 = CTest SKIPPED). A file that is
// PRESENT but refused by the loader is a FAIL carrying the loader's own message: the loader throws,
// and RunGuardedMain below turns an exception escaping a test's body into "[FAIL] <what()>" and
// exit 1. Without it the exception leaves main(), std::terminate calls abort(), and the MSVC CRT's
// abort() ends the process with __fastfail -- ctest then reports exit 0xc0000409
// (STATUS_STACK_BUFFER_OVERRUN) and the message is lost.
//
// Every path returned is a const char* that lives for the process (a literal, or a std::deque
// element -- deque element addresses never move), so the call sites stay `const char* kContainerPath = ...` / `#define ...` exactly as they were.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <string>
#include <utility>

#include "r4d.h"  // r4d_gemm_w4a16_nt_m64_group()

namespace r4dx_test {

// Where every container this repo's tests and tools default to lives on this machine.
inline constexpr const char* kModelRoot = "D:/models/r4dx";

// The w4a16 group this binary's GEMM kernel reads -- the R4DX_W4A16_GROUP it was built with.
inline int BuildW4a16Group() { return r4d_gemm_w4a16_nt_m64_group(); }

namespace detail {

inline const char* Own(std::string s) {
  static std::deque<std::string> owned;  // stable element addresses, unlike std::vector
  owned.push_back(std::move(s));
  return owned.back().c_str();
}

// R4DX_TEST_CONTAINER_DIR, or "" when unset/empty.
inline std::string OverrideDir() {
#ifdef _MSC_VER
  // getenv is deprecated under the MSVC CRT headers clang-cl uses; _dupenv_s is the sanctioned
  // spelling and is what avoids a -Wdeprecated-declarations warning in every TU that includes this.
  char* dir = nullptr;
  size_t dir_len = 0;
  std::string out;
  if (_dupenv_s(&dir, &dir_len, "R4DX_TEST_CONTAINER_DIR") == 0 && dir != nullptr) out = dir;
  std::free(dir);
  return out;
#else
  const char* dir = std::getenv("R4DX_TEST_CONTAINER_DIR");
  return dir != nullptr ? std::string(dir) : std::string();
#endif
}

inline std::string Basename(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

inline std::string Join(std::string dir, const std::string& name) {
  if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir.push_back('/');
  return dir + name;
}

}  // namespace detail

// The directory holding the fixed test containers packed at `group`.
inline std::string GroupContainerDir(int group) {
  if (group == 128) return kModelRoot;
  return std::string(kModelRoot) + "/g" + std::to_string(group);
}

// A fixed test container, resolved as described in this file's header comment.
inline const char* ContainerPath(const char* default_path) {
  const std::string override_dir = detail::OverrideDir();
  if (!override_dir.empty()) return detail::Own(detail::Join(override_dir, detail::Basename(default_path)));
  const int group = BuildW4a16Group();
  if (group == 128) return default_path;
  return detail::Own(detail::Join(GroupContainerDir(group), detail::Basename(default_path)));
}

// The real 64-layer production container matching this build's w4a16 group.
inline const char* ProductionTargetPath() {
  return BuildW4a16Group() == 128 ? "D:/models/r4dx/qwen38-27b-v3.r4dx"
                                  : "D:/models/r4dx/qwen38-27b-v6.r4dx";
}

// ProductionTargetPath()'s w4a16 DFlash2 drafter, packed at the same group.
inline const char* ProductionDrafterPath() {
  return BuildW4a16Group() == 128 ? "D:/models/r4dx/qwen38-27b-dflash2-w4a16.r4dx"
                                  : "D:/models/r4dx/qwen38-27b-dflash2-w4a16-g64.r4dx";
}

// Runs a test's body, turning an exception that would otherwise escape main() into a FAIL that
// prints its message (see "WHEN A CONTAINER CANNOT BE OPENED" above). Usage:
//   int main() { return r4dx_test::RunGuardedMain("test_x", [] { return RunTest(); }); }
template <class Body>
int RunGuardedMain(const char* test_name, Body&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s: uncaught exception: %s\n", test_name, e.what());
  } catch (...) {
    std::fprintf(stderr, "[FAIL] %s: uncaught non-std exception\n", test_name);
  }
  return 1;
}

}  // namespace r4dx_test
