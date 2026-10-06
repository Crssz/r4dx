// tests/model/test_container_path.h -- where the tests (and the tests/**/tool_* diagnostics) find
// the containers they open, and what they do when one cannot be opened. In its own header, not
// test_common.h, so the tests that cannot include test_common.h (test_dflash_draft.cpp: both it and
// tests/kernels/npy_fixture.hpp define r4dx_test::kSkipReturnCode) and tests/vision (its own
// vision_test_common.h) can still use it.
//
// WHY THE FIXED TEST CONTAINERS LIVE IN g64\. A container and the binaries that read it are a
// matched pair: the w4a16 group a container was PACKED with (__metadata__.quant.w4a16.group) must
// equal the group this build's r4d_gemm_w4a16_nt_m64 was COMPILED with (64), or Container::Load /
// DflashDraftWeights::Open refuse it (CheckW4a16Group, src/model/quant_linear.h). The fixed test
// containers are therefore the group-64 ones, <R4DX_MODELS_ROOT>\r4dx\g64\<name> (root default E:\models).
//
// RESOLUTION ORDER -- ContainerPath(default_path), for a fixed test container:
//   1. R4DX_TEST_CONTAINER_DIR, if set and non-empty: <that dir>/<basename of default_path>.
//   2. otherwise <R4DX_MODELS_ROOT>/r4dx/g64/<basename of default_path>. (Only the basename of
//      default_path is used, so callers pass "r4dx/<name>.r4dx" -- no root literal anywhere.)
// So `ctest` needs NO environment variable; R4DX_TEST_CONTAINER_DIR is only for pointing the fixed
// test containers somewhere else (pointing it at copies of the wrong group makes the affected tests
// FAIL on the loader's group guard -- deliberately, not silently).
//
// ProductionTargetPath() / ProductionDrafterPath() / ProductionLayoutName() -- the real 64-layer
// container, the body layout it loads with, and its w4a16 DFlash2 drafter, for test_dflash_e2e,
// test_vision_tower, test_tp_emulation / test_tp_real_vs_emulation's real-container cases and the
// tool_* diagnostics' defaults:
//   huihui-qwen38-27b-abl-trellis-mix45m.r4dx (layout "trellis": the Huihui abliterated trellis
//   mix4.5m, docs/huihui.md) + qwen38-27b-dflash2-w4a16-g64.r4dx, both in <R4DX_MODELS_ROOT>/r4dx. The
// trellis container's non-trellis tensors (heads, embeddings) are packed at w4a16 group 64. The
// previous production container (qwen38-27b-v6.r4dx, the base Qwen3.8-27B at w4a16 g64) and the base
// trellis mix4.5m were retired with the base checkpoint (docs/huihui.md "Default container").
// R4DX_TEST_CONTAINER_DIR does NOT apply to the production pair: its job is to point the FIXED test
// containers at a set of same-basename copies, and the production pair is not copied anywhere.
// (Honouring it here would turn the manual recipe, R4DX_TEST_CONTAINER_DIR=<root>\r4dx\g64, into
// a silent SKIP of every real-container test, since no 64-layer container lives in g64\.) Tools take
// --model instead.
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

#include "r4dx/models_root.h"

namespace r4dx_test {

// Where every container this repo's tests and tools default to lives on this machine.
// <R4DX_MODELS_ROOT>/r4dx (r4dx/models_root.h; the root defaults to E:\models).
inline std::string ContainerRoot() { return r4dx::ModelsPath("r4dx"); }

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

// A fixed test container, resolved as described in this file's header comment.
inline const char* ContainerPath(const char* default_path) {
  const std::string override_dir = detail::OverrideDir();
  if (!override_dir.empty()) return detail::Own(detail::Join(override_dir, detail::Basename(default_path)));
  return detail::Own(detail::Join(ContainerRoot() + "/g64", detail::Basename(default_path)));
}

// The real 64-layer production container.
inline const char* ProductionTargetPath() {
  return detail::Own(detail::Join(ContainerRoot(), "huihui-qwen38-27b-abl-trellis-mix45m.r4dx"));
}

// The body layout name (r4dx::model::LayoutFromName) ProductionTargetPath() loads with: a trellis
// container loads only with "trellis" (Container::Load refuses every other layout by name).
inline const char* ProductionLayoutName() { return "trellis"; }

// ProductionTargetPath()'s w4a16 DFlash2 drafter, packed at group 64.
inline const char* ProductionDrafterPath() {
  return detail::Own(detail::Join(ContainerRoot(), "qwen38-27b-dflash2-w4a16-g64.r4dx"));
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
