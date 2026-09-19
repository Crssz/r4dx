// tests/server/test_sse.cpp -- pure CPU unit test for src/server/sse.h's SSE chunk formatting.
#include <cstdio>
#include <string>

#include "nlohmann/json.hpp"
#include "sse.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                                    \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

void TestFormatSseEvent() {
  const nlohmann::json payload = {{"id", "abc"}, {"n", 1}};
  const std::string event = r4dx::server::FormatSseEvent(payload);
  CHECK(event.rfind("data: ", 0) == 0);
  CHECK(event.size() >= 2 && event.substr(event.size() - 2) == "\n\n");
  // Exactly one "data: " prefix, one trailing blank line, and the compact (no-whitespace) JSON
  // dump in between -- round-trips back to the same value.
  const std::string json_part = event.substr(6, event.size() - 6 - 2);
  CHECK(nlohmann::json::parse(json_part) == payload);
}

void TestFormatSseDone() {
  CHECK(r4dx::server::FormatSseDone() == "data: [DONE]\n\n");
}

}  // namespace

int main() {
  TestFormatSseEvent();
  TestFormatSseDone();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all sse formatting checks passed\n");
  return 0;
}
