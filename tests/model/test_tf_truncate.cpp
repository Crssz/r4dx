// CPU-only check of r4dx_tf::TruncateSegment / TokenIdsSha256 (tests/model/teacher_forced.h): the
// truncation behind tool_teacher_forced_logprobs --max-tokens N must hash exactly like kl_report.py's
// `token_ids[:N]` + token_ids_sha256 (--raw-max-tokens N). The digests below were produced by
// hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(). No GPU, no container.
#include <cstdio>
#include <stdexcept>

#include "teacher_forced.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
  const char* kSha123 = "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4";
  const char* kSha56 = "2f9cf80b937f44b41379ae3765c65668e5e96241d19d2088e76d72d18ea324b2";
  CHECK(r4dx_tf::TokenIdsSha256({1, 2, 3}) == kSha123);
  CHECK(r4dx_tf::TokenIdsSha256({5, 6}) == kSha56);

  r4dx_tf::Segment s{"x", {1, 2, 3, 4, 5}};
  r4dx_tf::TruncateSegment(s, 3);
  CHECK(s.token_ids.size() == 3 && r4dx_tf::TokenIdsSha256(s.token_ids) == kSha123);

  r4dx_tf::Segment t{"y", {5, 6, 7}};
  r4dx_tf::TruncateSegment(t, 0);    // 0: off
  CHECK(t.token_ids.size() == 3);
  r4dx_tf::TruncateSegment(t, 99);   // longer than the segment: unchanged (python slice semantics)
  CHECK(t.token_ids.size() == 3);
  r4dx_tf::TruncateSegment(t, 2);
  CHECK(r4dx_tf::TokenIdsSha256(t.token_ids) == kSha56);

  bool threw = false;
  try { r4dx_tf::Segment u{"z", {1, 2, 3}}; r4dx_tf::TruncateSegment(u, 1); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);

  if (fails == 0) std::printf("[PASS] test_tf_truncate\n");
  return fails == 0 ? 0 : 1;
}
