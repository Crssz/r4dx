// tests/kernels/test_attn_prefill_gqa2.cpp -- r4d_attn_prefill_h256_gqa2_fp8kv, libr4d's sliding-window
// prefill (docs/gemma4-plan.md 3.3, task M1-32), against the stage-0 reference kernel on the SAME fp8
// cache bytes (tests/kernels/gemma_attn_fixture.hpp). Gemma 4's sliding geometry: 16 q / 8 KV heads,
// head_dim 256, scale 1.0.
//   * window 1024 on a 1536-token RING (the shared block table T[i] = i % 96) at contexts that wrap it
//     (3000, 4000), chunk sizes 17 / 64 / 100 / 256 (the tail block of a 64-row workgroup is partial);
//   * small windows around the 48-key tile (1, 17, 47, 48, 49, 100) and a window longer than the context;
//   * the first-wrap edge (ctx 1537), a contiguous (non-ring) cache, and a chunk whose rows are all of a
//     short sequence;
//   * klimit_ext: an image block (rows 10..50) that sees keys up to its own last row;
//   * argument refusals.
// GPU test (ctest sets HIP_VISIBLE_DEVICES=1).
#include "gemma_attn_fixture.hpp"

using namespace gemma_attn;

namespace {
int g_fail = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%-96s %s\n", what.c_str(), cond ? "ok" : "FAIL");
  if (!cond) ++g_fail;
}

void Run(const Shape& s, std::mt19937& rng) {
  Fixture f(s, rng);
  const std::vector<uint16_t> ref = f.Reference();
  R4DArgsW a = f.Args(f.OutK(), f.Q(), s.q_len, s.ctx, f.Seqused());
  f.ZeroOutK();
  const int rc = r4d_attn_prefill_h256_gqa2_fp8kv(&a, nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  char name[160];
  std::snprintf(name, sizeof name, "prefill ctx %d q_len %d W %d %s%s", s.ctx, s.q_len, s.window,
                s.ring ? "ring" : "contiguous", s.ext ? " +klimit_ext" : "");
  if (rc != 0) { Check(false, std::string(name) + ": returned " + std::to_string(rc)); return; }
  const Err e = Compare(f.OutKHost(), ref, 0, s.q_len);
  std::printf("    norm_rel %.3e max_abs %.3e (max |ref| %.3e)\n", e.norm_rel, e.max_abs, e.max_ref);
  Check(Within(e), std::string(name) + ": matches the reference kernel");
}
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(61);

  const Shape shapes[] = {
      // window 1024 on the ring, contexts that wrap it
      {3000, 64, 1024, true, false},  {3000, 100, 1024, true, false}, {3000, 256, 1024, true, false},
      {4000, 17, 1024, true, false},  {1537, 40, 1024, true, false},  {1600, 1, 1024, true, false},
      // short sequences / chunks that are the whole sequence
      {700, 300, 1024, true, false},  {200, 200, 1024, true, false},
      // small windows around the tile size and the 64-row CTA
      {2000, 128, 1, true, false},    {2000, 128, 17, true, false},   {2000, 128, 47, true, false},
      {2000, 128, 48, true, false},   {2000, 128, 49, true, false},   {2000, 128, 100, true, false},
      // contiguous (non-ring) cache
      {1500, 64, 1024, false, false}, {1500, 64, 300, false, false},
      // a bidirectional image block inside the chunk
      {2500, 128, 1024, true, true},  {900, 100, 1024, true, true},
  };
  for (const Shape& s : shapes) Run(s, rng);

  // Refusals (a negative code, nothing launched).
  {
    Shape s{300, 20, 1024, true, false};
    Fixture f(s, rng);
    R4DArgsW a = f.Args(f.OutK(), f.Q(), s.q_len, s.ctx, f.Seqused());
    R4DArgsW b = a; b.window = 0;
    Check(r4d_attn_prefill_h256_gqa2_fp8kv(&b, nullptr) == -6, "window 0 refused (-6)");
    b = a; b.head_dim = 128;
    Check(r4d_attn_prefill_h256_gqa2_fp8kv(&b, nullptr) == -1, "head_dim 128 refused (-1)");
    b = a; b.q_heads = 24;   // gqa 3
    Check(r4d_attn_prefill_h256_gqa2_fp8kv(&b, nullptr) == -2, "gqa 3 refused (-2)");
    b = a; b.block_size = 32;
    Check(r4d_attn_prefill_h256_gqa2_fp8kv(&b, nullptr) == -1, "block 32 refused (-1)");
  }

  // The geometry registry lists both families.
  {
    bool g6 = false, g2 = false;
    for (int i = 0; i < r4d_attn_geom_count(); ++i) {
      const R4DAttnGeom* g = r4d_attn_geom_at(i);
      g6 = g6 || (g->gqa == 6 && !g->windowed);
      g2 = g2 || (g->gqa == 2 && g->windowed && g->head_dim == 256 && g->block_size == 16);
    }
    Check(g6 && g2, "r4d_attn_geom lists the gqa6 and the windowed gqa2 geometry");
  }

  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail ? 1 : 0;
}
