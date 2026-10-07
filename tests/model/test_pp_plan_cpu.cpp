// tests/model/test_pp_plan_cpu.cpp -- CPU-only checks of src/model/pp_plan.h, the arithmetic of the
// pipeline-parallel prefill track (docs/pp-prefill.md): the KV blocks a chunk touches (what crosses the stage
// boundary), the R4DX_PP_EMULATE parser, the two-stage pipeline simulation with a bounded slot ring, the stage
// bench's CSV and the projection / go rule built on them. No HIP call, no container, always runs.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "pp_plan.h"
#include "prefill_chunk.h"

using namespace r4dx::model;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
bool Near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

void KvBlocks() {
  const auto r = [](int64_t pos, int64_t rows) { return pp::KvBlocksTouched(pos, rows, 16); };
  Check(r(0, 256).block0 == 0 && r(0, 256).block1 == 16, "256 rows at 0 = blocks [0,16)");
  Check(r(256, 256).block0 == 16 && r(256, 256).block1 == 32, "256 rows at 256 = blocks [16,32)");
  Check(r(5, 10).block0 == 0 && r(5, 10).block1 == 1, "rows 5..14 live in block 0");
  Check(r(15, 2).block0 == 0 && r(15, 2).block1 == 2, "rows 15..16 straddle blocks 0 and 1");
  Check(r(16, 16).Blocks() == 1, "one aligned block");
  Check(r(255, 1).block0 == 15 && r(255, 1).block1 == 16, "row 255 is in block 15");
  Check(r(300, 1).Blocks() == 1, "a one-row chunk touches one block");
  // A mid-block start (a warm turn) costs one block more than the aligned chunk: 17 blocks for 256 rows.
  Check(r(8, 256).Blocks() == 17, "256 rows from mid-block touch 17 blocks");
  // Property: over the real chunk grid of a call, the touched ranges cover every block of the sequence and each
  // chunk's range starts no later than the previous chunk's end (no row is left uncopied between chunks).
  for (int64_t start : {int64_t{0}, int64_t{300}, int64_t{8145}}) {
    for (int n : {1, 63, 64, 65, 255, 256, 257, 511, 8145}) {
      int64_t pos = start, last_block1 = start / 16;
      bool covered = true;
      for (int64_t off = 0; off < n;) {
        const int64_t rows = PrefillNextChunk(n - off, kPrefillChunkWide);
        const pp::KvBlockRange b = pp::KvBlocksTouched(pos, rows, 16);
        if (b.block0 > last_block1) covered = false;
        last_block1 = std::max(last_block1, b.block1);
        pos += rows;
        off += rows;
      }
      if (last_block1 != (start + n + 15) / 16) covered = false;
      Check(covered, "the chunk grid's touched KV blocks cover [start, start + n) without a gap");
    }
  }
}

void SplitAndEnv() {
  Check(pp::ValidSplit(1, 64) && pp::ValidSplit(33, 64) && pp::ValidSplit(63, 64), "valid splits");
  Check(!pp::ValidSplit(0, 64) && !pp::ValidSplit(64, 64) && !pp::ValidSplit(-1, 64), "invalid splits");
  Check(pp::ParseEmulateSplit(nullptr) == 0 && pp::ParseEmulateSplit("") == 0, "unset / empty = off");
  Check(pp::ParseEmulateSplit("0") == 0 && pp::ParseEmulateSplit("off") == 0 && pp::ParseEmulateSplit("false") == 0,
        "0 / off / false = off");
  Check(pp::ParseEmulateSplit("33") == 33 && pp::ParseEmulateSplit("1") == 1, "a positive integer is the split");
  Check(pp::ParseEmulateSplit("abc") == -1 && pp::ParseEmulateSplit("-3") == -1 && pp::ParseEmulateSplit("3.5") == -1 &&
            pp::ParseEmulateSplit("1234567") == -1,
        "anything else is refused");
}

void Pipeline() {
  const auto uniform = [](size_t n, double v) { return std::vector<double>(n, v); };
  // Equal stages, a roomy ring: one fill then lockstep -- (N + 1) T.
  {
    const pp::PipelineResult r = pp::SimulatePipeline(uniform(32, 56.0), uniform(32, 56.0), 3, 0.0);
    Check(Near(r.total_ms, 33 * 56.0), "equal stages: (N + 1) T");
    Check(Near(r.a_busy_ms, 32 * 56.0) && Near(r.b_busy_ms, 32 * 56.0), "busy times are the sums");
  }
  // A slower than B: B starves, the call is A's N chunks plus B's last one.
  {
    const pp::PipelineResult r = pp::SimulatePipeline(uniform(20, 60.0), uniform(20, 50.0), 3, 0.0);
    Check(Near(r.total_ms, 20 * 60.0 + 50.0), "A-bound: N a + b");
    Check(r.a_stall_ms == 0.0 && r.b_stall_ms > 0.0, "A-bound: B waits, A never does");
  }
  // B slower than A with a ring of 3: A runs ahead three slots then stalls; the call is a + N b.
  {
    const pp::PipelineResult r = pp::SimulatePipeline(uniform(20, 50.0), uniform(20, 60.0), 3, 0.0);
    Check(Near(r.total_ms, 50.0 + 20 * 60.0), "B-bound: a + N b");
    Check(r.a_stall_ms > 0.0, "B-bound: A stalls on a full ring");
  }
  // A ring of one slot is serial: A cannot start chunk c + 1 before B finished chunk c.
  {
    const pp::PipelineResult r = pp::SimulatePipeline(uniform(10, 40.0), uniform(10, 40.0), 1, 0.0);
    Check(Near(r.total_ms, 10 * 80.0), "one slot: N (a + b)");
  }
  // Two slots are enough for equal stages (A finishes chunk c + 1 while B runs chunk c).
  {
    const pp::PipelineResult r = pp::SimulatePipeline(uniform(10, 40.0), uniform(10, 40.0), 2, 0.0);
    Check(Near(r.total_ms, 11 * 40.0), "two slots, equal stages: (N + 1) T");
  }
  // The tail is added once; a mismatched length uses the shorter; nothing gives zero.
  Check(Near(pp::SimulatePipeline(uniform(4, 10.0), uniform(4, 10.0), 3, 7.0).total_ms, 5 * 10.0 + 7.0), "tail added");
  Check(pp::SimulatePipeline({}, {}, 3, 5.0).total_ms == 0.0, "no chunks");
  Check(Near(pp::SimulatePipeline(uniform(5, 10.0), uniform(3, 10.0), 3, 0.0).total_ms, 4 * 10.0), "shorter side");
  // Varying costs: a deeper-context chunk is slower on both stages; hand-computed three-chunk case.
  //   A: 10, 20, 30 ; B: 15, 15, 15 ; ring 3.
  //   a_done = 10, 30, 60 ; b: start 10 -> 25 ; start 30 -> 45 ; start 60 -> 75.
  {
    const pp::PipelineResult r = pp::SimulatePipeline({10, 20, 30}, {15, 15, 15}, 3, 0.0);
    Check(Near(r.total_ms, 75.0), "hand-computed three-chunk pipeline");
  }
  // Closed form vs simulation at the design's operating point (docs/pp-prefill.md 6.1): half the monolithic
  // time per stage, f = 1, h = 0, s = 1 -- (1 + 1/N) T / 2.
  {
    const int n = 32;
    const double t_total = 3630.0;
    const double stage = t_total / 2 / n;
    const pp::PipelineResult r = pp::SimulatePipeline(uniform(n, stage), uniform(n, stage), 3, 0.0);
    Check(Near(r.total_ms, pp::DesignFormulaMs(t_total, n, 1.0, 0.0, 0.0, 1.0), 1e-9), "simulation == the design's closed form");
    Check(Near(pp::DesignFormulaMs(3630.0, 32, 1.0, 0.0, 0.0, 1.0), 1871.71875, 1e-9), "8k ideal is ~1.87 s (design table)");
  }
}

void Csv() {
  pp::BenchFile f;
  std::string err;
  pp::Sample s0;
  s0.pos = 0;
  s0.rows = 256;
  s0.a_ms = 28.5;
  s0.d2h_ms = 1.25;
  s0.h2d_ms = 0.75;
  s0.b_ms = 27.0;
  s0.epilogue_ms = 0.5;
  s0.mtp_ms = 0.125;
  s0.inject_ms = 2.5;
  s0.hop_bytes = 12345678;
  pp::Sample s1 = s0;
  s1.pos = 256;
  s1.a_ms = 29.0;
  const std::string text = "# a comment\n" + pp::FormatMonoLine("mono-8k", 3627.5) + "\n" +
                           pp::FormatChunkLine("emu33-8k", 0, s0) + "\r\n" + pp::FormatChunkLine("emu33-8k", 1, s1) + "\n";
  Check(pp::ParseBenchFile(text, &f, &err), "a well-formed file parses");
  Check(f.mono_ms.count("mono-8k") == 1 && Near(f.mono_ms["mono-8k"], 3627.5), "mono line");
  Check(f.chunks["emu33-8k"].size() == 2 && f.chunks["emu33-8k"][1].pos == 256 && Near(f.chunks["emu33-8k"][1].a_ms, 29.0) &&
            f.chunks["emu33-8k"][0].hop_bytes == 12345678 && Near(f.chunks["emu33-8k"][0].inject_ms, 2.5),
        "chunk lines round-trip");
  pp::BenchFile bad;
  Check(!pp::ParseBenchFile("chunk,x,1,0,256,1,1,1,1,1,1,1,1\n", &bad, &err), "an out-of-order chunk index is refused");
  Check(!pp::ParseBenchFile("chunk,x,0,0,256,abc,1,1,1,1,1,1,1\n", &bad, &err), "a bad number is refused");
  Check(!pp::ParseBenchFile("nonsense,1\n", &bad, &err), "an unknown record is refused");
}

void Hop() {
  std::vector<pp::HopRow> rows;
  std::string err;
  const std::string text =
      "# comment\n"
      "hop,d0to1,12.75MiB,13369344,d2h,1.2000,11.141\n"
      "hop,d0to1,12.75MiB,13369344,h2d,1.0000,13.369\n"
      "hop,d0to1,12.75MiB,13369344,pipe_d2h,2.0000,6.685\n"
      "hop,d0to1,2.5MiB,2621440,d2h,0.5000,5.243\n"
      "hop,d1to0,12.75MiB,13369344,d2h,1.5000,8.913\n";
  Check(pp::ParseHopFile(text, &rows, &err) && rows.size() == 5, "a hop file parses");
  Check(Near(pp::MinGbps(rows, "d0to1", {"d2h", "h2d"}, "12.75MiB"), 11.141), "the slowest plain leg of the 12.75 MiB size");
  Check(Near(pp::MinGbps(rows, "d0to1", {"d2h", "h2d", "pipe_d2h"}, "12.75MiB"), 6.685), "the pipe leg counts when asked");
  Check(Near(pp::MinGbps(rows, "d0to1", {"d2h"}), 5.243), "no size filter: every size");
  Check(pp::MinGbps(rows, "d1to0", {"h2d"}) == 0.0, "no row, no bandwidth");
  std::vector<pp::HopRow> bad;
  Check(!pp::ParseHopFile("hop,d0to1,x\n", &bad, &err), "a short hop record is refused");
  Check(!pp::ParseHopFile("hop,d0to1,a,xx,d2h,1,1\n", &bad, &err), "a bad number is refused");
}

void Projection() {
  // 8 chunks, stage A 50 ms + 1 ms export, stage B 49 ms + 1 ms import + 1 ms epilogue; mono 8 * 100 ms.
  std::vector<pp::Sample> a(8), b(8);
  for (int c = 0; c < 8; ++c) {
    a[c].pos = b[c].pos = c * 256;
    a[c].rows = b[c].rows = 256;
    a[c].a_ms = 50.0;
    a[c].d2h_ms = 1.0;
    b[c].b_ms = 49.0;
    b[c].h2d_ms = 1.0;
    b[c].epilogue_ms = 1.0;
  }
  pp::ProjectionParams p;
  p.fixed_ms = 0.0;
  const pp::Projection r = pp::Project(a, b, 800.0, p);
  Check(r.ok, "projection ok");
  // A costs 51 per chunk, B 51: lockstep, (N + 1) * 51.
  Check(Near(r.pp_ms, 9 * 51.0), "projected TTFT");
  Check(Near(r.speedup, 800.0 / (9 * 51.0)), "speedup = mono / pp");
  Check(Near(r.a_over_b, 1.0), "balanced stages");
  // Doubling the hop cost (hop_scale 2) makes both stages 52 per chunk.
  p.d2h_scale = 2.0;
  p.h2d_scale = 2.0;
  Check(Near(pp::Project(a, b, 800.0, p).pp_ms, 9 * 52.0), "hop scale 2: A = 50 + 2, B = 2 + 49 + 1");
  p.d2h_scale = 3.0;  // A alone slower now: 50 + 3 = 53 against B's 52 -> A-bound, N a + b
  Check(Near(pp::Project(a, b, 800.0, p).pp_ms, 8 * 53.0 + 52.0), "an A-bound hop: N a + b");
  p.d2h_scale = 1.0;
  p.h2d_scale = 1.0;
  // Mismatched sides and a missing mono are refused, not guessed.
  std::vector<pp::Sample> short_b(b.begin(), b.begin() + 7);
  Check(!pp::Project(a, short_b, 800.0, p).ok, "different chunk counts");
  Check(!pp::Project(a, b, 0.0, p).ok, "no mono TTFT");
  std::vector<pp::Sample> b_moved = b;
  b_moved[3].rows = 64;
  Check(!pp::Project(a, b_moved, 800.0, p).ok, "different rows on the two sides");
  // The go rule: both 8k and 32k at 1.6x; 64k does not decide.
  Check(pp::GoRule(1.60, 1.60) && !pp::GoRule(1.59, 1.9) && !pp::GoRule(1.9, 1.59) && pp::GoRule(1.8, 1.7, 1.6),
        "go rule");
}

}  // namespace

int main() {
  KvBlocks();
  SplitAndEnv();
  Pipeline();
  Csv();
  Hop();
  Projection();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_pp_plan_cpu: %d check(s) FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_pp_plan_cpu: PASS\n");
  return 0;
}
