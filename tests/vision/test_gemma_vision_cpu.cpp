// tests/vision/test_gemma_vision_cpu.cpp -- the golden-free CPU checks of src/vision/gemma_vision.{h,cpp}:
// target-size arithmetic, prompt expansion, the image-block finder, the prefill chunk planner, klimit_ext, the
// dense mask oracle (self-consistency), and, when tools/reference/gemma/vision_masks_golden.py's output is
// present, the dense mask against transformers' own create_masks_for_generate for 8 layouts. Pure host code:
// links r4dx_vision only.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gemma_test_util.h"
#include "gemma_vision.h"
#include "image_prompt.h"

using namespace r4dx::vision;

namespace {

void TestTargetSize() {
  GemmaImageConfig cfg;  // 280
  struct Row { int64_t h, w, th, tw; int mst; };
  // Values produced by transformers' get_aspect_ratio_preserving_size (vision_preproc_golden.py's manifest).
  const Row rows[] = {{48, 48, 768, 768, 280},    {480, 672, 672, 912, 280}, {300, 777, 480, 1248, 280},
                      {901, 233, 1536, 384, 280}, {1080, 1920, 576, 1056, 280}, {64, 2000, 96, 4464, 280},
                      {100, 130, 672, 912, 280},  {512, 512, 384, 384, 70},  {640, 480, 624, 480, 140},
                      {1000, 1000, 1104, 1104, 560}};
  for (const Row& r : rows) {
    cfg.max_soft_tokens = r.mst;
    int64_t th = 0, tw = 0;
    GemmaTargetSize(r.h, r.w, cfg, &th, &tw);
    GCHECK(th == r.th && tw == r.tw);
    GCHECK(th % 48 == 0 && tw % 48 == 0);
    GCHECK((th / 48) * (tw / 48) <= r.mst);
  }
  // A 1 x 100000 strip: height rounds to 0, width does not -> side x min(floor(w/h)*48, max_side).
  cfg.max_soft_tokens = 280;
  int64_t th = 0, tw = 0;
  GemmaTargetSize(10, 100000, cfg, &th, &tw);
  GCHECK(th == 48 && tw == 280 * 48);  // max_side_length = (2520 / 9) * 48
  bool threw = false;
  try {
    CheckGemmaSoftTokens(100);
  } catch (const std::exception&) {
    threw = true;
  }
  GCHECK(threw);
}

void TestExpansion() {
  const int32_t B = 2, T1 = 100, T2 = 101;
  const std::vector<int32_t> raw = {B, T1, kGemmaImageTokenId, T2, kGemmaImageTokenId, T1};
  const GemmaExpandedPrompt e = ExpandGemmaImagePlaceholders(raw, {3, 2});
  const std::vector<int32_t> want = {B,  T1, kGemmaBoiTokenId, kGemmaImageTokenId, kGemmaImageTokenId, kGemmaImageTokenId,
                                     kGemmaEoiTokenId, T2, kGemmaBoiTokenId, kGemmaImageTokenId, kGemmaImageTokenId,
                                     kGemmaEoiTokenId, T1};
  GCHECK(e.tokens == want);
  GCHECK(e.spans.size() == 2 && e.spans[0].offset == 3 && e.spans[0].tokens == 3 && e.spans[1].offset == 9 &&
         e.spans[1].tokens == 2);
  bool threw = false;
  try {
    ExpandGemmaImagePlaceholders(raw, {3});
  } catch (const std::exception&) {
    threw = true;
  }
  GCHECK(threw);
  threw = false;
  try {
    ExpandGemmaImagePlaceholders(raw, {1, 2, 3});
  } catch (const std::exception&) {
    threw = true;
  }
  GCHECK(threw);
  // image_prompt.h's ExpandImagePlaceholders (what the server calls) with the boi / eoi wrapper agrees, with the
  // Gemma grid convention: merge_size 1, grid in merged cells.
  {
    std::vector<ImagePlaceholderSpan> in(2);
    in[0].grid = GridThw{1, 1, 3};
    in[1].grid = GridThw{1, 2, 1};
    const ExpandedImagePrompt q = ExpandImagePlaceholders(raw, kGemmaImageTokenId, in, 1, kGemmaBoiTokenId, kGemmaEoiTokenId);
    GCHECK(q.tokens == want);
    GCHECK(q.spans.size() == 2 && q.spans[0].offset == 3 && q.spans[0].tokens == 3 && q.spans[1].offset == 9 &&
           q.spans[1].tokens == 2);
    // Qwen default (no wrapper) is unchanged.
    const ExpandedImagePrompt plain = ExpandImagePlaceholders(raw, kGemmaImageTokenId, in, 1);
    GCHECK(plain.tokens.size() == raw.size() - 2 + 3 + 2);
  }
  // The blocks found in the expansion are the spans (boi / eoi are not part of a block).
  const std::vector<ImageBlock> blocks = FindImageBlocks(e.tokens, kGemmaImageTokenId, 0);
  GCHECK(blocks.size() == 2 && blocks[0].start == 3 && blocks[0].end == 6 && blocks[1].start == 9 && blocks[1].end == 11);
  // Absolute base offset.
  const std::vector<ImageBlock> shifted = FindImageBlocks(e.tokens, kGemmaImageTokenId, 100);
  GCHECK(shifted[0].start == 103 && shifted[1].end == 111);
}

// A chunk plan must cover [0, total) exactly, never exceed max_rows unless it is one whole image block, and
// never cut a block.
void CheckPlan(int64_t total, const std::vector<ImageBlock>& blocks, int64_t max_rows, int64_t max_block) {
  const std::vector<PrefillChunk> chunks = PlanPrefillChunks(total, blocks, max_rows, max_block);
  int64_t pos = 0;
  for (const PrefillChunk& c : chunks) {
    GCHECK(c.start == pos && c.len >= 1);
    GCHECK(c.len <= max_block);
    bool whole_block = false;
    for (const ImageBlock& b : blocks) {
      if (b.start >= c.start && b.end <= c.start + c.len && c.len > max_rows) whole_block = true;
      // no cut
      const int64_t cs = c.start, ce = c.start + c.len;
      const bool cut = (b.start < ce && b.end > ce && b.start >= cs) || (b.start < cs && b.end > cs);
      GCHECK(!cut);
    }
    if (c.len > max_rows) GCHECK(whole_block);
    pos += c.len;
  }
  GCHECK(pos == total);
}

void TestPlanner() {
  CheckPlan(10, {}, 4, 288);
  CheckPlan(1000, {}, 256, 288);
  CheckPlan(300, {{10, 290}}, 256, 288);       // 280-token block after 10 text rows: chunk 0 = 10 rows, chunk 1 = block
  CheckPlan(600, {{0, 280}}, 256, 288);        // block at the start: one 280-row chunk
  CheckPlan(900, {{3, 283}, {290, 570}}, 256, 288);
  CheckPlan(900, {{250, 530}}, 256, 288);      // straddles the 256 cut: chunk 0 ends at 250
  CheckPlan(900, {{256, 536}}, 256, 288);      // starts exactly at the cut
  CheckPlan(900, {{200, 256}}, 256, 288);      // ends exactly at the cut
  CheckPlan(40, {{5, 6}}, 256, 288);           // one-token image
  // exact numbers for the common shape
  const std::vector<PrefillChunk> c = PlanPrefillChunks(300, {{10, 290}}, 256, 288);
  GCHECK(c.size() == 3 && c[0].start == 0 && c[0].len == 10 && c[1].start == 10 && c[1].len == 280 &&
         c[2].start == 290 && c[2].len == 10);
  bool threw = false;
  try {
    PlanPrefillChunks(700, {{0, 400}}, 256, 288);
  } catch (const std::exception&) {
    threw = true;
  }
  GCHECK(threw);
}

void TestKlimit() {
  const std::vector<ImageBlock> blocks = {{4, 8}, {12, 14}};
  // chunk covering [2, 14): absolute positions
  const std::vector<int32_t> ext = BuildKlimitExt(2, 12, blocks);
  const std::vector<int32_t> want = {-1, -1, 7, 7, 7, 7, -1, -1, -1, -1, 13, 13};
  GCHECK(ext == want);
  // The effective bound max(qpos, ext) is non-decreasing (the kernel's contract).
  int32_t prev = -1;
  for (size_t i = 0; i < ext.size(); ++i) {
    const int32_t q = static_cast<int32_t>(2 + i);
    const int32_t eff = ext[i] > q ? ext[i] : q;
    GCHECK(eff >= prev);
    prev = eff;
  }
  // A chunk that starts inside a block's tail never happens (the planner keeps blocks whole) but must not crash.
  const std::vector<int32_t> tail = BuildKlimitExt(6, 4, blocks);
  GCHECK(tail[0] == 7 && tail[1] == 7 && tail[2] == -1 && tail[3] == -1);
}

// klimit_ext + window + causal == the dense sliding mask, for the semantics the kernels implement
// (r4d_attn_window.h: lower bound qpos - W + 1, upper bound max(qpos, ext)).
void TestKernelContractMatchesDense() {
  const std::vector<ImageBlock> blocks = {{3, 9}, {12, 15}};
  const int64_t T = 20, W = 4;
  const std::vector<uint8_t> dense = BuildDenseMask(T, W, true, blocks);
  const std::vector<int32_t> ext = BuildKlimitExt(0, T, blocks);
  for (int64_t q = 0; q < T; ++q) {
    const int64_t lo = q - W + 1 > 0 ? q - W + 1 : 0;
    const int64_t hi = ext[static_cast<size_t>(q)] > q ? ext[static_cast<size_t>(q)] : q;
    for (int64_t k = 0; k < T; ++k) {
      const bool kernel = k >= lo && k <= hi;
      GCHECK(kernel == (dense[static_cast<size_t>(q * T + k)] != 0));
    }
  }
  // Full layers / decode: pure causal.
  const std::vector<uint8_t> full = BuildDenseMask(T, 0, false, blocks);
  for (int64_t q = 0; q < T; ++q)
    for (int64_t k = 0; k < T; ++k) GCHECK((full[static_cast<size_t>(q * T + k)] != 0) == (k <= q));
}

// The doc table of docs/gemma4-semantics.md section 2 (window 4, image at 2..5, 8 tokens), generate() path.
void TestDocTable() {
  const std::vector<ImageBlock> blocks = {{2, 6}};
  const char* sliding[8] = {"10000000", "11000000", "11111100", "11111100", "01111100", "00111100", "00011110", "00001111"};
  const char* full[8] = {"10000000", "11000000", "11100000", "11110000", "11111000", "11111100", "11111110", "11111111"};
  const std::vector<uint8_t> s = BuildDenseMask(8, 4, true, blocks);
  const std::vector<uint8_t> f = BuildDenseMask(8, 0, false, blocks);
  for (int q = 0; q < 8; ++q) {
    for (int k = 0; k < 8; ++k) {
      GCHECK((s[static_cast<size_t>(q * 8 + k)] != 0) == (sliding[q][k] == '1'));
      GCHECK((f[static_cast<size_t>(q * 8 + k)] != 0) == (full[q][k] == '1'));
    }
  }
}

int TestHfMasks() {
  std::ifstream f(gemma_test::GoldenDir() + "/vision_masks.txt");
  if (!f) {
    std::fprintf(stderr, "[note] vision_masks.txt not found: HF mask comparison skipped (generate with "
                         "tools/reference/gemma/vision_masks_golden.py)\n");
    return 0;
  }
  std::string tag;
  int cases = 0;
  while (f >> tag) {
    if (tag != "case") {
      GCHECK(false);
      return 1;
    }
    std::string name;
    int64_t T = 0, W = 0;
    f >> name >> T >> W;
    f >> tag;  // "types"
    std::vector<int32_t> tokens(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) {
      int t = 0;
      f >> t;
      tokens[static_cast<size_t>(i)] = t == 1 ? kGemmaImageTokenId : 7;
    }
    const std::vector<ImageBlock> blocks = FindImageBlocks(tokens, kGemmaImageTokenId, 0);
    std::vector<uint8_t> hf_s, hf_f;
    f >> tag;  // sliding
    for (int64_t q = 0; q < T; ++q) {
      std::string row;
      f >> row;
      for (char c : row) hf_s.push_back(c == '1');
    }
    f >> tag;  // full
    for (int64_t q = 0; q < T; ++q) {
      std::string row;
      f >> row;
      for (char c : row) hf_f.push_back(c == '1');
    }
    const std::vector<uint8_t> mine_s = BuildDenseMask(T, W, true, blocks);
    const std::vector<uint8_t> mine_f = BuildDenseMask(T, 0, false, blocks);
    const bool ok_s = mine_s == hf_s, ok_f = mine_f == hf_f;
    if (!ok_s || !ok_f) std::fprintf(stderr, "  mask mismatch in case %s (sliding %d full %d)\n", name.c_str(), ok_s, ok_f);
    GCHECK(ok_s);
    GCHECK(ok_f);
    // klimit_ext + window reproduce the HF sliding mask too.
    const std::vector<int32_t> ext = BuildKlimitExt(0, T, blocks);
    for (int64_t q = 0; q < T; ++q) {
      const int64_t lo = q - W + 1 > 0 ? q - W + 1 : 0;
      const int64_t hi = ext[static_cast<size_t>(q)] > q ? ext[static_cast<size_t>(q)] : q;
      for (int64_t k = 0; k < T; ++k) GCHECK(((k >= lo && k <= hi) ? 1 : 0) == hf_s[static_cast<size_t>(q * T + k)]);
    }
    ++cases;
  }
  GCHECK(cases == 8);
  std::printf("compared %d layouts with transformers' create_masks_for_generate\n", cases);
  return 0;
}

}  // namespace

int main() {
  TestTargetSize();
  TestExpansion();
  TestPlanner();
  TestKlimit();
  TestKernelContractMatchesDense();
  TestDocTable();
  TestHfMasks();
  if (gemma_test::g_failures != 0) {
    std::fprintf(stderr, "test_gemma_vision_cpu: %d failure(s)\n", gemma_test::g_failures);
    return 1;
  }
  std::printf("test_gemma_vision_cpu: OK\n");
  return 0;
}
