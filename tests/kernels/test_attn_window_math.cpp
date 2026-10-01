// tests/kernels/test_attn_window_math.cpp -- pure CPU (no HIP, no kernel library). Pins the sliding-
// window arithmetic of third_party/libr4d/r4d_attn_window.h (docs/gemma4-plan.md 3.3) against brute
// force over every (qpos, key) pair of a few thousand random shapes, so the GPU kernels that call it
// (the WIN variants of the libr4d prefill / decode kernels) can only be wrong in their own code:
//   * the visibility predicate, the first visible key and the visible count (min(qpos + 1, W));
//   * prefill: a CTA's tile range [first_tile, end_tile) contains every visible key of every row of
//     the CTA (also with a bidirectional klimit_ext), and is tight at both ends;
//   * the first tile's over-read below the window is at most TILE - 1 keys;
//   * decode: the split-KV segments [t0 + s*tps, ...) tile [t0, ntl) exactly, every row's visible keys
//     lie in its first r4d_win_decode_row_used segments, and that count is exactly the segments that
//     hold a key <= the row's qpos;
//   * the decode split-law bound W + q_len + 16 covers the number of window tiles at any alignment.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "r4d_attn_window.h"

namespace {

int g_fail = 0;
void Fail(const std::string& what) {
  if (++g_fail <= 20) std::printf("FAIL: %s\n", what.c_str());
}
#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) Fail(std::string(msg) + "  [" #cond "]");            \
  } while (0)

std::string Ctx(int ctx, int q_len, int W, int TILE) {
  return "ctx=" + std::to_string(ctx) + " q_len=" + std::to_string(q_len) + " W=" +
         std::to_string(W) + " TILE=" + std::to_string(TILE);
}

void TestVisibility() {
  for (int W : {1, 2, 5, 16, 1024}) {
    for (int qpos : {0, 1, 3, 15, 16, 17, 1023, 1024, 1025, 5000}) {
      int count = 0, first = -1, last = -1;
      for (int k = 0; k <= qpos + 40; ++k) {
        if (r4d_win_visible(k, qpos, W)) {
          ++count;
          if (first < 0) first = k;
          last = k;
        }
      }
      const std::string c = "W=" + std::to_string(W) + " qpos=" + std::to_string(qpos);
      CHECK(count == std::min(qpos + 1, W), c + " visible count");
      CHECK(first == r4d_win_klow(qpos, W), c + " klow is the first visible key");
      CHECK(last == qpos, c + " the query itself is visible");
    }
  }
  CHECK(r4d_win_klow(7, 0) == 0 && r4d_win_visible(0, 7, 0), "W = 0 means no window");
  CHECK(r4d_win_khigh(10, -1) == 10 && r4d_win_khigh(10, 14) == 14 && r4d_win_khigh(10, 3) == 10,
        "khigh: ext only raises the bound");
}

// A monotone ext array (end of the image block a row is in, else none), as the contract requires.
void FillExt(std::mt19937& rng, int ctx, int q_len, std::vector<int>* ext) {
  ext->assign(q_len, -1);
  int t = 0;
  while (t < q_len) {
    if (rng() % 4 == 0) {
      const int len = 2 + static_cast<int>(rng() % 40);
      const int end_t = std::min(q_len - 1, t + len - 1);
      // The block's last key is the absolute position of its last row (the image block lies inside
      // this chunk), so every row of the block sees keys up to it.
      for (int u = t; u <= end_t; ++u) (*ext)[u] = ctx - q_len + end_t;
      t = end_t + 1;
    } else {
      ++t;
    }
  }
}

void TestPrefillTiles(std::mt19937& rng) {
  for (int iter = 0; iter < 4000; ++iter) {
    const int TILE = (iter & 1) ? 48 : 16;
    const int BQ = (iter & 2) ? 64 : 21;   // 21 = a GQA-6 block, 64 = a GQA-2 block of 8 warps
    const int W = 1 + static_cast<int>(rng() % 1500);
    const int q_len = 1 + static_cast<int>(rng() % 300);
    const int ctx = q_len + static_cast<int>(rng() % 4000);
    const bool use_ext = (iter % 3) == 0;
    std::vector<int> ext;
    if (use_ext) FillExt(rng, ctx, q_len, &ext); else ext.assign(q_len, -1);

    for (int qb = 0; qb * BQ < q_len; ++qb) {
      const int r_lo = qb * BQ, r_hi = std::min(r_lo + BQ - 1, q_len - 1);
      const int qpos_lo = ctx - q_len + r_lo, qpos_hi = ctx - q_len + r_hi;
      const int ft = r4d_win_first_tile(qpos_lo, W, TILE);
      const int khigh_max = r4d_win_khigh(qpos_hi, ext[r_hi]);   // monotone ext: the last row's bound
      const int et = r4d_win_end_tile(khigh_max, ctx, TILE);
      const std::string c = Ctx(ctx, q_len, W, TILE) + " qb=" + std::to_string(qb);

      int min_vis = ctx, max_vis = -1;
      for (int r = r_lo; r <= r_hi; ++r) {
        const int qpos = ctx - q_len + r;
        const int hi = std::min(r4d_win_khigh(qpos, ext[r]), ctx - 1);
        const int lo = r4d_win_klow(qpos, W);
        CHECK(lo / TILE >= ft, c + " row's first key is below the first tile");
        CHECK(hi / TILE < et, c + " row's last key is past the end tile");
        min_vis = std::min(min_vis, lo);
        max_vis = std::max(max_vis, hi);
      }
      CHECK(min_vis / TILE == ft, c + " first tile is tight (it holds the lowest first key)");
      CHECK(max_vis / TILE == et - 1, c + " end tile is tight");
      CHECK(ft * TILE >= min_vis - r4d_win_max_overread(TILE), c + " over-read bound");
      // The effective bound is monotone in the row (the ext contract the end tile relies on).
      for (int r = r_lo + 1; r <= r_hi; ++r) {
        CHECK(r4d_win_khigh(ctx - q_len + r, ext[r]) >= r4d_win_khigh(ctx - q_len + r - 1, ext[r - 1]),
              c + " ext contract: non-decreasing bound");
      }
    }
  }
}

void TestDecode(std::mt19937& rng) {
  const int TILE = 16;
  for (int iter = 0; iter < 6000; ++iter) {
    const int W = 1 + static_cast<int>(rng() % 1100);
    const int q_len = 1 + static_cast<int>(rng() % 32);
    const int ctx = q_len + static_cast<int>(rng() % 6000);
    const int splits = 1 << (rng() % 6);   // 1..32
    const std::string c = Ctx(ctx, q_len, W, TILE) + " splits=" + std::to_string(splits);

    const int t0 = r4d_win_decode_t0(ctx, q_len, W, TILE);
    const int ntl = r4d_win_decode_ntl(ctx, TILE);
    const int tps = r4d_win_decode_tps(ctx, q_len, W, TILE, splits);
    CHECK(t0 >= 0 && t0 < ntl, c + " t0 in range");
    CHECK(tps >= 1, c + " tps >= 1");
    // Segments [t0 + s*tps, min(.. + tps, ntl)) tile [t0, ntl) exactly, the used ones non-empty.
    int covered = t0, used = 0;
    for (int s = 0; s < splits; ++s) {
      const int lo = t0 + s * tps;
      if (lo >= ntl) break;
      CHECK(lo == covered, c + " segments are contiguous");
      covered = std::min(lo + tps, ntl);
      ++used;
    }
    CHECK(covered == ntl, c + " segments reach the last tile");
    CHECK(used <= splits, c + " segment count");

    // Window-tile count is within the split law's bound.
    const int law_ctx = r4d_win_decode_max_ctx(ctx, W, q_len);
    CHECK(ntl - t0 <= (law_ctx + TILE - 1) / TILE, c + " window tiles exceed the split-law bound");

    for (int qrow = 0; qrow < q_len; ++qrow) {
      const int qpos = ctx - q_len + qrow;
      const int u = r4d_win_decode_row_used(ctx, q_len, qrow, W, TILE, splits);
      // Brute force: segments holding any key <= qpos (the row's causal range; the window's lower
      // part is masked inside segments, never skipped, so a segment is "used" by tile membership).
      int u_ref = 0;
      for (int s = 0; s < splits; ++s) {
        const int lo = t0 + s * tps;
        if (lo >= ntl) break;
        if (lo * TILE <= qpos) u_ref = s + 1;
      }
      CHECK(u == u_ref, c + " row_used qrow=" + std::to_string(qrow));
      // Every visible key of the row lies in segments [0, u).
      const int k_lo = r4d_win_klow(qpos, W);
      const int seg_of = (k_lo / TILE - t0) / tps;
      CHECK(k_lo / TILE >= t0, c + " row's first key precedes t0");
      CHECK(seg_of < u, c + " row's first key outside its used segments");
      const int seg_hi = (qpos / TILE - t0) / tps;
      CHECK(seg_hi == u - 1, c + " row's last key is in the last used segment");
    }
  }
}

}  // namespace

int main() {
  std::mt19937 rng(2026);
  TestVisibility();
  TestPrefillTiles(rng);
  TestDecode(rng);
  if (g_fail) {
    std::printf("FAIL (%d checks)\n", g_fail);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
