// tests/kernels/test_ring_math.cpp -- pure CPU (no HIP). The sliding KV ring of docs/gemma4-plan.md
// 3.3: SlidingRingGeometry (src/model/attention/include/r4dx/model/attention/sliding_ring.hpp) and
// the slot / block-table arithmetic of third_party/libr4d/r4d_attn_window.h, against a simulated ring:
//   * Gemma 4's numbers: 96 blocks = 1536 tokens, 252 MB of fp8 across 40 layers, a 64 KiB shared
//     block table at 262144 context, 512 rows of chunk headroom (so a 288-row image block, a 256-row
//     prefill chunk and a 16-row verify window all fit);
//   * the write slot (pos + t) % ring equals ((p / bs) % RB) * bs + p % bs and the slot the shared
//     block table T[i] = i % RB makes the kernels read for key p;
//   * the safety rule by simulation: every chunk is written BEFORE its attention, then every query
//     reads every key of its window back through the block-table slot and must find that key. Holds
//     at ring = W + T - 1 (exactly enough), breaks at W + T - 2, across wraps far past 1536;
//   * a speculative rollback (re-writing an earlier position with different data) leaves every key the
//     re-run needs intact;
//   * the constructor / FillSlots refusals.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/model/attention/sliding_ring.hpp"

using r4dx::model::attention::SlidingRingGeometry;

namespace {

int g_fail = 0;
void Fail(const std::string& what) {
  if (++g_fail <= 20) std::printf("FAIL: %s\n", what.c_str());
}
#define CHECK(cond, msg)                                    \
  do {                                                      \
    if (!(cond)) Fail(std::string(msg) + "  [" #cond "]");  \
  } while (0)

template <class F>
bool Throws(F f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void TestGemmaNumbers() {
  const SlidingRingGeometry g(1024, 288, 16, 262144);
  CHECK(g.RingBlocks() == 96 && g.RingTokens() == 1536, "default ring is 96 blocks / 1536 tokens");
  CHECK(g.MaxChunkRows() == 512, "512 rows of chunk headroom");
  CHECK(g.ChunkFits(288) && g.ChunkFits(256) && g.ChunkFits(16) && g.ChunkFits(512), "image / prefill / verify chunks fit");
  CHECK(!g.ChunkFits(513), "513 rows do not fit");
  CHECK(g.BlockTableEntries() == 16384, "block table indexes 262144 / 16 blocks");
  CHECK(g.BlockTableEntries() * 4 == 65536, "the shared block table is 64 KiB");
  const int64_t per_layer = int64_t{g.RingBlocks()} * 8 /*kv heads*/ * 16 * 2 * 256;
  CHECK(per_layer * 40 == 251658240, "40 sliding layers x 8 kv heads x 1536 tokens x 512 B = 252 MB fp8");
  const SlidingRingGeometry h(1024, 16, 16, 131072);   // a verify-only geometry still gets slack
  CHECK(h.RingTokens() % 16 == 0 && h.ChunkFits(16), "small-chunk geometry");
  // Explicit sizes.
  CHECK(SlidingRingGeometry(1024, 288, 16, 8192, 1312).RingBlocks() == 82, "exact ring 1312 = 82 blocks");
  CHECK(Throws([] { SlidingRingGeometry(1024, 288, 16, 8192, 1296); }), "ring below W + chunk is refused");
  CHECK(Throws([] { SlidingRingGeometry(1024, 288, 16, 8192, 1500); }), "ring not a block multiple is refused");
  CHECK(Throws([] { SlidingRingGeometry(0, 288, 16, 8192); }), "zero window is refused");
}

void TestSlotsAndTable() {
  const SlidingRingGeometry g(1024, 288, 16, 20000);
  const std::vector<int32_t> table = g.BuildBlockTable();
  const int RB = g.RingBlocks(), bs = g.BlockSize();
  CHECK(static_cast<int>(table.size()) == g.BlockTableEntries(), "table size");
  bool ok = true;
  for (size_t i = 0; i < table.size(); ++i) ok = ok && table[i] == static_cast<int32_t>(i % RB);
  CHECK(ok, "T[i] == i % RB");

  std::mt19937 rng(8);
  for (int iter = 0; iter < 2000; ++iter) {
    const int64_t pos = rng() % 19000;
    const int rows = 1 + static_cast<int>(rng() % 300);
    const std::vector<int32_t> slots = g.Slots(pos, rows);
    for (int t = 0; t < rows; ++t) {
      const int64_t p = pos + t;
      const int via_table = table[p / bs] * bs + static_cast<int>(p % bs);
      const int formula = static_cast<int>(((p / bs) % RB) * bs + p % bs);
      CHECK(slots[t] == via_table && slots[t] == formula && slots[t] == static_cast<int>(p % g.RingTokens()),
            "write slot == table-addressed read slot == ((p/bs)%RB)*bs + p%bs");
      CHECK(slots[t] >= 0 && slots[t] < g.RingTokens(), "slot in range");
    }
  }
  CHECK(Throws([&] { g.Slots(0, 513); }), "FillSlots refuses a chunk past the headroom");
  CHECK(Throws([&] { g.Slots(-1, 4); }), "FillSlots refuses a negative position");
}

// Simulated ring of `ring` tokens holding key positions (-1 = never written). Chunks of <= max_t
// rows; returns true when every window read finds the key it asks for.
bool Simulate(int ring, int W, int max_t, int bs, int64_t total, std::mt19937& rng, bool with_rollback) {
  std::vector<int64_t> slot_key(static_cast<size_t>(ring), -1);
  const int RB = ring / bs;
  int64_t pos = 0;
  while (pos < total) {
    const int T = 1 + static_cast<int>(rng() % max_t);
    for (int t = 0; t < T; ++t) {
      const int64_t p = pos + t;
      slot_key[(p / bs % RB) * bs + p % bs] = p;   // the write slot, via the block formula
    }
    for (int t = 0; t < T; ++t) {
      const int64_t qpos = pos + t;
      for (int64_t k = std::max<int64_t>(0, qpos - W + 1); k <= qpos; ++k) {
        // Read through the shared block table: block k/bs -> ring block (k/bs) % RB.
        if (slot_key[((k / bs) % RB) * bs + k % bs] != k) return false;
      }
    }
    pos += T;
    if (with_rollback && T <= 16 && (rng() % 4) == 0 && pos > 20) {
      // A speculative window of the last T keys is rejected: positions [pos - T, pos) are re-run with
      // new data. Re-write them, then re-read their windows.
      const int64_t base = pos - T;
      for (int t = 0; t < T; ++t) {
        const int64_t p = base + t;
        slot_key[(p / bs % RB) * bs + p % bs] = p;
      }
      for (int t = 0; t < T; ++t) {
        const int64_t qpos = base + t;
        for (int64_t k = std::max<int64_t>(0, qpos - W + 1); k <= qpos; ++k) {
          if (slot_key[((k / bs) % RB) * bs + k % bs] != k) return false;
        }
      }
    }
  }
  return true;
}

void TestSafetyBySimulation() {
  std::mt19937 rng(77);
  const int bs = 16;
  // Gemma 4's ring with every chunk size it serves, far past several wraps.
  CHECK(Simulate(1536, 1024, 512, bs, 20000, rng, true), "1536-token ring serves chunks up to 512 rows");
  CHECK(Simulate(1536, 1024, 288, bs, 20000, rng, true), "... image blocks of 288");
  CHECK(Simulate(1536, 1024, 16, bs, 20000, rng, true), "... verify windows of 16 (with rollbacks)");
  // The edge for a fixed chunk size T: ring = W + T - 1 is exactly enough, W + T - 2 is not.
  for (int W : {32, 64, 1024}) {
    for (int T : {1, 5, 16, 64}) {
      const int need = ((W + T - 1) + bs - 1) / bs * bs;           // blocks
      // Exactly W + T - 1 only when it is a block multiple; check the block-rounded size passes and a
      // ring one block smaller than W + T - 1 fails for a full-size chunk.
      bool pass_rounded = true, fail_small = false;
      // Fixed-size chunks: T-row chunks only.
      {
        std::vector<int64_t> slot_key(static_cast<size_t>(need), -1);
        const int RB = need / bs;
        for (int64_t pos = 0; pos < 6000; pos += T) {
          for (int t = 0; t < T; ++t) slot_key[(pos + t) / bs % RB * bs + (pos + t) % bs] = pos + t;
          for (int t = 0; t < T; ++t) {
            const int64_t q = pos + t;
            for (int64_t k = std::max<int64_t>(0, q - W + 1); k <= q; ++k) {
              if (slot_key[(k / bs) % RB * bs + k % bs] != k) pass_rounded = false;
            }
          }
        }
      }
      CHECK(pass_rounded, "ring of round_up(W + T - 1) blocks passes W=" + std::to_string(W) + " T=" + std::to_string(T));
      const int small = need - bs;
      if (small >= bs && small < W + T - 1) {
        std::vector<int64_t> slot_key(static_cast<size_t>(small), -1);
        const int RB = small / bs;
        for (int64_t pos = 0; pos < 6000 && !fail_small; pos += T) {
          for (int t = 0; t < T; ++t) slot_key[(pos + t) / bs % RB * bs + (pos + t) % bs] = pos + t;
          for (int t = 0; t < T && !fail_small; ++t) {
            const int64_t q = pos + t;
            for (int64_t k = std::max<int64_t>(0, q - W + 1); k <= q; ++k) {
              if (slot_key[(k / bs) % RB * bs + k % bs] != k) { fail_small = true; break; }
            }
          }
        }
        CHECK(fail_small, "a ring below W + T - 1 loses a window key W=" + std::to_string(W) + " T=" + std::to_string(T));
      }
    }
  }
  // The rule itself.
  CHECK(r4d_ring_fits(1536, 1024, 512) && !r4d_ring_fits(1536, 1024, 513), "r4d_ring_fits at the edge");
  CHECK(r4d_ring_max_chunk(1536, 1024) == 512, "r4d_ring_max_chunk");
}

}  // namespace

int main() {
  TestGemmaNumbers();
  TestSlotsAndTable();
  TestSafetyBySimulation();
  if (g_fail) {
    std::printf("FAIL (%d checks)\n", g_fail);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
