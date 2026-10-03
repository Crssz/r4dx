// R4D sliding-window and KV-ring arithmetic (docs/gemma4-plan.md 3.3, "Sliding KV ring by block-table
// aliasing"). A pure header: no device state, no HIP include, so the SAME functions are called by the
// windowed attention kernels (r4d_attn_{prefill,decode}_h256_gqa6.hip's WIN variants, via
// r4d_attn_paged_h256_gqa2.hip) and by the CPU tests (tests/kernels/test_attn_window_math.cpp,
// test_ring_math.cpp) that pin them against brute force.
//
// Conventions. A query's ABSOLUTE position is `qpos` (the kernels' `klimit`: ctx - q_len + row, the
// last key it may see causally). A key's absolute position is `kpos`. The sliding mask is
//
//      kpos <= qpos  and  kpos > qpos - W          (W = the window; a query sees W keys, itself included)
//
// i.e. HF's `dist < sliding_window`. `klimit_ext` (bidirectional image blocks, prefill only) can only
// RAISE a row's upper bound: the effective upper bound is max(qpos, ext), the lower bound stays
// qpos - W + 1. Its contract: the effective upper bound is non-decreasing in the query position
// (true for "end of the image block this row is in, else causal"), which is what lets the CTA tile
// range be taken at the CTA's last row.
//
// The ring. Sliding layers keep only the last RB * BS keys. One int32 block table T[i] = i % RB,
// shared by every sliding layer, makes the kernels' `bt[kpos / BS]` land on the ring block of key
// kpos, so the kernels' addressing is unchanged. A key's ring slot is kpos % (RB * BS).
#pragma once
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define R4D_WIN_HD __host__ __device__ __forceinline__
#else
#define R4D_WIN_HD inline
#endif

// ---- visibility ----------------------------------------------------------------------------------
// First key row `qpos` may see (clamped at 0). W <= 0 means "no window".
R4D_WIN_HD int r4d_win_klow(int qpos, int W) {
    if (W <= 0) return 0;
    const int lo = qpos - W + 1;
    return lo > 0 ? lo : 0;
}
R4D_WIN_HD bool r4d_win_visible(int kpos, int qpos, int W) {
    return kpos <= qpos && (W <= 0 || kpos > qpos - W);
}
// The upper bound with a bidirectional extension (ext < 0 = none).
R4D_WIN_HD int r4d_win_khigh(int qpos, int ext) { return ext > qpos ? ext : qpos; }

// ---- prefill tile range --------------------------------------------------------------------------
// The kernels walk TILE-key tiles. A CTA's rows span qpos in [qpos_lo, qpos_hi] (absolute); the first
// tile it needs holds the LOWEST row's first visible key, the last one holds the highest row's last
// key (clamped to ctx). Tiles are numbered from 0 so the staging code's k0 = ti * TILE is unchanged.
R4D_WIN_HD int r4d_win_first_tile(int qpos_lo, int W, int TILE) {
    return r4d_win_klow(qpos_lo, W) / TILE;
}
R4D_WIN_HD int r4d_win_end_tile(int khigh_max, int ctx, int TILE) {   // exclusive
    const int last = (khigh_max + 1 < ctx ? khigh_max + 1 : ctx);
    return (last + TILE - 1) / TILE;
}
// Keys of the first tile that lie BELOW the lowest row's window (they are staged, then masked): at
// most TILE - 1. Their ring slots alias NEWER keys, which is why the ring cache must be zero-
// initialised (a masked P = 0 times an fp8 NaN would still be NaN; any finite value is harmless).
R4D_WIN_HD int r4d_win_max_overread(int TILE) { return TILE - 1; }

// ---- decode split-KV geometry --------------------------------------------------------------------
// The decode kernel cuts the KV tile range into `splits` segments and the combine kernel merges
// them in fixed order. With a window the range is [t0, ntl): t0 is the first tile of the EARLIEST
// row's window (row 0 of the q_len rows, qpos = ctx - q_len), ntl the tile count of ctx. Every
// segment size, segment start and per-row used-segment count below is a function of (ctx, q_len, W,
// TILE, splits) alone, so the decode and combine kernels agree without passing t0 between them.
R4D_WIN_HD int r4d_win_decode_t0(int ctx, int q_len, int W, int TILE) {
    return r4d_win_first_tile(ctx - q_len, W, TILE);
}
R4D_WIN_HD int r4d_win_decode_ntl(int ctx, int TILE) { return (ctx + TILE - 1) / TILE; }
R4D_WIN_HD int r4d_win_decode_tps(int ctx, int q_len, int W, int TILE, int splits) {
    const int n = r4d_win_decode_ntl(ctx, TILE) - r4d_win_decode_t0(ctx, q_len, W, TILE);
    return (n + splits - 1) / splits;
}
// Segments that hold any key of row `qrow` (the combine's `u`): the row's last key is qpos = ctx -
// q_len + qrow, so its tiles end at ceil((qpos + 1) / TILE).
R4D_WIN_HD int r4d_win_decode_row_used(int ctx, int q_len, int qrow, int W, int TILE, int splits) {
    const int t0 = r4d_win_decode_t0(ctx, q_len, W, TILE);
    const int tps = r4d_win_decode_tps(ctx, q_len, W, TILE, splits);
    const int row_end = (ctx - q_len + 1 + qrow + TILE - 1) / TILE;     // exclusive tile bound
    const int n = row_end - t0;
    return n <= 0 ? 0 : (n + tps - 1) / tps;
}
// The context the split LAW sees: the window plus the verify rows plus one tile of slack (the over-
// read and the alignment of the last tile), however long the sequence is. The segment count is a
// function of this bound only, so decode cost stops growing with the context.
R4D_WIN_HD int r4d_win_decode_max_ctx(int ctx, int W, int q_len) {
    const int cap = W + q_len + 16;
    return ctx < cap ? ctx : cap;
}

// ---- the KV ring ---------------------------------------------------------------------------------
// Ring slot of key kpos in a ring of `ring_tokens` keys (a write slot AND, with the shared block
// table, the slot the kernels read): the same value as ((kpos / BS) % RB) * BS + kpos % BS.
R4D_WIN_HD int r4d_ring_slot(int64_t kpos, int ring_tokens) {
    return (int)(kpos % ring_tokens);
}
// The shared block table: entry i points at ring block i % RB.
R4D_WIN_HD int r4d_ring_block(int64_t block_index, int ring_blocks) {
    return (int)(block_index % ring_blocks);
}
// Ring capacity rule: a chunk of T new keys is written BEFORE its attention runs, and its earliest
// query (position p) still reads key p - W + 1, so the ring must hold the window plus the chunk:
// ring_tokens >= W + T. (The newest key, p + T - 1, overwrites the slot of the key ring_tokens older;
// that key must be below p - W + 1. W + T - 1 is already exactly enough -- test_ring_math simulates
// both edges -- and the rule keeps one key of margin.)
R4D_WIN_HD bool r4d_ring_fits(int ring_tokens, int W, int T) { return ring_tokens >= W + T; }
// The largest chunk (prefill chunk, verify window, image block) a ring serves.
R4D_WIN_HD int r4d_ring_max_chunk(int ring_tokens, int W) { return ring_tokens - W; }
