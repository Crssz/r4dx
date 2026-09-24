// r4dx::model::tp -- the tensor-parallel sharding rules (docs/tp.md sections 4 and 5.2): which
// tensor splits along which axis, and the exact byte ranges of every on-disk part of a rank's
// slice.
//
// CPU-only and HIP-free on purpose (target r4dx_tp_shard): the loader (Container::Load with
// tp_world > 1) is the only production caller, and tests/model/test_tp_shard.cpp proves
// `Gather(full, Plan*(...)) == pack(slice(W))` byte for byte against the converter's real packers
// (src/convert/include/r4dx_convert/quant_int4.hpp / quant_mxfp4.hpp) without a device.
//
// Two layers:
//   * RuleFor/RankRows/RankCols -- LOGICAL: the global row/col indices of a rank's slice of
//     W[N, K]. A column-parallel tensor is a concatenation of fused SEGMENTS (in_proj_qkv =
//     q|k|v, gate_up = gate|up, ...); each segment is split into `world` equal contiguous parts
//     and rank r takes part r of every segment, concatenated in segment order.
//   * PlanRows/PlanCols/Gather -- PHYSICAL: the byte runs of one on-disk part (bf16 row-major, the
//     w4 fragment-order wq, the per-(tile, group, row) uint32 scale dwords, mxfp4's [K/32][N] ws,
//     ...) whose concatenation is the rank's packed slice, in the order that slice stores them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model_config.h"

namespace r4dx::model::tp {

struct Segment {
  int64_t begin = 0;  // global row of the segment's first row
  int64_t rows = 0;
};
struct Range {
  int64_t begin = 0;
  int64_t count = 0;
};

enum class Split {
  kReplicate,  // every rank loads the full tensor
  kRows,       // column-parallel: output dim N split per segment
  kCols,       // row-parallel: input dim K split
  kRank0Only,  // vision tower weights
};

struct ShardRule {
  Split split = Split::kReplicate;
  std::vector<Segment> segments;  // kRows only, in on-disk order
  int64_t k_total = 0;            // kCols only: the full K
};

// `base`: the container base name WITHOUT its ".{layout}.{part}" suffix, e.g.
// "text.layers.3.gdn.in_proj_qkv", "mtp.attn.o", "lm_head", "text.layers.3.gdn.A_log". A linear's
// bare pre-R1 form and its `.bf16.w` form share one base, so one rule covers every on-disk form.
// `global`: the UNSHARDED config. Throws std::invalid_argument for a name it does not know (a new
// tensor must be classified, never silently replicated) -- including any name, on every path, that
// still carries its ".{bf16|w4a16|w4a8|mxfp4}.{part}" suffix -- for a layer index out of range,
// and for a GDN tensor on a full-attention layer or the reverse. The two exceptions are
// whole-prefix rules by design (docs/tp.md 4.2): every other `vision.*` name is rank-0-only and
// every other `dflash.*` name (a separate, fully replicated container) is replicated.
ShardRule RuleFor(const std::string& base, const ModelConfig& global);

// Rank `rank`'s global row ranges of a kRows rule, in concatenation order (one per segment).
// Throws std::invalid_argument unless r.split == kRows and every segment divides by `world`.
std::vector<Range> RankRows(const ShardRule& r, int world, int rank);
// Rank `rank`'s global K range of a kCols rule. Throws std::invalid_argument unless
// r.split == kCols and r.k_total divides by `world`.
Range RankCols(const ShardRule& r, int world, int rank);

// One on-disk part of a tensor (docs/tp.md 4.3 has the physical layout of each).
enum class Part {
  kBf16,      // row-major [N, K] bf16: `.bf16.w`, the bare pre-R1 form, 2-D raw bf16 (in_proj_a/b)
  kW4Wq,      // w4a16.wq / w4a8.wq: (t, kb, lh, r, s), 512 B per (16-row tile, 64-K block)
  kW4a16Wsz,  // uint32 per (t, g, r), g = the container's w4a16 group
  kW4a8Ws,    // uint32 per (t, g, r), g = 128 -- the same 4-byte stride as wsz, NOT uint16
  kMxWq,      // (nt, ks, lane), 128 B per (16-row tile, 16-K step)
  kMxWs,      // uint8 [K/32][N]
  kMxWref,    // int8 [N], each row's max E8M0 exponent
  kElem,      // [N] rows of `row_bytes` each: 1-D vectors, conv1d_weight ([conv_dim][4] bf16)
};
struct PartShape {
  Part part = Part::kBf16;
  int64_t N = 0, K = 0;   // the FULL logical W[N, K] the part belongs to (kElem: N rows, K unused)
  int group = 0;          // kW4a16Wsz: container w4a16 group; kW4a8Ws: 128; kMxWs: 32
  int64_t row_bytes = 0;  // kElem only: bytes per row (8 for conv1d_weight, 4 for fp32 vectors)
};
struct ByteRun {
  size_t src_off = 0;
  size_t bytes = 0;
};

// Byte runs into the full part whose concatenation is that part of W[rows], rows = RankRows(...)
// (any list of global row ranges works). Adjacent runs are merged, so a single run means "one
// contiguous range of the mmap: upload it directly, no staging". Throws std::invalid_argument on a
// shape the part cannot have, a range out of bounds, or a range misaligned for the part (the
// 16-row-tile parts need begin and count % 16).
std::vector<ByteRun> PlanRows(const PartShape& shape, const std::vector<Range>& rows);
// Byte runs whose concatenation is that part of W[:, cols]. kMxWref returns the FULL [N]
// (docs/tp.md 4.3 "The mxfp4 wref exception"). kElem has no column axis and throws. Throws
// std::invalid_argument on a misaligned range (w4 wq: % 64; w4 scale dwords: % group; mxfp4 wq
// and ws: % 32).
std::vector<ByteRun> PlanCols(const PartShape& shape, Range cols);
// Concatenation of `runs` out of `full` (`full_bytes` long). Throws std::out_of_range if a run
// reaches past the end.
std::vector<uint8_t> Gather(const uint8_t* full, size_t full_bytes,
                            const std::vector<ByteRun>& runs);

}  // namespace r4dx::model::tp
