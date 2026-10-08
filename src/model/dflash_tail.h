// The device half of the hybrid mode's DFlash tail (docs/pp-tp2-hybrid.md 6): inject host feature rows into a drafter's ring in the
// 64-row slices Model::RunChunk injects. A free function over a DflashDraft (not a Model member) so Model::TpInjectDflashTail and the
// drafter-only identity test (tests/model/test_dflash_tail.cpp: two drafters, ~4 GiB, no target) run the very same loop. The
// rules (start, slices, rope rows) are dflash_tail_plan.h's.
#pragma once

#include <cstdint>

#include "dflash_draft.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

// `host`: [rows][draft.FeatureCols()] bf16 feature rows of positions [start_pos, start_pos + rows). `staging_dev`: device memory for
// one slice, slice_rows * FeatureCols() bf16 (a Model passes its dflash_features_dev_). `rope_t`: host int32[rows], the temporal rope
// row of each position, or nullptr (the drafter then ropes at start + t + its rope delta). Every slice is its own InjectFeatures call
// (so the fc / k_proj / v_proj GEMMs see M = slice_rows, as in a full run), followed by an arena reset and a stream synchronize (the
// drafter stages each slice's positions in one pinned array). slice_rows must not exceed the drafter's max_inject_rows. Throws what
// InjectFeatures throws (a start below the drafter's frontier).
void InjectFeatureRowsFromHost(DflashDraft& draft, core::Stream& stream, core::Arena& arena, uint16_t* staging_dev, int64_t slice_rows,
                               const uint16_t* host, int64_t rows, int64_t start_pos, const int32_t* rope_t);

}  // namespace r4dx::model
