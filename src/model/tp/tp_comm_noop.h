// r4dx::model::tp::MakeNoopComm -- the single-rank, no-op TpComm (docs/tp.md 6.4, `--tp-mode noop`).
//
// It lets ONE rank's shard (world 2) run on one device with no peer: AllReduceSumBf16 launches
// nothing (the "no AR-slot kernel" baseline tp_bench's L_vs_no_ar_kernel is defined against,
// docs/tp.md 1.4), HostAllGather copies this rank's bytes into every rank's slot, and
// CheckLockstep/CheckHealthy do nothing. The tokens such a run produces are meaningless; only its
// timing is (tool_tp_step_bench, gate G3). It needs no TpGroup, so a bare Model can use it before
// any threading code exists.
#pragma once

#include <memory>

#include "r4dx/core/tp_comm.hpp"

namespace r4dx::model::tp {

// Throws std::invalid_argument unless world >= 1 and 0 <= rank < world.
std::unique_ptr<core::TpComm> MakeNoopComm(int world, int rank);

}  // namespace r4dx::model::tp
