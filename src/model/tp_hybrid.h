// TpModel's hybrid-mode types (docs/pp-tp2-hybrid.md 7, 15): the facade-side state `Hybrid` and the per-call shared block `HyCall`. Private to
// tp_model.cpp / tp_model_hybrid.cpp (tp_model.h only forward-declares them). HIP-free: every device handle lives in the RankSlots.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hybrid_dispatch.h"
#include "hybrid_ring.h"
#include "hybrid_sync.h"
#include "pp_channel.h"
#include "pp_stage_buffers.h"
#include "pp_stage_runner.h"
#include "reshard_exec.h"
#include "reshard_plan.h"
#include "stage_sync.h"
#include "tp_model.h"

namespace r4dx::model {

// The facade's side of the hybrid: what the load resolved, the stage geometry, the TP-master tracker, the stage channel and its pinned
// buffers, the counters. Written only on the facade thread; the rank threads read it (immutable fields, the channel and the buffers'
// pointers) while a command runs, and the facade does not touch the mutable parts then.
struct TpModel::Hybrid {
  // ---- resolved at load
  int64_t split = 0;          // k
  int64_t stage_ctx = 0;      // S: tokens of stage KV per attention layer
  int64_t min_rows = hybrid::kHybridMinRowsDefault;
  int64_t block = 16;         // KV rows per block
  int64_t hidden = 0;
  int64_t chunk_rows = 0;     // the prefill super-chunk both stages agreed on
  int64_t layers = 0;         // loaded layers
  bool mtp = false;           // the ranks (and stage Y) have an MTP head
  bool dflash = false;        // the ranks have a drafter
  std::vector<int64_t> targets;  // the drafter's target layers (stage Y captures all of them, stage X those below k)
  int64_t dfl_cols_x = 0;        // ... of which stage X captures
  int64_t gather_bytes_per_row[2] = {0, 0};  // the warm gather's bytes per stale KV row, per stage (the remote half: hybrid::GatherBytesPerRow)
  std::chrono::milliseconds step_timeout{30000};  // every wait between the two threads
  int slot_of_stage[2] = {0, 1};                  // ranks_ index of stage A (rank 1) / stage B (rank 0)
  hybrid::StateGeometry geo;
  hybrid::PlanParams plan_params;
  pp::StageRunOptions run;
  std::unique_ptr<hybrid::TpMasterTracker> tracker;
  std::unique_ptr<pp::StageBuffers> bufs;      // slots (stage A's thread), nothing else is used by the hybrid
  std::unique_ptr<pp::StageChannel> channel;
  // ---- per call
  int64_t call_id = 0;
  HybridStats stats;
  // test-only fault injection (TpModel::ArmHybridFault): set on the facade before a command is posted, read by the rank threads during it
  int fault_rank = -1;
  HybridFaultPhase fault_phase = HybridFaultPhase::kNone;
  // test-only negative control (TpModel::SetHybridNegControl): persistent, same threading rule as the fault above
  HybridNegControl neg_control = HybridNegControl::kNone;
};

// One pipelined call, shared by the two rank closures (heap, co-owned: a rank that stalls past the watchdog never touches the facade's
// stack, N53). Everything above the line is set before the command is posted and read-only afterwards; below it, each field has ONE writer
// thread and its readers wait on the gate that publishes it.
struct TpModel::HyCall {
  // ---- inputs
  std::vector<int32_t> ids;
  std::vector<ImageSpan> spans;                 // host rows (embeds_on_host); point into `rows`
  std::shared_ptr<std::vector<uint16_t>> rows;  // their storage
  bool multimodal = false;
  int64_t p0 = 0, n = 0, call_id = 0;
  StageSyncState sync;        // TP rank 0's scalars + MTP seed after the collapse: into BOTH stages
  bool inject = true;         // the DFlash injection policy the ranks run under
  hybrid::ReshardPlan gather, scatter;
  std::vector<hybrid::RingBatch> gather_batches[2];   // by the stage whose card they LEAVE (hybrid::PackCrossOps source_stage)
  std::vector<hybrid::RingBatch> scatter_batches[2];
  hybrid::TailPlan tail;
  uint16_t* tail_host = nullptr;                      // stage Y's capture buffer (rank 0's pinned memory)
  pp::StageChannel* channel = nullptr;
  std::chrono::milliseconds timeout{30000};
  // ---- shared sync. pipe[s]: the ring out of the card of stage s. Batch numbers keep counting across the gather and the scatter.
  // The pipes' depth IS the physical ring's (kHybridRingSlots: HybridExec's slot_of): one constant, so a grant never outruns the memory.
  hybrid::RingPipe pipe[2] = {hybrid::RingPipe(hybrid::kHybridRingSlots), hybrid::RingPipe(hybrid::kHybridRingSlots)};
  hybrid::HostGate end_gate;  // stage Y's end state (`end`, `logits`, `tail_rope`) is published
  // ---- written by stage Y's thread (rank 0), read by stage X's after the gate
  StageSyncState end;
  std::vector<float> logits;
  std::vector<int32_t> rope_rows;  // PrefillMultimodal's [3, n] rope rows (multimodal calls with a tail)
  std::vector<int32_t> tail_rope;  // the tail's temporal rope row ([tail.rows]); empty = the delta shortcut
  // ---- per stage thread
  pp::StageCall xcall;             // stage X's call
  pp::StageBCallState ystate;
  int64_t pos_after[2] = {0, 0};
  int64_t fallback_after[2] = {0, 0};
  double gather_ms[2] = {0, 0}, prefill_ms[2] = {0, 0}, reshard_ms[2] = {0, 0}, tail_ms[2] = {0, 0}, total_ms[2] = {0, 0};

  // Wakes everything the other thread may be blocked on: the stage channel, both ring pipes, the gate.
  void PoisonAll(const std::string& why) {
    if (channel != nullptr) channel->Poison(why);
    pipe[0].Poison(why);
    pipe[1].Poison(why);
    end_gate.Poison(why);
  }
};

// Load's first look at a hybrid request, before any weight is read: whether the hybrid is wanted at all (ModelOptions::pp on, then the
// kill switch PpOptions::hybrid / R4DX_HYBRID) and, if so, the checks that need no device: real mode, the placement against
// --pp-devices / R4DX_PP_DEVICES, the PpOptions ranges. Throws std::invalid_argument; logs one line when the kill switch turns the
// hybrid off under `--pp 2`. `devices`: TP rank r -> HIP ordinal.
bool ResolveHybridRequest(const ModelOptions& opts, const TpOptions& tp, const PpOptions& pp, const std::vector<int>& devices);

}  // namespace r4dx::model
