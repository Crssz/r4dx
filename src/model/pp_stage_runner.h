// r4dx::model::pp -- the thread-agnostic stage runners of the two-stage prefill (docs/pp-tp2-hybrid.md 7 "Who owns what", critique item 3).
//
// PpModel's orchestration of one pipelined call used to be welded to its threads: stage B ran on the facade, the pinned slots and the
// sync buffers were allocated there, kv_sync_ lazily. The runners below are what that orchestration is made of, each taking
// (Model&, StageChannel&, StageBuffers&, options) and assuming nothing about the thread it runs on except that the thread is bound to
// the Model's device (and, in the hybrid, inside the Model's Tp2TuningScope -- the caller's business, like hipSetDevice):
//   StageBExportSyncBack  B's side of the warm-turn sync-back: its GDN state and the KV rows A is missing, into B-produced buffers;
//   HostImageSpans        the image rows A reads: host copies of the spans (a D2H on the thread that owns the rows' device);
//   RunStageA             A's whole command: apply the sync-back, Prefill / PrefillMultimodal (the chunks go into the channel), export the
//                         GDN live state, publish it. Poisons the channel and rethrows on failure;
//   RunStageB             B's side of the prefill: engage the channel, Prefill / PrefillMultimodal, import A's GDN state. Poisons the channel
//                         on failure and RETURNS the error (the owner must still join A before it reports);
//   StageBOnLastChunk     the GDN import's early start (PpStageSetup::before_last_chunk).
// PpModel calls them exactly as before (A on its worker, B on the facade); the hybrid posts them to the two rank workers. The channel and
// the buffers are the owner's: StageBuffers::AllocateAProduced / AllocateBProduced are explicit steps for the thread that owns the
// producing device, never done here.
//
// Options: `carry_kv` false is the hybrid's slot layout (no KV payload: stage Y has no layers below the split); `gdn_handoff` false drops
// the --pp 2 GDN hand-off (the hybrid reshards each stage's GDN state to the TP ranks directly). Both stages of a call, and their PpAttach
// (MakeStageSetup), must agree on carry_kv.
#pragma once

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "pp_channel.h"
#include "pp_stage_buffers.h"
#include "pp_sync.h"

namespace r4dx::model::pp {

struct StageRunOptions {
  int64_t split = 0;
  int timeout_ms = 30000;           // PpStageSetup::timeout_ms and the bound of B's wait for A's GDN state
  bool carry_kv = true;             // the slot carries A's KV blocks (--pp 2); false = the carry only (hybrid)
  bool gdn_handoff = true;          // A exports its GDN live state at the end of the call and B imports it (--pp 2)
  bool skip_gdn_import = false;     // negative control: B does not import A's GDN state
};

// A PpAttach setup for `role` from the options: split, channel, timeout, carry_kv. The caller adds the hooks it wants
// (on_chunk, before_last_chunk).
Model::PpStageSetup MakeStageSetup(Model::PpRole role, const StageRunOptions& opt, StageChannel* channel);

// The pinned-memory allocator of StageBuffers (hipHostMalloc, portable). Allocate from the thread that owns the producing device.
HostAllocator PinnedAllocator();

// One call's inputs and results for stage A, shared between the thread that prepares it and the one that runs RunStageA. On the heap when
// the preparing thread may give up on a stalled stage A and unwind (A's closure still holds it).
struct StageCall {
  std::vector<int32_t> ids;
  std::vector<Model::ImageSpan> a_spans;               // stage A's image spans (host rows)
  std::vector<std::vector<uint16_t>> a_rows;           // ... their storage
  bool multimodal = false;
  std::optional<Model::PpSyncState> sync;              // the scalars to adopt before the prefill (--pp 2); empty = the Model already has them
  std::optional<bool> dflash_injection;                // the injection policy to set before the prefill; empty = leave
  MirrorTracker::SyncPlan plan;                        // the sync-back to import (empty plan = nothing)
  int64_t call_id = 0;
  bool verify = false;                                 // R4DX_PP_VERIFY: digest the live state after the sync-back and at the end
  std::vector<std::pair<std::string, uint64_t>> b_digest_sync;    // B's live state at the call's start (verify)
  std::vector<std::pair<std::string, uint64_t>> a_digest_end;     // A's live state at the call's end (verify)
  double a_sync_ms = 0;
  double a_gdn_export_ms = 0;
};

// Fills call->a_spans / a_rows from `images`: spans whose rows are on a device get a host copy (stage A reads host rows). Runs on the
// thread bound to the device that holds the rows.
void HostImageSpans(const std::vector<Model::ImageSpan>& images, int64_t hidden, StageCall* call);

// B's side of the sync-back for `call->plan`: its GDN live state into bufs.GdnSync(), the KV rows into bufs.KvSync() (which must fit:
// KvSyncFits), and the digest when call->verify. B's device thread; the buffers are B-produced.
void StageBExportSyncBack(Model& b, StageBuffers& bufs, const StageRunOptions& opt, StageCall* call);

// Stage A's command. Throws after poisoning `ch` (and logging "[r4dx-pp] stage A failed").
void RunStageA(Model& a, StageChannel& ch, StageBuffers& bufs, const StageRunOptions& opt, StageCall& call);

// State stage B's two entry points share within a call (both run on B's thread, one inside the other).
struct StageBCallState {
  int64_t call_id = -1;
  bool early_import = false;   // the GDN import was started before the last chunk computed
};
// PpStageSetup::before_last_chunk for stage B: if A's GDN export is already complete, start importing it on B's side stream now.
void StageBOnLastChunk(Model& b, StageChannel& ch, StageBuffers& bufs, const StageRunOptions& opt, StageBCallState& st);

struct StageBResult {
  std::vector<float> logits;
  std::exception_ptr error;    // non-null: B failed (the channel is poisoned); the owner joins A, then reports PickRootCause
  double gdn_wait_ms = 0;      // B's wait for A's GDN state after its last chunk (0 when the import overlapped)
};
// Stage B's prefill: engages the channel around the prefill, then imports A's GDN state unless it started early. Never throws.
StageBResult RunStageB(Model& b, StageChannel& ch, StageBuffers& bufs, const StageRunOptions& opt, StageBCallState& st,
                       const std::vector<int32_t>& ids, const std::vector<Model::ImageSpan>* images);

}  // namespace r4dx::model::pp
