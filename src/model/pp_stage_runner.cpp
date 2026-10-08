#include "pp_stage_runner.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "r4dx/core/error.hpp"

namespace r4dx::model::pp {

namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// Engages stage B's channel path for the scope of a prefill (Model::PpSetActive).
struct ActiveGuard {
  Model& m;
  explicit ActiveGuard(Model& mm) : m(mm) { m.PpSetActive(true); }
  ~ActiveGuard() {
    try {
      m.PpSetActive(false);
    } catch (...) {
    }
  }
};

}  // namespace

Model::PpStageSetup MakeStageSetup(Model::PpRole role, const StageRunOptions& opt, StageChannel* channel) {
  Model::PpStageSetup st;
  st.role = role;
  st.split = opt.split;
  st.channel = channel;
  st.timeout_ms = opt.timeout_ms;
  st.carry_kv = opt.carry_kv;
  return st;
}

HostAllocator PinnedAllocator() {
  HostAllocator a;
  a.alloc = [](size_t bytes) {
    void* p = nullptr;
    R4DX_HIP_CHECK(hipHostMalloc(&p, bytes, hipHostMallocPortable));
    return static_cast<uint8_t*>(p);
  };
  a.release = [](uint8_t* p) { static_cast<void>(hipHostFree(p)); };
  return a;
}

void HostImageSpans(const std::vector<Model::ImageSpan>& images, int64_t hidden, StageCall* call) {
  call->a_spans = images;
  call->a_rows.clear();
  call->a_rows.reserve(images.size());
  for (Model::ImageSpan& sp : call->a_spans) {
    if (sp.embeds_on_host) continue;
    const size_t n = static_cast<size_t>(sp.tokens * hidden);
    call->a_rows.emplace_back(n);
    R4DX_HIP_CHECK(hipMemcpy(call->a_rows.back().data(), sp.embeds, n * sizeof(uint16_t), hipMemcpyDeviceToHost));
    sp.embeds = call->a_rows.back().data();
    sp.embeds_on_host = true;
  }
}

void StageBExportSyncBack(Model& b, StageBuffers& bufs, const StageRunOptions& opt, StageCall* call) {
  const MirrorTracker::SyncPlan& plan = call->plan;
  if (plan.gdn) b.PpExportGdn(bufs.GdnSync(), opt.split);
  const size_t kv_need = plan.kv_row1 > plan.kv_row0 ? b.PpKvWireBytes(opt.split, plan.kv_row0, plan.kv_row1) : 0;
  if (kv_need > 0) {
    if (!KvSyncFits(kv_need, bufs.KvSyncBytes())) {
      throw std::logic_error("pp::StageBExportSyncBack: the sync-back needs " + std::to_string(kv_need) + " bytes of KV, the sync buffer holds " +
                             std::to_string(bufs.KvSyncBytes()));
    }
    b.PpExportKvRows(bufs.KvSync(), opt.split, plan.kv_row0, plan.kv_row1);
  }
  if (call->verify) call->b_digest_sync = b.PpLiveDigest(opt.split);
}

void RunStageA(Model& a, StageChannel& ch, StageBuffers& bufs, const StageRunOptions& opt, StageCall& call) {
  try {
    const auto t0 = Clock::now();
    if (call.sync) a.PpSetSyncState(*call.sync);
    if (call.dflash_injection) a.SetDflashInjectionEnabled(*call.dflash_injection);
    if (call.plan.gdn) a.PpImportGdn(bufs.GdnSync(), opt.split);
    if (call.plan.kv_row1 > call.plan.kv_row0) a.PpImportKvRows(bufs.KvSync(), opt.split, call.plan.kv_row0, call.plan.kv_row1);
    call.a_sync_ms = Ms(t0, Clock::now());
    if (call.verify) {
      const auto d = a.PpLiveDigest(opt.split);
      if (d != call.b_digest_sync) {
        throw std::runtime_error("R4DX_PP_VERIFY: stage A's state after the sync-back differs from stage B's (position " +
                                 std::to_string(call.sync ? call.sync->pos : a.PositionCount()) + ")");
      }
    }
    if (call.multimodal) {
      (void)a.PrefillMultimodal(call.ids, call.a_spans);
    } else {
      (void)a.Prefill(call.ids);
    }
    if (opt.gdn_handoff) {
      // The GDN live state of A's layers follows the last chunk (compactly); B imports it while it computes its own last chunk when
      // it is ready in time (StageBOnLastChunk), else right after.
      const auto t_exp = Clock::now();
      a.PpExportGdn(bufs.GdnHand(), opt.split);
      call.a_gdn_export_ms = Ms(t_exp, Clock::now());
      ch.PublishBulk(call.call_id);
    }
    if (call.verify) call.a_digest_end = a.PpLiveDigest(opt.split);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[r4dx-pp] stage A failed: %s\n", e.what());
    ch.Poison(std::string("stage A failed: ") + e.what());
    throw;
  } catch (...) {
    ch.Poison("stage A failed");
    throw;
  }
}

void StageBOnLastChunk(Model& b, StageChannel& ch, StageBuffers& bufs, const StageRunOptions& opt, StageBCallState& st) {
  if (!opt.gdn_handoff || opt.skip_gdn_import) return;
  if (ch.BulkReady(st.call_id)) {
    b.PpImportGdnAsync(bufs.GdnHand(), opt.split);
    st.early_import = true;
  }
}

StageBResult RunStageB(Model& b, StageChannel& ch, StageBuffers& bufs, const StageRunOptions& opt, StageBCallState& st,
                       const std::vector<int32_t>& ids, const std::vector<Model::ImageSpan>* images, std::vector<int32_t>* rope_rows_out) {
  StageBResult res;
  st.early_import = false;
  try {
    {
      ActiveGuard active(b);
      res.logits = images != nullptr ? b.PrefillMultimodal(ids, *images, nullptr, rope_rows_out) : b.Prefill(ids);
    }
    if (opt.gdn_handoff) {
      const auto t_gdn0 = Clock::now();
      if (!st.early_import) {
        ch.WaitBulk(st.call_id, std::chrono::milliseconds(opt.timeout_ms));
        if (!opt.skip_gdn_import) b.PpImportGdnAsync(bufs.GdnHand(), opt.split);
      }
      b.PpImportFence();
      res.gdn_wait_ms = Ms(t_gdn0, Clock::now());
    }
  } catch (...) {
    res.error = std::current_exception();
    std::fprintf(stderr, "[r4dx-pp] stage B failed: %s\n", WhatOf(res.error).c_str());
    ch.Poison("stage B failed: " + WhatOf(res.error));
    // An early GDN import (StageBOnLastChunk) may still be copying out of the hand-off buffer on the side stream: let it land before the
    // recovery (Reset) zeroes the state it writes.
    try {
      b.PpImportFence();
    } catch (...) {
    }
  }
  return res;
}

}  // namespace r4dx::model::pp
