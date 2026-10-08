#include "pp_model.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "pp_plan.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

// Thread name for the debugger / ETW, declared by hand rather than through <windows.h> (see tp_model.cpp).
extern "C" __declspec(dllimport) long __stdcall SetThreadDescription(void* thread, const wchar_t* description);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread();

namespace r4dx::model {

namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::chrono::seconds kShutdownWait{30};
// A sync-back larger than this (KV rows stage A is missing x 16 KiB x ...) is not worth a pinned buffer of its own: the
// call runs on the decode Model alone. 512 MiB = 32k rows of 8 attention layers.
constexpr size_t kMaxKvSyncBytes = 512ull * 1024 * 1024;

std::string GetEnvVar(const char* name) {
  char* v = nullptr;
  size_t len = 0;
  std::string out;
  if (_dupenv_s(&v, &len, name) == 0 && v != nullptr) out = v;
  std::free(v);
  return out;
}

bool EnvFlag(const char* name) {
  const std::string v = GetEnvVar(name);
  return v == "1" || v == "on" || v == "true";
}

std::string WhatOf(const std::exception_ptr& e) {
  try {
    std::rethrow_exception(e);
  } catch (const std::exception& x) {
    return x.what();
  } catch (...) {
    return "unknown exception";
  }
}

bool IsChannelPoisoned(const std::exception_ptr& e) {
  try {
    std::rethrow_exception(e);
  } catch (const pp::ChannelPoisoned&) {
    return true;
  } catch (...) {
    return false;
  }
}

VramReport MakeVramReport(int rank) {
  VramReport r;
  r.rank = rank;
  R4DX_HIP_CHECK(hipGetDevice(&r.device));
  size_t free_b = 0, total_b = 0;
  R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  r.used_gib = static_cast<double>(total_b - free_b) / kGiB;
  r.free_gib = static_cast<double>(free_b) / kGiB;
  r.total_gib = static_cast<double>(total_b) / kGiB;
  r.buffers_gib = static_cast<double>(core::DeviceBufferBytes(r.device)) / kGiB;
  return r;
}

}  // namespace

bool ResolvePpSwitch(int option) {
  if (option > 0) return true;
  if (option == 0) return false;
  const std::string env = GetEnvVar("R4DX_PP");
  const int v = pp::ParsePpEnable(env.c_str());
  if (v < 0) throw std::invalid_argument("R4DX_PP='" + env + "' is not 0 / off or 1 / on / 2");
  return v == 1;
}

// ---- load ---------------------------------------------------------------------------------------------------------

std::unique_ptr<PpModel> PpModel::Load(const ModelOptions& opts, const PpOptions& pp) {
  {
    const TpRankOptions d;
    if (opts.tp.world != d.world || opts.tp.rank != d.rank || opts.tp.comm != nullptr || opts.tp.shared_embed_host ||
        opts.tp.embed_device_resident_decided != -1 || !opts.tp.vision_weights_on_this_rank ||
        opts.tp.dflash_codebooks || opts.tp.submit_layers != 0 || opts.tp.max_inflight_units != 0) {
      throw std::invalid_argument("PpModel::Load: ModelOptions::tp must be default (the pipeline replaces TP)");
    }
  }
  if (pp.split < 0) throw std::invalid_argument("PpModel::Load: split must be >= 0 (0 = auto)");
  if (pp.min_rows < 1) throw std::invalid_argument("PpModel::Load: min_rows must be >= 1");
  if (pp.slots < 2 || pp.slots > 8) throw std::invalid_argument("PpModel::Load: slots must be in [2, 8]");
  if (pp.timeout_ms < 100) throw std::invalid_argument("PpModel::Load: timeout_ms must be >= 100");
  if (pp.submit_layers < 0 || pp.submit_layers > 64 || pp.max_inflight < 0 || pp.max_inflight > 64) {
    throw std::invalid_argument("PpModel::Load: submit_layers and max_inflight must be in [0, 64]");
  }
  if (opts.pp_emulate_split > 0) {
    throw std::invalid_argument("PpModel::Load: --pp 2 and the PP-emulate mode (R4DX_PP_EMULATE) are mutually exclusive");
  }
  // Stage B (decode) must be process ordinal 0 -- the ordinal every existing code path and every thread starts on -- and be
  // the physical headless card: HIP_VISIBLE_DEVICES=1,0 (docs/pp-prefill.md 1.2).
  if (const std::string why = pp::CheckDeviceOrder(GetEnvVar("HIP_VISIBLE_DEVICES")); !why.empty()) {
    throw std::runtime_error("PpModel::Load: " + why);
  }
  int visible = 0;
  R4DX_HIP_CHECK(hipGetDeviceCount(&visible));
  const int dev_b = 0, dev_a = pp.stage_a_device;
  if (visible < 2 || dev_a < 1 || dev_a >= visible) {
    throw std::runtime_error("PpModel::Load: the pipeline needs two visible HIP devices (stage A on ordinal " +
                             std::to_string(dev_a) + "), HIP_VISIBLE_DEVICES exposes " + std::to_string(visible));
  }
  {
    hipDeviceProp_t pb, pa;
    R4DX_HIP_CHECK(hipGetDeviceProperties(&pb, dev_b));
    R4DX_HIP_CHECK(hipGetDeviceProperties(&pa, dev_a));
    if (pb.pciDomainID == pa.pciDomainID && pb.pciBusID == pa.pciBusID) {
      throw std::runtime_error("PpModel::Load: ordinals 0 and " + std::to_string(dev_a) +
                               " are the same physical GPU (pci bus " + std::to_string(pb.pciBusID) + ")");
    }
    if (std::string(pb.gcnArchName) != std::string(pa.gcnArchName)) {
      throw std::runtime_error(std::string("PpModel::Load: the stages must run on GPUs of the same architecture, got ") +
                               pb.gcnArchName + " and " + pa.gcnArchName + " (the same code objects, the same bits)");
    }
    std::fprintf(stderr, "[r4dx-pp] stage B (decode) -> HIP device %d (%s, pci %02x:%02x); stage A -> HIP device %d (%s, pci %02x:%02x)\n",
                 dev_b, pb.name, pb.pciBusID, pb.pciDeviceID, dev_a, pa.name, pa.pciBusID, pa.pciDeviceID);
  }

  std::unique_ptr<PpModel> m(new PpModel());
  m->opts_ = opts;
  m->opts_.pp_emulate_split = 0;
  m->pp_ = pp;
  m->min_rows_ = pp.min_rows;
  m->verify_ = pp.verify || EnvFlag("R4DX_PP_VERIFY");

  // Stage B: the ordinary full Model, loaded on this (the facade) thread, as LocalTextModel's is.
  m->b_.emplace(Model::Load(m->opts_));
  Model& b = *m->b_;
  if (b.GetContainer().HasRotation()) {
    throw std::invalid_argument("PpModel::Load: not on a rotated (quant2) container: the stack-entry and -exit rotations "
                                "straddle the stages (docs/pp-prefill.md 4)");
  }
  const int64_t layers = b.GetContainer().NumLoadedLayers();
  m->split_ = pp.split > 0 ? pp.split : pp::DefaultSplit(b.DflashEnabled(), layers);
  if (!pp::ValidSplit(m->split_, layers)) {
    throw std::invalid_argument("PpModel::Load: split " + std::to_string(m->split_) + " must leave at least one of the " +
                                std::to_string(layers) + " loaded layers on each side");
  }
  const int64_t split = m->split_;
  // Stage A is loaded for the largest split it will ever run (PpOptions::reserve_split; tests move the split between
  // scenarios with SetSplit): the ring, the GDN buffers and the layer count are sized for it.
  m->reserve_split_ = std::max(split, static_cast<int64_t>(pp.reserve_split));
  if (!pp::ValidSplit(m->reserve_split_, layers)) {
    throw std::invalid_argument("PpModel::Load: reserve_split " + std::to_string(m->reserve_split_) +
                                " must leave at least one of the " + std::to_string(layers) + " loaded layers on stage B");
  }
  const int64_t hidden = b.Config().hidden_size;
  m->targets_ = b.DflashTargetLayers();
  std::vector<int64_t> targets_a = m->TargetsBelow(split);  // the drafter's target layers that stage A owns (a prefix)
  int64_t dfl_cols_max = 0;
  for (const int64_t l : m->targets_) {
    if (l < m->reserve_split_) ++dfl_cols_max;
  }
  dfl_cols_max = std::max<int64_t>(dfl_cols_max, 4);  // room for a test capture (AttachDflashFeatureCapture)
  const std::vector<int64_t> attn = b.PpAttnLayers(split);
  const std::vector<int64_t> attn_max = b.PpAttnLayers(m->reserve_split_);
  const size_t cap = pp::MaxSlotBytes(b.PrefillChunkRows(), hidden, dfl_cols_max, static_cast<int64_t>(attn_max.size()),
                                      b.PpKvBlockSize(), b.PpKvBlockStrideBytes());
  m->hidden_ = hidden;
  std::vector<uint8_t*> ptrs;
  for (int i = 0; i < pp.slots; ++i) {
    m->slots_.emplace_back(cap, hipHostMallocPortable);
    ptrs.push_back(m->slots_.back().data());
  }
  const size_t gdn_bytes = b.PpGdnWireBytes(m->reserve_split_);
  m->gdn_hand_ = core::PinnedBuffer<uint8_t>(gdn_bytes, hipHostMallocPortable);
  m->gdn_sync_ = core::PinnedBuffer<uint8_t>(gdn_bytes, hipHostMallocPortable);
  m->channel_ = std::make_unique<pp::StageChannel>(ptrs, cap);
  std::fprintf(stderr,
               "[r4dx-pp] split k=%lld (%lld attention layers and %zu DFlash columns on stage A), %d slots of %.1f MiB, GDN "
               "hand-off %.1f MiB, min rows %d, bounded submission %d/%d on stage A%s\n",
               static_cast<long long>(split), static_cast<long long>(attn.size()), targets_a.size(), pp.slots,
               static_cast<double>(cap) / (1024.0 * 1024.0), static_cast<double>(gdn_bytes) / (1024.0 * 1024.0), pp.min_rows,
               pp.submit_layers, pp.max_inflight, m->verify_ ? ", VERIFY on" : "");

  // Stage A's thread, and its Model, built there.
  m->worker_ = std::make_unique<tp::RankWorker>(
      1,
      [dev_a] {
        R4DX_HIP_CHECK(hipSetDevice(dev_a));
        (void)SetThreadDescription(GetCurrentThread(), L"r4dx-pp-stage-a");
      },
      &m->done_, m->timing_);
  PpModel* const self = m.get();
  const auto kNoStall = tp::ProgressWatchdog::kNoStallLimit;
  {
    ModelOptions ao = m->opts_;
    ao.layer_limit = m->reserve_split_ + 1;  // [0, split) run; layer `split` only supplies the input_layernorm layer split - 1 fuses
    ao.mtp_draft_k = 0;
    ao.dflash_container.clear();
    ao.dflash_draft_k = 0;
    ao.prompt_checkpoint = false;
    ao.pp = 0;
    const int stage_b_chunk = b.PrefillChunkRows();
    const bool stage_b_i8 = b.PrefillInt8Enabled();
    const int submit_layers = pp.submit_layers, inflight = pp.max_inflight;
    const int64_t reserve = m->reserve_split_;
    m->worker_->Post([self, ao, split, reserve, stage_b_chunk, stage_b_i8, targets_a, gdn_bytes, submit_layers, inflight] {
      self->a_.emplace(Model::Load(ao));
      Model& a = *self->a_;
      if (a.PrefillChunkRows() != stage_b_chunk) {
        throw std::runtime_error("PpModel::Load: the stages chose different prefill chunk sizes (" +
                                 std::to_string(a.PrefillChunkRows()) + " vs " + std::to_string(stage_b_chunk) + ")");
      }
      if (a.PrefillInt8Enabled() != stage_b_i8) {
        throw std::runtime_error("PpModel::Load: the stages disagree on int8 prefill (A " +
                                 std::to_string(a.PrefillInt8Enabled()) + ", B " + std::to_string(stage_b_i8) +
                                 "): the bits would differ");
      }
      if (a.PpGdnWireBytes(reserve) != gdn_bytes) {
        throw std::runtime_error("PpModel::Load: the stages disagree on the GDN hand-off size");
      }
      a.PpAttach(self->StageSetup(Model::PpRole::kStageA));
      a.PpEnableBounding(submit_layers, inflight);
      if (!targets_a.empty()) a.AttachDflashFeatureCapture(targets_a);
    });
    const std::exception_ptr err = self->JoinStageA(kNoStall);
    if (err) std::rethrow_exception(err);
  }
  b.PpAttach(m->StageSetup(Model::PpRole::kStageB));
  for (const VramReport& r : m->Vram()) {
    std::fprintf(stderr, "[r4dx-pp] stage %c (rank %d, HIP device %d): %.2f GiB used of %.2f GiB (%.2f GiB in this process's buffers)\n",
                 r.rank == 0 ? 'B' : 'A', r.rank, r.device, r.used_gib, r.total_gib, r.buffers_gib);
  }

  // Warm-up (docs/pp-prefill.md 4): one pipelined call over two super-chunks and a 8-row tail touches every first-use path
  // of both stages (kernel modules, GdnControlCache keys, the int8 / M = 256 launches, the ring), then everything is reset.
  {
    std::vector<int32_t> ids(520);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(300 + (i * 37) % 5000);
    const auto t0 = Clock::now();
    (void)m->PipelinedPrefill(ids, nullptr);
    m->Reset();
    std::fprintf(stderr, "[r4dx-pp] warm-up pipelined prefill (520 rows): %.0f ms\n", Ms(t0, Clock::now()));
    m->stats_ = Stats{};
    m->channel_->ResetStats();
  }
  m->dflash_injection_ = true;
  return m;
}

PpModel::~PpModel() {
  if (!worker_) return;
  if (channel_) channel_->Poison("shutdown");
  const auto wait_idle = [&](const char* what) {
    std::unique_lock<std::mutex> lk(done_.mu);
    if (!done_.cv.wait_for(lk, kShutdownWait, [&] { return worker_->Idle(); })) {
      std::fprintf(stderr, "[r4dx-pp] stage A did not finish %s in 30 s at shutdown\n", what);
      std::fflush(stderr);
      std::quick_exit(3);
    }
  };
  wait_idle("its command");
  if (a_) {
    // Stage A's Model (and every device buffer it owns) dies on its own thread, as a TP rank's does (docs/tp.md 2.6).
    try {
      worker_->Post([this] { a_.reset(); });
    } catch (...) {
    }
    wait_idle("tearing down");
  }
  if (b_) {
    try {
      b_->PpDetach();
    } catch (...) {
    }
  }
}

// ---- stage A plumbing ------------------------------------------------------------------------------------------------

std::exception_ptr PpModel::JoinStageA(std::chrono::milliseconds stall) {
  tp::ProgressWatchdog wd(stall);
  const tp::WaitResult r = tp::WaitAllIdle({worker_.get()}, done_, timing_, wd);
  if (r == tp::WaitResult::kStalled) {
    state_ = State::kFatal;
    if (channel_) channel_->Poison("stage A made no progress");
    std::fprintf(stderr, "[r4dx-pp] stage A made no progress for %lld ms (posted %llu, done %llu): fatal\n",
                 static_cast<long long>(stall.count()), static_cast<unsigned long long>(worker_->Posted()),
                 static_cast<unsigned long long>(worker_->Done()));
    throw std::runtime_error("pp: stage A made no progress; restart the process");
  }
  return worker_->TakeError();
}

void PpModel::RunA(const std::function<void(Model&)>& fn, std::chrono::milliseconds stall) {
  PpModel* const self = this;
  worker_->Post([self, fn] { fn(*self->a_); });
  const std::exception_ptr err = JoinStageA(stall);
  if (err) std::rethrow_exception(err);
}

void PpModel::RunOnStageA(const std::function<void(Model&)>& fn) {
  if (state_ == State::kFatal) throw std::runtime_error("pp: fatal, restart the process");
  RunA(fn, std::chrono::seconds(60));
}

void PpModel::RequireReady() const {
  if (state_ == State::kFatal) throw std::runtime_error("pp: fatal, restart the process");
  if (state_ == State::kNeedsRecovery) {
    throw std::runtime_error("pp: a pipelined prefill failed earlier; call Reset() first");
  }
}

void PpModel::Fail(std::exception_ptr cause) {
  if (state_ != State::kFatal) state_ = State::kNeedsRecovery;
  std::rethrow_exception(cause);
}

// ---- identity -----------------------------------------------------------------------------------------------------------

int64_t PpModel::VisionMergeSize() const {
  return b_->HasVision() ? static_cast<int64_t>(b_->GetContainer().Vision().config.spatial_merge_size) : 2;
}

std::vector<VramReport> PpModel::Vram() const {
  PpModel* const self = const_cast<PpModel*>(this);
  VramReport rb = MakeVramReport(0);
  VramReport ra;
  ra.rank = 1;
  if (state_ != State::kFatal) {
    self->RunA([&ra](Model&) { ra = MakeVramReport(1); }, std::chrono::seconds(60));
  }
  return {rb, ra};
}

// ---- sequence state -----------------------------------------------------------------------------------------------------

void PpModel::Reset() {
  if (state_ == State::kFatal) throw std::runtime_error("pp: fatal, restart the process");
  try {
    b_->PpSetActive(false);
    channel_->Reset();
    RunA([](Model& a) { a.Reset(); }, std::chrono::seconds(60));
    b_->Reset();
  } catch (...) {
    if (state_ != State::kFatal) state_ = State::kNeedsRecovery;
    throw;
  }
  tracker_.Reset();
  state_ = State::kReady;
}

void PpModel::SetDflashInjectionEnabled(bool enabled) {
  dflash_injection_ = enabled;
  b_->SetDflashInjectionEnabled(enabled);  // stage A takes the policy with every pipelined call
}

void PpModel::SaveCheckpoint() {
  RequireReady();
  b_->SaveCheckpoint();  // B alone holds the whole conversation's state; the mirror is untouched
}

void PpModel::RestoreCheckpoint() {
  RequireReady();
  tracker_.BRestored();
  b_->RestoreCheckpoint();
}

void PpModel::EncodeImages(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
                           ImageRows* out, vision::VisionEncodeStats* stats) {
  RequireReady();
  b_->EncodeImages(pixel_values, total_patches, grids, &out->dev, stats);
  int64_t rows = 0;
  const int64_t merge = VisionMergeSize();
  for (const vision::GridThw& g : grids) rows += g.MergedTokenCount(merge);
  out->SetFilled(/*on_host=*/false, rows);
}

// ---- prefill ---------------------------------------------------------------------------------------------------------------

std::vector<float> PpModel::Prefill(const std::vector<int32_t>& token_ids) {
  RequireReady();
  if (token_ids.empty()) throw std::runtime_error("PpModel::Prefill: token_ids is empty");
  if (!pp::ShouldPipeline(static_cast<int64_t>(token_ids.size()), min_rows_)) {
    ++stats_.b_only_calls;
    NoteBOnly();
    return b_->Prefill(token_ids);
  }
  return PipelinedPrefill(token_ids, nullptr);
}

std::vector<float> PpModel::PrefillMultimodal(const std::vector<int32_t>& token_ids, const std::vector<ImageSpan>& images) {
  RequireReady();
  if (token_ids.empty()) throw std::runtime_error("PpModel::PrefillMultimodal: token_ids is empty");
  if (!pp::ShouldPipeline(static_cast<int64_t>(token_ids.size()), min_rows_)) {
    ++stats_.b_only_calls;
    NoteBOnly();
    return b_->PrefillMultimodal(token_ids, images);
  }
  return PipelinedPrefill(token_ids, &images);
}

std::vector<float> PpModel::PipelinedPrefill(const std::vector<int32_t>& ids, const std::vector<ImageSpan>* images) {
  RequireReady();
  Model& b = *b_;
  const auto t_call0 = Clock::now();
  b.PpPrepareCall();  // move the live GDN state to window 0 if a speculative round left it elsewhere (before the export)
  const int64_t p0 = b.PositionCount();
  const pp::MirrorTracker::SyncPlan plan =
      fault_ == TestFault::kSkipSyncBack ? pp::MirrorTracker::SyncPlan{} : tracker_.PlanSync(p0);
  const size_t kv_need = plan.kv_row1 > plan.kv_row0 ? b.PpKvWireBytes(split_, plan.kv_row0, plan.kv_row1) : 0;
  if (kv_need > kMaxKvSyncBytes) {
    // Stage A is that far behind: the sync-back would dwarf the gain. The decode Model runs the call alone, which leaves
    // the mirror stale exactly as a decode step does.
    ++stats_.b_only_calls;
    NoteBOnly();
    return images != nullptr ? b.PrefillMultimodal(ids, *images) : b.Prefill(ids);
  }

  auto st = std::make_shared<CallState>();
  st->ids = ids;
  st->multimodal = images != nullptr;
  st->sync = b.PpGetSyncState();
  st->dflash_injection = dflash_injection_;
  st->plan = plan;
  st->call_id = ++call_id_;
  st->verify = verify_;

  // ---- the sync-back (B -> A): host-side export now, A imports at the start of its command -------------------------
  if (plan.gdn) b.PpExportGdn(gdn_sync_.data(), split_);
  if (kv_need > 0) {
    if (kv_sync_.bytes() < kv_need) {
      kv_sync_ = core::PinnedBuffer<uint8_t>(std::max<size_t>(kv_need, 16ull << 20), hipHostMallocPortable);
    }
    b.PpExportKvRows(kv_sync_.data(), split_, plan.kv_row0, plan.kv_row1);
  }
  if (st->verify) st->b_digest_sync = b.PpLiveDigest(split_);

  // ---- stage A's image spans: rows A can read (host); B keeps the caller's (device rows) ------------------------------
  if (images != nullptr) {
    const int64_t hidden = b.Config().hidden_size;
    st->a_spans = *images;
    st->a_rows.reserve(images->size());
    for (ImageSpan& sp : st->a_spans) {
      if (sp.embeds_on_host) continue;
      const size_t n = static_cast<size_t>(sp.tokens * hidden);
      st->a_rows.emplace_back(n);
      R4DX_HIP_CHECK(hipMemcpy(st->a_rows.back().data(), sp.embeds, n * sizeof(uint16_t), hipMemcpyDeviceToHost));
      sp.embeds = st->a_rows.back().data();
      sp.embeds_on_host = true;
    }
  }
  const double sync_ms = Ms(t_call0, Clock::now());

  // ---- stage A's command ------------------------------------------------------------------------------------------------
  pp::StageChannel* const ch = channel_.get();
  channel_->BeginCall(st->call_id);
  early_import_ = false;
  cur_call_ = st->call_id;
  PpModel* const self = this;
  const int64_t split = split_;
  worker_->Post([self, st, ch, split] {
    try {
      Model& a = *self->a_;
      const auto t0 = Clock::now();
      a.PpSetSyncState(st->sync);
      a.SetDflashInjectionEnabled(st->dflash_injection);
      if (st->plan.gdn) a.PpImportGdn(self->gdn_sync_.data(), split);
      if (st->plan.kv_row1 > st->plan.kv_row0) a.PpImportKvRows(self->kv_sync_.data(), split, st->plan.kv_row0, st->plan.kv_row1);
      st->a_sync_ms = Ms(t0, Clock::now());
      if (st->verify) {
        const auto d = a.PpLiveDigest(split);
        if (d != st->b_digest_sync) {
          throw std::runtime_error("R4DX_PP_VERIFY: stage A's state after the sync-back differs from stage B's (position " +
                                   std::to_string(st->sync.pos) + ")");
        }
      }
      if (st->multimodal) {
        (void)a.PrefillMultimodal(st->ids, st->a_spans);
      } else {
        (void)a.Prefill(st->ids);
      }
      // The GDN live state of A's layers follows the last chunk (compactly); B imports it while it computes its own last
      // chunk when it is ready in time (OnLastChunk), else right after.
      const auto t_exp = Clock::now();
      a.PpExportGdn(self->gdn_hand_.data(), split);
      st->a_gdn_export_ms = Ms(t_exp, Clock::now());
      ch->PublishBulk(st->call_id);
      if (st->verify) st->a_digest_end = a.PpLiveDigest(split);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[r4dx-pp] stage A failed: %s\n", e.what());
      ch->Poison(std::string("stage A failed: ") + e.what());
      throw;
    } catch (...) {
      ch->Poison("stage A failed");
      throw;
    }
  });

  // ---- stage B, on this thread ---------------------------------------------------------------------------------------------
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
  std::vector<float> logits;
  std::exception_ptr berr;
  double gdn_wait_ms = 0;
  const auto t_b0 = Clock::now();
  try {
    {
      ActiveGuard active(b);
      logits = images != nullptr ? b.PrefillMultimodal(ids, *images) : b.Prefill(ids);
    }
    const auto t_gdn0 = Clock::now();
    if (!early_import_) {
      channel_->WaitBulk(st->call_id, std::chrono::milliseconds(pp_.timeout_ms));
      if (fault_ != TestFault::kSkipGdnImport) b.PpImportGdnAsync(gdn_hand_.data(), split_);
    }
    b.PpImportFence();
    gdn_wait_ms = Ms(t_gdn0, Clock::now());
  } catch (...) {
    berr = std::current_exception();
    std::fprintf(stderr, "[r4dx-pp] stage B failed: %s\n", WhatOf(berr).c_str());
    channel_->Poison("stage B failed: " + WhatOf(berr));
    // An early GDN import (OnLastChunk) may still be copying out of gdn_hand_ on the side stream: let it land before the
    // recovery (Reset) zeroes the state it writes.
    try {
      b.PpImportFence();
    } catch (...) {
    }
  }
  const auto t_tail0 = Clock::now();

  // ---- join stage A -------------------------------------------------------------------------------------------------------------
  std::exception_ptr aerr = JoinStageA(std::chrono::seconds(60));
  if (berr || aerr) {
    // The root cause: stage A's own error unless stage B's is not just the channel telling it A had failed.
    if (aerr && (!berr || IsChannelPoisoned(berr))) Fail(aerr);
    Fail(berr);
  }
  const int64_t p_end = b.PositionCount();
  if (st->verify) {
    const auto db = b.PpLiveDigest(split_);
    if (db != st->a_digest_end) {
      state_ = State::kNeedsRecovery;
      throw std::runtime_error("R4DX_PP_VERIFY: stage A's live state at the end of the call (position " +
                               std::to_string(p_end) + ") differs from stage B's after the hand-off");
    }
  }
  tracker_.AfterPipelined(p_end);

  ++stats_.pipelined_calls;
  if (plan.gdn) ++stats_.sync_gdn;
  stats_.sync_kv_rows += std::max<int64_t>(0, plan.kv_row1 - plan.kv_row0);
  if (early_import_) ++stats_.early_gdn_imports;
  stats_.last_sync_ms = sync_ms + st->a_sync_ms;
  stats_.last_tail_ms = Ms(t_tail0, Clock::now());
  stats_.last_total_ms = Ms(t_call0, Clock::now());
  stats_.last_gdn_export_ms = st->a_gdn_export_ms;
  stats_.last_gdn_wait_ms = gdn_wait_ms;
  (void)t_b0;
  return logits;
}

// Stage B's last chunk of a call has taken its slot: if stage A's GDN export is already complete, start importing it on
// B's side stream now, so it overlaps this chunk's compute (the import writes only layers < split's state, which B's own
// layers [split, N) never read). Otherwise PipelinedPrefill imports it after the chunk.
void PpModel::OnLastChunk() {
  if (fault_ == TestFault::kSkipGdnImport) return;
  if (channel_->BulkReady(cur_call_)) {
    b_->PpImportGdnAsync(gdn_hand_.data(), split_);
    early_import_ = true;
  }
}

// ---- everything that moves B alone ------------------------------------------------------------------------------------------

std::vector<float> PpModel::DecodeStep(int32_t token_id) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStep(token_id);
}
int32_t PpModel::DecodeStepGreedy(int32_t token_id) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepGreedy(token_id);
}
int32_t PpModel::DecodeStepGreedyOverlap(int32_t token_id, const std::function<void()>& while_busy) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepGreedyOverlap(token_id, while_busy);
}
int32_t PpModel::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepSampled(token_id, params, rng);
}
std::vector<int32_t> PpModel::DecodeStepMtpGreedy(int32_t token_id, int64_t k) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepMtpGreedy(token_id, k);
}
std::vector<int32_t> PpModel::DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                                   std::mt19937_64& rng) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepMtpSampled(token_id, k, params, rng);
}
std::vector<int32_t> PpModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                     int64_t* walk_len_out) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepDflashGreedy(token_id, k, p_min, n_min, walk_len_out);
}
std::vector<int32_t> PpModel::DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                      const kernels::SampleParams& params, std::mt19937_64& rng,
                                                      int64_t* walk_len_out) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepDflashSampled(token_id, k, p_min, n_min, params, rng, walk_len_out);
}
StepProfile PpModel::DecodeStepProfiled(int32_t token_id) {
  RequireReady();
  NoteBOnly();
  return b_->DecodeStepProfiled(token_id);
}
StepProfile PpModel::PrefillProfiled(const std::vector<int32_t>& token_ids) {
  RequireReady();
  NoteBOnly();
  return b_->PrefillProfiled(token_ids);
}

Model::PpStageSetup PpModel::StageSetup(Model::PpRole role) {
  Model::PpStageSetup st;
  st.role = role;
  st.split = split_;
  st.channel = channel_.get();
  st.timeout_ms = pp_.timeout_ms;
  if (role == Model::PpRole::kStageA) {
    tp::RankWorker* w = worker_.get();
    st.on_chunk = [w] { w->Heartbeat().fetch_add(1, std::memory_order_relaxed); };
  } else {
    st.before_last_chunk = [this] { OnLastChunk(); };
  }
  return st;
}

std::vector<int64_t> PpModel::TargetsBelow(int64_t split) const {
  std::vector<int64_t> out;
  for (const int64_t l : targets_) {
    if (l < split) out.push_back(l);
  }
  return out;
}

// Stage A captures the DFlash target layers below the split (a prefix of B's ascending list: its columns land in
// columns [0, n) of B's feature buffer).
void PpModel::UpdateStageACapture() {
  const std::vector<int64_t> below = TargetsBelow(split_);
  RunA([below](Model& a) {
         if (below.empty()) {
           a.DetachDflashFeatureCapture();
         } else {
           a.AttachDflashFeatureCapture(below);
         }
       },
       std::chrono::seconds(60));
}

void PpModel::SetSplit(int64_t k) {
  // Not RequireReady(): SetSplit ends in Reset(), which is also the recovery from kNeedsRecovery -- and the identity test's
  // negative controls end a variant exactly there (a failed call), then move the split for the next one.
  if (state_ == State::kFatal) throw std::runtime_error("pp: fatal, restart the process");
  if (!pp::ValidSplit(k, NumLoadedLayers()) || k > reserve_split_) {
    throw std::invalid_argument("PpModel::SetSplit: split " + std::to_string(k) + " must be in [1, " +
                                std::to_string(reserve_split_) + "] (stage A was loaded for splits up to " +
                                std::to_string(reserve_split_) + ")");
  }
  const int64_t old = split_;
  split_ = k;
  try {
    RunA([this](Model& a) { a.PpAttach(StageSetup(Model::PpRole::kStageA)); }, std::chrono::seconds(60));
    b_->PpAttach(StageSetup(Model::PpRole::kStageB));
    UpdateStageACapture();
  } catch (...) {
    split_ = old;
    state_ = State::kNeedsRecovery;
    throw;
  }
  Reset();  // the mirror covered layers [0, old); it knows nothing of [old, k)
}

void PpModel::SetMinRows(int64_t min_rows) {
  if (min_rows < 1) throw std::invalid_argument("PpModel::SetMinRows: min_rows must be >= 1");
  min_rows_ = min_rows;
}

void PpModel::AttachDflashFeatureCapture(std::vector<int64_t> target_layers) {
  RequireReady();
  const std::vector<int64_t> old = targets_;
  targets_ = target_layers;
  const int64_t cols = static_cast<int64_t>(TargetsBelow(reserve_split_).size());
  const size_t need = pp::MaxSlotBytes(b_->PrefillChunkRows(), hidden_, cols,
                                       static_cast<int64_t>(b_->PpAttnLayers(reserve_split_).size()), b_->PpKvBlockSize(),
                                       b_->PpKvBlockStrideBytes());
  if (need > channel_->Capacity()) {
    targets_ = old;
    throw std::invalid_argument("PpModel::AttachDflashFeatureCapture: " + std::to_string(cols) +
                                " DFlash columns below the largest split do not fit the ring slots");
  }
  b_->AttachDflashFeatureCapture(std::move(target_layers));
  UpdateStageACapture();
}

// ---- diagnostics ---------------------------------------------------------------------------------------------------------------

PpModel::Stats PpModel::GetStats() const {
  Stats s = stats_;
  const pp::StageChannel::Stats c = channel_->GetStats();
  s.chunks = c.taken;
  s.stage_a_wait_ms = c.producer_wait_ms;
  s.stage_b_wait_ms = c.consumer_wait_ms;
  return s;
}

std::string PpModel::StatsLine() const {
  const Stats s = GetStats();
  char buf[640];
  std::snprintf(buf, sizeof(buf),
                "pp: split k=%lld, %lld pipelined / %lld decode-only prefill calls, %lld chunks through the ring, sync-backs: %lld "
                "GDN + %lld KV rows, GDN import overlapped the last chunk in %lld calls; stage A blocked %.0f ms on a full "
                "ring, stage B %.0f ms on an empty one; last call %.1f ms (sync %.1f ms, tail %.1f ms, GDN export %.1f ms, B waited %.1f ms for it)",
                static_cast<long long>(split_), static_cast<long long>(s.pipelined_calls), static_cast<long long>(s.b_only_calls),
                static_cast<long long>(s.chunks), static_cast<long long>(s.sync_gdn), static_cast<long long>(s.sync_kv_rows),
                static_cast<long long>(s.early_gdn_imports), s.stage_a_wait_ms, s.stage_b_wait_ms, s.last_total_ms,
                s.last_sync_ms, s.last_tail_ms, s.last_gdn_export_ms, s.last_gdn_wait_ms);
  return buf;
}

}  // namespace r4dx::model
