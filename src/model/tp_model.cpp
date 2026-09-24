#include "tp_model.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "local_text_model.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "tp/tp_comm_noop.h"

// Thread names for the debugger / ETW (docs/tp.md 2.2: "r4dx-tp-rank<r>"), declared by hand rather
// than through <windows.h>, whose macros would leak into every header below. Same signature as
// processthreadsapi.h (HRESULT = long, HANDLE = void*, PCWSTR = const wchar_t*).
extern "C" __declspec(dllimport) long __stdcall SetThreadDescription(void* thread, const wchar_t* description);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread();

namespace r4dx::model {

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::chrono::seconds kShutdownWait{30};

const char* ModeName(TpOptions::Mode m) {
  switch (m) {
    case TpOptions::Mode::kReal: return "real";
    case TpOptions::Mode::kEmulate: return "emulate";
    case TpOptions::Mode::kNoop: return "noop";
  }
  return "?";
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

bool IsTpAborted(const std::exception_ptr& e) {
  try {
    std::rethrow_exception(e);
  } catch (const core::TpAbortedError&) {
    return true;
  } catch (...) {
    return false;
  }
}

// hipGetDeviceCount on a helper thread: the facade thread itself makes no HIP call (docs/tp.md 2.1,
// 2.9 step 1).
int VisibleDeviceCount() {
  int n = 0;
  hipError_t err = hipSuccess;
  std::thread t([&] {
    err = hipGetDeviceCount(&n);
    (void)hipGetLastError();
  });
  t.join();
  if (err != hipSuccess) n = 0;
  return n;
}

// The process's HIP_VISIBLE_DEVICES, or "" (a CRT call, not HIP). _dupenv_s: getenv is deprecated
// under the MSVC CRT.
std::string HipVisibleDevices() {
  char* v = nullptr;
  size_t len = 0;
  std::string out;
  if (_dupenv_s(&v, &len, "HIP_VISIBLE_DEVICES") == 0 && v != nullptr) out = v;
  std::free(v);
  return out;
}

struct DeviceProbe {
  std::string name, arch;
  int pci_domain = -1, pci_bus = -1, pci_device = -1;
  int can_map_host = 0;
  size_t total = 0, free_bytes = 0;
  int wall_clock_khz = 0;
};

}  // namespace

// ---- load ---------------------------------------------------------------------------------------

std::unique_ptr<TpModel> TpModel::Load(const ModelOptions& opts, const TpOptions& tp) {
  // 1. Validate (docs/tp.md 2.9 step 1), including the staged rejections -- enforced here and not
  //    only in the arg parsers, so a test or tool that builds a TpModel directly cannot reach a path
  //    whose TP hooks are not in yet.
  if (tp.world != 2) {
    throw std::invalid_argument("TpModel::Load: TpOptions::world must be 2, got " + std::to_string(tp.world));
  }
  {
    const TpRankOptions d;
    if (opts.tp.world != d.world || opts.tp.rank != d.rank || opts.tp.comm != nullptr || opts.tp.shared_embed_host ||
        opts.tp.embed_device_resident_decided != -1 || !opts.tp.vision_weights_on_this_rank ||
        opts.tp.dflash_codebooks) {
      throw std::invalid_argument("TpModel::Load: ModelOptions::tp must be default -- TpModel fills in every rank's "
                                  "TpRankOptions itself");
    }
  }
  if (opts.mtp_draft_k > 0) {
    throw core::TpUnsupportedError("TpModel::Load: MTP (--mtp > 0) is not supported under tensor parallelism yet "
                                   "(docs/tp.md P5)");
  }
  if (!opts.dflash_container.empty()) {
    throw core::TpUnsupportedError("TpModel::Load: DFlash2 (--dflash) is not supported under tensor parallelism "
                                   "yet (docs/tp.md P5)");
  }
  if (opts.vision == ModelOptions::VisionMode::kOn) {
    throw core::TpUnsupportedError("TpModel::Load: --vision on is not supported under tensor parallelism yet "
                                   "(docs/tp.md P5)");
  }
  // The load-time warm-up (2.9 step 9, Model::TpWarmup) prefills one full 64-token chunk and
  // decodes one token: 65 positions of KV. Refuse a smaller context here, by name, instead of
  // failing inside the warm-up with the KV cache's capacity error (Appendix B N53).
  constexpr int64_t kWarmupPositions = 65;
  if (opts.max_ctx < kWarmupPositions) {
    throw std::invalid_argument("TpModel::Load: --max-ctx must be at least " + std::to_string(kWarmupPositions) +
                                " under tensor parallelism (the load-time warm-up prefills a 64-token chunk and "
                                "decodes one token), got " + std::to_string(opts.max_ctx));
  }
  if (tp.ar_timeout_ms < tp::kArTimeoutMinMs || tp.ar_timeout_ms > tp::kArTimeoutMaxMs) {
    throw std::invalid_argument("TpModel::Load: ar_timeout_ms must be in [10, 1500], got " +
                                std::to_string(tp.ar_timeout_ms));
  }
  (void)tp::MailboxLayout::Make(tp.ar_nb_small, tp.ar_nb_large);  // throws unless both nb are in [1, 64]
  if (tp.submit_layers < 0 || tp.submit_layers > 64 || tp.max_inflight_units < 0 || tp.max_inflight_units > 64) {
    throw std::invalid_argument("TpModel::Load: submit_layers and max_inflight_units must be in [0, 64], got " +
                                std::to_string(tp.submit_layers) + " and " + std::to_string(tp.max_inflight_units));
  }
  const bool noop = tp.mode == TpOptions::Mode::kNoop;
  const bool real = tp.mode == TpOptions::Mode::kReal;
  if (noop && (tp.noop_rank < 0 || tp.noop_rank >= tp.world)) {
    throw std::invalid_argument("TpModel::Load: noop_rank must be 0 or 1, got " + std::to_string(tp.noop_rank));
  }
  if (!noop && tp.noop_rank != 0) {
    throw std::invalid_argument("TpModel::Load: noop_rank is a --tp-mode noop option");
  }
  if (tp.fault_rank != -1) {
    if (tp.fault_rank < 0 || tp.fault_rank >= tp.world || tp.fault_at_allreduce <= 0 ||
        (tp.fault_kind != tp::kFaultThrow && tp.fault_kind != tp::kFaultStall)) {
      throw std::invalid_argument("TpModel::Load: fault injection needs 0 <= fault_rank < 2, fault_at_allreduce > 0 "
                                  "and fault_kind 0 or 1");
    }
    if (noop) {
      throw std::invalid_argument("TpModel::Load: fault injection needs a real transport or the emulated one "
                                  "(--tp-mode emulate); NoopComm moves nothing to fault");
    }
  }

  // Devices (docs/tp.md 9.2): emulate and noop use ONE device, the last visible ordinal by default
  // -- so HIP_VISIBLE_DEVICES=1 puts every single-device TP run on physical device 1. Real mode uses
  // two, by default in descending order from the last visible one: with HIP_VISIBLE_DEVICES unset,
  // rank 0 = device 1 (headless), rank 1 = device 0 (the desktop card).
  const int visible = VisibleDeviceCount();
  if (visible < 1) throw core::TpError("TpModel::Load: no visible HIP device");
  std::vector<int> devices;  // per rank slot
  if (real) {
    if (visible < 2) {
      const std::string hvd = HipVisibleDevices();
      throw core::TpError("TpModel::Load: --tp 2 --tp-mode real needs two visible HIP devices; HIP_VISIBLE_DEVICES=" +
                          (hvd.empty() ? std::string("<unset>") : hvd) + " exposes " + std::to_string(visible) +
                          ". Unset it (or set it to 0,1).");
    }
    if (tp.devices.empty()) {
      devices = {visible - 1, visible - 2};
    } else if (tp.devices.size() != 2 || tp.devices[0] == tp.devices[1]) {
      throw std::invalid_argument("TpModel::Load: --tp-mode real needs --tp-devices with two different ordinals "
                                  "(rank 0, rank 1), or auto");
    } else {
      devices = tp.devices;
    }
  } else {
    int device = visible - 1;
    if (!tp.devices.empty()) {
      if (tp.devices.size() > 2 || (tp.devices.size() == 2 && (noop || tp.devices[0] != tp.devices[1]))) {
        throw std::invalid_argument(std::string("TpModel::Load: --tp-mode ") + ModeName(tp.mode) +
                                    " runs on ONE device; give --tp-devices a single ordinal");
      }
      device = tp.devices[0];
    }
    devices.assign(noop ? 1 : 2, device);
  }
  for (int d : devices) {
    if (d < 0 || d >= visible) {
      throw std::invalid_argument("TpModel::Load: HIP device " + std::to_string(d) + " is not visible (" +
                                  std::to_string(visible) + " visible device(s))");
    }
  }
  if (opts.vision == ModelOptions::VisionMode::kAuto) {
    std::fprintf(stderr, "[r4dx-tp] --vision auto loads text-only under tensor parallelism until docs/tp.md P5\n");
  }

  // Everything the load commands' rank closures touch is declared BEFORE `m` (Appendix B N53): if a
  // command throws -- including the progress watchdog's TpTimeoutError while a rank is still inside
  // it -- `m` is destroyed first, and ~TpModel waits for every rank to go idle (or ends the process)
  // before these are.
  const int n_slots = noop ? 1 : 2;
  std::vector<DeviceProbe> probe(static_cast<size_t>(n_slots));
  std::vector<double> clock_khz(static_cast<size_t>(n_slots), 0.0);
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> embed;
  bool resident = opts.embed_device_resident;
  double latency_us = -1.0;
  struct Caps {
    std::string id;
    int64_t image_token = 0, merge = 2, layers = 0, vocab = 0, hidden = 0;
    bool vision = false, mtp = false, mtp_reduced = false, dflash = false;
  };
  std::vector<Caps> caps(static_cast<size_t>(n_slots));
  ModelConfig rank0_global_config;

  std::unique_ptr<TpModel> m(new TpModel());
  m->opts_ = opts;
  m->tp_ = tp;
  m->stall_limit_ = noop ? tp::ProgressWatchdog::kNoStallLimit : std::chrono::milliseconds(60000);
  const auto kNoStall = tp::ProgressWatchdog::kNoStallLimit;  // load commands move no heartbeat (N8)

  // The facade must never drop the LAST reference to the pinned embedding table: that would
  // hipHostFree it on a thread with no hipSetDevice (docs/tp.md 2.1). On success the reference is
  // dropped on rank 0's thread at the end of Load; on any failure this guard (destroyed before `m`)
  // does the same, and if rank 0 cannot run a command any more (a stall), it leaks the table rather
  // than free it here.
  struct OnExit {
    std::function<void()> f;
    ~OnExit() { f(); }
  } release_embed{[&] {
    if (!embed) return;
    if (m->state_ != State::kFatal) {
      try {
        m->Run({0}, [&embed](RankSlot&) { embed.reset(); }, CmdKind::kPlain, kNoStall);
      } catch (...) {
      }
    }
    if (embed) (void)new std::shared_ptr<const core::PinnedBuffer<uint16_t>>(std::move(embed));
  }};

  // 2. Spawn the rank threads (noop: one, for rank noop_rank; emulate: two on the same device; real:
  //    one per device).
  for (int i = 0; i < n_slots; ++i) {
    auto s = std::make_unique<RankSlot>();
    s->index = i;
    s->rank = noop ? tp.noop_rank : i;
    s->device = devices[static_cast<size_t>(i)];
    const int dev = s->device, rank = s->rank;
    s->worker = std::make_unique<tp::RankWorker>(
        rank,
        [dev, rank] {
          R4DX_HIP_CHECK(hipSetDevice(dev));
          const std::wstring name = L"r4dx-tp-rank" + std::to_wstring(rank);
          (void)SetThreadDescription(GetCurrentThread(), name.c_str());
        },
        &m->done_, m->timing_);
    m->ranks_.push_back(std::move(s));
  }

  // 3. Probe every rank's device (docs/tp.md 2.9 step 3); the free bytes feed the embedding decision
  //    below. Real mode requires two physical devices (distinct PCI bus), the same gcnArchName (the
  //    ranks run the same code objects and tunings) and canMapHostMemory (the mailbox, 6.3).
  m->Run(m->AllSlots(),
         [&](RankSlot& s) {
           hipDeviceProp_t p;
           R4DX_HIP_CHECK(hipGetDeviceProperties(&p, s.device));
           DeviceProbe& d = probe[static_cast<size_t>(s.index)];
           d.name = p.name;
           d.arch = p.gcnArchName;
           d.pci_domain = p.pciDomainID;
           d.pci_bus = p.pciBusID;
           d.pci_device = p.pciDeviceID;
           d.can_map_host = p.canMapHostMemory;
           R4DX_HIP_CHECK(hipDeviceGetAttribute(&d.wall_clock_khz, hipDeviceAttributeWallClockRate, s.device));
           R4DX_HIP_CHECK(hipMemGetInfo(&d.free_bytes, &d.total));
         },
         CmdKind::kPlain, kNoStall);
  for (const auto& s : m->ranks_) {
    const DeviceProbe& d = probe[static_cast<size_t>(s->index)];
    std::fprintf(stderr, "[r4dx-tp] rank %d -> HIP device %d (%s, %s, pci %02x:%02x, %.2f/%.2f GiB free) [mode %s]\n",
                 s->rank, s->device, d.name.c_str(), d.arch.c_str(), d.pci_bus, d.pci_device,
                 static_cast<double>(d.free_bytes) / kGiB, static_cast<double>(d.total) / kGiB, ModeName(tp.mode));
  }
  if (real) {
    const DeviceProbe &a = probe[0], &b = probe[1];
    if (a.pci_domain == b.pci_domain && a.pci_bus == b.pci_bus) {
      throw core::TpError("TpModel::Load: --tp-mode real needs two physical GPUs, but HIP devices " +
                          std::to_string(devices[0]) + " and " + std::to_string(devices[1]) +
                          " are the same one (pci bus " + std::to_string(a.pci_bus) +
                          "); for both ranks on one device use --tp-mode emulate");
    }
    if (a.arch != b.arch) {
      throw core::TpError("TpModel::Load: --tp-mode real needs two GPUs of the same architecture, got " + a.arch +
                          " (HIP device " + std::to_string(devices[0]) + ") and " + b.arch + " (HIP device " +
                          std::to_string(devices[1]) + ")");
    }
    for (size_t i = 0; i < probe.size(); ++i) {
      if (probe[i].can_map_host != 1) {
        throw core::TpError("TpModel::Load: HIP device " + std::to_string(devices[i]) +
                            " cannot map host memory (canMapHostMemory = 0); the two-GPU all-reduce needs it");
      }
    }
    // 4. Wall-clock check (docs/tp.md 2.9 step 4), before any spinning kernel: every all-reduce
    //    timeout converts milliseconds to wall_clock64 ticks with hipDeviceAttributeWallClockRate.
    m->Run(m->AllSlots(), [&](RankSlot& s) { clock_khz[static_cast<size_t>(s.index)] = tp::CheckWallClockRate(); },
           CmdKind::kPlain, kNoStall);
    for (const auto& s : m->ranks_) {
      const size_t i = static_cast<size_t>(s->index);
      std::fprintf(stderr, "[r4dx-tp] rank %d: device wall clock %.0f kHz measured (%d kHz reported)\n", s->rank,
                   clock_khz[i], probe[i].wall_clock_khz);
    }
  }

  // 6. The process's ONE pinned host copy of text.embed_tokens (docs/tp.md 2.9 step 6, 5.3), loaded
  //    on rank 0's thread with hipHostMallocPortable so every device may DMA from it.
  m->Run({0}, [&](RankSlot&) { embed = Container::LoadEmbedTokensHost(opts.container_path); }, CmdKind::kPlain,
         kNoStall);

  // 5. The embedding device-mirror decision, taken ONCE for every rank (docs/tp.md 2.9 step 5): the
  //    2x-headroom heuristic of the single-device loader, applied per PHYSICAL device against every
  //    rank that device holds -- both ranks' mirrors under emulation -- so every rank takes the same
  //    gather path.
  {
    std::map<int, std::pair<size_t, int>> per_bus;  // pci bus -> (free bytes, ranks)
    for (const auto& s : m->ranks_) {
      const DeviceProbe& d = probe[static_cast<size_t>(s->index)];
      auto& e = per_bus[d.pci_bus];
      e.first = d.free_bytes;
      e.second += 1;
    }
    for (const auto& kv : per_bus) {
      if (kv.second.first < 2 * embed->bytes() * static_cast<size_t>(kv.second.second)) resident = false;
    }
    std::fprintf(stderr, "[r4dx-tp] text.embed_tokens (%.2f GiB, one pinned host copy): device mirror %s\n",
                 static_cast<double>(embed->bytes()) / kGiB,
                 resident ? "on every rank"
                          : (opts.embed_device_resident ? "OFF on every rank (not enough free VRAM for every rank)"
                                                        : "off (--embed-device-resident off)"));
  }

  // 7. The comm group and each rank's endpoint, created on its own thread (docs/tp.md 2.9 step 7).
  if (noop) {
    m->Run(m->AllSlots(), [&](RankSlot& s) { s.noop = tp::MakeNoopComm(tp.world, s.rank); }, CmdKind::kPlain,
           kNoStall);
  } else {
    m->group_ = tp::TpGroup::Create(real ? tp::TpGroup::Mode::kReal : tp::TpGroup::Mode::kEmulate, tp.world,
                                    tp::TpGroup::Geometry{tp.ar_nb_small, tp.ar_nb_large}, tp.ar_timeout_ms);
    tp::TpGroup* const group = m->group_.get();
    if (real) {
      // The pinned mailbox both GPUs map (docs/tp.md 6.3.2), allocated on rank 0's thread and reset
      // in the same call (flags := the session base, N32): hipHostMalloc does not promise zeroed
      // memory. ~TpModel frees it on rank 0's thread too, after both endpoints are gone.
      m->Run({0}, [group](RankSlot&) { group->AllocateMailbox(); }, CmdKind::kPlain, kNoStall);
    }
    m->Run(m->AllSlots(),
           [group](RankSlot& s) { s.endpoint = group->CreateEndpoint(s.rank, &s.worker->Heartbeat()); },
           CmdKind::kPlain, kNoStall);
  }
  for (const auto& s : m->ranks_) s->facade_comm = s->Comm();  // every rank idle: no race

  // 8. Every rank's Model, loaded in parallel (no collectives yet, so a failure on one rank does not
  //    affect the other; ~TpModel destroys whatever was built, on the right threads).
  m->Run(m->AllSlots(),
         [&](RankSlot& s) {
           ModelOptions ro = opts;
           ro.tp.world = tp.world;
           ro.tp.rank = s.rank;
           ro.tp.comm = s.Comm();
           ro.tp.shared_embed_host = embed;
           ro.tp.embed_device_resident_decided = resident ? 1 : 0;
           ro.tp.vision_weights_on_this_rank = s.rank == 0;
           ro.tp.submit_layers = tp.submit_layers;
           ro.tp.max_inflight_units = tp.max_inflight_units;
           s.model.emplace(Model::Load(ro));
         },
         CmdKind::kPlain, kNoStall);

  // 9. Warm-up, ONE collective command (docs/tp.md 2.9 step 9): the comm self-test and a latency
  //    sample, then every enabled path once under the relaxed 1500 ms all-reduce timeout, then back
  //    to the configured one.
  m->Run(m->AllSlots(),
         [&](RankSlot& s) {
           core::TpComm* c = s.Comm();
           c->SelfTest();
           {
             // 200 back-to-back 10 KiB all-reduces, timed by the first rank thread (rank 0; noop's
             // only rank) with hipEvents. The stream lives as long as the endpoint (it tracks every
             // stream an all-reduce was queued on).
             s.aux_stream.emplace();
             const hipStream_t st = s.aux_stream->get();
             constexpr int kIters = 200;
             constexpr int64_t kElems = 5120;  // one hidden row: 10 KiB
             core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(kElems));
             R4DX_HIP_CHECK(hipMemsetAsync(buf.data(), 0, buf.bytes(), st));
             hipEvent_t e0 = nullptr, e1 = nullptr;
             const bool timer = s.index == 0;
             if (timer) {
               R4DX_HIP_CHECK(hipEventCreate(&e0));
               R4DX_HIP_CHECK(hipEventCreate(&e1));
               R4DX_HIP_CHECK(hipEventRecord(e0, st));
             }
             for (int i = 0; i < kIters; ++i) c->AllReduceSumBf16(buf.data(), kElems, st);
             if (timer) R4DX_HIP_CHECK(hipEventRecord(e1, st));
             if (!tp::SyncWithWatchdog(st)) throw core::TpTimeoutError("tp: warm-up latency sample stuck > 30 s");
             if (timer) {
               float ms = 0.0f;
               R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
               (void)hipEventDestroy(e0);
               (void)hipEventDestroy(e1);
               latency_us = 1000.0 * static_cast<double>(ms) / kIters;
             }
             c->CheckHealthy();
           }
           c->SetAllReduceTimeoutMs(tp::kArTimeoutMaxMs);
           s.model->TpWarmup();
           c->SetAllReduceTimeoutMs(tp.ar_timeout_ms);
         },
         CmdKind::kCollective);
  std::fprintf(stderr, "[r4dx-tp] all-reduce self-test OK (%s), isolated L(10 KiB) = %.2f us\n", ModeName(tp.mode),
               latency_us);

  // 10. The host-only values every state may read (docs/tp.md 2.9 step 10), from rank 0; every other
  //     rank must agree on all of them except HasVision (the tower is rank 0's alone).
  m->Run(m->AllSlots(),
         [&](RankSlot& s) {
           const Model& mm = *s.model;
           Caps& c = caps[static_cast<size_t>(s.index)];
           c.id = mm.GetContainer().ModelId();
           c.image_token = mm.GetContainer().ImageTokenId();
           c.merge = mm.GetContainer().HasVisionConfig()
                         ? static_cast<int64_t>(mm.GetContainer().VisionCfg().spatial_merge_size)
                         : 2;
           c.layers = mm.GetContainer().NumLoadedLayers();
           c.vocab = mm.GlobalConfig().vocab_size;
           c.hidden = mm.GlobalConfig().hidden_size;
           c.vision = mm.HasVision();
           c.mtp = mm.MtpEnabled();
           c.mtp_reduced = mm.MtpUsingReducedVocabDraft();
           c.dflash = mm.DflashEnabled();
           if (s.index == 0) rank0_global_config = mm.GlobalConfig();
         },
         CmdKind::kPlain, kNoStall);
  m->global_config_ = rank0_global_config;
  for (size_t i = 1; i < caps.size(); ++i) {
    const Caps &a = caps[0], &b = caps[i];
    if (a.id != b.id || a.image_token != b.image_token || a.merge != b.merge || a.layers != b.layers ||
        a.vocab != b.vocab || a.hidden != b.hidden || a.mtp != b.mtp || a.mtp_reduced != b.mtp_reduced ||
        a.dflash != b.dflash) {
      throw core::TpDivergenceError("TpModel::Load: the ranks disagree on the loaded model's identity or capabilities");
    }
  }
  m->model_id_ = caps[0].id;
  m->image_token_id_ = caps[0].image_token;
  m->vision_merge_size_ = caps[0].merge;
  m->has_vision_ = caps[0].vision;
  m->mtp_enabled_ = caps[0].mtp;
  m->mtp_reduced_vocab_ = caps[0].mtp_reduced;
  m->dflash_enabled_ = caps[0].dflash;
  m->num_loaded_layers_ = caps[0].layers;
  m->dflash_injection_ = true;  // the Model default (model.h)
  m->cached_position_ = 0;
  m->cached_fallback_rows_ = 0;

  // Test-only fault injection, counted from the END of warm-up (TpOptions::fault_*).
  if (tp.fault_rank >= 0) m->ArmFaultInjection(tp.fault_rank, tp.fault_at_allreduce, tp.fault_kind);

  // The facade drops its reference to the pinned embedding table on rank 0's thread, so the last
  // owner (a rank's Container) frees it on a rank thread, never the facade (docs/tp.md 2.1). (On a
  // failed load, `release_embed` above does the same.)
  m->Run({0}, [&embed](RankSlot&) { embed.reset(); }, CmdKind::kPlain, kNoStall);
  m->state_ = State::kReady;
  return m;
}

// ---- shutdown -----------------------------------------------------------------------------------

TpModel::~TpModel() {
  if (ranks_.empty()) return;
  // 1. A group that is not healthy may still have a rank blocked in a comm wait: poison it (host-side
  //    stores only) so that rank wakes up and finishes its command (docs/tp.md 2.6 step 1).
  if (state_ != State::kReady) AbortAllFromFacade(core::kAbortShutdown, "tp: shutdown");
  const auto wait_idle = [&](const char* what) {
    std::unique_lock<std::mutex> lk(done_.mu);
    const bool ok = done_.cv.wait_until(lk, std::chrono::steady_clock::now() + kShutdownWait, [&] {
      for (const auto& s : ranks_) {
        if (!s->worker->Idle()) return false;
      }
      return true;
    });
    if (!ok) {
      lk.unlock();
      for (const auto& s : ranks_) {
        if (!s->worker->Idle()) {
          std::fprintf(stderr, "[r4dx-tp] rank %d did not finish in 30 s at shutdown (%s)\n", s->rank, what);
        }
      }
      std::fflush(stderr);
      std::quick_exit(3);
    }
  };
  // 2. Never write the command of a busy worker: wait for whatever each rank is still running.
  wait_idle("running command");
  // 3. Drain every rank's streams BEFORE any rank frees anything: under emulation one rank's add
  //    reads the other rank's exchange buffers (docs/tp.md 6.5).
  for (const auto& s : ranks_) {
    RankSlot* sp = s.get();
    sp->worker->Post([sp] {
      try {
        if (sp->endpoint) sp->endpoint->SyncStreams();
      } catch (...) {
      }
    });
  }
  wait_idle("stream drain");
  // 4. The teardown closure on each rank's own thread: the Model first (it owns the streams the
  //    endpoint tracked), then the endpoint (frees its device memory on the right device).
  for (const auto& s : ranks_) {
    s->facade_comm = nullptr;
    RankSlot* sp = s.get();
    sp->worker->Post([sp] {
      sp->model.reset();
      sp->endpoint.reset();
      sp->noop.reset();
      sp->aux_stream.reset();
      (void)hipDeviceSynchronize();
    });
  }
  wait_idle("teardown");
  for (const auto& s : ranks_) (void)s->worker->TakeError();
  // 5. Free the group's shared state (docs/tp.md 2.6 step 3) -- in real mode the pinned mailbox, a
  //    hipHostFree -- on rank 0's thread, now that every endpoint is gone and both devices were
  //    synchronized, so no kernel can still touch it: the facade thread never calls HIP (2.1;
  //    Appendix B N58). ~TpGroup itself leaks the region if an endpoint could not be torn down or a
  //    stream watchdog fired. Then join the threads (~RankWorker).
  if (group_ != nullptr) {
    std::unique_ptr<tp::TpGroup>* g = &group_;
    ranks_[0]->worker->Post([g] { g->reset(); });
    wait_idle("mailbox free");
    (void)ranks_[0]->worker->TakeError();
  }
  ranks_.clear();
}

// ---- command plumbing ---------------------------------------------------------------------------

std::vector<int> TpModel::AllSlots() const {
  std::vector<int> v;
  for (const auto& s : ranks_) v.push_back(s->index);
  return v;
}

void TpModel::Run(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind,
                  std::chrono::milliseconds stall) {
  if (state_ == State::kFatal) throw core::TpStateError("tp: fatal, restart the process");
  // The closures share one copy of `body`: if a rank ever stalls past the watchdog, the facade
  // throws while that rank may still run it (docs/tp.md 2.2 step 6: the group is then fatal).
  auto shared = std::make_shared<std::function<void(RankSlot&)>>(body);
  std::vector<tp::RankWorker*> workers;
  for (int i : slots) {
    RankSlot* s = ranks_[static_cast<size_t>(i)].get();
    workers.push_back(s->worker.get());
    s->worker->Post([shared, s, kind] {
      try {
        (*shared)(*s);
        if (kind == CmdKind::kCollective) {
          if (core::TpComm* c = s->Comm()) c->CheckHealthy();
        }
      } catch (...) {
        // docs/tp.md 2.2 step 3: poison the group so a peer spinning in an all-reduce or blocked in
        // a host exchange wakes up -- unless this rank's error IS that poison.
        if (kind == CmdKind::kCollective) {
          const std::exception_ptr e = std::current_exception();
          if (!IsTpAborted(e)) {
            if (core::TpComm* c = s->Comm()) c->Abort(core::kAbortHost, WhatOf(e));
          }
        }
        throw;
      }
    });
  }
  tp::ProgressWatchdog watchdog(stall);
  if (tp::WaitAllIdle(workers, done_, timing_, watchdog) == tp::WaitResult::kStalled) {
    state_ = State::kFatal;
    // Appendix B N53: poison the group before giving up on the stalled command, so a rank that
    // eventually returns from the stuck HIP call throws at its next comm operation instead of
    // carrying on with the forward. What it touches until then is heap state its closure co-owns
    // (CmdState), never this stack frame.
    AbortAllFromFacade(core::kAbortShutdown, "tp: progress watchdog -- no rank made progress; the group is fatal");
    for (int i : slots) {
      const RankSlot& s = *ranks_[static_cast<size_t>(i)];
      std::fprintf(stderr, "[r4dx-tp] rank %d (dev %d): no progress for %lld s -- posted %llu, done %llu, heartbeat %llu\n",
                   s.rank, s.device, static_cast<long long>(stall.count() / 1000),
                   static_cast<unsigned long long>(s.worker->Posted()),
                   static_cast<unsigned long long>(s.worker->Done()),
                   static_cast<unsigned long long>(s.worker->Heartbeat().load(std::memory_order_relaxed)));
    }
    std::fflush(stderr);
    throw core::TpTimeoutError("tp: no rank made progress for " + std::to_string(stall.count() / 1000) +
                               " s (a rank is stuck inside a HIP call); the group is fatal, restart the process");
  }
  // docs/tp.md 2.4: the root cause is the lowest rank's exception that is not a TpAbortedError
  // (the aborts are its consequences), else the lowest rank's TpAbortedError.
  std::exception_ptr root, first_aborted;
  bool any = false;
  for (int i : slots) {
    RankSlot& s = *ranks_[static_cast<size_t>(i)];
    std::exception_ptr e = s.worker->TakeError();
    if (!e) continue;
    any = true;
    std::fprintf(stderr, "[r4dx-tp] rank %d (dev %d): %s\n", s.rank, s.device, WhatOf(e).c_str());
    if (IsTpAborted(e)) {
      if (!first_aborted) first_aborted = e;
    } else if (!root) {
      root = e;
    }
  }
  if (!any) return;
  std::fflush(stderr);
  std::rethrow_exception(root ? root : first_aborted);
}

void TpModel::RunGuarded(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind) {
  try {
    Run(slots, body, kind);
  } catch (...) {
    if (state_ != State::kFatal) state_ = State::kNeedsRecovery;
    throw;
  }
}

void TpModel::AbortAllFromFacade(uint32_t code, const char* why) noexcept {
  for (const auto& s : ranks_) {
    if (s->facade_comm != nullptr) s->facade_comm->Abort(code, why);
  }
}

void TpModel::RequireReady() const {
  if (state_ == State::kNeedsRecovery) {
    throw core::TpStateError("tp: group aborted by an earlier error; call Reset() first");
  }
  if (state_ == State::kFatal) throw core::TpStateError("tp: fatal, restart the process");
}

void TpModel::Diverged(const std::string& what) {
  state_ = State::kNeedsRecovery;
  std::fprintf(stderr, "[r4dx-tp] divergence: %s\n", what.c_str());
  throw core::TpDivergenceError("tp: " + what);
}

template <class T>
void TpModel::RequireAllEqual(const std::vector<T>& v, const char* what) {
  for (size_t i = 1; i < v.size(); ++i) {
    bool same;
    if constexpr (std::is_same_v<T, std::vector<float>>) {
      same = v[i].size() == v[0].size() &&
             (v[0].empty() || std::memcmp(v[i].data(), v[0].data(), v[0].size() * sizeof(float)) == 0);
    } else {
      same = v[i] == v[0];
    }
    if (!same) Diverged(std::string("ranks returned different ") + what + " (rank 0 vs rank " + std::to_string(i) + ")");
  }
}

void TpModel::RequireRngsEqual(const std::vector<std::mt19937_64>& rngs, const char* what) {
  for (size_t i = 1; i < rngs.size(); ++i) {
    if (!(rngs[i] == rngs[0])) {
      Diverged(std::string("ranks consumed different draws in ") + what + " (rank 0 vs rank " + std::to_string(i) +
               ")");
    }
  }
}

// ---- recovery and sequence state ----------------------------------------------------------------

void TpModel::Recover() {
  if (group_ != nullptr) {
    // docs/tp.md 2.5 steps 1-7 through the rank threads (plain commands: the comm is aborted, and
    // every step talks to the endpoint directly). The endpoints know every stream an all-reduce was
    // queued on, which includes each Model's stream (warm-up ran all-reduces on it).
    // The step closures are copied into the commands (N53: TpGroup::Recover's own closures co-own
    // their state), so a rank still running one after a watchdog stall touches no dead frame.
    tp::TpGroup::RankRunner rr;
    rr.run_all = [this](const std::function<void(int)>& f) {
      Run(AllSlots(), [f](RankSlot& s) { f(s.rank); }, CmdKind::kPlain);
    };
    rr.run_one = [this](int rank, const std::function<void()>& f) {
      for (const auto& s : ranks_) {
        if (s->rank == rank) {
          Run({s->index}, [f](RankSlot&) { f(); }, CmdKind::kPlain);
          return;
        }
      }
      throw std::logic_error("TpModel::Recover: no rank " + std::to_string(rank));
    };
    group_->Recover(rr);
  } else {
    // noop: no peer, no seq counters -- only the endpoint's host counters restart.
    Run(AllSlots(), [](RankSlot& s) { s.noop->ResetCounters(); }, CmdKind::kPlain);
  }
}

void TpModel::Reset() {
  if (state_ == State::kFatal) throw core::TpStateError("tp: fatal, restart the process");
  const bool inject = dflash_injection_;
  // docs/tp.md 8.4: Model::Reset (no collectives) and the cached injection policy, on every rank.
  const auto reset_ranks = [this, inject] {
    RunAll([inject](Model& m, int) {
      m.Reset();
      m.SetDflashInjectionEnabled(inject);
    });
  };
  if (state_ == State::kNeedsRecovery) {
    try {
      Recover();
      reset_ranks();
    } catch (const std::exception& e) {
      state_ = State::kFatal;
      std::fprintf(stderr, "[r4dx-tp] recovery FAILED, the group is fatal (restart the process): %s\n", e.what());
      throw;
    }
    std::fprintf(stderr, "[r4dx-tp] recovered: group reset after an earlier error\n");
    state_ = State::kReady;
  } else {
    try {
      reset_ranks();
    } catch (...) {
      if (state_ != State::kFatal) state_ = State::kNeedsRecovery;
      throw;
    }
  }
  cached_position_ = 0;
}

// ---- host-only / diagnostics --------------------------------------------------------------------

// The result slots of the diagnostic commands below are heap blocks the closures co-own (N53).
std::vector<VramReport> TpModel::VramImpl() {
  auto out = std::make_shared<std::vector<VramReport>>(ranks_.size());
  Run(AllSlots(),
      [out](RankSlot& s) {
        size_t free_b = 0, total_b = 0;
        R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
        VramReport& r = (*out)[static_cast<size_t>(s.index)];
        r.rank = s.rank;
        r.device = s.device;
        r.used_gib = static_cast<double>(total_b - free_b) / kGiB;
        r.free_gib = static_cast<double>(free_b) / kGiB;
        r.total_gib = static_cast<double>(total_b) / kGiB;
        r.buffers_gib = static_cast<double>(core::DeviceBufferBytes(s.device)) / kGiB;
      },
      CmdKind::kPlain);
  return *out;
}

std::vector<VramReport> TpModel::Vram() const {
  if (state_ == State::kFatal) return cached_vram_;
  // Host-only in the docs/tp.md 2.4 sense (no TpComm, every rank idle between commands), so it is
  // legal in kNeedsRecovery and a failure here changes no state. The dispatch itself is not const.
  cached_vram_ = const_cast<TpModel*>(this)->VramImpl();
  return cached_vram_;
}

std::vector<std::array<uint64_t, 2>> TpModel::CallCounts() {
  if (state_ == State::kFatal) throw core::TpStateError("tp: fatal, restart the process");
  auto out = std::make_shared<std::vector<std::array<uint64_t, 2>>>(ranks_.size());
  Run(AllSlots(), [out](RankSlot& s) { (*out)[static_cast<size_t>(s.index)] = s.Comm()->CallCounts(); },
      CmdKind::kPlain);
  return *out;
}

std::vector<core::TpCommStats> TpModel::CommStats() {
  if (state_ == State::kFatal) throw core::TpStateError("tp: fatal, restart the process");
  auto out = std::make_shared<std::vector<core::TpCommStats>>(ranks_.size());
  Run(AllSlots(), [out](RankSlot& s) { (*out)[static_cast<size_t>(s.index)] = s.Comm()->Stats(); }, CmdKind::kPlain);
  return *out;
}

std::vector<tp::SubmitBounder::Stats> TpModel::SubmitStats() {
  if (state_ == State::kFatal) throw core::TpStateError("tp: fatal, restart the process");
  auto out = std::make_shared<std::vector<tp::SubmitBounder::Stats>>(ranks_.size());
  Run(AllSlots(), [out](RankSlot& s) { (*out)[static_cast<size_t>(s.index)] = s.model->TpSubmitStats(); },
      CmdKind::kPlain);
  return *out;
}

std::string TpModel::StatsLine() {
  // docs/tp.md 9.1's `--stats` line (Appendix B N59). Both ranks make the same all-reduce calls
  // (6.3.6), so the per-channel counts are rank 0's; the exchange wait and the bounding's cap wait are
  // the maximum over ranks (whichever rank arrives first at an exchange does the waiting). Two time
  // bases: ar_calls, host_exchanges and max_exchange_wait restart at every recovery (the endpoint's
  // ResetCounters, 2.5 step 6); aborts and the bounding's counters are cumulative since load.
  const std::vector<core::TpCommStats> cs = CommStats();
  const std::vector<tp::SubmitBounder::Stats> ss = SubmitStats();
  std::string devs;
  for (const auto& s : ranks_) devs += (devs.empty() ? "" : ",") + std::to_string(s->device);
  double wait_max = 0, cap_wait_max = 0;
  uint64_t aborts = 0, units = 0, cap_waits = 0;
  for (const core::TpCommStats& c : cs) {
    wait_max = std::max(wait_max, c.host_exchange_wait_us_max);
    aborts += c.aborts;
  }
  for (const tp::SubmitBounder::Stats& s : ss) {
    units = std::max(units, s.units);
    cap_waits = std::max(cap_waits, s.waits);
    cap_wait_max = std::max(cap_wait_max, s.wait_us_max);
  }
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "tp: mode=%s devices=%s ar_calls=%llu/%llu host_exchanges=%llu max_exchange_wait=%.0fus aborts=%llu "
                "submit=%d/%d units=%llu cap_waits=%llu max_cap_wait=%.0fus",
                ModeName(tp_.mode), devs.c_str(), static_cast<unsigned long long>(cs[0].ar_calls[0]),
                static_cast<unsigned long long>(cs[0].ar_calls[1]),
                static_cast<unsigned long long>(cs[0].host_exchanges), wait_max,
                static_cast<unsigned long long>(aborts), tp_.submit_layers, tp_.max_inflight_units,
                static_cast<unsigned long long>(units), static_cast<unsigned long long>(cap_waits), cap_wait_max);
  return buf;
}

void TpModel::ArmFaultInjection(int rank, int64_t at_allreduce, int kind) {
  RequireReady();
  for (const auto& s : ranks_) {
    if (s->rank != rank) continue;
    if (!s->endpoint) {
      throw std::invalid_argument("TpModel::ArmFaultInjection: fault injection needs --tp-mode emulate or real");
    }
    Run({s->index}, [at_allreduce, kind](RankSlot& x) { x.endpoint->ArmFaultInjection(at_allreduce, kind); },
        CmdKind::kPlain);
    return;
  }
  throw std::invalid_argument("TpModel::ArmFaultInjection: no rank " + std::to_string(rank) + " in this group");
}

void TpModel::RunCollectiveForTest(const std::function<void(Model&, int)>& fn) {
  RunCollective([this, fn](Model& m, int slot) { fn(m, ranks_[static_cast<size_t>(slot)]->rank); });
}

// ---- vision / profiling -------------------------------------------------------------------------

void TpModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                           vision::VisionEncodeStats*) {
  RequireReady();
  throw core::TpUnsupportedError("TpModel::EncodeImages: vision is not supported under tensor parallelism yet "
                                 "(docs/tp.md P5)");
}

StepProfile TpModel::DecodeStepProfiled(int32_t) {
  RequireReady();
  throw core::TpUnsupportedError("TpModel::DecodeStepProfiled: profiling is not supported under tensor "
                                 "parallelism (docs/tp.md 1.2)");
}

StepProfile TpModel::PrefillProfiled(const std::vector<int32_t>&) {
  RequireReady();
  throw core::TpUnsupportedError("TpModel::PrefillProfiled: profiling is not supported under tensor "
                                 "parallelism (docs/tp.md 1.2)");
}

// ---- forward ------------------------------------------------------------------------------------
//
// Every closure below owns what it reads (Appendix B N53): arguments are captured by value (a
// Prefill copies its token vector once -- microseconds against the prefill), and the per-rank rng
// copies and walk lengths live in heap blocks the closure co-owns, so a rank still running after a
// watchdog stall never reads the caller's (possibly freed) arguments.

std::vector<float> TpModel::Prefill(const std::vector<int32_t>& token_ids) {
  std::vector<std::vector<float>> r =
      RunCollective([ids = token_ids](Model& m, int) { return m.Prefill(ids); });
  RequireAllEqual(r, "Prefill logits");
  return std::move(r[0]);
}

std::vector<float> TpModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                              const std::vector<ImageSpan>& images) {
  std::vector<std::vector<float>> r = RunCollective(
      [ids = token_ids, spans = images](Model& m, int) { return m.PrefillMultimodal(ids, spans); });
  RequireAllEqual(r, "PrefillMultimodal logits");
  return std::move(r[0]);
}

std::vector<float> TpModel::DecodeStep(int32_t token_id) {
  std::vector<std::vector<float>> r = RunCollective([token_id](Model& m, int) { return m.DecodeStep(token_id); });
  RequireAllEqual(r, "DecodeStep logits");
  return std::move(r[0]);
}

int32_t TpModel::DecodeStepGreedy(int32_t token_id) {
  const std::vector<int32_t> r =
      RunCollective([token_id](Model& m, int) { return m.DecodeStepGreedy(token_id); });
  RequireAllEqual(r, "DecodeStepGreedy token");
  return r[0];
}

int32_t TpModel::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
  // docs/tp.md 2.3: every rank draws from its own copy of the caller's generator; the copies must
  // end in the same state (same number of draws), and the caller's generator then takes it.
  auto rngs = std::make_shared<std::vector<std::mt19937_64>>(ranks_.size(), rng);
  const std::vector<int32_t> r = RunCollective([token_id, params, rngs](Model& m, int slot) {
    return m.DecodeStepSampled(token_id, params, (*rngs)[static_cast<size_t>(slot)]);
  });
  RequireAllEqual(r, "DecodeStepSampled token");
  RequireRngsEqual(*rngs, "DecodeStepSampled");
  rng = (*rngs)[0];
  return r[0];
}

std::vector<int32_t> TpModel::DecodeStepMtpGreedy(int32_t token_id, int64_t k) {
  RequireReady();
  if (!mtp_enabled_) {
    throw std::runtime_error("TpModel::DecodeStepMtp{Greedy,Sampled}: MTP is not enabled on this model");
  }
  std::vector<std::vector<int32_t>> r =
      RunCollective([token_id, k](Model& m, int) { return m.DecodeStepMtpGreedy(token_id, k); });
  RequireAllEqual(r, "DecodeStepMtpGreedy round");
  return std::move(r[0]);
}

std::vector<int32_t> TpModel::DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                                   std::mt19937_64& rng) {
  RequireReady();
  if (!mtp_enabled_) {
    throw std::runtime_error("TpModel::DecodeStepMtp{Greedy,Sampled}: MTP is not enabled on this model");
  }
  auto rngs = std::make_shared<std::vector<std::mt19937_64>>(ranks_.size(), rng);
  std::vector<std::vector<int32_t>> r = RunCollective([token_id, k, params, rngs](Model& m, int slot) {
    return m.DecodeStepMtpSampled(token_id, k, params, (*rngs)[static_cast<size_t>(slot)]);
  });
  RequireAllEqual(r, "DecodeStepMtpSampled round");
  RequireRngsEqual(*rngs, "DecodeStepMtpSampled");
  rng = (*rngs)[0];
  return std::move(r[0]);
}

std::vector<int32_t> TpModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                     int64_t* walk_len_out) {
  RequireReady();
  if (!dflash_enabled_) {
    throw std::runtime_error("TpModel::DecodeStepDflash{Greedy,Sampled}: DFlash2 is not enabled on this model");
  }
  auto walks = std::make_shared<std::vector<int64_t>>(ranks_.size(), 0);
  std::vector<std::vector<int32_t>> r = RunCollective([token_id, k, p_min, n_min, walks](Model& m, int slot) {
    return m.DecodeStepDflashGreedy(token_id, k, p_min, n_min, &(*walks)[static_cast<size_t>(slot)]);
  });
  RequireAllEqual(r, "DecodeStepDflashGreedy round");
  RequireAllEqual(*walks, "DecodeStepDflashGreedy walk_len");
  if (walk_len_out != nullptr) *walk_len_out = (*walks)[0];
  return std::move(r[0]);
}

std::vector<int32_t> TpModel::DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                      const kernels::SampleParams& params, std::mt19937_64& rng,
                                                      int64_t* walk_len_out) {
  RequireReady();
  if (!dflash_enabled_) {
    throw std::runtime_error("TpModel::DecodeStepDflash{Greedy,Sampled}: DFlash2 is not enabled on this model");
  }
  auto rngs = std::make_shared<std::vector<std::mt19937_64>>(ranks_.size(), rng);
  auto walks = std::make_shared<std::vector<int64_t>>(ranks_.size(), 0);
  std::vector<std::vector<int32_t>> r =
      RunCollective([token_id, k, p_min, n_min, params, rngs, walks](Model& m, int slot) {
        return m.DecodeStepDflashSampled(token_id, k, p_min, n_min, params, (*rngs)[static_cast<size_t>(slot)],
                                         &(*walks)[static_cast<size_t>(slot)]);
      });
  RequireAllEqual(r, "DecodeStepDflashSampled round");
  RequireAllEqual(*walks, "DecodeStepDflashSampled walk_len");
  RequireRngsEqual(*rngs, "DecodeStepDflashSampled");
  rng = (*rngs)[0];
  if (walk_len_out != nullptr) *walk_len_out = (*walks)[0];
  return std::move(r[0]);
}

// ---- factory ------------------------------------------------------------------------------------

std::unique_ptr<TextModel> LoadTextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world == 1) {
    const TpOptions d;
    if (tp.mode != d.mode || !tp.devices.empty() || tp.noop_rank != d.noop_rank || tp.ar_timeout_ms != d.ar_timeout_ms ||
        tp.ar_nb_small != d.ar_nb_small || tp.ar_nb_large != d.ar_nb_large || tp.submit_layers != d.submit_layers ||
        tp.max_inflight_units != d.max_inflight_units || tp.fault_rank != d.fault_rank ||
        tp.fault_at_allreduce != d.fault_at_allreduce || tp.fault_kind != d.fault_kind) {
      throw std::invalid_argument("LoadTextModel: the TpOptions besides `world` are tensor-parallel options "
                                  "(world == 2 only)");
    }
    return std::make_unique<LocalTextModel>(Model::Load(opts));
  }
  if (tp.world == 2) return TpModel::Load(opts, tp);
  throw std::invalid_argument("LoadTextModel: TpOptions::world must be 1 or 2, got " + std::to_string(tp.world));
}

}  // namespace r4dx::model
