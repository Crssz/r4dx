// tests/model/test_dflash_tail.cpp -- the device half of the hybrid mode's DFlash tail (docs/pp-tp2-hybrid.md 3, 4 "DFlash reference", 6, 9 "Gate C";
// the rules are dflash_tail_plan.h's, tested on the CPU by test_dflash_tail_cpu). Three independent parts; R4DX_TEST_ONLY=<a|b|c> picks one.
//   a. DRAFTERS ONLY (the production DFlash2 drafter, two copies ~4 GiB, no target): the tail-fed ring == the fully fed ring. Two fresh
//      drafters run with the thread's tuning flag TRUE (the TP-thread numerics the rank drafters use): F is fed every row of a call in the
//      64-row slices a prefill makes, T only the tail [DflashTailStart, n) through hybrid InjectFeatureRowsFromHost (the loop
//      Model::TpInjectDflashTail runs). Their visible windows (the K and V rows of every draft layer over [lo, injected)), injected
//      counts and lower bounds must be equal -- cold call, warm call (p0 > 0), a call with an image run (rope rows != positions, inside
//      the tail window), a call shorter than the window. Before each case both rings are filled with a poison window and reset, so a slot
//      the tail failed to write cannot hide behind an earlier equal write. NEGATIVE CONTROLS: the image rope rows dropped, a tail one
//      slice late, a skipped slice, a start off the call's grid each change the window.
//   b. THE STAGE-SIDE CAPTURE (the 4-layer container, feature capture on layers {0, 1, 3}, no drafter): Model::StageArmDflashTail /
//      StageDisarmDflashTail collect, during an ordinary Prefill (cold, then a warm second call), PP-emulate (stage A / stage B composition
//      with the carry through the host), the rows [s, n) of the features a reference run's capture observer saw, byte for byte; a call
//      that stops short, a double arm, a taken observer slot and Reset() are handled; PrefillMultimodal's rope_rows_out
//      gives the temporal row (hybrid::TemporalRopeRows) the tail ships.
//   c. Model::TpInjectDflashTail END TO END (the production target + drafter loaded as ONE TP=1 Model, ~17 GiB; set R4DX_TEST_ONLY=c or leave
//      unset on a machine that has the containers): prefill 2600 tokens with injection on and the tail ARMED (so the capture and the full
//      injection of the same run are the reference); Reset, prefill again with injection OFF (the drafter ring stays empty); inject the
//      captured tail with TpInjectDflashTail (rope rows nullptr, then explicit positions): the drafter's window digest records equal the
//      reference run's. Negative controls: shifted rope rows, a tail that does not end at the Model's position, a gap with a short tail.
// Device 1 (HIP_VISIBLE_DEVICES=1 from CMake), SKIP 77 when no part's files exist. Links the R4DX_TP_TESTING variant. Written, NOT run by its
// author (CPU-only session).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "dflash_draft.h"
#include "dflash_draft_weights.h"
#include "dflash_tail.h"
#include "dflash_tail_plan.h"
#include "linear.h"
#include "model.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::core::Arena;
using r4dx::core::DeviceBuffer;
using r4dx::core::PinnedBuffer;
using r4dx::core::Stream;
using r4dx::model::DflashDraft;
using r4dx::model::DflashDraftOptions;
using r4dx::model::InjectFeatureRowsFromHost;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
namespace hybrid = r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fails;
  }
}
template <class E = std::exception, class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}
bool Part(const char* name) {
  const char* only = std::getenv("R4DX_TEST_ONLY");
  return only == nullptr || *only == '\0' || std::string(only).find(name) != std::string::npos;
}

std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 100 + (i * 41 + salt * 977) % 5000;
  return v;
}

std::vector<uint16_t> Features(int64_t rows, int64_t cols, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<uint16_t> v(static_cast<size_t>(rows * cols));
  for (uint16_t& x : v) {
    const float f = (static_cast<float>(rng() % 2001) - 1000.0f) / 2000.0f;  // [-0.5, 0.5]
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    x = static_cast<uint16_t>(bits >> 16);
  }
  return v;
}

// ======================================================================================================================================
// Part a -- drafters only
// ======================================================================================================================================
struct Window {
  int64_t lo = 0, injected = 0;
  std::vector<uint64_t> k, v;
  bool operator==(const Window& o) const { return lo == o.lo && injected == o.injected && k == o.k && v == o.v; }
};

Window WindowOf(const DflashDraft& d, Stream& stream) {
  const r4dx::model::Dflash2Config& cfg = d.Config();
  Window w;
  int64_t hi = 0;
  hybrid::DflashWindow(d.InjectedCount(), d.ValidFrom(), cfg.attention.sliding_window, &w.lo, &hi);
  w.injected = d.InjectedCount();
  const int64_t row_elems = cfg.attention.head_count_kv * cfg.attention.key_length;
  for (int64_t l = 0; l < cfg.block_count; ++l) {
    const std::vector<uint16_t> k = d.DebugStoreK(stream, l, w.lo, hi - w.lo), v = d.DebugStoreV(stream, l, w.lo, hi - w.lo);
    w.k.push_back(hybrid::DigestDflashRows(k.data(), hi - w.lo, row_elems, w.lo));
    w.v.push_back(hybrid::DigestDflashRows(v.data(), hi - w.lo, row_elems, w.lo));
  }
  return w;
}

struct DrafterCase {
  const char* name;
  int64_t p0, n;
  int64_t image_at;  // an image run [image_at, image_at + 300) ropes at one temporal value; -1: text only
};

struct Drafters {
  DflashDraft f, t;  // F: fully fed; T: tail-fed
  Stream stream;
  Arena arena;
  DeviceBuffer<uint16_t> staging;
  int64_t cols = 0;
  std::vector<uint16_t> poison;

  // Poison both rings with a full window of other data, reset, then (warm) feed the same prior rows [0, p0) to both.
  void Prepare(const std::vector<uint16_t>& feat, int64_t p0) {
    for (DflashDraft* d : {&f, &t}) {
      d->Reset();  // (a monotonic drafter cannot take the poison below its frontier)
      InjectFeatureRowsFromHost(*d, stream, arena, staging.data(), 64, poison.data(), static_cast<int64_t>(poison.size()) / cols, 0, nullptr);
      d->Reset();
      if (p0 > 0) InjectFeatureRowsFromHost(*d, stream, arena, staging.data(), 64, feat.data(), p0, 0, nullptr);
    }
  }
};

std::vector<int32_t> RopeOf(const DrafterCase& c) {
  std::vector<int32_t> t(static_cast<size_t>(c.n));
  for (int64_t i = 0; i < c.n; ++i) {
    t[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    if (c.image_at >= 0 && i >= c.image_at + 300) t[static_cast<size_t>(i)] = static_cast<int32_t>(i - 299);
    if (c.image_at >= 0 && i >= c.image_at && i < c.image_at + 300) t[static_cast<size_t>(i)] = static_cast<int32_t>(c.image_at);
  }
  return t;
}

void PartA() {
  const char* path = ProductionDrafterPath();
  if (!FileExists(path)) {
    std::printf("SKIP part a: %s missing\n", path);
    return;
  }
  const r4dx::model::DflashDraftWeights peek = r4dx::model::DflashDraftWeights::Open(path);
  DflashDraftOptions o;
  o.container_path = path;
  o.layout = r4dx::model::LayoutFromName(peek.Config().layout);
  r4dx::model::Tp2TuningScope tp_thread(true);  // the rank drafters run with the TP-thread flag; the reference must too
  Drafters dr{DflashDraft::Load(o), DflashDraft::Load(o), Stream(), Arena(), DeviceBuffer<uint16_t>(), 0, {}};
  dr.arena.Reserve(256ull << 20);
  dr.cols = dr.f.FeatureCols();
  dr.staging = DeviceBuffer<uint16_t>(static_cast<size_t>(64 * dr.cols));
  dr.poison = Features(2100, dr.cols, 999);
  const int64_t window = dr.f.Config().attention.sliding_window;
  Check(window == hybrid::kDflashWindowRows, "the drafter's sliding window is the 2048 the tail rules assume");

  const DrafterCase cases[] = {
      {"cold, image run inside the tail window", 0, 2600, 1000},
      {"warm (p0 = 700), image run inside the tail window", 700, 3400, 1500},
      {"cold, shorter than the window", 0, 1500, -1},
      {"warm, text only, tail on a gap", 300, 2900, -1},
  };
  for (const DrafterCase& c : cases) {
    const std::vector<uint16_t> feat = Features(c.n, dr.cols, static_cast<uint64_t>(c.n));
    const std::vector<int32_t> rope = RopeOf(c);
    const bool images = c.image_at >= 0;
    const int64_t s = hybrid::DflashTailStart(c.p0, c.n, window);
    // the rope block the stage hands back (compact [3, n - p0], call-local) and the temporal row of the tail out of it
    std::vector<int32_t> rope3(static_cast<size_t>(3 * (c.n - c.p0)));
    for (int64_t i = 0; i < c.n - c.p0; ++i) {
      rope3[static_cast<size_t>(i)] = rope[static_cast<size_t>(c.p0 + i)];
      rope3[static_cast<size_t>(c.n - c.p0 + i)] = 7;
      rope3[static_cast<size_t>(2 * (c.n - c.p0) + i)] = 9;
    }
    const std::vector<int32_t> tail_rope = hybrid::TemporalRopeRows(rope3, c.n - c.p0, s - c.p0, c.n - s);

    const auto full_window = [&]() {
      dr.Prepare(feat, c.p0);
      InjectFeatureRowsFromHost(dr.f, dr.stream, dr.arena, dr.staging.data(), 64, feat.data() + c.p0 * dr.cols, c.n - c.p0, c.p0, images ? rope.data() + c.p0 : nullptr);
      return WindowOf(dr.f, dr.stream);
    };
    const Window ref = full_window();
    // The tail run. `skip` skips the slice with that index; `rope_on` false drops the rope rows; start is s unless given.
    const auto tail_window = [&](int64_t start, bool rope_on, int skip) {
      dr.Prepare(feat, c.p0);
      const std::vector<hybrid::TailSlice> sl = hybrid::TailSlices(c.n - start);
      for (size_t k = 0; k < sl.size(); ++k) {
        if (static_cast<int>(k) == skip) continue;
        const int64_t row0 = start + sl[k].row0;
        std::vector<int32_t> rr;
        if (rope_on && images) rr.assign(rope.begin() + row0, rope.begin() + row0 + sl[k].rows);
        InjectFeatureRowsFromHost(dr.t, dr.stream, dr.arena, dr.staging.data(), 64, feat.data() + row0 * dr.cols, sl[k].rows, row0, rope_on && images ? rr.data() : nullptr);
      }
      return WindowOf(dr.t, dr.stream);
    };
    // the real shape: ONE call over the whole tail, the rope rows out of rope_rows_out
    dr.Prepare(feat, c.p0);
    InjectFeatureRowsFromHost(dr.t, dr.stream, dr.arena, dr.staging.data(), 64, feat.data() + s * dr.cols, c.n - s, s, images ? tail_rope.data() : nullptr);
    const Window tail = WindowOf(dr.t, dr.stream);
    Check(tail == ref, std::string("Gate C [") + c.name + "]: the tail-fed drafter ring (K and V of every draft layer over its window, frontier, lower bound) == the fully fed one");
    Check(ref.injected == c.n, std::string("[") + c.name + "]: the reference drafter's frontier is n");

    if (c.n - c.p0 > window + 128) {
      // negative controls
      if (images) Check(!(tail_window(s, /*rope_on=*/false, -1) == ref), std::string("NEGATIVE CONTROL [") + c.name + "]: dropping the image rope rows changes the ring");
      Check(!(tail_window(s + 64, true, -1) == ref), std::string("NEGATIVE CONTROL [") + c.name + "]: a tail one slice late changes the window");
      Check(!(tail_window(s, true, 5) == ref), std::string("NEGATIVE CONTROL [") + c.name + "]: a skipped slice changes the ring");
      const int64_t off = s + 13 <= c.n - window ? s + 13 : s - 13;
      if (off >= c.p0 && (c.n - off) % 64 != (c.n - c.p0) % 64) {
        Check(!(tail_window(off, true, -1) == ref), std::string("NEGATIVE CONTROL [") + c.name + "]: a tail starting off the call's 64-row grid changes the ring");
      }
      Check(tail_window(s, true, -1) == ref, std::string("[") + c.name + "]: the slice-by-slice tail run equals the reference too (controls did not disturb the harness)");
    }
  }

  // flag sensitivity: informational (does the TP-thread tuning table change the drafter's GEMM bytes at these shapes?)
  {
    const DrafterCase& c = cases[0];
    const std::vector<uint16_t> feat = Features(c.n, dr.cols, static_cast<uint64_t>(c.n));
    dr.Prepare(feat, 0);
    InjectFeatureRowsFromHost(dr.f, dr.stream, dr.arena, dr.staging.data(), 64, feat.data(), c.n, 0, nullptr);
    const Window with_flag = WindowOf(dr.f, dr.stream);
    Window without;
    {
      r4dx::model::Tp2TuningScope tp1(false);
      dr.Prepare(feat, 0);
      InjectFeatureRowsFromHost(dr.f, dr.stream, dr.arena, dr.staging.data(), 64, feat.data(), c.n, 0, nullptr);
      without = WindowOf(dr.f, dr.stream);
    }
    std::printf("  info: the drafter's injected K/V window %s with the thread's tuning flag (TP=2 table) vs without\n",
                with_flag == without ? "does NOT change" : "CHANGES");
  }
}

// ======================================================================================================================================
// Part b -- the stage-side capture
// ======================================================================================================================================
constexpr int64_t kTargets[] = {0, 1, 3};

ModelOptions L4Options() {
  ModelOptions o;
  o.container_path = ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
  o.layout = Layout::kBf16;
  o.max_ctx = 4096;
  o.layer_limit = 4;
  o.vision = ModelOptions::VisionMode::kOff;
  return o;
}

// A Model with the capture attached and an observer that keeps EVERY row of every call in `all` (absolute positions).
struct Reference {
  Model m;
  std::vector<uint16_t> all;
  int64_t cols = 0;
  explicit Reference(const ModelOptions& o) : m(Model::Load(o)) {
    m.AttachDflashFeatureCapture({kTargets[0], kTargets[1], kTargets[2]});
    cols = m.DflashFeatureCols();
    all.assign(static_cast<size_t>(4096 * cols), 0);
    std::vector<uint16_t>* sink = &all;
    const int64_t c = cols;
    m.SetDflashCaptureObserver([sink, c](const uint16_t* features, int64_t rows, int64_t start) {
      if (hipMemcpy(sink->data() + start * c, features, static_cast<size_t>(rows * c) * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("reference capture: D2H failed");
      }
    });
  }
};

bool RowsEqual(const PinnedBuffer<uint16_t>& tail, const std::vector<uint16_t>& all, int64_t s, int64_t rows, int64_t cols) {
  return std::memcmp(tail.data(), all.data() + s * cols, static_cast<size_t>(rows * cols) * sizeof(uint16_t)) == 0;
}

void PartB() {
  const ModelOptions o = L4Options();
  if (!FileExists(o.container_path)) {
    std::printf("SKIP part b: %s missing\n", o.container_path.c_str());
    return;
  }
  constexpr int64_t kN = 2400, kWarm = 1000;
  const std::vector<int32_t> prompt = Tokens(static_cast<int>(kN), 4);
  const std::vector<int32_t> first(prompt.begin(), prompt.begin() + kWarm), second(prompt.begin() + kWarm, prompt.end());

  Reference cold(o), warm(o);
  (void)cold.m.Prefill(prompt);
  (void)warm.m.Prefill(first);
  (void)warm.m.Prefill(second);
  const int64_t cols = cold.cols;
  Check(cols == 3 * cold.m.Config().hidden_size, "the capture has the 3 target layers' columns");

  const auto capture = [&](Model& m, const std::vector<int32_t>& ids, int64_t p0, int64_t n, PinnedBuffer<uint16_t>* buf, bool multimodal, std::vector<int32_t>* rope_rows) {
    const int64_t s = hybrid::DflashTailStart(p0, n);
    *buf = PinnedBuffer<uint16_t>(static_cast<size_t>((n - s + 64) * cols));
    m.StageArmDflashTail(buf->data(), n - s + 64, s, n);
    Check(m.StageDflashTailArmed(), "armed after StageArmDflashTail");
    if (multimodal) {
      (void)m.PrefillMultimodal(ids, {}, nullptr, rope_rows);
    } else {
      (void)m.Prefill(ids);
    }
    const int64_t rows = m.StageDisarmDflashTail();
    Check(!m.StageDflashTailArmed() && rows == n - s, "disarmed, and the number of rows captured is n - s");
    return s;
  };

  // b1. cold call on a plain Model
  {
    Model m = Model::Load(o);
    m.AttachDflashFeatureCapture({kTargets[0], kTargets[1], kTargets[2]});
    PinnedBuffer<uint16_t> buf;
    const int64_t s = capture(m, prompt, 0, kN, &buf, false, nullptr);
    Check(s == 320 && RowsEqual(buf, cold.all, s, kN - s, cols), "b1 cold call: the captured tail rows [320, 2400) equal the reference observer's rows byte for byte");
  }
  // b2. warm second call (p0 = 1000, 1400 rows < the window: the tail is the whole call), through PrefillMultimodal for the rope rows
  {
    Model m = Model::Load(o);
    m.AttachDflashFeatureCapture({kTargets[0], kTargets[1], kTargets[2]});
    (void)m.Prefill(first);
    PinnedBuffer<uint16_t> buf;
    std::vector<int32_t> rope_rows;
    const int64_t s = capture(m, second, kWarm, kN, &buf, true, &rope_rows);
    Check(s == kWarm && RowsEqual(buf, warm.all, s, kN - s, cols), "b2 warm call: the whole second call is the tail and equals the reference rows");
    const int64_t N = kN - kWarm;
    Check(static_cast<int64_t>(rope_rows.size()) == 3 * N, "b2: PrefillMultimodal's rope_rows_out is [3, n - p0]");
    const std::vector<int32_t> t = hybrid::TemporalRopeRows(rope_rows, N, s - kWarm, kN - s);
    bool positions = true;
    for (int64_t i = 0; i < kN - s; ++i) positions = positions && t[static_cast<size_t>(i)] == s + i;
    Check(positions, "b2: a text-only conversation's temporal rope row of the tail is the absolute positions");
  }
  // b3. PP-emulate (stage A / stage B composition with the carry through the host): the observer fires in stage B's drain
  {
    Model m = Model::Load(o);
    m.AttachDflashFeatureCapture({kTargets[0], kTargets[1], kTargets[2]});
    Model::PpEmulateConfig pe;
    pe.split = 2;
    m.SetPpEmulate(pe);
    PinnedBuffer<uint16_t> buf;
    const int64_t s = capture(m, prompt, 0, kN, &buf, false, nullptr);
    Check(m.PpEmulatedChunksRun() > 0 && RowsEqual(buf, cold.all, s, kN - s, cols), "b3 PP-emulate: the tail captured through the stage A / B composition equals the monolithic reference");
  }
  // b4. protocol: an incomplete call, a double arm, a taken observer slot, Reset
  {
    Model m = Model::Load(o);
    m.AttachDflashFeatureCapture({kTargets[0], kTargets[1], kTargets[2]});
    PinnedBuffer<uint16_t> buf(static_cast<size_t>(3000 * cols));
    m.StageArmDflashTail(buf.data(), 3000, 320, kN + 64);  // the call will end at 2400, short of 2464
    (void)m.Prefill(prompt);
    Check(Throws<std::runtime_error>([&] { (void)m.StageDisarmDflashTail(); }) && !m.StageDflashTailArmed(), "b4: a call that stops short of the tail end is reported by Disarm, which still disarms");
    m.StageArmDflashTail(buf.data(), 3000, 5000, 5200);
    Check(Throws<std::logic_error>([&] { m.StageArmDflashTail(buf.data(), 3000, 5000, 5200); }), "b4: arming twice is refused");
    m.Reset();
    Check(!m.StageDflashTailArmed(), "b4: Reset() drops an armed capture (it belonged to the dropped call)");
    m.SetDflashCaptureObserver([](const uint16_t*, int64_t, int64_t) {});
    Check(Throws<std::logic_error>([&] { m.StageArmDflashTail(buf.data(), 3000, 0, 100); }), "b4: a taken observer slot is refused");
    m.ClearDflashCaptureObserver();
    Model plain = Model::Load(o);
    Check(Throws<std::logic_error>([&] { plain.StageArmDflashTail(buf.data(), 3000, 0, 100); }), "b4: a Model without a feature capture is refused");
    Check(Throws<std::invalid_argument>([&] { m.StageArmDflashTail(buf.data(), 50, 0, 100); }), "b4: a buffer smaller than the tail is refused");
    Check(Throws<std::logic_error>([&] { (void)m.StageDisarmDflashTail(); }), "b4: Disarm when not armed is refused");
  }
}

// ======================================================================================================================================
// Part c -- Model::TpInjectDflashTail on the real target + drafter
// ======================================================================================================================================
void PartC() {
  const char* target = ProductionTargetPath();
  const char* drafter = ProductionDrafterPath();
  if (!FileExists(target) || !FileExists(drafter)) {
    std::printf("SKIP part c: %s or %s missing\n", target, drafter);
    return;
  }
  ModelOptions o;
  o.container_path = target;
  o.layout = r4dx::model::LayoutFromName(ProductionLayoutName());
  o.max_ctx = 4096;
  o.vision = ModelOptions::VisionMode::kOff;
  o.dflash_container = drafter;
  o.dflash_draft_k = 7;
  Model m = Model::Load(o);
  constexpr int64_t kN = 2600;
  const std::vector<int32_t> prompt = Tokens(static_cast<int>(kN), 6);
  const int64_t cols = m.DflashFeatureCols();
  const int64_t s = hybrid::DflashTailStart(0, kN);
  PinnedBuffer<uint16_t> buf(static_cast<size_t>((kN - s) * cols));

  // the reference: the ordinary prefill injects every row; the armed capture keeps the tail
  m.StageArmDflashTail(buf.data(), kN - s, s, kN);
  (void)m.Prefill(prompt);
  Check(m.StageDisarmDflashTail() == kN - s, "c: the armed capture holds the tail of the real prefill");
  const auto dflash_records = [](const hybrid::DigestList& all) {
    hybrid::DigestList out;
    for (const auto& kv : all) {
      if (kv.first.rfind("dflash.", 0) == 0) out.push_back(kv);
    }
    return out;
  };
  const hybrid::DigestList ref = dflash_records(m.DebugLiveStateDigest());
  Check(!ref.empty() && m.DflashInjectedCount() == kN, "c: the reference run injected every row");

  // the hybrid's shape: the state is there, the drafter's ring is not
  m.Reset();
  m.SetDflashInjectionEnabled(false);
  (void)m.Prefill(prompt);
  m.SetDflashInjectionEnabled(true);
  Check(m.DflashInjectedCount() == 0 && m.PositionCount() == kN, "c: after a prefill with injection off the drafter is empty and the Model is at the end of the call");
  m.TpInjectDflashTail(buf.data(), kN - s, s, nullptr);
  Check(hybrid::DiffDigests(ref, dflash_records(m.DebugLiveStateDigest())).empty(), "c: TpInjectDflashTail (rope rows nullptr) leaves the drafter window digest equal to the fully fed run's");
  // explicit rope rows = the positions: the same bytes
  m.Reset();
  m.SetDflashInjectionEnabled(false);
  (void)m.Prefill(prompt);
  m.SetDflashInjectionEnabled(true);
  std::vector<int32_t> pos_rows(static_cast<size_t>(kN - s));
  for (int64_t i = 0; i < kN - s; ++i) pos_rows[static_cast<size_t>(i)] = static_cast<int32_t>(s + i);
  m.TpInjectDflashTail(buf.data(), kN - s, s, pos_rows.data());
  Check(hybrid::DiffDigests(ref, dflash_records(m.DebugLiveStateDigest())).empty(), "c: with explicit rope rows (the positions) the digest is the same");
  // negative controls
  {
    m.Reset();
    m.SetDflashInjectionEnabled(false);
    (void)m.Prefill(prompt);
    m.SetDflashInjectionEnabled(true);
    for (int64_t i = 0; i < kN - s; ++i) pos_rows[static_cast<size_t>(i)] += 5;
    Check(Throws<std::logic_error>([&] { m.TpInjectDflashTail(buf.data(), 64, s, nullptr); }), "c: a tail that does not end at the Model's position is refused");
    Check(Throws<std::logic_error>([&] { m.TpInjectDflashTail(buf.data() + (kN - s - 1000) * cols, 1000, kN - 1000, nullptr); }), "c: a gap with a tail shorter than the window is refused");
    m.TpInjectDflashTail(buf.data(), kN - s, s, pos_rows.data());
    Check(!hybrid::DiffDigests(ref, dflash_records(m.DebugLiveStateDigest())).empty(), "NEGATIVE CONTROL c: rope rows shifted by 5 change the drafter window");
  }
}

}  // namespace

static int RunTest() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::fprintf(stderr, "[SKIP] no HIP device\n");
    return kSkipReturnCode;
  }
  const bool any = FileExists(ProductionDrafterPath()) || FileExists(ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx")) || FileExists(ProductionTargetPath());
  if (!any) return SkipMissing(ProductionDrafterPath());
  if (Part("a")) PartA();
  if (Part("b")) PartB();
  if (Part("c")) PartC();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_dflash_tail: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_dflash_tail: PASS\n");
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_dflash_tail", RunTest); }
