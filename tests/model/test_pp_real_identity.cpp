// tests/model/test_pp_real_identity.cpp -- the REAL two-GPU pipeline-parallel prefill (docs/pp-prefill.md Phase 2, gate
// G2a) leaves exactly the bits the monolithic prefill on the decode Model leaves. Needs BOTH cards and
// HIP_VISIBLE_DEVICES unset (or 0,1): stage B = the headless decode card = the last visible ordinal (physical device 1, pci
// 07), stage A = the desktop card = ordinal 0 (pci 03); R4DX_PP_DEVICES=B,A overrides. SKIPs (77) with fewer than two
// visible devices, and without the containers.
//
// One PpModel is loaded per configuration. Every scenario runs twice on it, from a fresh state: first MONOLITHIC -- the
// decode Model (stage B's Model) driven directly, its pipeline role inactive, which is today's Model::Prefill -- and then
// through the PpModel (min_rows = 1, so every prefill call is pipelined, or the threshold variant's mix of pipelined and
// decode-Model-only calls). Every observable is compared bit for bit: the last row's logits of each Prefill call, the
// greedy / speculative tokens that follow, the digest of ALL the decode Model's per-sequence state
// (Model::DebugStateDigest, with the KV caches of BOTH stages zeroed at the scenario start so a stale page cannot depend on
// what an earlier scenario left), the DFlash feature capture, and -- after every pipelined call -- the live state of stage A
// against stage B's (Model::PpLiveDigest: the GDN hand-off and the streamed KV rows are exact). PpModel itself runs with
// verify on, which makes the same comparison after every sync-back and hand-off.
//
// The scenarios are test_pp_emulate_identity's shapes (1, 63, 64, 65, 255, 256, 257, 511, 8145 rows; the prefix-reuse shapes
// 300 + 333, 64 + 511, 257 + 1, 1 + 255 + 257) plus the WARM TURNS the pipeline's mirror exists for: prefill, decode, prefill
// again (the sync-back of the GDN state and the generated KV rows), speculative rounds in between, a checkpoint save /
// restore, one-row tails, and calls below the pipelining threshold that run on the decode Model alone.
//   * 4-layer container (bf16, w4a16): splits 1, 2, 3 -- plain, a feature capture of layers 0..3, --mtp 3;
//   * the real 64-layer trellis container: split 33 (the design's default) and 5, plain / --mtp 3 / --dflash (35, 21, 5),
//     image rows;
//   * NEGATIVE CONTROLS (each must change an observable or fail the call, or the comparison proves nothing): no sync-back to
//     stage A on a warm turn (a stale mirror), no GDN import into stage B.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "pp_model.h"
#include "prefill_chunk.h"
#include "r4dx/core/dtype.hpp"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::PpModel;
using r4dx::model::PpOptions;
using TestFault = r4dx::model::PpModel::TestFault;

namespace {

const char* kL4Container = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");

using Trace = std::vector<std::pair<std::string, uint64_t>>;

uint64_t Fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}
uint64_t HashLogits(const std::vector<float>& x) { return Fnv(x.data(), x.size() * sizeof(float)); }
uint64_t HashTokens(const std::vector<int32_t>& x) { return Fnv(x.data(), x.size() * sizeof(int32_t)); }

int32_t Argmax(const std::vector<float>& v) {
  size_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) best = i;
  }
  return static_cast<int32_t>(best);
}

std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 200 + (i * 53 + salt * 17 + (i / 7) * 3) % 5000;
  return ids;
}

enum class Mode { kPlain, kMtp, kDflash, kCapture };

struct Op {
  enum class Kind { kPrefill, kDecode, kSpec, kSave, kRestore } kind;
  int n = 0;
};
struct Scenario {
  std::string name;
  std::vector<Op> ops;
  bool needs_checkpoint = false;
};

Scenario Calls(const std::string& name, std::vector<int> calls) {
  Scenario s{name, {}, false};
  for (int c : calls) s.ops.push_back({Op::Kind::kPrefill, c});
  return s;
}
Scenario Ops(const std::string& name, std::vector<Op> ops, bool ckpt = false) { return Scenario{name, std::move(ops), ckpt}; }
Op P(int n) { return {Op::Kind::kPrefill, n}; }
Op D(int n) { return {Op::Kind::kDecode, n}; }
Op S(int n) { return {Op::Kind::kSpec, n}; }
Op Save() { return {Op::Kind::kSave, 0}; }
Op Restore() { return {Op::Kind::kRestore, 0}; }

// The pipelined chunks one call of `rows` rows makes: the chunk grid is anchored at the call's start (PrefillNextChunk).
int64_t GridChunksOf(int rows) {
  int64_t n = 0;
  for (long long off = 0; off < rows;) {
    off += r4dx::model::PrefillNextChunk(rows - off, r4dx::model::kPrefillChunkWide);
    ++n;
  }
  return n;
}

// How a scenario drives a Model: the monolithic reference drives the decode Model directly, the pipeline through the PpModel.
struct Driver {
  std::function<void()> reset;  // a fresh sequence, KV caches zeroed (on both stages for the pipeline)
  std::function<std::vector<float>(const std::vector<int32_t>&)> prefill;
  std::function<std::vector<float>(const std::vector<int32_t>&, const std::vector<Model::ImageSpan>&)> prefill_mm;
  std::function<int32_t(int32_t)> decode;
  std::function<std::vector<int32_t>(int32_t, int64_t)> spec;
  std::function<void()> save, restore;
  std::function<std::vector<std::pair<std::string, uint64_t>>()> state;  // the decode Model's full digest
  // After a Prefill call: the pipeline compares stage A's live state with stage B's (when the call was pipelined) and records
  // one observable; the reference records its expected value.
  std::function<void(Trace*, const std::string&)> after_prefill;
};

struct Capture {
  uint64_t chain = 1469598103934665603ull;
  int64_t rows = 0, calls = 0;
  bool on = false;
};

void RunScenario(Driver& d, Mode mode, int64_t spec_k, const Scenario& sc, Capture* cap, Trace* tr) {
  d.reset();
  int total = 0;
  for (const Op& op : sc.ops) {
    if (op.kind == Op::Kind::kPrefill) total += op.n;
  }
  const std::vector<int32_t> ids = Tokens(total, /*salt=*/1);
  const auto add = [&](const std::string& what, uint64_t v) { tr->emplace_back(sc.name + "/" + what, v); };
  const auto add_state = [&](const std::string& tag) {
    for (const auto& kv : d.state()) add(tag + "/" + kv.first, kv.second);
  };
  if (cap != nullptr) *cap = Capture{};
  size_t off = 0;
  int calls = 0, decodes = 0, specs = 0;
  int32_t tok = 0;
  for (const Op& op : sc.ops) {
    switch (op.kind) {
      case Op::Kind::kPrefill: {
        const std::vector<int32_t> part(ids.begin() + static_cast<ptrdiff_t>(off),
                                        ids.begin() + static_cast<ptrdiff_t>(off + static_cast<size_t>(op.n)));
        if (cap != nullptr) cap->on = true;
        const std::vector<float> logits = d.prefill(part);
        if (cap != nullptr) cap->on = false;
        off += static_cast<size_t>(op.n);
        const std::string tag = "call" + std::to_string(calls++);
        add(tag + "/logits", HashLogits(logits));
        add_state(tag);
        d.after_prefill(tr, sc.name + "/" + tag);
        tok = Argmax(logits);
        break;
      }
      case Op::Kind::kDecode: {
        std::vector<int32_t> toks;
        for (int i = 0; i < op.n; ++i) {
          tok = d.decode(tok);
          toks.push_back(tok);
        }
        add("decode" + std::to_string(decodes++) + "/tokens", HashTokens(toks));
        break;
      }
      case Op::Kind::kSpec: {
        std::vector<int32_t> toks;
        for (int i = 0; i < op.n; ++i) {
          const std::vector<int32_t> round = d.spec(tok, spec_k);
          toks.insert(toks.end(), round.begin(), round.end());
          tok = round.back();
        }
        add("spec" + std::to_string(specs++) + "/tokens", HashTokens(toks));
        break;
      }
      case Op::Kind::kSave:
        d.save();
        break;
      case Op::Kind::kRestore:
        d.restore();
        break;
    }
  }
  (void)mode;
  if (cap != nullptr) {
    add("capture/rows", static_cast<uint64_t>(cap->rows));
    add("capture/callbacks", static_cast<uint64_t>(cap->calls));
    add("capture/content", cap->chain);
  }
  add_state("end");
}

size_t Differing(const Trace& a, const Trace& b, const std::string& who, bool print) {
  if (a.size() != b.size()) {
    if (print) std::fprintf(stderr, "FAIL %s: trace sizes differ (%zu vs %zu)\n", who.c_str(), a.size(), b.size());
    return a.size() > b.size() ? a.size() : b.size();
  }
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first || a[i].second != b[i].second) {
      if (print && bad < 12) {
        std::fprintf(stderr, "FAIL %s: %s differs: monolithic %016llx, pipelined %016llx\n", who.c_str(), a[i].first.c_str(),
                     static_cast<unsigned long long>(a[i].second), static_cast<unsigned long long>(b[i].second));
      }
      ++bad;
    }
  }
  return bad;
}

struct Variant {
  std::string name;
  int64_t split;
  int64_t min_rows = 1;
  TestFault fault = TestFault::kNone;
  bool expect_equal = true;  // false: a negative control, the comparison must FAIL (or the call must throw)
  std::vector<Scenario> scenarios;
};

bool Only(const std::string& name) {
  // R4DX_TEST_ONLY=<substring>: run only the configurations whose name contains it (a debugging aid).
  if (const char* only = std::getenv("R4DX_TEST_ONLY")) {
    if (*only != '\0' && name.find(only) == std::string::npos) return false;
  }
  return true;
}

// The two drivers over one PpModel.
struct Drivers {
  Driver mono, pipe;
};

Drivers MakeDrivers(PpModel& pp, Mode mode, Capture* cap_state) {
  Model& b = pp.DecodeModel();
  Drivers d;
  const auto spec = [mode](auto& m, int32_t tok, int64_t k) {
    return mode == Mode::kMtp ? m.DecodeStepMtpGreedy(tok, k) : m.DecodeStepDflashGreedy(tok, k, /*p_min=*/0.0f, /*n_min=*/0);
  };

  // Monolithic: the decode Model directly (its pipeline role is inactive outside a pipelined call).
  d.mono.reset = [&b] {
    b.Reset();
    b.DebugZeroKvState();
  };
  d.mono.prefill = [&b](const std::vector<int32_t>& ids) { return b.Prefill(ids); };
  d.mono.prefill_mm = [&b](const std::vector<int32_t>& ids, const std::vector<Model::ImageSpan>& sp) {
    return b.PrefillMultimodal(ids, sp);
  };
  d.mono.decode = [&b](int32_t t) { return b.DecodeStepGreedy(t); };
  d.mono.spec = [&b, spec](int32_t t, int64_t k) { return spec(b, t, k); };
  d.mono.save = [&b] { b.SaveCheckpoint(); };
  d.mono.restore = [&b] { b.RestoreCheckpoint(); };
  d.mono.state = [&b] { return b.DebugStateDigest(); };
  d.mono.after_prefill = [](Trace* tr, const std::string& tag) { tr->emplace_back(tag + "/stage A == stage B", 1); };

  // Pipelined: through the PpModel.
  auto last_piped = std::make_shared<int64_t>(0);
  d.pipe.reset = [&pp, &b] {
    pp.Reset();
    b.DebugZeroKvState();
    pp.RunOnStageA([](Model& a) { a.DebugZeroKvState(); });
  };
  d.pipe.prefill = [&pp](const std::vector<int32_t>& ids) { return pp.Prefill(ids); };
  d.pipe.prefill_mm = [&pp](const std::vector<int32_t>& ids, const std::vector<Model::ImageSpan>& sp) {
    return pp.PrefillMultimodal(ids, sp);
  };
  d.pipe.decode = [&pp](int32_t t) { return pp.DecodeStepGreedy(t); };
  d.pipe.spec = [&pp, mode](int32_t t, int64_t k) {
    return mode == Mode::kMtp ? pp.DecodeStepMtpGreedy(t, k) : pp.DecodeStepDflashGreedy(t, k, 0.0f, 0);
  };
  d.pipe.save = [&pp] { pp.SaveCheckpoint(); };
  d.pipe.restore = [&pp] { pp.RestoreCheckpoint(); };
  d.pipe.state = [&b] { return b.DebugStateDigest(); };
  d.pipe.after_prefill = [&pp, &b, last_piped](Trace* tr, const std::string& tag) {
    const int64_t now = pp.GetStats().pipelined_calls;
    uint64_t ok = 1;
    if (now != *last_piped) {  // this call went through the pipeline: the hand-off must be exact
      std::vector<std::pair<std::string, uint64_t>> da;
      const int64_t split = pp.Split();
      pp.RunOnStageA([&da, split](Model& a) { da = a.PpLiveDigest(split); });
      ok = da == b.PpLiveDigest(split) ? 1 : 0;
      if (ok == 0) std::fprintf(stderr, "FAIL %s: stage A's live state differs from stage B's after the hand-off\n", tag.c_str());
    }
    *last_piped = now;
    tr->emplace_back(tag + "/stage A == stage B", ok);
  };
  (void)cap_state;
  return d;
}

// One PpModel per configuration; the monolithic baseline of every scenario any variant uses, then each variant against it.
bool RunConfig(const std::string& name, ModelOptions opts, Mode mode, int64_t spec_k, int64_t reserve_split,
               const std::vector<Variant>& variants, bool vision, int64_t vision_split);

// ---- image prompts (test_pp_emulate_identity's synthetic rows, driven through both paths) ---------------------------------
struct ImageRows {
  r4dx::core::DeviceBuffer<uint16_t> dev;
  int64_t tokens = 0;
  int64_t side = 0;  // patch grid h == w
};

ImageRows MakeImageRows(int64_t merge, int64_t tokens_side, int64_t hidden, int salt) {
  ImageRows r;
  r.side = tokens_side * merge;
  r.tokens = tokens_side * tokens_side;
  std::vector<uint16_t> host(static_cast<size_t>(r.tokens * hidden));
  uint32_t x = 12345u + static_cast<uint32_t>(salt);
  for (auto& v : host) {
    x = x * 1664525u + 1013904223u;
    v = r4dx::core::FloatToBf16((static_cast<float>(x >> 9) / static_cast<float>(1u << 23) - 0.5f) * 0.5f);
  }
  r.dev = r4dx::core::DeviceBuffer<uint16_t>(host.size());
  r.dev.CopyFromHost(host);
  return r;
}

void RunVisionScenarios(Driver& d, Model& b, Mode mode, int64_t spec_k, Trace* tr) {
  const int64_t hidden = b.Config().hidden_size;
  const int64_t merge = b.GetContainer().VisionCfg().spatial_merge_size;
  const int32_t image_id = static_cast<int32_t>(b.GetContainer().ImageTokenId());
  const auto add = [&](const std::string& n, uint64_t v) { tr->emplace_back(n, v); };
  const auto add_state = [&](const std::string& tag) {
    for (const auto& kv : d.state()) add(tag + "/" + kv.first, kv.second);
  };
  const auto image_span = [&](const ImageRows& rows, int64_t offset) {
    Model::ImageSpan sp;
    sp.offset = offset;
    sp.tokens = rows.tokens;
    sp.grid.t = 1;
    sp.grid.h = rows.side;
    sp.grid.w = rows.side;
    sp.embeds = rows.dev.data();
    sp.embeds_on_host = false;
    return sp;
  };
  const auto tail_steps = [&](const std::string& tag, const std::vector<float>& logits) {
    int32_t tok = Argmax(logits);
    std::vector<int32_t> toks;
    for (int i = 0; i < 2; ++i) {
      toks.push_back(d.decode(tok));
      tok = toks.back();
    }
    for (int i = 0; i < 2 && mode != Mode::kPlain; ++i) {
      const std::vector<int32_t> round = d.spec(tok, spec_k);
      toks.insert(toks.end(), round.begin(), round.end());
      tok = round.back();
    }
    add(tag + "/tokens", HashTokens(toks));
    add_state(tag + "/end");
  };
  const auto build = [&](int text_before, const ImageRows* img, int text_after, std::vector<int32_t>* ids, int64_t* offset) {
    const std::vector<int32_t> a = Tokens(text_before, 3), c = Tokens(text_after, 5);
    ids->assign(a.begin(), a.end());
    *offset = static_cast<int64_t>(ids->size());
    if (img != nullptr) ids->insert(ids->end(), static_cast<size_t>(img->tokens), image_id);
    ids->insert(ids->end(), c.begin(), c.end());
  };

  const ImageRows big = MakeImageRows(merge, 16, hidden, 1);    // 256 tokens
  const ImageRows small = MakeImageRows(merge, 12, hidden, 2);  // 144 tokens
  std::vector<int32_t> ids;
  int64_t off = 0;

  d.reset();  // A: text 100 + image 256 + text 200
  build(100, &big, 200, &ids, &off);
  tail_steps("img-mid", d.prefill_mm(ids, {image_span(big, off)}));

  d.reset();  // B: image at position 0 + text 300
  build(0, &small, 300, &ids, &off);
  tail_steps("img-first", d.prefill_mm(ids, {image_span(small, off)}));

  d.reset();  // C: a conversation -- image prompt, text continuation, a text-only multimodal call, decode between
  build(60, &small, 100, &ids, &off);
  std::vector<float> logits = d.prefill_mm(ids, {image_span(small, off)});
  add("conv/call0/logits", HashLogits(logits));
  add_state("conv/call0");
  d.after_prefill(tr, "conv/call0");
  int32_t tok = Argmax(logits);
  for (int i = 0; i < 3; ++i) tok = d.decode(tok);  // the rope rows now carry the image's delta; stage A must learn it
  logits = d.prefill(Tokens(300, 7));
  add("conv/call1/logits", HashLogits(logits));
  add_state("conv/call1");
  d.after_prefill(tr, "conv/call1");
  logits = d.prefill_mm(Tokens(270, 9), {});
  add("conv/call2/logits", HashLogits(logits));
  d.after_prefill(tr, "conv/call2");
  tail_steps("conv", logits);
}

bool RunConfig(const std::string& name, ModelOptions opts, Mode mode, int64_t spec_k, int64_t reserve_split,
               const std::vector<Variant>& variants, bool vision, int64_t vision_split) {
  if (!Only(name)) return true;
  std::fprintf(stderr, "[pp-real] %s: %zu variants%s\n", name.c_str(), variants.size(), vision ? " + image rows" : "");
  opts.pp_emulate_split = 0;
  PpOptions po;
  po.split = variants.empty() ? vision_split : variants[0].split;
  po.reserve_split = static_cast<int>(reserve_split);
  po.min_rows = 1;
  po.verify = true;  // the PpModel digests both stages after every sync-back and hand-off, too
  po.timeout_ms = 120000;
  std::unique_ptr<PpModel> pp = PpModel::Load(opts, po);
  Model& b = pp->DecodeModel();
  Capture cap;
  if (mode == Mode::kCapture) {
    pp->AttachDflashFeatureCapture({0, 1, 2, 3});
    b.SetDflashCaptureObserver([&cap, &b](const uint16_t* features, int64_t rows, int64_t) {
      if (!cap.on) return;
      const int64_t elems = rows * b.DflashFeatureCols();
      ++cap.calls;
      cap.rows += rows;
      if (elems > 0) {
        std::vector<uint16_t> host(static_cast<size_t>(elems));
        (void)hipMemcpy(host.data(), features, host.size() * 2, hipMemcpyDeviceToHost);
        cap.chain = (cap.chain ^ Fnv(host.data(), host.size() * 2)) * 1099511628211ull;
      }
    });
  }
  Drivers drv = MakeDrivers(*pp, mode, &cap);
  Capture* capp = mode == Mode::kCapture ? &cap : nullptr;
  bool ok = true;

  // The monolithic baseline of every scenario any variant uses.
  std::map<std::string, Trace> base;
  for (const Variant& v : variants) {
    for (const Scenario& sc : v.scenarios) {
      if (base.count(sc.name) != 0) continue;
      RunScenario(drv.mono, mode, spec_k, sc, capp, &base[sc.name]);
    }
  }
  for (const Variant& v : variants) {
    pp->SetSplit(v.split);
    pp->SetMinRows(v.min_rows);
    pp->SetTestFault(v.fault);
    size_t observables = 0, bad = 0;
    int64_t expect_chunks = 0, expect_calls = 0;
    const PpModel::Stats before = pp->GetStats();
    bool threw = false;
    for (const Scenario& sc : v.scenarios) {
      for (const Op& op : sc.ops) {
        if (op.kind == Op::Kind::kPrefill && op.n >= v.min_rows) {
          expect_chunks += GridChunksOf(op.n);
          ++expect_calls;
        }
      }
      Trace t;
      try {
        RunScenario(drv.pipe, mode, spec_k, sc, capp, &t);
      } catch (const std::exception& e) {
        threw = true;
        if (v.expect_equal) {
          std::fprintf(stderr, "FAIL %s/%s/%s: %s\n", name.c_str(), v.name.c_str(), sc.name.c_str(), e.what());
          ok = false;
        }
        // A pipeline left in kNeedsRecovery heals with Reset() (the next scenario's reset does exactly that).
        continue;
      }
      observables += t.size();
      bad += Differing(base[sc.name], t, name + "/" + v.name + "/" + sc.name, v.expect_equal);
    }
    pp->SetTestFault(TestFault::kNone);
    const PpModel::Stats after = pp->GetStats();
    if (v.expect_equal) {
      if (after.pipelined_calls - before.pipelined_calls != expect_calls || after.chunks - before.chunks != expect_chunks) {
        std::fprintf(stderr, "FAIL %s/%s: %lld calls / %lld chunks went through the pipeline, the scenarios have %lld / %lld\n",
                     name.c_str(), v.name.c_str(), static_cast<long long>(after.pipelined_calls - before.pipelined_calls),
                     static_cast<long long>(after.chunks - before.chunks), static_cast<long long>(expect_calls),
                     static_cast<long long>(expect_chunks));
        ok = false;
      }
      if (bad != 0) {
        std::fprintf(stderr, "FAIL %s/%s: %zu of %zu observables differ from the monolithic run\n", name.c_str(), v.name.c_str(),
                     bad, observables);
        ok = false;
      } else if (!threw) {
        std::fprintf(stderr, "[PASS] %s/%s: %zu observables bit-identical, monolithic vs pipelined (%lld calls, %lld chunks; "
                             "sync-backs: %lld GDN + %lld KV rows)\n",
                     name.c_str(), v.name.c_str(), observables, static_cast<long long>(after.pipelined_calls - before.pipelined_calls),
                     static_cast<long long>(after.chunks - before.chunks), static_cast<long long>(after.sync_gdn - before.sync_gdn),
                     static_cast<long long>(after.sync_kv_rows - before.sync_kv_rows));
      }
    } else if (bad == 0 && !threw) {
      std::fprintf(stderr, "FAIL %s/%s: NEGATIVE CONTROL passed -- the fault changed nothing, so the comparison cannot see the "
                           "part of the pipeline it breaks\n", name.c_str(), v.name.c_str());
      ok = false;
    } else {
      std::fprintf(stderr, "[PASS] %s/%s: negative control -- the fault %s\n", name.c_str(), v.name.c_str(),
                   threw ? "makes a call fail" : "changes observables");
    }
  }

  if (vision && Only(name + "/vision")) {
    if (!b.GetContainer().HasVisionConfig()) {
      std::fprintf(stderr, "[SKIP] %s: the container has no vision config\n", name.c_str());
    } else {
      pp->SetSplit(vision_split);
      pp->SetMinRows(1);
      Trace mono, pipe;
      RunVisionScenarios(drv.mono, b, mode, spec_k, &mono);
      const PpModel::Stats before = pp->GetStats();
      RunVisionScenarios(drv.pipe, b, mode, spec_k, &pipe);
      const PpModel::Stats after = pp->GetStats();
      const size_t bad = Differing(mono, pipe, name + "/vision", true);
      if (bad != 0) {
        std::fprintf(stderr, "FAIL %s/vision: %zu of %zu observables differ\n", name.c_str(), bad, mono.size());
        ok = false;
      } else if (after.pipelined_calls == before.pipelined_calls) {
        std::fprintf(stderr, "FAIL %s/vision: no call went through the pipeline\n", name.c_str());
        ok = false;
      } else {
        std::fprintf(stderr, "[PASS] %s/vision: %zu observables bit-identical at split %lld (%lld pipelined calls)\n",
                     name.c_str(), mono.size(), static_cast<long long>(vision_split),
                     static_cast<long long>(after.pipelined_calls - before.pipelined_calls));
      }
    }
  }
  std::fprintf(stderr, "[pp-real] %s: %s\n", name.c_str(), pp->StatsLine().c_str());
  return ok;
}

std::vector<Scenario> TailScenarios(bool with_8k) {
  std::vector<Scenario> s;
  for (int n : {1, 63, 64, 65, 255, 256, 257, 511}) s.push_back(Calls("len" + std::to_string(n), {n}));
  if (with_8k) s.push_back(Calls("len8145", {8145}));
  s.push_back(Calls("split300+333", {300, 333}));
  s.push_back(Calls("split64+511", {64, 511}));
  s.push_back(Calls("split257+1", {257, 1}));
  s.push_back(Calls("split1+255+257", {1, 255, 257}));
  return s;
}

// The warm turns: what the mirror (stage A's copy of the conversation) exists for.
std::vector<Scenario> WarmScenarios(bool speculative, bool checkpoint) {
  std::vector<Scenario> s;
  s.push_back(Ops("turn", {P(700), D(6), P(400), D(4), P(1100), D(2)}));
  s.push_back(Ops("tails", {P(600), D(1), P(1), D(1), P(257), D(1), P(63), D(2)}));
  s.push_back(Ops("below-threshold", {P(1200), D(2), P(100), D(2), P(1500), D(2)}));
  if (speculative) s.push_back(Ops("turn-spec", {P(600), S(3), P(300), S(2), P(520), S(2)}));
  if (checkpoint) s.push_back(Ops("ckpt", {P(600), Save(), D(5), Restore(), P(500), D(3), Restore(), P(70), D(1)}, true));
  return s;
}

}  // namespace

static int RunTest() {
  int fails = 0;
  int ran = 0;
  const auto run = [&](bool ok) {
    ++ran;
    if (!ok) ++fails;
  };

  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) {
    std::fprintf(stderr, "[SKIP] the real pipeline needs two visible HIP devices (HIP_VISIBLE_DEVICES unset), found %d\n", devices);
    return kSkipReturnCode;
  }
  // Placement is PpModel's auto rule (stage B = the last visible ordinal = physical device 1, the headless card, stage A =
  // ordinal 0 = the desktop card) unless R4DX_PP_DEVICES=B,A says otherwise; every PpModel::Load logs the PCI buses.

  if (FileExists(kL4Container)) {
    for (Layout layout : {Layout::kBf16, Layout::kW4a16}) {
      const std::string ln = layout == Layout::kBf16 ? "bf16" : "w4a16";
      ModelOptions base;
      base.container_path = kL4Container;
      base.layout = layout;
      base.max_ctx = 8448;
      base.layer_limit = 4;
      base.prompt_checkpoint = true;

      // plain: splits 1, 2, 3 of the 4 layers over the whole shape list, the warm turns (sync-back), the threshold mix
      // (calls below 1024 rows run on the decode Model alone), and the two negative controls.
      std::vector<Variant> plain;
      for (int64_t k : {1, 2, 3}) {
        std::vector<Scenario> sc = TailScenarios(/*with_8k=*/true);
        for (Scenario& w : WarmScenarios(false, true)) sc.push_back(std::move(w));
        plain.push_back({"split" + std::to_string(k), k, 1, TestFault::kNone, true, sc});
      }
      plain.push_back({"split2+threshold1024", 2, 1024, TestFault::kNone, true, WarmScenarios(false, true)});
      plain.push_back({"neg/skip-sync-back", 2, 1, TestFault::kSkipSyncBack, false, {Ops("turn", {P(700), D(6), P(400), D(4)})}});
      plain.push_back({"neg/skip-gdn-import", 2, 1, TestFault::kSkipGdnImport, false, {Calls("len300", {300})}});
      run(RunConfig("l4/" + ln + "/plain", base, Mode::kPlain, 0, 3, plain, false, 0));

      // a target feature capture of layers 0..3, drained per chunk: at split 2 the columns of layers 0, 1 are stage A's.
      std::vector<Variant> capture;
      for (int64_t k : {1, 2, 3}) {
        std::vector<Scenario> sc = TailScenarios(false);
        sc.push_back(Ops("turn", {P(700), D(6), P(400), D(2)}));
        capture.push_back({"split" + std::to_string(k), k, 1, TestFault::kNone, true, sc});
      }
      run(RunConfig("l4/" + ln + "/capture", base, Mode::kCapture, 0, 3, capture, false, 0));

      ModelOptions mo = base;
      mo.mtp_draft_k = 3;
      std::vector<Scenario> mtp_sc = TailScenarios(false);
      for (Scenario& w : WarmScenarios(true, true)) mtp_sc.push_back(std::move(w));
      run(RunConfig("l4/" + ln + "/mtp3", mo, Mode::kMtp, 3, 3, {{"split2", 2, 1, TestFault::kNone, true, mtp_sc}}, false, 0));
    }
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the 4-layer part did not run\n", kL4Container);
  }

  const char* real = r4dx_test::ProductionTargetPath();
  const char* drafter = r4dx_test::ProductionDrafterPath();
  if (FileExists(real)) {
    ModelOptions base;
    base.container_path = real;
    base.layout = r4dx::model::LayoutFromName(r4dx_test::ProductionLayoutName());
    base.max_ctx = 8448;
    base.prompt_checkpoint = true;
    std::vector<Scenario> sc;
    for (int n : {63, 64, 255, 256, 257, 511, 1100}) sc.push_back(Calls("len" + std::to_string(n), {n}));
    sc.push_back(Calls("split300+333", {300, 333}));
    sc.push_back(Calls("split257+1", {257, 1}));
    std::vector<Scenario> sc33 = sc;
    sc33.push_back(Calls("len8145", {8145}));
    for (Scenario& w : WarmScenarios(false, true)) {
      sc33.push_back(w);
      sc.push_back(w);
    }
    const std::vector<Scenario> neg_turn = {Ops("turn", {P(700), D(6), P(400), D(4)})};

    std::vector<Variant> plain;
    plain.push_back({"split33", 33, 1, TestFault::kNone, true, sc33});
    plain.push_back({"split5", 5, 1, TestFault::kNone, true, sc});
    plain.push_back({"split33+threshold1024", 33, 1024, TestFault::kNone, true, WarmScenarios(false, true)});
    plain.push_back({"neg/skip-sync-back", 33, 1, TestFault::kSkipSyncBack, false, neg_turn});
    plain.push_back({"neg/skip-gdn-import", 33, 1, TestFault::kSkipGdnImport, false, {Calls("len300", {300})}});
    run(RunConfig("real/plain", base, Mode::kPlain, 0, 33, plain, /*vision=*/true, /*vision_split=*/33));

    std::vector<Scenario> spec_sc = {Calls("len257", {257}), Calls("len511", {511}), Calls("split300+333", {300, 333})};
    for (Scenario& w : WarmScenarios(true, true)) spec_sc.push_back(std::move(w));
    ModelOptions mo = base;
    mo.mtp_draft_k = 3;
    run(RunConfig("real/mtp3", mo, Mode::kMtp, 3, 33,
                  {{"split33", 33, 1, TestFault::kNone, true, spec_sc}, {"split5", 5, 1, TestFault::kNone, true, spec_sc}},
                  false, 0));

    if (FileExists(drafter)) {
      ModelOptions dof = base;
      dof.dflash_container = drafter;
      dof.dflash_draft_k = 7;
      dof.prompt_checkpoint = true;
      // The drafter's target layers are {6, 20, 34, 48, 62}: split 35 (the default with --dflash) hands stage A the columns
      // of layers 6, 20 and 34; split 21 those of 6 and 20; split 5 none.
      run(RunConfig("real/dflash7", dof, Mode::kDflash, 7, 35,
                    {{"split35", 35, 1, TestFault::kNone, true, spec_sc},
                     {"split21", 21, 1, TestFault::kNone, true, spec_sc},
                     {"split5", 5, 1, TestFault::kNone, true, spec_sc}},
                    /*vision=*/true, /*vision_split=*/35));
    } else {
      std::fprintf(stderr, "[SKIP] %s not found -- the DFlash part did not run\n", drafter);
    }
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the real-container part did not run\n", real);
  }

  if (ran == 0) return SkipMissing(kL4Container);
  if (fails != 0) {
    std::fprintf(stderr, "test_pp_real_identity: %d of %d configurations FAILED\n", fails, ran);
    return 1;
  }
  std::fprintf(stderr, "test_pp_real_identity: PASS (%d configurations)\n", ran);
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_pp_real_identity", RunTest); }
