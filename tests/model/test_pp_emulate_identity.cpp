// tests/model/test_pp_emulate_identity.cpp -- the PP-emulate mode (docs/pp-prefill.md, Phase 1 gate G1b) leaves
// exactly the bits the monolithic prefill leaves. ONE Model is loaded per configuration; every scenario runs
// first with the emulation off (RunChunk = prologue + one layer range + epilogue) and then with it on at a split
// layer: every prompt-prefill chunk runs as stage A = layers [0, split) and stage B = [split, N) with the
// inter-stage carry (the residual stream, the fused-norm pair, the DFlash feature columns stage A captured, the
// KV rows its attention layers wrote) exported to pinned host memory, the device-side destinations poisoned (0xFF)
// and the carry imported again -- what the two-GPU pipeline does over PCIe. Every observable is compared bit for
// bit: the last row's logits of each Prefill call, the greedy tokens that follow, a digest of ALL the
// per-sequence state (Model::DebugStateDigest, with the KV caches zeroed at the start of every scenario so a
// stale page cannot depend on what an earlier scenario left), the speculative rounds' tokens and the DFlash
// capture drained through Prefill's per-chunk callback.
//
// The shape list is test_prefill_chunk_identity's (1, 63, 64, 65, 255, 256, 257, 511, 8145 rows; the prefix-reuse
// shapes 300 + 333, 64 + 511, 257 + 1, 1 + 255 + 257), plain / --mtp 3 / --dflash, and synthetic image rows. The
// default production paths run (int8 prefill, split-KV attention): both sides of a comparison make the same
// choices, so nothing is pinned.
//
// What each part proves:
//   * 4-layer container (bf16, w4a16): splits 1, 2, 3 of 4 -- the carry, the DFlash feature columns (a capture of
//     layers 0..3 splits between the stages); its one attention layer (layer 3) is always stage B's;
//   * the real 64-layer trellis container (SKIPs without it): split 33 (the design's default: 8 attention layers
//     per stage, the KV rows of stage A's go through the host), split 5 (one attention layer in stage A, an odd
//     ping-pong parity: `cur` leaves stage A in buf_b_ and arrives in buf_a_), split 35 with the drafter;
//   * the arena poison controls: 0xFF in the whole activation arena at the stage boundary, and after every
//     layer's scratch release -- a kernel that reads scratch it did not write would change a byte;
//   * NEGATIVE CONTROLS (each must FAIL the comparison, or the comparison proves nothing): the residual stream
//     imported one byte off, the fused-norm buffer dropped, the KV rows dropped, the DFlash columns dropped.
// SKIPs (CTest SKIPPED) for a container that is missing; an exception from a present one is a FAIL.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "prefill_chunk.h"
#include "r4dx/core/dtype.hpp"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using Fault = r4dx::model::Model::PpEmulateConfig::Fault;

namespace {

const char* kL4Container = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");

using Trace = std::vector<std::pair<std::string, uint64_t>>;

uint64_t Fnv(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = 1469598103934665603ull;
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

struct Scenario {
  std::string name;
  std::vector<int> calls;  // Prefill call lengths, in order (each continues the previous)
};

enum class Mode { kPlain, kMtp, kDflash, kCapture };

// How many chunks the grid makes of all the calls of a scenario: the emulation must have run exactly these.
int64_t GridChunks(const Scenario& sc) {
  int64_t n = 0;
  for (int c : sc.calls) {
    for (long long off = 0; off < c;) {
      off += r4dx::model::PrefillNextChunk(c - off, r4dx::model::kPrefillChunkWide);
      ++n;
    }
  }
  return n;
}

// One scenario on `m`: reset (and zero the KV caches), the Prefill calls, then 2 plain decode steps and
// (drafter configurations) 3 speculative rounds; every observable is appended to `tr` under `<scenario>/<what>`.
void RunScenario(Model& m, Mode mode, int64_t spec_k, const Scenario& sc, Trace* tr) {
  m.Reset();
  m.DebugZeroKvState();
  int total = 0;
  for (int c : sc.calls) total += c;
  const std::vector<int32_t> ids = Tokens(total, /*salt=*/1);
  const auto add = [&](const std::string& what, uint64_t v) { tr->emplace_back(sc.name + "/" + what, v); };
  const auto add_state = [&](const std::string& tag) {
    for (const auto& kv : m.DebugStateDigest()) add(tag + "/" + kv.first, kv.second);
  };
  uint64_t feature_chain = 1469598103934665603ull;
  int64_t feature_rows = 0, feature_calls = 0;
  const auto drain = [&] {
    const int64_t rows = m.DflashFeatureRows();
    const int64_t elems = rows * m.DflashFeatureCols();
    ++feature_calls;
    feature_rows += rows;
    if (elems > 0) {
      std::vector<uint16_t> host(static_cast<size_t>(elems));
      (void)hipMemcpy(host.data(), m.DflashFeatureBuffer(), host.size() * 2, hipMemcpyDeviceToHost);
      const uint64_t h = Fnv(host.data(), host.size() * 2);
      feature_chain = (feature_chain ^ h) * 1099511628211ull;
    }
  };
  const std::function<void()> cb = mode == Mode::kCapture ? std::function<void()>(drain) : nullptr;

  std::vector<float> logits;
  size_t off = 0;
  int k = 0;
  for (int c : sc.calls) {
    const std::vector<int32_t> part(ids.begin() + static_cast<ptrdiff_t>(off),
                                    ids.begin() + static_cast<ptrdiff_t>(off + static_cast<size_t>(c)));
    logits = m.Prefill(part, cb);
    off += static_cast<size_t>(c);
    add("call" + std::to_string(k) + "/logits", HashLogits(logits));
    add_state("call" + std::to_string(k));
    ++k;
  }
  if (mode == Mode::kCapture) {
    add("capture/rows", static_cast<uint64_t>(feature_rows));
    add("capture/callbacks", static_cast<uint64_t>(feature_calls));
    add("capture/content", feature_chain);
  }
  int32_t tok = Argmax(logits);
  std::vector<int32_t> plain;
  for (int i = 0; i < 2; ++i) {
    plain.push_back(m.DecodeStepGreedy(tok));
    tok = plain.back();
  }
  add("decode/tokens", HashTokens(plain));
  std::vector<int32_t> spec;
  for (int i = 0; i < 3 && mode != Mode::kPlain && mode != Mode::kCapture; ++i) {
    const std::vector<int32_t> round = mode == Mode::kMtp
                                           ? m.DecodeStepMtpGreedy(tok, spec_k)
                                           : m.DecodeStepDflashGreedy(tok, spec_k, /*p_min=*/0.0f, /*n_min=*/0);
    spec.insert(spec.end(), round.begin(), round.end());
    tok = round.back();
  }
  if (!spec.empty()) add("spec/tokens", HashTokens(spec));
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
        std::fprintf(stderr, "FAIL %s: %s differs: monolithic %016llx, emulated %016llx\n", who.c_str(),
                     a[i].first.c_str(), static_cast<unsigned long long>(a[i].second),
                     static_cast<unsigned long long>(b[i].second));
      }
      ++bad;
    }
  }
  return bad;
}

struct Variant {
  std::string name;
  int64_t split;
  int poison = 0;
  Fault fault = Fault::kNone;
  bool expect_equal = true;  // false: a negative control, the comparison must FAIL
  std::vector<Scenario> scenarios;
};

bool Only(const std::string& name) {
  // R4DX_TEST_ONLY=<substring>: run only the configurations whose name contains it (a debugging aid).
  if (const char* only = std::getenv("R4DX_TEST_ONLY")) {
    if (*only != '\0' && name.find(only) == std::string::npos) return false;
  }
  return true;
}

// One Model, the monolithic baseline of every scenario any variant uses, then each variant against it.
bool RunConfig(const std::string& name, ModelOptions opts, Mode mode, int64_t spec_k, const std::vector<Variant>& variants) {
  if (!Only(name)) return true;
  std::fprintf(stderr, "[pp-identity] %s: %zu variants\n", name.c_str(), variants.size());
  opts.pp_emulate_split = 0;  // off, whatever R4DX_PP_EMULATE says: the variants turn it on themselves
  Model m = Model::Load(opts);
  if (mode == Mode::kCapture) m.AttachDflashFeatureCapture({0, 1, 2, 3});
  bool ok = true;

  std::map<std::string, Trace> base;
  for (const Variant& v : variants) {
    for (const Scenario& sc : v.scenarios) {
      if (base.count(sc.name) != 0) continue;
      const int64_t before = m.PpEmulatedChunksRun();
      RunScenario(m, mode, spec_k, sc, &base[sc.name]);
      if (m.PpEmulatedChunksRun() != before) {
        std::fprintf(stderr, "FAIL %s: the monolithic baseline ran emulated chunks\n", name.c_str());
        ok = false;
      }
    }
  }
  for (const Variant& v : variants) {
    Model::PpEmulateConfig cfg;
    cfg.split = v.split;
    cfg.poison_arena = v.poison;
    cfg.fault = v.fault;
    m.SetPpEmulate(cfg);
    size_t observables = 0, bad = 0;
    int64_t expect_chunks = 0;
    const int64_t before = m.PpEmulatedChunksRun();
    for (const Scenario& sc : v.scenarios) {
      Trace t;
      RunScenario(m, mode, spec_k, sc, &t);
      expect_chunks += GridChunks(sc);
      observables += t.size();
      bad += Differing(base[sc.name], t, name + "/" + v.name + "/" + sc.name, v.expect_equal);
    }
    m.SetPpEmulate(Model::PpEmulateConfig{});
    const int64_t ran = m.PpEmulatedChunksRun() - before;
    if (ran != expect_chunks) {
      std::fprintf(stderr, "FAIL %s/%s: %lld chunks ran through the two-stage path, the grid has %lld\n", name.c_str(),
                   v.name.c_str(), static_cast<long long>(ran), static_cast<long long>(expect_chunks));
      ok = false;
    }
    if (v.expect_equal) {
      if (bad != 0) {
        std::fprintf(stderr, "FAIL %s/%s: %zu of %zu observables differ from the monolithic run\n", name.c_str(),
                     v.name.c_str(), bad, observables);
        ok = false;
      } else {
        std::fprintf(stderr, "[PASS] %s/%s: %zu observables bit-identical, monolithic vs two-stage (%lld chunks)\n",
                     name.c_str(), v.name.c_str(), observables, static_cast<long long>(ran));
      }
    } else if (bad == 0) {
      std::fprintf(stderr, "FAIL %s/%s: NEGATIVE CONTROL passed -- the fault changed nothing, so the comparison cannot "
                           "see the part of the carry it breaks\n", name.c_str(), v.name.c_str());
      ok = false;
    } else {
      std::fprintf(stderr, "[PASS] %s/%s: negative control -- the fault changes %zu of %zu observables\n", name.c_str(),
                   v.name.c_str(), bad, observables);
    }
  }
  return ok;
}

// ---- image prompts ---------------------------------------------------------------------------------------
// The same synthetic image rows test_prefill_chunk_identity splices (the vision tower's output is replaced by
// deterministic bf16 rows; the splice does not care): PrefillMultimodal walks the same chunk grid, its image
// chunks run f16, and the splice happens in the prologue (stage A) while the 3-axis rope rows are read by the
// attention layers of BOTH stages.
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

void RunVisionScenarios(Model& m, Mode mode, int64_t spec_k, Trace* tr) {
  const int64_t hidden = m.Config().hidden_size;
  const int64_t merge = m.GetContainer().VisionCfg().spatial_merge_size;
  const int32_t image_id = static_cast<int32_t>(m.GetContainer().ImageTokenId());
  const auto add = [&](const std::string& n, uint64_t v) { tr->emplace_back(n, v); };
  const auto add_state = [&](const std::string& tag) {
    for (const auto& kv : m.DebugStateDigest()) add(tag + "/" + kv.first, kv.second);
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
      toks.push_back(m.DecodeStepGreedy(tok));
      tok = toks.back();
    }
    for (int i = 0; i < 2 && mode != Mode::kPlain; ++i) {
      const std::vector<int32_t> round = mode == Mode::kMtp ? m.DecodeStepMtpGreedy(tok, spec_k)
                                                            : m.DecodeStepDflashGreedy(tok, spec_k, 0.0f, 0);
      toks.insert(toks.end(), round.begin(), round.end());
      tok = round.back();
    }
    add(tag + "/tokens", HashTokens(toks));
    add_state(tag + "/end");
  };
  const auto build = [&](int text_before, const ImageRows* img, int text_after, std::vector<int32_t>* ids,
                         int64_t* offset) {
    const std::vector<int32_t> a = Tokens(text_before, 3), b = Tokens(text_after, 5);
    ids->assign(a.begin(), a.end());
    *offset = static_cast<int64_t>(ids->size());
    if (img != nullptr) ids->insert(ids->end(), static_cast<size_t>(img->tokens), image_id);
    ids->insert(ids->end(), b.begin(), b.end());
  };
  const auto fresh = [&] {
    m.Reset();
    m.DebugZeroKvState();
  };

  const ImageRows big = MakeImageRows(merge, 16, hidden, 1);    // 256 tokens
  const ImageRows small = MakeImageRows(merge, 12, hidden, 2);  // 144 tokens
  std::vector<int32_t> ids;
  int64_t off = 0;

  fresh();  // A: text 100 + image 256 + text 200
  build(100, &big, 200, &ids, &off);
  tail_steps("img-mid", m.PrefillMultimodal(ids, {image_span(big, off)}));

  fresh();  // B: image at position 0 + text 300
  build(0, &small, 300, &ids, &off);
  tail_steps("img-first", m.PrefillMultimodal(ids, {image_span(small, off)}));

  fresh();  // C: a conversation -- image prompt, text continuation, text-only multimodal call
  build(60, &small, 100, &ids, &off);
  std::vector<float> logits = m.PrefillMultimodal(ids, {image_span(small, off)});
  add("conv/call0/logits", HashLogits(logits));
  add_state("conv/call0");
  logits = m.Prefill(Tokens(300, 7));  // rope rows now carry the image's delta
  add("conv/call1/logits", HashLogits(logits));
  add_state("conv/call1");
  logits = m.PrefillMultimodal(Tokens(270, 9), {});
  add("conv/call2/logits", HashLogits(logits));
  tail_steps("conv", logits);
}

bool RunVision(const std::string& name, ModelOptions opts, Mode mode, int64_t spec_k, int64_t split) {
  if (!Only(name)) return true;
  std::fprintf(stderr, "[pp-identity] %s\n", name.c_str());
  opts.pp_emulate_split = 0;
  Model m = Model::Load(opts);
  if (!m.GetContainer().HasVisionConfig()) {
    std::fprintf(stderr, "[SKIP] %s: the container has no vision config\n", name.c_str());
    return true;
  }
  Trace mono, emu;
  RunVisionScenarios(m, mode, spec_k, &mono);
  Model::PpEmulateConfig cfg;
  cfg.split = split;
  m.SetPpEmulate(cfg);
  const int64_t before = m.PpEmulatedChunksRun();
  RunVisionScenarios(m, mode, spec_k, &emu);
  const int64_t ran = m.PpEmulatedChunksRun() - before;
  const size_t bad = Differing(mono, emu, name, true);
  if (bad != 0) {
    std::fprintf(stderr, "FAIL %s: %zu of %zu observables differ\n", name.c_str(), bad, mono.size());
    return false;
  }
  if (ran <= 0) {
    std::fprintf(stderr, "FAIL %s: no chunk ran through the two-stage path\n", name.c_str());
    return false;
  }
  std::fprintf(stderr, "[PASS] %s: %zu observables bit-identical, monolithic vs two-stage at split %lld (%lld chunks)\n",
               name.c_str(), mono.size(), static_cast<long long>(split), static_cast<long long>(ran));
  return true;
}

std::vector<Scenario> TailScenarios(bool with_8k) {
  std::vector<Scenario> s;
  for (int n : {1, 63, 64, 65, 255, 256, 257, 511}) s.push_back({"len" + std::to_string(n), {n}});
  if (with_8k) s.push_back({"len8145", {8145}});
  s.push_back({"split300+333", {300, 333}});     // a prefix, then a suffix over one super-chunk
  s.push_back({"split64+511", {64, 511}});       // grid anchored at the second call, not at row 0
  s.push_back({"split257+1", {257, 1}});         // a one-token suffix after a wide + tail call
  s.push_back({"split1+255+257", {1, 255, 257}});  // three calls
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

  if (FileExists(kL4Container)) {
    for (Layout layout : {Layout::kBf16, Layout::kW4a16}) {
      const std::string ln = layout == Layout::kBf16 ? "bf16" : "w4a16";
      ModelOptions base;
      base.container_path = kL4Container;
      base.layout = layout;
      base.max_ctx = 8448;
      base.layer_limit = 4;

      // plain: splits 1, 2, 3 of the 4 layers; the arena poison at the boundary and after every layer; the
      // negative controls (the residual stream a byte off, the fused-norm buffer dropped).
      const std::vector<Scenario> tails = TailScenarios(/*with_8k=*/true);
      const std::vector<Scenario> neg = {{"len300", {300}}};
      std::vector<Variant> plain;
      for (int64_t k : {1, 2, 3}) plain.push_back({"split" + std::to_string(k), k, 0, Fault::kNone, true, tails});
      plain.push_back({"split2+poison-boundary", 2, 1, Fault::kNone, true, TailScenarios(false)});
      plain.push_back({"split2+poison-layers", 2, 2, Fault::kNone, true, TailScenarios(false)});
      plain.push_back({"neg/shift-cur", 2, 0, Fault::kShiftCurOneByte, false, neg});
      plain.push_back({"neg/skip-normed", 2, 0, Fault::kSkipNormed, false, neg});
      run(RunConfig("l4/" + ln + "/plain", base, Mode::kPlain, 0, plain));

      // a target feature capture of layers 0..3, drained per chunk: at split 2 the columns of layers 0, 1 are
      // stage A's, 2, 3 stage B's; the dropped DFlash columns must be seen.
      std::vector<Variant> capture;
      for (int64_t k : {1, 2, 3}) capture.push_back({"split" + std::to_string(k), k, 0, Fault::kNone, true, TailScenarios(false)});
      capture.push_back({"neg/skip-dflash", 2, 0, Fault::kSkipDflash, false, neg});
      run(RunConfig("l4/" + ln + "/capture", base, Mode::kCapture, 0, capture));

      ModelOptions mo = base;
      mo.mtp_draft_k = 3;
      run(RunConfig("l4/" + ln + "/mtp3", mo, Mode::kMtp, 3,
                    {{"split2", 2, 0, Fault::kNone, true, TailScenarios(false)}}));
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
    std::vector<Scenario> sc;
    for (int n : {63, 64, 255, 256, 257, 511, 1100}) sc.push_back({"len" + std::to_string(n), {n}});
    sc.push_back({"split300+333", {300, 333}});
    sc.push_back({"split257+1", {257, 1}});
    std::vector<Scenario> sc33 = sc;
    sc33.push_back({"len8145", {8145}});
    const std::vector<Scenario> some = {{"len256", {256}}, {"len511", {511}}, {"split300+333", {300, 333}}};
    const std::vector<Scenario> neg = {{"len300", {300}}};

    std::vector<Variant> plain;
    plain.push_back({"split33", 33, 0, Fault::kNone, true, sc33});
    plain.push_back({"split5", 5, 0, Fault::kNone, true, sc});
    plain.push_back({"split33+poison-boundary", 33, 1, Fault::kNone, true, some});
    plain.push_back({"split33+poison-layers", 33, 2, Fault::kNone, true, some});
    plain.push_back({"neg/shift-cur", 33, 0, Fault::kShiftCurOneByte, false, neg});
    plain.push_back({"neg/skip-normed", 33, 0, Fault::kSkipNormed, false, neg});
    plain.push_back({"neg/skip-kv", 33, 0, Fault::kSkipKv, false, neg});
    run(RunConfig("real/plain", base, Mode::kPlain, 0, plain));

    const std::vector<Scenario> spec_sc = {{"len257", {257}}, {"len511", {511}}, {"split300+333", {300, 333}}};
    ModelOptions mo = base;
    mo.mtp_draft_k = 3;
    run(RunConfig("real/mtp3", mo, Mode::kMtp, 3,
                  {{"split33", 33, 0, Fault::kNone, true, spec_sc}, {"split5", 5, 0, Fault::kNone, true, spec_sc}}));

    // image prompts: the splice in stage A's prologue, the 3-axis rope read by both stages' attention layers.
    run(RunVision("real/vision/plain", base, Mode::kPlain, 0, 33));

    if (FileExists(drafter)) {
      ModelOptions dof = base;
      dof.dflash_container = drafter;
      dof.dflash_draft_k = 7;
      // The drafter's target layers are {6, 20, 34, 48, 62} (docs/dflash2.md): split 35, the design's choice with
      // --dflash, hands stage A the columns of layers 6, 20 and 34; split 21 those of 6 and 20; split 5 none (the
      // drafter's whole feature block is stage B's).
      run(RunConfig("real/dflash7", dof, Mode::kDflash, 7,
                    {{"split35", 35, 0, Fault::kNone, true, spec_sc},
                     {"split21", 21, 0, Fault::kNone, true, spec_sc},
                     {"split5", 5, 0, Fault::kNone, true, spec_sc},
                     {"neg/skip-dflash", 35, 0, Fault::kSkipDflash, false, neg}}));
      run(RunVision("real/vision/dflash7", dof, Mode::kDflash, 7, 35));
    } else {
      std::fprintf(stderr, "[SKIP] %s not found -- the DFlash part did not run\n", drafter);
    }
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the real-container part did not run\n", real);
  }

  if (ran == 0) return SkipMissing(kL4Container);
  if (fails != 0) {
    std::fprintf(stderr, "test_pp_emulate_identity: %d of %d configurations FAILED\n", fails, ran);
    return 1;
  }
  std::fprintf(stderr, "test_pp_emulate_identity: PASS (%d configurations)\n", ran);
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_pp_emulate_identity", RunTest); }
