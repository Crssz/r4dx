// tests/model/test_hybrid_real_identity.cpp -- the FULL-DEPTH two-GPU gate of the hybrid serving mode (`--tp 2 --pp 2`; docs/pp-tp2-hybrid.md 4, 15, 16): the
// real production path -- TpModel with two rank threads on two cards, a stage Model next to each rank, the pinned host ring between the cards, the
// production trellis container at all 64 layers, MTP and the DFlash2 drafter -- against a TP=1 reference. test_tp_hybrid_identity does the same at 16 layers
// with the reference resident next to the hybrid; at full depth the two do not fit on one card, so the reference lives in EARLIER and LATER phases of this
// exe (or of separate processes of it) and only its results travel (small digest records, plus state files on disk):
//
//   phase ref     a TP=1 Model on the last visible card (the headless card, rank 0's) prefills the cold shapes (1024, 2125, 8145 rows): the logits hash, the
//                 digests of ReshardRef(state, rank) for both ranks, and (2125, 8145) the whole end state, DebugExportFullState, for G-H2. Freed.
//   phase hybrid  the real TpModel with the hybrid engaged (one load with --mtp 3; one with --dflash 7 when the drafter exists), the scenarios below. Every
//                 call it makes is recorded: the logits hash and each rank's DebugLiveStateDigest, and for a WARM call (after TP=2 work: a decode, a TP
//                 prefill, a checkpoint restore) the ranks' GATHERED pre-call state, written to a state file. Cold calls are checked at once against phase ref.
//   phase verify  the TP=1 Model again: for each recorded warm call it loads the recorded pre-call state, prefills the same tokens and compares.
//
//   G-H1  each pipelined call's logits equal the TP=1 reference's byte for byte and each rank's live-state digest equals ReshardRef(reference state, rank).
//   G-H2  16 greedy tokens (plain, then --mtp 3 / --dflash 7 rounds) decoded at TP=2 by the hybrid-prefilled ranks equal those decoded by the same ranks
//         after DebugImportFullState of the reference's end state (cold 2125 and 8145 rows). The speculative modes are lossless, so for --dflash 7 the
//         import run (an empty drafter ring) must agree too; the drafter itself is checked by its frontier (injected == position on both ranks), by
//         both ranks' windows agreeing, and -- because both drafters are fed from ONE assembled tail buffer, so equal windows cannot expose a wrong
//         tail -- by that buffer's digest (TpModel::HybridTailDigest: stage X's carried columns + stage Y's) against the TP=1 run's
//         StageArmDflashTail capture of the same call (the ref phase writes refs_tail.txt when the drafter exists). The injection itself
//         (tail buffer -> drafter ring, TP-thread numerics) is test_dflash_tail parts a and c.
//   Dispatch   a call below --pp-min-rows (300 rows) and a call over the stage-KV capacity S take the TP=2 prefill and are COUNTED as such; a warm pipelined
//         call continues from the TP-prefilled state (G-H1 in the verify phase); decode works after both.
//   Warm turns  call (1024 rows), decode 32 tokens at TP=2 (plain / --mtp 3 rounds), then a 1500-row hybrid call; the same after SaveCheckpoint, a decode
//         and RestoreCheckpoint.
//   NEGATIVE CONTROLS  (TpModel::SetHybridNegControl; each must be DETECTED by the comparison it targets, or the comparison proves nothing):
//         wrong-rank   rank 0's first recurrent-state op of the scatter reads rank 1's v-heads         (cold 2125, G-H1 rank digest)
//         stale-seed   stage Y keeps its own MTP seed instead of rank 0's, on a warm turn after --mtp rounds  (G-H1, the MTP KV)
//         stale-mirror the warm gather is skipped on a warm turn                                                  (G-H1, the logits)
//         skip-tail    the DFlash tail is not injected (dflash load)                                              (drafter frontier + window check)
//
//   P4 additions (dflash load; the design's "DFlash warm-turn case" and "vision + DFlash case with an image straddling the tail window"):
//     dflash/warm-dflash     call (1024 rows) -> 32 tokens decoded at TP=2 in --dflash 7 ROUNDS -> a 1500-row hybrid call (the warm gather with the drafters
//                            holding decode-time state; the DFlash tail on top of it). Checked: G-H1 from the recorded pre-call state (verify phase, a TP=1
//                            reference without an MTP head -- the dflash load has none), both drafters at the call's end with equal windows, the hybrid's
//                            assembled TAIL digest (HybridTailDigest: stage X's columns + stage Y's) equal to the TP=1 reference's capture of the same warm call
//                            (verify phase), and G-H2: the 16 plain + 16 --dflash 7 tokens decoded after the call equal those decoded from the reference's end
//                            state (the decode phase, a plain TP=2 load, DebugImportFullState of the file the verify phase wrote).
//     dflash/image-straddle  a cold PrefillMultimodal call: 1000 text rows, a 256-token synthetic image (the vision tower's output replaced by deterministic bf16
//                            rows, as test_hybrid_emulate_identity does), 1900 text rows; the 2048-row tail window starts INSIDE the image (rows 1000..1255), so the
//                            tail's temporal rope rows must carry the image's mrope positions. G-H1 vs the TP=1 reference (PrefillMultimodal), the assembled tail's
//                            features vs the reference capture, the drafters' frontier and windows, and G-H2 (the mrope delta rides the adoption; the plain and
//                            --dflash 7 decode must equal those from the reference image). The rope rows themselves are test_hybrid_emulate_identity Gate C's NC11.
//                            Needs the production container's vision config (skipped without it); the vision tower is loaded for this scenario.
//   Phases: ref -> hybrid -> verify -> decode (the last one only for the dflash warm records: a plain --tp 2 load decoding from the reference's end state).
//
// Needs BOTH cards visible (CMake sets HIP_VISIBLE_DEVICES=0,1, the natural order = unset): rank 0 = the last visible ordinal = the headless card (the
// reference also lives there), rank 1 = ordinal 0 = the desktop card. ~14 GiB per card for the hybrid at 64 layers; the reference alone ~16 GiB on one.
// Usage: test_hybrid_real_identity [--phase all|ref|hybrid|verify|decode] [--dir D] [--max-ctx N] [--hybrid-ctx N] [--keep]; R4DX_TEST_ONLY=<substring> filters
// scenarios ("mtp/", "mtp/warm", "dflash/", "dflash/warm-dflash", "dflash/image", ...). Output "[PASS] ..." / "FAIL ..." and "test_hybrid_real_identity: PASS (N configurations)"; exit 0 when
// every check passed, 1 on a failure, 77 with fewer than two devices or a missing container. Written, NOT run by its author (CPU-only session).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dflash_draft_weights.h"
#include "dflash_tail_plan.h"
#include "hybrid_dispatch.h"
#include "live_digest.h"
#include "live_state.h"
#include "model.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "reshard_plan.h"
#include "test_common.h"
#include "tp_model.h"

#ifndef R4DX_TP_TESTING
#error "test_hybrid_real_identity needs the R4DX_TP_TESTING hooks: link r4dx_model_tptest (src/model/CMakeLists.txt)"
#endif

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::PpOptions;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace hybrid = r4dx::model::hybrid;
namespace fs = std::filesystem;

namespace {

constexpr int kDecodeTokens = 16;  // G-H2
constexpr int kWarmDecode = 32;    // the warm turns' TP=2 decode between two hybrid calls
constexpr int kMtpK = 3;
constexpr int kDflashK = 7;

int g_fails = 0;
int g_configs = 0;

bool Only(const std::string& name) {
  const char* only = std::getenv("R4DX_TEST_ONLY");
  if (only == nullptr || *only == '\0') return true;
  return name.find(only) != std::string::npos;
}

// The image-straddle scenario (dflash load) needs the vision config on the reference and on the ranks (and loads the tower, +0.9 GiB on
// rank 0): only when R4DX_TEST_ONLY names it. An unfiltered run keeps the earlier footprint and leaves the scenario out ([SKIP]).
bool ImageWanted() {
  const char* only = std::getenv("R4DX_TEST_ONLY");
  return only != nullptr && *only != '\0' && Only("dflash/image-straddle");
}

class Group {
 public:
  explicit Group(std::string name) : name_(std::move(name)) {}
  void Ok(bool cond, const std::string& what) {
    ++checks_;
    if (cond) return;
    ++fails_;
    ++g_fails;
    std::fprintf(stderr, "FAIL %s: %s\n", name_.c_str(), what.c_str());
  }
  void Done(const std::string& note = "") {
    if (fails_ == 0) std::fprintf(stderr, "[PASS] %s: %d checks%s%s\n", name_.c_str(), checks_, note.empty() ? "" : " -- ", note.c_str());
  }
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  int checks_ = 0, fails_ = 0;
};

uint64_t HashLogits(const std::vector<float>& x) { return hybrid::FnvUpdate(hybrid::kFnvInit, x.data(), x.size() * sizeof(float)); }
int32_t Argmax(const std::vector<float>& v) {
  size_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) best = i;
  }
  return static_cast<int32_t>(best);
}
std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 100 + (i * 41 + salt * 977) % 5000;
  return v;
}
// ---- synthetic image rows (test_hybrid_emulate_identity's: the vision tower's output replaced by deterministic bf16 rows) --------------
// A cold image call: `before` text rows, an image of side x side merged tokens, `after` text rows.
struct ImageSpec {
  int before = 0, after = 0, side = 0, salt = 1;
  int64_t ImageTokens() const { return static_cast<int64_t>(side) * side; }
  int64_t Total() const { return before + ImageTokens() + after; }
};
// The first call of the dflash load's image scenario: the image's rows 1000..1255 contain the tail window's start (n = 3156, n - 2048 = 1108, on the
// call's 64-row grid 1088).
constexpr ImageSpec kStraddle{1000, 1900, 16, 1};
constexpr const char* kStraddleId = "img-straddle";

struct ImageCallData {
  std::vector<int32_t> ids;
  std::vector<r4dx::model::ImageSpan> spans;
  std::vector<uint16_t> host;               // [tokens][hidden] bf16
  r4dx::core::DeviceBuffer<uint16_t> dev;   // the same rows on the calling thread's device (a TP=1 Model's spans; empty for the TpModel's)
};

// `on_host`: the rows stay in `host` (TpModel::PrefillMultimodal copies them under its own lock); else they are uploaded to `dev` for a TP=1 Model.
ImageCallData BuildImageCall(const ImageSpec& s, int64_t image_id, int64_t merge, int64_t hidden, bool on_host) {
  ImageCallData d;
  const int64_t tokens = s.ImageTokens();
  d.host.resize(static_cast<size_t>(tokens * hidden));
  uint32_t x = 12345u + static_cast<uint32_t>(s.salt);
  for (uint16_t& v : d.host) {
    x = x * 1664525u + 1013904223u;
    v = r4dx::core::FloatToBf16((static_cast<float>(x >> 9) / static_cast<float>(1u << 23) - 0.5f) * 0.5f);
  }
  if (!on_host) {
    d.dev = r4dx::core::DeviceBuffer<uint16_t>(d.host.size());
    d.dev.CopyFromHost(d.host);
  }
  const std::vector<int32_t> a = Tokens(s.before, s.salt + 3), b = Tokens(s.after, s.salt + 5);
  d.ids.assign(a.begin(), a.end());
  r4dx::model::ImageSpan sp;
  sp.offset = static_cast<int64_t>(d.ids.size());
  sp.tokens = tokens;
  sp.grid.t = 1;
  sp.grid.h = static_cast<int64_t>(s.side) * merge;
  sp.grid.w = static_cast<int64_t>(s.side) * merge;
  sp.embeds = on_host ? d.host.data() : d.dev.data();
  sp.embeds_on_host = on_host;
  d.spans.push_back(sp);
  d.ids.insert(d.ids.end(), static_cast<size_t>(tokens), static_cast<int32_t>(image_id));
  d.ids.insert(d.ids.end(), b.begin(), b.end());
  return d;
}

std::string Hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}
std::string SafeName(std::string s) {
  for (char& c : s) {
    if (c == '/' || c == '\\' || c == ' ' || c == ':') c = '_';
  }
  return s;
}

struct Opts {
  std::string phase = "all";
  std::string dir;
  int64_t max_ctx = 20000;     // the ranks' and the reference's KV capacity (the "over S" call needs room above S)
  int64_t hybrid_ctx = 16384;  // --hybrid-ctx: the stage-KV capacity S (the smallest the mode accepts)
  bool keep = false;
  int ref_device = 1;          // the reference's card: the last visible ordinal (rank 0's)
  std::string Path(const std::string& name) const { return (fs::path(dir) / name).string(); }
};

// ======================================================================================================================================
// One checked call: what the hybrid did, or (the same struct) what the reference did
// ======================================================================================================================================
struct Rec {
  std::string id;        // unique
  std::string ref_key;   // the reference record this call is compared with; "" = a TP-path call (no TP=1 bit oracle)
  std::string pre_file;  // warm: the ranks' gathered pre-call state (the reference is loaded with it); "" = a fresh conversation
  int rows = 0, salt = 0;
  bool expect_mismatch = false;  // a negative control: the comparison must FAIL
  bool pipelined = false;
  int64_t pos = 0;
  uint64_t logits = 0;
  int32_t first = 0;
  hybrid::DigestList dig[2];
  // ---- dflash-load extras (saved on an "ext" / "tok" line after the record, absent for the others) ----
  int img_before = 0, img_after = 0, img_side = 0, img_salt = 1;  // an image call (img_side > 0): the ImageSpec the hybrid prefilled
  int64_t tail_rows = 0;      // the hybrid's assembled DFlash tail, rows ...
  uint64_t tail_digest = 0;   // ... and its FNV-1a (TpModel::HybridTailDigest), for the verify phase's TP=1 capture of the same call
  std::vector<int32_t> tok[2];  // G-H2 of a warm record: the tokens decoded after the call at TP=2 (0: plain, 1: --dflash 7 rounds)
  bool HasExt() const { return img_side > 0 || tail_rows > 0 || !tok[0].empty() || !tok[1].empty(); }
  ImageSpec Image() const { return ImageSpec{img_before, img_after, img_side, img_salt}; }
};

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> out;
  size_t at = 0;
  for (;;) {
    const size_t t = line.find('\t', at);
    if (t == std::string::npos) {
      out.push_back(line.substr(at));
      return out;
    }
    out.push_back(line.substr(at, t - at));
    at = t + 1;
  }
}

void SaveRecs(const std::string& path, const std::vector<Rec>& recs) {
  std::ofstream f(path, std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + path);
  for (const Rec& r : recs) {
    f << "rec\t" << r.id << '\t' << r.ref_key << '\t' << r.pre_file << '\t' << r.rows << '\t' << r.salt << '\t' << (r.expect_mismatch ? 1 : 0) << '\t'
      << (r.pipelined ? 1 : 0) << '\t' << r.pos << '\t' << Hex(r.logits) << '\t' << r.first << '\n';
    for (int k = 0; k < 2; ++k) {
      for (const auto& [name, v] : r.dig[k]) f << "dig\t" << k << '\t' << name << '\t' << Hex(v) << '\n';
    }
    if (r.HasExt()) f << "ext\t" << r.img_before << '\t' << r.img_after << '\t' << r.img_side << '\t' << r.img_salt << '\t' << r.tail_rows << '\t' << Hex(r.tail_digest) << '\n';
    for (int mode = 0; mode < 2; ++mode) {
      if (r.tok[mode].empty()) continue;
      f << "tok\t" << mode;
      for (const int32_t t : r.tok[mode]) f << '\t' << t;
      f << '\n';
    }
  }
  f.flush();
  if (!f) throw std::runtime_error("writing " + path + " failed");
}

std::vector<Rec> LoadRecs(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path + " (run the earlier phase first)");
  std::vector<Rec> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> p = SplitTabs(line);
    if (p[0] == "rec" && p.size() == 11) {
      Rec r;
      r.id = p[1];
      r.ref_key = p[2];
      r.pre_file = p[3];
      r.rows = std::stoi(p[4]);
      r.salt = std::stoi(p[5]);
      r.expect_mismatch = p[6] == "1";
      r.pipelined = p[7] == "1";
      r.pos = std::stoll(p[8]);
      r.logits = std::stoull(p[9], nullptr, 16);
      r.first = static_cast<int32_t>(std::stol(p[10]));
      out.push_back(std::move(r));
    } else if (p[0] == "dig" && p.size() == 4 && !out.empty()) {
      const int k = std::stoi(p[1]);
      if (k < 0 || k > 1) throw std::runtime_error("bad rank in " + path);
      out.back().dig[k].emplace_back(p[2], std::stoull(p[3], nullptr, 16));
    } else if (p[0] == "ext" && p.size() == 7 && !out.empty()) {
      Rec& r = out.back();
      r.img_before = std::stoi(p[1]);
      r.img_after = std::stoi(p[2]);
      r.img_side = std::stoi(p[3]);
      r.img_salt = std::stoi(p[4]);
      r.tail_rows = std::stoll(p[5]);
      r.tail_digest = std::stoull(p[6], nullptr, 16);
    } else if (p[0] == "tok" && p.size() >= 2 && !out.empty()) {
      const int mode = std::stoi(p[1]);
      if (mode < 0 || mode > 1) throw std::runtime_error("bad token mode in " + path);
      for (size_t i = 2; i < p.size(); ++i) out.back().tok[mode].push_back(static_cast<int32_t>(std::stol(p[i])));
    } else {
      throw std::runtime_error("unreadable line in " + path + ": " + line);
    }
  }
  return out;
}

hybrid::DigestList DropPrefix(const hybrid::DigestList& l, const std::string& prefix) {
  hybrid::DigestList out;
  for (const auto& kv : l) {
    if (kv.first.rfind(prefix, 0) != 0) out.push_back(kv);
  }
  return out;
}

// "" when the hybrid call is identical to the reference's, else the first difference. `has_mtp`: the ranks carry an MTP head (the --dflash load has none,
// the reference always has one, so its mtp.kv record is dropped).
std::string Mismatch(const Rec& hyb, const Rec& ref, bool has_mtp) {
  if (hyb.pos != ref.pos) return "position " + std::to_string(hyb.pos) + " vs " + std::to_string(ref.pos);
  if (hyb.logits != ref.logits) return "logits hash " + Hex(hyb.logits) + " vs " + Hex(ref.logits);
  for (int k = 0; k < 2; ++k) {
    hybrid::DigestList want = ref.dig[k];
    if (!has_mtp) want = DropPrefix(want, "mtp.kv");
    const std::string why = hybrid::DiffDigests(want, hyb.dig[k]);
    if (!why.empty()) return "rank " + std::to_string(k) + ": " + why;
  }
  return "";
}

// G-H1 for one call (or, for a negative control, the proof that the damage shows).
void CheckAgainst(Group& grp, const Rec& hyb, const Rec& ref, bool has_mtp) {
  const std::string tag = hyb.id;
  if (!hyb.pipelined) {
    grp.Ok(false, tag + ": the call did not run through the pipeline");
    return;
  }
  const std::string why = Mismatch(hyb, ref, has_mtp);
  if (hyb.expect_mismatch) {
    grp.Ok(!why.empty(), tag + ": negative control NOT detected -- the damaged call is identical to the TP=1 reference (the comparison cannot fail)");
    if (!why.empty()) std::fprintf(stderr, "[control] %s: detected (%s)\n", tag.c_str(), why.c_str());
    return;
  }
  grp.Ok(hyb.pos == ref.pos, tag + ": the position is " + std::to_string(ref.pos));
  grp.Ok(hyb.logits == ref.logits, tag + ": the pipelined prefill's logits equal the TP=1 reference's byte for byte");
  for (int k = 0; k < 2; ++k) {
    hybrid::DigestList want = ref.dig[k];
    if (!has_mtp) want = DropPrefix(want, "mtp.kv");
    const std::string d = hybrid::DiffDigests(want, hyb.dig[k]);
    grp.Ok(d.empty(), tag + ": G-H1 rank " + std::to_string(k) + "'s live-state digest equals ReshardRef(reference state): " + d);
  }
}

// ======================================================================================================================================
// The TP=1 reference
// ======================================================================================================================================
ModelOptions BaseOptions(const Opts& o, int64_t mtp_k, bool dflash, bool vision = false) {
  ModelOptions mo;
  mo.container_path = ProductionTargetPath();
  mo.layout = r4dx::model::LayoutFromName(ProductionLayoutName());
  mo.max_ctx = o.max_ctx;
  mo.mtp_draft_k = mtp_k;
  if (dflash) {
    mo.dflash_container = ProductionDrafterPath();
    mo.dflash_draft_k = kDflashK;
  }
  // kAuto: the tower is loaded (rank 0 / the reference) and every Model parses the vision geometry, which PrefillMultimodal's span validation needs
  // (the image-straddle scenario); kOff otherwise, as before.
  mo.vision = vision ? ModelOptions::VisionMode::kAuto : ModelOptions::VisionMode::kOff;
  return mo;
}

class RefRig {
 public:
  // `mtp_k`: the reference's MTP head (kMtpK for the mtp load's comparisons; 0 for the dflash load's warm calls, whose gathered pre-call state has no MTP KV);
  // `vision`: parse the vision geometry (the image scenario).
  explicit RefRig(const Opts& o, int64_t mtp_k = kMtpK, bool vision = false) {
    if (hipSetDevice(o.ref_device) != hipSuccess) throw std::runtime_error("hipSetDevice failed for the reference");
    std::fprintf(stderr, "[hybrid-real] loading the TP=1 reference (64 layers, %s%s) on HIP device %d\n", mtp_k > 0 ? "MTP head" : "no MTP head",
                 vision ? ", vision" : "", o.ref_device);
    m_ = std::make_unique<Model>(Model::Load(BaseOptions(o, mtp_k, false, vision)));
    r4dx::model::ModelConfig gc = m_->GlobalConfig();
    gc.num_hidden_layers = m_->GetContainer().NumLoadedLayers();
    gc.layer_types.resize(static_cast<size_t>(gc.num_hidden_layers));
    g_ = hybrid::StateGeometry::FromRules(gc, m_->PpKvBlockSize());
    for (int k = 0; k < 2; ++k) shape_[k] = {g_.block_tokens, g_.ranks[static_cast<size_t>(k)].kv_heads, g_.KvTokenHeadBytes(), g_.conv_live};
  }

  // A fresh conversation. `full_state_file` (may be empty): the end state as a DebugExportFullState file for G-H2.
  Rec Fresh(const std::string& id, int rows, int salt, const std::string& full_state_file) {
    m_->Reset();
    m_->DebugZeroKvState();
    Rec r = Finish(id, m_->Prefill(Tokens(rows, salt)));
    r.rows = rows;
    r.salt = salt;
    if (!full_state_file.empty()) m_->DebugExportFullState(full_state_file);
    return r;
  }
  // The same prefill from the ranks' gathered pre-call state of a recorded hybrid warm call.
  Rec Warm(const Rec& hyb) {
    const hybrid::LiveState pre = hybrid::ReadLiveState(hyb.pre_file);
    m_->Reset();
    m_->DebugZeroKvState();
    m_->DebugImportLiveState(pre);
    Rec r = Finish(hyb.id, m_->Prefill(Tokens(hyb.rows, hyb.salt)));
    r.rows = hyb.rows;
    r.salt = hyb.salt;
    return r;
  }
  // The DFlash tail of a fresh call as the TP=1 run captures it (Model::StageArmDflashTail, the capture stage Y runs): FNV-1a of the [tail rows][cols]
  // bf16 block. The reference for the hybrid's assembled tail (stage X's carried columns + stage Y's), which the drafters are fed from. `targets`:
  // the drafter's target layers. Run after the cold jobs (it attaches the feature capture to the reference Model).
  uint64_t TailDigest(int rows, int salt, const std::vector<int64_t>& targets) {
    EnsureCapture(targets);
    m_->Reset();
    m_->DebugZeroKvState();
    return CaptureTail(0, rows, [&] { (void)m_->Prefill(Tokens(rows, salt)); }, nullptr);
  }

  // ---- P4: the same for an image call and for a warm call -----------------------------------------------------------------------------------
  bool VisionSplice() const { return m_->VisionSpliceEnabled(); }

  // A fresh image conversation (PrefillMultimodal with the synthetic rows on the device). `full_state_file` (may be empty): the end state for G-H2.
  Rec FreshImage(const std::string& id, const ImageSpec& s, const std::string& full_state_file) {
    const ImageCallData d = ImageData(s);
    m_->Reset();
    m_->DebugZeroKvState();
    Rec r = Finish(id, m_->PrefillMultimodal(d.ids, d.spans));
    r.rows = static_cast<int>(d.ids.size());
    r.img_before = s.before;
    r.img_after = s.after;
    r.img_side = s.side;
    r.img_salt = s.salt;
    if (!full_state_file.empty()) m_->DebugExportFullState(full_state_file);
    return r;
  }
  // The DFlash tail features of that call as the TP=1 run captures them (the reference of the hybrid's assembled tail).
  uint64_t TailDigestImage(const ImageSpec& s, const std::vector<int64_t>& targets) {
    EnsureCapture(targets);
    const ImageCallData d = ImageData(s);
    m_->Reset();
    m_->DebugZeroKvState();
    return CaptureTail(0, static_cast<int64_t>(d.ids.size()), [&] { (void)m_->PrefillMultimodal(d.ids, d.spans); }, nullptr);
  }
  // A recorded WARM dflash-load call from its gathered pre-call state, with the tail capture of that very prefill (digest and row count) and, when
  // `end_state_file` is not empty, the reference's end state (DebugExportFullState) for the decode phase's G-H2.
  Rec WarmTail(const Rec& hyb, const std::vector<int64_t>& targets, uint64_t* tail_digest, int64_t* tail_rows, const std::string& end_state_file) {
    const hybrid::LiveState pre = hybrid::ReadLiveState(hyb.pre_file);
    EnsureCapture(targets);
    m_->Reset();
    m_->DebugZeroKvState();
    m_->DebugImportLiveState(pre);
    std::vector<float> logits;
    *tail_digest = CaptureTail(pre.scalars.pos, pre.scalars.pos + hyb.rows, [&] { logits = m_->Prefill(Tokens(hyb.rows, hyb.salt)); }, tail_rows);
    Rec r = Finish(hyb.id, logits);
    r.rows = hyb.rows;
    r.salt = hyb.salt;
    if (!end_state_file.empty()) m_->DebugExportFullState(end_state_file);
    return r;
  }

 private:
  ImageCallData ImageData(const ImageSpec& s) {
    return BuildImageCall(s, m_->GetContainer().ImageTokenId(), m_->GetContainer().VisionCfg().spatial_merge_size, m_->Config().hidden_size, /*on_host=*/false);
  }
  void EnsureCapture(const std::vector<int64_t>& targets) {
    if (capture_attached_) return;
    m_->AttachDflashFeatureCapture(targets);
    capture_attached_ = true;
  }
  // Arms the tail capture of the call [p0, n_end) (the call's own 64-row grid, hybrid::DflashTailStart), runs `prefill`, and returns the FNV-1a of the
  // captured [tail rows][cols] bf16 block -- what stage Y's StageArmDflashTail captures. The caller has Reset() and loaded the pre-call state.
  uint64_t CaptureTail(int64_t p0, int64_t n_end, const std::function<void()>& prefill, int64_t* rows_out) {
    const int64_t start = hybrid::DflashTailStart(p0, n_end), n_tail = n_end - start, cols = m_->DflashFeatureCols();
    r4dx::core::PinnedBuffer<uint16_t> buf(static_cast<size_t>(hybrid::kTailCapacityRows * cols), hipHostMallocPortable);
    m_->StageArmDflashTail(buf.data(), hybrid::kTailCapacityRows, start, n_end);
    try {
      prefill();
    } catch (...) {
      try {
        (void)m_->StageDisarmDflashTail();
      } catch (...) {
      }
      throw;
    }
    if (m_->StageDisarmDflashTail() != n_tail) throw std::runtime_error("the reference captured a different number of tail rows than the call's tail");
    if (rows_out != nullptr) *rows_out = n_tail;
    return hybrid::FnvUpdate(hybrid::kFnvInit, buf.data(), static_cast<size_t>(n_tail * cols) * sizeof(uint16_t));
  }
  bool capture_attached_ = false;
  Rec Finish(const std::string& id, const std::vector<float>& logits) {
    const hybrid::LiveState st = m_->DebugExportLiveState();
    Rec r;
    r.id = id;
    r.ref_key = id;
    r.pos = st.scalars.pos;
    r.logits = HashLogits(logits);
    r.first = Argmax(logits);
    for (int k = 0; k < 2; ++k) r.dig[k] = hybrid::DigestLiveImage(hybrid::ReshardRef(g_, st.image, k), shape_[k], r.pos);
    return r;
  }
  std::unique_ptr<Model> m_;
  hybrid::StateGeometry g_;
  hybrid::DigestShape shape_[2];
};

struct ColdJob {
  const char* id;
  int rows;
  bool full_state;  // G-H2 needs the reference's whole end state
};
constexpr ColdJob kColdJobs[] = {{"c1024", 1024, false}, {"c2125", 2125, true}, {"c8145", 8145, true}};
constexpr int kColdSalt = 1;

int RunRefPhase(const Opts& o) {
  Group grp("ref/cold");
  fs::create_directories(o.dir);
  std::vector<Rec> recs, tails;
  {
    const bool image = ImageWanted() && FileExists(ProductionDrafterPath());  // the dflash load's image-straddle scenario
    RefRig ref(o, kMtpK, image);
    for (const ColdJob& j : kColdJobs) {
      Rec r = ref.Fresh(j.id, j.rows, kColdSalt, j.full_state ? o.Path(std::string("ref_") + j.id + ".state") : std::string());
      grp.Ok(r.pos == j.rows, std::string(j.id) + ": the reference ended at position " + std::to_string(j.rows));
      std::fprintf(stderr, "[hybrid-real] reference %s: %d rows, logits %s\n", j.id, j.rows, Hex(r.logits).c_str());
      recs.push_back(std::move(r));
    }
    // the DFlash tail features the drafters are fed from (only where the --dflash load will run): the reference for DflashTail's assembled-tail check
    if (FileExists(ProductionDrafterPath())) {
      const std::vector<int64_t> targets = r4dx::model::DflashDraftWeights::Open(ProductionDrafterPath()).Config().target_layers;
      for (const ColdJob& j : kColdJobs) {
        if (j.rows < 2048) continue;  // the dflash load runs the two long cold calls only
        Rec t;
        t.id = std::string("tail_") + j.id;
        t.rows = j.rows;
        t.salt = kColdSalt;
        t.logits = ref.TailDigest(j.rows, kColdSalt, targets);
        std::fprintf(stderr, "[hybrid-real] reference DFlash tail %s: %d rows, features %s\n", j.id, j.rows, Hex(t.logits).c_str());
        tails.push_back(std::move(t));
      }
    }
    // the image-straddle call: the reference's PrefillMultimodal (its logits, digests, end state for G-H2) and its tail features
    if (image) {
      if (!ref.VisionSplice()) {
        std::fprintf(stderr, "[SKIP] ref/%s: the container has no vision config (no image rows can be spliced)\n", kStraddleId);
      } else {
        Rec r = ref.FreshImage(kStraddleId, kStraddle, o.Path(std::string("ref_") + kStraddleId + ".state"));
        grp.Ok(r.pos == kStraddle.Total(), std::string(kStraddleId) + ": the reference ended at position " + std::to_string(kStraddle.Total()));
        std::fprintf(stderr, "[hybrid-real] reference %s: %lld rows (image at %d..%lld), logits %s\n", kStraddleId, static_cast<long long>(kStraddle.Total()),
                     kStraddle.before, static_cast<long long>(kStraddle.before + kStraddle.ImageTokens()), Hex(r.logits).c_str());
        recs.push_back(std::move(r));
        const std::vector<int64_t> targets = r4dx::model::DflashDraftWeights::Open(ProductionDrafterPath()).Config().target_layers;
        Rec t;
        t.id = std::string("tail_") + kStraddleId;
        t.rows = static_cast<int>(kStraddle.Total());
        t.logits = ref.TailDigestImage(kStraddle, targets);
        std::fprintf(stderr, "[hybrid-real] reference DFlash tail %s: features %s\n", kStraddleId, Hex(t.logits).c_str());
        tails.push_back(std::move(t));
      }
    }
  }  // the reference Model is freed here
  SaveRecs(o.Path("refs_cold.txt"), recs);
  SaveRecs(o.Path("refs_tail.txt"), tails);
  grp.Done("reference records and state files in " + o.dir);
  return 0;
}

// ======================================================================================================================================
// The real TpModel with the hybrid
// ======================================================================================================================================
struct HCfg {
  std::string name;  // "mtp" / "dflash"
  bool dflash = false;
  bool hybrid = true;  // false: a plain --tp 2 load of the same container (the decode phase: ranks only, decoding from a reference image)
};

class HybridRig {
 public:
  HCfg cfg;
  Opts o;
  std::unique_ptr<TpModel> tpm;
  std::vector<Rec> recs;
  std::map<std::string, Rec> cold_refs;
  std::map<std::string, Rec> tail_refs;  // the TP=1 run's DFlash tail features (digest in .logits), by cold job id

  HybridRig(HCfg c, Opts opts) : cfg(std::move(c)), o(std::move(opts)) {
    // The dflash load also loads the vision tower (rank 0) when the image-straddle scenario is selected.
    ModelOptions mo = BaseOptions(o, cfg.dflash ? 0 : kMtpK, cfg.dflash, /*vision=*/cfg.dflash && cfg.hybrid && ImageWanted());
    mo.prompt_checkpoint = true;
    mo.pp = 2;  // --pp 2
    TpOptions tp;
    tp.world = 2;
    tp.mode = TpOptions::Mode::kReal;  // --tp 2, two cards, devices auto: rank 0 = the last visible ordinal, rank 1 = ordinal 0
    PpOptions pp;                      // the split: the default (pp::DefaultSplit, 29 / 30 with a drafter)
    pp.min_rows = 1024;                // the production default
    pp.min_rows_given = true;
    pp.hybrid = 1;
    pp.hybrid_ctx = o.hybrid_ctx;
    std::fprintf(stderr, "[hybrid-real] %s: loading the two-GPU TpModel %s (64 layers)\n", cfg.name.c_str(), cfg.hybrid ? "with the hybrid" : "(plain --tp 2, no hybrid)");
    if (cfg.hybrid) {
      tpm = TpModel::Load(mo, tp, pp);
      if (!tpm->HybridEngaged()) throw std::runtime_error("the hybrid did not engage: " + tpm->HybridRefusal() + " (lower --max-ctx / --hybrid-ctx or free VRAM)");
    } else {
      mo.pp = 0;
      tpm = TpModel::Load(mo, tp);
    }
    if (cfg.dflash && !tpm->DflashEnabled()) throw std::runtime_error("the DFlash drafter did not load");
    r4dx::model::ModelConfig gc = tpm->Config();
    gc.num_hidden_layers = tpm->NumLoadedLayers();
    gc.layer_types.resize(static_cast<size_t>(gc.num_hidden_layers));
    int64_t block = 0;
    OnRanks([&](Model& m, int rank) {
      if (rank == 0) block = m.PpKvBlockSize();
    }, /*read_only=*/true);
    g_ = hybrid::StateGeometry::FromRules(gc, block);
    std::fprintf(stderr, "[hybrid-real] %s: ready (split %lld, S %lld, min rows %lld)\n", cfg.name.c_str(), static_cast<long long>(tpm->HybridSplit()),
                 static_cast<long long>(tpm->HybridStageCtx()), static_cast<long long>(tpm->HybridMinRows()));
  }

  bool HasMtp() const { return !cfg.dflash; }
  void OnRanks(const std::function<void(Model&, int)>& fn, bool read_only = false) { tpm->RunCollectiveForTest(fn, read_only); }
  std::vector<hybrid::DigestList> RankDigests() {
    std::vector<hybrid::DigestList> out(2);
    OnRanks([&](Model& m, int rank) { out[static_cast<size_t>(rank)] = m.DebugLiveStateDigest(); }, /*read_only=*/true);
    return out;
  }
  std::vector<hybrid::LiveState> SnapshotRanks() {
    std::vector<hybrid::LiveState> rs(2);
    OnRanks(
        [&](Model& m, int rank) {
          m.ReshardCollapse();
          m.ReshardSync();
          rs[static_cast<size_t>(rank)] = m.DebugExportLiveState();
        },
        /*read_only=*/true);
    return rs;
  }
  using Stats = TpModel::HybridStats;

  // ---- one TpModel::Prefill, recorded ---------------------------------------------------------------------------------------------------
  // `warm`: the ranks did TP-only work since the last pipelined call, so their gathered state is written to a file for the verify phase. `ref_key`: the
  // reference to compare with ("" = none). `expect`: -1 any route, 1 must be pipelined, 0 must be the TP prefill.
  // `img` (P4): a cold PrefillMultimodal call with a synthetic image (`rows` and `salt` are then taken from it). On a dflash load the hybrid's assembled
  // DFlash tail (rows and FNV-1a) of every pipelined call is recorded, for the verify phase's TP=1 capture of the same warm call.
  Rec Call(Group& grp, const std::string& id, int rows, int salt, const std::string& ref_key, bool warm, int expect, bool expect_mismatch = false,
           const ImageSpec* img = nullptr) {
    ImageCallData img_data;
    if (img != nullptr) {
      img_data = BuildImageCall(*img, tpm->ImageTokenId(), tpm->VisionMergeSize(), tpm->Config().hidden_size, /*on_host=*/true);
      rows = static_cast<int>(img_data.ids.size());
      salt = img->salt;
    }
    const std::vector<int32_t> ids = img != nullptr ? img_data.ids : Tokens(rows, salt);
    Rec r;
    r.id = id;
    r.ref_key = ref_key;
    r.rows = rows;
    r.salt = salt;
    if (img != nullptr) {
      r.img_before = img->before;
      r.img_after = img->after;
      r.img_side = img->side;
      r.img_salt = img->salt;
    }
    r.expect_mismatch = expect_mismatch;
    const int64_t p0 = tpm->PositionCount();
    if (warm) {
      const std::vector<hybrid::LiveState> rs = SnapshotRanks();
      hybrid::LiveState full;
      full.image = hybrid::GatherRef(g_, rs[0].image, rs[1].image);
      full.scalars = rs[0].scalars;
      r.pre_file = o.Path("pre_" + SafeName(cfg.name + "_" + id) + ".state");
      hybrid::WriteLiveState(r.pre_file, full);
    }
    const Stats before = tpm->GetHybridStats();
    const std::vector<float> logits = img != nullptr ? tpm->PrefillMultimodal(ids, img_data.spans) : tpm->Prefill(ids);
    const Stats after = tpm->GetHybridStats();
    r.pipelined = after.pipelined_calls == before.pipelined_calls + 1;
    grp.Ok(r.pipelined || after.tp_prefill_calls == before.tp_prefill_calls + 1, id + ": the call is counted as pipelined or as a TP prefill");
    if (expect >= 0) grp.Ok(r.pipelined == (expect == 1), id + ": the dispatch rule chose the " + (expect == 1 ? "pipeline" : "TP prefill"));
    r.pos = tpm->PositionCount();
    grp.Ok(r.pos == p0 + rows, id + ": the position is " + std::to_string(p0 + rows));
    r.logits = HashLogits(logits);
    r.first = Argmax(logits);
    if (r.pipelined) {
      const std::vector<hybrid::DigestList> d = RankDigests();
      for (int k = 0; k < 2; ++k) r.dig[k] = DropPrefix(d[static_cast<size_t>(k)], "dflash.");
      if (cfg.dflash) {
        r.tail_rows = (p0 + rows) - hybrid::DflashTailStart(p0, p0 + rows);
        r.tail_digest = tpm->HybridTailDigest(r.tail_rows);
      }
    }
    if (r.pipelined && !ref_key.empty()) {
      const auto it = cold_refs.find(ref_key);
      if (it != cold_refs.end()) CheckAgainst(grp, r, it->second, HasMtp());  // a cold call: the reference is known; a warm one is checked by the verify phase
    }
    recs.push_back(r);
    return r;
  }

  // ---- the DFlash drafter: its frontier and its window --------------------------------------------------------------------------------------
  // "" when both drafters sit at the call's end and hold the same window; else why not.
  std::string DflashProblem() {
    int64_t injected[2] = {-1, -1}, pos[2] = {-1, -1};
    hybrid::DigestList dfl[2];
    OnRanks(
        [&](Model& m, int rank) {
          injected[rank] = m.DflashInjectedCount();
          pos[rank] = m.PositionCount();
          dfl[rank] = m.DebugLiveStateDigest();
        },
        /*read_only=*/true);
    for (int k = 0; k < 2; ++k) {
      if (injected[k] != pos[k]) return "rank " + std::to_string(k) + "'s drafter is at " + std::to_string(injected[k]) + ", the position is " + std::to_string(pos[k]);
    }
    hybrid::DigestList a, b;
    for (const auto& kv : dfl[0]) {
      if (kv.first.rfind("dflash.", 0) == 0) a.push_back(kv);
    }
    for (const auto& kv : dfl[1]) {
      if (kv.first.rfind("dflash.", 0) == 0) b.push_back(kv);
    }
    if (a.empty()) return "the drafter reports no window";
    const std::string why = hybrid::DiffDigests(a, b);
    if (!why.empty()) return "the two drafters' windows differ: " + why;
    return "";
  }

  // ---- G-H2 -------------------------------------------------------------------------------------------------------------------------------------
  struct DecodeOutcome {
    std::vector<int32_t> tok[2];
    hybrid::DigestList dig[2][2];  // [mode][rank]
  };
  // mode 0: plain greedy; mode 1: --mtp 3 (spec 1) or --dflash 7 (spec 2) rounds, rewound by the checkpoint. `digests`: keep the ranks' state after each.
  DecodeOutcome DecodeModes(int32_t first, bool digests) {
    tpm->SaveCheckpoint();
    DecodeOutcome out;
    for (int mode = 0; mode < 2; ++mode) {
      if (mode > 0) tpm->RestoreCheckpoint();
      int32_t tok = first;
      while (static_cast<int>(out.tok[mode].size()) < kDecodeTokens) {
        if (mode == 0) {
          tok = tpm->DecodeStepGreedy(tok);
          out.tok[0].push_back(tok);
          continue;
        }
        int64_t walk = 0;
        const std::vector<int32_t> round = cfg.dflash ? tpm->DecodeStepDflashGreedy(tok, kDflashK, 0.0f, 0, &walk) : tpm->DecodeStepMtpGreedy(tok, kMtpK);
        out.tok[1].insert(out.tok[1].end(), round.begin(), round.end());
        tok = round.back();
      }
      if (digests) {
        const std::vector<hybrid::DigestList> d = RankDigests();
        out.dig[mode][0] = d[0];
        out.dig[mode][1] = d[1];
      }
    }
    return out;
  }
  // The reference's end state as the ranks' state: DebugImportFullState on both ranks (the --dflash load has no MTP head, so the MTP part is stripped).
  void ImportReference(const std::string& file) {
    std::string use = file;
    if (!HasMtp()) {
      hybrid::LiveState st = hybrid::ReadLiveState(file);
      st.image.mtp_kv.clear();
      st.scalars.mtp_seed_valid = false;
      st.scalars.mtp_seed.clear();
      use = o.Path("ref_nomtp.state");
      hybrid::WriteLiveState(use, st);
    }
    tpm->Reset();
    OnRanks([&](Model& m, int) { m.DebugImportFullState(use); });
  }
  void CheckDecode(Group& grp, const Rec& ref, const std::string& ref_state_file) {
    const std::string spec = cfg.dflash ? "--dflash 7" : "--mtp 3";
    const DecodeOutcome h = DecodeModes(ref.first, /*digests=*/HasMtp());
    ImportReference(ref_state_file);
    const DecodeOutcome w = DecodeModes(ref.first, /*digests=*/HasMtp());
    for (int mode = 0; mode < 2; ++mode) {
      const std::string name = mode == 0 ? "plain" : spec;
      grp.Ok(h.tok[mode].size() >= static_cast<size_t>(kDecodeTokens), "G-H2 " + name + ": decoded " + std::to_string(h.tok[mode].size()) + " tokens");
      const size_t n = std::min<size_t>(static_cast<size_t>(kDecodeTokens), std::min(h.tok[mode].size(), w.tok[mode].size()));
      const bool same = std::equal(h.tok[mode].begin(), h.tok[mode].begin() + static_cast<std::ptrdiff_t>(n), w.tok[mode].begin());
      grp.Ok(same && n == static_cast<size_t>(kDecodeTokens),
             "G-H2 " + name + ": the first " + std::to_string(kDecodeTokens) + " tokens decoded at TP=2 after the hybrid prefill equal those decoded from the reference image");
      if (HasMtp()) {
        grp.Ok(hybrid::DiffDigests(h.dig[mode][0], w.dig[mode][0]).empty() && hybrid::DiffDigests(h.dig[mode][1], w.dig[mode][1]).empty(),
               "G-H2 " + name + ": both ranks' live state after the decode equals the reference-image run's");
      }
    }
  }

  // ---- TP=2 work between two hybrid calls -------------------------------------------------------------------------------------------------------
  int32_t DecodePlain(int32_t tok, int n) {
    for (int i = 0; i < n; ++i) tok = tpm->DecodeStepGreedy(tok);
    return tok;
  }
  int32_t DecodeMtp(int32_t tok, int n) {
    int got = 0;
    while (got < n) {
      const std::vector<int32_t> round = tpm->DecodeStepMtpGreedy(tok, kMtpK);
      got += static_cast<int>(round.size());
      tok = round.back();
    }
    return tok;
  }
  // --dflash 7 rounds at TP=2 (the drafters inject what they accept: the decode-time state a following warm call must continue from)
  int32_t DecodeDflash(int32_t tok, int n) {
    int got = 0;
    while (got < n) {
      int64_t walk = 0;
      const std::vector<int32_t> round = tpm->DecodeStepDflashGreedy(tok, kDflashK, 0.0f, 0, &walk);
      got += static_cast<int>(round.size());
      tok = round.back();
    }
    return tok;
  }

  // ---- scenarios ----------------------------------------------------------------------------------------------------------------------------------
  void Guard(const std::string& name, const std::function<void(Group&)>& body) {
    const std::string full = cfg.name + "/" + name;
    if (!Only(full)) return;
    Group grp(full);
    try {
      body(grp);
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
      try {
        tpm->Reset();  // heals a kNeedsRecovery group for the next scenario
        tpm->SetHybridNegControl(TpModel::HybridNegControl::kNone);  // (it needs a kReady group)
      } catch (const std::exception& e2) {
        std::fprintf(stderr, "FAIL %s: the group did not recover: %s\n", full.c_str(), e2.what());
        ++g_fails;
      }
    }
    tpm->Reset();
  }

  void Cold(const char* id, int rows, bool gh2) {
    Guard(id, [&](Group& grp) {
      tpm->Reset();
      const Rec r = Call(grp, std::string(cfg.name) + "." + id, rows, kColdSalt, id, false, 1);
      const auto it = cold_refs.find(id);
      if (cfg.dflash) {
        const std::string why = DflashProblem();
        grp.Ok(why.empty(), std::string(id) + ": the DFlash tail left both drafters at the call's end with the same window: " + why);
        grp.Ok(tpm->GetHybridStats().tail_rows > 0, std::string(id) + ": the tail rows were counted");
        // The drafters are both fed from the one assembled tail buffer, so equal windows cannot tell a wrong tail from a right one: the buffer itself
        // (stage X's carried columns + stage Y's, in the drafter's column order) must equal what the TP=1 run captures for the same call.
        const auto tr = tail_refs.find(id);
        grp.Ok(tr != tail_refs.end(), std::string(id) + ": a TP=1 DFlash tail reference exists (run the ref phase with the drafter present)");
        if (tr != tail_refs.end()) {
          const int64_t n_tail = rows - hybrid::DflashTailStart(0, rows);
          const uint64_t got = tpm->HybridTailDigest(n_tail);
          grp.Ok(got == tr->second.logits, std::string(id) + ": the hybrid's assembled DFlash tail (" + std::to_string(n_tail) + " rows) equals the TP=1 run's capture, got " +
                                               Hex(got) + " want " + Hex(tr->second.logits));
          grp.Ok(tpm->HybridTailDigest(n_tail - 1) != tr->second.logits, std::string(id) + ": control -- the digest of a tail one row short does not match the reference");
        }
      }
      if (gh2 && it != cold_refs.end()) CheckDecode(grp, it->second, o.Path(std::string("ref_") + id + ".state"));
      (void)r;
      grp.Done(std::to_string(rows) + " rows cold" + (gh2 ? ", G-H2" : ""));
    });
  }

  // a call below --pp-min-rows takes the TP prefill (counted), decode works, and a long call afterwards pipelines from that state
  void BelowMin() {
    Guard("below-min", [&](Group& grp) {
      tpm->Reset();
      const Stats s0 = tpm->GetHybridStats();
      const Rec a = Call(grp, "below-min.a", 300, 3, "", false, 0);
      const Stats s1 = tpm->GetHybridStats();
      const int why = static_cast<int>(hybrid::Why::kBelowMinRows);
      grp.Ok(s1.declined[why] == s0.declined[why] + 1, "the decline is counted as 'below min rows'");
      const int32_t tok = DecodePlain(a.first, 8);
      grp.Ok(tok >= 0, "decode continues after the TP prefill");
      (void)Call(grp, "below-min.b", 1500, 4, "below-min.b", true, 1);
      grp.Done("the TP path taken and said so; the next long call pipelines from the TP-prefilled state (verify phase)");
    });
  }

  // call, TP=2 decode, then a 1500-row hybrid call (the warm gather, the MTP seed sync, the primed-block rule)
  void Warm(const char* name, bool mtp) {
    Guard(name, [&](Group& grp) {
      tpm->Reset();
      const Rec a = Call(grp, std::string(name) + ".a", 1024, kColdSalt, "c1024", false, 1);
      (void)(mtp ? DecodeMtp(a.first, kWarmDecode) : DecodePlain(a.first, kWarmDecode));
      (void)Call(grp, std::string(name) + ".b", 1500, 2, std::string(name) + ".b", true, 1);
      grp.Done(std::string("1024 rows, ") + std::to_string(kWarmDecode) + " tokens at TP=2 (" + (mtp ? "--mtp 3 rounds" : "plain") + "), 1500 rows (verify phase)");
    });
  }

  // SaveCheckpoint, a decode, RestoreCheckpoint, then a warm hybrid call from the restored state
  void Checkpoint() {
    Guard("checkpoint", [&](Group& grp) {
      tpm->Reset();
      const Rec a = Call(grp, "checkpoint.a", 1024, kColdSalt, "c1024", false, 1);
      const std::vector<hybrid::DigestList> saved = RankDigests();
      tpm->SaveCheckpoint();
      (void)DecodePlain(a.first, 20);
      tpm->RestoreCheckpoint();
      const std::vector<hybrid::DigestList> d = RankDigests();
      grp.Ok(tpm->PositionCount() == a.pos, "RestoreCheckpoint returned the ranks to the saved position");
      for (int k = 0; k < 2; ++k) {
        grp.Ok(hybrid::DiffDigests(saved[static_cast<size_t>(k)], d[static_cast<size_t>(k)]).empty(),
               "rank " + std::to_string(k) + "'s digest after RestoreCheckpoint equals the one at SaveCheckpoint");
      }
      (void)Call(grp, "checkpoint.b", 1500, 2, "checkpoint.b", true, 1);
      grp.Done("restore, then a warm hybrid call (verify phase)");
    });
  }

  // P4, dflash load: a call, 32 tokens decoded at TP=2 in --dflash 7 rounds, then a 1500-row hybrid call -- the warm gather with the drafters holding
  // decode-time state, and the DFlash tail on top of it. Here: the call is pipelined, both drafters sit at its end with equal windows, the tail rows were
  // counted, and the 16 plain + 16 --dflash 7 tokens decoded after it are recorded for the decode phase (G-H2). The verify phase compares the call with a
  // TP=1 reference loaded with the recorded pre-call state (G-H1, and the assembled tail's digest against the reference's capture of the same warm call).
  void WarmDflash() {
    Guard("warm-dflash", [&](Group& grp) {
      tpm->Reset();
      const Rec a = Call(grp, "warm-dflash.a", 1024, kColdSalt, "c1024", false, 1);
      (void)DecodeDflash(a.first, kWarmDecode);
      const Rec b = Call(grp, "warm-dflash.b", 1500, 2, "warm-dflash.b", true, 1);
      const std::string why = DflashProblem();
      grp.Ok(why.empty(), "warm-dflash.b: the DFlash tail left both drafters at the call's end with the same window: " + why);
      grp.Ok(b.pipelined && b.tail_rows > 0 && b.tail_rows <= b.rows, "warm-dflash.b: the warm call's DFlash tail is " + std::to_string(b.tail_rows) + " rows of its " + std::to_string(b.rows));
      grp.Ok(tpm->GetHybridStats().tail_rows > 0, "warm-dflash.b: the tail rows were counted");
      const DecodeOutcome dec = DecodeModes(b.first, /*digests=*/false);  // G-H2's hybrid side; the reference side is the decode phase
      recs.back().tok[0] = dec.tok[0];
      recs.back().tok[1] = dec.tok[1];
      grp.Done("1024 rows, " + std::to_string(kWarmDecode) + " tokens at TP=2 in --dflash 7 rounds, 1500 rows; G-H1, the tail digest (verify phase) and G-H2 (decode phase) follow");
    });
  }

  // P4, dflash load: a cold PrefillMultimodal call whose image straddles the start of the 2048-row DFlash tail window (kStraddle: the image is rows
  // 1000..1255 of 3156, the window starts at 1088): the tail's temporal rope rows must carry the image's mrope positions.
  void ImageStraddle() {
    Guard("image-straddle", [&](Group& grp) {
      if (!ImageWanted()) {
        std::fprintf(stderr, "[SKIP] %s/image-straddle: not selected (run it with R4DX_TEST_ONLY=dflash/image-straddle, in the ref and the hybrid phase)\n", cfg.name.c_str());
        return;
      }
      bool splice = false;
      OnRanks([&](Model& m, int rank) {
        if (rank == 0) splice = m.VisionSpliceEnabled();
      }, /*read_only=*/true);
      if (!splice) {
        std::fprintf(stderr, "[SKIP] %s/image-straddle: the container has no vision config (no image rows can be spliced)\n", cfg.name.c_str());
        return;
      }
      const auto ref = cold_refs.find(kStraddleId);
      if (ref == cold_refs.end()) {
        grp.Ok(false, "no TP=1 reference for the image call: run the ref phase with this scenario selected (R4DX_TEST_ONLY=dflash/image-straddle)");
        return;
      }
      tpm->Reset();
      const Rec r = Call(grp, std::string(cfg.name) + "." + kStraddleId, 0, 0, kStraddleId, false, 1, false, &kStraddle);
      grp.Ok(r.rows == kStraddle.Total(), std::string(kStraddleId) + ": the call had " + std::to_string(kStraddle.Total()) + " rows");
      const std::string why = DflashProblem();
      grp.Ok(why.empty(), std::string(kStraddleId) + ": the DFlash tail left both drafters at the call's end with the same window: " + why);
      const int64_t start = hybrid::DflashTailStart(0, r.rows);
      grp.Ok(start > kStraddle.before && start < kStraddle.before + kStraddle.ImageTokens(),
             std::string(kStraddleId) + ": the tail window starts at row " + std::to_string(start) + ", inside the image (rows " + std::to_string(kStraddle.before) + ".." +
                 std::to_string(kStraddle.before + kStraddle.ImageTokens()) + ")");
      const auto tr = tail_refs.find(kStraddleId);
      grp.Ok(tr != tail_refs.end(), std::string(kStraddleId) + ": a TP=1 DFlash tail reference exists");
      if (tr != tail_refs.end()) {
        grp.Ok(r.tail_digest == tr->second.logits, std::string(kStraddleId) + ": the hybrid's assembled DFlash tail (" + std::to_string(r.tail_rows) + " rows) equals the TP=1 run's capture, got " +
                                                       Hex(r.tail_digest) + " want " + Hex(tr->second.logits));
      }
      CheckDecode(grp, ref->second, o.Path(std::string("ref_") + kStraddleId + ".state"));
      grp.Done("image rows 1000..1255, tail window from " + std::to_string(start) + "; G-H1, tail features, G-H2 (the mrope delta rides the adoption)");
    });
  }

  // n_total > S: the TP prefill (counted), decode works
  void OverS() {
    Guard("over-s", [&](Group& grp) {
      const int64_t s = tpm->HybridStageCtx();
      const int64_t n = s + 600;
      if (n + 64 > o.max_ctx) {
        std::fprintf(stderr, "[SKIP] %s/over-s: S = %lld plus 600 rows does not fit --max-ctx %lld\n", cfg.name.c_str(), static_cast<long long>(s),
                     static_cast<long long>(o.max_ctx));
        return;
      }
      tpm->Reset();
      const Stats s0 = tpm->GetHybridStats();
      const Rec a = Call(grp, "over-s", static_cast<int>(n), 5, "", false, 0);
      const Stats s1 = tpm->GetHybridStats();
      const int why = static_cast<int>(hybrid::Why::kOverStageCtx);
      grp.Ok(s1.declined[why] == s0.declined[why] + 1, "the decline is counted as 'over the stage-KV capacity' (S = " + std::to_string(s) + ")");
      grp.Ok(DecodePlain(a.first, 4) >= 0, "decode continues after the TP prefill");
      const std::string line = tpm->HybridStatsLine();
      std::fprintf(stderr, "[stats] %s\n", line.c_str());
      grp.Ok(line.rfind("hybrid:", 0) == 0, "the hybrid stats line is there");
      grp.Done("n_total > S falls back to the TP prefill and says so");
    });
  }

  // ---- negative controls -----------------------------------------------------------------------------------------------------------------------------
  void Controls() {
    using NC = TpModel::HybridNegControl;
    if (HasMtp()) {  // the --mtp load: its warm calls are verified by the verify phase (a TP=1 reference with an MTP head)
      Guard("nc-wrong-rank", [&](Group& grp) {
        tpm->Reset();
        tpm->SetHybridNegControl(NC::kWrongRankGdn);
        (void)Call(grp, "nc-wrong-rank", 2125, kColdSalt, "c2125", false, 1, /*expect_mismatch=*/true);
        tpm->SetHybridNegControl(NC::kNone);
        grp.Done("rank 0's recurrent state read from rank 1's heads is detected by the rank digest");
      });
      Guard("nc-stale-mirror", [&](Group& grp) {
        tpm->Reset();
        const Rec a = Call(grp, "nc-stale-mirror.a", 1024, kColdSalt, "c1024", false, 1);
        (void)DecodePlain(a.first, kWarmDecode);
        tpm->SetHybridNegControl(NC::kSkipGather);
        (void)Call(grp, "nc-stale-mirror.b", 1500, 2, "nc-stale-mirror.b", true, 1, /*expect_mismatch=*/true);
        tpm->SetHybridNegControl(NC::kNone);
        grp.Done("a skipped warm gather (stale mirror) is detected (verify phase)");
      });
      Guard("nc-stale-seed", [&](Group& grp) {
        tpm->Reset();
        const Rec a = Call(grp, "nc-stale-seed.a", 1024, kColdSalt, "c1024", false, 1);
        (void)DecodeMtp(a.first, kWarmDecode);
        tpm->SetHybridNegControl(NC::kStaleSeed);
        (void)Call(grp, "nc-stale-seed.b", 1500, 2, "nc-stale-seed.b", true, 1, /*expect_mismatch=*/true);
        tpm->SetHybridNegControl(NC::kNone);
        grp.Done("a skipped MTP seed sync is detected (verify phase)");
      });
    } else {
      Guard("nc-skip-tail", [&](Group& grp) {
        tpm->Reset();
        tpm->SetHybridNegControl(NC::kSkipDflashTail);
        const Rec a = Call(grp, "nc-skip-tail", 2125, kColdSalt, "c2125", false, 1);
        tpm->SetHybridNegControl(NC::kNone);
        (void)a;
        const std::string why = DflashProblem();
        grp.Ok(!why.empty(), "negative control NOT detected -- with the DFlash tail skipped both drafters are still at the call's end with equal windows");
        if (!why.empty()) std::fprintf(stderr, "[control] nc-skip-tail: detected (%s)\n", why.c_str());
        grp.Done("a skipped DFlash tail is detected by the drafter frontier");
      });
    }
  }

 private:
  hybrid::StateGeometry g_;
};

// R4DX_TEST_ONLY is a substring of "<cfg>/<scenario>"; the configuration is loaded when some scenario of it can match.
bool ConfigWanted(const std::string& cfg) {
  const char* only = std::getenv("R4DX_TEST_ONLY");
  if (only == nullptr || *only == '\0') return true;
  const std::string f = only, prefix = cfg + "/";
  return prefix.find(f) != std::string::npos || f.rfind(prefix, 0) == 0 || f.find('/') == std::string::npos;
}

int RunHybridConfig(const Opts& o, const HCfg& cfg) {
  if (!ConfigWanted(cfg.name)) return 0;
  const std::vector<Rec> refs = LoadRecs(o.Path("refs_cold.txt"));
  std::unique_ptr<HybridRig> rig;
  try {
    rig = std::make_unique<HybridRig>(cfg, o);
  } catch (const std::exception& e) {
    ++g_fails;
    std::fprintf(stderr, "FAIL %s: the rig did not load: %s\n", cfg.name.c_str(), e.what());
    return 1;
  }
  ++g_configs;
  for (const Rec& r : refs) rig->cold_refs[r.id] = r;
  if (cfg.dflash && FileExists(o.Path("refs_tail.txt"))) {
    for (const Rec& r : LoadRecs(o.Path("refs_tail.txt"))) rig->tail_refs[r.id.rfind("tail_", 0) == 0 ? r.id.substr(5) : r.id] = r;
  }
  if (!cfg.dflash) {
    rig->Cold("c1024", 1024, false);
    rig->Cold("c2125", 2125, true);
    rig->Cold("c8145", 8145, true);
    rig->BelowMin();
    rig->Warm("warm-plain", false);
    rig->Warm("warm-mtp", true);
    rig->Checkpoint();
    rig->OverS();
  } else {
    rig->Cold("c2125", 2125, true);
    rig->Cold("c8145", 8145, false);
    rig->WarmDflash();
    rig->ImageStraddle();
  }
  rig->Controls();
  SaveRecs(o.Path("hyb_" + cfg.name + ".txt"), rig->recs);
  std::fprintf(stderr, "[stats] %s\n", rig->tpm->HybridStatsLine().c_str());
  return 0;
}

// ======================================================================================================================================
// The verify phase: the TP=1 reference from each recorded warm call's pre-call state
// ======================================================================================================================================
// The dflash load's WARM calls (P4): a TP=1 reference WITHOUT an MTP head (the dflash ranks have none, so their gathered pre-call state carries no MTP KV)
// loaded with the recorded pre-call state prefills the same tokens with the tail capture armed: G-H1 (logits, digests), the assembled DFlash tail's
// digest against the hybrid's, and the reference's end state is written for the decode phase's G-H2.
int RunVerifyDflash(const Opts& o) {
  const std::string path = o.Path("hyb_dflash.txt");
  if (!FileExists(path) || !FileExists(ProductionDrafterPath())) return 0;
  std::vector<Rec> warm;
  for (Rec& r : LoadRecs(path)) {
    if (!r.pre_file.empty() && r.pipelined && Only("dflash/" + r.id)) warm.push_back(std::move(r));
  }
  if (warm.empty()) return 0;
  const std::vector<int64_t> targets = r4dx::model::DflashDraftWeights::Open(ProductionDrafterPath()).Config().target_layers;
  RefRig ref(o, /*mtp_k=*/0, /*vision=*/false);
  for (const Rec& hyb : warm) {
    Group grp("verify/dflash/" + hyb.id);
    try {
      uint64_t tail = 0;
      int64_t tail_rows = 0;
      const Rec want = ref.WarmTail(hyb, targets, &tail, &tail_rows, o.Path("ref_warm_" + SafeName(hyb.id) + ".state"));
      CheckAgainst(grp, hyb, want, /*has_mtp=*/false);
      grp.Ok(tail_rows == hyb.tail_rows, "the reference's tail of the warm call has " + std::to_string(tail_rows) + " rows, the hybrid's " + std::to_string(hyb.tail_rows));
      grp.Ok(tail == hyb.tail_digest, "the hybrid's assembled DFlash tail equals the TP=1 capture of the same warm call, got " + Hex(hyb.tail_digest) + " want " + Hex(tail));
      grp.Done("G-H1 and the DFlash tail digest from the recorded warm pre-call state; the reference's end state is in " + o.dir + " for the decode phase");
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
    }
  }
  return 0;
}

int RunVerifyPhase(const Opts& o) {
  RunVerifyDflash(o);
  const std::string path = o.Path("hyb_mtp.txt");
  if (!FileExists(path)) {
    std::fprintf(stderr, "[hybrid-real] verify: %s not found (the hybrid phase did not run for the --mtp load): nothing to verify\n", path.c_str());
    return 0;
  }
  std::vector<Rec> warm;
  for (Rec& r : LoadRecs(path)) {
    if (!r.pre_file.empty() && r.pipelined && Only("mtp/" + r.id)) warm.push_back(std::move(r));
  }
  if (warm.empty()) return 0;
  RefRig ref(o);
  for (const Rec& hyb : warm) {
    Group grp("verify/" + hyb.id);
    try {
      const Rec want = ref.Warm(hyb);
      CheckAgainst(grp, hyb, want, /*has_mtp=*/true);
      grp.Done(hyb.expect_mismatch ? "negative control detected" : "G-H1 from the recorded warm pre-call state");
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
    }
  }
  return 0;
}

// ======================================================================================================================================
// The decode phase (P4): G-H2 of the dflash warm records. A plain --tp 2 load (ranks only) is loaded with the reference's END state of the warm call
// (the file the verify phase wrote) and decodes; its tokens must equal those the hybrid's ranks decoded after the call itself (recorded in the hybrid phase).
// ======================================================================================================================================
int RunDecodePhase(const Opts& o) {
  const std::string path = o.Path("hyb_dflash.txt");
  if (!FileExists(path) || !FileExists(ProductionDrafterPath())) return 0;
  std::vector<Rec> warm;
  for (Rec& r : LoadRecs(path)) {
    if (!r.pre_file.empty() && r.pipelined && !r.tok[0].empty() && Only("dflash/" + r.id)) warm.push_back(std::move(r));
  }
  if (warm.empty()) return 0;
  std::unique_ptr<HybridRig> rig;
  try {
    HCfg cfg;
    cfg.name = "dflash";
    cfg.dflash = true;
    cfg.hybrid = false;
    rig = std::make_unique<HybridRig>(cfg, o);
  } catch (const std::exception& e) {
    ++g_fails;
    std::fprintf(stderr, "FAIL decode/dflash: the plain --tp 2 rig did not load: %s\n", e.what());
    return 1;
  }
  for (const Rec& hyb : warm) {
    Group grp("decode/dflash/" + hyb.id);
    try {
      const std::string ref_state = o.Path("ref_warm_" + SafeName(hyb.id) + ".state");
      if (!FileExists(ref_state)) {
        grp.Ok(false, ref_state + " not found (run the verify phase first)");
        continue;
      }
      rig->ImportReference(ref_state);
      const HybridRig::DecodeOutcome w = rig->DecodeModes(hyb.first, /*digests=*/false);
      for (int mode = 0; mode < 2; ++mode) {
        const std::string name = mode == 0 ? "plain" : "--dflash 7";
        const size_t n = static_cast<size_t>(kDecodeTokens);
        const bool enough = hyb.tok[mode].size() >= n && w.tok[mode].size() >= n;
        grp.Ok(enough, "G-H2 " + name + ": both sides decoded " + std::to_string(n) + " tokens (hybrid " + std::to_string(hyb.tok[mode].size()) + ", reference image " +
                           std::to_string(w.tok[mode].size()) + ")");
        grp.Ok(enough && std::equal(hyb.tok[mode].begin(), hyb.tok[mode].begin() + static_cast<std::ptrdiff_t>(n), w.tok[mode].begin()),
               "G-H2 " + name + ": the first " + std::to_string(n) + " tokens decoded at TP=2 after the WARM hybrid call equal those decoded from the reference's end state");
      }
      grp.Done("G-H2 plain + --dflash 7 after a warm hybrid call that followed a TP=2 DFlash decode");
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
    }
  }
  return 0;
}

void RemoveStateFiles(const Opts& o) {
  std::error_code ec;
  for (const fs::directory_entry& e : fs::directory_iterator(o.dir, ec)) {
    if (e.path().extension() == ".state") fs::remove(e.path(), ec);
  }
}

}  // namespace

static int RunTest(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
      return argv[++i];
    };
    if (a == "--phase") {
      o.phase = next();
    } else if (a == "--dir") {
      o.dir = next();
    } else if (a == "--max-ctx") {
      o.max_ctx = std::stoll(next());
    } else if (a == "--hybrid-ctx") {
      o.hybrid_ctx = std::stoll(next());
    } else if (a == "--keep") {
      o.keep = true;
    } else {
      throw std::invalid_argument("unknown argument " + a);
    }
  }
  if (o.phase != "all" && o.phase != "ref" && o.phase != "hybrid" && o.phase != "verify" && o.phase != "decode") {
    throw std::invalid_argument("--phase must be all, ref, hybrid, verify or decode");
  }
  if (o.dir.empty()) o.dir = (fs::temp_directory_path() / "r4dx_hybrid_real_identity").string();
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) {
    std::fprintf(stderr, "[SKIP] the two-GPU hybrid needs two visible HIP devices (%d visible)\n", devices);
    return kSkipReturnCode;
  }
  o.ref_device = devices - 1;
  if (!FileExists(ProductionTargetPath())) return SkipMissing(ProductionTargetPath());
  fs::create_directories(o.dir);
  std::fprintf(stderr, "[hybrid-real] phase %s, state in %s, --max-ctx %lld, --hybrid-ctx %lld\n", o.phase.c_str(), o.dir.c_str(), static_cast<long long>(o.max_ctx),
               static_cast<long long>(o.hybrid_ctx));
  const bool all = o.phase == "all";
  if (all || o.phase == "ref") RunRefPhase(o);
  if (all || o.phase == "hybrid") {
    RunHybridConfig(o, {"mtp", false});
    if (FileExists(ProductionDrafterPath())) {
      RunHybridConfig(o, {"dflash", true});
    } else {
      std::fprintf(stderr, "[SKIP] dflash/: %s not found (the DFlash tail, G-H2 --dflash 7 and the skip-tail control are not exercised)\n", ProductionDrafterPath());
    }
  }
  if (all || o.phase == "verify") RunVerifyPhase(o);
  if (all || o.phase == "decode") RunDecodePhase(o);
  if (all && !o.keep && g_fails == 0) RemoveStateFiles(o);
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_real_identity: %d FAILED\n", g_fails);
    return 1;
  }
  if (g_configs == 0 && o.phase != "ref" && o.phase != "verify" && o.phase != "decode") {
    std::fprintf(stderr, "[SKIP] no hybrid configuration ran (R4DX_TEST_ONLY filtered everything out)\n");
    return kSkipReturnCode;
  }
  std::fprintf(stderr, "test_hybrid_real_identity: PASS (%d configurations)\n", g_configs);
  return 0;
}

int main(int argc, char** argv) { return RunGuardedMain("test_hybrid_real_identity", [&] { return RunTest(argc, argv); }); }
