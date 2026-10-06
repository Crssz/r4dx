// tests/model/test_gemma_vision_tp.cpp -- Gemma 4 vision and audio under tensor parallelism (docs/gemma4-plan.md, docs/tp.md
// Gemma section). TWO-GPU test, opt-in exactly like test_gemma_dflash_tp: SKIP (77) unless R4DX_TP2GPU=1, two visible HIP
// devices and the container. WRITTEN, NOT RUN by its author (no GPU work was allowed); run it with
// `tests\run_tests.ps1 -TwoGpu` (LABEL tp2gpu) only when both GPUs are free.
//
// The same prompts go through a TP=1 TextModel (device 1, the compute card) and a TP=2 real group, both from LoadTextModel:
//   V  vision: two synthetic images (12x16 and 16x16 merged cells) wrapped in boi / eoi inside a short prompt.
//        - TP=2 EncodeImages rows (host, rank 0) == TP=1 rows (the same kernels on the same weights: bitwise, hard gate)
//        - TP=2 PrefillMultimodal last-row logits vs TP=1: KL(TP=1 || TP=2) <= 2e-3, same top-1 (gated); cosine reported
//          (the text-only TP=2 gate's numerics: the vocab split and the row-parallel o/down all-reduces)
//        - 24 greedy tokens after the image: the divergence position is reported (near-tie flips are expected)
//        - a text-only Prefill after the multimodal one (Reset in between) still works; PositionCount tracks
//        - with R4DX_GEMMA_VISION_TP_DRAFTER set (a dflash2 container): DFlash rounds after the image == plain TP=2 greedy
//          (injection covers the image rows exactly as TP=1: features are replicated)
//   A  audio (only when R4DX_GEMMA_VISION_TP_AUDIO or the main container carries audio.*): a 2 s tone through EncodeAudio
//        (bitwise equal to TP=1: the CPU embedder on the facade) and PrefillAudio logits / greedy as for V.
// Env: R4DX_GEMMA_VISION_TP_CONTAINER (default <R4DX_MODELS_ROOT>\r4dx\huihui-gemma\bf16-vision.r4dx), R4DX_GEMMA_VISION_TP_LAYOUT
// (default bf16; trellis for a trellis container), R4DX_GEMMA_VISION_TP_AUDIO (default: the main container),
// R4DX_GEMMA_VISION_TP_DRAFTER (optional dflash2 container).
#include <hip/hip_runtime.h>
#include "r4dx/models_root.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "audio_embed.h"
#include "audio_frames.h"
#include "model.h"
#include "test_common.h"
#include "text_model.h"

using namespace r4dx;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string Env(const char* name, const char* def) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string(def);
}

constexpr int64_t kPatchDim = 6912, kGen = 24;

int32_t ArgmaxOf(const std::vector<float>& v) {
  return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());  // lowest index on ties
}

std::vector<float> Pixels(int64_t patches, float phase) {
  std::vector<float> p(static_cast<size_t>(patches * kPatchDim));
  for (size_t i = 0; i < p.size(); ++i) p[i] = 0.5f + 0.5f * std::sin(0.00137f * static_cast<float>(i) + phase);
  return p;
}

std::vector<float> Tone(int64_t samples, float hz, float amp) {
  std::vector<float> x(static_cast<size_t>(samples));
  for (size_t i = 0; i < x.size(); ++i) x[i] = amp * std::sin(2.0f * 3.14159265f * hz * static_cast<float>(i) / 16000.0f);
  return x;
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  return ab / std::sqrt(std::max(aa * bb, 1e-300));
}

std::vector<double> LogSoftmax(const std::vector<float>& l) {
  const double mx = *std::max_element(l.begin(), l.end());
  double s = 0;
  for (float v : l) s += std::exp(static_cast<double>(v) - mx);
  const double lse = mx + std::log(s);
  std::vector<double> o(l.size());
  for (size_t i = 0; i < l.size(); ++i) o[i] = static_cast<double>(l[i]) - lse;
  return o;
}

double Kl(const std::vector<float>& p_logits, const std::vector<float>& q_logits) {
  const std::vector<double> lp = LogSoftmax(p_logits), lq = LogSoftmax(q_logits);
  double kl = 0;
  for (size_t i = 0; i < lp.size(); ++i) kl += std::exp(lp[i]) * (lp[i] - lq[i]);
  return kl;
}

int64_t FirstDiff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return static_cast<int64_t>(i);
  }
  return a.size() == b.size() ? -1 : static_cast<int64_t>(n);
}

// What one model produces for the V and A prompts.
struct Captured {
  std::vector<uint16_t> image_rows;  // both images, concatenated, host bf16
  std::vector<float> v_logits;
  std::vector<int32_t> v_greedy, v_dflash;  // v_dflash: DFlash rounds over the same prompt (only when the model has a drafter)
  int64_t v_len = 0;
  bool has_audio = false;
  std::vector<uint16_t> audio_rows;
  std::vector<float> a_logits;
  std::vector<int32_t> a_greedy;
};

std::vector<uint16_t> RowsToHost(const ImageRows& r, int64_t hidden) {
  std::vector<uint16_t> v(static_cast<size_t>(r.rows() * hidden));
  if (r.on_host()) {
    std::copy(r.data(), r.data() + v.size(), v.begin());
  } else {
    r.dev.CopyToHost(v.data(), v.size());
  }
  return v;
}

std::vector<int32_t> GreedyAfter(TextModel& m, const std::vector<float>& first_logits) {
  std::vector<int32_t> out{ArgmaxOf(first_logits)};
  while (static_cast<int64_t>(out.size()) < kGen) out.push_back(m.DecodeStepGreedy(out.back()));
  return out;
}

Captured Capture(TextModel& m, bool want_audio) {
  Captured c;
  const int64_t hidden = m.Config().hidden_size;
  Check(m.HasVision() && m.VisionMergeSize() == 1, "HasVision() and VisionMergeSize() == 1 (the Gemma processor merges)");
  Check(m.ImageBoiTokenId() > 0 && m.ImageEoiTokenId() > 0, "the model reports Gemma's boi / eoi wrapper ids");
  // ---- V: two images -------------------------------------------------------------------------------------
  const vision::GridThw g0{1, 12, 16}, g1{1, 16, 16};
  const std::vector<float> px0 = Pixels(g0.PatchCount(), 0.3f), px1 = Pixels(g1.PatchCount(), 1.1f);
  ImageRows r0, r1;
  m.EncodeImages(px0.data(), g0.PatchCount(), {g0}, &r0);
  m.EncodeImages(px1.data(), g1.PatchCount(), {g1}, &r1);
  Check(r0.rows() == g0.PatchCount() && r1.rows() == g1.PatchCount(), "EncodeImages rows == soft-token counts");
  Check(r0.on_host() == (m.TpWorld() == 2), "image rows are host rows exactly under TP");
  const std::vector<uint16_t> h0 = RowsToHost(r0, hidden), h1 = RowsToHost(r1, hidden);
  c.image_rows = h0;
  c.image_rows.insert(c.image_rows.end(), h1.begin(), h1.end());

  const int32_t img = static_cast<int32_t>(m.ImageTokenId()), boi = static_cast<int32_t>(m.ImageBoiTokenId()),
                eoi = static_cast<int32_t>(m.ImageEoiTokenId());
  std::vector<int32_t> toks = {2, 105, 2364, 107, 5183, 17038};
  std::vector<ImageSpan> spans;
  for (const auto& [rows, grid, r] : {std::tuple<int64_t, vision::GridThw, const ImageRows*>{g0.PatchCount(), g0, &r0},
                                       std::tuple<int64_t, vision::GridThw, const ImageRows*>{g1.PatchCount(), g1, &r1}}) {
    toks.push_back(boi);
    ImageSpan sp;
    sp.offset = static_cast<int64_t>(toks.size());
    sp.tokens = rows;
    sp.grid = grid;
    sp.embeds = r->data();
    sp.embeds_on_host = r->on_host();
    spans.push_back(sp);
    toks.insert(toks.end(), static_cast<size_t>(rows), img);
    toks.push_back(eoi);
    toks.insert(toks.end(), {107, 5183});
  }
  toks.insert(toks.end(), {106, 107, 105, 4368, 107});
  c.v_len = static_cast<int64_t>(toks.size());
  m.Reset();
  c.v_logits = m.PrefillMultimodal(toks, spans);
  Check(m.PositionCount() == c.v_len, "PositionCount == the multimodal prompt's length");
  c.v_greedy = GreedyAfter(m, c.v_logits);
  if (m.DflashEnabled()) {
    m.Reset();
    c.v_dflash = {ArgmaxOf(m.PrefillMultimodal(toks, spans))};
    while (static_cast<int64_t>(c.v_dflash.size()) < kGen) {
      for (int32_t tk : m.DecodeStepDflashGreedy(c.v_dflash.back(), 7, 0.0f, 0)) {
        if (static_cast<int64_t>(c.v_dflash.size()) < kGen) c.v_dflash.push_back(tk);
      }
    }
  }
  // text-only after a multimodal one, and the empty-span path
  m.Reset();
  const std::vector<int32_t> text(toks.begin(), toks.begin() + 6);
  const std::vector<float> p1 = m.Prefill(text);
  m.Reset();
  const std::vector<float> p2 = m.PrefillMultimodal(text, {});
  Check(p1 == p2, "PrefillMultimodal with no spans is Prefill, bit for bit");

  // ---- A: audio ------------------------------------------------------------------------------------------
  if (want_audio && m.HasAudio()) {
    c.has_audio = true;
    int64_t n = 0;
    const std::vector<float> frames = audio::FrameWaveform(Tone(32000, 440.0f, 0.5f), &n);
    c.audio_rows = m.EncodeAudio(frames.data(), n);
    std::vector<int32_t> t = {2, 105, 2364, 107, 5183, 17038, 236787, audio::kBoaTokenId};
    AudioRowSpan sp;
    sp.offset = static_cast<int64_t>(t.size());
    sp.tokens = n;
    sp.rows = c.audio_rows.data();
    t.insert(t.end(), static_cast<size_t>(n), audio::kAudioTokenId);
    for (int32_t id : {audio::kEoaTokenId, 106, 107, 105, 4368, 107, 100, 45518, 107, 101}) t.push_back(id);
    m.Reset();
    c.a_logits = m.PrefillAudio(t, {sp});
    Check(m.PositionCount() == static_cast<int64_t>(t.size()), "PositionCount == the audio prompt's length");
    c.a_greedy = GreedyAfter(m, c.a_logits);
  }
  return c;
}

void CompareLogits(const char* what, const std::vector<float>& a, const std::vector<float>& b, const std::vector<int32_t>& ga,
                   const std::vector<int32_t>& gb) {
  Check(a.size() == b.size() && !a.empty(), std::string(what) + ": same vocab width");
  const double cos = Cosine(a, b), kl = Kl(a, b);
  const int64_t d = FirstDiff(ga, gb);
  std::printf("[report] %s: cosine %.6f, KL(TP=1||TP=2) %.3e, greedy first diff %lld of %lld\n", what, cos, kl,
              static_cast<long long>(d), static_cast<long long>(kGen));
  // Gated like the accepted text-only TP=2 gate (docs/gemma4-plan.md item 12, 2026-10-02): KL and top-1. Cosine of raw
  // logits is dominated by the low-probability tail (vision measured cosine 0.99886 at KL 3.4e-8), and a later greedy
  // flip is the expected near-tie effect of the all-reduce summation order (text TP2-vs-TP1 top-1 98.9-100%): both are
  // reported, not gated.
  Check(kl <= 2e-3, std::string(what) + ": KL <= 2e-3");
  Check(ArgmaxOf(a) == ArgmaxOf(b), std::string(what) + ": same top-1");
}

ModelOptions Options(const std::string& target, const std::string& layout, const std::string& drafter) {
  ModelOptions o;
  o.container_path = target;
  o.layout = LayoutFromName(layout);
  o.max_ctx = 2048;
  o.vision = ModelOptions::VisionMode::kOn;
  if (!drafter.empty()) {
    o.dflash_container = drafter;
    o.dflash_draft_k = 7;
  }
  return o;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (Env("R4DX_TP2GPU", "") != "1") {
    std::printf("SKIP: two-GPU test; run tests\\run_tests.ps1 -TwoGpu\n");
    return 77;
  }
  const std::string target = Env("R4DX_GEMMA_VISION_TP_CONTAINER", r4dx::ModelsPath("r4dx/huihui-gemma/bf16-vision.r4dx").c_str());
  const std::string layout = Env("R4DX_GEMMA_VISION_TP_LAYOUT", "bf16");
  const std::string drafter = Env("R4DX_GEMMA_VISION_TP_DRAFTER", "");
  if (!r4dx_test::FileExists(target)) return r4dx_test::SkipMissing(target);
  if (!drafter.empty() && !r4dx_test::FileExists(drafter)) return r4dx_test::SkipMissing(drafter);
  int visible = 0;
  if (hipGetDeviceCount(&visible) != hipSuccess || visible < 2) {
    std::printf("SKIP: needs two visible HIP devices (have %d)\n", visible);
    return 77;
  }
  try {
    Captured one, two;
    {  // TP=1 on the compute card
      (void)hipSetDevice(visible - 1);
      TpOptions t1;
      t1.world = 1;
      std::unique_ptr<TextModel> m = LoadTextModel(Options(target, layout, ""), t1);
      one = Capture(*m, /*want_audio=*/true);
    }
    {
      TpOptions t2;
      t2.world = 2;
      t2.mode = TpOptions::Mode::kReal;
      std::unique_ptr<TextModel> m = LoadTextModel(Options(target, layout, drafter), t2);
      Check(m->TpWorld() == 2 && m->HasVision(), "TP=2 group reports HasVision()");
      Check(m->HasAudio() == one.has_audio, "TP=2 group reports the same HasAudio() as TP=1");
      two = Capture(*m, /*want_audio=*/true);

      Check(one.image_rows == two.image_rows, "TP=2 EncodeImages rows are bitwise equal to TP=1's");
      CompareLogits("vision prefill", one.v_logits, two.v_logits, one.v_greedy, two.v_greedy);
      if (one.has_audio && two.has_audio) {
        Check(one.audio_rows == two.audio_rows, "TP=2 EncodeAudio rows are bitwise equal to TP=1's");
        CompareLogits("audio prefill", one.a_logits, two.a_logits, one.a_greedy, two.a_greedy);
      } else {
        std::printf("[report] audio: skipped (the container carries no audio.* tensors)\n");
      }

      if (m->DflashEnabled()) {
        // DFlash after an image: the injection covers the image rows (the captured features are replicated and the image rows
        // are spliced identically on both ranks), so greedy rounds equal plain TP=2 greedy over the same image prompt.
        const int64_t d = FirstDiff(two.v_greedy, two.v_dflash);
        std::printf("[report] TP=2 DFlash after an image: first diff vs plain greedy %lld of %lld\n", static_cast<long long>(d),
                    static_cast<long long>(kGen));
        Check(d < 0, "TP=2 DFlash greedy rounds after an image prompt are byte-identical to plain TP=2 greedy");
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gemma_vision_tp: exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
