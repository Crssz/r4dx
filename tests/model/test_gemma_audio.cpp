// tests/model/test_gemma_audio.cpp -- docs/gemma4-audio.md M3-2: the Gemma 4 audio splice on the real container.
// GPU test, HIP device 1 only. NOT RUN BY THE AUTHOR (no GPU work was allowed); exits 77 (SKIPPED) while the
// container is missing or was converted without `--audio on`.
//
//   $env:HIP_VISIBLE_DEVICES='1'
//   build\win-hip\tests\model\test_gemma_audio.exe          (or: ctest -R gemma_audio)
//   R4DX_GEMMA_AUDIO_CONTAINER=<container>   default <R4DX_MODELS_ROOT>\r4dx\huihui-gemma\bf16-audio.r4dx
//   (convert it first:  r4dx-convert --model-dir <R4DX_MODELS_ROOT>\Huihui-gemma-4-12B-it-abliterated
//                        --out <R4DX_MODELS_ROOT>\r4dx\huihui-gemma\bf16-audio.r4dx --audio on)
//
// The CPU half (framing, WAV, the embedder vs HF) is tests/audio; this is the device half, with the prompt layout
// tools/reference/gemma/audio_golden_gemma.py recorded from the real chat template + processor:
//   <bos><|turn>user\nTranscribe:<|audio><|audio|> x n<audio|><turn|>\n<|turn>model\n<|channel>thought\n<channel|>
// Checks:
//   1. PrefillAudio of a synthetic 1 s tone (25 tokens) gives finite, non-degenerate logits; 16 greedy tokens follow
//      (printed as ids -- use r4dx-server with an input_audio part to read them as text).
//   2. The audio matters: a different waveform of the same length gives different logits (max |diff| > 0.05) and the
//      same waveform again gives bit-identical logits (determinism of the splice).
//   3. A 750-token (30 s) clip works, and splitting the same prefill inside the audio span (a continuation call, the
//      engine's prefix-reuse shape: span offset 0 with the rows pointer advanced) agrees with the one-call prefill
//      (top-1 equal, max |logit diff| < 1.0 -- different chunk boundaries move bf16 rounding a little).
//   4. A span outside the call's tokens is refused.
#include <algorithm>
#include "r4dx/models_root.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "audio_embed.h"
#include "audio_frames.h"
#include "gemma_model.h"
#include "test_common.h"

using namespace r4dx;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::vector<float> Tone(int64_t samples, float hz, float amp) {
  std::vector<float> x(static_cast<size_t>(samples));
  for (size_t i = 0; i < x.size(); ++i) x[i] = amp * std::sin(2.0f * 3.14159265f * hz * static_cast<float>(i) / 16000.0f);
  return x;
}

// The template's prompt with `n` audio soft tokens (ids recorded by audio_golden_gemma.py; BOA/EOA/AUDIO from
// audio_frames.h). Returns the token vector and fills the span (offset of the first soft token).
std::vector<int32_t> Prompt(int64_t n, int64_t* offset) {
  std::vector<int32_t> t = {2, 105, 2364, 107, 5183, 17038, 236787, audio::kBoaTokenId};
  *offset = static_cast<int64_t>(t.size());
  t.insert(t.end(), static_cast<size_t>(n), audio::kAudioTokenId);
  for (int32_t id : {audio::kEoaTokenId, 106, 107, 105, 4368, 107, 100, 45518, 107, 101}) t.push_back(id);
  return t;
}

int64_t Argmax(const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); }
float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
bool Finite(const std::vector<float>& v) {
  for (float x : v) if (!std::isfinite(x)) return false;
  return true;
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_gemma_audio", []() -> int {
    const char* e = std::getenv("R4DX_GEMMA_AUDIO_CONTAINER");
    const std::string container = e && *e ? e : r4dx::ModelsPath("r4dx/huihui-gemma/bf16-audio.r4dx");
    if (!r4dx_test::FileExists(container)) return r4dx_test::SkipMissing(container);
    if (!audio::ContainerHasAudio(container)) return r4dx_test::SkipMissing(container + " (converted without --audio on)");

    const audio::AudioEmbedder emb = audio::LoadAudioEmbedder(container);
    GemmaModelOptions o;
    o.container_path = container;
    o.max_ctx = 4096;
    ApplyGemmaEnv(&o);
    GemmaModel m = GemmaModel::Load(o);

    auto prefill = [&](const std::vector<float>& wav, const std::vector<int32_t>* force_tokens = nullptr) {
      int64_t n = 0;
      const std::vector<float> frames = audio::FrameWaveform(wav, &n);
      const std::vector<uint16_t> rows = emb.Embed(frames.data(), n);
      int64_t off = 0;
      const std::vector<int32_t> toks = force_tokens ? *force_tokens : Prompt(n, &off);
      if (force_tokens) off = 8;
      AudioRowSpan sp;
      sp.offset = off;
      sp.tokens = n;
      sp.rows = rows.data();
      m.Reset();
      return m.PrefillAudio(toks, {sp});
    };

    // 1. a 1 s tone
    const std::vector<float> a = prefill(Tone(16000, 440.0f, 0.5f));
    Check(static_cast<int64_t>(a.size()) == m.Config().vocab_size && Finite(a), "1 s tone: finite logits over the full vocab");
    {
      std::vector<float> sorted = a;
      std::sort(sorted.begin(), sorted.end());
      Check(sorted.back() - sorted[sorted.size() / 2] > 1.0f, "1 s tone: the logits are not flat");
    }
    std::printf("first greedy ids after the tone:");
    int32_t next = static_cast<int32_t>(Argmax(a));
    for (int i = 0; i < 16; ++i) {
      std::printf(" %d", next);
      next = m.DecodeStepGreedy(next);
    }
    std::printf("\n");

    // 2. audio matters; determinism
    const std::vector<float> b = prefill(Tone(16000, 1750.0f, 0.1f));
    const std::vector<float> a2 = prefill(Tone(16000, 440.0f, 0.5f));
    Check(MaxAbsDiff(a, b) > 0.05f, "a different waveform of the same length changes the logits");
    Check(MaxAbsDiff(a, a2) == 0.0f, "the same waveform gives bit-identical logits");

    // 3. 30 s, split inside the span
    const std::vector<float> long_wav = Tone(audio::kMaxAudioSamples, 300.0f, 0.3f);
    const std::vector<float> one = prefill(long_wav);
    Check(Finite(one), "30 s clip (750 tokens): finite logits");
    {
      int64_t n = 0;
      const std::vector<float> frames = audio::FrameWaveform(long_wav, &n);
      const std::vector<uint16_t> rows = emb.Embed(frames.data(), n);
      int64_t off = 0;
      const std::vector<int32_t> toks = Prompt(n, &off);
      const int64_t split = off + 300;  // inside the audio span, not a multiple of the 256-row chunk
      m.Reset();
      AudioRowSpan s1{off, 300, rows.data()};
      m.PrefillAudio(std::vector<int32_t>(toks.begin(), toks.begin() + split), {s1});
      AudioRowSpan s2{0, n - 300, rows.data() + 300 * audio::kEmbedHidden};
      const std::vector<float> two = m.PrefillAudio(std::vector<int32_t>(toks.begin() + split, toks.end()), {s2});
      Check(Argmax(one) == Argmax(two), "split inside the span: same top-1 as the one-call prefill");
      const float d = MaxAbsDiff(one, two);
      std::printf("split vs one-call: max |logit diff| = %g\n", d);
      Check(d < 1.0f, "split inside the span: max |logit diff| < 1.0");
    }

    // 4. a bad span is refused
    {
      int64_t n = 0;
      const std::vector<float> frames = audio::FrameWaveform(Tone(640, 440.0f, 0.5f), &n);
      const std::vector<uint16_t> rows = emb.Embed(frames.data(), n);
      m.Reset();
      bool threw = false;
      try {
        m.PrefillAudio({2, 105, 106}, {AudioRowSpan{2, 4, rows.data()}});
      } catch (const std::exception&) {
        threw = true;
      }
      Check(threw, "a span past the end of the tokens is refused");
    }
    return g_failures == 0 ? 0 : 1;
  });
}
