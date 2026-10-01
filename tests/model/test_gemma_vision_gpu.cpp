// tests/model/test_gemma_vision_gpu.cpp -- docs/gemma4-plan.md M2-3: the Gemma 4 vision path on the GPU against the
// reference golden tools/reference/gemma/vision_golden_gemma.py writes (transformers' image processor, the REAL
// vision embedder weights, and layers 0 / 5 with an image block). GPU test, HIP device 1 only. NOT RUN BY ITS AUTHOR
// (no GPU work was allowed); exits 77 (SKIPPED) while the container or the golden is missing.
//
//   container  R4DX_GEMMA_VISION_CONTAINER (default D:\models\r4dx\huihui-gemma\bf16-vision.r4dx): an UNROTATED bf16
//              container converted with `r4dx-convert --vision on` (a rotated one runs the layer in the rotated
//              basis, which this test's direct layer calls do not model).
//   golden     R4DX_GEMMA_VISION_GOLDEN (default tools/reference/golden_out/gemma/vision_golden_gemma.safetensors)
//
// Checks (6-layer model, bf16 KV, reference attention; layers 0 and 5 are the sliding and the full layer):
//   1. EncodeImages on both golden images (non-square tall and wide) == the HF embedder's bf16 rows (rel L2 <= 2e-2;
//      the device adds the Linear bias after the GEMM's bf16 rounding, ~2^-9 relative).
//   2. DebugLayerForward(0, x_in, klimit_ext) == layer0_block, and is NOT layer0_causal: the bidirectional block works
//      on the sliding layer. DebugLayerForward(5, x_in) == layer5_full even when klimit_ext is given (full layers stay
//      causal), and is not layer5_full_if_bidir.
//   3. PrefillMultimodal end to end: finite logits, PositionCount == T; the same prompt fed as two calls (image A
//      first, then image B with spans relative to the tail) gives the same last-row logits (the chunk planner, spans
//      and continuation are consistent); a decode step afterwards is finite. PrefillMultimodal({}) == Prefill
//      bit for bit (the text path is untouched).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gemma_model.h"
#include "gemma_vision.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"

#ifndef R4DX_SOURCE_DIR_STR
#define R4DX_SOURCE_DIR_STR "."
#endif

using namespace r4dx;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::vector<int32_t> ReadI32(const r4dx_convert::SafetensorsReader& r, const std::string& name) {
  const auto& m = r.Meta(name);
  if (m.dtype != "I32") throw std::runtime_error(name + " is not I32");
  const auto* p = reinterpret_cast<const int32_t*>(r.Data(name));
  return std::vector<int32_t>(p, p + m.ElemCount());
}

std::vector<float> Widen(const std::vector<uint16_t>& v) { return r4dx_test::WidenBf16(v); }

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_gemma_vision_gpu", []() -> int {
    const auto env = [](const char* k, const std::string& d) { const char* v = std::getenv(k); return v && *v ? std::string(v) : d; };
    const std::string container = env("R4DX_GEMMA_VISION_CONTAINER", "D:\\models\\r4dx\\huihui-gemma\\bf16-vision.r4dx");
    const std::string golden_path = env("R4DX_GEMMA_VISION_GOLDEN", std::string(R4DX_SOURCE_DIR_STR) +
                                        "/tools/reference/golden_out/gemma/vision_golden_gemma.safetensors");
    if (!r4dx_test::FileExists(container)) return r4dx_test::SkipMissing(container);
    if (!r4dx_test::FileExists(golden_path)) return r4dx_test::SkipMissing(golden_path);

    r4dx_convert::SafetensorsReader g(r4dx_convert::Utf8ToWide(golden_path));
    const std::vector<int32_t> ids = ReadI32(g, "token_ids");
    const std::vector<int32_t> klimit = ReadI32(g, "klimit_ext");
    const int64_t T = static_cast<int64_t>(ids.size());
    const std::vector<uint16_t> x_in = r4dx_test::ReadGoldenRawBf16(g, "x_in");
    const std::vector<float> l0_block = r4dx_test::ReadGoldenAsFloat(g, "layer0_block");
    const std::vector<float> l0_causal = r4dx_test::ReadGoldenAsFloat(g, "layer0_causal");
    const std::vector<float> l5_full = r4dx_test::ReadGoldenAsFloat(g, "layer5_full");
    const std::vector<float> l5_bidir = r4dx_test::ReadGoldenAsFloat(g, "layer5_full_if_bidir");

    GemmaModelOptions o;
    o.container_path = container;
    o.layout = Layout::kBf16;
    o.layer_limit = 6;
    o.max_ctx = 2048;
    o.kv = GemmaKvMode::kBf16;
    o.attn = attention::GemmaAttnBackend::kReference;
    o.vision = GemmaVisionLoad::kOn;
    GemmaModel m = GemmaModel::Load(o);
    const int64_t hidden = m.Config().hidden_size;
    Check(m.HasVision(), "the container carries the vision embedder");

    // ---- 1. the embedder ----
    std::vector<core::DeviceBuffer<uint16_t>> emb(2);
    const char* tags[2] = {"A", "B"};
    for (int k = 0; k < 2; ++k) {
      const std::string t = tags[k];
      const std::vector<float> pix = r4dx_test::ReadGoldenAsFloat(g, "pixel_values_" + t);
      const std::vector<int32_t> pos = ReadI32(g, "positions_" + t);
      const int64_t n = static_cast<int64_t>(pos.size() / 2);
      int64_t gw = 0, gh = 0;
      for (int64_t i = 0; i < n; ++i) {
        gw = std::max<int64_t>(gw, pos[static_cast<size_t>(i) * 2] + 1);
        gh = std::max<int64_t>(gh, pos[static_cast<size_t>(i) * 2 + 1] + 1);
      }
      Check(gw * gh == n, "image " + t + ": positions form a full raster grid");
      const std::vector<int32_t> raster = GemmaGridPositions(gh, gw);
      Check(raster == pos, "image " + t + ": raster-order (x, y) positions equal HF's image_position_ids");
      m.EncodeImages(pix.data(), n, {vision::GridThw{1, gh, gw}}, &emb[static_cast<size_t>(k)]);
      std::vector<uint16_t> got(static_cast<size_t>(n * hidden));
      emb[static_cast<size_t>(k)].CopyToHost(got.data(), got.size());
      const double rel = r4dx_test::RelL2(Widen(got), r4dx_test::ReadGoldenAsFloat(g, "embed_" + t));
      std::printf("embedder %s: %lld tokens, rel L2 vs HF %.3e\n", tags[k], static_cast<long long>(n), rel);
      Check(rel <= 2e-2, std::string("EncodeImages == HF embedder rows, image ") + tags[k]);
    }

    // ---- 2. layers with an image block ----
    {
      const std::vector<uint16_t> out = m.DebugLayerForward(0, x_in, T, 0, &klimit);
      const double e_block = r4dx_test::RelL2(Widen(out), l0_block);
      const double e_causal = r4dx_test::RelL2(Widen(out), l0_causal);
      std::printf("layer 0 (sliding): rel L2 vs block-mask golden %.3e, vs causal control %.3e\n", e_block, e_causal);
      Check(e_block <= 3e-2, "layer 0 with klimit_ext == the HF block-mask layer output");
      Check(e_causal >= 0.08, "layer 0 with klimit_ext is clearly not the causal output (the block is bidirectional)");
      const std::vector<uint16_t> out_plain = m.DebugLayerForward(0, x_in, T, 0, nullptr);
      Check(r4dx_test::RelL2(Widen(out_plain), l0_causal) <= 3e-2, "layer 0 without klimit_ext == the causal control");
    }
    {
      const std::vector<uint16_t> out = m.DebugLayerForward(5, x_in, T, 0, &klimit);
      const double e_full = r4dx_test::RelL2(Widen(out), l5_full);
      const double e_bidir = r4dx_test::RelL2(Widen(out), l5_bidir);
      std::printf("layer 5 (full): rel L2 vs causal golden %.3e, vs forward()-path bidirectional %.3e\n", e_full, e_bidir);
      Check(e_full <= 3e-2, "layer 5 stays causal even when klimit_ext is passed (sliding-only decision)");
      Check(e_bidir >= 0.04, "layer 5 is not the forward()-path full-layer-bidirectional output");
    }

    // ---- 3. end to end ----
    std::vector<ImageSpan> spans(2);
    const int64_t a0 = 4, a1 = 4 + emb[0].size() / static_cast<size_t>(hidden);
    const int64_t b0 = a1 + 4, b1 = b0 + static_cast<int64_t>(emb[1].size() / static_cast<size_t>(hidden));
    spans[0].offset = a0;
    spans[0].tokens = a1 - a0;
    spans[0].embeds = emb[0].data();
    spans[1].offset = b0;
    spans[1].tokens = b1 - b0;
    spans[1].embeds = emb[1].data();
    Check(b1 + 1 + 3 == T, "golden layout matches the spans (eoi + 3 text tokens after image B)");
    m.Reset();
    const std::vector<float> one = m.PrefillMultimodal(ids, spans);
    Check(m.PositionCount() == T, "PositionCount == T after the multimodal prefill");
    bool finite = true;
    for (float v : one) finite = finite && std::isfinite(v);
    Check(finite, "multimodal prefill logits are finite");
    // two calls: head = tokens [0, b0 - 1) (everything through image A's eoi and the text after it, before B's boi)
    const int64_t split = b0 - 1;
    m.Reset();
    const std::vector<int32_t> head(ids.begin(), ids.begin() + split), tail(ids.begin() + split, ids.end());
    std::vector<ImageSpan> s_head = {spans[0]}, s_tail = {spans[1]};
    s_tail[0].offset = b0 - split;
    m.PrefillMultimodal(head, s_head);
    const std::vector<float> two = m.PrefillMultimodal(tail, s_tail);
    double num = 0, den = 0;
    for (size_t i = 0; i < one.size(); ++i) {
      num += (one[i] - two[i]) * (static_cast<double>(one[i]) - two[i]);
      den += static_cast<double>(one[i]) * one[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-9));
    std::printf("two-call vs one-call logits: rel L2 %.3e\n", rel);
    Check(rel <= 2e-2, "splitting the multimodal prefill at an image boundary does not change the logits");
    const std::vector<float> d = m.DecodeStep(106);
    bool dfin = true;
    for (float v : d) dfin = dfin && std::isfinite(v);
    Check(dfin && m.PositionCount() == T + 1, "a decode step after the image prefill is finite");
    // text-only: PrefillMultimodal({}) is Prefill
    const std::vector<int32_t> text(ids.begin(), ids.begin() + 3);
    m.Reset();
    const std::vector<float> p1 = m.Prefill(text);
    m.Reset();
    const std::vector<float> p2 = m.PrefillMultimodal(text, {});
    Check(p1 == p2, "PrefillMultimodal with no spans is Prefill, bit for bit");

    std::printf("%s (%d failure(s))\n", g_failures == 0 ? "ALL PASS" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
  });
}
