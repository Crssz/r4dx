// Ornith (qwen3_5_moe) image input on the tiny container + the REAL vision tower (docs/qwen35moe-vision.md). Links r4dx_model.
//
// Fixtures (exit 77 = skip when one is missing; tools/reference/qwen35moe/tiny_vision.py builds them, r4dx-convert --vision on the container):
//   R4DX_Q35MOE_VISION_CONTAINER  default E:\models\r4dx\tiny-q35moe-vision\tiny-bf16-vision.r4dx  (tiny bf16 text stack + 333 vision.* tensors)
//   R4DX_Q35MOE_VISION_GOLDEN     default E:\models\r4dx\tiny-q35moe-vision\golden        (one_image / two_images: HF bf16, fp8 KV)
//   R4DX_Q35MOE_TEXT_CONTAINER    default E:\models\r4dx\tiny-q35moe\tiny-bf16.r4dx        (the same text stack without a tower)
//
// What it checks, per golden case:
//   1. the tower alone: Qwen35MoeModel::EncodeImages over the golden's own pixel_values against the HF merger output;
//   2. the whole model: PrefillMultimodal with a capture (every layer's residual, the logits of every row, which include the spliced
//      image rows and the 3-axis rope) against the HF run, the last row's argmax, and the greedy continuation after the image (the
//      decode path ropes at index + mrope delta);
//   3. the same prompt fed in two calls (text, then the image part): continuation blocks, offsets relative to the call;
//   4. text only: with a tower loaded, a text prompt's logits and a few decode steps are BIT-identical to the model without one.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "q35moe_model_util.h"

using namespace q35m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string EnvOr(const char* name, const char* dflt) {
  const char* e = std::getenv(name);
  return e != nullptr && e[0] != 0 ? e : dflt;
}

struct Case {
  std::unique_ptr<r4dx_convert::SafetensorsReader> r;
  std::vector<int32_t> ids;
  std::vector<float> pixels;
  std::vector<r4dx::vision::GridThw> grids;
  std::vector<int32_t> cont;
  int64_t delta = 0;
  int64_t image_token = 0;
};

Case LoadCase(const std::string& dir, const std::string& name) {
  Case c;
  c.r = std::make_unique<r4dx_convert::SafetensorsReader>(r4dx_convert::Utf8ToWide(dir + "\\" + name + ".safetensors"));
  const int64_t n = c.r->Meta("input_ids").ElemCount();
  const int64_t* p = reinterpret_cast<const int64_t*>(c.r->Data("input_ids"));
  for (int64_t i = 0; i < n; ++i) c.ids.push_back(static_cast<int32_t>(p[i]));
  const auto& pm = c.r->Meta("pixel_values");
  c.pixels.resize(static_cast<size_t>(pm.ElemCount()));
  std::memcpy(c.pixels.data(), c.r->Data("pixel_values"), c.pixels.size() * 4);
  const int64_t* g = reinterpret_cast<const int64_t*>(c.r->Data("image_grid_thw"));
  for (int64_t i = 0; i < c.r->Meta("image_grid_thw").ElemCount() / 3; ++i) c.grids.push_back({g[3 * i], g[3 * i + 1], g[3 * i + 2]});
  std::ifstream f(dir + "\\" + name + ".json", std::ios::binary);
  const nlohmann::json j = nlohmann::json::parse(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
  c.cont = j.at("continuation").get<std::vector<int32_t>>();
  c.delta = j.at("mrope_delta").get<int64_t>();
  c.image_token = j.at("image_token_id").get<int64_t>();
  return c;
}

// The spans of ids[first, last): every maximal run of the image token, in order, with `rows` (host bf16 [tokens, hidden], images
// back to back in `grids` order, `first_image` = index of the first image in this window).
std::vector<r4dx::model::ImageSpan> SpansOf(const Case& c, const std::vector<uint16_t>& rows, int64_t hidden, size_t first, size_t last,
                                           size_t first_image) {
  std::vector<r4dx::model::ImageSpan> out;
  size_t img = first_image;
  int64_t row0 = 0;
  for (size_t i = 0; i < first_image; ++i) row0 += c.grids[i].MergedTokenCount(2);
  for (size_t i = first; i < last;) {
    if (c.ids[i] != c.image_token) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < last && c.ids[j] == c.image_token) ++j;
    r4dx::model::ImageSpan sp;
    sp.offset = static_cast<int64_t>(i - first);
    sp.tokens = static_cast<int64_t>(j - i);
    sp.grid = c.grids.at(img);
    if (sp.grid.MergedTokenCount(2) != sp.tokens) throw std::runtime_error("golden run length does not match its grid");
    sp.embeds = rows.data() + row0 * hidden;
    sp.embeds_on_host = true;
    out.push_back(sp);
    row0 += sp.tokens;
    ++img;
    i = j;
  }
  return out;
}

struct Tol {
  double mean_norm, atol_rel, rtol, max_bad;
};
Tol TolFor(const std::string& name) {
  if (name == "logits") return {0.15, 0.25, 0.10, 0.02};
  if (name.find("residual") != std::string::npos) return {0.15, 0.25, 0.10, 0.02};
  return {0.15, 0.25, 0.10, 0.02};
}

void CompareCaptures(const CaptureStore& store, const Case& c, const std::string& tag) {
  PrintStatsHeader();
  for (const auto& [name, cap] : store.all()) {
    if (!c.r->Has(name) || cap.dtype != Qwen35MoeCapture::Dtype::kBf16) continue;
    if (name.find("residual_out") == std::string::npos && name != "layer0.residual_in" && name != "logits" && name != "norm_out") continue;
    const auto& m = c.r->Meta(name);
    const int64_t n = m.ElemCount();
    if (static_cast<int64_t>(cap.bf16.size()) != n) {
      Check(false, tag + " " + name + ": element count " + std::to_string(cap.bf16.size()) + " vs golden " + std::to_string(n));
      continue;
    }
    const uint16_t* ref = reinterpret_cast<const uint16_t*>(c.r->Data(name));
    const Stats s = CompareBf16(cap.bf16.data(), ref, n);
    PrintStats(name, s);
    const Tol t = TolFor(name);
    const double bad = BadFraction(cap.bf16.data(), ref, n, t.atol_rel, t.rtol);
    Check(!s.nan, tag + " " + name + ": NaN / Inf");
    Check(s.mean_norm <= t.mean_norm && bad <= t.max_bad,
          tag + " " + name + ": above its bound (mean/rms " + std::to_string(s.mean_norm) + ", " + std::to_string(100.0 * bad) + "% outside)");
  }
}

int64_t LastArgmaxGolden(const Case& c) {
  const auto& m = c.r->Meta("logits");
  const int64_t T = m.shape[0], V = m.shape[1];
  const std::vector<float> row = WidenBf16(reinterpret_cast<const uint16_t*>(c.r->Data("logits")) + (T - 1) * V, V);
  return ArgMax(row.data(), V);
}

}  // namespace

int RunTest();
int main() {
  Unbuffer();
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 1;
  }
}

int RunTest() {
  const std::string vcont = EnvOr("R4DX_Q35MOE_VISION_CONTAINER", "E:\\models\\r4dx\\tiny-q35moe-vision\\tiny-bf16-vision.r4dx");
  const std::string tcont = EnvOr("R4DX_Q35MOE_TEXT_CONTAINER", "E:\\models\\r4dx\\tiny-q35moe\\tiny-bf16.r4dx");
  const std::string gdir = EnvOr("R4DX_Q35MOE_VISION_GOLDEN", "E:\\models\\r4dx\\tiny-q35moe-vision\\golden");
  for (const std::string& p : {vcont, tcont, gdir + "\\one_image.safetensors", gdir + "\\two_images.safetensors"}) {
    if (!Exists(p)) {
      std::fprintf(stderr, "[SKIP] %s not found\n", p.c_str());
      return kSkip;
    }
  }
  const int dev = TestDevice();
  Qwen35MoeModelOptions o;
  o.container_path = vcont;
  o.max_ctx = 4096;
  o.devices = {dev, dev};
  o.vision = 1;
  auto model = Qwen35MoeModel::Load(o);
  Check(model->HasVision(), "the model has no tower");
  Check(model->ImageTokenId() == 4090, "image token id from the container");
  const int64_t hidden = model->Config().hidden_size;
  const int64_t vocab = model->Config().vocab_size;

  for (const char* name : {"one_image", "two_images"}) {
    const Case c = LoadCase(gdir, name);
    const std::string tag = name;
    std::printf("== %s: T=%zu images=%zu delta(golden)=%lld\n", name, c.ids.size(), c.grids.size(), static_cast<long long>(c.delta));

    // 1. the tower alone
    std::vector<uint16_t> rows, gold_rows;  // our tower's rows / the HF tower's rows (what the model checks below splice: isolates the text side)
    r4dx::vision::VisionEncodeStats st;
    model->EncodeImages(c.pixels.data(), static_cast<int64_t>(c.pixels.size()) / 1536, c.grids, &rows, &st);
    {
      const auto& m = c.r->Meta("image_embeds");
      Check(static_cast<int64_t>(rows.size()) == m.ElemCount(), tag + ": tower row count");
      if (static_cast<int64_t>(rows.size()) == m.ElemCount()) {
        const uint16_t* ref = reinterpret_cast<const uint16_t*>(c.r->Data("image_embeds"));
        const Stats s = CompareBf16(rows.data(), ref, m.ElemCount());
        PrintStatsHeader();
        PrintStats("image_embeds", s);
        std::printf("  tower: %lld patches -> %lld tokens in %.1f ms\n", static_cast<long long>(st.total_patches),
                    static_cast<long long>(st.merged_tokens), st.encode_ms);
        // bf16 tower, 27 blocks, against HF's bf16 (eager scores rounded to bf16): docs/vision.md measured mean/rms ~1e-2 at the merger
        // docs/vision.md "Why the deep-block disagreement is not an r4dx error": the reference's eager attention rounds scores to bf16, so the
        // tower sits at rel-L2 ~6.7e-2 from it at the merger (7.8e-2 at block 26) on the 27B; 0.12 is that measured band.
        double num = 0, den = 0;
        for (int64_t i = 0; i < m.ElemCount(); ++i) {
          const double d = static_cast<double>(Bf(rows[static_cast<size_t>(i)])) - Bf(ref[i]);
          num += d * d;
          den += static_cast<double>(Bf(ref[i])) * Bf(ref[i]);
        }
        std::printf("  tower rel-L2 vs HF: %.4f (band 0.12)\n", std::sqrt(num / den));
        Check(!s.nan && std::sqrt(num / den) <= 0.12, tag + ": tower output rel-L2 above 0.12");
        gold_rows.assign(ref, ref + m.ElemCount());
      }
    }

    // 2. the whole model, capture on
    {
      model->Reset();
      CaptureStore store;
      model->SetCapture(store.Fn());
      const auto spans = SpansOf(c, gold_rows, hidden, 0, c.ids.size(), 0);
      const std::vector<float> logits = model->PrefillMultimodal(c.ids, spans);
      model->SetCapture({});
      CompareCaptures(store, c, tag);
      const int64_t mine = ArgMax(logits.data(), vocab), theirs = LastArgmaxGolden(c);
      std::printf("  last-row argmax ours %lld golden %lld; mrope delta ours %lld golden %lld\n", static_cast<long long>(mine),
                  static_cast<long long>(theirs), static_cast<long long>(model->MropeDelta()), static_cast<long long>(c.delta));
      Check(mine == theirs, tag + ": last-row argmax differs from the golden");
      Check(model->MropeDelta() == c.delta, tag + ": mrope delta differs from get_rope_index's");
    }
    // the continuation after the image, TEACHER-FORCED with HF's own tokens (decode ropes at index + delta): our argmax must be HF's
    // token, or a near-tie of our own logits (the tiny random network's bf16 noise flips close calls; a free run would then diverge).
    {
      model->Reset();
      const auto spans = SpansOf(c, gold_rows, hidden, 0, c.ids.size(), 0);
      const std::vector<float> first = model->PrefillMultimodal(c.ids, spans);
      int agree = ArgMax(first.data(), vocab) == c.cont[0] ? 1 : 0, ties = 0, bad = 0;
      for (size_t i = 1; i < c.cont.size(); ++i) {
        const std::vector<float> lg = model->DecodeStep(c.cont[i - 1]);
        const int64_t mine = ArgMax(lg.data(), vocab);
        if (mine == c.cont[i]) {
          ++agree;
        } else {
          const double gap = lg[static_cast<size_t>(mine)] - lg[static_cast<size_t>(c.cont[i])];
          std::printf("    step %zu: ours %lld, HF %d: our own logit gap %.3f\n", i, static_cast<long long>(mine), c.cont[i], gap);
          (gap <= 0.35 ? ties : bad) += 1;
        }
      }
      std::printf("  teacher-forced continuation: %d of %zu agree, %d near-ties, %d real disagreements\n", agree, c.cont.size(), ties, bad);
      Check(bad == 0, tag + ": the continuation after the image disagrees with HF beyond near-ties");
    }    // 3. the prompt in two calls: text first (a plain prefill), then the rest with image offsets relative to the call
    {
      size_t first_img = 0;
      while (c.ids[first_img] != c.image_token) ++first_img;
      const size_t cut = first_img >= 2 ? first_img - 2 : 0;  // inside the text before the first image
      if (cut > 0) {
        model->Reset();
        const std::vector<int32_t> head(c.ids.begin(), c.ids.begin() + static_cast<ptrdiff_t>(cut));
        const std::vector<int32_t> tail(c.ids.begin() + static_cast<ptrdiff_t>(cut), c.ids.end());
        (void)model->PrefillMultimodal(head, {});
        const auto spans = SpansOf(c, gold_rows, hidden, cut, c.ids.size(), 0);
        const std::vector<float> logits = model->PrefillMultimodal(tail, spans);
        const int64_t mine = ArgMax(logits.data(), vocab);
        std::printf("  two calls (cut %zu): argmax ours %lld golden %lld, delta %lld\n", cut, static_cast<long long>(mine),
                    static_cast<long long>(LastArgmaxGolden(c)), static_cast<long long>(model->MropeDelta()));
        Check(mine == LastArgmaxGolden(c), tag + ": two-call prefill's last-row argmax differs from the golden");
        Check(model->MropeDelta() == c.delta, tag + ": two-call prefill's mrope delta differs");
      }
    }
    // 3b. two images in two calls (the second block continues the first block's mrope state)
    if (c.grids.size() == 2) {
      size_t second = 0, seen = 0;
      for (size_t i = 0; i < c.ids.size(); ++i) {
        if (c.ids[i] == c.image_token && (i == 0 || c.ids[i - 1] != c.image_token) && ++seen == 2) second = i;
      }
      const size_t cut = second - 3;  // after the first image's end marker, before the second image's start marker
      model->Reset();
      const std::vector<int32_t> head(c.ids.begin(), c.ids.begin() + static_cast<ptrdiff_t>(cut));
      const std::vector<int32_t> tail(c.ids.begin() + static_cast<ptrdiff_t>(cut), c.ids.end());
      (void)model->PrefillMultimodal(head, SpansOf(c, gold_rows, hidden, 0, cut, 0));
      const std::vector<float> logits = model->PrefillMultimodal(tail, SpansOf(c, gold_rows, hidden, cut, c.ids.size(), 1));
      const int64_t mine = ArgMax(logits.data(), vocab);
      std::printf("  image per call (cut %zu): argmax ours %lld golden %lld, delta %lld golden %lld\n", cut, static_cast<long long>(mine),
                  static_cast<long long>(LastArgmaxGolden(c)), static_cast<long long>(model->MropeDelta()), static_cast<long long>(c.delta));
      Check(mine == LastArgmaxGolden(c), tag + ": image-per-call prefill's last-row argmax differs from the golden");
      Check(model->MropeDelta() == c.delta, tag + ": image-per-call prefill's mrope delta differs");
    }
  }

  // 4. text only: bit-identical with and without a tower
  {
    auto plain = ([&] {
      Qwen35MoeModelOptions t;
      t.container_path = tcont;
      t.max_ctx = 4096;
      t.devices = {dev, dev};
      t.vision = 0;
      return Qwen35MoeModel::Load(t);
    })();
    Check(!plain->HasVision(), "vision=0 still loaded a tower");
    std::vector<int32_t> ids;
    for (int i = 0; i < 150; ++i) ids.push_back((i * 37 + 11) % 4000);
    model->Reset();
    const std::vector<float> a = model->PrefillMultimodal(ids, {});
    const std::vector<float> b = plain->Prefill(ids);
    Check(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "text-only prefill logits are not bit-identical with a tower loaded");
    int32_t ta = static_cast<int32_t>(ArgMax(a.data(), vocab)), tb = ta;
    bool same = true;
    for (int i = 0; i < 8; ++i) {
      ta = model->DecodeStepGreedy(ta);
      tb = plain->DecodeStepGreedy(tb);
      same &= ta == tb;
    }
    Check(same, "text-only greedy decode differs with a tower loaded");
    std::printf("== text only: prefill logits %s, 8 decode steps %s\n", a == b ? "bit-identical" : "DIFFER", same ? "identical" : "DIFFER");
  }

  std::printf("%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
