// tests/vision/tool_vision_chat.cpp -- the end-to-end "does the model actually SEE the image"
// harness for the text-side splicing stage (docs/vision.md "Text-side splicing").
//
// Built but deliberately NOT registered with add_test(), the same convention tool_vision_bench /
// tests/model's tool_hseed_drift / tests/kernels' tool_sampler_bench follow: it needs the real 27B
// container and real image files, it prints a generated answer for a human to read, and it asserts
// no pass/fail contract of its own. It is the splice/mrope driver: `--show-positions`,
// `--dump-prompt` and the `--turn2` prefix-reuse rehearsal have no equivalent on r4dx-cli's own
// `--image` path, which is what a user actually runs.
//
//   $env:HIP_VISIBLE_DEVICES='1'
//   build\win-hip\tests\vision\tool_vision_chat.exe [--model D:/models/r4dx/qwen38-27b-v6.r4dx]
//       --tokenizer C:/AI/models/Qwen3.8-27B --layout w4a16 --image pic.png
//       --prompt "Describe this image." [--max-tokens 128] [--mtp 3] [--dflash <container> --k 7]
//       [--turn2 "..."] [--show-positions] [--dump-prompt out.json]
//
// What it exercises, beyond "produce text":
//   * the whole splice path -- preprocess -> Model::EncodeImages -> placeholder expansion ->
//     Model::PrefillMultimodal with real ImageSpans;
//   * `--show-positions`, which prints the engine's own 3-axis rope rows for the REAL rendered
//     prompt so they can be diffed against the reference's get_rope_index (see
//     tools/reference/rope_index_golden.py and tests/vision/test_position_ids.cpp);
//   * `--turn2`, which re-renders the conversation with the assistant's own reply appended and
//     prefills ONLY the new tail, i.e. exactly the server's prefix-reuse path across a turn that
//     contains an image;
//   * `--mtp` / `--dflash`, whose draft and verify steps have to rope at `sequence index + delta`
//     while their KV slots stay the sequence index.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "chat_template.h"
#include "image_decode.h"
#include "image_prompt.h"
#include "../model/test_container_path.h"  // r4dx_test::ProductionTargetPath
#include "model.h"
#include "position_ids.h"
#include "preprocess.h"
#include "tokenizer.h"

namespace {

double Seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// The chat template emits ONE `<|image_pad|>` per image; expanding it into `merged_token_count`
// copies before the model ever sees it is `r4dx::vision::ExpandImagePlaceholders`' job
// (src/vision/image_prompt.h) -- the SAME call r4dx-cli's `--image` and the server's `image_url`
// path make, so this driver exercises the shipping routine rather than a lookalike copy of it.
// All that is left here is the field-for-field conversion to Model::ImageSpan that every caller
// which already includes model.h does (image_prompt.h's own header comment explains why
// src/vision cannot name that type itself), plus slicing the one encode output buffer per image.
struct ExpandResult {
  std::vector<int32_t> tokens;
  std::vector<r4dx::model::Model::ImageSpan> spans;
};

ExpandResult ExpandForPrefill(const std::vector<r4dx::TokenId>& raw, int32_t image_token_id,
                              const std::vector<r4dx::vision::GridThw>& grids, int merge_size,
                              const uint16_t* embeds_base, int64_t hidden) {
  std::vector<r4dx::vision::ImagePlaceholderSpan> in;
  int64_t embed_row = 0;
  for (const r4dx::vision::GridThw& g : grids) {
    r4dx::vision::ImagePlaceholderSpan sp;
    sp.grid = g;
    sp.embeds = embeds_base + embed_row * hidden;
    in.push_back(sp);
    embed_row += g.MergedTokenCount(merge_size);
  }
  r4dx::vision::ExpandedImagePrompt ex =
      r4dx::vision::ExpandImagePlaceholders(raw, image_token_id, in, merge_size);
  ExpandResult out;
  out.tokens = std::move(ex.tokens);
  for (const r4dx::vision::ImagePlaceholderSpan& sp : ex.spans) {
    r4dx::model::Model::ImageSpan ms;
    ms.offset = sp.offset;
    ms.tokens = sp.tokens;
    ms.grid = sp.grid;
    ms.embeds = sp.embeds;
    out.spans.push_back(ms);
  }
  return out;
}

void PrintPositions(const std::vector<int32_t>& tokens,
                     const std::vector<r4dx::model::Model::ImageSpan>& spans, int merge_size,
                     int32_t image_token_id) {
  std::vector<uint8_t> mm(tokens.size(), r4dx::vision::kMmTokenTypeText);
  std::vector<r4dx::vision::GridThw> grids;
  for (const auto& sp : spans) {
    for (int64_t i = sp.offset; i < sp.offset + sp.tokens; ++i) {
      mm[static_cast<size_t>(i)] = r4dx::vision::kMmTokenTypeImage;
    }
    grids.push_back(sp.grid);
  }
  const auto mp = r4dx::vision::BuildMropePositionIds(mm, grids, merge_size);
  std::printf("[positions] seq_len=%lld delta=%lld\n", static_cast<long long>(mp.seq_len),
              static_cast<long long>(mp.mrope_position_delta));
  // Print every text position plus the first/last few of each image run -- a 196-row image would
  // otherwise bury the interesting boundaries.
  for (int64_t p = 0; p < mp.seq_len; ++p) {
    bool in_image = mm[static_cast<size_t>(p)] == r4dx::vision::kMmTokenTypeImage;
    if (in_image) {
      int64_t run_start = p, run_end = p;
      while (run_end < mp.seq_len && mm[static_cast<size_t>(run_end)] ==
                                          r4dx::vision::kMmTokenTypeImage) {
        ++run_end;
      }
      for (int64_t q = run_start; q < run_end; ++q) {
        if (q - run_start >= 3 && run_end - q > 3) {
          if (q - run_start == 3) std::printf("  ... %lld image rows ...\n",
                                               static_cast<long long>(run_end - run_start - 6));
          continue;
        }
        std::printf("  %4lld IMG  t=%d h=%d w=%d\n", static_cast<long long>(q), mp.At(0, q),
                    mp.At(1, q), mp.At(2, q));
      }
      p = run_end - 1;
      continue;
    }
    std::printf("  %4lld txt  t=%d h=%d w=%d  tok=%d\n", static_cast<long long>(p), mp.At(0, p),
                mp.At(1, p), mp.At(2, p), tokens[static_cast<size_t>(p)]);
  }
  (void)image_token_id;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = r4dx_test::ProductionTargetPath();  // group-matched v6 (64) / v3 (128)
  std::string tokenizer_dir = "C:/AI/models/Qwen3.8-27B";
  std::string layout = "w4a16";
  std::string prompt = "Describe this image.";
  std::string turn2;
  std::string dflash_container;
  std::string dump_prompt;
  std::vector<std::string> image_paths;
  int64_t max_tokens = 128, max_ctx = 8192, image_max_pixels = 0, mtp_k = 0, dflash_k = 0;
  bool show_positions = false, thinking = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() { return std::string(argv[++i]); };
      if (a == "--model") model_path = next();
      else if (a == "--tokenizer") tokenizer_dir = next();
      else if (a == "--layout") layout = next();
      else if (a == "--image") image_paths.push_back(next());
      else if (a == "--prompt") prompt = next();
      else if (a == "--turn2") turn2 = next();
      else if (a == "--max-tokens") max_tokens = std::stoll(next());
      else if (a == "--max-ctx") max_ctx = std::stoll(next());
      else if (a == "--image-max-pixels") image_max_pixels = std::stoll(next());
      else if (a == "--mtp") mtp_k = std::stoll(next());
      else if (a == "--dflash") dflash_container = next();
      else if (a == "--k") dflash_k = std::stoll(next());
      else if (a == "--show-positions") show_positions = true;
      else if (a == "--dump-prompt") dump_prompt = next();
      else if (a == "--thinking") thinking = true;
      else {
        std::fprintf(stderr, "unrecognized argument: %s\n", a.c_str());
        return 2;
      }
    }

    r4dx::model::ModelOptions opts;
    opts.container_path = model_path;
    opts.layout = r4dx::model::LayoutFromName(layout);
    opts.max_ctx = max_ctx;
    opts.vision = r4dx::model::ModelOptions::VisionMode::kAuto;
    opts.mtp_draft_k = mtp_k;
    if (!dflash_container.empty()) {
      opts.dflash_container = dflash_container;
      opts.dflash_draft_k = dflash_k > 0 ? dflash_k : 7;
    }

    const auto load_t0 = std::chrono::steady_clock::now();
    r4dx::model::Model model = r4dx::model::Model::Load(opts);
    const auto load_t1 = std::chrono::steady_clock::now();
    std::printf("[load] %.2fs, vision=%s\n", Seconds(load_t0, load_t1),
                model.HasVision() ? "on" : "off");

    r4dx::Tokenizer::Options tok_options;
    tok_options.allow_unimplemented_normalizer = true;
    const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tokenizer_dir, tok_options);
    r4dx::ChatTemplate tmpl = r4dx::ChatTemplate::from_directory(tokenizer_dir);

    const int64_t hidden = model.Config().hidden_size;
    const int32_t image_token_id = static_cast<int32_t>(model.GetContainer().ImageTokenId());
    const int merge_size =
        model.HasVision()
            ? static_cast<int>(model.GetContainer().Vision().config.spatial_merge_size)
            : 2;

    // ---- encode every image once; the spans point into this one buffer --------------------------
    r4dx::core::DeviceBuffer<uint16_t> image_embeds;
    std::vector<r4dx::vision::GridThw> grids;
    if (!image_paths.empty()) {
      if (!model.HasVision()) throw std::runtime_error("--image needs a vision-capable container");
      r4dx::vision::ImageProcessorConfig pcfg =
          r4dx::vision::MakeImageProcessorConfig(image_max_pixels);
      std::vector<r4dx::vision::DecodedImage> decoded;
      for (const std::string& p : image_paths) decoded.push_back(r4dx::vision::DecodeImageFile(p));
      const r4dx::vision::PreprocessedImages pre = r4dx::vision::PreprocessImages(decoded, pcfg);
      grids = pre.grid_thw;
      r4dx::vision::VisionEncodeStats stats;
      model.EncodeImages(pre.pixel_values.data(), pre.TotalPatches(), grids, &image_embeds, &stats);
      std::printf("[vision] %zu image(s), %lld patches -> %lld merged tokens, encode %.1f ms\n",
                  image_paths.size(), static_cast<long long>(pre.TotalPatches()),
                  static_cast<long long>(stats.merged_tokens), stats.encode_ms);
      for (size_t i = 0; i < grids.size(); ++i) {
        std::printf("         image %zu: grid %lldx%lld -> %lld merged tokens\n", i,
                    static_cast<long long>(grids[i].h), static_cast<long long>(grids[i].w),
                    static_cast<long long>(grids[i].MergedTokenCount(merge_size)));
      }
    }

    // ---- render the conversation --------------------------------------------------------------
    r4dx::ChatJson messages = r4dx::ChatJson::array();
    r4dx::ChatJson content = r4dx::ChatJson::array();
    for (size_t i = 0; i < image_paths.size(); ++i) {
      content.push_back({{"type", "image"}});
    }
    content.push_back({{"type", "text"}, {"text", prompt}});
    messages.push_back({{"role", "user"}, {"content", content}});
    r4dx::ChatJson extra = r4dx::ChatJson::object();
    extra["enable_thinking"] = thinking;

    auto render_and_expand = [&](const r4dx::ChatJson& msgs) {
      const std::string rendered =
          tmpl.render(msgs, /*add_generation_prompt=*/true, r4dx::ChatJson::array(), extra);
      const std::vector<r4dx::TokenId> raw = tok.encode(rendered, /*parse_special=*/true);
      return ExpandForPrefill(raw, image_token_id, grids, merge_size, image_embeds.data(), hidden);
    };

    ExpandResult ex = render_and_expand(messages);
    std::printf("[prompt] %zu tokens after placeholder expansion, %zu image span(s)\n",
                ex.tokens.size(), ex.spans.size());
    if (show_positions) PrintPositions(ex.tokens, ex.spans, merge_size, image_token_id);

    // ---- generate -------------------------------------------------------------------------------
    // Greedy throughout: an answer that is wrong under greedy decoding is a bug, not a sample.
    const std::vector<r4dx::TokenId>& eos_ids = tok.eos_ids();
    auto is_eos = [&](int32_t id) {
      return std::find(eos_ids.begin(), eos_ids.end(), static_cast<r4dx::TokenId>(id)) !=
             eos_ids.end();
    };
    std::vector<int32_t> engine_rope_rows;
    auto generate = [&](const ExpandResult& block, std::vector<int32_t>* committed_out) {
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<float> logits =
          model.PrefillMultimodal(block.tokens, block.spans, nullptr, &engine_rope_rows);
      const auto t1 = std::chrono::steady_clock::now();
      int32_t next = static_cast<int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      std::vector<int32_t> produced;
      std::printf("[mrope] active=%d delta=%lld\n", model.MropeActive() ? 1 : 0,
                  static_cast<long long>(model.MropeDelta()));
      int64_t rounds = 0, accepted = 0, drafted = 0;
      const auto t2 = std::chrono::steady_clock::now();
      while (static_cast<int64_t>(produced.size()) < max_tokens) {
        produced.push_back(next);
        committed_out->push_back(next);
        if (is_eos(next)) break;
        if (mtp_k > 0) {
          const std::vector<int32_t> round = model.DecodeStepMtpGreedy(next, mtp_k);
          ++rounds;
          drafted += mtp_k;
          accepted += static_cast<int64_t>(round.size()) - 1;
          for (size_t i = 0; i + 1 < round.size(); ++i) {
            produced.push_back(round[i]);
            committed_out->push_back(round[i]);
          }
          next = round.back();
        } else if (!dflash_container.empty()) {
          const std::vector<int32_t> round =
              model.DecodeStepDflashGreedy(next, opts.dflash_draft_k, /*p_min=*/0.0f, /*n_min=*/0);
          ++rounds;
          drafted += opts.dflash_draft_k;
          accepted += static_cast<int64_t>(round.size()) - 1;
          for (size_t i = 0; i + 1 < round.size(); ++i) {
            produced.push_back(round[i]);
            committed_out->push_back(round[i]);
          }
          next = round.back();
        } else {
          next = model.DecodeStepGreedy(next);
        }
      }
      const auto t3 = std::chrono::steady_clock::now();
      std::printf("[stats] prefill %zu tok in %.3fs | decode %zu tok in %.3fs (%.2f tok/s)\n",
                  block.tokens.size(), Seconds(t0, t1), produced.size(), Seconds(t2, t3),
                  produced.size() / std::max(1e-9, Seconds(t2, t3)));
      if (rounds > 0) {
        std::printf("[spec]  %lld rounds, %lld/%lld drafts accepted (%.1f%%), %.2f tokens/round\n",
                    static_cast<long long>(rounds), static_cast<long long>(accepted),
                    static_cast<long long>(drafted),
                    100.0 * static_cast<double>(accepted) / std::max<int64_t>(1, drafted),
                    static_cast<double>(produced.size()) / static_cast<double>(rounds));
      }
      return produced;
    };

    std::vector<int32_t> committed;
    const std::vector<int32_t> answer = generate(ex, &committed);

    // The engine's OWN rope rows for the REAL rendered prompt, dumped for
    // `rope_index_golden.py --verify-prompt` to check against the unmodified reference
    // get_rope_index (docs/vision.md). Hand-written JSON rather than a nlohmann dependency: this
    // TU already sits on the delicate v3.11.3-vs-v3.12.0 include ordering the CMakeLists explains,
    // and three integer arrays do not justify adding another user of it.
    if (!dump_prompt.empty()) {
      std::FILE* f = std::fopen(dump_prompt.c_str(), "wb");
      if (f == nullptr) throw std::runtime_error("cannot write " + dump_prompt);
      auto put_ints = [&](const char* key, const int32_t* v, size_t n, bool last) {
        std::fprintf(f, "  \"%s\": [", key);
        for (size_t i = 0; i < n; ++i) std::fprintf(f, "%s%d", i ? "," : "", v[i]);
        std::fprintf(f, "]%s\n", last ? "" : ",");
      };
      std::vector<int32_t> mm(ex.tokens.size(), 0);
      std::vector<int32_t> grid_flat;
      for (const auto& sp : ex.spans) {
        for (int64_t i = sp.offset; i < sp.offset + sp.tokens; ++i) mm[static_cast<size_t>(i)] = 1;
        grid_flat.push_back(static_cast<int32_t>(sp.grid.t));
        grid_flat.push_back(static_cast<int32_t>(sp.grid.h));
        grid_flat.push_back(static_cast<int32_t>(sp.grid.w));
      }
      std::fprintf(f, "{\n  \"mrope_position_delta\": %lld,\n",
                   static_cast<long long>(model.MropeDelta()));
      std::fprintf(f, "  \"spatial_merge_size\": %d,\n", merge_size);
      put_ints("input_ids", ex.tokens.data(), ex.tokens.size(), false);
      put_ints("mm_token_type_ids", mm.data(), mm.size(), false);
      put_ints("image_grid_thw", grid_flat.data(), grid_flat.size(), false);
      put_ints("engine_position_ids", engine_rope_rows.data(), engine_rope_rows.size(), true);
      std::fprintf(f, "}\n");
      std::fclose(f);
      std::printf("[dump] wrote %s (%zu tokens, %zu rope values)\n", dump_prompt.c_str(),
                  ex.tokens.size(), engine_rope_rows.size());
    }
    const std::string text = tok.decode(answer, /*skip_special_tokens=*/true);
    std::printf("\n=== answer ===\n%s\n==============\n\n", text.c_str());

    // ---- turn 2: prefix reuse across a turn that contained an image -----------------------------
    if (!turn2.empty()) {
      messages.push_back({{"role", "assistant"}, {"content", text}});
      messages.push_back({{"role", "user"}, {"content", turn2}});
      const ExpandResult full = render_and_expand(messages);
      std::vector<int32_t> fed = ex.tokens;
      fed.insert(fed.end(), committed.begin(), committed.end());
      if (full.tokens.size() <= fed.size() ||
          !std::equal(fed.begin(), fed.end(), full.tokens.begin())) {
        std::printf("[turn2] the re-rendered conversation does not extend what was fed "
                    "(%zu fed vs %zu rendered) -- re-prefilling from scratch\n",
                    fed.size(), full.tokens.size());
        model.Reset();
        ExpandResult whole = full;
        std::vector<int32_t> c2;
        const std::vector<int32_t> a2 = generate(whole, &c2);
        std::printf("\n=== turn 2 answer (full re-prefill) ===\n%s\n=======================\n",
                    tok.decode(a2, true).c_str());
      } else {
        // The tail is pure text (the image was in turn 1), so it carries no spans -- which is
        // exactly the continuation case: positions must resume at `sequence index + delta`, not
        // at the sequence index.
        ExpandResult tail;
        tail.tokens.assign(full.tokens.begin() + static_cast<ptrdiff_t>(fed.size()),
                           full.tokens.end());
        std::printf("[turn2] prefix reuse: %zu of %zu tokens already fed, feeding %zu new\n",
                    fed.size(), full.tokens.size(), tail.tokens.size());
        std::vector<int32_t> c2;
        const std::vector<int32_t> a2 = generate(tail, &c2);
        std::printf("\n=== turn 2 answer (prefix reuse) ===\n%s\n=======================\n",
                    tok.decode(a2, true).c_str());
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_vision_chat: %s\n", e.what());
    return 1;
  }
  return 0;
}
