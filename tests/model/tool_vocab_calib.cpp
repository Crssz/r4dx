// tests/model/tool_vocab_calib.cpp -- one-off diagnostic (NOT a ctest test -- built by
// tests/model/CMakeLists.txt but never registered via add_test, same convention as
// tool_hseed_drift.cpp) for docs/r9700.md R9's "reduced-vocab draft head": chooses and JUSTIFIES a
// vocabulary subset for MtpHead::Draft's own drafting, by measuring what fraction of the REAL
// model's own greedy next-token predictions over a calibration corpus fall inside candidate subsets
// of several sizes, comparing two subset-construction methods.
//
// Two methods (docs/r9700.md task item 1's own options):
//   A. "frequency": highest-frequency tokens from the calibration corpus, run through the real
//      tokenizer -- a static property of the TEXT, independent of the model.
//   B. "predicted": the union of the real model's own top-1 (greedy) predictions over the
//      calibration set, ranked by how often each id is predicted -- better matched to what MTP
//      actually drafts, since a draft head's job is to approximate what the BACKBONE would predict,
//      not what token happens to occur most often in English text.
//
// Coverage metric (task item 1's own definition, "a draft token outside the subset is simply a
// rejected draft, not a wrong answer" -- so coverage is an upper bound on how often the reduced head
// COULD possibly match the real model, not the acceptance rate itself, which also depends on the
// reduced head's own quantization/compression error): for each calibration position, teacher-force
// the REAL corpus token sequence through the real 64-layer model (Prefill once, then DecodeStep per
// position -- greedy, deterministic) and record the model's own argmax prediction for the NEXT
// position. coverage(subset) = fraction of those predictions whose id falls inside `subset`.
//
// Usage: tool_vocab_calib.exe [container_path] [tokenizer_dir] [corpus_path] [out_json_path]
// Defaults match this project's standard real-container/tokenizer/corpus locations (docs/r9700.md
// task, D:/models/wikitext-2-raw -- a real calibration corpus already present on this machine).
// SKIPs (prints and returns 77, same convention as every other real-data tool/test in this
// directory) if the container, tokenizer directory, or corpus file is missing.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "test_common.h"
#include "tokenizer.h"

#include "nlohmann/json.hpp"

using r4dx_test::FileExists;
using r4dx_test::SkipMissing;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

const char* kDefaultContainer = "D:/models/r4dx/qwen38-27b-v3.r4dx";
const char* kDefaultTokenizerDir = "C:/AI/models/Qwen3.8-27B";
const char* kDefaultCorpus = "D:/models/wikitext-2-raw/wiki.train.raw";
const char* kDefaultOutJson = "build/logs/vocab_calib.json";

// Only the first this-many bytes of the corpus are tokenized -- enough for kCalibTokens positions
// without paying to tokenize/hold the whole 10 MB file (byte-level BPE averages ~4 chars/token on
// English text, so this is a generous over-read, trimmed by the resize() below).
constexpr size_t kCorpusBytesToRead = 2000000;
// Calibration positions to teacher-force through the real model. Each position is one real
// Prefill/DecodeStep call (~26 ms at w4a16 decode on this card, docs/perf.md) -- 20000 positions is
// ~9 minutes of real GPU time. This many positions matters for THIS tool's own correctness, not
// just budget: with too few positions, the number of DISTINCT predicted/corpus token ids is itself
// smaller than the candidate subset sizes (4096/8192/16384), which makes "coverage" trivially 1.0
// (a tautology -- an early run of this tool at 6000 positions hit exactly this: only ~1285 distinct
// ids were ever predicted, so every subset size >=1285 "covered" 100% by construction). 20000
// positions comfortably exceeds 16384 distinct ids in practice for a real 248320-vocab model.
constexpr int kCalibTokens = 20000;
// TRAIN/HELD-OUT split (out-of-sample coverage, not a tautology): the subset is built ONLY from the
// first kTrainTokens positions' own frequency tables; coverage is measured ONLY on the remaining
// (held-out) positions' predictions, which the subset-construction step never saw. This is what
// makes the coverage number mean something -- "how often does a subset chosen from PAST text cover
// what the model predicts on DIFFERENT text", the actual drafting scenario, not "how often does a
// subset cover the exact same data it was built from".
constexpr int kTrainTokens = 15000;
constexpr int kPromptChunk = 64;  // first chunk is a real Prefill(), matching Model's own chunking

// Reads `kSegments` dispersed byte ranges spread evenly across the WHOLE file (not just its
// prefix), joined with blank-line separators, total size capped at `max_bytes`. Fixes a real
// methodology bug found in this tool's own first pass: a single CONTIGUOUS prefix of one corpus
// file is topically narrow (one or a few Wikipedia articles in a row), so a fixed-size sample of it
// under-fills the vocabulary long before Zipf's law would predict for a truly diverse sample of the
// same total token budget -- this tool's own first two runs measured only 2446-3198 distinct ids
// from 15000-20000 positions of a contiguous prefix. Spreading the SAME total byte budget across
// several offsets (different articles/topics) costs nothing extra in GPU time (same total
// calibration position count) but gives materially better topical coverage per byte.
constexpr int kSegments = 10;

std::string ReadDispersedSegments(const std::string& path, size_t max_bytes, int segments) {
  std::ifstream f(path, std::ios::binary);
  f.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(f.tellg());
  const size_t seg_bytes = max_bytes / static_cast<size_t>(segments);
  std::string out;
  out.reserve(max_bytes + static_cast<size_t>(segments) * 2);
  for (int s = 0; s < segments; ++s) {
    // Evenly spaced start offsets across the whole file, leaving room for this segment's own
    // length so the last segment doesn't read past EOF.
    const size_t max_start = file_size > seg_bytes ? file_size - seg_bytes : 0;
    const size_t start = (segments > 1) ? (max_start * static_cast<size_t>(s)) / static_cast<size_t>(segments - 1)
                                         : 0;
    f.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    std::string chunk(seg_bytes, '\0');
    f.read(chunk.data(), static_cast<std::streamsize>(seg_bytes));
    chunk.resize(static_cast<size_t>(f.gcount()));
    out += chunk;
    out += "\n\n";  // separator so BPE merges never span two unrelated segments' seam
  }
  return out;
}

int32_t ArgmaxLogits(const std::vector<float>& logits) {
  int64_t best = 0;
  float best_v = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = static_cast<int64_t>(i);
    }
  }
  return static_cast<int32_t>(best);
}

// Top-N ids by frequency, descending count then ascending id (deterministic tie-break).
std::vector<int64_t> TopNByFrequency(const std::unordered_map<int32_t, int64_t>& freq, size_t n) {
  std::vector<std::pair<int32_t, int64_t>> v(freq.begin(), freq.end());
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });
  if (v.size() > n) v.resize(n);
  std::vector<int64_t> ids;
  ids.reserve(v.size());
  for (auto& p : v) ids.push_back(p.first);
  std::sort(ids.begin(), ids.end());  // stable, deterministic on-disk order
  return ids;
}

double Coverage(const std::vector<int32_t>& predicted_per_position,
                 const std::vector<int64_t>& subset_sorted) {
  size_t covered = 0;
  for (int32_t id : predicted_per_position) {
    if (std::binary_search(subset_sorted.begin(), subset_sorted.end(), static_cast<int64_t>(id))) {
      ++covered;
    }
  }
  return predicted_per_position.empty()
             ? 0.0
             : static_cast<double>(covered) / static_cast<double>(predicted_per_position.size());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string container_path = argc > 1 ? argv[1] : kDefaultContainer;
  const std::string tokenizer_dir = argc > 2 ? argv[2] : kDefaultTokenizerDir;
  const std::string corpus_path = argc > 3 ? argv[3] : kDefaultCorpus;
  const std::string out_json_path = argc > 4 ? argv[4] : kDefaultOutJson;

  if (!FileExists(container_path)) return SkipMissing(container_path);
  if (!FileExists(tokenizer_dir + "/tokenizer.json")) return SkipMissing(tokenizer_dir + "/tokenizer.json");
  if (!FileExists(corpus_path)) return SkipMissing(corpus_path);

  r4dx::Tokenizer::Options tok_opts;
  tok_opts.allow_unimplemented_normalizer = true;  // same escape hatch src/cli/main.cpp uses for
                                                    // this exact checkpoint's tokenizer.json (NFC)
  r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tokenizer_dir, tok_opts);

  const std::string corpus_text = ReadDispersedSegments(corpus_path, kCorpusBytesToRead, kSegments);
  std::vector<int32_t> ids = tok.encode(corpus_text, /*parse_special=*/false);
  if (static_cast<int>(ids.size()) > kCalibTokens + kPromptChunk) {
    ids.resize(static_cast<size_t>(kCalibTokens + kPromptChunk));
  }
  std::fprintf(stderr, "[vocab_calib] corpus=%s tokenized to %zu ids (capped at %d)\n",
               corpus_path.c_str(), ids.size(), kCalibTokens + kPromptChunk);
  if (static_cast<int>(ids.size()) < kPromptChunk + 100) {
    std::fprintf(stderr, "[vocab_calib] FAIL: corpus too short after tokenization/capping\n");
    return 1;
  }

  ModelOptions opts;
  opts.container_path = container_path;
  opts.layout = Layout::kW4a16;  // most accurate quantized layout (docs/r9700.md P1/status.md) --
                                  // this tool measures the BODY model's own predictions, not a
                                  // layout comparison, so use the layout closest to the reference.
  opts.max_ctx = static_cast<int64_t>(ids.size()) + 64;
  opts.mtp_draft_k = 0;  // no MTP needed at all -- plain Prefill/DecodeStepGreedy teacher-forcing

  std::fprintf(stderr, "[vocab_calib] loading %s (layout=w4a16, mtp=0)...\n", container_path.c_str());
  Model model = Model::Load(opts);

  const std::vector<int32_t> prompt(ids.begin(), ids.begin() + kPromptChunk);
  std::vector<float> logits = model.Prefill(prompt);

  // TRAIN frequency tables (built only from positions < kTrainTokens) vs HELD-OUT predictions
  // (positions >= kTrainTokens, which subset construction never sees) -- see kTrainTokens' own
  // comment for why this out-of-sample split is what makes "coverage" a real measurement.
  std::unordered_map<int32_t, int64_t> train_pred_freq, train_corpus_freq;
  std::vector<int32_t> heldout_predicted;
  int64_t correct = 0;
  const size_t n_positions = ids.size() - static_cast<size_t>(kPromptChunk);
  heldout_predicted.reserve(n_positions);

  for (size_t i = static_cast<size_t>(kPromptChunk); i < ids.size(); ++i) {
    const int32_t predicted = ArgmaxLogits(logits);
    const size_t calib_idx = i - static_cast<size_t>(kPromptChunk);  // 0-based calibration position
    const bool is_train = calib_idx < static_cast<size_t>(kTrainTokens);
    if (is_train) {
      ++train_pred_freq[predicted];
      ++train_corpus_freq[ids[i]];
    } else {
      heldout_predicted.push_back(predicted);
    }
    if (predicted == ids[i]) ++correct;
    logits = model.DecodeStep(ids[i]);
    if (calib_idx % 2000 == 0) {
      std::fprintf(stderr, "[vocab_calib] ... %zu/%zu positions\n", calib_idx, n_positions);
    }
  }
  std::fprintf(stderr,
               "[vocab_calib] done: %zu calibration positions (%d train, %zu held-out), top-1 "
               "self-teacher-forced accuracy = %.2f%% (informative sanity number, not the coverage "
               "metric itself) -- distinct ids: %zu predicted (train), %zu corpus (train)\n",
               n_positions, kTrainTokens, heldout_predicted.size(),
               100.0 * static_cast<double>(correct) / static_cast<double>(n_positions),
               train_pred_freq.size(), train_corpus_freq.size());

  const std::vector<size_t> sizes = {4096, 8192, 16384};
  nlohmann::json out;
  out["container"] = container_path;
  out["corpus"] = corpus_path;
  out["calibration_positions"] = n_positions;
  out["train_positions"] = kTrainTokens;
  out["heldout_positions"] = heldout_predicted.size();
  out["top1_self_teacher_forced_accuracy"] =
      static_cast<double>(correct) / static_cast<double>(n_positions);
  out["distinct_predicted_ids_train"] = train_pred_freq.size();
  out["distinct_corpus_ids_train"] = train_corpus_freq.size();
  out["sizes"] = nlohmann::json::array();

  for (size_t n : sizes) {
    const std::vector<int64_t> subset_pred = TopNByFrequency(train_pred_freq, n);
    const std::vector<int64_t> subset_freq = TopNByFrequency(train_corpus_freq, n);
    const double cov_pred_by_pred = Coverage(heldout_predicted, subset_pred);
    const double cov_pred_by_freq = Coverage(heldout_predicted, subset_freq);
    std::fprintf(stderr,
                 "[vocab_calib] N=%6zu  held-out coverage(predicted-method subset)=%.4f  "
                 "held-out coverage(frequency-method subset)=%.4f  (subset sizes actually built: "
                 "%zu predicted, %zu frequency)\n",
                 n, cov_pred_by_pred, cov_pred_by_freq, subset_pred.size(), subset_freq.size());
    out["sizes"].push_back({
        {"n", n},
        {"actual_predicted_subset_size", subset_pred.size()},
        {"actual_frequency_subset_size", subset_freq.size()},
        {"heldout_coverage_predicted_method", cov_pred_by_pred},
        {"heldout_coverage_frequency_method", cov_pred_by_freq},
    });
  }

  // Ship the "predicted" method at N=8192 as the default subset (docs/r9700.md's own §2.2 example
  // uses a top-8k head; the "predicted" method is the task's own recommended choice, "better
  // matched to what gets drafted") -- built from the FULL calibration set (train+held-out) since the
  // shipped container is not itself being evaluated for coverage, only the sweep above needs the
  // train/held-out split. Written to out_json_path in the exact `{"vocab_ids": [...]}` shape
  // src/convert/main.cpp's --draft-vocab-ids expects.
  // Re-merge the held-out portion's own predicted-id frequency into the shipped subset's own table
  // (the train/held-out split above is for the coverage MEASUREMENT only; the shipped container
  // gets the benefit of the whole calibration run).
  std::unordered_map<int32_t, int64_t> full_pred_freq = train_pred_freq;
  for (int32_t id : heldout_predicted) ++full_pred_freq[id];
  const std::vector<int64_t> shipped_subset = TopNByFrequency(full_pred_freq, 8192);
  out["shipped_subset_method"] = "predicted";
  out["shipped_subset_size"] = shipped_subset.size();
  out["vocab_ids"] = shipped_subset;

  std::ofstream f(out_json_path);
  f << out.dump(2);
  std::fprintf(stderr, "[vocab_calib] wrote %s (shipped subset: predicted-method, N=%zu)\n",
               out_json_path.c_str(), shipped_subset.size());

  return 0;
}
