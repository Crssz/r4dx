// tests/model/teacher_forced.h -- the teacher-forced per-position log-prob pass shared by
// tests/model/tool_teacher_forced_logprobs.cpp (the Rung 4 dump tool, docs/validation.md) and
// tests/model/test_teacher_forced_logprobs.cpp (the 4-layer self-consistency ctest). Both need the
// SAME pass, byte for byte -- a tool whose numbers a test never checks, and a test that checks a
// second implementation of the tool's math, are each worth very little -- so the pass lives here
// once and both TUs call it.
//
// WHAT THE PASS IS. Given a fixed token sequence `ids` of length T, produce
// `rows = T-1` vectors of per-position next-token log-probabilities:
//
//     row i = log_softmax(logits at position i) = log p(next token | ids[0..i]),  i in [0, T-2]
//
// The sequence is FIXED: nothing is ever sampled, and `ids[T-1]` is deliberately never fed (there
// is no row that would read its logits). The engine calls used are exactly the ones a real
// generation makes for the positions it generates:
//
//     logits = TextModel::Prefill({ids[0]})      -> row 0
//     logits = TextModel::DecodeStep(ids[i])     -> row i,  i = 1 .. T-2
//
// through r4dx::model::TextModel (docs/tp.md 2.8): a LocalTextModel -- today's Model, call for
// call -- at TP=1, or a TpModel under `--tp 2` (each rank computes its vocab shard and the full row
// comes back gathered, docs/tp.md 7.5).
//
// i.e. the same GDN recurrent-update decode kernels, the same paged fp8 KV cache with its
// calibrated descales, and the same fused quant epilogues the engine uses when it decodes. It is
// NOT the chunked-prefill path: a real generation runs its PROMPT through the chunked-scan GDN
// kernels (Model::Prefill with more than one token) and only its GENERATED positions through the
// per-token path this pass uses. Feeding the whole sequence one token at a time is what makes every
// row of the dump comparable to every other row (and to the reference's own uniform pass); the
// price is that row `p-1` of a "prompt of p tokens + generated tail" sequence is computed here by
// the decode path where the generation computed it by the prefill path. The two agree to GPU
// reduction-order noise, not bit for bit -- see CheckGreedyConsistency's own note.
//
// NUMERICS. `logits` come back from the engine as fp32. The row reduction is:
//   m   = max_j logits[j]                              (fp32 values, exact comparison)
//   S   = sum_j exp(logits[j] - m)                     (accumulated in double -- see below)
//   lse = m + log(S)                                   (rounded to fp32)
//   lp[j] = logits[j] - lse                            (fp32 subtraction)
// then `lp[j] = max(lp[j], -1e4f)` and finally the fp16 (IEEE binary16, round-to-nearest-even)
// cast. The double accumulator is not a deviation from "computed in fp32": a naive fp32 sum over
// 248320 terms loses several digits to cancellation, while torch's own fp32 log_softmax uses a
// blocked/pairwise reduction that is far closer to the exactly-rounded answer. Accumulating in
// double and rounding the RESULT to fp32 is the cheapest way to land on the same fp32 value the
// reference does, rather than on a different one for a reason that has nothing to do with the
// model. The -1e4 clamp exists because fp16 cannot represent a log-prob below about -65504 (and
// loses all resolution long before that); every token it touches has probability below e^-10000,
// i.e. exactly zero in any arithmetic either side can do, so no KL sum moves measurably.
//
// FILE FORMAT (the shared format both halves of the Rung 4 KL comparison must produce -- see
// docs/validation.md "Rung 4 tooling"):
//   <out-dir>/<segment>.logprobs.f16   raw little-endian float16, row-major [T-1, V], no header
//   <out-dir>/<segment>.meta.json      {"T","V","dtype","rows","source","layout",
//                                       "sha256_of_token_ids_json", ...}
// `sha256_of_token_ids_json` is the SHA-256, lowercase hex, of the COMPACT JSON serialization of
// the segment's token id array -- `[1,2,3]`, UTF-8, no spaces, no trailing newline, exactly what
// Python's `json.dumps(ids, separators=(",", ":")).encode("utf-8")` produces. It is what lets the
// KL report prove the r4dx half and the reference half scored the same tokens.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "text_model.h"  // also pulls in nlohmann/json.hpp via model_config.h

namespace r4dx_tf {

// ---- SHA-256 (FIPS 180-4) ----------------------------------------------------------------------
// Self-contained and ~60 lines; the repo has no crypto dependency and this is the only hash any
// r4dx C++ code needs. Only used for the provenance field above, never for anything security
// relevant.
class Sha256 {
 public:
  Sha256() { Reset(); }

  void Update(const uint8_t* data, size_t len) {
    Absorb(data, len);
    bits_ += static_cast<uint64_t>(len) * 8u;
  }
  void Update(const std::string& s) {
    Update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  }

  std::string HexDigest() {
    // FIPS 180-4 padding: 0x80, then zeros up to a 56-byte boundary, then the ORIGINAL message
    // length in bits, big-endian. Absorb() (not Update()) so the padding never counts toward it.
    const uint64_t bits = bits_;
    const uint8_t pad = 0x80;
    Absorb(&pad, 1);
    const uint8_t zero = 0x00;
    while (buf_len_ != 56) Absorb(&zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[i] = static_cast<uint8_t>((bits >> (56 - 8 * i)) & 0xffu);
    Absorb(len_be, 8);

    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; ++i) {
      for (int b = 3; b >= 0; --b) {
        const uint8_t byte = static_cast<uint8_t>((h_[i] >> (8 * b)) & 0xffu);
        out.push_back(kHex[byte >> 4]);
        out.push_back(kHex[byte & 0x0fu]);
      }
    }
    return out;
  }

 private:
  void Reset() {
    static const uint32_t kInit[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::memcpy(h_, kInit, sizeof(h_));
    buf_len_ = 0;
    bits_ = 0;
  }
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  // Feeds bytes through the block function WITHOUT touching the message-length counter.
  void Absorb(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buf_[buf_len_++] = data[i];
      if (buf_len_ == 64) {
        Block(buf_);
        buf_len_ = 0;
      }
    }
  }

  void Block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(p[4 * i]) << 24) | (static_cast<uint32_t>(p[4 * i + 1]) << 16) |
             (static_cast<uint32_t>(p[4 * i + 2]) << 8) | static_cast<uint32_t>(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }

  uint32_t h_[8];
  uint8_t buf_[64];
  size_t buf_len_ = 0;
  uint64_t bits_ = 0;
};

// The canonical `sha256_of_token_ids_json` of a segment: SHA-256 of `[1,2,3]`-style compact JSON,
// UTF-8, no spaces, no trailing newline. Matches Python's
// `hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()`.
inline std::string TokenIdsSha256(const std::vector<int32_t>& ids) {
  std::string s = "[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) s.push_back(',');
    s += std::to_string(ids[i]);
  }
  s.push_back(']');
  Sha256 h;
  h.Update(s);
  return h.HexDigest();
}

// ---- tokens.json (the shared input format) -----------------------------------------------------
struct Segment {
  std::string name;
  std::vector<int32_t> token_ids;
};
struct TokensFile {
  std::string tokenizer;
  std::vector<Segment> segments;
};

inline TokensFile ReadTokensJson(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open tokens file: " + path);
  nlohmann::json j;
  f >> j;
  TokensFile out;
  out.tokenizer = j.value("tokenizer", std::string());
  if (!j.contains("segments") || !j["segments"].is_array()) {
    throw std::runtime_error(path + ": missing \"segments\" array");
  }
  for (const auto& s : j["segments"]) {
    Segment seg;
    seg.name = s.at("name").get<std::string>();
    for (const auto& t : s.at("token_ids")) seg.token_ids.push_back(t.get<int32_t>());
    if (seg.token_ids.size() < 2) {
      throw std::runtime_error(path + ": segment \"" + seg.name +
                                "\" has fewer than 2 token ids (a teacher-forced pass over T tokens "
                                "produces T-1 rows, so T must be at least 2)");
    }
    out.segments.push_back(std::move(seg));
  }
  if (out.segments.empty()) throw std::runtime_error(path + ": \"segments\" is empty");
  return out;
}

// ---- the pass ----------------------------------------------------------------------------------
struct SegmentResult {
  int64_t T = 0;
  int64_t V = 0;
  int64_t rows = 0;
  double wall_s = 0.0;
  // max over rows of |log sum_j exp(row[j])| -- 0 for an exact log_softmax, so this is the pass's
  // own arithmetic self-check (the gate's "logsumexp(row) within 1e-2 of 0" assertion).
  double max_abs_lse = 0.0;
  // Greedy consistency (see CheckGreedyConsistency below): how many rows were checked and how many
  // disagreed with the token that actually follows them.
  int64_t greedy_checked = 0;
  int64_t greedy_mismatches = 0;
  int64_t first_mismatch_row = -1;
  int32_t first_mismatch_got = -1;
  int32_t first_mismatch_want = -1;
  std::string sha256;
  // Per-row fingerprints of the RAW engine logits, one entry per row, always filled (they cost a
  // vector push each -- the two reductions they report were computed anyway). `argmax` is the
  // greedy token at that row; `lse` is logsumexp over the whole raw row, i.e. a single scalar that
  // depends on all V logits. A caller that ran the same positions through the same engine calls by
  // some other route can compare these row for row: they are the cheapest evidence that the dump's
  // row i really is the model's prediction at position i (and not at i-1 or i+1), which is what
  // test_teacher_forced_logprobs cross-checks against its own decode loop.
  std::vector<int32_t> argmax;
  std::vector<double> lse;
};

struct SegmentOptions {
  // Empty => compute everything but write no files (what the ctest wants).
  std::string out_dir;
  // Written into the sidecar's "layout" field.
  std::string layout_name = "w4a16";
  // The last N tokens of the sequence were produced by GREEDY decoding on this same model/container
  // /layout, so for every row that predicts one of them, argmax(row) must equal that token. 0
  // disables the check. See the caveat in this header's top comment about the single row that sits
  // on the prompt/generation boundary.
  int64_t check_greedy_last_n = 0;
  // Print a progress line every this many rows (0 = silent).
  int64_t progress_every = 64;
  // Extra provenance copied verbatim into the sidecar.
  std::string container_path;
};

// Runs the whole pass for one segment on `model`, which it Reset()s first so the segment is always
// evaluated from a fresh context at position 0.
inline SegmentResult RunSegment(r4dx::model::TextModel& model, const Segment& seg,
                                 const SegmentOptions& opts) {
  using Clock = std::chrono::steady_clock;
  SegmentResult r;
  r.T = static_cast<int64_t>(seg.token_ids.size());
  r.V = model.Config().vocab_size;
  r.rows = r.T - 1;
  r.sha256 = TokenIdsSha256(seg.token_ids);
  if (r.T < 2) throw std::runtime_error("RunSegment: segment '" + seg.name + "' needs T >= 2");

  std::FILE* out = nullptr;
  std::string bin_path;
  if (!opts.out_dir.empty()) {
    bin_path = opts.out_dir + "/" + seg.name + ".logprobs.f16";
    out = std::fopen(bin_path.c_str(), "wb");
    if (out == nullptr) throw std::runtime_error("cannot write " + bin_path);
  }

  // Rows [first_greedy_row, rows-1] are the ones whose successor token was greedily generated.
  const int64_t first_greedy_row =
      opts.check_greedy_last_n > 0
          ? std::max<int64_t>(0, r.rows - opts.check_greedy_last_n)
          : r.rows;  // == no row qualifies

  std::vector<float> lp(static_cast<size_t>(r.V));
  std::vector<uint16_t> half(static_cast<size_t>(r.V));
  r.argmax.reserve(static_cast<size_t>(r.rows));
  r.lse.reserve(static_cast<size_t>(r.rows));

  const auto t0 = Clock::now();
  model.Reset();
  std::vector<float> logits = model.Prefill({seg.token_ids[0]});

  for (int64_t i = 0; i < r.rows; ++i) {
    if (static_cast<int64_t>(logits.size()) != r.V) {
      if (out) std::fclose(out);
      throw std::runtime_error("RunSegment: engine returned " + std::to_string(logits.size()) +
                                " logits, expected vocab_size=" + std::to_string(r.V));
    }
    // log_softmax in fp32 (double accumulator for the exp sum -- see this header's NUMERICS note).
    float m = logits[0];
    int64_t argmax = 0;
    for (int64_t j = 1; j < r.V; ++j) {
      if (logits[static_cast<size_t>(j)] > m) {
        m = logits[static_cast<size_t>(j)];
        argmax = j;
      }
    }
    double sum = 0.0;
    for (int64_t j = 0; j < r.V; ++j) {
      sum += std::exp(static_cast<double>(logits[static_cast<size_t>(j)]) - static_cast<double>(m));
    }
    const float lse = static_cast<float>(static_cast<double>(m) + std::log(sum));
    for (int64_t j = 0; j < r.V; ++j) lp[static_cast<size_t>(j)] = logits[static_cast<size_t>(j)] - lse;
    r.argmax.push_back(static_cast<int32_t>(argmax));
    r.lse.push_back(static_cast<double>(lse));

    // The pass's own arithmetic check: logsumexp of the RESULT must be 0.
    double check_max = lp[0];
    for (int64_t j = 1; j < r.V; ++j) check_max = std::max(check_max, static_cast<double>(lp[static_cast<size_t>(j)]));
    double check_sum = 0.0;
    for (int64_t j = 0; j < r.V; ++j) {
      check_sum += std::exp(static_cast<double>(lp[static_cast<size_t>(j)]) - check_max);
    }
    r.max_abs_lse = std::max(r.max_abs_lse, std::fabs(check_max + std::log(check_sum)));

    if (i >= first_greedy_row) {
      ++r.greedy_checked;
      const int32_t want = seg.token_ids[static_cast<size_t>(i + 1)];
      if (static_cast<int32_t>(argmax) != want) {
        ++r.greedy_mismatches;
        if (r.first_mismatch_row < 0) {
          r.first_mismatch_row = i;
          r.first_mismatch_got = static_cast<int32_t>(argmax);
          r.first_mismatch_want = want;
        }
      }
    }

    if (out != nullptr) {
      for (int64_t j = 0; j < r.V; ++j) {
        float v = lp[static_cast<size_t>(j)];
        if (!(v > -1e4f)) v = -1e4f;  // also maps a NaN to the floor rather than writing a NaN
        half[static_cast<size_t>(j)] = r4dx::core::FloatToF16(v);
      }
      if (std::fwrite(half.data(), sizeof(uint16_t), static_cast<size_t>(r.V), out) !=
          static_cast<size_t>(r.V)) {
        std::fclose(out);
        throw std::runtime_error("short write to " + bin_path);
      }
    }

    if (opts.progress_every > 0 && ((i + 1) % opts.progress_every == 0 || i + 1 == r.rows)) {
      const double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::printf("  [%s] row %lld/%lld  %.1fs  (%.1f ms/row)\n", seg.name.c_str(),
                  static_cast<long long>(i + 1), static_cast<long long>(r.rows), el,
                  1000.0 * el / static_cast<double>(i + 1));
      std::fflush(stdout);
    }

    // Feed the next token. ids[T-1] is deliberately never fed: no row would read its logits.
    if (i + 1 < r.rows) logits = model.DecodeStep(seg.token_ids[static_cast<size_t>(i + 1)]);
  }
  r.wall_s = std::chrono::duration<double>(Clock::now() - t0).count();

  if (out != nullptr) {
    std::fclose(out);
    nlohmann::json meta;
    meta["T"] = r.T;
    meta["V"] = r.V;
    meta["dtype"] = "float16";
    meta["rows"] = r.rows;
    meta["source"] = "r4dx";
    meta["layout"] = opts.layout_name;
    meta["sha256_of_token_ids_json"] = r.sha256;
    // Beyond the shared contract, purely provenance for whoever reads the pair later.
    meta["container"] = opts.container_path;
    meta["clamp_min"] = -1e4;
    meta["engine"] = "r4dx tool_teacher_forced_logprobs";
    meta["path"] = "Prefill(ids[0]) + DecodeStep(ids[1..T-2])";
    if (model.TpWorld() > 1) meta["tp_world"] = model.TpWorld();  // absent at TP=1: same bytes as before
    meta["mtp"] = 0;
    meta["dflash"] = false;
    meta["wall_seconds"] = r.wall_s;
    const std::string meta_path = opts.out_dir + "/" + seg.name + ".meta.json";
    std::ofstream mf(meta_path, std::ios::binary);
    if (!mf) throw std::runtime_error("cannot write " + meta_path);
    mf << meta.dump(2) << "\n";
  }
  return r;
}

}  // namespace r4dx_tf
