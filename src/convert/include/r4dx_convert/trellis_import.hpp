// r4dx_convert::trellis -- `r4dx-convert --trellis-from <oracle dir>` (docs/trellis-kernel.md
// sections 2 and 3): imports the EXL3/QTIP trellis bits that tools/reference/trellis_quant.py
// (quantize-model or mix; docs/trellis.md) wrote for the decoder linears, moves whole tiles into
// the container's pair grid, and checks the result by a CPU reconstruction of the bytes it wrote.
//
// Nothing here re-encodes (3.1: the shipped bits are the measured bits). The words inside a tile,
// EXL3's position order and the ring encoding are the oracle's, byte for byte; only the tile grid
// changes, from the oracle's [K/16][N/16][8 KB] to the pair grid of 2.1.
//
//   TrellisSource      the run's one import: weights_override.json in both forms (quantize-model:
//                      relative `file`s, stale_layers / layers_done / code_sha256; mix: absolute
//                      `file`s, float K, no stale_layers, the code sha under
//                      allocation.source_code_sha256), every check of 3.3 (steps 1-8 before the
//                      header is written), the regrid (9), the metadata (2.3, 3.3) and the
//                      reconstruction check of the written container (10).
//   PairGridIndex / RegridToPairGrid
//                      the pair grid (2.1). tools/reference/trellis_golden.py's to_pair_grid is the
//                      reference; tests/convert/test_trellis_import.cpp holds the two byte-equal
//                      (tools/reference/trellis_import_golden.py writes the expected bytes).
//   Mul1Codebook / DecodeTile / Fwht128*
//                      the decode (docs/trellis.md 3-5; trellis_quant.py codebook_np,
//                      unpack_states, tensor_core_perm, decode_words) and the natural-order
//                      FWHT-128 the reconstruction W_hat = diag(suh) P_K Q P_N diag(svh) needs.
#pragma once

#include <io.h>     // _commit, _fileno (PatchContainerHeader)
#include <share.h>  // _SH_DENYWR (PatchContainerHeader)

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/reuse_guard.hpp"  // Sha256Files, IsSha256Hex
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/threadpool.hpp"

namespace r4dx_convert {
namespace trellis {

constexpr int kTile = 256;      // 16 x 16 weights per tile, one tail-biting ring
constexpr int kStateBits = 16;  // L
constexpr int kHad = 128;       // the Hadamard block on both sides
constexpr uint32_t kMul1Mult = 0x83DCD12Du;
constexpr uint16_t kMul1KInvF16 = 0x1EEE;   // 0.00676727294921875
constexpr uint16_t kMul1KBiasF16 = 0xC931;  // -10.3828125

inline int WordsPerTile(int kb) { return 8 * kb; }

// ---- the pair grid (docs/trellis-kernel.md 2.1) -------------------------------------------------

// uint32 index of word `w` of tile (tn, tk) -- HF rows 16 tn.., columns 16 tk.. -- in the pair
// grid.
inline uint64_t PairGridIndex(int64_t tn, int64_t tk, int w, int64_t K, int kb) {
  return ((static_cast<uint64_t>(tn >> 1) * static_cast<uint64_t>(K / 16) +
           static_cast<uint64_t>(tk)) *
              2 +
          static_cast<uint64_t>(tn & 1)) *
             static_cast<uint64_t>(WordsPerTile(kb)) +
         static_cast<uint64_t>(w);
}

// One part's words in the ORACLE layout: [K/16][n/16][8 KB] little-endian uint32 (the oracle's I32
// tensor `<hf>.trellis`), `n` rows of the fused linear. Bytes, not uint32_t*: a safetensors mapping
// promises no alignment.
struct OracleWords {
  const uint8_t* data = nullptr;
  int64_t n = 0;
};

// The parts, concatenated along N in order, permuted into the pair grid: `out` holds
// sum(n) * K * kb / 32 words. Whole tiles move; the bits inside a tile do not. A tile pair never
// straddles two parts (every part is a multiple of 32 rows -- checked).
inline void RegridToPairGrid(const std::vector<OracleWords>& parts, int64_t K, int kb,
                             uint32_t* out, int threads) {
  const int nw = WordsPerTile(kb);
  const int64_t tk_count = K / 16;
  std::vector<int64_t> tn0;  // first tile row of each part
  int64_t tn_total = 0;
  for (const auto& p : parts) {
    if (p.n % 32 != 0)
      throw std::runtime_error("RegridToPairGrid: part of " + std::to_string(p.n) +
                               " rows is not a whole number of tile pairs (32 rows)");
    tn0.push_back(tn_total);
    tn_total += p.n / 16;
  }
  ParallelFor(0, tn_total, threads, [&](int64_t lo, int64_t hi) {
    for (int64_t tn = lo; tn < hi; ++tn) {
      size_t pi = parts.size() - 1;
      while (tn < tn0[pi]) --pi;
      const int64_t tn_p = parts[pi].n / 16, tnl = tn - tn0[pi];
      for (int64_t tk = 0; tk < tk_count; ++tk) {
        const uint8_t* src = parts[pi].data + static_cast<uint64_t>(tk * tn_p + tnl) * nw * 4;
        std::memcpy(out + PairGridIndex(tn, tk, 0, K, kb), src, static_cast<size_t>(nw) * 4);
      }
    }
  });
}

// ---- the mul1 codebook and the tile decode (docs/trellis.md 3-5) --------------------------------

// `v` rounded to the nearest fp16 value (ties to even), as a double. `v` finite, |v| < 65504.
inline double RoundToF16Value(double v) {
  if (v == 0.0) return v;
  int e = 0;
  std::frexp(std::fabs(v), &e);           // |v| = m 2^e, m in [0.5, 1): the leading bit is 2^(e-1)
  const int lead = std::max(e - 1, -14);  // below 2^-14 the step is the subnormal 2^-24
  const double ulp = std::ldexp(1.0, lead - 10);
  const double x = v / ulp;  // exact: a power-of-two scale
  double q = std::floor(x);
  const double frac = x - q;
  if (frac > 0.5 || (frac == 0.5 && std::fmod(q, 2.0) != 0.0)) q += 1.0;
  return q * ulp;
}

// The value of every 16-bit state (trellis_quant.py codebook_np("mul1")): x = s * 0x83DCD12D mod
// 2^32; v = fp16((1024 + bytesum(x)) * fp16(0x1EEE) + fp16(0xC931)) with ONE rounding (the double
// product and sum are exact).
inline const std::vector<float>& Mul1Codebook() {
  static const std::vector<float> table = [] {
    const double kinv = r4dx::core::F16ToFloat(kMul1KInvF16);
    const double kbias = r4dx::core::F16ToFloat(kMul1KBiasF16);
    std::vector<float> t(size_t{1} << kStateBits);
    for (uint32_t s = 0; s < t.size(); ++s) {
      const uint32_t x = s * kMul1Mult;
      const uint32_t bsum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
      t[s] = static_cast<float>(RoundToF16Value((1024.0 + bsum) * kinv + kbias));
    }
    return t;
  }();
  return table;
}

// Q::tensor_core_perm: sequence position p = 8 t + j holds row-major tile element perm[p]
// (row = k inside the tile, column = n inside the tile).
inline const std::array<uint8_t, kTile>& TensorCorePerm() {
  static const std::array<uint8_t, kTile> perm = [] {
    std::array<uint8_t, kTile> p{};
    int i = 0;
    for (int t = 0; t < 32; ++t) {
      const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
      const int c0 = t / 4, c1 = c0 + 8;
      for (int c : {c0, c1})
        for (int r : {r0, r1, r2, r3})
          p[static_cast<size_t>(i++)] = static_cast<uint8_t>(r * 16 + c);
    }
    return p;
  }();
  return perm;
}

// Where state(p) sits in the ring (integer rates only: every position owns KB stream bits): the 16
// bits ending at stream bit S(p) = (p + 1) KB, mod the 256 KB-bit ring, read from words i0, i1 at
// offset `off` (trellis_quant.py _ring_tables / unpack_states; word w holds stream bits 32w.. with
// the earliest at bit 31).
struct RingTables {
  int kb = 0;
  int nw = 0;
  std::array<uint16_t, kTile> i0{}, i1{};
  std::array<uint8_t, kTile> off{};
};

inline RingTables MakeRingTables(int kb) {
  if (kb != 4 && kb != 5)
    throw std::runtime_error("trellis: KB=" + std::to_string(kb) + " is not 4 or 5");
  RingTables t;
  t.kb = kb;
  t.nw = WordsPerTile(kb);
  const int64_t R = static_cast<int64_t>(kTile) * kb;
  for (int p = 0; p < kTile; ++p) {
    const int64_t s_end = static_cast<int64_t>(p + 1) * kb;
    const int64_t start = ((s_end - kStateBits) % R + R) % R;
    t.i0[static_cast<size_t>(p)] = static_cast<uint16_t>(start / 32);
    t.i1[static_cast<size_t>(p)] = static_cast<uint16_t>((start / 32 + 1) % t.nw);
    t.off[static_cast<size_t>(p)] = static_cast<uint8_t>(start % 32);
  }
  return t;
}

inline const RingTables& Ring(int kb) {
  static const RingTables r4 = MakeRingTables(4), r5 = MakeRingTables(5);
  if (kb == 4) return r4;
  if (kb == 5) return r5;
  throw std::runtime_error("trellis: KB=" + std::to_string(kb) + " is not 4 or 5");
}

// One tile's 8 KB ring words -> its 256 values, ROW-MAJOR (out[r * 16 + c], r = k, c = n inside the
// tile): trellis_quant.decode_words for one tile, bit for bit (the values are exact fp16 values).
inline void DecodeTile(const uint32_t* w, const RingTables& rt, float* out) {
  const std::vector<float>& cb = Mul1Codebook();
  const std::array<uint8_t, kTile>& perm = TensorCorePerm();
  for (int p = 0; p < kTile; ++p) {
    const uint64_t w0 = w[rt.i0[static_cast<size_t>(p)]], w1 = w[rt.i1[static_cast<size_t>(p)]];
    const int off = rt.off[static_cast<size_t>(p)];
    const uint64_t st = ((((w0 << off) & 0xFFFFFFFFull) >> 16) | (w1 >> (48 - off))) & 0xFFFFull;
    out[perm[static_cast<size_t>(p)]] = cb[static_cast<size_t>(st)];
  }
}

// ---- the FWHT (unnormalized, natural order: stages h = 1..64, (a, b) -> (a + b, a - b)) ---------

inline void Fwht128(float* v) {
  for (int h = 1; h < kHad; h <<= 1)
    for (int i = 0; i < kHad; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
}

// The same transform across 128 rows of `cols` floats each (row r at t + r * stride), per column.
inline void Fwht128Rows(float* t, int64_t stride, int64_t cols) {
  for (int h = 1; h < kHad; h <<= 1)
    for (int i = 0; i < kHad; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        float* a = t + static_cast<int64_t>(j) * stride;
        float* b = t + static_cast<int64_t>(j + h) * stride;
        for (int64_t c = 0; c < cols; ++c) {
          const float x = a[c], y = b[c];
          a[c] = x + y;
          b[c] = x - y;
        }
      }
}

// ---- small helpers ----------------------------------------------------------------------------

// Rewrites the one occurrence of `from` in `path`'s JSON header as `to` (same length: no offset
// moves), durably on both sides, like ContainerWriter::PatchHeader: the file's data is flushed to
// the device BEFORE the patch is written (so a result patched "pass" never reaches the disk ahead
// of the tensors it vouches for), and the patch itself before this returns. The reconstruction
// check runs on the closed file (SafetensorsReader cannot share it with the writer's handle), so
// this reopens it.
inline void PatchContainerHeader(const std::string& path, const std::string& from,
                                 const std::string& to) {
  if (from.size() != to.size())
    throw std::logic_error("PatchContainerHeader: replacement differs in length");
  std::FILE* f = _wfsopen(Utf8ToWide(path).c_str(), L"r+b", _SH_DENYWR);
  if (!f) throw std::runtime_error("cannot reopen " + path + " to patch its header");
  auto fail = [&](const std::string& why) {
    std::fclose(f);
    throw std::runtime_error("patching " + path + "'s header: " + why);
  };
  if (_commit(_fileno(f)) != 0) fail("flushing the data to disk before the patch failed");
  uint64_t n = 0;
  if (std::fread(&n, 1, 8, f) != 8 || n == 0 || n > (uint64_t{1} << 30)) fail("bad header length");
  std::string h(static_cast<size_t>(n), '\0');
  if (std::fread(h.data(), 1, h.size(), f) != h.size()) fail("short header");
  const size_t pos = h.find(from);
  if (pos == std::string::npos || h.find(from, pos + 1) != std::string::npos)
    fail("the text to patch does not occur exactly once");
  if (_fseeki64(f, static_cast<int64_t>(8 + pos), SEEK_SET) != 0 ||
      std::fwrite(to.data(), 1, to.size(), f) != to.size() || std::fflush(f) != 0 ||
      _commit(_fileno(f)) != 0)
    fail("write failed");
  if (std::fclose(f) != 0)
    throw std::runtime_error("closing " + path + " after the header patch failed");
}

inline std::string PadTo(std::string s, size_t width) {
  if (s.size() > width) s.resize(width);
  s.resize(width, ' ');
  return s;
}

// "model.language_model.layers.7.mlp.down_proj.weight" -> 7, or -1.
inline int HfLayerIndex(const std::string& hf) {
  static const std::string prefix = "model.language_model.layers.";
  if (hf.compare(0, prefix.size(), prefix) != 0) return -1;
  size_t i = prefix.size();
  int v = 0;
  bool any = false;
  while (i < hf.size() && hf[i] >= '0' && hf[i] <= '9') {
    v = v * 10 + (hf[i] - '0');
    any = true;
    ++i;
  }
  return any && i < hf.size() && hf[i] == '.' ? v : -1;
}

inline std::string JsonShort(const nlohmann::json& j, size_t n = 120) {
  std::string s = j.dump();
  return s.size() <= n ? s : s.substr(0, n - 3) + "...";
}

// A decoder body linear: the only container bases --trellis-from may cover (2.2). lm_head, mtp.*,
// the draft head and everything bf16-only are outside it.
inline bool IsBodyLinearBase(const std::string& base) {
  static const std::string prefix = "text.layers.";
  return base.compare(0, prefix.size(), prefix) == 0;
}

// ---- the import --------------------------------------------------------------------------------

struct TrellisOptions {
  std::string from;             // --trellis-from: an oracle directory or its weights_override.json
  std::string manifest_sha256;  // --trellis-manifest-sha256 (empty: no pin)
  std::string verify = "full";  // --trellis-verify full|none
  std::string allow_basis;      // --trellis-allow-basis ("" or "exl3")
  int prescale_log2 = 0;        // --trellis-prescale-log2
  // The run's own rotation (Gemma rotated trellis, docs/gemma4-plan.md 9.7 and 4.6): null for an
  // unrotated run, else {"kind", "seed", "tensors_sha256"} -- what RotationSource::Fingerprint() says
  // about the Q / Hb this conversion folds. The oracle quantized the FOLDED weights against the
  // rotated Hessians (trellis_quant.py --rotation), so the manifest must carry exactly this
  // fingerprint under "rotation"; an unrotated run refuses a manifest that carries one and vice versa.
  nlohmann::json rotation = nullptr;
};

// Per-linear fold the reconstruction check applies to the CHECKPOINT weight before comparing it with
// the container's W_hat, for a rotated run: `rows` is a [nrows, K] row block of the HF tensor `hf`
// (rows are independent under every fold the converter has: W diag(w) Q on the K side, W Hb), folded
// in place. Called from worker threads: must be thread-safe. Null = compare against the raw weight.
using VerifyFoldFn =
    std::function<void(const std::string& hf, std::vector<float>& rows, int64_t nrows, int64_t K)>;

// One HF tensor's manifest record, checked (3.3 step 4, minus the checkpoint shape:
// TrellisSource::Plan).
struct TrellisRecord {
  std::string name;  // HF name
  int kb = 0;
  int64_t k = 0, n = 0;
  std::string file;         // as the manifest writes it
  std::string path;         // resolved: absolute, or relative to the manifest's directory
  std::string file_sha256;  // lowercase hex
  double rel_weight_err = 0.0;
};

struct TrellisVerifySummary {
  int64_t checked = 0, failed = 0;
  double worst_dev = 0.0;  // max over tensors of |rel - rec| / rec
  std::string worst_name;
  std::vector<std::string> failures;
};

class TrellisSource {
 public:
  // The reconstruction check's STRICT bound, on top of the spec's tolerance (docs/trellis-kernel.md
  // 3.3 step 10: |rel - rec| <= 0.02 rec + 1e-4, full_logits_golden.py's): |rel - rec| / rec <=
  // 1e-4 (10.3). The spec's 2% cannot see one wrong 16 x 16 tile of a large tensor (about 6e-4 of
  // rel on an mlp.gate_up) nor a check that skipped part of a tensor; the CPU reconstruction
  // reproduces the oracle's rel to 2e-8 on the real containers (2e-6 on the test fixture), so 1e-4
  // is safe and catches a single wrong tile in every real tensor.
  static constexpr double kStrictRelDev = 1e-4;
  // The fixed width of verify.result's placeholder; VerifyPatch() fits the result text and the
  // numeric fields into it.
  static constexpr size_t kVerifyWidth = 320;

  TrellisSource(const TrellisOptions& opt, const std::string& config_text) : opt_(opt) {
    if (opt_.from.empty()) return;
    enabled_ = true;
    namespace fs = std::filesystem;
    fs::path p = fs::u8path(opt_.from);
    std::error_code ec;
    if (fs::is_directory(p, ec)) p /= "weights_override.json";
    if (!fs::is_regular_file(p, ec))
      throw std::runtime_error("--trellis-from " + opt_.from + ": no weights_override.json there");
    manifest_path_ = fs::absolute(p, ec).u8string();
    manifest_dir_ = fs::absolute(p, ec).parent_path().u8string();
    const std::string text = ReadAll(manifest_path_);
    manifest_sha256_ = Sha256Hex(text);
    if (!opt_.manifest_sha256.empty() && opt_.manifest_sha256 != manifest_sha256_)
      throw std::runtime_error("--trellis-manifest-sha256: " + manifest_path_ + " hashes to " +
                               manifest_sha256_ + ", not the pinned " + opt_.manifest_sha256);
    try {
      man_ = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
      throw std::runtime_error(Who() + ": not JSON: " + e.what());
    }
    if (!man_.is_object()) throw std::runtime_error(Who() + ": not a JSON object");
    CheckTopLevel(Sha256Hex(config_text));
  }

  bool Enabled() const { return enabled_; }
  const std::string& ManifestPath() const { return manifest_path_; }
  const std::string& ManifestSha256() const { return manifest_sha256_; }
  bool VerifyFull() const { return opt_.verify == "full"; }
  bool IsMix() const { return mix_; }
  size_t LinearCount() const { return order_.size(); }

  // add_linear (job construction), for a body linear (IsBodyLinearBase). Covered = the manifest has
  // every one of its HF names: then `*trellis_ls` becomes its trellis LayoutSet and the records are
  // checked (step 4 minus the checkpoint shapes), including that every part shares one K (gate K ==
  // up K). Returns true when the linear is IMPORTED. `kept` (--keep-bf16 matched) wins: its entries
  // are skipped (logged) and it returns false -- `*trellis_ls` is still filled when the entries are
  // usable, so --keep-bf16's byte accounting prices the linear at the layout it replaced. An
  // uncovered or partly covered base is remembered for CheckCoverage (step 5).
  bool Resolve(const std::string& base, const std::vector<std::string>& hf_names, bool kept,
               LayoutSet* trellis_ls, std::ostream& log) {
    const nlohmann::json& tensors = man_.at("tensors");
    std::vector<std::string> have, lack;
    for (const auto& h : hf_names) (tensors.contains(h) ? have : lack).push_back(h);
    if (kept) {
      kept_.push_back(base);
      if (lack.empty()) {
        try {
          const Linear L = MakeLinear(base, hf_names);
          *trellis_ls = TrellisLayoutSet(L.kb, L.parts);
        } catch (const std::exception&) {
          // Unusable entries of a kept linear are not this run's concern (it never reads them).
        }
      }
      log << "[r4dx-convert] trellis: " << base << " is --keep-bf16: "
          << (have.empty() ? std::string("not in the manifest either")
                           : std::to_string(have.size()) + " manifest entr" +
                                 (have.size() == 1 ? "y" : "ies") + " skipped")
          << ", written as bf16\n";
      return false;
    }
    if (have.empty()) {
      missing_.push_back(base);
      return false;
    }
    if (!lack.empty()) {
      std::string s = base + " (has";
      for (const auto& h : have) s += " " + h;
      s += "; lacks";
      for (const auto& h : lack) s += " " + h;
      partial_.push_back(s + ")");
      return false;
    }
    Linear L = MakeLinear(base, hf_names);
    for (const auto& r : L.recs) {
      layers_used_.insert(HfLayerIndex(r.name));
      used_names_.insert(r.name);
    }
    *trellis_ls = TrellisLayoutSet(L.kb, L.parts);
    if (!linears_.emplace(base, std::move(L)).second)
      throw std::logic_error("TrellisSource: " + base + " resolved twice");
    order_.push_back(base);
    return true;
  }

  // After every add_linear, before planning: step 5 (coverage: no body linear missing or partly
  // covered) and step 1's stale layers (only the layers this conversion takes tensors from count).
  // Entries this run does not use -- layers past --layers, --keep-bf16 -- are logged, not refused.
  void CheckCoverage(std::ostream& log, std::ostream& warn) const {
    if (!missing_.empty() || !partial_.empty()) {
      std::string msg = Who() + ": " + std::to_string(missing_.size() + partial_.size()) +
                        " body linear(s) not (fully) covered by the manifest:";
      size_t shown = 0;
      for (const auto& m : missing_)
        if (shown++ < 12) msg += "\n  " + m + " (absent)";
      for (const auto& m : partial_)
        if (shown++ < 12) msg += "\n  " + m;
      if (shown > 12) msg += "\n  ... " + std::to_string(shown - 12) + " more";
      throw std::runtime_error(
          msg +
          "\npartial coverage is refused: v1 has no mixed w4a16/trellis body "
          "(docs/trellis-kernel.md 3.3 step 5); --keep-bf16 a base to write it "
          "as bf16 instead");
    }
    if (man_.contains("stale_layers") && !man_["stale_layers"].is_null()) {
      const nlohmann::json& st = man_["stale_layers"];
      if (!st.is_array()) throw std::runtime_error(Who() + ": stale_layers is not an array");
      for (const auto& s : st) {
        const std::string name = s.is_string() ? s.get<std::string>() : s.dump();
        int layer = -1;
        if (name.size() >= 7 && name[0] == 'L' && name.compare(name.size() - 5, 5, ".json") == 0) {
          const std::string digits = name.substr(1, name.size() - 6);
          if (!digits.empty() && digits.size() < 6 &&
              digits.find_first_not_of("0123456789") == std::string::npos)
            layer = std::stoi(digits);
        }
        if (layer < 0)
          throw std::runtime_error(Who() + ": stale_layers entry " + name +
                                   " is not an L<ii>.json name -- cannot tell which layer it is");
        if (layers_used_.count(layer))
          throw std::runtime_error(Who() + ": layer " + std::to_string(layer) +
                                   " is listed in stale_layers (" + name +
                                   ": written by another job, other code or another K) and this "
                                   "conversion uses it -- re-run the oracle for it");
        log << "[r4dx-convert] trellis: stale layer " << name
            << " is not used by this conversion\n";
      }
    }
    std::vector<std::string> unused;
    for (auto it = man_.at("tensors").begin(); it != man_.at("tensors").end(); ++it)
      if (!used_names_.count(it.key())) unused.push_back(it.key());
    if (!unused.empty()) {
      std::ostream& os = kept_.empty() ? warn : log;
      os << "[r4dx-convert] " << (kept_.empty() ? "WARNING: " : "") << "trellis: " << unused.size()
         << " manifest tensor(s) not used by this conversion (--layers, --keep-bf16):";
      for (size_t i = 0; i < unused.size() && i < 6; ++i) os << " " << unused[i];
      os << (unused.size() > 6 ? " ..." : "") << "\n";
    }
  }

  // The plan job: the manifest's [n, k] of every part against the checkpoint's (the rest of step
  // 4).
  void Plan(const std::string& base, const std::vector<int64_t>& part_n, int64_t K) {
    Linear& L = linears_.at(base);
    if (part_n.size() != L.recs.size()) throw std::logic_error("TrellisSource::Plan: part count");
    for (size_t i = 0; i < L.recs.size(); ++i) {
      const TrellisRecord& r = L.recs[i];
      if (r.k != K || r.n != part_n[i])
        throw std::runtime_error(Who() + ": " + r.name + " is [" + std::to_string(r.n) + ", " +
                                 std::to_string(r.k) + "] in the manifest but [" +
                                 std::to_string(part_n[i]) + ", " + std::to_string(K) +
                                 "] in the checkpoint");
    }
    L.K = K;
    L.planned = true;
  }

  // After the planning pass, before the header is written (steps 7-8, moved ahead of the emit pass
  // so a bad file costs no output): every file the plan uses hashes to its records' file_sha256
  // (each file once, `threads` at a time), and every used tensor is I32 [k/16, n/16, 8K], F16 [k],
  // F16 [n].
  void CheckFiles(int threads, std::ostream& log) {
    std::map<std::string, std::string> want;  // path -> sha256 its records claim
    std::map<std::string, std::string> file_of;
    for (const auto& base : order_) {
      const Linear& L = linears_.at(base);
      if (!L.planned)
        throw std::logic_error("TrellisSource::CheckFiles: " + base + " was never planned");
      for (const auto& r : L.recs) {
        auto it = want.find(r.path);
        if (it == want.end()) {
          want[r.path] = r.file_sha256;
          file_of[r.path] = r.file;
        } else if (it->second != r.file_sha256) {
          throw std::runtime_error(Who() + ": " + r.file +
                                   " is recorded with two different sha256 (" + it->second + ", " +
                                   r.file_sha256 + ")");
        }
      }
    }
    std::vector<std::string> paths;
    for (const auto& kv : want) paths.push_back(kv.first);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<FileDigest> dig = Sha256Files(paths, threads);
    uint64_t bytes = 0;
    std::string bad;
    for (size_t i = 0; i < paths.size(); ++i) {
      bytes += dig[i].bytes;
      files_[file_of[paths[i]]] = dig[i].sha256;
      if (dig[i].sha256 != want[paths[i]])
        bad += "\n  " + paths[i] + ": sha256 " + dig[i].sha256 + ", the manifest says " +
               want[paths[i]];
    }
    if (!bad.empty())
      throw std::runtime_error(Who() +
                               ": file(s) do not hash to their records' file_sha256 -- the oracle "
                               "directory changed after the manifest was written:" +
                               bad);
    log << "[r4dx-convert] trellis: " << paths.size() << " oracle file(s), " << bytes
        << " B, hashed in "
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
        << " s: every sha256 matches the manifest\n";
    for (const auto& p : paths) readers_[p] = std::make_unique<SafetensorsReader>(Utf8ToWide(p));
    for (const auto& base : order_) {
      for (const auto& r : linears_.at(base).recs) {
        const SafetensorsReader& rd = *readers_.at(r.path);
        auto need = [&](const std::string& t, const char* dtype,
                        const std::vector<int64_t>& shape) {
          if (!rd.Has(t)) throw std::runtime_error(Who() + ": " + r.file + " has no tensor " + t);
          const TensorMeta& m = rd.Meta(t);
          uint64_t elems = 1;
          for (int64_t d : shape) elems *= static_cast<uint64_t>(d);
          const uint64_t width = std::string(dtype) == "I32" ? 4 : 2;
          if (m.dtype != dtype || m.shape != shape || m.end - m.begin != elems * width)
            throw std::runtime_error(Who() + ": " + r.file + " tensor " + t + " is " + m.dtype +
                                     " " + nlohmann::json(m.shape).dump() + ", expected " + dtype +
                                     " " + nlohmann::json(shape).dump());
        };
        need(r.name + ".trellis", "I32", {r.k / 16, r.n / 16, WordsPerTile(r.kb)});
        need(r.name + ".suh", "F16", {r.k});
        need(r.name + ".svh", "F16", {r.n});
      }
    }
  }

  // After CheckFiles: the one-line plan summary (and a note when a mix's sources were written by
  // different oracle code -- recorded, not refused: the bits describe themselves and step 10 checks
  // them).
  void ReportPlan(std::ostream& log) const {
    std::map<int, int64_t> hf_per_kb;
    int64_t hf = 0;
    for (const auto& base : order_) {
      const Linear& L = linears_.at(base);
      hf_per_kb[L.kb] += static_cast<int64_t>(L.recs.size());
      hf += static_cast<int64_t>(L.recs.size());
    }
    log << "[r4dx-convert] trellis: " << manifest_path_ << " (sha256 " << manifest_sha256_ << ", "
        << (mix_ ? "mix" : "quantize-model") << " form, hessian_basis " << basis_
        << "): " << order_.size() << " linear(s) / " << hf << " HF tensor(s) imported";
    for (const auto& kv : hf_per_kb) log << ", KB=" << kv.first << " x" << kv.second;
    if (!kept_.empty()) log << "; " << kept_.size() << " kept bf16 (--keep-bf16)";
    log << "; verify " << opt_.verify << "; LDLQ / --quant never touch them\n";
    const nlohmann::json code = CodeSha256();
    if (code.is_object() && mix_) {
      std::set<std::string> distinct;
      for (auto it = code.begin(); it != code.end(); ++it) distinct.insert(it.value().dump());
      if (distinct.size() > 1)
        log << "[r4dx-convert] trellis: NOTE: the mix's sources were written by different oracle "
               "code "
               "(allocation.source_code_sha256, recorded in r4dx_convert_run.trellis.code_sha256); "
               "the "
               "reconstruction check is authoritative\n";
    }
  }

  // __metadata__.quant.trellis (docs/trellis-kernel.md 2.3).
  nlohmann::json QuantMetadata() const {
    nlohmann::json lin = nlohmann::json::object();
    for (const auto& base : order_) {
      const Linear& L = linears_.at(base);
      nlohmann::json e = {{"bits", L.kb}};
      if (!L.parts.empty()) e["parts"] = L.parts;
      lin[base] = e;
    }
    return {{"format", "r4dx-trellis"},
            {"version", 1},
            {"codebook", "mul1"},
            {"codebook_consts",
             {{"mult", "0x83dcd12d"}, {"k_inv_f16", "0x1eee"}, {"k_bias_f16", "0xc931"}}},
            {"state_bits", kStateBits},
            {"tail_biting", true},
            {"position_order", "exl3-tensor-core"},
            {"bitstream", "ring-u32-msb-first"},
            {"tile_grid", "n32-pairs-k-major"},
            {"hadamard",
             {{"block", kHad},
              {"order", "sylvester-natural"},
              {"scale", "1/sqrt(128)"},
              {"input", "x*suh then H"},
              {"output", "H then *svh"}}},
            {"prescale_log2", opt_.prescale_log2},
            {"linears", lin}};
  }

  // quant_summary's line, e.g. "trellis mul1 KB=4 x400 (imported)" (HF tensors per rate).
  std::string SummaryLine() const {
    std::map<int, int64_t> per;
    for (const auto& base : order_)
      per[linears_.at(base).kb] += static_cast<int64_t>(linears_.at(base).recs.size());
    std::string s = "trellis mul1";
    bool first = true;
    for (const auto& kv : per) {
      s +=
          (first ? " KB=" : " + KB=") + std::to_string(kv.first) + " x" + std::to_string(kv.second);
      first = false;
    }
    return s + " (imported)";
  }

  // verify.result as the header is first written: a fixed-width placeholder that the caller patches
  // in place once Verify() ran (VerifyNeedle -> VerifyPatch), or the final text for
  // --trellis-verify none, which only a debug build accepts (main.cpp). A result that does not
  // start with "pass" -- this "pending" (the conversion died before the check finished), "FAILED"
  // or "not run" -- is not a verified container.
  std::string VerifyPlaceholder() const {
    return VerifyFull() ? PadTo(
                              "pending: the reconstruction check (--trellis-verify full) has not "
                              "finished",
                              kVerifyWidth)
                        : std::string("not run (--trellis-verify none, a debug build)");
  }

  // The exact header text the result patch replaces: `"result":"<placeholder>"` (the header is
  // nlohmann's compact dump; main.cpp checks that it occurs exactly once before the emit pass).
  std::string VerifyNeedle() const {
    return "\"result\":" + nlohmann::json(VerifyPlaceholder()).dump();
  }

  // Its replacement, the same length: verify.result's text, then the numbers -- worst |rel - rec| /
  // rec, its tensor, checked, failed (spec 3.3's `verify: {mode, worst}`) -- as further members of
  // the verify object, padded with JSON whitespace after the last value. The text is shortened if
  // it must be; the numbers never are.
  std::string VerifyPatch(const TrellisVerifySummary& s) const {
    const std::string needle = VerifyNeedle();
    std::string text = (s.failed == 0 ? "pass " + std::to_string(s.checked) + "/"
                                      : "FAILED " + std::to_string(s.failed) + "/") +
                       std::to_string(s.checked) + " HF tensors; worst |rel - rec| / rec " +
                       Num(s.worst_dev) + " (" + s.worst_name + ")";
    const std::string tail = ",\"worst\":" + nlohmann::json(s.worst_dev).dump() +
                             ",\"worst_tensor\":" + nlohmann::json(s.worst_name).dump() +
                             ",\"checked\":" + std::to_string(s.checked) +
                             ",\"failed\":" + std::to_string(s.failed);
    std::string out = "\"result\":" + nlohmann::json(text).dump() + tail;
    while (out.size() > needle.size() && !text.empty()) {
      text.resize(text.size() - std::min(text.size(), out.size() - needle.size()));
      out = "\"result\":" + nlohmann::json(text).dump() + tail;
    }
    if (out.size() > needle.size())
      throw std::logic_error("TrellisSource::VerifyPatch: the numbers alone overflow the field");
    out.resize(needle.size(), ' ');
    return out;
  }

  // __metadata__.r4dx_convert_run.trellis (docs/trellis-kernel.md 3.3).
  nlohmann::json RunMetadata() const {
    std::map<std::string, int64_t> per;
    int64_t hf = 0;
    for (const auto& base : order_) {
      per[std::to_string(linears_.at(base).kb)] +=
          static_cast<int64_t>(linears_.at(base).recs.size());
      hf += static_cast<int64_t>(linears_.at(base).recs.size());
    }
    nlohmann::json j = {
        {"manifest", manifest_path_},
        {"manifest_sha256", manifest_sha256_},
        {"manifest_form", mix_ ? "mix" : "quantize-model"},
        {"encoding", StrOr(man_, "encoding", "")},
        {"bpw_target", man_.value("bpw_target", nlohmann::json(nullptr))},
        {"K_uniform", man_.value("K_uniform", nlohmann::json(nullptr))},
        {"hessian_basis", basis_},
        {"hessian_manifest_sha256", man_.value("hessian_manifest_sha256", nlohmann::json(nullptr))},
        {"config_sha256", StrOr(man_, "config_sha256", "")},
        {"code_sha256", CodeSha256()},
        {"recipe", man_.value("recipe", nlohmann::json(nullptr))},
        {"allow_basis",
         opt_.allow_basis.empty() ? nlohmann::json(nullptr) : nlohmann::json(opt_.allow_basis)},
        {"prescale_log2", opt_.prescale_log2},
        {"linears", static_cast<int64_t>(order_.size())},
        {"hf_tensors", hf},
        {"hf_tensors_per_K", per},
        {"kept_bf16", kept_},
        {"files", files_},
        {"verify",
         {{"mode", opt_.verify},
          {"tolerance",
           "|rel - rec| <= 0.02 rec + 1e-4 and |rel - rec| / rec <= " + Num(kStrictRelDev) +
               ", rec = the manifest's rel_weight_err, rel = ||W_hat - W|| / ||W|| "
               "against the bf16 checkpoint, W_hat decoded from this container's bytes"},
          {"result", VerifyPlaceholder()}}},
    };
    if (man_.contains("allocation") && man_["allocation"].is_object())
      j["allocation_sources"] = man_["allocation"].value("sources", nlohmann::json(nullptr));
    // Rotated trellis only: the fingerprint the manifest and this run agreed on (CheckRotation).
    if (!opt_.rotation.is_null()) j["rotation"] = opt_.rotation;
    return j;
  }

  // The emit job of one trellis linear (step 9): the pair grid, suh [P][K], svh [N] -- nothing else
  // is computed; every byte comes from the oracle file.
  void Emit(ContainerWriter& writer, const std::string& base, int threads) const {
    const Linear& L = linears_.at(base);
    if (!L.planned) throw std::logic_error("TrellisSource::Emit: " + base + " was never planned");
    int64_t N = 0;
    std::vector<OracleWords> parts;
    for (const auto& r : L.recs) {
      parts.push_back({readers_.at(r.path)->Data(r.name + ".trellis"), r.n});
      N += r.n;
    }
    const uint64_t words =
        static_cast<uint64_t>(N) * static_cast<uint64_t>(L.K) * static_cast<uint64_t>(L.kb) / 32;
    std::vector<uint32_t> w(static_cast<size_t>(words));
    RegridToPairGrid(parts, L.K, L.kb, w.data(), threads);
    writer.WriteTensor(base + ".trellis.w", w.data(), words * 4);
    std::vector<uint8_t> suh(L.recs.size() * static_cast<size_t>(L.K) * 2),
        svh(static_cast<size_t>(N) * 2);
    size_t so = 0, vo = 0;
    for (const auto& r : L.recs) {
      const SafetensorsReader& rd = *readers_.at(r.path);
      std::memcpy(suh.data() + so, rd.Data(r.name + ".suh"), static_cast<size_t>(r.k) * 2);
      std::memcpy(svh.data() + vo, rd.Data(r.name + ".svh"), static_cast<size_t>(r.n) * 2);
      so += static_cast<size_t>(r.k) * 2;
      vo += static_cast<size_t>(r.n) * 2;
    }
    writer.WriteTensor(base + ".trellis.suh", suh.data(), suh.size());
    writer.WriteTensor(base + ".trellis.svh", svh.data(), svh.size());
  }

  // Step 10, on the CLOSED container: every imported HF tensor's W_hat = diag(suh) P_K Q P_N
  // diag(svh) decoded from `container`'s own bytes (so the regrid is covered too), on `threads` CPU
  // threads -- fp32 decode and FWHT, fp64 sums -- against the checkpoint's weight: rel = ||W_hat -
  // W|| / ||W|| must be within 0.02 rel_weight_err + 1e-4 of the oracle's own figure (the spec's
  // tolerance) AND within kStrictRelDev of it relatively. One log line per tensor. The work item is
  // one 128-row block of one part, taken in 128 x 128 blocks (L2-resident); every sum is per item
  // in a fixed order and the items are added in row order, so the result does not depend on
  // `threads`.
  TrellisVerifySummary Verify(const std::string& container, ShardedModel& model, int threads,
                              std::ostream& log, const VerifyFoldFn& fold = nullptr) const {
    const SafetensorsReader c(Utf8ToWide(container));
    struct Item {
      size_t li, part;
      int64_t block;
    };
    std::vector<Item> items;
    std::vector<const Linear*> lins;
    // The checkpoint weights, looked up here, single-threaded (ShardedModel opens shards lazily and
    // is not thread-safe); the workers only read through these pointers.
    struct Src {
      const uint8_t* data;
      bool bf16;
    };
    std::vector<std::vector<Src>> wsrc;  // [linear][part]
    for (const auto& base : order_) {
      const Linear& L = linears_.at(base);
      for (const char* suffix : {".trellis.w", ".trellis.suh", ".trellis.svh"})
        if (!c.Has(base + suffix))
          throw std::runtime_error("trellis verify: the container has no " + base + suffix);
      wsrc.emplace_back();
      for (size_t p = 0; p < L.recs.size(); ++p) {
        const TrellisRecord& r = L.recs[p];
        const TensorMeta& m = model.Meta(r.name);
        if ((m.dtype != "BF16" && m.dtype != "F32") || m.shape != std::vector<int64_t>{r.n, r.k})
          throw std::runtime_error("trellis verify: checkpoint tensor " + r.name + " is " +
                                   m.dtype + " " + nlohmann::json(m.shape).dump());
        wsrc.back().push_back({model.Data(r.name), m.dtype == "BF16"});
        for (int64_t b = 0; b < r.n / kHad; ++b) items.push_back({lins.size(), p, b});
      }
      lins.push_back(&L);
    }
    std::vector<double> err2(items.size(), 0.0), ref2(items.size(), 0.0);
    ParallelEach(items.size(), threads, [&](size_t i) {
      const Item& it = items[i];
      const Linear& L = *lins[it.li];
      const int64_t K = L.K;
      const int64_t row0 =
          PartRow0(L, it.part) + it.block * kHad;  // first container row of this block
      const RingTables& rt = Ring(L.kb);
      const uint8_t* grid = c.Data(L.base + ".trellis.w");
      const uint8_t* suh_b =
          c.Data(L.base + ".trellis.suh") + static_cast<uint64_t>(it.part) * K * 2;
      const uint8_t* svh_b = c.Data(L.base + ".trellis.svh") + static_cast<uint64_t>(row0) * 2;
      std::vector<float> blk(static_cast<size_t>(kHad * kHad));  // [n in block][k in block]
      float sv[kHad];
      for (int64_t row = 0; row < kHad; ++row) {
        uint16_t hv;
        std::memcpy(&hv, svh_b + row * 2, 2);
        sv[row] = r4dx::core::F16ToFloat(hv) * (1.0f / 128.0f);  // both Hadamards' 1/sqrt(128)
      }
      const Src& src = wsrc[it.li][it.part];
      uint32_t words[40];
      float vals[kTile];
      double e2 = 0.0, w2 = 0.0;
      // Rotated run: the weight the oracle quantized is fold(W), so the 128-row block is read whole,
      // folded (rows are independent) and compared from this buffer; an unrotated run reads W as is.
      std::vector<float> folded;
      if (fold) {
        folded.resize(static_cast<size_t>(kHad * K));
        for (int64_t row = 0; row < kHad; ++row) {
          const int64_t wrow = it.block * kHad + row;
          for (int64_t k = 0; k < K; ++k) {
            if (src.bf16) {
              uint16_t b;
              std::memcpy(&b, src.data + (wrow * K + k) * 2, 2);
              folded[static_cast<size_t>(row * K + k)] = r4dx::core::Bf16ToFloat(b);
            } else {
              std::memcpy(&folded[static_cast<size_t>(row * K + k)], src.data + (wrow * K + k) * 4, 4);
            }
          }
        }
        fold(L.recs[it.part].name, folded, kHad, K);
      }
      for (int64_t k0 = 0; k0 < K; k0 += kHad) {
        for (int64_t t8 = 0; t8 < kHad / 16; ++t8) {
          const int64_t tn = row0 / 16 + t8;
          for (int64_t q8 = 0; q8 < kHad / 16; ++q8) {
            const int64_t tk = k0 / 16 + q8;
            std::memcpy(words, grid + PairGridIndex(tn, tk, 0, K, L.kb) * 4,
                        static_cast<size_t>(rt.nw) * 4);
            DecodeTile(words, rt, vals);
            for (int cc = 0; cc < 16;
                 ++cc)  // block row = n (tile column cc), block column = k (tile row rr)
              for (int rr = 0; rr < 16; ++rr)
                blk[static_cast<size_t>((t8 * 16 + cc) * kHad + q8 * 16 + rr)] = vals[rr * 16 + cc];
          }
        }
        for (int64_t row = 0; row < kHad; ++row) Fwht128(blk.data() + row * kHad);  // P_K
        Fwht128Rows(blk.data(), kHad, kHad);                                        // P_N
        float su[kHad];
        for (int64_t k = 0; k < kHad; ++k) {
          uint16_t h;
          std::memcpy(&h, suh_b + (k0 + k) * 2, 2);
          su[k] = r4dx::core::F16ToFloat(h);
        }
        for (int64_t row = 0; row < kHad; ++row) {
          const int64_t wrow =
              it.block * kHad + row;  // row inside the checkpoint tensor (this part)
          const float* br = blk.data() + row * kHad;
          for (int64_t k = 0; k < kHad; ++k) {
            float wv;
            if (fold) {
              wv = folded[static_cast<size_t>(row * K + k0 + k)];
            } else if (src.bf16) {
              uint16_t b;
              std::memcpy(&b, src.data + (wrow * K + k0 + k) * 2, 2);
              wv = r4dx::core::Bf16ToFloat(b);
            } else {
              std::memcpy(&wv, src.data + (wrow * K + k0 + k) * 4, 4);
            }
            const double d = static_cast<double>(br[k] * su[k] * sv[row]) - wv;
            e2 += d * d;
            w2 += static_cast<double>(wv) * wv;
          }
        }
      }
      err2[i] = e2;
      ref2[i] = w2;
    });
    TrellisVerifySummary s;
    size_t i = 0;
    for (const Linear* L : lins) {
      for (size_t p = 0; p < L->recs.size(); ++p) {
        const TrellisRecord& r = L->recs[p];
        double e2 = 0.0, w2 = 0.0;  // the blocks in row order
        for (int64_t b = 0; b < r.n / kHad; ++b, ++i) {
          e2 += err2[i];
          w2 += ref2[i];
        }
        const double rel = w2 > 0.0 ? std::sqrt(e2 / w2) : 0.0;
        const double tol = 0.02 * r.rel_weight_err + 1e-4;
        const double dev = std::fabs(rel - r.rel_weight_err);
        const double rdev = r.rel_weight_err > 0.0 ? dev / r.rel_weight_err : dev;
        const bool ok = dev <= tol && rdev <= kStrictRelDev;  // false for a NaN rel
        ++s.checked;
        if (!ok) {
          ++s.failed;
          s.failures.push_back(r.name + " (" + L->base + "): rel " + Num(rel) +
                               " vs the oracle's " + Num(r.rel_weight_err) + ", |d|/rec " +
                               Num(rdev) + " (tolerance " + Num(tol) +
                               " and |d|/rec <= " + Num(kStrictRelDev) + ")");
        }
        if (!(rdev <= s.worst_dev) || s.worst_name.empty()) {
          s.worst_dev = rdev;
          s.worst_name = r.name;
        }
        log << "[r4dx-convert] trellis verify: " << r.name << " -> " << L->base << " KB=" << L->kb
            << " rel " << Num(rel) << " (oracle " << Num(r.rel_weight_err) << ", |d|/rec "
            << Num(rdev) << ") " << (ok ? "OK" : "FAIL") << "\n";
      }
    }
    return s;
  }

 private:
  struct Linear {
    std::string base;
    std::vector<std::string> hf;
    std::vector<TrellisRecord> recs;
    std::vector<int64_t> parts;  // empty for one part
    int kb = 0;
    int64_t K = 0;
    bool planned = false;
  };

  static std::string ReadAll(const std::string& path) {
    std::ifstream f(Utf8ToWide(path).c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }

  static std::string StrOr(const nlohmann::json& j, const char* key, const std::string& dflt) {
    return j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : dflt;
  }

  static std::string Num(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.6g", v);
    return b;
  }

  static int64_t PartRow0(const Linear& L, size_t part) {
    int64_t n0 = 0;
    for (size_t i = 0; i < part; ++i) n0 += L.recs[i].n;
    return n0;
  }

  std::string Who() const { return "--trellis-from " + manifest_path_; }

  // A quantize-model manifest's top-level code_sha256, or a mix's allocation.source_code_sha256.
  nlohmann::json CodeSha256() const {
    if (man_.contains("code_sha256")) return man_["code_sha256"];
    if (man_.contains("allocation") && man_["allocation"].is_object() &&
        man_["allocation"].contains("source_code_sha256"))
      return man_["allocation"]["source_code_sha256"];
    return nullptr;
  }

  // The records of every HF name of one linear, checked, sharing one K.
  Linear MakeLinear(const std::string& base, const std::vector<std::string>& hf_names) const {
    const nlohmann::json& tensors = man_.at("tensors");
    Linear L;
    L.base = base;
    L.hf = hf_names;
    for (const auto& h : hf_names) L.recs.push_back(CheckRecord(h, tensors.at(h)));
    L.kb = L.recs[0].kb;
    for (const auto& r : L.recs) {
      if (r.kb != L.kb) {
        std::string msg = Who() + ": the parts of " + base + " have different K (";
        for (size_t i = 0; i < L.recs.size(); ++i)
          msg += (i ? ", " : "") + L.recs[i].name + " K=" + std::to_string(L.recs[i].kb);
        throw std::runtime_error(
            msg +
            "): gate K must equal up K -- they are one GEMM at one rate "
            "(docs/trellis-kernel.md 2.3; the allocator promotes them together)");
      }
    }
    if (L.recs.size() > 1)
      for (const auto& r : L.recs) L.parts.push_back(r.n);
    return L;
  }

  // Step 1 (format, completeness), 2 (codebook, basis) and 3 (config sha) -- the manifest as a
  // whole.
  void CheckTopLevel(const std::string& config_sha256) {
    auto need_str = [&](const char* key, const std::string& want) {
      const std::string got = StrOr(man_, key, "(absent)");
      if (got != want)
        throw std::runtime_error(Who() + ": " + key + " is '" + got + "', expected '" + want + "'");
    };
    need_str("format", "r4dx-weights-override");
    if (!man_.contains("version") || !man_["version"].is_number_integer() ||
        man_["version"].get<int64_t>() != 1)
      throw std::runtime_error(Who() + ": version is " +
                               (man_.contains("version") ? man_["version"].dump() : "(absent)") +
                               ", expected 1");
    need_str("encoding", "trellis-exl3");
    if (!man_.contains("complete") || !man_["complete"].is_boolean() ||
        !man_["complete"].get<bool>())
      throw std::runtime_error(Who() + ": complete is " +
                               (man_.contains("complete") ? man_["complete"].dump() : "(absent)") +
                               " -- the oracle run did not finish every tensor");
    if (!man_.contains("missing_count") || !man_["missing_count"].is_number_integer() ||
        man_["missing_count"].get<int64_t>() != 0)
      throw std::runtime_error(
          Who() + ": missing_count is " +
          (man_.contains("missing_count") ? man_["missing_count"].dump() : "(absent)") +
          ", expected 0");
    need_str("codebook", "mul1");
    basis_ = StrOr(man_, "hessian_basis", "(absent)");
    if (basis_ != "matched" && !(basis_ == "exl3" && opt_.allow_basis == "exl3"))
      throw std::runtime_error(
          Who() + ": hessian_basis is '" + basis_ + "': only 'matched' is imported (" +
          "docs/trellis.md 14 (b) measured it 7-73% better on q/k/v); 'exl3' needs "
          "--trellis-allow-basis exl3");
    CheckRotation();
    const std::string cs = StrOr(man_, "config_sha256", "(absent)");
    if (cs != config_sha256)
      throw std::runtime_error(Who() + ": config_sha256 " + cs +
                               " is not this checkpoint's config.json (" + config_sha256 +
                               ") -- the oracle quantized another checkpoint");
    if (!man_.contains("tensors") || !man_["tensors"].is_object())
      throw std::runtime_error(Who() + ": no tensors object");
    // The two forms (trellis_quant.py write_manifest / cmd_mix): a mix carries `allocation` and
    // none of stale_layers / layers_done / top-level code_sha256.
    mix_ = man_.contains("allocation") && !man_.contains("layers_done");
  }

  // Rotated trellis (Gemma): the oracle's "rotation" fingerprint against this run's. Neither side may
  // have one the other lacks: rotated bits fed to an unrotated container (or the reverse) are
  // garbage, and a different seed or different sign tensors is a different Q.
  void CheckRotation() const {
    const bool want = !opt_.rotation.is_null();
    const bool have = man_.contains("rotation") && !man_["rotation"].is_null();
    if (!want && !have) return;
    if (want && !have)
      throw std::runtime_error(Who() + ": this run folds a rotation (--rotate " +
                               StrOr(opt_.rotation, "kind", "?") +
                               ") but the manifest has none: its bits quantized the UNROTATED weights "
                               "(re-run trellis_quant.py with --rotation <r4dx-convert --rotation-out file>)");
    if (!want)
      throw std::runtime_error(Who() + ": the manifest quantized ROTATED weights (rotation " +
                               JsonShort(man_["rotation"]) +
                               ") but this run has no --rotate: pass the same --rotate / --rotation-seed");
    const nlohmann::json& r = man_["rotation"];
    if (!r.is_object()) throw std::runtime_error(Who() + ": rotation is not an object");
    for (const char* key : {"kind", "seed", "tensors_sha256"}) {
      if (!r.contains(key) || r[key] != opt_.rotation[key])
        throw std::runtime_error(Who() + ": rotation." + key + " is " +
                                 (r.contains(key) ? r[key].dump() : std::string("(absent)")) +
                                 " in the manifest but " + opt_.rotation[key].dump() +
                                 " in this run -- the oracle folded another Q (same --rotate kind, "
                                 "--rotation-seed and checkpoint config are needed; compare "
                                 "`r4dx-convert --rotation-out`'s tensors_sha256)");
    }
  }

  // Step 4 for one HF tensor, before its checkpoint shape is known.
  TrellisRecord CheckRecord(const std::string& name, const nlohmann::json& rec) const {
    auto fail = [&](const std::string& why) -> void {
      throw std::runtime_error(Who() + ": tensor " + name + ": " + why + " (record " +
                               JsonShort(rec) + ")");
    };
    if (!rec.is_object()) fail("not an object");
    TrellisRecord r;
    r.name = name;
    if (StrOr(rec, "encoding", "(absent)") != "trellis-exl3")
      fail("encoding is '" + StrOr(rec, "encoding", "(absent)") + "', expected 'trellis-exl3'");
    if (!rec.contains("K") || !rec["K"].is_number()) fail("no numeric K");
    const double K = rec["K"].get<double>();
    if (K != 4.0 && K != 5.0) fail("K=" + Num(K) + " is not a rate the kernel instantiates (4, 5)");
    r.kb = static_cast<int>(K);
    if (StrOr(rec, "codebook", "(absent)") != "mul1") fail("codebook is not 'mul1'");
    if (rec.contains("hessian_basis") && StrOr(rec, "hessian_basis", "") != basis_)
      fail("hessian_basis '" + StrOr(rec, "hessian_basis", "") + "' differs from the manifest's '" +
           basis_ + "'");
    if (!rec.contains("k") || !rec["k"].is_number_integer() || !rec.contains("n") ||
        !rec["n"].is_number_integer())
      fail("no integer k / n");
    r.k = rec["k"].get<int64_t>();
    r.n = rec["n"].get<int64_t>();
    if (r.k <= 0 || r.n <= 0 || r.k % 16 != 0 || r.n % 16 != 0)
      fail("k / n are not positive multiples of 16");
    if (!rec.contains("shape_hf") || rec["shape_hf"] != nlohmann::json::array({r.n, r.k}))
      fail("shape_hf is not [n, k]");
    if (!rec.contains("words_shape") ||
        rec["words_shape"] != nlohmann::json::array({r.k / 16, r.n / 16, WordsPerTile(r.kb)}))
      fail("words_shape is not [k/16, n/16, 8K]");
    r.file = StrOr(rec, "file", "");
    if (r.file.empty()) fail("no file");
    namespace fs = std::filesystem;
    const fs::path fp = fs::u8path(r.file);
    r.path = (fp.is_absolute() ? fp : fs::u8path(manifest_dir_) / fp).u8string();
    r.file_sha256 = StrOr(rec, "file_sha256", "");
    std::transform(r.file_sha256.begin(), r.file_sha256.end(), r.file_sha256.begin(), [](char ch) {
      return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    });
    if (!IsSha256Hex(r.file_sha256)) fail("file_sha256 is not a sha256");
    if (!rec.contains("rel_weight_err") || !rec["rel_weight_err"].is_number())
      fail("no rel_weight_err");
    r.rel_weight_err = rec["rel_weight_err"].get<double>();
    if (!std::isfinite(r.rel_weight_err) || r.rel_weight_err < 0.0)
      fail("rel_weight_err is not finite >= 0");
    return r;
  }

  TrellisOptions opt_;
  bool enabled_ = false;
  bool mix_ = false;
  std::string manifest_path_, manifest_dir_, manifest_sha256_, basis_;
  nlohmann::json man_;
  std::map<std::string, Linear> linears_;
  std::vector<std::string> order_;  // container (= planning) order
  std::vector<std::string> missing_, partial_, kept_;
  std::set<int> layers_used_;
  std::set<std::string> used_names_;
  std::map<std::string, std::string> files_;  // manifest `file` -> sha256
  std::map<std::string, std::unique_ptr<SafetensorsReader>> readers_;
};

}  // namespace trellis
}  // namespace r4dx_convert
