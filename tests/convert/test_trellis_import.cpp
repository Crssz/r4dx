// r4dx-convert --trellis-from (docs/trellis-kernel.md sections 2, 3 and 6 `convert_trellis_import`;
// src/convert/include/r4dx_convert/trellis_import.hpp, main.cpp). CPU-only.
//
//   (a) always: the pair-grid index against a naive tile-by-tile placement (a bijection; parts
//       concatenated along N; a part that is not whole tile pairs refused), the mul1 codebook's
//       closed-form values and fp16 exactness, the ring tables, the trellis LayoutSet (id round
//       trip, exclusivity, byte accounting against what ContainerWriter really plans, the shape
//       refusals, EmitLinearLayouts refusing to quantize one) and IsBodyLinearBase.
//   (b) against tools/reference/trellis_import_golden.py's fixture (golden_out/trellis_import):
//       the codebook's fp16 bytes equal codebook_np("mul1")'s; DecodeTile of every oracle tensor
//       at K = 4 and 5 equals trellis_quant.decode_words bit for bit; RegridToPairGrid of every
//       linear, in both forms, equals trellis_golden.to_pair_grid byte for byte.
//   (c) r4dx-convert (when built) on the fixture's synthetic 2-layer checkpoint (GDN + full
//       attention, hidden 256, the real model's head shapes in miniature):
//         the quantize-model form (oracle_k4) and the mix form (mix/, absolute files, float K) --
//           every .trellis.w / .suh / .svh byte-equal to the Python regrid, no other layout of a
//           body linear, __metadata__.quant.trellis, the quant_summary line,
//           r4dx_convert_run.trellis (form, code sha, files, per-K counts), --trellis-verify full
//           passing 13/13 with the numbers recorded (worst |rel - rec| / rec <= 1e-4, checked,
//           failed), no .partial left; the model_config as the runtime reads it
//           (ModelConfig::FromJson, every trellis linear's [N, K] what Container::Load asks for,
//           ModelConfig::Shard at TP = 2 with every trellis rank range 128-aligned);
//         everything that is not a trellis linear byte-identical to a conversion without
//           --trellis-from; the --lm-head bf16 twin's body identical; --keep-bf16 skipping a
//           manifest entry (bf16 only, priced against its trellis bytes); the sha pin; an unused
//           stale layer;
//           --trellis-prescale-log2 recorded;
//         every refusal of 3.3, each before an output file exists: an oracle file whose sha256
//           differs, one file recorded with two different sha256, complete false, missing_count >
//           0, a used stale layer, K = 3.5, the exl3 basis without --trellis-allow-basis (and
//           accepted with it), --rotate q2ab, a missing linear and a partly covered one, gate K !=
//           up K, another checkpoint's config sha, a manifest shape other than the checkpoint's, a
//           wrong format, a wrong pin; --trellis-from with --selftest / --dflash-gguf /
//           --reuse-tensors-from / --record-reuse-guard, a --trellis-* option without it, a
//           prescale outside [-16, 16], and (release builds) --trellis-verify none;
//         a reconstruction that misses its rel_weight_err -- grossly (rec doubled), and by 1e-3
//           relative, inside the spec's 2% tolerance but outside the strict bound: the output
//           renamed to <output>.verify-failed with the failure and its numbers patched into its
//           header, no <output> (an earlier file there removed) and no <output>.partial.
// The quantize-model and mix containers of (c) are kept in R4DX_TRELLIS_TINY_DIR (the build tree)
// as tiny_k4.r4dx / tiny_mix.r4dx for the runtime's tests (docs/trellis-kernel.md 6,
// test_trellis_linear / test_tp_loader: declare FIXTURES_REQUIRED trellis_tiny); everything else is
// written under %TEMP% and removed.
//
// (b) and (c) need the fixture: & $py tools\reference\trellis_import_golden.py. Without it the test
// runs (a) and exits 77 (CTest SKIPPED).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"  // src/model: the runtime's own config parse and TP = 2 rules (read only)
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/trellis_import.hpp"

namespace {

using namespace r4dx_convert;
using namespace r4dx_convert::trellis;
namespace fs = std::filesystem;
using json = nlohmann::json;

int g_failures = 0;
constexpr int kThreads = 6;  // this machine is shared

bool Gate(bool cond, const std::string& label) {
  if (cond) {
    std::printf("OK   %s\n", label.c_str());
  } else {
    std::fprintf(stderr, "FAIL %s\n", label.c_str());
    ++g_failures;
  }
  std::fflush(stdout);
  return cond;
}

template <typename F>
void ExpectThrow(const std::string& label, const std::string& needle, F&& f) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    Gate(needle.empty() || msg.find(needle) != std::string::npos,
         label + " (threw: " + msg.substr(0, 200) + ")" +
             (needle.empty() ? "" : " [must mention '" + needle + "']"));
    return;
  }
  Gate(false, label + " (did not throw)");
}

bool Contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.u8string());
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void WriteWhole(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!f) throw std::runtime_error("cannot write " + p.u8string());
}

fs::path TempDir(const std::string& tag) {
  const fs::path d = fs::temp_directory_path() /
                     ("r4dx_test_trellis_import_" + tag + "_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(d);
  return d;
}

std::string Sha(const void* p, size_t n) {
  return Sha256Hex(std::string(static_cast<const char*>(p), n));
}

// ---- (a) --------------------------------------------------------------------------------------

void TestPairGrid() {
  std::printf("---- (a) pair grid ----\n");
  for (int kb : {4, 5}) {
    const int nw = WordsPerTile(kb);
    const int64_t K = 64;
    const std::vector<int64_t> part_n = {64, 32};  // two parts: 4 + 2 tile rows
    int64_t N = 0;
    for (int64_t n : part_n) N += n;
    // Oracle layout per part: [K/16][n/16][nw], word = (global tile row << 20) | (tk << 10) | w.
    std::vector<std::vector<uint32_t>> src;
    int64_t tn0 = 0;
    for (int64_t n : part_n) {
      std::vector<uint32_t> v(static_cast<size_t>(K / 16 * n / 16 * nw));
      for (int64_t tk = 0; tk < K / 16; ++tk)
        for (int64_t t = 0; t < n / 16; ++t)
          for (int w = 0; w < nw; ++w)
            v[static_cast<size_t>((tk * (n / 16) + t) * nw + w)] =
                static_cast<uint32_t>(((tn0 + t) << 20) | (tk << 10) | w);
      src.push_back(std::move(v));
      tn0 += n / 16;
    }
    std::vector<OracleWords> parts;
    for (size_t i = 0; i < src.size(); ++i)
      parts.push_back({reinterpret_cast<const uint8_t*>(src[i].data()), part_n[i]});
    std::vector<uint32_t> out(static_cast<size_t>(N * K * kb / 32), 0xFFFFFFFFu);
    RegridToPairGrid(parts, K, kb, out.data(), 3);
    bool ok = true;
    std::set<uint64_t> seen;
    for (int64_t tn = 0; tn < N / 16; ++tn)
      for (int64_t tk = 0; tk < K / 16; ++tk)
        for (int w = 0; w < nw; ++w) {
          // The closed form of docs/trellis-kernel.md 2.1, spelled out independently.
          const uint64_t idx =
              static_cast<uint64_t>((((tn / 2) * (K / 16) + tk) * 2 + tn % 2) * 8 * kb + w);
          ok = ok && idx == PairGridIndex(tn, tk, w, K, kb) && seen.insert(idx).second &&
               out[idx] == static_cast<uint32_t>((tn << 20) | (tk << 10) | w);
        }
    Gate(ok && seen.size() == out.size(),
         "KB=" + std::to_string(kb) +
             ": RegridToPairGrid puts word w of tile (tn, tk) at "
             "(((tn>>1) K/16 + tk) 2 + (tn&1)) 8KB + w, a bijection, parts concatenated along N");
    // One k-tile of one tile pair is 64 KB contiguous bytes (2.1).
    Gate(PairGridIndex(1, 0, 0, K, kb) == static_cast<uint64_t>(nw) &&
             PairGridIndex(0, 1, 0, K, kb) == static_cast<uint64_t>(2 * nw),
         "KB=" + std::to_string(kb) +
             ": the two tiles of a pair are adjacent, then the next k-tile");
  }
  std::vector<uint32_t> dummy(16 * 48 * 32 / 32 * 4);
  ExpectThrow("a part of 48 rows (not whole tile pairs) is refused", "32 rows", [&] {
    RegridToPairGrid({{reinterpret_cast<const uint8_t*>(dummy.data()), 48}}, 16, 4, dummy.data(),
                     1);
  });
}

void TestCodebookAndRing() {
  std::printf("---- (a) codebook, ring tables ----\n");
  const std::vector<float>& cb = Mul1Codebook();
  // s = 0: x = 0, bytesum 0 -> 1024 * 0.00676727294921875 - 10.3828125 = -3.453125 (exact).
  bool exact = cb.size() == 65536 && cb[0] == -3.453125f;
  // Every value is an fp16 value, and |v| <= (1024 + 1020) kinv + kbias.
  for (size_t s = 0; s < cb.size() && exact; ++s)
    exact =
        r4dx::core::F16ToFloat(r4dx::core::FloatToF16(cb[s])) == cb[s] && std::fabs(cb[s]) < 3.5f;
  Gate(exact, "Mul1Codebook: 65536 fp16-exact values, state 0 = -3.453125");
  Gate(RoundToF16Value(1.0 + std::ldexp(1.0, -11)) == 1.0 &&
           RoundToF16Value(1.0 + 3 * std::ldexp(1.0, -11)) == 1.0 + std::ldexp(1.0, -9) &&
           RoundToF16Value(-(1.0 + 3 * std::ldexp(1.0, -11))) == -(1.0 + std::ldexp(1.0, -9)) &&
           RoundToF16Value(std::ldexp(3.0, -26)) == std::ldexp(1.0, -24),
       "RoundToF16Value: ties to even (both signs), subnormal step 2^-24");
  bool ring = true;
  for (int kb : {4, 5}) {
    const RingTables& rt = Ring(kb);
    for (int p = 0; p < kTile; ++p) {
      const int64_t R = 256 * kb, start = (((p + 1) * kb - 16) % R + R) % R;
      ring = ring && rt.i0[static_cast<size_t>(p)] == start / 32 &&
             rt.off[static_cast<size_t>(p)] == start % 32 &&
             rt.i1[static_cast<size_t>(p)] == (start / 32 + 1) % (8 * kb);
    }
  }
  // Position 0 wraps: its window starts 16 - KB bits before the ring's end (tail-biting).
  Gate(ring && Ring(4).i0[0] == 31 && Ring(4).off[0] == 20 && Ring(5).i0[0] == 39 &&
           Ring(5).off[0] == 21,
       "Ring(4/5): state(p) = the 16 bits ending at (p + 1) KB mod 256 KB, position 0 tail-biting");
  ExpectThrow("Ring(3) is refused", "KB=3", [] { (void)Ring(3); });
}

void TestLayoutSet() {
  std::printf("---- (a) trellis LayoutSet ----\n");
  const LayoutSet one = TrellisLayoutSet(4), two = TrellisLayoutSet(5, {256, 256});
  Gate(one.trellis && !one.bf16 && !one.w4a16 && !one.w4a8 && !one.mxfp4 &&
           TrellisPartCount(one) == 1 && TrellisPartCount(two) == 2,
       "TrellisLayoutSet: the trellis layout only, no bf16 (the old-binary guard)");
  LayoutSet back;
  Gate(LayoutSetId(one) == "trellis.k4" && LayoutSetId(two) == "trellis.k5" &&
           ParseLayoutSetId("trellis.k4", &back) && back.trellis && back.trellis_bits == 4 &&
           !ParseLayoutSetId("bf16+trellis.k4", &back) && !ParseLayoutSetId("trellis.k3", &back) &&
           !ParseLayoutSetId("w4a16.g64+trellis.k5", &back),
       "LayoutSetId/ParseLayoutSetId: trellis.k4 / trellis.k5, exclusive, K in {4, 5}");
  Gate(LinearLayoutTensorNames("x", two) ==
           std::vector<std::string>{"x.trellis.w", "x.trellis.suh", "x.trellis.svh"},
       "LinearLayoutTensorNames: .trellis.w, .trellis.suh, .trellis.svh");
  bool bytes_ok = true;
  for (const auto& c :
       std::vector<std::tuple<int, int, LayoutSet>>{{384, 256, TrellisLayoutSet(4)},
                                                    {512, 256, TrellisLayoutSet(5, {256, 256})},
                                                    {256, 384, TrellisLayoutSet(5)}}) {
    ContainerWriter w;
    PlanLinearLayouts(w, "x", std::get<0>(c), std::get<1>(c), std::get<2>(c));
    const int64_t N = std::get<0>(c), K = std::get<1>(c);
    const LayoutSet& ls = std::get<2>(c);
    bytes_ok =
        bytes_ok &&
        w.PlannedDataBytes() == LinearLayoutBytes(static_cast<int>(N), static_cast<int>(K), ls) &&
        w.PlannedTensorCount() == 3 &&
        w.PlannedBytes(0) == static_cast<uint64_t>(N * K * ls.trellis_bits / 8) &&
        w.PlannedShape(1) == std::vector<int64_t>{TrellisPartCount(ls) * K, 2} &&
        w.PlannedShape(2) == std::vector<int64_t>{N, 2};
  }
  Gate(bytes_ok, "PlanLinearLayouts == LinearLayoutBytes: w [N K KB/8], suh [P K, 2], svh [N, 2]");
  auto plan = [](int N, int K, LayoutSet ls) {
    ContainerWriter w;
    PlanLinearLayouts(w, "x", N, K, ls);
  };
  ExpectThrow("K % 128 != 0 is refused", "K is not a multiple of 128",
              [&] { plan(256, 192, TrellisLayoutSet(4)); });
  ExpectThrow("N % 128 != 0 is refused", "N is not a multiple of 128",
              [&] { plan(160, 256, TrellisLayoutSet(4)); });
  ExpectThrow("a part % 128 != 0 is refused", "part N=192",
              [&] { plan(384, 256, TrellisLayoutSet(4, {192, 192})); });
  ExpectThrow("parts not summing to N are refused", "sum to 256",
              [&] { plan(384, 256, TrellisLayoutSet(4, {128, 128})); });
  ExpectThrow("KB = 3 is refused", "KB=3", [&] { plan(256, 256, TrellisLayoutSet(3)); });
  LayoutSet mixed = TrellisLayoutSet(4);
  mixed.bf16 = true;
  ExpectThrow("trellis next to bf16 is refused", "trellis layout only",
              [&] { plan(256, 256, mixed); });
  ExpectThrow("EmitLinearLayouts refuses to quantize a trellis linear", "imported", [&] {
    ContainerWriter w;
    EmitLinearLayouts(w, "x", std::vector<float>(256 * 256), 256, 256, TrellisLayoutSet(4), 1);
  });
  Gate(IsBodyLinearBase("text.layers.3.attn.k") && !IsBodyLinearBase("lm_head") &&
           !IsBodyLinearBase("mtp.mlp.gate_up") && !IsBodyLinearBase("mtp.draft_head.lm_head"),
       "IsBodyLinearBase: text.layers.* only (lm_head, mtp.*, the draft head stay w4a16/bf16)");
}

// ---- (b) --------------------------------------------------------------------------------------

struct Golden {
  fs::path dir;
  json exp;
};

// Q [K][N] fp16 bytes of one oracle tensor (oracle layout [K/16][N/16][8 KB]) by DecodeTile.
std::string DecodeOracle(const uint8_t* words, int64_t K, int64_t N, int kb) {
  const RingTables& rt = Ring(kb);
  std::vector<uint16_t> q(static_cast<size_t>(K * N));
  uint32_t w[40];
  float vals[kTile];
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn) {
      std::memcpy(w, words + static_cast<uint64_t>((tk * (N / 16) + tn) * rt.nw) * 4,
                  static_cast<size_t>(rt.nw) * 4);
      DecodeTile(w, rt, vals);
      for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
          q[static_cast<size_t>((tk * 16 + r) * N + tn * 16 + c)] =
              r4dx::core::FloatToF16(vals[r * 16 + c]);
    }
  return std::string(reinterpret_cast<const char*>(q.data()), q.size() * 2);
}

void TestGoldenDecodeAndRegrid(const Golden& g) {
  std::printf("---- (b) decode and regrid against the Python references ----\n");
  const std::vector<float>& cb = Mul1Codebook();
  std::vector<uint16_t> cb16(cb.size());
  for (size_t s = 0; s < cb.size(); ++s) cb16[s] = r4dx::core::FloatToF16(cb[s]);
  Gate(Sha(cb16.data(), cb16.size() * 2) == g.exp["codebook_f16_sha256"].get<std::string>(),
       "Mul1Codebook as fp16 == trellis_quant.codebook_np(\"mul1\"), all 65536 states");
  // Every oracle tensor at both rates.
  int decoded = 0, decode_ok = 0;
  for (int kb : {4, 5}) {
    const fs::path d = g.dir / (kb == 4 ? "oracle_k4" : "oracle_k5");
    const json man = json::parse(ReadWhole(d / "weights_override.json"));
    std::map<std::string, std::unique_ptr<SafetensorsReader>> rd;
    for (auto it = man["tensors"].begin(); it != man["tensors"].end(); ++it) {
      const std::string file = it.value()["file"].get<std::string>();
      if (!rd.count(file)) rd[file] = std::make_unique<SafetensorsReader>((d / file).wstring());
      const int64_t k = it.value()["k"].get<int64_t>(), n = it.value()["n"].get<int64_t>();
      const std::string q = DecodeOracle(rd[file]->Data(it.key() + ".trellis"), k, n, kb);
      ++decoded;
      if (Sha256Hex(q) == g.exp["decode"][it.key() + "@" + std::to_string(kb)].get<std::string>())
        ++decode_ok;
    }
  }
  Gate(decoded == 2 * g.exp["hf_tensors"].get<int>() && decode_ok == decoded,
       "DecodeTile == trellis_quant.decode_words bit for bit: " + std::to_string(decode_ok) + "/" +
           std::to_string(decoded) + " tensors (K = 4 and 5)");
  // The regrid of every linear, in both forms, from the files the manifests name.
  for (const char* form : {"k4", "mix"}) {
    const fs::path mp =
        g.dir / (std::string(form) == "k4" ? "oracle_k4" : "mix") / "weights_override.json";
    const json man = json::parse(ReadWhole(mp));
    std::map<std::string, std::unique_ptr<SafetensorsReader>> rd;
    int ok = 0, n_lin = 0;
    for (auto it = g.exp["forms"][form].begin(); it != g.exp["forms"][form].end(); ++it) {
      ++n_lin;
      const json& e = it.value();
      std::vector<OracleWords> parts;
      for (const auto& h : e["hf"]) {
        const json& rec = man["tensors"][h.get<std::string>()];
        fs::path f = fs::u8path(rec["file"].get<std::string>());
        if (!f.is_absolute()) f = mp.parent_path() / f;
        const std::string key = f.u8string();
        if (!rd.count(key)) rd[key] = std::make_unique<SafetensorsReader>(f.wstring());
        parts.push_back(
            {rd[key]->Data(h.get<std::string>() + ".trellis"), rec["n"].get<int64_t>()});
      }
      const int64_t K = e["K"].get<int64_t>(), N = e["N"].get<int64_t>();
      const int kb = e["bits"].get<int>();
      std::vector<uint32_t> out(static_cast<size_t>(N * K * kb / 32));
      RegridToPairGrid(parts, K, kb, out.data(), 3);
      if (Sha(out.data(), out.size() * 4) == e["w_sha256"].get<std::string>() &&
          out.size() * 4 == e["w_bytes"].get<size_t>())
        ++ok;
    }
    Gate(n_lin > 0 && ok == n_lin, std::string(form) +
                                       ": RegridToPairGrid == trellis_golden.to_pair_grid byte for "
                                       "byte for " +
                                       std::to_string(ok) + "/" + std::to_string(n_lin) +
                                       " linears");
  }
}

// ---- (c) --------------------------------------------------------------------------------------

#if defined(R4DX_CONVERT_EXE)

struct Container {
  bool ok = false;
  json header;  // without __metadata__
  json meta;
  std::map<std::string, std::string> tensors;
};

Container ReadContainer(const fs::path& path) {
  Container c;
  std::string bytes;
  try {
    bytes = ReadWhole(path);
  } catch (const std::exception&) {
    return c;
  }
  if (bytes.size() < 8) return c;
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  if (8 + hl > bytes.size()) return c;
  c.header = json::parse(bytes.substr(8, static_cast<size_t>(hl)));
  c.meta = c.header["__metadata__"];
  c.header.erase("__metadata__");
  const std::string data = bytes.substr(static_cast<size_t>(8 + hl));
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    const uint64_t b = it.value()["data_offsets"][0].get<uint64_t>();
    const uint64_t e = it.value()["data_offsets"][1].get<uint64_t>();
    c.tensors[it.key()] = data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b));
  }
  c.ok = true;
  return c;
}

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" +
                          log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

std::string Q(const fs::path& p) { return "\"" + p.u8string() + "\""; }

std::string Str(const json& j) { return j.is_string() ? j.get<std::string>() : std::string(); }

std::string G(double v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.3g", v);
  return b;
}

class ExeTest {
 public:
  explicit ExeTest(const Golden& g) : g_(g), root_(TempDir("exe")) {}
  ~ExeTest() {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  void Run() {
    std::printf("---- (c) r4dx-convert --trellis-from on the synthetic checkpoint ----\n");
    const fs::path ck4 = g_.dir / "ckpt_k4", ckmix = g_.dir / "ckpt_mix";
    const fs::path ok4 = g_.dir / "oracle_k4", omix = g_.dir / "mix";

    // ---- the two forms ----
    Container k4 = Convert("k4", ck4, "--trellis-from " + Q(ok4), true);
    CheckForm(k4, "k4", "quantize-model");
    Gate(k4.ok && k4.meta["r4dx_convert_run"]["trellis"]["code_sha256"] ==
                      json::parse(ReadWhole(ok4 / "weights_override.json"))["code_sha256"],
         "k4: r4dx_convert_run.trellis.code_sha256 == the manifest's code_sha256");
    const std::string mix_sha = g_.exp["manifest_sha256"]["mix"].get<std::string>();
    Container mix = Convert("mix", ckmix,
                            "--trellis-from " + Q(omix / "weights_override.json") +
                                " --trellis-manifest-sha256 " + mix_sha,
                            true);
    CheckForm(mix, "mix", "mix");
    if (mix.ok) {
      json& t = mix.meta["r4dx_convert_run"]["trellis"];
      const json m = json::parse(ReadWhole(omix / "weights_override.json"));
      Gate(t["code_sha256"] == m["allocation"]["source_code_sha256"] &&
               t["manifest_sha256"] == mix_sha &&
               t["hf_tensors_per_K"] == json({{"4", 7}, {"5", 6}}) && t["files"].size() == 2 &&
               t["allocation_sources"] == m["allocation"]["sources"] && t["bpw_target"] == 4.5,
           "mix: code_sha256 = allocation.source_code_sha256, the pinned sha, 6 x K5 + 7 x K4 HF "
           "tensors, 2 "
           "absolute files (oracle_k5 L00, oracle_k4 L01) hashed, allocation sources recorded");
    }
    Gate(k4.ok && mix.ok && !fs::exists(root_ / "k4.r4dx.partial") &&
             !fs::exists(root_ / "mix.r4dx.partial"),
         "k4, mix: written as <output>.partial and renamed once checked (no .partial left)");
    CheckTinyForRuntime(k4, "k4");
    CheckTinyForRuntime(mix, "mix");
    KeepTiny(k4, "tiny_k4.r4dx");
    KeepTiny(mix, "tiny_mix.r4dx");

    // ---- everything else is written exactly as without the flag ----
    Container plain = Convert("plain", ck4, "", true);
    if (k4.ok && plain.ok) {
      int same = 0, other = 0;
      bool all = true;
      for (const auto& kv : k4.tensors) {
        if (Contains(kv.first, ".trellis.")) continue;
        ++other;
        const auto it = plain.tensors.find(kv.first);
        const bool eq = it != plain.tensors.end() && it->second == kv.second &&
                        plain.header[kv.first]["shape"] == k4.header[kv.first]["shape"];
        same += eq;
        all = all && eq;
      }
      bool plain_clean = !plain.meta["quant"].contains("trellis") &&
                         !plain.meta["r4dx_convert_run"].contains("trellis") &&
                         plain.meta["quant_summary"].contains("text.layers.*.mlp.gate_up|down");
      for (const auto& kv : plain.tensors)
        plain_clean = plain_clean && !Contains(kv.first, ".trellis.");
      Gate(all && other == 18, "every non-trellis tensor (" + std::to_string(same) + "/" +
                                   std::to_string(other) +
                                   ": embeddings, norms, gdn a/b/conv1d/A_log/dt_bias, descales, "
                                   "lm_head w4a16) byte-identical to "
                                   "the conversion without --trellis-from");
      Gate(
          plain_clean,
          "without --trellis-from: no trellis tensor, no quant.trellis / r4dx_convert_run.trellis, "
          "the usual body quant_summary");
    }
    Container twin = Convert("lmbf16", ck4, "--trellis-from " + Q(ok4), true, "--lm-head bf16");
    if (k4.ok && twin.ok) {
      bool body_same = true;
      for (const auto& kv : k4.tensors)
        if (Contains(kv.first, ".trellis."))
          body_same =
              body_same && twin.tensors.count(kv.first) && twin.tensors.at(kv.first) == kv.second;
      Gate(body_same && twin.tensors.count("lm_head.bf16.w") &&
               !twin.tensors.count("lm_head.w4a16.wq"),
           "--lm-head bf16 twin: every trellis tensor identical, lm_head bf16 only");
    }

    // ---- --keep-bf16 skips a manifest entry ----
    {
      Container c =
          Convert("keep", ck4, "--trellis-from " + Q(ok4) + " --keep-bf16 \"attn\\.k$\"", true);
      const std::string log = ReadWhole(root_ / "keep.log");
      const std::string base = "text.layers.1.attn.k";
      // attn.k is [512, 256]: bf16 N K 2 bytes instead of the KB = 4 trellis N K / 2 + suh K 2 +
      // svh N 2.
      const int64_t n = 512, k = 256, extra = n * k * 2 - (n * k / 2 + k * 2 + n * 2);
      if (c.ok) {
        json& run = c.meta["r4dx_convert_run"];
        Gate(c.tensors.count(base + ".bf16.w") && !c.tensors.count(base + ".trellis.w") &&
                 !c.meta["quant"]["trellis"]["linears"].contains(base) &&
                 c.meta["quant"]["trellis"]["linears"].size() == 10 &&
                 run["trellis"]["kept_bf16"] == json::array({base}) &&
                 run["keep_bf16_linears"] == json::array({base}) &&
                 run["keep_bf16_extra_bytes"] == extra &&
                 c.header[base + ".bf16.w"]["shape"][0] == n &&
                 Contains(log, base + " is --keep-bf16: 1 manifest entry skipped") &&
                 Contains(Str(run["trellis"]["verify"]["result"]), "pass 12/12") &&
                 run["trellis"]["verify"]["checked"] == 12,
             "--keep-bf16 \"attn\\.k$\": " + base +
                 " bf16 only, its manifest entry skipped and logged, "
                 "priced against its trellis bytes (+" +
                 std::to_string(extra) + " B), verify 12/12");
      } else {
        Gate(false, "--keep-bf16: conversion failed:\n" + log);
      }
    }

    // ---- --trellis-prescale-log2 is recorded (2.3); outside [-16, 16] it is refused ----
    {
      Container c = Convert("prescale", ck4,
                            "--trellis-from " + Q(ok4) + " --trellis-prescale-log2 -3", true);
      Gate(c.ok && c.meta["quant"]["trellis"]["prescale_log2"] == -3 &&
               c.meta["r4dx_convert_run"]["trellis"]["prescale_log2"] == -3,
           "--trellis-prescale-log2 -3: quant.trellis.prescale_log2 and r4dx_convert_run.trellis "
           "record -3");
      Refused("prescale_range", ck4, "--trellis-from " + Q(ok4) + " --trellis-prescale-log2 17",
              "must be an integer in [-16, 16]");
    }

    // ---- --trellis-verify none: a debug build's option only (3.3 step 10) ----
#if defined(NDEBUG)
    Refused("verify_none", ck4, "--trellis-from " + Q(ok4) + " --trellis-verify none",
            "debug builds only");
#else
    {
      Container c =
          Convert("verify_none", ck4, "--trellis-from " + Q(ok4) + " --trellis-verify none", true);
      Gate(c.ok &&
               Str(c.meta["r4dx_convert_run"]["trellis"]["verify"]["result"]).rfind("not run", 0) ==
                   0 &&
               !c.meta["r4dx_convert_run"]["trellis"]["verify"].contains("worst"),
           "--trellis-verify none (debug build): converts, verify.result \"not run ...\", no "
           "numbers");
    }
#endif

    // ---- an unused stale layer is fine, a used one is not ----
    {
      const fs::path d = CopyK4("stale1", [](json& m) { m["stale_layers"] = {"L01.json"}; });
      Container c = Convert("stale_unused", ck4, "--trellis-from " + Q(d) + " --layers 1", true);
      const std::string log = ReadWhole(root_ / "stale_unused.log");
      Gate(c.ok && c.meta["quant"]["trellis"]["linears"].size() == 5 &&
               Contains(log, "stale layer L01.json is not used") &&
               Contains(log, "7 manifest tensor(s) not used"),
           "--layers 1 with stale_layers [L01.json]: converts layer 0 (5 linears), warns about the "
           "7 unused entries");
      Refused("stale_used", ck4, "--trellis-from " + Q(d), "layer 1 is listed in stale_layers");
    }

    // ---- refusals, each before an output exists ----
    {
      const fs::path d = CopyK4("badsha", [](json&) {});
      std::string st = ReadWhole(d / "L01.safetensors");
      st.back() = static_cast<char>(st.back() ^ 0x5A);
      WriteWhole(d / "L01.safetensors", st);
      Refused("badsha", ck4, "--trellis-from " + Q(d), "do not hash to their records' file_sha256");
    }
    Refused("twosha", ck4,
            "--trellis-from " +
                Q(CopyK4("twosha",
                         [](json& m) {
                           // o_proj and k_proj both live in L01.safetensors; give one record
                           // another (valid) sha.
                           m["tensors"]["model.language_model.layers.1.self_attn.o_proj.weight"]
                            ["file_sha256"] = std::string(64, 'b');
                         })),
            "L01.safetensors is recorded with two different sha256");
    Refused("incomplete", ck4,
            "--trellis-from " + Q(CopyK4("incomplete", [](json& m) { m["complete"] = false; })),
            "complete is false");
    Refused("missing_count", ck4,
            "--trellis-from " + Q(CopyK4("missing_count", [](json& m) { m["missing_count"] = 3; })),
            "missing_count is 3");
    Refused(
        "k35", ck4,
        "--trellis-from " +
            Q(CopyK4("k35",
                     [](json& m) {
                       m["tensors"]["model.language_model.layers.1.self_attn.v_proj.weight"]["K"] =
                           3.5;
                     })),
        "K=3.5 is not a rate the kernel instantiates");
    {
      const fs::path d = CopyK4("exl3", [](json& m) {
        m["hessian_basis"] = "exl3";
        for (auto it = m["tensors"].begin(); it != m["tensors"].end(); ++it)
          it.value()["hessian_basis"] = "exl3";
      });
      Refused("exl3", ck4, "--trellis-from " + Q(d), "--trellis-allow-basis exl3");
      Container c = Convert("exl3_allowed", ck4,
                            "--trellis-from " + Q(d) + " --trellis-allow-basis exl3", true);
      Gate(c.ok && c.meta["r4dx_convert_run"]["trellis"]["hessian_basis"] == "exl3" &&
               c.meta["r4dx_convert_run"]["trellis"]["allow_basis"] == "exl3",
           "hessian_basis exl3 with --trellis-allow-basis exl3: converts and records the basis");
    }
    Refused("rotate", ck4, "--trellis-from " + Q(ok4) + " --rotate q2ab",
            "--trellis-from needs --rotate none");
    Refused(
        "missing_linear", ck4,
        "--trellis-from " +
            Q(CopyK4("missing_linear",
                     [](json& m) {
                       m["tensors"].erase("model.language_model.layers.1.self_attn.k_proj.weight");
                     })),
        "text.layers.1.attn.k (absent)");
    Refused("partial", ck4,
            "--trellis-from " +
                Q(CopyK4("partial",
                         [](json& m) {
                           m["tensors"].erase("model.language_model.layers.0.mlp.up_proj.weight");
                         })),
            "lacks model.language_model.layers.0.mlp.up_proj.weight");
    {
      // gate at K = 4 (oracle_k4), up at K = 5 (oracle_k5): both records valid, one linear.
      json m = json::parse(ReadWhole(omix / "weights_override.json"));
      const json k5 = json::parse(ReadWhole(g_.dir / "oracle_k5" / "weights_override.json"));
      const std::string up = "model.language_model.layers.1.mlp.up_proj.weight";
      json rec = k5["tensors"][up];
      rec["file"] = (g_.dir / "oracle_k5" / rec["file"].get<std::string>()).u8string();
      m["tensors"][up] = rec;
      const fs::path d = root_ / "mix_gateup";
      fs::create_directories(d);
      WriteWhole(d / "weights_override.json", m.dump(2));
      Refused("gate_up_k", ckmix, "--trellis-from " + Q(d), "gate K must equal up K");
    }
    Refused("config_sha", ck4,
            "--trellis-from " +
                Q(CopyK4("config_sha", [](json& m) { m["config_sha256"] = std::string(64, '0'); })),
            "is not this checkpoint's config.json");
    Refused(
        "shape", ck4,
        "--trellis-from " +
            Q(CopyK4("shape",
                     [](json& m) {
                       json& r =
                           m["tensors"]["model.language_model.layers.1.self_attn.o_proj.weight"];
                       r["n"] = 128;  // the checkpoint's o_proj is [256, 1024]
                       r["shape_hf"] = {128, 1024};
                       r["words_shape"] = {64, 8, 32};
                     })),
        "is [128, 1024] in the manifest but [256, 1024] in the checkpoint");
    Refused(
        "format", ck4,
        "--trellis-from " + Q(CopyK4("format", [](json& m) { m["format"] = "something-else"; })),
        "format is 'something-else'");
    Refused("pin", ck4,
            "--trellis-from " + Q(ok4) + " --trellis-manifest-sha256 " + std::string(64, 'a'),
            "not the pinned");
    // Argument errors.
    Refused("selftest", ck4, "--trellis-from " + Q(ok4) + " --selftest",
            "not to --selftest or --dflash-gguf");
    Refused("dflash", ck4, "--trellis-from " + Q(ok4) + " --dflash-gguf x.gguf",
            "not to --selftest or --dflash-gguf");
    Refused("reuse", ck4, "--trellis-from " + Q(ok4) + " --reuse-tensors-from x.r4dx",
            "--reuse-tensors-from");
    Refused("guard", ck4, "--trellis-from " + Q(ok4) + " --record-reuse-guard",
            "--record-reuse-guard");
    Refused("option_alone", ck4, "--trellis-prescale-log2 1", "need --trellis-from");

    // ---- a reconstruction that misses its record: grossly, and just outside the strict bound ----
    VerifyFails("verify", 2.0, "rel_weight_err doubled");
    // 1e-3 relative is inside the spec's tolerance (0.02 rec + 1e-4) but 10x the strict 1e-4.
    VerifyFails("verify_strict", 1.001,
                "rel_weight_err x 1.001 (inside the spec's 2%, outside 1e-4)");
  }

 private:
  std::string Common(const fs::path& ckpt, const fs::path& out,
                     const std::string& lm_head = "--lm-head w4a16") {
    return "--input " + Q(ckpt) + " --output " + Q(out) + " --layouts w4a16 --no-bf16 " + lm_head +
           " --vision off --mtp off --quant search --threads " + std::to_string(kThreads);
  }

  Container Convert(const std::string& tag, const fs::path& ckpt, const std::string& extra,
                    bool expect_ok, const std::string& lm_head = "--lm-head w4a16") {
    const fs::path out = root_ / (tag + ".r4dx"), log = root_ / (tag + ".log");
    const int rc = ::Run(Common(ckpt, out, lm_head) + " " + extra, log);
    Container c = rc == 0 ? ReadContainer(out) : Container{};
    if (expect_ok && (rc != 0 || !c.ok)) {
      Gate(false, tag + ": exit " + std::to_string(rc));
      std::printf("%s", ReadWhole(log).c_str());
    }
    return c;
  }

  // A plan-phase refusal: a non-zero exit, no output file, the log naming the reason.
  void Refused(const std::string& tag, const fs::path& ckpt, const std::string& extra,
               const std::string& needle) {
    const fs::path out = root_ / (tag + ".r4dx"), log = root_ / (tag + ".log");
    const int rc = ::Run(Common(ckpt, out) + " " + extra, log);
    const std::string text = ReadWhole(log);
    const size_t at = text.find("error: ");
    Gate(rc != 0 && !fs::exists(out) && Contains(text, needle),
         "refused (" + tag + "): exit " + std::to_string(rc) + ", no output, log: " +
             (at == std::string::npos ? text.substr(0, 160) : text.substr(at, 160)));
  }

  // One tensor's rel_weight_err scaled by `factor` in a copy of oracle_k4: the conversion must fail
  // the reconstruction check -- a non-zero exit, the file renamed to <output>.verify-failed (an
  // earlier <output> removed, no <output>.partial left), "FAILED 1/13" and the numbers (checked 13,
  // failed 1, worst = the scaled tensor's |rel - rec| / rec) patched into its header, the tensor
  // named in the log.
  void VerifyFails(const std::string& tag, double factor, const std::string& what) {
    const std::string v = "model.language_model.layers.1.self_attn.v_proj.weight";
    const fs::path d = CopyK4(tag, [&](json& m) {
      m["tensors"][v]["rel_weight_err"] = factor * m["tensors"][v]["rel_weight_err"].get<double>();
    });
    const fs::path out = root_ / (tag + ".r4dx");
    const fs::path failed = root_ / (tag + ".r4dx.verify-failed"),
                   partial = root_ / (tag + ".r4dx.partial");
    WriteWhole(out, "an earlier conversion's output");
    const int rc =
        ::Run(Common(g_.dir / "ckpt_k4", out) + " --trellis-from " + Q(d), root_ / (tag + ".log"));
    const std::string log = ReadWhole(root_ / (tag + ".log"));
    Container c = ReadContainer(failed);
    json vf = c.ok ? c.meta["r4dx_convert_run"]["trellis"]["verify"] : json::object();
    const std::string result = Str(vf["result"]);
    // |rel - rec| / rec against the scaled record, with rel == the original rec to ~1e-6.
    const double expect_dev = std::fabs(1.0 - factor) / factor;
    const bool numbers =
        vf["checked"] == 13 && vf["failed"] == 1 && vf["worst_tensor"] == v &&
        vf["worst"].is_number() &&
        std::fabs(vf["worst"].get<double>() - expect_dev) <= 1e-3 * expect_dev + 1e-5;
    Gate(rc != 0 && !fs::exists(out) && !fs::exists(partial) && c.ok &&
             result.rfind("FAILED 1/13 HF tensors", 0) == 0 && numbers &&
             Contains(log, v + " (text.layers.1.attn.v): rel") &&
             Contains(log, "|d|/rec <= 0.0001"),
         what + ": exit " + std::to_string(rc) +
             ", renamed to .verify-failed (no output, no .partial), "
             "verify.result \"" +
             result.substr(0, 40) + "...\", worst " +
             (vf["worst"].is_number() ? G(vf["worst"].get<double>()) : std::string("?")) +
             " (expected " + G(expect_dev) + "), the tensor named");
  }

  // The tiny container as the runtime will read it (docs/trellis-kernel.md 6: M4's
  // test_trellis_linear and test_tp_loader cases load tiny_k4 / tiny_mix): ModelConfig::FromJson
  // parses its model_config; every trellis linear's [N, K] is what Container::Load asks for
  // (container.cpp's per-layer shapes); and ModelConfig::Shard accepts it at TP = 2, with every
  // trellis rank range -- column-parallel segments and row-parallel rank K -- a multiple of 128
  // (the loader's rule, 2.4-2.5).
  void CheckTinyForRuntime(Container& c, const std::string& form) {
    if (!c.ok) return;
    using r4dx::model::ModelConfig;
    std::string why;
    int linears = 0;
    try {
      const ModelConfig g = ModelConfig::FromJson(c.meta["model_config"]["text_config"]);
      const int64_t H = g.hidden_size, attn = g.num_attention_heads * g.head_dim;
      const std::map<std::string, std::pair<int64_t, int64_t>> want = {
          {"gdn.in_proj_qkv", {2 * g.KeyDim() + g.ValueDim(), H}},
          {"gdn.in_proj_z", {g.ValueDim(), H}},
          {"gdn.out_proj", {H, g.ValueDim()}},
          {"attn.qg", {2 * attn, H}},
          {"attn.k", {g.num_key_value_heads * g.head_dim, H}},
          {"attn.v", {g.num_key_value_heads * g.head_dim, H}},
          {"attn.o", {H, attn}},
          {"mlp.gate_up", {2 * g.intermediate_size, H}},
          {"mlp.down", {H, g.intermediate_size}}};
      const json& lin = c.meta["quant"]["trellis"]["linears"];
      for (auto it = lin.begin(); it != lin.end(); ++it) {
        const std::string& b = it.key();
        const std::string suffix = b.substr(b.find('.', std::string("text.layers.").size()) + 1);
        const int64_t P =
            it.value().contains("parts") ? static_cast<int64_t>(it.value()["parts"].size()) : 1;
        const int64_t N = c.header[b + ".trellis.svh"]["shape"][0].get<int64_t>();
        const int64_t K = c.header[b + ".trellis.suh"]["shape"][0].get<int64_t>() / P;
        if (!want.count(suffix) || want.at(suffix) != std::make_pair(N, K))
          why += " " + b + " is [" + std::to_string(N) + ", " + std::to_string(K) + "]";
        ++linears;
      }
      for (int rank = 0; rank < 2; ++rank) {
        const ModelConfig r = ModelConfig::Shard(g, 2, rank);  // throws naming the failing rule
        const int64_t ranges[] = {r.KeyDim(),                  // gdn q, k segments
                                  r.ValueDim(),                // gdn v, z; out_proj K
                                  2 * r.num_attention_heads * r.head_dim,  // attn.qg rows
                                  r.num_key_value_heads * r.head_dim,      // attn.k / v rows
                                  r.num_attention_heads * r.head_dim,      // attn.o K
                                  r.intermediate_size};                    // gate / up rows; down K
        for (int64_t v : ranges)
          if (v <= 0 || v % 128 != 0)
            why += " rank " + std::to_string(rank) + " range " + std::to_string(v);
      }
    } catch (const std::exception& e) {
      why += std::string(" threw: ") + e.what();
    }
    Gate(why.empty() && linears == 11, form + ": model_config parses (ModelConfig::FromJson), " +
                                           std::to_string(linears) +
                                           " trellis linears at Container::Load's [N, K], "
                                           "ModelConfig::Shard(TP = 2) legal with every "
                                           "trellis rank range 128-aligned" +
                                           (why.empty() ? "" : " --" + why));
  }

  // A copy of oracle_k4 (relative files) with its manifest edited by `f`.
  fs::path CopyK4(const std::string& tag, const std::function<void(json&)>& f) {
    const fs::path d = root_ / ("k4_" + tag);
    fs::copy(g_.dir / "oracle_k4", d, fs::copy_options::recursive);
    json m = json::parse(ReadWhole(d / "weights_override.json"));
    f(m);
    WriteWhole(d / "weights_override.json", m.dump(2));
    return d;
  }

  // (Non-const: json's non-const operator[] turns a missing key into null, which then fails the
  // gate, where the const one would be undefined behaviour.)
  void CheckForm(Container& c, const std::string& form, const std::string& manifest_form) {
    if (!c.ok) return;
    const json& exp = g_.exp["forms"][form];
    int ok = 0, n = 0;
    json linears = json::object();
    for (auto it = exp.begin(); it != exp.end(); ++it) {
      ++n;
      const std::string& b = it.key();
      const json& e = it.value();
      bool good = true;
      for (const char* t : {"w", "suh", "svh"}) {
        const std::string name = b + ".trellis." + t;
        good = good && c.tensors.count(name) &&
               Sha256Hex(c.tensors.at(name)) == e[std::string(t) + "_sha256"].get<std::string>() &&
               c.tensors.at(name).size() == e[std::string(t) + "_bytes"].get<size_t>();
      }
      for (const auto& kv : c.tensors)  // nothing else of this linear
        if (kv.first.rfind(b + ".", 0) == 0 && !Contains(kv.first, ".trellis.")) good = false;
      good = good && c.header[b + ".trellis.w"]["shape"] == json::array({e["w_bytes"]}) &&
             c.header[b + ".trellis.suh"]["shape"] ==
                 json::array({e["suh_bytes"].get<int64_t>() / 2, 2}) &&
             c.header[b + ".trellis.svh"]["shape"] == json::array({e["N"], 2});
      ok += good;
      json le = {{"bits", e["bits"]}};
      if (!e["parts"].is_null()) le["parts"] = e["parts"];
      linears[b] = le;
    }
    Gate(n == 11 && ok == n,
         form + ": " + std::to_string(ok) + "/" + std::to_string(n) +
             " linears' .trellis.w/.suh/.svh byte-equal to the Python pair grid and scales, "
             "shapes [N K KB/8] / [P K, 2] / [N, 2], no other layout");
    json& q = c.meta["quant"]["trellis"];
    Gate(
        q["format"] == "r4dx-trellis" && q["version"] == 1 && q["codebook"] == "mul1" &&
            q["codebook_consts"] ==
                json({{"mult", "0x83dcd12d"}, {"k_inv_f16", "0x1eee"}, {"k_bias_f16", "0xc931"}}) &&
            q["state_bits"] == 16 && q["tail_biting"] == true &&
            q["position_order"] == "exl3-tensor-core" && q["bitstream"] == "ring-u32-msb-first" &&
            q["tile_grid"] == "n32-pairs-k-major" && q["hadamard"]["block"] == 128 &&
            q["hadamard"]["order"] == "sylvester-natural" && q["prescale_log2"] == 0 &&
            q["linears"] == linears,
        form +
            ": __metadata__.quant.trellis (2.3), linears {bits, parts} as the manifest allocates");
    json& run = c.meta["r4dx_convert_run"]["trellis"];
    const std::string summary =
        Str(c.meta["quant_summary"]
                  ["text.layers.*.attn.qg|k|v|o, gdn.in_proj_qkv|z|out_proj, mlp.gate_up|down"]);
    const std::string result = Str(run["verify"]["result"]);
    Gate(run["manifest_form"] == manifest_form && run["hf_tensors"] == 13 && run["linears"] == 11 &&
             run["config_sha256"] == g_.exp["config_sha256"] && run["hessian_basis"] == "matched" &&
             run["encoding"] == "trellis-exl3" && run["verify"]["mode"] == "full" &&
             result.rfind("pass 13/13 HF tensors", 0) == 0 &&
             Contains(summary, "body: trellis mul1 KB=") &&
             !c.meta["quant_summary"].contains("text.layers.*.attn.k|v") &&
             c.meta["config_sha256"] == g_.exp["config_sha256"],
         form + ": r4dx_convert_run.trellis (" + manifest_form + " form, verify \"" +
             result.substr(0, 60) + "...\"), quant_summary \"" + summary.substr(0, 40) + "...\"");
    // The check itself, not just its verdict: every tensor's rel reproduces the oracle's to the
    // strict bound (the fixture's error is uneven, so a check that skipped part of a tensor would
    // not), and the numbers are recorded as numbers (spec 3.3 `verify: {mode, worst}`).
    const json& v = run["verify"];
    const double worst = v["worst"].is_number() ? v["worst"].get<double>() : -1.0;
    Gate(
        v["checked"] == 13 && v["failed"] == 0 && worst >= 0.0 &&
            worst <= TrellisSource::kStrictRelDev && v["worst_tensor"].is_string() &&
            Contains(Str(v["tolerance"]), "/ rec <= 0.0001"),
        form + ": verify.worst " + G(worst) + " <= 1e-4 (|rel - rec| / rec), checked 13, failed 0");
  }

  void KeepTiny(const Container& c, const std::string& name) {
#if defined(R4DX_TRELLIS_TINY_DIR)
    if (!c.ok) return;
    std::error_code ec;
    const fs::path dir = fs::u8path(R4DX_TRELLIS_TINY_DIR);
    fs::create_directories(dir, ec);
    const std::string tag = name == "tiny_k4.r4dx" ? "k4" : "mix";
    fs::copy_file(root_ / (tag + ".r4dx"), dir / name, fs::copy_options::overwrite_existing, ec);
    std::printf("     kept %s (%s)\n", (dir / name).u8string().c_str(),
                ec ? ec.message().c_str() : "ok");
#else
    (void)c;
    (void)name;
#endif
  }

  const Golden& g_;
  fs::path root_;
};

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  try {
    TestPairGrid();
    TestCodebookAndRing();
    TestLayoutSet();
    Golden g;
    g.dir = fs::u8path(std::string(R4DX_SOURCE_DIR) + "/tools/reference/golden_out/trellis_import");
    if (!fs::exists(g.dir / "expected.json")) {
      std::printf(
          "SKIP (b), (c): no fixture at %s -- run tools\\reference\\trellis_import_golden.py\n",
          g.dir.u8string().c_str());
      if (g_failures) return 1;
      return 77;
    }
    g.exp = json::parse(ReadWhole(g.dir / "expected.json"));
    TestGoldenDecodeAndRegrid(g);
#if defined(R4DX_CONVERT_EXE)
    ExeTest(g).Run();
#else
    std::printf("SKIP (c) r4dx-convert is not part of this build (R4DX_BUILD_CONVERT off)\n");
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures) {
    std::fprintf(stderr, "convert_trellis_import: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("convert_trellis_import: all gates passed\n");
  return 0;
}
