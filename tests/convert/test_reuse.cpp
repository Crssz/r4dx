// convert_reuse -- r4dx-convert --record-reuse-guard / --reuse-tensors-from (docs/quant2.md 5.2;
// src/convert/main.cpp's BuildReuseGuard / OpenReuseBaseline / CheckReuseBaseline / PlanReuse;
// reuse_guard.hpp). CPU-only.
//
// A reuse run must write EXACTLY the container a full run with the same flags writes: every data
// byte, every tensor entry of the header, and the whole __metadata__ except
// r4dx_convert_run.reused_from (which only a reuse run writes; both write reuse_guard). The gates
// compare the reuse output's header text, reused_from removed, against the full run's header text as
// written, and the data sections byte for byte.
//
//   (0) the library pieces: Sha256File against the FIPS vector and against Sha256Hex across its read
//       buffer; Sha256Files (order, missing file); ContainerDataSha256; CpuIdentity;
//       ReuseGuardMismatches (the completion fields and the per-linear record ignored, changed /
//       absent fields named); LayoutSetId / ParseLayoutSetId (round trip, every non-canonical
//       spelling refused) and LinearLayoutTensorNames == what PlanLinearLayouts plans;
//       ContainerWriter::WriteTensorChunked, ReadBackDigests (another handle while the writer holds
//       the file), HeaderOccurrences, PatchHeader, Close. Prints the sha256 throughput (the 27B
//       checkpoint's 55.6 GB are hashed one shard per worker).
//   (a) on a synthetic 2-shard checkpoint (layer 0 GDN, layer 1 full attention, the MTP layer, a
//       reduced-vocab draft head, two vision tensors; hidden 5120, row counts shrunk) with the Q3
//       sweep's recipe (--layouts w4a16 --lm-head w4a16 --no-bf16 --mtp on --vision on, --kv-calib,
//       --quant search --imatrix, --keep-bf16 attn.k/v, plus --draft-vocab-ids): rule R =
//       "mlp\.down$=32" + "draft_head=32". The premise (a full run with R differs from the no-rule
//       full run ONLY in R's four linears), then full(R) == reuse(R) from the no-rule baseline, with
//       reused_from naming exactly those four linears; a full run without --record-reuse-guard is
//       the guarded one minus reuse_guard; the guard's emit_complete is 1 on disk and its
//       data_sha256 the digest of the file's own tensors (reuse == full; reused_from carries the
//       baseline's).
//   (b) the reverse (the baseline has R, this run has no rule) == the no-rule full run; the same from
//       the reuse output of (a) (a reuse output is itself a baseline); a rule on other linears
//       (gdn.out_proj=32 over R) == its full run; a rule naming the default group recomputes
//       nothing; a moved (copied) checkpoint and another --threads are accepted.
//   (c) the guard: every real flag / input file / checkpoint file / binary difference is refused
//       before an output file exists, naming its reuse_guard field; every leaf of a recorded guard,
//       changed in a copy of the baseline, is refused naming that leaf (the per-linear record
//       reuse_guard.linears included); so are an unfinished
//       baseline (emit_complete 0), one without a guard, a truncated one, one whose data does not
//       hash to its data_sha256 (placeholder, another digest, the second half zeroed at full length,
//       one bit flipped), one whose header is not JSON, --output == the baseline (which stays
//       untouched), a missing file, a quant block from another build, a group map naming an unknown
//       linear or disagreeing with the per-linear record, a keep list disagreeing with it, a record
//       without `linears`, naming a linear this run does not write or missing one, a valid record
//       that names tensors the baseline does not have (with and without a matching group map), and a
//       tensor directory with a tensor missing, an
//       extra one or another shape (with a digest recomputed for it, so PlanReuse is what refuses);
//       --reuse-tensors-from with --selftest is an argument error.
//   (d) LDLQ + rotation: --rotate q2ab --ldlq ".*" against synthetic Hessians (keys + rms_keys):
//       premise, full(R) == reuse(R) and the reverse, the reuse log showing LDLQ only for the
//       recomputed mlp.down linears; attn.k alone at g32 (the full run takes its shared-tap factor
//       from HessianStore's cache, the reuse run factors it afresh) == its full run; --ldlq,
//       --ldlq-damp, the Hessian manifest, --rotate and --rotation-seed each refused by the guard;
//       the guard hashes all 7 .hess files; a .hess payload edited in place under an unchanged
//       hessian.json (it passes every Hessian check and changes the mlp.down bytes) is refused by
//       the guard; two same-K files swapped under it are refused even by a full run (header trace).
//   (e) --keep-bf16 may differ (docs/quant2.md 5.3): the guard records no keep regex, only each
//       linear's resolved layout set, and a reuse recomputes the linears whose set differs. On the
//       sweep recipe: (i) a keep added (layer 0 mlp.down) == its full run, one linear recomputed, the
//       keep_bf16_extra_bytes delta exact; (ii) a keep removed (attn.v quantized) == its full run, and
//       the reverse of (i); another spelling of the same keep set recomputes nothing; (iii) a keep plus
//       a group rule on the same class (keep wins for layer 0) == its full run, from the no-rule
//       baseline and from full(R); a keep change together with any other difference is still
//       refused. (iv) on the q2ab + LDLQ fixture: layer 0 mlp.down (Hadamard-folded, K side), layer 1
//       attn.o (Hadamard) and attn.k (in-projection, shared rms tap) kept in bf16 -- their bf16 bytes
//       are the folded weights, not the checkpoint's -- reuse == full, and the reverse (un-keeping
//       them LDLQs exactly those three, attn.k's factor fresh where the full run took it from the
//       cache) == the no-keep full run.
//
// Every file this test writes goes to a fresh directory under %TEMP% (~450 MB, most of it the
// K = 5120 Hessians) and is removed at the end. `convert_reuse --make-fixture <dir>` writes the
// fixture alone and keeps it.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/reuse_guard.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/threadpool.hpp"

namespace {

using namespace r4dx_convert;
namespace fs = std::filesystem;
using json = nlohmann::json;

int g_failures = 0;
constexpr int kThreads = 6;  // this machine is shared; every parallel section here uses at most this

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
         label + " (threw: " + msg + ")" + (needle.empty() ? "" : " [must mention '" + needle + "']"));
    return;
  }
  Gate(false, label + " (did not throw)");
}

bool Contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// The first line of `text` containing `needle`, or "".
std::string LineWith(const std::string& text, const std::string& needle) {
  const size_t p = text.find(needle);
  if (p == std::string::npos) return std::string();
  const size_t b = text.rfind('\n', p);
  const size_t e = text.find('\n', p);
  return text.substr(b == std::string::npos ? 0 : b + 1,
                     (e == std::string::npos ? text.size() : e) - (b == std::string::npos ? 0 : b + 1));
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
                     ("r4dx_test_reuse_" + tag + "_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(d);
  return d;
}

double Since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// ---- (0) library pieces ----------------------------------------------------------------------------

void TestLibrary() {
  std::printf("---- (0) reuse_guard.hpp / ContainerWriter pieces ----\n");
  const fs::path dir = TempDir("lib");
  try {
    WriteWhole(dir / "abc.txt", "abc");
    Gate(Sha256File((dir / "abc.txt").u8string()) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "(0) Sha256File(\"abc\") == the FIPS 180-4 vector");
    std::string big(static_cast<size_t>(20) * 1024 * 1024 + 37, '\0');
    std::mt19937_64 rng(7);
    for (auto& c : big) c = static_cast<char>(rng());
    WriteWhole(dir / "big.bin", big);
    uint64_t n = 0;
    const auto t = std::chrono::steady_clock::now();
    const std::string hf = Sha256File((dir / "big.bin").u8string(), &n);
    const double secs = Since(t);
    Gate(hf == Sha256Hex(big) && n == big.size(),
         "(0) Sha256File == Sha256Hex over 20 MiB + 37 B (streamed across its 8 MiB buffer), length too");
    std::printf("     sha256 throughput %.0f MB/s per worker (file, cached)\n",
                static_cast<double>(big.size()) / 1e6 / std::max(secs, 1e-9));

    {
      WriteWhole(dir / "small.bin", "abc");
      const std::vector<FileDigest> fd = Sha256Files(
          {(dir / "small.bin").u8string(), (dir / "big.bin").u8string(), (dir / "abc.txt").u8string()}, 3);
      Gate(fd.size() == 3 && fd[0].sha256 == Sha256Hex("abc") && fd[0].bytes == 3 &&
               fd[1].sha256 == hf && fd[1].bytes == big.size() && fd[2].sha256 == fd[0].sha256,
           "(0) Sha256Files: one digest + length per path, in the paths' order (hashed largest first)");
      ExpectThrow("(0) Sha256Files names a missing file", "nope.bin", [&]() {
        Sha256Files({(dir / "abc.txt").u8string(), (dir / "nope.bin").u8string()}, 2);
      });
    }
    {
      const std::string ha = Sha256Hex("A"), hb = Sha256Hex("B");
      Gate(ContainerDataSha256({{"b", hb}, {"a", ha}}) == Sha256Hex("a\n" + ha + "\nb\n" + hb + "\n") &&
               ContainerDataSha256({{"a", ha}, {"b", hb}}) == ContainerDataSha256({{"b", hb}, {"a", ha}}) &&
               ContainerDataSha256({{"a", hb}, {"b", ha}}) != ContainerDataSha256({{"a", ha}, {"b", hb}}),
           "(0) ContainerDataSha256: sha256 of \"<name>\\n<digest>\\n\" in name order; binds names to bytes");
      const json cpu = CpuIdentity();
      Gate(cpu.contains("vendor") && cpu.contains("brand") && cpu.contains("fma3") &&
               cpu.contains("avx512f") && cpu.contains("crt_fma3_enable") && cpu.contains("xcr0"),
           "(0) CpuIdentity: vendor, brand, signature, feature bits, XCR0, ucrt's FMA3 switch -- " +
               cpu.dump());
    }

    const json a = {{"version", 1},
                    {"emit_complete", 1},
                    {"data_sha256", std::string(64, 'a')},
                    {"args", {{"layers", 2}, {"quant", "search"}}},
                    {"checkpoint", {{"shards", {{"m.safetensors", {{"sha256", "aa"}}}}}}}};
    json b = a;
    b["emit_complete"] = 0;
    b["data_sha256"] = ReuseDataPlaceholder();
    b[kReuseLinearsKey] = {{"text.layers.0.mlp.down", "bf16"}};
    Gate(ReuseGuardMismatches(a, b).empty(),
         "(0) ReuseGuardMismatches ignores the completion fields (emit_complete, data_sha256) and the "
         "per-linear record (linears)");
    b["args"]["quant"] = "rtn";
    b["checkpoint"]["shards"]["m.safetensors"]["sha256"] = "bb";
    std::vector<std::string> d = ReuseGuardMismatches(a, b);
    Gate(d.size() == 2 &&
             d[0] == "reuse_guard.args.quant: baseline \"search\", this run \"rtn\"" &&
             d[1] == "reuse_guard.checkpoint.shards.m.safetensors.sha256: baseline \"aa\", this run \"bb\"",
         "(0) ReuseGuardMismatches names each changed leaf with both values");
    json c = a;
    c["args"].erase("layers");
    c["extra"] = true;
    d = ReuseGuardMismatches(a, c);
    Gate(d.size() == 2 && d[0] == "reuse_guard.args.layers: baseline 2, this run (absent)" &&
             d[1] == "reuse_guard.extra: baseline (absent), this run true",
         "(0) ReuseGuardMismatches names fields present on one side only");

    {
      // reuse_guard.linears' vocabulary: one canonical string per resolved LayoutSet.
      LayoutSet q;
      q.bf16 = false;
      q.w4a16 = true;
      LayoutSet g32 = q;
      g32.w4a16_group = 32;
      LayoutSet all = g32;
      all.bf16 = true;
      all.w4a16_group = 32;
      LayoutSet none;
      none.bf16 = false;
      const std::vector<std::pair<LayoutSet, std::string>> sets = {
          {KeptBf16LayoutSet(), "bf16"},
          {q, "w4a16.g" + std::to_string(kW4A16Group)},
          {g32, "w4a16.g32"},
          {all, "bf16+w4a16.g32"},
          {none, "none"}};
      bool ids = true, names = true;
      for (const auto& s : sets) {
        LayoutSet back;
        ids = ids && LayoutSetId(s.first) == s.second && ParseLayoutSetId(s.second, &back) &&
              LayoutSetId(back) == s.second && back.bf16 == s.first.bf16 &&
              back.w4a16 == s.first.w4a16 &&
              (!back.w4a16 || back.w4a16_group == s.first.w4a16_group);
        ContainerWriter w;
        PlanLinearLayouts(w, "x.y", 32, 256, s.first);
        std::vector<std::string> planned;
        for (size_t i = 0; i < w.PlannedTensorCount(); ++i) planned.push_back(w.PlannedName(i));
        names = names && planned == LinearLayoutTensorNames("x.y", s.first);
      }
      Gate(ids, "(0) LayoutSetId: bf16 / w4a16.g<default> / w4a16.g32 / "
                "bf16+w4a16.g32 / none, and ParseLayoutSetId round-trips each");
      Gate(names, "(0) LinearLayoutTensorNames == the names PlanLinearLayouts plans, for each set");
      bool refused = true;
      std::string accepted;
      for (const char* bad : {"", "bf16+", "+bf16", "w4a8+bf16", "bf16+bf16", "w4a16.g48",
                              "w4a16.g064", "w4a16.g", "w4a16", "w4a16.g99999", "fp8", "none+bf16",
                              "bf16 ", "BF16", "mxfp4+w4a8", "w4a8", "mxfp4",
                              "w4a16.g32+w4a8+mxfp4"}) {
        LayoutSet ls;
        if (ParseLayoutSetId(bad, &ls)) {
          refused = false;
          accepted += std::string(" '") + bad + "'";
        }
      }
      Gate(refused, "(0) ParseLayoutSetId refuses every non-canonical or unknown spelling" +
                        (accepted.empty() ? std::string() : " -- accepted:" + accepted));
    }

    const fs::path cp = dir / "c.r4dx";
    std::vector<uint8_t> ta(20);
    for (size_t i = 0; i < ta.size(); ++i) ta[i] = static_cast<uint8_t>(i * 7 + 1);
    const uint8_t tb[3] = {9, 8, 7};
    {
      ContainerWriter w;
      w.Plan("a", {5, 4}, 20);
      w.Plan("b", {3}, 3);
      w.FinalizeHeader(cp.u8string(), {{"r4dx_convert_run", {{"reuse_guard", {{"emit_complete", 0}}}}}});
      Gate(w.HeaderOccurrences(ReuseCompleteNeedle(0)) == 1 &&
               w.HeaderOccurrences(ReuseCompleteNeedle(1)) == 0,
           "(0) HeaderOccurrences finds the marker once");
      w.WriteTensorChunked("a", ta.data(), ta.size(), 3);  // 7 chunks, the last one short
      w.WriteTensor("b", tb, 3);
      w.Finish();
      // Read back through other handles while the writer still holds the file (deny-write only).
      const std::vector<std::string> rb = w.ReadBackDigests(2);
      Gate(rb.size() == 2 && rb[0] == Sha256Hex(std::string(ta.begin(), ta.end())) &&
               rb[1] == Sha256Hex(std::string(reinterpret_cast<const char*>(tb), 3)),
           "(0) ReadBackDigests: each planned tensor's sha256 as on disk, plan order");
      w.PatchHeader(ReuseCompleteNeedle(0), ReuseCompleteNeedle(1));
      ExpectThrow("(0) PatchHeader refuses a needle that no longer occurs", "exactly once",
                  [&]() { w.PatchHeader(ReuseCompleteNeedle(0), ReuseCompleteNeedle(1)); });
      ExpectThrow("(0) PatchHeader refuses a length change", "differ in length",
                  [&]() { w.PatchHeader(ReuseCompleteNeedle(1), "\"emit_complete\":10"); });
      ExpectThrow("(0) WriteTensorChunked refuses a size mismatch", "size mismatch",
                  [&]() { w.WriteTensorChunked("a", ta.data(), 19); });
      w.Close();
      w.Close();  // idempotent
      ExpectThrow("(0) PatchHeader after Close() is an error, not a crash", "after Close()",
                  [&]() { w.PatchHeader(ReuseCompleteNeedle(1), ReuseCompleteNeedle(0)); });
      ExpectThrow("(0) WriteTensor after Close() is an error, not a crash", "after Close()",
                  [&]() { w.WriteTensor("b", tb, 3); });
    }
    const std::string bytes = ReadWhole(cp);
    uint64_t hl = 0;
    std::memcpy(&hl, bytes.data(), 8);
    const json h = json::parse(bytes.substr(8, static_cast<size_t>(hl)));
    Gate(h["__metadata__"]["r4dx_convert_run"]["reuse_guard"]["emit_complete"] == 1 &&
             bytes.substr(static_cast<size_t>(8 + hl)) ==
                 std::string(ta.begin(), ta.end()) + std::string(reinterpret_cast<const char*>(tb), 3),
         "(0) the patched header reads emit_complete 1 and the chunked tensor's bytes are exact");
  } catch (const std::exception& e) {
    Gate(false, std::string("(0) threw unexpectedly: ") + e.what());
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- the synthetic checkpoint and its inputs ---------------------------------------------------------

constexpr int64_t kHidden = 5120;  // the rotation is defined for 5 x 1024 only
constexpr int64_t kHeadDim = 256;  // q2ab: one attn.o Hadamard block per head
constexpr int64_t kVDim = 128;     // q2ab: one gdn.out_proj block per head
constexpr int64_t kInter = 512;    // q2ab: mlp.down's block
constexpr int64_t kVocab = 64;
constexpr uint64_t kRows = 4096;
const char* const kL0 = "model.language_model.layers.0.";
const char* const kL1 = "model.language_model.layers.1.";
const char* const kMtp = "mtp.layers.0.";

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

struct SynthTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> v;  // bf16-exact
};

std::vector<SynthTensor> MakeCheckpoint() {
  std::vector<SynthTensor> s;
  std::mt19937_64 rng(0x5EED0526ull);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto rnd = [&](int64_t n, double sigma) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = Bf16Round(static_cast<float>(sigma * nd(rng)));
    return v;
  };
  auto add = [&](const std::string& name, std::vector<int64_t> shape, double sigma) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    s.push_back({name, std::move(shape), rnd(n, sigma)});
  };
  // Zero-centred norms, clamped away from -1 so a rotated in-projection could also divide them out.
  auto add_norm = [&](const std::string& name) {
    std::vector<float> v = rnd(kHidden, 0.3);
    for (auto& x : v) x = Bf16Round(std::min(0.9f, std::max(-0.9f, x)));
    s.push_back({name, {kHidden}, std::move(v)});
  };
  for (const char* L : {kL0, kL1, kMtp}) {
    add_norm(std::string(L) + "input_layernorm.weight");
    add_norm(std::string(L) + "post_attention_layernorm.weight");
    add(std::string(L) + "mlp.gate_proj.weight", {64, kHidden}, 0.02);
    add(std::string(L) + "mlp.up_proj.weight", {64, kHidden}, 0.02);
    add(std::string(L) + "mlp.down_proj.weight", {kHidden, kInter}, 0.02);
  }
  const std::string g = std::string(kL0) + "linear_attn.";
  add(g + "in_proj_qkv.weight", {64, kHidden}, 0.02);
  add(g + "in_proj_z.weight", {kVDim, kHidden}, 0.02);
  add(g + "in_proj_b.weight", {1, kHidden}, 0.02);
  add(g + "in_proj_a.weight", {1, kHidden}, 0.02);
  add(g + "conv1d.weight", {64, 1, 4}, 0.2);
  add(g + "A_log", {1}, 0.5);
  add(g + "dt_bias", {1}, 0.5);
  add(g + "norm.weight", {kVDim}, 0.1);
  add(g + "out_proj.weight", {kHidden, kVDim}, 0.02);
  for (const char* L : {kL1, kMtp}) {
    const std::string a = std::string(L) + "self_attn.";
    add(a + "q_proj.weight", {128, kHidden}, 0.02);
    add(a + "k_proj.weight", {64, kHidden}, 0.02);
    add(a + "v_proj.weight", {64, kHidden}, 0.02);
    add(a + "o_proj.weight", {kHidden, kHeadDim}, 0.02);
    add(a + "q_norm.weight", {kHeadDim}, 0.1);
    add(a + "k_norm.weight", {kHeadDim}, 0.1);
  }
  add("mtp.fc.weight", {16, 32}, 0.02);
  add("mtp.norm.weight", {kHidden}, 0.1);
  add("mtp.pre_fc_norm_embedding.weight", {kHidden}, 0.1);
  add("mtp.pre_fc_norm_hidden.weight", {kHidden}, 0.1);
  add("model.visual.patch_embed.proj.weight", {16, 12}, 0.02);
  add("model.visual.blocks.0.norm1.weight", {16}, 0.1);
  add("model.language_model.embed_tokens.weight", {kVocab, kHidden}, 0.02);
  add("model.language_model.norm.weight", {kHidden}, 0.1);
  add("lm_head.weight", {kVocab, kHidden}, 0.02);
  return s;
}

// Two shards (the guard hashes each), the tensors split in half, bf16 throughout.
void WriteCheckpoint(const std::vector<SynthTensor>& s, const fs::path& dir) {
  fs::create_directories(dir);
  json weight_map = json::object();
  const size_t half = s.size() / 2;
  for (int shard = 0; shard < 2; ++shard) {
    const std::string file = "model-0000" + std::to_string(shard + 1) + "-of-00002.safetensors";
    json hdr = json::object();
    std::string data;
    for (size_t i = shard == 0 ? 0 : half; i < (shard == 0 ? half : s.size()); ++i) {
      const auto& t = s[i];
      const uint64_t b = data.size();
      for (float x : t.v) {
        const uint16_t h = r4dx::core::FloatToBf16(x);
        data.append(reinterpret_cast<const char*>(&h), 2);
      }
      hdr[t.name] = {{"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", {b, data.size()}}};
      weight_map[t.name] = file;
    }
    const std::string h = hdr.dump();
    const uint64_t hl = h.size();
    std::string bytes(reinterpret_cast<const char*>(&hl), 8);
    bytes += h;
    bytes += data;
    WriteWhole(dir / file, bytes);
  }
  WriteWhole(dir / "model.safetensors.index.json", json{{"weight_map", weight_map}}.dump(2));
  const json config = {
      {"architectures", {"SyntheticReuseTest"}},
      {"text_config",
       {{"hidden_size", kHidden},
        {"num_hidden_layers", 2},
        {"layer_types", {"linear_attention", "full_attention"}},
        {"num_attention_heads", 1},
        {"num_key_value_heads", 1},
        {"head_dim", kHeadDim},
        {"linear_num_value_heads", 1},
        {"linear_value_head_dim", kVDim},
        {"intermediate_size", kInter},
        {"mtp_num_hidden_layers", 1}}}};
  WriteWhole(dir / "config.json", config.dump(2));
}

// hessian_store.hpp's .hess format from a packed upper triangle; returns the header trace.
double WriteHessPacked(const fs::path& path, int64_t K, const std::vector<float>& packed) {
  double trace = 0.0;
  size_t off = 0;
  for (int64_t i = 0; i < K; ++i) {
    trace += static_cast<double>(packed[off]);
    off += static_cast<size_t>(K - i);
  }
  std::string bytes(64, '\0');
  std::memcpy(&bytes[0], "R4DXHES1", 8);
  const uint32_t k32 = static_cast<uint32_t>(K), flags = 1;
  const uint64_t rows = kRows;
  std::memcpy(&bytes[8], &k32, 4);
  std::memcpy(&bytes[12], &flags, 4);
  std::memcpy(&bytes[16], &rows, 8);
  std::memcpy(&bytes[24], &trace, 8);
  bytes.append(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(float));
  WriteWhole(path, bytes);
  return trace;
}

// A positive definite H = diag(s) + V V^T (V: K x r), lognormal channel scales, packed upper.
std::vector<float> LowRankPdPacked(uint64_t seed, int64_t K, int r) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ud(0.0, 1.0);
  std::vector<double> V(static_cast<size_t>(K * r)), s(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) {
    const double c = std::exp(0.5 * nd(rng));
    for (int t = 0; t < r; ++t) V[static_cast<size_t>(k * r + t)] = nd(rng) * c / std::sqrt(static_cast<double>(r));
    s[static_cast<size_t>(k)] = (0.05 + 0.1 * ud(rng)) * c * c;
  }
  std::vector<float> p(static_cast<size_t>(K * (K + 1) / 2));
  ParallelFor(0, K, kThreads, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      size_t off = static_cast<size_t>(i * K - i * (i - 1) / 2);
      const double* vi = V.data() + i * r;
      for (int64_t j = i; j < K; ++j) {
        const double* vj = V.data() + j * r;
        double h = (i == j) ? s[static_cast<size_t>(i)] : 0.0;
        for (int t = 0; t < r; ++t) h += vi[t] * vj[t];
        p[off++] = static_cast<float>(h);
      }
    }
  });
  return p;
}

// hessian.json + .hess files for every quantized text linear of (d)'s recipe, with rms_keys for the
// seven norm-fed in-projections (what hessian-v1 has). `hess2` is the same set with a manifest that
// differs only in its "tool" string: another Hessian set identity with the same numbers.
void WriteHessianDirs(const fs::path& hess, const fs::path& hess2) {
  fs::create_directories(hess);
  fs::create_directories(hess2);
  json files = json::object();
  auto put = [&](const std::string& name, int64_t K, uint64_t seed, int r) {
    const double tr = WriteHessPacked(hess / name, K, LowRankPdPacked(seed, K, r));
    files[name] = {{"K", K}, {"rows", kRows}, {"trace", tr}};
    std::error_code ec;
    fs::create_hard_link(hess / name, hess2 / name, ec);
    if (ec) fs::copy_file(hess / name, hess2 / name, fs::copy_options::overwrite_existing);
  };
  put("in.hess", kHidden, 201, 16);
  put("in.rms.hess", kHidden, 202, 16);
  put("mlp_in.hess", kHidden, 203, 16);
  put("mlp_in.rms.hess", kHidden, 204, 16);
  put("L00.out.hess", kVDim, 205, 8);
  put("L01.out.hess", kHeadDim, 206, 8);
  put("mlp_mid.hess", kInter, 207, 8);
  const json keys = {
      {"text.layers.0.gdn.in_proj_qkv", "in.hess"}, {"text.layers.0.gdn.in_proj_z", "in.hess"},
      {"text.layers.0.gdn.out_proj", "L00.out.hess"}, {"text.layers.0.mlp.gate_up", "mlp_in.hess"},
      {"text.layers.0.mlp.down", "mlp_mid.hess"},     {"text.layers.1.attn.qg", "in.hess"},
      {"text.layers.1.attn.k", "in.hess"},            {"text.layers.1.attn.v", "in.hess"},
      {"text.layers.1.attn.o", "L01.out.hess"},       {"text.layers.1.mlp.gate_up", "mlp_in.hess"},
      {"text.layers.1.mlp.down", "mlp_mid.hess"}};
  json rms = json::object();
  for (auto it = keys.begin(); it != keys.end(); ++it) {
    const std::string f = it.value().get<std::string>();
    if (f == "in.hess") rms[it.key()] = "in.rms.hess";
    if (f == "mlp_in.hess") rms[it.key()] = "mlp_in.rms.hess";
  }
  json m = {{"format", "r4dx-hessian"}, {"version", 1}, {"files", files}, {"keys", keys},
            {"rms_keys", rms}, {"tool", "tests/convert/test_reuse.cpp"}};
  WriteWhole(hess / "hessian.json", m.dump(2));
  m["tool"] = "tests/convert/test_reuse.cpp (second set)";
  WriteWhole(hess2 / "hessian.json", m.dump(2));
}

// numpy.savez's shape, as far as npz_reader.hpp reads it: STORED zip members holding .npy v1 '<f4'
// 1-D arrays, then an (empty) end-of-central-directory record.
void WriteNpz(const fs::path& p, const std::map<std::string, std::vector<float>>& vecs) {
  std::string zip;
  auto u16 = [&](uint16_t x) { zip.append(reinterpret_cast<const char*>(&x), 2); };
  auto u32 = [&](uint32_t x) { zip.append(reinterpret_cast<const char*>(&x), 4); };
  for (const auto& kv : vecs) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                      std::to_string(kv.second.size()) + ",), }";
    hdr += std::string((64 - (10 + hdr.size() + 1) % 64) % 64, ' ') + "\n";
    std::string npy = "\x93NUMPY";
    npy += '\x01';
    npy += '\x00';
    const uint16_t hl = static_cast<uint16_t>(hdr.size());
    npy.append(reinterpret_cast<const char*>(&hl), 2);
    npy += hdr;
    npy.append(reinterpret_cast<const char*>(kv.second.data()), kv.second.size() * 4);
    const std::string fname = kv.first + ".npy";
    u32(0x04034b50u);
    u16(20);
    u16(0);  // flags
    u16(0);  // stored
    u16(0);
    u16(0);
    u32(0);  // crc32: npz_reader does not check it
    u32(static_cast<uint32_t>(npy.size()));
    u32(static_cast<uint32_t>(npy.size()));
    u16(static_cast<uint16_t>(fname.size()));
    u16(0);
    zip += fname;
    zip += npy;
  }
  u32(0x06054b50u);
  zip.append(18, '\0');
  WriteWhole(p, zip);
}

std::map<std::string, std::vector<float>> ImatrixVectors(uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> ud(0.2, 3.0);
  auto vec = [&](int64_t K) {
    std::vector<float> v(static_cast<size_t>(K));
    for (auto& x : v) x = static_cast<float>(ud(rng));
    return v;
  };
  std::map<std::string, std::vector<float>> m;
  for (const std::string L : {"text.layers.0.", "text.layers.1.", "mtp."}) {
    m[L + "mlp.gate_up"] = vec(kHidden);
    m[L + "mlp.down"] = vec(kInter);
  }
  m["text.layers.0.gdn.in_proj_qkv"] = vec(kHidden);
  m["text.layers.0.gdn.in_proj_z"] = vec(kHidden);
  m["text.layers.0.gdn.out_proj"] = vec(kVDim);
  for (const std::string L : {"text.layers.1.", "mtp."}) {
    m[L + "attn.qg"] = vec(kHidden);
    m[L + "attn.o"] = vec(kHeadDim);
  }
  m["lm_head"] = vec(kHidden);
  m["mtp.draft_head.lm_head"] = vec(kHidden);
  return m;
}

struct Fixture {
  fs::path root, ckpt, ckpt_moved, ckpt_shard, ckpt_cfg, ckpt_idx, hess, hess2;
  fs::path imatrix, imatrix2, kv, kv2, draft, draft2;
};

void CopyDir(const fs::path& from, const fs::path& to) {
  fs::create_directories(to);
  for (const auto& e : fs::directory_iterator(from))
    fs::copy_file(e.path(), to / e.path().filename(), fs::copy_options::overwrite_existing);
}

// `to` as a hard link of `from` (a copy where linking fails). Never written through afterwards.
void LinkOrCopy(const fs::path& from, const fs::path& to) {
  std::error_code ec;
  fs::create_hard_link(from, to, ec);
  if (ec) fs::copy_file(from, to, fs::copy_options::overwrite_existing);
}

// `from`'s files linked into `to`, except the names in `except` (which the caller writes itself).
void LinkDir(const fs::path& from, const fs::path& to, const std::set<std::string>& except) {
  fs::create_directories(to);
  for (const auto& e : fs::directory_iterator(from))
    if (!except.count(e.path().filename().u8string()))
      LinkOrCopy(e.path(), to / e.path().filename());
}

Fixture MakeFixture(const fs::path& root) {
  Fixture f;
  f.root = root;
  fs::create_directories(root);
  f.ckpt = root / "ckpt";
  WriteCheckpoint(MakeCheckpoint(), f.ckpt);
  // The same checkpoint elsewhere (accepted: the identity is content, not the path), and three
  // copies with one file changed each (refused).
  f.ckpt_moved = root / "ckpt_moved";
  CopyDir(f.ckpt, f.ckpt_moved);
  f.ckpt_shard = root / "ckpt_shard";
  CopyDir(f.ckpt, f.ckpt_shard);
  {
    const fs::path p = f.ckpt_shard / "model-00002-of-00002.safetensors";
    std::string b = ReadWhole(p);
    b[b.size() - 1] = static_cast<char>(b[b.size() - 1] ^ 0x01);  // one bit of lm_head's last value
    WriteWhole(p, b);
  }
  f.ckpt_cfg = root / "ckpt_cfg";
  CopyDir(f.ckpt, f.ckpt_cfg);
  WriteWhole(f.ckpt_cfg / "config.json", ReadWhole(f.ckpt / "config.json") + "\n");
  f.ckpt_idx = root / "ckpt_idx";
  CopyDir(f.ckpt, f.ckpt_idx);
  WriteWhole(f.ckpt_idx / "model.safetensors.index.json",
             json::parse(ReadWhole(f.ckpt / "model.safetensors.index.json")).dump(4));
  f.hess = root / "hess";
  f.hess2 = root / "hess2";
  WriteHessianDirs(f.hess, f.hess2);
  f.imatrix = root / "imatrix.npz";
  f.imatrix2 = root / "imatrix2.npz";
  WriteNpz(f.imatrix, ImatrixVectors(11));
  WriteNpz(f.imatrix2, ImatrixVectors(12));
  f.kv = root / "kv.json";
  f.kv2 = root / "kv2.json";
  WriteWhole(f.kv, json{{"1", {{"k_amax", {3.5}}, {"v_amax", {2.25}}}}}.dump());
  WriteWhole(f.kv2, json{{"1", {{"k_amax", {3.5}}, {"v_amax", {2.5}}}}}.dump());
  f.draft = root / "draft.json";
  f.draft2 = root / "draft2.json";
  std::vector<int> ids, ids2;
  for (int i = 0; i < 16; ++i) {
    ids.push_back(i * 3);
    ids2.push_back(i * 3 + 1);
  }
  WriteWhole(f.draft, json{{"vocab_ids", ids}}.dump());
  WriteWhole(f.draft2, json{{"vocab_ids", ids2}}.dump());
  return f;
}

// ---- containers ------------------------------------------------------------------------------------

struct Container {
  bool ok = false;
  std::string header_text;  // exactly as written
  json header;
  std::string data;
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
  c.header_text = bytes.substr(8, static_cast<size_t>(hl));
  c.header = json::parse(c.header_text);
  c.data = bytes.substr(static_cast<size_t>(8 + hl));
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0].get<uint64_t>();
    const uint64_t e = it.value()["data_offsets"][1].get<uint64_t>();
    c.tensors[it.key()] = c.data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b));
  }
  c.ok = true;
  return c;
}

// `c` with its header replaced (the data section and every data_offsets untouched unless `header`
// says otherwise), written to `path`.
void WriteWithHeader(const Container& c, const json& header, const fs::path& path) {
  const std::string h = header.dump();
  const uint64_t hl = h.size();
  std::string bytes(reinterpret_cast<const char*>(&hl), 8);
  bytes += h;
  bytes += c.data;
  WriteWhole(path, bytes);
}

// ContainerDataSha256 of the tensors `header` describes, read from `data` at the header's offsets.
std::string DataDigestOf(const json& header, const std::string& data) {
  std::vector<std::pair<std::string, std::string>> named;
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0].get<uint64_t>();
    const uint64_t e = it.value()["data_offsets"][1].get<uint64_t>();
    named.push_back({it.key(), Sha256Hex(data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b)))});
  }
  return ContainerDataSha256(named);
}

// `header` with its reuse_guard.data_sha256 recomputed for its own tensor directory over c's data:
// a baseline as a converter that wrote that directory would have recorded it, so the check under
// test is PlanReuse's, not the digest's.
json Redigest(const Container& c, json header) {
  header["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseDataKey] = DataDigestOf(header, c.data);
  return header;
}

// A copy of __metadata__.r4dx_convert_run<ptr> ("/reused_from/path"), null when absent.
json RunField(const Container& c, const std::string& ptr) {
  if (!c.ok) return json();
  return c.header.value(json::json_pointer("/__metadata__/r4dx_convert_run" + ptr), json());
}

// The tensor names whose bytes differ between `a` and `b`, or exist in one only.
std::set<std::string> DifferingTensors(const Container& a, const Container& b) {
  std::set<std::string> out;
  for (const auto& kv : a.tensors) {
    auto it = b.tensors.find(kv.first);
    if (it == b.tensors.end() || it->second != kv.second) out.insert(kv.first);
  }
  for (const auto& kv : b.tensors)
    if (!a.tensors.count(kv.first)) out.insert(kv.first);
  return out;
}

bool OwnedBy(const std::string& tensor, const std::set<std::string>& bases) {
  for (const auto& b : bases)
    if (tensor.compare(0, b.size() + 1, b + ".") == 0) return true;
  return false;
}

// A reuse output against the full run with the same flags: every data byte equal, and the header
// text equal to the full run's AS WRITTEN once r4dx_convert_run.reused_from is removed (and, with
// `ignore_threads`, `threads` set to the full run's).
bool SameAsFull(const Container& reuse, const Container& full, std::string* why,
                bool ignore_threads = false) {
  if (!reuse.ok || !full.ok) {
    *why = "a container is unreadable";
    return false;
  }
  if (reuse.data != full.data) {
    const std::set<std::string> d = DifferingTensors(reuse, full);
    *why = "data differs (" + std::to_string(d.size()) + " tensor(s), first " +
           (d.empty() ? std::string("?") : *d.begin()) + ")";
    return false;
  }
  json h = reuse.header;
  json& run = h["__metadata__"]["r4dx_convert_run"];
  if (!run.contains("reused_from")) {
    *why = "no reused_from";
    return false;
  }
  run.erase("reused_from");
  if (ignore_threads) run["threads"] = full.header["__metadata__"]["r4dx_convert_run"]["threads"];
  if (h.dump() != full.header_text) {
    *why = "header differs beyond reused_from";
    return false;
  }
  return true;
}

#if defined(R4DX_CONVERT_EXE)

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
int RunExe(const std::string& exe, const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + exe + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

std::string Q(const fs::path& p) { return "\"" + p.u8string() + "\""; }

// Command lines as ordered (flag, value) lists, so a case can replace or drop one flag.
using Opts = std::vector<std::pair<std::string, std::string>>;

Opts With(Opts o, const std::string& flag, const std::string& value) {
  for (auto& kv : o) {
    if (kv.first == flag) {
      kv.second = value;
      return o;
    }
  }
  o.push_back({flag, value});
  return o;
}

Opts Without(Opts o, const std::string& flag) {
  o.erase(std::remove_if(o.begin(), o.end(), [&](const auto& kv) { return kv.first == flag; }),
          o.end());
  return o;
}

std::string Join(const Opts& o) {
  std::string s;
  for (const auto& kv : o) s += " " + kv.first + (kv.second.empty() ? "" : " " + kv.second);
  return s;
}

// tools/quant2/group_sweep.ps1's -Recipe on the fixture, plus the draft head.
Opts RecipeOpts(const Fixture& f) {
  return {{"--input", Q(f.ckpt)},
          {"--layouts", "w4a16"},
          {"--lm-head", "w4a16"},
          {"--no-bf16", ""},
          {"--mtp", "on"},
          {"--vision", "on"},
          {"--kv-calib", Q(f.kv)},
          {"--quant", "search"},
          {"--imatrix", Q(f.imatrix)},
          {"--keep-bf16", "\"^text\\.layers\\.[0-9]+\\.attn\\.[kv]$\""},
          {"--draft-vocab-ids", Q(f.draft)},
          {"--threads", std::to_string(kThreads)}};
}

// (d): q2ab_ldlq's shape -- w4a16 only, residual rotation + Hadamards, every quantized linear LDLQ'd
// against the rms Hessians -- on the text stack (the synthetic Hessians cover no lm_head / MTP).
Opts LdlqOpts(const Fixture& f) {
  return {{"--input", Q(f.ckpt)},    {"--layouts", "w4a16"},  {"--lm-head", "bf16"},
          {"--no-bf16", ""},         {"--mtp", "off"},        {"--vision", "off"},
          {"--rotate", "q2ab"},      {"--hessian-dir", Q(f.hess)},
          {"--ldlq", "\".*\""},      {"--threads", std::to_string(kThreads)}};
}

struct Result {
  int rc = -1;
  std::string log;
  Container c;
};

class Runner {
 public:
  explicit Runner(const fs::path& root) : root_(root) {}

  Result Convert(const std::string& label, const Opts& opts, const fs::path& out,
                 const std::string& extra = "", const std::string& exe = R4DX_CONVERT_EXE) {
    std::error_code ec;
    fs::remove(out, ec);
    const fs::path log = root_ / "convert.log";
    const auto t = std::chrono::steady_clock::now();
    Result r;
    r.rc = RunExe(exe, Join(opts) + " --output " + Q(out) + extra, log);
    r.log = ReadWhole(log);
    r.c = ReadContainer(out);
    std::printf("     %s: rc=%d in %.1f s\n", label.c_str(), r.rc, Since(t));
    std::fflush(stdout);
    return r;
  }

  // A run that must be refused before any output exists, naming `needle` in its error.
  void Refused(const std::string& label, const Opts& opts, const std::string& extra,
               const std::string& needle, const std::string& exe = R4DX_CONVERT_EXE) {
    const fs::path out = root_ / "refused.r4dx";
    const Result r = Convert(label, opts, out, extra, exe);
    const bool ok = r.rc != 0 && !fs::exists(out) && Contains(r.log, needle);
    Gate(ok, "(c) refused, no output file, error names '" + needle + "': " + label);
    if (!ok) std::printf("%s\n", r.log.c_str());
  }

 private:
  fs::path root_;
};

std::set<std::string> RecomputedSet(const Container& c) {
  std::set<std::string> s;
  const json v = RunField(c, "/reused_from/linears_recomputed");
  if (v.is_array())
    for (const auto& b : v) s.insert(b.get<std::string>());
  return s;
}

void TestExe(const Fixture& f) {
  const fs::path& root = f.root;
  Runner run(root);
  std::string why;
  const std::string exe = R4DX_CONVERT_EXE;
  const Opts recipe = RecipeOpts(f);
  const std::string rule_r = " --w4a16-group-rule \"mlp\\.down$=32\" --w4a16-group-rule \"draft_head=32\"";
  const std::set<std::string> r_bases = {"text.layers.0.mlp.down", "text.layers.1.mlp.down",
                                         "mtp.mlp.down", "mtp.draft_head.lm_head"};
  const std::set<std::string> down_bases = {"text.layers.0.mlp.down", "text.layers.1.mlp.down",
                                            "mtp.mlp.down"};

  // ---- (a) -------------------------------------------------------------------------------------
  std::printf("---- (a) full(R) == reuse(R) from a no-rule baseline (the sweep's recipe) ----\n");
  const fs::path base0 = root / "base0.r4dx", full_r = root / "full_r.r4dx",
                 reuse_r = root / "reuse_r.r4dx", noguard = root / "noguard.r4dx";
  const Result b0 = run.Convert("baseline: no rule, --record-reuse-guard", recipe, base0,
                                " --record-reuse-guard");
  const Result fr = run.Convert("full: rule R, --record-reuse-guard", recipe, full_r,
                                rule_r + " --record-reuse-guard");
  const Result rr = run.Convert("reuse: rule R from the baseline", recipe, reuse_r,
                                rule_r + " --reuse-tensors-from " + Q(base0));
  Gate(b0.rc == 0 && fr.rc == 0 && rr.rc == 0 && b0.c.ok && fr.c.ok && rr.c.ok,
       "(a) baseline, full and reuse conversions exit 0");
  if (b0.rc != 0) std::printf("%s\n", b0.log.c_str());
  if (rr.rc != 0) std::printf("%s\n", rr.log.c_str());
  if (!(b0.c.ok && fr.c.ok && rr.c.ok)) return;
  {
    const std::set<std::string> d = DifferingTensors(b0.c, fr.c);
    bool only_r = !d.empty();
    for (const auto& t : d) only_r = only_r && OwnedBy(t, r_bases);
    Gate(only_r && d.count("text.layers.0.mlp.down.w4a16.wsz.g32") &&
             d.count("mtp.draft_head.lm_head.w4a16.wsz.g32"),
         "(a) premise: full(R) differs from the no-rule full run in " + std::to_string(d.size()) +
             " tensor(s), all of them R's four linears' (the test is not vacuous)");
  }
  Gate(SameAsFull(rr.c, fr.c, &why), "(a) reuse(R) == full(R): data and header text, reused_from aside" +
                                         (why.empty() ? "" : " -- " + why));
  {
    const size_t total = fr.c.tensors.size();
    Gate(RecomputedSet(rr.c) == r_bases && RunField(rr.c, "/reused_from/tensors_recomputed") == 9 &&
             RunField(rr.c, "/reused_from/tensors_copied") == total - 9 &&
             RunField(rr.c, "/reused_from/header_sha256") == Sha256Hex(b0.c.header_text) &&
             RunField(rr.c, "/reused_from/path") == base0.u8string(),
         "(a) reused_from: the four R linears (8 tensors + the draft head's vocab_ids) recomputed, "
         "the other " + std::to_string(total - 9) + " copied, the baseline's path and header sha256");
    Gate(Contains(rr.log, "matches this run's guard") && Contains(rr.log, "recomputed 4 linear(s)"),
         "(a) the reuse log says the guard matched and 4 linears were recomputed");
  }
  Gate(RunField(b0.c, "/reuse_guard/emit_complete") == 1 &&
           RunField(rr.c, "/reuse_guard/emit_complete") == 1 &&
           Contains(b0.c.header_text, "\"emit_complete\":1") &&
           !Contains(b0.c.header_text, "\"emit_complete\":0"),
       "(a) the guard's emit_complete is 1 on disk after a completed run (baseline and reuse output)");
  {
    const json d0 = RunField(b0.c, "/reuse_guard/data_sha256");
    Gate(d0.is_string() && d0 == DataDigestOf(b0.c.header, b0.c.data) &&
             RunField(fr.c, "/reuse_guard/data_sha256") == DataDigestOf(fr.c.header, fr.c.data) &&
             RunField(rr.c, "/reuse_guard/data_sha256") == RunField(fr.c, "/reuse_guard/data_sha256") &&
             d0 != RunField(fr.c, "/reuse_guard/data_sha256") &&
             RunField(rr.c, "/reused_from/data_sha256") == d0 &&
             !Contains(b0.c.header_text, ReuseDataPlaceholder()),
         "(a) data_sha256 on disk == the digest of the file's own tensors (baseline, full, reuse -- "
         "reuse == full), reused_from names the baseline's; no placeholder left");
    Gate(RunField(b0.c, "/reuse_guard/version") == kReuseGuardVersion &&
             RunField(b0.c, "/reuse_guard/converter/cpu").is_object() &&
             RunField(b0.c, "/reuse_guard/inputs/hessian_files") == "none",
         "(a) guard v" + std::to_string(kReuseGuardVersion) +
             ": converter.cpu recorded; inputs.hessian_files \"none\" without --ldlq");
    // The per-linear record: 17 linears (layer 0: 5, layer 1: 6, lm_head, MTP: 4, the draft head),
    // k/v "bf16" (the recipe's --keep-bf16), R's four at their groups, the rest at the default; the
    // keep regex's text is not part of the guard.
    const json lin0 = RunField(b0.c, "/reuse_guard/linears"), linr = RunField(fr.c, "/reuse_guard/linears");
    const std::string gdef = "w4a16.g" + std::to_string(kW4A16Group);
    Gate(lin0.is_object() && lin0.size() == 17 && lin0.value("text.layers.1.attn.k", "") == "bf16" &&
             lin0.value("text.layers.1.attn.v", "") == "bf16" &&
             lin0.value("text.layers.0.mlp.down", "") == gdef && lin0.value("lm_head", "") == gdef &&
             lin0.value("mtp.draft_head.lm_head", "") == gdef &&
             linr.value("text.layers.0.mlp.down", "") == "w4a16.g32" &&
             linr.value("mtp.mlp.down", "") == "w4a16.g32" &&
             linr.value("mtp.draft_head.lm_head", "") == "w4a16.g32" &&
             linr.value("text.layers.0.mlp.gate_up", "") == gdef &&
             RunField(b0.c, "/reuse_guard/args").is_object() &&
             !RunField(b0.c, "/reuse_guard/args").contains("keep_bf16"),
         "(a) reuse_guard.linears records all 17 linears' resolved layout sets (k/v bf16, R's four at "
         "32 in full(R)); args has no keep_bf16 -- " + lin0.dump());
  }
  const Result ng = run.Convert("full: no rule, no guard", recipe, noguard);
  {
    json h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"].erase("reuse_guard");
    Gate(ng.rc == 0 && ng.c.ok && ng.c.data == b0.c.data && ng.c.header_text == h.dump() &&
             RunField(ng.c, "/reuse_guard").is_null(),
         "(a) --record-reuse-guard adds exactly r4dx_convert_run.reuse_guard (same data; without it "
         "no guard key)");
  }

  // ---- (b) -------------------------------------------------------------------------------------
  std::printf("---- (b) the reverse and the other rule shapes ----\n");
  const Result r0 = run.Convert("reuse: no rule from full(R)", recipe, root / "reuse_0.r4dx",
                                " --reuse-tensors-from " + Q(full_r));
  Gate(r0.rc == 0 && SameAsFull(r0.c, b0.c, &why) && RecomputedSet(r0.c) == r_bases,
       "(b) reverse: reuse(no rule) from a baseline WITH R == the no-rule full run; R's four linears "
       "recomputed" + (why.empty() ? "" : " -- " + why));
  why.clear();
  const Result rc0 = run.Convert("reuse: no rule from reuse(R) (chained)", recipe,
                                 root / "reuse_0c.r4dx", " --reuse-tensors-from " + Q(reuse_r));
  Gate(rc0.rc == 0 && SameAsFull(rc0.c, b0.c, &why),
       "(b) a reuse output is itself a baseline: reuse(no rule) from reuse(R) == the no-rule full run" +
           (why.empty() ? "" : " -- " + why));
  why.clear();
  const std::string rule_alt = " --w4a16-group-rule \"gdn\\.out_proj$=32\"";
  const Result f_alt = run.Convert("full: gdn.out_proj=32, --record-reuse-guard", recipe,
                                   root / "full_alt.r4dx", rule_alt + " --record-reuse-guard");
  const Result r_alt = run.Convert("reuse: gdn.out_proj=32 from full(R)", recipe,
                                   root / "reuse_alt.r4dx", rule_alt + " --reuse-tensors-from " + Q(full_r));
  {
    std::set<std::string> alt_bases = r_bases;
    alt_bases.insert("text.layers.0.gdn.out_proj");
    Gate(f_alt.rc == 0 && r_alt.rc == 0 && SameAsFull(r_alt.c, f_alt.c, &why) &&
             RecomputedSet(r_alt.c) == alt_bases,
         "(b) a rule on other linears (gdn.out_proj=32) over a baseline with R == the full run; R's "
         "four (g32 -> default) and the out_proj (default -> g32) recomputed" +
             (why.empty() ? "" : " -- " + why));
  }
  why.clear();
  const std::string rule_def = " --w4a16-group-rule \"lm_head$=" + std::to_string(kW4A16Group) + "\"";
  const Result rdef = run.Convert("reuse: a rule naming the default group", recipe,
                                  root / "reuse_def.r4dx", rule_def + " --reuse-tensors-from " + Q(base0));
  Gate(rdef.rc == 0 && rdef.c.ok && rdef.c.data == b0.c.data && RecomputedSet(rdef.c).empty() &&
           RunField(rdef.c, "/reused_from/tensors_recomputed") == 0,
       "(b) a rule naming the default group recomputes nothing (every tensor copied, data == baseline)");
  const Result rmv = run.Convert("reuse: rule R, the checkpoint copied elsewhere", With(recipe, "--input", Q(f.ckpt_moved)),
                                 root / "reuse_moved.r4dx", rule_r + " --reuse-tensors-from " + Q(base0));
  Gate(rmv.rc == 0 && SameAsFull(rmv.c, fr.c, &why),
       "(b) a moved checkpoint (same bytes, another directory) is accepted and == full(R)" +
           (why.empty() ? "" : " -- " + why));
  why.clear();
  const Result rth = run.Convert("reuse: rule R, --threads 3", With(recipe, "--threads", "3"),
                                 root / "reuse_t3.r4dx", rule_r + " --reuse-tensors-from " + Q(base0));
  Gate(rth.rc == 0 && SameAsFull(rth.c, fr.c, &why, /*ignore_threads=*/true) &&
           RunField(rth.c, "/threads") == 3,
       "(b) another --threads is accepted and == full(R) apart from the recorded threads" +
           (why.empty() ? "" : " -- " + why));
  why.clear();

  // ---- (c) -------------------------------------------------------------------------------------
  std::printf("---- (c) the guard ----\n");
  const std::string from0 = rule_r + " --reuse-tensors-from " + Q(base0);
  run.Refused("--layers 1", With(recipe, "--layers", "1"), from0, "reuse_guard.args.layers:");
  run.Refused("--vision off", With(recipe, "--vision", "off"), from0, "reuse_guard.args.vision:");
  run.Refused("--mtp off", With(recipe, "--mtp", "off"), from0, "reuse_guard.args.mtp:");
  run.Refused("--lm-head w4a16,bf16", With(recipe, "--lm-head", "w4a16,bf16"), from0,
              "reuse_guard.args.lm_head.bf16:");
  run.Refused("without --no-bf16", Without(recipe, "--no-bf16"), from0, "reuse_guard.args.layouts.bf16:");
  run.Refused("--quant rtn (no --imatrix)", Without(With(recipe, "--quant", "rtn"), "--imatrix"), from0,
              "reuse_guard.args.quant:");
  // (--keep-bf16 another regex is NOT refused any more: the linears it changes are recomputed -- (e).)
  run.Refused("--ldlq (with --hessian-dir)",
              With(With(recipe, "--hessian-dir", Q(f.hess)), "--ldlq", "\"^text\\.layers\\.0\\.\""), from0,
              "reuse_guard.args.ldlq:");
  run.Refused("--rotate q2a", With(recipe, "--rotate", "q2a"), from0, "reuse_guard.args.rotate:");
  run.Refused("--imatrix another file", With(recipe, "--imatrix", Q(f.imatrix2)), from0,
              "reuse_guard.inputs.imatrix_sha256:");
  run.Refused("--kv-calib another file", With(recipe, "--kv-calib", Q(f.kv2)), from0,
              "reuse_guard.inputs.kv_calib_sha256:");
  run.Refused("--draft-vocab-ids another file", With(recipe, "--draft-vocab-ids", Q(f.draft2)), from0,
              "reuse_guard.inputs.draft_vocab_ids_sha256:");
  run.Refused("checkpoint: one bit of shard 2", With(recipe, "--input", Q(f.ckpt_shard)), from0,
              "reuse_guard.checkpoint.shards.model-00002-of-00002.safetensors.sha256:");
  run.Refused("checkpoint: config.json", With(recipe, "--input", Q(f.ckpt_cfg)), from0,
              "reuse_guard.checkpoint.config_sha256:");
  run.Refused("checkpoint: model.safetensors.index.json", With(recipe, "--input", Q(f.ckpt_idx)), from0,
              "reuse_guard.checkpoint.index_sha256:");
  {
    // Another binary: this one's bytes plus one trailing byte (PE files ignore trailing data).
    const fs::path alt_dir = root / "alt_exe";
    fs::create_directories(alt_dir);
    const fs::path alt = alt_dir / "r4dx-convert.exe";
    WriteWhole(alt, ReadWhole(fs::u8path(exe)) + std::string(1, '\0'));
    const fs::path probe_log = root / "probe.log";
    const int probe_rc = RunExe(alt.u8string(), "", probe_log);  // no args: its own usage error
    if (Contains(ReadWhole(probe_log), "r4dx-convert: error:")) {
      run.Refused("another converter binary", recipe, from0, "reuse_guard.converter.exe_sha256:",
                  alt.u8string());
    } else {
      std::printf("SKIP (c) another converter binary: the copied exe does not start here (rc=%d) -- "
                  "the exe_sha256 leaf is still covered by the mutation cases below\n",
                  probe_rc);
    }
  }
  run.Refused("baseline without a guard", recipe, rule_r + " --reuse-tensors-from " + Q(noguard),
              "has no __metadata__.r4dx_convert_run.reuse_guard");
  // ... from its header alone: nothing of this run's guard (the checkpoint, the Hessians) is hashed
  // for a baseline that could never be used.
  Gate(!Contains(ReadWhole(root / "convert.log"), "reuse guard: binary, checkpoint"),
       "(c) a baseline without a guard is refused before this run's guard is hashed");
  run.Refused("missing baseline", recipe, rule_r + " --reuse-tensors-from " + Q(root / "nope.r4dx"),
              "no such file");
  {
    const std::string before = ReadWhole(base0);
    const Result r = run.Convert("--output == the baseline", recipe, root / "unused.r4dx",
                                 rule_r + " --reuse-tensors-from " + Q(base0) + " --output " + Q(base0));
    Gate(r.rc != 0 && Contains(r.log, "--output is the baseline itself") && ReadWhole(base0) == before,
         "(c) --output == the baseline is refused and the baseline is untouched");
  }
  {
    const fs::path log = root / "selftest.log";
    const int rc = RunExe(exe, "--selftest --selftest-input x --selftest-output y --reuse-tensors-from " +
                                   Q(base0), log);
    Gate(rc != 0 && Contains(ReadWhole(log), "apply only to the HF-checkpoint conversion"),
         "(c) --reuse-tensors-from with --selftest is an argument error");
  }

  // Every recorded guard leaf, changed in a copy of the baseline: refused, naming the leaf.
  const fs::path mut = root / "mutated.r4dx";
  const std::string from_mut = rule_r + " --reuse-tensors-from " + Q(mut);
  auto mutated = [&](const Container& src, const json& header, const std::string& label,
                     const std::string& needle) {
    WriteWithHeader(src, header, mut);
    run.Refused(label, recipe, from_mut, needle);
  };
  {
    // Every leaf of the guard is a primitive (flatten() maps each to a JSON pointer), so changing
    // one and unflattening gives the guard with exactly that field different.
    const json flat = b0.c.header.at("__metadata__").at("r4dx_convert_run").at("reuse_guard").flatten();
    int leaves = 0;
    for (auto it = flat.begin(); it != flat.end(); ++it) {
      const std::string rel = it.key().substr(1);  // "/args/quant" -> "args/quant"
      if (rel == kReuseCompleteKey || rel == kReuseDataKey) continue;  // the completion fields: below
      json v = it.value();
      if (v.is_string()) v = v.get<std::string>() + "x";
      else if (v.is_boolean()) v = !v.get<bool>();
      else if (v.is_number_unsigned()) v = v.get<uint64_t>() + 1;
      else if (v.is_number_integer()) v = v.get<int64_t>() + 1;
      else if (v.is_number_float()) v = v.get<double>() * 2.0 + 1.0;
      else v = "x";
      json fl = flat;
      fl[it.key()] = v;
      json h = b0.c.header;
      h["__metadata__"]["r4dx_convert_run"]["reuse_guard"] = fl.unflatten();
      std::string dotted = rel;
      std::replace(dotted.begin(), dotted.end(), '/', '.');
      mutated(b0.c, h, "baseline's reuse_guard." + dotted + " changed", "reuse_guard." + dotted + ":");
      ++leaves;
    }
    Gate(leaves >= 30, "(c) " + std::to_string(leaves) + " guard leaves mutated one at a time");
  }
  {
    json h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"]["emit_complete"] = 0;
    mutated(b0.c, h, "unfinished baseline (emit_complete 0)", "emit pass never completed");
  }
  {
    json h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseDataKey] = ReuseDataPlaceholder();
    mutated(b0.c, h, "emit_complete 1 over the placeholder data_sha256", "is not a recorded digest");
    std::string d = b0.c.header["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseDataKey];
    d[0] = d[0] == 'f' ? '0' : 'f';
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseDataKey] = d;
    mutated(b0.c, h, "another data_sha256", "does not match its reuse_guard.data_sha256");
  }
  {
    // A completed baseline copied by a tool that pre-sizes the destination and died half way: full
    // length, header intact (emit_complete 1), the second half of the data zeros.
    Container half = b0.c;
    std::fill(half.data.begin() + static_cast<std::ptrdiff_t>(half.data.size() / 2), half.data.end(), '\0');
    WriteWithHeader(half, b0.c.header, mut);
    Gate(fs::file_size(mut) == fs::file_size(base0), "(c) the half-zeroed copy has the baseline's length");
    run.Refused("a completed baseline, second half of its data zeroed", recipe, from_mut,
                "does not match its reuse_guard.data_sha256");
    // ... and one byte flipped deep inside one tensor.
    Container flip = b0.c;
    flip.data[flip.data.size() / 3] = static_cast<char>(flip.data[flip.data.size() / 3] ^ 0x10);
    WriteWithHeader(flip, b0.c.header, mut);
    run.Refused("a completed baseline, one data bit flipped", recipe, from_mut,
                "does not match its reuse_guard.data_sha256");
  }
  {
    std::string bytes(8, '\0');
    const uint64_t hl = 16;
    std::memcpy(&bytes[0], &hl, 8);
    bytes += "not json at all!";
    WriteWhole(mut, bytes);
    run.Refused("a baseline whose header is not JSON", recipe, from_mut,
                "--reuse-tensors-from: " + mut.u8string() + " is not a");
  }
  {
    json h = b0.c.header;
    h["__metadata__"]["quant"]["w4a16"]["group"] = 32;
    mutated(b0.c, h, "quant block of another build", "__metadata__.quant");
  }
  {
    json h = b0.c.header;
    h["__metadata__"]["quant"]["w4a16"]["groups"] = {{"text.layers.9.mlp.down", 32}};
    mutated(b0.c, h, "group map naming a linear this run does not write",
            "lists 'text.layers.9.mlp.down', which this run does not write as a linear");
  }
  // The directory cases carry a data_sha256 recomputed for their own directory (Redigest), so what
  // refuses them is PlanReuse; without it the digest already would.
  {
    json h = b0.c.header;
    h.erase("text.final_norm");
    mutated(b0.c, h, "a tensor missing from the baseline (digest not updated)",
            "does not match its reuse_guard.data_sha256");
    mutated(b0.c, Redigest(b0.c, h), "a tensor missing from the baseline",
            "this run writes 'text.final_norm'");
  }
  {
    json h = b0.c.header;
    h["bogus.tensor"] = {{"dtype", "U8"}, {"shape", {2}}, {"data_offsets", {0, 2}}};
    mutated(b0.c, Redigest(b0.c, h), "an extra tensor in the baseline",
            "the baseline has tensor 'bogus.tensor'");
  }
  {
    json h = b0.c.header;
    h["text.final_norm"]["shape"] = {kHidden / 2, 4};
    mutated(b0.c, h, "a tensor of another shape", "tensor 'text.final_norm' is U8 [2560,4]");
  }
  {
    // full(R)'s map entry removed: the map says default, the record (and the tensors) say g32.
    json h = fr.c.header;
    h["__metadata__"]["quant"]["w4a16"]["groups"].erase("text.layers.0.mlp.down");
    WriteWithHeader(fr.c, h, mut);
    run.Refused("group map without a linear the record has at g32", recipe,
                " --reuse-tensors-from " + Q(mut),
                "quant.w4a16.groups maps 'text.layers.0.mlp.down' to nothing (the default group");
    // ... and saying 64 where the record (and the tensor) say g32.
    h = fr.c.header;
    h["__metadata__"]["quant"]["w4a16"]["groups"]["text.layers.0.mlp.down"] = 64;
    WriteWithHeader(fr.c, h, mut);
    run.Refused("group map disagreeing with the record's group", recipe,
                " --reuse-tensors-from " + Q(mut),
                "maps 'text.layers.0.mlp.down' to group 64 but its reuse_guard.linears records "
                "\"w4a16.g32\"");
  }
  {
    // The per-linear record itself: missing, naming a linear this run does not write, missing one.
    json h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"].erase(kReuseLinearsKey);
    mutated(b0.c, h, "a guard without the per-linear record", "reuse_guard.linears is missing");
    h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseLinearsKey]["text.layers.9.mlp.down"] =
        "w4a16.g" + std::to_string(kW4A16Group);
    mutated(b0.c, h, "a record naming a linear this run does not write",
            "reuse_guard.linears.text.layers.9.mlp.down: baseline");
    h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseLinearsKey].erase("lm_head");
    mutated(b0.c, h, "a record without one of this run's linears",
            "reuse_guard.linears.lm_head: baseline (absent)");
    // The keep list disagreeing with the record (attn.k kept, the record says quantized).
    h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseLinearsKey]["text.layers.1.attn.k"] =
        "w4a16.g" + std::to_string(kW4A16Group);
    mutated(b0.c, h, "keep_bf16_linears disagreeing with the record",
            "keep_bf16_linears lists \"text.layers.1.attn.k\" but its reuse_guard.linears records");
    // A valid record that lies about an AFFECTED linear (rule R recomputes mlp.down): the baseline's
    // tensors must be the ones its record names.
    h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseLinearsKey]["text.layers.0.mlp.down"] = "bf16";
    mutated(b0.c, h, "a record naming tensors the baseline does not have (recomputed linear)",
            "records 'text.layers.0.mlp.down' as \"bf16\" (reuse_guard.linears) but has no tensor "
            "'text.layers.0.mlp.down.bf16.w'");
    // ... and about a COPIED one, group map edited to agree: the directory refuses it.
    h = b0.c.header;
    h["__metadata__"]["r4dx_convert_run"]["reuse_guard"][kReuseLinearsKey]["text.layers.0.gdn.in_proj_z"] =
        "w4a16.g32";
    h["__metadata__"]["quant"]["w4a16"]["groups"] = {{"text.layers.0.gdn.in_proj_z", 32}};
    WriteWithHeader(b0.c, h, mut);
    run.Refused("record + group map both claiming g32 for a g64 linear this run copies", recipe,
                rule_r + " --w4a16-group-rule \"in_proj_z$=32\" --reuse-tensors-from " + Q(mut),
                "this run writes 'text.layers.0.gdn.in_proj_z.w4a16.wsz.g32' but the baseline has no "
                "such tensor");
  }
  {
    const std::string b = ReadWhole(base0);
    WriteWhole(mut, b.substr(0, b.size() - 10));
    run.Refused("truncated baseline", recipe, from_mut, "truncated");
  }
}

// The container's r4dx_convert_run.keep_bf16_linears as a list (container order).
std::vector<std::string> KeptList(const Container& c) {
  std::vector<std::string> v;
  const json k = RunField(c, "/keep_bf16_linears");
  if (k.is_array())
    for (const auto& b : k) v.push_back(b.get<std::string>());
  return v;
}

// (e) i-iii: --keep-bf16 differing between the baseline and this run, on the sweep's recipe. Uses
// (a)'s base0.r4dx (keep k/v, no rule) and full_r.r4dx (keep k/v, rule R) from `root`.
void TestKeep(const Fixture& f) {
  std::printf("---- (e) --keep-bf16 differs: reuse == full (sweep recipe) ----\n");
  const fs::path& root = f.root;
  Runner run(root);
  std::string why;
  const Opts recipe = RecipeOpts(f);
  const fs::path base0 = root / "base0.r4dx", full_r = root / "full_r.r4dx";
  const Container b0 = ReadContainer(base0), fr = ReadContainer(full_r);
  if (!Gate(b0.ok && fr.ok, "(e) (a)'s baseline and full(R) are there")) return;
  const std::string from0 = " --reuse-tensors-from " + Q(base0);
  const std::string gdef = "w4a16.g" + std::to_string(kW4A16Group);
  const std::string l0down = "text.layers.0.mlp.down";

  // (i) A keep added: the recipe's k/v plus layer 0's mlp.down, as ONE merged regex.
  const Opts k1 = With(recipe, "--keep-bf16", "\"^text\\.layers\\.(?:[0-9]+\\.attn\\.[kv]|0\\.mlp\\.down)$\"");
  const fs::path full_k1 = root / "full_k1.r4dx";
  const Result fk1 = run.Convert("full: keep k/v + layer 0 mlp.down, --record-reuse-guard", k1, full_k1,
                                 " --record-reuse-guard");
  const Result rk1 = run.Convert("reuse: keep k/v + layer 0 mlp.down from the baseline", k1,
                                 root / "reuse_k1.r4dx", from0);
  if (!Gate(fk1.rc == 0 && rk1.rc == 0 && fk1.c.ok && rk1.c.ok, "(e-i) full and reuse exit 0")) {
    std::printf("%s\n%s\n", fk1.log.c_str(), rk1.log.c_str());
    return;
  }
  {
    const std::set<std::string> d = DifferingTensors(b0, fk1.c);
    bool only = !d.empty();
    for (const auto& t : d) only = only && OwnedBy(t, {l0down});
    const int64_t nk = kHidden * kInter;
    const int64_t per = 2 * nk - (nk / 2 + nk / kW4A16Group * 4);
    Gate(only && d.count(l0down + ".bf16.w") && d.count(l0down + ".w4a16.wq") &&
             !fk1.c.tensors.count(l0down + ".w4a16.wq") &&
             KeptList(fk1.c) == std::vector<std::string>{l0down, "text.layers.1.attn.k",
                                                         "text.layers.1.attn.v"} &&
             RunField(fk1.c, "/reuse_guard/linears/" + l0down) == "bf16" &&
             RunField(fk1.c, "/keep_bf16_extra_bytes").get<int64_t>() -
                     RunField(b0, "/keep_bf16_extra_bytes").get<int64_t>() == per,
         "(e-i) premise: the added keep changes only layer 0 mlp.down (bf16.w only, no w4a16), the keep "
         "list gains it, and keep_bf16_extra_bytes grows by exactly 2NK - w4a16(g" +
             std::to_string(kW4A16Group) + ") = " + std::to_string(per) + " B");
  }
  Gate(SameAsFull(rk1.c, fk1.c, &why) && RecomputedSet(rk1.c) == std::set<std::string>{l0down} &&
           Contains(rk1.log, "reuse: recompute " + l0down + " (baseline " + gdef + " -> bf16)"),
       "(e-i) a keep added: reuse == full, only layer 0 mlp.down recomputed (" + gdef + " -> bf16)" +
           (why.empty() ? "" : " -- " + why));
  why.clear();

  // (ii) A keep removed: only attn.k kept, attn.v quantized (LDLQ-free recipe: search + imatrix).
  const Opts k2 = With(recipe, "--keep-bf16", "\"^text\\.layers\\.[0-9]+\\.attn\\.k$\"");
  const Result fk2 = run.Convert("full: keep attn.k only, --record-reuse-guard", k2, root / "full_k2.r4dx",
                                 " --record-reuse-guard");
  const Result rk2 = run.Convert("reuse: keep attn.k only from the baseline", k2, root / "reuse_k2.r4dx", from0);
  Gate(fk2.rc == 0 && rk2.rc == 0 && SameAsFull(rk2.c, fk2.c, &why) &&
           RecomputedSet(rk2.c) == std::set<std::string>{"text.layers.1.attn.v"} &&
           fk2.c.tensors.count("text.layers.1.attn.v.w4a16.wq") &&
           !fk2.c.tensors.count("text.layers.1.attn.v.bf16.w") &&
           Contains(rk2.log, "reuse: recompute text.layers.1.attn.v (baseline bf16 -> " + gdef + ")"),
       "(e-ii) a keep removed: reuse == full, only attn.v recomputed (bf16 -> " + gdef + ")" +
           (why.empty() ? "" : " -- " + why));
  why.clear();
  const Result rk0 = run.Convert("reuse: the recipe's keep from full(keep + layer 0 mlp.down)", recipe,
                                 root / "reuse_k0.r4dx", " --reuse-tensors-from " + Q(full_k1));
  Gate(rk0.rc == 0 && SameAsFull(rk0.c, b0, &why) && RecomputedSet(rk0.c) == std::set<std::string>{l0down},
       "(e-ii) reverse of (i): reuse(recipe) from a baseline keeping one more linear == the baseline's "
       "full run" + (why.empty() ? "" : " -- " + why));
  why.clear();
  {
    // Another spelling of the SAME keep set: nothing to recompute; the header differs from the
    // baseline's only in the recorded pattern (and reused_from).
    const std::string alt = "\"attn\\.[kv]$\"";
    const Result rs = run.Convert("reuse: the same keep set spelled another way", With(recipe, "--keep-bf16", alt),
                                  root / "reuse_same.r4dx", from0);
    json h = rs.c.header;
    bool ok = rs.rc == 0 && rs.c.ok && rs.c.data == b0.data && RecomputedSet(rs.c).empty() &&
              RunField(rs.c, "/reused_from/tensors_recomputed") == 0 &&
              RunField(rs.c, "/keep_bf16") == "attn\\.[kv]$";
    if (ok) {
      h["__metadata__"]["r4dx_convert_run"].erase("reused_from");
      h["__metadata__"]["r4dx_convert_run"]["keep_bf16"] = RunField(b0, "/keep_bf16");
      ok = h.dump() == b0.header_text;
    }
    Gate(ok, "(e-ii) another spelling of the same keep set recomputes nothing; data == the baseline's, "
             "header == the baseline's apart from the pattern text and reused_from");
  }

  // (iii) A keep and a group rule on the same class: layer 0's mlp.down kept (keep wins over the
  // rule), layer 1's and the MTP's at 32.
  const Opts k1r = k1;
  const std::string rule32 = " --w4a16-group-rule \"mlp\\.down$=32\"";
  const Result fkr = run.Convert("full: keep + layer 0 mlp.down, mlp.down=32, --record-reuse-guard", k1r,
                                 root / "full_k1r.r4dx", rule32 + " --record-reuse-guard");
  const Result rkr0 = run.Convert("reuse: keep + rule from the no-rule baseline", k1r, root / "reuse_k1r0.r4dx",
                                  rule32 + from0);
  const Result rkrr = run.Convert("reuse: keep + rule from full(R)", k1r, root / "reuse_k1rr.r4dx",
                                  rule32 + " --reuse-tensors-from " + Q(full_r));
  if (!Gate(fkr.rc == 0 && rkr0.rc == 0 && rkrr.rc == 0, "(e-iii) full and both reuses exit 0")) {
    std::printf("%s\n%s\n%s\n", fkr.log.c_str(), rkr0.log.c_str(), rkrr.log.c_str());
    return;
  }
  Gate(RunField(fkr.c, "/w4a16_groups") == json{{"mtp.mlp.down", 32}, {"text.layers.1.mlp.down", 32}} &&
           RunField(fkr.c, "/reuse_guard/linears/" + l0down) == "bf16",
       "(e-iii) premise: keep wins over the rule for layer 0 (bf16, not in the group map); layer 1 and "
       "the MTP's mlp.down at 32");
  Gate(SameAsFull(rkr0.c, fkr.c, &why) &&
           RecomputedSet(rkr0.c) ==
               std::set<std::string>{l0down, "text.layers.1.mlp.down", "mtp.mlp.down"},
       "(e-iii) keep + rule from the no-rule baseline == full; the three mlp.down recomputed" +
           (why.empty() ? "" : " -- " + why));
  why.clear();
  Gate(SameAsFull(rkrr.c, fkr.c, &why) &&
           RecomputedSet(rkrr.c) == std::set<std::string>{l0down, "mtp.draft_head.lm_head"} &&
           Contains(rkrr.log, "reuse: recompute " + l0down + " (baseline w4a16.g32 -> bf16)") &&
           Contains(rkrr.log, "reuse: recompute mtp.draft_head.lm_head (baseline w4a16.g32 -> " + gdef + ")"),
       "(e-iii) keep + rule from full(R) == full: layer 0 mlp.down (g32 -> bf16) and the draft head "
       "(R's g32 -> default) recomputed, layer 1 / MTP mlp.down (g32 both) copied" +
           (why.empty() ? "" : " -- " + why));
  why.clear();
  // A keep change does not open the guard for anything else.
  run.Refused("keep change + --quant rtn (no --imatrix)", Without(With(k1, "--quant", "rtn"), "--imatrix"),
              from0, "reuse_guard.args.quant:");
  run.Refused("keep change + --kv-calib another file", With(k1, "--kv-calib", Q(f.kv2)), from0,
              "reuse_guard.inputs.kv_calib_sha256:");
}

// Raw tensor bytes of the synthetic checkpoint (both shards), by HF name.
std::map<std::string, std::string> CheckpointTensors(const fs::path& ckpt) {
  std::map<std::string, std::string> m;
  for (const char* s : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}) {
    const Container c = ReadContainer(ckpt / s);
    m.insert(c.tensors.begin(), c.tensors.end());
  }
  return m;
}

void TestLdlqRotation(const Fixture& f) {
  std::printf("---- (d) LDLQ + q2ab rotation ----\n");
  const fs::path& root = f.root;
  Runner run(root);
  std::string why;
  const Opts opts = LdlqOpts(f);
  const std::string rule = " --w4a16-group-rule \"mlp\\.down$=32\"";
  const std::set<std::string> down = {"text.layers.0.mlp.down", "text.layers.1.mlp.down"};
  const fs::path d0p = root / "ldlq_base.r4dx", drp = root / "ldlq_full_r.r4dx";
  const Result d0 = run.Convert("ldlq baseline: no rule, --record-reuse-guard", opts, d0p,
                                " --record-reuse-guard");
  const Result dr = run.Convert("ldlq full: mlp.down=32, --record-reuse-guard", opts, drp,
                                rule + " --record-reuse-guard");
  const Result du = run.Convert("ldlq reuse: mlp.down=32 from the baseline", opts,
                                root / "ldlq_reuse_r.r4dx", rule + " --reuse-tensors-from " + Q(d0p));
  const Result db = run.Convert("ldlq reuse: no rule from full(mlp.down=32)", opts,
                                root / "ldlq_reuse_0.r4dx", " --reuse-tensors-from " + Q(drp));
  Gate(d0.rc == 0 && dr.rc == 0 && du.rc == 0 && db.rc == 0, "(d) all four conversions exit 0");
  if (d0.rc != 0) std::printf("%s\n", d0.log.c_str());
  if (du.rc != 0) std::printf("%s\n", du.log.c_str());
  if (!(d0.c.ok && dr.c.ok && du.c.ok && db.c.ok)) return;
  Gate(RunField(d0.c, "/rotate") == "q2ab" && RunField(d0.c, "/ldlq_linears").size() == 11 &&
           RunField(d0.c, "/ldlq_rms_linears").size() == 7,
       "(d) the baseline is rotated (q2ab) with 11 LDLQ'd linears, 7 against the rms Hessians");
  {
    const std::set<std::string> d = DifferingTensors(d0.c, dr.c);
    bool only = !d.empty();
    for (const auto& t : d) only = only && OwnedBy(t, down);
    Gate(only, "(d) premise: the rule changes only the two mlp.down linears' tensors (" +
                   std::to_string(d.size()) + ")");
  }
  Gate(SameAsFull(du.c, dr.c, &why) && RecomputedSet(du.c) == down,
       "(d) reuse(mlp.down=32) == full(mlp.down=32); only the two mlp.down recomputed" +
           (why.empty() ? "" : " -- " + why));
  why.clear();
  Gate(SameAsFull(db.c, d0.c, &why) && RecomputedSet(db.c) == down,
       "(d) reverse: reuse(no rule) from full(mlp.down=32) == the no-rule full run" +
           (why.empty() ? "" : " -- " + why));
  Gate(Contains(du.log, "ldlq: text.layers.0.mlp.down [5120,512]") &&
           Contains(du.log, "ldlq: text.layers.1.mlp.down [5120,512]") &&
           !Contains(du.log, "ldlq: text.layers.0.gdn.in_proj_qkv") &&
           !Contains(du.log, "ldlq: text.layers.1.attn.qg") &&
           Contains(du.log, "ldlq: 2 linear(s) quantized with LDLQ"),
       "(d) the reuse run factors and rounds only the two recomputed mlp.down linears");
  why.clear();
  // attn.k shares its tap (and, rotated, its Q^T H_rms Q factor) with attn.qg / attn.v: the full run
  // takes it from HessianStore's cache, the reuse run -- attn.qg copied -- factors it afresh. The
  // bytes must not care.
  const std::string rule_k = " --w4a16-group-rule \"attn\\.k$=32\"";
  const Result dkf = run.Convert("ldlq full: attn.k=32, --record-reuse-guard", opts,
                                 root / "ldlq_full_k.r4dx", rule_k + " --record-reuse-guard");
  const Result dku = run.Convert("ldlq reuse: attn.k=32 from the baseline", opts,
                                 root / "ldlq_reuse_k.r4dx", rule_k + " --reuse-tensors-from " + Q(d0p));
  const std::string k_line = "ldlq: text.layers.1.attn.k [64,5120] rms Hessian in.rms.hess: factor";
  Gate(dkf.rc == 0 && dku.rc == 0 && SameAsFull(dku.c, dkf.c, &why) &&
           RecomputedSet(dku.c) == std::set<std::string>{"text.layers.1.attn.k"} &&
           Contains(LineWith(dkf.log, k_line), "(shared tap, cached)") &&
           !LineWith(dku.log, k_line).empty() && !Contains(LineWith(dku.log, k_line), "cached"),
       "(d) a linear whose shared-tap factor the full run took from the cache and the reuse run "
       "recomputed: reuse(attn.k=32) == full(attn.k=32)" + (why.empty() ? "" : " -- " + why));
  const std::string from = rule + " --reuse-tensors-from " + Q(d0p);
  run.Refused("ldlq: --ldlq another regex", With(opts, "--ldlq", "\"mlp\\.\""), from, "reuse_guard.args.ldlq:");
  run.Refused("ldlq: --ldlq-damp 0.02", With(opts, "--ldlq-damp", "0.02"), from, "reuse_guard.args.ldlq_damp:");
  run.Refused("ldlq: another Hessian set", With(opts, "--hessian-dir", Q(f.hess2)), from,
              "reuse_guard.inputs.hessian_manifest_sha256:");
  run.Refused("ldlq: --rotate q2a", With(opts, "--rotate", "q2a"), from, "reuse_guard.args.rotate:");
  run.Refused("ldlq: --rotation-seed 12345", With(opts, "--rotation-seed", "12345"), from,
              "reuse_guard.args.rotation_seed:");

  // The Hessian payloads themselves. hessian.json pins each file's K, rows and header trace, and
  // ReadHessFile ties the trace to the diagonal -- so an in-place edit of the off-diagonals passes
  // every Hessian check with the manifest (and its sha256) unchanged. Only the guard's hashes of the
  // .hess files catch it, and they must: a reuse would copy linears rounded against the old H.
  {
    const json hf = RunField(d0.c, "/reuse_guard/inputs/hessian_files");
    Gate(hf.is_object() && hf.size() == 7 && hf.contains("mlp_mid.hess") &&
             hf.contains("in.rms.hess") && hf["mlp_mid.hess"]["bytes"] == 64 + 4 * kInter * (kInter + 1) / 2,
         "(d) the guard hashes every .hess file --ldlq \".*\" can read (7: keys and rms_keys files)");
  }
  {
    const fs::path he = root / "hess_edit";
    LinkDir(f.hess, he, {"mlp_mid.hess"});
    std::string b = ReadWhole(f.hess / "mlp_mid.hess");
    float* p = reinterpret_cast<float*>(&b[64]);
    size_t off = 0;
    for (int64_t i = 0; i < kInter; ++i) {
      for (int64_t j = 1; j < kInter - i; ++j) p[off + static_cast<size_t>(j)] *= 0.5f;  // diagonal kept
      off += static_cast<size_t>(kInter - i);
    }
    WriteWhole(he / "mlp_mid.hess", b);
    const Result fe = run.Convert("ldlq full: mlp.down=32, mlp_mid.hess edited in place",
                                  With(opts, "--hessian-dir", Q(he)), root / "ldlq_full_edit.r4dx",
                                  rule + " --record-reuse-guard");
    Gate(fe.rc == 0 && fe.c.ok &&
             RunField(fe.c, "/hessian_manifest_sha256") == RunField(dr.c, "/hessian_manifest_sha256") &&
             fe.c.tensors.at("text.layers.1.mlp.down.w4a16.wq") !=
                 dr.c.tensors.at("text.layers.1.mlp.down.w4a16.wq"),
         "(d) premise: the edited payload passes every Hessian check (same hessian.json sha256) and "
         "changes the mlp.down bytes");
    run.Refused("ldlq: a .hess payload edited in place, hessian.json unchanged",
                With(opts, "--hessian-dir", Q(he)), from,
                "reuse_guard.inputs.hessian_files.mlp_mid.hess.sha256:");
  }
  {
    // Two same-K, same-rows files swapped under an unchanged manifest: CheckFile holds each header's
    // trace to hessian.json's exactly, so even a full run refuses it at planning.
    const fs::path hs = root / "hess_swap";
    LinkDir(f.hess, hs, {"in.rms.hess", "mlp_in.rms.hess"});
    LinkOrCopy(f.hess / "mlp_in.rms.hess", hs / "in.rms.hess");
    LinkOrCopy(f.hess / "in.rms.hess", hs / "mlp_in.rms.hess");
    run.Refused("ldlq: in.rms.hess and mlp_in.rms.hess swapped, hessian.json unchanged (full run)",
                With(opts, "--hessian-dir", Q(hs)), "", "header says trace=");
  }

  // (e-iv) --keep-bf16 on the rotated q2ab + LDLQ container: layer 0 mlp.down (Q^T W Hb, the down
  // Hadamard), layer 1 attn.o (Q^T W Hb, the o Hadamard) and attn.k (W diag(1+w) Q, sharing the
  // input_layernorm rms tap with attn.qg / attn.v) written as bf16 of the folded fp32.
  std::printf("---- (e-iv) --keep-bf16 differs on q2ab + LDLQ ----\n");
  const std::set<std::string> kb = {"text.layers.0.mlp.down", "text.layers.1.attn.k",
                                    "text.layers.1.attn.o"};
  const Opts ko = With(opts, "--keep-bf16", "\"^text\\.layers\\.(?:0\\.mlp\\.down|1\\.attn\\.[ko])$\"");
  const fs::path kfp = root / "ldlq_full_keep.r4dx";
  const Result kf = run.Convert("ldlq full: keep 0.mlp.down + 1.attn.k/o, --record-reuse-guard", ko, kfp,
                                " --record-reuse-guard");
  const Result ku = run.Convert("ldlq reuse: the keep from the no-keep baseline", ko,
                                root / "ldlq_reuse_keep.r4dx", " --reuse-tensors-from " + Q(d0p));
  const Result kr = run.Convert("ldlq reuse: no keep from full(keep)", opts, root / "ldlq_reuse_unkeep.r4dx",
                                " --reuse-tensors-from " + Q(kfp));
  if (!Gate(kf.rc == 0 && ku.rc == 0 && kr.rc == 0 && kf.c.ok && ku.c.ok && kr.c.ok,
            "(e-iv) full(keep) and both reuses exit 0")) {
    std::printf("%s\n%s\n%s\n", kf.log.c_str(), ku.log.c_str(), kr.log.c_str());
    return;
  }
  {
    const std::set<std::string> d = DifferingTensors(d0.c, kf.c);
    bool only = !d.empty();
    for (const auto& t : d) only = only && OwnedBy(t, kb);
    bool bf16_only = true;
    for (const auto& b : kb)
      bf16_only = bf16_only && kf.c.tensors.count(b + ".bf16.w") && !kf.c.tensors.count(b + ".w4a16.wq");
    Gate(only && bf16_only && RunField(kf.c, "/ldlq_linears").size() == 8 &&
             RunField(kf.c, "/ldlq_rms_linears").size() == 6 && KeptList(kf.c).size() == 3 &&
             RunField(kf.c, "/reuse_guard/linears/text.layers.1.attn.o") == "bf16" &&
             RunField(kf.c, "/reuse_guard/inputs/hessian_files") ==
                 RunField(d0.c, "/reuse_guard/inputs/hessian_files"),
         "(e-iv) premise: only the three kept linears change (bf16.w only); LDLQ drops them (8 of 11 "
         "LDLQ'd, 6 against rms Hessians); the guard's .hess set is the same (it follows --ldlq only)");
    // The kept bf16 is the FOLDED weight, not the checkpoint's bytes: the Hadamard / Q folds ran.
    const std::map<std::string, std::string> raw = CheckpointTensors(f.ckpt);
    const std::string& o = kf.c.tensors.at("text.layers.1.attn.o.bf16.w");
    const std::string& dn = kf.c.tensors.at("text.layers.0.mlp.down.bf16.w");
    const std::string& k = kf.c.tensors.at("text.layers.1.attn.k.bf16.w");
    const std::string& o_raw = raw.at(std::string(kL1) + "self_attn.o_proj.weight");
    const std::string& dn_raw = raw.at(std::string(kL0) + "mlp.down_proj.weight");
    const std::string& k_raw = raw.at(std::string(kL1) + "self_attn.k_proj.weight");
    Gate(o.size() == o_raw.size() && o != o_raw && dn.size() == dn_raw.size() && dn != dn_raw &&
             k.size() == k_raw.size() && k != k_raw,
         "(e-iv) the kept attn.o / mlp.down / attn.k bf16 are the rotated (folded) weights: same size "
         "as the checkpoint's, different bytes");
  }
  Gate(SameAsFull(ku.c, kf.c, &why) && RecomputedSet(ku.c) == kb &&
           Contains(ku.log, "ldlq: 0 linear(s) quantized with LDLQ"),
       "(e-iv) reuse(keep) from the no-keep baseline == full(keep); the three recomputed as bf16, no "
       "LDLQ" + (why.empty() ? "" : " -- " + why));
  why.clear();
  const std::string k_rms = "ldlq: text.layers.1.attn.k [64,5120] rms Hessian in.rms.hess: factor";
  Gate(SameAsFull(kr.c, d0.c, &why) && RecomputedSet(kr.c) == kb &&
           Contains(kr.log, "ldlq: 3 linear(s) quantized with LDLQ") &&
           Contains(LineWith(d0.log, k_rms), "(shared tap, cached)") &&
           !LineWith(kr.log, k_rms).empty() && !Contains(LineWith(kr.log, k_rms), "cached"),
       "(e-iv) reverse: un-keeping them from full(keep) == the no-keep full run; exactly those three "
       "LDLQ'd, attn.k's rms factor fresh (cached in the full run)" + (why.empty() ? "" : " -- " + why));
  why.clear();
  run.Refused("ldlq: keep change + --ldlq-damp 0.02", With(ko, "--ldlq-damp", "0.02"),
              " --reuse-tensors-from " + Q(d0p), "reuse_guard.args.ldlq_damp:");
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main(int argc, char** argv) {
  // `convert_reuse --make-fixture <dir>`: writes (a)-(d)'s checkpoint and input files into <dir>
  // and keeps them (e.g. to drive tools/quant2/group_sweep.ps1 -Convert on something small).
  if (argc == 3 && std::string(argv[1]) == "--make-fixture") {
    const Fixture f = MakeFixture(fs::u8path(argv[2]));
    std::printf("checkpoint %s\nimatrix %s\nkv-calib %s\ndraft-vocab-ids %s\nhessian-dir %s\n",
                f.ckpt.u8string().c_str(), f.imatrix.u8string().c_str(), f.kv.u8string().c_str(),
                f.draft.u8string().c_str(), f.hess.u8string().c_str());
    return 0;
  }
  try {
    TestLibrary();
#if defined(R4DX_CONVERT_EXE)
    const fs::path root = TempDir("exe");
    try {
      const auto t = std::chrono::steady_clock::now();
      const Fixture f = MakeFixture(root);
      std::printf("     fixture written in %.1f s\n", Since(t));
      TestExe(f);
      TestKeep(f);
      TestLdlqRotation(f);
    } catch (const std::exception& e) {
      Gate(false, std::string("(a-d) threw unexpectedly: ") + e.what());
    }
    std::error_code ec;
    fs::remove_all(root, ec);
#else
    std::printf("SKIP (a)-(d) r4dx-convert is not part of this build (R4DX_BUILD_CONVERT off)\n");
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures) {
    std::fprintf(stderr, "convert_reuse: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("convert_reuse: all gates passed\n");
  return 0;
}
