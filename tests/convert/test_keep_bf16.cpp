// tests/convert/test_keep_bf16.cpp -- r4dx-convert's `--keep-bf16 <regex>`
// (src/convert/include/r4dx_convert/keep_bf16.hpp, docs/validation.md "Milestone 11 /
// sensitivity"). The flag exists to build per-tensor-class sensitivity containers: everything
// quantized except one class, which stays bf16, so the KL difference against the all-quantized
// baseline is that class's share of the error. Three things have to hold for that measurement to
// mean anything, and this test gates all three on the CPU, with no checkpoint and no GPU:
//
//   1. SELECTION: the regex picks exactly the intended bases. It is a regex_SEARCH, which is what
//      makes "attn\.o$" select every layer's attention output without the caller writing out a
//      `^text\.layers\.\d+\.` prefix -- and is also why an unanchored pattern can over-select, so
//      the cases below pin both behaviours down rather than leaving them to the reader.
//   2. EMISSION: a matched linear is written as `<base>.bf16.w` and NOTHING else. If a quantized
//      layout leaked through, the loader (container.cpp's LoadQuantLinearWithFallback tier 1)
//      would silently pick it and the whole experiment would measure the baseline again.
//   3. ACCOUNTING: the reported byte delta is the real one -- LinearLayoutBytes must agree with
//      what ContainerWriter actually plans, for every LayoutSet, at whatever R4DX_W4A16_GROUP this
//      build uses.
//
// Plus the two failure modes the CLI contract names: an invalid regex is a hard error, and a valid
// regex that matches nothing is a WARNING (a scripted sweep over a list of class regexes must not
// die mid-batch on a class this checkpoint happens not to have).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/keep_bf16.hpp"
#include "r4dx_convert/linear_layouts.hpp"

using r4dx_convert::ContainerWriter;
using r4dx_convert::KeepBf16Selector;
using r4dx_convert::KeptBf16LayoutSet;
using r4dx_convert::LayoutSet;
using r4dx_convert::LinearLayoutBytes;

namespace {

int g_failures = 0;

bool Check(const std::string& label, bool cond) {
  std::printf("%-58s %s\n", label.c_str(), cond ? "OK" : "FAIL");
  if (!cond) ++g_failures;
  return cond;
}

LayoutSet Set(bool mxfp4, bool w4a16, bool w4a8, bool bf16) {
  LayoutSet ls;
  ls.mxfp4 = mxfp4;
  ls.w4a16 = w4a16;
  ls.w4a8 = w4a8;
  ls.bf16 = bf16;
  return ls;
}

// The container base names r4dx-convert actually emits (src/convert/main.cpp's add_linear calls),
// for a model with both layer kinds -- the vocabulary every --keep-bf16 regex is written against.
std::vector<std::string> RealBaseNames() {
  return {
      "text.layers.0.gdn.in_proj_qkv", "text.layers.0.gdn.in_proj_z", "text.layers.0.gdn.out_proj",
      "text.layers.0.mlp.gate_up",      "text.layers.0.mlp.down",
      "text.layers.3.attn.qg",          "text.layers.3.attn.k",       "text.layers.3.attn.v",
      "text.layers.3.attn.o",           "text.layers.3.mlp.gate_up",  "text.layers.3.mlp.down",
      "text.layers.17.attn.o",          "text.layers.63.attn.o",
      "mtp.attn.qg",                    "mtp.attn.o",                 "mtp.mlp.down",
      "mtp.draft_head.lm_head",         "lm_head",
  };
}

std::vector<std::string> Selected(const std::string& pattern) {
  KeepBf16Selector sel(pattern);
  std::vector<std::string> out;
  for (const auto& name : RealBaseNames()) {
    if (sel.Matches(name)) out.push_back(name);
  }
  return out;
}

bool SameSet(const std::vector<std::string>& got, const std::vector<std::string>& want) {
  if (got.size() != want.size()) return false;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) return false;
  }
  return true;
}

std::string Join(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + v[i];
  return s;
}

std::string TempPath(const char* leaf) {
  const char* t = std::getenv("TEMP");
  return std::string(t ? t : ".") + "\\" + leaf;
}

// Reads back a just-written container's tensor-name list (safetensors shell: 8-byte LE header
// length + JSON header).
std::vector<std::string> TensorNames(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  std::string header(static_cast<size_t>(header_len), '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  const nlohmann::json j = nlohmann::json::parse(header);
  std::vector<std::string> names;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() != "__metadata__") names.push_back(it.key());
  }
  return names;
}

}  // namespace

int main() {
  // ---- 1. selection -------------------------------------------------------------------------
  {
    KeepBf16Selector off("");
    Check("empty pattern -> disabled", !off.Enabled());
    Check("disabled selector matches nothing", !off.Matches("text.layers.3.attn.o"));
    std::ostringstream log, warn;
    off.Report(log, warn);
    Check("disabled selector reports nothing", log.str().empty() && warn.str().empty());
  }

  // The exact class regexes docs/validation.md's sensitivity sweep uses.
  Check("'attn\\.o$' selects every attn.o, incl. mtp, not attn.qg",
        SameSet(Selected("attn\\.o$"),
                {"text.layers.3.attn.o", "text.layers.17.attn.o", "text.layers.63.attn.o",
                 "mtp.attn.o"}));
  Check("'^text\\.layers\\.\\d+\\.attn\\.o$' excludes mtp",
        SameSet(Selected("^text\\.layers\\.\\d+\\.attn\\.o$"),
                {"text.layers.3.attn.o", "text.layers.17.attn.o", "text.layers.63.attn.o"}));
  Check("'attn\\.(k|v)$' selects k and v only",
        SameSet(Selected("attn\\.(k|v)$"), {"text.layers.3.attn.k", "text.layers.3.attn.v"}));
  Check("'^text\\.layers\\.[0-3]\\.' selects a layer band's every linear",
        SameSet(Selected("^text\\.layers\\.[0-3]\\."),
                {"text.layers.0.gdn.in_proj_qkv", "text.layers.0.gdn.in_proj_z",
                 "text.layers.0.gdn.out_proj", "text.layers.0.mlp.gate_up", "text.layers.0.mlp.down",
                 "text.layers.3.attn.qg", "text.layers.3.attn.k", "text.layers.3.attn.v",
                 "text.layers.3.attn.o", "text.layers.3.mlp.gate_up", "text.layers.3.mlp.down"}));
  // Anchoring matters: `^lm_head$` is the real head alone; the bare substring also drags in the
  // optional reduced-vocab draft head. Both behaviours are intentional -- this pins them so a
  // future "helpful" switch to regex_match (which would break every unanchored pattern above)
  // cannot land unnoticed.
  Check("'^lm_head$' selects the real head only", SameSet(Selected("^lm_head$"), {"lm_head"}));
  Check("unanchored 'lm_head' also selects the draft head",
        SameSet(Selected("lm_head"), {"mtp.draft_head.lm_head", "lm_head"}));
  Check("'gdn\\.in_proj_z$' does not match in_proj_qkv",
        SameSet(Selected("gdn\\.in_proj_z$"), {"text.layers.0.gdn.in_proj_z"}));

  // ---- 2. failure modes ---------------------------------------------------------------------
  {
    bool threw = false;
    std::string msg;
    try {
      KeepBf16Selector bad("text\\.layers\\.[0-7");  // unterminated bracket expression
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    Check("invalid regex throws", threw);
    Check("...and the message names the flag and the pattern",
          msg.find("--keep-bf16") != std::string::npos && msg.find("[0-7") != std::string::npos);
    std::printf("  %s\n", msg.c_str());
  }
  {
    KeepBf16Selector sel("text\\.layers\\.\\d+\\.attn\\.nonesuch$");
    for (const auto& name : RealBaseNames()) (void)sel.Matches(name);
    std::ostringstream log, warn;
    sel.Report(log, warn);  // must not throw
    Check("regex matching nothing is a warning, not an error",
          sel.MatchedCount() == 0 && warn.str().find("WARNING") != std::string::npos &&
              warn.str().find("matched no linear") != std::string::npos && log.str().empty());
    std::printf("  %s", warn.str().c_str());
  }

  // ---- 3. accounting ------------------------------------------------------------------------
  // LinearLayoutBytes must equal what ContainerWriter really plans, for every LayoutSet -- at this
  // build's R4DX_W4A16_GROUP, whatever it is. N,K chosen divisible by 16 and by every group size.
  {
    const int N = 64, K = 512;
    const LayoutSet sets[] = {
        Set(false, false, false, true),  Set(false, true, false, false),
        Set(false, false, true, false),  Set(true, false, false, false),
        Set(true, true, true, true),     Set(false, true, false, true),
    };
    bool all_agree = true;
    for (const LayoutSet& ls : sets) {
      ContainerWriter w;
      r4dx_convert::PlanLinearLayouts(w, "t", N, K, ls);
      const uint64_t predicted = LinearLayoutBytes(N, K, ls);
      if (w.PlannedDataBytes() != predicted) {
        std::printf("  mismatch: planned=%llu predicted=%llu\n",
                    static_cast<unsigned long long>(w.PlannedDataBytes()),
                    static_cast<unsigned long long>(predicted));
        all_agree = false;
      }
    }
    Check("LinearLayoutBytes == ContainerWriter::PlannedDataBytes (6 sets)", all_agree);
    std::printf("  w4a16 group=%d, w4a8 group=%d, mxfp4 group=%d\n", r4dx_convert::kW4A16Group,
                r4dx_convert::kW4A8Group, r4dx_convert::kMxfp4Group);
  }
  {
    // The delta a sensitivity run pays: bf16 instead of the requested quantized set.
    const int N = 64, K = 512;
    const LayoutSet requested = Set(false, true, false, false);  // --layouts w4a16 --no-bf16
    KeepBf16Selector sel("attn\\.o$");
    std::ostringstream log, warn;
    sel.Record("text.layers.3.attn.o", N, K, requested, log);
    sel.Record("text.layers.17.attn.o", N, K, requested, log);
    const int64_t per_linear = static_cast<int64_t>(LinearLayoutBytes(N, K, KeptBf16LayoutSet())) -
                               static_cast<int64_t>(LinearLayoutBytes(N, K, requested));
    Check("ExtraBytes == sum of per-linear deltas", sel.ExtraBytes() == 2 * per_linear);
    Check("ExtraBytes is positive (bf16 costs more than 4-bit)", sel.ExtraBytes() > 0);
    sel.Report(log, warn);
    Check("summary names the pattern and the count",
          warn.str().empty() && log.str().find("attn\\.o$") != std::string::npos &&
              log.str().find("2 linear(s) kept as bf16") != std::string::npos);
    std::printf("%s", log.str().c_str());
  }

  // ---- 4. emission --------------------------------------------------------------------------
  // A kept linear must produce EXACTLY `<base>.bf16.w`, with the plain row-major bf16 bytes
  // (tests/convert/test_bf16_layout.cpp owns the byte-level layout contract; what matters here is
  // that no quantized layout leaked through alongside it).
  {
    const int N = 64, K = 512;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> w(static_cast<size_t>(N) * K);
    for (auto& v : w) v = dist(rng);

    // Ask for everything, then let the selector override it -- the situation in a real run
    // (`--layouts w4a16 --lm-head 4bit --keep-bf16 ...`).
    const LayoutSet requested = Set(true, true, true, false);
    KeepBf16Selector sel("^t$");
    const LayoutSet effective = sel.Matches("t") ? KeptBf16LayoutSet() : requested;
    Check("selector overrides the requested set",
          effective.bf16 && !effective.w4a16 && !effective.w4a8 && !effective.mxfp4);

    const std::string path = TempPath("r4dx_test_keep_bf16.r4dx");
    {
      ContainerWriter writer;
      r4dx_convert::PlanLinearLayouts(writer, "t", N, K, effective);
      Check("kept linear plans exactly 1 tensor", writer.PlannedTensorCount() == 1);
      Check("kept linear plans N*K*2 bytes",
            writer.PlannedDataBytes() == static_cast<uint64_t>(N) * K * 2);
      nlohmann::json meta;
      meta["r4dx_format_version"] = "1";
      writer.FinalizeHeader(path, meta);
      r4dx_convert::EmitLinearLayouts(writer, "t", w, N, K, effective, /*nthreads=*/1);
      writer.Finish();
    }
    const std::vector<std::string> names = TensorNames(path);
    Check("container carries only t.bf16.w", SameSet(names, {"t.bf16.w"}));
    if (names.size() != 1 || names[0] != "t.bf16.w") std::printf("  got: %s\n", Join(names).c_str());
    std::remove(path.c_str());
  }

  if (g_failures != 0) {
    std::printf("FAIL (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
