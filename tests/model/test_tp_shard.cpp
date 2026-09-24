// tests/model/test_tp_shard.cpp -- CPU-only (no GPU, no container). docs/tp.md 4, 5.2, 10.2.
//
// The tensor-parallel slicer (src/model/tp/tp_shard.{h,cpp}) against the CONVERTER'S OWN packers
// (src/convert/include/r4dx_convert/quant_int4.hpp, quant_mxfp4.hpp, quant_search.hpp,
// tensor_codec.hpp's EncodeBf16). The contract under test, for every layout and every on-disk part:
//
//     Gather(pack(W), PlanRows/PlanCols(...)) == pack(W[rank rows or cols])   byte for byte
//
// i.e. the loader can cut a rank's shard straight out of the one converted container and get
// exactly the bytes the converter would have written for the shard itself -- with ONE deliberate
// exception, the mxfp4 K-slice `wref`, which keeps the FULL row's exponent (docs/tp.md 4.3); there
// the assertion is `== full.wref`, plus a CPU dequant, by the kernel's own formula
// (r4d_gemm_mxfp4a8_nt_m64.hip: dsh = clamp(wref - ws, 0, 15), factor 2^(wref-127)), of
// (sliced wq, sliced ws, full wref) == the matching columns of the full tensor's dequant.
//
// Coverage:
//   1. RuleFor on the REAL v6 config: every tensor family of docs/tp.md 4.2 (text layers, MTP head,
//      global tensors, vision, DFlash2), bare vs `.bf16.w` bases, and the names that must throw.
//   2. Plans on the REAL per-layer shapes of every linear, every layout, both ranks: legal, the
//      exact rank byte count, and exactly the contiguous-vs-staged split docs/tp.md 5.1 claims.
//   3. End to end on a small config with the real model's structure (test_tp_config.cpp's
//      SmallConfig): every sharded tensor of a GDN layer, an attention layer and lm_head --
//      qg's per-head interleave, in_proj_qkv's 3 segments, gate_up's 2 halves, conv1d's 3 channel
//      ranges, in_proj_a/b, A_log/dt_bias, the descales, and the three row-parallel K splits --
//      both ranks, every layout, real packers on random W with outliers.
//   4. docs/tp.md 10.2's explicit shapes (single segment, fused qkv/gate_up/qg-shaped, K = 1536
//      and the 17 x 64 per-rank K that mirrors mlp.down's 8704 = 17 x 512).
//   5. The mxfp4 wref exception, with rows built so the full-row wref clamps a group the shard's
//      own wref would not.
//   6. Misalignment throws; Gather bounds.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/tensor_codec.hpp"
#include "tp/tp_shard.h"

using r4dx::model::ModelConfig;
using namespace r4dx::model::tp;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}
void Pass(const std::string& what) { std::printf("PASS: %s\n", what.c_str()); }

template <class Fn>
bool Throws(Fn fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

const int kThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

// ---- configs -----------------------------------------------------------------------------------

ModelConfig RealConfig() {
  ModelConfig c;
  c.hidden_size = 5120;
  c.num_hidden_layers = 64;
  for (int i = 0; i < 64; ++i) {
    c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  }
  c.num_attention_heads = 24;
  c.num_key_value_heads = 4;
  c.head_dim = 256;
  c.intermediate_size = 17408;
  c.linear_num_key_heads = 16;
  c.linear_num_value_heads = 48;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.vocab_size = 248320;
  return c;
}

// test_tp_config.cpp's SmallConfig: KeyDim 256, ValueDim 1024, ConvDim 1536, qg 2048 rows (4 heads
// x [q 256 | gate 256]), kv 512 rows, intermediate 1024, vocab 1024, hidden 512. Layer 3 is the
// full-attention layer.
ModelConfig SmallConfig() {
  ModelConfig c;
  c.hidden_size = 512;
  c.num_hidden_layers = 4;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 256;
  c.intermediate_size = 1024;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 8;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.vocab_size = 1024;
  return c;
}

// ---- layouts: the converter's packers -----------------------------------------------------------

enum class Scheme { kBf16, kW4a16, kW4a16Search, kW4a8, kW4a8Search, kMxfp4, kMxfp4Search };
struct Variant {
  const char* name;
  Scheme scheme;
  int group;
};
const Variant kVariants[] = {
    {"bf16", Scheme::kBf16, 0},
    {"w4a16 g64", Scheme::kW4a16, 64},
    {"w4a16 g128", Scheme::kW4a16, 128},
    {"w4a16 search+imatrix g64", Scheme::kW4a16Search, 64},
    {"w4a16 search+imatrix g128", Scheme::kW4a16Search, 128},
    {"w4a8", Scheme::kW4a8, r4dx_convert::kW4A8Group},
    {"w4a8 search+imatrix", Scheme::kW4a8Search, r4dx_convert::kW4A8Group},
    {"mxfp4", Scheme::kMxfp4, r4dx_convert::kMxfp4Group},
    {"mxfp4 search+imatrix", Scheme::kMxfp4Search, r4dx_convert::kMxfp4Group},
};

struct PackedPart {
  Part part;
  int group;
  std::vector<uint8_t> bytes;
};

std::vector<uint8_t> AsBytes(const std::vector<uint32_t>& v) {
  std::vector<uint8_t> b(v.size() * 4);
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// Whether the converter can pack W[N, K] in this layout at all.
bool Packable(const Variant& v, int64_t N, int64_t K) {
  if (v.scheme == Scheme::kBf16) return true;
  if (N % 16 != 0) return false;
  if (v.scheme == Scheme::kMxfp4 || v.scheme == Scheme::kMxfp4Search) return K % 32 == 0;
  return K % 64 == 0 && K % v.group == 0;
}

// Every on-disk part of `v` for W[N, K], exactly as r4dx_convert::EmitLinearLayouts writes them.
// `imp` (length K) is the importance vector the *Search variants weight by.
std::vector<PackedPart> Pack(const Variant& v, const std::vector<float>& w, int N, int K,
                             const std::vector<float>& imp) {
  using namespace r4dx_convert;
  const ImportanceVector iv{imp.data(), static_cast<int64_t>(imp.size())};
  std::vector<PackedPart> out;
  switch (v.scheme) {
    case Scheme::kBf16:
      out.push_back({Part::kBf16, 0, EncodeBf16(w)});
      break;
    case Scheme::kW4a16:
    case Scheme::kW4a16Search: {
      std::vector<uint8_t> q, zero;
      std::vector<float> scale;
      if (v.scheme == Scheme::kW4a16Search) {
        QuantizeInt4AsymmetricSearch(w.data(), N, K, v.group, iv, kThreads, q, scale, zero);
      } else {
        QuantizeInt4Asymmetric(w.data(), N, K, v.group, kThreads, q, scale, zero);
      }
      out.push_back({Part::kW4Wq, 0, AsBytes(PackW4Nibbles(q, N, K, kThreads))});
      out.push_back(
          {Part::kW4a16Wsz, v.group, AsBytes(PackW4A16Scales(scale, zero, N, K, v.group))});
      break;
    }
    case Scheme::kW4a8:
    case Scheme::kW4a8Search: {
      std::vector<uint8_t> q;
      std::vector<float> scale;
      if (v.scheme == Scheme::kW4a8Search) {
        QuantizeInt4Pinned8Search(w.data(), N, K, v.group, iv, kThreads, q, scale);
      } else {
        QuantizeInt4SymmetricPinned8(w.data(), N, K, v.group, kThreads, q, scale);
      }
      out.push_back({Part::kW4Wq, 0, AsBytes(PackW4Nibbles(q, N, K, kThreads))});
      out.push_back({Part::kW4a8Ws, v.group, AsBytes(PackW4A8Scales(scale, N, K, v.group))});
      break;
    }
    case Scheme::kMxfp4:
    case Scheme::kMxfp4Search: {
      const Mxfp4Quantized mq = (v.scheme == Scheme::kMxfp4Search)
                                    ? QuantizeMxfp4Search(w.data(), N, K, kMxfp4Group, iv, kThreads)
                                    : QuantizeMxfp4(w.data(), N, K, kMxfp4Group, kThreads);
      out.push_back({Part::kMxWq, kMxfp4Group, PackMxfp4Wq(mq.packed, N, K, kThreads)});
      out.push_back({Part::kMxWs, kMxfp4Group, PackMxfp4Ws(mq.escale, N, K, kMxfp4Group)});
      out.push_back({Part::kMxWref, kMxfp4Group, mq.wref});
      break;
    }
  }
  return out;
}

// Column alignment each part needs (docs/tp.md 4.3's "Alignment required" column).
int64_t ColAlign(const PackedPart& p) {
  switch (p.part) {
    case Part::kW4Wq: return 64;
    case Part::kW4a16Wsz:
    case Part::kW4a8Ws:
    case Part::kMxWs: return p.group;
    case Part::kMxWq: return 32;
    default: return 1;
  }
}

// ---- random weights ----------------------------------------------------------------------------

// N(0, 0.02) with ~0.5% outliers at 25x, plus one exactly-constant 64-wide group (the quantizers'
// degenerate branch) in row 0.
std::vector<float> RandomW(int64_t N, int64_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 0.02f);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(N * K));
  for (float& x : w) {
    x = d(rng);
    if (u(rng) < 0.005f) x *= 25.0f;
  }
  for (int64_t k = 0; k < std::min<int64_t>(64, K); ++k) w[static_cast<size_t>(k)] = 0.0123f;
  return w;
}

std::vector<float> RandomImportance(int64_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.1f, 10.0f);
  std::vector<float> imp(static_cast<size_t>(K));
  for (float& x : imp) x = u(rng);
  return imp;
}

std::vector<float> SliceRows(const std::vector<float>& w, int64_t K,
                             const std::vector<Range>& rows) {
  std::vector<float> out;
  for (const Range& r : rows) {
    out.insert(out.end(), w.begin() + r.begin * K, w.begin() + (r.begin + r.count) * K);
  }
  return out;
}

std::vector<float> SliceCols(const std::vector<float>& w, int64_t N, int64_t K, Range c) {
  std::vector<float> out(static_cast<size_t>(N * c.count));
  for (int64_t n = 0; n < N; ++n) {
    std::copy(w.begin() + n * K + c.begin, w.begin() + n * K + c.begin + c.count,
              out.begin() + n * c.count);
  }
  return out;
}

int64_t TotalRows(const std::vector<Range>& rows) {
  int64_t n = 0;
  for (const Range& r : rows) n += r.count;
  return n;
}

std::string RangesStr(const std::vector<Range>& rows) {
  std::string s;
  for (const Range& r : rows) {
    s += "[" + std::to_string(r.begin) + "," + std::to_string(r.begin + r.count) + ")";
  }
  return s;
}

bool SameBytes(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want,
               std::string* why) {
  if (got.size() != want.size()) {
    *why = "size " + std::to_string(got.size()) + " vs " + std::to_string(want.size());
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      *why = "first difference at byte " + std::to_string(i);
      return false;
    }
  }
  return true;
}

// ---- the core assertions ------------------------------------------------------------------------

// Row slice of W[N, K] to `rows`, every packable layout (or only bf16 when `bf16_only`).
void CheckRowSlice(const std::string& what, const std::vector<float>& w, int64_t N, int64_t K,
                   const std::vector<Range>& rows, bool bf16_only = false) {
  const std::vector<float> imp = RandomImportance(K, 99);
  const std::vector<float> ws = SliceRows(w, K, rows);
  const int64_t n = TotalRows(rows);
  int parts = 0;
  for (const Variant& v : kVariants) {
    if (bf16_only && v.scheme != Scheme::kBf16) continue;
    if (!Packable(v, N, K) || !Packable(v, n, K)) continue;
    const auto full = Pack(v, w, static_cast<int>(N), static_cast<int>(K), imp);
    const auto want = Pack(v, ws, static_cast<int>(n), static_cast<int>(K), imp);
    for (size_t i = 0; i < full.size(); ++i) {
      const PartShape shape{full[i].part, N, K, full[i].group, 0};
      const auto runs = PlanRows(shape, rows);
      const auto got = Gather(full[i].bytes.data(), full[i].bytes.size(), runs);
      std::string why;
      Check(SameBytes(got, want[i].bytes, &why),
            what + " rows " + RangesStr(rows) + " " + v.name + " part " + std::to_string(i) +
                ": Gather(PlanRows(pack(W))) == pack(W[rows]) -- " + why);
      ++parts;
    }
  }
  Pass(what + " rows " + RangesStr(rows) + ": " + std::to_string(parts) +
       " packed parts byte-exact");
}

// Column slice of W[N, K] to `cols`, every packable layout. When the range breaks the alignment of
// any part of a layout (e.g. a 17 x 64 K range against group-128 scales), that layout cannot be
// split at all: every misaligned part must make PlanCols throw, and the layout is skipped.
// Returns the number of layouts refused.
int CheckColSlice(const std::string& what, const std::vector<float>& w, int64_t N, int64_t K,
                  Range cols) {
  const std::vector<float> imp = RandomImportance(K, 98);
  const std::vector<float> imp_s(imp.begin() + cols.begin, imp.begin() + cols.begin + cols.count);
  const std::vector<float> ws = SliceCols(w, N, K, cols);
  const std::string range =
      " cols [" + std::to_string(cols.begin) + "," + std::to_string(cols.begin + cols.count) + ")";
  int exact = 0, refused = 0;
  for (const Variant& v : kVariants) {
    if (!Packable(v, N, K)) continue;
    const auto full = Pack(v, w, static_cast<int>(N), static_cast<int>(K), imp);
    bool aligned = true;
    for (const PackedPart& p : full) {
      const int64_t a = ColAlign(p);
      if (cols.begin % a != 0 || cols.count % a != 0) {
        aligned = false;
        const PartShape shape{p.part, N, K, p.group, 0};
        Check(Throws([&] { PlanCols(shape, cols); }),
              what + range + " " + v.name + ": a misaligned part throws");
      }
    }
    if (!aligned) {
      ++refused;
      continue;
    }
    const bool slice_packable = Packable(v, N, cols.count);
    Check(slice_packable, what + range + " " + v.name +
                              ": the converter can pack every slice the planner accepts");
    if (!slice_packable) continue;
    const auto want = Pack(v, ws, static_cast<int>(N), static_cast<int>(cols.count), imp_s);
    for (size_t i = 0; i < full.size(); ++i) {
      const PartShape shape{full[i].part, N, K, full[i].group, 0};
      const std::string tag = what + range + " " + v.name + " part " + std::to_string(i);
      const auto got = Gather(full[i].bytes.data(), full[i].bytes.size(), PlanCols(shape, cols));
      std::string why;
      if (full[i].part == Part::kMxWref) {
        Check(SameBytes(got, full[i].bytes, &why), tag + ": wref K-slice == FULL wref -- " + why);
      } else {
        Check(SameBytes(got, want[i].bytes, &why),
              tag + ": Gather(PlanCols(pack(W))) == pack(W[:, cols]) -- " + why);
      }
      ++exact;
    }
  }
  Pass(what + range + ": " + std::to_string(exact) + " parts byte-exact, " +
       std::to_string(refused) + " layouts refused as misaligned");
  return refused;
}

// kElem tensors (1-D vectors, conv1d_weight): random bytes, expected rows cut independently.
void CheckElemRows(const std::string& what, int64_t N, int64_t row_bytes,
                   const std::vector<Range>& rows) {
  std::mt19937 rng(static_cast<uint32_t>(N * 131 + row_bytes));
  std::vector<uint8_t> full(static_cast<size_t>(N * row_bytes));
  for (auto& b : full) b = static_cast<uint8_t>(rng());
  std::vector<uint8_t> want;
  for (const Range& r : rows) {
    want.insert(want.end(), full.begin() + r.begin * row_bytes,
                full.begin() + (r.begin + r.count) * row_bytes);
  }
  const PartShape shape{Part::kElem, N, 0, 0, row_bytes};
  std::string why;
  Check(SameBytes(Gather(full.data(), full.size(), PlanRows(shape, rows)), want, &why),
        what + " elem rows " + RangesStr(rows) + " -- " + why);
  Check(Throws([&] { PlanCols(shape, {0, 1}); }), what + ": an elem part has no column axis");
  Pass(what + " elem rows " + RangesStr(rows) + " byte-exact");
}

bool SameRanges(const std::vector<Range>& a, const std::vector<Range>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].begin != b[i].begin || a[i].count != b[i].count) return false;
  }
  return true;
}

// ---- 1. RuleFor on the real config --------------------------------------------------------------

struct RuleCase {
  std::string base;
  Split split;
  std::vector<Segment> segments;  // kRows
  int64_t k_total;                // kCols
};

bool SameRule(const ShardRule& r, const RuleCase& c) {
  if (r.split != c.split || r.k_total != c.k_total || r.segments.size() != c.segments.size()) {
    return false;
  }
  for (size_t i = 0; i < r.segments.size(); ++i) {
    if (r.segments[i].begin != c.segments[i].begin || r.segments[i].rows != c.segments[i].rows) {
      return false;
    }
  }
  return true;
}

void TestRuleForReal() {
  const ModelConfig g = RealConfig();
  const Split R = Split::kReplicate, ROWS = Split::kRows, COLS = Split::kCols;
  const std::vector<Segment> qkv = {{0, 2048}, {2048, 2048}, {4096, 6144}};
  const std::vector<Segment> gate_up = {{0, 17408}, {17408, 17408}};
  std::vector<RuleCase> cases = {
      // GDN layer 0
      {"text.layers.0.input_layernorm", R, {}, 0},
      {"text.layers.0.post_attention_layernorm", R, {}, 0},
      {"text.layers.0.gdn.in_proj_qkv", ROWS, qkv, 0},
      {"text.layers.0.gdn.in_proj_z", ROWS, {{0, 6144}}, 0},
      {"text.layers.0.gdn.in_proj_a", ROWS, {{0, 48}}, 0},
      {"text.layers.0.gdn.in_proj_b", ROWS, {{0, 48}}, 0},
      {"text.layers.0.gdn.conv1d_weight", ROWS, qkv, 0},
      {"text.layers.0.gdn.A_log", ROWS, {{0, 48}}, 0},
      {"text.layers.0.gdn.dt_bias", ROWS, {{0, 48}}, 0},
      {"text.layers.0.gdn.norm_weight", R, {}, 0},
      {"text.layers.0.gdn.out_proj", COLS, {}, 6144},
      {"text.layers.0.mlp.gate_up", ROWS, gate_up, 0},
      {"text.layers.0.mlp.down", COLS, {}, 17408},
      // full-attention layer 63
      {"text.layers.63.input_layernorm", R, {}, 0},
      {"text.layers.63.attn.qg", ROWS, {{0, 12288}}, 0},
      {"text.layers.63.attn.k", ROWS, {{0, 1024}}, 0},
      {"text.layers.63.attn.v", ROWS, {{0, 1024}}, 0},
      {"text.layers.63.attn.o", COLS, {}, 6144},
      {"text.layers.63.attn.q_norm", R, {}, 0},
      {"text.layers.63.attn.k_norm", R, {}, 0},
      {"text.layers.63.attn.k_descale", ROWS, {{0, 4}}, 0},
      {"text.layers.63.attn.v_descale", ROWS, {{0, 4}}, 0},
      {"text.layers.63.mlp.gate_up", ROWS, gate_up, 0},
      {"text.layers.63.mlp.down", COLS, {}, 17408},
      // global tensors
      {"text.embed_tokens", R, {}, 0},
      {"text.final_norm", R, {}, 0},
      {"lm_head", ROWS, {{0, 248320}}, 0},
      // MTP head: an attention layer + MLP, sharded like the body; the rest replicated.
      // mtp.attn.k/v are BARE bf16 on disk in v6, text.layers.*.attn.k/v `.bf16.w` -- one base,
      // one rule.
      {"mtp.input_layernorm", R, {}, 0},
      {"mtp.post_attention_layernorm", R, {}, 0},
      {"mtp.attn.qg", ROWS, {{0, 12288}}, 0},
      {"mtp.attn.k", ROWS, {{0, 1024}}, 0},
      {"mtp.attn.v", ROWS, {{0, 1024}}, 0},
      {"mtp.attn.o", COLS, {}, 6144},
      {"mtp.attn.q_norm", R, {}, 0},
      {"mtp.attn.k_norm", R, {}, 0},
      {"mtp.attn.k_descale", ROWS, {{0, 4}}, 0},
      {"mtp.attn.v_descale", ROWS, {{0, 4}}, 0},
      {"mtp.mlp.gate_up", ROWS, gate_up, 0},
      {"mtp.mlp.down", COLS, {}, 17408},
      {"mtp.fc", R, {}, 0},
      {"mtp.norm", R, {}, 0},
      {"mtp.pre_fc_norm_hidden", R, {}, 0},
      {"mtp.pre_fc_norm_embedding", R, {}, 0},
      {"mtp.draft_head.lm_head", R, {}, 0},
      {"mtp.draft_head.vocab_ids", R, {}, 0},
      // vision: rank 0 only; DFlash2 drafter: replicated
      {"vision.blocks.0.attn.qkv.weight", Split::kRank0Only, {}, 0},
      {"vision.merger.linear_fc2.weight", Split::kRank0Only, {}, 0},
      {"dflash.fc", R, {}, 0},
      {"dflash.layers.0.self_attn.q_proj", R, {}, 0},
      {"dflash.selector.predecessor", R, {}, 0},
  };
  int ok = 0;
  for (const RuleCase& c : cases) {
    bool same = false;
    std::string err;
    try {
      same = SameRule(RuleFor(c.base, g), c);
    } catch (const std::exception& e) {
      err = e.what();
    }
    Check(same, "RuleFor(" + c.base + ") has the docs/tp.md 4.2 rule " + err);
    ok += same ? 1 : 0;
  }
  Pass("RuleFor: " + std::to_string(ok) + "/" + std::to_string(cases.size()) +
       " tensor families classified as docs/tp.md 4.2 says");

  const char* must_throw[] = {
      "text.layers.0.gdn.in_proj_qkv.w4a16.wq",  // layout suffix not stripped
      "text.layers.63.attn.k.bf16.w",            // likewise the `.bf16.w` form
      "lm_head.mxfp4.wref",
      "text.layers.0.attn.qg",       // layer 0 is GDN
      "text.layers.3.gdn.out_proj",  // layer 3 is full attention
      "text.layers.64.mlp.down",     // out of range
      "text.layers.x.mlp.down",
      "text.layers..mlp.down",
      "text.layers.0",
      "text.layers.0.mlp.gate_proj",  // an HF name, not a container one
      "text.layers.0.self_attn.q_proj",
      "mtp.gdn.in_proj_qkv",  // the MTP layer has no GDN block
      "mtp.layers.0.attn.qg",
      "mtp.draft_head.lm_head.w4a16.wq",  // suffixed names throw on the replicated paths too
      "mtp.draft_head.vocab",             // only the two known draft_head tensors replicate
      "dflash.fc.w4a16.wq",
      "dflash.layers.0.self_attn.q_proj.bf16.w",
      "vision.blocks.0.attn.qkv.mxfp4.ws",
      "text.norm",
      "embed_tokens",
      "lm_head2",
      "",
  };
  int threw = 0;
  for (const char* name : must_throw) {
    const bool t = Throws([&] { RuleFor(name, g); });
    Check(t, std::string("RuleFor('") + name + "') throws std::invalid_argument");
    threw += t ? 1 : 0;
  }
  Pass("RuleFor: " + std::to_string(threw) + "/" +
       std::to_string(sizeof(must_throw) / sizeof(must_throw[0])) +
       " unknown / suffixed / misplaced names throw");
  Check(Throws([&] { RuleFor("lm_head", ModelConfig::Shard(g, 2, 0)); }),
        "RuleFor refuses a rank config (it needs the global one)");

  // Rank ranges on the real shapes (docs/tp.md 4.2's "Rank r rows / cols" column).
  const auto rows = [&](const char* base, int r) { return RankRows(RuleFor(base, g), 2, r); };
  for (int r = 0; r < 2; ++r) {
    const std::string p = "real rank " + std::to_string(r) + ": ";
    Check(SameRanges(rows("text.layers.0.gdn.in_proj_qkv", r),
                     {{1024 * r, 1024}, {2048 + 1024 * r, 1024}, {4096 + 3072 * r, 3072}}),
          p + "in_proj_qkv rows [1024r,+1024) u [2048+1024r,+1024) u [4096+3072r,+3072)");
    Check(SameRanges(rows("text.layers.0.gdn.conv1d_weight", r),
                     {{1024 * r, 1024}, {2048 + 1024 * r, 1024}, {4096 + 3072 * r, 3072}}),
          p + "conv1d_weight channels as in_proj_qkv");
    Check(SameRanges(rows("text.layers.0.gdn.in_proj_z", r), {{3072 * r, 3072}}),
          p + "in_proj_z rows [3072r,+3072)");
    Check(SameRanges(rows("text.layers.0.gdn.A_log", r), {{24 * r, 24}}),
          p + "A_log [24r,+24)");
    Check(SameRanges(rows("text.layers.0.mlp.gate_up", r),
                     {{8704 * r, 8704}, {17408 + 8704 * r, 8704}}),
          p + "gate_up rows [8704r,+8704) u [17408+8704r,+8704)");
    Check(SameRanges(rows("text.layers.3.attn.qg", r), {{6144 * r, 6144}}),
          p + "attn.qg rows [6144r,+6144) = heads 12r..12r+11");
    Check(SameRanges(rows("text.layers.3.attn.k", r), {{512 * r, 512}}),
          p + "attn.k rows [512r,+512) = kv heads 2r, 2r+1");
    Check(SameRanges(rows("text.layers.3.attn.k_descale", r), {{2 * r, 2}}),
          p + "k_descale [2r,+2)");
    Check(SameRanges(rows("lm_head", r), {{124160 * r, 124160}}), p + "lm_head [124160r,+124160)");
    const Range o = RankCols(RuleFor("text.layers.3.attn.o", g), 2, r);
    const Range out = RankCols(RuleFor("text.layers.0.gdn.out_proj", g), 2, r);
    const Range down = RankCols(RuleFor("text.layers.0.mlp.down", g), 2, r);
    Check(o.begin == 3072 * r && o.count == 3072 && out.begin == 3072 * r && out.count == 3072 &&
              down.begin == 8704 * r && down.count == 8704,
          p + "attn.o / gdn.out_proj K [3072r,+3072), mlp.down K [8704r,+8704)");
  }
  // The rank ranges must reassemble exactly the rank config's shapes (config split vs slicer).
  for (int r = 0; r < 2; ++r) {
    const ModelConfig s = ModelConfig::Shard(g, 2, r);
    Check(TotalRows(rows("text.layers.0.gdn.in_proj_qkv", r)) == s.ConvDim() &&
              TotalRows(rows("text.layers.0.gdn.in_proj_z", r)) == s.ValueDim() &&
              TotalRows(rows("text.layers.0.gdn.in_proj_a", r)) == s.linear_num_value_heads &&
              TotalRows(rows("text.layers.3.attn.qg", r)) ==
                  2 * s.num_attention_heads * s.head_dim &&
              TotalRows(rows("text.layers.3.attn.k", r)) == s.num_key_value_heads * s.head_dim &&
              TotalRows(rows("text.layers.3.attn.v_descale", r)) == s.num_key_value_heads &&
              TotalRows(rows("text.layers.0.mlp.gate_up", r)) == 2 * s.intermediate_size &&
              TotalRows(rows("lm_head", r)) == s.VocabShardSize() &&
              rows("lm_head", r)[0].begin == s.VocabShardBegin(),
          "real rank " + std::to_string(r) + ": rank rows == ModelConfig::Shard's shapes");
  }
  Check(Throws([&] { RankRows(RuleFor("text.layers.0.mlp.down", g), 2, 0); }) &&
            Throws([&] { RankCols(RuleFor("lm_head", g), 2, 0); }) &&
            Throws([&] { RankRows(RuleFor("text.final_norm", g), 2, 0); }) &&
            Throws([&] { RankRows(RuleFor("lm_head", g), 2, 2); }),
        "RankRows/RankCols refuse the wrong split kind and a bad rank");
  Pass("RankRows/RankCols on the real shapes match docs/tp.md 4.2 and ModelConfig::Shard");
}

// ---- 2. plans on the real per-layer shapes ------------------------------------------------------

struct LayoutParts {
  const char* name;
  std::vector<std::pair<Part, int>> parts;  // (part, group)
};

void TestRealPlans() {
  const ModelConfig g = RealConfig();
  const LayoutParts layouts[] = {
      {"bf16", {{Part::kBf16, 0}}},
      {"w4a16 g64", {{Part::kW4Wq, 0}, {Part::kW4a16Wsz, 64}}},
      {"w4a16 g128", {{Part::kW4Wq, 0}, {Part::kW4a16Wsz, 128}}},
      {"w4a8", {{Part::kW4Wq, 0}, {Part::kW4a8Ws, 128}}},
      {"mxfp4", {{Part::kMxWq, 32}, {Part::kMxWs, 32}, {Part::kMxWref, 32}}},
  };
  struct Linear {
    const char* base;
    int64_t N, K;
  };
  const Linear linears[] = {
      {"text.layers.0.gdn.in_proj_qkv", 10240, 5120}, {"text.layers.0.gdn.in_proj_z", 6144, 5120},
      {"text.layers.0.gdn.out_proj", 5120, 6144},     {"text.layers.0.mlp.gate_up", 34816, 5120},
      {"text.layers.0.mlp.down", 5120, 17408},        {"text.layers.3.attn.qg", 12288, 5120},
      {"text.layers.3.attn.k", 1024, 5120},           {"text.layers.3.attn.v", 1024, 5120},
      {"text.layers.3.attn.o", 5120, 6144},           {"lm_head", 248320, 5120},
      {"mtp.attn.qg", 12288, 5120},                   {"mtp.attn.o", 5120, 6144},
      {"mtp.mlp.gate_up", 34816, 5120},               {"mtp.mlp.down", 5120, 17408},
  };
  int plans = 0;
  for (const Linear& lin : linears) {
    const ShardRule rule = RuleFor(lin.base, g);
    for (int r = 0; r < 2; ++r) {
      std::vector<Range> rows;
      Range cols{};
      int64_t rn = lin.N, rk = lin.K;
      if (rule.split == Split::kRows) {
        rows = RankRows(rule, 2, r);
        rn = TotalRows(rows);
      } else {
        cols = RankCols(rule, 2, r);
        rk = cols.count;
      }
      for (const LayoutParts& lp : layouts) {
        for (const auto& [part, group] : lp.parts) {
          const PartShape shape{part, lin.N, lin.K, group, 0};
          std::vector<ByteRun> runs;
          const std::string tag = std::string(lin.base) + " rank " + std::to_string(r) + " " +
                                  lp.name + " part " + std::to_string(static_cast<int>(part));
          try {
            runs = rule.split == Split::kRows ? PlanRows(shape, rows) : PlanCols(shape, cols);
          } catch (const std::exception& e) {
            Check(false, tag + ": plan throws on a real shape: " + e.what());
            continue;
          }
          size_t total = 0;
          for (const ByteRun& b : runs) total += b.bytes;
          int64_t want = 0;
          size_t want_runs = 0;
          const size_t segs = rule.split == Split::kRows ? rows.size() : 0;
          switch (part) {
            case Part::kBf16:
              want = rn * rk * 2;
              want_runs = segs ? segs : static_cast<size_t>(lin.N);
              break;
            case Part::kW4Wq:
            case Part::kMxWq:
              want = rn * rk / 2;
              want_runs = segs ? segs : static_cast<size_t>(lin.N / 16);
              break;
            case Part::kW4a16Wsz:
            case Part::kW4a8Ws:
              want = rn * rk / group * 4;  // 4 B per (t, g, r) dword, never a uint16 stride
              want_runs = segs ? segs : static_cast<size_t>(lin.N / 16);
              break;
            case Part::kMxWs:
              want = rk / 32 * rn;
              want_runs = segs ? segs * static_cast<size_t>(lin.K / 32) : 1;
              break;
            case Part::kMxWref:
              want = segs ? rn : lin.N;  // a K-slice keeps the full row's wref
              want_runs = segs ? segs : 1;
              break;
            default:
              break;
          }
          Check(static_cast<int64_t>(total) == want,
                tag + ": planned " + std::to_string(total) + " B, rank tensor is " +
                    std::to_string(want) + " B");
          Check(runs.size() == want_runs, tag + ": " + std::to_string(runs.size()) +
                                              " runs, expected " + std::to_string(want_runs));
          int64_t full = 0;  // the whole on-disk part
          switch (part) {
            case Part::kBf16: full = lin.N * lin.K * 2; break;
            case Part::kW4Wq:
            case Part::kMxWq: full = lin.N * lin.K / 2; break;
            case Part::kW4a16Wsz:
            case Part::kW4a8Ws: full = lin.N * (lin.K / group) * 4; break;
            case Part::kMxWs: full = lin.K / 32 * lin.N; break;
            case Part::kMxWref: full = lin.N; break;
            default: break;
          }
          bool ordered = true;
          size_t end = 0;
          for (const ByteRun& b : runs) {
            if (b.src_off < end || b.src_off + b.bytes > static_cast<size_t>(full)) ordered = false;
            end = b.src_off + b.bytes;
          }
          Check(ordered, tag + ": runs ascending, disjoint and inside the " + std::to_string(full) +
                             "-B part");
          ++plans;
        }
      }
    }
  }
  // docs/tp.md 5.1's "no staging copy" list is exactly the single-run plans: qg, z, k/v, lm_head
  // (every part but mxfp4 ws), a/b, A_log, dt_bias, descales, mxfp4 ws cols.
  const auto single = [&](const char* base, const PartShape& shape) {
    const ShardRule rule = RuleFor(base, g);
    for (int r = 0; r < 2; ++r) {
      const auto runs = rule.split == Split::kRows ? PlanRows(shape, RankRows(rule, 2, r))
                                                   : PlanCols(shape, RankCols(rule, 2, r));
      if (runs.size() != 1) return false;
    }
    return true;
  };
  Check(single("text.layers.0.gdn.in_proj_a", {Part::kBf16, 48, 5120, 0, 0}) &&
            single("text.layers.0.gdn.A_log", {Part::kElem, 48, 0, 0, 4}) &&
            single("text.layers.0.gdn.dt_bias", {Part::kElem, 48, 0, 0, 4}) &&
            single("text.layers.3.attn.k_descale", {Part::kElem, 4, 0, 0, 4}) &&
            single("text.layers.0.gdn.out_proj", {Part::kMxWs, 5120, 6144, 32, 0}),
        "a/b, A_log, dt_bias, descales and mxfp4 ws K-slices are one contiguous range");
  Check(!single("text.layers.0.gdn.conv1d_weight", {Part::kElem, 10240, 0, 0, 8}),
        "conv1d_weight's 3 channel ranges go through staging");

  // Byte-exact offsets at the real sizes, from docs/tp.md 4.3's formulas: the small config keeps
  // every offset under 2 MB, so only these catch a narrowed intermediate or a size-dependent branch
  // (lm_head's rank-1 bf16 slice starts at 1.27 GB).
  const auto plan1 = [&](const char* base, const PartShape& shape) {  // rank 1
    const ShardRule rule = RuleFor(base, g);
    return rule.split == Split::kRows ? PlanRows(shape, RankRows(rule, 2, 1))
                                      : PlanCols(shape, RankCols(rule, 2, 1));
  };
  const auto ends = [](const std::vector<ByteRun>& runs, size_t n, size_t first_off,
                       size_t last_off, size_t run_bytes) {
    return runs.size() == n && runs.front().src_off == first_off &&
           runs.front().bytes == run_bytes && runs.back().src_off == last_off &&
           runs.back().bytes == run_bytes;
  };
  const size_t lm_bf16 = size_t{124160} * 5120 * 2;
  Check(ends(plan1("lm_head", {Part::kBf16, 248320, 5120, 0, 0}), 1, lm_bf16, lm_bf16, lm_bf16),
        "real offsets: lm_head bf16 rank 1 = [124160*5120*2, +124160*5120*2)");
  Check(ends(plan1("lm_head", {Part::kMxWs, 248320, 5120, 32, 0}), 160, 124160,
             size_t{159} * 248320 + 124160, 124160),
        "real offsets: lm_head mxfp4 ws rank 1 = [kg*248320 + 124160, +124160), kg = 0..159");
  // mlp.down K = 17408: 272 64-K blocks per tile, rank 1 takes blocks [136, 272) of each of the
  // 320 tiles -- 512 B per (tile, block) in wq, 64 B per (tile, g64 group) in wsz.
  Check(ends(plan1("text.layers.0.mlp.down", {Part::kW4Wq, 5120, 17408, 0, 0}), 320, 136 * 512,
             (size_t{319} * 272 + 136) * 512, 136 * 512),
        "real offsets: mlp.down w4 wq rank 1 = [(t*272 + 136)*512, +136*512), t = 0..319");
  Check(ends(plan1("text.layers.0.mlp.down", {Part::kW4a16Wsz, 5120, 17408, 64, 0}), 320, 136 * 64,
             (size_t{319} * 272 + 136) * 64, 136 * 64),
        "real offsets: mlp.down w4a16 g64 wsz rank 1 = [(t*272 + 136)*64, +136*64), t = 0..319");
  // mlp.gate_up rank 1 rows [8704, +8704) u [26112, +8704): 160 B per row of w4a8 ws (40 groups x
  // 4 B), 2560 B per row of mxfp4 wq.
  Check(ends(plan1("text.layers.0.mlp.gate_up", {Part::kW4a8Ws, 34816, 5120, 128, 0}), 2,
             size_t{8704} * 160, size_t{26112} * 160, size_t{8704} * 160),
        "real offsets: mlp.gate_up w4a8 ws rank 1 = rows 8704.. and 26112.. at 160 B per row");
  Check(ends(plan1("text.layers.0.mlp.gate_up", {Part::kMxWq, 34816, 5120, 32, 0}), 2,
             size_t{8704} * 2560, size_t{26112} * 2560, size_t{8704} * 2560),
        "real offsets: mlp.gate_up mxfp4 wq rank 1 = rows 8704.. and 26112.. at 2560 B per row");
  Pass("real shapes: " + std::to_string(plans) +
       " (tensor, rank, layout, part) plans legal, exact size, ascending and in bounds, contiguous "
       "exactly where 5.1 says; 6 rank-1 offsets pinned to 4.3's formulas");
}

// ---- 3. end to end on the small config ----------------------------------------------------------

void TestSmallModelSlices() {
  const ModelConfig g = SmallConfig();
  const int64_t H = g.hidden_size;  // 512
  for (int r = 0; r < 2; ++r) {
    const std::string p = "small rank " + std::to_string(r) + " ";
    const auto rule_rows = [&](const std::string& base, const std::vector<Range>& expect) {
      const auto got = RankRows(RuleFor(base, g), 2, r);
      Check(SameRanges(got, expect), p + base + ": RankRows " + RangesStr(got) + " == expected " +
                                         RangesStr(expect));
      return expect;
    };

    // gdn.in_proj_qkv: q [0,256) k [256,512) v [512,1536); rank r: q/k 128 rows, v 512 rows.
    const std::vector<Range> qkv = {{128 * r, 128}, {256 + 128 * r, 128}, {512 + 512 * r, 512}};
    rule_rows("text.layers.0.gdn.in_proj_qkv", qkv);
    CheckRowSlice(p + "gdn.in_proj_qkv", RandomW(1536, H, 1), 1536, H, qkv);
    // gdn.conv1d_weight: the same 3 channel ranges, [conv_dim][4] bf16 = 8 B per channel.
    rule_rows("text.layers.0.gdn.conv1d_weight", qkv);
    CheckElemRows(p + "gdn.conv1d_weight", 1536, 8, qkv);
    // gdn.in_proj_z
    const std::vector<Range> z = {{512 * r, 512}};
    rule_rows("text.layers.0.gdn.in_proj_z", z);
    CheckRowSlice(p + "gdn.in_proj_z", RandomW(1024, H, 2), 1024, H, z);
    // gdn.in_proj_a/b: bf16 [8, 512] (never quantized); A_log/dt_bias fp32 [8].
    const std::vector<Range> heads = {{4 * r, 4}};
    rule_rows("text.layers.0.gdn.in_proj_a", heads);
    rule_rows("text.layers.0.gdn.in_proj_b", heads);
    CheckRowSlice(p + "gdn.in_proj_a", RandomW(8, H, 3), 8, H, heads, /*bf16_only=*/true);
    rule_rows("text.layers.0.gdn.A_log", heads);
    rule_rows("text.layers.0.gdn.dt_bias", heads);
    CheckElemRows(p + "gdn.A_log", 8, 4, heads);
    // gdn.out_proj: row-parallel, K = ValueDim 1024 -> [512r, +512).
    const Range out = RankCols(RuleFor("text.layers.0.gdn.out_proj", g), 2, r);
    Check(out.begin == 512 * r && out.count == 512, p + "gdn.out_proj K range");
    CheckColSlice(p + "gdn.out_proj", RandomW(H, 1024, 4), H, 1024, out);

    // attn.qg: per-head interleave [q 256 | gate 256] x 4 heads; rank r = heads 2r, 2r+1 -- built
    // head by head so a slicer that split inside a head would not match.
    std::vector<Range> qg;
    for (int h = 2 * r; h < 2 * r + 2; ++h) {
      if (!qg.empty() && qg.back().begin + qg.back().count == 512 * h) {
        qg.back().count += 512;
      } else {
        qg.push_back({512 * h, 512});
      }
    }
    rule_rows("text.layers.3.attn.qg", qg);
    CheckRowSlice(p + "attn.qg", RandomW(2048, H, 5), 2048, H, qg);
    // attn.k / attn.v: kv head r (256 rows).
    const std::vector<Range> kv = {{256 * r, 256}};
    rule_rows("text.layers.3.attn.k", kv);
    rule_rows("text.layers.3.attn.v", kv);
    CheckRowSlice(p + "attn.k", RandomW(512, H, 6), 512, H, kv);
    // attn.k_descale / v_descale: fp32 [2] -> [1].
    const std::vector<Range> ds = {{r, 1}};
    rule_rows("text.layers.3.attn.k_descale", ds);
    rule_rows("text.layers.3.attn.v_descale", ds);
    CheckElemRows(p + "attn.k_descale", 2, 4, ds);
    // attn.o: row-parallel, K = 1024 -> [512r, +512).
    const Range o = RankCols(RuleFor("text.layers.3.attn.o", g), 2, r);
    Check(o.begin == 512 * r && o.count == 512, p + "attn.o K range");
    CheckColSlice(p + "attn.o", RandomW(H, 1024, 7), H, 1024, o);

    // mlp.gate_up: gate [0,1024) up [1024,2048); rank r: 512 of each.
    const std::vector<Range> gu = {{512 * r, 512}, {1024 + 512 * r, 512}};
    rule_rows("text.layers.0.mlp.gate_up", gu);
    rule_rows("text.layers.3.mlp.gate_up", gu);
    CheckRowSlice(p + "mlp.gate_up", RandomW(2048, H, 8), 2048, H, gu);
    // mlp.down: K = 1024 -> [512r, +512).
    const Range down = RankCols(RuleFor("text.layers.0.mlp.down", g), 2, r);
    Check(down.begin == 512 * r && down.count == 512, p + "mlp.down K range");
    CheckColSlice(p + "mlp.down", RandomW(H, 1024, 9), H, 1024, down);

    // lm_head: vocab 1024 -> [512r, +512).
    const std::vector<Range> vocab = {{512 * r, 512}};
    rule_rows("lm_head", vocab);
    CheckRowSlice(p + "lm_head", RandomW(1024, H, 10), 1024, H, vocab);

    // The MTP head's layer tensors use the body attention-layer rules.
    Check(SameRanges(RankRows(RuleFor("mtp.attn.qg", g), 2, r), qg) &&
              SameRanges(RankRows(RuleFor("mtp.attn.k", g), 2, r), kv) &&
              SameRanges(RankRows(RuleFor("mtp.mlp.gate_up", g), 2, r), gu) &&
              RankCols(RuleFor("mtp.attn.o", g), 2, r).begin == o.begin &&
              RankCols(RuleFor("mtp.mlp.down", g), 2, r).begin == down.begin,
          p + "mtp.* layer tensors split exactly like the body's");
  }
}

// ---- 4. docs/tp.md 10.2's explicit shapes -------------------------------------------------------

void TestDesignCases() {
  // rows, single segment: N = 256, K = 1024.
  {
    ShardRule rule;
    rule.split = Split::kRows;
    rule.segments = {{0, 256}};
    const auto w = RandomW(256, 1024, 21);
    for (int r = 0; r < 2; ++r) {
      const auto rows = RankRows(rule, 2, r);
      Check(SameRanges(rows, {{128 * r, 128}}), "10.2 single segment: rank rows");
      CheckRowSlice("10.2 single segment N=256 K=1024", w, 256, 1024, rows);
    }
  }
  // rows, fused: qkv-shaped {64, 64, 192}, gate_up-shaped {96, 96}, qg-shaped 4 heads x 32 rows.
  struct Fused {
    const char* name;
    std::vector<int64_t> segs;
    std::vector<Range> rank0;
  };
  const Fused fused[] = {
      {"10.2 qkv-shaped {64,64,192}", {64, 64, 192}, {{0, 32}, {64, 32}, {128, 96}}},
      {"10.2 gate_up-shaped {96,96}", {96, 96}, {{0, 48}, {96, 48}}},
      {"10.2 qg-shaped 4 heads x 32", {128}, {{0, 64}}},
  };
  for (const Fused& f : fused) {
    ShardRule rule;
    rule.split = Split::kRows;
    int64_t n = 0;
    for (int64_t s : f.segs) {
      rule.segments.push_back({n, s});
      n += s;
    }
    const auto w = RandomW(n, 512, 22);
    Check(SameRanges(RankRows(rule, 2, 0), f.rank0), std::string(f.name) + ": rank 0 rows");
    for (int r = 0; r < 2; ++r) CheckRowSlice(f.name, w, n, 512, RankRows(rule, 2, r));
  }
  // cols: N = 64, K = 1536 -> [0,768) / [768,1536).
  {
    const auto w = RandomW(64, 1536, 23);
    Check(CheckColSlice("10.2 N=64 K=1536", w, 64, 1536, {0, 768}) == 0 &&
              CheckColSlice("10.2 N=64 K=1536", w, 64, 1536, {768, 768}) == 0,
          "10.2 K=1536: every layout splits at 768");
  }
  // cols: per-rank K = 1088 = 17 x 64 (full K 2176), mirroring mlp.down's per-rank 8704 = 17 x 512:
  // legal for bf16, w4a16 at g=64 and mxfp4; the four group-128 layouts (w4a16 g128 RTN/search,
  // w4a8 RTN/search) must refuse.
  {
    const auto w = RandomW(64, 2176, 24);
    Check(CheckColSlice("10.2 N=64 K=2176 (rank K 17x64)", w, 64, 2176, {0, 1088}) == 4 &&
              CheckColSlice("10.2 N=64 K=2176 (rank K 17x64)", w, 64, 2176, {1088, 1088}) == 4,
          "10.2 rank K 17 x 64: exactly the four group-128 layouts are refused");
    Check(Throws([&] { PlanCols({Part::kW4a8Ws, 64, 2176, 128, 0}, {0, 1088}); }) &&
              Throws([&] { PlanCols({Part::kW4a16Wsz, 64, 2176, 128, 0}, {1088, 1088}); }),
          "10.2: a 17 x 64 rank K cannot split a group-128 scale tensor");
  }
}

// ---- 5. the mxfp4 wref exception ----------------------------------------------------------------

// Dequant of packed mxfp4 by the kernel's own arithmetic (r4d_gemm_mxfp4a8_nt_m64.hip lines
// 190-192, 225): the lane word of (tile nt, k-step ks) holds k = ks*16 + 8*(lane>>4) + 2j (+1);
// dsh = clamp(wref[n] - ws[k/32][n], 0, 15); value = e2m1 * 2^-dsh * 2^(wref[n]-127).
std::vector<double> DequantMxfp4(const std::vector<uint8_t>& wq, const std::vector<uint8_t>& ws,
                                 const std::vector<uint8_t>& wref, int64_t N, int64_t K) {
  std::vector<double> out(static_cast<size_t>(N * K));
  const int64_t ksteps = K / 16;
  for (int64_t nt = 0; nt < N / 16; ++nt) {
    for (int64_t ks = 0; ks < ksteps; ++ks) {
      for (int lane = 0; lane < 32; ++lane) {
        const int64_t row = nt * 16 + (lane & 15);
        const size_t base = static_cast<size_t>(((nt * ksteps + ks) * 32 + lane) * 4);
        for (int j = 0; j < 4; ++j) {
          for (int half = 0; half < 2; ++half) {
            const uint8_t code = half ? (wq[base + j] >> 4) & 0xF : wq[base + j] & 0xF;
            const int64_t k = ks * 16 + 8 * (lane >> 4) + 2 * j + half;
            const int ref = wref[static_cast<size_t>(row)];
            int dsh = ref - ws[static_cast<size_t>((k / 32) * N + row)];
            dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
            const double mag = r4dx_convert::kE2M1Magnitude[code & 0x7];
            out[static_cast<size_t>(row * K + k)] =
                ((code & 0x8) ? -1.0 : 1.0) * std::ldexp(mag, ref - 127 - dsh);
          }
        }
      }
    }
  }
  return out;
}

void TestMxfp4WrefException() {
  const int64_t N = 32, K = 1024, half = 512;
  std::vector<float> w = RandomW(N, K, 31);
  // Row 5: rank 0's K half is ~1e-7 (E8M0 ~103) against ~0.02-0.5 in rank 1's half (~121-124):
  // TP=1 clamps rank 0's groups at dsh 15; the shard's own wref (~103) would not. Row 20: reverse.
  std::mt19937 rng(32);
  std::normal_distribution<float> tiny(0.0f, 1e-7f);
  for (int64_t k = 0; k < half; ++k) w[static_cast<size_t>(5 * K + k)] = tiny(rng);
  for (int64_t k = half; k < K; ++k) w[static_cast<size_t>(20 * K + k)] = tiny(rng);
  const std::vector<float> imp = RandomImportance(K, 33);

  for (const Variant& v : kVariants) {
    if (v.scheme != Scheme::kMxfp4 && v.scheme != Scheme::kMxfp4Search) continue;
    const auto full = Pack(v, w, N, K, imp);  // wq, ws, wref
    const auto dq_full = DequantMxfp4(full[0].bytes, full[1].bytes, full[2].bytes, N, K);
    for (int r = 0; r < 2; ++r) {
      const Range cols{r * half, half};
      const std::string p = std::string(v.name) + " rank " + std::to_string(r) + ": ";
      std::vector<std::vector<uint8_t>> sliced;
      for (int i = 0; i < 3; ++i) {
        const PartShape shape{full[i].part, N, K, full[i].group, 0};
        sliced.push_back(Gather(full[i].bytes.data(), full[i].bytes.size(), PlanCols(shape, cols)));
      }
      Check(sliced[2] == full[2].bytes, p + "the K-slice keeps the FULL wref");
      const auto dq = DequantMxfp4(sliced[0], sliced[1], sliced[2], N, half);
      bool equal = true;
      for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < half; ++k) {
          equal = equal && dq[static_cast<size_t>(n * half + k)] ==
                               dq_full[static_cast<size_t>(n * K + cols.begin + k)];
        }
      }
      Check(equal, p + "dequant(sliced wq, sliced ws, full wref) == the full dequant's columns, "
                       "every element");

      // Why the exception exists: the shard's own pack has a smaller wref on the tiny row and
      // un-clamps groups TP=1 clamps.
      const std::vector<float> imp_s(imp.begin() + cols.begin, imp.begin() + cols.begin + half);
      const auto own = Pack(v, SliceCols(w, N, K, cols), N, half, imp_s);
      const size_t tiny_row = r == 0 ? 5 : 20;
      Check(own[0].bytes == sliced[0] && own[1].bytes == sliced[1],
            p + "wq and ws K-slices are the shard's own bytes");
      const int own_ref = own[2].bytes[tiny_row], full_ref = full[2].bytes[tiny_row];
      Check(full_ref - own_ref > 15, p + "the shard's own wref on row " +
                                         std::to_string(tiny_row) + " (" +
                                         std::to_string(own_ref) + ") is > 15 below the full "
                                         "row's (" + std::to_string(full_ref) +
                                         "): TP=1 clamps there (test setup)");
      const auto dq_own = DequantMxfp4(own[0].bytes, own[1].bytes, own[2].bytes, N, half);
      bool differs = false;
      for (int64_t k = 0; k < half; ++k) {
        differs = differs || dq_own[tiny_row * half + static_cast<size_t>(k)] !=
                                 dq_full[tiny_row * K + static_cast<size_t>(cols.begin + k)];
      }
      Check(differs, p + "dequant with the shard's own wref would differ from TP=1 on row " +
                         std::to_string(tiny_row));
    }
  }
  Pass("mxfp4 wref exception: K-slices keep the full wref and dequantize exactly like TP=1");
}

// ---- 6. misalignment, bounds --------------------------------------------------------------------

void TestMisalignment() {
  const int64_t N = 256, K = 1024;
  const std::vector<Range> bad_rows = {{0, 120}};  // not whole 16-row tiles
  const std::vector<Range> bad_begin = {{8, 128}};
  for (Part p : {Part::kW4Wq, Part::kMxWq, Part::kW4a16Wsz, Part::kW4a8Ws}) {
    const PartShape s{p, N, K, 128, 0};
    Check(Throws([&] { PlanRows(s, bad_rows); }) && Throws([&] { PlanRows(s, bad_begin); }),
          "PlanRows: a tile part refuses a row range not % 16 (part " +
              std::to_string(static_cast<int>(p)) + ")");
  }
  // bf16 / mxfp4 ws / wref / elem rows need no tile alignment.
  Check(!Throws([&] { PlanRows({Part::kBf16, N, K, 0, 0}, bad_rows); }) &&
            !Throws([&] { PlanRows({Part::kMxWs, N, K, 32, 0}, bad_rows); }) &&
            !Throws([&] { PlanRows({Part::kMxWref, N, K, 32, 0}, bad_rows); }) &&
            !Throws([&] { PlanRows({Part::kElem, N, 0, 0, 4}, bad_rows); }),
        "PlanRows: bf16 / mxfp4 ws / wref / elem accept any row range");
  // Column alignment: w4 wq % 64, scale dwords % group, mxfp4 wq/ws % 32.
  Check(Throws([&] { PlanCols({Part::kW4Wq, N, K, 0, 0}, {32, 512}); }) &&
            !Throws([&] { PlanCols({Part::kW4Wq, N, K, 0, 0}, {64, 512}); }),
        "PlanCols: w4 wq needs % 64");
  Check(Throws([&] { PlanCols({Part::kW4a16Wsz, N, K, 64, 0}, {32, 512}); }) &&
            !Throws([&] { PlanCols({Part::kW4a16Wsz, N, K, 64, 0}, {64, 512}); }) &&
            Throws([&] { PlanCols({Part::kW4a16Wsz, N, K, 128, 0}, {64, 512}); }),
        "PlanCols: w4a16 wsz needs % group (64 or 128)");
  Check(Throws([&] { PlanCols({Part::kW4a8Ws, N, K, 128, 0}, {64, 512}); }),
        "PlanCols: w4a8 ws needs % 128");
  Check(Throws([&] { PlanCols({Part::kMxWq, N, K, 32, 0}, {16, 512}); }) &&
            Throws([&] { PlanCols({Part::kMxWs, N, K, 32, 0}, {16, 512}); }) &&
            !Throws([&] { PlanCols({Part::kMxWq, N, K, 32, 0}, {32, 512}); }),
        "PlanCols: mxfp4 wq/ws need % 32");
  Check(!Throws([&] { PlanCols({Part::kBf16, N, K, 0, 0}, {3, 5}); }),
        "PlanCols: bf16 accepts any column range");
  // Bounds and impossible shapes.
  Check(Throws([&] { PlanRows({Part::kBf16, N, K, 0, 0}, {{240, 32}}); }) &&
            Throws([&] { PlanRows({Part::kBf16, N, K, 0, 0}, {{-16, 16}}); }) &&
            Throws([&] { PlanCols({Part::kBf16, N, K, 0, 0}, {1000, 64}); }),
        "PlanRows/PlanCols refuse ranges outside the tensor");
  Check(Throws([&] { PlanRows({Part::kW4Wq, 250, K, 0, 0}, {{0, 16}}); }) &&
            Throws([&] { PlanRows({Part::kW4a16Wsz, N, 1000, 64, 0}, {{0, 16}}); }) &&
            Throws([&] { PlanRows({Part::kElem, N, 0, 0, 0}, {{0, 16}}); }),
        "PlanRows refuses shapes the part cannot have");
  // Gather bounds.
  const std::vector<uint8_t> buf(64, 7);
  bool oob = false;
  try {
    Gather(buf.data(), buf.size(), {{60, 8}});
  } catch (const std::out_of_range&) {
    oob = true;
  }
  Check(oob, "Gather refuses a run past the end of the tensor");
  Check(Gather(buf.data(), buf.size(), {{0, 4}, {60, 4}}).size() == 8, "Gather concatenates runs");
  Pass("misaligned / out-of-bounds plans and runs are refused");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("r4dx_convert::kW4A16Group = %d (build default); packing at g=64 and g=128 below\n",
              r4dx_convert::kW4A16Group);
  TestRuleForReal();
  TestRealPlans();
  TestSmallModelSlices();
  TestDesignCases();
  TestMxfp4WrefException();
  TestMisalignment();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("ALL PASS (%d checks)\n", g_checks);
  return 0;
}
