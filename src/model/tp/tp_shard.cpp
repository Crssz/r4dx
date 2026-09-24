#include "tp/tp_shard.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace r4dx::model::tp {

namespace {

[[noreturn]] void Fail(const std::string& what) {
  throw std::invalid_argument("r4dx::model::tp::" + what);
}

bool StartsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

// True iff `base` still ends in an on-disk ".{layout}.{part}" tail (".w4a16.wq", ".bf16.w", ...).
bool HasLayoutSuffix(const std::string& base) {
  const size_t last = base.rfind('.');
  if (last == std::string::npos || last == 0) return false;
  const size_t prev = base.rfind('.', last - 1);
  const size_t begin = prev == std::string::npos ? 0 : prev + 1;
  const std::string layout = base.substr(begin, last - begin);
  return layout == "bf16" || layout == "w4a16" || layout == "w4a8" || layout == "mxfp4";
}

ShardRule Replicate() { return ShardRule{}; }

ShardRule Rows(const std::vector<int64_t>& segment_rows) {
  ShardRule r;
  r.split = Split::kRows;
  int64_t begin = 0;
  for (int64_t rows : segment_rows) {
    r.segments.push_back({begin, rows});
    begin += rows;
  }
  return r;
}

ShardRule Cols(int64_t k_total) {
  ShardRule r;
  r.split = Split::kCols;
  r.k_total = k_total;
  return r;
}

// Which layer type a per-layer suffix belongs to (a GDN tensor on a full-attention layer means the
// name or the config is wrong, not something to shard).
enum class LayerKind { kAny, kGdn, kAttn };

// The per-layer table shared by `text.layers.{i}.` and `mtp.` (docs/tp.md 5.2), keyed by the suffix
// after that prefix. Every rule's segments are in on-disk row order (the converter's fusion order:
// in_proj_qkv = q | k | v, conv1d_weight the same channels, gate_up = gate | up, qg = per-head
// [q | gate] interleave, so a contiguous block of heads is a contiguous block of rows).
bool LookupLayerRule(const std::string& s, const ModelConfig& g, ShardRule* rule, LayerKind* kind) {
  const int64_t attn_out = g.num_attention_heads * g.head_dim;
  const int64_t kv_rows = g.num_key_value_heads * g.head_dim;
  *kind = LayerKind::kAny;
  if (s == "input_layernorm" || s == "post_attention_layernorm") {
    *rule = Replicate();
  } else if (s == "mlp.gate_up") {
    *rule = Rows({g.intermediate_size, g.intermediate_size});
  } else if (s == "mlp.down") {
    *rule = Cols(g.intermediate_size);
  } else if (s == "gdn.in_proj_qkv" || s == "gdn.conv1d_weight") {
    *kind = LayerKind::kGdn;
    *rule = Rows({g.KeyDim(), g.KeyDim(), g.ValueDim()});
  } else if (s == "gdn.in_proj_z") {
    *kind = LayerKind::kGdn;
    *rule = Rows({g.ValueDim()});
  } else if (s == "gdn.in_proj_a" || s == "gdn.in_proj_b" || s == "gdn.A_log" ||
             s == "gdn.dt_bias") {
    *kind = LayerKind::kGdn;
    *rule = Rows({g.linear_num_value_heads});
  } else if (s == "gdn.norm_weight") {
    *kind = LayerKind::kGdn;
    *rule = Replicate();
  } else if (s == "gdn.out_proj") {
    *kind = LayerKind::kGdn;
    *rule = Cols(g.ValueDim());
  } else if (s == "attn.qg") {
    *kind = LayerKind::kAttn;
    *rule = Rows({2 * attn_out});
  } else if (s == "attn.k" || s == "attn.v") {
    *kind = LayerKind::kAttn;
    *rule = Rows({kv_rows});
  } else if (s == "attn.k_descale" || s == "attn.v_descale") {
    *kind = LayerKind::kAttn;
    *rule = Rows({g.num_key_value_heads});
  } else if (s == "attn.o") {
    *kind = LayerKind::kAttn;
    *rule = Cols(attn_out);
  } else if (s == "attn.q_norm" || s == "attn.k_norm") {
    *kind = LayerKind::kAttn;
    *rule = Replicate();
  } else {
    return false;
  }
  return true;
}

void CheckWorldRank(const char* fn, int world, int rank) {
  if (world < 1 || rank < 0 || rank >= world) {
    Fail(std::string(fn) + ": need world >= 1 and 0 <= rank < world, got world " +
         std::to_string(world) + ", rank " + std::to_string(rank));
  }
}

const char* PartName(Part p) {
  switch (p) {
    case Part::kBf16: return "bf16";
    case Part::kW4Wq: return "w4 wq";
    case Part::kW4a16Wsz: return "w4a16 wsz";
    case Part::kW4a8Ws: return "w4a8 ws";
    case Part::kMxWq: return "mxfp4 wq";
    case Part::kMxWs: return "mxfp4 ws";
    case Part::kMxWref: return "mxfp4 wref";
    case Part::kElem: return "elem";
  }
  return "?";
}

bool IsTilePart(Part p) {
  return p == Part::kW4Wq || p == Part::kW4a16Wsz || p == Part::kW4a8Ws || p == Part::kMxWq;
}

void RequireDiv(const char* fn, const PartShape& sh, const char* what, int64_t value,
                int64_t divisor) {
  if (divisor <= 0 || value % divisor != 0) {
    Fail(std::string(fn) + ": " + PartName(sh.part) + " " + what + " = " + std::to_string(value) +
         " is not a multiple of " + std::to_string(divisor));
  }
}

// The shape itself must be one the part can have at all (the packers' own RequireDivisible rules),
// otherwise the per-tile / per-group arithmetic below would silently truncate.
void CheckShape(const char* fn, const PartShape& sh) {
  if (sh.N <= 0) Fail(std::string(fn) + ": " + PartName(sh.part) + " needs N > 0");
  if (sh.part == Part::kElem) {
    if (sh.row_bytes <= 0) Fail(std::string(fn) + ": elem part needs row_bytes > 0");
    return;
  }
  if (sh.K <= 0) Fail(std::string(fn) + ": " + PartName(sh.part) + " needs K > 0");
  switch (sh.part) {
    case Part::kW4Wq:
      RequireDiv(fn, sh, "N", sh.N, 16);
      RequireDiv(fn, sh, "K", sh.K, 64);
      break;
    case Part::kW4a16Wsz:
    case Part::kW4a8Ws:
      RequireDiv(fn, sh, "N", sh.N, 16);
      RequireDiv(fn, sh, "K", sh.K, sh.group);
      break;
    case Part::kMxWq:
      RequireDiv(fn, sh, "N", sh.N, 16);
      RequireDiv(fn, sh, "K", sh.K, 32);
      break;
    case Part::kMxWs:
      RequireDiv(fn, sh, "K", sh.K, sh.group);
      break;
    default:
      break;
  }
}

void CheckRange(const char* fn, const PartShape& sh, const char* axis, Range r, int64_t extent) {
  if (r.begin < 0 || r.count < 0 || r.begin + r.count > extent) {
    Fail(std::string(fn) + ": " + PartName(sh.part) + " " + axis + " range [" +
         std::to_string(r.begin) + ", " + std::to_string(r.begin + r.count) +
         ") is outside [0, " + std::to_string(extent) + ")");
  }
}

void Append(std::vector<ByteRun>& runs, int64_t off, int64_t bytes) {
  if (bytes <= 0) return;
  const size_t o = static_cast<size_t>(off), b = static_cast<size_t>(bytes);
  if (!runs.empty() && runs.back().src_off + runs.back().bytes == o) {
    runs.back().bytes += b;
  } else {
    runs.push_back({o, b});
  }
}

}  // namespace

ShardRule RuleFor(const std::string& base, const ModelConfig& global) {
  if (global.tp_world != 1) {
    Fail("RuleFor: `global` must be the unsharded config (tp_world " +
         std::to_string(global.tp_world) + ")");
  }
  const auto unknown = [&]() {
    Fail("RuleFor: unknown tensor '" + base +
         "' -- pass the base name without its .{layout} suffix; a new tensor must be classified "
         "in src/model/tp/tp_shard.cpp (docs/tp.md 4.2), never replicated by default");
  };
  // Every path, the vision.* / dflash.* prefix rules included.
  if (HasLayoutSuffix(base)) unknown();

  static constexpr char kText[] = "text.layers.";
  if (StartsWith(base, kText)) {
    const size_t p = sizeof(kText) - 1;
    size_t q = p;
    while (q < base.size() && q - p < 9 && base[q] >= '0' && base[q] <= '9') ++q;
    if (q == p || q >= base.size() || base[q] != '.') unknown();
    const int64_t layer = std::stoll(base.substr(p, q - p));
    if (layer >= global.num_hidden_layers) {
      Fail("RuleFor: '" + base + "' names layer " + std::to_string(layer) + " of a " +
           std::to_string(global.num_hidden_layers) + "-layer model");
    }
    ShardRule rule;
    LayerKind kind;
    if (!LookupLayerRule(base.substr(q + 1), global, &rule, &kind)) unknown();
    const bool gdn_layer = global.IsGdnLayer(layer);
    if ((kind == LayerKind::kGdn && !gdn_layer) || (kind == LayerKind::kAttn && gdn_layer)) {
      Fail("RuleFor: '" + base + "' is a " + (gdn_layer ? "full-attention" : "GDN") +
           " tensor on a " + (gdn_layer ? "GDN" : "full-attention") + " layer");
    }
    return rule;
  }

  if (StartsWith(base, "mtp.")) {
    // The MTP head is one full-attention decoder layer (sharded exactly like a body attention
    // layer + MLP) plus four replicated tensors and the optional replicated reduced-vocab head.
    const std::string s = base.substr(4);
    if (s == "fc" || s == "norm" || s == "pre_fc_norm_hidden" || s == "pre_fc_norm_embedding" ||
        s == "draft_head.lm_head" || s == "draft_head.vocab_ids") {
      return Replicate();
    }
    ShardRule rule;
    LayerKind kind;
    if (!LookupLayerRule(s, global, &rule, &kind) || kind == LayerKind::kGdn) unknown();
    return rule;
  }

  if (base == "lm_head") return Rows({global.vocab_size});
  if (base == "text.embed_tokens" || base == "text.final_norm") return Replicate();
  if (StartsWith(base, "vision.")) {
    ShardRule r;
    r.split = Split::kRank0Only;
    return r;
  }
  if (StartsWith(base, "dflash.")) return Replicate();  // replicated drafter (docs/tp.md 8.2)
  unknown();
  return {};  // unreachable
}

std::vector<Range> RankRows(const ShardRule& r, int world, int rank) {
  CheckWorldRank("RankRows", world, rank);
  if (r.split != Split::kRows) Fail("RankRows: the rule is not a row (column-parallel) split");
  std::vector<Range> out;
  out.reserve(r.segments.size());
  for (const Segment& s : r.segments) {
    if (s.rows % world != 0) {
      Fail("RankRows: a segment of " + std::to_string(s.rows) + " rows does not split into " +
           std::to_string(world) + " equal parts");
    }
    const int64_t part = s.rows / world;
    out.push_back({s.begin + rank * part, part});
  }
  return out;
}

Range RankCols(const ShardRule& r, int world, int rank) {
  CheckWorldRank("RankCols", world, rank);
  if (r.split != Split::kCols) Fail("RankCols: the rule is not a column (row-parallel) split");
  if (r.k_total % world != 0) {
    Fail("RankCols: K = " + std::to_string(r.k_total) + " does not split into " +
         std::to_string(world) + " equal parts");
  }
  const int64_t part = r.k_total / world;
  return {rank * part, part};
}

std::vector<ByteRun> PlanRows(const PartShape& sh, const std::vector<Range>& rows) {
  CheckShape("PlanRows", sh);
  const int64_t N = sh.N, K = sh.K;
  std::vector<ByteRun> runs;
  for (const Range& r : rows) {
    CheckRange("PlanRows", sh, "row", r, N);
    if (IsTilePart(sh.part) && (r.begin % 16 != 0 || r.count % 16 != 0)) {
      Fail("PlanRows: " + std::string(PartName(sh.part)) + " row range [" +
           std::to_string(r.begin) + ", " + std::to_string(r.begin + r.count) +
           ") is not whole 16-row tiles");
    }
    const int64_t a = r.begin, n = r.count;
    switch (sh.part) {
      case Part::kBf16:
        Append(runs, a * K * 2, n * K * 2);
        break;
      case Part::kW4Wq:  // tile-major, K/2 bytes per row: rows [a, a+n) are one byte range
      case Part::kMxWq:
        Append(runs, a * K / 2, n * K / 2);
        break;
      case Part::kW4a16Wsz:  // (t, g, r) dwords, K/g per row, tile-major
      case Part::kW4a8Ws:
        Append(runs, a * (K / sh.group) * 4, n * (K / sh.group) * 4);
        break;
      case Part::kMxWref:
        Append(runs, a, n);
        break;
      case Part::kElem:
        Append(runs, a * sh.row_bytes, n * sh.row_bytes);
        break;
      case Part::kMxWs:
        break;  // below: the row ranges are gathered inside every k-group row
    }
  }
  if (sh.part == Part::kMxWs) {
    for (int64_t kg = 0; kg < K / sh.group; ++kg) {
      for (const Range& r : rows) Append(runs, kg * N + r.begin, r.count);
    }
  }
  return runs;
}

std::vector<ByteRun> PlanCols(const PartShape& sh, Range cols) {
  CheckShape("PlanCols", sh);
  if (sh.part == Part::kElem) Fail("PlanCols: an elem part has no column axis");
  const int64_t N = sh.N, K = sh.K;
  CheckRange("PlanCols", sh, "column", cols, K);
  int64_t align = 1;
  switch (sh.part) {
    case Part::kW4Wq: align = 64; break;
    case Part::kW4a16Wsz:
    case Part::kW4a8Ws:
    case Part::kMxWs: align = sh.group; break;
    case Part::kMxWq: align = 32; break;
    default: break;
  }
  if (cols.begin % align != 0 || cols.count % align != 0) {
    Fail("PlanCols: " + std::string(PartName(sh.part)) + " column range [" +
         std::to_string(cols.begin) + ", " + std::to_string(cols.begin + cols.count) +
         ") is not a multiple of " + std::to_string(align));
  }
  const int64_t k0 = cols.begin, k = cols.count;
  std::vector<ByteRun> runs;
  switch (sh.part) {
    case Part::kBf16:
      for (int64_t n = 0; n < N; ++n) Append(runs, (n * K + k0) * 2, k * 2);
      break;
    case Part::kW4Wq:  // 512 B per (tile, 64-K block)
      for (int64_t t = 0; t < N / 16; ++t) {
        Append(runs, (t * (K / 64) + k0 / 64) * 512, (k / 64) * 512);
      }
      break;
    case Part::kW4a16Wsz:  // 16 dwords per (tile, group)
    case Part::kW4a8Ws: {
      const int64_t g = sh.group;
      for (int64_t t = 0; t < N / 16; ++t) {
        Append(runs, (t * (K / g) + k0 / g) * 16 * 4, (k / g) * 16 * 4);
      }
      break;
    }
    case Part::kMxWq:  // 128 B per (tile, 16-K step)
      for (int64_t t = 0; t < N / 16; ++t) {
        Append(runs, (t * (K / 16) + k0 / 16) * 128, (k / 16) * 128);
      }
      break;
    case Part::kMxWs:  // [K/32][N]: a K range is a contiguous block of k-group rows
      Append(runs, (k0 / sh.group) * N, (k / sh.group) * N);
      break;
    case Part::kMxWref:  // the FULL row max, see the header
      Append(runs, 0, N);
      break;
    case Part::kElem:
      break;
  }
  return runs;
}

std::vector<uint8_t> Gather(const uint8_t* full, size_t full_bytes,
                            const std::vector<ByteRun>& runs) {
  size_t total = 0;
  for (const ByteRun& r : runs) {
    if (r.src_off > full_bytes || r.bytes > full_bytes - r.src_off) {
      throw std::out_of_range("r4dx::model::tp::Gather: run [" + std::to_string(r.src_off) + ", " +
                              std::to_string(r.src_off + r.bytes) + ") is past the " +
                              std::to_string(full_bytes) + "-byte tensor");
    }
    total += r.bytes;
  }
  std::vector<uint8_t> out(total);
  size_t o = 0;
  for (const ByteRun& r : runs) {
    std::memcpy(out.data() + o, full + r.src_off, r.bytes);
    o += r.bytes;
  }
  return out;
}

}  // namespace r4dx::model::tp
