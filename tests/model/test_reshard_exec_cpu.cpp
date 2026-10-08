// tests/model/test_reshard_exec_cpu.cpp -- CPU-only checks of src/model/reshard_exec.h, the address arithmetic of the hybrid mode's
// reshard EXECUTOR (docs/pp-tp2-hybrid.md 3, 7): what Model::ReshardExport / ReshardImport / ReshardCopyLocal run on the device is
// hybrid::RectOf over a plan's CopyOp and a row slice, so this test runs the same RectOf over host buffers in device layout.
//   * SplitOp tiles every run of every op of the real 27B plans exactly once, in order, within the byte budget; PackCrossOps packs
//     the cross-card ops of each source stage into ring batches that cover each row exactly once, are full to within one row, rotate
//     the slots and add up to PlanTotals;
//   * a host executor (export through RectOf into a ring batch, import through RectOf) rebuilds, for scatter and for gather, the
//     tiny model's rank / stage buffers byte for byte against ReshardRef -- with stage B's conv lines LONGER than stage A's (stage B
//     carries the MTP head, so it is sized for the verify window: PlanParams::stage_b_conv_pitch) and a speculating rank's pitch;
//   * NEGATIVE CONTROLS: the wrong side, a skipped batch, a slice one row off, stage B's pitch left at stage A's, each break the result.
// No HIP call, no container; always runs.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "reshard_exec.h"

using namespace r4dx::model;
using namespace r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
template <class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

ModelConfig RealConfig() {
  ModelConfig c;
  c.hidden_size = 5120;
  c.num_hidden_layers = 64;
  for (int i = 0; i < 64; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
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
ModelConfig TinyConfig() {
  ModelConfig c;
  c.hidden_size = 32;
  c.num_hidden_layers = 8;
  for (int i = 0; i < 8; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  c.num_attention_heads = 8;
  c.num_key_value_heads = 4;
  c.head_dim = 4;
  c.intermediate_size = 16;
  c.linear_num_key_heads = 4;
  c.linear_num_value_heads = 8;
  c.linear_key_head_dim = 4;
  c.linear_value_head_dim = 4;
  c.vocab_size = 64;
  return c;
}

// ---- 1. slices and ring batches on the real geometry ------------------------------------------------------------------------
uint64_t MaxWidth(const ReshardPlan& plan) {
  uint64_t w = 0;
  for (const CopyOp& op : plan.ops) {
    for (const Run2D& r : op.runs) w = std::max(w, r.width);
  }
  return w;
}

void SlicesTileRuns(const ReshardPlan& plan) {
  bool whole_ok = true, split_ok = true;
  for (const CopyOp& op : plan.ops) {
    uint64_t whole = 0;
    for (const OpSlice& s : WholeOp(op)) whole += SliceBytes(op, s);
    whole_ok = whole_ok && whole == op.Bytes();
    for (const uint64_t budget : {uint64_t{1}, uint64_t{4096}, uint64_t{1} << 20, uint64_t{200} << 20}) {
      std::vector<uint64_t> next(op.runs.size(), 0);
      uint64_t bytes = 0;
      size_t last_run = 0;
      for (const OpSlice& s : SplitOp(op, budget)) {
        const Run2D& r = op.runs[s.run];
        split_ok = split_ok && s.rows >= 1 && s.run >= last_run && s.row0 == next[s.run];  // in order, no gap, no overlap
        split_ok = split_ok && (SliceBytes(op, s) <= budget || s.rows == 1);               // within the budget unless one row is wider
        next[s.run] = s.row0 + s.rows;
        last_run = s.run;
        bytes += SliceBytes(op, s);
        (void)r;
      }
      for (size_t i = 0; i < op.runs.size(); ++i) split_ok = split_ok && next[i] == op.runs[i].height;
      split_ok = split_ok && bytes == op.Bytes();
    }
  }
  Check(whole_ok, "WholeOp: the slices of every op add up to the op's bytes");
  Check(split_ok, "SplitOp: for every op and budget the slices tile each run once, in order, within the budget (or one row)");
}

void BatchesCoverCrossOps(const ReshardPlan& plan, uint64_t piece, const char* label) {
  const PlanTotals tot = Totals(plan);
  for (int src = 0; src < 2; ++src) {
    const std::vector<RingBatch> batches = PackCrossOps(plan, src, piece);
    uint64_t sum = 0;
    bool within = true, full = true, slots = true, local_free = true;
    std::map<std::pair<size_t, std::pair<size_t, uint64_t>>, int> seen;  // (op, (run, row)) -> times
    for (size_t b = 0; b < batches.size(); ++b) {
      const RingBatch& rb = batches[b];
      uint64_t bytes = 0;
      for (const SliceRef& r : rb.slices) {
        const CopyOp& op = plan.ops.at(r.op);
        local_free = local_free && !op.local && SourceStage(op, plan.dir) == src;
        bytes += SliceBytes(op, r.slice);
        for (uint64_t i = 0; i < r.slice.rows; ++i) ++seen[{r.op, {r.slice.run, r.slice.row0 + i}}];
      }
      within = within && bytes == rb.bytes && bytes <= piece;
      if (b + 1 < batches.size()) full = full && bytes + MaxWidth(plan) > piece;
      slots = slots && rb.slot == static_cast<int>(b % kRingSlots);
      sum += bytes;
    }
    Check(sum == tot.CrossFrom(src), label);
    Check(within, "PackCrossOps: every batch is within the piece size and its bytes are the sum of its slices");
    Check(full, "PackCrossOps: every batch but the last is full to within one row width");
    Check(slots, "PackCrossOps: the ring slots rotate round robin");
    Check(local_free, "PackCrossOps: only cross-card ops of the requested source stage (never a local op)");
    // every row of every cross op from `src` exactly once
    bool once = true;
    uint64_t expect_rows = 0;
    for (size_t i = 0; i < plan.ops.size(); ++i) {
      const CopyOp& op = plan.ops[i];
      if (op.local || SourceStage(op, plan.dir) != src) continue;
      for (size_t ri = 0; ri < op.runs.size(); ++ri) {
        for (uint64_t row = 0; row < op.runs[ri].height; ++row) {
          ++expect_rows;
          const auto it = seen.find({i, {ri, row}});
          once = once && it != seen.end() && it->second == 1;
        }
      }
    }
    Check(once && seen.size() == expect_rows, "PackCrossOps: every row of every cross op is carried exactly once");
  }
}

void RealGeometry() {
  const StateGeometry g = StateGeometry::FromRules(RealConfig(), 16);
  for (const int64_t split : {32, 35}) {
    PlanParams p = PlanParams::ForSplit(g, split, /*rank_conv_pitch=*/10);
    p.stage_b_conv_pitch = 10;  // stage B with an MTP head k = 7: 2 + 8 entries
    const ReshardPlan scatter = ScatterPlan(g, p, 0, 8192);
    SlicesTileRuns(scatter);
    BatchesCoverCrossOps(scatter, kRingPieceBytes, "PackCrossOps (128 MiB pieces, 8k scatter): the batches add up to the plan's crossing bytes from this stage");
    BatchesCoverCrossOps(scatter, uint64_t{3} << 20, "PackCrossOps (3 MiB pieces): the batches add up to the plan's crossing bytes");
    BatchesCoverCrossOps(scatter, MaxWidth(scatter), "PackCrossOps (pieces of exactly the widest row, the 1.5 MiB recurrent half): the batches add up to the plan's crossing bytes");
    TpMasterTracker t(16, true);
    t.TpOnly(0);
    const ReshardPlan gather = GatherPlan(g, p, t.PlanSync(5000));
    SlicesTileRuns(gather);
    BatchesCoverCrossOps(gather, uint64_t{2} << 20, "PackCrossOps (gather, 2 MiB pieces): the batches add up to the plan's crossing bytes");
    // the 128 MiB ring pieces of the byte stream (ChunkRing) and the row-aligned batches agree on the total
    for (int src = 0; src < 2; ++src) {
      uint64_t ring = 0;
      for (const RingPiece& rp : RingPiecesFrom(scatter, src)) ring += rp.bytes;
      uint64_t batch = 0;
      for (const RingBatch& b : PackCrossOps(scatter, src)) batch += b.bytes;
      Check(ring == batch, "ChunkRing's byte pieces and PackCrossOps' row batches carry the same total");
    }
  }
  const PlanParams p0 = PlanParams::ForSplit(g, 32, 10);
  const ReshardPlan one = ScatterPlan(g, p0, 0, 64);
  Check(Throws([&] { (void)PackCrossOps(one, 0, 1000); }), "PackCrossOps: a piece smaller than one row is refused");
  Check(Throws([&] { (void)PackCrossOps(one, 0, 0); }) && Throws([&] { (void)SplitOp(one.ops.front(), 0); }), "PackCrossOps / SplitOp: a zero budget is refused");
  Check(Throws([&] { (void)RectOf(one.ops.front(), {0, one.ops.front().runs[0].height, 1}, Side::kFull); }), "RectOf: rows past the run are refused");
  // PlanParams::StageConvPitch
  PlanParams q = p0;
  q.stage_conv_pitch = 3;
  q.stage_b_conv_pitch = 0;
  Check(q.StageConvPitch(kStageA) == 3 && q.StageConvPitch(kStageB) == 3, "PlanParams: stage B's conv pitch defaults to stage A's");
  q.stage_b_conv_pitch = 10;
  Check(q.StageConvPitch(kStageA) == 3 && q.StageConvPitch(kStageB) == 10, "PlanParams: stage B's conv pitch can differ");
  // the ops carry the pitch: stage B's conv runs use 10 entries, stage A's 3
  bool pitches = true;
  const ReshardPlan with_b = ScatterPlan(g, q, 0, 64);
  for (const CopyOp& op : with_b.ops) {
    if (op.kind != StateKind::kGdnConv) continue;
    for (const Run2D& r : op.runs) pitches = pitches && r.full_pitch == (op.stage == kStageB ? 20u : 6u) && r.rank_pitch == 20u;
  }
  Check(pitches, "ScatterPlan: conv runs use stage A's pitch for stage A's layers and stage B's for stage B's");
}

// ---- 2. a host executor over RectOf, against the reference -------------------------------------------------------------------------
std::mt19937_64 g_rng(1008);
std::vector<uint8_t> RandBytes(size_t n) {
  std::vector<uint8_t> v(n);
  for (uint8_t& b : v) b = static_cast<uint8_t>(g_rng());
  return v;
}
std::vector<uint16_t> RandU16(size_t n) {
  std::vector<uint16_t> v(n);
  for (uint16_t& b : v) b = static_cast<uint16_t>(g_rng());
  return v;
}

constexpr uint16_t kFill = 0xBEEF;
constexpr int64_t kPitchA = 3, kPitchB = 6, kPitchRank = 5;

// One holder's per-layer buffers in DEVICE layout: KV whole blocks, the live recurrent slot, conv lines of `pitch` entries per channel.
struct Holder {
  std::map<int64_t, std::vector<uint8_t>> kv, rec, conv;
  std::vector<uint8_t> mtp;
};
std::vector<uint8_t> Pitched(const std::vector<uint16_t>& compact, int64_t channels, int64_t live, int64_t pitch) {
  std::vector<uint16_t> out(static_cast<size_t>(channels * pitch), kFill);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t e = 0; e < live; ++e) out[static_cast<size_t>(c * pitch + e)] = compact[static_cast<size_t>(c * live + e)];
  }
  std::vector<uint8_t> bytes(out.size() * 2);
  std::memcpy(bytes.data(), out.data(), bytes.size());
  return bytes;
}
std::vector<uint16_t> Unpitched(const std::vector<uint8_t>& bytes, int64_t channels, int64_t live, int64_t pitch) {
  std::vector<uint16_t> all(bytes.size() / 2);
  std::memcpy(all.data(), bytes.data(), bytes.size());
  std::vector<uint16_t> out(static_cast<size_t>(channels * live));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t e = 0; e < live; ++e) out[static_cast<size_t>(c * live + e)] = all[static_cast<size_t>(c * pitch + e)];
  }
  return out;
}
bool PadUntouched(const std::vector<uint8_t>& bytes, int64_t channels, int64_t live, int64_t pitch) {
  std::vector<uint16_t> all(bytes.size() / 2);
  std::memcpy(all.data(), bytes.data(), bytes.size());
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t e = live; e < pitch; ++e) {
      if (all[static_cast<size_t>(c * pitch + e)] != kFill) return false;
    }
  }
  return true;
}

LiveImage RandomImage(const StateGeometry& g, int64_t blocks) {
  LiveImage x;
  for (const int64_t l : g.attn_layers) x.kv[l] = RandBytes(static_cast<size_t>(blocks * g.KvBlockBytesFull()));
  x.mtp_kv = RandBytes(static_cast<size_t>(blocks * g.KvBlockBytesFull()));
  for (const int64_t l : g.gdn_layers) {
    x.gdn_rec[l] = RandBytes(static_cast<size_t>(g.RecurrentBytesFull()));
    x.gdn_conv[l] = RandU16(static_cast<size_t>(g.conv_dim_full * g.conv_live));
  }
  return x;
}

Holder StageHolder(const StateGeometry& g, const PlanParams& p, const LiveImage& img, int stage, int64_t pitch) {
  Holder h;
  for (const int64_t l : p.stage[stage].attn) h.kv[l] = img.kv.at(l);
  for (const int64_t l : p.stage[stage].gdn) {
    h.rec[l] = img.gdn_rec.at(l);
    h.conv[l] = Pitched(img.gdn_conv.at(l), g.conv_dim_full, g.conv_live, pitch);
  }
  if (stage == kStageB) h.mtp = img.mtp_kv;
  return h;
}
Holder RankHolder(const StateGeometry& g, const LiveImage& full, int rank) {
  const LiveImage r = ReshardRef(g, full, rank);
  Holder h;
  h.kv = r.kv;
  h.mtp = r.mtp_kv;
  h.rec = r.gdn_rec;
  for (const auto& [l, c] : r.gdn_conv) h.conv[l] = Pitched(c, g.ConvChannelsRank(rank), g.conv_live, kPitchRank);
  return h;
}

std::vector<uint8_t>& BufOf(Holder& h, const CopyOp& op) {
  switch (op.kind) {
    case StateKind::kKv: return h.kv.at(op.layer);
    case StateKind::kMtpKv: return h.mtp;
    case StateKind::kGdnRecurrent: return h.rec.at(op.layer);
    case StateKind::kGdnConv: return h.conv.at(op.layer);
  }
  throw std::logic_error("unreachable");
}

// The device executor's two halves over host buffers: rows of `rect.width` bytes, `rect.pitch` apart, to / from a compact byte string.
void ExportRect(Holder& h, const CopyOp& op, const OpSlice& s, Side side, uint8_t* out) {
  const Rect q = RectOf(op, s, side);
  std::vector<uint8_t>& buf = BufOf(h, op);
  if (q.End() > buf.size()) throw std::out_of_range("export reaches past its buffer");
  for (uint64_t i = 0; i < q.rows; ++i) std::memcpy(out + i * q.width, buf.data() + q.off + i * q.pitch, q.width);
}
void ImportRect(Holder& h, const CopyOp& op, const OpSlice& s, Side side, const uint8_t* in) {
  const Rect q = RectOf(op, s, side);
  std::vector<uint8_t>& buf = BufOf(h, op);
  if (q.End() > buf.size()) throw std::out_of_range("import reaches past its buffer");
  for (uint64_t i = 0; i < q.rows; ++i) std::memcpy(buf.data() + q.off + i * q.pitch, in + i * q.width, q.width);
}

struct Faults {
  bool wrong_side = false;     // export the first cross op from the other side's rectangle
  int skip_batch = -1;         // do not run this batch (source stage 0)
  bool slice_off_by_one = false;
};

// Runs a plan the way the hybrid will: cross ops through ring batches (export on the source, import on the destination), local ops in
// place. stage[] / rank[] are the holders; the source / destination of each op follows plan.dir.
void Run(const ReshardPlan& plan, Holder (&stage)[2], Holder (&rank)[2], const Faults& f = {}) {
  const bool scatter = plan.dir == Dir::kStageToRank;
  const auto src_holder = [&](const CopyOp& op) -> Holder& { return scatter ? stage[op.stage] : rank[op.rank]; };
  const auto dst_holder = [&](const CopyOp& op) -> Holder& { return scatter ? rank[op.rank] : stage[op.stage]; };
  const Side src_side = scatter ? Side::kFull : Side::kRank, dst_side = scatter ? Side::kRank : Side::kFull;
  bool first = true;
  for (int src = 0; src < 2; ++src) {
    int batch_index = 0;
    for (const RingBatch& b : PackCrossOps(plan, src, /*piece_bytes=*/4096)) {
      if (src == 0 && batch_index++ == f.skip_batch) continue;
      std::vector<uint8_t> ring(static_cast<size_t>(b.bytes));
      uint64_t at = 0;
      for (const SliceRef& r : b.slices) {
        const CopyOp& op = plan.ops[r.op];
        OpSlice sl = r.slice;
        // (the faults hit the first slice they can: a recurrent half of rank 0 has the same rectangle on both sides, a one-row run has no next row)
        if (f.slice_off_by_one && first && op.runs[sl.run].height > 1) {
          sl.row0 += 1;
          first = false;
        }
        Side s = src_side;
        if (f.wrong_side && first && op.rank == 1) {
          s = dst_side;
          first = false;
        }
        ExportRect(src_holder(op), op, sl, s, ring.data() + at);
        at += SliceBytes(op, r.slice);
      }
      at = 0;
      for (const SliceRef& r : b.slices) {
        const CopyOp& op = plan.ops[r.op];
        ImportRect(dst_holder(op), op, r.slice, dst_side, ring.data() + at);
        at += SliceBytes(op, r.slice);
      }
    }
  }
  for (const CopyOp& op : plan.ops) {
    if (!op.local) continue;
    for (const OpSlice& s : WholeOp(op)) {
      std::vector<uint8_t> tmp(static_cast<size_t>(SliceBytes(op, s)));
      ExportRect(src_holder(op), op, s, src_side, tmp.data());
      ImportRect(dst_holder(op), op, s, dst_side, tmp.data());
    }
  }
}

struct Harness {
  StateGeometry g;
  PlanParams p;
  int64_t blocks;
  LiveImage x, y;  // x: the new truth, y: the stale content of the destination
  Holder stage[2], rank[2];
  Harness(int64_t split, int64_t blocks_in, bool stage_b_pitch) : g(StateGeometry::FromRules(TinyConfig(), 4)), p(PlanParams::ForSplit(g, split, kPitchRank)), blocks(blocks_in) {
    p.stage_conv_pitch = kPitchA;
    p.stage_b_conv_pitch = stage_b_pitch ? kPitchB : 0;
    x = RandomImage(g, blocks);
    y = RandomImage(g, blocks);
  }
  int64_t StageBPitch() const { return kPitchB; }  // what the stage-B buffers really use, whatever the plan was told
  void SetupScatter() {
    for (int s = 0; s < 2; ++s) stage[s] = StageHolder(g, p, x, s, s == kStageB ? kPitchB : kPitchA);
    for (int r = 0; r < 2; ++r) rank[r] = RankHolder(g, y, r);
  }
  void SetupGather() {
    for (int s = 0; s < 2; ++s) stage[s] = StageHolder(g, p, y, s, s == kStageB ? kPitchB : kPitchA);
    for (int r = 0; r < 2; ++r) rank[r] = RankHolder(g, x, r);
  }
  bool RanksEqualX() const {
    for (int r = 0; r < 2; ++r) {
      const LiveImage xr = ReshardRef(g, x, r);
      const Holder& h = rank[r];
      if (h.kv != xr.kv || h.mtp != xr.mtp_kv || h.rec != xr.gdn_rec) return false;
      for (const int64_t l : g.gdn_layers) {
        if (Unpitched(h.conv.at(l), g.ConvChannelsRank(r), g.conv_live, kPitchRank) != xr.gdn_conv.at(l)) return false;
        if (!PadUntouched(h.conv.at(l), g.ConvChannelsRank(r), g.conv_live, kPitchRank)) return false;
      }
    }
    return true;
  }
  // gather: the stages equal x on the blocks rows [0, 14) cover (0..3) and on every GDN layer; elsewhere y
  bool StagesEqualX(const BlockRange& kvb, const BlockRange& mtpb) const {
    const size_t fb = static_cast<size_t>(g.KvBlockBytesFull());
    const auto blend = [&](std::vector<uint8_t> base, const std::vector<uint8_t>& over, const BlockRange& r) {
      for (int64_t b = r.first; b <= r.last; ++b) std::memcpy(base.data() + static_cast<size_t>(b) * fb, over.data() + static_cast<size_t>(b) * fb, fb);
      return base;
    };
    for (int s = 0; s < 2; ++s) {
      const Holder& h = stage[s];
      const int64_t pitch = s == kStageB ? kPitchB : kPitchA;
      for (const int64_t l : p.stage[s].attn) {
        if (h.kv.at(l) != blend(y.kv.at(l), x.kv.at(l), kvb)) return false;
      }
      for (const int64_t l : p.stage[s].gdn) {
        if (h.rec.at(l) != x.gdn_rec.at(l)) return false;
        if (Unpitched(h.conv.at(l), g.conv_dim_full, g.conv_live, pitch) != x.gdn_conv.at(l)) return false;
        if (!PadUntouched(h.conv.at(l), g.conv_dim_full, g.conv_live, pitch)) return false;
      }
    }
    return stage[kStageB].mtp == blend(y.mtp_kv, x.mtp_kv, mtpb);
  }
};

void ExecutorVsReference() {
  for (const int64_t split : {1, 3, 4, 5, 7}) {
    Harness h(split, 6, /*stage_b_pitch=*/true);
    h.SetupScatter();
    Run(ScatterPlan(h.g, h.p, 0, 24), h.stage, h.rank);
    Check(h.RanksEqualX(), "executor: a full scatter through ring batches of 4 KiB and RectOf rebuilds both ranks byte for byte (stage A pitch 3, stage B pitch 6, rank pitch 5)");

    // gather, as after a TP-only stretch: rows [0, 14) stale on both stages, the MTP block of row 13
    TpMasterTracker t(4, true);
    t.TpOnly(0);
    h.SetupGather();
    Run(GatherPlan(h.g, h.p, t.PlanSync(14)), h.stage, h.rank);
    Check(h.StagesEqualX(BlockRange{0, 3}, BlockRange{3, 3}), "executor: the warm gather rebuilds both stages (KV blocks 0..3, all GDN state, MTP block 3) from the two ranks");
  }
}

void NegativeControls() {
  const int64_t split = 3;
  {
    Harness h(split, 6, true);
    h.SetupScatter();
    Faults f;
    f.wrong_side = true;
    bool bad = false;
    try {
      Run(ScatterPlan(h.g, h.p, 0, 24), h.stage, h.rank, f);
      bad = !h.RanksEqualX();
    } catch (const std::out_of_range&) {
      bad = true;
    }
    Check(bad, "NEGATIVE CONTROL: exporting from the wrong side's rectangle breaks the result (or reaches past the buffer)");
  }
  {
    Harness h(split, 6, true);
    h.SetupScatter();
    Faults f;
    f.skip_batch = 0;
    Run(ScatterPlan(h.g, h.p, 0, 24), h.stage, h.rank, f);
    Check(!h.RanksEqualX(), "NEGATIVE CONTROL: a skipped ring batch leaves a rank unrebuilt");
  }
  {
    Harness h(split, 6, true);
    h.SetupScatter();
    Faults f;
    f.slice_off_by_one = true;
    bool bad = false;
    try {
      Run(ScatterPlan(h.g, h.p, 0, 24), h.stage, h.rank, f);
      bad = !h.RanksEqualX();
    } catch (const std::out_of_range&) {
      bad = true;
    }
    Check(bad, "NEGATIVE CONTROL: a slice one row off breaks the result");
  }
  {
    // stage B's buffers are 6 entries per channel but the plan was left at stage A's 3: the conv runs address the wrong entries.
    Harness h(split, 6, /*stage_b_pitch=*/false);
    h.SetupScatter();
    bool bad = false;
    try {
      Run(ScatterPlan(h.g, h.p, 0, 24), h.stage, h.rank);
      bad = !h.RanksEqualX();
    } catch (const std::out_of_range&) {
      bad = true;
    }
    Check(bad, "NEGATIVE CONTROL: a plan that assumes stage A's conv pitch for stage B's longer lines mis-addresses the conv state");
  }
}

}  // namespace

int main() {
  try {
    RealGeometry();
    ExecutorVsReference();
    NegativeControls();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_reshard_exec_cpu: uncaught exception: %s\n", e.what());
    return 1;
  }
  if (g_fails != 0) {
    std::fprintf(stderr, "test_reshard_exec_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_reshard_exec_cpu: PASS\n");
  return 0;
}
