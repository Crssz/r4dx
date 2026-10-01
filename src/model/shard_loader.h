// r4dx::model::container_util::ShardLoader -- the tensor-parallel (docs/tp.md 4.3, 5.1) per-rank tensor loader:
// every tensor is classified by tp::RuleFor on its base name and only this rank's byte runs are uploaded.
//
// MECHANICAL EXTRACTION (docs/gemma4-plan.md M1b-1): moved verbatim out of container.cpp's anonymous namespace so
// GemmaContainer::Load can shard through the SAME code Container::LoadShard runs (bodies unchanged; only the home
// and linkage -- inline, in container_util -- differ). RuleFor's `global` is the Qwen ModelConfig or, for Gemma,
// GemmaConfig::ToModelConfig() (arch kGemma4 selects the Gemma rule table).
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "container_load_util.h"
#include "tp/tp_shard.h"
#include "trellis_meta.h"

namespace r4dx::model::container_util {

struct ShardLoadStats {
  int sharded = 0;     // on-disk tensors this rank uploaded a slice of
  int replicated = 0;  // on-disk tensors this rank uploaded whole
  uint64_t uploaded_bytes = 0;
  uint64_t staged_bytes = 0;  // the part of uploaded_bytes gathered through the host staging buffer
};

// The byte size a WHOLE on-disk part of logical shape [N, K] has (docs/tp.md 4.3, the converter's
// packers). A tensor whose span disagrees is refused rather than sliced with the wrong strides.
inline uint64_t PartBytes(const tp::PartShape& s) {
  const uint64_t N = static_cast<uint64_t>(s.N), K = static_cast<uint64_t>(s.K);
  switch (s.part) {
    case tp::Part::kBf16: return N * K * 2;
    case tp::Part::kW4Wq: return N * K / 2;
    case tp::Part::kW4a16Wsz: return N * (K / static_cast<uint64_t>(s.group)) * 4;
    case tp::Part::kElem: return N * static_cast<uint64_t>(s.row_bytes);
    case tp::Part::kTrellisW: return N * K * static_cast<uint64_t>(s.rate) / 8;
  }
  return 0;
}

// One rank's view of the container: every tensor is classified by tp::RuleFor on its base name and
// only this rank's byte runs (tp::PlanRows/PlanCols) are uploaded -- straight from the mmap when the
// plan is one contiguous range, through one reusable host staging buffer otherwise (docs/tp.md 5.1
// step 4). Every name must be known to RuleFor: an unclassified tensor throws, it is never silently
// replicated.
class ShardLoader {
 public:
  // `meta.w4a16`: the container's w4a16 groups (quant2 Q3) -- each w4a16 linear's scale name and
  // the group its wsz slice is cut at (tp::Part::kW4a16Wsz strides by the group), per tensor.
  // `meta.trellis`: each trellis linear's rate and parts (docs/trellis-kernel.md 2.3, 5.1).
  ShardLoader(const SafetensorsReader& r, const ModelConfig& global, int world, int rank,
              const LinearLoadMeta& meta)
      : r_(r), global_(global), world_(world), rank_(rank), meta_(meta), w4a16_(meta.w4a16) {}

  // A tensor with one on-disk form and no layout suffix (norms, gdn.in_proj_a/b, conv1d_weight,
  // A_log, dt_bias, the descales, mtp.fc, ...): its whole bytes when the rule replicates, else this
  // rank's row ranges of the row-major [N, ...] array (N = the rule's total rows).
  template <class T>
  core::DeviceBuffer<T> Raw(const std::string& name) {
    const tp::ShardRule rule = tp::RuleFor(name, global_);
    const uint64_t span = Span(name);
    if (rule.split == tp::Split::kReplicate) return Upload<T>(name, {tp::ByteRun{0, span}}, false);
    if (rule.split != tp::Split::kRows) {
      throw std::logic_error("r4dx::model::Container: '" + name +
                             "' has no layout suffix but its tensor-parallel rule is not a row split "
                             "or a replication");
    }
    int64_t rows = 0;
    for (const tp::Segment& s : rule.segments) rows += s.rows;
    if (rows <= 0 || span % static_cast<uint64_t>(rows) != 0) {
      throw std::runtime_error("r4dx::model::Container: tensor '" + name + "' (" +
                               std::to_string(span) + " bytes) is not " + std::to_string(rows) +
                               " equal rows, the row count its tensor-parallel rule splits");
    }
    tp::PartShape shape;
    shape.part = tp::Part::kElem;
    shape.N = rows;
    shape.row_bytes = static_cast<int64_t>(span / static_cast<uint64_t>(rows));
    return Upload<T>(name, tp::PlanRows(shape, tp::RankRows(rule, world_, rank_)), true);
  }

  // UploadWidenedF32's counterpart (gdn.norm_weight, bf16 on disk, fp32 on device): replicated.
  core::DeviceBuffer<float> WidenedF32(const std::string& name) {
    if (tp::RuleFor(name, global_).split != tp::Split::kReplicate) {
      throw std::logic_error("r4dx::model::Container: '" + name + "' is widened on load, which "
                             "only a replicated tensor supports");
    }
    core::DeviceBuffer<float> buf = UploadWidenedF32(r_, name);
    ++stats_.replicated;
    stats_.uploaded_bytes += buf.bytes();
    return buf;
  }

  // LoadQuantLinearWithFallback's sharded counterpart: the same requested -> .bf16.w -> bare
  // on-disk form chain, then this rank's slice of every part of that form. `N`/`K` are the GLOBAL
  // logical shape; the returned QuantLinear carries the RANK's (docs/tp.md 5.1 step 4).
  QuantLinear Linear(const std::string& base, Layout requested, int64_t N, int64_t K,
                     int* fallbacks = nullptr) {
    Layout form = requested;
    bool bare = false;
    if (!HasLayout(r_, meta_, base, requested)) {
      // quant2 Q3: a mapped w4a16 linear never falls back (LoadQuantLinearWithFallback).
      if (requested == Layout::kW4a16 && w4a16_.groups.Mapped(base)) {
        throw std::runtime_error("r4dx::model::Container: '" + base + "' is listed in " +
                                 "__metadata__.quant.w4a16.groups at group " +
                                 std::to_string(w4a16_.groups.GroupFor(base)) + " but '" +
                                 w4a16_.groups.WszName(base) + "' is missing; not falling back");
      }
      if (fallbacks != nullptr) ++*fallbacks;
      form = Layout::kBf16;
      if (!HasLayout(r_, meta_, base, Layout::kBf16)) {
        if (!r_.Has(base)) {
          throw std::runtime_error("r4dx::model::Container: no tensor found for '" + base +
                                   "' in any known on-disk form (requested layout, bf16, or bare)");
        }
        bare = true;
      }
    }

    const tp::ShardRule rule = tp::RuleFor(base, global_);
    QuantLinear q;
    q.layout = form;
    q.N = N;
    q.K = K;
    std::vector<tp::Range> rows;
    tp::Range cols{0, K};
    if (rule.split == tp::Split::kRows) {
      int64_t total = 0;
      for (const tp::Segment& s : rule.segments) total += s.rows;
      if (total != N) {
        throw std::logic_error("r4dx::model::Container: '" + base + "' has N = " +
                               std::to_string(N) + " but its tensor-parallel rule splits " +
                               std::to_string(total) + " rows");
      }
      rows = tp::RankRows(rule, world_, rank_);
      q.N = 0;
      for (const tp::Range& r : rows) q.N += r.count;
    } else if (rule.split == tp::Split::kCols) {
      if (rule.k_total != K) {
        throw std::logic_error("r4dx::model::Container: '" + base + "' has K = " +
                               std::to_string(K) + " but its tensor-parallel rule splits K = " +
                               std::to_string(rule.k_total));
      }
      cols = tp::RankCols(rule, world_, rank_);
      q.K = cols.count;
    } else if (rule.split != tp::Split::kReplicate) {
      throw std::logic_error("r4dx::model::Container: '" + base +
                             "' is not a text/mtp linear (rank-0-only rule)");
    }

    const auto shape = [&](tp::Part p, int group) {
      tp::PartShape s;
      s.part = p;
      s.N = N;
      s.K = K;
      s.group = group;
      return s;
    };
    switch (form) {
      case Layout::kBf16:
        q.bf16_w = Part<uint16_t>(bare ? base : base + ".bf16.w", shape(tp::Part::kBf16, 0), rule,
                                  rows, cols);
        break;
      case Layout::kW4a16:
        // quant2 Q3: the linear's own group sets both the scale tensor's name and the dword
        // stride its K slice is cut at; a rank's K range is whole 64-K blocks (wq's rule), so it
        // is whole groups at 32 and 64 alike.
        q.w4a16_group = w4a16_.Resolve(base);
        q.wq = Part<uint8_t>(base + ".w4a16.wq", shape(tp::Part::kW4Wq, 0), rule, rows, cols);
        q.w4a16_wsz = Part<uint32_t>(w4a16_.groups.WszName(base),
                                     shape(tp::Part::kW4a16Wsz, w4a16_.KernelGroup(base)), rule,
                                     rows, cols);
        break;
      case Layout::kTrellis:
        TrellisSlice(q, base, rule, rows, cols, N, K);
        break;
    }
    return q;
  }

  // The embedding's device mirror is uploaded by LoadShard itself; this only counts it.
  void CountReplicated(uint64_t bytes) {
    ++stats_.replicated;
    stats_.uploaded_bytes += bytes;
  }
  const ShardLoadStats& Stats() const { return stats_; }

 private:
  uint64_t Span(const std::string& name) const {
    const auto& m = r_.Meta(name);
    return m.end - m.begin;
  }

  // docs/trellis-kernel.md 2.4, 5.1, 5.5: this rank's slice of trellis linear `base` (GLOBAL
  // [N, K]; q.N / q.K are already the rank's). The words go through Part() with kTrellisW; suh and
  // svh have explicit plans, because Part() picks the plan by the linear's rule, which is right for
  // neither:
  //   column-parallel (kRows): suh replicated (every part, full K), svh cut by the rank's rows;
  //   row-parallel (kCols): suh cut by the rank's K range (one part only), svh replicated.
  // Both are gathered as fp16 and widened on the host. Every rank range must be whole 128-blocks
  // (both Hadamards work in them), and each part's rank-local width is the rank's rows inside it.
  void TrellisSlice(QuantLinear& q, const std::string& base, const tp::ShardRule& rule,
                    const std::vector<tp::Range>& rows, tp::Range cols, int64_t N, int64_t K) {
    const TrellisLinearSpec& t = TrellisFor(meta_, base);
    CheckTrellisShape(r_, base, t, N, K, meta_.path);
    const auto fail = [&](const std::string& why) {
      throw std::runtime_error("r4dx::model::Container: trellis linear '" + base + "' at TP=" +
                               std::to_string(world_) + " rank " + std::to_string(rank_) + ": " +
                               why + " (docs/trellis-kernel.md 2.4)");
    };
    for (const tp::Range& r : rows) {
      if (r.begin % kTrellisBlock != 0 || r.count % kTrellisBlock != 0) {
        fail("the rank's rows [" + std::to_string(r.begin) + ", " +
             std::to_string(r.begin + r.count) + ") are not whole 128-blocks");
      }
    }
    if (cols.begin % kTrellisBlock != 0 || cols.count % kTrellisBlock != 0) {
      fail("the rank's K range [" + std::to_string(cols.begin) + ", " +
           std::to_string(cols.begin + cols.count) + ") is not whole 128-blocks");
    }
    SetTrellisFields(q, t);
    tp::PartShape wshape;
    wshape.part = tp::Part::kTrellisW;
    wshape.N = N;
    wshape.K = K;
    wshape.rate = t.bits;
    q.trellis_w = Part<uint32_t>(base + ".trellis.w", wshape, rule, rows, cols);

    const std::string suh = base + ".trellis.suh", svh = base + ".trellis.svh";
    const auto elem = [](int64_t n) {
      tp::PartShape s;
      s.part = tp::Part::kElem;
      s.N = n;
      s.row_bytes = 2;  // fp16
      return s;
    };
    if (rule.split == tp::Split::kCols) {
      if (t.Parts() != 1) fail("a row-parallel linear with two input transforms");
      q.trellis_suh = WidenedF16(suh, tp::PlanRows(elem(K), {cols}), true);
    } else {
      q.trellis_suh = WidenedF16(suh, {tp::ByteRun{0, Span(suh)}}, false);
    }
    if (rule.split == tp::Split::kRows) {
      q.trellis_svh = WidenedF16(svh, tp::PlanRows(elem(N), rows), true);
    } else {
      q.trellis_svh = WidenedF16(svh, {tp::ByteRun{0, Span(svh)}}, false);
    }

    // Rank-local parts: the rank's rows inside each part's global range, in concatenation order
    // (every rank range lies inside one part, and the part index never decreases along them).
    const int parts = t.Parts();
    const int64_t global_n[2] = {parts > 1 ? t.parts[0] : N, parts > 1 ? t.parts[1] : 0};
    if (rule.split != tp::Split::kRows) {
      q.trellis_part_n[0] = global_n[0];
      q.trellis_part_n[1] = global_n[1];
      return;
    }
    int64_t local[2] = {0, 0};
    int last_part = 0;
    for (const tp::Range& r : rows) {
      const int p = (parts > 1 && r.begin >= global_n[0]) ? 1 : 0;
      const int64_t part_end = p == 0 ? global_n[0] : global_n[0] + global_n[1];
      if (r.begin + r.count > part_end || p < last_part) {
        fail("the rank's rows [" + std::to_string(r.begin) + ", " +
             std::to_string(r.begin + r.count) + ") cross a part boundary");
      }
      local[p] += r.count;
      last_part = p;
    }
    if (parts > 1 && (local[0] == 0 || local[1] == 0)) fail("a part has no rows on this rank");
    q.trellis_part_n[0] = local[0];
    q.trellis_part_n[1] = local[1];
  }

  // The fp16 bytes `runs` of `name`, gathered and widened to fp32 (trellis suh / svh).
  core::DeviceBuffer<float> WidenedF16(const std::string& name,
                                       const std::vector<tp::ByteRun>& runs, bool sharded) {
    const std::vector<uint8_t> bytes = tp::Gather(r_.Data(name), Span(name), runs);
    if (bytes.empty() || bytes.size() % 2 != 0) {
      throw std::runtime_error("r4dx::model::Container: this rank's slice of '" + name + "' is " +
                               std::to_string(bytes.size()) + " bytes, not whole fp16 values");
    }
    core::DeviceBuffer<float> buf = UploadWidenedF16Bytes(bytes.data(), bytes.size());
    ++(sharded ? stats_.sharded : stats_.replicated);
    stats_.uploaded_bytes += buf.bytes();  // the fp32 device bytes, as WidenedF32 counts them
    if (runs.size() > 1) stats_.staged_bytes += bytes.size();
    return buf;
  }

  template <class T>
  core::DeviceBuffer<T> Part(const std::string& name, const tp::PartShape& shape,
                             const tp::ShardRule& rule, const std::vector<tp::Range>& rows,
                             tp::Range cols) {
    const uint64_t span = Span(name);
    if (span != PartBytes(shape)) {
      throw std::runtime_error("r4dx::model::Container: tensor '" + name + "' is " +
                               std::to_string(span) + " bytes, but its [" +
                               std::to_string(shape.N) + ", " + std::to_string(shape.K) +
                               "] layout needs " + std::to_string(PartBytes(shape)));
    }
    switch (rule.split) {
      case tp::Split::kRows: return Upload<T>(name, tp::PlanRows(shape, rows), true);
      case tp::Split::kCols: return Upload<T>(name, tp::PlanCols(shape, cols), true);
      default: return Upload<T>(name, {tp::ByteRun{0, span}}, false);
    }
  }

  template <class T>
  core::DeviceBuffer<T> Upload(const std::string& name, const std::vector<tp::ByteRun>& runs,
                               bool sharded) {
    const uint64_t span = Span(name);
    const uint8_t* src = r_.Data(name);
    size_t total = 0;
    for (const tp::ByteRun& run : runs) {
      if (run.src_off > span || run.bytes > span - run.src_off) {
        throw std::out_of_range("r4dx::model::Container: a byte run of '" + name +
                                "' reaches past its " + std::to_string(span) + " bytes");
      }
      total += run.bytes;
    }
    if (total == 0 || total % sizeof(T) != 0) {
      throw std::runtime_error("r4dx::model::Container: this rank's slice of '" + name + "' is " +
                               std::to_string(total) + " bytes, not a positive multiple of " +
                               std::to_string(sizeof(T)));
    }
    const size_t count = total / sizeof(T);
    core::DeviceBuffer<T> buf(count);
    if (runs.size() == 1) {
      // One contiguous range: straight from the mmap, no staging copy.
      buf.CopyFromHost(reinterpret_cast<const T*>(src + runs[0].src_off), count);
    } else {
      if (staging_.size() < total) staging_.resize(total);
      size_t o = 0;
      for (const tp::ByteRun& run : runs) {
        std::memcpy(staging_.data() + o, src + run.src_off, run.bytes);
        o += run.bytes;
      }
      buf.CopyFromHost(reinterpret_cast<const T*>(staging_.data()), count);
      stats_.staged_bytes += total;
    }
    ++(sharded ? stats_.sharded : stats_.replicated);
    stats_.uploaded_bytes += total;
    return buf;
  }

  const SafetensorsReader& r_;
  const ModelConfig& global_;
  int world_, rank_;
  const LinearLoadMeta& meta_;
  const W4a16LoadGroups& w4a16_;
  std::vector<uint8_t> staging_;  // grown to the largest gathered tensor; freed with the loader
  ShardLoadStats stats_;
};

}  // namespace r4dx::model::container_util

