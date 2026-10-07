// r4dx::model::pp -- the HIP-free arithmetic of the pipeline-parallel prefill track (docs/pp-prefill.md):
// which KV blocks a chunk touches (what crosses the stage boundary), the two-stage pipeline simulation with a
// bounded slot ring, the CSV the stage bench writes and the projection that turns two devices' measured stage
// times into a projected TTFT and the Phase 0 go rule. Header-only so a CPU test (tests/model/test_pp_plan_cpu.cpp)
// and the offline projection tool (tests/model/tool_pp_project.cpp) run without a device.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace r4dx::model::pp {

// ---- what crosses the boundary -----------------------------------------------------------------------------
// The KV blocks of ONE full-attention layer that a chunk of `rows` rows starting at absolute position `pos`
// writes: [block0, block1) -- whole blocks, so a chunk that starts or ends mid-block copies the neighbouring
// rows of its first and last block too (they are valid rows, and the copy puts them back unchanged). Blocks of
// one layer are contiguous in memory (slot(t) == t, paged_kv_cache.hpp), so this is one byte range.
struct KvBlockRange {
  int64_t block0 = 0, block1 = 0;
  int64_t Blocks() const { return block1 - block0; }
};
inline KvBlockRange KvBlocksTouched(int64_t pos, int64_t rows, int64_t block_size) {
  KvBlockRange r;
  r.block0 = pos / block_size;
  r.block1 = (pos + rows + block_size - 1) / block_size;
  return r;
}

// A stage split is a layer index with at least one layer on each side.
inline bool ValidSplit(int64_t split, int64_t num_layers) { return split >= 1 && split < num_layers; }

// R4DX_PP_EMULATE (ModelOptions::pp_emulate_split): unset, empty, "0", "off" or "false" = 0 (off, the default); a
// positive integer = the split layer; anything else = -1 (the caller refuses it).
inline int64_t ParseEmulateSplit(const char* e) {
  if (e == nullptr || *e == '\0') return 0;
  const std::string s = e;
  if (s == "0" || s == "off" || s == "false") return 0;
  for (char ch : s) {
    if (ch < '0' || ch > '9') return -1;
  }
  if (s.size() > 6) return -1;
  return std::stoll(s);
}

// ---- the pipeline ------------------------------------------------------------------------------------------
// Two stages, one thread each, a ring of `slots` hand-over slots (docs/pp-prefill.md 1.3): stage A runs chunk c
// (a_ms[c], its own export included), publishes it, and may start chunk c + 1 only once a slot is free -- the
// slot of chunk c + 1 - slots, which stage B frees when its chunk finishes; stage B runs chunk c (b_ms[c], its
// own import included) after A published it and after its own chunk c - 1. The call ends `tail_ms` after B's
// last chunk (thread wake-up, the GDN state join). Returns the call's wall time.
struct PipelineResult {
  double total_ms = 0.0;
  double a_busy_ms = 0.0, b_busy_ms = 0.0;
  double a_stall_ms = 0.0;  // A waiting for a free slot (back-pressure: B is the slower stage)
  double b_stall_ms = 0.0;  // B waiting for A (starvation, the fill included)
};
inline PipelineResult SimulatePipeline(const std::vector<double>& a_ms, const std::vector<double>& b_ms,
                                       int slots, double tail_ms) {
  PipelineResult out;
  const size_t n = std::min(a_ms.size(), b_ms.size());
  if (n == 0) return out;
  const size_t ring = static_cast<size_t>(std::max(1, slots));
  std::vector<double> a_done(n, 0.0), b_done(n, 0.0);
  for (size_t c = 0; c < n; ++c) {
    double start = c > 0 ? a_done[c - 1] : 0.0;
    if (c >= ring) {
      const double free_at = b_done[c - ring];
      if (free_at > start) {
        out.a_stall_ms += free_at - start;
        start = free_at;
      }
    }
    a_done[c] = start + a_ms[c];
    out.a_busy_ms += a_ms[c];
    // Stage B's chunk c depends on a_done[c] only, but it is needed to start A's chunk c + ring: B's finishing
    // times are computed here, in order, which is possible because b_done[c] needs nothing later than a_done[c].
    const double b_ready = c > 0 ? b_done[c - 1] : 0.0;
    const double b_start = std::max(a_done[c], b_ready);
    if (a_done[c] > b_ready) out.b_stall_ms += a_done[c] - b_ready;
    b_done[c] = b_start + b_ms[c];
    out.b_busy_ms += b_ms[c];
  }
  out.total_ms = b_done[n - 1] + tail_ms;
  return out;
}

// The design's closed form (docs/pp-prefill.md 6.1): [(1 + 1/N) (T_total / 2) f + N h + c] s. A cross-check of
// the simulation, not an input to it.
inline double DesignFormulaMs(double t_total_ms, int n_chunks, double f, double h_ms, double c_ms, double s) {
  return ((1.0 + 1.0 / n_chunks) * (t_total_ms / 2.0) * f + n_chunks * h_ms + c_ms) * s;
}

// ---- the stage bench's CSV ---------------------------------------------------------------------------------
// tool_pp_stage_bench writes, per measured run, one `mono,<label>,<ttft_ms>` line (the monolithic prefill) or one
// `chunk,<label>,<index>,<pos>,<rows>,<a>,<d2h>,<h2d>,<b>,<epilogue>,<mtp>,<inject>,<hop_bytes>` line per
// emulated chunk (Model::PpChunkTimes, ms); `#` lines are comments.
struct Sample {
  int64_t pos = 0, rows = 0;
  double a_ms = 0, d2h_ms = 0, h2d_ms = 0, b_ms = 0, epilogue_ms = 0, mtp_ms = 0, inject_ms = 0;
  int64_t hop_bytes = 0;
};
struct BenchFile {
  std::map<std::string, double> mono_ms;               // label -> monolithic TTFT
  std::map<std::string, std::vector<Sample>> chunks;   // label -> per-chunk samples, in chunk order
};

inline std::string FormatChunkLine(const std::string& label, size_t index, const Sample& s) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), "chunk,%s,%zu,%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%lld", label.c_str(), index,
                static_cast<long long>(s.pos), static_cast<long long>(s.rows), s.a_ms, s.d2h_ms, s.h2d_ms, s.b_ms,
                s.epilogue_ms, s.mtp_ms, s.inject_ms, static_cast<long long>(s.hop_bytes));
  return buf;
}
inline std::string FormatMonoLine(const std::string& label, double ms) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "mono,%s,%.3f", label.c_str(), ms);
  return buf;
}

inline std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> f;
  std::string cur;
  for (char ch : line) {
    if (ch == ',') {
      f.push_back(cur);
      cur.clear();
    } else if (ch != '\r') {
      cur.push_back(ch);
    }
  }
  f.push_back(cur);
  return f;
}

// Parses a stage bench file; on a malformed line returns false and sets *err. Chunks of one label must come in
// index order (0, 1, 2, ...), which the tool writes and this checks.
inline bool ParseBenchFile(const std::string& text, BenchFile* out, std::string* err) {
  std::istringstream in(text);
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> f = SplitCsv(line);
    try {
      if (f[0] == "mono" && f.size() == 3) {
        out->mono_ms[f[1]] = std::stod(f[2]);
      } else if (f[0] == "chunk" && f.size() == 13) {
        std::vector<Sample>& v = out->chunks[f[1]];
        if (static_cast<size_t>(std::stoll(f[2])) != v.size()) {
          if (err) *err = "line " + std::to_string(line_no) + ": chunk index out of order for " + f[1];
          return false;
        }
        Sample s;
        s.pos = std::stoll(f[3]);
        s.rows = std::stoll(f[4]);
        s.a_ms = std::stod(f[5]);
        s.d2h_ms = std::stod(f[6]);
        s.h2d_ms = std::stod(f[7]);
        s.b_ms = std::stod(f[8]);
        s.epilogue_ms = std::stod(f[9]);
        s.mtp_ms = std::stod(f[10]);
        s.inject_ms = std::stod(f[11]);
        s.hop_bytes = std::stoll(f[12]);
        v.push_back(s);
      } else {
        if (err) *err = "line " + std::to_string(line_no) + ": not a mono / chunk record";
        return false;
      }
    } catch (const std::exception&) {
      if (err) *err = "line " + std::to_string(line_no) + ": bad number";
      return false;
    }
  }
  return true;
}

// ---- the hop bench's CSV -----------------------------------------------------------------------------------
// tool_pp_hop_bench: `hop,<dir>,<size label>,<bytes>,<leg>,<median_ms>,<GB/s>` -- dir is d<src>to<dst> (visible
// ordinals), leg d2h / h2d / e2e / pipe_d2h / pipe_h2d (see that tool).
struct HopRow {
  std::string dir, size_label, leg;
  int64_t bytes = 0;
  double ms = 0.0, gbps = 0.0;
};
inline bool ParseHopFile(const std::string& text, std::vector<HopRow>* out, std::string* err) {
  std::istringstream in(text);
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> f = SplitCsv(line);
    if (f.size() != 7 || f[0] != "hop") {
      if (err) *err = "line " + std::to_string(line_no) + ": not a hop record";
      return false;
    }
    try {
      HopRow r;
      r.dir = f[1];
      r.size_label = f[2];
      r.bytes = std::stoll(f[3]);
      r.leg = f[4];
      r.ms = std::stod(f[5]);
      r.gbps = std::stod(f[6]);
      out->push_back(r);
    } catch (const std::exception&) {
      if (err) *err = "line " + std::to_string(line_no) + ": bad number";
      return false;
    }
  }
  return true;
}
// The slowest bandwidth among the rows of `dir` whose leg is one of `legs` (and, when `size_label` is not empty,
// of that size); 0 when there is none.
inline double MinGbps(const std::vector<HopRow>& rows, const std::string& dir, const std::vector<std::string>& legs,
                      const std::string& size_label = "") {
  double best = 0.0;
  bool any = false;
  for (const HopRow& r : rows) {
    if (r.dir != dir || (!size_label.empty() && r.size_label != size_label)) continue;
    if (std::find(legs.begin(), legs.end(), r.leg) == legs.end()) continue;
    if (!any || r.gbps < best) best = r.gbps;
    any = true;
  }
  return any ? best : 0.0;
}

// ---- the projection ----------------------------------------------------------------------------------------
// Stage A runs on device 0 and stage B on device 1, so A's per-chunk cost is the A-side bench's (prologue +
// layers [0, k), plus the export half of the hop) and B's is the B-side bench's (the import half of the hop,
// layers [k, N), the epilogue: MTP priming, lm_head, DFlash injection). `d2h_scale` / `h2d_scale` stretch the two
// hop halves: the emulation's own host copy against the measured cross-card hop bandwidth under load (>= 1: a
// slower measured hop costs more, a faster one is never credited).
struct ProjectionParams {
  int slots = 3;
  double fixed_ms = 2.0;   // after B's last chunk: thread wake-up, the GDN state join
  double d2h_scale = 1.0;
  double h2d_scale = 1.0;
};
struct Projection {
  bool ok = false;
  std::string error;
  int chunks = 0;
  double mono_ms = 0.0;
  double pp_ms = 0.0;
  double speedup = 0.0;
  double a_cost_ms = 0.0, b_cost_ms = 0.0;  // sum over the chunks (stage busy time)
  double a_stall_ms = 0.0, b_stall_ms = 0.0;
  double a_over_b = 0.0;                    // stage balance (A total / B total)
};
inline Projection Project(const std::vector<Sample>& a_side, const std::vector<Sample>& b_side, double mono_ms,
                          const ProjectionParams& p) {
  Projection r;
  r.mono_ms = mono_ms;
  if (a_side.empty() || a_side.size() != b_side.size()) {
    r.error = "the two sides measured a different number of chunks (" + std::to_string(a_side.size()) + " vs " +
              std::to_string(b_side.size()) + ")";
    return r;
  }
  if (!(mono_ms > 0.0)) {
    r.error = "no monolithic TTFT";
    return r;
  }
  std::vector<double> a(a_side.size()), b(b_side.size());
  for (size_t c = 0; c < a.size(); ++c) {
    if (a_side[c].rows != b_side[c].rows || a_side[c].pos != b_side[c].pos) {
      r.error = "chunk " + std::to_string(c) + " covers different rows on the two sides";
      return r;
    }
    a[c] = a_side[c].a_ms + a_side[c].d2h_ms * p.d2h_scale;
    b[c] = b_side[c].h2d_ms * p.h2d_scale + b_side[c].b_ms + b_side[c].epilogue_ms;
  }
  const PipelineResult sim = SimulatePipeline(a, b, p.slots, p.fixed_ms);
  r.ok = true;
  r.chunks = static_cast<int>(a.size());
  r.pp_ms = sim.total_ms;
  r.speedup = mono_ms / sim.total_ms;
  r.a_cost_ms = sim.a_busy_ms;
  r.b_cost_ms = sim.b_busy_ms;
  r.a_stall_ms = sim.a_stall_ms;
  r.b_stall_ms = sim.b_stall_ms;
  r.a_over_b = sim.b_busy_ms > 0 ? sim.a_busy_ms / sim.b_busy_ms : 0.0;
  return r;
}

// The Phase 0 go rule (docs/pp-prefill.md 8): the projected TTFT speedup must reach `need` at BOTH 8k and 32k.
// 64k is reported (it is part of the table) but does not decide.
inline bool GoRule(double speedup_8k, double speedup_32k, double need = 1.6) {
  return speedup_8k >= need && speedup_32k >= need;
}

}  // namespace r4dx::model::pp
