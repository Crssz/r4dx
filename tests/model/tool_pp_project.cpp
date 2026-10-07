// tests/model/tool_pp_project.cpp -- pipeline-parallel prefill, Phase 0 (docs/pp-prefill.md section 8): turns the
// measurements of tool_pp_stage_bench (one CSV per card) and tool_pp_hop_bench into the projected TTFT of the
// two-GPU pipeline at 8k / 32k / 64k and the design's go rule, and writes the summary the Phase 0 script
// (pp_phase0.ps1) leaves in <R4DX_MODELS_ROOT>\r4dx\pp\phase0\summary.txt. CPU only: no HIP call, no device.
//
// The projection (pp_plan.h): stage A's per-chunk cost is device 0's prologue + layers [0, k) + the export half
// of the hop, stage B's is device 1's import half of the hop + layers [k, N) + the epilogue (MTP priming, lm_head,
// DFlash injection); the pipeline simulation runs the real chunk grid through two stages and a 3-slot ring. The
// baseline is the monolithic TTFT measured on device 1 (decode's card). Two projections are made when the inputs
// exist: "alone" (each card measured by itself) and "both busy" (both cards measured at the same time, which
// carries the dual-GPU clock / power sag) -- the go rule uses the both-busy one when present, since that is
// what the pipeline will run at. The hop's two halves are stretched by the ratio of the emulation's own host copy
// to the measured cross-card bandwidth under load, when that is slower (never credited when faster).
//
// GO RULE (docs/pp-prefill.md 8, the Phase 0 table): projected speedup >= 1.6x at BOTH 8k and 32k; otherwise STOP.
// The other Phase 0 gates are reported next to it (stage balance, device skew, hop bandwidth, DFlash + MTP cost
// per chunk) but only the go rule decides. Exit code: 0 = GO, 3 = STOP, 2 = unusable input.
//
// Usage:
//   tool_pp_project --stage-a stage_dev0.csv --stage-b stage_dev1.csv [--conc-a conc_dev0.csv --conc-b conc_dev1.csv]
//       [--dflash-a ... --dflash-b ...] [--mtp-b ...] [--hop hop_alone.csv] [--hop-conc hop_conc.csv]
//       [--split 33] [--dflash-split 35] [--slots 3] [--fixed-ms 2.0] [--need 1.6] [--notes notes.txt] --out summary.txt
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "pp_plan.h"

namespace pp = r4dx::model::pp;

namespace {

struct Args {
  std::string stage_a, stage_b, conc_a, conc_b, dflash_a, dflash_b, mtp_b, hop, hop_conc, notes, out;
  int64_t split = 33;
  int64_t dflash_split = 35;
  int slots = 3;
  double fixed_ms = 2.0;
  double need = 1.6;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "tool_pp_project: %s\nusage: tool_pp_project --stage-a F --stage-b F [--conc-a F --conc-b F] [--dflash-a F "
               "--dflash-b F] [--mtp-b F] [--hop F] [--hop-conc F] [--split 33] [--dflash-split 35] [--slots 3] "
               "[--fixed-ms 2] [--need 1.6] [--notes F] --out summary.txt\n", why.c_str());
  std::exit(2);
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(s + " needs a value");
      return argv[++i];
    };
    if (s == "--stage-a") a.stage_a = next();
    else if (s == "--stage-b") a.stage_b = next();
    else if (s == "--conc-a") a.conc_a = next();
    else if (s == "--conc-b") a.conc_b = next();
    else if (s == "--dflash-a") a.dflash_a = next();
    else if (s == "--dflash-b") a.dflash_b = next();
    else if (s == "--mtp-b") a.mtp_b = next();
    else if (s == "--hop") a.hop = next();
    else if (s == "--hop-conc") a.hop_conc = next();
    else if (s == "--notes") a.notes = next();
    else if (s == "--out") a.out = next();
    else if (s == "--split") a.split = std::stoll(next());
    else if (s == "--dflash-split") a.dflash_split = std::stoll(next());
    else if (s == "--slots") a.slots = std::stoi(next());
    else if (s == "--fixed-ms") a.fixed_ms = std::stod(next());
    else if (s == "--need") a.need = std::stod(next());
    else Usage("unknown argument " + s);
  }
  if (a.stage_a.empty() || a.stage_b.empty() || a.out.empty()) Usage("--stage-a, --stage-b and --out are required");
  return a;
}

std::string ReadAll(const std::string& path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *ok = false;
    return {};
  }
  *ok = true;
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool LoadBench(const std::string& path, pp::BenchFile* out) {
  if (path.empty()) return false;
  bool ok = false;
  const std::string text = ReadAll(path, &ok);
  if (!ok) {
    std::fprintf(stderr, "tool_pp_project: cannot read %s\n", path.c_str());
    std::exit(2);
  }
  std::string err;
  if (!pp::ParseBenchFile(text, out, &err)) {
    std::fprintf(stderr, "tool_pp_project: %s: %s\n", path.c_str(), err.c_str());
    std::exit(2);
  }
  return true;
}
bool LoadHop(const std::string& path, std::vector<pp::HopRow>* out) {
  if (path.empty()) return false;
  bool ok = false;
  const std::string text = ReadAll(path, &ok);
  if (!ok) {
    std::fprintf(stderr, "tool_pp_project: cannot read %s\n", path.c_str());
    std::exit(2);
  }
  std::string err;
  if (!pp::ParseHopFile(text, out, &err)) {
    std::fprintf(stderr, "tool_pp_project: %s: %s\n", path.c_str(), err.c_str());
    std::exit(2);
  }
  return true;
}

double Sum(const std::vector<pp::Sample>& v, double pp::Sample::*f) {
  double s = 0;
  for (const pp::Sample& x : v) s += x.*f;
  return s;
}

std::string Fmt(const char* fmt, double a, double b = 0, double c = 0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), fmt, a, b, c);
  return buf;
}

// The hop scales from a loaded hop file: the emulation moved `hop_bytes` through its own host copy in d2h_ms /
// h2d_ms (GB/s = that), the cross-card bench says the 12.75 MiB hop sustains `gbps`; scale = max(1, emu / bench).
struct HopScale {
  double d2h = 1.0, h2d = 1.0;
  double bench_d2h = 0.0, bench_h2d = 0.0, emu_d2h = 0.0, emu_h2d = 0.0;
};
HopScale ScalesFrom(const std::vector<pp::HopRow>& rows, const std::vector<pp::Sample>& a_side,
                    const std::vector<pp::Sample>& b_side) {
  HopScale s;
  if (rows.empty() || a_side.empty() || b_side.empty()) return s;
  // The pipe_ legs are the steady state (export of chunk c + 1 while the import of chunk c runs); the plain legs
  // are the cross-check. The slower of the two is the bandwidth the pipeline gets.
  s.bench_d2h = pp::MinGbps(rows, "d0to1", {"d2h", "pipe_d2h"}, "12.75MiB");
  s.bench_h2d = pp::MinGbps(rows, "d0to1", {"h2d", "pipe_h2d"}, "12.75MiB");
  double bytes = 0, d2h_ms = 0;
  for (const pp::Sample& x : a_side) {
    bytes += static_cast<double>(x.hop_bytes);
    d2h_ms += x.d2h_ms;
  }
  double bytes_b = 0, h2d_ms = 0;
  for (const pp::Sample& x : b_side) {
    bytes_b += static_cast<double>(x.hop_bytes);
    h2d_ms += x.h2d_ms;
  }
  if (d2h_ms > 0) s.emu_d2h = bytes / (d2h_ms * 1e-3) / 1e9;
  if (h2d_ms > 0) s.emu_h2d = bytes_b / (h2d_ms * 1e-3) / 1e9;
  if (s.bench_d2h > 0 && s.emu_d2h > 0) s.d2h = std::max(1.0, s.emu_d2h / s.bench_d2h);
  if (s.bench_h2d > 0 && s.emu_h2d > 0) s.h2d = std::max(1.0, s.emu_h2d / s.bench_h2d);
  return s;
}

struct SizeResult {
  std::string tag;
  bool have_alone = false, have_conc = false;
  pp::Projection alone, conc;
  double mono_dev1 = 0, mono_dev0 = 0;
  int64_t tokens = 0;
};

}  // namespace

int main(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  pp::BenchFile a, b, ca, cb, da, db, mb;
  LoadBench(args.stage_a, &a);
  LoadBench(args.stage_b, &b);
  const bool have_conc = LoadBench(args.conc_a, &ca) && LoadBench(args.conc_b, &cb);
  const bool have_dflash = LoadBench(args.dflash_a, &da) && LoadBench(args.dflash_b, &db);
  const bool have_mtp = LoadBench(args.mtp_b, &mb);
  std::vector<pp::HopRow> hop_alone, hop_conc;
  const bool have_hop = LoadHop(args.hop, &hop_alone);
  const bool have_hop_conc = LoadHop(args.hop_conc, &hop_conc);

  pp::ProjectionParams params;
  params.slots = args.slots;
  params.fixed_ms = args.fixed_ms;
  const std::string emu = "emu" + std::to_string(args.split);

  std::vector<SizeResult> results;
  for (const char* tag : {"8k", "32k", "64k"}) {
    const std::string elabel = emu + "-" + tag, mlabel = std::string("mono-") + tag;
    SizeResult r;
    r.tag = tag;
    const auto ea = a.chunks.find(elabel), eb = b.chunks.find(elabel);
    if (ea == a.chunks.end() || eb == b.chunks.end() || b.mono_ms.count(mlabel) == 0) continue;
    r.mono_dev1 = b.mono_ms[mlabel];
    r.mono_dev0 = a.mono_ms.count(mlabel) != 0 ? a.mono_ms[mlabel] : 0.0;
    for (const pp::Sample& s : ea->second) r.tokens += s.rows;
    // The hop scales: the alone projection uses the alone hop file, the both-busy one the concurrent hop file.
    pp::ProjectionParams p_alone = params, p_conc = params;
    if (have_hop) {
      const HopScale s = ScalesFrom(hop_alone, ea->second, eb->second);
      p_alone.d2h_scale = s.d2h;
      p_alone.h2d_scale = s.h2d;
    }
    r.alone = pp::Project(ea->second, eb->second, r.mono_dev1, p_alone);
    r.have_alone = true;
    if (have_conc) {
      const auto xa = ca.chunks.find(elabel), xb = cb.chunks.find(elabel);
      if (xa != ca.chunks.end() && xb != cb.chunks.end()) {
        if (have_hop_conc || have_hop) {
          const HopScale s = ScalesFrom(have_hop_conc ? hop_conc : hop_alone, xa->second, xb->second);
          p_conc.d2h_scale = s.d2h;
          p_conc.h2d_scale = s.h2d;
        }
        // The baseline stays the card's own (alone) monolithic TTFT: what a user gets today.
        r.conc = pp::Project(xa->second, xb->second, r.mono_dev1, p_conc);
        r.have_conc = true;
      }
    }
    results.push_back(r);
  }

  std::ostringstream out;
  const std::time_t now = std::time(nullptr);
  char stamp[64] = "";
  std::tm tmv{};
#ifdef _WIN32
  localtime_s(&tmv, &now);
#else
  localtime_r(&now, &tmv);
#endif
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tmv);
  out << "PP phase 0 summary (tool_pp_project, " << stamp << ")\n";
  out << "design: docs/pp-prefill.md section 8; split k = " << args.split << ", " << args.slots << " hand-over slots, fixed tail "
      << args.fixed_ms << " ms\n";
  out << "stage A = layers [0, k) on device 0 (the desktop card); stage B = [k, N), final norm, lm_head on device 1 "
         "(headless, decode's card)\n";
  if (!args.notes.empty()) {
    bool ok = false;
    const std::string notes = ReadAll(args.notes, &ok);
    if (ok) out << "\n" << notes << (notes.empty() || notes.back() == '\n' ? "" : "\n");
  }

  out << "\n---- projected cold TTFT, plain prefill (int8 + split-KV defaults), ms ----\n";
  out << "size  tokens  chunks  mono dev1   mono dev0   stage A sum  stage B sum  A/B    PP alone (x)       PP both-busy (x)\n";
  bool have_8k = false, have_32k = false;
  double sp8 = 0, sp32 = 0, sp64 = 0;
  bool have_basis_conc = true;  // judged on the sizes the rule uses: both-busy numbers for 8k and 32k, or none
  for (const SizeResult& r : results) {
    const pp::Projection& basis = r.have_conc ? r.conc : r.alone;
    if (!r.have_conc && (r.tag == "8k" || r.tag == "32k")) have_basis_conc = false;
    char line[512];
    std::snprintf(line, sizeof(line), "%-5s %-7lld %-7d %-11.1f %-11.1f %-12.1f %-12.1f %-6.3f %-8.1f (%.3fx)   ", r.tag.c_str(),
                  static_cast<long long>(r.tokens), r.alone.chunks, r.mono_dev1, r.mono_dev0, r.alone.a_cost_ms,
                  r.alone.b_cost_ms, r.alone.a_over_b, r.alone.pp_ms, r.alone.speedup);
    out << line;
    if (r.have_conc) {
      char l2[128];
      std::snprintf(l2, sizeof(l2), "%-8.1f (%.3fx)", r.conc.pp_ms, r.conc.speedup);
      out << l2;
    } else {
      out << "(not measured)";
    }
    out << "\n";
    if (!r.alone.ok) out << "      ! alone: " << r.alone.error << "\n";
    if (r.have_conc && !r.conc.ok) out << "      ! both-busy: " << r.conc.error << "\n";
    if (basis.ok) {
      if (r.tag == "8k") { have_8k = true; sp8 = basis.speedup; }
      if (r.tag == "32k") { have_32k = true; sp32 = basis.speedup; }
      if (r.tag == "64k") sp64 = basis.speedup;
    }
  }
  if (results.empty()) out << "(no size has an emu" << args.split << "- run on both cards and a monolithic baseline)\n";
  for (const SizeResult& r : results) {
    if (!r.alone.ok) continue;
    out << "  " << r.tag << ": stage stalls (alone) A waits for a free slot " << Fmt("%.1f", r.alone.a_stall_ms)
        << " ms, B waits for A " << Fmt("%.1f", r.alone.b_stall_ms) << " ms (the fill is B's first wait)\n";
  }

  // ---- the go rule -----------------------------------------------------------------------------------------
  out << "\n---- go rule (Phase 0 gate: projected TTFT speedup >= " << Fmt("%.2f", args.need) << "x at 8k AND 32k) ----\n";
  int exit_code = 3;
  if (!have_8k || !have_32k) {
    out << "GO RULE: INCONCLUSIVE -- need emu" << args.split << " runs on both cards at 8k and 32k (found:"
        << (have_8k ? " 8k" : "") << (have_32k ? " 32k" : "") << ")\n";
    exit_code = 2;
  } else {
    const bool go = pp::GoRule(sp8, sp32, args.need);
    out << "basis: " << (have_basis_conc ? "both cards busy" : "each card alone (no both-busy run)") << "\n";
    out << "  8k  " << Fmt("%.3f", sp8) << "x   32k  " << Fmt("%.3f", sp32) << "x   64k  " << Fmt("%.3f", sp64)
        << "x (64k is reported, not part of the rule)\n";
    out << (go ? "GO RULE: GO -- build Phase 2 (the real two-GPU pipeline)\n"
               : "GO RULE: STOP -- below the design's gate; do not start Phase 2 (design section 8)\n");
    exit_code = go ? 0 : 3;
  }

  // ---- the other Phase 0 gates (informational) --------------------------------------------------------------
  out << "\n---- the design's other Phase 0 gates (informational; only the go rule decides) ----\n";
  const std::string e8 = emu + "-8k";
  if (a.chunks.count(e8) != 0 && a.mono_ms.count("mono-8k") != 0) {
    const double ratio = Sum(a.chunks[e8], &pp::Sample::a_ms) / a.mono_ms["mono-8k"];
    out << "stage A cost / full prefill, device 0, 8k: " << Fmt("%.3f", ratio) << "  (gate [0.49, 0.54]; k = " << args.split
        << " puts " << Fmt("%.1f", 100.0 * static_cast<double>(args.split) / 64.0) << "% of the layers in A)\n";
  }
  if (a.mono_ms.count("mono-8k") != 0 && b.mono_ms.count("mono-8k") != 0) {
    const double skew = a.mono_ms["mono-8k"] / b.mono_ms["mono-8k"] - 1.0;
    out << "device skew (monolithic 8k TTFT, device 0 vs device 1): " << Fmt("%+.1f", 100.0 * skew)
        << " %  (gate |skew| <= 6 %)\n";
  }
  if (have_hop || have_hop_conc) {
    const std::vector<pp::HopRow>& rows = have_hop_conc ? hop_conc : hop_alone;
    const double g = std::min(pp::MinGbps(rows, "d0to1", {"d2h", "h2d", "pipe_d2h", "pipe_h2d"}),
                              pp::MinGbps(rows, "d1to0", {"d2h", "h2d", "pipe_d2h", "pipe_h2d"}));
    out << "hop bandwidth, slowest leg over both card orders (" << (have_hop_conc ? "concurrent with prefill" : "idle cards")
        << "): " << Fmt("%.2f", g) << " GB/s  (gate >= 3 GB/s concurrent)\n";
    if (have_hop && have_hop_conc) {
      out << "  idle cards: " << Fmt("%.2f", std::min(pp::MinGbps(hop_alone, "d0to1", {"d2h", "h2d", "pipe_d2h", "pipe_h2d"}),
                                                       pp::MinGbps(hop_alone, "d1to0", {"d2h", "h2d", "pipe_d2h", "pipe_h2d"})))
          << " GB/s\n";
    }
  } else {
    out << "hop bandwidth: (not measured)\n";
  }
  const auto epilogue_cost = [&](const pp::BenchFile& f, const std::string& label, const char* what) {
    const auto it = f.chunks.find(label);
    if (it == f.chunks.end()) return;
    double sum = 0;
    int n = 0;
    for (const pp::Sample& s : it->second) {
      if (s.rows != 256) continue;
      sum += s.mtp_ms + s.inject_ms;
      ++n;
    }
    if (n > 0) {
      out << what << " per 256-row chunk on stage B (MTP priming + DFlash injection): " << Fmt("%.2f", sum / n)
          << " ms over " << n << " chunks  (gate < 12 ms, else plan k = 35 and the ring-window skip)\n";
    }
  };
  if (have_dflash) {
    epilogue_cost(db, "emu" + std::to_string(args.dflash_split) + "-8k", "--dflash:");
    for (const char* tag : {"8k", "32k"}) {
      const std::string label = "emu" + std::to_string(args.dflash_split) + "-" + tag;
      const auto xa = da.chunks.find(label), xb = db.chunks.find(label);
      const std::string mlabel = std::string("mono-") + tag;
      if (xa == da.chunks.end() || xb == db.chunks.end() || db.mono_ms.count(mlabel) == 0) continue;
      const pp::Projection p = pp::Project(xa->second, xb->second, db.mono_ms[mlabel], params);
      if (p.ok) {
        out << "--dflash at k = " << args.dflash_split << ", " << tag << ": mono " << Fmt("%.1f", db.mono_ms[mlabel])
            << " ms -> PP " << Fmt("%.1f", p.pp_ms) << " ms (" << Fmt("%.3f", p.speedup) << "x, each card alone, hop unscaled)\n";
      }
    }
  }
  if (have_mtp) epilogue_cost(mb, "emu" + std::to_string(args.split) + "-8k", "--mtp:");
  if (!have_dflash && !have_mtp) out << "DFlash injection / MTP priming per chunk: (not measured)\n";

  out << "\nnotes: the emulation times each stage with extra stream synchronizes and the host hop on ONE card; the real\n"
         "pipeline's copies cross two cards and its stages overlap, which the simulation (3 slots) models and the\n"
         "both-busy run (two processes at once, a start barrier) exercises on the clocks. Stage numbers are medians.\n";

  const std::string text = out.str();
  std::ofstream f(args.out, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "tool_pp_project: cannot write %s\n", args.out.c_str());
    return 2;
  }
  f << text;
  f.close();
  std::fputs(text.c_str(), stdout);
  return exit_code;
}
