// tests/kernels/tool_tp_ar_latency.cpp -- the in-engine all-reduce latency in tools/tp_bench's decode
// pattern (docs/tp.md 1.4, 10.1, gate G5).
//
// Per token: 128 slots, each preceded by a filler that streams --filler-mb MiB (60) from a 512 MiB
// hash-filled VRAM ring and dirties --dirty-mb MiB (4) of L2. Three conditions run INTERLEAVED
// token by token (tp_bench's decode mode, tests/kernels/tp_ar_harness.h):
//   (a) the real all-reduce in every slot -- through the engine's HostMailboxComm endpoint,
//   (b) a local stand-in kernel on the same grid (no host traffic),
//   (c) fillers only (no slot kernel: exactly NoopComm's baseline, docs/tp.md 1.4).
// The input-generating kernel runs before every token of every condition and the verify kernel
// after every (a) token, both outside the hipEvent pair (excluded, not subtracted); every (a) output
// is verified bit-exact. T = per-token ms, max over ranks. Reported, per message size and nb:
//   L_vs_no_ar_kernel = (T_a - T_c) / 128   -- the G5 number (<= 9.0 us at 10 KiB, <= 16.0 at 80 KiB)
//   L_vs_standin      = (T_a - T_b) / 128   -- recorded next to it
// with standard errors of the paired per-token differences (max over ranks). The 640 KiB row (channel
// 1) is measured for every nb in --nb-large-list, one TpGroup per nb.
//
//   tool_tp_ar_latency.exe [--tokens 50] [--sizes 10240,81920,655360] [--nb 4]
//       [--nb-large-list 4,8,16] [--filler-mb 60] [--dirty-mb 4] [--timeout-ms 500]
//       [--devices auto|a,b] [--need-gib 1.5] [--json path]
// Built, never add_test()'d. Exit 0 when every run verified; the G5 thresholds are printed, not gated.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "tp_ar_harness.h"

using namespace tp_harness;

namespace {

constexpr int kArsPerToken = 128;

struct Args {
  int tokens = 50;
  std::vector<size_t> sizes = {10240, 81920, 655360};
  int nb = 4;
  std::vector<int> nb_large = {4, 8, 16};
  int filler_mb = 60, dirty_mb = 4;
  int timeout_ms = tp::kArTimeoutDefaultMs;
  std::string devices;
  double need_gib = 1.5;
  std::string json;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "%s\nusage: tool_tp_ar_latency.exe [--tokens N] [--sizes a,b,...] [--nb N] [--nb-large-list a,b,...] "
               "[--filler-mb N] [--dirty-mb N] [--timeout-ms N] [--devices auto|a,b] [--need-gib X] [--json path]\n",
               why.c_str());
  std::exit(1);
}

template <class T>
std::vector<T> ParseList(const std::string& s) {
  std::vector<T> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(static_cast<T>(std::stoll(item)));
  return v;
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(s + " needs a value");
      return argv[++i];
    };
    try {
      if (s == "--tokens") a.tokens = std::stoi(next());
      else if (s == "--sizes") a.sizes = ParseList<size_t>(next());
      else if (s == "--nb") a.nb = std::stoi(next());
      else if (s == "--nb-large-list") a.nb_large = ParseList<int>(next());
      else if (s == "--filler-mb") a.filler_mb = std::stoi(next());
      else if (s == "--dirty-mb") a.dirty_mb = std::stoi(next());
      else if (s == "--timeout-ms") a.timeout_ms = std::stoi(next());
      else if (s == "--devices") a.devices = next();
      else if (s == "--need-gib") a.need_gib = std::stod(next());
      else if (s == "--json") a.json = next();
      else Usage("unknown option " + s);
    } catch (const std::exception&) {
      Usage("bad value for " + s);
    }
  }
  if (a.tokens < 1 || a.tokens > 100000) Usage("--tokens must be in [1, 100000]");
  if (a.sizes.empty() || a.nb_large.empty()) Usage("--sizes and --nb-large-list must not be empty");
  for (size_t b : a.sizes)
    if (b < 16 || b % 16 != 0 || b > tp::kMaxAllReduceBytes) Usage("--sizes: multiples of 16 in [16, 655360]");
  for (int nb : a.nb_large)
    if (nb < 1 || nb > 64) Usage("--nb-large-list entries must be in [1, 64]");
  if (a.nb < 1 || a.nb > 64) Usage("--nb must be in [1, 64]");
  return a;
}

struct Row {
  size_t bytes = 0;
  int channel = 0, nb = 0, nb_eff = 0;
  bool ok = false;
  std::string failure;
  double ta = NAN, tb = NAN, tc = NAN;           // max over ranks
  double L_nok = NAN, L_nok_se = NAN;            // vs no AR kernel (G5)
  double L_sb = NAN, L_sb_se = NAN;              // vs stand-in
  std::vector<double> quintiles[2];
  size_t tokens = 0;
};

double G5Limit(size_t bytes) { return bytes == 10240 ? 9.0 : bytes == 81920 ? 16.0 : NAN; }

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  const std::vector<int> devices = ParseDevices(a.devices);
  if (devices.size() != 2) {
    std::fprintf(stderr, "need two HIP devices (HIP_VISIBLE_DEVICES exposes %d); unset it or pass --devices a,b\n",
                 VisibleDevices());
    return 1;
  }
  PreflightVram(devices, a.need_gib);
  std::vector<Row> rows;
  std::vector<double> khz(2, 0.0);
  std::string error;
  try {
    RankPool pool(devices);
    std::printf("[latency] rank 0 -> %s, rank 1 -> %s\n", DeviceLabel(devices[0]).c_str(),
                DeviceLabel(devices[1]).c_str());
    std::printf("[latency] decode pattern: %d tokens x 128 slots per condition, filler %d MiB, dirty %d MiB\n",
                a.tokens, a.filler_mb, a.dirty_mb);
    std::fflush(stdout);
    for (size_t gi = 0; gi < a.nb_large.size(); ++gi) {
      const int nbl = a.nb_large[gi];
      TpSetup T;
      pool.on_error = [&](int r, const std::string& what) {
        std::fprintf(stderr, "[latency] rank %d (HIP device %d) error: %s\n", r, devices[static_cast<size_t>(r)],
                     what.c_str());
        if (T.eps.size() > static_cast<size_t>(r) && T.eps[static_cast<size_t>(r)]) {
          T.eps[static_cast<size_t>(r)]->Abort(core::kAbortHost, what);
        }
      };
      T.Create(pool, tp::TpGroup::Geometry{a.nb, nbl}, a.timeout_ms);
      khz = T.clock_khz;
      pool.RunAll([&](int r) { T.eps[static_cast<size_t>(r)]->SelfTest(); });
      for (size_t bytes : a.sizes) {
        const int ch = tp::MailboxLayout::ChannelFor(bytes);
        if (gi > 0 && ch == 0) continue;  // channel 0 does not depend on nb_large: measured once
        DriverConfig cfg;
        cfg.sizes = {bytes};
        cfg.K = kArsPerToken;
        cfg.conditions = 3;
        cfg.batches = 3LL * (a.tokens + 1);
        cfg.discard = 3;  // one warm-up token per condition
        cfg.fillers = true;
        cfg.filler_rd_bytes = static_cast<size_t>(a.filler_mb) << 20;
        cfg.filler_dirty_bytes = static_cast<size_t>(a.dirty_mb) << 20;
        cfg.nb[0] = a.nb;
        cfg.nb[1] = nbl;
        std::vector<DriverResult> res(2);
        pool.RunAll([&](int r) {
          res[static_cast<size_t>(r)] = RunDriver(*T.eps[static_cast<size_t>(r)], r, T.Stream(r), cfg);
        });
        Row row;
        row.bytes = bytes;
        row.channel = ch;
        row.nb = ch == 0 ? a.nb : nbl;
        row.nb_eff = tp::ChunkCall(bytes, row.nb).nb;
        row.ok = res[0].ok && res[1].ok;
        for (int r = 0; r < 2; ++r) {
          const std::string f = DescribeFailure(res[static_cast<size_t>(r)]);
          if (!f.empty()) row.failure += "rank " + std::to_string(r) + ": " + f + "; ";
        }
        double se_nok = NAN, se_sb = NAN;
        const auto upd = [](double& m, double v) {
          if (std::isfinite(v)) m = std::isfinite(m) ? std::max(m, v) : v;
        };
        for (int r = 0; r < 2; ++r) {
          const DecodeStats d = Decode(res[static_cast<size_t>(r)], kArsPerToken);
          upd(row.ta, d.ta);
          upd(row.tb, d.tb);
          upd(row.tc, d.tc);
          upd(se_nok, d.Lnok_se);
          upd(se_sb, d.L_se);
          row.quintiles[r] = d.L_quintiles;
          row.tokens = d.n;
        }
        row.L_nok = (row.ta - row.tc) * 1000.0 / kArsPerToken;
        row.L_sb = (row.ta - row.tb) * 1000.0 / kArsPerToken;
        row.L_nok_se = se_nok;
        row.L_sb_se = se_sb;
        char g5[48] = "";
        if (std::isfinite(G5Limit(bytes))) {
          std::snprintf(g5, sizeof g5, "  (G5 limit %.1f us: %s)", G5Limit(bytes),
                        row.L_nok <= G5Limit(bytes) ? "within" : "MISSED");
        }
        const std::string verdict = row.ok ? std::string("verified") : "FAILED: " + row.failure;
        std::printf("[decode] %7zu B ch%d nb=%-2d: per-token ms  AR %.3f  stand-in %.3f  fillers-only %.3f  ->  "
                    "L_vs_no_ar_kernel %.2f +- %.2f us, L_vs_standin %.2f +- %.2f us%s  [%s]\n",
                    bytes, ch, row.nb, row.ta, row.tb, row.tc, row.L_nok, row.L_nok_se, row.L_sb, row.L_sb_se, g5,
                    verdict.c_str());
        std::fflush(stdout);
        rows.push_back(row);
      }
      T.Destroy(pool);
      pool.on_error = nullptr;  // it captured this iteration's T
    }
  } catch (const std::exception& e) {
    error = e.what();
    std::printf("ERROR: %s\n", e.what());
  }

  // best nb for the 640 KiB (channel-1) row
  int best_nb = -1;
  double best_L = INFINITY, nb4_L = NAN;
  for (const Row& r : rows) {
    if (r.channel != 1 || r.bytes != tp::kMaxAllReduceBytes || !r.ok) continue;
    if (r.nb == 4) nb4_L = r.L_nok;
    if (r.L_nok < best_L) {
      best_L = r.L_nok;
      best_nb = r.nb;
    }
  }
  bool all_ok = error.empty() && !rows.empty();
  for (const Row& r : rows) all_ok = all_ok && r.ok;
  if (best_nb > 0) {
    std::printf("[latency] 640 KiB: best nb %d (L_vs_no_ar_kernel %.2f us); nb 4: %.2f us; %s\n", best_nb, best_L, nb4_L,
                std::isfinite(nb4_L) && best_L <= 0.9 * nb4_L
                    ? "at least 10% better than nb 4 -- stress it (1M ARs) before changing --tp-ar-nb-large"
                    : "keep --tp-ar-nb-large 4");
  }
  if (!a.json.empty()) {
    std::string j = "{\n  \"mode\": \"latency\",\n";
    j += "  \"status\": " + JsonStr(all_ok ? "ok" : "failed") + ",\n";
    j += "  \"config\": {\"pattern\": \"decode\", \"tokens\": " + std::to_string(a.tokens) +
         ", \"ars_per_token\": 128, \"filler_mb\": " + std::to_string(a.filler_mb) + ", \"dirty_mb\": " +
         std::to_string(a.dirty_mb) + ", \"nb\": " + std::to_string(a.nb) + ", \"nt\": 256, \"timeout_ms\": " +
         std::to_string(a.timeout_ms) + ", \"devices\": [" + std::to_string(devices[0]) + ", " +
         std::to_string(devices[1]) + "]},\n";
    j += "  \"wallclock_khz\": " + JsonArr(khz) + ",\n";
    j += "  \"L_formula\": " +
         JsonStr("L_vs_no_ar_kernel = (T_a - T_c) / 128 (the G5 number; condition (c) is NoopComm's no-kernel "
                 "baseline); L_vs_standin = (T_a - T_b) / 128. T = per-token ms, max over ranks; se = standard error of "
                 "the paired per-token differences, max over ranks.") +
         ",\n";
    j += "  \"rows\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      j += "    {\"bytes\": " + std::to_string(r.bytes) + ", \"channel\": " + std::to_string(r.channel) +
           ", \"nb\": " + std::to_string(r.nb) + ", \"nb_effective\": " + std::to_string(r.nb_eff) +
           ", \"verified\": " + (r.ok ? "true" : "false") + ", \"tokens_per_condition\": " + std::to_string(r.tokens) +
           ", \"per_token_ms_allreduce\": " + JsonNum(r.ta) + ", \"per_token_ms_standin\": " + JsonNum(r.tb) +
           ", \"per_token_ms_fillers_only\": " + JsonNum(r.tc) + ", \"L_vs_no_ar_kernel_us\": " + JsonNum(r.L_nok) +
           ", \"L_vs_no_ar_kernel_se_us\": " + JsonNum(r.L_nok_se) + ", \"L_vs_standin_us\": " + JsonNum(r.L_sb) +
           ", \"L_vs_standin_se_us\": " + JsonNum(r.L_sb_se) + ", \"g5_limit_us\": " + JsonNum(G5Limit(r.bytes)) +
           ", \"L_vs_no_ar_kernel_by_quintile_rank0\": " + JsonArr(r.quintiles[0]) +
           ", \"L_vs_no_ar_kernel_by_quintile_rank1\": " + JsonArr(r.quintiles[1]) + ", \"failure\": " +
           (r.ok ? std::string("null") : JsonStr(r.failure)) + "}" + (i + 1 < rows.size() ? "," : "") + "\n";
    }
    j += "  ],\n  \"best_nb_640KiB\": " + (best_nb > 0 ? std::to_string(best_nb) : std::string("null")) +
         ",\n  \"error\": " + (error.empty() ? std::string("null") : JsonStr(error)) + "\n}\n";
    if (!WriteFile(a.json, j)) {
      std::fprintf(stderr, "cannot write %s\n", a.json.c_str());
      return 1;
    }
  }
  return all_ok ? 0 : 1;
}
