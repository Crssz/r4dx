// tests/kernels/tool_tp_ar_stress.cpp -- N verified all-reduces through the engine's HostMailboxComm
// on both GPUs (docs/tp.md 10.1, gate G5: `--count 10000000 --pattern decode`).
//
// A port of tools/tp_bench's `stress` mode onto TpGroup / HostMailboxComm (tests/kernels/
// tp_ar_harness.h): every output element of every all-reduce is checked against tp_bench's hash
// pattern on the device, both ranks vote after every batch, and the run stops at the first mismatch,
// timeout, protocol violation or host error with a report of which rank / call failed.
//   --pattern isolated  back-to-back all-reduces in verified batches of 50
//   --pattern decode    tp_bench's decode pattern: tokens of 128 all-reduces, each preceded by a
//                       filler that streams --filler-mb MiB and dirties --dirty-mb MiB of L2; the count
//                       is rounded up to whole tokens
//   --mixed             sizes cycle over both channels instead of the fixed --bytes
// Built, never add_test()'d: it takes the desktop GPU for minutes.
//
//   tool_tp_ar_stress.exe [--count N=10000000] [--pattern isolated|decode] [--bytes B=10240]
//       [--mixed] [--nb 4] [--nb-large 4] [--timeout-ms 500] [--filler-mb 60] [--dirty-mb 4]
//       [--devices auto|a,b] [--need-gib X] [--json path]
// Exit codes (tp_bench's): 0 ok, 1 error, 2 data mismatch / protocol violation, 3 timeout / abort.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tp_ar_harness.h"

using namespace tp_harness;

namespace {

struct Args {
  int64_t count = 10000000;
  std::string pattern = "isolated";
  size_t bytes = 10240;
  bool mixed = false;
  int nb = 4, nb_large = 4;
  int timeout_ms = tp::kArTimeoutDefaultMs;
  int filler_mb = 60, dirty_mb = 4;
  std::string devices;
  double need_gib = -1;
  std::string json;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "%s\nusage: tool_tp_ar_stress.exe [--count N] [--pattern isolated|decode] [--bytes B] [--mixed] "
               "[--nb N] [--nb-large N] [--timeout-ms N] [--filler-mb N] [--dirty-mb N] [--devices auto|a,b] "
               "[--need-gib X] [--json path]\n",
               why.c_str());
  std::exit(1);
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
      if (s == "--count") a.count = std::stoll(next());
      else if (s == "--pattern") a.pattern = next();
      else if (s == "--bytes") a.bytes = static_cast<size_t>(std::stoll(next()));
      else if (s == "--mixed") a.mixed = true;
      else if (s == "--nb") a.nb = std::stoi(next());
      else if (s == "--nb-large") a.nb_large = std::stoi(next());
      else if (s == "--timeout-ms") a.timeout_ms = std::stoi(next());
      else if (s == "--filler-mb") a.filler_mb = std::stoi(next());
      else if (s == "--dirty-mb") a.dirty_mb = std::stoi(next());
      else if (s == "--devices") a.devices = next();
      else if (s == "--need-gib") a.need_gib = std::stod(next());
      else if (s == "--json") a.json = next();
      else Usage("unknown option " + s);
    } catch (const std::exception&) {
      Usage("bad value for " + s);
    }
  }
  if (a.pattern != "isolated" && a.pattern != "decode") Usage("--pattern must be isolated|decode");
  if (a.count < 1 || a.count > 2000000000LL) Usage("--count must be in [1, 2e9]");
  if (a.bytes < 16 || a.bytes % 16 != 0 || a.bytes > tp::kMaxAllReduceBytes) Usage("--bytes: multiple of 16 in [16, 655360]");
  if (a.nb < 1 || a.nb > 64 || a.nb_large < 1 || a.nb_large > 64) Usage("--nb / --nb-large must be in [1, 64]");
  if (a.filler_mb < 0 || a.filler_mb > 1024 || a.dirty_mb < 0 || a.dirty_mb > 256) Usage("bad --filler-mb/--dirty-mb");
  if (a.need_gib < 0) a.need_gib = a.pattern == "decode" ? 1.5 : 0.5;
  return a;
}

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
  const bool decode = a.pattern == "decode";
  const std::vector<size_t> sizes =
      a.mixed ? std::vector<size_t>{10240, 81920, 174080, 184320, 655360, 20480, 40960, 122880}
              : std::vector<size_t>{a.bytes};
  DriverConfig cfg;
  cfg.sizes = sizes;
  cfg.K = decode ? 128 : 50;
  cfg.batches = (a.count + cfg.K - 1) / cfg.K;
  cfg.fillers = decode;
  cfg.filler_rd_bytes = static_cast<size_t>(a.filler_mb) << 20;
  cfg.filler_dirty_bytes = static_cast<size_t>(a.dirty_mb) << 20;
  cfg.nb[0] = a.nb;
  cfg.nb[1] = a.nb_large;
  const int64_t requested_rounded = cfg.batches * cfg.K;

  int code = 0;
  std::vector<DriverResult> res(2);
  std::vector<double> khz(2, 0.0);
  std::string error;
  double secs = 0;
  try {
    RankPool pool(devices);
    TpSetup T;
    pool.on_error = [&](int r, const std::string& what) {
      // RunAll rethrows one root cause; keep every rank's own error in the log.
      std::fprintf(stderr, "[stress] rank %d (HIP device %d) error: %s\n", r, devices[static_cast<size_t>(r)], what.c_str());
      if (T.eps.size() > static_cast<size_t>(r) && T.eps[static_cast<size_t>(r)]) {
        T.eps[static_cast<size_t>(r)]->Abort(core::kAbortHost, what);
      }
    };
    T.Create(pool, tp::TpGroup::Geometry{a.nb, a.nb_large}, a.timeout_ms);
    khz = T.clock_khz;
    std::printf("[stress] rank 0 -> %s, rank 1 -> %s\n", DeviceLabel(devices[0]).c_str(), DeviceLabel(devices[1]).c_str());
    std::printf("[stress] pattern %s, %lld all-reduces (%lld requested), %s, nb %d / %d, timeout %d ms\n",
                a.pattern.c_str(), static_cast<long long>(requested_rounded), static_cast<long long>(a.count),
                a.mixed ? "mixed sizes" : (std::to_string(a.bytes) + " B").c_str(), a.nb, a.nb_large, a.timeout_ms);
    std::fflush(stdout);
    pool.RunAll([&](int r) { T.eps[static_cast<size_t>(r)]->SelfTest(); });
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    pool.RunAll([&](int r) {
      DriverConfig c = cfg;
      if (r == 0) {
        c.on_batch = [&](int64_t done) {
          if (SecondsSince(last) >= 5.0) {
            last = std::chrono::steady_clock::now();
            const double el = SecondsSince(t0);
            std::printf("[stress] %lld / %lld calls verified OK, %.0f calls/s, %.0f s elapsed\n",
                        static_cast<long long>(done), static_cast<long long>(requested_rounded), done / el, el);
            std::fflush(stdout);
          }
        };
      }
      res[static_cast<size_t>(r)] = RunDriver(*T.eps[static_cast<size_t>(r)], r, T.Stream(r), c);
    });
    secs = SecondsSince(t0);
    T.Destroy(pool);
  } catch (const std::exception& e) {
    error = e.what();
  }

  const int64_t verified = std::min(res[0].calls_verified, res[1].calls_verified);
  // A host error on any rank comes first (RunDriver then Abort()s, so the ranks also report an
  // abort, but the protocol is not to blame); a timeout is a device-claimed kAbortTimeout, not
  // merely "some abort word was set".
  bool host_error = !error.empty(), mismatch = false, violation = false, timeout = false, aborted = false;
  for (const DriverResult& R : res) {
    host_error = host_error || !R.error.empty();
    mismatch = mismatch || R.verify.mismatches != 0;
    violation = violation || (R.comm_status.claimed && R.comm_status.code == core::kAbortProtocol);
    timeout = timeout || (R.comm_status.claimed && R.comm_status.code == core::kAbortTimeout);
    aborted = aborted || !R.abort_message.empty();
  }
  const bool all_ok = error.empty() && res[0].ok && res[1].ok;
  code = all_ok ? 0 : host_error ? 1 : (violation || mismatch) ? 2 : (timeout || aborted) ? 3 : 1;
  const char* status = code == 0   ? "ok"
                       : code == 1 ? "error"
                       : violation ? "protocol_violation"
                       : code == 2 ? "mismatch"
                       : timeout   ? "timeout"
                                   : "aborted";
  std::printf("[stress %s] %s: %lld calls verified in %.1f s (%.0f calls/s)\n", a.pattern.c_str(), status,
              static_cast<long long>(verified), secs, secs > 0 ? verified / secs : 0.0);
  if (!error.empty()) std::printf("    ERROR: %s\n", error.c_str());
  for (int r = 0; r < 2; ++r) {
    const std::string f = DescribeFailure(res[static_cast<size_t>(r)]);
    if (!f.empty()) std::printf("    rank %d: %s\n", r, f.c_str());
  }

  if (!a.json.empty()) {
    std::string j = "{\n";
    j += "  \"mode\": \"stress\",\n";
    j += "  \"status\": " + JsonStr(status) + ",\n  \"exit_code\": " + std::to_string(code) + ",\n";
    j += "  \"config\": {\"pattern\": " + JsonStr(a.pattern) + ", \"mixed\": " + (a.mixed ? "true" : "false") +
         ", \"bytes\": " + std::to_string(a.bytes) + ", \"nb\": " + std::to_string(a.nb) +
         ", \"nb_large\": " + std::to_string(a.nb_large) + ", \"nt\": 256, \"timeout_ms\": " +
         std::to_string(a.timeout_ms) + ", \"filler_mb\": " + std::to_string(decode ? a.filler_mb : 0) +
         ", \"dirty_mb\": " + std::to_string(decode ? a.dirty_mb : 0) + ", \"ars_per_verified_batch\": " +
         std::to_string(cfg.K) + ", \"devices\": [" + std::to_string(devices[0]) + ", " + std::to_string(devices[1]) +
         "]},\n";
    j += "  \"wallclock_khz\": " + JsonArr(khz) + ",\n";
    j += "  \"result\": {\"calls_verified\": " + std::to_string(verified) + ", \"calls_requested_rounded\": " +
         std::to_string(requested_rounded) + ", \"seconds\": " + JsonNum(secs) +
         ", \"calls_per_s\": " + JsonNum(secs > 0 ? verified / secs : NAN) + "},\n";
    j += "  \"error\": " + (error.empty() ? std::string("null") : JsonStr(error)) + ",\n  \"ranks\": [\n";
    for (int r = 0; r < 2; ++r) {
      const DriverResult& R = res[static_cast<size_t>(r)];
      j += "    {\"rank\": " + std::to_string(r) + ", \"device\": " + std::to_string(devices[static_cast<size_t>(r)]) +
           ", \"ok\": " + (R.ok ? "true" : "false") + ", \"calls_verified\": " + std::to_string(R.calls_verified) +
           ", \"mismatches\": " + std::to_string(R.verify.mismatches) + ", \"failure\": " +
           (R.ok ? std::string("null") : JsonStr(DescribeFailure(R))) + ", \"n_abort_exits\": " +
           std::to_string(R.comm_status.n_abort_exits) + ", \"n_skipped_blocks\": " +
           std::to_string(R.comm_status.n_skipped) + "}" + (r == 0 ? "," : "") + "\n";
    }
    j += "  ]\n}\n";
    if (!WriteFile(a.json, j)) {
      std::fprintf(stderr, "cannot write %s\n", a.json.c_str());
      if (code == 0) code = 1;
    }
  }
  return code;
}
