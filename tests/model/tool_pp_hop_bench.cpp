// tests/model/tool_pp_hop_bench.cpp -- pipeline-parallel prefill, Phase 0 (docs/pp-prefill.md section 8): the
// bandwidth of the stage-boundary hop. Stage A's device exports each chunk with a hipMemcpyAsync device-to-host
// into pinned (hipHostMallocPortable) memory, stage B's device imports it with a hipMemcpyAsync host-to-device:
// no peer copy, no GPU-side wait (docs/tp.md 1.2: hipMemcpyPeerAsync corrupts on this box). This tool times
// exactly those two legs, for both card orders, at the sizes the pipeline moves per chunk:
//   2.5 MiB     one 256 x 5120 bf16 buffer (the residual stream),
//   7.5 MiB     the three carry buffers (residual, the two fused-norm buffers),
//   12.75 MiB   the carry plus a chunk's 4.25 MiB of KV blocks,
//   24 x 3 MiB  the GDN recurrent states (one call of 24 async copies, one synchronize),
// and a `pipe` leg where the export on the source device and the import on the destination device run
// at the same time from two threads (the pipeline's steady state: chunk c + 1 leaving A while chunk c enters B).
// Times are host wall clock around enqueue + synchronize (what the pipeline's threads pay), median of --reps.
//
// "Concurrent with compute": run it while two tool_pp_stage_bench processes (one per card) are mid-prefill
// (tools: the Phase 0 script does) -- this tool queues only copies and holds ~100 MiB per device. --duration
// repeats the whole sweep until that many seconds have passed, half of them per card order (the median then covers
// the busy window).
//
// Output (--out): lines `hop,<dir>,<size label>,<bytes>,<leg>,<median_ms>,<GB/s>` read by tool_pp_project; dir is
// d<src>to<dst> (visible ordinals), leg is d2h (on the source device), h2d (on the destination device), e2e
// (D2H then H2D, serial) or pipe_d2h / pipe_h2d (the two legs of the simultaneous run). GB/s is 1e9 bytes/s.
// Built, never add_test()'d (GPU). Usage (both cards visible):
//   $env:HIP_VISIBLE_DEVICES = '0,1'
//   tool_pp_hop_bench --src 0 --dst 1 --out hop_alone.csv [--reps 40] [--duration 0] [--mode alone]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/core/error.hpp"
#include "test_common.h"

namespace {

struct Args {
  int src = 0, dst = 1;
  int reps = 40;
  double duration_s = 0.0;
  std::string out;
  std::string mode = "alone";
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr, "tool_pp_hop_bench: %s\nusage: tool_pp_hop_bench --out file.csv [--src 0] [--dst 1] [--reps 40] "
                       "[--duration 0] [--mode alone|conc]\n", why.c_str());
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
    if (s == "--src") a.src = std::stoi(next());
    else if (s == "--dst") a.dst = std::stoi(next());
    else if (s == "--reps") a.reps = std::stoi(next());
    else if (s == "--duration") a.duration_s = std::stod(next());
    else if (s == "--out") a.out = next();
    else if (s == "--mode") a.mode = next();
    else Usage("unknown argument " + s);
  }
  if (a.out.empty()) Usage("--out is required");
  if (a.src == a.dst) Usage("--src and --dst must differ");
  if (a.reps < 3) Usage("--reps must be >= 3");
  return a;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double NowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One card's side of the hop: a stream, a device buffer and a pinned staging buffer of `bytes`.
struct Side {
  int device = 0;
  hipStream_t stream = nullptr;
  void* dev = nullptr;
  void* host = nullptr;
  size_t bytes = 0;
  Side(int d, size_t b) : device(d), bytes(b) {
    R4DX_HIP_CHECK(hipSetDevice(d));
    R4DX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    R4DX_HIP_CHECK(hipMalloc(&dev, bytes));
    R4DX_HIP_CHECK(hipMemset(dev, 0x5A, bytes));
    R4DX_HIP_CHECK(hipHostMalloc(&host, bytes, hipHostMallocPortable));
    std::fill_n(static_cast<unsigned char*>(host), bytes, static_cast<unsigned char>(0xA5));
  }
  ~Side() {
    (void)hipSetDevice(device);
    if (host != nullptr) (void)hipHostFree(host);
    if (dev != nullptr) (void)hipFree(dev);
    if (stream != nullptr) (void)hipStreamDestroy(stream);
  }
  Side(const Side&) = delete;
  Side& operator=(const Side&) = delete;
  // `pieces` copies of bytes / pieces each, one synchronize.
  void D2H(size_t n, int pieces) {
    R4DX_HIP_CHECK(hipSetDevice(device));
    const size_t each = n / static_cast<size_t>(pieces);
    for (int p = 0; p < pieces; ++p) {
      R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(host) + p * each, static_cast<char*>(dev) + p * each, each,
                                    hipMemcpyDeviceToHost, stream));
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(stream));
  }
  void H2D(size_t n, int pieces) {
    R4DX_HIP_CHECK(hipSetDevice(device));
    const size_t each = n / static_cast<size_t>(pieces);
    for (int p = 0; p < pieces; ++p) {
      R4DX_HIP_CHECK(hipMemcpyAsync(static_cast<char*>(dev) + p * each, static_cast<char*>(host) + p * each, each,
                                    hipMemcpyHostToDevice, stream));
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(stream));
  }
};

struct Size {
  std::string label;
  size_t bytes;
  int pieces;
};

int RunTool(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  int count = 0;
  R4DX_HIP_CHECK(hipGetDeviceCount(&count));
  if (a.src >= count || a.dst >= count) {
    std::fprintf(stderr, "tool_pp_hop_bench: needs two visible devices (HIP_VISIBLE_DEVICES=0,1); %d visible\n", count);
    return r4dx_test::kSkipReturnCode;
  }
  for (const int ord : {a.src, a.dst}) {
    char bus[64] = "?";
    (void)hipDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), ord);
    std::fprintf(stderr, "[pp-hop] visible ordinal %d is PCI %s\n", ord, bus);
  }
  constexpr size_t MiB = 1024 * 1024;
  const std::vector<Size> sizes = {{"2.5MiB", 5 * MiB / 2, 1},
                                   {"7.5MiB", 15 * MiB / 2, 3},
                                   {"12.75MiB", 51 * MiB / 4, 4},
                                   {"24x3MiB", 72 * MiB, 24}};
  size_t max_bytes = 0;
  for (const Size& s : sizes) max_bytes = std::max(max_bytes, s.bytes);
  std::string csv = "# tool_pp_hop_bench: mode " + a.mode + ", reps " + std::to_string(a.reps) + ", visible ordinals " +
                    std::to_string(a.src) + " -> " + std::to_string(a.dst) + "\n";
  const double t_begin = NowMs();

  // Both card orders: the pipeline wants src = stage A's card, dst = stage B's; the reverse is the sync-back.
  for (int order = 0; order < 2; ++order) {
    const int s = order == 0 ? a.src : a.dst;
    const int d = order == 0 ? a.dst : a.src;
    Side from(s, max_bytes), to(d, max_bytes);
    const std::string dir = "d" + std::to_string(s) + "to" + std::to_string(d);
    // pass collects the samples of every sweep until --duration is up (one sweep when it is 0)
    struct Acc {
      std::vector<double> d2h, h2d, e2e, pipe_d2h, pipe_h2d;
    };
    std::vector<Acc> acc(sizes.size());
    int sweeps = 0;
    do {
      for (size_t si = 0; si < sizes.size(); ++si) {
        const Size& sz = sizes[si];
        for (int w = 0; w < 3; ++w) {  // warm-up
          from.D2H(sz.bytes, sz.pieces);
          to.H2D(sz.bytes, sz.pieces);
        }
        for (int r = 0; r < a.reps; ++r) {
          const double t0 = NowMs();
          from.D2H(sz.bytes, sz.pieces);
          const double t1 = NowMs();
          to.H2D(sz.bytes, sz.pieces);
          const double t2 = NowMs();
          acc[si].d2h.push_back(t1 - t0);
          acc[si].h2d.push_back(t2 - t1);
          acc[si].e2e.push_back(t2 - t0);
        }
        // Both legs at once, two threads, a barrier-free start (a few ms of skew is the pipeline's too).
        std::vector<double> pd, ph;
        pd.resize(static_cast<size_t>(a.reps));
        ph.resize(static_cast<size_t>(a.reps));
        std::atomic<int> go{0};
        std::thread tb([&] {
          while (go.load() == 0) std::this_thread::yield();
          for (int r = 0; r < a.reps; ++r) {
            const double t0 = NowMs();
            to.H2D(sz.bytes, sz.pieces);
            ph[static_cast<size_t>(r)] = NowMs() - t0;
          }
        });
        go.store(1);
        for (int r = 0; r < a.reps; ++r) {
          const double t0 = NowMs();
          from.D2H(sz.bytes, sz.pieces);
          pd[static_cast<size_t>(r)] = NowMs() - t0;
        }
        tb.join();
        acc[si].pipe_d2h.insert(acc[si].pipe_d2h.end(), pd.begin(), pd.end());
        acc[si].pipe_h2d.insert(acc[si].pipe_h2d.end(), ph.begin(), ph.end());
      }
      ++sweeps;
      // --duration is split between the two card orders: each sweeps until its own half of it is up (a single clock
      // from the tool's start would let the first order use it all and leave the second one a lone sweep, outside
      // the busy window it is meant to cover).
    } while ((NowMs() - t_begin) / 1000.0 < a.duration_s * static_cast<double>(order + 1) / 2.0);

    for (size_t si = 0; si < sizes.size(); ++si) {
      const Size& sz = sizes[si];
      const auto line = [&](const char* leg, const std::vector<double>& v) {
        const double med = Median(v);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "hop,%s,%s,%zu,%s,%.4f,%.3f\n", dir.c_str(), sz.label.c_str(), sz.bytes, leg, med,
                      static_cast<double>(sz.bytes) / (med * 1e-3) / 1e9);
        csv += buf;
        std::fprintf(stderr, "[pp-hop] %s %-9s %-8s %8.3f ms  %6.2f GB/s\n", dir.c_str(), sz.label.c_str(), leg, med,
                     static_cast<double>(sz.bytes) / (med * 1e-3) / 1e9);
      };
      line("d2h", acc[si].d2h);
      line("h2d", acc[si].h2d);
      line("e2e", acc[si].e2e);
      line("pipe_d2h", acc[si].pipe_d2h);
      line("pipe_h2d", acc[si].pipe_h2d);
    }
    csv += "# " + dir + ": " + std::to_string(sweeps) + " sweep(s)\n";
  }
  std::ofstream f(a.out, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + a.out);
  f << csv;
  std::fprintf(stderr, "[pp-hop] wrote %s\n", a.out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return r4dx_test::RunGuardedMain("tool_pp_hop_bench", [&] { return RunTool(argc, argv); });
}
