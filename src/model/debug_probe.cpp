#include "debug_probe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kernels/model_kernels.h"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

namespace {

// Host time in ms since the first call (steady clock).
double HostMs() {
  using Clock = std::chrono::steady_clock;
  static const Clock::time_point t0 = Clock::now();
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

double Quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double x = q * static_cast<double>(v.size() - 1);
  const size_t i = static_cast<size_t>(x);
  const double f = x - static_cast<double>(i);
  return i + 1 < v.size() ? v[i] * (1.0 - f) + v[i + 1] * f : v[i];
}
double Median(const std::vector<double>& v) { return Quantile(v, 0.5); }
double Mean(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

bool IsRound(const std::string& kind) { return kind == "mtp_round" || kind == "dflash_round"; }

// A span's class for the report: the ProfiledCall name without its "gemm:" prefix.
const char* ClassOf(const std::string& name) {
  return name.compare(0, 5, "gemm:") == 0 ? name.c_str() + 5 : name.c_str();
}

}  // namespace

DebugProbe* DebugProbe::Get() {
  static DebugProbe* const probe = []() -> DebugProbe* {
    const bool clock = EnvOn("R4DX_CLOCK_PROBE");
    int linears = 0;
    if (EnvOn("R4DX_PROFILE_LINEARS")) {
      linears = std::strcmp(std::getenv("R4DX_PROFILE_LINEARS"), "all") == 0 ? 2 : 1;
    }
    if (!clock && linears == 0) return nullptr;
    const char* json = std::getenv("R4DX_PROBE_JSON");
    static DebugProbe instance(clock, linears, json != nullptr ? json : "");
    instance.Start();
    if (linears != 0) linears_ = &instance;
    return &instance;
  }();
  return probe;
}

DebugProbe::DebugProbe(bool clock, int linears, std::string json_path)
    : clock_(clock), linears_mode_(linears), json_path_(std::move(json_path)) {
  const char* tl = std::getenv("R4DX_PROBE_TIMELINE");
  if (tl != nullptr && tl[0] != '\0') {
    timeline_ = std::fopen(tl, "w");
    if (timeline_ == nullptr) std::fprintf(stderr, "[probe] cannot write R4DX_PROBE_TIMELINE %s\n", tl);
  }
}

// The bench's calibration (tool_trellis_gemm_bench.cpp's CalibrateClocks and Timer::Calibrate),
// on a private stream: the REALTIME rate from pairs of 1e5 / 1e6-iteration probes against host
// time (the pair's difference cancels launch and synchronize), the empty stamp bracket, and the
// probe at its in-model length on the idle device.
void DebugProbe::Start() {
  mirror_.assign(static_cast<size_t>(cap_), 0);
  R4DX_HIP_CHECK(hipMalloc(&dev_, static_cast<size_t>(cap_) * sizeof(unsigned long long)));
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0) == hipSuccess && khz > 0) {
    wall_mhz_attr_ = khz / 1000.0;
  }
  hipStream_t cs = nullptr;
  R4DX_HIP_CHECK(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking));
  // One probe: host microseconds around launch + synchronize, and d(clock64), d(wall_clock64).
  auto probe = [&](int iters, double* dc, double* dw) {
    const double h0 = HostMs();
    r4dx_model_clock_probe(P(dev_), iters, P(cs));
    R4DX_HIP_CHECK(hipStreamSynchronize(cs));
    const double us = (HostMs() - h0) * 1000.0;
    unsigned long long k[4];
    R4DX_HIP_CHECK(hipMemcpy(k, dev_, sizeof k, hipMemcpyDeviceToHost));
    *dc = static_cast<double>(k[2] - k[0]);
    *dw = static_cast<double>(k[3] - k[1]);
    return us;
  };
  double dc0, dw0, dc1, dw1;
  probe(1000, &dc0, &dw0);
  std::vector<double> wall, per_add;
  for (int i = 0; i < 3; ++i) {
    const double ua = probe(100000, &dc0, &dw0);
    const double ub = probe(1000000, &dc1, &dw1);
    wall.push_back((dw1 - dw0) / (ub - ua));
    per_add.push_back(dc1 / 8e6);
  }
  wall_mhz_ = Median(wall);
  idle_cycles_per_add_ = Median(per_add);
  std::vector<double> bracket;
  for (int i = 0; i < 64; ++i) {
    r4dx_model_wall_stamp(P(dev_), P(cs));
    r4dx_model_wall_stamp(P(dev_ + 1), P(cs));
    R4DX_HIP_CHECK(hipStreamSynchronize(cs));
    unsigned long long t[2];
    R4DX_HIP_CHECK(hipMemcpy(t, dev_, sizeof t, hipMemcpyDeviceToHost));
    bracket.push_back(static_cast<double>(t[1] - t[0]) / wall_mhz_ / 1000.0);
  }
  bracket_ms_ = Median(bracket);
  std::vector<double> mhz, us;
  for (int i = 0; i < 16; ++i) {
    probe(probe_iters_, &dc0, &dw0);
    if (dw0 <= 0) continue;
    mhz.push_back(dc0 / dw0 * wall_mhz_);
    us.push_back(dw0 / wall_mhz_);
  }
  idle_probe_mhz_ = Median(mhz);
  idle_probe_us_ = Median(us);
  R4DX_HIP_CHECK(hipStreamDestroy(cs));
  std::fprintf(stderr,
               "[probe] on:%s%s; REALTIME %.3f MHz against host time (attribute %.1f), empty stamp "
               "bracket %.2f us, idle probe (%d iters) %.2f us at %.0f MHz\n",
               clock_ ? " clock probe after every Mlp + round composition" : "",
               linears_mode_ == 2   ? " + every ProfiledCall span"
               : linears_mode_ == 1 ? " + linear spans"
                                    : "",
               wall_mhz_, wall_mhz_attr_, bracket_ms_ * 1000.0, probe_iters_, idle_probe_us_,
               idle_probe_mhz_);
}

uint64_t DebugProbe::Alloc(uint64_t n) {
  const uint64_t pos = next_ % cap_;
  if (pos + n > cap_) next_ += cap_ - pos;  // a record's slots never wrap
  const uint64_t slot = next_;
  next_ += n;
  return slot;
}

uint64_t DebugProbe::Stamp(hipStream_t s) {
  const uint64_t slot = Alloc(1);
  r4dx_model_wall_stamp(P(dev_ + slot % cap_), P(s));
  return slot;
}

double DebugProbe::Ms(uint64_t from, uint64_t to) const {
  const auto d = static_cast<int64_t>(mirror_[static_cast<size_t>(to % cap_)] -
                                      mirror_[static_cast<size_t>(from % cap_)]);
  return static_cast<double>(d) / wall_mhz_ / 1000.0 - bracket_ms_;
}

int DebugProbe::Begin(hipStream_t s, const char* kind, int64_t T) {
  Call c;
  c.kind = kind;
  c.T = T;
  c.serial = ++serial_;
  c.parent = stack_.empty() ? -1 : stack_.back();
  c.host_b = HostMs();
  c.b = Stamp(s);
  calls_.push_back(std::move(c));
  const int id = static_cast<int>(calls_.size()) - 1;
  stack_.push_back(id);
  return id;
}

void DebugProbe::End(hipStream_t s, bool host_done) {
  if (stack_.empty()) return;
  const int id = stack_.back();
  stack_.pop_back();
  Call& c = calls_[static_cast<size_t>(id)];
  c.e = Stamp(s);
  c.host_e = host_done ? HostMs() : -1.0;
  c.closed = true;
  closed_.push_back(id);
}

void DebugProbe::Abandon(int id) {
  const auto it = std::find(stack_.begin(), stack_.end(), id);
  if (it == stack_.end()) return;
  abandoned_ += static_cast<int64_t>(stack_.end() - it);
  stack_.erase(it, stack_.end());
  span_open_ = false;
}

void DebugProbe::SetCommitted(int64_t tokens) {
  if (!stack_.empty()) calls_[static_cast<size_t>(stack_.back())].committed = tokens;
}

void DebugProbe::AfterMlp(hipStream_t s) {
  if (!clock_ || stack_.empty()) return;
  const uint64_t slot = Alloc(5);
  r4dx_model_clock_probe(P(dev_ + slot % cap_), probe_iters_, P(s));
  probes_.push_back({stack_.back(), layer_, slot});
}

bool DebugProbe::WantsSpan(const char* name) const {
  return !stack_.empty() && (linears_mode_ == 2 || std::strncmp(name, "gemm:", 5) == 0);
}

void DebugProbe::BeginSpan(hipStream_t s, const char* name) {
  cur_ = Span{stack_.back(), name, layer_, Stamp(s), 0, 0, false};
  span_open_ = true;
}

void DebugProbe::MarkInput(hipStream_t s) {
  if (!span_open_ || cur_.has_mid) return;
  cur_.mid = Stamp(s);
  cur_.has_mid = true;
}

void DebugProbe::EndSpan(hipStream_t s) {
  if (!span_open_) return;
  cur_.e = Stamp(s);
  spans_.push_back(cur_);
  span_open_ = false;
}

void DebugProbe::Collect(hipStream_t s) {
  R4DX_HIP_CHECK(hipStreamSynchronize(s));
  const double now = HostMs();
  if (next_ != copied_) {
    uint64_t from = copied_;
    if (next_ - from > cap_) {
      from = next_ - cap_;
      ++overflows_;
    }
    while (from < next_) {
      const uint64_t pos = from % cap_;
      const uint64_t n = std::min(cap_ - pos, next_ - from);
      R4DX_HIP_CHECK(hipMemcpy(mirror_.data() + pos, dev_ + pos,
                               static_cast<size_t>(n) * sizeof(unsigned long long),
                               hipMemcpyDeviceToHost));
      from += n;
    }
    copied_ = next_;
  }

  // Calls, in the order they closed: a child before its parent, the top-level calls in time order.
  for (int id : closed_) {
    Call& c = calls_[static_cast<size_t>(id)];
    if (c.host_e < 0) c.host_e = now;
    if (!Valid(c.b) || !Valid(c.e)) {
      ++dropped_;
      continue;
    }
    const double gpu = Ms(c.b, c.e);
    if (c.parent >= 0) {
      Call& p = calls_[static_cast<size_t>(c.parent)];
      p.child_ms[c.kind] += gpu;
      p.child_T[c.kind] = c.T;
    }
    const Key key{c.kind, c.T};
    CallAgg& a = call_agg_[key];
    a.gpu_ms.push_back(gpu);
    // The call's own GPU span as a timeline row (span name "call", layer -1, start 0), so a reader
    // can take "other" = call - sum(spans) per call (tools/prefill/probe_depth.py).
    if (timeline_ != nullptr) {
      std::fprintf(timeline_, "%s,%lld,%llu,call,-1,0.000,%.3f\n", c.kind.c_str(),
                   static_cast<long long>(c.T), static_cast<unsigned long long>(c.serial),
                   gpu * 1000.0);
    }
    a.host_ms.push_back(c.host_e - c.host_b);
    double gap = -1.0;
    if (c.parent < 0) {
      // Host time outside the model between two top-level calls of one kind (the caller's own work
      // per step or round); another kind in between (a prefill) starts a new run.
      if (last_top_done_ >= 0 && last_top_kind_ == c.kind) {
        gap = c.host_b - last_top_done_;
        a.gap_ms.push_back(gap);
      }
      last_top_done_ = c.host_e;
      last_top_kind_ = c.kind;
    }
    if (IsRound(c.kind)) {
      const auto ms = [&](const char* k) {
        const auto it = c.child_ms.find(k);
        return it == c.child_ms.end() ? 0.0 : it->second;
      };
      const auto vt = c.child_T.find("verify");
      rounds_[key].push_back(RoundRow{c.host_e - c.host_b, gap,
                                      ms("mtp_draft") + ms("dflash_draft"), ms("verify"),
                                      ms("dflash_inject"),
                                      vt == c.child_T.end() ? 0 : vt->second, c.committed});
    }
  }
  closed_.clear();

  for (const Span& sp : spans_) {
    if (!Valid(sp.b) || !Valid(sp.e) || (sp.has_mid && !Valid(sp.mid))) {
      ++dropped_;
      continue;
    }
    const Call& c = calls_[static_cast<size_t>(sp.call)];
    SpanAgg& a = span_agg_[{c.kind, c.T, sp.name}];
    const double total = Ms(sp.b, sp.e);
    a.n += 1;
    a.total_ms += total;
    a.total_us.push_back(total * 1000.0);
    if (sp.has_mid) {
      a.with_input += 1;
      a.in_ms += Ms(sp.b, sp.mid);
      a.gemm_ms += Ms(sp.mid, sp.e);
    } else {
      a.gemm_ms += total;
    }
    auto& l = a.by_layer[sp.layer];
    l.first += 1;
    l.second += total;
    if (timeline_ != nullptr && Valid(c.b)) {
      std::fprintf(timeline_, "%s,%lld,%llu,%s,%lld,%.3f,%.3f\n", c.kind.c_str(),
                   static_cast<long long>(c.T), static_cast<unsigned long long>(c.serial), sp.name,
                   static_cast<long long>(sp.layer), Ms(c.b, sp.b) * 1000.0, total * 1000.0);
    }
  }
  spans_.clear();

  for (const ProbeRec& p : probes_) {
    if (!Valid(p.slot) || !Valid(p.slot + 3)) {
      ++dropped_;
      continue;
    }
    const auto at = [&](uint64_t i) { return mirror_[static_cast<size_t>((p.slot + i) % cap_)]; };
    const double dc = static_cast<double>(at(2) - at(0));
    const double dw = static_cast<double>(at(3) - at(1));
    if (dw <= 0) continue;
    const Call& c = calls_[static_cast<size_t>(p.call)];
    clock_agg_[{c.kind, c.T}].push_back(
        ClockSample{c.serial, p.layer, dc / dw * wall_mhz_, dw / wall_mhz_,
                    dc / (8.0 * probe_iters_)});
  }
  probes_.clear();

  if (stack_.empty() && !span_open_) calls_.clear();
}

DebugProbe::~DebugProbe() {
  // No HIP call here: the runtime may already be shutting down. The ring stays allocated until the
  // process ends; whatever was never collected is reported as unfinished.
  Report();
  if (!json_path_.empty()) WriteJson(json_path_);
  if (timeline_ != nullptr) std::fclose(timeline_);
}

void DebugProbe::Report() const {
  const int64_t unfinished = static_cast<int64_t>(closed_.size() + stack_.size() + spans_.size() +
                                                  probes_.size());
  std::fprintf(stderr,
               "[probe] ---- summary (src/model/debug_probe.h): REALTIME %.3f MHz, empty bracket "
               "%.2f us (subtracted), idle probe %.2f us at %.0f MHz ----\n",
               wall_mhz_, bracket_ms_ * 1000.0, idle_probe_us_, idle_probe_mhz_);
  std::fprintf(stderr,
               "[probe] calls: GPU span (first stamp to last) and host time per call, ms\n");
  for (const auto& kv : call_agg_) {
    const CallAgg& a = kv.second;
    std::fprintf(stderr,
                 "[probe]   %-13s T=%-3lld n=%-6zu gpu median %8.3f [p10 %8.3f p90 %8.3f] "
                 "mean %8.3f | host median %8.3f",
                 kv.first.first.c_str(), static_cast<long long>(kv.first.second), a.gpu_ms.size(),
                 Median(a.gpu_ms), Quantile(a.gpu_ms, 0.1), Quantile(a.gpu_ms, 0.9), Mean(a.gpu_ms),
                 Median(a.host_ms));
    if (!a.gap_ms.empty()) std::fprintf(stderr, " | gap to previous median %.3f", Median(a.gap_ms));
    std::fprintf(stderr, "\n");
  }
  if (!clock_agg_.empty()) {
    std::fprintf(stderr,
                 "[probe] clock (probe after every Mlp), MHz: median [p10 p90] min | median by "
                 "layer quarter | probe us, cycles per add\n");
  }
  for (const auto& kv : clock_agg_) {
    std::vector<double> mhz, us, cpa, q[4];
    int64_t max_layer = 0;
    for (const ClockSample& c : kv.second) max_layer = std::max(max_layer, c.layer);
    for (const ClockSample& c : kv.second) {
      mhz.push_back(c.mhz);
      us.push_back(c.us);
      cpa.push_back(c.cycles_per_add);
      if (c.layer >= 0) q[std::min<int64_t>(3, c.layer * 4 / (max_layer + 1))].push_back(c.mhz);
    }
    std::fprintf(stderr,
                 "[probe]   %-13s T=%-3lld %7zu probes: %5.0f [%5.0f %5.0f] %5.0f | %5.0f %5.0f "
                 "%5.0f %5.0f | %.2f us, %.3f\n",
                 kv.first.first.c_str(), static_cast<long long>(kv.first.second), mhz.size(),
                 Median(mhz), Quantile(mhz, 0.1), Quantile(mhz, 0.9),
                 *std::min_element(mhz.begin(), mhz.end()), Median(q[0]), Median(q[1]),
                 Median(q[2]), Median(q[3]), Median(us), Median(cpa));
  }
  if (!rounds_.empty()) {
    std::fprintf(stderr,
                 "[probe] rounds, ms per round (mean): host | drafter, verify (mean T), inject GPU "
                 "spans, other = host - those | tokens per round, host ms per token | gap\n");
  }
  for (const auto& kv : rounds_) {
    std::vector<double> host, gap, dr, ve, vt, in, other, com;
    for (const RoundRow& r : kv.second) {
      host.push_back(r.host_ms);
      if (r.gap_ms >= 0) gap.push_back(r.gap_ms);
      dr.push_back(r.drafter_ms);
      ve.push_back(r.verify_ms);
      vt.push_back(static_cast<double>(r.verify_T));
      in.push_back(r.inject_ms);
      other.push_back(r.host_ms - r.drafter_ms - r.verify_ms - r.inject_ms);
      com.push_back(static_cast<double>(r.committed));
    }
    const double tok = Mean(com);
    std::fprintf(stderr,
                 "[probe]   %-13s k=%-3lld n=%-5zu %7.3f | %6.3f %7.3f (%.2f) %6.3f %6.3f | "
                 "%.3f %7.3f | %.3f\n",
                 kv.first.first.c_str(), static_cast<long long>(kv.first.second), host.size(),
                 Mean(host), Mean(dr), Mean(ve), Mean(vt), Mean(in), Mean(other), tok,
                 tok > 0 ? (Mean(host) + Mean(gap)) / tok : 0.0, Mean(gap));
  }
  // Linear spans, grouped by call kind and T.
  std::map<Key, std::vector<std::pair<std::string, const SpanAgg*>>> by_call;
  for (const auto& kv : span_agg_) {
    by_call[{std::get<0>(kv.first), std::get<1>(kv.first)}].push_back(
        {std::get<2>(kv.first), &kv.second});
  }
  for (const auto& kv : by_call) {
    const auto ca = call_agg_.find(kv.first);
    const size_t calls = ca == call_agg_.end() ? 0 : ca->second.gpu_ms.size();
    if (calls == 0) continue;
    const double per = 1.0 / static_cast<double>(calls);
    std::fprintf(stderr,
                 "[probe] spans in %s T=%lld (%zu calls): per call, count and ms; per span, us "
                 "mean (median), in = cast / input transform, gemm = the rest; ms per call by "
                 "layer half\n",
                 kv.first.first.c_str(), static_cast<long long>(kv.first.second), calls);
    double sum = 0.0;
    for (const auto& e : kv.second) {
      const SpanAgg& a = *e.second;
      double lo = 0.0, hi = 0.0;
      int64_t max_layer = 0;
      for (const auto& l : a.by_layer) max_layer = std::max(max_layer, l.first);
      for (const auto& l : a.by_layer) (l.first * 2 <= max_layer ? lo : hi) += l.second.second;
      const double n = static_cast<double>(a.n);
      std::fprintf(stderr,
                   "[probe]   %-26s %5.1f x %8.2f (%8.2f) us  in %6.2f  gemm %8.2f | %7.3f ms  "
                   "(%6.3f + %6.3f)\n",
                   ClassOf(e.first), n * per, 1000.0 * a.total_ms / n, Median(a.total_us),
                   a.with_input ? 1000.0 * a.in_ms / static_cast<double>(a.with_input) : 0.0,
                   1000.0 * a.gemm_ms / n, a.total_ms * per, lo * per, hi * per);
      sum += a.total_ms * per;
    }
    std::fprintf(stderr, "[probe]   %-26s %7.3f ms of %7.3f ms mean GPU span per call\n", "sum",
                 sum, Mean(ca->second.gpu_ms));
  }
  if (dropped_ || abandoned_ || overflows_ || unfinished) {
    std::fprintf(stderr,
                 "[probe] not reported: %lld dropped (overwritten), %lld abandoned, %lld ring "
                 "overflows, %lld unfinished at exit\n",
                 static_cast<long long>(dropped_), static_cast<long long>(abandoned_),
                 static_cast<long long>(overflows_), static_cast<long long>(unfinished));
  }
}

void DebugProbe::WriteJson(const std::string& path) const {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "[probe] R4DX_PROBE_JSON: cannot write %s\n", path.c_str());
    return;
  }
  const auto spread = [&](const std::vector<double>& v) {
    std::fprintf(f,
                 "{\"n\": %zu, \"mean\": %.6g, \"median\": %.6g, \"p10\": %.6g, \"p25\": %.6g, "
                 "\"p75\": %.6g, \"p90\": %.6g, \"min\": %.6g, \"max\": %.6g}",
                 v.size(), Mean(v), Median(v), Quantile(v, 0.1), Quantile(v, 0.25),
                 Quantile(v, 0.75), Quantile(v, 0.9), Quantile(v, 0.0), Quantile(v, 1.0));
  };
  std::fprintf(f,
               "{\"format\": \"r4dx-debug-probe\", \"version\": 1,\n \"clock_probe\": %s, "
               "\"profile_linears\": \"%s\",\n \"calibration\": {\"wall_clock_mhz\": %.6f, "
               "\"wall_clock_mhz_attribute\": %.3f, \"empty_bracket_us\": %.4f, "
               "\"probe_iters\": %d, \"idle_probe_us\": %.4f, \"idle_probe_mhz\": %.3f, "
               "\"idle_cycles_per_add_1e6\": %.5f},\n",
               clock_ ? "true" : "false",
               linears_mode_ == 2 ? "all" : linears_mode_ == 1 ? "gemm" : "off", wall_mhz_,
               wall_mhz_attr_, bracket_ms_ * 1000.0, probe_iters_, idle_probe_us_, idle_probe_mhz_,
               idle_cycles_per_add_);
  std::fprintf(f, " \"calls\": [");
  bool first = true;
  for (const auto& kv : call_agg_) {
    std::fprintf(f, "%s\n  {\"kind\": \"%s\", \"T\": %lld, \"gpu_ms\": ", first ? "" : ",",
                 kv.first.first.c_str(), static_cast<long long>(kv.first.second));
    spread(kv.second.gpu_ms);
    std::fprintf(f, ", \"host_ms\": ");
    spread(kv.second.host_ms);
    std::fprintf(f, ", \"gap_ms\": ");
    spread(kv.second.gap_ms);
    std::fprintf(f, "}");
    first = false;
  }
  std::fprintf(f, "],\n \"clock\": [");
  first = true;
  for (const auto& kv : clock_agg_) {
    std::vector<double> mhz, us, cpa;
    std::map<int64_t, std::vector<double>> by_layer;
    std::map<uint64_t, std::vector<double>> by_call;
    for (const ClockSample& c : kv.second) {
      mhz.push_back(c.mhz);
      us.push_back(c.us);
      cpa.push_back(c.cycles_per_add);
      by_layer[c.layer].push_back(c.mhz);
      by_call[c.serial].push_back(c.mhz);
    }
    std::fprintf(f, "%s\n  {\"kind\": \"%s\", \"T\": %lld, \"mhz\": ", first ? "" : ",",
                 kv.first.first.c_str(), static_cast<long long>(kv.first.second));
    spread(mhz);
    std::fprintf(f, ", \"probe_us\": ");
    spread(us);
    std::fprintf(f, ", \"cycles_per_add\": ");
    spread(cpa);
    std::fprintf(f, ",\n   \"median_mhz_by_layer\": [");
    bool fl = true;
    for (const auto& l : by_layer) {
      std::fprintf(f, "%s[%lld, %.1f]", fl ? "" : ", ", static_cast<long long>(l.first),
                   Median(l.second));
      fl = false;
    }
    std::fprintf(f, "],\n   \"median_mhz_per_call\": [");
    fl = true;
    for (const auto& c : by_call) {
      std::fprintf(f, "%s%.1f", fl ? "" : ", ", Median(c.second));
      fl = false;
    }
    std::fprintf(f, "]}");
    first = false;
  }
  std::fprintf(f, "],\n \"rounds\": [");
  first = true;
  for (const auto& kv : rounds_) {
    std::fprintf(f,
                 "%s\n  {\"kind\": \"%s\", \"k\": %lld, \"columns\": [\"host_ms\", \"gap_ms\", "
                 "\"drafter_ms\", \"verify_ms\", \"verify_T\", \"inject_ms\", \"committed\"],\n   "
                 "\"rows\": [",
                 first ? "" : ",", kv.first.first.c_str(), static_cast<long long>(kv.first.second));
    bool fr = true;
    for (const RoundRow& r : kv.second) {
      std::fprintf(f, "%s[%.4f, %.4f, %.4f, %.4f, %lld, %.4f, %lld]", fr ? "" : ", ", r.host_ms,
                   r.gap_ms, r.drafter_ms, r.verify_ms, static_cast<long long>(r.verify_T),
                   r.inject_ms, static_cast<long long>(r.committed));
      fr = false;
    }
    std::fprintf(f, "]}");
    first = false;
  }
  std::fprintf(f, "],\n \"spans\": [");
  first = true;
  for (const auto& kv : span_agg_) {
    const SpanAgg& a = kv.second;
    const auto ca = call_agg_.find({std::get<0>(kv.first), std::get<1>(kv.first)});
    std::fprintf(f,
                 "%s\n  {\"kind\": \"%s\", \"T\": %lld, \"name\": \"%s\", \"calls_of_kind\": %zu, "
                 "\"n\": %lld, \"total_ms\": %.6f, \"with_input\": %lld, \"in_ms\": %.6f, "
                 "\"gemm_ms\": %.6f, \"us\": ",
                 first ? "" : ",", std::get<0>(kv.first).c_str(),
                 static_cast<long long>(std::get<1>(kv.first)), std::get<2>(kv.first).c_str(),
                 ca == call_agg_.end() ? size_t{0} : ca->second.gpu_ms.size(),
                 static_cast<long long>(a.n), a.total_ms, static_cast<long long>(a.with_input),
                 a.in_ms, a.gemm_ms);
    spread(a.total_us);
    std::fprintf(f, ",\n   \"by_layer\": [");
    bool fl = true;
    for (const auto& l : a.by_layer) {
      std::fprintf(f, "%s[%lld, %lld, %.6f]", fl ? "" : ", ", static_cast<long long>(l.first),
                   static_cast<long long>(l.second.first), l.second.second);
      fl = false;
    }
    std::fprintf(f, "]}");
    first = false;
  }
  std::fprintf(f,
               "],\n \"not_reported\": {\"dropped\": %lld, \"abandoned\": %lld, "
               "\"overflows\": %lld, \"unfinished\": %zu}}\n",
               static_cast<long long>(dropped_), static_cast<long long>(abandoned_),
               static_cast<long long>(overflows_),
               closed_.size() + stack_.size() + spans_.size() + probes_.size());
  std::fclose(f);
  std::fprintf(stderr, "[probe] R4DX_PROBE_JSON: wrote %s\n", path.c_str());
}

}  // namespace r4dx::model
