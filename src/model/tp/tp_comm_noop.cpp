#include "tp/tp_comm_noop.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace r4dx::model::tp {

namespace {

// docs/tp.md 6.3.2: the two all-reduce channels' byte bounds. NoopComm moves no bytes, but it keeps
// the same per-channel accounting and argument checks as the real transport, so a caller that
// passes a size the real one would refuse fails here first.
constexpr size_t kChannel0MaxBytes = 174080;
constexpr size_t kMaxAllReduceBytes = 655360;

class NoopComm final : public core::TpComm {
 public:
  NoopComm(int world, int rank) : world_(world), rank_(rank) {}

  int Rank() const override { return rank_; }
  int World() const override { return world_; }

  void AllReduceSumBf16(uint16_t* buf, int64_t n, hipStream_t /*stream*/) override {
    const int64_t bytes = n * 2;
    if (buf == nullptr || n <= 0 || bytes % 16 != 0 ||
        static_cast<size_t>(bytes) > kMaxAllReduceBytes) {
      throw std::invalid_argument("tp::NoopComm::AllReduceSumBf16: n = " + std::to_string(n) +
                                  " bf16 elements (need a non-null buffer and 0 < 2n <= " +
                                  std::to_string(kMaxAllReduceBytes) + ", 2n % 16 == 0)");
    }
    const int ch = static_cast<size_t>(bytes) <= kChannel0MaxBytes ? 0 : 1;
    ++stats_.ar_calls[ch];
    stats_.ar_bytes[ch] += static_cast<uint64_t>(bytes);
    ++calls_[static_cast<size_t>(ch)];
  }

  void HostAllGather(const void* mine, size_t bytes, void* out) override {
    auto* dst = static_cast<uint8_t*>(out);
    for (int r = 0; r < world_; ++r) {
      std::memmove(dst + static_cast<size_t>(r) * bytes, mine, bytes);
    }
    ++stats_.host_exchanges;
  }

  void CheckLockstep(const uint64_t* /*fingerprint*/) override {}
  void CheckHealthy() override {}

  void Abort(uint32_t /*code*/, const std::string& /*why*/) noexcept override {
    aborted_ = true;
    ++stats_.aborts;
  }
  bool Aborted() const noexcept override { return aborted_; }

  size_t MaxAllReduceBytes() const override { return kMaxAllReduceBytes; }
  void SelfTest() override {}
  core::TpCommStats Stats() const override { return stats_; }
  void SetAllReduceTimeoutMs(int /*ms*/) override {}  // nothing ever spins
  void ResetCounters() override { calls_ = {0, 0}; }
  std::array<uint64_t, 2> CallCounts() const override { return calls_; }

 private:
  int world_, rank_;
  bool aborted_ = false;
  core::TpCommStats stats_;
  std::array<uint64_t, 2> calls_ = {0, 0};
};

}  // namespace

std::unique_ptr<core::TpComm> MakeNoopComm(int world, int rank) {
  if (world < 1 || rank < 0 || rank >= world) {
    throw std::invalid_argument("tp::MakeNoopComm: need world >= 1 and 0 <= rank < world, got world " +
                                std::to_string(world) + ", rank " + std::to_string(rank));
  }
  return std::make_unique<NoopComm>(world, rank);
}

}  // namespace r4dx::model::tp
