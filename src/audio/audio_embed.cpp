#include "audio_embed.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace r4dx::audio {

AudioEmbedder::AudioEmbedder(const std::vector<uint16_t>& weight_bf16, int in_dim, int out_dim, float eps)
    : in_(in_dim), out_(out_dim), eps_(eps) {
  if (in_ < 1 || out_ < 1 || weight_bf16.size() != static_cast<size_t>(in_) * static_cast<size_t>(out_)) {
    throw std::runtime_error("AudioEmbedder: weight has " + std::to_string(weight_bf16.size()) +
                             " elements, expected " + std::to_string(static_cast<int64_t>(in_) * out_));
  }
  w_.resize(weight_bf16.size());
  for (size_t i = 0; i < w_.size(); ++i) w_[i] = Bf16ToF32(weight_bf16[i]);
}

std::vector<uint16_t> AudioEmbedder::Embed(const float* frames, int64_t n, int threads) const {
  if (n < 0) throw std::runtime_error("AudioEmbedder::Embed: negative row count");
  std::vector<uint16_t> out(static_cast<size_t>(n) * static_cast<size_t>(out_));
  if (n == 0) return out;
  if (threads <= 0) threads = static_cast<int>(std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), 8u));
  threads = static_cast<int>(std::min<int64_t>(threads, n));

  const auto rows = [&](int64_t r0, int64_t r1) {
    std::vector<float> x(static_cast<size_t>(in_));
    for (int64_t r = r0; r < r1; ++r) {
      const float* f = frames + r * in_;
      // bf16 cast of the waveform first, then RMSNorm in fp32 with torch's pow(mean + eps, -0.5), bf16 result.
      float sumsq = 0.0f;
      for (int k = 0; k < in_; ++k) {
        x[static_cast<size_t>(k)] = Bf16ToF32(F32ToBf16(f[k]));
        sumsq += x[static_cast<size_t>(k)] * x[static_cast<size_t>(k)];
      }
      const float scale = std::pow(sumsq / static_cast<float>(in_) + eps_, -0.5f);
      for (int k = 0; k < in_; ++k) x[static_cast<size_t>(k)] = Bf16ToF32(F32ToBf16(x[static_cast<size_t>(k)] * scale));
      // Linear, no bias: fp32 accumulate, one bf16 rounding.
      uint16_t* o = out.data() + static_cast<size_t>(r) * static_cast<size_t>(out_);
      for (int j = 0; j < out_; ++j) {
        const float* w = w_.data() + static_cast<size_t>(j) * static_cast<size_t>(in_);
        // 8 independent partial sums (auto-vectorizes, and keeps the fp32 summation error near a pairwise
        // sum's, like the BLAS the reference runs on); deterministic: the order is fixed by in_ alone.
        float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int k = 0;
        for (; k + 8 <= in_; k += 8) {
          for (int l = 0; l < 8; ++l) lane[l] += x[static_cast<size_t>(k + l)] * w[k + l];
        }
        float tail = 0.0f;
        for (; k < in_; ++k) tail += x[static_cast<size_t>(k)] * w[k];
        const float acc = (((lane[0] + lane[4]) + (lane[1] + lane[5])) + ((lane[2] + lane[6]) + (lane[3] + lane[7]))) + tail;
        o[j] = F32ToBf16(acc);
      }
    }
  };

  if (threads <= 1) {
    rows(0, n);
  } else {
    std::vector<std::thread> pool;
    const int64_t per = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
      const int64_t r0 = t * per, r1 = std::min<int64_t>(n, r0 + per);
      if (r0 < r1) pool.emplace_back(rows, r0, r1);
    }
    for (auto& th : pool) th.join();
  }
  return out;
}

}  // namespace r4dx::audio
