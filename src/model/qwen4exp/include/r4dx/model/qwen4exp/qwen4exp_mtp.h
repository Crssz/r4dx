#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "quant_linear.h"
#include "qwen4exp_config.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model::qwen4exp {

class Qwen4ExpMtp {
 public:
  static std::unique_ptr<Qwen4ExpMtp> Load(const std::string& path,
                                           const Qwen4ExpConfig& cfg,
                                           int device,
                                           hipStream_t stream);

  ~Qwen4ExpMtp();

  // Drafts up to `k` candidate tokens using the MTP neural head.
  // Returns drafted candidate token IDs.
  std::vector<int32_t> Draft(int32_t token_id,
                             const uint16_t* h_seed,
                             const uint16_t* embed_table,
                             const QuantLinear& lm_head,
                             int64_t k,
                             int64_t current_pos);

 private:
  Qwen4ExpMtp() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace r4dx::model::qwen4exp
