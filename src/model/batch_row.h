// r4dx::model::BatchDecodeRow -- one row of a batched decode step (Model::DecodeBatch / TextModel::DecodeBatch,
// docs/batch-decode.md). Its own header, HIP-free, so the server's batch executor (src/server/batch_executor.h) and its CPU tests can
// name it without pulling in model_types.h's device buffers.
#pragma once

#include <cstdint>
#include <random>

#include "r4dx/kernels/sampler.hpp"  // kernels::SampleParams (header-only, HIP-free)

namespace r4dx::model {

// One row of a batched decode step (Model::DecodeBatch / TextModel::DecodeBatch, docs/batch-decode.md): feed `token` into
// batch slot `slot` and get the sequence's next token back. `params.temperature <= 0` is greedy and consumes no draw;
// otherwise `rng` (required) supplies exactly one draw, the same contract as DecodeStepSampled.
struct BatchDecodeRow {
  int slot = 0;
  int32_t token = 0;
  kernels::SampleParams params;
  std::mt19937_64* rng = nullptr;
};

}  // namespace r4dx::model
