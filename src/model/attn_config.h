// r4dx::model::MakeAttnConfig -- the one place a ModelConfig is turned into the
// attention::AttnConfig every full-attention call site needs.
//
// Five call sites used to spell the same seven assignments out by hand (Model::RunChunk,
// Model::VerifyWindow, Model::DecodeStepProfiled, Model::PrefillProfiled and MtpHead's
// constructor). That was survivable while the field list was fixed; the vision milestone added
// mrope_section_{t,h,w} to it (docs/vision.md "Text-side splicing"), and a site that forgot those
// would not fail to compile -- it would silently rope image tokens with the DEFAULT section split,
// i.e. produce plausible-looking garbage on exactly the path that is hardest to notice. One
// function makes that impossible.
#pragma once

#include "model_config.h"
#include "r4dx/model/attention/types.hpp"

namespace r4dx::model {

// `comm` (docs/tp.md 6.2): the tensor-parallel communicator the layer all-reduces its o_proj output
// through, or nullptr (TP=1, and MtpHead::PrimeKv's K/V-only layer under TP).
inline attention::AttnConfig MakeAttnConfig(const ModelConfig& cfg, core::TpComm* comm = nullptr) {
  attention::AttnConfig acfg;
  acfg.comm = comm;
  acfg.hidden = static_cast<int>(cfg.hidden_size);
  acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
  acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
  acfg.head_dim = static_cast<int>(cfg.head_dim);
  acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
  acfg.rope_theta = static_cast<float>(cfg.rope_theta);
  acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
  cfg.MropeSections(&acfg.mrope_section_t, &acfg.mrope_section_h, &acfg.mrope_section_w);
  return acfg;
}

}  // namespace r4dx::model
