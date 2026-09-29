// r4dx-convert -- HF checkpoint (safetensors + config.json) -> r4dx container
// (docs/container-format.md). See docs/build-windows.md for the toolchain and README.md for the
// project. Usage:
//
//   r4dx-convert --input <HF checkpoint dir> --output <container path>
//                [--layouts w4a16] [--lm-head 4bit+bf16]
//                [--layers N] [--threads T] [--vision on|off] [--mtp on|off] [--no-bf16]
//                [--kv-calib <tools/reference/kv_calibrate.py JSON>]
//                [--draft-vocab-ids <tests/model/tool_vocab_calib.cpp output JSON>]
//                [--quant {rtn,search}] [--imatrix <tools/reference/imatrix_capture.py .npz>]
//                [--keep-bf16 <ECMAScript regex over container base names>]
//                [--hessian-dir <tools/reference/hessian_capture.py output dir>
//                 --ldlq <ECMAScript regex over container base names> [--ldlq-damp 0.01]]
//                [--rotate {none,q2a,q2ab}] [--rotation-seed <u64, decimal or 0x hex>]
//                [--w4a16-group-rule "<ECMAScript regex over container base names>=<32|64|128>"]...
//                [--record-reuse-guard] [--reuse-tensors-from <baseline container>]
//                [--trellis-from <oracle dir or its weights_override.json>
//                 [--trellis-manifest-sha256 <hex>] [--trellis-verify full|none]
//                 [--trellis-allow-basis exl3] [--trellis-prescale-log2 <int>]]
//
// --trellis-from (docs/trellis-kernel.md sections 2-3, trellis_import.hpp) IMPORTS the EXL3/QTIP
// trellis bits tools/reference/trellis_quant.py wrote (a quantize-model directory or a mix
// manifest) for every text-layer body linear -- attn.qg/k/v/o, gdn.in_proj_qkv/z/out_proj,
// mlp.gate_up/down -- as `<base>.trellis.w` (the oracle's words, whole tiles moved into the pair
// grid, nothing re-encoded), `.trellis.suh` and `.trellis.svh`, plus __metadata__.quant.trellis.
// Those linears get the trellis layout ONLY, whatever --layouts says (no `.bf16.w` next to them:
// its absence is what makes an old binary refuse the container); --layouts / --lm-head / --ldlq /
// --w4a16-group-rule keep governing every other linear (lm_head, mtp.*, the draft head), exactly
// as without the flag. --keep-bf16 still wins for a base it matches (its manifest entries are
// skipped). Refused, before anything is written: a manifest whose format / version / encoding /
// codebook is not the oracle's, complete false or missing_count > 0, a used layer in
// stale_layers, hessian_basis other than "matched" (unless --trellis-allow-basis exl3),
// config_sha256 other than this checkpoint's, a used tensor with K not in {4, 5}, a shape other
// than the checkpoint's, gate K != up K, any body linear not (fully) covered, an oracle file whose
// sha256 differs from its record, --rotate other than none, and the combination with --selftest /
// --dflash-gguf / --reuse-tensors-from / --record-reuse-guard.
// The container is written as <output>.partial and gets its name only once it is complete and
// checked: --trellis-verify full (the default; `none` is accepted by debug builds only)
// reconstructs every imported tensor from the CLOSED file's bytes on the CPU and requires
// |rel - rel_weight_err| <= 0.02 rel_weight_err + 1e-4 (the spec's tolerance) and
// |rel - rel_weight_err| / rel_weight_err <= 1e-4 (trellis_import.hpp kStrictRelDev) against the
// bf16 checkpoint. The result is patched into __metadata__.r4dx_convert_run.trellis.verify (result,
// worst, worst_tensor, checked, failed); a failure renames the file to <output>.verify-failed, and
// a conversion that dies leaves only the .partial file -- so <output> itself only ever holds a
// checked container.
//
// --reuse-tensors-from <baseline> (docs/quant2.md 5.2, reuse_guard.hpp) writes exactly the container
// a full run with the same flags would write, but copies every tensor whose bytes cannot depend on
// the difference between this run and the baseline's verbatim from the baseline instead of computing
// it. The ONLY differences it allows are --w4a16-group-rule and --keep-bf16 (plus --output, --threads
// and this flag), and it does not compare either flag's text: both runs record every linear's
// RESOLVED layout set (reuse_guard.linears: "w4a16.g64", "bf16", ...), and the linears whose set
// differs -- a group either run's rules moved, a linear either run keeps in bf16 and the other
// quantizes -- are recomputed, every layout of them, and everything else is copied (each linear is
// quantized on its own, so nothing else can depend on another linear's group or precision; every
// header field a keep or a rule changes -- the group map, keep_bf16_linears, ldlq_linears, the byte
// totals -- comes from the planning pass, which a reuse run executes in full). The baseline must carry
// __metadata__.r4dx_convert_run.reuse_guard, which a run given
// --record-reuse-guard (or --reuse-tensors-from, which implies it) writes: the sha256 of this
// converter's own executable and runtime DLLs, the CPU's identity, the sha256 of config.json,
// model.safetensors.index.json and every shard, of the --imatrix / --kv-calib / --draft-vocab-ids
// files, of hessian.json and of every .hess file the --ldlq regex can read (whichever linears are
// kept: the set follows --ldlq alone), and every other flag, resolved. Any difference -- a rebuilt
// binary included; there is deliberately no override --
// refuses the reuse before the first shard is read, naming each field. So does a baseline whose
// emit pass never finished (the guard's emit_complete marker, patched from 0 to 1 only after every
// tensor is durably on disk), one whose tensor data no longer hashes to the guard's data_sha256
// (patched in with the marker, from every tensor read back from the file), and a baseline whose
// tensor directory (or group map, or keep list) does not match its own per-linear record and what
// this run writes. The output records reused_from (the
// baseline's path, header sha256 and data_sha256, the copied / recomputed counts and the recomputed
// linears) next to its own reuse_guard; apart from reused_from (and `threads`, when it differs) its
// header and data are byte-identical to a full run's with --record-reuse-guard.
//
// --w4a16-group-rule (docs/quant2.md section 5, w4a16_groups.hpp; repeatable, first matching rule
// wins, unmatched linears keep the build default R4DX_W4A16_GROUP) packs the w4a16 layout of every
// linear whose container base name the regex matches (regex_SEARCH, like --keep-bf16) at that group
// instead: its scale tensor becomes `<base>.w4a16.wsz.g<g>` and __metadata__.quant.w4a16.groups
// lists it, so the runtime dispatches r4d_gemm_w4a16_nt_m64_g for it and a binary that predates
// per-tensor groups refuses the container instead of reading the scales at the default stride.
// That refusal only happens if the linear has no `.bf16.w` to fall back to, so a non-default group
// on a linear that keeps bf16 is a planning error (use --no-bf16, and an --lm-head spec without
// bf16). K must be a multiple of the group and of 64, N of 16 -- checked while planning. Without
// the flag every container is byte-identical to one built before it existed.
// __metadata__.r4dx_convert_run records the rules, the resolved map and the .wsz byte delta.
//
// --keep-bf16 (docs/validation.md "Milestone 11 / sensitivity", keep_bf16.hpp) writes every linear
// whose container base name the regex matches as `<base>.bf16.w` ONLY, skipping every quantized
// layout the run asked for; src/model/container.cpp's LoadQuantLinearWithFallback then loads those
// -- and only those -- as bf16. It is the per-tensor-class sensitivity instrument: convert one
// class in bf16, re-run the KL harness, and the KL that disappears is that class's share.
// e.g. --keep-bf16 "attn\.o$", --keep-bf16 "^text\.layers\.[0-7]\.". Matching is regex_SEARCH, so
// anchor when a substring would over-select. A regex that matches nothing warns and converts
// normally (a scripted sweep must not die mid-batch on a class this checkpoint has none of).
//
// --quant selects HOW the 4-bit values are chosen; it does NOT change a single byte of the on-disk
// layout (docs/container-format.md, "How the quantized values are chosen"). `rtn` (the DEFAULT) is
// the historical min/max grid + round-to-nearest and reproduces any pre-existing container byte for
// byte; `search` minimizes the squared reconstruction error per (row, 128-K group) over a small
// candidate grid (src/convert/include/r4dx_convert/quant_search.hpp). --imatrix additionally weights
// that error by each input channel's mean activation energy, from the .npz
// tools/reference/imatrix_capture.py writes (keyed by these same container base names); it requires
// --quant search, and a linear with no entry in the file falls back to unweighted MSE with a
// warning plus an end-of-run coverage line.
//
// `rtn` stays the default deliberately: `--quant search` WITHOUT --imatrix was measured on the real
// checkpoint at mean KL 0.0713 / top-1 87.71% against rtn's 0.0724 / 88.4% -- i.e. no better, and
// marginally worse on top-1 -- while costing ~2x the conversion wall time, and --imatrix cannot be
// a default because it needs a capture file. The pair that IS worth using is
// `--quant search --imatrix <npz>` (w4a16 0.0534 / 89.30%, docs/validation.md "Milestone 10"), and
// it has to be asked for explicitly so that a plain `r4dx-convert --input ... --output ...` keeps
// reproducing every container built before these flags existed.
//
// --ldlq <regex> (docs/quant2.md section 2, quant_ldlq.hpp) is a third way of choosing the values,
// again with the byte layout untouched: every linear whose container base name the regex matches
// (regex_SEARCH, ECMAScript, exactly like --keep-bf16; ".*" = every linear) is rounded column by
// column with GPTQ/LDLQ error feedback against its input Hessian H = E[x x^T], read from the
// directory tools/reference/hessian_capture.py writes (--hessian-dir, required; its hessian.json maps
// these same base names to .hess files, several bases sharing one file where they share an input).
// Each group's (scale, zero) comes from the `search` grid weighted by diag(H), on the error-updated
// weights, without the final refit. For those linears --quant/--imatrix are ignored; every other
// linear follows --quant/--imatrix exactly as without the flag. --ldlq-damp (default 0.01) is the
// relative Tikhonov damping, H += damp * mean(diag H) * I, retried at 10x and 100x on a failed
// Cholesky. Precedence and failure modes, all deliberate:
//   - --keep-bf16 wins over --ldlq for a linear both match (it is not quantized at all);
//   - a matching linear with no manifest key, or whose manifest K differs from the checkpoint's, is
//     a hard error during PLANNING -- before the header is written or a single shard read -- not a
//     silent fallback: a partially-LDLQ'd container measured as "LDLQ" would poison the KL table;
//   - a VALID regex that matches nothing warns (stderr) and converts normally, like --keep-bf16;
//   - --ldlq without --hessian-dir, or with --dflash-gguf (no Hessians exist for the drafter), is an
//     argument error.
// __metadata__.r4dx_convert_run records the pattern, the damp, the directory, hessian.json's sha256
// and the resolved list of LDLQ'd bases; each LDLQ'd linear logs its factor/emit seconds, damp_used
// and retry count, and the run ends with a one-line total.
//
// --rotate (docs/quant2.md sections 3-4, rotation.hpp; default `none`, which leaves every byte of
// the container exactly as without the flag) folds an orthogonal residual-stream rotation Q into
// the 64-layer text stack, in fp32 before any quantization: `q2a` stores both layer norms of every
// text layer as 0 and folds W diag(1 + w_norm) Q into every in-projection (attn.qg/k/v,
// gdn.in_proj_qkv/z/a/b, mlp.gate_up -- the bf16-only in_proj_a/b and any --keep-bf16 linear are
// rounded to bf16 once, from the folded fp32) and Q^T W into every out-projection (attn.o,
// gdn.out_proj, mlp.down); `q2ab` additionally folds a block Hadamard W Hb into the K side of those
// three out-projections (blocks 256 / 128 / 512). The embedding table, final_norm, lm_head, vision.*
// and mtp.* are untouched: the runtime applies x Q at the stack's entry and x Q^T at its exit, and
// Hb on the three inputs, from the tensors this writes (rotation.signs, rotation.mix5 and, for q2ab,
// rotation.had_{down,o,gdn_out}_signs, generated from --rotation-seed, default 0x5EED2025) and the
// __metadata__.rotation block that tells it to (docs/container-format.md "Residual rotation"). It
// composes with every way of choosing values: --ldlq rounds a rotated linear against its Hessian
// carried into the new basis (Hb^T H Hb for the q2ab K side; for an in-projection Q^T H_rms Q when
// hessian.json's "rms_keys" has the weightless rms(x) Hessian of its norm's input -- the folded
// input is exactly rms(x) Q -- and otherwise Q^T D^-1 H D^-1 Q from the post-norm capture, which is
// refused during planning if any |1 + w| < 1e-3); --imatrix vectors are carried over under their
// own diagonal model (rotation.hpp's header says exactly what that is and is not).
// r4dx_convert_run.ldlq_rms_linears lists the in-projections that used H_rms. --layers N folds all
// N converted layers. Not accepted with --selftest or --dflash-gguf.
//
// --draft-vocab-ids (docs/r9700.md R9, "reduced-vocab draft head"): bakes an OPTIONAL smaller
// lm_head (mtp.draft_head.lm_head.{layout} + mtp.draft_head.vocab_ids, docs/container-format.md
// "mtp.*") into the container for MtpHead::Draft's own drafting -- ignored unless --mtp is also on;
// a container built without this flag simply has no draft_head.* tensors (old-container-compatible,
// loader falls back to the full-vocab head automatically, src/model/container.cpp).
//
// --kv-calib fills text.layers.{i}.attn.k_descale/.v_descale (full-attention layers only) from a
// tools/reference/kv_calibrate.py merged JSON: descale[head] = k_amax|v_amax[head] / 448.0 (see
// r4dx_convert::ResolveKvDescale, kv_calib.hpp). A layer absent from the JSON (or with a
// missing/wrong-length amax array) falls back to descale=1.0 with a WARNING on stderr, not a hard
// error -- omitting --kv-calib entirely also yields the 1.0 placeholder, silently (no calibration
// was ever requested, so there is nothing to warn about).
//
//   r4dx-convert --selftest --selftest-input <small .safetensors, one 2D bf16 tensor "w">
//                --selftest-output <container path> [--layouts w4a16] [--threads T]
//                [--no-bf16] [--w4a16-group-rule "<regex>=<g>"]...
//
// --selftest packs exactly one tensor through every requested layout and writes it as
// "selftest.{layout}.*" -- tools/convert_ref/selftest_compare.py runs the matching Python
// reference on the same input and diffs the two containers byte for byte. Its container base is
// "selftest", which --keep-bf16 / --ldlq / --w4a16-group-rule match against like any other.
//
//   r4dx-convert --dflash-gguf <DFlash2 draft .gguf> --out <container path>
//                [--layout {w4a16,bf16}] [--threads T]
//
// Converts a DFlash2 speculative-decoding draft model (docs/container-format.md "DFlash2 draft
// container") from its GGUF v3 source into its own r4dx container (container_kind
// "dflash2_draft", separate file from the main text-model container). Produces ONE container per
// invocation carrying exactly the requested `--layout` (bf16 is only included when
// `--layout bf16` is requested; unlike the main HF-checkpoint mode, there is no automatic bf16
// side-by-side -- see DflashLayoutSet).
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4d.h"  // r4d_gemm_w4a16_nt_m64_group() -- cross-checked against this converter's own
                  // kW4A16Group at startup (ValidateKernelGroupSizes).
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/dflash2_container.hpp"
#include "r4dx_convert/gguf_reader.hpp"
#include "r4dx_convert/hessian_store.hpp"
#include "r4dx_convert/keep_bf16.hpp"
#include "r4dx_convert/kv_calib.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/npz_reader.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/reuse_guard.hpp"
#include "r4dx_convert/rotation.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/tensor_codec.hpp"
#include "r4dx_convert/trellis_import.hpp"
#include "r4dx_convert/w4a16_groups.hpp"

namespace {

using r4dx_convert::ContainerWriter;
using r4dx_convert::LayoutSet;
using r4dx_convert::ShardedModel;

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

size_t Count(const std::string& hay, const std::string& needle) {
  size_t n = 0;
  for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + 1)) ++n;
  return n;
}

// Review finding (major, quant_int4.hpp): the group size this converter packs with
// (r4dx_convert::kW4A16Group) is a compile-time constant that must match the group size the GPU
// kernel was actually built with (w4a16's R4D_GEMM_W4_GROUP is the R4DX_W4A16_GROUP build option).
// A silent mismatch here produces a container that GEMMs read at the wrong stride with no error,
// ever. r4d_core exports the group the kernel was actually compiled with, so assert against it once
// at startup instead of trusting the -D flags stayed in sync.
void ValidateKernelGroupSizes() {
  auto check = [](const char* label, int expected, int actual) {
    if (actual != expected) {
      throw std::runtime_error(
          std::string("r4dx-convert: group size mismatch for ") + label + ": this converter packs "
          "with group=" + std::to_string(expected) + " but r4d_core's kernel was built with group=" +
          std::to_string(actual) + " (check third_party/CMakeLists.txt's R4D_EXTRA_* -D flags and "
          "the R4DX_W4A16_GROUP cache variable against "
          "src/convert/include/r4dx_convert/quant_int4.hpp's kW4A16Group)");
    }
  };
  check("w4a16", r4dx_convert::kW4A16Group, r4d_gemm_w4a16_nt_m64_group());
}

// --w4a16-group-rule: every group a rule can hand out must be one this build's
// r4d_gemm_w4a16_nt_m64_g actually instantiates -- the same "ask the kernel, do not trust the
// constant" discipline as ValidateKernelGroupSizes, for the per-tensor groups.
void ValidateKernelRuleGroups(const r4dx_convert::W4a16GroupRules& rules) {
  for (const auto& r : rules.Rules()) {
    if (r4d_gemm_w4a16_nt_m64_has_group(r.group) == 0) {
      throw std::runtime_error("--w4a16-group-rule '" + r.spec + "': this build's r4d_core has no "
                               "r4d_gemm_w4a16_nt_m64 instantiation at group " +
                               std::to_string(r.group) + " (r4d_gemm_w4a16_nt_m64_has_group)");
    }
  }
}

// Builds the __metadata__["quant"] block (review finding, major): the task brief asked for
// per-tensor quant layout/group/permutation in __metadata__ and the original run only had prose
// in quant_summary. This is the authoritative, loader-checkable record -- src/model should assert
// its own kernel's group() matches these before trusting the container.
//
// `w4a16_groups` is W4a16GroupRules::GroupsJson(): the linears packed at a group other than
// `w4a16.group` (docs/quant2.md section 5). Written as `w4a16.groups` only when non-empty, so a
// container converted without --w4a16-group-rule keeps exactly the header it had before.
nlohmann::json BuildQuantMetadata(const nlohmann::json& w4a16_groups = nlohmann::json::object()) {
  nlohmann::json q = {
      {"w4a16",
       {{"group", r4dx_convert::kW4A16Group},
        {"zero_mode", "free_0_15"},
        {"nibble_encoding", "offset_binary_xor8"},
        {"fragment_permutation", "wmma16x16x16_lane16_koff_0_8_1_9_2_10_3_11"}}},
  };
  if (!w4a16_groups.empty()) q["w4a16"]["groups"] = w4a16_groups;
  return q;
}

// Parses a comma/plus-separated list of layout tokens ("w4a16", "bf16", "4bit" meaning the
// quantized layout (w4a16), and "none"/"" meaning bf16 only). The retired "mxfp4" / "w4a8" tokens
// and any unknown token are a hard error -- silently ignoring a typo'd --layouts value would produce a container missing a
// layout the caller thinks it asked for.
LayoutSet ParseLayoutList(const std::string& spec, bool bf16_default_on) {
  LayoutSet ls;
  ls.bf16 = bf16_default_on;
  std::string cur;
  std::vector<std::string> tokens;
  for (char c : spec) {
    if (c == ',' || c == '+') {
      if (!cur.empty()) tokens.push_back(cur);
      cur.clear();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  if (!cur.empty()) tokens.push_back(cur);
  for (auto& t : tokens) {
    if (t == "w4a16") ls.w4a16 = true;
    else if (t == "bf16") ls.bf16 = true;
    else if (t == "4bit") { ls.w4a16 = true; }
    else if (t == "none" || t.empty()) { /* no-op */ }
    else if (t == "mxfp4" || t == "w4a8")
      throw std::runtime_error("layout token '" + t + "' was retired (supported: w4a16, bf16, "
                               "trellis via --trellis-from)");
    else throw std::runtime_error("unknown layout token: " + t);
  }
  return ls;
}

struct AppArgs {
  bool selftest = false;
  std::string input, output;
  std::string selftest_input, selftest_output, selftest_name = "w";
  std::string layouts_spec = "w4a16";
  std::string lm_head_spec = "4bit+bf16";
  bool lm_head_spec_explicit = false;  // --lm-head was given: --no-bf16 then leaves it alone
  int layers = -1;  // -1 = every text layer
  int threads = 0;  // 0 = hardware_concurrency
  int vision = -1;  // -1 auto (on iff full run), 0 off, 1 on
  int mtp = -1;
  bool no_bf16 = false;  // drop the bf16 layout for body weights even if --layouts/--lm-head asked
  std::string kv_calib;  // path to tools/reference/kv_calibrate.py's merged JSON; empty = no calib
  // Reduced-vocab MTP draft head (docs/r9700.md R9): path to a JSON file `{"vocab_ids": [...]}`
  // (real vocabulary ids, produced by tests/model/tool_vocab_calib.cpp's calibration pass) --
  // when non-empty AND --mtp is on, slices those rows out of lm_head.weight and emits them as
  // mtp.draft_head.lm_head.{layout} + mtp.draft_head.vocab_ids (docs/container-format.md "mtp.*").
  // Empty (default): no draft_head.* tensors are written -- MtpHead::Draft falls back to the
  // full-vocab head unconditionally, byte-identical to every container built before this flag
  // existed.
  std::string draft_vocab_ids;

  // DFlash2 draft-model mode (docs/dflash2.md): --dflash-gguf <gguf> --out <container> --layout {...}.
  // Mutually exclusive with the HF-checkpoint (--input/--output) and --selftest modes. Uses its
  // own --out/--layout flag names (not --output/--layouts) since this mode packs exactly one
  // requested layout per run (no automatic bf16 companion -- pass --layout bf16 for an
  // exact-precision container), not the HF mode's "every layout side by side" model.
  std::string dflash_gguf;
  std::string dflash_out;
  std::string dflash_layout = "w4a16";  // one of w4a16, bf16

  // How the 4-bit quantizers choose their (scale, zero) values -- the on-disk BYTE LAYOUT is
  // identical either way (src/convert/include/r4dx_convert/quant_search.hpp). "rtn" (default) is
  // the historical min/max + round-to-nearest grid and reproduces every container built before this
  // flag existed byte for byte; "search" minimizes the (optionally importance-weighted) squared
  // reconstruction error per (row, group). See this file's header comment for why the default is
  // rtn and not search.
  std::string quant = "rtn";
  // Importance matrix ("imatrix") .npz from tools/reference/imatrix_capture.py, keyed by the
  // converter's own container base names. Weights the search's error term by the mean activation
  // energy of each input channel. Only meaningful with --quant search; a linear with no entry in
  // the file falls back to unweighted MSE (with a count reported at the end of the run).
  std::string imatrix;

  // ECMAScript regex over container base names; matching linears are written as `<base>.bf16.w`
  // only (r4dx_convert::KeepBf16Selector, keep_bf16.hpp). Empty (default) = every linear is
  // quantized exactly as before this flag existed, byte for byte.
  std::string keep_bf16;

  // LDLQ error-feedback rounding (docs/quant2.md section 2; this file's header comment). `ldlq` is an
  // ECMAScript regex over container base names, empty (default) = no linear is LDLQ'd and every
  // container is byte-identical to one built before these flags existed. `hessian_dir` is
  // tools/reference/hessian_capture.py's output directory (hessian.json + *.hess). `ldlq_damp` is the
  // RELATIVE damping (times mean(diag H)); 0 is allowed (the H = I gate in convert_quant_ldlq uses
  // it), negative/non-finite is rejected.
  std::string ldlq;
  std::string hessian_dir;
  float ldlq_damp = 0.01f;

  // Residual-stream rotation (docs/quant2.md sections 3-4; this file's header comment). "none"
  // (default) = no fold, no rotation tensors, no metadata key: byte-identical to a container built
  // before these flags existed.
  std::string rotate = "none";
  uint64_t rotation_seed = r4dx_convert::kDefaultRotationSeed;
  bool rotation_seed_explicit = false;

  // Per-tensor w4a16 groups (docs/quant2.md section 5; w4a16_groups.hpp): every
  // --w4a16-group-rule "<regex>=<g>" in command-line order, first match wins. Empty (default) =
  // every w4a16 linear at the build default and a byte-identical container.
  std::vector<std::string> w4a16_group_rules;

  // docs/quant2.md 5.2 (this file's header comment, reuse_guard.hpp). `reuse_from`: a baseline
  // container whose unaffected tensors are copied instead of recomputed; implies
  // `record_reuse_guard`, which writes r4dx_convert_run.reuse_guard. Both off (default) = no hashing
  // and a header byte-identical to one written before these flags existed.
  std::string reuse_from;
  bool record_reuse_guard = false;

  // --trellis-from and its options (docs/trellis-kernel.md 3.2; this file's header comment). Empty
  // `trellis.from` (default) = no trellis linear, and a container byte-identical to one built
  // before these flags existed. `trellis_options` = any --trellis-* option other than
  // --trellis-from was given (refused without it).
  r4dx_convert::trellis::TrellisOptions trellis;
  bool trellis_options = false;
};

// --rotation-seed: a full u64, decimal or 0x-prefixed hex. std::stoull alone would accept "-1"
// (wrapping to 2^64-1) and read a leading-zero value as octal, so both are ruled out explicitly.
uint64_t ParseSeed(const std::string& v) {
  const bool hex = v.size() > 2 && v[0] == '0' && (v[1] == 'x' || v[1] == 'X');
  const std::string digits = hex ? v.substr(2) : v;
  bool ok = !digits.empty();
  for (char c : digits)
    ok = ok && (hex ? std::isxdigit(static_cast<unsigned char>(c)) != 0
                    : std::isdigit(static_cast<unsigned char>(c)) != 0);
  uint64_t out = 0;
  if (ok) {
    try {
      size_t used = 0;
      out = std::stoull(digits, &used, hex ? 16 : 10);
      ok = used == digits.size();
    } catch (const std::exception&) {
      ok = false;  // out of range
    }
  }
  if (!ok)
    throw std::runtime_error("--rotation-seed must be an unsigned 64-bit integer (decimal or 0x hex), "
                             "got '" + v + "'");
  return out;
}

// Strict on/off parser for --vision/--mtp (review finding, minor): the previous `(v == "on") ? 1
// : 0` silently treated any typo ("yes", "true", "1", a fat-fingered value) as OFF, which could
// quietly drop the vision tower or MTP head from a multi-hour conversion run with no diagnostic.
int ParseOnOff(const std::string& v, const char* flag) {
  if (v == "on") return 1;
  if (v == "off") return 0;
  throw std::runtime_error(std::string(flag) + " must be 'on' or 'off', got '" + v + "'");
}

AppArgs ParseArgs(int argc, char** argv) {
  AppArgs a;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--selftest") a.selftest = true;
    else if (arg == "--input") a.input = next(i);
    else if (arg == "--output") a.output = next(i);
    else if (arg == "--selftest-input") a.selftest_input = next(i);
    else if (arg == "--selftest-output") a.selftest_output = next(i);
    else if (arg == "--selftest-name") a.selftest_name = next(i);
    else if (arg == "--layouts") a.layouts_spec = next(i);
    else if (arg == "--lm-head") { a.lm_head_spec = next(i); a.lm_head_spec_explicit = true; }
    else if (arg == "--layers") a.layers = std::stoi(next(i));
    else if (arg == "--threads") a.threads = std::stoi(next(i));
    else if (arg == "--vision") a.vision = ParseOnOff(next(i), "--vision");
    else if (arg == "--mtp") a.mtp = ParseOnOff(next(i), "--mtp");
    else if (arg == "--no-bf16") a.no_bf16 = true;
    else if (arg == "--kv-calib") a.kv_calib = next(i);
    else if (arg == "--draft-vocab-ids") a.draft_vocab_ids = next(i);
    else if (arg == "--dflash-gguf") a.dflash_gguf = next(i);
    else if (arg == "--out") a.dflash_out = next(i);
    else if (arg == "--layout") a.dflash_layout = next(i);
    else if (arg == "--quant") a.quant = next(i);
    else if (arg == "--imatrix") a.imatrix = next(i);
    else if (arg == "--keep-bf16") a.keep_bf16 = next(i);
    else if (arg == "--ldlq") a.ldlq = next(i);
    else if (arg == "--hessian-dir") a.hessian_dir = next(i);
    else if (arg == "--ldlq-damp") {
      const std::string v = next(i);
      size_t used = 0;
      try {
        a.ldlq_damp = std::stof(v, &used);
      } catch (const std::exception&) {
        used = 0;
      }
      if (used != v.size() || !std::isfinite(a.ldlq_damp) || a.ldlq_damp < 0.0f)
        throw std::runtime_error("--ldlq-damp must be a finite number >= 0, got '" + v + "'");
    }
    else if (arg == "--rotate") a.rotate = next(i);
    else if (arg == "--rotation-seed") {
      a.rotation_seed = ParseSeed(next(i));
      a.rotation_seed_explicit = true;
    }
    else if (arg == "--w4a16-group-rule") a.w4a16_group_rules.push_back(next(i));
    else if (arg == "--reuse-tensors-from") a.reuse_from = next(i);
    else if (arg == "--record-reuse-guard") a.record_reuse_guard = true;
    else if (arg == "--trellis-from") a.trellis.from = next(i);
    else if (arg == "--trellis-manifest-sha256") {
      a.trellis.manifest_sha256 = next(i);
      std::transform(a.trellis.manifest_sha256.begin(), a.trellis.manifest_sha256.end(),
                     a.trellis.manifest_sha256.begin(), [](char c) {
                       return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                     });
      if (!r4dx_convert::IsSha256Hex(a.trellis.manifest_sha256))
        throw std::runtime_error("--trellis-manifest-sha256 must be 64 hex digits, got '" +
                                 a.trellis.manifest_sha256 + "'");
      a.trellis_options = true;
    }
    else if (arg == "--trellis-verify") {
      a.trellis.verify = next(i);
      if (a.trellis.verify != "full" && a.trellis.verify != "none")
        throw std::runtime_error("--trellis-verify must be 'full' or 'none', got '" +
                                 a.trellis.verify + "'");
#if defined(NDEBUG)
      // docs/trellis-kernel.md 3.3 step 10: an unverified trellis container is a debug artefact; a
      // release converter always runs the check (it costs well under a minute on the 27B).
      if (a.trellis.verify == "none")
        throw std::runtime_error("--trellis-verify none exists for debug builds only "
                                 "(docs/trellis-kernel.md 3.3 step 10): this release r4dx-convert "
                                 "always runs the reconstruction check");
#endif
      a.trellis_options = true;
    }
    else if (arg == "--trellis-allow-basis") {
      a.trellis.allow_basis = next(i);
      if (a.trellis.allow_basis != "exl3")
        throw std::runtime_error("--trellis-allow-basis accepts only 'exl3' (the matched basis "
                                 "needs no flag), got '" + a.trellis.allow_basis + "'");
      a.trellis_options = true;
    }
    else if (arg == "--trellis-prescale-log2") {
      const std::string v = next(i);
      size_t used = 0;
      int s = 0;
      try {
        s = std::stoi(v, &used);
      } catch (const std::exception&) {
        used = 0;
      }
      // 2^s scales the f16 activations the kernel reads (docs/trellis-kernel.md 4.8): far outside
      // +-16 every activation over- or underflows f16.
      if (used != v.size() || s < -16 || s > 16)
        throw std::runtime_error("--trellis-prescale-log2 must be an integer in [-16, 16], got '" +
                                 v + "'");
      a.trellis.prescale_log2 = s;
      a.trellis_options = true;
    }
    else throw std::runtime_error("unknown argument: " + arg);
  }
  // --trellis-from (docs/trellis-kernel.md 3.3 step 6): an import of oracle bits into the Qwen
  // body. The selftest and the drafter have no body to import into; a reuse has nothing to reuse
  // (an import is I/O-bound) and its guard does not cover the oracle files (refused in v1, 3.4).
  if (a.trellis_options && a.trellis.from.empty())
    throw std::runtime_error("--trellis-manifest-sha256 / --trellis-verify / --trellis-allow-basis "
                             "/ --trellis-prescale-log2 need --trellis-from");
  if (!a.trellis.from.empty() && (a.selftest || !a.dflash_gguf.empty()))
    throw std::runtime_error("--trellis-from applies only to the HF-checkpoint conversion "
                             "(--input/--output), not to --selftest or --dflash-gguf");
  if (!a.trellis.from.empty() && (!a.reuse_from.empty() || a.record_reuse_guard))
    throw std::runtime_error("--trellis-from cannot be combined with --reuse-tensors-from / "
                             "--record-reuse-guard (docs/trellis-kernel.md 3.4: the reuse guard "
                             "does not cover the oracle files, and an import has nothing to "
                             "reuse)");
  // The guard describes an HF-checkpoint conversion (checkpoint files, its flags); the selftest and
  // the drafter have neither a sweep to share a baseline across nor a guard to check one against.
  if ((!a.reuse_from.empty() || a.record_reuse_guard) && (a.selftest || !a.dflash_gguf.empty()))
    throw std::runtime_error("--reuse-tensors-from / --record-reuse-guard apply only to the "
                             "HF-checkpoint conversion (--input/--output), not to --selftest or "
                             "--dflash-gguf");
  // Parsed here (syntax, group, regex) so a typo is an argument error, not a mid-plan one; the
  // selector itself is rebuilt by the run that uses it.
  if (!a.w4a16_group_rules.empty()) (void)r4dx_convert::W4a16GroupRules(a.w4a16_group_rules);
  // The DFlash2 drafter's loader (src/model/dflash_draft.cpp) reads every w4a16 linear at the
  // container's one default group, and its tensors share none of the Qwen base names the rules are
  // written against -- reject the combination rather than build a drafter it silently ignored.
  if (!a.w4a16_group_rules.empty() && !a.dflash_gguf.empty())
    throw std::runtime_error("--w4a16-group-rule does not apply to --dflash-gguf (the drafter is "
                             "packed at the build's default w4a16 group only)");
  const r4dx_convert::RotationKind rotate_kind = r4dx_convert::ParseRotationKind(a.rotate);
  // The rotation is a property of the Qwen text stack (Q is 5120 wide, folded into its layers and
  // undone at its exit by the runtime): the selftest's one bare tensor has no residual stream to
  // rotate and no runtime to undo it, and the DFlash2 drafter is un-rotated by design
  // (docs/quant2.md 1.2) -- a rotated drafter would be silently wrong, so both are argument errors.
  if (rotate_kind != r4dx_convert::RotationKind::kNone && (a.selftest || !a.dflash_gguf.empty()))
    throw std::runtime_error("--rotate applies only to the HF-checkpoint conversion (--input/--output), "
                             "not to --selftest or --dflash-gguf");
  // A trellis linear quantizes the UNROTATED W (its own RHT does the incoherence); fed x Q with the
  // norms removed it would be garbage, and the runtime cannot rotate per linear
  // (docs/trellis-kernel.md 3.4). The loader refuses the combination too.
  if (rotate_kind != r4dx_convert::RotationKind::kNone && !a.trellis.from.empty())
    throw std::runtime_error("--trellis-from needs --rotate none (got --rotate " + a.rotate +
                             "): the oracle quantized the unrotated weights, and a rotated "
                             "container would feed them x Q with the norms folded away");
  if (a.quant != "rtn" && a.quant != "search")
    throw std::runtime_error("--quant must be 'rtn' or 'search', got '" + a.quant + "'");
  if (!a.imatrix.empty() && a.quant != "search")
    throw std::runtime_error("--imatrix requires --quant search (got --quant " + a.quant + ")");
  // The imatrix .npz is keyed by the QWEN container's own add_linear base names
  // (tools/reference/imatrix_capture.py); the DFlash2 drafter's tensors share none of them, so an
  // --imatrix passed alongside --dflash-gguf could only ever be a no-op. Reject it instead of
  // silently converting the drafter with unweighted MSE and reporting "imatrix-weighted".
  if (!a.imatrix.empty() && !a.dflash_gguf.empty())
    throw std::runtime_error(
        "--imatrix does not apply to --dflash-gguf (the drafter has no importance matrix; "
        "tools/reference/imatrix_capture.py only captures the main model's linears)");
  // --keep-bf16 is wired into the HF-checkpoint and --selftest paths only. Accepting it silently
  // alongside --dflash-gguf would produce a drafter container in which it did nothing at all -- the
  // exact failure mode the "matched nothing" warning exists to make visible, so reject it outright
  // rather than emit a warning nobody reads in a sweep log. (--layout bf16 is how you get a bf16
  // drafter.)
  if (!a.keep_bf16.empty() && !a.dflash_gguf.empty())
    throw std::runtime_error("--keep-bf16 does not apply to --dflash-gguf (use --layout bf16 for a "
                             "bf16 drafter container)");
  // --ldlq has nothing to round against without the Hessians; failing here rather than at the first
  // matching linear keeps "forgot a flag" a one-second error instead of a mid-plan one.
  if (!a.ldlq.empty() && a.hessian_dir.empty())
    throw std::runtime_error("--ldlq requires --hessian-dir <tools/reference/hessian_capture.py "
                             "output directory>");
  // hessian_capture.py captures the main model's linear inputs only (docs/quant2.md 1.2: "LDLQ for
  // the DFlash2 drafter" is a non-goal), and its keys are the Qwen container's base names -- the
  // drafter shares none of them, so the flags could only ever be a no-op there. Same reasoning as
  // --imatrix/--keep-bf16 above.
  if ((!a.ldlq.empty() || !a.hessian_dir.empty()) && !a.dflash_gguf.empty())
    throw std::runtime_error("--ldlq/--hessian-dir do not apply to --dflash-gguf (no Hessians are "
                             "captured for the drafter; tools/reference/hessian_capture.py only "
                             "covers the main model's linears)");
  return a;
}

// Loads the imatrix .npz once (or nothing) and hands out per-linear weight vectors by container
// base name. Also tallies coverage so a run against a stale/partial imatrix is visible in the log
// rather than silently degrading to unweighted MSE for half the model.
class ImatrixSource {
 public:
  ImatrixSource(const std::string& path, r4dx_convert::QuantMode mode) : mode_(mode) {
    if (path.empty()) return;
    npz_ = std::make_unique<r4dx_convert::NpzReader>(path);
    std::cout << "[r4dx-convert] imatrix=" << path << " (" << npz_->Count() << " vector(s))\n";
  }

  r4dx_convert::QuantOptions For(const std::string& container_base, int64_t K) {
    r4dx_convert::QuantOptions opts;
    opts.mode = mode_;
    if (!npz_) return opts;
    int64_t len = 0;
    const float* v = npz_->Vector(container_base, &len);
    if (v == nullptr) {
      std::lock_guard<std::mutex> lk(mu_);
      ++missing_;
      if (missing_ <= 8)
        std::cerr << "[r4dx-convert] WARNING: imatrix has no vector for '" << container_base
                  << "' -- that linear is quantized with unweighted MSE\n";
      return opts;
    }
    if (len != K) {
      // A length mismatch means the imatrix was captured against a different checkpoint/shape;
      // using it would weight the wrong channels, which is worse than not weighting at all.
      throw std::runtime_error("r4dx-convert: imatrix vector '" + container_base + "' has length " +
                                std::to_string(len) + " but the linear's K is " +
                                std::to_string(K));
    }
    opts.importance.data = v;
    opts.importance.size = len;
    {
      std::lock_guard<std::mutex> lk(mu_);
      ++hit_;
    }
    return opts;
  }

  // A partial imatrix is a silent quality regression -- the container still converts, still says
  // "imatrix-weighted" at the top of a 250 s log, and just quietly quantizes some fraction of the
  // model with unweighted MSE. So the coverage line goes to stderr and says WARNING when anything
  // fell back, next to the (capped) per-linear warnings, instead of being one more stdout line.
  void ReportCoverage() const {
    if (!npz_) return;
    std::ostream& os = (missing_ > 0) ? std::cerr : std::cout;
    os << "[r4dx-convert] " << (missing_ > 0 ? "WARNING: " : "")
       << "imatrix coverage: " << hit_ << " linear(s) weighted, " << missing_
       << " fell back to unweighted MSE\n";
  }

 private:
  r4dx_convert::QuantMode mode_;
  std::unique_ptr<r4dx_convert::NpzReader> npz_;
  std::mutex mu_;
  int64_t hit_ = 0, missing_ = 0;
};

bool HasQuantizedLayout(const LayoutSet& ls) { return ls.w4a16; }

double SecondsBetween(std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// The `--ldlq <regex>` selector plus the HessianStore behind it (docs/quant2.md 2.2-2.3).
//
// The decision "is this linear LDLQ'd" is made ONCE per linear by the caller (add_linear), from
// Matches() + the linear's resolved LayoutSet, and captured by value in both the plan and the emit
// lambda -- the same discipline --keep-bf16 uses, so the two passes cannot disagree. Plan() is the
// planning-pass half: it checks the manifest (key present, K equal, K a whole number of 128-column
// LDLQ blocks, every emitted layout's group dividing the block) and the key's .hess file on disk
// (header and exact size, no payload read), and records the base, so a stale or partial Hessian
// directory dies before the header is written and before a single shard is read. (A file whose
// payload is corrupt but correctly sized -- non-finite values, trace mismatch -- is still only
// caught when Factor() reads it.) Factor()/Record() are the emit-pass half.
//
// Not thread-safe, and it does not need to be: both passes run their job lists sequentially (the
// parallelism is INSIDE each quantizer), and the store's one-entry cache relies on that order --
// adjacent linears sharing a tap (gdn.in_proj_qkv/z, attn.qg/k/v) are emitted back to back, so they
// factor once.
class LdlqSource {
 public:
  // quant_ldlq.hpp's column-block width (docs/quant2.md 2.2: B = 128). The quantizers throw on
  // K % 128 != 0 themselves; checking it here moves that failure from the emit pass to planning.
  static constexpr int64_t kBlock = 128;

  LdlqSource(const std::string& pattern, const std::string& hessian_dir, float damp)
      : pattern_(pattern), dir_(hessian_dir), damp_(damp) {
    if (pattern_.empty()) {
      // ParseArgs already rejects --ldlq without --hessian-dir; the reverse is merely pointless, but
      // it usually means the --ldlq half of a copy-pasted command line went missing.
      if (!dir_.empty())
        std::cerr << "[r4dx-convert] WARNING: --hessian-dir given without --ldlq -- ignored, no "
                     "linear is LDLQ'd\n";
      return;
    }
    try {
      re_ = std::regex(pattern_, std::regex::ECMAScript);
    } catch (const std::regex_error& e) {
      throw std::runtime_error("--ldlq: invalid ECMAScript regex '" + pattern_ + "': " + e.what());
    }
    // Parses (and validates the format/version of) hessian.json right here, at startup: a wrong
    // directory is a one-second error, not a mid-plan one.
    store_ = std::make_unique<r4dx_convert::HessianStore>(dir_);
    enabled_ = true;
  }

  bool Enabled() const { return enabled_; }
  const std::string& Pattern() const { return pattern_; }
  const std::string& Dir() const { return dir_; }
  float Damp() const { return damp_; }
  std::string ManifestSha256() const { return store_ ? store_->ManifestSha256() : std::string(); }
  // Every .hess file this run could read (HessianStore::FilesFor over the bases the regex selects),
  // as (name in hessian.json, path). Empty when --ldlq is off. The reuse guard hashes them.
  std::vector<std::pair<std::string, std::string>> HessianFiles() const {
    std::vector<std::pair<std::string, std::string>> out;
    if (!enabled_) return out;
    for (const auto& f : store_->FilesFor([this](const std::string& b) { return Matches(b); }))
      out.push_back({f, r4dx_convert::hessian_detail::JoinPath(dir_, f)});  // the path Factor opens
    return out;
  }
  // The LDLQ'd bases, in container (= planning) order. Filled by Plan(); complete once the planning
  // pass is over, which is before the metadata that records it is built.
  const std::vector<std::string>& Linears() const { return planned_; }
  // The subset of Linears() rounded against their "rms_keys" Hessian (rotated in-projections only;
  // see HasRms), same order.
  const std::vector<std::string>& RmsLinears() const { return planned_rms_; }

  bool Matches(const std::string& container_base) const {
    return enabled_ && std::regex_search(container_base, re_);
  }

  // hessian.json has a weightless rms Hessian ("rms_keys", hessian_store.hpp) for this base. Only a
  // ROTATED in-projection may use it -- its folded input is exactly rms(x) Q -- so the caller
  // (add_linear) combines this with the linear's fold, once, and hands the result to Plan() and
  // Factor(). An unrotated linear never reads the rms file, whatever the manifest holds.
  bool HasRms(const std::string& container_base) const {
    return enabled_ && store_->HasRms(container_base);
  }

  // Planning pass, for a linear the caller decided to LDLQ. Throws on any manifest problem. `rms`:
  // the caller's decision that this linear rounds against its "rms_keys" file (then checked on disk
  // too).
  void Plan(const std::string& container_base, int64_t K, const LayoutSet& ls, bool rms = false) {
    if (!store_->Has(container_base)) {
      throw std::runtime_error(
          "--ldlq '" + pattern_ + "' selects '" + container_base + "' but " + dir_ +
          "/hessian.json has no key for it -- capture it with tools/reference/hessian_capture.py "
          "(the MTP draft head needs its --draft-head) or narrow the regex");
    }
    const int64_t hk = store_->KOf(container_base);
    if (hk != K) {
      throw std::runtime_error("--ldlq: '" + container_base + "' has K=" + std::to_string(K) +
                               " in this checkpoint but " + dir_ + "/hessian.json says K=" +
                               std::to_string(hk) + " (" + store_->File(container_base) +
                               ") -- the Hessians were captured against a different checkpoint");
    }
    if (K % kBlock != 0) {
      throw std::runtime_error("--ldlq: '" + container_base + "' has K=" + std::to_string(K) +
                               ", not a multiple of the LDLQ block width " +
                               std::to_string(kBlock));
    }
    // quant_ldlq.hpp also needs every emitted layout's group to tile the block (a group never
    // straddles two blocks). w4a16's group is this linear's
    // own (`ls.w4a16_group`: a --w4a16-group-rule's 32/64/128, all fine, or the build's
    // R4DX_W4A16_GROUP, which may be any multiple of 64 -- 192, 256, ... do not divide 128).
    auto require_group = [&](bool on, int group, const char* layout) {
      if (on && kBlock % group != 0) {
        throw std::runtime_error("--ldlq: '" + container_base + "' emits " + layout +
                                 " with group " + std::to_string(group) +
                                 ", which does not divide the LDLQ block width " +
                                 std::to_string(kBlock) + " -- LDLQ needs a build with group <= " +
                                 std::to_string(kBlock) + " or drop " + layout + " from --layouts");
      }
    };
    require_group(ls.w4a16, ls.w4a16_group, "w4a16");
    // The file itself (header, size, K/rows vs the manifest), each distinct file once.
    store_->CheckFile(container_base);
    if (rms) {
      // HessianStore already refused an "rms_keys" entry whose K differs from the "keys" one.
      store_->CheckFile(container_base, r4dx_convert::HessianSource::kRms);
      planned_rms_.push_back(container_base);
    }
    planned_.push_back(container_base);
  }

  // Planning pass, for a linear the regex matched but that is NOT LDLQ'd: --keep-bf16 won, or the
  // run asked for no quantized layout of it (e.g. `--lm-head bf16`). Counted, so the summary line
  // can say why the selection is smaller than the regex suggests.
  void NoteSkipped() { ++skipped_; }

  // Emit pass. `*reused` = this base shares its Hessian file AND its --rotate change of basis with
  // the linear emitted just before it (same tap, same fold), i.e. the store's cached factorization
  // was returned rather than a new one computed. `xf` is RotationSource::HessianFor's transform, or
  // nullptr for a linear whose input basis the rotation does not change. `rms` = Plan()'s: factor
  // the base's "rms_keys" file instead of its "keys" one.
  const r4dx_convert::LdlqFactor& Factor(const std::string& container_base, int64_t K, int threads,
                                         bool* reused,
                                         const r4dx_convert::HessianTransform* xf = nullptr,
                                         bool rms = false) {
    const r4dx_convert::HessianSource src =
        rms ? r4dx_convert::HessianSource::kRms : r4dx_convert::HessianSource::kKeys;
    const std::string file = store_->FileFor(container_base, src);
    const std::string xf_id = xf ? xf->id : std::string();
    *reused = (file == last_file_ && xf_id == last_xf_id_);
    last_file_ = file;
    last_xf_id_ = xf_id;
    return store_->Factor(container_base, K, damp_, threads, xf, src);
  }

  // Emit pass, after the linear is written: the per-linear log line and the run totals.
  void Record(const std::string& container_base, int64_t N, int64_t K, double factor_s,
              double emit_s, const r4dx_convert::LdlqFactor& f, bool reused, std::ostream& log,
              bool rms = false) {
    ++done_;
    if (!reused) ++factorizations_;
    if (f.retries > 0) ++retried_;
    factor_s_ += factor_s;
    emit_s_ += emit_s;
    log << "[r4dx-convert] ldlq: " << container_base << " [" << N << "," << K << "]"
        << (rms ? " rms Hessian " + store_->RmsFile(container_base) + ":" : std::string())
        << " factor " << factor_s << " s" << (reused ? " (shared tap, cached)" : "")
        << ", quantize+write " << emit_s << " s, damp_used=" << f.damp_used
        << " retries=" << f.retries << "\n";
  }

  // After the planning pass (before the long emit pass), like KeepBf16Selector::Report. A valid
  // regex that selects nothing is a WARNING on stderr, not an error: same sweep-script reasoning as
  // --keep-bf16, and a container built that way is still an ordinary --quant container.
  void ReportPlan(std::ostream& log, std::ostream& warn) const {
    if (!enabled_) return;
    if (planned_.empty()) {
      warn << "[r4dx-convert] WARNING: --ldlq '" << pattern_ << "' selected no quantized linear"
           << (skipped_ > 0 ? " (" + std::to_string(skipped_) +
                                  " match(es) have no 4-bit layout: --keep-bf16, bf16-only or "
                                  "trellis)"
                            : std::string())
           << " -- every linear follows --quant/--imatrix as usual\n";
      return;
    }
    log << "[r4dx-convert] ldlq '" << pattern_ << "': " << planned_.size()
        << " linear(s) selected, damp=" << damp_ << ", hessian-dir=" << dir_
        << " (hessian.json sha256 " << store_->ManifestSha256() << ")";
    if (skipped_ > 0)
      log << "; " << skipped_
          << " other match(es) have no 4-bit layout (bf16-only or trellis), not LDLQ'd";
    log << "\n";
    if (!planned_rms_.empty())
      log << "[r4dx-convert] ldlq: " << planned_rms_.size()
          << " rotated in-projection(s) round against the weightless rms Hessian "
             "(hessian.json rms_keys): H' = Q^T H_rms Q\n";
  }

  // End of run. LDLQ'd linears never consult --imatrix, so they are in neither count of the imatrix
  // coverage line -- said here so the two lines add up.
  void ReportRun(std::ostream& log) const {
    if (!enabled_ || planned_.empty()) return;
    log << "[r4dx-convert] ldlq: " << done_ << " linear(s) quantized with LDLQ ("
        << factorizations_ << " factorization(s), " << factor_s_ << " s; quantize+write "
        << emit_s_ << " s; " << retried_ << " needed a damping retry) -- not counted in the "
        << "imatrix coverage line\n";
  }

 private:
  std::string pattern_, dir_;
  float damp_ = 0.0f;
  bool enabled_ = false;
  std::regex re_;
  std::unique_ptr<r4dx_convert::HessianStore> store_;
  std::vector<std::string> planned_, planned_rms_;
  int64_t skipped_ = 0, done_ = 0, factorizations_ = 0, retried_ = 0;
  double factor_s_ = 0.0, emit_s_ = 0.0;
  std::string last_file_, last_xf_id_;
};

// ---- --rotate (docs/quant2.md sections 3-4, rotation.hpp) ------------------------------------------

// What --rotate does to ONE linear. Resolved at the call site in RunConvert's text-layer loop -- the
// only loop that folds; the MTP head, lm_head, the draft head and the DFlash2 drafter all take the
// default (kNone) -- and handed to add_linear as a VARIABLE, never as a string literal:
// tools/reference/imatrix_capture.py's audit_converter_source parses every add_linear call and
// requires exactly one quoted string after the HF-name list.
enum class HadSite { kNone, kDown, kO, kGdnOut };

struct LinearFold {
  enum Kind { kNone, kIn, kOut };
  Kind kind = kNone;
  std::string norm_hf;           // kIn: the HF zero-centred norm whose (1 + w) folds into this linear
  HadSite had = HadSite::kNone;  // kOut under q2ab: the block Hadamard folded into the K side
  // The linear's INPUT basis changes (so its Hessian / imatrix vector must follow it).
  bool RotatesInput() const { return kind == kIn || had != HadSite::kNone; }
};

const char* HadSiteName(HadSite s) {
  switch (s) {
    case HadSite::kDown: return "down";
    case HadSite::kO: return "o";
    case HadSite::kGdnOut: return "gdn_out";
    default: return "none";
  }
}

// Owns the generated rotation (r4dx_convert::RotationSet) for the run and applies it per linear:
// shape checks at planning time, the weight fold, the Hessian and imatrix changes of basis at emit
// time, and the metadata block. A disabled source (--rotate none) hands out kNone folds only, so
// every call site below degenerates to exactly the pre-rotation code path.
class RotationSource {
 public:
  RotationSource(const std::string& kind, uint64_t seed, bool seed_explicit,
                 const nlohmann::json& text_cfg)
      : kind_(r4dx_convert::ParseRotationKind(kind)) {
    using namespace r4dx_convert;
    if (kind_ == RotationKind::kNone) {
      if (seed_explicit)
        std::cerr << "[r4dx-convert] WARNING: --rotation-seed given without --rotate -- ignored, the "
                     "container is not rotated\n";
      return;
    }
    RotationShape shape;
    shape.hidden = text_cfg.at("hidden_size").get<int64_t>();
    // The runtime's online op (src/kernels, one workgroup per 5120-wide row) and the tensor name
    // rotation.mix5 are both this exact factorization, so anything else is refused here rather than
    // written as a container no binary can run.
    if (shape.hidden != 5 * kRotationBlock)
      throw std::runtime_error("--rotate: hidden_size=" + std::to_string(shape.hidden) +
                               ", but the rotation is defined for 5120 = 5 x 1024 only");
    if (kind_ == RotationKind::kQ2ab) {
      const int64_t head_dim = text_cfg.at("head_dim").get<int64_t>();
      const int64_t v_head_dim = text_cfg.at("linear_value_head_dim").get<int64_t>();
      // attn.o's and gdn.out_proj's Hadamard block IS one head (the runtime's fused kernels run one
      // workgroup per (token, head)); a checkpoint with other head widths needs other kernels.
      if (head_dim != kHadBlockO || v_head_dim != kHadBlockGdnOut)
        throw std::runtime_error("--rotate q2ab: needs head_dim=" + std::to_string(kHadBlockO) +
                                 " and linear_value_head_dim=" + std::to_string(kHadBlockGdnOut) +
                                 " (got " + std::to_string(head_dim) + ", " +
                                 std::to_string(v_head_dim) + ")");
      shape.k_down = text_cfg.at("intermediate_size").get<int64_t>();
      shape.k_o = text_cfg.at("num_attention_heads").get<int64_t>() * head_dim;
      shape.k_gdn_out = text_cfg.at("linear_num_value_heads").get<int64_t>() * v_head_dim;
    }
    set_ = GenerateRotationSet(kind_, seed, shape);
  }

  bool Enabled() const { return kind_ != r4dx_convert::RotationKind::kNone; }
  bool Hadamard() const { return kind_ == r4dx_convert::RotationKind::kQ2ab; }
  const char* KindName() const { return r4dx_convert::RotationKindName(kind_); }
  uint64_t Seed() const { return set_.seed; }
  const r4dx_convert::RotationSet& Set() const { return set_; }

  LinearFold In(const std::string& norm_hf) const {
    LinearFold f;
    if (Enabled()) {
      f.kind = LinearFold::kIn;
      f.norm_hf = norm_hf;
    }
    return f;
  }
  LinearFold Out(HadSite site) const {
    LinearFold f;
    if (Enabled()) {
      f.kind = LinearFold::kOut;
      if (Hadamard()) f.had = site;
    }
    return f;
  }

  const r4dx_convert::BlockHadamard& Had(HadSite s) const {
    switch (s) {
      case HadSite::kDown: return set_.had_down;
      case HadSite::kO: return set_.had_o;
      case HadSite::kGdnOut: return set_.had_gdn_out;
      default: throw std::logic_error("RotationSource::Had(kNone)");
    }
  }

  // Planning pass: the linear's shape against the fold (and the norm tensor's), so a mismatch dies
  // before the header is written.
  void CheckPlan(const LinearFold& f, const std::string& base, int64_t N, int64_t K,
                 ShardedModel& model) const {
    if (f.kind == LinearFold::kNone) return;
    const int64_t hidden = set_.q.hidden;
    if (f.kind == LinearFold::kIn) {
      if (K != hidden)
        throw std::runtime_error("--rotate: in-projection '" + base + "' has K=" +
                                 std::to_string(K) + ", expected hidden=" + std::to_string(hidden));
      const auto& m = model.Meta(f.norm_hf);
      if (m.shape.size() != 1 || m.shape[0] != hidden)
        throw std::runtime_error("--rotate: norm '" + f.norm_hf + "' folded into '" + base +
                                 "' is not a [" + std::to_string(hidden) + "] vector");
      return;
    }
    if (N != hidden)
      throw std::runtime_error("--rotate: out-projection '" + base + "' has N=" + std::to_string(N) +
                               ", expected hidden=" + std::to_string(hidden));
    if (f.had != HadSite::kNone && K != Had(f.had).K)
      throw std::runtime_error("--rotate q2ab: '" + base + "' has K=" + std::to_string(K) +
                               " but rotation.had_" + HadSiteName(f.had) + "_signs is " +
                               std::to_string(Had(f.had).K) + " long");
  }

  // Emit pass. The fp32 norm weight of a kIn fold (empty otherwise), read once per linear and shared
  // by Fold / Importance / HessianFor.
  std::vector<float> ReadNorm(const LinearFold& f, ShardedModel& model) const {
    if (f.kind != LinearFold::kIn) return {};
    return r4dx_convert::ReadTensorAsFloat(model, f.norm_hf);
  }

  // W [N, K] in place, fp32 -> (double math) -> fp32, before any rounding to the container's forms.
  // A q2ab out-projection is two passes (Q^T on the columns, then Hb on the rows) with the weight
  // held in fp32 between them: ~1e-7 relative, far below even bf16's step, and it keeps the peak at
  // one fp32 copy of mlp.down instead of a 713 MB double one.
  void Fold(const LinearFold& f, const std::vector<float>& norm, std::vector<float>& w, int64_t N,
            int64_t K, int threads) {
    using namespace r4dx_convert;
    if (f.kind == LinearFold::kIn) {
      FoldRowsQ(w, N, K, norm.data(), set_.q, threads);
      ++folded_in_;
    } else if (f.kind == LinearFold::kOut) {
      FoldColumnsQt(w, N, K, set_.q, threads);
      ++folded_out_;
      if (f.had != HadSite::kNone) {
        FoldRowsHadamard(w, N, K, Had(f.had), threads);
        ++folded_had_;
      }
    }
  }

  // The --imatrix vector carried into the folded linear's input basis under the diagonal model
  // (rotation.hpp). Only called for f.RotatesInput().
  std::vector<float> Importance(const LinearFold& f, const std::vector<float>& norm,
                                const r4dx_convert::ImportanceVector& v, int64_t K) {
    ++imatrix_rotated_;
    if (f.kind == LinearFold::kIn)
      return r4dx_convert::TransformImportanceQ(v.data, K, norm.data(), set_.q);
    return r4dx_convert::TransformImportanceHadamard(v.data, K, Had(f.had));
  }

  // Planning pass, for an in-projection that --ldlq rounds against its POST-norm Hessian (no
  // "rms_keys" entry): TransformHessianQ will divide (1 + w) out of it, which needs every
  // |1 + w| >= 1e-3. Checked here, from the [hidden] norm vector alone, so a dead channel fails the
  // conversion before the header is written instead of at that layer's emit, and the message says
  // how to get the rms Hessian that needs no division.
  void CheckNormDivisible(const LinearFold& f, const std::string& base, ShardedModel& model) {
    if (f.kind != LinearFold::kIn) return;
    const std::vector<float> norm = r4dx_convert::ReadTensorAsFloat(model, f.norm_hf);
    r4dx_convert::CheckNormInvertible(
        norm.data(), static_cast<int64_t>(norm.size()),
        "--rotate " + std::string(KindName()) + " --ldlq: '" + base + "' (norm '" + f.norm_hf +
            "') has no rms Hessian in hessian.json, and its post-norm Hessian cannot be used",
        r4dx_convert::kRmsHessianHint);
    ++ldlq_divided_;
  }

  // The LDLQ Hessian's change of basis for a folded linear, or nullptr when its input basis is
  // unchanged (every out-projection under q2a, and every unfolded linear). The id -- part of
  // HessianStore's factor cache key, next to the file -- is the kind, the seed, the site and, for an
  // in-projection rounded against its post-norm Hessian, the norm's name and the sha256 of its fp32
  // bytes; for one rounded against its weightless rms Hessian (`rms`, the "rms_keys" file) it is
  // "in-rms", since Q^T H_rms Q depends on nothing else. Either way attn.qg / k / v (one
  // input_layernorm tap) and gdn.in_proj_qkv / z still share one factorization, and nothing else can.
  std::unique_ptr<r4dx_convert::HessianTransform> HessianFor(const LinearFold& f,
                                                             const std::vector<float>& norm,
                                                             bool rms = false) {
    if (!f.RotatesInput()) return nullptr;
    ++ldlq_rotated_;
    auto xf = std::make_unique<r4dx_convert::HessianTransform>();
    const std::string prefix =
        std::string(KindName()) + ":seed=" + std::to_string(set_.seed) + ":";
    if (f.kind == LinearFold::kIn && rms) {
      xf->id = prefix + "in-rms";
      const r4dx_convert::ResidualRotation* q = &set_.q;
      xf->apply = [q](std::vector<float>& H, int64_t K, int t) {
        r4dx_convert::TransformHessianRmsQ(H, K, *q, t);
      };
    } else if (f.kind == LinearFold::kIn) {
      const std::string bytes(reinterpret_cast<const char*>(norm.data()), norm.size() * sizeof(float));
      xf->id = prefix + "in:" + f.norm_hf + ":" + r4dx_convert::Sha256Hex(bytes);
      const r4dx_convert::ResidualRotation* q = &set_.q;
      xf->apply = [q, norm](std::vector<float>& H, int64_t K, int t) {
        r4dx_convert::TransformHessianQ(H, K, norm.data(), *q, t);
      };
    } else {
      xf->id = prefix + "had:" + HadSiteName(f.had);
      const r4dx_convert::BlockHadamard* hb = &Had(f.had);
      xf->apply = [hb](std::vector<float>& H, int64_t K, int t) {
        r4dx_convert::TransformHessianHadamard(H, K, *hb, t);
      };
    }
    return xf;
  }

  // __metadata__.rotation (docs/container-format.md "Residual rotation"). Only written when enabled.
  nlohmann::json Metadata() const {
    nlohmann::json j = {{"kind", KindName()},
                        {"seed", set_.seed},
                        {"hidden", set_.q.hidden},
                        {"block", set_.q.block}};
    if (Hadamard())
      j["had"] = {{"down", set_.had_down.block}, {"o", set_.had_o.block},
                  {"gdn_out", set_.had_gdn_out.block}};
    return j;
  }

  void ReportRun(std::ostream& log) const {
    if (!Enabled()) return;
    log << "[r4dx-convert] rotate " << KindName() << ": folded " << folded_in_
        << " in-projection(s) (W diag(1+w) Q), " << folded_out_ << " out-projection(s) (Q^T W)";
    if (Hadamard()) log << ", " << folded_had_ << " of them also W Hb on the K side";
    log << "; " << ldlq_rotated_ << " LDLQ Hessian(s) and " << imatrix_rotated_
        << " imatrix vector(s) carried into the rotated basis";
    if (ldlq_divided_ > 0)
      log << " (" << ldlq_divided_ << " in-projection Hessian(s) by dividing the norm out of the "
          << "post-norm capture: no rms_keys entry)";
    log << "\n";
  }

 private:
  r4dx_convert::RotationKind kind_;
  r4dx_convert::RotationSet set_;
  int64_t folded_in_ = 0, folded_out_ = 0, folded_had_ = 0, ldlq_rotated_ = 0,
          imatrix_rotated_ = 0, ldlq_divided_ = 0;
};

r4dx_convert::QuantMode ParseQuantMode(const std::string& s) {
  return s == "search" ? r4dx_convert::QuantMode::kSearch : r4dx_convert::QuantMode::kRtn;
}

int ResolveThreads(int requested) {
  if (requested > 0) return requested;
  unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(hw) : 4;
}

std::string ReadFile(const std::string& path) {
  // Wide open (MSVC ifstream(const wchar_t*) extension), not narrow fopen/ifstream(path) --
  // consistent with SafetensorsReader's CreateFileW so a non-ASCII --input path on a non-UTF-8
  // ANSI codepage resolves to the same file both places (review finding, minor).
  std::ifstream f(r4dx_convert::Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---- --record-reuse-guard / --reuse-tensors-from (docs/quant2.md 5.2, reuse_guard.hpp) ------------

nlohmann::json LayoutSetJson(const LayoutSet& ls) {
  // Without w4a16_group: the group is the one thing a reuse may change, and it is resolved per
  // linear (the per-tensor decision in RunConvert), never at the LayoutSet the flags give.
  return {{"bf16", ls.bf16}, {"w4a16", ls.w4a16}};
}

// __metadata__.r4dx_convert_run.reuse_guard: everything this run's bytes can depend on EXCEPT the
// per-linear layout choices -- --w4a16-group-rule and --keep-bf16 -- (and --output, --threads, the
// reuse flags), in resolved form -- so two spellings of one setting (--layers 99 on a 64-layer
// checkpoint, --vision's auto default) compare equal while any real difference does not. Files are
// identified by content; a flag whose file the run never reads is recorded as unused rather than
// hashed. Every field is compared, whole: a new input the converter learns to read must be added here
// (and kReuseGuardVersion bumped). The per-linear choices are recorded resolved, not as regex text:
// RunConvert adds reuse_guard.linears (every linear's LayoutSetId) once add_linear has resolved them,
// and PlanReuse compares that linear by linear instead. The two completion fields (emit_complete,
// data_sha256) are written as placeholders and patched once the data is on disk (RunConvert's end;
// reuse_guard.hpp).
nlohmann::json BuildReuseGuard(const AppArgs& args, const std::string& config_text, int layers,
                               bool do_vision, bool do_mtp, const LayoutSet& layouts,
                               const LayoutSet& lm_head_layouts, const LdlqSource& ldlq,
                               const RotationSource& rot, int threads) {
  using namespace r4dx_convert;
  auto file_id = [](const std::string& path, bool used) {
    if (path.empty()) return std::string("none");
    if (!used) return std::string("unused");
    return Sha256File(path);
  };
  nlohmann::json g = nlohmann::json::object();
  g["version"] = kReuseGuardVersion;
  g[kReuseCompleteKey] = 0;
  g[kReuseDataKey] = ReuseDataPlaceholder();
  // The binary itself, not a version string: a rebuild from any other source, with other build
  // options (R4DX_W4A16_GROUP, the vendored r4d_core it links statically) or none at all hashes
  // differently, and is refused. Deliberately no override: a baseline costs one conversion, while a
  // container mixing two converters' tensors would measure as a group-rule effect.
  g["converter"] = {{"exe_sha256", Sha256File(ExecutablePath())},
                    // the CRT DLLs the exe hash does not cover (rotation.mix5 goes through ucrt's
                    // log/cos/sin)
                    {"runtime_dlls", RuntimeModulesIdentity()},
                    // ... and the CPU, which picks ucrt's FMA3 or SSE2 transcendentals (and
                    // dense_linalg's path) at run time
                    {"cpu", CpuIdentity()},
                    {"w4a16_group", kW4A16Group},
                    // bit-identical by construction and gated (dense_linalg.hpp), recorded anyway
                    {"avx512", linalg::UseAvx512()}};
  // The two large file sets -- the checkpoint's shards and the .hess files LDLQ can read, often on
  // different drives -- in one parallel pass, so neither waits for the other.
  const CheckpointFiles cf = ReadCheckpointIndex(args.input);
  const std::vector<std::pair<std::string, std::string>> hess = ldlq.HessianFiles();
  std::vector<std::string> paths;
  for (const auto& s : cf.shards) paths.push_back(args.input + "\\" + s);
  for (const auto& h : hess) paths.push_back(h.second);
  const std::vector<FileDigest> dig = Sha256Files(paths, threads);
  g["checkpoint"] = CheckpointIdentity(
      config_text, cf,
      std::vector<FileDigest>(dig.begin(), dig.begin() + static_cast<std::ptrdiff_t>(cf.shards.size())));
  nlohmann::json hess_files = ldlq.Enabled() ? nlohmann::json::object() : nlohmann::json("none");
  for (size_t i = 0; i < hess.size(); ++i) {
    const FileDigest& d = dig[cf.shards.size() + i];
    hess_files[hess[i].first] = {{"bytes", d.bytes}, {"sha256", d.sha256}};
  }
  g["args"] = {
      {"layers", layers},
      {"vision", do_vision},
      {"mtp", do_mtp},
      {"layouts", LayoutSetJson(layouts)},
      {"lm_head", LayoutSetJson(lm_head_layouts)},
      {"quant", args.quant},
      // No keep_bf16: what it keeps is per-linear, in reuse_guard.linears (RunConvert adds it).
      {"ldlq", ldlq.Enabled() ? args.ldlq : std::string()},
      {"ldlq_damp", ldlq.Enabled() ? nlohmann::json(ldlq.Damp()) : nlohmann::json(nullptr)},
      {"rotate", rot.KindName()},
      {"rotation_seed", rot.Enabled() ? nlohmann::json(rot.Seed()) : nlohmann::json(nullptr)},
  };
  g["inputs"] = {
      {"imatrix_sha256", file_id(args.imatrix, true)},
      {"kv_calib_sha256", file_id(args.kv_calib, true)},
      {"draft_vocab_ids_sha256", file_id(args.draft_vocab_ids, do_mtp)},
      // hessian.json by content, AND every .hess file the --ldlq regex can make this run read
      // (HessianStore::FilesFor: the selected bases' keys and rms_keys files), each by length +
      // sha256: the manifest pins each file's K, rows and header trace (CheckFile), but nothing
      // ties a payload to it -- an in-place edit that keeps the diagonal passes every check. A
      // reuse run never reads the files of the linears it copies, so only their hashes can say the
      // baseline rounded against the same Hessians. The directory's path is not part of it.
      {"hessian_manifest_sha256", ldlq.Enabled() ? ldlq.ManifestSha256() : std::string("none")},
      {"hessian_files", hess_files},
  };
  return g;
}

// --reuse-tensors-from, first half, before this run's guard is hashed (so a baseline that could never
// be used -- an older converter's, without a guard -- is refused at once, not after the checkpoint
// and the Hessians are read): opens the baseline (its header only) and refuses it unless it is a
// completed conversion that recorded a guard and a data digest. CheckReuseBaseline does the rest.
std::unique_ptr<r4dx_convert::BaselineContainer> OpenReuseBaseline(const AppArgs& args) {
  namespace fs = std::filesystem;
  const std::string who = "--reuse-tensors-from " + args.reuse_from;
  std::error_code ec;
  const fs::path base_path = fs::u8path(args.reuse_from), out_path = fs::u8path(args.output);
  if (!fs::is_regular_file(base_path, ec))
    throw std::runtime_error(who + ": no such file");
  // FinalizeHeader truncates --output: never let it be the file this run copies from.
  if (fs::exists(out_path, ec) && fs::equivalent(out_path, base_path, ec))
    throw std::runtime_error(who + ": --output is the baseline itself");
  auto b = r4dx_convert::BaselineContainer::Open(args.reuse_from);
  const nlohmann::json& md = b->metadata;
  if (!md.contains("r4dx_convert_run") || !md["r4dx_convert_run"].is_object() ||
      !md["r4dx_convert_run"].contains("reuse_guard")) {
    throw std::runtime_error(who + ": the baseline has no __metadata__.r4dx_convert_run.reuse_guard "
                             "-- convert it with --record-reuse-guard (the guard records what the "
                             "baseline was made from; without it nothing can be checked)");
  }
  const nlohmann::json& bg = md["r4dx_convert_run"]["reuse_guard"];
  const auto done = bg.is_object() ? bg.find(r4dx_convert::kReuseCompleteKey) : bg.end();
  if (!bg.is_object() || done == bg.end() || !done->is_number_integer() || done->get<int64_t>() != 1) {
    throw std::runtime_error(who + ": the baseline's emit pass never completed (reuse_guard." +
                             std::string(r4dx_convert::kReuseCompleteKey) + " is not 1) -- it is "
                             "an interrupted or failed conversion; convert it again");
  }
  const auto data = bg.find(r4dx_convert::kReuseDataKey);
  const std::string recorded = data != bg.end() && data->is_string() ? data->get<std::string>() : "";
  if (!r4dx_convert::IsSha256Hex(recorded) || recorded == r4dx_convert::ReuseDataPlaceholder())
    throw std::runtime_error(who + ": the baseline's reuse_guard." + r4dx_convert::kReuseDataKey +
                             " is not a recorded digest -- an interrupted or failed conversion; "
                             "convert it again");
  return b;
}

// --reuse-tensors-from, second half, once this run's guard is built: refuses the baseline unless its
// guard equals `guard` field for field, its quant block is this build's, and its tensor data still
// hashes to the digest it recorded (read in full, `threads` tensors at a time -- last, being the
// expensive part). Still before the first shard is opened.
void CheckReuseBaseline(r4dx_convert::BaselineContainer& b, const nlohmann::json& guard, int threads) {
  const std::string who = "--reuse-tensors-from " + b.path;
  const nlohmann::json& md = b.metadata;
  const nlohmann::json& bg = md.at("r4dx_convert_run").at("reuse_guard");
  const std::vector<std::string> diff = r4dx_convert::ReuseGuardMismatches(bg, guard);
  if (!diff.empty()) {
    std::string msg = who + ": refused -- the baseline was not converted from the same inputs with "
                            "the same flags by this binary (only --w4a16-group-rule, --keep-bf16, "
                            "--output and --threads may differ); " + std::to_string(diff.size()) +
                      " guard field(s) differ:";
    for (const auto& d : diff) msg += "\n  " + d;
    throw std::runtime_error(msg);
  }
  // The format block, apart from the per-tensor map a reuse may change (the guard's binary hash
  // already implies it; checked explicitly because the copy trusts it).
  nlohmann::json q = md.contains("quant") ? md["quant"] : nlohmann::json();
  if (q.is_object() && q.contains("w4a16") && q["w4a16"].is_object()) q["w4a16"].erase("groups");
  if (q != BuildQuantMetadata())
    throw std::runtime_error(who + ": the baseline's __metadata__.quant " + q.dump() +
                             " differs from this build's " + BuildQuantMetadata().dump());
  // Last (it reads the whole file): the bytes are still the ones its conversion wrote and digested.
  b.VerifyData(bg.at(r4dx_convert::kReuseDataKey).get<std::string>(), threads);
}

// The per-job decision: which emit jobs this run copies from the baseline and which it runs.
struct ReusePlan {
  std::vector<bool> copy;                       // per plan/emit job
  std::vector<std::string> recomputed_linears;  // container order
  std::vector<std::string> recomputed_why;      // same order: "<baseline set> -> <this run's set>"
  int64_t tensors_copied = 0, tensors_recomputed = 0;
  uint64_t bytes_copied = 0, bytes_recomputed = 0;
};

// `job_tensors[j]` = the writer's plan indices job j planned; `linear_jobs` = job -> container base,
// for every add_linear (and the MTP draft head) job; `layouts_now` = base -> the LayoutSet this run
// resolved for it (after --w4a16-group-rule and --keep-bf16), which this run's guard records as
// reuse_guard.linears. A linear is AFFECTED when its LayoutSetId here differs from the one the
// baseline's guard records for it: a rule or a keep in either run that the other lacks, or two
// different groups. An affected job runs as usual -- all its layouts -- and every other job is copied,
// after its every tensor was checked to exist in the baseline under the same name with dtype U8, the
// same shape and the same size.
//
// Nothing is inferred from either run's regex text, and the baseline's record is not trusted on its
// own: it must name exactly this run's linears, each with a canonical layout set; the baseline's other
// records of the same decisions (quant.w4a16.groups, which the runtime reads, and
// r4dx_convert_run.keep_bf16_linears) must agree with it; an affected linear's baseline tensors must
// be exactly the ones its recorded set names; and the baseline must hold nothing else. Any
// inconsistency refuses the reuse (the guard matched, so it means the guard missed something --
// nothing is trusted then).
ReusePlan PlanReuse(const r4dx_convert::BaselineContainer& b, const ContainerWriter& writer,
                    const std::vector<std::pair<size_t, size_t>>& job_tensors,
                    const std::map<size_t, std::string>& linear_jobs,
                    const std::map<std::string, LayoutSet>& layouts_now) {
  using r4dx_convert::kReuseLinearsKey;
  using r4dx_convert::LayoutSetId;
  const std::string who = "--reuse-tensors-from " + b.path;
  const std::string inconsistent = " -- refusing an inconsistent baseline";
  const r4dx_convert::SafetensorsReader& r = *b.reader;
  for (const auto& kv : linear_jobs) {
    if (!layouts_now.count(kv.second))
      throw std::logic_error("PlanReuse: linear job '" + kv.second + "' has no resolved LayoutSet");
  }

  // (1) The baseline's per-linear record: exactly this run's linears, each a canonical layout set.
  const nlohmann::json& bg = b.metadata.at("r4dx_convert_run").at("reuse_guard");
  const auto rec = bg.find(kReuseLinearsKey);
  if (rec == bg.end() || !rec->is_object())
    throw std::runtime_error(who + ": the baseline's reuse_guard." + kReuseLinearsKey +
                             " is missing or not an object" + inconsistent);
  std::map<std::string, LayoutSet> then;
  for (auto it = rec->begin(); it != rec->end(); ++it) {
    const std::string field = "reuse_guard." + std::string(kReuseLinearsKey) + "." + it.key() + ": ";
    const auto now = layouts_now.find(it.key());
    if (now == layouts_now.end())
      throw std::runtime_error(who + ": " + field + "baseline " + it.value().dump() +
                               ", this run (absent) -- the baseline records a linear this run does "
                               "not write" + inconsistent);
    LayoutSet ls;
    if (!it.value().is_string() || !r4dx_convert::ParseLayoutSetId(it.value().get<std::string>(), &ls))
      throw std::runtime_error(who + ": " + field + "the baseline records " + it.value().dump() +
                               ", which is not a layout set this converter writes (this run: \"" +
                               LayoutSetId(now->second) + "\")" + inconsistent);
    then.emplace(it.key(), ls);
  }
  for (const auto& kv : layouts_now) {
    if (!then.count(kv.first))
      throw std::runtime_error(who + ": reuse_guard." + std::string(kReuseLinearsKey) + "." + kv.first +
                               ": baseline (absent), this run \"" + LayoutSetId(kv.second) + "\"" +
                               inconsistent);
  }

  // (2) The baseline's other records of the same decisions agree with (1): its group map (what the
  // runtime dispatches on) and its keep list.
  const nlohmann::json& w16 = b.metadata.at("quant").at("w4a16");
  std::map<std::string, int> groups_then;
  if (w16.contains("groups")) {
    if (!w16["groups"].is_object())
      throw std::runtime_error(who + ": the baseline's quant.w4a16.groups is not an object");
    for (auto it = w16["groups"].begin(); it != w16["groups"].end(); ++it) {
      if (!it.value().is_number_integer())
        throw std::runtime_error(who + ": the baseline's quant.w4a16.groups[\"" + it.key() +
                                 "\"] is not an integer");
      if (!then.count(it.key()))
        throw std::runtime_error(who + ": the baseline's quant.w4a16.groups lists '" + it.key() +
                                 "', which this run does not write as a linear");
      groups_then[it.key()] = it.value().get<int>();
    }
  }
  for (const auto& kv : then) {
    const bool mapped = kv.second.w4a16 && kv.second.w4a16_group != r4dx_convert::kW4A16Group;
    const auto g = groups_then.find(kv.first);
    if (mapped == (g != groups_then.end()) && (!mapped || g->second == kv.second.w4a16_group)) continue;
    throw std::runtime_error(
        who + ": the baseline's quant.w4a16.groups maps '" + kv.first + "' to " +
        (g == groups_then.end() ? "nothing (the default group " + std::to_string(r4dx_convert::kW4A16Group) + ")"
                                : "group " + std::to_string(g->second)) +
        " but its reuse_guard." + kReuseLinearsKey + " records \"" + LayoutSetId(kv.second) + "\"" +
        inconsistent);
  }
  const nlohmann::json& brun = b.metadata.at("r4dx_convert_run");
  if (brun.contains("keep_bf16_linears")) {
    const nlohmann::json& kept = brun["keep_bf16_linears"];
    if (!kept.is_array())
      throw std::runtime_error(who + ": the baseline's r4dx_convert_run.keep_bf16_linears is not an array");
    for (const auto& k : kept) {
      const auto t = k.is_string() ? then.find(k.get<std::string>()) : then.end();
      if (t == then.end() || LayoutSetId(t->second) != LayoutSetId(r4dx_convert::KeptBf16LayoutSet()))
        throw std::runtime_error(
            who + ": the baseline's r4dx_convert_run.keep_bf16_linears lists " + k.dump() + " but its "
            "reuse_guard." + kReuseLinearsKey + " records " +
            (t == then.end() ? std::string("no such linear") : "\"" + LayoutSetId(t->second) + "\"") +
            inconsistent);
    }
  }

  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? "," : "") + std::to_string(s[i]);
    return o + "]";
  };

  // (3) Per job.
  ReusePlan p;
  p.copy.assign(job_tensors.size(), false);
  std::set<std::string> accounted;
  for (size_t j = 0; j < job_tensors.size(); ++j) {
    const auto lj = linear_jobs.find(j);
    if (lj != linear_jobs.end()) {
      const std::string& base = lj->second;
      const LayoutSet& ls_now = layouts_now.at(base);
      const LayoutSet& ls_then = then.at(base);
      const std::string id_now = LayoutSetId(ls_now), id_then = LayoutSetId(ls_then);
      if (id_now != id_then) {
        p.recomputed_linears.push_back(base);
        p.recomputed_why.push_back(id_then + " -> " + id_now);
        // The baseline's own tensors of this linear: exactly the ones its record names (a leftover
        // or a missing one is refused by the directory check below / here).
        for (const std::string& name : r4dx_convert::LinearLayoutTensorNames(base, ls_then)) {
          if (!r.Has(name))
            throw std::runtime_error(who + ": the baseline records '" + base + "' as \"" + id_then +
                                     "\" (reuse_guard." + kReuseLinearsKey + ") but has no tensor '" +
                                     name + "'" + inconsistent);
          accounted.insert(name);
        }
        // This job's other tensors (the MTP draft head's vocab_ids) are recomputed with it; the
        // baseline's copy of one is its own, not a stray.
        const std::vector<std::string> own = r4dx_convert::LinearLayoutTensorNames(base, ls_now);
        for (size_t t = job_tensors[j].first; t < job_tensors[j].second; ++t) {
          const std::string& name = writer.PlannedName(t);
          if (std::find(own.begin(), own.end(), name) == own.end() && r.Has(name)) accounted.insert(name);
          ++p.tensors_recomputed;
          p.bytes_recomputed += writer.PlannedBytes(t);
        }
        continue;
      }
    }
    for (size_t t = job_tensors[j].first; t < job_tensors[j].second; ++t) {
      const std::string& name = writer.PlannedName(t);
      if (!r.Has(name))
        throw std::runtime_error(who + ": this run writes '" + name + "' but the baseline has no "
                                 "such tensor -- refusing an inconsistent baseline");
      const auto& m = r.Meta(name);
      const uint64_t bytes = m.end - m.begin;
      if (m.dtype != "U8" || m.shape != writer.PlannedShape(t) || bytes != writer.PlannedBytes(t)) {
        throw std::runtime_error(
            who + ": tensor '" + name + "' is " + m.dtype + " " + shape_str(m.shape) + " " +
            std::to_string(bytes) + " B in the baseline but U8 " + shape_str(writer.PlannedShape(t)) +
            " " + std::to_string(writer.PlannedBytes(t)) + " B in this run -- refusing an "
            "inconsistent baseline");
      }
      accounted.insert(name);
      ++p.tensors_copied;
      p.bytes_copied += bytes;
    }
    p.copy[j] = true;
  }
  for (const auto& name : r.Names()) {
    if (!accounted.count(name))
      throw std::runtime_error(who + ": the baseline has tensor '" + name + "', which this run "
                               "does not write and its record does not name" + inconsistent);
  }
  return p;
}

// ---- normal mode --------------------------------------------------------------------------

int RunConvert(const AppArgs& args) {
  const int threads = ResolveThreads(args.threads);
  const std::string config_text = ReadFile(args.input + "\\config.json");
  const nlohmann::json config = nlohmann::json::parse(config_text);
  const nlohmann::json& text_cfg = config.at("text_config");

  const int hidden = text_cfg.at("hidden_size").get<int>();
  const int num_layers_total = text_cfg.at("num_hidden_layers").get<int>();
  const int kv_heads = text_cfg.at("num_key_value_heads").get<int>();
  const std::vector<std::string> layer_types =
      text_cfg.at("layer_types").get<std::vector<std::string>>();
  const int layers = args.layers > 0 ? std::min(args.layers, num_layers_total) : num_layers_total;
  const bool full_run = (layers == num_layers_total);
  const bool do_vision = args.vision >= 0 ? (args.vision == 1) : full_run;
  const bool do_mtp = args.mtp >= 0 ? (args.mtp == 1) : full_run;

  LayoutSet layouts = ParseLayoutList(args.layouts_spec, /*bf16_default_on=*/true);
  LayoutSet lm_head_layouts = ParseLayoutList(args.lm_head_spec, /*bf16_default_on=*/false);
  if (args.no_bf16) {
    // --layouts/--lm-head have no way to say "and NOT bf16" (bf16 defaults on for body weights,
    // and "none" in a spec is a no-op rather than a subtractive token) -- --no-bf16 is the
    // explicit override so a quantized-only conversion is possible (review finding, minor).
    layouts.bf16 = false;
    // An explicit `--lm-head bf16` (or `4bit+bf16`) is a deliberate request for a bf16 lm_head in
    // an otherwise quantized-only container (rung 4 follow-up: the 4-bit lm_head is where the
    // vocab-tail KL loss sits) -- only the DEFAULT lm_head spec is subject to --no-bf16.
    if (!args.lm_head_spec_explicit) lm_head_layouts.bf16 = false;
  }

  std::cout << "[r4dx-convert] input=" << args.input << " output=" << args.output << "\n"
            << "[r4dx-convert] hidden=" << hidden << " layers=" << layers << "/" << num_layers_total
            << " threads=" << threads << " vision=" << (do_vision ? "on" : "off")
            << " mtp=" << (do_mtp ? "on" : "off") << "\n";

  // The BYTE LAYOUT is identical in both modes (docs/container-format.md "How the quantized values
  // are chosen"); only the (scale, zero) values differ.
  const r4dx_convert::QuantMode quant_mode = ParseQuantMode(args.quant);
  std::cout << "[r4dx-convert] quant=" << args.quant
            << (args.imatrix.empty() ? " (unweighted MSE)" : " (imatrix-weighted)") << "\n";
  ImatrixSource imatrix(args.imatrix, quant_mode);

  // --keep-bf16 (keep_bf16.hpp): constructed here so an invalid regex fails before the first shard
  // is opened, not 250 s into the emit pass.
  r4dx_convert::KeepBf16Selector keep_bf16(args.keep_bf16);
  if (keep_bf16.Enabled())
    std::cout << "[r4dx-convert] keep-bf16=" << keep_bf16.Pattern()
              << " (matching linears written as <base>.bf16.w only)\n";

  // --w4a16-group-rule (w4a16_groups.hpp): rules parsed and checked against the kernel's own
  // instantiations here; each linear's group is resolved once in add_linear, and its shape and the
  // bf16-companion guard are checked in the planning pass below.
  r4dx_convert::W4a16GroupRules w4a16_groups(args.w4a16_group_rules);
  if (w4a16_groups.Enabled()) {
    ValidateKernelRuleGroups(w4a16_groups);
    for (const auto& r : w4a16_groups.Rules())
      std::cout << "[r4dx-convert] w4a16-group-rule " << r.spec << " (regex '" << r.pattern
                << "' -> group " << r.group << "; first match wins, default "
                << r4dx_convert::kW4A16Group << ")\n";
  }

  // --ldlq (LdlqSource): regex compiled and hessian.json parsed here, before the first shard is
  // opened; the per-linear manifest checks run in the planning pass below.
  LdlqSource ldlq(args.ldlq, args.hessian_dir, args.ldlq_damp);
  if (ldlq.Enabled())
    std::cout << "[r4dx-convert] ldlq=" << ldlq.Pattern() << " damp=" << ldlq.Damp()
              << " hessian-dir=" << ldlq.Dir()
              << " (matching linears: LDLQ error-feedback rounding; --quant/--imatrix ignored for "
                 "them)\n";

  // --rotate (RotationSource): Q / Hb generated here from the seed, and the checkpoint's shape checked
  // against them, before the first shard is opened.
  RotationSource rot(args.rotate, args.rotation_seed, args.rotation_seed_explicit, text_cfg);
  if (rot.Enabled()) {
    std::cout << "[r4dx-convert] rotate=" << rot.KindName() << " seed=0x" << std::hex << rot.Seed()
              << std::dec << " (text layers folded; norms stored as 0; rotation.* tensors + "
              << "__metadata__.rotation written)\n";
    if (!args.imatrix.empty())
      std::cout << "[r4dx-convert] rotate: --imatrix vectors of rotated linears are carried into the "
                   "new basis under the diagonal model (rotation.hpp) -- use --ldlq for the exact "
                   "Hessian\n";
  }

  // --trellis-from (trellis_import.hpp, docs/trellis-kernel.md 3): the manifest is read and its
  // whole-manifest checks (format, completeness, codebook, basis, config sha, the
  // --trellis-manifest-sha256 pin) run here, before the first shard is opened; the per-linear
  // checks run as add_linear resolves each body linear and in the planning pass, the file hashes
  // right after it.
  r4dx_convert::trellis::TrellisSource trellis(args.trellis, config_text);
  if (trellis.Enabled()) {
    std::cout << "[r4dx-convert] trellis-from=" << trellis.ManifestPath() << " (sha256 "
              << trellis.ManifestSha256() << ", " << (trellis.IsMix() ? "mix" : "quantize-model")
              << " form): every text-layer body linear is imported in the trellis layout only; "
                 "--layouts / --lm-head / --ldlq / --w4a16-group-rule govern the rest\n";
    if (!trellis.VerifyFull())
      std::cerr << "[r4dx-convert] WARNING: --trellis-verify none (debug build): the imported "
                   "tensors are NOT checked against the checkpoint; the header records "
                   "\"not run\"\n";
  }

  const bool have_kv_calib = !args.kv_calib.empty();
  nlohmann::json kv_calib_json;
  if (have_kv_calib) {
    kv_calib_json = nlohmann::json::parse(ReadFile(args.kv_calib));
    std::cout << "[r4dx-convert] kv-calib=" << args.kv_calib << " (" << kv_calib_json.size()
              << " layer(s) in file)\n";
  }

  // --record-reuse-guard / --reuse-tensors-from (docs/quant2.md 5.2): the guard is computed -- and a
  // baseline checked against it -- before the first shard is opened, so a refused baseline costs the
  // hashing (disk-bound: the whole checkpoint is read once), not a conversion.
  const bool want_guard = args.record_reuse_guard || !args.reuse_from.empty();
  nlohmann::json reuse_guard;
  std::unique_ptr<r4dx_convert::BaselineContainer> baseline;
  if (want_guard) {
    if (!args.reuse_from.empty()) baseline = OpenReuseBaseline(args);  // header checks only
    const auto tg = std::chrono::steady_clock::now();
    reuse_guard = BuildReuseGuard(args, config_text, layers, do_vision, do_mtp, layouts,
                                  lm_head_layouts, ldlq, rot, threads);
    const nlohmann::json& hf = reuse_guard["inputs"]["hessian_files"];
    std::cout << "[r4dx-convert] reuse guard: binary, checkpoint ("
              << reuse_guard["checkpoint"]["shards"].size() << " shard(s)), "
              << (hf.is_object() ? hf.size() : 0) << " Hessian file(s) and input files hashed in "
              << SecondsBetween(tg, std::chrono::steady_clock::now()) << " s\n";
    if (baseline) {
      const auto tb = std::chrono::steady_clock::now();
      CheckReuseBaseline(*baseline, reuse_guard, threads);
      std::cout << "[r4dx-convert] reuse: baseline " << args.reuse_from
                << " matches this run's guard (every field; the per-linear layout sets -- "
                   "--w4a16-group-rule, --keep-bf16 -- may differ), and its data matches its "
                   "recorded data_sha256 (read in "
                << SecondsBetween(tb, std::chrono::steady_clock::now()) << " s)\n";
    }
  }

  ShardedModel model(args.input);
  ContainerWriter writer;

  // Every add_* below pushes exactly ONE plan job and ONE emit job, so job j's emit writes exactly the
  // tensors job j's plan registered (checked when planning runs). `linear_jobs`: job index -> container
  // base for the jobs that write a linear's layouts (add_linear, the MTP draft head) -- the only jobs
  // a --w4a16-group-rule or --keep-bf16 can change, so the only ones --reuse-tensors-from may have to
  // recompute. `linear_layouts`: base -> the LayoutSet it is written in, resolved once where the job
  // is added (the guard records it as reuse_guard.linears; PlanReuse compares it).
  std::vector<std::function<void()>> plan_jobs, emit_jobs;
  std::map<size_t, std::string> linear_jobs;
  std::map<std::string, LayoutSet> linear_layouts;
  auto note_linear = [&](const std::string& base, const LayoutSet& ls) {
    if (!linear_layouts.emplace(base, ls).second)
      throw std::logic_error("RunConvert: linear '" + base + "' added twice");
    linear_jobs[plan_jobs.size()] = base;
  };

  auto add_bf16 = [&](std::string hf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &model, hf_name, container_name]() {
      const auto& meta = model.Meta(hf_name);
      std::vector<int64_t> shape = meta.shape;
      shape.push_back(2);
      writer.Plan(container_name, shape, static_cast<uint64_t>(meta.ElemCount()) * 2);
    });
    emit_jobs.push_back([&writer, &model, hf_name, container_name]() {
      auto bytes = r4dx_convert::CopyRawBf16(model, hf_name);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  // --rotate: a zero-centred layer norm whose (1 + w) has been folded into the next linear(s) is
  // stored as 0 (bf16 +0.0, all bytes zero) in the checkpoint tensor's shape, so the kernel computes
  // rms(x) * 1 -- which commutes with Q. Without --rotate this is add_bf16, byte for byte.
  // Rotated, the tensor is renamed `<name>.rotated` on purpose: a pre-quant2 binary never reads
  // __metadata__.rotation, and the missing bare name is what stops it from running rotated weights
  // in the unrotated basis.
  auto add_norm = [&](std::string hf_name, std::string container_name) {
    if (!rot.Enabled()) {
      add_bf16(hf_name, container_name);
      return;
    }
    container_name += ".rotated";
    plan_jobs.push_back([&writer, &model, hf_name, container_name]() {
      const auto& meta = model.Meta(hf_name);
      std::vector<int64_t> shape = meta.shape;
      shape.push_back(2);
      writer.Plan(container_name, shape, static_cast<uint64_t>(meta.ElemCount()) * 2);
    });
    emit_jobs.push_back([&writer, &model, hf_name, container_name]() {
      const std::vector<uint8_t> zeros(static_cast<size_t>(model.Meta(hf_name).ElemCount()) * 2, 0);
      writer.WriteTensor(container_name, zeros.data(), zeros.size());
    });
  };
  // A bf16-only linear (gdn.in_proj_a/b) that --rotate folds: read as fp32, fold, round to bf16
  // ONCE. Unfolded (the default) it is add_bf16's raw byte copy, so nothing changes without --rotate.
  auto add_bf16_linear = [&](std::string hf_name, std::string container_name, LinearFold fold) {
    if (fold.kind == LinearFold::kNone) {
      add_bf16(hf_name, container_name);
      return;
    }
    plan_jobs.push_back([&writer, &model, &rot, hf_name, container_name, fold]() {
      const auto& meta = model.Meta(hf_name);
      if (meta.shape.size() != 2)
        throw std::runtime_error("--rotate: '" + hf_name + "' is not a 2-D linear weight");
      rot.CheckPlan(fold, container_name, meta.shape[0], meta.shape[1], model);
      writer.Plan(container_name, {meta.shape[0], meta.shape[1], 2},
                  static_cast<uint64_t>(meta.ElemCount()) * 2);
    });
    emit_jobs.push_back([&writer, &model, &rot, hf_name, container_name, fold, threads]() {
      std::vector<float> w = r4dx_convert::ReadTensorAsFloat(model, hf_name);
      const auto& meta = model.Meta(hf_name);
      rot.Fold(fold, rot.ReadNorm(fold, model), w, meta.shape[0], meta.shape[1], threads);
      auto bytes = r4dx_convert::EncodeBf16(w);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  // Raw fp32 values the converter itself produced (the rotation.* tensors), `shape` without the
  // trailing element-width dim, which is appended exactly as add_fp32_widen does.
  auto add_fp32_values = [&](std::string container_name, std::vector<int64_t> shape,
                             const std::vector<float>* values) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    if (n != static_cast<int64_t>(values->size()))
      throw std::logic_error("add_fp32_values: shape/value count mismatch for " + container_name);
    shape.push_back(4);
    plan_jobs.push_back([&writer, container_name, shape, n]() {
      writer.Plan(container_name, shape, static_cast<uint64_t>(n) * 4);
    });
    emit_jobs.push_back([&writer, container_name, values]() {
      auto bytes = r4dx_convert::EncodeFp32(*values);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  auto add_fp32_widen = [&](std::string hf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &model, hf_name, container_name]() {
      const auto& meta = model.Meta(hf_name);
      std::vector<int64_t> shape = meta.shape;
      shape.push_back(4);
      writer.Plan(container_name, shape, static_cast<uint64_t>(meta.ElemCount()) * 4);
    });
    emit_jobs.push_back([&writer, &model, hf_name, container_name]() {
      auto w = r4dx_convert::ReadTensorAsFloat(model, hf_name);
      auto bytes = r4dx_convert::EncodeFp32(w);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  // `layer_idx` is the checkpoint's text-layer index (matches kv_calibrate.py's --layer);
  // `calib_applicable` should be false for any layer kv_calibrate.py never covers by construction
  // (the MTP layer has no calibration entry -- it isn't one of the 64 text-layer indices) so that
  // case falls back to 1.0 silently instead of emitting a spurious "no entry for layer N" warning
  // that would actually be about an unrelated text layer sharing the same numeric index.
  auto add_descale = [&](std::string container_name, int n_kv, int layer_idx, const char* kind,
                          bool calib_applicable) {
    plan_jobs.push_back([&writer, container_name, n_kv]() {
      writer.Plan(container_name, {n_kv, 4}, static_cast<uint64_t>(n_kv) * 4);
    });
    emit_jobs.push_back([&writer, container_name, n_kv, layer_idx, kind, calib_applicable,
                          have_kv_calib, &kv_calib_json]() {
      auto result = r4dx_convert::ResolveKvDescale(
          kv_calib_json, have_kv_calib && calib_applicable, layer_idx, n_kv, kind);
      if (!result.warning.empty()) std::cerr << "[r4dx-convert] WARNING: " << result.warning << "\n";
      auto bytes = r4dx_convert::EncodeFp32(result.values);
      writer.WriteTensor(container_name, bytes.data(), bytes.size());
    });
  };
  // The one emit path for every linear that may be quantized (add_linear and the MTP draft head), so
  // the LDLQ-vs-imatrix choice cannot drift between the two sites. `use_ldlq` is the caller's
  // already-resolved decision. Every add_linear fuses on the OUTPUT axis only (mlp.gate_up is the
  // only fusion and both halves share K), so one length-K importance vector -- and one K x K Hessian
  // -- per container base is well defined; tools/reference/imatrix_capture.py and
  // hessian_capture.py assert the same thing from the other side.
  //
  // `w` arrives ALREADY folded by --rotate (the caller did it); `fold` / `norm` only say how this
  // linear's input basis changed, so the imatrix vector and the LDLQ Hessian can follow it. With
  // `fold.kind == kNone` (always, without --rotate) both paths are exactly the pre-rotation code.
  // `use_rms` (the caller's, like `use_ldlq`; only ever true for a rotated in-projection): round
  // against Q^T H_rms Q from the base's "rms_keys" file instead of dividing the norm out of H.
  auto emit_linear = [&writer, &imatrix, &ldlq, &rot, quant_mode, threads](
                         const std::string& container_base, const std::vector<float>& w, int64_t N,
                         int64_t K, const LayoutSet& ls, bool use_ldlq, bool use_rms,
                         const LinearFold& fold, const std::vector<float>& norm) {
    if (!use_ldlq) {
      r4dx_convert::QuantOptions opts = imatrix.For(container_base, K);
      std::vector<float> rotated_importance;  // must outlive EmitLinearLayouts (opts points at it)
      if (fold.RotatesInput() && !opts.importance.empty()) {
        rotated_importance = rot.Importance(fold, norm, opts.importance, K);
        opts.importance.data = rotated_importance.data();
      }
      r4dx_convert::EmitLinearLayouts(writer, container_base, w, static_cast<int>(N),
                                       static_cast<int>(K), ls, threads, opts);
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool reused = false;
    const std::unique_ptr<r4dx_convert::HessianTransform> xf = rot.HessianFor(fold, norm, use_rms);
    // A reference into the store's one-entry cache: valid until the next Factor() call, which is
    // the next LDLQ'd linear's emit -- i.e. strictly after this EmitLinearLayouts returns.
    const r4dx_convert::LdlqFactor& f =
        ldlq.Factor(container_base, K, threads, &reused, xf.get(), use_rms);
    const auto t1 = std::chrono::steady_clock::now();
    r4dx_convert::QuantOptions opts;
    opts.mode = quant_mode;  // ignored once opts.ldlq is set; kept so the struct reads truthfully
    opts.ldlq = &f;
    r4dx_convert::EmitLinearLayouts(writer, container_base, w, static_cast<int>(N),
                                     static_cast<int>(K), ls, threads, opts);
    const auto t2 = std::chrono::steady_clock::now();
    ldlq.Record(container_base, N, K, SecondsBetween(t0, t1), SecondsBetween(t1, t2), f, reused,
                std::cout, use_rms);
  };

  // `fold` is what --rotate does to this linear (RotationSource::In / Out); the default, kNone, is
  // what every call outside the text-layer loop -- and every call without --rotate -- gets.
  auto add_linear = [&](std::vector<std::string> hf_names, std::string container_base,
                         LayoutSet requested, LinearFold fold = LinearFold{}) {
    // --keep-bf16 and --ldlq decide per container base name, which is known here -- so plan and
    // emit cannot disagree about what this linear is (the two lambdas below capture the SAME
    // resolved `ls` / `use_ldlq`, rather than each re-evaluating a regex). --keep-bf16 wins: a kept
    // linear has no quantized layout left to round, and neither has one whose run asked for bf16
    // only (`--lm-head bf16`), so LDLQ applies only when some 4-bit layout survives.
    // --w4a16-group-rule resolves the w4a16 group into `requested` first, so --keep-bf16's "would
    // have been" bytes are priced at the group the linear would really have had.
    requested = w4a16_groups.Apply(container_base, requested);
    const bool kept = keep_bf16.Matches(container_base);
    // --trellis-from (docs/trellis-kernel.md 3.4): a body linear the manifest covers is written in
    // the trellis layout ONLY, whatever --layouts says -- so it has no 4-bit layout for LDLQ or a
    // group rule to act on, and no `.bf16.w` next to it (the old-binary guard). --keep-bf16 still
    // wins; a kept linear's "would have been" bytes are then priced at the trellis layout it
    // replaced.
    bool imported = false;
    if (trellis.Enabled() && r4dx_convert::trellis::IsBodyLinearBase(container_base)) {
      LayoutSet tls;
      imported = trellis.Resolve(container_base, hf_names, kept, &tls, std::cout);
      if (tls.trellis) requested = tls;
    }
    const LayoutSet ls = kept ? r4dx_convert::KeptBf16LayoutSet() : requested;
    const bool ldlq_matched = ldlq.Matches(container_base);
    const bool use_ldlq = ldlq_matched && HasQuantizedLayout(ls);
    // A rotated in-projection rounds against the weightless rms Hessian when hessian.json has one
    // for it ("rms_keys"): its folded input is exactly rms(x) Q, so H' = Q^T H_rms Q needs no
    // division by (1 + w). Without one it falls back to dividing the post-norm H, which the plan job
    // below checks is possible. Never for an unrotated linear (fold.kind is kNone without --rotate),
    // so an unrotated container reads exactly the Hessians it did before rms_keys existed.
    const bool rotated_in = fold.kind == LinearFold::kIn;
    const bool use_rms = use_ldlq && rotated_in && ldlq.HasRms(container_base);
    note_linear(container_base, ls);
    plan_jobs.push_back([&writer, &model, &keep_bf16, &ldlq, &rot, &w4a16_groups, &trellis,
                          hf_names, container_base, ls, requested, kept, ldlq_matched, use_ldlq,
                          rotated_in, use_rms, fold, imported]() {
      int64_t N = 0, K = 0;
      std::vector<int64_t> part_n;
      for (auto& n : hf_names) {
        const auto& m = model.Meta(n);
        N += m.shape[0];
        K = m.shape[1];
        part_n.push_back(m.shape[0]);
      }
      rot.CheckPlan(fold, container_base, N, K, model);
      // The manifest's [n, k] of every part against the checkpoint's (docs/trellis-kernel.md 3.3
      // step 4).
      if (imported) trellis.Plan(container_base, part_n, K);
      if (kept) {
        keep_bf16.Record(container_base, static_cast<int>(N), static_cast<int>(K), requested,
                         std::cout);
      }
      // The manifest + file check lives HERE, in planning, so a missing key / wrong K / missing or
      // truncated .hess / group not dividing the LDLQ block aborts before
      // FinalizeHeader and before the first weight is read. (A rotated in-projection without an rms
      // Hessian also has its [hidden] norm vector read here, to refuse a (1 + w) it cannot divide.)
      w4a16_groups.Plan(container_base, N, K, ls);
      if (use_ldlq) {
        ldlq.Plan(container_base, K, ls, use_rms);
        if (rotated_in && !use_rms) rot.CheckNormDivisible(fold, container_base, model);
      } else if (ldlq_matched) {
        ldlq.NoteSkipped();
      }
      r4dx_convert::PlanLinearLayouts(writer, container_base, static_cast<int>(N),
                                       static_cast<int>(K), ls);
    });
    if (imported) {
      // Every byte of an imported linear comes from the oracle files: the checkpoint weight is not
      // read here (the reconstruction check reads it once the container is closed).
      emit_jobs.push_back([&writer, &trellis, container_base, threads]() {
        trellis.Emit(writer, container_base, threads);
      });
      return;
    }
    emit_jobs.push_back([&model, &emit_linear, &rot, hf_names, container_base, ls, use_ldlq,
                         use_rms, fold, threads]() {
      std::vector<float> w;
      int64_t K = 0;
      for (auto& n : hf_names) {
        auto part = r4dx_convert::ReadTensorAsFloat(model, n);
        K = model.Meta(n).shape[1];
        w.insert(w.end(), part.begin(), part.end());
      }
      const int64_t N = static_cast<int64_t>(w.size()) / K;
      // The fold is per ROW on the K side (in-projections, the q2ab Hadamard) and per COLUMN on the
      // N side (out-projections), so the output-axis fusions -- mlp.gate_up's gate|up rows, attn.qg's
      // per-head query/gate interleave -- need no special case. Every layout of this linear,
      // --keep-bf16's bf16-only one included, is then produced from the folded fp32.
      const std::vector<float> norm = rot.ReadNorm(fold, model);
      rot.Fold(fold, norm, w, N, K, threads);
      emit_linear(container_base, w, N, K, ls, use_ldlq, use_rms, fold, norm);
    });
  };

  for (int i = 0; i < layers; ++i) {
    const std::string hf = "model.language_model.layers." + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    // --rotate folds (all kNone without it): input_layernorm's (1 + w) goes into this layer's token
    // mixer in-projections, post_attention_layernorm's into mlp.gate_up, and Q^T (plus, for q2ab,
    // the per-site Hadamard) into the three out-projections. Plain variables on purpose -- see
    // LinearFold.
    const LinearFold mixer_in_fold = rot.In(hf + "input_layernorm.weight");
    const LinearFold mlp_in_fold = rot.In(hf + "post_attention_layernorm.weight");
    const LinearFold o_fold = rot.Out(HadSite::kO);
    const LinearFold gdn_out_fold = rot.Out(HadSite::kGdnOut);
    const LinearFold down_fold = rot.Out(HadSite::kDown);
    add_norm(hf + "input_layernorm.weight", base + "input_layernorm");
    add_norm(hf + "post_attention_layernorm.weight", base + "post_attention_layernorm");

    if (layer_types.at(i) == "full_attention") {
      // q_proj is already the fused query+output-gate matrix in this checkpoint (HF's
      // Qwen3_5Attention builds it as Linear(hidden, num_heads*head_dim*2)); attn.qg is a direct
      // copy/quantize of it, no fusion needed at convert time.
      add_linear({hf + "self_attn.q_proj.weight"}, base + "attn.qg", layouts, mixer_in_fold);
      // attn.k/v (R1, docs/r9700.md): join the quantized-linear family -- 1024x5120 x16 layers,
      // 0.336 GB/token, previously forced bf16 regardless of --layout. Same LayoutSet as every
      // other body linear, so a run without --layouts w4a16 (say) simply omits that variant here
      // too, exactly like attn.qg/o already do.
      add_linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", layouts, mixer_in_fold);
      add_linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", layouts, mixer_in_fold);
      add_linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", layouts, o_fold);
      add_bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm");
      add_bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm");
      add_descale(base + "attn.k_descale", kv_heads, i, "k", /*calib_applicable=*/true);
      add_descale(base + "attn.v_descale", kv_heads, i, "v", /*calib_applicable=*/true);
    } else {
      add_linear({hf + "linear_attn.in_proj_qkv.weight"}, base + "gdn.in_proj_qkv", layouts,
                 mixer_in_fold);
      // gdn.in_proj_z (R1, docs/r9700.md): joins the quantized-linear family -- 6144x5120 x48
      // layers, 3.02 GB/token, the single largest bf16-only tensor in the model (more bytes/token
      // than the entire quantized GDN weight set combined). in_proj_a/in_proj_b stay bf16 (too
      // small to matter, feed the decay path).
      add_linear({hf + "linear_attn.in_proj_z.weight"}, base + "gdn.in_proj_z", layouts,
                 mixer_in_fold);
      add_bf16_linear(hf + "linear_attn.in_proj_b.weight", base + "gdn.in_proj_b", mixer_in_fold);
      add_bf16_linear(hf + "linear_attn.in_proj_a.weight", base + "gdn.in_proj_a", mixer_in_fold);
      add_bf16(hf + "linear_attn.conv1d.weight", base + "gdn.conv1d_weight");
      add_fp32_widen(hf + "linear_attn.A_log", base + "gdn.A_log");
      add_fp32_widen(hf + "linear_attn.dt_bias", base + "gdn.dt_bias");
      add_bf16(hf + "linear_attn.norm.weight", base + "gdn.norm_weight");
      add_linear({hf + "linear_attn.out_proj.weight"}, base + "gdn.out_proj", layouts,
                 gdn_out_fold);
    }
    add_linear({hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"}, base + "mlp.gate_up",
                layouts, mlp_in_fold);
    add_linear({hf + "mlp.down_proj.weight"}, base + "mlp.down", layouts, down_fold);
  }

  // Everything from here on is outside the rotated stack and is never folded: the runtime applies
  // x Q after the embedding gather and x Q^T before final_norm (docs/quant2.md section 3).
  add_bf16("model.language_model.embed_tokens.weight", "text.embed_tokens");
  add_bf16("model.language_model.norm.weight", "text.final_norm");
  add_linear({"lm_head.weight"}, "lm_head", lm_head_layouts);

  // --rotate: the transforms themselves, fp32, exactly the values every fold above used (the runtime
  // reads these; it never regenerates them from the seed). No .{layout} suffix, like every other
  // single-layout tensor. Written only when rotated -- an unrotated container has none of them.
  if (rot.Enabled()) {
    const r4dx_convert::RotationSet& rs = rot.Set();
    add_fp32_values("rotation.signs", {rs.q.hidden}, &rs.q.signs);
    add_fp32_values("rotation.mix5", {rs.q.nblk, rs.q.nblk}, &rs.q.mix);
    if (rot.Hadamard()) {
      add_fp32_values("rotation.had_down_signs", {rs.had_down.K}, &rs.had_down.signs);
      add_fp32_values("rotation.had_o_signs", {rs.had_o.K}, &rs.had_o.signs);
      add_fp32_values("rotation.had_gdn_out_signs", {rs.had_gdn_out.K}, &rs.had_gdn_out.signs);
    }
  }

  if (do_vision) {
    for (const auto& name : model.AllNames()) {
      if (StartsWith(name, "model.visual.")) {
        add_bf16(name, "vision." + name.substr(std::string("model.visual.").size()));
      }
    }
  }

  if (do_mtp) {
    // Review finding (major): mtp.layers.0 is a COMPLETE full-attention decoder layer -- verified
    // against the real checkpoint's model.safetensors.index.json/shard header, not assumed --
    // self_attn.{q,k,v,o}_proj (q_proj [12288,5120], fused q+gate exactly like a text full-
    // attention layer), q_norm/k_norm, mlp.{gate,up,down}_proj ([17408,5120]x2,[5120,17408]), both
    // layernorms. It is NOT a small bespoke head, so it gets the identical add_linear/add_bf16
    // routing a text full_attention layer gets (w4a16/bf16 on qg/o and the MLP, bf16 on
    // k/v/norms) rather than a blanket bf16-only passthrough -- the whole point of the 4-bit
    // layouts is to serve the M=1..64 band MTP self-speculation runs in.
    const int mtp_layers = text_cfg.value("mtp_num_hidden_layers", 1);
    for (int i = 0; i < mtp_layers; ++i) {
      const std::string hf = "mtp.layers." + std::to_string(i) + ".";
      // The real checkpoint has exactly one MTP layer; docs/container-format.md's "mtp.*" section
      // documents that single-layer case with a bare "mtp." prefix (no index), so collapse to that
      // when there is only one layer and fall back to a text-layer-shaped "mtp.layers.{i}."
      // prefix if a future checkpoint ever has more.
      const std::string base = (mtp_layers == 1) ? "mtp." : ("mtp.layers." + std::to_string(i) + ".");
      add_bf16(hf + "input_layernorm.weight", base + "input_layernorm");
      add_bf16(hf + "post_attention_layernorm.weight", base + "post_attention_layernorm");
      add_linear({hf + "self_attn.q_proj.weight"}, base + "attn.qg", layouts);
      add_bf16(hf + "self_attn.k_proj.weight", base + "attn.k");
      add_bf16(hf + "self_attn.v_proj.weight", base + "attn.v");
      add_linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", layouts);
      add_bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm");
      add_bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm");
      // MTP's inner layer is not one of the 64 text-layer indices kv_calibrate.py calibrates --
      // calib_applicable=false so this always falls back to descale=1.0, silently.
      add_descale(base + "attn.k_descale", kv_heads, i, "k", /*calib_applicable=*/false);
      add_descale(base + "attn.v_descale", kv_heads, i, "v", /*calib_applicable=*/false);
      add_linear({hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"}, base + "mlp.gate_up",
                  layouts);
      add_linear({hf + "mlp.down_proj.weight"}, base + "mlp.down", layouts);
    }
    // mtp.fc / mtp.norm / mtp.pre_fc_norm_embedding / mtp.pre_fc_norm_hidden have no per-text-
    // layer analogue (the hidden+embedding concat -> fc projection around the MTP layer, per HF's
    // Qwen3_5MTPLayer) -- bf16 passthrough by exact HF name. docs/container-format.md's "mtp.*"
    // section does not mention these four tensors; that is outside this component's owned paths
    // (src/convert/**, tests/convert/**, tools/convert_ref/**) so it is flagged in open_issues
    // rather than edited here.
    add_bf16("mtp.fc.weight", "mtp.fc");
    add_bf16("mtp.norm.weight", "mtp.norm");
    add_bf16("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding");
    add_bf16("mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden");

    // Reduced-vocab draft head (docs/r9700.md R9): OPTIONAL, only when the caller supplied a
    // calibration-chosen vocabulary subset (tests/model/tool_vocab_calib.cpp's own output JSON).
    // `mtp.draft_head.lm_head` is planned/emitted through the SAME PlanLinearLayouts/
    // EmitLinearLayouts helpers every other quantized linear uses (a plain [draft_vocab_size,
    // hidden] slice of lm_head.weight's ROWS, same K as the real lm_head), in the same LayoutSet as
    // every other mtp.* head linear (`layouts`) so Container::Load's mtp_head_layout selection at
    // load time works identically to mtp.attn.qg/o and mtp.mlp.gate_up/down. `vocab_ids` is a plain
    // raw int32 tensor (subset index -> real vocab id), written directly (not through
    // EncodeFp32/EncodeBf16 -- it is already the on-disk bit pattern, no dtype conversion needed).
    if (!args.draft_vocab_ids.empty()) {
      const nlohmann::json ids_json = nlohmann::json::parse(ReadFile(args.draft_vocab_ids));
      const std::vector<int64_t> draft_ids = ids_json.at("vocab_ids").get<std::vector<int64_t>>();
      if (draft_ids.empty()) {
        throw std::runtime_error("--draft-vocab-ids: 'vocab_ids' array is empty in " +
                                  args.draft_vocab_ids);
      }
      const int64_t draft_vocab_size = static_cast<int64_t>(draft_ids.size());
      std::cout << "[r4dx-convert] draft-vocab-ids=" << args.draft_vocab_ids << " ("
                << draft_vocab_size << " ids)\n";

      // --keep-bf16 applies here too, resolved on the same base name the regex sees everywhere
      // else -- so a pattern like "lm_head$" that happens to reach this optional head keeps it in
      // bf16 rather than quantizing it while claiming otherwise.
      // --w4a16-group-rule likewise (add_linear's order: the group first, then --keep-bf16).
      const LayoutSet draft_head_requested = w4a16_groups.Apply("mtp.draft_head.lm_head", layouts);
      const bool draft_head_kept = keep_bf16.Matches("mtp.draft_head.lm_head");
      const LayoutSet draft_head_layouts =
          draft_head_kept ? r4dx_convert::KeptBf16LayoutSet() : draft_head_requested;
      // --ldlq likewise, resolved once with add_linear's rule. The draft head's Hessian is the MTP
      // layer's post-`mtp.norm` hidden (hessian_capture.py writes mtp.draft_head.hess only under
      // --draft-head), never lm_head's -- so a regex like "lm_head$" that reaches this head against
      // a directory captured without --draft-head is the planning-time "no key" error, by design.
      const bool draft_head_ldlq_matched = ldlq.Matches("mtp.draft_head.lm_head");
      const bool draft_head_ldlq =
          draft_head_ldlq_matched && HasQuantizedLayout(draft_head_layouts);
      note_linear("mtp.draft_head.lm_head", draft_head_layouts);
      plan_jobs.push_back([&writer, &model, &keep_bf16, &ldlq, &w4a16_groups, draft_vocab_size,
                            draft_head_layouts, draft_head_requested, draft_head_kept,
                            draft_head_ldlq_matched, draft_head_ldlq]() {
        const auto& m = model.Meta("lm_head.weight");
        const int64_t hidden_k = m.shape[1];
        if (draft_head_kept) {
          keep_bf16.Record("mtp.draft_head.lm_head", static_cast<int>(draft_vocab_size),
                           static_cast<int>(hidden_k), draft_head_requested, std::cout);
        }
        w4a16_groups.Plan("mtp.draft_head.lm_head", draft_vocab_size, hidden_k, draft_head_layouts);
        if (draft_head_ldlq) ldlq.Plan("mtp.draft_head.lm_head", hidden_k, draft_head_layouts);
        else if (draft_head_ldlq_matched) ldlq.NoteSkipped();
        r4dx_convert::PlanLinearLayouts(writer, "mtp.draft_head.lm_head",
                                         static_cast<int>(draft_vocab_size),
                                         static_cast<int>(hidden_k), draft_head_layouts);
        writer.Plan("mtp.draft_head.vocab_ids", {draft_vocab_size, 4},
                    static_cast<uint64_t>(draft_vocab_size) * 4);
      });
      emit_jobs.push_back([&writer, &model, &emit_linear, draft_ids, draft_head_layouts,
                           draft_head_ldlq]() {
        const auto full = r4dx_convert::ReadTensorAsFloat(model, "lm_head.weight");
        const int64_t hidden_k = model.Meta("lm_head.weight").shape[1];
        const int64_t vocab_full = static_cast<int64_t>(full.size()) / hidden_k;
        std::vector<float> sliced(draft_ids.size() * static_cast<size_t>(hidden_k));
        std::vector<int32_t> ids32(draft_ids.size());
        for (size_t i = 0; i < draft_ids.size(); ++i) {
          const int64_t id = draft_ids[i];
          if (id < 0 || id >= vocab_full) {
            throw std::runtime_error("--draft-vocab-ids: id " + std::to_string(id) +
                                      " out of range [0," + std::to_string(vocab_full) + ")");
          }
          std::copy(full.begin() + id * hidden_k, full.begin() + (id + 1) * hidden_k,
                    sliced.begin() + static_cast<int64_t>(i) * hidden_k);
          ids32[i] = static_cast<int32_t>(id);
        }
        // Its input distribution is the MTP layer's post-`mtp.norm` hidden, NOT the backbone's, so
        // it must never borrow `lm_head`'s vector -- imatrix_capture.py emits its own
        // `mtp.draft_head.lm_head` only under --draft-head, and without that key this falls back to
        // unweighted MSE (with the usual warning) rather than silently mis-weighting. (Same rule
        // for its Hessian under --ldlq, see above.)
        // Never folded by --rotate: it reads the MTP head's own (un-rotated) hidden.
        emit_linear("mtp.draft_head.lm_head", sliced, static_cast<int64_t>(draft_ids.size()),
                    hidden_k, draft_head_layouts, draft_head_ldlq, /*use_rms=*/false, LinearFold{},
                    std::vector<float>{});
        writer.WriteTensor("mtp.draft_head.vocab_ids", ids32.data(), ids32.size() * 4);
      });
    }
  }

  // --trellis-from: every body linear add_linear resolved is covered (or kept bf16), and no layer
  // this run takes tensors from is stale -- before a single shard is read.
  if (trellis.Enabled()) trellis.CheckCoverage(std::cout, std::cerr);

  const auto t0 = std::chrono::steady_clock::now();

  if (plan_jobs.size() != emit_jobs.size())
    throw std::logic_error("RunConvert: " + std::to_string(plan_jobs.size()) + " plan jobs but " +
                           std::to_string(emit_jobs.size()) + " emit jobs");
  // job j planned the writer's tensors [first, second) -- exactly what emit job j writes.
  std::vector<std::pair<size_t, size_t>> job_tensors(plan_jobs.size());
  for (size_t j = 0; j < plan_jobs.size(); ++j) {
    const size_t first = writer.PlannedTensorCount();
    plan_jobs[j]();
    job_tensors[j] = {first, writer.PlannedTensorCount()};
  }
  // After planning (every add_linear's shapes are known by now), before the header is written --
  // so the summary/"matched nothing" warning lands ahead of the long emit pass rather than after it.
  keep_bf16.Report(std::cout, std::cerr);
  ldlq.ReportPlan(std::cout, std::cerr);
  w4a16_groups.Report(std::cout, std::cerr);
  // --trellis-from: the oracle files' sha256 and tensor shapes (docs/trellis-kernel.md 3.3 steps
  // 7-8), moved ahead of the header so a changed oracle directory costs no output.
  if (trellis.Enabled()) {
    trellis.CheckFiles(threads, std::cout);
    trellis.ReportPlan(std::cout);
  }

  // The guard's per-linear record (reuse_guard.hpp's kReuseLinearsKey): every linear's resolved
  // layout set, which a later --reuse-tensors-from of this container compares linear by linear.
  if (want_guard) {
    nlohmann::json rec = nlohmann::json::object();
    for (const auto& kv : linear_layouts) rec[kv.first] = r4dx_convert::LayoutSetId(kv.second);
    reuse_guard[r4dx_convert::kReuseLinearsKey] = rec;
  }

  // --reuse-tensors-from: which jobs are copied (PlanReuse), decided before the header is written so
  // the header can record it and an inconsistent baseline dies before any output exists.
  ReusePlan reuse_plan;
  if (baseline) {
    reuse_plan = PlanReuse(*baseline, writer, job_tensors, linear_jobs, linear_layouts);
    for (size_t i = 0; i < reuse_plan.recomputed_linears.size(); ++i)
      std::cout << "[r4dx-convert] reuse: recompute " << reuse_plan.recomputed_linears[i]
                << " (baseline " << reuse_plan.recomputed_why[i] << ")\n";
    std::cout << "[r4dx-convert] reuse: " << reuse_plan.tensors_copied << " tensor(s) ("
              << reuse_plan.bytes_copied << " B) copied from the baseline, "
              << reuse_plan.tensors_recomputed << " tensor(s) (" << reuse_plan.bytes_recomputed
              << " B) of " << reuse_plan.recomputed_linears.size()
              << " linear(s) whose layout set (w4a16 group or --keep-bf16) differs recomputed\n";
  }

  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["model_id"] = "Qwen/Qwen3.8-27B";
  metadata["config_sha256"] = r4dx_convert::Sha256Hex(config_text);
  metadata["produced_by"] = "r4dx-convert (r4dx dev build)";
  metadata["quant_summary"] = {
      // Review finding (minor): this previously said "text.layers.*.attn.qg|k|v|o": "bf16", which
      // is factually wrong for qg/o (they go through add_linear with the full LayoutSet, same as
      // mlp.gate_up/down) -- only k/v are bf16-only. A loader trusting the old string would pick
      // bf16 for attn.o and silently lose the 4-bit A/B the container exists to enable.
      {"text.layers.*.mlp.gate_up|down", "w4a16|bf16 (as requested by --layouts; pick at load time)"},
      {"text.layers.*.attn.qg|o", "w4a16|bf16 (as requested by --layouts)"},
      // R1 (docs/r9700.md): attn.k/v and gdn.in_proj_z now join the quantized-linear family (were
      // "bf16" unconditionally before this pass) -- old containers built before this change still
      // have the bare, unsuffixed bf16-only tensor; src/model/container.cpp's
      // LoadQuantLinearWithFallback handles both on-disk forms.
      {"text.layers.*.attn.k|v", "w4a16|bf16 (as requested by --layouts; pick at load time)"},
      {"text.layers.*.gdn.in_proj_z", "w4a16|bf16 (as requested by --layouts; pick at load time)"},
      {"text.layers.*.gdn.in_proj_a|b", "bf16 (never quantized -- feeds the decay path)"},
      {"mtp.attn.qg|o", "w4a16|bf16 (as requested by --layouts, when --mtp on)"},
      {"mtp.attn.k|v", "bf16 (when --mtp on)"},
      {"mtp.mlp.gate_up|down", "w4a16|bf16 (as requested by --layouts, when --mtp on)"},
      {"mtp.fc|norm|pre_fc_norm_embedding|pre_fc_norm_hidden", "bf16 (when --mtp on)"},
      {"mtp.draft_head.lm_head", "w4a16|bf16 (as requested by --layouts, OPTIONAL -- only when "
                                  "--draft-vocab-ids was given, docs/r9700.md R9)"},
      {"mtp.draft_head.vocab_ids", "raw int32[draft_vocab_size] (OPTIONAL, same condition)"},
      {"lm_head", "w4a16|bf16 (as requested by --layouts / --lm-head)"},
  };
  // --rotate: every key below is written ONLY for a rotated container, so an unrotated one keeps
  // exactly the header bytes it had before the flag existed.
  if (rot.Enabled()) {
    metadata["quant_summary"]["text.layers.*.input_layernorm|post_attention_layernorm"] =
        "bf16 zeros: rotated container, (1 + w) folded into the next linear (__metadata__.rotation)";
    metadata["quant_summary"]["rotation.*"] =
        "fp32 (signs, mix5; q2ab also had_down/o/gdn_out_signs) -- see __metadata__.rotation";
    metadata["rotation"] = rot.Metadata();
  }
  // --trellis-from: the body's summary lines say what the body now is (the four entries above
  // describe the multi-layout body, which a trellis container does not have), and
  // __metadata__.quant.trellis is what the loader parses (docs/trellis-kernel.md 2.3, 2.5). Only
  // then, so every other container keeps its header byte for byte.
  if (trellis.Enabled()) {
    for (const char* k : {"text.layers.*.mlp.gate_up|down", "text.layers.*.attn.qg|o",
                          "text.layers.*.attn.k|v", "text.layers.*.gdn.in_proj_z"})
      metadata["quant_summary"].erase(k);
    metadata["quant_summary"]
            ["text.layers.*.attn.qg|k|v|o, gdn.in_proj_qkv|z|out_proj, mlp.gate_up|down"] =
        "body: " + trellis.SummaryLine() +
        " -- the trellis layout only (<base>.trellis.w|suh|svh, __metadata__.quant.trellis); "
        "--keep-bf16 bases (r4dx_convert_run.keep_bf16_linears) bf16 only";
  }
  metadata["quant"] = BuildQuantMetadata(w4a16_groups.GroupsJson());
  if (trellis.Enabled()) metadata["quant"]["trellis"] = trellis.QuantMetadata();
  metadata["model_config"] = config;
  metadata["r4dx_convert_run"] = {
      {"layers_converted", layers},
      {"layers_total", num_layers_total},
      {"vision", do_vision},
      {"mtp", do_mtp},
      {"layouts", args.layouts_spec},
      {"lm_head", args.lm_head_spec},
      {"threads", threads},
      {"kv_calib", have_kv_calib ? args.kv_calib : std::string("none (descale placeholder 1.0)")},
      {"draft_vocab_ids",
       args.draft_vocab_ids.empty() ? std::string("none (no reduced-vocab draft head)")
                                     : args.draft_vocab_ids},
      // How the quantized values were CHOSEN (the byte layout is the same either way) -- so a
      // container on disk says which quantizer produced it without re-deriving it from the bytes.
      {"quant_values", args.quant},
      {"imatrix", args.imatrix.empty() ? std::string("none (unweighted MSE)") : args.imatrix},
      // --keep-bf16 (keep_bf16.hpp): which linears in THIS container are bf16 passthroughs rather
      // than quantized. Recorded as the pattern plus the resolved name list, because the pattern
      // alone cannot be re-evaluated later without the converter's own base-name vocabulary, and a
      // sensitivity table is only readable next to the exact set of tensors each row covers.
      {"keep_bf16", args.keep_bf16.empty() ? std::string("none (every linear quantized)")
                                            : args.keep_bf16},
      {"keep_bf16_linears", keep_bf16.Matched()},
      {"keep_bf16_extra_bytes", keep_bf16.ExtraBytes()},
      // --ldlq (docs/quant2.md 2.3): pattern + resolved base list for the same reason as keep_bf16
      // above, plus the damp and the exact Hessian set (directory AND hessian.json's sha256 -- the
      // path alone says nothing once the directory is re-captured). Linears NOT in ldlq_linears
      // were chosen by quant_values/imatrix exactly as before.
      {"ldlq", ldlq.Enabled() ? args.ldlq : std::string("none")},
      {"ldlq_damp", ldlq.Enabled() ? nlohmann::json(ldlq.Damp()) : nlohmann::json(nullptr)},
      {"hessian_dir", ldlq.Enabled() ? args.hessian_dir : std::string("none")},
      {"hessian_manifest_sha256",
       ldlq.Enabled() ? ldlq.ManifestSha256() : std::string("none")},
      {"ldlq_linears", ldlq.Linears()},
  };
  if (rot.Enabled()) {
    metadata["r4dx_convert_run"]["rotate"] = rot.KindName();
    metadata["r4dx_convert_run"]["rotation_seed"] = rot.Seed();
    // How an --imatrix vector reached a rotated linear (rotation.hpp): not diag(H'), which the
    // vector cannot give, but the diagonal model carried into the new basis.
    if (!args.imatrix.empty())
      metadata["r4dx_convert_run"]["imatrix_rotated"] = "diagonal model: (M o M)^T v";
    // Which LDLQ'd in-projections were rounded against the weightless rms Hessian (hessian.json
    // "rms_keys": H' = Q^T H_rms Q). Every other LDLQ'd in-projection of a rotated container divided
    // the norm out of its post-norm Hessian (Q^T D^-1 H D^-1 Q). Rotated containers only: an
    // unrotated one never reads an rms Hessian, and its header keeps exactly its old keys.
    metadata["r4dx_convert_run"]["ldlq_rms_linears"] = ldlq.RmsLinears();
  }
  // --w4a16-group-rule: only when given, so a container converted without it keeps its header byte
  // for byte. The rules as typed (order matters: first match wins), the resolved map (the same one
  // __metadata__.quant.w4a16.groups carries -- recorded here too so the run record is complete on
  // its own) and the .w4a16.wsz byte delta against the default group.
  if (w4a16_groups.Enabled()) {
    metadata["r4dx_convert_run"]["w4a16_group_rules"] = w4a16_groups.RulesJson();
    metadata["r4dx_convert_run"]["w4a16_groups"] = w4a16_groups.GroupsJson();
    metadata["r4dx_convert_run"]["w4a16_group_extra_bytes"] = w4a16_groups.ExtraBytes();
  }
  // --record-reuse-guard / --reuse-tensors-from: only then, so a container converted without them
  // keeps its header byte for byte. `reuse_guard` is identical to a full run's with the same flags
  // (it is written with emit_complete 0 and a zero data_sha256, both patched below once the data is
  // on disk -- to the same values a full run patches in, the data being identical); `reused_from`
  // exists only in a reuse run's output and is the one header difference against that full run
  // (docs/quant2.md 5.2).
  if (want_guard) {
    metadata["r4dx_convert_run"]["reuse_guard"] = reuse_guard;
    // The completion patches below rewrite these exact texts in place; they must be unambiguous.
    for (const std::string& needle :
         {r4dx_convert::ReuseCompleteNeedle(0),
          r4dx_convert::ReuseDataNeedle(r4dx_convert::ReuseDataPlaceholder())}) {
      if (Count(metadata.dump(), needle) != 1)
        throw std::runtime_error("reuse guard: '" + needle + "' does not occur exactly once in __metadata__");
    }
  }
  if (trellis.Enabled()) metadata["r4dx_convert_run"]["trellis"] = trellis.RunMetadata();
  if (baseline) {
    metadata["r4dx_convert_run"]["reused_from"] = {
        {"path", args.reuse_from},
        {"header_sha256", r4dx_convert::Sha256Hex(baseline->header_bytes)},
        {"data_sha256", baseline->data_sha256},
        {"tensors_copied", reuse_plan.tensors_copied},
        {"bytes_copied", reuse_plan.bytes_copied},
        {"tensors_recomputed", reuse_plan.tensors_recomputed},
        {"bytes_recomputed", reuse_plan.bytes_recomputed},
        {"linears_recomputed", reuse_plan.recomputed_linears},
    };
  }
  // --trellis-from: the container is written as <output>.partial and renamed to <output> only once
  // it is complete and checked (after the emit pass), so <output> never holds an unfinished or
  // unchecked trellis container -- not even an earlier run's, which goes now, with a stale
  // .verify-failed (a file another process holds open fails the run here, not after the emit
  // pass).
  const std::string write_path = trellis.Enabled() ? args.output + ".partial" : args.output;
  if (trellis.Enabled()) {
    for (const std::string& p : {args.output, write_path, args.output + ".verify-failed"}) {
      std::error_code ec;
      std::filesystem::remove(std::filesystem::u8path(p), ec);
      if (ec)
        throw std::runtime_error("--trellis-from: cannot remove the existing " + p + " (" +
                                 ec.message() + ")");
    }
  }
  writer.FinalizeHeader(write_path, metadata);
  std::cout << "[r4dx-convert] planned " << writer.PlannedTensorCount() << " tensors, "
            << writer.PlannedDataBytes() << " data bytes\n";
  // The reconstruction check's result is patched over this placeholder once the file is closed, so
  // it must be unambiguous (checked now, not after the emit pass).
  if (trellis.Enabled() && trellis.VerifyFull() &&
      writer.HeaderOccurrences(trellis.VerifyNeedle()) != 1)
    throw std::logic_error(
        "trellis: the verify placeholder does not occur exactly once in the header");

  for (size_t j = 0; j < emit_jobs.size(); ++j) {
    if (baseline && reuse_plan.copy[j]) {
      // Verbatim, straight out of the baseline's mapping (PlanReuse checked name, dtype, shape and
      // size); WriteTensorChunked streams it, so a multi-GB tensor is never held in memory twice.
      for (size_t t = job_tensors[j].first; t < job_tensors[j].second; ++t) {
        const std::string& name = writer.PlannedName(t);
        writer.WriteTensorChunked(name, baseline->reader->Data(name), writer.PlannedBytes(t));
      }
    } else {
      emit_jobs[j]();
    }
  }
  writer.Finish();
  // Every planned tensor was written (Finish throws otherwise): only now can this container become a
  // --reuse-tensors-from baseline. ReadBackDigests syncs the file to disk and hashes every tensor as
  // it is there; a copied tensor must read back as exactly the baseline's verified bytes. Then the
  // digest and the marker, each patched durably (after the data, before returning), in that order:
  // a crash between them leaves emit_complete 0, which is refused.
  if (want_guard) {
    const auto td = std::chrono::steady_clock::now();
    const std::vector<std::string> digests = writer.ReadBackDigests(threads);
    std::vector<std::pair<std::string, std::string>> named(digests.size());
    for (size_t t = 0; t < digests.size(); ++t) named[t] = {writer.PlannedName(t), digests[t]};
    if (baseline) {
      for (size_t j = 0; j < emit_jobs.size(); ++j) {
        if (!reuse_plan.copy[j]) continue;
        for (size_t t = job_tensors[j].first; t < job_tensors[j].second; ++t) {
          if (digests[t] != baseline->tensor_sha256.at(writer.PlannedName(t)))
            throw std::runtime_error("reuse: the copy of '" + writer.PlannedName(t) +
                                     "' does not read back as the baseline's bytes -- the output is "
                                     "left unfinished (emit_complete 0)");
        }
      }
    }
    const std::string data_sha256 = r4dx_convert::ContainerDataSha256(std::move(named));
    writer.PatchHeader(r4dx_convert::ReuseDataNeedle(r4dx_convert::ReuseDataPlaceholder()),
                       r4dx_convert::ReuseDataNeedle(data_sha256));
    writer.PatchHeader(r4dx_convert::ReuseCompleteNeedle(0), r4dx_convert::ReuseCompleteNeedle(1));
    std::cout << "[r4dx-convert] reuse guard: data_sha256 " << data_sha256 << " (every tensor read "
              << "back in " << SecondsBetween(td, std::chrono::steady_clock::now())
              << " s); emit_complete 1\n";
  }
  // --trellis-from: every byte on the device before the file is closed -- the check below reads it
  // back, and the result patched into the header must never reach the disk ahead of the data.
  if (trellis.Enabled()) writer.Sync();
  writer.Close();
  if (trellis.Enabled()) {
    namespace fs = std::filesystem;
    // --trellis-verify full (docs/trellis-kernel.md 3.3 step 10): every imported tensor
    // reconstructed from the CLOSED file's bytes against the checkpoint. The result replaces the
    // placeholder in r4dx_convert_run.trellis.verify; a failure renames the file to
    // <output>.verify-failed, an error inside the check leaves it as <output>.partial -- either way
    // no loader ever picks it up under its intended name.
    if (trellis.VerifyFull()) {
      const auto tv = std::chrono::steady_clock::now();
      r4dx_convert::trellis::TrellisVerifySummary vs;
      try {
        vs = trellis.Verify(write_path, model, threads, std::cout);
        r4dx_convert::trellis::PatchContainerHeader(write_path, trellis.VerifyNeedle(),
                                                    trellis.VerifyPatch(vs));
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) + "\n(--trellis-verify full did not finish: "
                                 "the unchecked output is left as " + write_path + ")");
      }
      std::cout << "[r4dx-convert] trellis verify: " << (vs.checked - vs.failed) << "/"
                << vs.checked << " HF tensors within tolerance, worst |rel - rec| / rec "
                << vs.worst_dev << " (" << vs.worst_name << "), in "
                << SecondsBetween(tv, std::chrono::steady_clock::now()) << " s\n";
      if (vs.failed > 0) {
        const fs::path failed = fs::u8path(args.output + ".verify-failed");
        std::error_code ec;
        fs::rename(fs::u8path(write_path), failed, ec);
        std::string msg = "--trellis-verify full: " + std::to_string(vs.failed) + " of " +
                          std::to_string(vs.checked) +
                          " imported tensor(s) do not reconstruct to the oracle's rel_weight_err; "
                          "the output is " +
                          (ec ? write_path + " (renaming it failed: " + ec.message() + ")"
                              : failed.u8string()) +
                          ":";
        for (size_t i = 0; i < vs.failures.size() && i < 12; ++i) msg += "\n  " + vs.failures[i];
        throw std::runtime_error(msg);
      }
    }
    std::error_code ec;
    fs::rename(fs::u8path(write_path), fs::u8path(args.output), ec);
    if (ec)
      throw std::runtime_error("--trellis-from: renaming the finished " + write_path + " to " +
                               args.output + " failed: " + ec.message());
  }
  if (baseline)
    std::cout << "[r4dx-convert] reuse: copied " << reuse_plan.tensors_copied << " tensor(s) from "
              << args.reuse_from << "; recomputed " << reuse_plan.recomputed_linears.size()
              << " linear(s)\n";
  imatrix.ReportCoverage();
  ldlq.ReportRun(std::cout);
  rot.ReportRun(std::cout);

  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[r4dx-convert] wrote " << args.output << " in " << secs << " s\n";
  return 0;
}

// ---- --selftest mode -----------------------------------------------------------------------

int RunSelftest(const AppArgs& args) {
  const int threads = ResolveThreads(args.threads);
  // --w4a16-group-rule on the base "selftest", resolved before --keep-bf16 exactly as add_linear
  // does: the renamed `.w4a16.wsz.g<g>` tensor, the metadata map and the bf16-companion guard are
  // all reachable here without a checkpoint (tests/convert/test_w4a16_groups.cpp drives them).
  // --no-bf16 is honoured here too (it used to be ignored by the selftest, whose bf16 is always
  // on otherwise), because that guard refuses a non-default group next to a `.bf16.w`.
  r4dx_convert::W4a16GroupRules w4a16_groups(args.w4a16_group_rules);
  if (w4a16_groups.Enabled()) ValidateKernelRuleGroups(w4a16_groups);
  LayoutSet parsed = ParseLayoutList(args.layouts_spec, /*bf16_default_on=*/true);
  if (args.no_bf16) parsed.bf16 = false;
  const LayoutSet requested = w4a16_groups.Apply("selftest", parsed);
  // The selftest's single tensor has container base "selftest", so --keep-bf16 is exercisable end
  // to end here in milliseconds, against tests/convert/fixtures/input.safetensors and without a
  // 27B checkpoint -- the quickest way to see the whole CLI wiring (regex -> LayoutSet -> emitted
  // tensor set -> warning path) by hand. The automated gates are split: tests/convert/
  // test_keep_bf16.cpp owns selection/emission/accounting through the library API (no exe, no
  // checkpoint), and tests/model/test_keep_bf16.cpp runs the real binary on a 4-layer container.
  r4dx_convert::KeepBf16Selector keep_bf16(args.keep_bf16);
  const bool kept = keep_bf16.Matches("selftest");
  const LayoutSet layouts = kept ? r4dx_convert::KeptBf16LayoutSet() : requested;

  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(args.selftest_input));
  if (!reader.Has(args.selftest_name))
    throw std::runtime_error("--selftest-input has no tensor named " + args.selftest_name);
  const auto& meta = reader.Meta(args.selftest_name);
  if (meta.shape.size() != 2) throw std::runtime_error("--selftest tensor must be 2D [N,K]");
  const int N = static_cast<int>(meta.shape[0]), K = static_cast<int>(meta.shape[1]);
  std::vector<float> w(static_cast<size_t>(N) * K);
  if (meta.dtype == "BF16") {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(reader.Data(args.selftest_name));
    for (size_t i = 0; i < w.size(); ++i) w[i] = r4dx::core::Bf16ToFloat(src[i]);
  } else if (meta.dtype == "F32") {
    std::memcpy(w.data(), reader.Data(args.selftest_name), w.size() * 4);
  } else {
    throw std::runtime_error("--selftest tensor must be BF16 or F32");
  }

  // The selftest packs one tensor named `--selftest-name`'s container base "selftest", so an
  // --imatrix passed here must carry a "selftest" vector of length K -- that is what
  // tools/convert_ref/selftest_compare.py generates for its weighted pass.
  ImatrixSource imatrix(args.imatrix, ParseQuantMode(args.quant));
  // --ldlq works here too, against a --hessian-dir whose hessian.json has a "selftest" key of this
  // K: the whole CLI path (regex -> manifest check -> factor -> *Ldlq quantizers -> packers) on one
  // small tensor, without a checkpoint or a capture run. Same resolution rule as add_linear.
  LdlqSource ldlq(args.ldlq, args.hessian_dir, args.ldlq_damp);
  const bool ldlq_matched = ldlq.Matches("selftest");
  const bool use_ldlq = ldlq_matched && HasQuantizedLayout(layouts);
  r4dx_convert::QuantOptions opts;
  if (!use_ldlq) opts = imatrix.For("selftest", K);

  ContainerWriter writer;
  if (kept) keep_bf16.Record("selftest", N, K, requested, std::cout);
  keep_bf16.Report(std::cout, std::cerr);
  w4a16_groups.Plan("selftest", N, K, layouts);
  w4a16_groups.Report(std::cout, std::cerr);
  if (use_ldlq) ldlq.Plan("selftest", K, layouts);
  else if (ldlq_matched) ldlq.NoteSkipped();
  ldlq.ReportPlan(std::cout, std::cerr);
  r4dx_convert::PlanLinearLayouts(writer, "selftest", N, K, layouts);
  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["model_id"] = "selftest";
  metadata["produced_by"] = "r4dx-convert --selftest";
  metadata["quant"] = BuildQuantMetadata(w4a16_groups.GroupsJson());
  // Only with rules, as in RunConvert (the selftest has no r4dx_convert_run block otherwise).
  if (w4a16_groups.Enabled()) {
    metadata["r4dx_convert_run"] = {{"w4a16_group_rules", w4a16_groups.RulesJson()},
                                    {"w4a16_groups", w4a16_groups.GroupsJson()},
                                    {"w4a16_group_extra_bytes", w4a16_groups.ExtraBytes()}};
  }
  writer.FinalizeHeader(args.selftest_output, metadata);
  if (use_ldlq) {
    const auto t0 = std::chrono::steady_clock::now();
    bool reused = false;
    const r4dx_convert::LdlqFactor& f = ldlq.Factor("selftest", K, threads, &reused);
    const auto t1 = std::chrono::steady_clock::now();
    opts.mode = ParseQuantMode(args.quant);
    opts.ldlq = &f;
    r4dx_convert::EmitLinearLayouts(writer, "selftest", w, N, K, layouts, threads, opts);
    const auto t2 = std::chrono::steady_clock::now();
    ldlq.Record("selftest", N, K, SecondsBetween(t0, t1), SecondsBetween(t1, t2), f, reused,
                std::cout);
  } else {
    r4dx_convert::EmitLinearLayouts(writer, "selftest", w, N, K, layouts, threads, opts);
  }
  writer.Finish();

  std::cout << "[r4dx-convert --selftest] N=" << N << " K=" << K << " quant="
            << (use_ldlq ? std::string("ldlq (damp ") + std::to_string(args.ldlq_damp) + ")"
                         : args.quant)
            << (use_ldlq                   ? " (Hessian-weighted)"
                : opts.importance.empty() ? " (unweighted MSE)"
                                          : " (imatrix-weighted)")
            << " -> " << args.selftest_output << "\n";
  imatrix.ReportCoverage();
  ldlq.ReportRun(std::cout);
  return 0;
}

// ---- --dflash-gguf mode (task A1) -----------------------------------------------------------

LayoutSet DflashLayoutSet(const std::string& layout_name) {
  // Exactly the ONE requested layout, no automatic bf16 side-by-side copy (deliberately UNLIKE
  // the main text-model container's "every layout side by side in one file" convention) -- the
  // task spec's own expected container sizes (~1.2 GB for w4a16, ~3.9 GB for
  // bf16) only make sense as single-layout files; `--dflash-gguf ... --layout X` produces ONE
  // container per invocation, matching `r4dx-cli --layout`'s per-run layout selection rather than
  // `r4dx-convert`'s (HF-mode) `--layouts a,b,c` side-by-side list.
  LayoutSet ls;
  ls.bf16 = false;
  if (layout_name == "bf16") ls.bf16 = true;
  else if (layout_name == "w4a16") ls.w4a16 = true;
  else throw std::runtime_error("--layout must be one of w4a16, bf16 (mxfp4 and w4a8 were retired; got '" + layout_name + "')");
  return ls;
}

int RunDflashConvert(const AppArgs& args) {
  using namespace r4dx_convert;
  if (args.dflash_out.empty()) throw std::runtime_error("--dflash-gguf requires --out");
  const int threads = ResolveThreads(args.threads);
  const LayoutSet layouts = DflashLayoutSet(args.dflash_layout);

  GgufReader gguf(Utf8ToWide(args.dflash_gguf));
  Dflash2Metadata meta = ReadDflash2Metadata(gguf);

  // The DFlash2 drafter has no imatrix of its own (imatrix_capture.py's keys are the main model's
  // container bases), so --quant search here is the unweighted-MSE search.
  r4dx_convert::QuantOptions quant_opts;
  quant_opts.mode = ParseQuantMode(args.quant);

  std::cout << "[r4dx-convert --dflash-gguf] input=" << args.dflash_gguf << " out=" << args.dflash_out
            << " layout=" << args.dflash_layout << " threads=" << threads
            << " quant=" << args.quant << "\n"
            << "[r4dx-convert --dflash-gguf] hidden=" << meta.hidden << " block_count=" << meta.block_count
            << " vocab=" << meta.vocab_size << " n_rot=" << meta.n_rot << "\n";

  ContainerWriter writer;
  std::vector<std::function<void()>> plan_jobs, emit_jobs;

  auto add_linear = [&](std::string gguf_name, std::string container_base) {
    plan_jobs.push_back([&writer, &gguf, gguf_name, container_base, layouts]() {
      PlanDflash2Linear(writer, gguf, Dflash2LinearSpec{gguf_name, container_base}, layouts);
    });
    emit_jobs.push_back([&writer, &gguf, gguf_name, container_base, layouts, threads, quant_opts]() {
      EmitDflash2Linear(writer, gguf, Dflash2LinearSpec{gguf_name, container_base}, layouts, threads,
                        quant_opts);
    });
  };
  auto add_f32 = [&](std::string gguf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      PlanDflash2F32(writer, gguf, gguf_name, container_name);
    });
    emit_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      EmitDflash2F32(writer, gguf, gguf_name, container_name);
    });
  };
  auto add_bf16 = [&](std::string gguf_name, std::string container_name) {
    plan_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      PlanDflash2Bf16(writer, gguf, gguf_name, container_name);
    });
    emit_jobs.push_back([&writer, &gguf, gguf_name, container_name]() {
      EmitDflash2Bf16(writer, gguf, gguf_name, container_name);
    });
  };

  add_linear("fc.weight", "dflash.fc");
  add_f32("enc.output_norm.weight", "dflash.enc_output_norm");
  add_f32("output_norm.weight", "dflash.output_norm");
  add_linear("selector_hidden.weight", "dflash.selector.hidden");
  // Row-gather codebooks (task A1: "NOT quantized to a GEMM layout"), bf16, GGUF's own
  // [vocab][rank] flat order preserved verbatim (see dflash2_container.hpp's file comment).
  add_bf16("selector_predecessor.weight", "dflash.selector.predecessor");
  add_bf16("selector_successor.weight", "dflash.selector.successor");

  for (int64_t i = 0; i < meta.block_count; ++i) {
    const std::string g = "blk." + std::to_string(i) + ".";
    const std::string base = "dflash.layers." + std::to_string(i) + ".";
    add_f32(g + "attn_norm.weight", base + "input_layernorm");
    add_linear(g + "attn_q.weight", base + "self_attn.q_proj");
    add_linear(g + "attn_k.weight", base + "self_attn.k_proj");
    add_linear(g + "attn_v.weight", base + "self_attn.v_proj");
    add_linear(g + "attn_output.weight", base + "self_attn.o_proj");
    add_f32(g + "attn_q_norm.weight", base + "self_attn.q_norm");
    add_f32(g + "attn_k_norm.weight", base + "self_attn.k_norm");
    // conv_base: F32 in the GGUF, ne=[hidden,2,2] = flat [side][tap][hidden] (hidden fastest) --
    // exactly third_party/libr4d/r4d_dflash_conv_body.h's expected per-side [taps,H] slicing, no
    // permutation needed; stored bf16 per the task spec (the libr4d kernel reads bf16 x/base/delta).
    add_bf16(g + "attn_conv_base", base + "self_attn.conv.base");
    add_linear(g + "attn_conv_proj.weight", base + "self_attn.conv.proj");
    add_f32(g + "ffn_norm.weight", base + "post_attention_layernorm");
    add_linear(g + "ffn_gate.weight", base + "mlp.gate_proj");
    add_linear(g + "ffn_up.weight", base + "mlp.up_proj");
    add_linear(g + "ffn_down.weight", base + "mlp.down_proj");
    add_bf16(g + "ffn_conv_base", base + "mlp.conv.base");
    add_linear(g + "ffn_conv_proj.weight", base + "mlp.conv.proj");
  }

  for (auto& j : plan_jobs) j();

  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["container_kind"] = "dflash2_draft";
  metadata["model_id"] = "z-lab/Qwen3.8-27B-DFlash2";
  metadata["produced_by"] = "r4dx-convert --dflash-gguf (r4dx dev build)";
  metadata["source_gguf"] = {
      {"filename", args.dflash_gguf.substr(args.dflash_gguf.find_last_of("/\\") + 1)},
      {"sha256_first_1mib", Sha256HexOfFilePrefix(args.dflash_gguf, 1024 * 1024)},
  };
  metadata["dflash2"] = BuildDflash2MetadataJson(meta);
  metadata["dflash2"]["layout"] = args.dflash_layout;
  metadata["quant"] = BuildQuantMetadata();

  const auto t0 = std::chrono::steady_clock::now();
  writer.FinalizeHeader(args.dflash_out, metadata);
  std::cout << "[r4dx-convert --dflash-gguf] planned " << writer.PlannedTensorCount() << " tensors, "
            << writer.PlannedDataBytes() << " data bytes\n";

  for (auto& j : emit_jobs) j();
  writer.Finish();

  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[r4dx-convert --dflash-gguf] wrote " << args.dflash_out << " in " << secs << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ValidateKernelGroupSizes();
    AppArgs args = ParseArgs(argc, argv);
    if (!args.dflash_gguf.empty()) return RunDflashConvert(args);
    if (args.selftest) {
      if (args.selftest_input.empty() || args.selftest_output.empty())
        throw std::runtime_error("--selftest requires --selftest-input and --selftest-output");
      return RunSelftest(args);
    }
    if (args.input.empty() || args.output.empty())
      throw std::runtime_error("--input and --output are required (or pass --selftest or --dflash-gguf)");
    return RunConvert(args);
  } catch (const std::exception& e) {
    std::cerr << "r4dx-convert: error: " << e.what() << "\n";
    return 1;
  }
}
