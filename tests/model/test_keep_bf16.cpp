// tests/model/test_keep_bf16.cpp -- end-to-end gate for r4dx-convert's `--keep-bf16 <regex>`
// (docs/validation.md "Milestone 11 / sensitivity"): convert a REAL 4-layer container in which one
// hand-picked set of linears is kept in bf16 while everything else is w4a16, then load it and
// generate with it.
//
// tests/convert/test_keep_bf16.cpp already owns the selection/emission/accounting contract on the
// CPU. What only a real container can show is the other half of the mechanism -- that
// src/model/container.cpp's LoadQuantLinearWithFallback notices a base carrying only `.bf16.w` and
// falls THAT ONE LINEAR back to bf16, while every other linear in the same container still loads in
// the requested w4a16, and that the resulting mixed-layout model runs. Before Milestone 11 the body
// linears loaded through LoadQuantLinear directly and such a container was a hard "tensor not
// found" throw at load.
//
// The 4 layers of the real checkpoint cover both layer kinds (0-2 GDN, 3 full attention), so the
// regex below selects a GDN-side linear (layer 1's gdn.out_proj + mlp.down) and an attention-side
// one (layer 3's attn.o + mlp.down) and leaves each one's neighbours quantized -- i.e. it checks
// the per-tensor granularity, not just "the flag did something".
//
// SKIPs (CTest SKIPPED, exit 77) when the HF checkpoint is not on this machine, same convention as
// every other tests/model test. Overrides: R4DX_HF_CHECKPOINT (checkpoint dir),
// R4DX_TEST_SCRATCH_DIR (where the ~3 GiB throwaway container is written; %TEMP% by default). The
// container this test writes is deleted before it returns, whether it passes or fails.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "container.h"
#include "model.h"
#include "test_common.h"

using r4dx::model::Container;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx_test::kSkipReturnCode;

namespace {

int g_failures = 0;

bool Check(const std::string& label, bool cond) {
  std::printf("%-64s %s\n", label.c_str(), cond ? "OK" : "FAIL");
  if (!cond) ++g_failures;
  return cond;
}

std::string Env(const char* name, const std::string& fallback) {
#ifdef _MSC_VER
  char* v = nullptr;
  size_t len = 0;
  const bool have = (_dupenv_s(&v, &len, name) == 0 && v != nullptr && *v != '\0');
  const std::string out = have ? std::string(v) : fallback;
  std::free(v);
  return out;
#else
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::string(v) : fallback;
#endif
}

bool DirExists(const std::string& path) {
  // A directory cannot be opened as a file, so probe a file that must exist inside it.
  return r4dx_test::FileExists(path + "\\config.json");
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// std::system() on Windows runs `cmd /c <s>`, and cmd strips the first and last character of <s>
// when both are quotes -- so a command whose executable path is quoted needs one EXTRA outer pair
// or the quoted path is eaten. (The build tree's path routinely contains spaces.)
int RunCommand(const std::string& command_line) {
  const std::string wrapped = "\"" + command_line + "\"";
  return std::system(wrapped.c_str());
}

}  // namespace

int main() {
  const std::string checkpoint = Env("R4DX_HF_CHECKPOINT", "C:/AI/models/Qwen3.8-27B");
  if (!DirExists(checkpoint)) return r4dx_test::SkipMissing(checkpoint + "\\config.json");
#ifndef R4DX_CONVERT_EXE
  std::fprintf(stderr, "[SKIP] this build has no r4dx-convert target (R4DX_BUILD_CONVERT=OFF)\n");
  return kSkipReturnCode;
#else
  const std::string convert_exe = R4DX_CONVERT_EXE;
  if (!r4dx_test::FileExists(convert_exe)) return r4dx_test::SkipMissing(convert_exe);

  const std::string scratch = Env("R4DX_TEST_SCRATCH_DIR", Env("TEMP", "."));
  const std::string container = scratch + "\\r4dx_test_keep_bf16_l4.r4dx";
  const std::string log_path = scratch + "\\r4dx_test_keep_bf16_convert.log";

  // Layer 1 is GDN, layer 3 is full attention (the real checkpoint's [linear,linear,linear,full]
  // repeat) -- so this keeps one linear on each side plus both of their MLP down-projections, and
  // leaves layers 0/2 and every other linear in layers 1/3 quantized.
  const std::string regex = "^text\\.layers\\.[13]\\.(mlp\\.down|attn\\.o|gdn\\.out_proj)$";
  const int threads = static_cast<int>(std::thread::hardware_concurrency() > 0
                                            ? std::thread::hardware_concurrency()
                                            : 4u);

  std::ostringstream cmd;
  cmd << "\"" << convert_exe << "\""
      << " --input \"" << checkpoint << "\""
      << " --output \"" << container << "\""
      << " --layers 4 --vision off --mtp off --layouts w4a16 --lm-head w4a16 --no-bf16"
      << " --threads " << threads << " --keep-bf16 \"" << regex << "\""
      << " > \"" << log_path << "\" 2>&1";
  std::printf("[convert] %s\n", cmd.str().c_str());
  const int rc = RunCommand(cmd.str());
  const std::string log = ReadWholeFile(log_path);
  if (rc != 0) {
    std::fprintf(stderr, "FAIL: r4dx-convert exited %d\n%s\n", rc, log.c_str());
    std::remove(container.c_str());
    std::remove(log_path.c_str());
    return 1;
  }
  std::printf("%s", log.c_str());

  // The flag must SAY what it did -- a sweep reads these lines to confirm the class it meant.
  Check("convert log names each kept linear",
        log.find("keep-bf16: text.layers.1.gdn.out_proj") != std::string::npos &&
            log.find("keep-bf16: text.layers.1.mlp.down") != std::string::npos &&
            log.find("keep-bf16: text.layers.3.attn.o") != std::string::npos &&
            log.find("keep-bf16: text.layers.3.mlp.down") != std::string::npos);
  Check("convert log summarizes the count and the byte cost",
        log.find("4 linear(s) kept as bf16") != std::string::npos);

  {
    // On disk: the kept bases carry ONLY bf16, their neighbours ONLY w4a16.
    r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(container));
    Check("kept base has .bf16.w", r.Has("text.layers.3.attn.o.bf16.w"));
    Check("kept base has no .w4a16.wq", !r.Has("text.layers.3.attn.o.w4a16.wq"));
    Check("neighbour base has .w4a16.wq", r.Has("text.layers.3.attn.qg.w4a16.wq"));
    Check("neighbour base has no .bf16.w", !r.Has("text.layers.3.attn.qg.bf16.w"));
    Check("same class, unmatched layer stays quantized",
          r.Has("text.layers.2.mlp.down.w4a16.wq") && !r.Has("text.layers.2.mlp.down.bf16.w"));
  }

  {
    // At load: exactly the kept linears report kBf16, everything else the requested kW4a16.
    Container c = Container::Load(container, Layout::kW4a16, Layout::kW4a16, /*layer_limit=*/4);
    Check("layer 1 gdn.out_proj loaded bf16", c.Layer(1).gdn->out_proj.layout == Layout::kBf16);
    Check("layer 1 mlp.down loaded bf16", c.Layer(1).mlp.down.layout == Layout::kBf16);
    Check("layer 3 attn.o loaded bf16", c.Layer(3).attn->o.layout == Layout::kBf16);
    Check("layer 3 mlp.down loaded bf16", c.Layer(3).mlp.down.layout == Layout::kBf16);
    Check("layer 1 gdn.in_proj_qkv still w4a16",
          c.Layer(1).gdn->in_proj_qkv.layout == Layout::kW4a16);
    Check("layer 1 gdn.in_proj_z still w4a16", c.Layer(1).gdn->in_proj_z.layout == Layout::kW4a16);
    Check("layer 1 mlp.gate_up still w4a16", c.Layer(1).mlp.gate_up.layout == Layout::kW4a16);
    Check("layer 3 attn.qg still w4a16", c.Layer(3).attn->qg.layout == Layout::kW4a16);
    Check("layer 0 mlp.down still w4a16", c.Layer(0).mlp.down.layout == Layout::kW4a16);
    Check("layer 2 gdn.out_proj still w4a16", c.Layer(2).gdn->out_proj.layout == Layout::kW4a16);
    Check("lm_head still w4a16", c.LmHead().layout == Layout::kW4a16);
  }

  {
    // And it generates: 80-token chunked prefill + 5 decode steps, finite throughout, with the
    // prefill-vs-decode state-handoff equivalence test_forward_smoke.cpp value-gates (same 8e-2
    // tolerance it uses for the quantized layouts) -- a mixed-layout container must be exactly as
    // self-consistent as a uniform one.
    ModelOptions opts;
    opts.container_path = container;
    opts.layout = Layout::kW4a16;  // Model::Load passes this as the lm_head layout too
    opts.max_ctx = 256;
    opts.layer_limit = 4;

    std::vector<int32_t> prompt(80);
    for (int i = 0; i < 80; ++i) prompt[static_cast<size_t>(i)] = 100 + (i * 37) % 5000;

    Model model = Model::Load(opts);
    std::vector<float> logits = model.Prefill(prompt);
    bool finite = logits.size() == static_cast<size_t>(model.Config().vocab_size);
    for (float x : logits) finite = finite && std::isfinite(x);
    int32_t tok = 42;
    for (int step = 0; step < 5 && finite; ++step) {
      logits = model.DecodeStep(tok);
      for (float x : logits) finite = finite && std::isfinite(x);
      int64_t best = 0;
      for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[static_cast<size_t>(best)]) best = static_cast<int64_t>(i);
      }
      tok = static_cast<int32_t>(best % model.Config().vocab_size);
    }
    Check("mixed-layout model: prefill(80)+decode(5) finite logits", finite);
    Check("...and consumed every position",
          model.PositionCount() == static_cast<int64_t>(prompt.size()) + 5);

    Model a = Model::Load(opts);
    const std::vector<float> full = a.Prefill(prompt);
    Model b = Model::Load(opts);
    b.Prefill(std::vector<int32_t>(prompt.begin(), prompt.end() - 1));
    const std::vector<float> split = b.DecodeStep(prompt.back());
    const double rel = r4dx_test::RelL2(split, full);
    Check("...and prefill/decode state handoff agrees (rel L2 < 8e-2)", rel < 8e-2);
    std::printf("  prefill-vs-decode rel L2 = %.4e (layout=%s + 4 bf16 passthroughs)\n", rel,
                LayoutName(Layout::kW4a16));
  }

  std::remove(container.c_str());
  std::remove(log_path.c_str());

  if (g_failures != 0) {
    std::printf("FAIL (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
#endif
}
