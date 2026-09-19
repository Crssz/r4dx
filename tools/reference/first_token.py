"""tools/reference/first_token.py

End-to-end first-token cross-check: runs the REAL Qwen3.8-27B checkpoint through the reference
`transformers` install (no r4dx code at all) for the exact same chat-templated prompt the assembly
task's perf run uses, and prints the top-5 next-token logits/ids/probabilities for the prompt's
last position. Compare this by eye against `r4dx-cli --layout bf16 ...`'s first generated token
(the CLI's --stats output plus its printed first token) -- this script does not itself talk to the
r4dx C++ engine or parse its output, since the two run in entirely separate processes/toolchains;
the comparison is a manual "do the top-5 ids and rough logit ordering agree" check, not an
automated pass/fail gate.

Usage (reference venv only -- see common.py's file comment, read-only against that venv and the
checkpoint):
    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\reference\\first_token.py ^
        --device cpu --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences."

Runtime / feasibility note (see docs/perf.md "Correctness evidence"): a full 27B-parameter forward
pass on CPU (--device cpu, the only option that does not compete with the same HIP device 1 the
r4dx engine itself needs) is multiple minutes at minimum and was NOT executed as part of this
task's run -- the task brief explicitly allows falling back to the layer-golden tests
(tests/model/test_gdn_layer.cpp, test_final_lm_head.cpp, tests/model/attention/test_attn_layer.cpp,
all passing against real per-layer transformers goldens under docs/perf.md's measured tolerances)
as the correctness evidence when this script is not feasible inside the task's time budget. This
file is provided so that check can be run later, e.g. overnight or via --device cpu with
`--max-new-context` trimmed down, without requiring a second GPU-holding process.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import DEFAULT_MODEL_DIR, resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                     help="cpu (default, safe to run alongside the GPU-resident r4dx engine) or "
                          "cuda (device 1 only, per resolve_device's HIP_VISIBLE_DEVICES check -- "
                          "will contend with r4dx-cli for the same 32GB card, so only use this "
                          "once the r4dx run has exited)")
    ap.add_argument("--prompt", default="Write a haiku about GPUs, then explain what a GPU is in "
                                          "two sentences.")
    ap.add_argument("--system", default=None)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=True (default off, "
                                                              "matching the r4dx perf run)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[first_token] loading tokenizer + model from {args.model_dir} on {device} "
          f"(this can take several minutes for a 27B-parameter checkpoint) ...", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    print(f"[first_token] model loaded in {time.time() - t0:.1f}s", flush=True)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    input_ids = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
        return_tensors="pt",
    ).to(device)
    print(f"[first_token] prompt token count: {input_ids.shape[-1]}", flush=True)

    t1 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1, :].float()
    print(f"[first_token] forward pass in {time.time() - t1:.1f}s", flush=True)

    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(logits, args.top_k)
    print(f"[first_token] top-{args.top_k} next-token candidates:")
    for rank, (logit, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        piece = tok.decode([tid])
        print(f"  #{rank + 1}: id={tid:>7d}  logit={logit: .4f}  prob={probs[tid].item(): .4f}  "
              f"piece={piece!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
