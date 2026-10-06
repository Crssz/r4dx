"""tools/prefill/warm_delta.py -- warm-turn prefill cost at a deep offset, through r4dx-server's prefix
reuse (prefill M0: "a 4k delta appended at 32k and 64k offsets").

For each base length L (default 32k, 64k), against a running server:
  1. cold: messages = [user: prompts/ttft_<L>.txt], max_tokens 8 -> the full prompt is prefilled.
  2. warm: messages = [user: same, assistant: the reply from 1, user: a ~4k-token docs passage],
     max_tokens 8 -> only the tail after the reused prefix is prefilled (the reply's re-rendered
     turn plus the new user turn; the prompt-end checkpoint covers a mismatch inside the reply).
Also a cold run of the same ~4k delta on its own (offset 0) as the shallow reference.
timings.prompt_n / prompt_ms of each request are recorded to <out>/warm_delta.jsonl.

Usage (server already up, e.g. started by warm_delta.ps1):
  python tools\\prefill\\warm_delta.py --port 8094 --bases 32k,64k --out <models root>\\r4dx\\prefill-m0\\profile\\warm
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pf_common as C


def post(url: str, body: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def delta_text(tok: C.Tok, target: int, tag: str) -> str:
    paras = C.load_prose_paragraphs()
    # Start deep into the prose so the passage is not the same text the base prompts open with.
    paras = paras[len(paras) // 2:] + paras[: len(paras) // 2]

    def build(n):
        return f"[delta {tag}]\n" + "\n\n".join(paras[:n]) + "\n\nSummarize the passage above in one sentence."

    n, text, count = C.fit_units(min(len(paras), 4000), build, lambda s: len(tok.encode(s)), target)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--tasks-dir", type=Path, default=C.DEFAULT_OUT / "tasks")
    ap.add_argument("--bases", default="32k,64k")
    ap.add_argument("--delta-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    args.out.mkdir(parents=True, exist_ok=True)
    tok = C.Tok()
    common = {"temperature": 0, "seed": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
    out = open(args.out / "warm_delta.jsonl", "a", encoding="utf-8", newline="\n")

    def run(label: str, messages: list, **extra) -> dict:
        t0 = time.time()
        resp = post(url, {"messages": messages, **common})
        wall = time.time() - t0
        t = resp.get("timings") or {}
        rec = {"label": label, "usage": resp.get("usage"), "timings": t, "wall_s": round(wall, 3),
               "reply": resp["choices"][0]["message"].get("content") or "", **extra}
        pn, pms = t.get("prompt_n"), t.get("prompt_ms")
        rec["prompt_tok_s"] = round(pn / (pms / 1000.0), 1) if pn and pms else None
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[warm] {label:<22} prompt_tokens={(rec['usage'] or {}).get('prompt_tokens')} "
              f"prefilled={pn} in {pms / 1000.0 if pms else 0:.2f}s ({rec['prompt_tok_s']} tok/s) wall {wall:.1f}s",
              flush=True)
        return rec

    d0 = delta_text(tok, args.delta_tokens, "offset0")
    run("cold_delta_offset0", [{"role": "user", "content": d0}],
        delta_content_tokens=len(tok.encode(d0)))
    for base in C.parse_lengths(args.bases):
        content = (args.tasks_dir / "prompts" / f"ttft_{base}.txt").read_text(encoding="utf-8")
        cold = run(f"cold_{base}", [{"role": "user", "content": content}], base=base)
        d = delta_text(tok, args.delta_tokens, base)
        run(f"warm_delta_at_{base}",
            [{"role": "user", "content": content}, {"role": "assistant", "content": cold["reply"]},
             {"role": "user", "content": d}],
            base=base, delta_content_tokens=len(tok.encode(d)),
            prefix_tokens=(cold["usage"] or {}).get("prompt_tokens"))
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
