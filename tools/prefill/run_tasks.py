"""tools/prefill/run_tasks.py -- sends the frozen task set to a running r4dx-server and records the
answers plus the server's own prefill/decode timings. Normally driven by run_tasks.ps1 (which starts
and stops the server on the requested device); usable directly against any server.

Each item is one non-streaming /v1/chat/completions request: temperature 0, the item's max_tokens,
chat_template_kwargs.enable_thinking=false, seed 0. Results are appended to <out>/results.jsonl one
line per item as they finish, so an interrupted run resumes where it stopped (items already in the
file are skipped). The server's usage.prompt_tokens is checked against the builder's own count of
the same templated prompt (a mismatch means the tokenizer/template disagree and is reported).

Usage:
  python tools\\prefill\\run_tasks.py --port 8093 --tasks-dir D:\\models\\r4dx\\prefill-m0\\tasks `
      --lengths 8k,32k --out D:\\models\\r4dx\\prefill-m0\\runs\\dense-tp1
Options: --task niah_single,vt (subset), --limit N (first N items per task per length), --max-tokens N
(override every item's, e.g. 1 for a pure TTFT sweep).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pf_common as C
import score_tasks


def post(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--tasks-dir", type=Path, default=C.DEFAULT_OUT / "tasks")
    ap.add_argument("--lengths", default="8k,32k,64k,128k")
    ap.add_argument("--task", default="", help="comma-separated task subset")
    ap.add_argument("--limit", type=int, default=0, help="first N items per task per length (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=0, help="override every item's max_tokens")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    args.out.mkdir(parents=True, exist_ok=True)
    res_path = args.out / "results.jsonl"
    done = {r["id"] for r in C.read_jsonl(res_path)}
    subset = {t for t in args.task.split(",") if t}
    items = []
    for length in C.parse_lengths(args.lengths):
        rows = C.read_jsonl(args.tasks_dir / f"tasks_{length}.jsonl")
        if not rows:
            raise SystemExit(f"no {args.tasks_dir / f'tasks_{length}.jsonl'} -- run build_tasks.py first")
        if subset:
            rows = [r for r in rows if r["task"] in subset]
        if args.limit:
            seen: dict[str, int] = {}
            kept = []
            for r in rows:
                seen[r["task"]] = seen.get(r["task"], 0) + 1
                if seen[r["task"]] <= args.limit:
                    kept.append(r)
            rows = kept
        items.extend(rows)
    todo = [r for r in items if r["id"] not in done]
    print(f"[run] {len(items)} items, {len(items) - len(todo)} already done, {len(todo)} to run -> {res_path}",
          flush=True)
    mismatches = 0
    t_start = time.time()
    with open(res_path, "a", encoding="utf-8", newline="\n") as f:
        for k, item in enumerate(todo):
            body = {"messages": [{"role": "user", "content": item["content"]}],
                    "temperature": 0, "seed": 0,
                    "max_tokens": args.max_tokens or item["max_tokens"],
                    "chat_template_kwargs": {"enable_thinking": False}}
            rec = {key: item[key] for key in ("id", "task", "subtype", "length", "target_tokens",
                                                "prompt_tokens", "answers", "match", "content_sha256")}
            t0 = time.time()
            try:
                resp = post(url, body, args.timeout)
                msg = resp["choices"][0]["message"]
                rec.update(output=msg.get("content") or "", finish_reason=resp["choices"][0].get("finish_reason"),
                           usage=resp.get("usage"), timings=resp.get("timings"), error=None)
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = e.read().decode("utf-8", "replace")[:500]
                    except OSError:
                        pass
                rec.update(output="", usage=None, timings=None, error=f"{type(e).__name__}: {e} {detail}")
            rec["wall_s"] = round(time.time() - t0, 3)
            rec["score"] = 0.0 if rec["error"] else score_tasks.score_item(rec, rec["output"])
            server_pt = (rec.get("usage") or {}).get("prompt_tokens")
            if server_pt is not None and server_pt != item["prompt_tokens"]:
                mismatches += 1
                rec["prompt_tokens_server"] = server_pt
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            t = rec.get("timings") or {}
            pps = t["prompt_n"] / (t["prompt_ms"] / 1000.0) if t.get("prompt_ms") else 0.0
            print(f"[run] {k + 1}/{len(todo)} {item['id']:<28} score={rec['score']:.2f} "
                  f"prefill={t.get('prompt_n', '?')} tok {t.get('prompt_ms', 0) / 1000.0:.2f}s ({pps:.1f} tok/s) "
                  f"wall={rec['wall_s']:.1f}s out={rec['output'][:60]!r}"
                  + (f"  ERROR {rec['error'][:200]}" if rec["error"] else "")
                  + (f"  [prompt_tokens server={server_pt} builder={item['prompt_tokens']}]"
                     if server_pt is not None and server_pt != item["prompt_tokens"] else ""),
                  flush=True)
    print(f"[run] finished {len(todo)} items in {time.time() - t_start:.1f}s; "
          f"prompt-token mismatches vs builder: {mismatches}")
    s = score_tasks.summarize(score_tasks.load_results(res_path))
    score_tasks.print_table(s, str(res_path))
    C.write_json(args.out / "summary.json", s)
    return 1 if any(r.get("error") for r in C.read_jsonl(res_path)) else 0


if __name__ == "__main__":
    sys.exit(main())
