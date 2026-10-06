"""tools/prefill/build_tasks.py -- builds the RULER-style long-context task set, from local text only.

Tasks (each item has ground truth; see score_tasks.py for the matching rules):

  niah_single     one "special magic number" needle in prose, at a swept depth; recall it
  niah_multikey   8 needles with different keys (7 distractors); recall the asked key's number
  niah_multivalue 4 needles sharing one key; list all 4 numbers
  niah_multiquery 4 needles with different keys; give all 4 numbers
  vt              variable tracking: a 5-hop chain VAR A = n, VAR B = VAR A, ... plus a distractor
                  chain, scattered through prose; list every variable holding n
  cwe             aggregation (RULER common-words extraction): a numbered word list in which 10
                  words appear far more often than the rest; list the 10 most common
  code_qa         real repository source files as the haystack; auto-extracted questions:
                    py_caller  "which function in <file> calls <name>?"   (Python AST)
                    const      "what is the value of constant <name> in <file>?"
                               (C++ constexpr k-names and Python UPPER_CASE literals)
                    def_file   "which file defines function <name>?"

Every prompt is ONE user turn rendered by the checkpoint's own chat template with thinking off --
exactly what r4dx-server builds for {"messages":[{"role":"user",...}], enable_thinking:false} and
what r4dx-cli builds for --prompt-file with its default --think off. Prompt length (the whole
templated prompt) is fitted to at most the length tag's token target by binary search on the
haystack size. Every item starts with a unique "[item <id>]" line so a server's prefix cache can
never reuse one item's prefill for the next (each request is a cold prefill of ~all its tokens).

Output (under --out, default <models root>\\r4dx\\prefill-m0\\tasks):
  tasks_<len>.jsonl     one item per line: id, task, subtype, length, target_tokens, prompt_tokens,
                        content (the user message), answers, match, max_tokens, meta
  prompts\\ttft_<len>.txt the first niah_single item's user message (for r4dx-cli --prompt-file)
  manifest.json         seeds, counts, sha256 of every jsonl

Usage (Python312 has tokenizers/transformers/numpy):
  python tools\\prefill\\build_tasks.py --lengths 8k,32k,64k,128k
  python tools\\prefill\\build_tasks.py --lengths 8k --items 2 --out <dir>   (smoke)

Default items per task: 8 at <=32k, 4 at 64k, 2 at 128k (--items N overrides all lengths).
"""

from __future__ import annotations

import argparse
import ast
import random
import re
import time
import zlib
from pathlib import Path

import pf_common as C

TASKS = ["niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt", "cwe", "code_qa"]
DEFAULT_ITEMS = {"4k": 8, "8k": 8, "16k": 8, "32k": 8, "64k": 4, "128k": 2}

PROSE_INSTR = ("Some special magic numbers and variable assignments are hidden inside the following "
               "document. Read it carefully; a question about it follows the document.")


def item_seed(base: int, task: str, length: str, i: int) -> int:
    return zlib.crc32(f"{base}:{task}:{length}:{i}".encode()) & 0x7FFFFFFF


def depth_for(i: int, n: int) -> float:
    """Needle depths swept evenly over [0.05, 0.95]."""
    return 0.5 if n == 1 else 0.05 + 0.9 * i / (n - 1)


class Builder:
    def __init__(self, tok: C.Tok, seed: int):
        self.tok = tok
        self.seed = seed
        self.paras = C.load_prose_paragraphs()
        self.vocab = C.prose_vocabulary()
        self.code_files = C.load_code_files()
        self._code_tok = None

    # ---- shared ----------------------------------------------------------------------------
    def count(self, content: str) -> int:
        return self.tok.prompt_tokens(content)

    def rand_key(self, rng: random.Random) -> str:
        return f"{rng.choice(self.vocab)}-{rng.choice(self.vocab)}"

    @staticmethod
    def rand_number(rng: random.Random) -> str:
        return str(rng.randint(1_000_000, 9_999_999))

    def prose_haystack(self, rng: random.Random, target: int, header: str, question: str,
                       inserts: list[tuple[float, str]]) -> tuple[str, int, dict]:
        """Prose paragraphs from a random start, with `inserts` (depth, sentence) placed at
        paragraph boundaries at those relative depths, fitted to `target` prompt tokens."""
        start = rng.randrange(len(self.paras))
        order = [self.paras[(start + k) % len(self.paras)] for k in range(len(self.paras))]

        def build(n: int) -> str:
            body = order[:n]
            cum = [0]
            for p in body:
                cum.append(cum[-1] + len(p))
            placed: list[list[str]] = [[] for _ in range(n + 1)]
            for depth, sent in inserts:
                target_c = depth * cum[-1]
                slot = min(range(n + 1), key=lambda k: abs(cum[k] - target_c))
                placed[slot].append(sent)
            parts = []
            for k in range(n + 1):
                parts.extend(placed[k])
                if k < n:
                    parts.append(body[k])
            return (f"{header}\n{PROSE_INSTR}\n\n<document>\n" + "\n\n".join(parts) +
                    f"\n</document>\n\n{question}")

        n, content, c = C.fit_units(len(order), build, self.count, target)
        return content, c, {"paragraphs": n, "start_paragraph": start}

    # ---- NIAH ------------------------------------------------------------------------------
    @staticmethod
    def needle(key: str, value: str) -> str:
        return f"One of the special magic numbers for {key} is: {value}."

    def niah_single(self, rng, target, header, depth):
        key, val = self.rand_key(rng), self.rand_number(rng)
        q = (f"Question: What is the special magic number for {key} mentioned in the document? "
             f"Answer with the number only.")
        content, c, meta = self.prose_haystack(rng, target, header, q, [(depth, self.needle(key, val))])
        meta.update(depth=round(depth, 3), key=key)
        return content, c, [val], "all", 48, meta, None

    def niah_multikey(self, rng, target, header, depth):
        keys = set()
        while len(keys) < 8:
            keys.add(self.rand_key(rng))
        keys = sorted(keys)
        rng.shuffle(keys)
        vals = [self.rand_number(rng) for _ in keys]
        depths = [depth] + [rng.uniform(0.02, 0.98) for _ in keys[1:]]
        q = (f"Question: What is the special magic number for {keys[0]} mentioned in the document? "
             f"Answer with the number only.")
        ins = [(d, self.needle(k, v)) for d, k, v in zip(depths, keys, vals)]
        content, c, meta = self.prose_haystack(rng, target, header, q, ins)
        meta.update(depth=round(depth, 3), key=keys[0], distractors=7)
        return content, c, [vals[0]], "all", 48, meta, None

    def niah_multivalue(self, rng, target, header, depth):
        key = self.rand_key(rng)
        vals = sorted({self.rand_number(rng) for _ in range(4)})
        depths = sorted([depth] + [rng.uniform(0.02, 0.98) for _ in vals[1:]])
        q = (f"Question: What are all the special magic numbers for {key} mentioned in the document? "
             f"List every one of them, separated by commas.")
        ins = [(d, self.needle(key, v)) for d, v in zip(depths, vals)]
        content, c, meta = self.prose_haystack(rng, target, header, q, ins)
        meta.update(depths=[round(d, 3) for d in depths], key=key)
        return content, c, vals, "all", 96, meta, None

    def niah_multiquery(self, rng, target, header, depth):
        keys = set()
        while len(keys) < 4:
            keys.add(self.rand_key(rng))
        keys = sorted(keys)
        vals = [self.rand_number(rng) for _ in keys]
        depths = [depth] + [rng.uniform(0.02, 0.98) for _ in keys[1:]]
        q = ("Question: What are the special magic numbers for " + ", ".join(keys[:-1]) +
             f" and {keys[-1]} mentioned in the document? Give each key with its number.")
        ins = [(d, self.needle(k, v)) for d, k, v in zip(depths, keys, vals)]
        content, c, meta = self.prose_haystack(rng, target, header, q, ins)
        meta.update(depths=[round(d, 3) for d in depths], keys=keys)
        return content, c, vals, "all", 128, meta, None

    # ---- variable tracking -----------------------------------------------------------------
    def vt(self, rng, target, header, depth):
        names = set()
        while len(names) < 10:
            names.add("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)))
        names = sorted(names)
        rng.shuffle(names)
        chain, distract = names[:5], names[5:]
        v1 = str(rng.randint(10000, 99999))
        v2 = str(rng.randint(10000, 99999))
        while v2 == v1:
            v2 = str(rng.randint(10000, 99999))

        def stmts(ch, v):
            return [f"VAR {ch[0]} = {v}"] + [f"VAR {ch[k]} = VAR {ch[k - 1]}" for k in range(1, len(ch))]

        # The target chain spans [depth*0.5, depth*0.5 + 0.5] in order, so its first hop sweeps
        # with `depth`; the distractor chain is spread over the whole document, also in order.
        lo = 0.5 * depth
        d_target = [lo + 0.5 * k / 4 for k in range(5)]
        d_dis = sorted(rng.uniform(0.02, 0.98) for _ in range(5))
        ins = list(zip(d_target, stmts(chain, v1))) + list(zip(d_dis, stmts(distract, v2)))
        q = (f"Question: Find all variables that are assigned the value {v1} in the document above, "
             f"directly or through other variables. Answer with the variable names only, separated "
             f"by spaces.")
        content, c, meta = self.prose_haystack(rng, target, header, q, ins)
        meta.update(value=v1, chain=chain, distractor_chain=distract, first_hop_depth=round(lo, 3))
        return content, c, chain, "all_words", 64, meta, None

    # ---- aggregation: common words extraction ------------------------------------------------
    def cwe(self, rng, target, header, depth):
        words = list(self.vocab)
        rng.shuffle(words)
        common, pool = words[:10], words[10:]
        instr = ("Below is a numbered list of words. In these words, some appear more often than "
                 "others. Memorize the ones that appear most often.")
        q = ("Question: What are the 10 most common words in the above list? Answer with the 10 "
             "words only, separated by commas.")
        # ~4-6 tokens per "N. word" entry: size the uncommon pool for 1.5x the estimate so the fit
        # below is never starved, then cut only uncommon entries -- every common word keeps all
        # cw_freq occurrences and each uncommon word appears at most ucw_freq times (< cw_freq / 3).
        est_entries = max(200, (target - 150) // 4)
        need_unc = int(1.5 * est_entries)
        ucw_freq = max(3, -(-need_unc // len(pool)))
        cw_freq = max(30, 3 * ucw_freq + 1)
        entries = [w for w in common for _ in range(cw_freq)]
        unc_only = [w for w in pool for _ in range(ucw_freq)]
        rng.shuffle(unc_only)
        salt = rng.randrange(1 << 30)

        def build(n_unc: int) -> str:
            lst = unc_only[:n_unc] + entries
            random.Random(salt + n_unc).shuffle(lst)
            body = "\n".join(f"{k + 1}. {w}" for k, w in enumerate(lst))
            return f"{header}\n{instr}\n\n{body}\n\n{q}"
        n, content, c = C.fit_units(len(unc_only), build, self.count, target)
        meta = {"cw_freq": cw_freq, "ucw_freq_max": ucw_freq, "entries": n + len(entries)}
        return content, c, common, "all_words", 128, meta, None

    # ---- code QA ---------------------------------------------------------------------------
    def code_tok(self) -> dict[str, int]:
        if self._code_tok is None:
            self._code_tok = {rel: len(self.tok.encode(t)) for rel, t in self.code_files}
        return self._code_tok

    @staticmethod
    def file_block(rel: str, text: str) -> str:
        return f"===== FILE: {rel} =====\n{text.rstrip()}\n===== END FILE: {rel} ====="

    def code_questions(self, rel: str, text: str) -> list[dict]:
        qs = []
        if rel.endswith(".py"):
            try:
                tree = ast.parse(text)
            except SyntaxError:
                tree = None
            if tree is not None:
                callers: dict[str, set[str]] = {}
                defs = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs.append(node.name)
                        for sub in ast.walk(node):
                            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                                callers.setdefault(sub.func.id, set()).add(node.name)
                import builtins
                for callee, who in sorted(callers.items()):
                    # A callee called from exactly one function (nested defs count for both the
                    # inner and outer function, so require one distinct NON-nested owner).
                    if len(who) != 1 or len(callee) < 6 or hasattr(builtins, callee):
                        continue
                    caller = next(iter(who))
                    if (len(caller) < 5 or caller.startswith("__") or caller in ("main",)
                            or caller in rel or callee in rel):
                        continue
                    if defs.count(caller) != 1:
                        continue
                    qs.append({"subtype": "py_caller", "callee": callee, "answer": caller,
                               "question": f"Question: Which function in the file {rel} calls "
                                           f"`{callee}`? Answer with the function name only."})
                for name in sorted(set(defs)):
                    if defs.count(name) == 1 and len(name) >= 8 and not name.startswith("_"):
                        qs.append({"subtype": "def_file", "name": name, "answer": rel,
                                   "question": f"Question: Which file defines the function "
                                               f"`{name}`? Answer with the file path only."})
            for m in re.finditer(r"^([A-Z][A-Z0-9_]{3,})\s*(?::\s*[\w\[\], ]+)?\s*=\s*"
                                 r"(-?\d[\d_]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:#.*)?$", text, re.M):
                name, val = m.group(1), m.group(2)
                if len(re.findall(rf"^{name}\s*[:=]", text, re.M)) == 1 and _interesting_number(val):
                    qs.append({"subtype": "const", "name": name, "answer": val,
                               "question": f"Question: What is the value of the constant {name} "
                                           f"defined in the file {rel}? Answer with the value only."})
        else:
            for m in re.finditer(r"constexpr\s+[\w:<>\s]+?\s(k[A-Z]\w*)\s*=\s*"
                                 r"([-+]?(?:0x[0-9a-fA-F']+|\d[\d']*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                                 r"[uUlLfF]*)\s*;", text):
                name, val = m.group(1), m.group(2)
                if len(re.findall(rf"\b{name}\s*=", text)) == 1 and _interesting_number(val):
                    qs.append({"subtype": "const", "name": name, "answer": val,
                               "question": f"Question: What is the value of the constant {name} "
                                           f"defined in the file {rel}? Answer with the value only."})
        return qs

    def code_qa(self, rng, target, header, depth, subtype_idx: int):
        sizes = self.code_tok()
        budget = target - 200
        subtypes = ["py_caller", "const", "def_file"]
        want = subtypes[subtype_idx % 3]
        cands = [(rel, t) for rel, t in self.code_files if sizes[rel] <= 0.5 * budget]
        rng.shuffle(cands)
        pick = None
        for rel, text in cands:
            qs = [q for q in self.code_questions(rel, text) if q["subtype"] == want]
            if qs:
                pick = (rel, text, rng.choice(qs))
                break
        if pick is None:
            raise RuntimeError(f"no {want} question fits {target} tokens")
        rel, text, q = pick
        fillers = [(r, t) for r, t in self.code_files if r != rel and sizes[r] <= 0.35 * budget]
        rng.shuffle(fillers)
        if q["subtype"] == "def_file":
            pat = re.compile(rf"\bdef\s+{re.escape(q['name'])}\s*\(")
            base = rel.rsplit("/", 1)[-1]
            fillers = [(r, t) for r, t in fillers
                       if not pat.search(t) and r.rsplit("/", 1)[-1] != base]
        instr = ("The following are source files from a C++/HIP/Python code repository. Read them "
                 "carefully; a question about the code follows the files.")

        def build_from(files_in: list) -> str:
            pos = round(depth * len(files_in))
            files = files_in[:pos] + [(rel, text)] + files_in[pos:]
            body = "\n\n".join(self.file_block(r, t) for r, t in files)
            return f"{header}\n{instr}\n\n{body}\n\n{q['question']}"

        n, content, c = C.fit_units(len(fillers), lambda n: build_from(fillers[:n]), self.count, target)
        # Whole files leave a gap below the target: top it up with the smallest-fitting later
        # fillers (by the cached per-file counts, then checked exactly).
        chosen = list(fillers[:n])
        for r, t in sorted(fillers[n:], key=lambda f: -sizes[f[0]]):
            if c + sizes[r] + 20 > target:
                continue
            trial = build_from(chosen + [(r, t)])
            tc = self.count(trial)
            if tc <= target:
                chosen.append((r, t))
                content, c = trial, tc
            if target - c < 150:
                break
        n = len(chosen)
        meta = {"subtype": q["subtype"], "file": rel, "files": n + 1, "depth": round(depth, 3)}
        meta.update({k: v for k, v in q.items() if k not in ("question", "answer", "subtype")})
        match = "number" if q["subtype"] == "const" else ("path" if q["subtype"] == "def_file" else "word")
        return content, c, [q["answer"]], match, 64, meta, q["subtype"]


def _interesting_number(val: str) -> bool:
    digits = re.sub(r"[^0-9]", "", val.lower().replace("0x", ""))
    return len(digits) >= 3 and len(digits.strip("0")) >= 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="8k,32k,64k,128k")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--items", type=int, default=None, help="items per task per length (all lengths)")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_OUT / "tasks")
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    lengths = C.parse_lengths(args.lengths)
    tasks = [t for t in args.tasks.split(",") if t]
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r}; known: {', '.join(TASKS)}")
    tok = C.Tok(args.model_dir)
    b = Builder(tok, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "tokenizer": tok.t.describe(), "lengths": {}, "tasks": tasks,
                "sources": "repo docs/*.md prose + repo source files (pf_common.py)",
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    for length in lengths:
        target = C.LENGTHS[length]
        n_items = args.items if args.items is not None else DEFAULT_ITEMS[length]
        rows = []
        t0 = time.time()
        for task in tasks:
            for i in range(n_items):
                rng = random.Random(item_seed(args.seed, task, length, i))
                iid = f"{task}-{length}-{i:02d}"
                header = f"[item {iid}]"
                depth = depth_for(i, n_items)
                if task == "code_qa":
                    res = b.code_qa(rng, target, header, depth, i)
                else:
                    res = getattr(b, task)(rng, target, header, depth)
                content, count, answers, match, max_tokens, meta, subtype = res
                rows.append({"id": iid, "task": task, "subtype": subtype, "length": length,
                             "target_tokens": target, "prompt_tokens": count, "content": content,
                             "answers": answers, "match": match, "max_tokens": max_tokens,
                             "meta": meta, "content_sha256": C.sha256_text(content)})
                print(f"[build] {iid:<28} {count:>7} tok  answers={answers if len(answers) < 5 else str(answers[:4]) + '...'}",
                      flush=True)
        path = args.out / f"tasks_{length}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")
        ttft = next((r for r in rows if r["task"] == "niah_single"), rows[0])
        (args.out / "prompts").mkdir(exist_ok=True)
        (args.out / "prompts" / f"ttft_{length}.txt").write_bytes(ttft["content"].encode("utf-8"))
        manifest["lengths"][length] = {
            "target_tokens": target, "items": len(rows), "items_per_task": n_items,
            "jsonl": path.name, "sha256": C.sha256_text(path.read_text(encoding="utf-8")),
            "prompt_tokens_min": min(r["prompt_tokens"] for r in rows),
            "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
            "ttft_prompt": f"prompts/ttft_{length}.txt", "ttft_prompt_tokens": ttft["prompt_tokens"],
            "build_seconds": round(time.time() - t0, 1)}
        print(f"[build] wrote {path} ({len(rows)} items, {time.time() - t0:.1f}s)")
    mpath = args.out / "manifest.json"
    if mpath.exists():
        old = __import__("json").loads(mpath.read_text(encoding="utf-8"))
        old_lengths = old.get("lengths", {})
        old_lengths.update(manifest["lengths"])
        manifest["lengths"] = old_lengths
    C.write_json(mpath, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
