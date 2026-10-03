"""CPU-only perplexity sanity: stock HF Gemma-4 vs tools/reference/gemma/ref.py on identical token ids.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\ppl_sanity.py --impl hf  --model-dir <dir> --tag huihui
    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\ppl_sanity.py --impl ref --model-dir <dir> --tag huihui
    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\ppl_sanity.py --compare out1.json out2.json

Cases: corpus english_prose/python_source (first N tokens, BOS only), a chat-templated user+model turn
(ppl on model-turn tokens only), and plain-BOS natural English. Writes per-token NLL / argmax to --out.
No GPU: hides every device in-process and aborts if torch sees one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

assert torch.cuda.is_available() is False, "GPU visible: refusing to run"

TOK_DIR = r"D:\models\Huihui-gemma-4-12B-it-abliterated-tok"
CORPUS = Path(__file__).resolve().parents[2] / "kl_corpus" / "tokens_gemma.json"

PROSE = (
    "The city of Lyon sits where the Rhone and the Saone rivers meet, and for two thousand years that "
    "meeting of waters has shaped its fortunes. The Romans founded it as a colony, and later it became the "
    "capital of the Gauls. In the Renaissance, silk merchants made it rich, and the narrow covered passages "
    "that workers used to carry fabric through the hillside neighborhoods are still in use today. Visitors "
    "often come for the food, which locals will tell you is the best in France, but the old town and the "
    "river banks are worth a long afternoon on their own.")
USER_Q = "Can you tell me a little about the city of Lyon in France?"
MODEL_A = (
    "Certainly! Lyon is a city in east-central France, located where the Rhone and Saone rivers meet. It was "
    "founded by the Romans over two thousand years ago and later served as the capital of Gaul. During the "
    "Renaissance it became an important center for the silk trade, and many of its historic neighborhoods "
    "date from that time. Today it is widely regarded as the gastronomic capital of France, famous for its "
    "traditional bouchons, and its old town is a UNESCO World Heritage Site.")


def build_cases(tok, n):
    doc = json.load(open(CORPUS, encoding="utf-8"))
    segs = {s["name"]: s["token_ids"] for s in doc["segments"]}
    cases = {}
    for name in ("english_prose", "python_source"):
        ids = segs[name][:n]
        cases[f"{name}[:{n}]"] = dict(ids=ids, start=1)
    # chat template, thinking off
    msgs = [{"role": "user", "content": USER_Q}]
    kw = dict(tokenize=False)
    try:
        pre = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, **kw)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": MODEL_A}],
                                       enable_thinking=False, **kw)
    except TypeError:
        pre = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": MODEL_A}], **kw)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    # The history rendering drops the empty thought channel that the generation prompt (thinking off)
    # appends, so build the scored sequence as generation-prompt + answer + end-of-turn (what inference sees).
    full = pre + MODEL_A + "<turn|>\n"
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(pre_ids)] == pre_ids, "template prefix mismatch:\n%r\n%r" % (pre, full)
    cases["chat_model_turn"] = dict(ids=full_ids, start=len(pre_ids), rendered=full, prefix_len=len(pre_ids))
    cases["bos_plain_english"] = dict(ids=[2] + tok(PROSE, add_special_tokens=False)["input_ids"], start=1)
    return cases


def nll_from_logits(logits, ids):
    lp = torch.log_softmax(logits.float(), -1)  # rows predict ids[1:]
    t = torch.tensor(ids[1:])
    nll = -lp[: len(ids) - 1].gather(1, t[:, None])[:, 0]
    return nll, logits[: len(ids) - 1].argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["hf", "ref"])
    ap.add_argument("--model-dir")
    ap.add_argument("--tag", default="x")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    cases = build_cases(tok, a.n)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 4))
    t0 = time.perf_counter()
    if a.impl == "hf":
        from transformers import Gemma4UnifiedForConditionalGeneration as M
        model = M.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map="cpu")
        model.eval()
        print(f"loaded HF in {time.perf_counter()-t0:.0f}s attn={model.config.get_text_config()._attn_implementation}", flush=True)

        def run(ids):
            with torch.no_grad():
                return model(input_ids=torch.tensor([ids])).logits[0]
    else:
        from gemma.ref import GemmaReference
        ref = GemmaReference(a.model_dir, torch.device("cpu"), resident=False, attn="eager", gemm_guard=None)

        def run(ids):
            h = ref.forward_hidden(ids)
            return ref.logits_rows(h, mode="hf-bf16")

    out = {"impl": a.impl, "model_dir": a.model_dir, "cases": {}}
    for name, c in cases.items():
        t1 = time.perf_counter()
        ids = c["ids"]
        logits = run(ids)
        nll, am = nll_from_logits(logits, ids)
        s = c["start"]
        sel = nll[s - 1:]  # nll[j] scores ids[j+1]
        ppl = math.exp(float(sel.mean()))
        r = dict(ids=ids, start=s, nll=[float(x) for x in nll], argmax=am.tolist(), ppl=ppl,
                 mean_nll=float(sel.mean()), n_scored=int(sel.numel()))
        if "rendered" in c:
            r["rendered"] = c["rendered"]
            # drop trailing end-of-turn/newline tokens from the scored span too
            r["ppl_excl_last2"] = math.exp(float(nll[s - 1:-2].mean()))
        out["cases"][name] = r
        print(f"{a.impl} {name}: T={len(ids)} scored={sel.numel()} ppl={ppl:.2f} ({time.perf_counter()-t1:.0f}s)", flush=True)
    p = a.out or Path(f"D:/models/r4dx/huihui-gemma/kl/ppl_sanity_{a.impl}_{a.tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"))
    print("wrote", p)


def compare(p1, p2):
    A, B = json.load(open(p1)), json.load(open(p2))
    for name in A["cases"]:
        if name not in B["cases"]:
            continue
        a, b = A["cases"][name], B["cases"][name]
        assert a["ids"] == b["ids"]
        d = [abs(x - y) for x, y in zip(a["nll"], b["nll"])]
        top1 = sum(x == y for x, y in zip(a["argmax"], b["argmax"])) / len(d)
        print(f"{name}: ppl {A['impl']}={a['ppl']:.2f} {B['impl']}={b['ppl']:.2f} | mean|dNLL|={sum(d)/len(d):.4f} max={max(d):.3f} top1-agree={top1:.3f}")


if __name__ == "__main__":
    main()
