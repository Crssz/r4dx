"""tools/reference/llama/full_logits_llama_cpu.py  (docs/nanbeige-plan.md N0-8 / 6.2: the fp32-truth KL gate)

CPU-only teacher-forced log-prob dumps for the Nanbeige4.3-33B gate, in kl_report.py's shared format
(`<seg>.logprobs.f16` rows 0..T-2 [T-1, V=166912] fp16 + `<seg>.meta.json`), for any mix of tokens files
(chat_nanbeige.json with score masks, tokens_nanbeige.json / tokens_nanbeige_long.json raw segments).

Two variants, both streaming the 52 decoder layers from the bf16 checkpoint ONE LAYER AT A TIME FOR ALL SEGMENTS IN
LOCKSTEP (each layer is read from disk once per run):

  truth     exact fp32 upcast of the bf16 weights; fp32 activations, eager attention (softmax in fp32, computed over
            query chunks so a 6k-token segment never materializes [heads, T, T]), fp32 rope tables, fp32 final norm and
            lm_head.
  bf16sdpa  a bf16 engine: bf16 weights / activations, torch scaled_dot_product_attention (query-chunked with an
            explicit causal mask), rope cos/sin computed in fp32 and cast to bf16, the head a bf16 matmul. The NOISE term.
            NOT literally `LlamaForCausalLM.to(bfloat16)`: that also casts the rope `inv_freq` buffer to bf16 (0.3 %
            relative error, measured), which at theta 1.2e8 puts ~0.8 rad of angle error on the fastest pair by position
            280 -- on the tiny fixture it is 12x further from the truth (KL 7.0e-2) than this variant (5.6e-3). Serving
            stacks keep inv_freq in fp32, so this variant is the fair yardstick, and the stricter one for the gate.

The layer math is HF's LlamaDecoderLayer written out (transformers 5.18 modeling_llama.py): RMSNorm computes in fp32 and
casts back before the weight multiply; rope inv_freq = 1 / theta^(arange(0, D, 2) / D) in fp32, angle = fp32 position x
inv_freq, cos/sin cast to the activation dtype; q / k / v / o and gate / up / down without bias; down(silu(gate) * up).
tools/reference/llama/tiny_e2e_golden.py checks this file against LlamaForCausalLM itself (--tiny).

No logit mask: the model has 166912 rows (166107 tokenizer ids) and neither HF nor r4dx masks the padding rows.

Resumable: the hidden states of all segments are checkpointed after every layer in <out-dir>/_ckpt/state.pt.

NO GPU: the environment variables below hide the GPU before torch is imported, and torch.cuda.is_available() must be
False. RAM peak: one fp32 layer (2.4 GB) + the hidden states (segments x T x 7168 x 4 B) + one query chunk of attention
scores + the head (the fp32 lm_head is 4.8 GB, streamed in row blocks of the table).

FULL RUN (one big host job, two variants one after the other):

    $py = 'D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe'
    $K = 'tools\\reference\\kl_corpus'; $O = 'E:\\models\\r4dx\\nanbeige\\kl'
    & $py -I tools\\reference\\llama\\full_logits_llama_cpu.py --variant truth --tokens $K\\chat_nanbeige.json `
        --tokens $K\\tokens_nanbeige.json --tokens $K\\tokens_nanbeige_long.json --out-dir $O\\truth --threads 24
    & $py -I tools\\reference\\llama\\full_logits_llama_cpu.py --variant bf16sdpa --tokens $K\\chat_nanbeige.json `
        --tokens $K\\tokens_nanbeige.json --tokens $K\\tokens_nanbeige_long.json --out-dir $O\\bf16sdpa --threads 24
"""

from __future__ import annotations

import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

assert torch.cuda.is_available() is False, "torch.cuda.is_available() is True: refusing to run (CPU only)"

DEFAULT_MODEL_DIR = Path(r"E:\models\Nanbeige\Nanbeige4.3-33B")
VARIANTS = ("truth", "bf16sdpa")
Q_CHUNK = 512


def token_ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids], separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Checkpoint:
    """Tensor access over a (sharded) safetensors checkpoint."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        idx = model_dir / "model.safetensors.index.json"
        if idx.exists():
            self.where = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        else:
            with safe_open(str(model_dir / "model.safetensors"), "pt") as f:
                self.where = {k: "model.safetensors" for k in f.keys()}
        self.cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.dir / self.where[name]), "pt") as f:
            return f.get_tensor(name)


class LlamaStream:
    def __init__(self, ck: Checkpoint, dtype: torch.dtype, attn: str):
        c = ck.cfg
        self.ck, self.dtype, self.attn = ck, dtype, attn
        self.H = c["hidden_size"]
        self.n_layers = c["num_hidden_layers"]
        self.nh, self.nkv = c["num_attention_heads"], c["num_key_value_heads"]
        self.hd = c.get("head_dim") or self.H // self.nh
        self.eps = float(c["rms_norm_eps"])
        theta = float(c["rope_parameters"]["rope_theta"] if "rope_parameters" in c else c["rope_theta"])
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.int64).float() / self.hd))
        self.vocab = c["vocab_size"]

    def rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return w * h.to(in_dtype)

    def rope_tables(self, T: int):
        pos = torch.arange(T, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]          # fp32 position x inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def load_layer(self, i: int) -> dict:
        p = f"model.layers.{i}."
        names = {"in": "input_layernorm.weight", "post": "post_attention_layernorm.weight", "q": "self_attn.q_proj.weight",
                 "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight",
                 "g": "mlp.gate_proj.weight", "u": "mlp.up_proj.weight", "d": "mlp.down_proj.weight"}
        return {k: self.ck.get(p + n).to(self.dtype) for k, n in names.items()}

    def attention(self, q, k, v):
        """q [nh, T, hd], k / v [nkv, T, hd] -> [T, nh * hd]; causal, GQA by repeat."""
        T = q.shape[1]
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        scale = self.hd ** -0.5
        out = torch.empty_like(q)
        for s in range(0, T, Q_CHUNK):
            e = min(s + Q_CHUNK, T)
            qc, kc, vc = q[:, s:e], k[:, :e], v[:, :e]
            mask = torch.ones(e - s, e, dtype=torch.bool).tril(diagonal=s)
            if self.attn == "eager":
                w = (qc @ kc.transpose(-1, -2)) * scale
                w = w.masked_fill(~mask, float("-inf"))
                w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
                out[:, s:e] = w @ vc
            else:
                out[:, s:e] = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask, scale=scale)
        return out.transpose(0, 1).reshape(T, self.nh * self.hd)

    def layer(self, L: dict, h: torch.Tensor, cos, sin) -> torch.Tensor:
        T = h.shape[0]
        x = self.rms(h, L["in"])
        q = F.linear(x, L["q"]).view(T, self.nh, self.hd).transpose(0, 1)
        k = F.linear(x, L["k"]).view(T, self.nkv, self.hd).transpose(0, 1)
        v = F.linear(x, L["v"]).view(T, self.nkv, self.hd).transpose(0, 1)
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        h = h + F.linear(self.attention(q, k, v), L["o"])
        x = self.rms(h, L["post"])
        return h + F.linear(F.silu(F.linear(x, L["g"])) * F.linear(x, L["u"]), L["d"])


def load_segments(paths: list[Path], max_tokens: int | None, long_max_tokens: int | None) -> list[dict]:
    segs, seen = [], set()
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        if doc.get("tokenizer_arch") != "llama":
            raise SystemExit(f"{p}: not a llama tokens file (tokenizer_arch={doc.get('tokenizer_arch')!r})")
        for s in doc["segments"]:
            chat = "score_mask" in s
            ids = [int(t) for t in s["token_ids"]]
            if not chat:
                cap = long_max_tokens if s["name"].endswith("_long") else max_tokens
                if cap:
                    ids = ids[:cap]
            if s["name"] in seen:
                raise SystemExit(f"duplicate segment name {s['name']}")
            seen.add(s["name"])
            segs.append({"name": s["name"], "token_ids": ids, "chat": chat, "file": str(p)})
    return segs


def write_head(st: LlamaStream, hid: dict, segs: list[dict], head: str, out_dir: Path, meta_common: dict,
               row_block: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = st.ck.get("lm_head.weight")
    table = table.float() if head == "fp32" else table.to(torch.bfloat16)
    for sg in segs:
        n, ids = sg["name"], sg["token_ids"]
        Hs = hid[n][:-1]
        r = Hs.shape[0]
        t0 = time.perf_counter()
        path = out_dir / f"{n}.logprobs.f16"
        nll, worst = 0.0, 0.0
        with open(path, "wb") as f:
            for s in range(0, r, row_block):
                e = min(s + row_block, r)
                lg = (Hs[s:e].float() @ table.T) if head == "fp32" else F.linear(Hs[s:e].to(torch.bfloat16), table).float()
                lp = torch.log_softmax(lg, dim=-1)
                worst = max(worst, torch.logsumexp(lp, dim=-1).abs().max().item())
                tgt = torch.as_tensor(ids[s + 1:e + 1])
                nll -= lp[torch.arange(e - s), tgt].sum().item()
                f.write(np.ascontiguousarray(lp.clamp_min(-65000.0).to(torch.float16).numpy()).tobytes())
        assert path.stat().st_size == r * st.vocab * 2
        meta = {"T": len(ids), "V": st.vocab, "dtype": "float16", "rows": r, "segment": n,
                "sha256_of_token_ids_json": token_ids_sha256(ids), "head": head,
                "mean_nll_next_token": nll / r, "perplexity": float(np.exp(nll / r)), "max_abs_logsumexp_fp16": worst,
                "row_semantics": "row i = log_softmax of the logits at position i = log p(next | ids[0..i])", **meta_common}
        (out_dir / f"{n}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[cpu-ref] {head} {n}: T={len(ids)} nll={nll / r:.4f} ppl={meta['perplexity']:.2f} "
              f"({time.perf_counter() - t0:.0f}s) -> {path.name}", flush=True)
    del table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path, action="append", default=None, help="repeatable")
    ap.add_argument("--max-tokens", type=int, default=512, help="truncation of RAW (mask-less) segments")
    ap.add_argument("--long-max-tokens", type=int, default=6144, help="truncation of raw segments named *_long")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--debug-max-layers", type=int, default=None, help="NOT a valid reference (timing probes)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tiny-ids", default=None, help="tests: comma list of ids, one segment named 'tiny' (no tokens file)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    if args.tiny_ids:
        segs = [{"name": "tiny", "token_ids": [int(v) for v in args.tiny_ids.split(",")], "chat": False, "file": "<tiny>"}]
        tokens_sha = None
    else:
        if not args.tokens:
            raise SystemExit("--tokens required")
        segs = load_segments(args.tokens, args.max_tokens, args.long_max_tokens)
        tokens_sha = {str(p): sha256_file(p) for p in args.tokens}
    truth = args.variant == "truth"
    dtype = torch.float32 if truth else torch.bfloat16
    st = LlamaStream(Checkpoint(args.model_dir), dtype, "eager" if truth else "sdpa")
    n_layers = st.n_layers if args.debug_max_layers is None else args.debug_max_layers
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cpu-ref] variant={args.variant} dtype={dtype} layers={n_layers} threads={torch.get_num_threads()} "
          f"segments={[(s['name'], len(s['token_ids'])) for s in segs]}", flush=True)

    t_start = time.perf_counter()
    names = [s["name"] for s in segs]
    ckpt = out_dir / "_ckpt" / "state.pt"
    start_layer, states = 0, {}
    if ckpt.exists() and not args.no_resume and not args.tiny_ids:
        c = torch.load(ckpt, weights_only=True)
        if c["names"] == names and c["variant"] == args.variant and c.get("debug") == args.debug_max_layers:
            start_layer, states = c["next_layer"], c["states"]
            print(f"[cpu-ref] resuming at layer {start_layer}", flush=True)
    with torch.no_grad():
        if not states:
            table = st.ck.get("model.embed_tokens.weight")
            states = {s["name"]: table[torch.as_tensor(s["token_ids"])].to(dtype) for s in segs}
            del table
        ropes = {s["name"]: st.rope_tables(len(s["token_ids"])) for s in segs}
        for i in range(start_layer, n_layers):
            t0 = time.perf_counter()
            L = st.load_layer(i)
            t_load = time.perf_counter() - t0
            for n in names:
                states[n] = st.layer(L, states[n], *ropes[n])
            del L
            hmax = max(float(states[n].abs().max()) for n in names)
            print(f"[cpu-ref] L{i:02d} load {t_load:.1f}s total {time.perf_counter() - t0:.1f}s max|h|={hmax:.1f} "
                  f"elapsed={(time.perf_counter() - t_start) / 60:.1f}min", flush=True)
            if not args.tiny_ids:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"names": names, "variant": args.variant, "debug": args.debug_max_layers, "next_layer": i + 1,
                            "states": states}, str(ckpt) + ".tmp")
                os.replace(str(ckpt) + ".tmp", ckpt)
        norm = st.ck.get("model.norm.weight").to(dtype)
        hid = {n: st.rms(states[n], norm) for n in names}
        meta_common = {"source": "reference-fp32-truth" if truth else "hf-bf16-sdpa", "variant": args.variant,
                       "model_dir": str(args.model_dir), "work_dtype": str(dtype), "n_layers": n_layers,
                       "debug_max_layers": args.debug_max_layers, "torch_version": torch.__version__,
                       "tokens_files": tokens_sha, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "stack_seconds": time.perf_counter() - t_start}
        write_head(st, hid, segs, "fp32" if truth else "hf-bf16", out_dir, meta_common)
    (out_dir / "run.json").write_text(json.dumps({"variant": args.variant,
                                                  "segments": {s["name"]: len(s["token_ids"]) for s in segs},
                                                  "total_seconds": time.perf_counter() - t_start, "n_layers": n_layers},
                                                 indent=2), encoding="utf-8")
    print(f"[cpu-ref] done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
