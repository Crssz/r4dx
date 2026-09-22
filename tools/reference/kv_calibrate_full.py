"""tools/reference/kv_calibrate_full.py

The REAL static fp8 KV-cache calibration for docs/container-format.md's "KV descale tables" --
the one `r4dx-convert --kv-calib <json>` is meant to ship with.

Unlike `kv_calibrate.py` (a prototype that feeds the calibration tokens' raw `embed_tokens` rows
straight into one layer, skipping every preceding layer's transform), this script runs the **whole
64-layer bf16 stack** over the calibration corpus and records, at each of the 16 `full_attention`
layers, exactly the tensors the paged fp8 cache stores:

  * **K, post-rope** -- captured by monkeypatching `apply_rotary_pos_emb` in the modeling module,
    the same technique `kv_calibrate.py` uses. The corpus is text-only, so mrope's three axes
    (t, h, w) all carry the plain token index and rope reduces to ordinary partial rope (see
    `full_logits_golden.py`'s "rope / mrope" note).
  * **V** -- the raw `v_proj` output (no rope is ever applied to V), captured with a forward hook.

The hidden states those two are computed from are the *true mid-stack activations*: this script
does not build its own forward at all, it reuses `full_logits_golden.StreamingReference`, which
materializes one real `Qwen3_5DecoderLayer`'s weights from the safetensors shards at a time onto
HIP device 1, runs it, and frees it (peak VRAM ~2-3 GiB against a 51.7 GiB checkpoint). The lm_head
is never touched -- calibration only needs the KV tensors, not the logits.

Per kv head it records `amax` (what the converter consumes: `descale = amax / 448.0`) and, purely
informationally, the 99.99th percentile of |K| / |V| over every calibration element, so the weight
of the tail the amax is chasing is visible rather than assumed.

Output: ONE merged JSON in the converter's layout, all 16 full-attention layers in one file --
`{"<layer_idx>": {"layer_idx", "kv_heads", "head_dim", "k_amax", "v_amax", ...provenance}}`.

Usage (reference venv only; read-only against the venv and the checkpoint):

    $env:HIP_VISIBLE_DEVICES = '1'
    <venv>\\Scripts\\python.exe tools\\reference\\kv_calibrate_full.py `
        --out D:\\models\\r4dx\\qwen38-27b.kvcalib-full.json

See tools/reference/README.md ("kv_calibrate_full.py") for the option list, the corpus, the gate
runs and the expected runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    load_text_config,
    resolve_device,
    sha256_bytes,
    sha256_file,
)
from full_logits_golden import StreamingReference  # noqa: E402

#: OCP e4m3fn finite max magnitude; `descale = amax / FP8_E4M3_MAX` (src/convert/.../kv_calib.hpp).
FP8_E4M3_MAX = 448.0

#: The informational tail statistic recorded alongside amax.
TAIL_PERCENTILE = 99.99

#: How many of the largest |values| per (layer, head, tensor) are kept so `TAIL_PERCENTILE` can be
#: computed *exactly* rather than from a histogram. The nearest-rank 99.99th percentile of N values
#: is the (N - ceil(0.9999*N))-th largest, i.e. it needs ~0.01% of N; with N = tokens * head_dim
#: (~2.1e6 for the default corpus) that is ~213 values, so 8192 is a wide margin. If the corpus
#: ever outgrows it the script says so in the JSON (`tail_exact: false`) instead of lying.
TOP_BUFFER = 8192

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = Path(__file__).parent / "kv_calib_corpus"
DEFAULT_CALIB_TXT = Path(__file__).parent / "calib.txt"
DEFAULT_OUT = Path(r"D:\models\r4dx\qwen38-27b.kvcalib-full.json")


def repo_relative(path: Path) -> str:
    """Path as written into the JSON. Repo-relative with forward slashes when the file lives in the
    repo (never an absolute path through a user profile -- see CLAUDE.md), else the path as given."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------


class HeadStats:
    """Running per-head `amax` and exact-tail percentile over an unbounded stream of [H, T, D]
    blocks. Everything is reduced on the GPU and only [H] / [H, TOP_BUFFER] ever comes back."""

    def __init__(self, num_heads: int, top_buffer: int = TOP_BUFFER):
        self.num_heads = num_heads
        self.top_buffer = top_buffer
        self.amax = torch.zeros(num_heads, dtype=torch.float32)
        self.top = torch.zeros(num_heads, 0, dtype=torch.float32)
        self.count = 0  # elements seen PER HEAD

    def update(self, x: torch.Tensor) -> None:
        """`x`: [H, T, D], any float dtype, on any device."""
        assert x.dim() == 3 and x.shape[0] == self.num_heads, f"unexpected shape {tuple(x.shape)}"
        a = x.detach().float().abs()
        self.amax = torch.maximum(self.amax, a.amax(dim=(1, 2)).cpu())
        flat = a.reshape(self.num_heads, -1)
        k = min(self.top_buffer, flat.shape[1])
        merged = torch.cat([self.top, flat.topk(k, dim=1).values.cpu()], dim=1)
        self.top = merged.topk(min(self.top_buffer, merged.shape[1]), dim=1).values
        self.count += flat.shape[1]

    def percentile(self, p: float) -> tuple[list[float] | None, bool]:
        """Nearest-rank percentile of |value| over every element seen, per head. Returns
        (values, exact); `exact` is False (and values None) if the kept buffer was too small."""
        if self.count == 0:
            return None, False
        rank_from_top = self.count - math.ceil((p / 100.0) * self.count)  # 0-based, descending
        if rank_from_top >= self.top.shape[1]:
            return None, False
        return self.top[:, rank_from_top].tolist(), True


class LayerStats:
    def __init__(self, kv_heads: int):
        self.k = HeadStats(kv_heads)
        self.v = HeadStats(kv_heads)


# --------------------------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------------------------


class KvCapture:
    """Installs the K (post-rope) / V (v_proj) taps around `StreamingReference`'s streaming forward.

    Layer identity: `LazyDecoderLayers` builds layer `i` immediately before the forward loop runs
    it and drops it immediately after, strictly in order, so wrapping the builder is enough to know
    which layer the global `apply_rotary_pos_emb` patch is currently serving.
    """

    def __init__(self, ref: StreamingReference, stats: dict[int, LayerStats],
                 rope_check_layer: int | None = None):
        self.ref = ref
        self.stats = stats
        self.layer_types = ref.layer_types
        self.kv_heads = ref.text_config.num_key_value_heads
        self.head_dim = ref.text_config.head_dim
        self.rope_check_layer = rope_check_layer
        self.rope_check: dict | None = None
        self._current = None
        self._orig_builder = None
        self._orig_rope = None
        self._handles: list = []

    # -- context manager ----------------------------------------------------------------------

    def __enter__(self) -> "KvCapture":
        mod = self.ref.m
        self._orig_rope = mod.apply_rotary_pos_emb
        orig_rope = self._orig_rope

        def patched_rope(*args, **kwargs):
            q_embed, k_embed = orig_rope(*args, **kwargs)
            layer_idx = self._current
            if layer_idx is not None and layer_idx in self.stats:
                k_pre = args[1] if len(args) > 1 else kwargs["k"]
                self._on_k(layer_idx, k_embed, k_pre, args, kwargs)
            return q_embed, k_embed

        mod.apply_rotary_pos_emb = patched_rope

        self._orig_builder = self.ref.lazy._builder

        def builder(i: int):
            layer = self._orig_builder(i)
            self._current = i
            if i in self.stats:
                assert self.layer_types[i] == "full_attention", (
                    f"layer {i} is {self.layer_types[i]!r}, not full_attention")
                self._handles.append(
                    layer.self_attn.v_proj.register_forward_hook(self._make_v_hook(i))
                )
            return layer

        self.ref.lazy._builder = builder
        return self

    def __exit__(self, *exc) -> None:
        self.ref.m.apply_rotary_pos_emb = self._orig_rope
        self.ref.lazy._builder = self._orig_builder
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._current = None

    # -- taps ---------------------------------------------------------------------------------

    def _on_k(self, layer_idx: int, k_embed, k_pre, args, kwargs) -> None:
        # [batch, kv_heads, T, head_dim]; batch is always 1 here (one sequence per forward).
        assert k_embed.dim() == 4 and k_embed.shape[0] == 1 and k_embed.shape[1] == self.kv_heads, (
            f"layer {layer_idx}: post-rope K has shape {tuple(k_embed.shape)}, expected "
            f"[1, {self.kv_heads}, T, {self.head_dim}]"
        )
        self.stats[layer_idx].k.update(k_embed[0])
        if self.rope_check_layer == layer_idx and self.rope_check is None:
            self.rope_check = self._compare_rope(layer_idx, k_pre, k_embed, args, kwargs)

    def _make_v_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            # v_proj output: [batch, T, kv_heads * head_dim]; no rope is applied to V.
            assert output.dim() == 3 and output.shape[0] == 1 and \
                output.shape[2] == self.kv_heads * self.head_dim, (
                    f"layer {layer_idx}: v_proj output has shape {tuple(output.shape)}")
            t = output.shape[1]
            v = output.detach()[0].view(t, self.kv_heads, self.head_dim).transpose(0, 1)
            self.stats[layer_idx].v.update(v)

        return hook

    def _compare_rope(self, layer_idx: int, k_pre, k_post, args, kwargs) -> dict:
        """Gate (c): prove the K tap really is POST-rope. Rope is a rotation applied to the first
        `rot_dim` of each head vector (partial rope) and the identity on the rest, so (1) the
        captured tensor must differ from the pre-rope `k_norm` output, and (2) every per-position,
        per-head vector norm -- both of the full head vector and of the rotated subspace alone --
        must be preserved to bf16 tolerance."""
        cos = args[2] if len(args) > 2 else kwargs["cos"]
        rot_dim = int(cos.shape[-1])
        pre = k_pre.detach().float()
        post = k_post.detach().float()

        def norm_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
            na = a.norm(dim=-1)
            nb = b.norm(dim=-1)
            rel = ((nb - na).abs() / na.clamp_min(1e-6))
            return {"max_abs_norm_diff": float((nb - na).abs().max()),
                    "max_rel_norm_diff": float(rel.max()),
                    "mean_rel_norm_diff": float(rel.mean()),
                    "min_norm": float(na.min()), "max_norm": float(na.max())}

        return {
            "layer_idx": layer_idx,
            "shape": list(k_post.shape),
            "rot_dim": rot_dim,
            "head_dim": self.head_dim,
            "differs_from_pre_rope": bool((post != pre).any()),
            "max_abs_elementwise_diff": float((post - pre).abs().max()),
            "fraction_elements_changed": float((post != pre).float().mean()),
            "fraction_elements_changed_rotary_dims": float(
                (post[..., :rot_dim] != pre[..., :rot_dim]).float().mean()),
            "passthrough_dims_bit_identical": bool(
                torch.equal(post[..., rot_dim:], pre[..., rot_dim:])),
            "full_vector_norm": norm_stats(pre, post),
            "rotary_subspace_norm": norm_stats(pre[..., :rot_dim], post[..., :rot_dim]),
        }


# --------------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------------


class CorpusFile:
    def __init__(self, name: str, path: Path, kind: str, text: str):
        self.name = name
        self.path = path
        self.kind = kind  # "text" | "chat_template"
        self.text = text
        self.token_ids: list[int] = []


def render_chat(path: Path, tokenizer) -> str:
    """A `*.messages.json` corpus entry: `{"messages": [{"role", "content"}, ...]}` rendered with
    the checkpoint's OWN chat template (the engine serves chat, so the calibration set has to
    contain the real control tokens in their real positions). `add_generation_prompt=False`: the
    conversation already ends with the assistant's turn."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    messages = doc["messages"] if isinstance(doc, dict) else doc
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def collect_corpus(corpus_dir: Path, calib_txt: Path | None, extra: list[Path],
                   tokenizer, max_tokens: int) -> list[CorpusFile]:
    paths: list[Path] = []
    if calib_txt is not None:
        if not calib_txt.exists():
            raise FileNotFoundError(f"--calib-txt {calib_txt} does not exist")
        paths.append(calib_txt)
    if corpus_dir is not None:
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"--corpus-dir {corpus_dir} is not a directory")
        paths += sorted(corpus_dir.glob("*.txt"))
        paths += sorted(corpus_dir.glob("*.messages.json"))
    paths += list(extra)
    if not paths:
        raise SystemExit("[kv_calib_full] empty corpus -- nothing to calibrate on")

    files: list[CorpusFile] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.name.endswith(".messages.json"):
            kind = "chat_template"
            text = render_chat(p, tokenizer)
            name = p.name[: -len(".messages.json")]
        else:
            kind = "text"
            text = p.read_text(encoding="utf-8")
            name = p.stem
        cf = CorpusFile(name, p, kind, text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        cf.token_ids = list(ids)[:max_tokens]  # one sequence per file, fresh context, truncated
        if len(cf.token_ids) < 2:
            raise SystemExit(f"[kv_calib_full] {p} tokenizes to {len(cf.token_ids)} ids; need >= 2")
        files.append(cf)
    return files


# --------------------------------------------------------------------------------------------


CAVEAT = (
    "Static per-kv-head fp8 e4m3 KV calibration from a REAL full forward: every amax below was "
    "measured on the true mid-stack activations of the original bf16 checkpoint -- the whole "
    "64-layer stack was run over the calibration corpus (tools/reference/kv_calibrate_full.py, "
    "reusing full_logits_golden.py's layer-streaming StreamingReference), and K was captured "
    "POST-rope (exactly what the paged fp8 cache stores) and V at the v_proj output. This "
    "supersedes tools/reference/kv_calibrate.py, whose amax came from feeding raw embed_tokens "
    "rows into a single layer with every preceding layer skipped. What this still is NOT: it is "
    "not per-token dynamic scaling -- one scalar per kv head is fixed at convert time and reused "
    "for every sequence, position and token at inference, so any activation whose magnitude "
    "exceeds this corpus's amax saturates at +-448 rather than getting its own scale. It is also "
    "only as representative as the corpus: a prompt distribution far outside it (a script, domain "
    "or control-token layout the corpus never shows) can exceed these amaxes. The recorded "
    "p99.99 fields are there to make that risk legible -- amax/p99.99 is how far the top 0.01% of "
    "the distribution reaches past the bulk."
)

CONVERTER_CONSUMPTION = (
    "r4dx-convert --kv-calib <json> (src/convert/main.cpp, r4dx_convert::ResolveKvDescale in "
    "src/convert/include/r4dx_convert/kv_calib.hpp) writes text.layers.{i}.attn.k_descale / "
    ".v_descale as fp32[kv_heads] = amax / fp8_e4m3_max (448.0), one scalar per kv head, replacing "
    "docs/container-format.md's placeholder 1.0. Only k_amax / v_amax are read; every other field "
    "here is provenance. r4dx_kv_write_paged_fp8_hnd (src/kernels/src/r4dx_kernels.hip) divides "
    "each K/V element by its head's descale before the fp8 cast on write (stored_fp8 = "
    "fp8e4m3(value / descale[head])); the attention kernel's dequant is the inverse (value = "
    "fp8_value * descale[head]). A too-small descale clips; a too-large one wastes fp8's few "
    "mantissa bits. NOTE: r4d.h declares the *runtime* k_descale/v_descale as (num_seqs, "
    "kv_heads), indexed seq*kv_heads+kvh -- this per-layer fp32[kv_heads] vector must be broadcast "
    "to all num_seqs rows when the loader builds that runtime array."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="directory of *.txt (plain) and *.messages.json (rendered with the "
                         "tokenizer's chat template) calibration files; 'none' to skip it")
    ap.add_argument("--calib-txt", type=Path, default=DEFAULT_CALIB_TXT,
                    help="the original mixed English/code/Thai corpus, always included; "
                         "pass 'none' to leave it out")
    ap.add_argument("--extra-files", type=Path, action="append", default=[],
                    help="additional corpus file (repeatable); *.messages.json is chat-rendered")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="truncate every corpus file to this many tokens (one sequence per file)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated full_attention layer indices (default: all 16)")
    ap.add_argument("--rope-check", type=int, default=None, metavar="LAYER",
                    help="gate (c): on LAYER, also capture the pre-rope K and verify the tap is "
                         "post-rope; writes <out>.ropecheck.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # The GPU rule, enforced before anything expensive happens (common.resolve_device re-checks it).
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible != "1":
        raise SystemExit(
            "[kv_calib_full] refusing to run: $env:HIP_VISIBLE_DEVICES must be exactly '1' (only "
            f"HIP device 1, the headless R9700, may be used). Got {visible!r}."
        )
    device = resolve_device("cuda")
    dtype = torch.bfloat16

    from transformers import AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))

    calib_txt = None if str(args.calib_txt).lower() == "none" else args.calib_txt
    corpus_dir = None if str(args.corpus_dir).lower() == "none" else args.corpus_dir
    corpus = collect_corpus(corpus_dir, calib_txt, args.extra_files, tokenizer, args.max_tokens)
    total_tokens = sum(len(c.token_ids) for c in corpus)
    print(f"[kv_calib_full] corpus: {len(corpus)} file(s), {total_tokens} tokens "
          f"(<= {args.max_tokens} each)")
    for c in corpus:
        print(f"    {c.name:<24} {len(c.token_ids):>5} tok  [{c.kind}]  {repo_relative(c.path)}")

    _, text_config = load_text_config(args.model_dir)
    kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    full_layers = [i for i, t in enumerate(text_config.layer_types) if t == "full_attention"]
    if args.layers:
        wanted = {int(x) for x in args.layers.split(",") if x.strip()}
        bad = wanted - set(full_layers)
        if bad:
            raise SystemExit(f"[kv_calib_full] {sorted(bad)} are not full_attention layers "
                             f"(those are {full_layers})")
        full_layers = [i for i in full_layers if i in wanted]
    if args.rope_check is not None and args.rope_check not in full_layers:
        raise SystemExit(f"[kv_calib_full] --rope-check {args.rope_check} is not a calibrated "
                         f"full_attention layer ({full_layers})")
    print(f"[kv_calib_full] calibrating {len(full_layers)} full_attention layer(s): {full_layers}")

    t0 = time.perf_counter()
    ref = StreamingReference(args.model_dir, device, dtype=dtype)
    print(f"[kv_calib_full] streaming skeleton ready in {time.perf_counter() - t0:.1f}s: "
          f"{ref.n_layers} layers, hidden={ref.text_config.hidden_size}, "
          f"kv_heads={kv_heads}, head_dim={head_dim}", flush=True)

    stats = {i: LayerStats(kv_heads) for i in full_layers}
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    with torch.no_grad(), KvCapture(ref, stats, rope_check_layer=args.rope_check) as cap:
        for c in corpus:
            t1 = time.perf_counter()
            hidden = ref.forward_hidden(c.token_ids, impl="model")  # full 64-layer stack
            del hidden
            torch.cuda.synchronize()
            print(f"[kv_calib_full] {c.name:<24} T={len(c.token_ids):>5} "
                  f"{time.perf_counter() - t1:6.1f}s", flush=True)
    wall = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"[kv_calib_full] forward done: {wall:.1f}s, peak VRAM {peak:.3f} GiB allocated / "
          f"{reserved:.3f} GiB reserved")

    for i in full_layers:
        if stats[i].k.count == 0 or stats[i].v.count == 0:
            raise SystemExit(f"[kv_calib_full] layer {i} captured nothing (k={stats[i].k.count} "
                             f"v={stats[i].v.count}) -- the K/V taps did not fire")

    corpus_meta = [{"name": c.name, "path": repo_relative(c.path), "kind": c.kind,
                    "sha256": sha256_file(c.path), "tokens": len(c.token_ids),
                    "rendered_sha256": sha256_bytes(c.text.encode("utf-8"))}
                   for c in corpus]
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    out: dict = {}
    for i in full_layers:
        k_tail, k_exact = stats[i].k.percentile(TAIL_PERCENTILE)
        v_tail, v_exact = stats[i].v.percentile(TAIL_PERCENTILE)
        out[str(i)] = {
            "layer_idx": i,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "num_calibration_tokens": total_tokens,
            "k_amax": stats[i].k.amax.tolist(),
            "v_amax": stats[i].v.amax.tolist(),
            # Informational only -- the converter reads k_amax/v_amax and nothing else.
            "k_p9999": k_tail,
            "v_p9999": v_tail,
            "tail_percentile": TAIL_PERCENTILE,
            "tail_exact": bool(k_exact and v_exact),
            "tail_elements_per_head": stats[i].k.count,
            "method": "full-forward",
            "method_detail": (
                "full 64-layer bf16 forward per corpus file from a fresh context (position 0, "
                "causal, no cache) via full_logits_golden.StreamingReference; K captured post-rope "
                "by patching apply_rotary_pos_emb, V at the v_proj output; amax is the max over "
                "every calibration token of every corpus file"
            ),
            "torch_dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "weights_source": "safetensors (streamed, real weights, all 64 layers)",
            "model_dir": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "corpus": corpus_meta,
            "total_calibration_tokens": total_tokens,
            "max_tokens_per_file": args.max_tokens,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "generated_at": generated_at,
            "caveat": CAVEAT,
            "converter_consumption": CONVERTER_CONSUMPTION,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[kv_calib_full] wrote {args.out} ({len(out)} layers)")

    print(f"{'layer':>5} {'head':>4} {'k_amax':>10} {'k_p99.99':>10} {'v_amax':>10} {'v_p99.99':>10}")
    for i in full_layers:
        e = out[str(i)]
        for h in range(kv_heads):
            kt = f"{e['k_p9999'][h]:10.4f}" if e["k_p9999"] else f"{'n/a':>10}"
            vt = f"{e['v_p9999'][h]:10.4f}" if e["v_p9999"] else f"{'n/a':>10}"
            print(f"{i:>5} {h:>4} {e['k_amax'][h]:10.4f} {kt} {e['v_amax'][h]:10.4f} {vt}")

    if cap.rope_check is not None:
        rc_path = args.out.parent / (args.out.name + ".ropecheck.json")
        rc = dict(cap.rope_check)
        rc["generated_at"] = generated_at
        with open(rc_path, "w", encoding="utf-8") as f:
            json.dump(rc, f, indent=2)
        print(f"[kv_calib_full] rope check (layer {rc['layer_idx']}): "
              f"differs_from_pre_rope={rc['differs_from_pre_rope']} "
              f"max|post-pre|={rc['max_abs_elementwise_diff']:.4f} "
              f"rot_dim={rc['rot_dim']}/{rc['head_dim']} "
              f"passthrough_bit_identical={rc['passthrough_dims_bit_identical']}")
        print(f"[kv_calib_full]   full-vector norm preserved: max rel diff "
              f"{rc['full_vector_norm']['max_rel_norm_diff']:.3e}, mean "
              f"{rc['full_vector_norm']['mean_rel_norm_diff']:.3e}")
        print(f"[kv_calib_full]   rotary-subspace norm preserved: max rel diff "
              f"{rc['rotary_subspace_norm']['max_rel_norm_diff']:.3e}, mean "
              f"{rc['rotary_subspace_norm']['mean_rel_norm_diff']:.3e}")
        print(f"[kv_calib_full] wrote {rc_path}")

    print(f"[kv_calib_full] total tokens={total_tokens} wall={wall:.1f}s peak={peak:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
