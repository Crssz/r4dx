"""`hessian_q4.py`: per-layer Hessian capture for qwen4_exp (plan 4.4 / Q12 / Q16, M1-11).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\hessian_q4.py capture --layer 3
        [--tokens E:\\models\\r4dx\\q4exp\\tokens_calib.json] [--root E:\\models\\r4dx\\q4exp] [--budget-seconds 840]
    ... hessian_q4.py status [--root ...]
    ... hessian_q4.py check --layer 3 [--eig-sample 4]      (symmetry / finite / PSD / fallback gates of a written layer)

GPU runs only through `tools\\with_lock.ps1 -Name gpu` (tools/quant2/qwen4exp/pipeline.ps1 does that, one bounded slice per
lock) with HIP_VISIBLE_DEVICES=1 (`common.resolve_device` refuses anything else).

What one `capture --layer L` does. The corpus (segments of <= 4096 tokens, corpus_q4.py) runs through layer L ONLY: the
input of layer L is the residual [T, 10240] (bf16) the previous step left in `<root>\\work\\b{L:02d}.bin` (a memmap; layer 0
embeds the tokens itself), the output goes to `b{L+1:02d}.bin`, so the whole stack is traversed once over the 48 steps and only
two residual boundaries (2 x 20 GB at 1M tokens) ever exist. Layer L is `Qwen4ExpStreamingReference._build_layer(L)` (HF's own
decoder layer on the AWQ weights dequantized to bf16, QSA ties by the `recent` rule of Q23, PLE through the n-gram sidecar)
called exactly as `run_layers` calls it, with `ReferenceHooks.linear_input` on every linear. The Hessians are those of the
bf16 reference forward, not sequential (inputs do not come from quantized layers): `H = sum_t x_t x_t^T / rows`.

Hessian per distinct linear INPUT (hessian_capture.py's taps; file `L{ii:02d}.<tap>.hess`, R4DXHES1 packed fp32 upper triangle):
    GDN      in   input of in_proj_qkv (= in_proj_z's; asserted equal on every segment)        K 2560
             out  input of out_proj                                                              K 6144
    QSA      in   input of q_proj (= k_proj's, v_proj's; asserted)                              K 2560
             out  input of o_proj                                                                K 6144
    experts  L{ii}.e{e:04d}.gu.hess   gate_proj and up_proj share ONE Hessian (the rows routed to e)   K 2560
             L{ii}.e{e:04d}.down.hess the expert's silu(gate) * up activation                           K 640
The router, the shared expert + its gate (bf16), in_proj_a/b, the indexer, the hyper-connection and PLE linears are not
quantized: no Hessian. `hessian.json` (`keys`: checkpoint base name -> file) is the contract of tools/quant2/qwen4exp/oracle_q4.py.

Accumulation. Dense taps: fp32 GEMM per <= 2048 rows added into an fp64 accumulator (hessian_capture.HessianAccum), result
fp32. Expert taps (512 x [2560, 2560] would be 27 GB in fp64, 13.4 GB in fp32 for gate/up): fp32 accumulators on the device,
one fp32 GEMM per (segment, expert) call (products of bf16 values are exact in fp32, the GEMM sums ~tens of rows) added into
the fp32 accumulator: ~hundreds of additions per expert, relative rounding ~1e-5 of the entry (tests/reference/test_q4exp_quant.py
measures it against an fp64 reference: orders below the damping 0.025 mean(diag) the oracle adds). The upper triangle is
what is stored; the accumulator's own asymmetry |H - H^T| / max diag is recorded (`asymmetry` in capture.json).

Fallback rule (Q16). An expert whose routed token count n_e is below the linear's input dimension (2560 gate_up, 640 down)
has a rank-deficient / noisy Hessian. It gets a DIAGONAL one built from layer-wide second moments instead
(formula in hessian.json `fallback.formula`):
    gate_up: m_j = (1 / T) sum over ALL T tokens of the layer's MoE input x_j^2          (the router's input rows)
    down:    m_j = (1 / P) sum over ALL P = T * top_k routed (token, expert) pairs of h_j^2   (h = silu(gate) * up, pooled)
    H_fb = diag(m) + 0.01 * mean(m) * I            (header rows = T resp. P: the rows behind the moments)
The oracle then adds trellis' own 0.025 * mean(diag) damping on top. The count n_e of EVERY expert is logged to
`routing_counts.json` with the list of fallback experts per class.

Probe rows. Per expert a seeded reservoir sample (Algorithm R) of PROBE_ROWS = 256 of its routed gate_up input rows is kept and
written as `L{ii}.probe.safetensors` (`x` [512, 256, 2560] bf16, `n` valid rows per expert): oracle_q4.py measures with them the
error of an expert's OUTPUT when one of its classes is quantized, the loss alloc_q4.py ranks classes by. `capture --propagate-only`
runs a layer without taps (writes only the next residual boundary): for layers whose Hessians are not wanted.

Resumable: the pass is sliced by `--budget-seconds` (default 840: one slice per GPU lock); after the budget the accumulators
(14 GB) go to `<root>\\work\\L{ii:02d}.state` with the next segment index and exit code 75 means "run me again". Residual
boundaries carry a crc32 per segment, checked when the next layer reads them. Done = `hessian\\L{ii:02d}\\DONE.json` (sha256 of
every written file); the layer's input boundary is deleted then (`--keep-boundary` keeps it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hessian_capture as hc  # noqa: E402
from qwen4exp.arch import GDN, QSA  # noqa: E402
from qwen4exp.common_qwen4exp import DEFAULT_MODEL_DIR, DEFAULT_SIDECAR, MODELS_ROOT, sha256_file  # noqa: E402
from qwen4exp.ref import Qwen4ExpStreamingReference, ReferenceHooks  # noqa: E402

Q4_ROOT = MODELS_ROOT / "r4dx" / "q4exp"
DEFAULT_TOKENS = Q4_ROOT / "tokens_calib.json"
FALLBACK_DAMP = 0.01
PROBE_ROWS = 256   # routed rows kept per expert (reservoir) to measure the output-referred quantization error
GU_K = 2560
DOWN_K = 640
EXIT_MORE = 75

# layer type -> [(tap, representative module, companions that must read the same input)]
TAPS = {
    GDN: [("in", "linear_attn.in_proj_qkv", ["linear_attn.in_proj_z"]), ("out", "linear_attn.out_proj", [])],
    QSA: [("in", "self_attn.q_proj", ["self_attn.k_proj", "self_attn.v_proj"]), ("out", "self_attn.o_proj", [])],
}
# tap -> checkpoint modules (relative to the layer) that the file serves
TAP_SERVES = {
    GDN: {"in": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], "out": ["linear_attn.out_proj"]},
    QSA: {"in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "out": ["self_attn.o_proj"]},
}
EXPERT_RE = re.compile(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


def hess_name(layer: int, tap: str) -> str:
    return f"L{layer:02d}.{tap}.hess"


def expert_file(layer: int, e: int, cls: str) -> str:
    return f"L{layer:02d}.e{e:04d}.{cls}.hess"


def layer_dir(root: Path, layer: int) -> Path:
    return root / "hessian" / f"L{layer:02d}"


def peak_rss_gib() -> float:
    import psutil

    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**30


def atomic_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def crc32_i16(a: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8)) & 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------------------
# residual boundaries
# ---------------------------------------------------------------------------------------------------------------


class Boundary:
    """`work\\b{L:02d}.bin`: the residual [N, width] (bf16 as int16) at the input of layer L for the concatenated segments,
    `work\\b{L:02d}.json`: {"complete", "crc": [crc32 per segment]}."""

    def __init__(self, work: Path, layer: int, lens: list[int], width: int):
        self.path = work / f"b{layer:02d}.bin"
        self.meta_path = work / f"b{layer:02d}.json"
        self.lens = lens
        self.offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.width = width
        self.mm: np.memmap | None = None
        self.meta = {"layer": layer, "complete": False, "crc": [None] * len(lens)}

    def exists(self) -> bool:
        return self.meta_path.exists() and self.path.exists()

    def load_meta(self) -> None:
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if len(self.meta["crc"]) != len(self.lens):
            raise RuntimeError(f"{self.meta_path}: {len(self.meta['crc'])} segments, the corpus has {len(self.lens)}")

    def open(self, create: bool) -> None:
        size = int(self.offsets[-1]) * self.width * 2
        if create and (not self.path.exists() or self.path.stat().st_size != size):
            with open(self.path, "wb") as f:
                f.truncate(size)
        if self.path.stat().st_size != size:
            raise RuntimeError(f"{self.path}: {self.path.stat().st_size} bytes, expected {size}")
        self.mm = np.memmap(self.path, dtype=np.int16, mode="r+", shape=(int(self.offsets[-1]), self.width))

    def write(self, seg: int, hidden: torch.Tensor) -> None:
        a = hidden.reshape(-1, self.width).contiguous().cpu().view(torch.int16).numpy()
        if a.shape[0] != self.lens[seg]:
            raise RuntimeError(f"segment {seg}: {a.shape[0]} rows, expected {self.lens[seg]}")
        self.mm[self.offsets[seg]:self.offsets[seg + 1]] = a
        self.meta["crc"][seg] = crc32_i16(a)

    def read(self, seg: int, device) -> torch.Tensor:
        a = np.array(self.mm[self.offsets[seg]:self.offsets[seg + 1]])
        want = self.meta["crc"][seg]
        if want is None or crc32_i16(a) != want:
            raise RuntimeError(f"{self.path}: segment {seg} fails its crc32 (stored {want}): redo layer {self.meta['layer'] - 1}")
        return torch.from_numpy(a).view(torch.bfloat16).reshape(1, -1, self.width).to(device)

    def save_meta(self, complete: bool | None = None) -> None:
        if self.mm is not None:
            self.mm.flush()
        if complete is not None:
            self.meta["complete"] = complete
        atomic_json(self.meta_path, self.meta)

    def close(self) -> None:
        if self.mm is not None:
            self.mm.flush()
            self.mm._mmap.close()
            self.mm = None

    def delete(self) -> None:
        self.close()
        for p in (self.path, self.meta_path):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------------------------------------------
# the capture hooks
# ---------------------------------------------------------------------------------------------------------------


class LayerTaps(ReferenceHooks):
    """linear_input / moe_routing callbacks of one layer: the dense taps, the expert Hessians, the layer-wide second moments."""

    capture_linear_inputs = True

    def __init__(self, ref: Qwen4ExpStreamingReference, layer: int, device, gu_k: int = GU_K, down_k: int = DOWN_K):
        self.layer = layer
        arch = ref.arch
        self.arch = arch
        self.lt = arch.layer_types[layer]
        self.prefix = arch.layer_prefix(layer)
        self.device = device
        self.E = arch.n_experts
        self.top_k = arch.top_k
        self.gu_k, self.down_k = gu_k, down_k
        self.dense: dict[str, hc.HessianAccum] = {}
        self.rep_of: dict[str, str] = {}     # full name -> representative's full name
        self.rep_tap: dict[str, str] = {}    # representative full name -> tap
        for tap, rep, comps in TAPS[self.lt]:
            self.rep_tap[self.prefix + rep] = tap
            for c in comps:
                self.rep_of[self.prefix + c] = self.prefix + rep
        self.gu = torch.zeros(self.E, gu_k, gu_k, dtype=torch.float32, device=device)
        self.dn = torch.zeros(self.E, down_k, down_k, dtype=torch.float32, device=device)
        self.n_gu = np.zeros(self.E, dtype=np.int64)
        self.n_dn = np.zeros(self.E, dtype=np.int64)
        self.routed = np.zeros(self.E, dtype=np.int64)       # from moe_routing
        self.sq_in = torch.zeros(gu_k, dtype=torch.float64, device=device)
        self.t_in = 0
        self.sq_h = torch.zeros(down_k, dtype=torch.float64, device=device)
        self.pairs = 0
        self.shared = {"checked": 0, "equal": True}
        self._rep_x: dict[str, torch.Tensor] = {}
        self.router_name = self.prefix + "mlp.gate"
        # probe rows: a reservoir sample (Algorithm R, seeded) of PROBE_ROWS rows routed to each expert, for the
        # output-referred error of its quantization (oracle_q4.py)
        self.probe = torch.zeros(self.E, PROBE_ROWS, gu_k, dtype=torch.bfloat16)
        self.rng = np.random.Generator(np.random.PCG64(1000 + layer))

    # -- state -------------------------------------------------------------------------------------------------

    def dense_acc(self, name: str, k: int) -> hc.HessianAccum:
        if name not in self.dense:
            self.dense[name] = hc.HessianAccum(k, self.device)
        return self.dense[name]

    # -- ReferenceHooks ------------------------------------------------------------------------------------------

    def moe_routing(self, layer, topk_ids, counts):
        self.routed += counts.detach().cpu().numpy().astype(np.int64)

    def linear_input(self, layer, name, x):
        if layer != self.layer:
            return
        rep_tap = self.rep_tap.get(name)
        if rep_tap is not None:
            x2 = x.reshape(-1, x.shape[-1])
            for s in range(0, x2.shape[0], 2048):
                self.dense_acc(name, x2.shape[1]).update(x2[s:s + 2048])
            self._rep_x[name] = x2
            return
        rep = self.rep_of.get(name)
        if rep is not None:
            x2 = x.reshape(-1, x.shape[-1])
            r = self._rep_x.get(rep)
            ok = r is not None and r.shape == x2.shape and bool(torch.equal(r, x2))
            self.shared["checked"] += 1
            self.shared["equal"] = self.shared["equal"] and ok
            return
        if name == self.router_name:
            xf = x.reshape(-1, x.shape[-1]).double()
            self.sq_in += xf.pow(2).sum(0)
            self.t_in += int(xf.shape[0])
            return
        if ".mlp.experts." in name:
            m = EXPERT_RE.search(name)
            if m is None:
                return
            e, proj = int(m.group(1)), m.group(2)
            if proj == "up_proj":  # same rows as gate_proj
                return
            xf = x.reshape(-1, x.shape[-1]).float()
            if proj == "gate_proj":
                self.sample_probe(e, x.reshape(-1, x.shape[-1]))
                self.gu[e].addmm_(xf.T, xf)
                self.n_gu[e] += xf.shape[0]
            else:
                self.dn[e].addmm_(xf.T, xf)
                self.n_dn[e] += xf.shape[0]
                self.sq_h += xf.double().pow(2).sum(0)
                self.pairs += int(xf.shape[0])

    def sample_probe(self, e: int, x: torch.Tensor) -> None:
        """Reservoir sampling over the rows expert `e` has seen so far (self.n_gu[e]) plus these `x` [n, K]."""
        n = int(x.shape[0])
        t = self.n_gu[e] + np.arange(n)
        slots = np.full(n, -1, dtype=np.int64)
        fill = t < PROBE_ROWS
        slots[fill] = t[fill]
        rest = ~fill
        if rest.any():
            j = np.floor(self.rng.random(int(rest.sum())) * (t[rest] + 1)).astype(np.int64)
            slots[np.flatnonzero(rest)[j < PROBE_ROWS]] = j[j < PROBE_ROWS]
        pos = np.flatnonzero(slots >= 0)
        if pos.size:
            rev = pos[::-1]  # a slot drawn twice keeps the LATER row (reservoir order): first occurrence in the reversed list
            uniq, first = np.unique(slots[rev], return_index=True)
            keep = rev[first]
            rows = x[torch.from_numpy(keep).to(x.device)].to(torch.bfloat16).cpu()
            self.probe[e, torch.from_numpy(uniq)] = rows

    def end_segment(self) -> None:
        self._rep_x.clear()

    # -- persistence -------------------------------------------------------------------------------------------

    def small_state(self) -> dict:
        return {"dense": {n: (a.acc.cpu(), a.rows) for n, a in self.dense.items()}, "n_gu": self.n_gu, "n_dn": self.n_dn,
                "routed": self.routed, "sq_in": self.sq_in.cpu(), "t_in": self.t_in, "sq_h": self.sq_h.cpu(),
                "pairs": self.pairs, "shared": self.shared, "probe": self.probe, "rng": self.rng.bit_generator.state}

    def load_small(self, st: dict) -> None:
        for n, (acc, rows) in st["dense"].items():
            a = hc.HessianAccum(acc.shape[0], self.device)
            a.acc = acc.to(self.device)
            a.rows = rows
            self.dense[n] = a
        self.n_gu, self.n_dn, self.routed = st["n_gu"], st["n_dn"], st["routed"]
        self.sq_in, self.t_in = st["sq_in"].to(self.device), st["t_in"]
        self.sq_h, self.pairs = st["sq_h"].to(self.device), st["pairs"]
        self.shared = st["shared"]
        self.probe = st["probe"]
        self.rng.bit_generator.state = st["rng"]


def save_state(work: Path, layer: int, taps: LayerTaps, next_seg: int, stats: dict) -> float:
    """Atomic state swap: write `L{ii}.state.new`, then replace `L{ii}.state`. Returns seconds."""
    t0 = time.perf_counter()
    cur, new, old = work / f"L{layer:02d}.state", work / f"L{layer:02d}.state.new", work / f"L{layer:02d}.state.old"
    for p in (new, old):
        if p.exists():
            shutil.rmtree(p)
    new.mkdir(parents=True)
    for name, t in (("gu.f32", taps.gu), ("dn.f32", taps.dn)):
        with open(new / name, "wb", buffering=16 << 20) as f:
            step = 8
            for e in range(0, t.shape[0], step):
                t[e:e + step].cpu().numpy().tofile(f)
    torch.save(taps.small_state(), new / "small.pt")
    atomic_json(new / "progress.json", {"next_seg": next_seg, "stats": stats, "layer": layer})
    if cur.exists():
        os.replace(cur, old)
    os.replace(new, cur)
    if old.exists():
        shutil.rmtree(old)
    return time.perf_counter() - t0


def load_state(work: Path, layer: int, taps: LayerTaps) -> tuple[int, dict] | None:
    cur, old = work / f"L{layer:02d}.state", work / f"L{layer:02d}.state.old"
    if not cur.exists() and old.exists():
        os.replace(old, cur)
    if not cur.exists():
        return None
    prog = json.loads((cur / "progress.json").read_text(encoding="utf-8"))
    for name, t in (("gu.f32", taps.gu), ("dn.f32", taps.dn)):
        n = t.shape[0]
        with open(cur / name, "rb") as f:
            for e in range(0, n, 8):
                m = min(8, n - e)
                a = np.fromfile(f, dtype=np.float32, count=m * t.shape[1] * t.shape[2]).reshape(m, t.shape[1], t.shape[2])
                t[e:e + m] = torch.from_numpy(a).to(t.device)
    taps.load_small(torch.load(cur / "small.pt", weights_only=False))
    return prog["next_seg"], prog.get("stats", {})


# ---------------------------------------------------------------------------------------------------------------
# the layer pass
# ---------------------------------------------------------------------------------------------------------------


class WeightCache:
    """`ref.reader.linear_weight` with the expert weights of the current layer kept on the device (5 GiB bf16 per layer): a
    segment calls every routed expert, 450 segments would dequantize each one 450 times otherwise."""

    def __init__(self, reader):
        self.reader = reader
        self.orig = reader.linear_weight
        self.cache: dict[str, torch.Tensor] = {}
        self.hits = self.misses = 0
        reader.linear_weight = self.linear_weight

    def linear_weight(self, base, device):
        if ".mlp.experts." not in base:
            return self.orig(base, device)
        w = self.cache.get(base)
        if w is None:
            w = self.orig(base, device)
            self.cache[base] = w
            self.misses += 1
        else:
            self.hits += 1
        return w

    def clear(self) -> None:
        self.cache.clear()


def layer_context(ref: Qwen4ExpStreamingReference, ids: list[int], embed: bool):
    """What `run_layers` builds once per sequence for its layer loop: (embeds, kwargs of the decoder layer call). `embed` False
    (layers > 0, whose input is a stored residual) skips the embedding read: the masks and rope only use its shape and dtype."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import create_causal_mask, create_recurrent_attention_mask

    seq = len(ids)
    embeds = (ref.embed(ids).unsqueeze(0) if embed
              else torch.zeros(1, seq, ref.arch.hidden, dtype=torch.bfloat16, device=ref.device))
    position_ids = torch.arange(seq, device=ref.device).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids, position_ids = position_ids[0], position_ids[1:]
    mask_kwargs = {"config": ref.text_config, "inputs_embeds": embeds, "attention_mask": None,
                   "past_key_values": None, "position_ids": text_position_ids, "allow_is_causal_skip": False}
    qsa_dense = seq <= ref.attn_query_chunk
    attention_mask = create_causal_mask(**mask_kwargs) if qsa_dense else None
    conv_mask = create_recurrent_attention_mask(**mask_kwargs)
    ple_ids = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=ref.device)
    if conv_mask is not None:
        ple_ids = torch.where(conv_mask.bool(), ple_ids, ref.arch.eos_token_id)
    position_embeddings = ref._rotary_emb()(embeds, position_ids)
    return embeds, dict(position_embeddings=position_embeddings, attention_mask=attention_mask, conv_mask=conv_mask,
                        past_key_values=None, ple_input_ids=ple_ids)


def load_corpus(path: Path) -> tuple[list[list[int]], dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    segs = [[int(t) for t in s["token_ids"]] for s in doc["segments"]]
    rec = {"path": str(path), "sha256": sha256_file(path), "segments": len(segs), "tokens": sum(len(s) for s in segs),
           "names": [s["name"] for s in doc["segments"]], "tokenizer_mode": doc.get("tokenizer_mode")}
    return segs, rec


def fallback_diag(m: torch.Tensor) -> torch.Tensor:
    """H_fb = diag(m) + FALLBACK_DAMP * mean(m) * I as a dense [K, K] float32 (diagonal) matrix."""
    d = m.double()
    d = d + FALLBACK_DAMP * d.mean()
    return torch.diag(d.float())


def write_hess(path: Path, h: torch.Tensor, rows: int) -> dict:
    return hc.write_hess_file(path, h.detach().cpu().numpy(), rows)


def finalize_layer(root: Path, layer: int, taps: LayerTaps, ref: Qwen4ExpStreamingReference, corpus: dict,
                   timing: dict) -> dict:
    """Write every .hess, hessian.json, routing_counts.json, capture.json, DONE.json for a layer whose pass is complete."""
    out = layer_dir(root, layer)
    out.mkdir(parents=True, exist_ok=True)
    arch, lt, E = ref.arch, taps.lt, taps.E
    files: dict[str, dict] = {}
    keys: dict[str, str] = {}
    asym: dict[str, float] = {}
    t0 = time.perf_counter()

    for rep_full, tap in taps.rep_tap.items():
        acc = taps.dense[rep_full]
        rows = acc.rows
        h = acc.finalize_fp32()
        asym[hess_name(layer, tap)] = float((h - h.T).abs().max() / h.diagonal().abs().max())
        files[hess_name(layer, tap)] = write_hess(out / hess_name(layer, tap), h, rows)
        del h
        for mod in TAP_SERVES[lt][tap]:
            keys[taps.prefix + mod] = hess_name(layer, tap)

    if taps.t_in == 0 or taps.pairs == 0:
        raise RuntimeError("no MoE rows were captured")
    m_in = (taps.sq_in / taps.t_in).float()
    m_h = (taps.sq_h / taps.pairs).float()
    fb_gu = fallback_diag(m_in)
    fb_dn = fallback_diag(m_h)
    fallback = {"gate_up": [], "down": []}
    worst_asym = 0.0
    for e in range(E):
        for cls, acc_t, n, k, fb, fb_rows in (("gu", taps.gu, int(taps.n_gu[e]), taps.gu_k, fb_gu, taps.t_in),
                                               ("down", taps.dn, int(taps.n_dn[e]), taps.down_k, fb_dn, taps.pairs)):
            fname = expert_file(layer, e, cls)
            if n < k:
                fallback["gate_up" if cls == "gu" else "down"].append(e)
                files[fname] = write_hess(out / fname, fb, fb_rows)
            else:
                h = acc_t[e] / float(n)
                worst_asym = max(worst_asym, float((h - h.T).abs().max() / h.diagonal().abs().max()))
                files[fname] = write_hess(out / fname, h, n)
            if cls == "gu":
                keys[arch.expert_base(layer, e, "gate_proj")] = fname
                keys[arch.expert_base(layer, e, "up_proj")] = fname
            else:
                keys[arch.expert_base(layer, e, "down_proj")] = fname
    asym["experts_worst"] = worst_asym

    if not np.array_equal(taps.n_gu, taps.routed) or not np.array_equal(taps.n_dn, taps.routed):
        raise RuntimeError("routed counts disagree with the captured expert rows")
    bad = {f: s for f, s in files.items() if not (s["finite"] and s["min_diag"] > 0.0)}
    fb_set = {expert_file(layer, e, "gu") for e in fallback["gate_up"]} | {expert_file(layer, e, "down") for e in fallback["down"]}
    nonpos = {f: s["nonpos_channels"][:8] for f, s in bad.items() if f not in fb_set}
    share = {"gate_up": len(fallback["gate_up"]) / E, "down": len(fallback["down"]) / E}
    routing = {"layer": layer, "tokens": taps.t_in, "top_k": taps.top_k, "pairs": taps.pairs,
               "threshold": {"gate_up": taps.gu_k, "down": taps.down_k},
               "tokens_routed": [int(v) for v in taps.routed], "fallback": fallback, "fallback_share": share,
               "experts_never_routed": [int(e) for e in np.flatnonzero(taps.routed == 0)],
               "min_routed": int(taps.routed.min()), "max_routed": int(taps.routed.max()),
               "median_routed": float(np.median(taps.routed)), "formula": FORMULA,
               "probe_rows": [int(min(v, PROBE_ROWS)) for v in taps.n_gu]}
    atomic_json(out / "routing_counts.json", routing)
    from safetensors.torch import save_file
    save_file({"x": taps.probe.contiguous(), "n": torch.tensor([min(int(v), PROBE_ROWS) for v in taps.n_gu], dtype=torch.int32)},
              str(out / f"L{layer:02d}.probe.safetensors"), metadata={"format": "r4dx-q4exp-probe", "layer": str(layer),
                                                                       "rows": str(PROBE_ROWS), "k": str(taps.gu_k)})
    extra = {"tool": "tools/reference/qwen4exp/hessian_q4.py", "arch": "qwen4_exp", "layer": layer, "layer_type": lt,
             "model_dir": str(ref.model_dir),
             "config_sha256": sha256_file(ref.model_dir / "config.json"), "tie_rule": ref.tie_rule,
             "semantics": "H = sum_t x_t x_t^T / rows (rows = tokens routed to the expert for expert files); the upper "
                          "triangle of the accumulator is stored; non-sequential: inputs of the bf16 (AWQ-dequantized) reference",
             "corpus": {k: corpus[k] for k in ("path", "sha256", "segments", "tokens", "tokenizer_mode")},
             "fallback": {"formula": FORMULA, "damp": FALLBACK_DAMP, "threshold": routing["threshold"],
                          "experts": fallback, "share": share},
             "shared_input_gate": taps.shared, "asymmetry": asym,
             "accumulation": {"dense": "fp32 GEMM (<= 2048 rows) into fp64, fp32 result",
                              "experts": "fp32 GEMM per (segment, expert) call into an fp32 accumulator"}}
    hc.write_manifest(out, files, keys, extra)
    timing["finalize_s"] = time.perf_counter() - t0
    cap = {"layer": layer, "layer_type": lt, "tokens": taps.t_in, "files": len(files), "bytes": sum(s["bytes"] for s in files.values()),
           "fallback_share": share, "fallback_count": {k: len(v) for k, v in fallback.items()},
           "nonpositive_diag_nonfallback": nonpos, "asymmetry": asym,
           "shared_input_gate": taps.shared, "timing": timing,
           "peak_rss_gib": peak_rss_gib(),
           "peak_vram_gib": (torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)}
    atomic_json(out / "capture.json", cap)
    if nonpos:
        raise RuntimeError(f"layer {layer}: non-finite or non-positive diagonals outside the fallback: {nonpos}")
    if not taps.shared["equal"]:
        raise RuntimeError(f"layer {layer}: a companion linear did not read its representative's input")
    done = {"layer": layer, "files": {f.name: sha256_file(f) for f in sorted(out.glob("*")) if f.name != "DONE.json"}}
    atomic_json(out / "DONE.json", done)
    return cap


FORMULA = ("gate_up: m_j = mean over all T tokens of the layer's MoE input x_j^2 (router input rows); down: m_j = mean over all "
           "P = T*top_k routed (token, expert) pairs of h_j^2 (h = silu(gate) * up, pooled over experts); "
           f"H_fb = diag(m) + {FALLBACK_DAMP} * mean(m) * I; header rows = T resp. P")


def capture_layer(args) -> int:
    if args.propagate_only:
        return propagate_layer(args)
    t_start = time.perf_counter()
    root = Path(args.root)
    layer = args.layer
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    out = layer_dir(root, layer)
    if (out / "DONE.json").exists() and not args.force:
        print(f"[hessian_q4] L{layer:02d}: DONE already ({out})", flush=True)
        return 0
    corpus_ids, corpus = load_corpus(Path(args.tokens))
    lens = [len(s) for s in corpus_ids]
    ref = Qwen4ExpStreamingReference(args.model_dir, args.device, tie_rule="recent", ngram_sidecar=args.ngram_sidecar)
    arch = ref.arch
    if layer >= arch.n_layers:
        raise SystemExit(f"layer {layer} >= {arch.n_layers}")
    device = ref.device
    width = arch.residual_width
    need_gib = sum(lens) * width * 2 / 2**30 * 2 + 16.0
    free_gib = shutil.disk_usage(work).free / 2**30
    if free_gib < need_gib + 4 and not args.force:
        raise SystemExit(f"[hessian_q4] only {free_gib:.0f} GiB free on {work.drive}; a layer step needs about {need_gib:.0f}")

    b_in = Boundary(work, layer, lens, width)
    if layer > 0:
        if not b_in.exists():
            raise SystemExit(f"[hessian_q4] no input boundary {b_in.path}: run layer {layer - 1} first")
        b_in.load_meta()
        if not b_in.meta["complete"]:
            raise SystemExit(f"[hessian_q4] {b_in.meta_path} is not complete")
        b_in.open(create=False)
    last = layer == arch.n_layers - 1 and not args.write_final
    b_out = None if last else Boundary(work, layer + 1, lens, width)

    hooks = LayerTaps(ref, layer, device)
    ref.hooks = hooks
    wc = WeightCache(ref.reader)
    timing = {"seconds_forward_total": 0.0, "seconds_state_io": 0.0, "slices": 0, "segments_per_slice": []}
    next_seg = 0
    t0 = time.perf_counter()
    st = load_state(work, layer, hooks)
    if st is not None:
        next_seg, old = st
        timing.update({k: v for k, v in old.items() if k in timing and k != "segments_per_slice"})
        timing["segments_per_slice"] = list(old.get("segments_per_slice", []))
        timing["seconds_state_io"] += time.perf_counter() - t0
        print(f"[hessian_q4] L{layer:02d}: resumed at segment {next_seg}/{len(lens)} ({time.perf_counter() - t0:.1f}s state load)",
              flush=True)
    if b_out is not None:
        if b_out.exists() and next_seg > 0:
            b_out.load_meta()
        b_out.open(create=True)
    seg_done = 0
    layer_mod = ref._build_layer(layer)
    for seg in range(next_seg, len(lens)):
        ids = corpus_ids[seg]
        t_seg = time.perf_counter()
        with torch.no_grad():
            embeds, kw = layer_context(ref, ids, embed=layer == 0)
            hidden = embeds.repeat(1, 1, arch.hc_count) if layer == 0 else b_in.read(seg, device)
            hidden = layer_mod(hidden, **kw)
        hooks.end_segment()
        if b_out is not None:
            b_out.write(seg, hidden)
        del hidden, embeds, kw
        seg_done += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        if args.verbose or seg % 25 == 0:
            print(f"[hessian_q4] L{layer:02d} segment {seg + 1}/{len(lens)} ({len(ids)} tok) {time.perf_counter() - t_seg:.2f}s "
                  f"[experts cached {len(wc.cache)}]", flush=True)
        timing["seconds_forward_total"] += time.perf_counter() - t_seg
        if seg + 1 < len(lens) and args.budget_seconds and time.perf_counter() - t_start > args.budget_seconds:
            timing["slices"] += 1
            timing["segments_per_slice"].append(seg_done)
            if b_out is not None:
                b_out.save_meta(False)
            timing["seconds_state_io"] += save_state(work, layer, hooks, seg + 1, timing)
            print(f"[hessian_q4] L{layer:02d}: budget reached after segment {seg + 1}/{len(lens)}; state saved "
                  f"({timing['seconds_state_io']:.0f}s io total); run again", flush=True)
            return EXIT_MORE
    timing["slices"] += 1
    timing["segments_per_slice"].append(seg_done)
    if b_out is not None:
        b_out.save_meta(True)
    # the pass is complete: keep the state until the files are final (a crash in finalize must not lose the pass)
    timing["seconds_state_io"] += save_state(work, layer, hooks, len(lens), timing)
    timing["cache"] = {"expert_hits": wc.hits, "expert_misses": wc.misses}
    wc.clear()
    cap = finalize_layer(root, layer, hooks, ref, corpus, timing)
    shutil.rmtree(work / f"L{layer:02d}.state", ignore_errors=True)
    if not args.keep_boundary and layer > 0:
        b_in.delete()
    print(f"[hessian_q4] L{layer:02d} DONE: {cap['files']} files, {cap['bytes'] / 2**30:.2f} GiB, fallback "
          f"gate_up {cap['fallback_count']['gate_up']}/{arch.n_experts} down {cap['fallback_count']['down']}/{arch.n_experts}, "
          f"forward {timing['seconds_forward_total']:.0f}s, finalize {timing['finalize_s']:.0f}s, "
          f"peak RSS {cap['peak_rss_gib']:.1f} GiB, peak VRAM {cap['peak_vram_gib']}", flush=True)
    return 0


def propagate_layer(args) -> int:
    """`capture --propagate-only`: run layer L over the corpus and write the layer L+1 residual boundary, no taps, no
    Hessians (a layer whose Hessians are not wanted, e.g. the layers between two measured ones). Resumes at the first
    segment without a crc in the output boundary's meta."""
    t_start = time.perf_counter()
    root, layer = Path(args.root), args.layer
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    corpus_ids, _ = load_corpus(Path(args.tokens))
    lens = [len(s) for s in corpus_ids]
    ref = Qwen4ExpStreamingReference(args.model_dir, args.device, tie_rule="recent", ngram_sidecar=args.ngram_sidecar)
    arch, device, width = ref.arch, ref.device, ref.arch.residual_width
    b_in = Boundary(work, layer, lens, width)
    if layer > 0:
        if not b_in.exists():
            raise SystemExit(f"[hessian_q4] no input boundary {b_in.path}: run layer {layer - 1} first")
        b_in.load_meta()
        if not b_in.meta["complete"]:
            raise SystemExit(f"[hessian_q4] {b_in.meta_path} is not complete")
        b_in.open(create=False)
    b_out = Boundary(work, layer + 1, lens, width)
    if b_out.exists():
        b_out.load_meta()
        if b_out.meta["complete"] and not args.force:
            print(f"[hessian_q4] L{layer:02d}: boundary {layer + 1} complete already", flush=True)
            return 0
    b_out.open(create=True)
    first = next((i for i, c in enumerate(b_out.meta["crc"]) if c is None), len(lens))
    ref.hooks = ReferenceHooks()
    wc = WeightCache(ref.reader)
    layer_mod = ref._build_layer(layer)
    secs = 0.0
    for seg in range(first, len(lens)):
        t_seg = time.perf_counter()
        with torch.no_grad():
            embeds, kw = layer_context(ref, corpus_ids[seg], embed=layer == 0)
            hidden = embeds.repeat(1, 1, arch.hc_count) if layer == 0 else b_in.read(seg, device)
            hidden = layer_mod(hidden, **kw)
        b_out.write(seg, hidden)
        del hidden, embeds, kw
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs += time.perf_counter() - t_seg
        if seg + 1 < len(lens) and args.budget_seconds and time.perf_counter() - t_start > args.budget_seconds:
            b_out.save_meta(False)
            print(f"[hessian_q4] L{layer:02d} propagate: budget reached after segment {seg + 1}/{len(lens)}; run again", flush=True)
            return EXIT_MORE
    b_out.save_meta(True)
    b_out.close()
    wc.clear()
    if not args.keep_boundary and layer > 0:
        b_in.delete()
    print(f"[hessian_q4] L{layer:02d} propagated (no Hessians): {len(lens) - first} segments, forward {secs:.0f}s, "
          f"peak RSS {peak_rss_gib():.1f} GiB", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# status / check
# ---------------------------------------------------------------------------------------------------------------


def cmd_status(args) -> int:
    root = Path(args.root)
    for d in sorted((root / "hessian").glob("L*")):
        done = (d / "DONE.json").exists()
        cap = json.loads((d / "capture.json").read_text(encoding="utf-8")) if (d / "capture.json").exists() else {}
        print(f"{d.name}: {'DONE' if done else 'partial'}  fallback {cap.get('fallback_count')}")
    for p in sorted((root / "work").glob("*")):
        print(f"work/{p.name}")
    return 0


def cmd_check(args) -> int:
    """Gates of a written layer: every key's file exists with matching K; symmetric expansion; finite; positive diagonal;
    rows >= K for every non-fallback expert file; a sample of files PSD (fp64 eigenvalues)."""
    out = layer_dir(Path(args.root), args.layer)
    man = json.loads((out / "hessian.json").read_text(encoding="utf-8"))
    routing = json.loads((out / "routing_counts.json").read_text(encoding="utf-8"))
    problems = []
    fb = {expert_file(args.layer, e, "gu") for e in routing["fallback"]["gate_up"]} | \
         {expert_file(args.layer, e, "down") for e in routing["fallback"]["down"]}
    sampled = 0
    for name, meta in man["files"].items():
        h, rows, trace = hc.read_hess_file(out / name)
        if not np.isfinite(h).all():
            problems.append(f"{name}: non-finite")
        if not np.array_equal(h, h.T):
            problems.append(f"{name}: not symmetric after expansion")
        if h.diagonal().min() <= 0:
            problems.append(f"{name}: min diag {h.diagonal().min()}")
        if h.shape[0] != meta["K"] or rows != meta["rows"]:
            problems.append(f"{name}: K/rows differ from hessian.json")
        if ".e" in name and name not in fb and rows < h.shape[0]:
            problems.append(f"{name}: {rows} rows < K {h.shape[0]} but not a fallback")
        if name in fb and not np.array_equal(h, np.diag(np.diag(h))):
            problems.append(f"{name}: fallback is not diagonal")
        if sampled < args.eig_sample and name not in fb:
            ev = torch.linalg.eigvalsh(torch.from_numpy(h).double())
            if float(ev[0]) < -1e-5 * float(ev[-1]):
                problems.append(f"{name}: min eigenvalue {float(ev[0])} (max {float(ev[-1])})")
            sampled += 1
    print(f"[hessian_q4] check L{args.layer:02d}: {len(man['files'])} files, {sampled} eigen-checked, "
          f"{len(problems)} problem(s)")
    for p in problems[:20]:
        print("  " + p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture", help="capture one layer's Hessians (one time slice; exit 75 = run again)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--tokens", default=str(DEFAULT_TOKENS))
    p.add_argument("--root", default=str(Q4_ROOT))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--ngram-sidecar", default=str(DEFAULT_SIDECAR))
    p.add_argument("--device", default="cuda")
    p.add_argument("--budget-seconds", type=float, default=840.0, help="0 = run to the end (no time slicing)")
    p.add_argument("--propagate-only", action="store_true",
                   help="run the layer and write the next boundary only (no Hessians): layers between measured ones")
    p.add_argument("--keep-boundary", action="store_true")
    p.add_argument("--write-final", action="store_true", help="also write the last layer's output boundary")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=capture_layer)
    p = sub.add_parser("status")
    p.add_argument("--root", default=str(Q4_ROOT))
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("check")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--root", default=str(Q4_ROOT))
    p.add_argument("--eig-sample", type=int, default=2)
    p.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
