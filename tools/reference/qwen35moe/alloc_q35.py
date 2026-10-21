"""`alloc_q35.py`: the per-card-budget allocation of trellis KB over qwen3_5_moe's tensor classes (docs/qwen35moe-plan.md M4; the\ncounterpart of tools/reference/qwen35moe/alloc_q35.py).

    python tools\\reference\\qwen35moe\\alloc_q35.py allocate --src E:\\models\\r4dx\\q35moe\\oracle-k3 --src ...\\oracle-k4
        --out-dir E:\\models\\r4dx\\q35moe\\mix [--cap-gib 26] [--floor p=4,gu=3,down=3]
    python tools\\reference\\qwen35moe\\alloc_q35.py compare --src ...oracle-k3 --src ...oracle-k4      (per-class K vs K table)

Pure python (json only): runs in any interpreter.

Classes. One class per (layer, kind): `p` = the attention or GDN projections (q/k/v/o resp. qkv/z/out), `gu` = every expert's
gate_proj + up_proj, `down` = every expert's down_proj. ONE trellis KB per class (plan 4.11: all experts of a layer's class
share it so one launch config serves them all). A class's candidate Ks are those the `--src` oracle directories have, at or above
its floor (`--floor`, default p=4, gu=3, down=3: projections never below KB 4, experts from KB 3).

Bytes (what the loader allocates, `qwen35moe_container.cpp` Inspect): per linear
`k * n * K / 8` trellis words + `4 k` (suh as fp32) + `4 n` (svh as fp32) + `4 * ceil(n / 128)` tickets; a gate+up linear is two
tensors (its suh/svh per part). Everything that is not a trellis linear (norms, router, shared expert, GDN / attention
small tensors, final norm, lm_head w4a16 g32) is `fixed_layer_bytes` / `tail_bytes`, derived from the config, bf16 as the
plan says. The card split is `choose_split`, a line-for-line port of `r4dx::model::ChooseSplit` (the k in [1, L-1] that
balances device 0 = layers [0, k) and device 1 = layers [k, L) + tail, among those keeping both <= the cap; the lower k on a tie).

Loss (`--loss out`, default). `loss(layer, class, K)` = the expected squared error of the BLOCK output per token caused by
quantizing the class at K, in the same unit for every class. Experts: `sum_e f_e * out_err_e`, `f_e = routed_e / tokens`, `out_err_e`
the mean squared error of the expert's own OUTPUT (3072) over its probe rows (a reservoir sample of 256 routed rows kept by
hessian_q35.py) with only this class quantized (oracle_q35.py `probe_errors`: gate+up reconstructed from the stored bits with the
down weights exact; down reconstructed with exact gate/up): it includes the silu*up path that a linear's own-output proxy
misses. Measured on real layers 0 and 3 (100k-token Hessians, the first 32 experts): per extra GiB of KB 3 -> 4 the local
proxy ranks gate_up 43x (L0) / 32x (L3) ABOVE down, the output-referred loss ranks down 2.9x / 2.7x above gate_up, i.e. the
plan's KB 3 gate_up / KB 4 down start. Projections: `E_mixer * sum_t rel_proxy_t` (relative output errors of the class's tensors added into the
relative error of the mixer output, E_mixer = the energy of out_proj / o_proj's output). `--loss local`: the earlier
definition, `sum` of routing-weighted `proxy_abs = tr(E^T H E)` per tensor, for comparison only. Either way this is a
layer-local proxy: it ranks the classes against each other, it is not a KL (plan M1-13's sweep measures KL).

Allocation. Start every class at its floor (an infeasible start is an error). Then repeat: among classes with a higher K
available, take the one with the largest `(loss(K) - loss(K_next)) / extra bytes` (ties: edge layers first, trellis_quant.allocate's
order min(layer, last - layer)) whose promotion keeps `choose_split` feasible under the cap on BOTH cards, promote it, until none
fits. A promotion that does not fit never fits later (bytes only grow), so it is dropped. The result is written as a mix
manifest (`weights_override.json`, schema of trellis_quant.py cmd_mix: tensors copied record by record from the source
directory of their class's K, `file` made absolute; `allocation` present, `layers_done` absent) plus `allocation.json` /
`allocation_report.md`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen35moe.arch import FULL, GDN, Qwen35MoeArch  # noqa: E402

GIB = 2**30
DEFAULT_CAP_GIB = 26.0
DEFAULT_FLOOR = {"p": 4.0, "gu": 3.0, "down": 3.0}
KINDS = ("p", "gu", "down")
OVERRIDE_FORMAT = "r4dx-weights-override"
MANIFEST_NAME = "weights_override.json"
LAYER_RE = "model.language_model.layers."


# ---------------------------------------------------------------------------------------------------------------
# bytes
# ---------------------------------------------------------------------------------------------------------------


def trellis_tensor_bytes(k: int, n: int, K: float) -> int:
    """One quantized tensor on the device: words + suh + svh (fp32) + tickets (4 B per 128 output rows)."""
    return int(k * n * K) // 8 + 4 * k + 4 * n + 4 * ((n + 127) // 128)


def class_shapes(arch: Qwen35MoeArch, layer: int, kind: str) -> list[tuple[int, int]]:
    """[(k, n)] of the tensors of one class (HF weight [n, k])."""
    q = arch.quantized_linears(layer)
    p = arch.layer_prefix(layer)
    out = []
    for base, (n, k) in q.items():
        rel = base[len(p):]
        is_expert = rel.startswith("mlp.experts.")
        cls = "p" if not is_expert else ("down" if rel.endswith("down_proj") else "gu")
        if cls == kind:
            out.append((k, n))
    return out


def class_bytes(arch: Qwen35MoeArch, layer: int, kind: str, K: float) -> int:
    return sum(trellis_tensor_bytes(k, n, K) for k, n in class_shapes(arch, layer, kind))


def fixed_layer_bytes(arch: Qwen35MoeArch, layer: int) -> int:
    """Everything of a layer that is not a trellis linear, as the container's loader holds it on the device
    (src/model/qwen35moe_container.cpp Inspect: norms + mixer small tensors + router / shared expert / shared gate, all bf16;
    the A_log / dt_bias vectors and the gated-norm weight are fp32)."""
    H, E, Si = arch.hidden, arch.n_experts, arch.shared_inter
    norms = 2 * H * 2
    dense = E * H * 2 + H * 2 + 3 * Si * H * 2                       # router, shared_expert_gate, shared expert gate/up/down
    if arch.layer_types[layer] == GDN:
        mixer = (2 * arch.gdn_value_heads * H * 2                      # in_proj_a, in_proj_b
                 + arch.gdn_qkv_rows * arch.gdn_conv * 2               # conv1d
                 + 2 * arch.gdn_value_heads * 4                        # A_log, dt_bias (fp32)
                 + arch.gdn_value_dim * 4)                             # norm weight widened to fp32
    else:
        mixer = 2 * arch.head_dim * 2 + 2 * arch.kv_heads * 4          # q_norm, k_norm; k/v descale (fp32)
    return norms + dense + mixer


def tail_bytes(arch: Qwen35MoeArch) -> int:
    """Device 1's non-layer bytes: final norm + the w4a16 g32 lm_head (the embedding table stays on the host)."""
    return arch.hidden * 2 + arch.vocab * arch.hidden // 2 + arch.vocab * arch.hidden // 32 * 4

def choose_split(per_layer: list[int], head: int, tail: int, cap: int) -> tuple[int, int, int]:
    """`r4dx::model::ChooseSplit`: (k, dev0_bytes, dev1_bytes). Raises ValueError / RuntimeError like the C++ throws."""
    L = len(per_layer)
    if L < 2:
        raise ValueError(f"need at least two layers to split, got {L}")
    if head < 0 or tail < 0 or cap <= 0:
        raise ValueError("head_bytes, tail_bytes must be >= 0 and cap_bytes > 0")
    if any(b < 0 for b in per_layer):
        raise ValueError("a layer has negative bytes")
    total = sum(per_layer)
    best, best_diff, below = None, 0, 0
    for k in range(1, L):
        below += per_layer[k - 1]
        d0, d1 = head + below, tail + (total - below)
        if d0 > cap or d1 > cap:
            continue
        diff = abs(d0 - d1)
        if best is None or diff < best_diff:
            best, best_diff = (k, d0, d1), diff
    if best is None:
        raise RuntimeError(f"no layer boundary keeps both cards within {cap} bytes: {head + tail + total} bytes of weights "
                           f"({total} in {L} layers, head {head}, tail {tail})")
    return best


class Model:
    """The byte model of one arch: fixed bytes and the class bytes for a K assignment."""

    def __init__(self, arch: Qwen35MoeArch):
        self.arch = arch
        self.L = arch.n_layers
        self.fixed = [fixed_layer_bytes(arch, i) for i in range(self.L)]
        self.tail = tail_bytes(arch)
        self._cb: dict[tuple, int] = {}

    def cbytes(self, layer: int, kind: str, K: float) -> int:
        key = (layer, kind, K)
        if key not in self._cb:
            self._cb[key] = class_bytes(self.arch, layer, kind, K)
        return self._cb[key]

    def per_layer(self, ks: dict[tuple, float]) -> list[int]:
        return [self.fixed[i] + sum(self.cbytes(i, kd, ks[(i, kd)]) for kd in KINDS) for i in range(self.L)]

    def total(self, ks: dict[tuple, float]) -> int:
        return self.tail + sum(self.per_layer(ks))

    def split(self, ks: dict[tuple, float], cap: int) -> tuple[int, int, int]:
        return choose_split(self.per_layer(ks), 0, self.tail, cap)


def uniform_assignment(L: int, p: float, gu: float, down: float) -> dict[tuple, float]:
    return {(i, kd): v for i in range(L) for kd, v in (("p", p), ("gu", gu), ("down", down))}


# ---------------------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------------------


def load_source(path: Path) -> dict:
    path = Path(path)
    man_path = path / MANIFEST_NAME if path.is_dir() else path
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("format") != OVERRIDE_FORMAT or man.get("version") != 1:
        raise ValueError(f"{man_path}: not a {OVERRIDE_FORMAT} v1 manifest")
    ks = {float(r["K"]) for r in man["tensors"].values()}
    if len(ks) != 1:
        raise ValueError(f"{man_path}: not a uniform-rate directory ({sorted(ks)})")
    man["_path"], man["_dir"], man["_K"] = str(man_path.resolve()), str(man_path.resolve().parent), ks.pop()
    return man


def parse_name(name: str) -> tuple[int, str, int | None]:
    """model.language_model.layers.7.mlp.experts.12.gate_proj.weight -> (7, 'gu', 12)."""
    head, _, rest = name[len(LAYER_RE):].partition(".")
    layer = int(head)
    if rest.startswith("mlp.experts."):
        e, _, proj = rest[len("mlp.experts."):].partition(".")
        return layer, ("down" if proj.startswith("down_proj") else "gu"), int(e)
    return layer, "p", None


def class_stats(man: dict, loss_mode: str = "out") -> dict[tuple, dict]:
    """(layer, kind) -> {loss, loss_local, loss_out, rel_err, proxy_rel, n, numel, bits, experts, fallback, names}.

    `loss_local` = sum of routing-weighted tr(E^T H E) of each tensor's own output (module docstring's first definition).
    `loss_out` (the default `loss`) is output-referred, in squared error of the BLOCK output per token:
      experts: sum over experts of f_e * (mean squared error of the expert's output with only this class quantized, on its
               probe rows), f_e = routed_e / tokens -- from the records' `probe` (oracle_q35.py);
      p:       E_mixer * sum over the class's tensors of the relative proxy tr(E^T H E) / tr(W^T H W), E_mixer = the energy
               tr(W^T H W) of the mixer's output projection (out_proj / o_proj) = E |mixer output|^2 per token: relative
               errors of the projections are taken to add into the relative error of the mixer output."""
    if loss_mode not in ("out", "local"):
        raise ValueError(f"loss mode {loss_mode!r}")
    out: dict[tuple, dict] = {}
    for name, r in man["tensors"].items():
        layer, kind, e = parse_name(name)
        c = out.setdefault((layer, kind), {"loss_local": 0.0, "loss_out": 0.0, "proxy_rel_sum": 0.0, "rel_num": 0.0,
                                           "numel": 0, "n": 0, "fallback": set(), "experts": set(), "names": [],
                                           "bits": 0, "probe_experts": set(), "rel_sum_t": 0.0, "mixer_energy": None,
                                           "refit_fb": 0, "refit_fb_rounds": 0})
        h = r.get("hessian", {})
        w = 1.0
        if kind != "p":
            tokens = h.get("tokens") or 0
            w = (h.get("routed", 0) / tokens) if tokens else 0.0
        if r.get("refit_fallbacks"):       # oracle_q35.py: scale-refit solves that raised and went down refit_fallback's ladder
            c["refit_fb"] += 1
            c["refit_fb_rounds"] += int(r["refit_fallbacks"])
        c["loss_local"] += w * r["proxy_abs"]
        c["proxy_rel_sum"] += r["proxy"] * r["numel"]
        c["rel_num"] += (r["rel_weight_err"] ** 2) * r["numel"]
        c["numel"] += r["numel"]
        c["n"] += 1
        c["bits"] += r["bits"]["total"]
        c["names"].append(name)
        if kind != "p":
            c["experts"].add(e)
            if h.get("fallback"):
                c["fallback"].add(e)
            pr = r.get("probe")
            if pr is not None and e not in c["probe_experts"]:  # gate and up carry the same group-level number: count it once
                c["probe_experts"].add(e)
                c["loss_out"] += w * pr["out_err"]
        else:
            c["rel_sum_t"] += r["proxy"]
            if r.get("tap") == "out":
                c["mixer_energy"] = r["energy"]
    for key, c in out.items():
        c["proxy_rel"] = c["proxy_rel_sum"] / c["numel"]       # numel-weighted mean of the relative proxy
        c["rel_err"] = math.sqrt(c["rel_num"] / c["numel"])    # rms of the relative weight error
        if key[1] == "p":
            if c["mixer_energy"] is None:
                raise ValueError(f"layer {key[0]}: no out_proj / o_proj record in the projection class")
            c["loss_out"] = c["mixer_energy"] * c["rel_sum_t"]
        elif loss_mode == "out" and c["probe_experts"] != c["experts"]:
            raise ValueError(f"layer {key[0]} {key[1]}: {len(c['experts'] - c['probe_experts'])} expert(s) have no probe error "
                             f"(oracle run without the layer's probe file): rerun, or --loss local")
        c["experts"] = len(c["experts"])
        c["fallback"] = len(c["fallback"])
        c["loss"] = c["loss_out"] if loss_mode == "out" else c["loss_local"]
    return out


# ---------------------------------------------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------------------------------------------


def parse_floor(s: str | None) -> dict[str, float]:
    floor = dict(DEFAULT_FLOOR)
    if s:
        for part in s.split(","):
            k, _, v = part.partition("=")
            if k not in floor:
                raise ValueError(f"--floor: unknown class {k!r} (p, gu, down)")
            floor[k] = float(v)
    return floor


def allocate(model: Model, stats: dict[float, dict[tuple, dict]], cap: int, floor: dict[str, float],
             layers: list[int] | None = None) -> tuple[dict[tuple, float], list[dict]]:
    """Greedy promotion (module docstring). `stats[K][(layer, kind)]` carry `loss`. Returns (K per class, promotion log)."""
    Ks = sorted(stats)
    L = model.L
    layers = list(range(L)) if layers is None else layers
    last = L - 1
    ks: dict[tuple, float] = {}
    for i in range(L):
        for kd in KINDS:
            cand = [K for K in Ks if K >= floor[kd] and (i, kd) in stats[K]]
            if not cand and i in layers:
                raise ValueError(f"no source has layer {i} class {kd} at K >= {floor[kd]:g}")
            ks[(i, kd)] = cand[0] if cand else floor[kd]
    model.split(ks, cap)  # raises when even the floor does not fit
    log: list[dict] = []
    dead: set[tuple] = set()
    while True:
        best, best_key = None, None
        for (i, kd), K in ks.items():
            if (i, kd) in dead:
                continue
            nxt = [x for x in Ks if x > K and (i, kd) in stats[x]]
            if not nxt:
                continue
            K2 = nxt[0]
            gain = stats[K][(i, kd)]["loss"] - stats[K2][(i, kd)]["loss"]
            extra = model.cbytes(i, kd, K2) - model.cbytes(i, kd, K)
            ratio = gain / extra * GIB if extra > 0 else float("inf")
            key = (-ratio, min(i, last - i), i, KINDS.index(kd))
            if gain <= 0:
                dead.add((i, kd))
                continue
            if best_key is None or key < best_key:
                best, best_key = (i, kd, K, K2, gain, extra, ratio), key
        if best is None:
            break
        i, kd, K, K2, gain, extra, ratio = best
        trial = dict(ks)
        trial[(i, kd)] = K2
        try:
            model.split(trial, cap)
        except RuntimeError:
            dead.add((i, kd))
            continue
        ks = trial
        log.append({"layer": i, "class": kd, "from": K, "to": K2, "gain": gain, "extra_bytes": extra,
                    "gain_per_gib": ratio})
    return ks, log


def total_loss(stats: dict[float, dict[tuple, dict]], ks: dict[tuple, float]) -> float:
    return sum(stats[K][key]["loss"] for key, K in ks.items())


# ---------------------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------------------


def write_json_atomic(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def build_report(model: Model, stats, ks, cap: int, log, sources: dict[float, dict], floor) -> tuple[dict, str]:
    L = model.L
    pl = model.per_layer(ks)
    k, d0, d1 = model.split(ks, cap)
    total = model.total(ks)
    by_kind = {kd: {"bytes": 0, "K": {}, "loss": 0.0} for kd in KINDS}
    for (i, kd), K in ks.items():
        b = by_kind[kd]
        b["bytes"] += model.cbytes(i, kd, K)
        b["K"][f"{K:g}"] = b["K"].get(f"{K:g}", 0) + 1
        b["loss"] += stats[K][(i, kd)]["loss"]
    fixed = sum(model.fixed) + model.tail
    loss = total_loss(stats, ks)
    baselines = {}
    for K in sorted(stats):
        try:
            u = {key: (K if key[1] != "p" or K >= floor["p"] else floor["p"]) for key in ks}
            if all(key in stats[u[key]] for key in u):
                baselines[f"uniform K{K:g} (p at max(K, floor))"] = {
                    "loss": total_loss(stats, u), "total_gib": model.total(u) / GIB}
        except KeyError:
            pass
    # fallback share per kind: experts whose Hessian is the fallback diagonal (count and loss share)
    rfb = {kd: {"tensors": 0, "classes": 0, "rounds": 0} for kd in KINDS}
    for (i, kd), K in ks.items():
        st = stats[K][(i, kd)]
        rfb[kd]["tensors"] += st["refit_fb"]
        rfb[kd]["rounds"] += st["refit_fb_rounds"]
        rfb[kd]["classes"] += 1 if st["refit_fb"] else 0
    fb = {}
    for kd in ("gu", "down"):
        n = nfb = 0
        lo = lo_fb = 0.0
        for (i, kd2), K in ks.items():
            if kd2 != kd:
                continue
            st = stats[K][(i, kd)]
            n += st["experts"]
            nfb += st["fallback"]
        fb[kd] = {"experts": n, "fallback": nfb, "share": (nfb / n if n else 0.0)}
    doc = {"cap_gib": cap / GIB, "k": k, "device0_gib": d0 / GIB, "device1_gib": d1 / GIB, "total_gib": total / GIB,
           "fixed_gib": fixed / GIB, "trellis_gib": (total - fixed) / GIB, "layers": L,
           "by_kind": {kd: {"gib": v["bytes"] / GIB, "K_counts": v["K"], "loss": v["loss"]} for kd, v in by_kind.items()},
           "expected_loss": loss, "baselines": baselines, "fallback": fb, "refit_fallbacks": rfb, "floor": floor,
           "sources": {f"{K:g}": s["_path"] for K, s in sources.items()}, "promotions": len(log),
           "per_layer": [{"layer": i, "gib": pl[i] / GIB, "K": {kd: ks[(i, kd)] for kd in KINDS},
                          "loss": {kd: stats[ks[(i, kd)]][(i, kd)]["loss"] for kd in KINDS}} for i in range(L)]}
    lines = ["# qwen3_5_moe trellis allocation", "",
             f"cap {cap / GIB:.2f} GiB per card; split k = {k}: device 0 {d0 / GIB:.3f} GiB (layers 0..{k - 1}), "
             f"device 1 {d1 / GIB:.3f} GiB (layers {k}..{L - 1} + final norm + lm_head); total {total / GIB:.3f} GiB "
             f"({fixed / GIB:.3f} fixed + {(total - fixed) / GIB:.3f} trellis linears incl. sign vectors and tickets)", "",
             "| class | GiB | K counts (layers) | expected loss |", "|---|--:|---|--:|"]
    for kd in KINDS:
        v = doc["by_kind"][kd]
        lines.append(f"| {kd} | {v['gib']:.3f} | {v['K_counts']} | {v['loss']:.6g} |")
    lines += ["", f"expected proxy loss (sum over classes, per token): {loss:.6g}", ""]
    for name, b in baselines.items():
        lines.append(f"- {name}: loss {b['loss']:.6g}, {b['total_gib']:.3f} GiB")
    lines += ["", f"fallback experts (diagonal Hessian): gate_up {fb['gu']['fallback']}/{fb['gu']['experts']} "
                  f"({fb['gu']['share']:.1%}), down {fb['down']['fallback']}/{fb['down']['experts']} ({fb['down']['share']:.1%})",
              "", "scale-refit solver fallbacks (refit_fallback.py: a solve that raised was redone in float64 / ridged / min-norm): "
              + (", ".join(f"{kd} {v['tensors']} tensor(s) in {v['classes']} class(es), {v['rounds']} round(s)" for kd, v in rfb.items() if v["tensors"])
                 if sum(v["tensors"] for v in rfb.values()) else "none"),
              "", f"{len(log)} promotions in order of loss reduction per GiB; the first 12:"]
    for p in log[:12]:
        lines.append(f"  - L{p['layer']:02d} {p['class']} K{p['from']:g} -> K{p['to']:g}: -{p['gain']:.4g} for "
                     f"{p['extra_bytes'] / GIB:.4f} GiB ({p['gain_per_gib']:.4g}/GiB)")
    lines += ["", "| layer | GiB | p | gu | down | loss p | loss gu | loss down |", "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in doc["per_layer"]:
        lines.append(f"| {r['layer']} | {r['gib']:.3f} | {r['K']['p']:g} | {r['K']['gu']:g} | {r['K']['down']:g} | "
                     f"{r['loss']['p']:.4g} | {r['loss']['gu']:.4g} | {r['loss']['down']:.4g} |")
    return doc, "\n".join(lines) + "\n"


def cmd_allocate(args) -> int:
    srcs = [load_source(Path(p)) for p in args.src]
    by_K = {s["_K"]: s for s in srcs}
    if len(by_K) != len(srcs):
        raise SystemExit("[alloc_q35] two sources with the same K")
    base = srcs[0]
    for s in srcs:
        for key in ("config_sha256", "hessian_basis", "codebook"):
            if s.get(key) != base.get(key):
                raise SystemExit(f"[alloc_q35] sources differ in {key}")
    arch = Qwen35MoeArch.from_config(Path(base["model_dir"]) / "config.json")
    model = Model(arch)
    stats = {K: class_stats(s, args.loss) for K, s in by_K.items()}
    cap = int(args.cap_gib * GIB) if args.cap_bytes is None else args.cap_bytes
    floor = parse_floor(args.floor)
    ks, log = allocate(model, stats, cap, floor)
    doc, text = build_report(model, stats, ks, cap, log, by_K, floor)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    counts: dict[str, int] = {}
    for (i, kd), K in sorted(ks.items()):
        src = by_K[K]
        for name in stats[K][(i, kd)]["names"]:
            rec = dict(src["tensors"][name])
            f = Path(rec["file"])
            rec["file"] = str(f if f.is_absolute() else Path(src["_dir"]) / f)
            tensors[name] = rec
            counts[f"{K:g}"] = counts.get(f"{K:g}", 0) + 1
    expected = {n + ".weight" for i in range(arch.n_layers) for n in arch.quantized_linears(i)}
    missing = sorted(expected - set(tensors))
    extra = sorted(set(tensors) - expected)
    if extra:
        raise SystemExit(f"[alloc_q35] tensors outside the model: {extra[:3]}")
    man = {"format": OVERRIDE_FORMAT, "version": 1, "encoding": "trellis-exl3", "complete": not missing,
           "missing": missing[:32], "missing_count": len(missing), "bpw_target": None, "K_uniform": None,
           **{k: base.get(k) for k in ("model_dir", "config_sha256", "hessian_basis", "codebook", "hessian_manifest_sha256",
                                       "recipe")},
           "allocation": {"rule": "tools/reference/qwen35moe/alloc_q35.py: classes (layer, p|gu|down), one KB per class, start at "
                                  "the floor, promote by (proxy loss reduction per GiB) under choose_split's per-card cap",
                          "cap_gib": args.cap_gib if args.cap_bytes is None else args.cap_bytes / GIB,
                          "floor": floor, "tensors_per_K": counts, "split": {"k": doc["k"], "device0_gib": doc["device0_gib"],
                                                                            "device1_gib": doc["device1_gib"]},
                          "sources": {f"{K:g}": s["_path"] for K, s in by_K.items()},
                          "source_code_sha256": {f"{K:g}": s.get("code_sha256") for K, s in by_K.items()},
                          "class_K": {f"{i}.{kd}": K for (i, kd), K in sorted(ks.items())}},
           "summary": {"n_tensors": len(tensors), "decode_gib_model_total": doc["total_gib"], "tensors_per_K": counts},
           "tensors": tensors,
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json_atomic(out / MANIFEST_NAME, man)
    write_json_atomic(out / "allocation.json", dict(doc, promotion_log=log))
    (out / "allocation_report.md").write_text(text, encoding="utf-8")
    print(text.split("\n\n")[1] if "\n\n" in text else text)
    print(f"[alloc_q35] {out / MANIFEST_NAME}: complete={man['complete']}, tensors per K {counts}, "
          f"k={doc['k']} dev0 {doc['device0_gib']:.3f} GiB dev1 {doc['device1_gib']:.3f} GiB")
    return 0


def cmd_compare(args) -> int:
    srcs = {s["_K"]: s for s in (load_source(Path(p)) for p in args.src)}
    stats = {K: class_stats(s, args.loss) for K, s in srcs.items()}
    keys = sorted(set.intersection(*[set(s) for s in stats.values()]))
    Ks = sorted(stats)
    print("layer class  " + "  ".join(f"K{K:g}: proxy_rel rel_err loss(w) bytes_MiB" for K in Ks) + "   fallback/experts")
    for key in keys:
        row = f"{key[0]:>5} {key[1]:<5}"
        for K in Ks:
            s = stats[K][key]
            row += f"  K{K:g}: {s['proxy_rel']:.5f} {s['rel_err']:.5f} {s['loss']:.4g} {s['bits'] / 8 / 2**20:8.1f}"
        s = stats[Ks[0]][key]
        row += f"   {s['fallback']}/{s['experts']}" if key[1] != "p" else ""
        print(row)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("allocate")
    p.add_argument("--src", action="append", required=True, help="a uniform-K oracle directory (repeat: K3, K4, [K5])")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cap-gib", type=float, default=DEFAULT_CAP_GIB)
    p.add_argument("--cap-bytes", type=int, default=None, help="overrides --cap-gib (tests)")
    p.add_argument("--floor", default=None, help="p=4,gu=3,down=3")
    p.add_argument("--loss", default="out", choices=["out", "local"], help="class loss: output-referred (default) or local proxy")
    p.set_defaults(fn=cmd_allocate)
    p = sub.add_parser("compare")
    p.add_argument("--src", action="append", required=True)
    p.add_argument("--loss", default="out", choices=["out", "local"])
    p.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
