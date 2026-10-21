"""tools/quant2/qwen35moe/oracle_q35.py -- the trellis oracle of one qwen3_5_moe layer at one rate (docs/qwen35moe-plan.md M4;
the counterpart of tools/quant2/qwen4exp/oracle_q4.py).

    <python>  tools\\quant2\\qwen35moe\\oracle_q35.py quantize --layer 3 --K 3 --out-dir E:\\models\\r4dx\\q35moe\\oracle-k3
    ... oracle_q35.py manifest --out-dir E:\\models\\r4dx\\q35moe\\oracle-k3
    ... oracle_q35.py verify --layer 3 --out-dir ...oracle-k3@3 --out-dir ...oracle-k4@4 [--delete-hessians]

`<python>` is the reference venv (D:\\venvs\\r4dx-qwen4-ref, torch 2.9.1+rocm): `trellis_quant.block_ldl` factors the K x K
Hessian on the CPU when the tensor lives on the GPU (torch 2.9.1 has no MAGMA), so no other venv is needed. GPU runs only
through `tools\\with_lock.ps1 -Name gpu` (HIP device 1, or 0 with R4DX_ALLOW_GPU0=1).

It reads the layer's Hessians from `<hessian-root>\\hessian\\L{ii}\\` (hessian_q35.py; keys: checkpoint base name -> .hess), the
bf16 weights through `qwen35moe.ref.CheckpointReader` (experts sliced out of the fused 3-D tensors under their virtual
per-expert names), and quantizes with `trellis_quant.quantize_group` -- the function `trellis_quant.py quantize-model` calls per
shared-Hessian group -- in the `matched` Hessian basis (the only one r4dx-convert imports), with the seeds quantize-model draws
(`su` per (layer, tap), `sv` per tensor). Groups of one shared input:
  projections:  GDN {in_proj_qkv, in_proj_z} and {out_proj}; full attention {q_proj, k_proj, v_proj} and {o_proj}
  experts:      {gate_proj, up_proj} (one Hessian) and {down_proj}, per expert
A layer is written as chunks (`p` = the projection groups, `e00`.. = --chunk-experts experts each): `L{ii}.{chunk}.safetensors`
(`<hf name>.trellis/.suh/.svh`) + `L{ii}.{chunk}.json` (the per-tensor records of the weights_override manifest, K, proxy, the
absolute proxy tr(E^T H E) and the energy tr(W^T H W), the Hessian provenance incl. the routed count and fallback flag). Chunks
are skipped when their record matches this job and the layer's Hessian DONE.json; a time budget (`--budget-seconds`) ends a
run between chunks with exit code 75 ("run me again").

`manifest` merges every usable chunk of the directory into `weights_override.json` (format r4dx-weights-override v1, encoding
trellis-exl3, `complete` true only when every linear of the classes the directory was run for is present).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "reference"))

import torch  # noqa: E402

import trellis_quant as tq  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refit_fallback  # noqa: E402
from qwen35moe.arch import FULL, GDN, Qwen35MoeArch  # noqa: E402
from qwen35moe.ref import CheckpointReader  # noqa: E402

MODELS = Path(r"E:\models")
Q4_ROOT = MODELS / "r4dx" / "q35moe"
MODEL_DIR = MODELS / "r4dx" / "Ornith-1.5-122B"
EXIT_MORE = 75
CLASSES = {"all": ("p", "e"), "proj": ("p",), "experts": ("e",)}
PROJ_GROUPS = {
    GDN: [("in", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]), ("out", ["linear_attn.out_proj"])],
    FULL: [("in", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), ("out", ["self_attn.o_proj"])],
}


def sha256_path(p: Path) -> str:
    return tq.sha256_path(p)


def peak_rss_gib() -> float:
    if sys.platform != "win32":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.kernel32.K32GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2**30


def class_of(base_module: str) -> str:
    """'p' (projection) | 'gu' | 'down' of a checkpoint module path under a layer."""
    if ".experts." in base_module or base_module.startswith("mlp.experts."):
        return "down" if base_module.endswith("down_proj") else "gu"
    return "p"


# ---------------------------------------------------------------------------------------------------------------
# job / chunks
# ---------------------------------------------------------------------------------------------------------------


def make_job(model_dir: Path, K: float) -> dict:
    return {"model_dir": str(model_dir), "config_sha256": sha256_path(model_dir / "config.json"), "K": float(K),
            "hessian_basis": "matched", "codebook": "mul1",
            "recipe": dict(tq.RECIPE, qwen35moe="one shared-Hessian group per distinct input; experts: {gate, up} + {down}"),
            "code_sha256": tq.code_sha256()}


def chunk_ids(arch: Qwen35MoeArch, classes: str, chunk_experts: int) -> list[str]:
    ids = []
    if "p" in CLASSES[classes]:
        ids.append("p")
    if "e" in CLASSES[classes]:
        ids += [f"e{c:02d}" for c in range((arch.n_experts + chunk_experts - 1) // chunk_experts)]
    return ids


def chunk_groups(arch: Qwen35MoeArch, layer: int, chunk: str, chunk_experts: int) -> list[dict]:
    """[{tap, bases [checkpoint base names], cls}] of one chunk."""
    prefix = arch.layer_prefix(layer)
    if chunk == "p":
        return [{"tap": tap, "bases": [prefix + m for m in mods], "cls": "p"}
                for tap, mods in PROJ_GROUPS[arch.layer_types[layer]]]
    c = int(chunk[1:])
    out = []
    for e in range(c * chunk_experts, min((c + 1) * chunk_experts, arch.n_experts)):
        out.append({"tap": f"e{e}.gu", "bases": [arch.expert_base(layer, e, "gate_proj"), arch.expert_base(layer, e, "up_proj")],
                    "cls": "gu", "expert": e})
        out.append({"tap": f"e{e}.down", "bases": [arch.expert_base(layer, e, "down_proj")], "cls": "down", "expert": e})
    return out


def chunk_paths(out_dir: Path, layer: int, chunk: str) -> tuple[Path, Path]:
    return out_dir / f"L{layer:02d}.{chunk}.safetensors", out_dir / f"L{layer:02d}.{chunk}.json"


def hessian_done_sha(hroot: Path, layer: int) -> str | None:
    p = hroot / "hessian" / f"L{layer:02d}" / "DONE.json"
    return sha256_path(p) if p.exists() else None


def eol_variants(path: Path) -> set[str]:
    """sha256 of a source file as it is, with LF endings, and with CRLF endings. `trellis_quant.code_sha256` hashes the working-tree
    BYTES, so the same source hashes differently in a checkout with CRLF files (the main checkout) and one with LF files (a worktree
    under .gitattributes eol=lf): a chunk written from one would look stale from the other."""
    data = Path(path).read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    return {hashlib.sha256(x).hexdigest() for x in (data, lf, lf.replace(b"\n", b"\r\n"))}


def code_hash_matches(stored: dict, current: dict) -> bool:
    """`job["code_sha256"]` equality up to line endings (and nothing else: a changed source still mismatches)."""
    files = {"trellis_quant": Path(tq.__file__), "trellis_viterbi": Path(tq.NATIVE_SRC)}
    if not isinstance(stored, dict) or stored.keys() != current.keys():
        return False
    return all(stored[k] == current[k] or (k in files and stored[k] in eol_variants(files[k])) for k in current)


def jobs_match(a: dict, b: dict) -> bool:
    from common import same_path

    return isinstance(a, dict) and a.keys() == b.keys() and all(
        same_path(a[k], b[k]) if k == "model_dir" else code_hash_matches(a[k], b[k]) if k == "code_sha256" else a[k] == b[k]
        for k in a)


def chunk_usable(out_dir: Path, layer: int, chunk: str, job: dict, hess_sha: str | None, chunk_experts: int) -> bool:
    st, js = chunk_paths(out_dir, layer, chunk)
    if not (st.exists() and js.exists()):
        return False
    doc = json.loads(js.read_text(encoding="utf-8"))
    if not jobs_match(doc.get("job"), job) or doc.get("chunk_experts") != chunk_experts:
        return False
    if hess_sha is not None and doc.get("hessian_done_sha256") != hess_sha:
        return False
    return all(r.get("file_sha256") for r in doc["tensors"].values())


# ---------------------------------------------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------------------------------------------


class CohortEncoder:
    """The TrellisEncoder surface `trellis_quant.quantize_group` uses (quantize / cb / codebook / tiles), shared by N worker
    threads each running `quantize_group` on its own tensors: an encode request waits until every ACTIVE worker has
    submitted one (or finished), then the tiles of all of them go to the real encoder in ONE call. The Viterbi encoder
    is exact per tile, so every worker gets the states it would have got alone (bit for bit; checked by
    tests/reference/test_q35moe_quant.py --e2e), but a 1024-column tensor's LDLQ calls carry 64 tiles each and leave the
    gfx1201 mostly idle: 32 experts at once make them 2,048."""

    def __init__(self, base, n_workers: int):
        import threading

        self.base = base
        self.cb, self.codebook, self.device = base.cb, base.codebook, base.device
        self.cv = threading.Condition()
        self.active = n_workers
        self.waiting: dict = {}
        self.results: dict = {}
        self._tl = threading.local()
        self.calls = 0
        self.tiles_batched = 0

    @property
    def tiles(self) -> int:  # per worker, so quantize_group's `enc.tiles - before` records its own tiles
        return getattr(self._tl, "tiles", 0)

    def quantize(self, x, K):
        import threading

        self._tl.tiles = self.tiles + int(x.shape[0])
        tid = threading.get_ident()
        with self.cv:
            self.waiting[tid] = (x, float(K))
            if len(self.waiting) >= self.active:
                self._dispatch()
            while tid not in self.results:
                self.cv.wait()
            res = self.results.pop(tid)
        if isinstance(res, BaseException):
            raise res
        return res

    def _dispatch(self) -> None:  # cv held
        items = list(self.waiting.items())
        self.waiting.clear()
        by_k: dict = {}
        for tid, (x, K) in items:
            by_k.setdefault(K, []).append((tid, x))
        try:
            for K, group in by_k.items():
                cat = torch.cat([x for _, x in group]) if len(group) > 1 else group[0][1]
                vals, st = self.base.quantize(cat, K)
                self.calls += 1
                self.tiles_batched += int(cat.shape[0])
                o = 0
                for tid, x in group:
                    n = int(x.shape[0])
                    self.results[tid] = (vals[o:o + n], st[o:o + n])
                    o += n
        except BaseException as e:  # noqa: BLE001 -- every waiting worker gets the failure
            for tid, _ in items:
                self.results[tid] = e
        self.cv.notify_all()

    def leave(self) -> None:
        with self.cv:
            self.active -= 1
            if self.waiting and len(self.waiting) >= self.active:
                self._dispatch()


_LA_LOCK = None


def guard_linear_algebra() -> None:
    """block_ldl (cholesky_ex + inverse) and refit_scales (solve) are serialized across worker threads: the GPU solvers are
    not promised thread-safe on ROCm; the encoder calls and the GEMMs of the other workers run meanwhile."""
    global _LA_LOCK
    import threading

    if _LA_LOCK is not None:
        return
    _LA_LOCK = threading.RLock()
    for name in ("block_ldl", "refit_scales"):
        orig = getattr(tq, name)

        def locked(*a, _orig=orig, **k):
            with _LA_LOCK:
                return _orig(*a, **k)

        setattr(tq, name, locked)


PROBE: dict | None = None   # {"x": [E, P, 3072] bf16 (cpu), "n": [E]}: the layer's routed-row sample (hessian_q35.py)


def probe_errors(arch, reader, layer: int, g: dict, res: dict, weights: dict, enc, device, K: float) -> dict | None:
    """Output-referred error of one expert group on its probe rows: the expert's OUTPUT (3072) with only this group
    quantized (gate+up: their stored-bits reconstruction, down in full precision; down: full-precision gate/up) against the
    unquantized expert, mean squared error and mean squared output per routed row. Commensurate across the expert classes
    (and with the mixer's output error), which the layer-local proxy tr(E^T H E) of a linear's own output is not."""
    if PROBE is None or g["cls"] == "p":
        return None
    import torch.nn.functional as F

    e = g["expert"]
    n = int(PROBE["n"][e])
    if n == 0:
        return {"rows": 0, "out_err": 0.0, "out_energy": 0.0}
    x = PROBE["x"][e, :n].to(device).float()

    def full(proj):
        return reader.linear_weight(arch.expert_base(layer, e, proj), device).float()      # [N, K]

    def hat(name):                                                                          # [k, n] (in, out)
        r = res[name]
        return tq.reconstruct(r["words"].to(device), r["suh"].to(device), r["svh"].to(device), K, enc.cb)

    names = [b + ".weight" for b in g["bases"]]
    if g["cls"] == "gu":
        wg, wu, wd = weights[names[0]].float(), weights[names[1]].float(), full("down_proj")
        y = (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T
        yh = (F.silu(x @ hat(names[0])) * (x @ hat(names[1]))) @ wd.T
    else:
        wg, wu, wd = full("gate_proj"), full("up_proj"), weights[names[0]].float()
        h = F.silu(x @ wg.T) * (x @ wu.T)
        y, yh = h @ wd.T, h @ hat(names[0])
    return {"rows": n, "out_err": float(((yh - y) ** 2).sum(1).mean()), "out_energy": float((y ** 2).sum(1).mean())}


def quantize_group_record(args, arch, reader, enc, layer: int, g: dict, routing: dict, hman: dict, hdir: Path, device,
                          st_name: str, K: float) -> tuple[dict, dict]:
    """One shared-Hessian group -> (tensors {name.trellis/.suh/.svh: tensor}, records {name: record})."""
    tokens = routing["tokens"]
    fb_gu, fb_dn = set(routing["fallback"]["gate_up"]), set(routing["fallback"]["down"])
    hfile = hman["keys"][g["bases"][0]]
    for b in g["bases"]:
        if hman["keys"][b] != hfile:
            raise RuntimeError(f"{b}: Hessian key {hman['keys'][b]} != {hfile} of its group")
    H, rows = tq.read_hess(hdir / hfile, device)
    names = [b + ".weight" for b in g["bases"]]
    weights = {n: reader.linear_weight(b, device) for n, b in zip(names, g["bases"])}
    res = tq.quantize_group(weights, H, {n: K for n in names}, enc,
                            seed_group=tq.stable_seed("su", layer, g["tap"]),
                            seeds={n: tq.stable_seed("sv", n) for n in names},
                            hessian_basis="matched", verbose=args.verbose)
    hrec = {"file": hfile, "rows": rows, "key": g["bases"][0], "tokens": tokens}
    n_fb = sum(refit_fallback.count_fallbacks(res[n]["record"]) for n in names)
    if n_fb:
        stages = sorted({r["fallback"]["stage"] for n in names for r in res[n]["record"]["refit"] if "fallback" in r})
        print(f"[oracle_q35] WARNING L{layer:02d} {g['tap']}: {n_fb} scale-refit solve(s) raised, solved by {stages} "
              f"(record: refit_fallbacks)", flush=True)
    probe = probe_errors(arch, reader, layer, g, res, weights, enc, device, K)
    if g["cls"] != "p":
        e = g["expert"]
        hrec.update({"routed": routing["tokens_routed"][e], "fallback": e in (fb_gu if g["cls"] == "gu" else fb_dn)})
    tensors_out, records = {}, {}
    for n, b in zip(names, g["bases"]):
        r = res[n]
        Wf = weights[n].float().T.contiguous()
        energy = tq.hessian_energy(Wf, H)
        rec = r["record"]
        if refit_fallback.count_fallbacks(rec):  # only then: a group that never fell back keeps the record it always had
            rec["refit_fallbacks"] = refit_fallback.count_fallbacks(rec)
        rec.update({"layer": layer, "class": g["cls"], "tap": g["tap"], "module": b[len(arch.layer_prefix(layer)):],
                    "numel": rec["k"] * rec["n"], "energy": energy, "proxy_abs": rec["proxy"] * energy,
                    "hessian": hrec, "file": st_name, **({"probe": probe} if probe is not None else {})})
        tensors_out[n + ".trellis"] = r["words"].contiguous()
        tensors_out[n + ".suh"] = r["suh"].contiguous()
        tensors_out[n + ".svh"] = r["svh"].contiguous()
        records[n] = rec
        del Wf
    return tensors_out, records


def quantize_chunk(args, arch, reader, enc, layer: int, chunk: str, job: dict, routing: dict, hman: dict, hess_sha: str,
                   hdir: Path, device) -> dict:
    t_chunk = time.perf_counter()
    out_dir = Path(args.out_dir)
    st_path, js_path = chunk_paths(out_dir, layer, chunk)
    K = float(args.K)
    tensors_out: dict[str, torch.Tensor] = {}
    records: dict[str, dict] = {}
    seconds_by_class = {"p": 0.0, "gu": 0.0, "down": 0.0}
    groups = chunk_groups(arch, layer, chunk, args.chunk_experts)
    batched = {"calls": 0, "tiles": 0}
    for cls in ("p", "gu", "down"):
        cohort = [g for g in groups if g["cls"] == cls]
        if not cohort:
            continue
        t0 = time.perf_counter()
        if args.cohort > 1 and len(cohort) > 1:
            import concurrent.futures as cf

            guard_linear_algebra()
            parts = []
            for s in range(0, len(cohort), args.cohort):
                sub = cohort[s:s + args.cohort]
                ce = CohortEncoder(enc, len(sub))

                def work(g, ce=ce):
                    try:
                        return quantize_group_record(args, arch, reader, ce, layer, g, routing, hman, hdir, device, st_path.name, K)
                    finally:
                        ce.leave()

                with cf.ThreadPoolExecutor(max_workers=len(sub)) as ex:
                    parts += [f.result() for f in [ex.submit(work, g) for g in sub]]
                batched["calls"] += ce.calls
                batched["tiles"] += ce.tiles_batched
        else:
            parts = [quantize_group_record(args, arch, reader, enc, layer, g, routing, hman, hdir, device, st_path.name, K)
                     for g in cohort]
        for t, r in parts:
            tensors_out.update(t)
            records.update(r)
        seconds_by_class[cls] += time.perf_counter() - t0
    from safetensors.torch import save_file

    tmp = st_path.with_name(st_path.name + ".tmp")
    save_file(tensors_out, str(tmp), metadata={"format": tq.OVERRIDE_FORMAT, "layer": str(layer), "chunk": chunk})
    os.replace(tmp, st_path)
    fsha = sha256_path(st_path)
    for r in records.values():
        r["file_sha256"] = fsha
    secs = time.perf_counter() - t_chunk
    tq.write_json_atomic(js_path, {
        "layer": layer, "chunk": chunk, "chunk_experts": args.chunk_experts, "job": job, "hessian_done_sha256": hess_sha,
        "hessian_json_sha256": sha256_path(hdir / "hessian.json"), "tensors": records, "seconds": secs,
        "seconds_by_class": seconds_by_class, "cohort": {"workers": args.cohort, **batched}, "peak_rss_gib": peak_rss_gib(),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
        "provenance": tq.provenance(enc)})
    return {"seconds": secs, "tensors": len(records), "bytes": st_path.stat().st_size}


def cmd_quantize(args) -> int:
    t_start = time.perf_counter()
    from common import resolve_device

    model_dir = Path(args.model_dir)
    arch = Qwen35MoeArch.from_config(model_dir / "config.json")
    layer = args.layer
    hroot = Path(args.hessian_root)
    hdir = hroot / "hessian" / f"L{layer:02d}"
    hess_sha = hessian_done_sha(hroot, layer)
    if hess_sha is None:
        raise SystemExit(f"[oracle_q35] {hdir} has no DONE.json: capture layer {layer} first")
    hman = json.loads((hdir / "hessian.json").read_text(encoding="utf-8"))
    routing = json.loads((hdir / "routing_counts.json").read_text(encoding="utf-8"))
    job = make_job(model_dir, args.K)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in chunk_ids(arch, args.classes, args.chunk_experts)
            if args.force or not chunk_usable(out_dir, layer, c, job, hess_sha, args.chunk_experts)]
    if not todo:
        print(f"[oracle_q35] L{layer:02d} K{args.K:g}: all chunks usable already", flush=True)
        return 0
    device = resolve_device(args.device)
    refit_fallback.install(tq)  # before guard_linear_algebra() wraps tq.refit_scales in the solver lock
    enc = tq.TrellisEncoder(args.backend, device, threads=args.threads, grid=args.grid)
    print(f"[oracle_q35] encoder {enc.describe()}", flush=True)
    enc.check_against_reference({float(args.K)})
    reader = CheckpointReader(model_dir)
    global PROBE
    probe_path = hdir / f"L{layer:02d}.probe.safetensors"
    if probe_path.exists():
        from safetensors.torch import load_file

        PROBE = load_file(str(probe_path))
        print(f"[oracle_q35] probe rows: {int(PROBE['n'].sum())} over {len(PROBE['n'])} experts ({probe_path.name})", flush=True)
    else:
        PROBE = None
        print(f"[oracle_q35] WARNING: no {probe_path.name}: expert records get no output-referred probe error", flush=True)
    for i, c in enumerate(todo):
        info = quantize_chunk(args, arch, reader, enc, layer, c, job, routing, hman, hess_sha, hdir, device)
        print(f"[oracle_q35] L{layer:02d} K{args.K:g} chunk {c} done in {info['seconds']:.1f}s ({info['tensors']} tensors, "
              f"{info['bytes'] / 2**20:.0f} MiB; total {time.perf_counter() - t_start:.0f}s, encoder {enc.tiles} tiles "
              f"{enc.tiles / max(enc.seconds, 1e-9):.0f} tiles/s)", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if i + 1 < len(todo) and args.budget_seconds and time.perf_counter() - t_start > args.budget_seconds:
            print(f"[oracle_q35] L{layer:02d} K{args.K:g}: budget reached, {len(todo) - i - 1} chunk(s) left; run again", flush=True)
            return EXIT_MORE
    write_manifest(out_dir, model_dir, args.K, args.classes, args.chunk_experts)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------------------------


def expected_names(arch: Qwen35MoeArch, classes: str) -> list[str]:
    want = CLASSES[classes]
    names = []
    for i in range(arch.n_layers):
        for base in arch.quantized_linears(i):
            is_proj = class_of(base[len(arch.layer_prefix(i)):]) == "p"
            if ("p" in want) if is_proj else ("e" in want):
                names.append(base + ".weight")
    return names


def summarize(tensors: dict[str, dict]) -> dict:
    by: dict[str, dict] = {}
    tot_bits = tot_numel = tot_trellis = 0
    for rec in tensors.values():
        c = by.setdefault(rec["class"], {"n": 0, "numel": 0, "bits": 0, "proxy_abs": 0.0, "K": {}})
        c["n"] += 1
        c["numel"] += rec["numel"]
        c["bits"] += rec["bits"]["total"]
        c["proxy_abs"] += rec["proxy_abs"]
        c["K"][str(rec["K"])] = c["K"].get(str(rec["K"]), 0) + 1
        tot_bits += rec["bits"]["total"]
        tot_trellis += rec["bits"]["trellis"]
        tot_numel += rec["numel"]
    for c in by.values():
        c["bpw"] = c["bits"] / c["numel"]
    return {"n_tensors": len(tensors), "numel": tot_numel, "bits": tot_bits, "bpw": tot_bits / tot_numel if tot_numel else None,
            "bpw_trellis_only": tot_trellis / tot_numel if tot_numel else None, "decode_gib": tot_bits / 8 / 2**30,
            "by_class": dict(sorted(by.items()))}


def combined_hessian_sha(chunk_docs: list[dict]) -> str:
    """sha256 of the per-layer hessian.json hashes (the Hessian dirs are deleted once their oracles are verified)."""
    per_layer = {}
    for d in chunk_docs:
        per_layer[d["layer"]] = d["hessian_json_sha256"]
    return hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(per_layer.items())).encode()).hexdigest()


def write_manifest(out_dir: Path, model_dir: Path, K: float, classes: str, chunk_experts: int) -> dict:
    arch = Qwen35MoeArch.from_config(model_dir / "config.json")
    job = make_job(model_dir, K)
    tensors: dict[str, dict] = {}
    docs, stale, done_layers = [], [], []
    per_layer_chunks: dict[int, set] = {}
    for p in sorted(out_dir.glob("L*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if (not jobs_match(doc.get("job"), job) or doc.get("chunk_experts") != chunk_experts
                or not (out_dir / p.with_suffix(".safetensors").name).exists()):
            stale.append(p.name)
            continue
        docs.append(doc)
        tensors.update(doc["tensors"])
        per_layer_chunks.setdefault(doc["layer"], set()).add(doc["chunk"])
    want = chunk_ids(arch, classes, chunk_experts)
    done_layers = sorted(l for l, cs in per_layer_chunks.items() if set(want) <= cs)
    expected = expected_names(arch, classes)
    missing = sorted(n for n in expected if n not in tensors)
    extra = sorted(n for n in tensors if n not in set(expected))
    man = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
           "complete": not missing, "missing": missing[:32], "missing_count": len(missing), "bpw_target": None,
           "K_uniform": float(K), "layers_done": done_layers, "stale_layers": stale, "classes": classes,
           "unexpected_tensors": extra[:8], **{k: job[k] for k in ("model_dir", "config_sha256", "hessian_basis", "codebook")},
           "hessian_manifest_sha256": combined_hessian_sha(docs) if docs else None, "recipe": job["recipe"],
           "code_sha256": job["code_sha256"], "summary": summarize(tensors), "tensors": tensors,
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "provenance": tq.provenance()}
    tq.write_json_atomic(out_dir / tq.MANIFEST_NAME, man)
    return man


def cmd_manifest(args) -> int:
    man = write_manifest(Path(args.out_dir), Path(args.model_dir), args.K, args.classes, args.chunk_experts)
    s = man["summary"]
    print(f"[oracle_q35] {args.out_dir}: {s['n_tensors']} tensors, {len(man['layers_done'])} layers done, complete={man['complete']}, "
          f"{s['decode_gib']:.3f} GiB, bpw {s['bpw']}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------------------


def verify_dir(out_dir: Path, model_dir: Path, layer: int, K: float, classes: str, chunk_experts: int, sample: int,
               device) -> list[str]:
    """Every chunk of the layer: file sha256 = the records', all three arrays of every tensor present with the right shapes,
    and `sample` tensors reconstructed from the stored bits only: relative weight error equal to the record's."""
    from safetensors import safe_open

    arch = Qwen35MoeArch.from_config(model_dir / "config.json")
    job = make_job(model_dir, K)
    problems = []
    reader = CheckpointReader(model_dir)
    cb_cache: dict = {}
    for c in chunk_ids(arch, classes, chunk_experts):
        st, js = chunk_paths(out_dir, layer, c)
        if not (st.exists() and js.exists()):
            problems.append(f"{out_dir.name}: L{layer:02d}.{c} missing")
            continue
        doc = json.loads(js.read_text(encoding="utf-8"))
        if not jobs_match(doc["job"], job) or doc.get("chunk_experts") != chunk_experts:
            problems.append(f"{js.name}: job or chunk size differs")
        sha = sha256_path(st)
        names = sorted(doc["tensors"])
        for n in names:
            if doc["tensors"][n]["file_sha256"] != sha:
                problems.append(f"{js.name}: {n} file_sha256 differs from the file")
                break
        with safe_open(str(st), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for n in names:
                r = doc["tensors"][n]
                for suf, shape in ((".trellis", r["words_shape"]), (".suh", [r["k"]]), (".svh", [r["n"]])):
                    if n + suf not in keys or list(f.get_slice(n + suf).get_shape()) != shape:
                        problems.append(f"{st.name}: {n}{suf} missing or wrong shape")
            step = max(1, len(names) // max(sample, 1))
            for n in names[::step][:sample]:
                r = doc["tensors"][n]
                words, suh, svh = f.get_tensor(n + ".trellis"), f.get_tensor(n + ".suh"), f.get_tensor(n + ".svh")
                cb = cb_cache.setdefault("cb", torch.from_numpy(tq.codebook_np(r["codebook"]).copy()))
                what = tq.reconstruct(words, suh, svh, r["K"], cb)  # [k, n]
                w = reader.linear_weight(n[:-len(".weight")], torch.device("cpu")).float().T
                rel, _ = tq.rel_cos(w, what)
                if abs(rel - r["rel_weight_err"]) > 1e-4 * max(r["rel_weight_err"], 1e-3):
                    problems.append(f"{n}: reconstructed rel err {rel} != recorded {r['rel_weight_err']}")
    return problems


def cmd_verify(args) -> int:
    model_dir = Path(args.model_dir)
    arch = Qwen35MoeArch.from_config(model_dir / "config.json")
    problems = []
    for spec in args.out_dir:
        d, _, k = spec.partition("@")  # <dir>@<K>
        K = float(k) if k else float(json.loads((Path(d) / tq.MANIFEST_NAME).read_text(encoding="utf-8"))["K_uniform"])
        problems += verify_dir(Path(d), model_dir, args.layer, K, args.classes, args.chunk_experts, args.sample, torch.device("cpu"))
    print(f"[oracle_q35] verify L{args.layer:02d} {args.out_dir}: {len(problems)} problem(s)")
    for p in problems[:20]:
        print("  " + p)
    if problems:
        return 1
    if args.delete_hessians:
        hdir = Path(args.hessian_root) / "hessian" / f"L{args.layer:02d}"
        gone = 0
        for f in [*hdir.glob("*.hess"), *hdir.glob("*.probe.safetensors")]:
            f.unlink()
            gone += 1
        # the small records stay: hessian.json, routing_counts.json, capture.json, DONE.json
        print(f"[oracle_q35] L{args.layer:02d}: {gone} Hessian / probe files deleted from {hdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model-dir", default=str(MODEL_DIR))
        p.add_argument("--classes", default="all", choices=sorted(CLASSES))
        p.add_argument("--chunk-experts", type=int, default=32)

    p = sub.add_parser("quantize", help="one layer at one rate (exit 75 = budget reached, run again)")
    common(p)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--K", type=float, required=True, choices=[3.0, 4.0, 5.0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hessian-root", default=str(Q4_ROOT))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--backend", default="auto", choices=["auto", "hip", "cpu", "torch"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--grid", type=int, default=0)
    p.add_argument("--budget-seconds", type=float, default=840.0, help="0 = no limit")
    p.add_argument("--cohort", type=int, default=32,
                   help="worker threads per class cohort inside a chunk whose encoder calls are batched (CohortEncoder, "
                        "bit-identical to 0 = off, ~18%% faster, ~8 GiB VRAM instead of 2)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_quantize)

    p = sub.add_parser("manifest", help="merge the usable chunks of a directory into weights_override.json")
    common(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--K", type=float, required=True)
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("verify", help="check a layer's chunks in one or more oracle dirs; optionally delete its Hessians")
    common(p)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out-dir", action="append", required=True, help="<dir> or <dir>@<K>; repeat")
    p.add_argument("--sample", type=int, default=2, help="tensors per chunk reconstructed from the stored bits")
    p.add_argument("--delete-hessians", action="store_true")
    p.add_argument("--hessian-root", default=str(Q4_ROOT))
    p.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
