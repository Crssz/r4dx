"""Layer-major, resumable runs of the qwen4_exp streaming reference on the REAL checkpoint (plan M0-5 full depth, M0-7).

`ref.Qwen4ExpStreamingReference.run_layers` runs one sequence through all 48 layers and dequantizes the whole
checkpoint (58 GiB of experts) for every sequence, in one process. For the real model this module runs a SET of
sequences ("jobset") LAYER-MAJOR instead: layer i is built and its weights are dequantized once, every sequence of
the set goes through it, the residuals are checkpointed to disk, then layer i + 1. The streaming reference's own
modules do the math (`ref._build_layer`, HF's decoder layer, `StreamingExperts`, `qsa_indexer`), so a sequence
gets exactly the bits `ref.run_layers` would give it (test_real_ref.py checks this on the tiny model, on CPU).

    prepare   <canon|probes|goldens|kvcalib>   CPU: build the jobset json (ids) under <root>\\jobsets
    step      --jobset NAME                    one bounded GPU step (take the gpu lock around it: real_ref_run.ps1)
    manifest  PATH...                          sizes + sha256 of the deliverables
    ngram-check                                random-access reader vs the np.memmap of the real sidecar

Jobs (`kind`): `logprobs` (teacher-forced log-softmax rows -> <out_dir>\\<segment>.logprobs.f16 + meta.json, the
`kl_report.py` format), `golden` (per-layer activations named like tiny_golden.py -> layer_goldens), `calib` (K/V
amax of the QSA layers -> kv_calib.json). A step returns exit code 0 when the jobset is finished and 3 when more
steps are needed (it stops after `--max-minutes` or `--max-layers` layers, always at a layer boundary).

State (resumable, per job, under <root>\\work\\<jobset>\\<job>\\): `state.json` {layers_done, residual file, its
sha256, ids sha256, rule, ...} and `resid_L<k>.bf16` (the residual [1, T, 10240] bf16 after layer k-1, raw
little-endian bytes). A restart verifies the sha256 of the residual and the ids before it continues; a step that
dies mid-way loses at most `--ckpt-every` layers.

N-gram rows: `UnbufferedSidecarRows` reads only the needed 320-byte rows of the 95 GiB sidecar with Windows
FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS overlapped reads (sector-aligned, adjacent rows merged, a thread
pool): nothing is mapped, nothing enters the OS file cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from qwen4exp import qsa_indexer  # noqa: E402
from qwen4exp.arch import QSA  # noqa: E402
from qwen4exp.common_qwen4exp import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_SIDECAR,
    KL_CORPUS_DIR,
    MODELS_ROOT,
    resolve_device,
    sha256_file,
)
from qwen4exp.ngram_rows import SidecarNGramRows  # noqa: E402
from qwen4exp.ngram_sidecar import HEADER_BYTES  # noqa: E402
from qwen4exp.ref import (  # noqa: E402
    DEFAULT_ATTN_QUERY_CHUNK,
    LOGPROB_CLAMP,
    Qwen4ExpStreamingReference,
    token_ids_sha256,
)

DEFAULT_ROOT = Path(os.environ.get("R4DX_Q4EXP_REF_ROOT", str(MODELS_ROOT / "r4dx" / "q4exp" / "ref")))
TOKENIZER_DIR = Path(os.environ.get("R4DX_CANON_TOKENIZER_DIR", str(MODELS_ROOT / "Huihui-Qwen3.8-27B-abliterated")))
EXIT_DONE, EXIT_MORE = 0, 3
STATE_VERSION = 1
FP8_E4M3_MAX = 448.0
TAIL_PERCENTILE = 99.99
TOP_BUFFER = 8192
GOLDEN_LAYERS = tuple(range(8)) + (47,)
GOLDEN_TIE_RULES = ("hf", "recent")


# --------------------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------------------


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_path(path: Path, bufsize: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def peak_ram_gib() -> float | None:
    """Peak working set of this process (Windows PROCESS_MEMORY_COUNTERS), GiB; None elsewhere."""
    if os.name != "nt":
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        except Exception:  # noqa: BLE001
            return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    k32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return pmc.PeakWorkingSetSize / 2**30


# --------------------------------------------------------------------------------------------------------
# n-gram rows by random-access, unbuffered reads
# --------------------------------------------------------------------------------------------------------

SECTOR = 4096
MAX_SPAN = 1 << 20


class _WinRandomReader:
    """Thread-safe `pread` on Windows: FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED."""

    def __init__(self, path: Path):
        import ctypes
        from ctypes import wintypes

        self.ct, self.wt = ctypes, wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32 = k32

        class OVERLAPPED(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", wintypes.DWORD),
                        ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

        self.OVERLAPPED = OVERLAPPED
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.HANDLE]
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                 ctypes.POINTER(OVERLAPPED)]
        k32.ReadFile.restype = wintypes.BOOL
        k32.GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.POINTER(OVERLAPPED), ctypes.POINTER(wintypes.DWORD),
                                            wintypes.BOOL]
        k32.GetOverlappedResult.restype = wintypes.BOOL
        k32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateEventW.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 1, 3
        flags = 0x20000000 | 0x10000000 | 0x40000000  # NO_BUFFERING | RANDOM_ACCESS | OVERLAPPED
        h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ | 2, None, OPEN_EXISTING, flags, None)
        if h is None or h == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), f"CreateFileW({path}) failed")
        self.h = h
        self.tls = threading.local()

    def _event(self):
        ev = getattr(self.tls, "ev", None)
        if ev is None:
            ev = self.k32.CreateEventW(None, True, False, None)
            self.tls.ev = ev
        return ev

    def read_into(self, offset: int, buf: np.ndarray) -> int:
        """Read len(buf) bytes at `offset` (sector aligned, buf data pointer sector aligned) into buf; returns bytes read."""
        ct, wt = self.ct, self.wt
        ov = self.OVERLAPPED()
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = offset >> 32
        ov.hEvent = self._event()
        got = wt.DWORD(0)
        ok = self.k32.ReadFile(self.h, buf.ctypes.data, buf.nbytes, None, ct.byref(ov))
        if not ok:
            err = ct.get_last_error()
            if err == 38:  # ERROR_HANDLE_EOF
                return 0
            if err != 997:  # ERROR_IO_PENDING
                raise OSError(err, f"ReadFile at {offset} failed")
        if not self.k32.GetOverlappedResult(self.h, ct.byref(ov), ct.byref(got), True):
            err = ct.get_last_error()
            if err == 38:
                return 0
            raise OSError(err, f"GetOverlappedResult at {offset} failed")
        return int(got.value)

    def close(self):
        if self.h:
            self.k32.CloseHandle(self.h)
            self.h = None


class _PlainRandomReader:
    """Portable fallback (tests on non-Windows): positional reads without mapping."""

    def __init__(self, path: Path):
        self.fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))

    def read_into(self, offset: int, buf: np.ndarray) -> int:
        data = os.pread(self.fd, buf.nbytes, offset)
        buf[: len(data)] = np.frombuffer(data, dtype=np.uint8)
        return len(data)

    def close(self):
        os.close(self.fd)


def aligned_buffer(nbytes: int) -> np.ndarray:
    raw = np.empty(nbytes + SECTOR, dtype=np.uint8)
    off = (-raw.ctypes.data) % SECTOR
    return raw[off: off + nbytes]


class UnbufferedSidecarRows(SidecarNGramRows):
    """`SidecarNGramRows` (same header check, same `rows` contract and bytes) without the memmap."""

    def __init__(self, path: str | Path, ngram_hash, threads: int = 16):
        super().__init__(path, ngram_hash)
        self.mm = None  # the base class memmap is never touched; drop the mapping
        self.row_bytes = self.header.row_bytes
        self.reader = (_WinRandomReader if os.name == "nt" else _PlainRandomReader)(self.path)
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.tls = threading.local()
        self.bytes_read = 0
        self.reads = 0

    def _buf(self) -> np.ndarray:
        b = getattr(self.tls, "buf", None)
        if b is None:
            b = aligned_buffer(MAX_SPAN + 2 * SECTOR)
            self.tls.buf = b
        return b

    def _spans(self, offs: np.ndarray):
        """[(start, end, first, last_exclusive)] sector-aligned spans covering rows at byte offsets `offs` (sorted)."""
        rb = self.row_bytes
        s = (offs // SECTOR) * SECTOR
        e = ((offs + rb + SECTOR - 1) // SECTOR) * SECTOR
        spans = []
        first = 0
        cur_s, cur_e = int(s[0]), int(e[0])
        for k in range(1, len(offs)):
            sk, ek = int(s[k]), int(e[k])
            if sk <= cur_e and ek - cur_s <= MAX_SPAN:
                cur_e = max(cur_e, ek)
            else:
                spans.append((cur_s, cur_e, first, k))
                first, cur_s, cur_e = k, sk, ek
        spans.append((cur_s, cur_e, first, len(offs)))
        return spans

    def rows(self, ids: np.ndarray) -> torch.Tensor:
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if flat.size == 0:
            return torch.empty((*ids.shape, self.head_dim), dtype=torch.bfloat16)
        if flat.min() < 0 or flat.max() >= self.header.rows:
            raise IndexError(f"n-gram row id outside [0, {self.header.rows})")
        uniq, inverse = np.unique(flat, return_inverse=True)
        offs = HEADER_BYTES + uniq * self.row_bytes
        out = np.empty((uniq.size, self.head_dim), dtype=np.uint16)
        rb = self.row_bytes

        def do(span):
            start, end, a, b = span
            buf = self._buf()[: end - start]
            got = self.reader.read_into(start, buf)
            need = int(offs[b - 1]) + rb - start
            if got < need:
                raise EOFError(f"{self.path}: short read at {start}: {got} < {need} bytes")
            for k in range(a, b):
                o = int(offs[k]) - start
                out[k] = buf[o: o + rb].view(np.uint16)
            return got

        total = 0
        for got in self.pool.map(do, self._spans(offs)):
            total += got
            self.reads += 1
        self.bytes_read += total
        data = out[inverse.reshape(-1)]
        return torch.from_numpy(data).view(torch.bfloat16).reshape(*ids.shape, self.head_dim)

    def describe(self) -> str:
        return (f"sidecar {self.path} ({self.header.rows} rows, sha256 {self.header.body_sha256.hex()[:12]}), "
                f"unbuffered random-access reads")


# --------------------------------------------------------------------------------------------------------
# dequantized weight cache (one layer at a time)
# --------------------------------------------------------------------------------------------------------


class WeightCache:
    """Memoizes `reader.linear_weight` for the layer being run, so every sequence of a jobset reuses the dequantized
    experts. Cleared before the next layer; bounded by `max_bytes` (entries past it are simply not kept)."""

    def __init__(self, reader, max_bytes: int):
        self.reader = reader
        self.orig = reader.linear_weight
        self.max_bytes = max_bytes
        self.d: dict = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        reader.linear_weight = self

    def __call__(self, base, device):
        key = (base, str(device))
        w = self.d.get(key)
        if w is not None:
            self.hits += 1
            return w
        self.misses += 1
        w = self.orig(base, device)
        nb = w.numel() * w.element_size()
        if self.bytes + nb <= self.max_bytes:
            self.d[key] = w
            self.bytes += nb
        return w

    def clear(self):
        self.d.clear()
        self.bytes = 0

    def close(self):
        self.clear()
        self.reader.linear_weight = self.orig


# --------------------------------------------------------------------------------------------------------
# golden capture (tensor names of tiny_golden.py)
# --------------------------------------------------------------------------------------------------------


def _golden_capture_cls():
    from qwen4exp import tiny_golden

    class LayerGolden(tiny_golden.Capture):
        """tiny_golden.Capture's hooks, registered on ONE decoder layer of the streaming reference."""

        def __init__(self, layer, i: int):
            self.t = {}
            self.handles = []
            p = f"layer{i}."
            self.handles.append(layer.register_forward_pre_hook(lambda m, a, k, p=p: self._put(p + "residual_in", a[0]), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(lambda m, a, o, p=p: self._put(p + "residual_out", o)))
            ah, mh = layer.attn_hyper_connection, layer.mlp_hyper_connection
            self.handles.append(ah.register_forward_hook(lambda m, a, o, p=p: self._hc(p + "attn_", o)))
            self.handles.append(mh.register_forward_hook(lambda m, a, o, p=p: self._hc(p + "mlp_", o)))
            self.handles.append(mh.register_forward_pre_hook(lambda m, a, p=p: self._put(p + "residual_attn", a[0])))
            blk = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            self.handles.append(blk.register_forward_hook(lambda m, a, o, p=p: self._put(p + "attn_out", o if torch.is_tensor(o) else o[0])))
            moe = layer.mlp
            self.handles.append(moe.register_forward_hook(lambda m, a, o, p=p: self._put(p + "moe_out", o)))
            self.handles.append(moe.gate.register_forward_hook(lambda m, a, o, p=p: self._router(p, o)))
            self.handles.append(moe.shared_expert.register_forward_hook(lambda m, a, o, p=p: self._put(p + "shared_out", o, False)))
            self.handles.append(moe.shared_expert_gate.register_forward_hook(lambda m, a, o, p=p: self._put(p + "shared_gate_logit", o, False)))
            if layer.ple is not None:
                self.handles.append(layer.ple.register_forward_hook(lambda m, a, o, p=p: self._put(p + "ple_out", o)))
                self.handles.append(layer.ple.ple_embedding.register_forward_hook(lambda m, a, o, p=p: self._put(p + "ple_embedding", o)))
            if layer.layer_type != "linear_attention":
                self.handles.append(layer.self_attn.indexer.register_forward_hook(lambda m, a, o, p=p: self._mask(p, o)))
                self.handles.append(layer.self_attn.register_forward_pre_hook(lambda m, a, k, p=p: self._qsa_inputs(p, m, a, k), with_kwargs=True))
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, p=p: self._put(p + "attn_pre_o", a[0])))

        def _qsa_inputs(self, prefix, attn, args, kwargs):
            """tiny_golden.Capture._qsa_inputs with the block start indices on the activations' device."""
            from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

            hidden, pos = args[0], args[1]
            cos_full, sin_full = pos
            seq = hidden.shape[1]
            ix = attn.indexer
            q, raw = qsa_indexer.project(ix, hidden, pos)
            self._put(prefix + "index_q", q)
            self._put(prefix + "index_raw_keys", raw)
            ratio = ix.compress_ratio
            nb = seq // ratio
            if nb > 0:
                starts = torch.arange(nb, device=hidden.device) * ratio
                bk = qsa_indexer._pooled_block_keys(raw[0, : nb * ratio], starts, cos_full[0], sin_full[0],
                                                    ix.k_layernorm.weight, ix.k_layernorm.eps, ratio)
            else:
                bk = raw.new_zeros((0, raw.shape[-1]))
            self.t[prefix + "index_block_keys"] = bk.clone()
            cos, sin = (x[:, -seq:, :] for x in pos)
            shape = (*hidden.shape[:-1], -1, attn.head_dim)
            qg = attn.q_proj(hidden).view(*hidden.shape[:-1], -1, attn.head_dim * 2)
            qs, gate = torch.chunk(qg, 2, dim=-1)
            self._put(prefix + "attn_gate", gate.reshape(*hidden.shape[:-1], -1))
            qs = attn.q_norm(qs.view(shape)).transpose(1, 2)
            ks = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
            vs = attn.v_proj(hidden).view(shape).transpose(1, 2)
            qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
            self._put(prefix + "attn_q", qs.transpose(1, 2))
            self._put(prefix + "attn_k", ks.transpose(1, 2))
            self._put(prefix + "attn_v", vs.transpose(1, 2))

    return LayerGolden


# --------------------------------------------------------------------------------------------------------
# K/V amax statistics (port of kv_calibrate_full.HeadStats)
# --------------------------------------------------------------------------------------------------------


class HeadStats:
    """Running per-head amax and the TOP_BUFFER largest |values| (exact tail percentile) over [H, T, D] blocks."""

    def __init__(self, num_heads: int, top_buffer: int = TOP_BUFFER):
        self.num_heads = num_heads
        self.top_buffer = top_buffer
        self.amax = torch.zeros(num_heads, dtype=torch.float32)
        self.top = torch.zeros(num_heads, 0, dtype=torch.float32)
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        assert x.dim() == 3 and x.shape[0] == self.num_heads, f"unexpected shape {tuple(x.shape)}"
        a = x.detach().float().abs()
        self.amax = torch.maximum(self.amax, a.amax(dim=(1, 2)).cpu())
        flat = a.reshape(self.num_heads, -1)
        k = min(self.top_buffer, flat.shape[1])
        merged = torch.cat([self.top, flat.topk(k, dim=1).values.cpu()], dim=1)
        self.top = merged.topk(min(self.top_buffer, merged.shape[1]), dim=1).values
        self.count += flat.shape[1]

    def percentile(self, p: float):
        if self.count == 0:
            return None, False
        rank_from_top = self.count - math.ceil((p / 100.0) * self.count)
        if rank_from_top >= self.top.shape[1]:
            return None, False
        return self.top[:, rank_from_top].tolist(), True

    def dump(self) -> dict:
        return {"amax": self.amax, "top": self.top, "count": self.count}

    @classmethod
    def load(cls, d: dict) -> "HeadStats":
        s = cls(d["amax"].shape[0])
        s.amax, s.top, s.count = d["amax"], d["top"], d["count"]
        return s


# --------------------------------------------------------------------------------------------------------
# jobsets
# --------------------------------------------------------------------------------------------------------


def jobset_path(root: Path, name: str) -> Path:
    return root / "jobsets" / f"{name}.json"


def load_jobset(root: Path, name: str) -> dict:
    p = jobset_path(root, name)
    if not p.exists():
        raise SystemExit(f"[real_ref] {p} does not exist: run `prepare` first")
    return json.loads(p.read_text(encoding="utf-8"))


def work_dir(root: Path, js: dict, job: dict | None = None) -> Path:
    d = root / "work" / js["name"]
    return d / job["name"] if job is not None else d


def new_state(js: dict, job: dict) -> dict:
    return {"version": STATE_VERSION, "job": job["name"], "ids_sha256": token_ids_sha256(job["ids"]), "T": len(job["ids"]),
            "rule": job["rule"], "layers_done": 0, "resid_file": None, "resid_sha256": None, "finalized": False,
            "layer_seconds": {}, "created": now_iso()}


def load_state(root: Path, js: dict, job: dict) -> dict:
    p = work_dir(root, js, job) / "state.json"
    if p.exists():
        st = json.loads(p.read_text(encoding="utf-8"))
        fresh = new_state(js, job)
        if (st.get("version"), st.get("ids_sha256"), st.get("rule")) != (fresh["version"], fresh["ids_sha256"], fresh["rule"]):
            raise SystemExit(f"[real_ref] {p} belongs to different ids/rule than the jobset; remove the work dir to restart")
        return st
    return new_state(js, job)


def save_state(root: Path, js: dict, job: dict, st: dict) -> None:
    atomic_write_json(work_dir(root, js, job) / "state.json", st)


def save_residual(root, js, job, st, hidden: torch.Tensor, layers_done: int) -> None:
    """Write the residual after `layers_done` layers, then commit state, then drop the previous residual."""
    d = work_dir(root, js, job)
    d.mkdir(parents=True, exist_ok=True)
    cpu = hidden.detach().contiguous().cpu()
    arr = cpu.view(torch.int16).numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()
    name = f"resid_L{layers_done}.bf16"
    tmp = d / (name + ".tmp")
    with open(tmp, "wb") as f:
        arr.tofile(f)
    sha = hashlib.sha256(arr.data).hexdigest()
    os.replace(tmp, d / name)
    old = st.get("resid_file")
    st.update({"layers_done": layers_done, "resid_file": name, "resid_sha256": sha, "resid_shape": list(hidden.shape),
               "resid_dtype": str(hidden.dtype).replace("torch.", ""),
               "updated": now_iso()})
    save_state(root, js, job, st)
    if old and old != name and (d / old).exists():
        (d / old).unlink()


def load_residual(root, js, job, st, device) -> torch.Tensor:
    d = work_dir(root, js, job)
    p = d / st["resid_file"]
    shape = tuple(st["resid_shape"])
    fp32 = st.get("resid_dtype", "bfloat16") == "float32"
    esz = 4 if fp32 else 2
    if p.stat().st_size != int(np.prod(shape)) * esz:
        raise SystemExit(f"[real_ref] {p} has {p.stat().st_size} bytes, expected {int(np.prod(shape)) * esz}")
    sha = sha256_path(p)
    if sha != st["resid_sha256"]:
        raise SystemExit(f"[real_ref] {p} sha256 {sha} != state {st['resid_sha256']}: corrupt checkpoint, remove the work dir")
    if fp32:
        return torch.from_numpy(np.fromfile(p, dtype=np.float32).reshape(shape)).to(device)
    arr = np.fromfile(p, dtype=np.int16).reshape(shape)
    return torch.from_numpy(arr).view(torch.bfloat16).to(device)


# --------------------------------------------------------------------------------------------------------
# the layer-major engine
# --------------------------------------------------------------------------------------------------------


class JobRT:
    """A job's in-step state: residual on the device and the per-sequence masks / rotary tables."""

    def __init__(self, spec: dict, st: dict, hidden: torch.Tensor):
        self.spec, self.st, self.hidden = spec, st, hidden
        self.ctx = None
        self.golden: dict[int, dict] = {}

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def layers_done(self) -> int:
        return self.st["layers_done"]


def make_ctx(ref: Qwen4ExpStreamingReference, ids: list[int], like: torch.Tensor) -> dict:
    """`position_embeddings`, masks and PLE ids exactly as `Qwen4ExpStreamingReference.run_layers` builds them."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import create_causal_mask, create_recurrent_attention_mask

    seq = len(ids)
    embeds = torch.empty((1, seq, ref.arch.hidden), dtype=like.dtype, device=like.device)  # shape/dtype holder
    position_ids = torch.arange(seq, device=ref.device).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids, position_ids = position_ids[0], position_ids[1:]
    mask_kwargs = {"config": ref.text_config, "inputs_embeds": embeds, "attention_mask": None, "past_key_values": None,
                   "position_ids": text_position_ids, "allow_is_causal_skip": False}
    attention_mask = create_causal_mask(**mask_kwargs) if seq <= ref.attn_query_chunk else None
    conv_mask = create_recurrent_attention_mask(**mask_kwargs)
    ple_ids = torch.tensor([[int(t) for t in ids]], dtype=torch.long).to(ref.device)
    if conv_mask is not None:
        ple_ids = torch.where(conv_mask.bool(), ple_ids, ref.arch.eos_token_id)
    return {"pe": ref._rotary_emb()(embeds, position_ids), "am": attention_mask, "cm": conv_mask, "ple_ids": ple_ids}


def make_ref(js: dict, device, *, tie_rule: str = "recent", verbose: bool = True) -> Qwen4ExpStreamingReference:
    sidecar = js.get("ngram_sidecar")
    dtype = getattr(torch, js.get("dtype", "bfloat16"))
    if dtype == torch.float32:
        torch.set_float32_matmul_precision("highest")
    ref = Qwen4ExpStreamingReference(js["model_dir"], device, tie_rule=tie_rule,
                                     ngram_sidecar=sidecar, attn=js.get("attn", "sdpa"),
                                     attn_query_chunk=int(js.get("attn_query_chunk", DEFAULT_ATTN_QUERY_CHUNK)), verbose=verbose,
                                     dtype=dtype, fp32_exact_weights=bool(js.get("fp32_exact_weights", True)))
    if sidecar and ref.arch.ple_layers():
        ref.ngram = UnbufferedSidecarRows(sidecar, ref.ngram_hash)
    return ref


def build_layer(ref: Qwen4ExpStreamingReference, i: int):
    """`ref._build_layer` with the QSA indexer's tie rule read from `ref.tie_rule` at call time (one layer instance
    serves jobs of both rules)."""
    layer = ref._build_layer(i)
    if ref.arch.layer_types[i] == QSA:
        ix = layer.self_attn.indexer
        ix.forward = lambda h, pe, m, pkv=None: qsa_indexer.indexer_forward(ix, h, pe, m, pkv, tie_rule=ref.tie_rule)
    return layer


class KvTap:
    """K (post k_norm, post rope) and V (v_proj output) amax statistics of one QSA layer, all jobs of the layer."""

    def __init__(self, ref, layer, i: int):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as mod

        self.mod = mod
        self.kv_heads = ref.arch.kv_heads
        self.head_dim = ref.arch.head_dim
        self.k, self.v = HeadStats(self.kv_heads), HeadStats(self.kv_heads)
        self.tokens = 0
        self.orig = mod.apply_rotary_pos_emb
        orig = self.orig

        def patched(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
            out = orig(q, k, cos, sin, unsqueeze_dim)
            if k is not None:
                k_embed = out[1]
                assert k_embed.dim() == 4 and k_embed.shape[0] == 1 and k_embed.shape[1] == self.kv_heads, tuple(k_embed.shape)
                self.k.update(k_embed[0])
                self.tokens += k_embed.shape[2]
            return out

        mod.apply_rotary_pos_emb = patched
        self.handle = layer.self_attn.v_proj.register_forward_hook(self._v_hook)

    def _v_hook(self, _m, _inputs, output):
        assert output.dim() == 3 and output.shape[0] == 1 and output.shape[2] == self.kv_heads * self.head_dim
        t = output.shape[1]
        self.v.update(output.detach()[0].view(t, self.kv_heads, self.head_dim).transpose(0, 1))

    def close(self):
        self.mod.apply_rotary_pos_emb = self.orig
        self.handle.remove()


def finalize_logprobs(ref: Qwen4ExpStreamingReference, js: dict, spec: dict, st: dict, hidden: torch.Tensor) -> dict:
    ids = spec["ids"]
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(ids)
    mode, chunk, row_block = js.get("logits_mode", "fp32"), int(js.get("lm_head_chunk", 32768)), int(js.get("row_block", 128))
    bin_path = out_dir / f"{spec['segment']}.logprobs.f16"
    part = bin_path.with_name(bin_path.name + ".partial")
    if ref.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    mixed = ref.mix(hidden)[0]
    nll_sum, worst = 0.0, 0.0
    with open(part, "wb") as f:
        for s in range(0, T - 1, row_block):
            e = min(s + row_block, T - 1)
            lg = ref.logits_rows(mixed[s:e], mode, chunk)
            lp = torch.log_softmax(lg, dim=-1).clamp_min(LOGPROB_CLAMP).to(torch.float16)
            worst = max(worst, torch.logsumexp(lp.float(), dim=-1).abs().max().item())
            tgt = torch.as_tensor([int(t) for t in ids[s + 1:e + 1]], device=lp.device)
            nll_sum -= lp[torch.arange(e - s, device=lp.device), tgt].float().sum().item()
            f.write(np.ascontiguousarray(lp.cpu().numpy()).tobytes())
            del lg, lp
    last = ref.logits_rows(mixed[-1:], mode, chunk)[0]
    top = torch.topk(last, min(5, last.numel()))
    probs = torch.softmax(last, dim=-1)[top.indices].tolist()
    if ref.device.type == "cuda":
        torch.cuda.synchronize()
    expect = (T - 1) * ref.vocab_size * 2
    if part.stat().st_size != expect:
        raise RuntimeError(f"{part} is {part.stat().st_size} bytes, expected {expect}")
    os.replace(part, bin_path)
    nll = nll_sum / (T - 1)
    seconds_layers = sum(st["layer_seconds"].values())
    tokens_file = js.get("tokens_file")
    meta = {
        "T": T, "V": ref.vocab_size, "dtype": "float16", "rows": T - 1, "source": ref.source_tag,
        "torch_dtype": str(ref.dtype).replace("torch.", ""), "fp32_exact_weights": ref.fp32_exact_weights if ref.dtype == torch.float32 else None,
        "sha256_of_token_ids_json": token_ids_sha256(ids), "segment": spec["segment"], "model_dir": str(ref.model_dir),
        "impl": f"qwen4exp-streaming-layermajor-{ref.attn}-{spec['rule']}", "logits_mode": mode, "attn": ref.attn,
        "attn_query_chunk": ref.attn_query_chunk, "lm_head_chunk": chunk, "row_block": row_block, "logprob_clamp": LOGPROB_CLAMP,
        "row_semantics": "row i = log_softmax(logits at position i) = log p(next | token_ids[0..i]), fp32 then fp16",
        "max_abs_logsumexp_fp16": worst, "mean_nll_next_token": nll, "perplexity": math.exp(nll),
        "last_position_top_k": [{"id": int(i), "logit": float(v), "prob": p}
                                for v, i, p in zip(top.values.tolist(), top.indices.tolist(), probs)],
        "seconds_stack": seconds_layers, "seconds_lm_head": time.perf_counter() - t1,
        "tokens_file": tokens_file, "tokens_file_sha256": js.get("tokens_file_sha256"),
        "generated_at": now_iso(), "tie_rule": spec["rule"], "n_layers": ref.n_layers, "debug_max_layers": None,
        "ngram": ref.ngram.describe() if ref.ngram else None, "jobset": js["name"], "layers_seconds": st["layer_seconds"],
        "weights": weights_label(ref),
    }
    atomic_write_json(out_dir / f"{spec['segment']}.meta.json", meta)
    write_dir_records(js, out_dir, spec["rule"])
    return meta


def weights_label(ref) -> str:
    if not ref.packed:
        return "bf16 checkpoint" if ref.dtype == torch.bfloat16 else "bf16 checkpoint, fp32 compute"
    if ref.dtype == torch.bfloat16:
        return "int4 AWQ dequantized to bf16"
    return ("int4 AWQ dequantized to fp32 (unrounded), fp32 compute" if ref.fp32_exact_weights
            else "int4 AWQ dequantized to bf16 then widened, fp32 compute")


def write_dir_records(js: dict, out_dir: Path, rule: str) -> None:
    """`reference_run.json` and `tokens_used.json` of a log-prob directory, from the jobs of the jobset that write there."""
    segs = [j for j in js["jobs"] if Path(j["out_dir"]) == out_dir and j["kind"] == "logprobs"]
    done = [j for j in segs if (out_dir / f"{j['segment']}.meta.json").exists()]
    metas = {j["segment"]: json.loads((out_dir / f"{j['segment']}.meta.json").read_text(encoding="utf-8")) for j in done}
    run = {"generated_at": now_iso(), "model_dir": js["model_dir"], "tokens_file": js.get("tokens_file"),
           "tokens_file_sha256": js.get("tokens_file_sha256"), "tie_rule": rule, "n_layers": next(iter(metas.values()))["n_layers"] if metas else None,
           "debug_max_layers": None, "attn": js.get("attn", "sdpa"), "attn_query_chunk": js.get("attn_query_chunk", DEFAULT_ATTN_QUERY_CHUNK),
           "logits_mode": js.get("logits_mode", "fp32"), "weights": next(iter(metas.values()))["weights"] if metas else None,
           "dtype": js.get("dtype", "bfloat16"), "torch_version": torch.__version__,
           "ngram_sidecar": js.get("ngram_sidecar"), "jobset": js["name"], "segments": metas}
    atomic_write_json(out_dir / "reference_run.json", run)
    atomic_write_json(out_dir / "tokens_used.json",
                      {**js.get("tokens_doc_header", {}), "segments": [{"name": j["segment"], "token_ids": j["ids"]} for j in done]})


def finalize_golden(ref: Qwen4ExpStreamingReference, js: dict, root: Path, rt: JobRT) -> None:
    from safetensors.torch import load_file, save_file

    spec, st = rt.spec, rt.st
    d = work_dir(root, js, spec)
    ids = spec["ids"]
    tensors: dict[str, torch.Tensor] = {"input_ids": torch.tensor(ids, dtype=torch.long), "embedding": ref.embed(ids).cpu()}
    for i in js["golden_layers"]:
        p = d / f"golden_L{i}.safetensors"
        tensors.update(load_file(str(p)))
    hidden = rt.hidden
    mixed = ref.mix(hidden)
    tensors["mixer_out"] = mixed[0].cpu()
    tensors["logits"] = ref.logits_rows(mixed, "hf-bf16")[0].to(torch.bfloat16).cpu()
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{spec['segment']}.{spec['rule']}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path),
              metadata={"format": "pt", "tie_rule": spec["rule"], "T": str(len(ids)), "layers": json.dumps(js["golden_layers"]),
                        "weights": "real AWQ checkpoint dequantized to bf16, streaming reference, layer-major"})
    st["golden_file"] = str(path)


def write_kv_calib(js: dict, root: Path, ref: Qwen4ExpStreamingReference) -> dict:
    out: dict = {}
    total_tokens = sum(len(j["ids"]) for j in js["jobs"])
    d = work_dir(root, js)
    qsa_layers = [i for i, t in enumerate(ref.arch.layer_types) if t == QSA]
    for i in qsa_layers:
        blob = torch.load(d / f"calib_stats_L{i}.pt", map_location="cpu")
        if blob["tokens"] != total_tokens:
            raise RuntimeError(f"layer {i}: statistics cover {blob['tokens']} tokens, the corpus has {total_tokens}")
        k, v = HeadStats.load(blob["k"]), HeadStats.load(blob["v"])
        k_tail, k_exact = k.percentile(TAIL_PERCENTILE)
        v_tail, v_exact = v.percentile(TAIL_PERCENTILE)
        out[str(i)] = {
            "layer_idx": i, "kv_heads": ref.arch.kv_heads, "head_dim": ref.arch.head_dim,
            "num_calibration_tokens": total_tokens, "k_amax": k.amax.tolist(), "v_amax": v.amax.tolist(),
            "k_p9999": k_tail, "v_p9999": v_tail, "tail_percentile": TAIL_PERCENTILE, "tail_exact": bool(k_exact and v_exact),
            "tail_elements_per_head": k.count, "method": "full-forward-streaming-qwen4exp",
            "method_detail": ("every calibration document is its own sequence from position 0 (causal, no cache) through all 48 layers of "
                              "the streaming reference (AWQ dequantized to bf16, QSA tie rule recent); K is captured after k_norm and "
                              "rope (what the paged fp8 cache stores), V at the v_proj output; amax is the max over every token of "
                              "every document; k_p9999/v_p9999 are informational"),
            "torch_dtype": "bfloat16", "weights_source": "Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ int4 dequantized to bf16",
            "model_dir": js["model_dir"], "corpus": js["corpus_meta"], "tokenizer": js["tokenizer_provenance"],
            "total_calibration_tokens": total_tokens, "torch_version": torch.__version__, "generated_at": now_iso(),
        }
    path = Path(js["out_json"])
    atomic_write_json(path, out)
    return out


def run_step(root: Path, js: dict, device: str, max_minutes: float, max_layers: int | None, ckpt_every: int,
             cache_gib: float, verbose: bool = True) -> int:
    t_start = time.perf_counter()
    started = now_iso()
    jobs = js["jobs"]
    states = {j["name"]: load_state(root, js, j) for j in jobs}
    todo = [j for j in jobs if not states[j["name"]]["finalized"]]
    status_path = work_dir(root, js) / "status.json"
    if not todo:
        atomic_write_json(status_path, {"status": "done", "updated": now_iso()})
        print(f"[real_ref] {js['name']}: nothing to do", flush=True)
        return EXIT_DONE
    dev = resolve_device(device)
    ref = make_ref(js, dev, verbose=verbose)
    n_layers = ref.n_layers
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # runtime jobs: those that still need layers, or need finalizing
    rts: dict[str, JobRT] = {}
    for j in todo:
        st = states[j["name"]]
        if st["layers_done"] == 0:
            hidden = ref.embed(j["ids"]).unsqueeze(0).repeat(1, 1, ref.arch.hc_count)
        else:
            hidden = load_residual(root, js, j, st, dev)
        rts[j["name"]] = JobRT(j, st, hidden)
    kind_calib = js["kind"] == "calib"
    if kind_calib and len({rt.layers_done for rt in rts.values()}) > 1:
        raise SystemExit("[real_ref] calib jobs are at different layers (an interrupted checkpoint): remove the work dir and restart")
    golden_layers = set(js.get("golden_layers", ())) if js["kind"] == "golden" else set()
    cache = WeightCache(ref.reader, int(cache_gib * 2**30)) if len(todo) > 1 and cache_gib > 0 else None
    deadline = t_start + max_minutes * 60.0
    layer_times = []
    stop_reason = None
    layers_run = 0
    start = min([rt.layers_done for rt in rts.values() if rt.layers_done < n_layers], default=n_layers)
    first_layer, last_layer = start, start
    GoldenCls = _golden_capture_cls() if golden_layers else None
    print(f"[real_ref] {js['name']}: {len(todo)}/{len(jobs)} jobs to go, from layer {start}/{n_layers}, "
          f"tokens {sum(len(j['ids']) for j in todo)}, budget {max_minutes} min", flush=True)

    def commit(i_done: int):
        for rt in rts.values():
            if rt.layers_done == i_done and rt.st["resid_file"] != f"resid_L{i_done}.bf16":
                save_residual(root, js, rt.spec, rt.st, rt.hidden, i_done)

    for i in range(start, n_layers):
        active = [rt for rt in rts.values() if rt.layers_done == i]
        if not active:
            continue
        t0 = time.perf_counter()
        layer = build_layer(ref, i)
        t_build = time.perf_counter() - t0
        tap = KvTap(ref, layer, i) if (kind_calib and ref.arch.layer_types[i] == QSA) else None
        try:
            for rt in active:
                if rt.ctx is None:
                    rt.ctx = make_ctx(ref, rt.spec["ids"], rt.hidden)
                ref.tie_rule = rt.spec["rule"]
                cap = GoldenCls(layer, i) if i in golden_layers else None
                t1 = time.perf_counter()
                with torch.no_grad():
                    out = layer(rt.hidden, position_embeddings=rt.ctx["pe"], attention_mask=rt.ctx["am"], conv_mask=rt.ctx["cm"],
                                past_key_values=None, ple_input_ids=rt.ctx["ple_ids"])
                if cap is not None:
                    cap.close()
                    t = cap.t
                    if layer.ple is not None:
                        t[f"layer{i}.ple_ngram_ids"] = torch.from_numpy(ref.last_ngram_ids.copy())
                        t[f"layer{i}.residual_ple"] = t[f"layer{i}.residual_in"] + t[f"layer{i}.ple_out"]
                    from safetensors.torch import save_file

                    d = work_dir(root, js, rt.spec)
                    d.mkdir(parents=True, exist_ok=True)
                    save_file({k: v.detach().cpu().contiguous() for k, v in t.items()}, str(d / f"golden_L{i}.safetensors"))
                rt.hidden = out
                rt.st["layer_seconds"][str(i)] = rt.st["layer_seconds"].get(str(i), 0.0) + (time.perf_counter() - t1)
                rt.st["layers_done"] = i + 1
        finally:
            if tap is not None:
                tap.close()
        if tap is not None:
            work_dir(root, js).mkdir(parents=True, exist_ok=True)
            torch.save({"k": tap.k.dump(), "v": tap.v.dump(), "tokens": tap.tokens}, work_dir(root, js) / f"calib_stats_L{i}.pt")
        del layer
        if cache is not None:
            cache.clear()
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        dt_layer = time.perf_counter() - t0
        layer_times.append(dt_layer)
        layers_run += 1
        last_layer = i + 1
        if verbose:
            extra = f" cache {cache.hits}h/{cache.misses}m" if cache is not None else ""
            print(f"[real_ref] layer {i} ({ref.arch.layer_types[i]}) {dt_layer:.1f}s (build {t_build:.1f}s, {len(active)} jobs){extra}",
                  flush=True)
        now = time.perf_counter()
        if max_layers is not None and layers_run >= max_layers:
            stop_reason = "max-layers"
        elif now + 1.25 * max(layer_times) > deadline and i + 1 < n_layers:
            stop_reason = "time budget"
        if stop_reason or (i + 1) % ckpt_every == 0 or i + 1 == n_layers:
            commit(i + 1)
        if stop_reason:
            break
    # finalize jobs that completed all layers (budget permitting: logprob finalization of a 32k job takes minutes)
    finalized = []
    for rt in rts.values():
        if rt.layers_done < n_layers or rt.st["finalized"]:
            continue
        if stop_reason == "time budget" or time.perf_counter() > deadline + 60:
            continue
        spec = rt.spec
        if spec["kind"] == "logprobs":
            finalize_logprobs(ref, js, spec, rt.st, rt.hidden)
            print(f"[real_ref] finalized {spec['name']}: {spec['out_dir']}\\{spec['segment']}.logprobs.f16", flush=True)
        elif spec["kind"] == "golden":
            finalize_golden(ref, js, root, rt)
            print(f"[real_ref] finalized {spec['name']}: {rt.st['golden_file']}", flush=True)
        elif spec["kind"] != "calib":
            raise RuntimeError(f"unknown job kind {spec['kind']}")
        rt.st["finalized"] = True
        rt.st["finalized_at"] = now_iso()
        d = work_dir(root, js, spec)
        old = rt.st.get("resid_file")
        rt.st["resid_file"] = None
        save_state(root, js, spec, rt.st)
        if old and (d / old).exists():
            (d / old).unlink()
        finalized.append(spec["name"])
    states_now = {j["name"]: rts[j["name"]].st if j["name"] in rts else states[j["name"]] for j in jobs}
    all_done = all(s["finalized"] for s in states_now.values())
    if all_done and js["kind"] == "calib":
        write_kv_calib(js, root, ref)
        print(f"[real_ref] wrote {js['out_json']}", flush=True)
    if cache is not None:
        cache.close()
    peak_vram = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None
    rec = {"started": started, "seconds": time.perf_counter() - t_start, "layers": [first_layer, last_layer], "layers_run": layers_run,
           "layer_seconds": layer_times, "stop_reason": stop_reason, "finalized": finalized, "peak_vram_gib": peak_vram,
           "peak_ram_gib": peak_ram_gib(), "jobs_active": len(todo), "ngram_bytes_read": getattr(ref.ngram, "bytes_read", None),
           "ngram_reads": getattr(ref.ngram, "reads", None)}
    steps = work_dir(root, js) / "steps.jsonl"
    steps.parent.mkdir(parents=True, exist_ok=True)
    with open(steps, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    status = "done" if all_done else "more"
    atomic_write_json(status_path, {"status": status, "updated": now_iso(), "last_step": rec})
    print(f"[real_ref] step {rec['seconds']:.1f}s layers [{first_layer},{last_layer}) peak VRAM "
          f"{peak_vram if peak_vram is None else round(peak_vram, 2)} GiB peak RAM {rec['peak_ram_gib']} GiB -> {status}", flush=True)
    return EXIT_DONE if all_done else EXIT_MORE


# --------------------------------------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------------------------------------


def base_jobset(root: Path, name: str, kind: str, model_dir: Path, sidecar: Path | None, **kw) -> dict:
    return {"format": "r4dx-real-ref-jobset", "version": 1, "name": name, "kind": kind, "model_dir": str(model_dir),
            "config_sha256": sha256_file(model_dir / "config.json"), "ngram_sidecar": str(sidecar) if sidecar else None,
            "attn": "sdpa", "attn_query_chunk": DEFAULT_ATTN_QUERY_CHUNK, "logits_mode": "fp32", "lm_head_chunk": 32768,
            "row_block": 128, "created": now_iso(), "jobs": [], **kw}


def write_jobset(root: Path, js: dict) -> Path:
    p = jobset_path(root, js["name"])
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if [(j["name"], token_ids_sha256(j["ids"]), j["rule"]) for j in old["jobs"]] != \
                [(j["name"], token_ids_sha256(j["ids"]), j["rule"]) for j in js["jobs"]]:
            raise SystemExit(f"[real_ref] {p} exists with different jobs; delete it (and its work dir) to rebuild")
        return p
    atomic_write_json(p, js)
    return p


def load_canon_doc() -> tuple[dict, Path]:
    path = KL_CORPUS_DIR / "tokens_canon.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def prepare_canon(root: Path, model_dir: Path, sidecar: Path | None) -> Path:
    doc, path = load_canon_doc()
    js = base_jobset(root, "canon", "logprobs", model_dir, sidecar, tokens_file=str(path), tokens_file_sha256=sha256_file(path),
                     tokens_doc_header={k: v for k, v in doc.items() if k != "segments"})
    for seg in doc["segments"]:
        for rule, d in (("recent", root / "canon"), ("hf", root / "ref-hf")):
            js["jobs"].append({"name": f"{rule}.{seg['name']}", "kind": "logprobs", "rule": rule, "segment": seg["name"],
                               "ids": seg["token_ids"], "out_dir": str(d)})
    return write_jobset(root, js)


def _doc_ids(tok, row: dict, max_tokens: int | None) -> list[int]:
    """Canonical ids of one corpus-v2 document (NFC-normalized first: the canonical tokenizer refuses other text)."""
    nfc = lambda t: unicodedata.normalize("NFC", t)  # noqa: E731
    if row["format"] == "chat":
        ids = tok.encode(nfc(tok.render_chat(row["messages"], add_generation_prompt=False)))
    else:
        ids = tok.encode(nfc(row["text"]))
    return ids if max_tokens is None else ids[:max_tokens]


def corpus_v2_rows() -> list[dict]:
    p = MODELS_ROOT / "r4dx" / "corpus-v2" / "samples.jsonl"
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    return [r for r in rows if not r["rejected"]]


def prepare_probes(root: Path, model_dir: Path, sidecar: Path | None) -> Path:
    """probe_32k = the four canon segments (cpp, english, python, thai: 4096 tokens) followed by corpus-v2 documents
    (round robin english_prose, code, chat, thai_prose, multilingual; each document encoded on its own, canonical
    tokenizer, documents concatenated), cut at 32768 ids. probe_8k = its first 8192 ids."""
    from common import RefTokenizer

    doc, cpath = load_canon_doc()
    tok = RefTokenizer(TOKENIZER_DIR, "canonical")
    rows = corpus_v2_rows()
    cats = ["english_prose", "code", "chat", "thai_prose", "multilingual"]
    queues = {c: [r for r in rows if r["category"] == c] for c in cats}
    ids: list[int] = []
    for seg in doc["segments"]:
        ids += seg["token_ids"]
    used = []
    k = 0
    while len(ids) < 32768:
        c = cats[k % len(cats)]
        k += 1
        if not queues[c]:
            if not any(queues.values()):
                raise SystemExit("[real_ref] corpus-v2 exhausted before 32768 tokens")
            continue
        r = queues[c].pop(0)
        d = _doc_ids(tok, r, None)
        ids += d
        used.append({"id": r["id"], "tokens": len(d)})
    ids = ids[:32768]
    probes = {"probe_8k": ids[:8192], "probe_32k": ids}
    out_dir = root / "probes"
    header = {"tokenizer": f"{TOKENIZER_DIR / 'tokenizer.json'} (tokenizers.Tokenizer.from_file, canonical)", "tokenizer_mode": "canonical",
              "tokenizer_provenance": tok.provenance(), "add_special_tokens": False, "chat_template": False,
              "construction": ("probe_32k = 4 canon segments (tokens_canon.json order: " + ", ".join(s["name"] for s in doc["segments"]) +
                               ") + corpus-v2 samples.jsonl documents (round robin english_prose, code, chat[chat template, "
                               "add_generation_prompt=False], thai_prose, multilingual; each encoded separately) cut at 32768; "
                               "probe_8k = first 8192 ids of probe_32k"),
              "corpus_documents": used, "canon_tokens_sha256": sha256_file(cpath)}
    tokens_doc = {**header, "segments": [{"name": n, "token_ids": v} for n, v in probes.items()]}
    tpath = out_dir / "tokens_probes.json"
    atomic_write_json(tpath, tokens_doc)
    made = []
    for n, v in probes.items():
        js = base_jobset(root, n.replace("probe_", "probe"), "logprobs", model_dir, sidecar, tokens_file=str(tpath),
                         tokens_file_sha256=sha256_file(tpath), tokens_doc_header=header)
        for rule in ("recent", "hf"):  # both rules: the canon segments are dense (< 2052 tokens), the rule noise shows up here
            js["jobs"].append({"name": f"{rule}.{n}", "kind": "logprobs", "rule": rule, "segment": n, "ids": v,
                               "out_dir": str(out_dir / rule)})
        made.append(write_jobset(root, js))
    # numerical-noise control: the same recent-rule run with ONE more token (8193): every selection of the first 8192 rows is
    # identical, so KL(probe_8k || probe_8k_p1) over those rows is pure bf16 shape noise (MoE / GEMM batch composition)
    p1 = {"name": "probe_8k_p1", "token_ids": ids[:8193]}
    p1_path = out_dir / "tokens_probes_p1.json"
    atomic_write_json(p1_path, {**header, "segments": [p1]})
    js = base_jobset(root, "probe8k_p1", "logprobs", model_dir, sidecar, tokens_file=str(p1_path),
                     tokens_file_sha256=sha256_file(p1_path), tokens_doc_header=header)
    js["jobs"].append({"name": "recent.probe_8k_p1", "kind": "logprobs", "rule": "recent", "segment": "probe_8k_p1",
                       "ids": p1["token_ids"], "out_dir": str(out_dir / "recent")})
    write_jobset(root, js)
    return made[0]


def prepare_goldens(root: Path, model_dir: Path, sidecar: Path | None) -> Path:
    doc, _ = load_canon_doc()
    base = next(s for s in doc["segments"] if s["name"] == "english_prose")["token_ids"][:70]
    with_eos = list(base)
    eos = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["text_config"]["eos_token_id"]
    eos = eos[0] if isinstance(eos, list) else eos
    for p in (17, 41):
        with_eos[p] = int(eos)
    js = base_jobset(root, "goldens", "golden", model_dir, sidecar, golden_layers=list(GOLDEN_LAYERS))
    for seg, ids in (("T70", base), ("T70eos", with_eos)):
        for rule in GOLDEN_TIE_RULES:
            js["jobs"].append({"name": f"{rule}.{seg}", "kind": "golden", "rule": rule, "segment": seg, "ids": ids,
                               "out_dir": str(root / "layer_goldens")})
    return write_jobset(root, js)


def prepare_kvcalib(root: Path, model_dir: Path, sidecar: Path | None, target_tokens: int = 50000, max_doc_tokens: int = 2048) -> Path:
    """~50k tokens, one sequence per document: the kv_calibrate_full corpus (kv_calib_corpus/*, calib.txt; the files and the
    chat conversation rendered with the chat template) plus corpus-v2 documents, round robin over categories, until
    `target_tokens`; every document truncated to `max_doc_tokens`."""
    from common import RefTokenizer

    tok = RefTokenizer(TOKENIZER_DIR, "canonical")
    ref_dir = HERE.parent
    docs: list[dict] = []

    def add(name, kind, ids, src_sha):
        docs.append({"name": name, "kind": kind, "ids": ids[:max_doc_tokens], "source_sha256": src_sha})

    cdir = ref_dir / "kv_calib_corpus"
    paths = [ref_dir / "calib.txt"] + sorted(cdir.glob("*.txt")) + sorted(cdir.glob("*.messages.json"))
    for p in paths:
        if p.name.endswith(".messages.json"):
            m = json.loads(p.read_text(encoding="utf-8"))
            m = m["messages"] if isinstance(m, dict) else m
            add("kvc/" + p.name[: -len(".messages.json")], "chat_template", tok.encode(tok.render_chat(m, add_generation_prompt=False)), sha256_file(p))
        else:
            add("kvc/" + p.stem, "text", tok.encode(p.read_text(encoding="utf-8")), sha256_file(p))
    rows = corpus_v2_rows()
    cats = ["english_prose", "code", "chat", "thai_prose", "multilingual"]
    queues = {c: [r for r in rows if r["category"] == c] for c in cats}
    total = sum(len(d["ids"]) for d in docs)
    k = 0
    while total < target_tokens and any(queues.values()):
        c = cats[k % len(cats)]
        k += 1
        if not queues[c]:
            continue
        r = queues[c].pop(0)
        ids = _doc_ids(tok, r, max_doc_tokens)
        if len(ids) < 16:
            continue
        add("v2/" + r["id"], "chat_template" if r["format"] == "chat" else "text", ids, None)
        total += len(ids)
    js = base_jobset(root, "kvcalib", "calib", model_dir, sidecar, out_json=str(root / "kv_calib.json"),
                     tokenizer_provenance=tok.provenance(), max_doc_tokens=max_doc_tokens,
                     corpus_meta=[{"name": d["name"], "kind": d["kind"], "tokens": len(d["ids"]), "source_sha256": d["source_sha256"],
                                   "ids_sha256": token_ids_sha256(d["ids"])} for d in docs])
    for n, d in enumerate(docs):
        js["jobs"].append({"name": f"{n:03d}", "kind": "calib", "rule": "recent", "segment": d["name"], "ids": d["ids"],
                           "out_dir": str(root / "work" / "kvcalib")})
    return write_jobset(root, js)


def prepare_canon_p1(root: Path, model_dir: Path, sidecar: Path | None) -> Path:
    """Shape-noise control of the canon corpus: every canon segment as the first 1024 + 1 tokens of its source text
    (kl_corpus/<name>.txt, canonical tokenizer; the canon ids are its first 1024), tie rule recent, `<segment>_p1`."""
    from common import RefTokenizer

    doc, path = load_canon_doc()
    tok = RefTokenizer(TOKENIZER_DIR, "canonical")
    segs = []
    for seg in doc["segments"]:
        ids = tok.encode((KL_CORPUS_DIR / f"{seg['name']}.txt").read_text(encoding="utf-8"))[:1025]
        if ids[:1024] != seg["token_ids"]:
            raise SystemExit(f"{seg['name']}: tokenization of the source text does not start with the canon ids")
        segs.append({"name": seg["name"] + "_p1", "token_ids": ids})
    header = {k: v for k, v in doc.items() if k != "segments"}
    tpath = root / "canon_p1" / "tokens_canon_p1.json"
    atomic_write_json(tpath, {**header, "segments": segs})
    js = base_jobset(root, "canon_p1", "logprobs", model_dir, sidecar, tokens_file=str(tpath), tokens_file_sha256=sha256_file(tpath),
                     tokens_doc_header=header)
    for s in segs:
        js["jobs"].append({"name": f"recent.{s['name']}", "kind": "logprobs", "rule": "recent", "segment": s["name"],
                           "ids": s["token_ids"], "out_dir": str(root / "canon_p1")})
    return write_jobset(root, js)


def prepare_truth(root: Path, model_dir: Path, sidecar: Path | None) -> Path:
    """fp32 truth jobsets (tie rule recent, whole stack in float32; ref.py `dtype=float32`): the canon segments and probe_8k.
    `truth_*`: unrounded AWQ dequant (out_dir truth-fp32). `truthw16_*`: the bf16-rounded AWQ weights the bf16 stack uses, widened,
    fp32 compute (out_dir truth-fp32-w16): separates the weight rounding from the activation rounding."""
    canon, cpath = load_canon_doc()
    probes_path = root / "probes" / "tokens_probes.json"
    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    p8 = next(s for s in probes["segments"] if s["name"] == "probe_8k")
    made = None
    for prefix, exact, out in (("truth", True, "truth-fp32"), ("truthw16", False, "truth-fp32-w16")):
        for name, doc, path, segs in (("canon", canon, cpath, canon["segments"]), ("probe8k", probes, probes_path, [p8])):
            js = base_jobset(root, f"{prefix}_{name}", "logprobs", model_dir, sidecar, tokens_file=str(path),
                             tokens_file_sha256=sha256_file(path), tokens_doc_header={k: v for k, v in doc.items() if k != "segments"},
                             dtype="float32", fp32_exact_weights=exact)
            for seg in segs:
                js["jobs"].append({"name": f"recent.{seg['name']}", "kind": "logprobs", "rule": "recent", "segment": seg["name"],
                                   "ids": seg["token_ids"], "out_dir": str(root / out)})
            made = write_jobset(root, js)
    # the tie rule in the fp32 world (no bf16 shape noise to hide behind): probe_8k with HF's own topk choice, to compare to truth-fp32
    js = base_jobset(root, "truth_probe8k_hf", "logprobs", model_dir, sidecar, tokens_file=str(probes_path),
                     tokens_file_sha256=sha256_file(probes_path), tokens_doc_header={k: v for k, v in probes.items() if k != "segments"},
                     dtype="float32", fp32_exact_weights=True)
    js["jobs"].append({"name": "hf.probe_8k", "kind": "logprobs", "rule": "hf", "segment": "probe_8k", "ids": p8["token_ids"],
                       "out_dir": str(root / "truth-fp32-hf")})
    return write_jobset(root, js)


PREPARERS = {"truth": prepare_truth, "canon_p1": prepare_canon_p1, "canon": prepare_canon, "probes": prepare_probes, "goldens": prepare_goldens, "kvcalib": prepare_kvcalib}
JOBSET_NAMES = {"truth": ["truth_canon", "truth_probe8k", "truthw16_canon", "truthw16_probe8k", "truth_probe8k_hf"], "canon_p1": ["canon_p1"], "canon": ["canon"], "probes": ["probe8k", "probe32k", "probe8k_p1"], "goldens": ["goldens"], "kvcalib": ["kvcalib"]}


# --------------------------------------------------------------------------------------------------------
# manifest / ngram check
# --------------------------------------------------------------------------------------------------------


def cmd_manifest(args) -> int:
    rows = []
    for p in args.paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and "work" not in f.relative_to(p).parts)
        for f in files:
            rows.append({"path": str(f), "bytes": f.stat().st_size, "sha256": sha256_path(f)})
            print(f"{rows[-1]['sha256']}  {rows[-1]['bytes']:>14}  {f}", flush=True)
    if args.out:
        atomic_write_json(Path(args.out), {"generated_at": now_iso(), "files": rows})
    return 0


def cmd_ngram_check(args) -> int:
    from qwen4exp.ngram_ids import NGramHash

    cfg = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
    nh = NGramHash.from_config(cfg, ple_layer_index=0)
    mm = SidecarNGramRows(args.sidecar, nh)
    un = UnbufferedSidecarRows(args.sidecar, nh)
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, nh.padded_rows, size=(args.tokens, 16), dtype=np.int64)
    ids[0, :4] = [0, 1, nh.padded_rows - 1, nh.padded_rows - 2]
    t0 = time.perf_counter()
    a = un.rows(ids)
    t1 = time.perf_counter()
    b = mm.rows(ids)
    t2 = time.perf_counter()
    same = torch.equal(a, b)
    print(f"[ngram-check] {ids.size} random rows ({np.unique(ids).size} unique): unbuffered {t1 - t0:.2f}s "
          f"({un.reads} reads, {un.bytes_read / 2**20:.1f} MiB), memmap {t2 - t1:.2f}s, equal={same}", flush=True)
    return 0 if same else 1


def cmd_prepare(args) -> int:
    root = Path(args.root)
    sidecar = Path(args.sidecar) if args.sidecar else (DEFAULT_SIDECAR if DEFAULT_SIDECAR.exists() else None)
    for what in args.what:
        p = PREPARERS[what](root, Path(args.model_dir), sidecar)
        for name in JOBSET_NAMES[what]:
            js = load_jobset(root, name)
            print(f"[real_ref] prepared {name}: {len(js['jobs'])} jobs, {sum(len(j['ids']) for j in js['jobs'])} tokens -> {jobset_path(root, name)}")
    return 0


def cmd_step(args) -> int:
    root = Path(args.root)
    js = load_jobset(root, args.jobset)
    if args.model_dir:
        js["model_dir"] = str(args.model_dir)
    return run_step(root, js, args.device, args.max_minutes, args.max_layers, args.ckpt_every, args.cache_gib)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("what", nargs="+", choices=sorted(PREPARERS))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--sidecar", default=None)
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("step")
    p.add_argument("--jobset", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model-dir", default=None)
    p.add_argument("--max-minutes", type=float, default=11.0, help="stop at a layer boundary when the next layer would pass this")
    p.add_argument("--max-layers", type=int, default=None, help="stop after this many layers of this step")
    p.add_argument("--ckpt-every", type=int, default=4, help="residual checkpoint every N layers (always at the end of a step)")
    p.add_argument("--cache-gib", type=float, default=10.0, help="dequantized-weight cache per layer when several sequences share it")
    p.set_defaults(fn=cmd_step)
    p = sub.add_parser("manifest")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("ngram-check")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    p.add_argument("--tokens", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_ngram_check)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

