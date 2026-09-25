"""tools/reference/hessian_capture.py

The **input Hessians** `H = X^T X / rows` that `r4dx-convert --ldlq` (docs/quant2.md section 2)
needs to round a linear's weights with GPTQ/LDLQ error feedback instead of independently.

For a linear `y = x W^T` with `W [N, K]`, quantizing `W` to `Wq` moves the output by
`x (W - Wq)^T`, so the mean squared output error over a calibration set is
`tr((W - Wq) H (W - Wq)^T)` with `H = mean over tokens of x x^T` -- a `[K, K]` matrix per distinct
linear INPUT. `imatrix_capture.py` measures only its diagonal (per-channel importance); LDLQ needs
the whole thing, because the correlations between input channels are exactly what lets the
rounding error of column `k` be compensated on the columns after it. Nothing here changes a byte
layout: like the imatrix, the Hessian is an input to a better choice of `q/scale/zero` for the
same containers and the same kernels.

**Layer-major, unlike `imatrix_capture.py`.** The imatrix streams the whole 64-layer stack once per
sequence and can afford to keep all 341 `[K]` accumulators resident. A Hessian accumulator is
`[K, K]` fp64 -- `mlp.down`'s alone is 17408^2 x 8 B = 2.3 GiB -- so all of them at once would be
~200 GiB. This script therefore turns the loop inside out: every calibration sequence is embedded
once, then for each layer `i` the layer is materialized ONCE (`StreamingReference.build_layer`) and
every sequence's hidden state is pushed through it with exactly the call
`StreamingReference._forward_manual` makes (`attention_mask=None` for `linear_attention` layers,
the additive causal mask otherwise, partial-rope `(cos, sin)` from `_manual_rope`). Only one
layer's taps are ever resident (<= 3.2 GiB fp64), and they are written to disk and freed before
the next layer is built. The hidden states of all sequences (~1.8 GiB bf16 for the default
corpus) are the only thing carried from layer to layer.

**Taps, not linears.** Several converter tensors read the same input: `gdn.in_proj_qkv` and
`gdn.in_proj_z` both read the post-`input_layernorm` hidden of a GDN layer; `attn.qg`, `attn.k`,
`attn.v` all read it on an attention layer; `mlp.gate_up` is `[gate; up]` on the row axis and
`gate_proj`/`up_proj` read one tensor. One Hessian per DISTINCT input ("tap") therefore serves
every key that shares it, and each tap is hooked on exactly ONE representative module:

    layer type | tap      | representative         | keys served             | file
    GDN        | in       | linear_attn.in_proj_qkv| gdn.in_proj_qkv, _z     | L{i:02d}.in.hess
    GDN        | out      | linear_attn.out_proj   | gdn.out_proj            | L{i:02d}.out.hess
    attention  | in       | self_attn.q_proj       | attn.qg, attn.k, attn.v | L{i:02d}.in.hess
    attention  | out      | self_attn.o_proj       | attn.o                  | L{i:02d}.out.hess
    both       | mlp_in   | mlp.gate_proj          | mlp.gate_up             | L{i:02d}.mlp_in.hess
    both       | mlp_mid  | mlp.down_proj          | mlp.down                | L{i:02d}.mlp_mid.hess
    head       | lm_head  | model.norm output      | lm_head                 | lm_head.hess
    MTP        | 4 taps   | the MTP layer's modules| mtp.attn.qg/o, mtp.mlp.*| mtp.{in,out,mlp_in,mlp_mid}.hess

"The companions really see the same tensor" is proved, not assumed: on the first GDN layer and the
first attention layer `in_proj_z` / `k_proj` / `v_proj` / `up_proj` are hooked too and must receive
the SAME storage (`data_ptr`) and equal values as their representative, on every sequence
(`gates.shared_input`). The key -> tap assignment itself comes from
`imatrix_capture.enumerate_quantized_linears`, whose table `audit_converter_source` re-derives from
`src/convert/main.cpp`'s text on every run, so a new `add_linear` cannot slip past silently.

**lm_head and MTP.** `lm_head`'s input is the last layer's output through the model's own final
norm module (`ref.model.norm`, the very module `forward_hidden` -- and so `imatrix_capture.py` --
applies). The MTP head (unless `--no-mtp`) is driven by `imatrix_capture.MtpTaps` unchanged, fed
the PRE-final-norm hidden per sequence: its hooks look accumulators up in `result.accums` by
container key and call `.update(x)`, so pre-populating that dict with Hessian accumulators is all
it takes (`T-1` rows per sequence there, see imatrix_capture's docstring).

**Numerics.** Per sequence, `x` (bf16) is upcast to fp32 and `x^T x` is formed by one fp32 GEMM
(TF32 explicitly off) -- a sum of <= 2048 products per entry -- and added into an fp64 device
accumulator, so the ~172k-row sum loses nothing to accumulation order. `H = acc / rows` is written
as fp32, packed upper triangle (`write_hess_file`, the exact header below), and the header's trace
is computed from the fp32 values actually written so a reader can check it bit-for-bit-ish.

File format (little-endian), one per tap, `<out-dir>/<file>.hess`:

    [8]  magic "R4DXHES1"
    [4]  u32 K
    [4]  u32 flags            (bit 0 = packed upper triangle; always set in v1)
    [8]  u64 rows             (tokens accumulated)
    [8]  f64 trace            (sum of the stored fp32 diagonal)
    [32] reserved (zero)
    then K*(K+1)/2 float32: H[i][j] for i = 0..K-1, j = i..K-1

plus `<out-dir>/hessian.json` (format "r4dx-hessian", version 1): `files` (K, rows, trace per
file), `keys` (container base name -> file), the corpus with sha256 per source, the checkpoint's
config sha256, the converter audit, and the gate results. The manifest is written LAST, after every
`.hess` file, and any old one is deleted before the capture starts, so a crashed run leaves a
directory the converter refuses to read rather than a half-valid one.

Usage (reference venv only -- read-only against the venv and the checkpoint):

    $env:HIP_VISIBLE_DEVICES = '1'
    <venv>\\Scripts\\python.exe tools\\reference\\hessian_capture.py --dry-run
    <venv>\\Scripts\\python.exe tools\\reference\\hessian_capture.py --out-dir D:\\models\\r4dx\\hessian-v1

`--dry-run` and `--write-fixture` never touch the GPU. See tools/reference/README.md
("hessian_capture.py") for the option list, the corpus, the gates and the expected runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# NOTE: torch, transformers and the sibling reference modules (all of which import torch) are
# imported lazily inside the functions that need them, so `--write-fixture` runs on any python that
# has numpy -- it needs no checkpoint, no torch and no GPU.

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_REF = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = TOOLS_REF / "kv_calib_corpus"
DEFAULT_CALIB_TXT = TOOLS_REF / "calib.txt"
DEFAULT_WIKITEXT = Path("D:/models/wikitext-2-raw/wiki.train.raw")
DEFAULT_OUT_DIR = Path(r"D:\models\r4dx\hessian-v1")

#: The held-out KL corpus (tools/reference/kl_corpus/) is made of excerpts of these two files; the
#: Hessian corpus must stay disjoint from the text the drift is measured on (same reasoning as the
#: imatrix corpus -- calibrating on the eval text would flatter the result).
KL_CORPUS_SOURCES = ("src/model/model.cpp", "tools/reference/layer_golden.py")
KL_CORPUS_DIR = "tools/reference/kl_corpus/"
CODE_SUFFIXES = (".cpp", ".h", ".hpp", ".hip", ".py")
#: Not "this repo's own" source: vendored libraries. Their style is not r4dx's and they are large
#: (src/tokenizer/vendor/ is llama.cpp's unicode tables, ~236 KB of mostly hex range literals).
CODE_EXCLUDE_PREFIXES = ("third_party/", "src/tokenizer/vendor/")

HESS_MAGIC = b"R4DXHES1"
HESS_FLAG_PACKED_UPPER = 1
#: magic, K, flags, rows, trace, 32 reserved bytes -- 64 bytes, no implicit padding under '<'.
HESS_HEADER = struct.Struct("<8sIIQd32x")
assert HESS_HEADER.size == 64

MANIFEST_NAME = "hessian.json"
#: Where the manifest goes instead when a gate fails: the provenance (and the gate report) is kept
#: for diagnosis, but under a name the converter's HessianStore never opens, so a failed set cannot
#: be converted against by accident.
FAILED_MANIFEST_NAME = "hessian.failed.json"
MANIFEST_FORMAT = "r4dx-hessian"
MANIFEST_VERSION = 1

#: The design's rank condition (docs/quant2.md gate G2): with fewer than a few rows per column the
#: Hessian of a K = 17408 linear is badly rank-deficient and LDLQ leans entirely on damping.
ROWS_PER_K_MIN = 4

#: Modules whose input is ANOTHER module's input (same tensor), and the representative that is
#: actually hooked for it. Anything a converter spec names that is neither here nor in TAP_OF is a
#: hard error (the converter audit should have caught it first).
SHARED_WITH = {
    "linear_attn.in_proj_z": "linear_attn.in_proj_qkv",
    "self_attn.k_proj": "self_attn.q_proj",
    "self_attn.v_proj": "self_attn.q_proj",
}
TAP_OF = {
    "linear_attn.in_proj_qkv": "in",
    "self_attn.q_proj": "in",
    "linear_attn.out_proj": "out",
    "self_attn.o_proj": "out",
    "mlp.gate_proj": "mlp_in",
    "mlp.down_proj": "mlp_mid",
}
MTP_TAP_OF = {
    "self_attn.q_proj": "in",
    "self_attn.o_proj": "out",
    "mlp.gate_proj": "mlp_in",
    "mlp.down_proj": "mlp_mid",
}
#: Shared-input gate: representative -> companions, per layer type, checked on the first layer of
#: each type. `up_proj` is the companion of `gate_proj` on both types.
SHARED_GATE = {
    "linear_attention": [("linear_attn.in_proj_qkv", ["linear_attn.in_proj_z"]),
                         ("mlp.gate_proj", ["mlp.up_proj"])],
    "full_attention": [("self_attn.q_proj", ["self_attn.k_proj", "self_attn.v_proj"]),
                       ("mlp.gate_proj", ["mlp.up_proj"])],
}

CAVEAT = (
    "H = mean over calibration tokens of x x^T (x = the linear's INPUT activation, bf16 upcast to "
    "fp32, fp64 accumulation, stored fp32 packed upper triangle), measured on a REAL forward of "
    "the original bf16 checkpoint: every layer was fed the true output of all the real bf16 layers "
    "before it (full_logits_golden.StreamingReference's layers, driven layer-major). What this is "
    "NOT: it is not a sequential-GPTQ Hessian -- the preceding layers are the bf16 originals, not "
    "their quantized versions (docs/quant2.md 1.2) -- and it is only as representative as the "
    "corpus (calib.txt + kv_calib_corpus/ + WikiText-2 train + this repo's own sources, disjoint "
    "from the held-out KL corpus tools/reference/kl_corpus/)."
)


# --------------------------------------------------------------------------------------------
# The file format (shared by the capture and --write-fixture)
# --------------------------------------------------------------------------------------------


def hess_file_bytes(k: int) -> int:
    return HESS_HEADER.size + (k * (k + 1) // 2) * 4


def write_hess_file(path: Path, h: np.ndarray, rows: int, want_sum: bool = False) -> dict:
    """Write one `.hess` file from a symmetric `[K, K]` matrix (only the upper triangle is read).

    The matrix is rounded to fp32 FIRST and everything recorded -- the header trace, the returned
    min/max diagonal and (with `want_sum`, meant for the small fixture: it holds every entry as a
    Python float) the exactly-rounded fp64 sum of all stored entries -- is computed from those fp32
    values, i.e. from exactly the bytes a reader will see. Written to
    `<path>.tmp` and renamed into place, so a crash never leaves a truncated file under the real
    name. Returns the stats dict the manifest and the gates are built from."""
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"write_hess_file: expected a square matrix, got shape {h.shape}")
    k = int(h.shape[0])
    if k <= 0 or k >= 2**32:
        raise ValueError(f"write_hess_file: K={k} out of range")
    if rows <= 0:
        raise ValueError(f"write_hess_file: rows={rows} (nothing accumulated)")
    h32 = np.ascontiguousarray(h, dtype="<f4")
    diag = np.diagonal(h32).astype(np.float64)
    trace = math.fsum(diag.tolist())
    finite = True
    sum_upper = 0.0
    partial_sums: list[float] = []
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb", buffering=16 << 20) as f:
        f.write(HESS_HEADER.pack(HESS_MAGIC, k, HESS_FLAG_PACKED_UPPER, int(rows), trace))
        for i in range(k):
            row = h32[i, i:]
            if finite and not np.isfinite(row).all():
                finite = False
            if want_sum:
                partial_sums.extend(row.astype(np.float64).tolist())
            f.write(row.tobytes())
    if want_sum:
        sum_upper = math.fsum(partial_sums)
    os.replace(tmp, path)
    size = path.stat().st_size
    if size != hess_file_bytes(k):
        raise RuntimeError(f"{path}: wrote {size} bytes, expected {hess_file_bytes(k)}")
    stats = {"K": k, "rows": int(rows), "trace": trace, "finite": bool(finite),
             "min_diag": float(diag.min()), "max_diag": float(diag.max()), "bytes": size}
    if want_sum:
        stats["sum_upper"] = sum_upper
    return stats


def read_hess_file(path: Path) -> tuple[np.ndarray, int, float]:
    """The Python mirror of `hessian_store.hpp`'s `ReadHessFile`: returns the full symmetric fp32
    `[K, K]` matrix, rows and the header trace, validating magic, flags, size and trace. Used by
    `--write-fixture` to round-trip what it just wrote."""
    data = path.read_bytes()
    if len(data) < HESS_HEADER.size:
        raise ValueError(f"{path}: {len(data)} bytes, shorter than the {HESS_HEADER.size}-byte header")
    magic, k, flags, rows, trace = HESS_HEADER.unpack_from(data, 0)
    if magic != HESS_MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if flags != HESS_FLAG_PACKED_UPPER:
        raise ValueError(f"{path}: flags {flags:#x}, only {HESS_FLAG_PACKED_UPPER:#x} is v1")
    if len(data) != hess_file_bytes(k):
        raise ValueError(f"{path}: {len(data)} bytes, K={k} needs {hess_file_bytes(k)}")
    packed = np.frombuffer(data, dtype="<f4", offset=HESS_HEADER.size)
    h = np.zeros((k, k), dtype=np.float32)
    iu = np.triu_indices(k)
    h[iu] = packed
    h.T[iu] = packed
    got = math.fsum(np.diagonal(h).astype(np.float64).tolist())
    if not math.isclose(got, trace, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(f"{path}: header trace {trace!r} but the stored diagonal sums to {got!r}")
    return h, int(rows), float(trace)


def write_manifest(out_dir: Path, files: dict[str, dict], keys: dict[str, str],
                   extra: dict, manifest_name: str = MANIFEST_NAME) -> Path:
    """`hessian.json` (or `manifest_name`): the contract's three fields first (format/version, files {K,
    rows, trace}, keys), then whatever provenance the caller passes. LF line endings on every
    platform (its sha256 is recorded in converted containers), written via a temp file + rename."""
    doc = {"format": MANIFEST_FORMAT, "version": MANIFEST_VERSION,
           "files": {name: {"K": int(v["K"]), "rows": int(v["rows"]), "trace": float(v["trace"])}
                     for name, v in sorted(files.items())},
           "keys": dict(sorted(keys.items()))}
    for name in keys.values():
        if name not in doc["files"]:
            raise RuntimeError(f"manifest key points at {name!r}, which is not in files")
    for k, v in extra.items():
        if k in doc:
            raise RuntimeError(f"manifest extra field {k!r} would overwrite a contract field")
        doc[k] = v
    path = out_dir / manifest_name
    tmp = out_dir / (manifest_name + ".tmp")
    payload = (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------------------------
# --write-fixture: the tiny deterministic set tests/convert/ reads (CPU, numpy only)
# --------------------------------------------------------------------------------------------

FIXTURE_SEED = 20260925
#: (file name, K, rows, keys). rows >= 4K so the fixture itself passes the rows gate.
FIXTURE_FILES = (("a.hess", 128, 1024, ("t.a",)),
                 ("bc.hess", 256, 2048, ("t.b", "t.c")))


def _fixture_x(rng: np.random.Generator, k: int, rows: int) -> np.ndarray:
    """Activations with the two properties real ones have and LDLQ cares about: strongly
    CORRELATED neighbouring channels (an AR(1) covariance, rho = 0.9) and a heavy-tailed
    per-channel scale (log-normal, plus a handful of 10x "massive activation" channels)."""
    idx = np.arange(k)
    cov = 0.9 ** np.abs(idx[:, None] - idx[None, :])
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((rows, k))
    scale = np.exp(0.5 * rng.standard_normal(k))
    scale[rng.choice(k, size=max(1, k // 64), replace=False)] *= 10.0
    return (z @ chol.T) * scale[None, :]


def write_fixture(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FIXTURE_SEED)
    files: dict[str, dict] = {}
    keys: dict[str, str] = {}
    expected: dict = {"files": {}}
    for name, k, rows, file_keys in FIXTURE_FILES:
        x = _fixture_x(rng, k, rows)
        h = (x.T @ x) / float(rows)  # fp64; write_hess_file rounds to fp32
        st = write_hess_file(out_dir / name, h, rows, want_sum=True)
        if not (st["finite"] and st["min_diag"] > 0.0):
            raise SystemExit(f"[hessian] fixture {name} is not finite/positive: {st}")
        files[name] = st
        for key in file_keys:
            keys[key] = name
        # Round-trip through the Python reader, and pull a few exact entries (both triangles, so a
        # C++ reader test checks the symmetric expansion too) for the test to compare against.
        back, back_rows, back_trace = read_hess_file(out_dir / name)
        want32 = np.asarray(h, dtype=np.float32)
        iu = np.triu_indices(k)
        if back_rows != rows or back_trace != st["trace"] or \
                not np.array_equal(back[iu], want32[iu]) or not np.array_equal(back, back.T):
            raise SystemExit(f"[hessian] fixture {name}: read-back mismatch")
        samples = [(0, 0), (0, 1), (1, 0), (k // 2, k // 3), (k // 3, k // 2), (k - 1, 0),
                   (k - 1, k - 1)]
        expected["files"][name] = {
            "K": k, "rows": rows, "trace": st["trace"], "sum_upper": st["sum_upper"],
            "min_diag": st["min_diag"], "max_diag": st["max_diag"], "bytes": st["bytes"],
            "sha256": hashlib.sha256((out_dir / name).read_bytes()).hexdigest(),
            "samples": [[i, j, float(back[i, j])] for i, j in samples],
        }
    manifest = write_manifest(out_dir, files, keys, {
        "tool": "tools/reference/hessian_capture.py --write-fixture",
        "fixture": {
            "seed": FIXTURE_SEED,
            "generator": ("numpy default_rng; X = (Z @ chol(AR1(0.9)).T) * lognormal channel "
                          "scales with k//64 channels x10; H = X^T X / rows in fp64, stored fp32"),
            "note": "synthetic test data for tests/convert (hessian_store.hpp); not a real Hessian",
        },
    })
    expected["keys"] = dict(sorted(keys.items()))
    expected["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected["semantics"] = {
        "trace": "f64 sum of the stored fp32 diagonal (== the header's trace field)",
        "sum_upper": "exactly-rounded (math.fsum) f64 sum of all K*(K+1)/2 stored fp32 entries",
        "samples": "[i, j, H[i][j]] of the full symmetric expansion; values are exact fp32",
        "manifest_sha256": "sha256 of hessian.json's bytes (HessianStore::ManifestSha256)",
    }
    with open(out_dir / "expected.json", "wb") as f:
        f.write((json.dumps(expected, indent=2) + "\n").encode("utf-8"))
    for name, st in files.items():
        print(f"[hessian] fixture {name}: K={st['K']} rows={st['rows']} trace={st['trace']:.9g} "
              f"sum_upper={st['sum_upper']:.9g} min_diag={st['min_diag']:.6g} ({st['bytes']} B)")
    print(f"[hessian] wrote {manifest} and {out_dir / 'expected.json'} "
          f"(manifest sha256 {expected['manifest_sha256']})")
    return 0


# --------------------------------------------------------------------------------------------
# The tap plan: converter keys -> files
# --------------------------------------------------------------------------------------------


@dataclass
class TapPlan:
    file: str                   #: e.g. "L03.in.hess"
    scope: str                  #: "layer" | "lm_head" | "mtp"
    layer: int | None           #: text layer index for scope "layer"
    module: str                 #: the ONE hooked module path (or "final_norm_out" / "mtp:...")
    k: int                      #: expected in-features (from the checkpoint's safetensors headers)
    keys: list[str] = field(default_factory=list)
    describe: str = ""


def build_tap_plan(specs, expect_k: dict[str, int]) -> list[TapPlan]:
    """Group `imatrix_capture.enumerate_quantized_linears`' specs by the input they read."""
    plans: dict[str, TapPlan] = {}
    for s in specs:
        if s.tap == "final_norm_out":
            p = plans.setdefault("lm_head.hess", TapPlan("lm_head.hess", "lm_head", None,
                                                         "final_norm_out", expect_k[s.key]))
            p.describe = "post-final-norm hidden (ref.model.norm of the last layer's output)"
        elif s.tap == "mtp:norm_out":
            p = plans.setdefault("mtp.draft_head.hess", TapPlan("mtp.draft_head.hess", "mtp", None,
                                                                s.tap, expect_k[s.key]))
            p.describe = "post-mtp.norm hidden of the MTP layer"
        elif s.tap.startswith("mtp:"):
            module = s.tap.split(":", 1)[1]
            if module not in MTP_TAP_OF:
                raise SystemExit(f"[hessian] no tap rule for MTP module {module!r} ({s.key})")
            name = f"mtp.{MTP_TAP_OF[module]}.hess"
            p = plans.setdefault(name, TapPlan(name, "mtp", None, s.tap, expect_k[s.key]))
            p.describe = f"forward_pre_hook on the MTP layer's {module}"
        elif s.tap.startswith("layer"):
            scope, module = s.tap.split(":", 1)
            i = int(scope[len("layer"):])
            rep = SHARED_WITH.get(module, module)
            if rep not in TAP_OF:
                raise SystemExit(f"[hessian] no tap rule for text module {module!r} ({s.key})")
            name = f"L{i:02d}.{TAP_OF[rep]}.hess"
            p = plans.setdefault(name, TapPlan(name, "layer", i, rep, expect_k[s.key]))
            p.describe = f"forward_pre_hook on layer {i}'s {rep}"
        else:
            raise SystemExit(f"[hessian] unknown tap {s.tap!r} for {s.key}")
        if expect_k[s.key] != p.k:
            raise SystemExit(f"[hessian] {s.key} has K={expect_k[s.key]} but shares tap {p.file} "
                             f"(K={p.k}) -- the shared-input table is wrong for this checkpoint")
        p.keys.append(s.key)
    return list(plans.values())


# --------------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------------


@dataclass
class Seq:
    name: str
    source: str
    token_ids: list[int]


def _tokenize_prefix(tokenizer, text: str, need: int, margin: int = 256) -> tuple[list[int], int]:
    """The first `need` tokens of `text` without tokenizing all of it. Tokenizes a growing prefix
    until it yields `need + margin` tokens: BPE pre-tokenization is local, so only the last few
    tokens before the cut can differ from a whole-text tokenization, and the margin is discarded.
    Returns (the first `need` ids -- fewer if the text runs out, chars tokenized)."""
    chars = max(4096, need * 8)
    while True:
        chunk = text[:chars]
        ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        if len(ids) >= need + margin or chars >= len(text):
            return list(ids)[:need], len(chunk)
        chars *= 2


def repo_code_files() -> tuple[list[str], str]:
    """This repo's own C++/HIP/Python sources, repo-relative posix paths, sorted. Tracked files
    only (`git ls-files`) so build trees and scratch files never leak in; falls back to walking the
    tree (skipping build*/, .git/, dot-dirs) when git is unavailable."""
    method = "git ls-files"
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"], check=True,
                             capture_output=True).stdout.decode("utf-8")
        paths = [p for p in out.split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        method = "filesystem walk (git unavailable)"
        paths = []
        for p in REPO_ROOT.rglob("*"):
            rel = p.relative_to(REPO_ROOT).as_posix()
            top = rel.split("/", 1)[0]
            if top.startswith(".") or top.startswith("build") or not p.is_file():
                continue
            paths.append(rel)
    keep = []
    for rel in paths:
        if not rel.endswith(CODE_SUFFIXES):
            continue
        if rel in KL_CORPUS_SOURCES or rel.startswith(KL_CORPUS_DIR):
            continue
        if rel.startswith(CODE_EXCLUDE_PREFIXES):
            continue
        keep.append(rel)
    return sorted(keep), method


def build_corpus(args, tokenizer) -> tuple[list[Seq], list[dict]]:
    from common import sha256_file
    from kv_calibrate_full import collect_corpus, repo_relative

    seqs: list[Seq] = []
    sources: list[dict] = []
    L = args.seq_len

    # 1. calib.txt + kv_calib_corpus/: exactly imatrix_capture's corpus, one sequence per file.
    calib_txt = None if str(args.calib_txt).lower() == "none" else args.calib_txt
    corpus_dir = None if str(args.corpus_dir).lower() == "none" else args.corpus_dir
    if calib_txt is not None or corpus_dir is not None:
        for c in collect_corpus(corpus_dir, calib_txt, [], tokenizer, L):
            seqs.append(Seq(f"calib/{c.name}", "calib", list(c.token_ids)))
            sources.append({"source": "calib", "name": c.name, "path": repo_relative(c.path),
                            "kind": c.kind, "sha256": sha256_file(c.path),
                            "tokens": len(c.token_ids), "sequences": 1})

    # 2. WikiText-2 train: the first N non-overlapping L-token windows, from the start.
    if str(args.wikitext).lower() != "none" and args.wikitext_seqs > 0:
        path = Path(args.wikitext)
        if not path.is_file():
            raise SystemExit(f"[hessian] --wikitext {path} does not exist ('none' to skip)")
        text = path.read_bytes().decode("utf-8")
        need = args.wikitext_seqs * L
        ids, chars = _tokenize_prefix(tokenizer, text, need)
        if len(ids) < need:
            raise SystemExit(f"[hessian] {path} has only {len(ids)} tokens, need {need} "
                             f"({args.wikitext_seqs} x {L})")
        for w in range(args.wikitext_seqs):
            seqs.append(Seq(f"wikitext/{w:03d}", "wikitext", ids[w * L:(w + 1) * L]))
        sources.append({"source": "wikitext", "path": str(path), "sha256": sha256_file(path),
                        "chars_tokenized": chars, "tokens": need, "sequences": args.wikitext_seqs,
                        "windows": f"non-overlapping, token offsets 0, {L}, ..., {need - L}"})

    # 3. This repo's own sources, concatenated (sorted, one header line per file), N windows spread
    #    evenly over the whole concatenation so every part of the tree -- kernels, model, server,
    #    tests, Python tooling -- is represented, not just the alphabetically first directory.
    if args.code_seqs > 0:
        files, method = repo_code_files()
        parts = []
        for rel in files:
            # CRLF -> LF: with core.autocrlf=true the checkout's line endings are a property of the
            # machine, not of the source, and must not change the tokens or the recorded sha256.
            body = (REPO_ROOT / rel).read_bytes().decode("utf-8", errors="replace")
            body = body.replace("\r\n", "\n")
            if not body.endswith("\n"):
                body += "\n"
            parts.append(f"==> {rel} <==\n{body}")
        text = "".join(parts)
        # verbose=False: silences the "longer than the model's maximum length" warning -- this is
        # tokenization of a whole source tree, not a model input; only L-token windows are run.
        ids = list(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
        n = args.code_seqs
        if len(ids) < n * L:
            raise SystemExit(f"[hessian] repo sources tokenize to {len(ids)} ids, need {n} x {L}")
        stride = (len(ids) - L) // (n - 1) if n > 1 else 0  # >= L because len(ids) >= n * L
        starts = [w * stride for w in range(n)]
        for w, s in enumerate(starts):
            seqs.append(Seq(f"code/{w:02d}", "code", ids[s:s + L]))
        sources.append({"source": "code", "file_list_method": method, "files": len(files),
                        "excluded": list(KL_CORPUS_SOURCES) + [KL_CORPUS_DIR + "**"] +
                        [p + "**" for p in CODE_EXCLUDE_PREFIXES],
                        "sha256_of_concatenation": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "concatenation_tokens": len(ids), "tokens": n * L, "sequences": n,
                        "window_starts": starts, "file_list": files})
    if not seqs:
        raise SystemExit("[hessian] empty corpus")
    for s in seqs:
        if len(s.token_ids) < 2:
            raise SystemExit(f"[hessian] sequence {s.name} has {len(s.token_ids)} tokens; need >= 2")
    return seqs, sources


# --------------------------------------------------------------------------------------------
# Accumulation (torch; imported lazily)
# --------------------------------------------------------------------------------------------


class HessianAccum:
    """`sum over tokens of x x^T` in fp64 on device, plus the row count.

    Duck-typed to `imatrix_capture.ChannelAccum` (`.k`, `.rows`, `.update(x)`), which is all
    `imatrix_capture.MtpTaps`' hooks touch -- so the MTP head is captured by that class unchanged.
    Each update is one fp32 GEMM over <= 2048 rows (products of bf16 values are exact in fp32; only
    the in-GEMM sum rounds), added into the fp64 accumulator by a mixed-dtype in-place add (no fp64
    temporary)."""

    __slots__ = ("k", "acc", "rows")

    def __init__(self, k: int, device):
        import torch

        self.k = k
        self.acc = torch.zeros(k, k, dtype=torch.float64, device=device)
        self.rows = 0

    def update(self, x) -> None:
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.shape[1] != self.k:
            raise RuntimeError(f"input width changed: accumulator K={self.k}, got {flat.shape[1]}")
        xf = flat.float()
        self.acc.add_(xf.t() @ xf)
        self.rows += int(flat.shape[0])

    def finalize_fp32(self):
        """`acc / rows` as fp32 on device; frees the fp64 accumulator (divides in place first)."""
        import torch

        if self.rows == 0:
            raise RuntimeError("no rows accumulated")
        self.acc.div_(float(self.rows))
        h = self.acc.to(torch.float32)
        self.acc = None
        return h


class SharedInputGate:
    """Proves the shared-tap assumption: on the first layer of each type, every companion module
    (`in_proj_z`, `k_proj`, `v_proj`, `up_proj`) must be handed the same storage, with equal values,
    as its representative, on every sequence. Inputs are held only until `check()` runs after each
    layer call, then dropped."""

    def __init__(self):
        self.results: dict[str, dict] = {}
        self._seen: dict[str, object] = {}

    def install(self, layer, layer_idx: int, layer_type: str) -> list:
        handles = []
        for rep, comps in SHARED_GATE[layer_type]:
            for path in [rep] + comps:
                mod = layer
                for part in path.split("."):
                    mod = getattr(mod, part)

                def pre_hook(_m, inputs, _path=path):
                    self._seen[_path] = inputs[0]
                handles.append(mod.register_forward_pre_hook(pre_hook))
        self._layer, self._type = layer_idx, layer_type
        return handles

    def check(self) -> None:
        import torch

        for rep, comps in SHARED_GATE[self._type]:
            r = self._seen.get(rep)
            for c in comps:
                t = self._seen.get(c)
                same = (r is not None and t is not None and r.data_ptr() == t.data_ptr()
                        and tuple(r.shape) == tuple(t.shape) and r.stride() == t.stride())
                equal = bool(r is not None and t is not None and torch.equal(r, t))
                key = f"layer{self._layer}:{c}"
                prev = self.results.get(key)
                self.results[key] = {
                    "layer": self._layer, "layer_type": self._type, "representative": rep,
                    "companion": c,
                    "same_storage": same and (prev["same_storage"] if prev else True),
                    "elementwise_equal": equal and (prev["elementwise_equal"] if prev else True),
                    "calls": (prev["calls"] + 1) if prev else 1,
                }
        self._seen.clear()


# --------------------------------------------------------------------------------------------
# Capture (GPU)
# --------------------------------------------------------------------------------------------


@dataclass
class CaptureState:
    written: dict[str, dict] = field(default_factory=dict)   #: file -> write_hess_file stats
    seconds_per_file: dict[str, float] = field(default_factory=dict)
    layer_seconds: list[float] = field(default_factory=list)
    shared_gate: SharedInputGate = field(default_factory=SharedInputGate)


def _write_accum(out_dir: Path, plan: TapPlan, acc: HessianAccum, st: CaptureState) -> None:
    import torch

    t0 = time.perf_counter()
    rows = acc.rows
    h = acc.finalize_fp32()
    host = h.cpu().numpy()
    del h
    torch.cuda.empty_cache()
    stats = write_hess_file(out_dir / plan.file, host, rows)
    del host
    st.written[plan.file] = stats
    st.seconds_per_file[plan.file] = time.perf_counter() - t0
    warn = "" if stats["finite"] and stats["min_diag"] > 0 else "   <-- NOT FINITE / NON-POSITIVE DIAG"
    print(f"[hessian]   {plan.file:<22} K={stats['K']:>5} rows={rows:>7} "
          f"trace={stats['trace']:.6g} min_diag={stats['min_diag']:.3e} "
          f"{stats['bytes'] / 2**20:8.1f} MiB {st.seconds_per_file[plan.file]:5.1f}s{warn}",
          flush=True)


def run_capture(ref, seqs: list[Seq], plans: list[TapPlan], specs, n_run: int, out_dir: Path,
                hidden_device, want_draft_head: bool) -> CaptureState:
    import torch
    from imatrix_capture import CaptureResult, MtpTaps

    st = CaptureState()
    dev = ref.device
    layer_types = ref.layer_types
    gate_layers = {}
    for t in ("linear_attention", "full_attention"):
        if t in layer_types[:n_run]:
            gate_layers[layer_types.index(t)] = t
    by_layer: dict[int, list[TapPlan]] = {}
    for p in plans:
        if p.scope == "layer":
            by_layer.setdefault(p.layer, []).append(p)

    rope_cache: dict[int, tuple] = {}
    mask_cache: dict[int, object] = {}

    def rope(T):
        if T not in rope_cache:
            rope_cache[T] = ref._manual_rope(T)
        return rope_cache[T]

    def mask(T):
        if T not in mask_cache:
            mask_cache[T] = ref._manual_mask(T)
        return mask_cache[T]

    with torch.no_grad():
        t0 = time.perf_counter()
        hidden = [ref.embed(s.token_ids).to(hidden_device) for s in seqs]  # [1, T, hidden] each
        print(f"[hessian] embedded {len(seqs)} sequences in {time.perf_counter() - t0:.1f}s "
              f"(hidden states on {hidden_device})", flush=True)

        for i in range(n_run):
            t_layer = time.perf_counter()
            layer = ref.build_layer(i)
            accs: dict[str, HessianAccum] = {}
            handles = []
            for p in by_layer.get(i, []):
                acc = HessianAccum(p.k, dev)
                accs[p.file] = acc
                mod = layer
                for part in p.module.split("."):
                    mod = getattr(mod, part)
                handles.append(mod.register_forward_pre_hook(
                    lambda _m, inputs, _acc=acc: _acc.update(inputs[0])))
            gated = i in gate_layers
            if gated:
                handles += st.shared_gate.install(layer, i, gate_layers[i])
            attn_is_linear = layer_types[i] == "linear_attention"
            try:
                for j, h in enumerate(hidden):
                    x = h.to(dev)
                    T = x.shape[1]
                    out = layer(hidden_states=x, position_embeddings=rope(T),
                                attention_mask=None if attn_is_linear else mask(T),
                                position_ids=None, past_key_values=None)
                    if gated:
                        st.shared_gate.check()
                    hidden[j] = out.to(hidden_device)
                    del x, out
            finally:
                for hd in handles:
                    hd.remove()
            del layer
            t_fwd = time.perf_counter() - t_layer
            print(f"[hessian] layer {i:>2}/{n_run} ({layer_types[i]}): forward+accumulate "
                  f"{t_fwd:.1f}s, writing {len(accs)} tap(s)", flush=True)
            for p in by_layer.get(i, []):
                _write_accum(out_dir, p, accs.pop(p.file), st)
            st.layer_seconds.append(time.perf_counter() - t_layer)

        # lm_head + MTP: both consume the stack's output, one pass over the sequences.
        head_plans = [p for p in plans if p.scope in ("lm_head", "mtp")]
        if head_plans:
            t_head = time.perf_counter()
            result = CaptureResult()
            lm_acc = None
            mtp = None
            for p in head_plans:
                if p.scope == "lm_head":
                    lm_acc = HessianAccum(p.k, dev)
                else:
                    # One key per MTP file; MtpTaps' hooks find it by container key.
                    result.accums[p.keys[0]] = HessianAccum(p.k, dev)
            mtp_specs = [s for s in specs if s.key in result.accums]
            if mtp_specs:
                mtp = MtpTaps(ref, mtp_specs, result,
                              want_draft_head and "mtp.draft_head.lm_head" in result.accums)
            try:
                for j, s in enumerate(seqs):
                    pre = hidden[j].to(dev)  # [1, T, hidden], the PRE-final-norm residual
                    if lm_acc is not None:
                        lm_acc.update(ref.model.norm(pre))
                    if mtp is not None:
                        mtp.run(pre[0], s.token_ids)
                    del pre
            finally:
                if mtp is not None:
                    mtp.close()
            del mtp
            print(f"[hessian] lm_head/mtp pass: {time.perf_counter() - t_head:.1f}s", flush=True)
            for p in head_plans:
                acc = lm_acc if p.scope == "lm_head" else result.accums.pop(p.keys[0])
                if acc.rows == 0:
                    print(f"[hessian]   {p.file}: hook never fired -- not written")
                    continue
                _write_accum(out_dir, p, acc, st)
            lm_acc = None
    return st


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def evaluate_gates(audit: dict, all_keys: list[str], plans: list[TapPlan], st: CaptureState,
                   selected_keys: list[str], shared_expected: list[int]) -> dict:
    served = {k for p in plans if p.file in st.written for k in p.keys}
    missing_all = sorted(set(all_keys) - served)
    missing_sel = sorted(set(selected_keys) - served)
    # The accumulator itself raises if a hook's input width ever differs from the checkpoint's
    # in-features (HessianAccum.update); this re-checks what actually landed on disk.
    k_bad = []
    for p in plans:
        w = st.written.get(p.file)
        if w is not None and w["K"] != p.k:
            k_bad.append((p.file, w["K"], p.k))
    not_finite = sorted(f for f, w in st.written.items() if not w["finite"])
    nonpos = sorted(f for f, w in st.written.items() if not (w["min_diag"] > 0.0))
    min_diag = min((w["min_diag"] for w in st.written.values()), default=float("nan"))
    min_diag_file = min(st.written, key=lambda f: st.written[f]["min_diag"]) if st.written else None
    rows_short = sorted((f, w["rows"], ROWS_PER_K_MIN * w["K"]) for f, w in st.written.items()
                        if w["rows"] < ROWS_PER_K_MIN * w["K"])
    worst_ratio = min(((w["rows"] / w["K"], f) for f, w in st.written.items()), default=(0.0, None))
    sg = st.shared_gate.results
    shared_ok = bool(sg) and all(v["same_storage"] and v["elementwise_equal"] for v in sg.values())
    shared_layers = sorted({v["layer"] for v in sg.values()})
    shared_complete = shared_layers == sorted(shared_expected)
    gates = {
        "converter_audit_ok": bool(audit["ok"]),
        "keys_selected_ok": not missing_sel,
        "keys_missing_selected": missing_sel,
        "complete": not missing_all and shared_complete,
        "keys_missing_all": missing_all[:32] + (["..."] if len(missing_all) > 32 else []),
        "keys_missing_all_count": len(missing_all),
        "k_ok": not k_bad,
        "k_mismatches": k_bad,
        "finite_ok": not not_finite,
        "not_finite": not_finite,
        "min_diag_ok": not nonpos,
        "min_diag": min_diag,
        "min_diag_file": min_diag_file,
        "nonpositive_diag_files": nonpos,
        "rows_ok": not rows_short,
        "rows_per_k_min": ROWS_PER_K_MIN,
        "rows_short": rows_short,
        "worst_rows_over_k": {"ratio": worst_ratio[0], "file": worst_ratio[1]},
        "shared_input_ok": shared_ok,
        "shared_input_complete": shared_complete,
        "shared_input": sg,
    }
    gates["ok"] = all(gates[g] for g in ("converter_audit_ok", "keys_selected_ok", "k_ok",
                                         "finite_ok", "min_diag_ok", "rows_ok", "shared_input_ok"))
    return gates


def print_gates(g: dict) -> None:
    def pf(ok):
        return "PASS" if ok else "FAIL"
    print(f"\n[gate] converter audit: {pf(g['converter_audit_ok'])}")
    print(f"[gate] selected keys covered: {pf(g['keys_selected_ok'])} "
          f"(missing {g['keys_missing_selected'][:8]})")
    print(f"[gate] every converter linear covered: {g['complete']} "
          f"({g['keys_missing_all_count']} missing{': ' + str(g['keys_missing_all'][:6]) if g['keys_missing_all_count'] else ''})")
    print(f"[gate] K matches checkpoint in-features: {pf(g['k_ok'])} {g['k_mismatches'][:8]}")
    print(f"[gate] finite: {pf(g['finite_ok'])} {g['not_finite'][:8]}")
    print(f"[gate] min diag > 0: {pf(g['min_diag_ok'])} (min {g['min_diag']:.4e} in "
          f"{g['min_diag_file']})")
    print(f"[gate] rows >= {g['rows_per_k_min']} x K: {pf(g['rows_ok'])} (worst rows/K "
          f"{g['worst_rows_over_k']['ratio']:.2f} in {g['worst_rows_over_k']['file']})")
    if not g["rows_ok"]:
        print("[gate] ********************************************************************")
        print(f"[gate] WARNING: {len(g['rows_short'])} Hessian(s) have fewer than "
              f"{g['rows_per_k_min']} x K rows -- badly rank-deficient; LDLQ on them leans on "
              f"damping alone:")
        for f, r, need in g["rows_short"][:16]:
            print(f"[gate]   {f}: rows={r} < {need}")
        print("[gate] ********************************************************************")
    for key, v in sorted(g["shared_input"].items()):
        print(f"[gate] shared input {key} vs {v['representative']}: same storage "
              f"{v['same_storage']}, equal {v['elementwise_equal']} over {v['calls']} call(s)")
    print(f"[gate] shared-input check: {pf(g['shared_input_ok'])}"
          f"{'' if g['shared_input_complete'] else ' (NOT every layer type was visited)'}")
    print(f"[gate] overall: {pf(g['ok'])}{'' if g['complete'] else '   -- INCOMPLETE SET (smoke/pilot)'}")


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def _existing_anchor(p: Path) -> Path:
    p = p.resolve()
    while not p.exists():
        p = p.parent
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-fixture", type=Path, default=None, metavar="DIR",
                    help="CPU only, no checkpoint: write the tiny deterministic test set "
                         "(tests/convert/fixtures/hess_small) and exit")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="checkpoint (default: common.DEFAULT_MODEL_DIR)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="allow an --out-dir that already holds a hessian.json (it is deleted "
                         "before the capture starts; stale .hess files are overwritten)")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="kv_calib_corpus-style directory; 'none' to skip")
    ap.add_argument("--calib-txt", type=Path, default=DEFAULT_CALIB_TXT, help="'none' to skip")
    ap.add_argument("--wikitext", default=str(DEFAULT_WIKITEXT),
                    help="WikiText-2 raw train file; 'none' to skip")
    ap.add_argument("--wikitext-seqs", type=int, default=64)
    ap.add_argument("--code-seqs", type=int, default=16,
                    help="windows of this repo's own sources (0 to skip)")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="window length, and the truncation of every calib file")
    ap.add_argument("--no-mtp", action="store_true", help="skip the mtp.* head")
    ap.add_argument("--draft-head", action="store_true",
                    help="also capture mtp.draft_head.lm_head (only with --draft-vocab-ids containers)")
    ap.add_argument("--layers", type=int, default=None,
                    help="SMOKE: run only the first N text layers (lm_head/MTP then see layer N-1's "
                         "output, so their files get no key in the manifest; the set is marked "
                         "smoke and incomplete)")
    ap.add_argument("--keys", default=None, metavar="REGEX",
                    help="write only taps serving a container key that matches (re.search), e.g. "
                         "'mlp\\.' for the G3 pilot; the manifest then lists only matching keys")
    ap.add_argument("--hidden-device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="where the inter-layer hidden states live (auto: device if <= 6 GiB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print corpus sizes, the file list and the disk estimate; no GPU")
    args = ap.parse_args()

    if args.write_fixture is not None:
        return write_fixture(args.write_fixture)

    # The GPU rule, before anything expensive (common.resolve_device re-checks it). --dry-run never
    # touches the GPU, so it is exempt.
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if not args.dry_run and visible != "1":
        raise SystemExit(
            "[hessian] refusing to run: $env:HIP_VISIBLE_DEVICES must be exactly '1' (only HIP "
            f"device 1, the headless R9700, may be used). Got {visible!r}. (--dry-run needs no GPU.)"
        )

    sys.path.insert(0, str(TOOLS_REF))
    from common import DEFAULT_MODEL_DIR, ShardIndex, load_text_config, sha256_file
    from imatrix_capture import (CONVERTER_MAIN, audit_converter_source,
                                 enumerate_quantized_linears, tensor_shapes, verify_no_k_concat)

    model_dir = args.model_dir or DEFAULT_MODEL_DIR

    audit = audit_converter_source(CONVERTER_MAIN)
    print(f"[hessian] converter audit ({audit['path']}): "
          f"text={len(audit['found']['text'])} mtp={len(audit['found']['mtp'])} add_linear shapes, "
          f"direct PlanLinearLayouts={audit['direct_plan_linear_layouts']}")
    for kind in ("missing", "unexpected", "mismatched"):
        if audit[kind]:
            print(f"[hessian] converter audit {kind.upper()}: {audit[kind]}")
    if not audit["ok"]:
        raise SystemExit("[hessian] src/convert/main.cpp's add_linear set no longer matches "
                         "imatrix_capture.enumerate_quantized_linears() -- update it first")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    t0 = time.perf_counter()
    seqs, sources = build_corpus(args, tokenizer)
    total_tokens = sum(len(s.token_ids) for s in seqs)
    print(f"[hessian] corpus: {len(seqs)} sequences, {total_tokens} tokens "
          f"({time.perf_counter() - t0:.1f}s to tokenize)")
    for src in sources:
        label = src.get("name") or src.get("path") or src["source"]
        print(f"    {src['source']:<9} {src['sequences']:>3} seq {src['tokens']:>7} tok  {label}")

    _, text_config = load_text_config(model_dir)
    n_layers = int(text_config.num_hidden_layers)
    n_run = n_layers if args.layers is None else max(1, min(args.layers, n_layers))
    do_mtp = not args.no_mtp
    specs = enumerate_quantized_linears(text_config, do_mtp, args.draft_head)
    index = ShardIndex.load(model_dir)
    shapes = tensor_shapes(index, sorted({n for s in specs for n in s.hf_names}))
    expect_k = verify_no_k_concat(specs, shapes)
    all_keys = [s.key for s in specs]
    plans_all = build_tap_plan(specs, expect_k)

    key_re = re.compile(args.keys) if args.keys else None
    plans: list[TapPlan] = []
    for p in plans_all:
        if p.scope == "layer" and p.layer >= n_run:
            continue
        keys = [k for k in p.keys if key_re is None or key_re.search(k)]
        if keys:
            plans.append(TapPlan(p.file, p.scope, p.layer, p.module, p.k, keys, p.describe))
    selected_keys = [k for p in plans for k in p.keys]
    if not plans:
        raise SystemExit("[hessian] --keys/--layers select no tap at all")
    # MTP taps carry T-1 rows per sequence (MtpTaps pairs row i with token i+1), all others T.
    predicted_rows = {p.file: total_tokens - (len(seqs) if p.scope == "mtp" else 0) for p in plans}
    estimate = sum(hess_file_bytes(p.k) for p in plans) + (1 << 20)

    print(f"[hessian] {len(all_keys)} converter linears -> {len(plans_all)} taps; selected "
          f"{len(plans)} file(s) serving {len(selected_keys)} key(s), text layers 0..{n_run - 1}"
          f"{'' if key_re is None else f', --keys {args.keys!r}'}")
    short = [(p.file, predicted_rows[p.file], ROWS_PER_K_MIN * p.k) for p in plans
             if predicted_rows[p.file] < ROWS_PER_K_MIN * p.k]
    if args.dry_run:
        for p in plans:
            print(f"    {p.file:<22} K={p.k:>5} rows~{predicted_rows[p.file]:>7} "
                  f"{hess_file_bytes(p.k) / 2**20:8.1f} MiB  {','.join(p.keys)}")
    print(f"[hessian] disk estimate {estimate / 2**30:.2f} GiB for {len(plans)} file(s)")
    if short:
        print(f"[hessian] WARNING: {len(short)} tap(s) would get fewer than {ROWS_PER_K_MIN} x K "
              f"rows: {short[:6]}")
    anchor = _existing_anchor(args.out_dir)
    free = shutil.disk_usage(anchor).free
    print(f"[hessian] free space at {anchor}: {free / 2**30:.2f} GiB "
          f"(need {1.1 * estimate / 2**30:.2f} GiB = 1.1 x estimate)")
    if args.dry_run:
        print("[hessian] --dry-run: nothing captured, nothing written")
        return 0
    if free < 1.1 * estimate:
        raise SystemExit("[hessian] not enough free space on the target drive -- refusing to start")

    manifest_path = args.out_dir / MANIFEST_NAME
    if manifest_path.exists():
        if not args.force:
            raise SystemExit(f"[hessian] {manifest_path} already exists; pass --force to replace "
                             f"the set (the old manifest is deleted before capturing)")
        manifest_path.unlink()
    # A previous failed run's report describes .hess files this run is about to overwrite.
    failed_path = args.out_dir / FAILED_MANIFEST_NAME
    if failed_path.exists():
        failed_path.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from common import resolve_device
    from full_logits_golden import StreamingReference

    device = resolve_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hidden_bytes = total_tokens * int(text_config.hidden_size) * 2
    if args.hidden_device == "auto":
        hidden_device = device if hidden_bytes <= 6 * 2**30 else torch.device("cpu")
    else:
        hidden_device = device if args.hidden_device == "cuda" else torch.device("cpu")

    t0 = time.perf_counter()
    ref = StreamingReference(model_dir, device, dtype=torch.bfloat16)
    print(f"[hessian] streaming skeleton ready in {time.perf_counter() - t0:.1f}s: {ref.n_layers} "
          f"layers, hidden={ref.text_config.hidden_size}; hidden states "
          f"{hidden_bytes / 2**30:.2f} GiB on {hidden_device}", flush=True)
    if n_run < n_layers:
        print(f"[hessian] WARNING --layers {n_run}: SMOKE RUN -- lm_head/MTP taps see layer "
              f"{n_run - 1}'s output, not the stack's; this set is NOT a valid Hessian set")

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    st = run_capture(ref, seqs, plans, specs, n_run, args.out_dir, hidden_device,
                     args.draft_head)
    seconds = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"[hessian] capture done: {seconds:.1f}s, peak VRAM {peak:.3f} GiB allocated / "
          f"{reserved:.3f} GiB reserved")

    # The first layer of each type, over the WHOLE stack: a --layers smoke that stops before the
    # first attention layer reports the shared-input check as incomplete.
    shared_expected = [ref.layer_types.index(t) for t in SHARED_GATE if t in ref.layer_types]
    gates = evaluate_gates(audit, all_keys, plans, st, selected_keys, shared_expected)
    print_gates(gates)

    files = {f: w for f, w in st.written.items()}
    # A --layers smoke run still captures lm_head/MTP (the code path is exercised), but from layer
    # n_run-1's output: the wrong input distribution. Those files stay listed under "files"/"taps"
    # for inspection, but no key points at them, so `r4dx-convert --ldlq` cannot select them.
    smoke = n_run < n_layers
    keys = {k: p.file for p in plans if p.file in files and not (smoke and p.scope != "layer")
            for k in p.keys}
    extra = {
        "tool": "tools/reference/hessian_capture.py",
        "semantics": ("H = X^T X / rows, X = the linear's INPUT activation over every calibration "
                      "token (bf16 checkpoint forward), fp64 accumulation, stored fp32 packed "
                      "upper triangle. One file per distinct input ('tap'); 'keys' maps each "
                      "converter container base name to the tap file it reads."),
        "header_layout": "magic[8] 'R4DXHES1', u32 K, u32 flags (1 = packed upper), u64 rows, "
                         "f64 trace (of the stored fp32 diagonal), 32 reserved bytes; then "
                         "K*(K+1)/2 f32 H[i][j], i=0..K-1, j=i..K-1; little-endian",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "activation_dtype": "bfloat16",
        "gemm_dtype": "float32 (TF32 off)",
        "accumulation_dtype": "float64 (on device)",
        "n_layers": n_layers,
        "layers_run": n_run,
        "smoke": smoke,
        "keys_filter": args.keys,
        "mtp": do_mtp,
        "draft_head": bool(args.draft_head),
        "corpus": {"sequences": len(seqs), "tokens": total_tokens, "seq_len": args.seq_len,
                   "sources": sources},
        "converter_audit": {"path": audit["path"], "sha256": audit["sha256"], "ok": audit["ok"]},
        "taps": {p.file: {"scope": p.scope, "layer": p.layer, "module": p.module,
                          "tap": p.describe, "keys": p.keys}
                 for p in plans if p.file in files},
        "file_stats": {f: {"min_diag": w["min_diag"], "max_diag": w["max_diag"],
                           "finite": w["finite"], "bytes": w["bytes"],
                           "seconds": st.seconds_per_file.get(f)} for f, w in sorted(files.items())},
        "gates": gates,
        "device": str(device),
        "hidden_device": str(hidden_device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seconds": seconds,
        "peak_vram_gib": peak,
        "peak_vram_reserved_gib": reserved,
        "caveat": CAVEAT,
    }
    # A failed set is written as hessian.failed.json, which HessianStore never opens: the gate report
    # survives for diagnosis, but `--hessian-dir` on this directory is a "cannot open hessian.json"
    # error rather than a conversion against Hessians the gates just called invalid.
    path = write_manifest(args.out_dir, files, keys, extra,
                          manifest_name=MANIFEST_NAME if gates["ok"] else FAILED_MANIFEST_NAME)
    total_bytes = sum(w["bytes"] for w in files.values())
    print(f"\n[hessian] wrote {len(files)} .hess file(s), {total_bytes / 2**30:.2f} GiB, and {path}")
    print(f"[hessian] tokens={total_tokens} wall={seconds:.1f}s peak={peak:.3f} GiB")
    if smoke:
        print("[hessian] smoke run: lm_head/MTP files written but given no key in the manifest")
    if not gates["ok"]:
        print(f"[hessian] GATES FAILED -- manifest written as {FAILED_MANIFEST_NAME}, not "
              f"{MANIFEST_NAME}; the converter will not load this set")
    return 0 if gates["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
