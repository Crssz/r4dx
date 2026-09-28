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

**Weightless rms taps (`--rms-taps`, `--rms-only`; docs/quant2.md 3.2).** A rotated container
(`r4dx-convert --rotate q2a/q2ab`) runs every text layer's zero-centred norm WITHOUT its weight and
folds `(1 + w)` into the next in-projections, so their input becomes `rms(x) Q` with
`rms(x) = x * rsqrt(mean(x^2) + eps)` -- no weight. The converter needs `H_rms = E[rms(x)^T rms(x)]`,
which the post-norm taps above give only divided by `(1 + w)`: impossible at a dead channel
(`(1 + w) == 0`, e.g. layer 7 post_attention_layernorm channel 3994) and noisy where `|1 + w|` is
small. So a forward_pre_hook on each text layer's `input_layernorm` / `post_attention_layernorm`
reads the norm's INPUT (the bf16 residual) and accumulates `rms(x)` computed exactly as
`Qwen3_5RMSNorm._norm(x.float())` does -- fp32, the module's own eps, NO `(1 + w)` multiply and NO
bf16 cast (`rms_weightless`) -- into `L{i:02d}.in.rms.hess` / `L{i:02d}.mlp_in.rms.hess`
(`rms_file_name`: the post-norm file of the same activations with `.rms` before `.hess`, a pairing
the converter's HessianStore enforces). `hessian.json` `rms_keys` maps gdn.in_proj_qkv/z and
attn.qg/k/v to the first and mlp.gate_up to the second (gdn.in_proj_a/b read the same input but stay
bf16 and are never LDLQ'd). `--rms-taps` adds them to a full capture; `--rms-only` captures ONLY them
(still a full layer-major forward) and merges them into the validated set already in `--out-dir`,
after checking the checkpoint config and the corpus against what that set records (`--code-rev`
reads the repo-source part of the corpus from the commit the set was captured at). Their gates
(`evaluate_rms_gates`: finite; min diag > 0 with NO dead-channel exemption; rows >= 4K and equal to
the post-norm file's; `diag(H_post)_i` within 2% of `(1 + w_i)^2 diag(H_rms)_i` on every channel with
`(1 + w_i) != 0` and exactly 0 where it is 0, which proves both captures saw the same activations)
must all pass before anything rms goes into a manifest; otherwise nothing rms is written into it and
the report goes to `hessian_rms.failed.json`.

**Self-generated corpus (`--gen-file`, corpus source 4; docs/quant2.md "Corpus v2").** A
`samples.jsonl` written by the corpus generator (text our own quantized model produced through
r4dx-server; one JSON object per line: id "<category>/<NNN>", category, format "raw"|"chat",
enable_thinking, the FULL `messages` incl. every generated assistant turn and its
`reasoning_content`, `text` = the final assistant content stripped for "raw", rejected /
reject_reason, ...). `load_gen_samples` validates that contract and drops the rejected samples;
`render_gen_sample` turns each remaining one, in id order, into the text the model would read: a raw
document verbatim, a chat through the checkpoint's own chat template with `enable_thinking` as
recorded and `add_generation_prompt=False`; `build_gen_corpus` tokenizes each category's
concatenation (raw documents joined by a blank line, chats back to back -- each ends in
`<|im_end|>\n`) with no special tokens added and cuts it into non-overlapping `--seq-len` windows
(the short tail is dropped and reported; `--gen-max-seqs` keeps an evenly spread subset). What the
template does with thinking (checked on every render, `check_rendered_chat`): EVERY assistant turn
becomes `<|im_start|>assistant\n<think>\n{reasoning_content|trim}\n</think>\n\n{content|trim}<|im_end|>\n`
unless the caller sets `preserve_thinking=false` (the generator does not, and neither does this
script), in which case only the turns after the last user message keep the block -- the final
assistant turn of a sample always does. A turn without reasoning (thinking off) gets the EMPTY block
`<think>\n\n</think>\n\n`, which is exactly the generation prompt the template appends for
`enable_thinking=false`, so the rendered chat is what the model was served plus what it generated.
`enable_thinking=true` also puts the template's default (`xhigh`) reasoning-effort sentence into
the system turn.

**Tokenization (`--tokenizer`, docs/quant2.md 3.4).** Every source -- calib files, WikiText,
code, gen raw documents and rendered chats alike -- is tokenized by one `common.RefTokenizer`:
`canonical` (the default for a new capture) is the checkpoint's tokenizer.json as r4dx and
r4dx-server tokenize it; `hf-auto` is transformers' AutoTokenizer, which splits Thai combining
marks off their consonants and is kept only to reproduce hessian-v1. Chats are rendered by the
checkpoint's template either way and then encoded with the selected mode. The mode is recorded as
`corpus.tokenizer` (RefTokenizer.provenance); a manifest without it predates the switch and is
hf-auto, which is what `--rms-only` then uses (an explicit `--tokenizer` that disagrees with the
set is refused). English and code text tokenizes identically in both modes; the code windows still
move by a few tokens, because a few repo files carry Thai (their stride follows the token count).

**Disjointness with the held-out KL corpus, every source** (`kl_index` / `find_kl_overlaps`): a
"hit" is a `KL_SHINGLE_CHARS`-character (whitespace-normalized) run shared with a
`tools/reference/kl_corpus/*.txt` file outside that file's `#include` / Python import lines
(`KL_BOILERPLATE_LINE`: the cpp/python KL excerpts open with the standard include/import blocks any
code shares, which is idiom, not eval text). Every non-rejected gen sample is checked BEFORE
tokenizing -- its rendered text, plus for a raw sample the prompt and every other message field
(`gen_gate_texts`) -- and ANY hit stops the run after listing every offending sample. Every token
window of sources 1-3 is checked too, decoded back to its text (`gate_windows`), but a hit there is
printed and recorded, not refused: those windows are fixed by the design, and the repo code they
cut from quotes model.cpp / layer_golden.py in places (build_corpus says where and why). Each
source entry records the check and its hits as `kl_disjointness`. Lines that can carry generated or
KL text go through `say`, which escapes what the console cannot encode.

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
    <venv>\\Scripts\\python.exe tools\\reference\\hessian_capture.py --rms-only --code-rev <commit> `
        --out-dir D:\\models\\r4dx\\hessian-v1
    <venv>\\Scripts\\python.exe tools\\reference\\hessian_capture.py --out-dir D:\\models\\r4dx\\hessian-v2 `
        --rms-taps --wikitext-seqs 128 --code-seqs 48 --gen-file D:\\models\\r4dx\\corpus-v2\\samples.jsonl

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
import unicodedata
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

#: --gen-file (samples.jsonl): the categories and formats the corpus generator writes. Anything else
#: is a contract violation and stops the run (a typo would otherwise become a category of its own).
GEN_CATEGORIES = ("thai_prose", "english_prose", "chat", "code", "multilingual")
GEN_FORMATS = ("raw", "chat")
#: What follows a raw document in a category's concatenation when anything comes after it. A
#: rendered chat needs no separator: it ends in "<|im_end|>\n", so the next sample starts fresh.
GEN_RAW_SEPARATOR = "\n\n"
#: The disjointness gate (every corpus source): no calibration text may share this many consecutive
#: characters (after collapsing every whitespace run to one space) with any
#: tools/reference/kl_corpus/*.txt file, outside that file's boilerplate lines (next).
KL_SHINGLE_CHARS = 50
#: The lines of a KL file the gate does NOT compare against: an `#include <x>` / `#include "x"`
#: directive or a Python import statement (`import a.b as c, d`, `from a import b, c`,
#: `from a import (b, c)`, and every line of a `from a import (` block up to its `)`), each with an
#: optional trailing comment. The cpp/python KL excerpts are the HEADERS of model.cpp and
#: layer_golden.py, so they open with sorted standard includes/imports that any C++ or Python text
#: shares: `\n#include <algorithm>\n#include <chrono>\n#include <` and
#: `from __future__ import annotations\n\nimport argparse` are 50 normalized characters on their
#: own. Such a match is idiom, not eval text. Each run of lines between boilerplate lines is
#: shingled on its own, so no shingle spans one either. The grammar is strict (a prose line such as
#: "import tariffs on steel" is not an import statement); the per-file count of skipped lines is
#: recorded, and is 0 for the prose files.
_PY_NAMES = r"\w+(?:\s+as\s+\w+)?(?:\s*,\s*\w+(?:\s+as\s+\w+)?)*\s*,?"
KL_BOILERPLATE_LINE = re.compile(
    r"\s*(?:#\s*include\s*(?:<[^<>]*>|\"[^\"]*\")\s*(?://.*|/\*.*)?"
    r"|import\s+[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*\s*(?:#.*)?"
    rf"|from\s+[\w.]+\s+import\s+(?:\*|{_PY_NAMES}|\(\s*{_PY_NAMES}\s*\)|(?P<open>\()\s*(?:{_PY_NAMES})?)"
    r"\s*(?:#.*)?)")
#: A line inside a `from a import (` block: names, a comment, the closing `)`, or blank.
KL_IMPORT_CONTINUATION = re.compile(rf"\s*(?:{_PY_NAMES})?\s*(?P<close>\))?\s*(?:#.*)?")

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

#: How many non-positive-diagonal channel indices a file's stats carry. A file with more than this
#: many fails the diagonal gate outright: that is a broken capture, not a handful of dead channels.
MAX_REPORTED_CHANNELS = 64

#: The zero-centred norm that feeds each representative module. A channel whose (1 + w) is exactly
#: 0 is identically zero after the norm, so its Hessian row/column is exactly zero: a property of
#: the checkpoint (e.g. layer 7 post_attention_layernorm channel 3994), not a capture bug. LDLQ is
#: unaffected -- damping keeps the factorization defined and the weight column sees a constant-zero
#: input. The diagonal gate therefore allows exactly those channels and nothing else.
NORM_BEFORE = {
    "linear_attn.in_proj_qkv": "input_layernorm",
    "self_attn.q_proj": "input_layernorm",
    "mlp.gate_proj": "post_attention_layernorm",
}

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

#: --rms-taps / --rms-only: per text layer, (the zero-centred norm whose INPUT is hooked, the tap
#: name). The rms file is L{i:02d}.<tap>.rms.hess; the post-norm file of the same activations is
#: L{i:02d}.<tap>.hess (the table above: in_proj_qkv / q_proj read input_layernorm's output, gate_proj
#: reads post_attention_layernorm's).
RMS_NORMS = (("input_layernorm", "in"), ("post_attention_layernorm", "mlp_in"))
#: The container keys (after "text.layers.{i}.") an rms tap may serve: the LDLQ'd in-projections a
#: rotated container feeds from that norm. gdn.in_proj_a/b read the same input but stay bf16 in the
#: container and are never LDLQ'd, so they have no key at all.
RMS_KEY_SUFFIXES = {"in": ("gdn.in_proj_qkv", "gdn.in_proj_z", "attn.qg", "attn.k", "attn.v"),
                    "mlp_in": ("mlp.gate_up",)}
#: A post-norm tap file an rms tap can pair with.
RMS_POST_FILE_RE = re.compile(r"L(\d{2,})\.(in|mlp_in)\.hess")
#: Where a failed rms capture's report goes: the manifest is left exactly as it was.
RMS_FAILED_MANIFEST_NAME = "hessian_rms.failed.json"
#: The consistency gate's tolerance: diag(H_post)_i vs (1 + w_i)^2 diag(H_rms)_i. Both are means over
#: the same activations; the only difference is that the post-norm tap's input was rounded to bf16
#: (<= 2^-9 relative per element, so <= ~0.4% on a mean square). A different corpus, layer, norm or
#: checkpoint moves channels by far more.
RMS_CONSISTENCY_RTOL = 0.02
#: Corpus-source fields that say HOW a source was read, not what it contains: --rms-only ignores
#: them when comparing its corpus with the recorded one (--code-rev changes file_list_method only;
#: a gen source's kl_disjointness records what the gate checked against, which a later edit of
#: kl_corpus/ may change without changing a single calibration token).
CORPUS_HOW_FIELDS = ("file_list_method", "kl_disjointness")

CAVEAT = (
    "H = mean over calibration tokens of x x^T (x = the linear's INPUT activation, bf16 upcast to "
    "fp32, fp64 accumulation, stored fp32 packed upper triangle), measured on a REAL forward of "
    "the original bf16 checkpoint: every layer was fed the true output of all the real bf16 layers "
    "before it (full_logits_golden.StreamingReference's layers, driven layer-major). What this is "
    "NOT: it is not a sequential-GPTQ Hessian -- the preceding layers are the bf16 originals, not "
    "their quantized versions (docs/quant2.md 1.2) -- and it is only as representative as the "
    "corpus (calib.txt + kv_calib_corpus/ + WikiText-2 train + this repo's own sources + with "
    "--gen-file text self-generated by the quantized model), tokenized as corpus.tokenizer records "
    "(canonical: the checkpoint's tokenizer.json, as r4dx serves it; hf-auto or no record: "
    "transformers' AutoTokenizer, whose Thai token sequences the served model never sees). None of "
    "it is the held-out KL corpus "
    "tools/reference/kl_corpus/, and model.cpp and layer_golden.py, the sources of its code "
    "excerpts, are left out. A 50-character whitespace-normalized run shared with kl_corpus/ "
    "outside its #include/import lines refuses a generated sample (prompts included); in the "
    "windows of the other sources, which can hold repo code quoting those two files, every such "
    "run is listed under corpus.sources[*].kl_disjointness.overlaps."
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
             "min_diag": float(diag.min()), "max_diag": float(diag.max()), "bytes": size,
             "nonpos_channels": np.flatnonzero(diag <= 0.0)[:MAX_REPORTED_CHANNELS].tolist()}
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


def rms_file_name(post_file: str) -> str:
    """The weightless rms tap paired with a post-norm tap file: `L03.in.hess` -> `L03.in.rms.hess`.
    `hessian_store.hpp` refuses any other pairing in `rms_keys`."""
    if not post_file.endswith(".hess") or post_file.endswith(".rms.hess"):
        raise ValueError(f"rms_file_name: {post_file!r} is not a post-norm .hess file name")
    return post_file[:-len(".hess")] + ".rms.hess"


def write_manifest(out_dir: Path, files: dict[str, dict], keys: dict[str, str],
                   extra: dict, manifest_name: str = MANIFEST_NAME,
                   rms_keys: dict[str, str] | None = None) -> Path:
    """`hessian.json` (or `manifest_name`): the contract's fields first (format/version, files {K,
    rows, trace}, keys, and `rms_keys` when given), then whatever provenance the caller passes. LF
    line endings on every platform (its sha256 is recorded in converted containers), written via a
    temp file + rename. `rms_keys` is checked the way HessianStore will: every value a listed file
    that is not a "keys" file, for a base with a "keys" entry, named `rms_file_name(keys[base])`."""
    doc = {"format": MANIFEST_FORMAT, "version": MANIFEST_VERSION,
           "files": {name: {"K": int(v["K"]), "rows": int(v["rows"]), "trace": float(v["trace"])}
                     for name, v in sorted(files.items())},
           "keys": dict(sorted(keys.items()))}
    for name in keys.values():
        if name not in doc["files"]:
            raise RuntimeError(f"manifest key points at {name!r}, which is not in files")
    if "rms_keys" in extra:
        raise RuntimeError("manifest extra field 'rms_keys': pass it as rms_keys= so it is checked")
    if rms_keys is not None:
        key_files = set(keys.values())
        for base, name in rms_keys.items():
            if name not in doc["files"] or name in key_files or base not in keys or \
                    name != rms_file_name(keys[base]):
                raise RuntimeError(f"rms_keys[{base!r}] = {name!r} is not the listed rms file "
                                   f"paired with keys[{base!r}] = {keys.get(base)!r}")
        doc["rms_keys"] = dict(sorted(rms_keys.items()))
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


@dataclass
class RmsTapPlan:
    file: str                   #: e.g. "L03.in.rms.hess"
    layer: int                  #: text layer index
    norm: str                   #: "input_layernorm" | "post_attention_layernorm" (its INPUT is hooked)
    post_file: str              #: the post-norm tap of the same activations, e.g. "L03.in.hess"
    k: int                      #: hidden size
    keys: list[str] = field(default_factory=list)


def build_rms_plans(post_keys: dict[str, list[str]], n_run: int, hidden: int) -> list[RmsTapPlan]:
    """One weightless rms tap per text layer < n_run and norm (RMS_NORMS) whose post-norm tap file is
    in `post_keys` (file -> the container keys reading it); those keys become its `rms_keys`. Refuses
    a key the rotated container does not feed from that norm (RMS_KEY_SUFFIXES)."""
    plans = []
    for i in range(n_run):
        for norm, tap in RMS_NORMS:
            post = f"L{i:02d}.{tap}.hess"
            keys = post_keys.get(post)
            if not keys:
                continue
            want = {f"text.layers.{i}.{s}" for s in RMS_KEY_SUFFIXES[tap]}
            bad = sorted(set(keys) - want)
            if bad:
                raise SystemExit(f"[hessian] {post} serves {bad}: not an in-projection fed by layer "
                                 f"{i}'s {norm} (expected a subset of {sorted(want)})")
            plans.append(RmsTapPlan(rms_file_name(post), i, norm, post, hidden, sorted(keys)))
    return plans


# --------------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------------


@dataclass
class Seq:
    name: str
    source: str
    token_ids: list[int]


def check_ref_tokenizer(tokenizer) -> str:
    """The corpus is tokenized ONLY through a `common.RefTokenizer` (`.mode`, `.encode`, `.decode`,
    `.render_chat`, `.chat_template`) -- never a bare transformers tokenizer, whose own `.encode`
    and `.decode` exist and would silently tokenize with AutoTokenizer's pre-tokenizer (hf-auto)
    whatever the caller meant. Returns the mode."""
    mode = getattr(tokenizer, "mode", None)
    if not isinstance(mode, str) or not callable(getattr(tokenizer, "render_chat", None)):
        raise SystemExit(f"[hessian] the corpus tokenizer must be a common.RefTokenizer "
                         f"(common.load_ref_tokenizer(model_dir, mode)), not "
                         f"{type(tokenizer).__name__}")
    return mode


def _tokenize_prefix(tokenizer, text: str, need: int, margin: int = 256) -> tuple[list[int], int]:
    """The first `need` tokens of `text` without tokenizing all of it. Tokenizes a growing prefix
    until it yields `need + margin` tokens: BPE pre-tokenization is local, so only the last few
    tokens before the cut can differ from a whole-text tokenization, and the margin is discarded.
    Returns (the first `need` ids -- fewer if the text runs out, chars tokenized)."""
    chars = max(4096, need * 8)
    while True:
        chunk = text[:chars]
        ids = tokenizer.encode(chunk)
        if len(ids) >= need + margin or chars >= len(text):
            return list(ids)[:need], len(chunk)
        chars *= 2


def _git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], input=stdin, check=True,
                          capture_output=True).stdout


def repo_code_files(rev: str | None = None) -> tuple[list[str], str, str | None]:
    """This repo's own C++/HIP/Python sources, repo-relative posix paths, sorted, plus how they were
    listed and (with `rev`) the resolved commit. Tracked files only (`git ls-files`) so build trees
    and scratch files never leak in; falls back to walking the tree (skipping build*/, .git/,
    dot-dirs) when git is unavailable. With `rev` (--code-rev) the files are those of that commit's
    tree instead, and git is required."""
    commit = None
    if rev is not None:
        try:
            commit = _git("rev-parse", "--verify", "--quiet", rev + "^{commit}").decode().strip()
            out = _git("ls-tree", "-r", "-z", "--name-only", commit).decode("utf-8")
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(f"[hessian] --code-rev {rev!r}: not a commit of {REPO_ROOT} ({e})")
        method = f"git ls-tree {commit} (--code-rev {rev})"
        paths = [p for p in out.split("\0") if p]
    else:
        method = "git ls-files"
        try:
            out = _git("ls-files", "-z").decode("utf-8")
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
    return sorted(keep), method, commit


def git_blobs(commit: str, paths: list[str]) -> dict[str, bytes]:
    """`<commit>:<path>` for every path, from one `git cat-file --batch`."""
    try:
        out = _git("cat-file", "--batch",
                   stdin="".join(f"{commit}:{p}\n" for p in paths).encode("utf-8"))
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"[hessian] git cat-file --batch at {commit} failed: {e}")
    blobs: dict[str, bytes] = {}
    pos = 0
    for p in paths:
        nl = out.index(b"\n", pos)
        head = out[pos:nl].decode("utf-8", errors="replace").split()
        if len(head) != 3 or head[1] != "blob":
            raise SystemExit(f"[hessian] {commit}:{p} is not a blob ({' '.join(head)})")
        size = int(head[2])
        blobs[p] = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1  # the content is followed by one LF
    return blobs


# --------------------------------------------------------------------------------------------
# Corpus source 4: self-generated samples (--gen-file, docs/quant2.md "Corpus v2")
# --------------------------------------------------------------------------------------------


def load_gen_samples(path: Path) -> tuple[list[dict], list[dict], str]:
    """Read a samples.jsonl and hold it to the generator's interface contract. Returns (the samples
    with rejected == false, sorted by id; the rejected ones, sorted by id; the file's sha256).

    Refused on any line (the line number is named): not a JSON object; an id that is not a unique
    "<category>/<rest>" string whose category is the sample's `category` and one of GEN_CATEGORIES;
    a format not in GEN_FORMATS; a non-boolean `rejected` or `enable_thinking`. Refused on a
    non-rejected sample only (a rejected one is counted, never rendered): `messages` that is not a
    list of {role: system|user|assistant, content: str} objects with a user turn and ending in an
    assistant turn; a raw sample whose `text` is empty or not the final assistant content stripped;
    a chat sample with a non-null `text`; a thinking chat whose FINAL assistant turn has no
    `reasoning_content` string (its thought would render as an empty block), or a non-thinking one
    with a non-empty `reasoning_content` on any turn (it would render a thought the model never had
    in context). An earlier assistant turn of a thinking chat may lack `reasoning_content`: it then
    renders with an empty block, which is exactly what the model was served if the generator did
    not replay that thought."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    used: list[dict] = []
    rejected: list[dict] = []
    seen: dict[str, int] = {}
    bad: list[str] = []
    # Split on "\n" only, NOT str.splitlines(): that also breaks at U+2028/U+2029, U+0085, \v, \f
    # and \x1c-\x1e, which json.dumps(ensure_ascii=False) leaves unescaped inside a string -- a
    # generated sample containing one would be cut in half. (json.loads ignores a trailing "\r".)
    for lineno, line in enumerate(raw.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            s = json.loads(line)
        except json.JSONDecodeError as e:
            bad.append(f"{where}: not JSON ({e})")
            continue
        if not isinstance(s, dict):
            bad.append(f"{where}: not a JSON object")
            continue
        sid, cat, fmt = s.get("id"), s.get("category"), s.get("format")
        if not isinstance(sid, str) or "/" not in sid or sid.split("/", 1)[0] != cat:
            bad.append(f"{where}: id {sid!r} is not '<category>/<NNN>' for category {cat!r}")
            continue
        where = f"{where} ({sid})"
        if sid in seen:
            bad.append(f"{where}: duplicate id (first on line {seen[sid]})")
            continue
        seen[sid] = lineno
        if cat not in GEN_CATEGORIES:
            bad.append(f"{where}: category {cat!r} is not one of {list(GEN_CATEGORIES)}")
        elif fmt not in GEN_FORMATS:
            bad.append(f"{where}: format {fmt!r} is not one of {list(GEN_FORMATS)}")
        elif not isinstance(s.get("rejected"), bool):
            bad.append(f"{where}: rejected {s.get('rejected')!r} is not a boolean")
        elif not isinstance(s.get("enable_thinking"), bool):
            bad.append(f"{where}: enable_thinking {s.get('enable_thinking')!r} is not a boolean")
        elif s["rejected"]:
            rejected.append(s)
        else:
            why = _gen_sample_problem(s)
            if why:
                bad.append(f"{where}: {why}")
            else:
                used.append(s)
    if bad:
        for b in bad[:16]:
            say(f"[hessian]   {b}")
        raise SystemExit(f"[hessian] --gen-file {path}: {len(bad)} line(s) break the samples.jsonl "
                         f"contract (docs/quant2.md 'Corpus v2'); first: {bad[0]}")
    if not used:
        raise SystemExit(f"[hessian] --gen-file {path}: no sample with rejected == false "
                         f"({len(rejected)} rejected)")
    return sorted(used, key=lambda s: s["id"]), sorted(rejected, key=lambda s: s["id"]), sha


def _gen_sample_problem(s: dict) -> str | None:
    """What makes a non-rejected sample unusable (load_gen_samples' per-sample rules), or None."""
    msgs = s.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(
            isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and
            isinstance(m.get("content"), str) for m in msgs):
        return "messages is not a non-empty list of {role: system|user|assistant, content: str}"
    if msgs[-1]["role"] != "assistant":
        return f"the last message is a {msgs[-1]['role']!r} turn, not the generated assistant turn"
    if not any(m["role"] == "user" for m in msgs):
        return "no user turn (the chat template refuses a conversation without one)"
    for m in msgs:
        rc = m.get("reasoning_content")
        if rc is not None and not isinstance(rc, str):
            return f"reasoning_content {type(rc).__name__} is not a string"
    final = msgs[-1]
    if s["format"] == "raw":
        text = s.get("text")
        if not isinstance(text, str) or not text:
            return "a raw sample needs a non-empty text"
        if text != final["content"].strip():
            return "text is not the final assistant content stripped of surrounding whitespace"
        return None
    if s.get("text") is not None:
        return "a chat sample's text must be null (the messages are what is rendered)"
    if s["enable_thinking"] and not isinstance(final.get("reasoning_content"), str):
        return "enable_thinking is true but the final assistant turn has no reasoning_content"
    if not s["enable_thinking"] and any(m.get("reasoning_content") for m in msgs):
        return "enable_thinking is false but a turn carries a non-empty reasoning_content"
    return None


def assistant_turn_rendering(message: dict) -> str:
    """How the checkpoint's chat_template.jinja writes an assistant turn with preserve_thinking
    unset: `<|im_start|>assistant\\n<think>\\n` + reasoning_content|trim + `\\n</think>\\n\\n` +
    content|trim + `<|im_end|>\\n` (jinja's trim is str.strip; a missing/non-string
    reasoning_content renders as the empty block `<think>\\n\\n</think>\\n\\n`)."""
    rc = message.get("reasoning_content")
    return ("<|im_start|>assistant\n<think>\n" + (rc if isinstance(rc, str) else "").strip() +
            "\n</think>\n\n" + message["content"].strip() + "<|im_end|>\n")


def check_rendered_chat(sample: dict, text: str) -> None:
    """Every assistant turn of a rendered chat must appear in order in the form
    `assistant_turn_rendering` gives, and the render must END with the final one: a template (or a
    transformers version) that dropped the final turn's thought, rendered a thinking-off turn
    without its empty block, or appended anything after the model's own turn stops the run instead
    of quietly calibrating on text the model was never served."""
    turns = [m for m in sample["messages"] if m["role"] == "assistant"]
    pos = 0
    for k, m in enumerate(turns):
        block = assistant_turn_rendering(m)
        at = text.find(block, pos)
        last = k == len(turns) - 1
        if at < 0 or (last and at + len(block) != len(text)):
            raise SystemExit(
                f"[hessian] --gen-file sample {sample['id']}: the chat template did not render "
                f"assistant turn {k + 1}/{len(turns)} as <think>\\n<reasoning>\\n</think>\\n\\n<content>"
                f"<|im_end|>{' at the end of the conversation' if last and at >= 0 else ''} -- a "
                f"different chat_template.jinja than the one this script was checked against?")
        pos = at + len(block)


def render_gen_sample(sample: dict, tokenizer, chat_template: str | None) -> str:
    """The text one non-rejected sample contributes. Raw: its `text` verbatim. Chat: its full
    `messages` through `chat_template` (the checkpoint's own) with `enable_thinking` as recorded and
    `add_generation_prompt=False` -- the conversation already ends with the model's turn -- then
    checked by `check_rendered_chat`. No other template variable is set: `reasoning_effort` stays
    the template's default (xhigh) and `preserve_thinking` unset, as for a request that sends
    neither."""
    if sample["format"] == "raw":
        return sample["text"]
    if not isinstance(chat_template, str):
        raise SystemExit(f"[hessian] --gen-file sample {sample['id']} is a chat but the tokenizer "
                         f"has no chat template")
    try:
        text = tokenizer.render_chat(sample["messages"], chat_template=chat_template,
                                     add_generation_prompt=False,
                                     enable_thinking=bool(sample["enable_thinking"]))
    except TypeError as e:  # RefTokenizer.render_chat: the template did not return a string
        raise SystemExit(f"[hessian] --gen-file sample {sample['id']}: {e}")
    check_rendered_chat(sample, text)
    return text


def ws_normalize(text: str) -> str:
    """The disjointness gate's normalization: every whitespace run (str.split's Unicode notion,
    CR/LF, tabs and NBSP included) becomes one space, and leading/trailing whitespace goes."""
    return " ".join(text.split())


def kl_gate_segments(text: str) -> tuple[list[tuple[str, list[int]]], int]:
    """How the gate reads one kl_corpus file: (the runs of consecutive lines that are not
    boilerplate (KL_BOILERPLATE_LINE and the continuation lines of a `from a import (` block), each
    whitespace-normalized exactly as ws_normalize normalizes the whole run -- the non-empty
    normalized lines joined by one space -- with the 1-based file line of every one of its
    characters; the number of boilerplate lines skipped). A file without boilerplate is ONE segment,
    equal to ws_normalize(text)."""
    segs: list[tuple[str, list[int]]] = []
    pieces: list[str] = []
    line_of: list[int] = []
    skipped = 0
    in_import = False

    def close() -> None:
        if pieces:
            segs.append((" ".join(pieces), line_of.copy()))
            pieces.clear()
            line_of.clear()

    for no, line in enumerate(text.split("\n"), 1):
        if in_import:
            m = KL_IMPORT_CONTINUATION.fullmatch(line)
            if m:
                skipped += 1
                in_import = m.group("close") is None
                continue
            in_import = False  # not an import line after all: an unclosed `(`, read on as text
        m = KL_BOILERPLATE_LINE.fullmatch(line)
        if m:
            close()
            skipped += 1
            in_import = m.group("open") is not None
            continue
        piece = ws_normalize(line)
        if piece:
            if pieces:
                line_of.append(no)  # the joining space
            pieces.append(piece)
            line_of.extend([no] * len(piece))
    close()
    return segs, skipped


@dataclass
class KlIndex:
    """The held-out KL corpus as the disjointness gate sees it (`kl_index`)."""
    shingles: dict[str, tuple[str, int]]  #: n-char normalized run -> (file name, 1-based line)
    shas: dict[str, str]                  #: file name -> sha256 of its bytes
    boilerplate_lines: dict[str, int]     #: file name -> lines skipped (KL_BOILERPLATE_LINE)
    label: str                            #: the directory, repo-relative when inside the repo

    def record(self, checked: str, overlaps: list[dict] | None = None) -> dict:
        """A source entry's `kl_disjointness` (a CORPUS_HOW_FIELDS field): what was compared against
        what, the overlaps found (find_kl_overlaps hits; a gen source never has any -- its gate
        refuses the run instead) and whether there were none."""
        return {"shingle_chars": KL_SHINGLE_CHARS,
                "normalization": "every whitespace run -> one space (str.split), then trimmed",
                "kl_dir": self.label, "kl_files": dict(self.shas),
                "kl_boilerplate": ("#include and Python import lines (trailing comment allowed) are "
                                   "not compared; no shingle spans one"),
                "kl_boilerplate_lines": dict(self.boilerplate_lines), "shingles": len(self.shingles),
                "checked": checked, "overlaps": list(overlaps or []), "ok": not overlaps}


def kl_index(kl_dir: Path | None = None, n: int = KL_SHINGLE_CHARS) -> KlIndex:
    """Every n-character substring of every `kl_gate_segments` segment of every `kl_dir/*.txt`
    (default: the repo's tools/reference/kl_corpus/) -> (file name, 1-based line where it starts):
    ~34k entries for the KL corpus. Refuses a directory without .txt files: the gate would pass
    vacuously."""
    kl_dir = kl_dir if kl_dir is not None else REPO_ROOT / KL_CORPUS_DIR
    files = sorted(kl_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"[hessian] disjointness gate: no *.txt under {kl_dir} to compare against")
    shingles: dict[str, tuple[str, int]] = {}
    shas: dict[str, str] = {}
    skipped: dict[str, int] = {}
    for p in files:
        raw = p.read_bytes()
        shas[p.name] = hashlib.sha256(raw).hexdigest()
        segs, skipped[p.name] = kl_gate_segments(raw.decode("utf-8"))
        for t, line_of in segs:
            for i in range(len(t) - n + 1):
                shingles.setdefault(t[i:i + n], (p.name, line_of[i]))
    try:
        label = kl_dir.resolve().relative_to(REPO_ROOT).as_posix() + "/"
    except ValueError:
        label = str(kl_dir)
    return KlIndex(shingles, shas, skipped, label)


def find_kl_overlaps(texts: list[tuple[str, str]], shingles: dict[str, tuple[str, int]],
                     n: int = KL_SHINGLE_CHARS) -> list[dict]:
    """For each (id, text) that shares an n-character whitespace-normalized substring with the KL
    corpus, its FIRST such position: {id, offset (in the normalized text), substring, kl_file,
    kl_line}. One hash lookup per character position -- ~1.5M for ~400k tokens of generated text,
    about a second."""
    hits = []
    for sid, text in texts:
        t = ws_normalize(text)
        for i in range(len(t) - n + 1):
            where = shingles.get(t[i:i + n])
            if where is not None:
                hits.append({"id": sid, "offset": i, "substring": t[i:i + n],
                             "kl_file": where[0], "kl_line": where[1]})
                break
    return hits


def say(msg: str) -> None:
    """print() for a line that may carry generated or KL text (Thai): a stream that cannot encode it
    -- on Windows a pipe or a file gets the ANSI code page (cp1252) with 'strict' errors, and ctest
    always pipes -- gets `\\uXXXX` escapes instead of a UnicodeEncodeError that would bury the
    gate's refusal under a traceback. main() reconfigures sys.stdout the same way; this also
    covers code that imports the module (the tests)."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(msg.encode(enc, errors="backslashreplace").decode(enc, errors="replace"), flush=True)


def report_kl_hits(hits: list[dict]) -> None:
    """One line per hit, EVERY hit: the ids are what the user acts on."""
    for h in hits:
        say(f"[hessian]   {h['id']}{' ' + h['where'] if h.get('where') else ''} (normalized char "
            f"{h['offset']}) shares {h['substring']!r} with {h['kl_file']} line {h['kl_line']}")


def gen_gate_texts(sample: dict, rendered: str) -> list[tuple[str, str]]:
    """What the gate reads of one non-rejected sample, as (where, text): the rendered text, and for
    a RAW sample also every message's `content` and `reasoning_content` except the final content
    (that IS the rendered text) -- the prompt is not calibrated on there, but it steered the text
    (a prompt quoting KL text makes the model write on in the eval's domain). A rendered chat
    already holds every message."""
    raw = sample["format"] == "raw"
    texts = [("text" if raw else "rendered chat", rendered)]
    if raw:
        last = len(sample["messages"]) - 1
        for k, m in enumerate(sample["messages"]):
            for fld in ("content", "reasoning_content"):
                if (k, fld) != (last, "content") and isinstance(m.get(fld), str) and m[fld].strip():
                    texts.append((f"messages[{k}].{fld}", m[fld]))
    return texts


def gate_windows(seqs: list[Seq], tokenizer, kl: KlIndex) -> list[dict]:
    """The disjointness check over token windows (corpus sources 1-3; build_corpus records its hits,
    it does not refuse them): each sequence's ids decoded back to text exactly (special tokens
    kept, no clean-up: RefTokenizer.decode) and scanned -- the text the Hessians are actually
    calibrated on. One hit per window at most, `id` the sequence name."""
    return find_kl_overlaps([(s.name, tokenizer.decode(s.token_ids)) for s in seqs], kl.shingles)


def gen_window_starts(n_tokens: int, seq_len: int, max_seqs: int | None) -> list[int]:
    """Token offsets of the windows cut from one category's `n_tokens`-token concatenation: all
    `n_tokens // seq_len` non-overlapping windows 0, L, 2L, ... (the short tail is dropped). With
    `max_seqs` below that, `max_seqs` of them spread evenly over the whole concatenation, first and
    last included like the code windows -- a subset of the uncapped windows, so the cap only ever
    removes windows, never moves one."""
    n_all = n_tokens // seq_len
    if max_seqs is None or max_seqs >= n_all:
        picked = list(range(n_all))
    elif max_seqs == 1:
        picked = [0]
    else:  # consecutive picks differ by >= floor((n_all - 1) / (max_seqs - 1)) >= 1: distinct
        picked = [w * (n_all - 1) // (max_seqs - 1) for w in range(max_seqs)]
    return [w * seq_len for w in picked]


def build_gen_corpus(path: Path, tokenizer, seq_len: int, max_seqs: int | None = None,
                     kl: KlIndex | None = None) -> tuple[list[Seq], list[dict]]:
    """Corpus source 4 (`--gen-file`): the non-rejected samples of `path` (load_gen_samples), each
    rendered (render_gen_sample), gated against the held-out KL corpus (`kl`, default kl_index())
    BEFORE any tokenizing (gen_gate_texts + find_kl_overlaps; a SystemExit after every overlapping
    sample has been listed), then per category (sorted), the
    rendered samples in id order are concatenated -- a raw document is followed by
    GEN_RAW_SEPARATOR when anything comes after it, a chat by nothing (it ends in `<|im_end|>\\n`) --
    tokenized as one text by `tokenizer.encode` (a common.RefTokenizer: NO special tokens added,
    the template's control tokens in the text mapped to their ids; the checkpoint adds no BOS), and
    cut into the `gen_window_starts` windows. Returns the sequences (`gen/<category>/<w>`) and one
    source entry per category present in the file (also a category whose every sample was
    rejected)."""
    check_ref_tokenizer(tokenizer)
    if max_seqs is not None and max_seqs < 1:
        raise SystemExit(f"[hessian] --gen-max-seqs {max_seqs}: must be >= 1 (omit it for no cap)")
    if not path.is_file():
        raise SystemExit(f"[hessian] --gen-file {path} does not exist")
    used, rejected, file_sha = load_gen_samples(path)
    template = getattr(tokenizer, "chat_template", None)
    rendered = [(s, render_gen_sample(s, tokenizer, template)) for s in used]

    kl = kl if kl is not None else kl_index()
    t0 = time.perf_counter()
    hits: list[dict] = []
    chars = 0
    for s, t in rendered:
        for where, text in gen_gate_texts(s, t):
            chars += len(text)
            h = find_kl_overlaps([(s["id"], text)], kl.shingles)
            if h:
                hits.append(dict(h[0], where=where))
                break
    gate_s = time.perf_counter() - t0
    if hits:
        report_kl_hits(hits)
        raise SystemExit(
            f"[hessian] --gen-file {path}: {len(hits)} sample(s), every one listed above, share a "
            f"{KL_SHINGLE_CHARS}-character whitespace-normalized run with the held-out KL corpus "
            f"{kl.label} (first: {hits[0]['id']} vs {hits[0]['kl_file']}). The Hessians must not be "
            f"calibrated on the text KL is measured on: mark those samples rejected (or regenerate "
            f"them) and rerun.")
    say(f"[hessian] gen disjointness gate: PASS -- {len(rendered)} sample(s), {chars} chars "
        f"(prompts included), no {KL_SHINGLE_CHARS}-char overlap with {len(kl.shas)} kl_corpus "
        f"file(s) ({len(kl.shingles)} shingles, {gate_s:.2f}s)")
    disjoint = kl.record("every non-rejected sample: its rendered text, and for a raw sample every "
                         "other message field (the prompt) too")
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest() \
        if isinstance(template, str) else None

    seqs: list[Seq] = []
    sources: list[dict] = []
    for cat in sorted({s["category"] for s in used} | {s["category"] for s in rejected}):
        items = [(s, t) for s, t in rendered if s["category"] == cat]
        rej = [s for s in rejected if s["category"] == cat]
        parts = []
        for k, (s, t) in enumerate(items):
            if k and items[k - 1][0]["format"] == "raw":
                parts.append(GEN_RAW_SEPARATOR)
            parts.append(t)
        text = "".join(parts)
        ids = tokenizer.encode(text) if text else []
        starts = gen_window_starts(len(ids), seq_len, max_seqs)
        for w, s0 in enumerate(starts):
            seqs.append(Seq(f"gen/{cat}/{w:03d}", "gen", ids[s0:s0 + seq_len]))
        formats = {f: sum(1 for s, _ in items if s["format"] == f) for f in GEN_FORMATS}
        # Counted by the "<kind>" of the generator's "<kind>: <detail>" reject_reason, the keys of
        # gen_manifest.json's reject_reasons: the detail is per sample (a length, a turn, and for a
        # "kl overlap" the matched run of KL text, which has no place in hessian.json). The full
        # reasons stay in the jsonl, whose sha256 is recorded.
        reasons: dict[str, int] = {}
        for s in rej:
            r = s.get("reject_reason")
            k = r.split(":", 1)[0].strip() if isinstance(r, str) and r.strip() else "unspecified"
            reasons[k] = reasons.get(k, 0) + 1
        finish: dict[str, int] = {}
        for s, _ in items:
            f = str(s.get("finish_reason"))
            finish[f] = finish.get(f, 0) + 1
        entry = {"source": "gen", "name": cat, "path": str(path), "sha256": file_sha,
                 "samples": len(items), "samples_rejected": len(rej),
                 "reject_reasons": dict(sorted(reasons.items())), "formats": formats,
                 "thinking_samples": sum(1 for s, _ in items
                                         if s["format"] == "chat" and s["enable_thinking"]),
                 "finish_reasons": dict(sorted(finish.items())),
                 "sample_ids": [s["id"] for s, _ in items]}
        if formats["chat"]:
            entry["chat_template_sha256"] = template_sha
        entry.update({
            "packing": ("rendered samples in id order; a raw document is followed by a blank line "
                        "when anything comes after it, a chat (ending in <|im_end|>\\n) by nothing; "
                        "tokenized as one text, add_special_tokens=False; non-overlapping windows"),
            "sha256_of_concatenation": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "concatenation_chars": len(text), "concatenation_tokens": len(ids),
            "tokens": len(starts) * seq_len, "sequences": len(starts),
            "windows_available": len(ids) // seq_len, "max_seqs": max_seqs,
            "window_starts": starts, "dropped_tail_tokens": len(ids) % seq_len,
            "kl_disjointness": disjoint})
        sources.append(entry)
        if not starts:
            print(f"[hessian] WARNING gen/{cat}: {len(ids)} token(s) from {len(items)} sample(s) "
                  f"make no full {seq_len}-token window -- this category contributes nothing")
    return seqs, sources


def build_corpus(args, tokenizer) -> tuple[list[Seq], list[dict]]:
    """Every corpus source, each tokenized by `tokenizer` (a common.RefTokenizer; the caller records
    its mode as the manifest's corpus.tokenizer, see corpus_record)."""
    from common import sha256_file
    from kv_calibrate_full import collect_corpus, repo_relative

    check_ref_tokenizer(tokenizer)
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
    code_parts: list[tuple[str, str]] = []  # (path, text) -- names the file behind a gate hit
    if args.code_seqs > 0:
        # --code-rev: that commit's tree, not the working tree -- how --rms-only reproduces the code
        # windows of a set captured at an older checkout (its sha256_of_concatenation must match).
        files, method, commit = repo_code_files(getattr(args, "code_rev", None))
        blobs = git_blobs(commit, files) if commit is not None else None
        parts = []
        nfc_files = []
        for rel in files:
            raw = blobs[rel] if blobs is not None else (REPO_ROOT / rel).read_bytes()
            # CRLF -> LF: with core.autocrlf=true the checkout's line endings are a property of the
            # machine, not of the source, and must not change the tokens or the recorded sha256.
            body = raw.decode("utf-8", errors="replace")
            body = body.replace("\r\n", "\n")
            if not body.endswith("\n"):
                body += "\n"
            # Canonical tokenization refuses non-NFC text (common.RefTokenizer); the tree carries
            # such text on purpose (tests/reference/test_hessian_corpus.py's non-NFC probe, from
            # f87a4b0 on), so the canonical path NFC-normalizes each file and names the ones it
            # changed. hf-auto keeps the bytes as they are (hessian-v1's corpus reproduces).
            if tokenizer.mode == "canonical" and not unicodedata.is_normalized("NFC", body):
                body = unicodedata.normalize("NFC", body)
                nfc_files.append(rel)
            parts.append(f"==> {rel} <==\n{body}")
            code_parts.append((rel, body))
        text = "".join(parts)
        # The whole source tree as one text, far past the model's maximum length on purpose: only
        # L-token windows of it are run.
        ids = tokenizer.encode(text)
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
                        "window_starts": starts, "file_list": files,
                        # only when non-empty, so a tree with no non-NFC file records exactly what
                        # it did before this field existed (hessian-v2 at 714955f; --rms-only).
                        **({"nfc_normalized_files": nfc_files} if nfc_files else {})})

    # The disjointness check over sources 1-3 (the gen samples are gated in build_gen_corpus, before
    # tokenizing): every window, decoded back to exactly the text the Hessians see, against
    # kl_corpus/ outside its #include/import lines. RECORDED per source (kl_disjointness.overlaps)
    # and printed, NOT refused: leaving model.cpp and layer_golden.py out of the code concatenation
    # keeps the KL excerpts' own sources away, but not the files sharing runs with them -- model.h
    # and three others repeat a model.cpp comment, container.cpp a block of its code, ten files the
    # venv usage line of layer_golden.py's docstring, the rest single idiomatic lines (25 of 277
    # files at the time of writing). Whether a
    # window meets one depends on --code-seqs and on the commit (--code-seqs 48 on the working tree
    # hit one window; 17 of 21 counts between 16 and 64 hit >= 1), and hessian-v1's own code/04 (at
    # 34d381a) holds that model.cpp comment, so refusing would make the corpus a lottery and
    # `--rms-only` on hessian-v1 impossible. A hit names the repo file(s) holding the shared run.
    kl = kl_index()
    t0 = time.perf_counter()
    hits = gate_windows(seqs, tokenizer, kl)
    for h in hits:
        if h["id"].startswith("code/"):
            files = [rel for rel, body in code_parts if h["substring"] in ws_normalize(body)]
            h["where"] = "in " + ", ".join(files) if files else "across a file boundary"
    if hits:
        say(f"[hessian] WARNING disjointness: {len(hits)} of {len(seqs)} window(s) of sources 1-3 "
            f"share a {KL_SHINGLE_CHARS}-character whitespace-normalized run with the held-out KL "
            f"corpus {kl.label} (repo code quoting model.cpp / layer_golden.py; recorded under the "
            f"source's kl_disjointness.overlaps, not refused):")
        report_kl_hits(hits)
    elif seqs:
        say(f"[hessian] disjointness: {len(seqs)} window(s) of sources 1-3, no "
            f"{KL_SHINGLE_CHARS}-char overlap with kl_corpus ({time.perf_counter() - t0:.2f}s)")
    for src in sources:
        mine = [h for h in hits if (h["id"] == f"calib/{src['name']}" if src["source"] == "calib"
                                    else h["id"].startswith(src["source"] + "/"))]
        src["kl_disjointness"] = kl.record("every token window of this source, decoded; overlaps "
                                           "recorded, not refused", mine)

    # 4. Text our own quantized model generated (--gen-file samples.jsonl): per category, the
    #    rendered non-rejected samples in id order, gated disjoint from kl_corpus/, cut into
    #    non-overlapping windows (build_gen_corpus). Appended after sources 1-3, whose tokens a run
    #    without --gen-file leaves exactly as they were.
    gen_file = getattr(args, "gen_file", None)
    if gen_file is not None and str(gen_file).lower() != "none":
        gen_seqs, gen_sources = build_gen_corpus(Path(gen_file), tokenizer, L,
                                                 getattr(args, "gen_max_seqs", None), kl)
        seqs += gen_seqs
        sources += gen_sources
    if not seqs:
        raise SystemExit("[hessian] empty corpus")
    for s in seqs:
        if len(s.token_ids) < 2:
            raise SystemExit(f"[hessian] sequence {s.name} has {len(s.token_ids)} tokens; need >= 2")
    return seqs, sources


def print_corpus_sources(sources: list[dict]) -> None:
    """The corpus report: one line per source, a second line per gen category (samples used and
    rejected, the concatenation, the windows taken and the dropped tail), and a subtotal per source
    kind."""
    for src in sources:
        label = src.get("name") or src.get("path") or src["source"]
        print(f"    {src['source']:<9} {src['sequences']:>3} seq {src['tokens']:>7} tok  {label}")
        if src["source"] == "gen":
            f = src["formats"]
            print(f"              {src['samples']} sample(s) ({f['raw']} raw, {f['chat']} chat, "
                  f"{src['thinking_samples']} thinking) + {src['samples_rejected']} rejected; "
                  f"{src['concatenation_tokens']} tok -> {src['sequences']} of "
                  f"{src['windows_available']} window(s); tail of {src['dropped_tail_tokens']} tok "
                  f"dropped")
    kinds: dict[str, list[int]] = {}
    for src in sources:
        k = kinds.setdefault(src["source"], [0, 0])
        k[0] += src["sequences"]
        k[1] += src["tokens"]
    print("    by source: " + "; ".join(f"{k} {n} seq {t} tok" for k, (n, t) in kinds.items()))


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


def rms_weightless(x, eps: float):
    """`Qwen3_5RMSNorm._norm(x.float())` exactly (transformers' modeling_qwen3_5.py): fp32
    `x * rsqrt(mean(x^2) + eps)` over the last axis -- WITHOUT the `(1 + w)` multiply and WITHOUT the
    `.type_as(x)` bf16 cast that follow it in the module's forward. That is what a rotated
    container's weightless norm feeds its folded in-projections (before Q). The runtime's
    RmsNormKernel computes the same `x * rsqrtf(sum(x^2)/hidden + eps)` in fp32."""
    import torch

    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def norm_eps(mod) -> float:
    """A norm module's own epsilon. `Qwen3_5RMSNorm` stores it as `eps`; `Qwen3_5RMSNormGated` (and
    Llama-style norms) as `variance_epsilon`."""
    for attr in ("eps", "variance_epsilon"):
        v = getattr(mod, attr, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    raise SystemExit(f"[hessian] {type(mod).__name__} has neither .eps nor .variance_epsilon")


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
    #: The weightless rms taps (RmsTapPlan): stats, the norm's eps, and the fp32 (1 + w) the norm
    #: module multiplies by (from the materialized layer), for evaluate_rms_gates.
    rms_written: dict[str, dict] = field(default_factory=dict)
    rms_eps: dict[str, float] = field(default_factory=dict)
    rms_scale: dict[str, np.ndarray] = field(default_factory=dict)


def _write_accum(out_dir: Path, plan, acc: HessianAccum, st: CaptureState,
                 rms: bool = False) -> None:
    import torch

    t0 = time.perf_counter()
    rows = acc.rows
    h = acc.finalize_fp32()
    on_device = h.is_cuda
    host = h.cpu().numpy()
    del h
    if on_device:
        torch.cuda.empty_cache()
    stats = write_hess_file(out_dir / plan.file, host, rows)
    del host
    (st.rms_written if rms else st.written)[plan.file] = stats
    st.seconds_per_file[plan.file] = time.perf_counter() - t0
    warn = "" if stats["finite"] and stats["min_diag"] > 0 else "   <-- NOT FINITE / NON-POSITIVE DIAG"
    print(f"[hessian]   {plan.file:<22} K={stats['K']:>5} rows={rows:>7} "
          f"trace={stats['trace']:.6g} min_diag={stats['min_diag']:.3e} "
          f"{stats['bytes'] / 2**20:8.1f} MiB {st.seconds_per_file[plan.file]:5.1f}s{warn}",
          flush=True)


def run_capture(ref, seqs: list[Seq], plans: list[TapPlan], specs, n_run: int, out_dir: Path,
                hidden_device, want_draft_head: bool, rms_plans: list[RmsTapPlan] = (),
                shared_gate: bool = True) -> CaptureState:
    """The layer-major pass. `rms_plans` adds the weightless rms taps (a forward_pre_hook on the
    layer's norm module, see rms_weightless); `--rms-only` passes no `plans` and `shared_gate=False`,
    so only those are hooked."""
    import torch

    st = CaptureState()
    dev = ref.device
    layer_types = ref.layer_types
    gate_layers = {}
    if shared_gate:
        for t in ("linear_attention", "full_attention"):
            if t in layer_types[:n_run]:
                gate_layers[layer_types.index(t)] = t
    by_layer: dict[int, list[TapPlan]] = {}
    for p in plans:
        if p.scope == "layer":
            by_layer.setdefault(p.layer, []).append(p)
    rms_by_layer: dict[int, list[RmsTapPlan]] = {}
    for rp in rms_plans:
        rms_by_layer.setdefault(rp.layer, []).append(rp)

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
            # The weightless rms taps: the norm's INPUT is the residual x; accumulate rms(x) with the
            # module's own eps, no weight (the scale it would apply is recorded for the gates).
            rms_accs: dict[str, HessianAccum] = {}
            for rp in rms_by_layer.get(i, []):
                norm = getattr(layer, rp.norm)
                eps = norm_eps(norm)
                acc = HessianAccum(rp.k, dev)
                rms_accs[rp.file] = acc
                st.rms_eps[rp.file] = eps
                st.rms_scale[rp.file] = (1.0 + norm.weight.detach().float()).cpu().numpy()
                handles.append(norm.register_forward_pre_hook(
                    lambda _m, inputs, _acc=acc, _eps=eps: _acc.update(rms_weightless(inputs[0], _eps))))
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
                  f"{t_fwd:.1f}s, writing {len(accs) + len(rms_accs)} tap(s)", flush=True)
            for p in by_layer.get(i, []):
                _write_accum(out_dir, p, accs.pop(p.file), st)
            for rp in rms_by_layer.get(i, []):
                _write_accum(out_dir, rp, rms_accs.pop(rp.file), st, rms=True)
            st.layer_seconds.append(time.perf_counter() - t_layer)

        # Diagnostics (huihui capture, 2026-09-28: mtp.out.hess came out with inf in heads 12-17,
        # dims 0-63, from activations ~1e20 that a synthetic MTP run does not reproduce).
        # $env:R4DX_HESSIAN_SAVE_STACK=<dir> saves the stack's output (the pre-final-norm residual
        # of every sequence, bf16) so the head pass can be re-run and studied without the stack.
        save_stack = os.environ.get("R4DX_HESSIAN_SAVE_STACK")
        if save_stack and n_run == ref.n_layers:
            sdir = Path(save_stack)
            sdir.mkdir(parents=True, exist_ok=True)
            t_save = time.perf_counter()
            torch.save({"hidden": [h.cpu() for h in hidden],
                        "token_ids": [list(s.token_ids) for s in seqs]}, sdir / "stack_output.pt")
            print(f"[hessian] saved the stack output to {sdir / 'stack_output.pt'} "
                  f"({time.perf_counter() - t_save:.1f}s)", flush=True)

        # lm_head + MTP: both consume the stack's output, one pass over the sequences.
        head_plans = [p for p in plans if p.scope in ("lm_head", "mtp")]
        if head_plans:
            from imatrix_capture import CaptureResult, MtpTaps

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
            # Report (never alter) any MTP o_proj input that is non-finite or above 1e4 in magnitude.
            anomalies: list[str] = []
            cur = {"j": -1}
            watch = None
            if mtp is not None:
                def _watch(_m, inputs):
                    x = inputs[0].detach()
                    bad = ~torch.isfinite(x) | (x.abs() > 1e4)
                    if bool(bad.any()):
                        r = torch.nonzero(bad.reshape(-1, x.shape[-1]).any(-1)).flatten().tolist()
                        c = torch.nonzero(bad.reshape(-1, x.shape[-1]).any(0)).flatten().tolist()
                        msg = (f"seq {cur['j']}: {len(r)} row(s) {r[:8]}, {len(c)} channel(s) "
                               f"{c[:4]}..{c[-4:]}, max |x| {x.float().abs().max().item():.3e}")
                        anomalies.append(msg)
                        print(f"[hessian] MTP o_proj input anomaly: {msg}", flush=True)
                watch = mtp.layer.self_attn.o_proj.register_forward_pre_hook(_watch)
            try:
                for j, s in enumerate(seqs):
                    cur["j"] = j
                    pre = hidden[j].to(dev)  # [1, T, hidden], the PRE-final-norm residual
                    if lm_acc is not None:
                        lm_acc.update(ref.model.norm(pre))
                    if mtp is not None:
                        mtp.run(pre[0], s.token_ids)
                    del pre
            finally:
                if watch is not None:
                    watch.remove()
                if mtp is not None:
                    mtp.close()
            print(f"[hessian] MTP o_proj input anomalies: {len(anomalies)}", flush=True)
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


def norm_weight_name(scope: str, layer: int | None, module: str) -> str | None:
    """HF name of the zero-centred norm whose output a tap's representative module reads, or None
    for taps not fed by a norm (out_proj / o_proj / down_proj inputs)."""
    if scope == "lm_head":
        return "model.language_model.norm.weight"
    if scope == "mtp":
        if module == "mtp:norm_out":
            return "mtp.norm.weight"
        n = NORM_BEFORE.get(module.split(":", 1)[1])
        return f"mtp.layers.0.{n}.weight" if n else None
    n = NORM_BEFORE.get(module)
    return f"model.language_model.layers.{layer}.{n}.weight" if n else None


def structural_zero_channels(model_dir: Path, taps: dict[str, tuple]) -> dict[str, list[int]]:
    """{file: channels whose preceding norm scale (1 + w) is exactly 0}. `taps` maps each file to
    (scope, layer, module). CPU only: reads the norm vectors from the checkpoint's shards."""
    import torch
    from safetensors import safe_open

    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, list[int]] = {}
    for f, (scope, layer, module) in taps.items():
        name = norm_weight_name(scope, layer, module)
        if name is None:
            continue
        with safe_open(str(model_dir / weight_map[name]), framework="pt", device="cpu") as sf:
            w = sf.get_tensor(name).to(torch.float32)
        out[f] = torch.nonzero(1.0 + w == 0.0).flatten().tolist()
    return out


def read_hess_diag(path: Path) -> np.ndarray:
    """The stored fp32 diagonal of a .hess file (row i of the packed upper triangle starts at
    element i*K - i*(i-1)/2), memory-mapped: no full read."""
    with open(path, "rb") as f:
        magic, k, flags, _rows, _trace = HESS_HEADER.unpack(f.read(HESS_HEADER.size))
    if magic != HESS_MAGIC or flags != HESS_FLAG_PACKED_UPPER:
        raise ValueError(f"{path}: not a v1 packed .hess file")
    tri = np.memmap(path, dtype="<f4", mode="r", offset=HESS_HEADER.size)
    i = np.arange(k, dtype=np.int64)
    return np.asarray(tri[i * k - i * (i - 1) // 2], dtype=np.float64)


def diag_gate(nonpos: dict[str, list[int]], structural: dict[str, list[int]]) -> dict:
    """Non-positive diagonal channels are allowed only where the preceding norm scale is exactly 0,
    and never more than MAX_REPORTED_CHANNELS per file (the stats would be truncated)."""
    bad = {}
    for f, ch in nonpos.items():
        extra = sorted(set(ch) - set(structural.get(f, [])))
        if extra or len(ch) >= MAX_REPORTED_CHANNELS:
            bad[f] = extra or ch
    return {"min_diag_ok": not bad,
            "nonpositive_diag_files": sorted(bad),
            "nonpositive_diag_unexplained": bad,
            "structural_zero_channels": {f: ch for f, ch in sorted(nonpos.items()) if ch and f not in bad}}


def evaluate_gates(audit: dict, all_keys: list[str], plans: list[TapPlan], st: CaptureState,
                   selected_keys: list[str], shared_expected: list[int],
                   structural: dict[str, list[int]]) -> dict:
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
    dg = diag_gate({f: w["nonpos_channels"] for f, w in st.written.items()}, structural)
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
        **dg,
        "min_diag": min_diag,
        "min_diag_file": min_diag_file,
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
    print(f"[gate] min diag > 0 except structurally dead channels: {pf(g['min_diag_ok'])} "
          f"(min {g['min_diag']:.4e} in {g['min_diag_file']}; dead (1+w)==0 channels allowed: "
          f"{g['structural_zero_channels'] or 'none'}; unexplained: "
          f"{g['nonpositive_diag_unexplained'] or 'none'})")
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


RMS_GATES = ("rms_complete", "rms_k_ok", "rms_finite_ok", "rms_min_diag_ok", "rms_rows_ok",
             "rms_rows_match_ok", "rms_consistency_ok")


def evaluate_rms_gates(rms_plans: list[RmsTapPlan], written: dict[str, dict], out_dir: Path,
                       post_rows: dict[str, int], scales: dict[str, np.ndarray]) -> dict:
    """The weightless rms files' gates. CPU only: the diagonals are read back from disk (what the
    converter will read), `scales[file]` is the fp32 (1 + w) of the tap's norm and `post_rows` the
    rows of each paired post-norm file.

    - finite; min diag > 0 with NO exemption (rms(x) of a real residual has no structurally zero
      channel -- a zero means the norm's OUTPUT, or a weighted input, was hooked);
    - rows >= 4 x K, and == the post-norm file's rows (the same tokens);
    - consistency, which proves both captures saw the same activations through the same norm: on
      every channel with (1 + w_i) != 0, |diag(H_post)_i - (1 + w_i)^2 diag(H_rms)_i| <=
      RMS_CONSISTENCY_RTOL x (1 + w_i)^2 diag(H_rms)_i, and diag(H_post)_i == 0 exactly where
      (1 + w_i) == 0. The worst channel is reported."""
    missing, k_bad, not_finite, rows_short, rows_mismatch = [], [], [], [], []
    nonpos: dict[str, list[int]] = {}
    inconsistent: list[str] = []
    dead_nonzero: dict[str, list[int]] = {}
    per_file: dict[str, dict] = {}
    worst = {"rel": 0.0, "file": None, "channel": None, "post_diag": None, "predicted_diag": None}
    for rp in rms_plans:
        w = written.get(rp.file)
        post_path = out_dir / rp.post_file
        if w is None or not (out_dir / rp.file).exists() or not post_path.exists():
            missing.append(rp.file if w is None else f"{rp.file} (or its pair {rp.post_file})")
            continue
        if w["K"] != rp.k:
            k_bad.append((rp.file, w["K"], rp.k))
        if not w["finite"]:
            not_finite.append(rp.file)
        if w["nonpos_channels"] or not w["min_diag"] > 0.0:
            nonpos[rp.file] = list(w["nonpos_channels"])
        if w["rows"] < ROWS_PER_K_MIN * w["K"]:
            rows_short.append((rp.file, w["rows"], ROWS_PER_K_MIN * w["K"]))
        if post_rows.get(rp.post_file) != w["rows"]:
            rows_mismatch.append((rp.file, w["rows"], rp.post_file, post_rows.get(rp.post_file)))
        d_rms = read_hess_diag(out_dir / rp.file)
        d_post = read_hess_diag(post_path)
        s = np.asarray(scales[rp.file], dtype=np.float64)
        if not (d_rms.shape == d_post.shape == s.shape):
            k_bad.append((rp.file, int(d_rms.shape[0]), int(d_post.shape[0]), int(s.shape[0])))
            continue
        live = s != 0.0
        pred = s * s * d_rms
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(d_post - pred) / pred
        rel = np.where(live, np.where(np.isfinite(rel), rel, np.inf), 0.0)
        j = int(np.argmax(rel))
        dead = np.flatnonzero(~live)
        dz = [int(c) for c in dead if d_post[c] != 0.0]
        per_file[rp.file] = {
            "post_file": rp.post_file, "worst_rel": float(rel[j]), "worst_channel": j,
            "post_diag": float(d_post[j]), "predicted_diag": float(pred[j]),
            "median_rel": float(np.median(rel[live])) if live.any() else 0.0,
            "dead_channels": dead[:MAX_REPORTED_CHANNELS].tolist(),
            "min_diag": w["min_diag"], "max_diag": w["max_diag"], "rows": w["rows"]}
        if not rel[j] <= RMS_CONSISTENCY_RTOL:
            inconsistent.append(rp.file)
        if dz:
            dead_nonzero[rp.file] = dz[:MAX_REPORTED_CHANNELS]
        if not rel[j] <= worst["rel"]:
            worst = {"rel": float(rel[j]), "file": rp.file, "channel": j,
                     "post_diag": float(d_post[j]), "predicted_diag": float(pred[j])}
    min_file = min(per_file, key=lambda f: per_file[f]["min_diag"]) if per_file else None
    g = {
        "rms_files": len(rms_plans),
        "rms_complete": not missing and bool(rms_plans),
        "rms_missing": missing,
        "rms_k_ok": not k_bad,
        "rms_k_mismatches": k_bad,
        "rms_finite_ok": not not_finite,
        "rms_not_finite": not_finite,
        "rms_min_diag_ok": not nonpos,
        "rms_nonpositive_diag": nonpos,
        "rms_min_diag": per_file[min_file]["min_diag"] if min_file else None,
        "rms_min_diag_file": min_file,
        "rms_rows_ok": not rows_short,
        "rms_rows_per_k_min": ROWS_PER_K_MIN,
        "rms_rows_short": rows_short,
        "rms_rows_match_ok": not rows_mismatch,
        "rms_rows_mismatch": rows_mismatch,
        "rms_consistency_ok": not inconsistent and not dead_nonzero,
        "rms_consistency_rtol": RMS_CONSISTENCY_RTOL,
        "rms_consistency_worst": worst,
        "rms_consistency_failed": inconsistent,
        "rms_dead_channels_nonzero_in_post": dead_nonzero,
        "rms_per_file": per_file,
    }
    g["ok"] = all(g[k] for k in RMS_GATES)
    return g


def print_rms_gates(g: dict) -> None:
    def pf(ok):
        return "PASS" if ok else "FAIL"
    print(f"\n[gate] rms files present: {pf(g['rms_complete'])} ({g['rms_files']} planned"
          f"{'; missing ' + str(g['rms_missing'][:6]) if g['rms_missing'] else ''})")
    print(f"[gate] rms K == hidden == post-norm K: {pf(g['rms_k_ok'])} {g['rms_k_mismatches'][:6]}")
    print(f"[gate] rms finite: {pf(g['rms_finite_ok'])} {g['rms_not_finite'][:8]}")
    print(f"[gate] rms min diag > 0, no exemption: {pf(g['rms_min_diag_ok'])} (min "
          f"{g['rms_min_diag']} in {g['rms_min_diag_file']}"
          f"{'; non-positive: ' + str(dict(list(g['rms_nonpositive_diag'].items())[:4])) if g['rms_nonpositive_diag'] else ''})")
    print(f"[gate] rms rows >= {g['rms_rows_per_k_min']} x K: {pf(g['rms_rows_ok'])} "
          f"{g['rms_rows_short'][:6]}")
    print(f"[gate] rms rows == post-norm rows: {pf(g['rms_rows_match_ok'])} {g['rms_rows_mismatch'][:4]}")
    w = g["rms_consistency_worst"]
    print(f"[gate] diag(H_post) vs (1+w)^2 diag(H_rms) within {g['rms_consistency_rtol']:.0%}: "
          f"{pf(g['rms_consistency_ok'])} (worst {w['rel']:.3e} at {w['file']} channel {w['channel']}: "
          f"post {w['post_diag']} vs predicted {w['predicted_diag']}"
          f"{'; failing ' + str(g['rms_consistency_failed'][:6]) if g['rms_consistency_failed'] else ''}"
          f"{'; dead channels non-zero in post ' + str(g['rms_dead_channels_nonzero_in_post']) if g['rms_dead_channels_nonzero_in_post'] else ''})")
    print(f"[gate] rms overall: {pf(g['ok'])}")


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def _existing_anchor(p: Path) -> Path:
    p = p.resolve()
    while not p.exists():
        p = p.parent
    return p


def regate(out_dir: Path, model_dir: Path) -> int:
    """CPU only: re-run the diagonal gate over a set a capture already wrote (reading each file's
    diagonal from disk and the norm vectors from the checkpoint), keep every other recorded gate,
    and promote hessian.failed.json to hessian.json if everything now passes. For a set written
    before the dead-channel rule existed -- no re-capture needed."""
    failed = out_dir / FAILED_MANIFEST_NAME
    if not failed.exists():
        raise SystemExit(f"[hessian] --regate: {failed} not found (nothing to re-gate)")
    doc = json.loads(failed.read_text(encoding="utf-8"))
    # A --rms-taps capture lists its rms files (and rms_keys) only if their own gates passed at
    # capture time; a manifest claiming otherwise is not something to promote.
    if doc.get("rms_keys") and not doc.get("rms_capture", {}).get("gates", {}).get("ok"):
        raise SystemExit("[hessian] --regate: rms_keys present but rms_capture.gates.ok is not true")
    gates = doc["gates"]
    nonpos = {}
    for f in doc["files"]:
        d = read_hess_diag(out_dir / f)
        nonpos[f] = np.flatnonzero(d <= 0.0)[:MAX_REPORTED_CHANNELS].tolist()
    # Only the post-norm taps are in "taps", so only they can be exempted by a dead (1 + w) == 0
    # channel; the rms files (listed in "files" too) are re-gated with no exemption.
    taps = {f: (t["scope"], t["layer"], t["module"]) for f, t in doc["taps"].items()}
    structural = structural_zero_channels(model_dir, taps)
    gates.pop("nonpositive_diag_files", None)
    gates.update(diag_gate(nonpos, structural))
    gates["ok"] = all(gates[g] for g in ("converter_audit_ok", "keys_selected_ok", "k_ok",
                                         "finite_ok", "min_diag_ok", "rows_ok", "shared_input_ok"))
    print_gates(gates)
    if not gates["ok"]:
        print("[hessian] --regate: still failing; hessian.failed.json left as is")
        return 1
    extra = {k: v for k, v in doc.items()
             if k not in ("format", "version", "files", "keys", "rms_keys")}
    extra["gates"] = gates
    extra["regated"] = {"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "rule": "non-positive diagonal allowed only on channels whose preceding "
                                "norm scale (1 + w) is exactly 0 (never on an rms file)"}
    path = write_manifest(out_dir, doc["files"], doc["keys"], extra, rms_keys=doc.get("rms_keys"))
    failed.unlink()
    print(f"[hessian] --regate: gates pass; wrote {path}")
    return 0


# --------------------------------------------------------------------------------------------
# --rms-only / --rms-taps: the manifest side (CPU)
# --------------------------------------------------------------------------------------------


def corpus_record(seqs: list[Seq], sources: list[dict], seq_len: int, tokenizer) -> dict:
    """hessian.json "corpus": totals, the tokenizer (RefTokenizer.provenance, mode first) and the
    sources."""
    return {"sequences": len(seqs), "tokens": sum(len(s.token_ids) for s in seqs),
            "seq_len": seq_len, "tokenizer": tokenizer.provenance(), "sources": sources}


def rms_only_tokenizer_mode(doc: dict, requested: str | None,
                            manifest: str = MANIFEST_NAME) -> tuple[str, str]:
    """The tokenizer mode `--rms-only` must use, and how it was chosen: the mode the set's corpus
    was tokenized with (`corpus.tokenizer.mode`). A manifest without one predates the switch --
    every capture then used AutoTokenizer -- so it is hf-auto (hessian-v1). The rms taps must see
    the post-norm taps' tokens, so an explicit `--tokenizer` that disagrees is refused."""
    from common import recorded_tokenizer_mode

    mode, recorded = recorded_tokenizer_mode(doc.get("corpus", {}).get("tokenizer"))
    how = (f"recorded in {manifest} (corpus.tokenizer.mode)" if recorded else
           f"{manifest} records no tokenizer mode: captured before the switch, with AutoTokenizer")
    if requested is not None and requested != mode:
        raise SystemExit(
            f"[hessian] --rms-only: --tokenizer {requested}, but the set was tokenized with {mode} "
            f"({how}). The rms taps must see exactly the post-norm taps' tokens: omit --tokenizer "
            f"(it then follows the set) or pass --tokenizer {mode}. A {requested} set needs a new "
            f"full capture into another --out-dir.")
    return mode, how


def corpus_mismatches(recorded: dict, sources: list[dict], n_seqs: int, n_tokens: int,
                      seq_len: int, tokenizer_mode: str | None = None) -> list[str]:
    """How this run's corpus differs from a manifest's recorded `corpus` (empty: identical). Every
    source field is compared except CORPUS_HOW_FIELDS -- sha256s, token counts, windows, file lists
    -- and, given `tokenizer_mode`, the recorded tokenizer mode (none recorded: hf-auto)."""
    diffs = []
    if tokenizer_mode is not None:
        from common import recorded_tokenizer_mode

        want = recorded_tokenizer_mode(recorded.get("tokenizer"))[0]
        if want != tokenizer_mode:
            diffs.append(f"corpus tokenizer mode: recorded {want!r}, this run {tokenizer_mode!r}")
    for name, want, got in (("seq_len", recorded.get("seq_len"), seq_len),
                            ("sequences", recorded.get("sequences"), n_seqs),
                            ("tokens", recorded.get("tokens"), n_tokens)):
        if want != got:
            diffs.append(f"corpus {name}: recorded {want!r}, this run {got!r}")
    rec = recorded.get("sources", [])
    if len(rec) != len(sources):
        diffs.append(f"corpus sources: recorded {len(rec)}, this run {len(sources)}")

    def short(v):
        s = repr(v)
        return s if len(s) <= 80 else s[:77] + "..."
    for a, b in zip(rec, sources):
        a2 = {k: v for k, v in a.items() if k not in CORPUS_HOW_FIELDS}
        b2 = {k: v for k, v in b.items() if k not in CORPUS_HOW_FIELDS}
        for k in sorted(set(a2) | set(b2)):
            if a2.get(k) != b2.get(k):
                label = a.get("name") or a.get("path") or ""
                diffs.append(f"source {a.get('source')}{'/' + label if label else ''} {k}: "
                             f"recorded {short(a2.get(k))}, this run {short(b2.get(k))}")
    return diffs


def rms_keys_of(rms_plans: list[RmsTapPlan]) -> dict[str, str]:
    return {k: rp.file for rp in rms_plans for k in rp.keys}


def manifest_without_rms(doc: dict) -> tuple[dict, dict, dict]:
    """(files, keys, extra) of a manifest with its rms entries (rms_keys, their files, rms_capture)
    removed -- what `--rms-only --force` rewrites before recapturing them."""
    rms_files = set(doc.get("rms_keys", {}).values())
    if rms_files & set(doc["keys"].values()):
        raise SystemExit(f"[hessian] manifest lists {sorted(rms_files & set(doc['keys'].values()))} "
                         f"under both keys and rms_keys -- refusing to touch it")
    files = {f: v for f, v in doc["files"].items() if f not in rms_files}
    extra = {k: v for k, v in doc.items()
             if k not in ("format", "version", "files", "keys", "rms_keys", "rms_capture")}
    return files, dict(doc["keys"]), extra


def merge_rms_manifest(out_dir: Path, doc: dict, rms_plans: list[RmsTapPlan],
                       written: dict[str, dict], rms_capture: dict) -> Path:
    """`doc` (the validated hessian.json, without rms entries) plus the rms files under "files",
    "rms_keys" and the "rms_capture" provenance, rewritten atomically (write_manifest: temp + rename,
    LF). Every other field is kept as it was, in its order."""
    files, keys, extra = manifest_without_rms(doc)
    for rp in rms_plans:
        if rp.file in files:
            raise RuntimeError(f"{rp.file} is already listed in the manifest")
        files[rp.file] = written[rp.file]
    extra["rms_capture"] = rms_capture
    return write_manifest(out_dir, files, keys, extra, rms_keys=rms_keys_of(rms_plans))


def write_rms_failed(out_dir: Path, rms_plans: list[RmsTapPlan], written: dict[str, dict],
                     rms_capture: dict) -> Path:
    """The report of an rms capture whose gates failed: its files and the rms_keys it would have
    added, under RMS_FAILED_MANIFEST_NAME (never opened by the converter). The manifest is untouched."""
    doc = {"format": MANIFEST_FORMAT + "-rms-failed", "version": MANIFEST_VERSION,
           "files": {f: {"K": int(w["K"]), "rows": int(w["rows"]), "trace": float(w["trace"])}
                     for f, w in sorted(written.items())},
           "rms_keys": dict(sorted(rms_keys_of(rms_plans).items())),
           "rms_capture": rms_capture}
    path = out_dir / RMS_FAILED_MANIFEST_NAME
    tmp = out_dir / (RMS_FAILED_MANIFEST_NAME + ".tmp")
    with open(tmp, "wb") as f:
        f.write((json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, path)
    return path


def rms_capture_record(tool: str, rms_plans: list[RmsTapPlan], st: CaptureState, gates: dict,
                       **provenance) -> dict:
    """hessian.json "rms_capture": what the rms taps are, how and from what they were captured."""
    return {
        "tool": tool,
        "semantics": ("H_rms = E[r^T r] over every calibration token, r = rms(x) = x * rsqrt(mean(x^2) "
                      "+ eps) of a text layer's input_layernorm / post_attention_layernorm INPUT x "
                      "(the bf16 residual), in fp32 exactly as Qwen3_5RMSNorm._norm(x.float()) with "
                      "the module's own eps -- NO (1 + w) multiply, NO bf16 cast. The input a "
                      "rotated container's weightless norm feeds its folded in-projections (before "
                      "Q). fp32 GEMM per sequence, fp64 accumulation, stored fp32 packed upper "
                      "triangle; 'rms_keys' maps each norm-fed in-projection to its file."),
        "eps": sorted(set(st.rms_eps.values())),
        "taps": {rp.file: {"layer": rp.layer, "norm": rp.norm, "post_file": rp.post_file,
                           "keys": rp.keys,
                           "tap": f"forward_pre_hook on layer {rp.layer}'s {rp.norm} (its input)"}
                 for rp in rms_plans},
        "file_stats": {f: {"min_diag": w["min_diag"], "max_diag": w["max_diag"],
                           "finite": w["finite"], "bytes": w["bytes"],
                           "seconds": st.seconds_per_file.get(f)}
                       for f, w in sorted(st.rms_written.items())},
        "gates": gates,
        **provenance,
    }



def _pick_hidden_device(args, device, hidden_bytes: int):
    import torch

    if args.hidden_device == "auto":
        return device if hidden_bytes <= 6 * 2**30 else torch.device("cpu")
    return device if args.hidden_device == "cuda" else torch.device("cpu")


def run_rms_only(args) -> int:
    """--rms-only: capture ONLY the weightless rms taps (a full layer-major forward, nothing else
    hooked) and merge them into the validated set already in --out-dir. Before any GPU work it
    refuses unless hessian.json exists, the checkpoint's config.json sha256 and this run's corpus
    (every source's sha256, token counts and windows, tokenized with the mode the set records --
    rms_only_tokenizer_mode) equal what the manifest records -- the rms
    taps must see the SAME activations as the post-norm taps they pair with -- and (without --force)
    the manifest has no rms_keys yet. hessian.json is rewritten only if every rms gate passes and the
    manifest did not change meanwhile; otherwise it is left byte-identical and the report goes to
    hessian_rms.failed.json."""
    sys.path.insert(0, str(TOOLS_REF))
    from common import DEFAULT_MODEL_DIR, load_text_config, sha256_file

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    out_dir = args.out_dir
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        hint = (f"; {FAILED_MANIFEST_NAME} is there -- promote it with --regate first"
                if (out_dir / FAILED_MANIFEST_NAME).exists() else "")
        raise SystemExit(f"[hessian] --rms-only: {manifest_path} not found (it merges into a "
                         f"validated set){hint}")
    raw = manifest_path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("format") != MANIFEST_FORMAT or doc.get("version") != MANIFEST_VERSION:
        raise SystemExit(f"[hessian] --rms-only: {manifest_path} is not an {MANIFEST_FORMAT} "
                         f"v{MANIFEST_VERSION} manifest")
    cfg_sha = sha256_file(model_dir / "config.json")
    if doc.get("config_sha256") != cfg_sha:
        raise SystemExit(f"[hessian] --rms-only: {model_dir}\\config.json sha256 {cfg_sha} is not "
                         f"the set's ({doc.get('config_sha256')}, captured from "
                         f"{doc.get('model_dir')}) -- pass the same --model-dir")
    if doc.get("rms_keys") and not args.force:
        raise SystemExit(f"[hessian] --rms-only: {manifest_path} already has rms_keys "
                         f"({len(doc['rms_keys'])} keys); pass --force to recapture them (the old "
                         f"rms entries are removed from it before the capture starts)")

    _, text_config = load_text_config(model_dir)
    n_layers = int(text_config.num_hidden_layers)
    hidden = int(text_config.hidden_size)
    n_run = n_layers if args.layers is None else max(1, min(args.layers, n_layers))
    key_re = re.compile(args.keys) if args.keys else None
    post_keys: dict[str, list[str]] = {}
    for k, f in doc["keys"].items():
        if RMS_POST_FILE_RE.fullmatch(f) and (key_re is None or key_re.search(k)):
            post_keys.setdefault(f, []).append(k)
    rms_plans = build_rms_plans(post_keys, n_run, hidden)
    if not rms_plans:
        raise SystemExit("[hessian] --rms-only: the manifest has no norm-fed tap "
                         "(L{i}.in / L{i}.mlp_in) selected by --layers/--keys")
    for rp in rms_plans:
        if int(doc["files"][rp.post_file]["K"]) != hidden:
            raise SystemExit(f"[hessian] {rp.post_file} has K={doc['files'][rp.post_file]['K']}, "
                             f"not hidden_size {hidden}")

    from common import load_ref_tokenizer

    mode, mode_how = rms_only_tokenizer_mode(doc, getattr(args, "tokenizer", None))
    tokenizer = load_ref_tokenizer(model_dir, mode)
    print(f"[hessian] tokenizer: {mode} ({mode_how})")
    t0 = time.perf_counter()
    seqs, sources = build_corpus(args, tokenizer)
    total_tokens = sum(len(s.token_ids) for s in seqs)
    print(f"[hessian] corpus: {len(seqs)} sequences, {total_tokens} tokens "
          f"({time.perf_counter() - t0:.1f}s to tokenize)")
    print_corpus_sources(sources)
    diffs = corpus_mismatches(doc.get("corpus", {}), sources, len(seqs), total_tokens, args.seq_len,
                              tokenizer_mode=mode)
    if diffs:
        for d in diffs[:24]:
            print(f"[hessian]   {d}")
        code = [s for s in doc.get("corpus", {}).get("sources", []) if s.get("source") == "code"]
        hint = (f" The code windows are read from this checkout's working tree unless --code-rev "
                f"names the commit the set was captured at (recorded sha256_of_concatenation "
                f"{code[0].get('sha256_of_concatenation')})." if code else "")
        gen = [s for s in doc.get("corpus", {}).get("sources", []) if s.get("source") == "gen"]
        if gen:
            cap = gen[0].get("max_seqs")
            hint += (f" The set includes self-generated text: pass --gen-file {gen[0].get('path')} "
                     f"(sha256 {gen[0].get('sha256')})"
                     f"{f' and --gen-max-seqs {cap}' if cap is not None else ''} as recorded.")
        raise SystemExit(f"[hessian] --rms-only: this corpus is not the one {manifest_path} was "
                         f"captured from, so the rms taps would not see the post-norm taps' "
                         f"activations. Pass that capture's corpus options.{hint}")
    print(f"[hessian] corpus matches {manifest_path.name} (sha256s, token counts, windows, "
          f"tokenizer mode {mode})")

    n_fwd = max(rp.layer for rp in rms_plans) + 1
    estimate = sum(hess_file_bytes(rp.k) for rp in rms_plans) + (1 << 20)
    print(f"[hessian] --rms-only: {len(rms_plans)} weightless rms tap(s) serving "
          f"{sum(len(rp.keys) for rp in rms_plans)} key(s), text layers 0..{n_fwd - 1} forwarded")
    if args.dry_run:
        for rp in rms_plans:
            print(f"    {rp.file:<22} K={rp.k:>5} rows~{total_tokens:>7} "
                  f"{hess_file_bytes(rp.k) / 2**20:8.1f} MiB  {rp.norm} input; pairs "
                  f"{rp.post_file}; {','.join(rp.keys)}")
    anchor = _existing_anchor(out_dir)
    free = shutil.disk_usage(anchor).free
    print(f"[hessian] disk estimate {estimate / 2**30:.2f} GiB; free at {anchor}: "
          f"{free / 2**30:.2f} GiB (need 1.1 x estimate)")
    if args.dry_run:
        print("[hessian] --dry-run: nothing captured, nothing written")
        return 0
    if free < 1.1 * estimate:
        raise SystemExit("[hessian] not enough free space on the target drive -- refusing to start")

    if doc.get("rms_keys"):  # --force: the old rms entries go first, so hessian.json never lists
        files, keys, extra = manifest_without_rms(doc)  # a file this run is about to overwrite
        write_manifest(out_dir, files, keys, extra)
        print(f"[hessian] --force: removed the old rms entries from {manifest_path}")
        raw = manifest_path.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
    sha_start = hashlib.sha256(raw).hexdigest()
    stale = out_dir / RMS_FAILED_MANIFEST_NAME
    if stale.exists():
        stale.unlink()

    import torch
    import transformers
    from common import resolve_device
    from full_logits_golden import StreamingReference

    device = resolve_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hidden_bytes = total_tokens * hidden * 2
    hidden_device = _pick_hidden_device(args, device, hidden_bytes)
    ref = StreamingReference(model_dir, device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    st = run_capture(ref, seqs, [], [], n_fwd, out_dir, hidden_device, False,
                     rms_plans=rms_plans, shared_gate=False)
    seconds = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[hessian] rms capture done: {seconds:.1f}s, peak VRAM {peak:.3f} GiB")

    return finish_rms_only(
        out_dir, doc, sha_start, rms_plans, st,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        model_dir=str(model_dir), config_sha256=cfg_sha,
        manifest_sha256_before=sha_start,
        corpus={"sequences": len(seqs), "tokens": total_tokens, "seq_len": args.seq_len,
                "code_rev": getattr(args, "code_rev", None),
                "gen_file": None if args.gen_file is None else str(args.gen_file),
                "tokenizer": dict(tokenizer.provenance(), chosen=mode_how),
                "matches": "the manifest's corpus: every source's sha256, token counts, windows, "
                           "tokenizer mode"},
        layers_forwarded=n_fwd, keys_filter=args.keys,
        device=str(device), hidden_device=str(hidden_device),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        seconds=seconds, peak_vram_gib=peak)


def finish_rms_only(out_dir: Path, doc: dict, sha_start: str, rms_plans: list[RmsTapPlan],
                    st: CaptureState, **provenance) -> int:
    """--rms-only after the capture (CPU): gate the rms files; merge them into hessian.json only if
    every gate passes AND hessian.json is still the file `doc` was read from (sha256 `sha_start`);
    otherwise leave it byte-identical and write hessian_rms.failed.json. Returns the exit code."""
    manifest_path = out_dir / MANIFEST_NAME
    post_rows = {rp.post_file: int(doc["files"][rp.post_file]["rows"]) for rp in rms_plans}
    gates = evaluate_rms_gates(rms_plans, st.rms_written, out_dir, post_rows, st.rms_scale)
    now = manifest_path.read_bytes() if manifest_path.exists() else b""
    gates["manifest_unchanged_ok"] = hashlib.sha256(now).hexdigest() == sha_start
    gates["ok"] = gates["ok"] and gates["manifest_unchanged_ok"]
    print_rms_gates(gates)
    if not gates["manifest_unchanged_ok"]:
        print(f"[gate] {manifest_path} changed during the capture: FAIL (not merging into it)")
    record = rms_capture_record("tools/reference/hessian_capture.py --rms-only", rms_plans, st,
                                gates, **provenance)
    if gates["ok"]:
        path = merge_rms_manifest(out_dir, doc, rms_plans, st.rms_written, record)
        print(f"\n[hessian] merged {len(rms_plans)} rms file(s) and "
              f"{len(rms_keys_of(rms_plans))} rms_keys into {path}")
        return 0
    path = write_rms_failed(out_dir, rms_plans, st.rms_written, record)
    print(f"\n[hessian] RMS GATES FAILED -- {manifest_path} left unchanged; report in {path}")
    return 1


def main() -> int:
    # Generated and KL text (Thai) can reach stdout, and on Windows a pipe or a file (a tee'd or
    # redirected capture) gets the ANSI code page with 'strict' errors: escape instead of dying.
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass
    sys.path.insert(0, str(TOOLS_REF))
    # common imports no torch at module level, so --write-fixture stays numpy-only.
    from common import DEFAULT_TOKENIZER_MODE, LEGACY_TOKENIZER_MODE, TOKENIZER_HELP, TOKENIZER_MODES

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regate", action="store_true",
                    help="CPU only: re-evaluate the gates of the set in --out-dir (after a gate "
                         "rule change) and promote hessian.failed.json to hessian.json if it passes")
    ap.add_argument("--write-fixture", type=Path, default=None, metavar="DIR",
                    help="CPU only, no checkpoint: write the tiny deterministic test set "
                         "(tests/convert/fixtures/hess_small) and exit")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="checkpoint (default: common.DEFAULT_MODEL_DIR)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="allow an --out-dir that already holds a hessian.json (it is deleted "
                         "before the capture starts; stale .hess files are overwritten). With "
                         "--rms-only: allow a hessian.json that already has rms_keys (those entries "
                         "are removed from it before the capture starts)")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="kv_calib_corpus-style directory; 'none' to skip")
    ap.add_argument("--calib-txt", type=Path, default=DEFAULT_CALIB_TXT, help="'none' to skip")
    ap.add_argument("--wikitext", default=str(DEFAULT_WIKITEXT),
                    help="WikiText-2 raw train file; 'none' to skip")
    ap.add_argument("--wikitext-seqs", type=int, default=64)
    ap.add_argument("--code-seqs", type=int, default=16,
                    help="windows of this repo's own sources (0 to skip)")
    ap.add_argument("--gen-file", type=Path, default=None, metavar="SAMPLES_JSONL",
                    help="corpus source 4: the corpus generator's samples.jsonl (text self-generated "
                         "by the quantized model; docs/quant2.md 'Corpus v2'). Non-rejected samples "
                         "are rendered (chats through the checkpoint's chat template), refused if "
                         "any (prompts included) shares a 50-char run with "
                         "tools/reference/kl_corpus/ outside its include/import lines, and cut "
                         "per category into non-overlapping --seq-len windows (default: none)")
    ap.add_argument("--gen-max-seqs", type=int, default=None, metavar="N",
                    help="with --gen-file: at most N windows per category, spread evenly over its "
                         "concatenation (default: every full window)")
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
    ap.add_argument("--rms-taps", action="store_true",
                    help="also capture the weightless rms taps L{i}.in.rms.hess / L{i}.mlp_in.rms.hess "
                         "(rms(x) of each text layer's input_layernorm / post_attention_layernorm "
                         "input, no weight) that r4dx-convert --rotate rounds in-projections against; "
                         "listed under hessian.json rms_keys only if their gates pass")
    ap.add_argument("--rms-only", action="store_true",
                    help="capture ONLY the weightless rms taps and merge them into the validated set "
                         "already in --out-dir (same checkpoint and corpus required; --force replaces "
                         "rms entries it already has)")
    ap.add_argument("--code-rev", default=None, metavar="COMMIT",
                    help="read the repo-source windows of the corpus from this commit's tree instead "
                         "of the working tree (for --rms-only against a set captured at an older "
                         "checkout; its code sha256_of_concatenation must match)")
    ap.add_argument("--tokenizer", choices=TOKENIZER_MODES, default=None,
                    help=TOKENIZER_HELP + f" Used for every corpus source. Default: "
                         f"{DEFAULT_TOKENIZER_MODE} for a new capture; with --rms-only, the mode the "
                         f"set records ({LEGACY_TOKENIZER_MODE} for a set that records none, e.g. "
                         f"hessian-v1) -- a different explicit one is refused")
    args = ap.parse_args()

    if args.write_fixture is not None:
        return write_fixture(args.write_fixture)
    if args.regate:
        sys.path.insert(0, str(TOOLS_REF))
        from common import DEFAULT_MODEL_DIR
        return regate(args.out_dir, args.model_dir or DEFAULT_MODEL_DIR)

    # The GPU rule, before anything expensive (common.resolve_device re-checks it). --dry-run never
    # touches the GPU, so it is exempt.
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if not args.dry_run and visible != "1":
        raise SystemExit(
            "[hessian] refusing to run: $env:HIP_VISIBLE_DEVICES must be exactly '1' (only HIP "
            f"device 1, the headless R9700, may be used). Got {visible!r}. (--dry-run needs no GPU.)"
        )
    if args.rms_only:
        return run_rms_only(args)

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

    from common import load_ref_tokenizer

    tokenizer = load_ref_tokenizer(model_dir, args.tokenizer or DEFAULT_TOKENIZER_MODE)
    print(f"[hessian] tokenizer: {tokenizer.mode} ({tokenizer.describe()})")
    t0 = time.perf_counter()
    seqs, sources = build_corpus(args, tokenizer)
    total_tokens = sum(len(s.token_ids) for s in seqs)
    print(f"[hessian] corpus: {len(seqs)} sequences, {total_tokens} tokens "
          f"({time.perf_counter() - t0:.1f}s to tokenize)")
    print_corpus_sources(sources)

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
    # --rms-taps: one weightless rms tap per selected norm-fed post-norm tap (same layer, same keys).
    rms_plans = (build_rms_plans({p.file: p.keys for p in plans if p.scope == "layer"}, n_run,
                                 int(text_config.hidden_size)) if args.rms_taps else [])
    if args.rms_taps and not rms_plans:
        raise SystemExit("[hessian] --rms-taps: --keys/--layers select no norm-fed tap (L{i}.in / "
                         "L{i}.mlp_in)")
    # MTP taps carry T-1 rows per sequence (MtpTaps pairs row i with token i+1), all others T.
    predicted_rows = {p.file: total_tokens - (len(seqs) if p.scope == "mtp" else 0) for p in plans}
    estimate = (sum(hess_file_bytes(p.k) for p in plans) +
                sum(hess_file_bytes(rp.k) for rp in rms_plans) + (1 << 20))

    print(f"[hessian] {len(all_keys)} converter linears -> {len(plans_all)} taps; selected "
          f"{len(plans)} file(s) serving {len(selected_keys)} key(s), text layers 0..{n_run - 1}"
          f"{'' if key_re is None else f', --keys {args.keys!r}'}"
          f"{f'; + {len(rms_plans)} weightless rms tap(s)' if rms_plans else ''}")
    short = [(p.file, predicted_rows[p.file], ROWS_PER_K_MIN * p.k) for p in plans
             if predicted_rows[p.file] < ROWS_PER_K_MIN * p.k]
    if args.dry_run:
        for p in plans:
            print(f"    {p.file:<22} K={p.k:>5} rows~{predicted_rows[p.file]:>7} "
                  f"{hess_file_bytes(p.k) / 2**20:8.1f} MiB  {','.join(p.keys)}")
        for rp in rms_plans:
            print(f"    {rp.file:<22} K={rp.k:>5} rows~{total_tokens:>7} "
                  f"{hess_file_bytes(rp.k) / 2**20:8.1f} MiB  rms_keys: {','.join(rp.keys)}")
    print(f"[hessian] disk estimate {estimate / 2**30:.2f} GiB for {len(plans) + len(rms_plans)} "
          f"file(s)")
    if short:
        print(f"[hessian] WARNING: {len(short)} tap(s) would get fewer than {ROWS_PER_K_MIN} x K "
              f"rows: {short[:6]}")
    anchor = _existing_anchor(args.out_dir)
    free = shutil.disk_usage(anchor).free
    print(f"[hessian] free space at {anchor}: {free / 2**30:.2f} GiB "
          f"(need {1.1 * estimate / 2**30:.2f} GiB = 1.1 x estimate)")
    manifest_path = args.out_dir / MANIFEST_NAME
    existing = None  # the set already in --out-dir: which tokenizer mode it holds (docs/quant2.md 3.4)
    if manifest_path.exists():
        from common import recorded_tokenizer_mode

        try:
            old = recorded_tokenizer_mode(json.loads(manifest_path.read_text(encoding="utf-8"))
                                          .get("corpus", {}).get("tokenizer"))[0]
        except (ValueError, OSError, AttributeError):
            old = "unreadable"
        existing = (f"{manifest_path} already exists (a {old} set; this run would capture "
                    f"{tokenizer.mode})")
    if args.dry_run:
        if existing:
            print(f"[hessian] NOTE: {existing}: the real run needs --force, or another --out-dir")
        print("[hessian] --dry-run: nothing captured, nothing written")
        return 0
    if free < 1.1 * estimate:
        raise SystemExit("[hessian] not enough free space on the target drive -- refusing to start")

    if existing:
        if not args.force:
            raise SystemExit(f"[hessian] {existing}; pass --force to replace the set (the old "
                             f"manifest is deleted before capturing), or another --out-dir")
        manifest_path.unlink()
    # A previous failed run's reports describe .hess files this run is about to overwrite.
    for stale in (FAILED_MANIFEST_NAME, RMS_FAILED_MANIFEST_NAME):
        if (args.out_dir / stale).exists():
            (args.out_dir / stale).unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from common import resolve_device
    from full_logits_golden import StreamingReference

    device = resolve_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hidden_bytes = total_tokens * int(text_config.hidden_size) * 2
    hidden_device = _pick_hidden_device(args, device, hidden_bytes)

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
                     args.draft_head, rms_plans=rms_plans)
    seconds = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"[hessian] capture done: {seconds:.1f}s, peak VRAM {peak:.3f} GiB allocated / "
          f"{reserved:.3f} GiB reserved")

    # The first layer of each type, over the WHOLE stack: a --layers smoke that stops before the
    # first attention layer reports the shared-input check as incomplete.
    shared_expected = [ref.layer_types.index(t) for t in SHARED_GATE if t in ref.layer_types]
    structural = structural_zero_channels(model_dir, {p.file: (p.scope, p.layer, p.module)
                                                      for p in plans if p.file in st.written})
    gates = evaluate_gates(audit, all_keys, plans, st, selected_keys, shared_expected, structural)
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
        "corpus": corpus_record(seqs, sources, args.seq_len, tokenizer),
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
    # --rms-taps: the rms files enter the manifest ("files", "rms_keys", "rms_capture") only if their
    # own gates pass -- whichever manifest name the post-norm gates choose, so --regate keeps them.
    # Otherwise the post-norm set is written without them and the rms report goes to
    # hessian_rms.failed.json (--rms-only can redo just that part).
    rms_keys = None
    rms_ok = True
    if rms_plans:
        rms_gates = evaluate_rms_gates(rms_plans, st.rms_written, args.out_dir,
                                       {f: w["rows"] for f, w in st.written.items()}, st.rms_scale)
        print_rms_gates(rms_gates)
        rms_ok = rms_gates["ok"]
        record = rms_capture_record("tools/reference/hessian_capture.py --rms-taps", rms_plans, st,
                                    rms_gates)
        if rms_ok:
            for rp in rms_plans:
                files[rp.file] = st.rms_written[rp.file]
            rms_keys = rms_keys_of(rms_plans)
            extra["rms_capture"] = record
        else:
            rms_failed = write_rms_failed(args.out_dir, rms_plans, st.rms_written, record)
            print(f"[hessian] RMS GATES FAILED -- the rms files are NOT listed in the manifest; "
                  f"report in {rms_failed}")
    # A failed set is written as hessian.failed.json, which HessianStore never opens: the gate report
    # survives for diagnosis, but `--hessian-dir` on this directory is a "cannot open hessian.json"
    # error rather than a conversion against Hessians the gates just called invalid.
    path = write_manifest(args.out_dir, files, keys, extra,
                          manifest_name=MANIFEST_NAME if gates["ok"] else FAILED_MANIFEST_NAME,
                          rms_keys=rms_keys)
    total_bytes = sum(w["bytes"] for w in files.values())
    print(f"\n[hessian] wrote {len(files)} .hess file(s), {total_bytes / 2**30:.2f} GiB, and {path}")
    print(f"[hessian] tokens={total_tokens} wall={seconds:.1f}s peak={peak:.3f} GiB")
    if smoke:
        print("[hessian] smoke run: lm_head/MTP files written but given no key in the manifest")
    if not gates["ok"]:
        print(f"[hessian] GATES FAILED -- manifest written as {FAILED_MANIFEST_NAME}, not "
              f"{MANIFEST_NAME}; the converter will not load this set")
    return 0 if gates["ok"] and rms_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
