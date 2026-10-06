"""Shared helpers for r4dx's Python reference/validation tooling.

Everything under tools/reference/** is read-only against:
  - the reference Python (torch + transformers + numpy): $env:R4DX_REFERENCE_VENV's
    Scripts\\python.exe when that is set, else python on PATH -- never pip/uv install into it, we
    only import from it
  - <models root>\\Huihui-Qwen3.8-27B-abliterated        (the checkpoint: DEFAULT_MODEL_DIR; shards may
    still be downloading, so every tensor read here is allowed to fail and fall back to random init).
    The base Qwen3.8-27B checkpoint (C:\\AI\\models\\Qwen3.8-27B) was retired on 2026-09-29; the Huihui
    fine-tune's tokenizer / config files are byte-identical to it (docs/huihui.md), only layers 17..51's
    o_proj / out_proj / down_proj weights differ

Run these scripts with that interpreter, e.g.:
    python tools\\reference\\layer_golden.py

torch is imported inside the functions that need it, so the tokenizer helpers below (and the
constants) can be imported by tools that must not pull torch in (hessian_capture.py's
`--write-fixture`, tools/quant2/test_gen_corpus.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Root of every model / checkpoint / derived artifact: $env:R4DX_MODELS_ROOT, default E:\models.
#: Every default path in tools/ and tests/reference/ is built from it (never a literal drive path).
MODELS_ROOT = Path(os.environ.get("R4DX_MODELS_ROOT", r"E:\models"))
DEFAULT_MODEL_DIR = MODELS_ROOT / "Huihui-Qwen3.8-27B-abliterated"


def same_path(a: Any, b: Any) -> bool:
    """True when `a` and `b` name the same location, junction/symlink- and case-aware (so a path
    recorded through a junction alias of the models root still matches the real location). Falls back to a
    normalized string comparison when a path cannot be resolved. None only equals None."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (OSError, RuntimeError, ValueError):
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
#: The reference venv when $env:R4DX_REFERENCE_VENV names one, else None (python on PATH).
DEFAULT_REFERENCE_VENV = (Path(os.environ["R4DX_REFERENCE_VENV"])
                          if os.environ.get("R4DX_REFERENCE_VENV") else None)


# --------------------------------------------------------------------------------------------
# Tokenization: canonical vs AutoTokenizer (docs/quant2.md 3.4)
# --------------------------------------------------------------------------------------------

#: `--tokenizer` of every tool here that turns text into token ids:
#:
#:   canonical  `tokenizers.Tokenizer.from_file(<model_dir>/tokenizer.json)`: the checkpoint's own
#:              on-disk declaration. It is what r4dx's C++ tokenizer produces by default
#:              (src/tokenizer/bpe_tokenizer.cpp, tests/tokenizer/golden.json), so it is what
#:              r4dx-server feeds the model and the tokenization the model generates in.
#:   hf-auto    `transformers.AutoTokenizer`: for this checkpoint a Qwen2Tokenizer whose __init__
#:              replaces tokenizer.json's pre-tokenizer regex (`[\p{L}\p{M}]+`) with an older one
#:              without `\p{M}`, so Thai, Lao, Khmer, Devanagari, Hebrew and diacritic Arabic
#:              combining marks are split off their base letters (transformers 5.17.0; see
#:              tools/tok_ref/gen_golden.py). Kept ONLY to reproduce what was made with it: every
#:              artifact from before 2026-09-26 (hessian-v1, the imatrix, the kvcalib json,
#:              kl_corpus/tokens.json).
#:
#: English, code and the other scripts tokenize identically either way; Thai does not (corpus v2's
#: 84,990 generated Thai tokens become 166,325 through AutoTokenizer).
#:
#: Two limits of "canonical == what r4dx produces":
#:   - NFC. tokenizer.json declares an NFC normalizer, which `tokenizers` applies and r4dx does not
#:     (src/tokenizer/tokenizer.h, KNOWN GAP). canonical therefore REFUSES text that is not already
#:     NFC (`encode` raises ValueError) instead of returning ids r4dx would not produce. No corpus in
#:     use has such text.
#:   - Special-token literals. Written in the text, they are recognized as their ids: r4dx's chat
#:     path (parse_special=true). r4dx-server's raw /v1/completions prompt is encoded with
#:     parse_special=false, which spells such a literal out in ordinary BPE pieces. This only
#:     matters for raw text that contains them: hessian_capture's repo-code concatenation does (80
#:     at 714955f), the prose, calib and KL files do not.
TOKENIZER_MODES = ("canonical", "hf-auto")
DEFAULT_TOKENIZER_MODE = "canonical"
#: What a provenance record WITHOUT a tokenizer mode means: it was written before the switch, and
#: every such tool tokenized with AutoTokenizer.
LEGACY_TOKENIZER_MODE = "hf-auto"
#: What `tokens_file_tokenizer_mode` returns for a tokens file whose mode it cannot establish.
UNKNOWN_TOKENIZER_MODE = "unknown"

#: "This place is not a peaceful city". Nearly every syllable carries a combining mark, which
#: transformers 5.17's AutoTokenizer splits off its consonant: 6 ids canonical, 14 hf-auto.
THAI_PROBE = "ที่นี่ไม่ใช่เมืองแห่งความสงบ"

#: RefTokenizer.describe() in canonical mode is "<model_dir>/tokenizer.json" + this.
_CANONICAL_DESCRIBE_SUFFIX = " (tokenizers.Tokenizer.from_file, canonical)"

#: Tokens files that predate the "tokenizer_mode" field and hold AutoTokenizer ids, by
#: `tokens_segments_sha256`. Only these are read as hf-auto; see tokens_file_tokenizer_mode.
LEGACY_HF_AUTO_TOKENS_FILES = {
    "bb8b71693e868f57e434e3b576dfa31bcb5760042fa69acebb630de637800c9d":
        "tools/reference/kl_corpus/tokens.json",
}

TOKENIZER_HELP = (
    "how text becomes token ids (docs/quant2.md 3.4). canonical: the checkpoint's tokenizer.json "
    "through the `tokenizers` library -- what r4dx / r4dx-server produce and what the model "
    "generates; it refuses text that is not Unicode NFC (r4dx applies no NFC). hf-auto: "
    "transformers' AutoTokenizer, which in transformers 5.17 splits Thai (and other combining-mark "
    "scripts) off their base letters; only to reproduce artifacts made before 2026-09-26, and "
    "refused under a transformers that does not split. Special tokens written in the text "
    "(<|im_start|>, <think>, ...) are recognized as their ids either way (r4dx's chat path, "
    "parse_special=true); none are added.")


def _package_version(pkg: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _load_raw_tokenizer(model_dir: Path):
    """tokenizer.json through the `tokenizers` library, special-token text recognized as its id."""
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    # The backend's `encode_special_tokens` is the OPPOSITE polarity of r4dx's `parse_special`:
    # True pushes special-token text through the ordinary BPE (r4dx parse_special=false); False,
    # its default, recognizes it as its id (parse_special=true). See tools/tok_ref/gen_golden.py
    # add_encode().
    raw.encode_special_tokens = False
    raw.no_truncation()
    raw.no_padding()
    return raw


def check_hf_auto_splits(hf, raw) -> tuple[int, int]:
    """hf-auto exists to reproduce transformers 5.17.0's AutoTokenizer ids, which split combining
    marks. Other transformers versions differ: 5.3.0's AutoTokenizer tokenizes canonically. Raises
    RuntimeError when `hf` (an AutoTokenizer) encodes THAI_PROBE exactly as `raw` (tokenizer.json)
    does, so an hf-auto label can never sit on canonical ids. Returns (canonical, hf) id counts."""
    c = list(raw.encode(THAI_PROBE, add_special_tokens=False).ids)
    h = list(hf(THAI_PROBE, add_special_tokens=False)["input_ids"])
    if h == c:
        raise RuntimeError(
            f"--tokenizer hf-auto: this AutoTokenizer ({type(hf).__name__}, transformers "
            f"{_package_version('transformers')}) tokenizes the Thai probe exactly as tokenizer.json "
            f"does ({len(c)} ids), so it cannot reproduce the legacy AutoTokenizer ids (transformers "
            f"5.17.0 splits the probe into 14; docs/quant2.md 3.4). Run under the reference venv "
            f"(transformers 5.17.0), or use --tokenizer canonical.")
    return len(c), len(h)


def _non_nfc_error(text: str) -> ValueError:
    nfc = unicodedata.normalize("NFC", text)
    i = len(os.path.commonprefix([text, nfc]))
    return ValueError(
        f"canonical tokenization refuses text that is not Unicode NFC (first difference at "
        f"character {i} of {len(text)}: {text[max(0, i - 12):i + 12]!r}). tokenizer.json normalizes "
        f"to NFC and r4dx / r4dx-server do not (src/tokenizer/tokenizer.h, KNOWN GAP), so no single "
        f"id sequence is 'what r4dx produces' for it. NFC-normalize the source first "
        f"(unicodedata.normalize('NFC', text)).")


class RefTokenizer:
    """The checkpoint's tokenizer as every tool in this directory uses it.

    - `encode(text)` -> ids: nothing ADDED (no BOS/EOS -- the checkpoint adds none anyway), and the
      special/added tokens written in the text (`<|im_start|>`, `<|im_end|>`, `<think>`, ...)
      RECOGNIZED as their ids -- r4dx's `Tokenizer::encode(text, parse_special=true)`, which is what
      a rendered chat needs, and what AutoTokenizer's `__call__` did by default. (r4dx-server
      encodes a raw /v1/completions prompt with parse_special=false instead; the two differ only
      on text that contains such literals.) canonical refuses text that is not NFC (ValueError).
    - `decode(ids)`: the exact inverse (special tokens kept, no clean-up of spaces).
    - `render_chat(messages, **template_kwargs)`: the checkpoint's chat template as TEXT, through
      transformers' `apply_chat_template(tokenize=False)` in both modes -- rendering only assembles
      a string and is unaffected by the pre-tokenizer bug (gen_golden.py renders the same way).
      `encode_chat` is that text through `encode`, i.e. canonically in canonical mode.
    - `provenance()`: the record an output file carries (`mode` first).

    `mode` is one of TOKENIZER_MODES. hf-auto reproduces the old call exactly
    (`AutoTokenizer(text, add_special_tokens=False)`), and refuses to load under a transformers whose
    AutoTokenizer does not split combining marks (check_hf_auto_splits).
    """

    def __init__(self, model_dir: Path | str | None = None, mode: str = DEFAULT_TOKENIZER_MODE):
        if mode not in TOKENIZER_MODES:
            raise ValueError(f"tokenizer mode {mode!r}: expected one of {list(TOKENIZER_MODES)}")
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self.mode = mode
        self._hf = None
        self._raw = None
        if mode == "canonical":
            self._raw = _load_raw_tokenizer(self.model_dir)
        else:
            # Loaded now, so a missing checkpoint or a non-splitting transformers fails here and
            # not mid-run.
            check_hf_auto_splits(self.hf, _load_raw_tokenizer(self.model_dir))

    @property
    def hf(self):
        """The transformers AutoTokenizer (loaded on first use): the chat template renderer in both
        modes, and the encoder/decoder in hf-auto mode."""
        if self._hf is None:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(str(self.model_dir))
        return self._hf

    def encode(self, text: str) -> list[int]:
        if self._raw is not None:
            if not unicodedata.is_normalized("NFC", text):
                raise _non_nfc_error(text)
            return list(self._raw.encode(text, add_special_tokens=False).ids)
        # verbose=False: the calibration tools tokenize whole files and concatenations far longer
        # than the model's maximum length on purpose (only windows of them are run); the ids are
        # the same as without it.
        return list(self.hf(text, add_special_tokens=False, verbose=False)["input_ids"])

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids]
        if self._raw is not None:
            return self._raw.decode(ids, skip_special_tokens=False)
        return self.hf.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    @property
    def chat_template(self) -> str | None:
        tpl = getattr(self.hf, "chat_template", None)
        return tpl if isinstance(tpl, str) else None

    def render_chat(self, messages, **template_kwargs) -> str:
        """`apply_chat_template(messages, tokenize=False, **template_kwargs)` -- e.g.
        `add_generation_prompt`, `enable_thinking`, `tools`, `chat_template`."""
        if "tokenize" in template_kwargs:
            raise TypeError("render_chat always renders text; use encode_chat for ids")
        text = self.hf.apply_chat_template(messages, tokenize=False, **template_kwargs)
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, "
                            f"not the rendered string")
        return text

    def encode_chat(self, messages, **template_kwargs) -> list[int]:
        return self.encode(self.render_chat(messages, **template_kwargs))

    def describe(self) -> str:
        """One line naming the tokenizer. hf-auto keeps the bare checkpoint directory that
        make_tokens_json.py always wrote into a tokens file's "tokenizer" field."""
        if self.mode == "canonical":
            return f"{self.model_dir / 'tokenizer.json'}{_CANONICAL_DESCRIBE_SUFFIX}"
        return str(self.model_dir)

    def provenance(self) -> dict:
        tj = self.model_dir / "tokenizer.json"
        if self.mode == "canonical":
            impl = "tokenizers.Tokenizer.from_file(tokenizer.json) -- the on-disk pre-tokenizer, as r4dx"
            norm = ("input must already be NFC (refused otherwise): tokenizer.json declares NFC, r4dx "
                    "applies none")
        else:
            impl = ("transformers.AutoTokenizer (Qwen2Tokenizer: its __init__ replaces tokenizer.json's "
                    "pre-tokenizer and splits combining marks) -- legacy, reproduction only")
            norm = "whatever this AutoTokenizer applies (legacy)"
        return {"mode": self.mode, "implementation": impl, "tokenizer_json": str(tj),
                "tokenizer_json_sha256": sha256_file(tj) if tj.is_file() else None,
                "special_tokens": "none added; special tokens in the text recognized as their ids "
                                  "(r4dx's chat path, parse_special=true; a raw /v1/completions "
                                  "prompt is parse_special=false)",
                "normalization": norm,
                "thai_probe_ids": len(self.encode(THAI_PROBE)),
                "chat_render": "transformers apply_chat_template(tokenize=False), then encoded as above",
                "tokenizers_version": _package_version("tokenizers"),
                "transformers_version": _package_version("transformers")}


def load_ref_tokenizer(model_dir: Path | str | None = None, mode: str | None = None) -> RefTokenizer:
    """`RefTokenizer(model_dir, mode)`, `mode` None meaning DEFAULT_TOKENIZER_MODE."""
    return RefTokenizer(model_dir, DEFAULT_TOKENIZER_MODE if mode is None else mode)


def recorded_tokenizer_mode(record) -> tuple[str, bool]:
    """(mode, whether it was recorded) of a provenance field written by `RefTokenizer.provenance()`
    (a dict with "mode"), or a bare mode string. Anything else -- None, i.e. no such field -- is a
    record from before the switch, whose tools all used AutoTokenizer: LEGACY_TOKENIZER_MODE."""
    mode = record.get("mode") if isinstance(record, dict) else record
    if mode is None:
        return LEGACY_TOKENIZER_MODE, False
    if mode not in TOKENIZER_MODES:
        raise ValueError(f"recorded tokenizer mode {mode!r} is not one of {list(TOKENIZER_MODES)}")
    return mode, True


def tokens_segments_sha256(doc: dict) -> str:
    """sha256 of a tokens file's segments as compact JSON `[[name, [ids...]], ...]`: what
    LEGACY_HF_AUTO_TOKENS_FILES identifies a file by (whitespace and other fields do not count)."""
    segs = [[s["name"], [int(t) for t in s["token_ids"]]] for s in doc.get("segments", [])]
    return sha256_bytes(json.dumps(segs, separators=(",", ":")).encode("utf-8"))


def tokens_file_tokenizer_mode(doc: dict) -> str:
    """The tokenizer mode of a KL tokens file (docs/validation.md's shared format): one of
    TOKENIZER_MODES, or UNKNOWN_TOKENIZER_MODE. Nothing is guessed from a substring:

    1. its "tokenizer_mode" field (make_tokens_json.py writes one since 2026-09-26);
    2. r4dx-cli --dump-token-ids's exact shape -- only "tokenizer" and "segments", one segment named
       "cli" (src/cli/main.cpp) -- is canonical: r4dx's own C++ tokenizer, which r4dx-cli always
       loads with its default (on-disk) pre-tokenizer;
    3. a "tokenizer" string of RefTokenizer.describe()'s exact canonical form (the lead's
       kl_corpus/tokens_thai_canon.json) is canonical;
    4. a legacy file identified by its token ids (LEGACY_HF_AUTO_TOKENS_FILES: kl_corpus/tokens.json)
       is hf-auto;
    5. anything else is unknown -- e.g. a make_tokens_json file from before the field existed,
       which may hold either."""
    if doc.get("tokenizer_mode") is not None:
        return recorded_tokenizer_mode(doc["tokenizer_mode"])[0]
    segments = doc.get("segments")
    if set(doc) == {"tokenizer", "segments"} and isinstance(segments, list) and \
            [s.get("name") for s in segments] == ["cli"]:
        return "canonical"
    tokenizer = doc.get("tokenizer")
    if isinstance(tokenizer, str) and tokenizer.endswith("tokenizer.json" + _CANONICAL_DESCRIBE_SUFFIX):
        return "canonical"
    if isinstance(segments, list) and tokens_segments_sha256(doc) in LEGACY_HF_AUTO_TOKENS_FILES:
        return LEGACY_TOKENIZER_MODE
    return UNKNOWN_TOKENIZER_MODE


def refuse_tokenizer_mode_change(tag: str, what: str, old_modes, new_mode: str,
                                 force: bool) -> None:
    """The overwrite guard of every tool whose default output path holds a pre-switch (hf-auto)
    artifact: refuse (SystemExit) to replace or extend `what`, whose token ids were made in
    `old_modes` (UNKNOWN_TOKENIZER_MODE: cannot tell), with `new_mode` ones -- unless `force`."""
    old = sorted(set(old_modes))
    if force or all(m == new_mode for m in old):
        return
    known = [m for m in old if m in TOKENIZER_MODES]
    how = (f"Pass --tokenizer {known[0]} to match it, " if len(old) == 1 and known else "Pass ")
    raise SystemExit(
        f"[{tag}] {what} holds {' + '.join(old)} token ids and this run would write {new_mode} "
        f"ones (docs/quant2.md 3.4). {how}--force to replace it anyway, or another --out.")


def resolve_device(requested: str) -> torch.device:  # noqa: F821 (annotations are lazy)
    """Resolve a --device argument.

    Enforces this repo's GPU rule: GPU work only ever touches HIP device 1 (the headless R9700),
    selected by setting $env:HIP_VISIBLE_DEVICES='1' *before* the process starts (torch/HIP reads
    it at import/init time). Once that's set, torch's ROCm build sees exactly one device at CUDA
    index 0, which is physical device 1.

    Opt-in exception: with $env:R4DX_ALLOW_GPU0='1' (set only when the user has granted device 0
    for the run, e.g. two independent reference jobs in parallel), HIP_VISIBLE_DEVICES='0' is
    accepted too. Device 0 drives the desktop; torch's kernels are short, so no TDR risk is known.
    """
    import torch

    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu", "hip"):
        visible = os.environ.get("HIP_VISIBLE_DEVICES")
        allow0 = os.environ.get("R4DX_ALLOW_GPU0") == "1"
        if visible != "1" and not (allow0 and visible == "0"):
            raise RuntimeError(
                "Refusing to touch the GPU: $env:HIP_VISIBLE_DEVICES must be '1' (only HIP device "
                "1, the headless R9700, may be used -- see CLAUDE.md 'use GPU device 1'). Got "
                f"HIP_VISIBLE_DEVICES={visible!r}. Set it before launching python, or pass "
                "--device cpu."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False -- no HIP/ROCm device visible to torch even "
                "though HIP_VISIBLE_DEVICES=1. Is another process holding the device?"
            )
        return torch.device("cuda:0")  # index 0 == physical device 1 once HIP_VISIBLE_DEVICES=1
    raise ValueError(f"unknown --device {requested!r}, expected 'cpu' or 'cuda'")


def set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_ST_DTYPES = {  # safetensors dtype tag -> torch dtype attribute name
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64", "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}


def raw_safetensors_header(path) -> tuple[dict, int]:
    """(header dict without __metadata__, data-section offset) of one .safetensors file, by plain file reads."""
    import struct

    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n <= 0 or n > 100_000_000:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        hdr = json.loads(fh.read(n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def raw_safetensors_read(path, name: str, header: tuple[dict, int] | None = None,
                         rows: tuple[int, int] | None = None) -> "torch.Tensor":
    """Read one tensor (or rows [a, b) of a >=1-D tensor) with seek+readinto into a fresh buffer: no mmap.

    Why this exists (docs: g4-load-crash): safetensors 0.8's torch path maps the WHOLE file with
    torch.UntypedStorage.from_file(shared=False), a copy-on-write view. On Windows that is PAGE_WRITECOPY and
    is charged against the system commit limit in full (23.9 GB for Gemma's single model.safetensors) on every
    open. When commit is near the limit (other jobs, big pagefile use) the mapping comes back unusable and
    torch dereferences it in Tensor::item<uint8> (0xC0000005 inside the Rust loader). Reading the bytes
    directly needs commit only for the tensor itself.
    """
    import math

    import torch

    hdr, off = header if header is not None else raw_safetensors_header(path)
    ent = hdr[name]
    dt = getattr(torch, _ST_DTYPES[ent["dtype"]])
    shape = list(ent["shape"])
    lo, hi = ent["data_offsets"]
    esz = torch.empty(0, dtype=dt).element_size()
    start = off + lo
    if rows is not None:
        if not shape:
            raise ValueError(f"{name}: scalar tensor cannot be row-sliced")
        r0, r1 = max(0, rows[0]), min(shape[0], rows[1])
        inner = math.prod(shape[1:])
        start += r0 * inner * esz
        nbytes = max(0, r1 - r0) * inner * esz
        shape = [max(0, r1 - r0)] + shape[1:]
    else:
        nbytes = hi - lo
    if nbytes == 0:
        return torch.empty(shape, dtype=dt)
    buf = bytearray(nbytes)
    with open(path, "rb", buffering=0) as fh:
        fh.seek(start)
        mv, got = memoryview(buf), 0
        while got < nbytes:
            n = fh.readinto(mv[got:got + (1 << 28)])
            if not n:
                raise EOFError(f"{path}: {name}: short read ({got}/{nbytes} bytes)")
            got += n
    return torch.frombuffer(buf, dtype=dt).reshape(shape)


@dataclass
class ShardIndex:
    """Lazy per-tensor lookup across a HF safetensors shard set, via model.safetensors.index.json.

    Every read is allowed to fail (shard not downloaded yet, or a partially-written file) --
    callers should treat any exception from get_tensor/get_row_slice as "not available" and fall
    back to random init, not as a hard error.
    """

    model_dir: Path
    weight_map: dict[str, str] = field(default_factory=dict)
    single_file: Path | None = None  # set if the checkpoint is a single model.safetensors, no index
    _hdr: tuple | None = None  # cached (header, data offset) of single_file

    @classmethod
    def load(cls, model_dir: Path) -> "ShardIndex":
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            return cls(model_dir=model_dir, weight_map=weight_map)
        single = model_dir / "model.safetensors"
        if single.exists():
            hdr = raw_safetensors_header(single)  # no mmap: see raw_safetensors_read
            weight_map = {k: "model.safetensors" for k in hdr[0]}
            return cls(model_dir=model_dir, weight_map=weight_map, single_file=single, _hdr=hdr)
        return cls(model_dir=model_dir, weight_map={})

    def available(self, name: str) -> bool:
        shard = self.weight_map.get(name)
        if shard is None:
            return False
        return (self.model_dir / shard).exists()

    def names_with_prefix(self, prefix: str) -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]

    def get_tensor(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[name]
        if self.single_file is not None:  # Gemma's one huge file: plain reads, never a whole-file mmap
            return raw_safetensors_read(self.single_file, name, self._hdr)
        # .clone(): this venv's safetensors/torch build returns a tensor whose storage aliases the
        # `safe_open` context manager's own mmap -- once `with` exits and unmaps the file, that
        # storage is dangling. Reading a handful of tensors (layer_golden.py's ~14-20 per component)
        # never surfaced this because Python's GC/OS page cache happened to keep the mapping alive
        # long enough; a tight loop over a few hundred tensors (tools/reference/vision_golden.py's
        # ~330 vision.* weights) reproducibly crashes the interpreter with SIGSEGV/access-violation,
        # or worse, silently reads whatever now occupies that address range instead. Found and fixed
        # 2026-09-20 while building vision_golden.py -- see docs/status.md's vision-tower entry.
        with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as f:
            return f.get_tensor(name).clone()

    def get_row_slice(self, name: str, start: int, stop: int) -> torch.Tensor:
        """Read rows [start:stop) of a 2D tensor without materializing the whole thing."""
        from safetensors import safe_open

        shard = self.weight_map[name]
        if self.single_file is not None:
            return raw_safetensors_read(self.single_file, name, self._hdr, rows=(start, stop))
        with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as f:
            sl = f.get_slice(name)
            return sl[start:stop, :].clone()  # same dangling-mmap risk as get_tensor above


def get_tensors_grouped(index: ShardIndex, names: list[str]) -> dict[str, torch.Tensor]:
    """`ShardIndex.get_tensor` for many names, opening each shard file once (a single-file checkpoint
    included). The same function as full_logits_golden.get_tensors_grouped, here so the Gemma tools do not
    need full_logits_golden's Qwen reference just to read weights."""
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for n in names:
        if n not in index.weight_map:
            raise KeyError(f"{n!r} is not in this checkpoint's weight map")
        by_shard.setdefault(index.weight_map[n], []).append(n)
    out: dict[str, torch.Tensor] = {}
    if index.single_file is not None:
        return {n: index.get_tensor(n) for n in names}
    for shard, shard_names in by_shard.items():
        with safe_open(str(index.model_dir / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                out[n] = f.get_tensor(n).clone()  # clone inside the `with`: see ShardIndex.get_tensor
    return out


def load_module_state(
    module: torch.nn.Module, index: ShardIndex, hf_prefix: str
) -> tuple[bool, list[str], str | None]:
    """Load `module`'s state dict from the shard index under `hf_prefix`, in place.

    Returns (loaded, missing_keys, error). `module` is left with whatever it was constructed with
    (the caller's seeded random init) unless every key loads cleanly.
    """
    sd = module.state_dict()
    missing = [k for k in sd if not index.available(hf_prefix + k)]
    if missing:
        return False, missing, None
    try:
        new_sd = {k: index.get_tensor(hf_prefix + k) for k in sd}
        module.load_state_dict(new_sd)
    except Exception as exc:  # shard present but unreadable/partial -- fall back, don't crash
        return False, list(sd.keys()), f"{type(exc).__name__}: {exc}"
    return True, [], None


def save_golden(path: Path, tensors: dict[str, torch.Tensor], extra_meta: dict[str, Any] | None = None) -> None:
    from safetensors.torch import save_file

    # .clone() (not just .contiguous(), which is a no-op on an already-contiguous *view*) so no two
    # entries alias the same storage -- safetensors refuses to save shared-memory tensors.
    cpu_tensors = {k: v.detach().to("cpu").contiguous().clone() for k, v in tensors.items()}
    meta = {"r4dx_reference_tool": "r4dx tools/reference"}
    if extra_meta:
        meta.update({k: json.dumps(v) if not isinstance(v, str) else v for k, v in extra_meta.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cpu_tensors, str(path), metadata=meta)


def tensor_manifest_entry(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}


def load_text_config(model_dir: Path):
    """Load Qwen3_5Config from the reference transformers install and return (config, text_config).

    Forces `_attn_implementation = "eager"` -- these scripts construct bare `nn.Module`s directly
    (not through `from_pretrained`/`AutoModel`), so there's no `PreTrainedModel.__init__` around to
    pick an attention backend, and eager is what we want anyway: plain tensor ops with no
    sdpa/flash-attn dependency, whose intermediates we can hook and monkeypatch.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    config = Qwen3_5Config.from_pretrained(str(model_dir))
    config.text_config._attn_implementation = "eager"
    return config, config.text_config


def force_eager(config) -> None:
    config._attn_implementation = "eager"
