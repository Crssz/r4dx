"""Shared helpers of the Gemma 4 reference package (docs/gemma4-plan.md 6.3).

Everything here is read-only against <models root>\\Huihui-gemma-4-12B-it-abliterated (the checkpoint,
`R4DX_MODEL_DIR` overrides) and <models root>\\Huihui-gemma-4-12B-it-abliterated-tok (google's tokenizer
files over Huihui's tokenizer.json, `R4DX_TOKENIZER_DIR` overrides). Run the scripts with the
reference venv:

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\<script>.py ...

The device rule (CLAUDE.md "use GPU device 1"): GPU work only ever touches the HIP devices that
`$env:R4DX_REF_ALLOWED_DEVICES` lists (default '1': the headless R9700) and
`$env:HIP_VISIBLE_DEVICES` must be set, before the process starts, to a subset of them. The default
behaves exactly like tools/reference/common.py's resolve_device. The override only exists for jobs
the user has launched on purpose, e.g. two generation shards ('0' and '1') or the drafter trainer
('0,1'); setting it never makes a script choose a device by itself.

torch / transformers are imported inside the functions that need them, so `arch`-level and
tokenizer-free users of this module stay light.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

# tools/reference is on sys.path so the model-agnostic helpers of common.py are shared, not forked.
_REF_DIR = Path(__file__).resolve().parents[1]
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from common import (  # noqa: E402,F401  (re-exported)
    MODELS_ROOT,
    ShardIndex,
    save_golden,
    set_seed,
    sha256_bytes,
    sha256_file,
    tensor_manifest_entry,
)
from gemma.arch import GemmaArch  # noqa: E402,F401

DEFAULT_MODEL_DIR = Path(os.environ.get("R4DX_MODEL_DIR", str(MODELS_ROOT / "Huihui-gemma-4-12B-it-abliterated")))
DEFAULT_TOKENIZER_DIR = Path(os.environ.get("R4DX_TOKENIZER_DIR",
                                            str(MODELS_ROOT / "Huihui-gemma-4-12B-it-abliterated-tok")))
#: Derived artifacts of the Gemma track (corpus, kvcalib, kl dumps, hessians, trellis dirs).
GEMMA_OUT_DIR = MODELS_ROOT / "r4dx" / "huihui-gemma"
DEFAULT_ALLOWED_DEVICES = "1"


def free_commit_bytes() -> int | None:
    """System commit charge still available (ullAvailPageFile = commit limit - committed), Windows only."""
    if os.name != "nt":
        return None
    import ctypes

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]

    ms = _MS()
    ms.dwLength = ctypes.sizeof(ms)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        return None
    return int(ms.apf)


def wait_for_commit(need_bytes: int, what: str = "model load", timeout_s: float = 900.0,
                    margin_bytes: int = 4 << 30, poll_s: float = 15.0, free_fn=None) -> bool:
    """Block until free commit >= need_bytes + margin (or timeout). Returns True when there is headroom.

    safetensors' torch path maps the whole checkpoint copy-on-write, which Windows charges to commit in full;
    with too little commit left, torch dereferences the failed mapping and the process dies with 0xC0000005
    (g4-load-crash). Waiting is the only in-process defence for the transformers from_pretrained path."""
    import time

    free_fn = free_fn or free_commit_bytes
    t0 = time.time()
    while True:
        free = free_fn()
        if free is None or free >= need_bytes + margin_bytes:
            return True
        msg = (f"[load-guard] {what}: free commit {free / 2**30:.1f} GiB < need "
               f"{(need_bytes + margin_bytes) / 2**30:.1f} GiB")
        if time.time() - t0 >= timeout_s:
            print(msg + " -- timed out, proceeding (a native crash is possible)", file=sys.stderr, flush=True)
            return False
        print(msg + f" -- waiting {poll_s:.0f}s", file=sys.stderr, flush=True)
        time.sleep(poll_s)


def guarded_from_pretrained(cls, model_dir, retries: int = 1, **kw):
    """`cls.from_pretrained(model_dir, **kw)` after waiting for commit headroom of the checkpoint size; a
    Python-level failure is retried `retries` times with a log line (an access violation cannot be caught)."""
    import time

    mdir = Path(model_dir)
    need = sum(p.stat().st_size for p in mdir.glob("*.safetensors"))
    for attempt in range(retries + 1):
        wait_for_commit(need, what=f"from_pretrained({mdir.name})")
        try:
            return cls.from_pretrained(str(mdir), **kw)
        except (OSError, RuntimeError, MemoryError) as e:
            if attempt >= retries:
                raise
            print(f"[load-guard] from_pretrained failed ({type(e).__name__}: {e}); retry {attempt + 1}/{retries}",
                  file=sys.stderr, flush=True)
            time.sleep(10)
GOLDEN_DIR = _REF_DIR / "golden_out" / "gemma"
KL_CORPUS_DIR = _REF_DIR / "kl_corpus"
PROMPTS_PATH = _REF_DIR.parents[1] / "tools" / "quant2" / "corpus_v2_prompts.json"

#: Token ids of the 12B tokenizer that the scripts name (docs/gemma4-semantics.md, config.json).
BOS_ID = 2
END_OF_TURN_ID = 106
EOS_IDS = (1, 106, 50)          # generation_config.json (docs/gemma4-plan.md 9.4)
THINK_OPEN = "<|channel>thought\n"
THINK_CLOSE = "<channel|>"
TURN_END = "<turn|>"


# --------------------------------------------------------------------------------------------
# Device rule
# --------------------------------------------------------------------------------------------


def allowed_devices() -> set[str]:
    """The HIP devices a GPU run may use: $R4DX_REF_ALLOWED_DEVICES (comma list, default '1'); the
    legacy $R4DX_ALLOW_GPU0=1 adds '0'."""
    raw = os.environ.get("R4DX_REF_ALLOWED_DEVICES", DEFAULT_ALLOWED_DEVICES)
    allowed = {t.strip() for t in raw.split(",") if t.strip()}
    if os.environ.get("R4DX_ALLOW_GPU0") == "1":
        allowed.add("0")
    return allowed


def check_visible_devices(visible: str | None, allowed: set[str] | None = None) -> list[str]:
    """The device list of $HIP_VISIBLE_DEVICES `visible`; raises RuntimeError unless it is set and
    every entry is allowed. Pure (testable without a GPU)."""
    allowed = allowed_devices() if allowed is None else allowed
    parts = [p.strip() for p in (visible or "").split(",") if p.strip()]
    if not parts or not set(parts) <= allowed:
        raise RuntimeError(
            "Refusing to touch the GPU: $env:HIP_VISIBLE_DEVICES must be set, before python starts, to "
            f"a subset of the allowed devices {sorted(allowed)} ($env:R4DX_REF_ALLOWED_DEVICES, default "
            f"'1': only the headless R9700 -- CLAUDE.md 'use GPU device 1'). Got "
            f"HIP_VISIBLE_DEVICES={visible!r}. Set it, or pass --device cpu.")
    return parts


def resolve_device(requested: str):
    """`--device` -> torch.device. 'cpu' never touches a GPU; 'cuda' applies the device rule."""
    import torch

    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu", "hip"):
        check_visible_devices(os.environ.get("HIP_VISIBLE_DEVICES"))
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False: no HIP device is visible to torch "
                               f"(HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r}). "
                               "Is another process holding it?")
        return torch.device("cuda:0")  # index 0 == the first entry of HIP_VISIBLE_DEVICES
    raise ValueError(f"unknown --device {requested!r}, expected 'cpu' or 'cuda'")


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def load_arch(model_dir: Path | str | None = None) -> GemmaArch:
    return GemmaArch.from_model_dir(model_dir or DEFAULT_MODEL_DIR)


def load_generation_config(model_dir: Path | str | None = None) -> dict:
    p = Path(model_dir or DEFAULT_MODEL_DIR) / "generation_config.json"
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sampling_from_generation_config(gen: dict) -> dict:
    """The generation_config.json sampling (temperature 1.0, top_k 64, top_p 0.95 for Huihui)."""
    return {"temperature": float(gen.get("temperature", 1.0)), "top_k": int(gen.get("top_k", 64)),
            "top_p": float(gen.get("top_p", 0.95)), "do_sample": bool(gen.get("do_sample", True))}


def load_configs(model_dir: Path | str | None = None):
    """(Gemma4UnifiedConfig, Gemma4UnifiedTextConfig) from `model_dir`, attention forced to eager:
    the scripts build bare modules (no PreTrainedModel around to choose a backend), and eager is
    plain tensor ops whose intermediates can be hooked."""
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig

    cfg = Gemma4UnifiedConfig.from_pretrained(str(model_dir or DEFAULT_MODEL_DIR))
    force_eager(cfg)
    return cfg, cfg.get_text_config()


def force_eager(cfg) -> None:
    cfg._attn_implementation = "eager"
    cfg.get_text_config()._attn_implementation = "eager"


def perturb_norm_weights(module, seed: int, sigma: float = 0.05) -> int:
    """Seeded noise on every RMSNorm weight under `module` (weight = 1 + sigma * N(0,1)): a random-init
    model has all-ones norm weights, so a missing `* w` in a port could never show. Own generator;
    global RNG untouched. Returns the number of weights perturbed."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    for sub in module.modules():
        if type(sub).__name__.endswith("RMSNorm") and getattr(sub, "with_scale", True) and hasattr(sub, "weight"):
            with torch.no_grad():
                noise = 1.0 + sigma * torch.randn(sub.weight.shape, generator=gen)
                sub.weight.copy_(noise.to(dtype=sub.weight.dtype, device=sub.weight.device))
            n += 1
    return n


# --------------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------------


class GemmaRefTokenizer:
    """The Gemma tokenizer as the reference tools use it.

    - `encode(text)`: `transformers.AutoTokenizer(text, add_special_tokens=False)`: NO BOS is
      added (HF never adds one to raw text: docs/gemma4-semantics.md item 10), special-token
      literals in the text (`<bos>`, `<|turn>`, ...) are recognized as their ids.
    - `encode_raw(text)`: the same through `tokenizers.Tokenizer.from_file(tokenizer.json)`, i.e.
      what r4dx's C++ tokenizer implements. `agrees(text)` compares the two; the Gemma tokenizer.json
      declares no NFC normalizer (Replace ' ' -> U+2581 only), so unlike Qwen nothing is refused.
    - `render_chat` / `encode_chat`: the checkpoint's chat template as text (it emits `<bos>`
      itself) and its ids. `enable_thinking=True` puts `<|think|>` in the system turn; with it False
      the generation prompt ends in an empty thought channel.
    - `with_bos(ids)`: BOS=2 prepended, the KL-token convention (both sides teacher-force BOS first).
    """

    def __init__(self, tok_dir: Path | str | None = None):
        self.tok_dir = Path(tok_dir) if tok_dir is not None else DEFAULT_TOKENIZER_DIR
        self._hf = None
        self._raw = None

    @property
    def hf(self):
        if self._hf is None:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(str(self.tok_dir))
        return self._hf

    @property
    def raw(self):
        if self._raw is None:
            from tokenizers import Tokenizer

            raw = Tokenizer.from_file(str(self.tok_dir / "tokenizer.json"))
            raw.encode_special_tokens = False  # special-token text -> its id (r4dx parse_special=true)
            raw.no_truncation()
            raw.no_padding()
            self._raw = raw
        return self._raw

    @property
    def bos_id(self) -> int:
        return int(self.hf.bos_token_id)

    @property
    def vocab_size(self) -> int:
        return len(self.hf)

    def encode(self, text: str) -> list[int]:
        return list(self.hf(text, add_special_tokens=False, verbose=False)["input_ids"])

    def encode_raw(self, text: str) -> list[int]:
        return list(self.raw.encode(text, add_special_tokens=False).ids)

    def agrees(self, text: str) -> bool:
        return self.encode(text) == self.encode_raw(text)

    def decode(self, ids) -> str:
        return self.hf.decode([int(i) for i in ids], skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)

    def with_bos(self, ids) -> list[int]:
        return [self.bos_id] + [int(i) for i in ids]

    def render_chat(self, messages, **template_kwargs) -> str:
        if "tokenize" in template_kwargs:
            raise TypeError("render_chat always renders text; use encode_chat for ids")
        text = self.hf.apply_chat_template(messages, tokenize=False, **template_kwargs)
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}")
        return text

    def encode_chat(self, messages, **template_kwargs) -> list[int]:
        return self.encode(self.render_chat(messages, **template_kwargs))

    def is_nfc(self, text: str) -> bool:
        return unicodedata.is_normalized("NFC", text)

    def describe(self) -> str:
        return f"{self.tok_dir / 'tokenizer.json'} (transformers AutoTokenizer == tokenizers, BOS explicit)"

    def provenance(self) -> dict:
        from importlib import metadata

        def ver(pkg):
            try:
                return metadata.version(pkg)
            except metadata.PackageNotFoundError:
                return None

        tj = self.tok_dir / "tokenizer.json"
        manifest = self.tok_dir / "MANIFEST.json"
        return {"tokenizer_dir": str(self.tok_dir),
                "tokenizer_json_sha256": sha256_file(tj) if tj.is_file() else None,
                "tokenizer_manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file()
                else None,
                "bos_rule": "BOS id 2 prepended explicitly to every KL segment (HF adds none to raw text); "
                            "chat-templated text carries its own <bos> literal",
                "special_tokens": "special-token literals in the text are recognized as their ids",
                "normalization": "none beyond the tokenizer.json Replace(' ' -> U+2581); no NFC",
                "tokenizers_version": ver("tokenizers"), "transformers_version": ver("transformers")}


class StubTokenizer:
    """Byte-level stand-in used by the `--tiny` CPU smoke paths (a tiny random model has a tiny
    vocabulary, real Gemma ids would not fit). Same surface as GemmaRefTokenizer, ids in [4, vocab)."""

    bos_id = BOS_ID

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.tok_dir = Path("<stub>")

    def encode(self, text: str) -> list[int]:
        return [4 + (b % (self.vocab_size - 4)) for b in text.encode("utf-8")]

    def encode_raw(self, text: str) -> list[int]:
        return self.encode(text)

    def agrees(self, text: str) -> bool:
        return True

    def decode(self, ids) -> str:
        return "".join(f"<{int(i)}>" if int(i) < 4 else chr(32 + (int(i) - 4) % 95) for i in ids)

    def with_bos(self, ids) -> list[int]:
        return [self.bos_id] + [int(i) for i in ids]

    def render_chat(self, messages, enable_thinking: bool = False, add_generation_prompt: bool = False,
                    **_ignored) -> str:
        out = "<bos>"
        if enable_thinking:
            out += "<|think|>\n"
        for m in messages:
            out += f"<|turn>{'model' if m['role'] == 'assistant' else m['role']}\n{m['content']}<turn|>\n"
        if add_generation_prompt:
            out += "<|turn>model\n" + ("" if enable_thinking else "<|channel>thought\n<channel|>")
        return out

    def encode_chat(self, messages, **kw) -> list[int]:
        text = self.render_chat(messages, **kw)
        body = text[len("<bos>"):] if text.startswith("<bos>") else text
        return [BOS_ID] + self.encode(body)

    def is_nfc(self, text: str) -> bool:
        return True

    def describe(self) -> str:
        return "stub byte tokenizer (tiny smoke only)"

    def provenance(self) -> dict:
        return {"stub": True}


def load_tokenizer(tok_dir: Path | str | None = None, tiny_vocab: int | None = None):
    """GemmaRefTokenizer, or the StubTokenizer when `tiny_vocab` (the --tiny smoke) is given."""
    return StubTokenizer(tiny_vocab) if tiny_vocab else GemmaRefTokenizer(tok_dir)


def split_model_output(text: str) -> dict[str, Any]:
    """A decoded (special tokens kept) model turn -> {reasoning, content, closed_thought}.

    Output shape: `<|channel>thought\\n...<channel|>answer<turn|>` (thinking on) or just
    `answer<turn|>` (thinking off: the generation prompt already closed an empty thought channel).
    """
    for end in ("<eos>", TURN_END, "<|tool_response>"):
        cut = text.find(end)
        if cut >= 0:
            text = text[:cut]
    reasoning, closed = None, True
    if text.startswith(THINK_OPEN):
        body = text[len(THINK_OPEN):]
        j = body.find(THINK_CLOSE)
        if j < 0:
            reasoning, text, closed = body, "", False
        else:
            reasoning, text = body[:j], body[j + len(THINK_CLOSE):]
    return {"reasoning": reasoning, "content": text.strip(), "closed_thought": closed}


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
