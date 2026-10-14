"""Shared helpers of the LFM2.5-24B-A2B reference package (docs/lfm2-plan.md 6.2).

The checkpoint (and its tokenizer files: tokenizer.json, tokenizer_config.json, chat_template.jinja) live in
one directory, `E:\\models\\LiquidAI\\LFM2.5-24B-A2B` (`R4DX_MODEL_DIR` overrides).  Run the scripts with the
reference venv (`-I` when they read downloaded files):

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\<script>.py ...

Device rule (same as the Gemma package): GPU work only ever touches HIP devices listed in
`$env:R4DX_REF_ALLOWED_DEVICES` (default '1'); every CPU-only script here sets HIP_VISIBLE_DEVICES=-1 itself.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

_REF_DIR = Path(__file__).resolve().parents[1]
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from common import ShardIndex, sha256_bytes, sha256_file  # noqa: E402,F401  (re-exported; torch-free import)

DEFAULT_MODEL_DIR = Path(os.environ.get("R4DX_MODEL_DIR", r"E:\models\LiquidAI\LFM2.5-24B-A2B"))
DEFAULT_OUT_DIR = Path(os.environ.get("R4DX_LFM2_OUT_DIR", r"E:\models\r4dx\lfm2"))
KL_CORPUS_DIR = _REF_DIR / "kl_corpus"
GOLDEN_DIR = _REF_DIR / "golden_out" / "lfm2"

#: ids of the real tokenizer (docs/lfm2-semantics.md): the embedding has 128000 rows but the highest added
#: token is 125016, so logit columns >= REAL_VOCAB are padding and are masked everywhere (decision 2026-10-14).
BOS_ID = 124894          # <|startoftext|>
PAD_ID = 124893
IM_START_ID = 124899
IM_END_ID = 124900       # eos
THINK_OPEN_ID = 124901
THINK_CLOSE_ID = 124902
REAL_VOCAB = 125017
LOGPROB_CLAMP = -1e4


def token_ids_sha256(token_ids) -> str:
    """Byte-identical to kl_report.token_ids_sha256 / the C++ TokenIdsSha256."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def allowed_devices() -> set[str]:
    raw = os.environ.get("R4DX_REF_ALLOWED_DEVICES", "1")
    return {t.strip() for t in raw.split(",") if t.strip()}


def force_eager(cfg) -> None:
    cfg._attn_implementation = "eager"


class Lfm2RefTokenizer:
    """The LFM tokenizer as the reference tools use it.

    - `encode(text)`: transformers AutoTokenizer, add_special_tokens=False (the post-processor adds no BOS; the
      chat template emits `<|startoftext|>` itself); special-token literals in the text map to their ids.
    - `encode_raw(text)`: the same through `tokenizers.Tokenizer.from_file(tokenizer.json)` (what r4dx's C++
      tokenizer implements).  `agrees(text)` compares both.
    - `render_chat` / `encode_chat`: the checkpoint's chat_template.jinja (text / ids).
    """

    def __init__(self, tok_dir: Path | str | None = None):
        self.tok_dir = Path(tok_dir) if tok_dir is not None else DEFAULT_MODEL_DIR
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
            raw.encode_special_tokens = False
            raw.no_truncation()
            raw.no_padding()
            self._raw = raw
        return self._raw

    @property
    def bos_id(self) -> int:
        return int(self.hf.bos_token_id)

    def encode(self, text: str) -> list[int]:
        return list(self.hf(text, add_special_tokens=False, verbose=False)["input_ids"])

    def encode_raw(self, text: str) -> list[int]:
        return list(self.raw.encode(text, add_special_tokens=False).ids)

    def agrees(self, text: str) -> bool:
        return self.encode(text) == self.encode_raw(text)

    def decode(self, ids) -> str:
        return self.hf.decode([int(i) for i in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)

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
        ct = self.tok_dir / "chat_template.jinja"
        return {"tokenizer_dir": str(self.tok_dir),
                "tokenizer_json_sha256": sha256_file(tj) if tj.is_file() else None,
                "chat_template_sha256": sha256_file(ct) if ct.is_file() else None,
                "bos_rule": "BOS id 124894 prepended explicitly to every raw KL segment (the post-processor adds "
                            "none); chat-templated text carries its own <|startoftext|> literal",
                "special_tokens": "special-token literals in the text are recognized as their ids",
                "normalization": "none (normalizer null); no NFC",
                "tokenizers_version": ver("tokenizers"), "transformers_version": ver("transformers")}


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
