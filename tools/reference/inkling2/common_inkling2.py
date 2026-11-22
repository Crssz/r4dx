"""Shared helpers of the Inkling-2-Mini (`inkling2_loop`) reference package (docs/inkling2-plan.md 3.3, 7).

The checkpoint (and its tokenizer files: tokenizer.json, tokenizer_config.json, chat_template.jinja) live in one
directory, `E:\\models\\OpenUncensored\\Inkling-2-Mini-Uncensored` (`R4DX_MODEL_DIR` overrides; otherwise
`$R4DX_MODELS_ROOT\\OpenUncensored\\Inkling-2-Mini-Uncensored`).  Run the scripts with the reference venv (`-I`
when they read downloaded files; the scripts add their own directory to sys.path):

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\reference\\inkling2\\<script>.py ...

CPU ONLY: importing this module sets HIP_VISIBLE_DEVICES=-1 and CUDA_VISIBLE_DEVICES=-1 (import it BEFORE torch);
`assert_cpu_only()` then checks that torch really sees no GPU.  `set_threads` refuses more than 24 threads.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_REF_DIR = Path(__file__).resolve().parents[1]
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from common import sha256_bytes, sha256_file  # noqa: E402,F401  (re-exported; torch-free import)

_MODELS_ROOT = Path(os.environ.get("R4DX_MODELS_ROOT", r"E:\models"))
DEFAULT_MODEL_DIR = Path(os.environ.get("R4DX_MODEL_DIR", str(_MODELS_ROOT / "OpenUncensored" / "Inkling-2-Mini-Uncensored")))
DEFAULT_OUT_DIR = Path(os.environ.get("R4DX_INKLING2_OUT_DIR", str(_MODELS_ROOT / "r4dx" / "inkling2")))
KL_CORPUS_DIR = _REF_DIR / "kl_corpus"
GOLDEN_DIR = _REF_DIR / "golden_out" / "inkling2"

#: ids of the real tokenizer (tokenizer_config.json added_tokens_decoder; docs/inkling2-plan.md 3.1)
BEGIN_OF_TEXT_ID = 200028   # <|begin_of_text|>: the raw-segment "BOS" (the chat template has none)
PAD_ID = 199999             # <|endoftext|>
MESSAGE_USER_ID = 200000
MESSAGE_MODEL_ID = 200001
MESSAGE_SYSTEM_ID = 200002
MESSAGE_TOOL_ID = 200003
CONTENT_TEXT_ID = 200004
EOS_ID = 200006             # <|content_model_end_sampling|>
CONTENT_THINKING_ID = 200008
END_MESSAGE_ID = 200010
CONTENT_INVOKE_TOOL_JSON_ID = 200049
EMBED_ROWS = 201024
REAL_VOCAB = 200058         # lm_head rows actually produced (config unpadded_vocab_size)
LOGPROB_CLAMP = -1e4
MAX_THREADS = 24


def token_ids_sha256(token_ids) -> str:
    """Byte-identical to kl_report.token_ids_sha256 / the C++ TokenIdsSha256."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def peak_rss_gb() -> float | None:
    """Peak working set of this process in GiB (Windows PROCESS_MEMORY_COUNTERS; resource.getrusage elsewhere)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            gpmi.restype = wintypes.BOOL
            ok = gpmi(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 2**30 if ok else None
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        return None


def assert_cpu_only() -> None:
    """Refuses to run when torch can see a GPU (call after `import torch`)."""
    import torch

    assert os.environ.get("HIP_VISIBLE_DEVICES") == "-1", "HIP_VISIBLE_DEVICES must be -1 (CPU-only reference)"
    assert torch.cuda.is_available() is False, "torch.cuda.is_available() is True: refusing to run (CPU only)"


def set_threads(n: int) -> int:
    """torch.set_num_threads(n) with the lane cap (24)."""
    import torch

    if n < 1 or n > MAX_THREADS:
        raise SystemExit(f"--threads {n}: must be in [1, {MAX_THREADS}]")
    torch.set_num_threads(n)
    return n


def force_eager(cfg) -> None:
    cfg._attn_implementation = "eager"


class Inkling2RefTokenizer:
    """The Inkling tokenizer as the reference tools use it.

    - `encode(text)`: transformers AutoTokenizer, add_special_tokens=False (the template emits every special token as
      a literal; the post-processor adds none); special-token literals in the text map to their ids.
    - `encode_raw(text)`: the same through `tokenizers.Tokenizer.from_file(tokenizer.json)` (what r4dx's C++
      tokenizer implements).  `agrees(text)` compares both.
    - `render_chat` / `encode_chat`: the checkpoint's chat_template.jinja (text / ids).  There is NO generation
      prompt: the template appends nothing for add_generation_prompt, the model emits its own `<|message_model|>`.
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

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self.hf(text, add_special_tokens=False, verbose=False)["input_ids"]]

    def encode_raw(self, text: str) -> list[int]:
        return [int(i) for i in self.raw.encode(text, add_special_tokens=False).ids]

    def agrees(self, text: str) -> bool:
        return self.encode(text) == self.encode_raw(text)

    def decode(self, ids) -> str:
        return self.hf.decode([int(i) for i in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def with_bos(self, ids) -> list[int]:
        return [BEGIN_OF_TEXT_ID] + [int(i) for i in ids]

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
        return f"{self.tok_dir / 'tokenizer.json'} (transformers AutoTokenizer == tokenizers, no BOS/EOS added)"

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
                "bos_rule": "chat segments start with the template's own first special token (<|message_system|> or "
                            "<|message_user|>); RAW segments get <|begin_of_text|> (id 200028) prepended explicitly",
                "special_tokens": "special-token literals in the text are recognized as their ids",
                "normalization": "none; no NFC",
                "tokenizers_version": ver("tokenizers"), "transformers_version": ver("transformers")}


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
