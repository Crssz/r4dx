"""Shared helpers of the qwen4_exp reference package: the model-agnostic ones of tools/reference/common.py,
re-exported (not forked) the way gemma/common_gemma.py does, plus this track's default paths.

Run the scripts with the reference venv (D:\\venvs\\r4dx-qwen4-ref). The device rule (CLAUDE.md "use GPU device 1")
is `common.resolve_device`: cuda needs HIP_VISIBLE_DEVICES='1' (or R4DX_ALLOW_GPU0=1 for '0'), set before the
process starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REF_DIR = Path(__file__).resolve().parents[1]
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from common import (  # noqa: E402,F401  (re-exported)
    MODELS_ROOT,
    ShardIndex,
    raw_safetensors_header,
    raw_safetensors_read,
    resolve_device,
    save_golden,
    set_seed,
    sha256_bytes,
    sha256_file,
    tensor_manifest_entry,
)
from qwen4exp.arch import Qwen4ExpArch  # noqa: E402,F401

DEFAULT_MODEL_DIR = Path(os.environ.get(
    "R4DX_MODEL_DIR", str(MODELS_ROOT / "r4dx" / "Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ")))
DEFAULT_SIDECAR = Path(os.environ.get(
    "R4DX_NGRAM_SIDECAR", str(MODELS_ROOT / "r4dx" / "swift15-flash-next.r4dx-ngram")))
TINY_DIR = MODELS_ROOT / "r4dx" / "tiny-q4exp"
KL_CORPUS_DIR = _REF_DIR / "kl_corpus"
QWEN4EXP_OUT_DIR = MODELS_ROOT / "r4dx" / "qwen4exp"
