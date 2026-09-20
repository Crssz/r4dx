"""Shared helpers for r4dx's Python reference/validation tooling.

Everything under tools/reference/** is read-only against:
  - C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10  (the reference transformers install -- never
    pip/uv install into it, we only import from it)
  - C:\\AI\\models\\Qwen3.8-27B                          (the checkpoint; shards may still be
    downloading, so every tensor read here is allowed to fail and fall back to random init)

Run these scripts with that venv's python.exe, e.g.:
    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\reference\\layer_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL_DIR = Path(r"C:\AI\models\Qwen3.8-27B")
DEFAULT_REFERENCE_VENV = Path(
    os.environ.get("R4DX_REFERENCE_VENV", Path.home() / "dev" / "vLLM_for_AMD" / ".venv-rocm10")
)


def resolve_device(requested: str) -> torch.device:
    """Resolve a --device argument.

    Enforces this repo's GPU rule: GPU work only ever touches HIP device 1 (the headless R9700),
    selected by setting $env:HIP_VISIBLE_DEVICES='1' *before* the process starts (torch/HIP reads
    it at import/init time). Once that's set, torch's ROCm build sees exactly one device at CUDA
    index 0, which is physical device 1.
    """
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu", "hip"):
        visible = os.environ.get("HIP_VISIBLE_DEVICES")
        if visible != "1":
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
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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

    @classmethod
    def load(cls, model_dir: Path) -> "ShardIndex":
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            return cls(model_dir=model_dir, weight_map=weight_map)
        single = model_dir / "model.safetensors"
        if single.exists():
            from safetensors import safe_open

            with safe_open(str(single), framework="pt", device="cpu") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}
            return cls(model_dir=model_dir, weight_map=weight_map, single_file=single)
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
        with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as f:
            sl = f.get_slice(name)
            return sl[start:stop, :].clone()  # same dangling-mmap risk as get_tensor above


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
