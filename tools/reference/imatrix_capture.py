"""tools/reference/imatrix_capture.py

The **importance matrix** (llama.cpp's "imatrix") for every linear `r4dx-convert` quantizes.

For a weight `W [N, K]` quantized per (output row, group of 128 input channels), the quantizer's
only real choice is how to spend 4 bits per weight, and *not every input channel matters equally*:
the error that leaves the layer is `sum_k (W[n,k] - Wq[n,k]) * x_k`, so a channel `k` whose
activation is routinely large deserves a tighter fit than one that is routinely near zero. This
script measures exactly that statistic, with llama.cpp's semantics:

    importance[k] = mean over calibration tokens of x_k^2      (x = the linear's INPUT activation)

and writes one `float32[K]` vector per container tensor, keyed by **the converter's own tensor base
name** (`text.layers.17.mlp.down`, `lm_head`, ...) so `src/convert/main.cpp` can look a linear's
vector up with zero name-mapping logic. Nothing here changes any byte layout: the imatrix is an
input to a *better choice of q/scale/zero* for the same containers and the same kernels.

**Where the activations come from.** This script does not build a forward of its own. It imports
`full_logits_golden.StreamingReference` unchanged (the same lazy layer streamer
`kv_calibrate_full.py` reuses) and hangs `forward_pre_hook`s on the real `nn.Linear` modules of
each `Qwen3_5DecoderLayer` as it is materialized on HIP device 1, so every `x` it squares is the
true mid-stack activation of the original bf16 checkpoint with all the real preceding layers in
front of it. Peak VRAM is ~2-4 GiB against a 51.7 GiB checkpoint.

**Fused projections.** Every `add_linear` call in `src/convert/main.cpp` concatenates its parts on
the OUTPUT (row / `N`) axis only -- `mlp.gate_up` = `[gate; up]`, and `attn.qg` is already fused in
the checkpoint (`q_proj` is `Linear(hidden, num_heads*head_dim*2)`). The `K` axis is never
concatenated, which is what lets ONE `[K]` importance vector serve both halves of a fused tensor.
`verify_no_k_concat()` re-checks that against the checkpoint's own shapes on every run rather than
trusting this paragraph, and `--fusion-check-layer` additionally proves `gate_proj` and `up_proj`
really do see the same input tensor.

**The MTP head** (`mtp.attn.qg`, `mtp.attn.o`, `mtp.mlp.gate_up`, `mtp.mlp.down`, and the optional
`mtp.draft_head.lm_head`) is quantized by the converter too, but `transformers` 5.17.0 has no MTP
class at all -- the checkpoint's `mtp.*` weights are dead tensors to it. Its one decoder layer is
therefore driven here by a small hand-rolled forward that mirrors `src/model/mtp_head.cpp`
exactly: `fc(concat(rmsnorm(embed(t_{i+1}), pre_fc_norm_embedding), rmsnorm(h_i,
pre_fc_norm_hidden)))` -- embedding in fc's FIRST `hidden` input columns, hidden state in the
second, both norms zero-centered `(1 + w)` like `r4dx_rmsnorm_bf16`, and `h_i` the **pre-final-norm**
residual stream (`src/model/model.cpp`'s `RunChunk` primes MTP from `cur`, before `FinalLmHead`
applies `final_norm`). Pass `--no-mtp` to skip it; the converter then simply finds no entry for
those four keys and quantizes them the way it does today.

Usage (reference venv only -- read-only against the venv and the checkpoint):

    $env:HIP_VISIBLE_DEVICES = '1'
    <venv>\\Scripts\\python.exe tools\\reference\\imatrix_capture.py `
        --out D:\\models\\r4dx\\qwen38-27b.imatrix.npz

See tools/reference/README.md ("imatrix_capture.py") for the option list, the corpus, the gate runs
and the expected runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    ShardIndex,
    load_text_config,
    resolve_device,
    sha256_file,
)
from full_logits_golden import (  # noqa: E402
    EMBED_NAME,
    LM_HEAD_NAME,
    StreamingReference,
    gather_embedding_rows,
    get_tensors_grouped,
)
from kv_calibrate_full import collect_corpus, repo_relative  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = Path(__file__).parent / "kv_calib_corpus"
DEFAULT_CALIB_TXT = Path(__file__).parent / "calib.txt"
DEFAULT_OUT = Path(r"D:\models\r4dx\qwen38-27b.imatrix.npz")
CONVERTER_MAIN = REPO_ROOT / "src" / "convert" / "main.cpp"

#: HF weight-name prefixes (same checkpoint layout `full_logits_golden.py` documents).
TEXT_LAYER_PREFIX = "model.language_model.layers."
MTP_PREFIX = "mtp."
MTP_LAYER_PREFIX = "mtp.layers.0."

CAVEAT = (
    "Per-input-channel importance (llama.cpp imatrix semantics: mean over calibration tokens of "
    "x_k^2, x = the linear's INPUT activation) measured on a REAL full forward of the original "
    "bf16 checkpoint -- the whole 64-layer stack was run over the calibration corpus via "
    "full_logits_golden.StreamingReference, with forward_pre_hooks on the real nn.Linear modules. "
    "What this is NOT: it is not a Hessian and not a per-weight sensitivity -- it is the diagonal "
    "second moment of the input activation only, so it captures 'this input channel is usually "
    "large' and nothing about correlations between channels. It is also only as representative as "
    "the corpus (tools/reference/calib.txt + kv_calib_corpus/, deliberately DISJOINT from the "
    "held-out KL eval set tools/reference/kl_corpus/): a prompt distribution far outside it "
    "weights the wrong channels. Weighting is monotone in this vector, so a mis-scaled but "
    "correctly-ordered vector is harmless; a systematically wrong ORDER is not."
)


# --------------------------------------------------------------------------------------------
# What the converter quantizes
# --------------------------------------------------------------------------------------------


@dataclass
class LinearSpec:
    """One `add_linear` call in `src/convert/main.cpp`, as seen from the activation side."""

    key: str                    #: the converter's container base name == the npz key
    hf_names: list[str]         #: the HF weight(s) the converter concatenates (row axis only)
    tap: str                    #: how the input activation is captured (see `describe_tap`)
    optional: bool = False      #: written only under a converter flag (draft head)

    @property
    def family(self) -> str:
        return self.key.split(".")[-1] if "." in self.key else self.key


def enumerate_quantized_linears(text_config, do_mtp: bool, do_draft_head: bool) -> list[LinearSpec]:
    """Every linear `src/convert/main.cpp` routes through `add_linear` (i.e. every linear that gets
    `mxfp4`/`w4a16`/`w4a8` variants), in converter order, with the HF module whose INPUT feeds it.

    Deliberately NOT included, because the converter never quantizes them: `gdn.in_proj_a`/`_b`,
    `gdn.conv1d_weight`, every `*_norm`, `text.embed_tokens`, `mtp.fc`, `mtp.norm`,
    `mtp.pre_fc_norm_*`, and `mtp.attn.k`/`mtp.attn.v` (those four `add_bf16`, see
    docs/container-format.md: "the MTP head stays bf16-only"). `audit_converter_source()` re-derives
    this list's shape from main.cpp's text so a new `add_linear` cannot slip past it silently.
    """
    specs: list[LinearSpec] = []
    for i, kind in enumerate(text_config.layer_types):
        hf = f"{TEXT_LAYER_PREFIX}{i}."
        base = f"text.layers.{i}."
        if kind == "full_attention":
            # q_proj is ALREADY the fused query+output-gate matrix in this checkpoint; qg is a
            # straight copy of it, so q and gate share this one input vector.
            specs.append(LinearSpec(base + "attn.qg", [hf + "self_attn.q_proj.weight"],
                                    f"layer{i}:self_attn.q_proj"))
            specs.append(LinearSpec(base + "attn.k", [hf + "self_attn.k_proj.weight"],
                                    f"layer{i}:self_attn.k_proj"))
            specs.append(LinearSpec(base + "attn.v", [hf + "self_attn.v_proj.weight"],
                                    f"layer{i}:self_attn.v_proj"))
            specs.append(LinearSpec(base + "attn.o", [hf + "self_attn.o_proj.weight"],
                                    f"layer{i}:self_attn.o_proj"))
        else:
            specs.append(LinearSpec(base + "gdn.in_proj_qkv", [hf + "linear_attn.in_proj_qkv.weight"],
                                    f"layer{i}:linear_attn.in_proj_qkv"))
            specs.append(LinearSpec(base + "gdn.in_proj_z", [hf + "linear_attn.in_proj_z.weight"],
                                    f"layer{i}:linear_attn.in_proj_z"))
            specs.append(LinearSpec(base + "gdn.out_proj", [hf + "linear_attn.out_proj.weight"],
                                    f"layer{i}:linear_attn.out_proj"))
        # gate_proj and up_proj are row-concatenated into one tensor and share one input.
        specs.append(LinearSpec(base + "mlp.gate_up",
                                [hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"],
                                f"layer{i}:mlp.gate_proj"))
        specs.append(LinearSpec(base + "mlp.down", [hf + "mlp.down_proj.weight"],
                                f"layer{i}:mlp.down_proj"))
    # lm_head's input is the post-final-norm hidden state, which is exactly what
    # StreamingReference.forward_hidden returns -- no hook needed.
    specs.append(LinearSpec("lm_head", [LM_HEAD_NAME], "final_norm_out"))

    if do_mtp:
        specs.append(LinearSpec("mtp.attn.qg", [MTP_LAYER_PREFIX + "self_attn.q_proj.weight"],
                                "mtp:self_attn.q_proj"))
        specs.append(LinearSpec("mtp.attn.o", [MTP_LAYER_PREFIX + "self_attn.o_proj.weight"],
                                "mtp:self_attn.o_proj"))
        specs.append(LinearSpec("mtp.mlp.gate_up",
                                [MTP_LAYER_PREFIX + "mlp.gate_proj.weight",
                                 MTP_LAYER_PREFIX + "mlp.up_proj.weight"],
                                "mtp:mlp.gate_proj"))
        specs.append(LinearSpec("mtp.mlp.down", [MTP_LAYER_PREFIX + "mlp.down_proj.weight"],
                                "mtp:mlp.down_proj"))
        if do_draft_head:
            # OPTIONAL (only with --draft-vocab-ids): a row slice of lm_head.weight applied to the
            # MTP layer's OWN post-mtp.norm hidden state, not the backbone's -- different input
            # distribution from `lm_head`, hence its own vector.
            specs.append(LinearSpec("mtp.draft_head.lm_head", [LM_HEAD_NAME], "mtp:norm_out",
                                    optional=True))
    return specs


def describe_tap(tap: str) -> str:
    if tap == "final_norm_out":
        return "post-final-norm hidden state (StreamingReference.forward_hidden's return)"
    if tap == "mtp:norm_out":
        return "post-mtp.norm hidden state of the MTP layer"
    scope, module = tap.split(":", 1)
    return f"forward_pre_hook on {scope}.{module} (its nn.Linear input)"


# --------------------------------------------------------------------------------------------
# Gate (a) part 1: the converter's source, not this file's memory of it
# --------------------------------------------------------------------------------------------

#: `(hf suffixes, container suffix)` pairs this script knows how to serve, per main.cpp scope.
EXPECTED_ADD_LINEAR = {
    "text": {
        ("self_attn.q_proj.weight",): "attn.qg",
        ("self_attn.k_proj.weight",): "attn.k",
        ("self_attn.v_proj.weight",): "attn.v",
        ("self_attn.o_proj.weight",): "attn.o",
        ("linear_attn.in_proj_qkv.weight",): "gdn.in_proj_qkv",
        ("linear_attn.in_proj_z.weight",): "gdn.in_proj_z",
        ("linear_attn.out_proj.weight",): "gdn.out_proj",
        ("mlp.gate_proj.weight", "mlp.up_proj.weight"): "mlp.gate_up",
        ("mlp.down_proj.weight",): "mlp.down",
        ("lm_head.weight",): "lm_head",
    },
    "mtp": {
        ("self_attn.q_proj.weight",): "attn.qg",
        ("self_attn.o_proj.weight",): "attn.o",
        ("mlp.gate_proj.weight", "mlp.up_proj.weight"): "mlp.gate_up",
        ("mlp.down_proj.weight",): "mlp.down",
    },
}

#: Quantized linears main.cpp emits WITHOUT going through `add_linear` (it calls
#: Plan/EmitLinearLayouts directly), so the regex scan cannot see them. `mtp.draft_head.lm_head` is
#: a real (optional) model tensor and is served by `--draft-head`; `selftest` is `r4dx-convert
#: --selftest`'s throwaway single-tensor container (`main.cpp`'s `RunSelftest`), not a model tensor
#: at all, so no activation exists for it and none is captured.
EXPECTED_DIRECT_LAYOUTS = {"mtp.draft_head.lm_head", "selftest"}

_QUOTED = re.compile(r'"([^"]*)"')


def audit_converter_source(path: Path) -> dict:
    """Re-derive main.cpp's `add_linear` call set from its text and diff it against
    `EXPECTED_ADD_LINEAR`. Only calls whose FIRST argument is a `{...}` initializer list are the
    Qwen container's (`add_linear(std::vector<std::string>, std::string, LayoutSet)`); the DFlash2
    draft container further down the file has its own `add_linear(std::string, std::string)` lambda
    for a completely different model, which this pattern skips by construction."""
    src = path.read_text(encoding="utf-8")
    mtp_at = src.find("if (do_mtp)")
    if mtp_at < 0:
        raise SystemExit(f"[imatrix] cannot find the `if (do_mtp)` block in {path}")
    found: dict[str, dict] = {"text": {}, "mtp": {}}
    for m in re.finditer(r"add_linear\(\{([^{}]*)\}\s*,(.*?)\);", src, re.S):
        hf = tuple(_QUOTED.findall(m.group(1)))
        cont = _QUOTED.findall(m.group(2))
        if len(cont) != 1:
            raise SystemExit(f"[imatrix] unparsable add_linear container name at offset {m.start()}: "
                             f"{m.group(2)!r}")
        scope = "mtp" if m.start() > mtp_at else "text"
        found[scope][hf] = cont[0]
    direct = set()
    for m in re.finditer(r"PlanLinearLayouts\(writer,\s*\"([^\"]+)\"", src):
        direct.add(m.group(1))
    missing, unexpected, mismatched = [], [], []
    for scope in ("text", "mtp"):
        for hf, cont in found[scope].items():
            want = EXPECTED_ADD_LINEAR[scope].get(hf)
            if want is None:
                unexpected.append((scope, hf, cont))
            elif want != cont:
                mismatched.append((scope, hf, cont, want))
        for hf, cont in EXPECTED_ADD_LINEAR[scope].items():
            if hf not in found[scope]:
                missing.append((scope, hf, cont))
    for name in sorted(direct - EXPECTED_DIRECT_LAYOUTS):
        unexpected.append(("direct", ("PlanLinearLayouts",), name))
    ok = not (missing or unexpected or mismatched)
    return {"path": repo_relative(path), "sha256": sha256_file(path), "found": found,
            "direct_plan_linear_layouts": sorted(direct), "missing": missing,
            "unexpected": unexpected, "mismatched": mismatched, "ok": ok}


# --------------------------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------------------------


class ChannelAccum:
    """Per-input-channel `sum(x^2)` plus the token (row) count, accumulated on device in fp64.

    fp64 because this is a sum of ~10^4 strictly non-negative terms whose magnitudes span several
    orders (the whole point of the measurement is that some channels are huge), and a fp32 running
    sum of such a stream loses low-order terms exactly where the "unimportant" channels live.
    Only `[K]` ever leaves the GPU."""

    __slots__ = ("k", "sum_x2", "rows")

    def __init__(self, k: int, device):
        self.k = k
        self.sum_x2 = torch.zeros(k, dtype=torch.float64, device=device)
        self.rows = 0

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.shape[1] != self.k:
            raise RuntimeError(f"channel count changed: accumulator K={self.k}, got {flat.shape[1]}")
        self.sum_x2 += flat.float().pow(2).sum(dim=0, dtype=torch.float64)
        self.rows += int(flat.shape[0])

    def mean_x2(self) -> np.ndarray:
        if self.rows == 0:
            raise RuntimeError("no rows accumulated")
        return (self.sum_x2 / float(self.rows)).to(torch.float32).cpu().numpy()


@dataclass
class CaptureResult:
    accums: dict[str, ChannelAccum] = field(default_factory=dict)
    fusion_check: dict | None = None
    total_tokens: int = 0
    seconds: float = 0.0
    peak_gib: float = 0.0
    reserved_gib: float = 0.0


class LinearTaps:
    """Installs the `forward_pre_hook`s on every streamed text layer.

    Layer identity works the same way `kv_calibrate_full.KvCapture` establishes it: `LazyDecoderLayers`
    builds layer `i` immediately before the forward loop runs it and drops it immediately after,
    strictly in order, so wrapping the builder is enough to know which layer a module belongs to
    (and the hooks die with the layer, which is freed one forward later)."""

    def __init__(self, ref: StreamingReference, specs: list[LinearSpec], result: CaptureResult,
                 fusion_check_layer: int | None):
        self.ref = ref
        self.result = result
        self.fusion_check_layer = fusion_check_layer
        self._orig_builder = None
        self._handles: list = []
        self._pending_gate: torch.Tensor | None = None
        # tap "layer{i}:module.path" -> container key
        self.by_layer: dict[int, dict[str, str]] = {}
        for s in specs:
            if not s.tap.startswith("layer"):
                continue
            scope, module = s.tap.split(":", 1)
            self.by_layer.setdefault(int(scope[len("layer"):]), {})[module] = s.key

    def _accum(self, key: str, x: torch.Tensor) -> None:
        acc = self.result.accums.get(key)
        if acc is None:
            acc = ChannelAccum(int(x.shape[-1]), x.device)
            self.result.accums[key] = acc
        acc.update(x)

    def _hook(self, key: str):
        def pre_hook(_module, inputs):
            self._accum(key, inputs[0])
        return pre_hook

    def _gate_check_hooks(self, layer):
        """Gate: `mlp.gate_up` is [gate; up] on the row axis, so gate_proj and up_proj must be fed
        the SAME tensor object. Prove it rather than cite docs/container-format.md."""
        def gate_hook(_m, inputs):
            self._pending_gate = inputs[0]

        def up_hook(_m, inputs):
            g = self._pending_gate
            same_obj = g is not None and g.data_ptr() == inputs[0].data_ptr()
            equal = bool(g is not None and torch.equal(g, inputs[0]))
            prev = self.result.fusion_check
            self.result.fusion_check = {
                "layer": self.fusion_check_layer,
                "same_storage": bool(same_obj) and (prev["same_storage"] if prev else True),
                "elementwise_equal": equal and (prev["elementwise_equal"] if prev else True),
                "shape": list(inputs[0].shape),
                "calls": (prev["calls"] + 1) if prev else 1,
            }
            self._pending_gate = None

        self._handles.append(layer.mlp.gate_proj.register_forward_pre_hook(gate_hook))
        self._handles.append(layer.mlp.up_proj.register_forward_pre_hook(up_hook))

    def __enter__(self) -> "LinearTaps":
        self._orig_builder = self.ref.lazy._builder

        def builder(i: int):
            layer = self._orig_builder(i)
            wanted = self.by_layer.get(i, {})
            for module_path, key in wanted.items():
                mod = layer
                for part in module_path.split("."):
                    mod = getattr(mod, part)
                self._handles.append(mod.register_forward_pre_hook(self._hook(key)))
            if self.fusion_check_layer == i:
                self._gate_check_hooks(layer)
            return layer

        self.ref.lazy._builder = builder
        return self

    def __exit__(self, *exc) -> None:
        self.ref.lazy._builder = self._orig_builder
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --------------------------------------------------------------------------------------------
# The MTP head (transformers has no class for it -- see this file's docstring)
# --------------------------------------------------------------------------------------------


def zero_centered_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5's RMSNorm, the variant `r4dx_rmsnorm_bf16` (`RmsNormKernel`) implements: normalize in
    fp32, scale by `(1 + w)`. Identical in form to `StreamingReference._manual_final_norm`."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).to(x.dtype)


class MtpTaps:
    """Loads `mtp.*` once and runs the head over each corpus file's (h_i, t_{i+1}) pairs."""

    def __init__(self, ref: StreamingReference, specs: list[LinearSpec], result: CaptureResult,
                 want_draft_head: bool):
        self.ref = ref
        self.result = result
        self.want_draft_head = want_draft_head
        self.eps = float(ref.text_config.rms_norm_eps)
        # The MTP layer is a COMPLETE full-attention decoder layer (src/convert/main.cpp's own
        # verified note), so build it against a full_attention layer index of the text config.
        self.layer_idx = list(ref.text_config.layer_types).index("full_attention")
        with torch.device("meta"):
            self.layer = ref.m.Qwen3_5DecoderLayer(ref.text_config, self.layer_idx)
        keys = list(self.layer.state_dict().keys())
        raw = get_tensors_grouped(ref.index, [MTP_LAYER_PREFIX + k for k in keys])
        self.layer.load_state_dict(
            {k: raw[MTP_LAYER_PREFIX + k].to(device=ref.device, dtype=ref.dtype) for k in keys},
            assign=True)
        self.layer.eval()
        for p in self.layer.parameters():
            p.requires_grad_(False)
        aux = get_tensors_grouped(ref.index, [MTP_PREFIX + n for n in (
            "fc.weight", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")])
        self.fc = aux[MTP_PREFIX + "fc.weight"].to(device=ref.device, dtype=ref.dtype)
        self.norm_w = aux[MTP_PREFIX + "norm.weight"].to(device=ref.device, dtype=ref.dtype)
        self.pre_emb_w = aux[MTP_PREFIX + "pre_fc_norm_embedding.weight"].to(
            device=ref.device, dtype=ref.dtype)
        self.pre_hid_w = aux[MTP_PREFIX + "pre_fc_norm_hidden.weight"].to(
            device=ref.device, dtype=ref.dtype)
        hidden = ref.text_config.hidden_size
        if list(self.fc.shape) != [hidden, 2 * hidden]:
            raise SystemExit(f"[imatrix] mtp.fc has shape {list(self.fc.shape)}, expected "
                             f"[{hidden}, {2 * hidden}] -- the concat order assumption in "
                             f"src/model/mtp_head.cpp no longer holds")
        self._handles: list = []
        for s in specs:
            if not s.tap.startswith("mtp:") or s.tap == "mtp:norm_out":
                continue
            mod = self.layer
            for part in s.tap.split(":", 1)[1].split("."):
                mod = getattr(mod, part)
            self._handles.append(mod.register_forward_pre_hook(self._hook(s.key)))

    def _hook(self, key: str):
        def pre_hook(_module, inputs):
            acc = self.result.accums.get(key)
            if acc is None:
                acc = ChannelAccum(int(inputs[0].shape[-1]), inputs[0].device)
                self.result.accums[key] = acc
            acc.update(inputs[0])
        return pre_hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def run(self, pre_norm_hidden: torch.Tensor, token_ids: list[int]) -> None:
        """`pre_norm_hidden`: [T, hidden], the residual stream BEFORE final_norm. Pairs row i with
        token i+1, exactly as `Model::RunChunk`'s `PrimeKv` does, so rows 0..T-2 are used."""
        ref = self.ref
        n = pre_norm_hidden.shape[0] - 1
        if n < 1:
            return
        h = pre_norm_hidden[:n]
        emb = gather_embedding_rows(ref.index, list(token_ids[1:]), ref.device, ref.dtype)[0]
        x = torch.cat([zero_centered_rmsnorm(emb, self.pre_emb_w, self.eps),
                       zero_centered_rmsnorm(h, self.pre_hid_w, self.eps)], dim=-1)
        fc_out = torch.nn.functional.linear(x, self.fc).unsqueeze(0)  # [1, n, hidden]
        cos, sin = ref._manual_rope(n)
        out = self.layer(hidden_states=fc_out, position_embeddings=(cos, sin),
                         attention_mask=ref._manual_mask(n), position_ids=None,
                         past_key_values=None)
        if self.want_draft_head:
            normed = zero_centered_rmsnorm(out[0], self.norm_w, self.eps)
            acc = self.result.accums.get("mtp.draft_head.lm_head")
            if acc is None:
                acc = ChannelAccum(int(normed.shape[-1]), normed.device)
                self.result.accums["mtp.draft_head.lm_head"] = acc
            acc.update(normed)


class FinalNormInputTap:
    """Captures the pre-final-norm residual stream (`Qwen3_5TextModel.forward`'s last op is
    `hidden_states = self.norm(hidden_states)`), which is what MTP is primed from."""

    def __init__(self, ref: StreamingReference):
        self.ref = ref
        self.captured: torch.Tensor | None = None
        self._h = None

    def __enter__(self) -> "FinalNormInputTap":
        def pre_hook(_m, inputs):
            self.captured = inputs[0]
        self._h = self.ref.model.norm.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, *exc) -> None:
        self._h.remove()
        self.captured = None


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def tensor_shapes(index: ShardIndex, names: list[str]) -> dict[str, list[int]]:
    """Shapes straight from the safetensors headers -- no tensor data is read."""
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for n in names:
        if n not in index.weight_map:
            raise KeyError(f"{n!r} is not in this checkpoint's model.safetensors.index.json")
        by_shard.setdefault(index.weight_map[n], []).append(n)
    out: dict[str, list[int]] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(index.model_dir / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                out[n] = [int(v) for v in f.get_slice(n).get_shape()]
    return out


def verify_no_k_concat(specs: list[LinearSpec], shapes: dict[str, list[int]]) -> dict[str, int]:
    """Gate (a) part 2: every `add_linear` must concatenate on the OUTPUT axis only, so all parts of
    a fused tensor share one `K`. Returns key -> expected K."""
    expect: dict[str, int] = {}
    for s in specs:
        ks = {shapes[n][1] for n in s.hf_names}
        if len(ks) != 1:
            raise SystemExit(f"[imatrix] K-CONCAT DETECTED for {s.key}: parts have in-features "
                             f"{ {n: shapes[n][1] for n in s.hf_names} } -- one importance vector "
                             f"cannot serve them and the converter's row-concat assumption is wrong")
        expect[s.key] = ks.pop()
    return expect


@torch.no_grad()
def run_capture(ref: StreamingReference, corpus, specs: list[LinearSpec], do_mtp: bool,
                want_draft_head: bool, fusion_check_layer: int | None,
                verbose: bool = True) -> CaptureResult:
    result = CaptureResult()
    mtp = MtpTaps(ref, specs, result, want_draft_head) if do_mtp else None
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    try:
        with LinearTaps(ref, specs, result, fusion_check_layer), FinalNormInputTap(ref) as fn_tap:
            for c in corpus:
                t1 = time.perf_counter()
                hidden = ref.forward_hidden(c.token_ids, impl="model")  # post-final-norm [T, hidden]
                pre_norm = fn_tap.captured
                if pre_norm is None:
                    raise SystemExit("[imatrix] the final-norm tap never fired")
                pre_norm = pre_norm[0] if pre_norm.dim() == 3 else pre_norm
                # lm_head's input is the post-final-norm hidden state.
                acc = result.accums.get("lm_head")
                if acc is None:
                    acc = ChannelAccum(int(hidden.shape[-1]), hidden.device)
                    result.accums["lm_head"] = acc
                acc.update(hidden)
                if mtp is not None:
                    mtp.run(pre_norm, c.token_ids)
                del hidden, pre_norm
                fn_tap.captured = None
                result.total_tokens += len(c.token_ids)
                torch.cuda.synchronize()
                if verbose:
                    print(f"[imatrix] {c.name:<24} T={len(c.token_ids):>5} "
                          f"{time.perf_counter() - t1:6.1f}s", flush=True)
    finally:
        if mtp is not None:
            mtp.close()
    result.seconds = time.perf_counter() - t_start
    result.peak_gib = torch.cuda.max_memory_allocated() / 2**30
    result.reserved_gib = torch.cuda.max_memory_reserved() / 2**30
    return result


# -- gates -------------------------------------------------------------------------------------


def gate_shapes(specs: list[LinearSpec], result: CaptureResult, expect_k: dict[str, int]) -> bool:
    want = {s.key for s in specs}
    got = set(result.accums)
    missing = sorted(want - got)
    extra = sorted(got - want)
    print(f"\n[gate a] converter linears expected: {len(want)}   captured: {len(got)}")
    print(f"[gate a] missing (hook never fired): {len(missing)}  {missing[:8]}"
          f"{' ...' if len(missing) > 8 else ''}")
    print(f"[gate a] extra (captured but not a converter linear): {len(extra)}  {extra[:8]}"
          f"{' ...' if len(extra) > 8 else ''}")
    bad_k = [(k, result.accums[k].k, expect_k[k]) for k in sorted(got & want)
             if result.accums[k].k != expect_k[k]]
    print(f"[gate a] K mismatches vs the checkpoint's in-features: {len(bad_k)}  {bad_k[:8]}")
    rows = {k: result.accums[k].rows for k in sorted(got)}
    by_count: dict[int, int] = {}
    for v in rows.values():
        by_count[v] = by_count.get(v, 0) + 1
    print(f"[gate a] token counts per key: {dict(sorted(by_count.items()))} (key count per row count)")
    ok = not missing and not extra and not bad_k and all(r > 0 for r in rows.values())
    print(f"[gate a] {'PASS' if ok else 'FAIL'}")
    return ok


def gate_sanity(result: CaptureResult, probe_key: str) -> tuple[bool, dict]:
    bad = []
    for k, acc in result.accums.items():
        v = acc.mean_x2()
        if not np.isfinite(v).all():
            bad.append((k, "non-finite"))
        elif (v < 0).any():
            bad.append((k, "negative"))
        elif not (v > 0).any():
            bad.append((k, "all zero"))
    print(f"\n[gate b] non-negative / finite / not-all-zero over {len(result.accums)} vectors: "
          f"{len(bad)} bad {bad[:8]}")
    info: dict = {"violations": bad}
    if probe_key in result.accums:
        v = result.accums[probe_key].mean_x2()
        order = np.argsort(v)[::-1]
        med = float(np.median(v))
        mx = float(v.max())
        info.update({"probe_key": probe_key, "K": int(v.size), "max": mx, "median": med,
                     "min": float(v.min()), "mean": float(v.mean()),
                     "max_over_median": mx / med if med > 0 else float("inf"),
                     "top10": [(int(i), float(v[i])) for i in order[:10]]})
        print(f"[gate b] {probe_key}: K={v.size} top-10 channels by importance "
              f"(mean over tokens of x_k^2):")
        for rank, i in enumerate(order[:10]):
            print(f"           #{rank + 1:>2} channel {int(i):>6}  {float(v[i]):.6g}  "
                  f"({float(v[i]) / med:.1f}x median)")
        print(f"[gate b] {probe_key}: max={mx:.6g} median={med:.6g} min={float(v.min()):.6g} "
              f"max/median={mx / med:.1f}x  (heavy tail expected)")
    # The same statistic across every captured vector: how heavy-tailed the input distribution of
    # each linear is, which is precisely how much an importance-weighted quantizer can win over the
    # plain min/max one. Recorded as data so stage 2 can target the worst offenders first.
    ratios = []
    for k, acc in result.accums.items():
        v = acc.mean_x2()
        med = float(np.median(v))
        ratios.append((float(v.max()) / med if med > 0 else float("inf"), k))
    ratios.sort(reverse=True)
    pct = np.percentile([r for r, _ in ratios], [10, 50, 90, 99])
    info["max_over_median_percentiles"] = {"p10": float(pct[0]), "p50": float(pct[1]),
                                           "p90": float(pct[2]), "p99": float(pct[3])}
    info["top_keys_by_max_over_median"] = [(k, r) for r, k in ratios[:5]]
    info["flattest_keys_by_max_over_median"] = [(k, r) for r, k in ratios[-5:]]
    print(f"[gate b] max/median across all {len(ratios)} vectors: p10={pct[0]:.1f} "
          f"p50={pct[1]:.1f} p90={pct[2]:.1f} p99={pct[3]:.1f}")
    print(f"[gate b] heaviest tails: " +
          ", ".join(f"{k} ({r:.0f}x)" for r, k in ratios[:3]))
    ok = not bad
    print(f"[gate b] {'PASS' if ok else 'FAIL'}")
    return ok, info


def compare_runs(a: CaptureResult, b: CaptureResult) -> tuple[bool, dict]:
    keys = sorted(set(a.accums) & set(b.accums))
    worst_rel, worst_key, exact = 0.0, "", 0
    for k in keys:
        va, vb = a.accums[k].mean_x2(), b.accums[k].mean_x2()
        if np.array_equal(va, vb):
            exact += 1
            continue
        denom = np.maximum(np.abs(va), 1e-30)
        rel = float((np.abs(va - vb) / denom).max())
        if rel > worst_rel:
            worst_rel, worst_key = rel, k
    #: bf16 has 8 mantissa bits (~3.9e-3 relative ulp); squaring doubles that to ~7.8e-3. Anything
    #: at or below that is arithmetic-order noise, not a different measurement.
    tol = 8e-3
    ok = worst_rel <= tol
    print(f"\n[gate c] determinism over {len(keys)} keys: {exact} bit-identical, "
          f"worst relative per-channel difference {worst_rel:.3e} on {worst_key or '(none)'} "
          f"(bf16 x^2 noise floor {tol:.1e})")
    print(f"[gate c] {'PASS' if ok else 'FAIL'}")
    return ok, {"keys": len(keys), "bit_identical": exact, "worst_rel": worst_rel,
                "worst_key": worst_key, "tolerance": tol}


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="directory of *.txt and *.messages.json calibration files; 'none' to skip")
    ap.add_argument("--calib-txt", type=Path, default=DEFAULT_CALIB_TXT,
                    help="the mixed English/code/Thai corpus, always included; 'none' to leave out")
    ap.add_argument("--extra-files", type=Path, action="append", default=[])
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="truncate every corpus file to this many tokens (one sequence per file)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="use only the first N corpus files (gate runs; default: all)")
    ap.add_argument("--no-mtp", action="store_true", help="skip the mtp.* head entirely")
    ap.add_argument("--draft-head", action="store_true",
                    help="also capture mtp.draft_head.lm_head (only useful when the container will "
                         "be converted with --draft-vocab-ids)")
    ap.add_argument("--fusion-check-layer", type=int, default=0,
                    help="prove mlp.gate_proj and mlp.up_proj share one input on this text layer "
                         "(-1 to skip)")
    ap.add_argument("--probe-key", default=None,
                    help="gate (b) top-10 probe; default: a mid-stack mlp.down")
    ap.add_argument("--determinism", action="store_true",
                    help="gate (c): run the whole capture TWICE and diff, write nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # The GPU rule, enforced before anything expensive happens (common.resolve_device re-checks it).
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible != "1":
        raise SystemExit(
            "[imatrix] refusing to run: $env:HIP_VISIBLE_DEVICES must be exactly '1' (only HIP "
            f"device 1, the headless R9700, may be used). Got {visible!r}."
        )
    device = resolve_device("cuda")
    dtype = torch.bfloat16

    audit = audit_converter_source(CONVERTER_MAIN)
    print(f"[imatrix] converter audit ({audit['path']}): "
          f"text={len(audit['found']['text'])} mtp={len(audit['found']['mtp'])} add_linear shapes, "
          f"direct PlanLinearLayouts={audit['direct_plan_linear_layouts']}")
    for kind in ("missing", "unexpected", "mismatched"):
        if audit[kind]:
            print(f"[imatrix] converter audit {kind.upper()}: {audit[kind]}")
    if not audit["ok"]:
        raise SystemExit("[imatrix] src/convert/main.cpp's add_linear set no longer matches this "
                         "script's table -- update enumerate_quantized_linears() before trusting "
                         "any imatrix it produces")
    print("[imatrix] converter audit: PASS (every quantized linear in main.cpp is accounted for)")

    from transformers import AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    calib_txt = None if str(args.calib_txt).lower() == "none" else args.calib_txt
    corpus_dir = None if str(args.corpus_dir).lower() == "none" else args.corpus_dir
    corpus = collect_corpus(corpus_dir, calib_txt, args.extra_files, tokenizer, args.max_tokens)
    if args.max_files is not None:
        corpus = corpus[: args.max_files]
    total_tokens = sum(len(c.token_ids) for c in corpus)
    print(f"[imatrix] corpus: {len(corpus)} file(s), {total_tokens} tokens (<= {args.max_tokens} each)")
    for c in corpus:
        print(f"    {c.name:<24} {len(c.token_ids):>5} tok  [{c.kind}]  {repo_relative(c.path)}")

    _, text_config = load_text_config(args.model_dir)
    do_mtp = not args.no_mtp
    specs = enumerate_quantized_linears(text_config, do_mtp, args.draft_head)
    index = ShardIndex.load(args.model_dir)
    shapes = tensor_shapes(index, sorted({n for s in specs for n in s.hf_names}))
    expect_k = verify_no_k_concat(specs, shapes)
    print(f"[imatrix] {len(specs)} quantized linears "
          f"({sum(1 for s in specs if s.key.startswith('text.'))} text, "
          f"{sum(1 for s in specs if s.key.startswith('mtp'))} mtp, lm_head), "
          f"no K-concat among any fused pair")

    t0 = time.perf_counter()
    ref = StreamingReference(args.model_dir, device, dtype=dtype)
    print(f"[imatrix] streaming skeleton ready in {time.perf_counter() - t0:.1f}s: {ref.n_layers} "
          f"layers, hidden={ref.text_config.hidden_size}, V={ref.vocab_size}", flush=True)

    fcl = None if args.fusion_check_layer < 0 else args.fusion_check_layer
    result = run_capture(ref, corpus, specs, do_mtp, args.draft_head, fcl)
    print(f"[imatrix] capture done: {result.seconds:.1f}s, peak VRAM {result.peak_gib:.3f} GiB "
          f"allocated / {result.reserved_gib:.3f} GiB reserved")

    if args.determinism:
        print("\n[imatrix] gate (c): second identical pass ...", flush=True)
        second = run_capture(ref, corpus, specs, do_mtp, args.draft_head, fcl)
        ok, det = compare_runs(result, second)
        print(f"[imatrix] determinism gate {'PASS' if ok else 'FAIL'}; nothing written "
              f"({det['bit_identical']}/{det['keys']} keys bit-identical)")
        return 0 if ok else 1

    probe = args.probe_key
    if probe is None:
        probe = "text.layers.32.mlp.down" if "text.layers.32.mlp.down" in result.accums else \
            next(k for k in result.accums if k.endswith("mlp.down"))
    ok_a = gate_shapes(specs, result, expect_k)
    ok_b, probe_info = gate_sanity(result, probe)
    fc = result.fusion_check
    if fc is not None:
        print(f"\n[imatrix] fused-input check (text layer {fc['layer']}): gate_proj and up_proj saw "
              f"the same storage on all {fc['calls']} calls: {fc['same_storage']}, elementwise "
              f"equal: {fc['elementwise_equal']}, shape {fc['shape']}")

    arrays = {k: result.accums[k].mean_x2() for k in sorted(result.accums)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    sidecar = args.out.parent / (args.out.stem + ".json")
    meta = {
        "tool": "tools/reference/imatrix_capture.py",
        "semantics": ("value[k] = mean over calibration tokens of x_k^2, x = the linear's INPUT "
                      "activation (llama.cpp imatrix). One float32[K] per container tensor, keyed "
                      "by the converter's own add_linear container base name."),
        "dtype": "float32",
        "activation_dtype": str(dtype).replace("torch.", ""),
        "accumulation_dtype": "float64 (on device)",
        "npz": str(args.out),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "config_sha256": sha256_file(args.model_dir / "config.json"),
        "embed_tokens_name": EMBED_NAME,
        "total_calibration_tokens": total_tokens,
        "max_tokens_per_file": args.max_tokens,
        "corpus": [{"name": c.name, "path": repo_relative(c.path), "kind": c.kind,
                    "sha256": sha256_file(c.path), "tokens": len(c.token_ids)} for c in corpus],
        "mtp": do_mtp,
        "draft_head": bool(args.draft_head),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seconds": result.seconds,
        "peak_vram_gib": result.peak_gib,
        "peak_vram_reserved_gib": result.reserved_gib,
        "converter_audit": {"path": audit["path"], "sha256": audit["sha256"], "ok": audit["ok"]},
        "fused_input_check": fc,
        "gates": {"shapes_ok": ok_a, "sanity_ok": ok_b, "probe": probe_info},
        "tensors": {s.key: {"K": expect_k[s.key], "rows": result.accums[s.key].rows,
                            "hf_names": s.hf_names, "tap": describe_tap(s.tap),
                            "optional": s.optional}
                    for s in specs if s.key in result.accums},
        "caveat": CAVEAT,
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[imatrix] wrote {args.out} ({len(arrays)} vectors, "
          f"{args.out.stat().st_size / 2**20:.1f} MiB)")
    print(f"[imatrix] wrote {sidecar}")
    print(f"[imatrix] tokens={total_tokens} wall={result.seconds:.1f}s "
          f"peak={result.peak_gib:.3f} GiB")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
