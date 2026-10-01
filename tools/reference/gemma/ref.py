"""GemmaReference: the bf16 text stack of a gemma4_unified checkpoint, composed from the HF
`modeling_gemma4_unified` decoder layers, for the golden / KL / quantization / validation scripts
(docs/gemma4-plan.md 6.3 "ref.py/arch.py", mode A and B of full_logits_gemma.py).

    ref = GemmaReference(model_dir, device, resident=True)      # mode A: all layers on the device
    ref = GemmaReference(model_dir, device, resident=False)     # mode B: one layer at a time
    h = ref.forward_hidden(token_ids)                           # [T, hidden] post final norm, bf16
    ref.logprobs_to_file(ids, f)                                # row i = log p(next | ids[0..i]), fp16

Why a manual layer loop instead of `model.forward`: it is the same computation (the loop below is
Gemma4UnifiedTextModel.forward minus the cache and mask plumbing, and `selftest_tiny` proves it equal
to the HF model on a tiny random config), but it can stream layers, hook any intermediate, run an
fp32 twin, and takes explicit masks (docs/gemma4-semantics.md: sliding keeps `kv > q - window`).

Numerics follow HF exactly: embeddings `* bf16(sqrt(hidden))`, RMSNorm `x*rsqrt(ms+eps)*w` in fp32
with one cast, attention scaling 1.0, `layer_scalar` applied once per layer, tied lm_head, final
softcap `30*tanh(x/30)`. `logits_mode` picks where the softcap runs (semantics doc item 14):
  fp32     lm_head accumulated in fp32, softcap in fp32 (more accurate than HF; the KL default)
  hf-bf16  lm_head output rounded to bf16 and softcapped in bf16 (what HF computes)

torch/transformers are imported lazily so `import gemma.ref` is cheap for table-only users.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma.arch import EMBED_NAME, FINAL_NORM_NAME, FULL, SLIDING, GemmaArch  # noqa: E402
from gemma.common_gemma import (  # noqa: E402
    ShardIndex,
    force_eager,
    perturb_norm_weights,
    sha256_bytes,
)

LOGPROB_CLAMP = -1e4
LOGITS_MODES = ("fp32", "hf-bf16")
ATTN_IMPLS = ("eager", "sdpa")


def token_ids_sha256(token_ids) -> str:
    """Byte-identical to full_logits_golden.token_ids_sha256 / make_tokens_json.token_ids_sha256."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def additive_mask(q_len: int, kv_len: int, past: int, window: int | None, device, dtype):
    """[1,1,q,kv] additive mask. Query j sits at position past+j; key k at position k. Causal
    (k <= pos) and, with `window`, sliding (k > pos - window): the query sees `window` keys
    including itself (docs/gemma4-semantics.md item 4)."""
    import torch

    qp = torch.arange(q_len, device=device).unsqueeze(1) + past
    kp = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = kp <= qp
    if window is not None:
        allowed &= kp > qp - window
    m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    return m.masked_fill(~allowed, torch.finfo(dtype).min).view(1, 1, q_len, kv_len)


class GemmGuardState:
    """Counters of install_gemm_guard. torch 2.9.1+rocmsdk on gfx1201 sporadically returned garbage
    GEMM tiles (|y| 1e7..1e36, ~1 in 1000 layer calls, never on a recompute; see
    tools/reference/full_logits_golden.py StreamingReference.gemm_guard_state). Clean outputs are
    untouched, so a run without events is identical to one without the guard."""

    LIMIT = 1e5

    def __init__(self):
        self.recomputed = 0
        self.confirmed_large = 0
        self.failed = 0
        self.events: list[str] = []

    def as_dict(self) -> dict:
        return {"limit": self.LIMIT, "recomputed": self.recomputed, "confirmed_large": self.confirmed_large,
                "failed": self.failed, "events": self.events[:50]}


def install_gemm_guard(module, state: GemmGuardState, tag: str) -> None:
    import torch
    import torch.nn.functional as F

    def hook(mod, args, out):
        if torch.isfinite(out).all() and out.abs().max().item() <= state.LIMIT:
            return None
        again = F.linear(args[0], mod.weight, mod.bias)
        if torch.equal(again, out):
            state.confirmed_large += 1  # real (bit-equal on recompute): keep
            return None
        ok = bool(torch.isfinite(again).all() and again.abs().max().item() <= state.LIMIT)
        state.recomputed += 1
        state.failed += 0 if ok else 1
        state.events.append(f"{tag}: recomputed a {tuple(out.shape)} linear output ({'ok' if ok else 'STILL BAD'})")
        return again

    for sub in module.modules():
        if type(sub).__name__ == "Linear":
            sub.register_forward_hook(hook)


class GemmaReference:
    source_tag = "reference"

    def __init__(self, model_dir: Path | str, device, dtype=None, max_layers: int | None = None,
                 resident: bool = True, attn: str = "eager", verbose: bool = True,
                 gemm_guard: GemmGuardState | None = None):
        import torch
        import transformers.models.gemma4_unified.modeling_gemma4_unified as modeling
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig

        if attn not in ATTN_IMPLS:
            raise ValueError(f"attn {attn!r}: expected one of {ATTN_IMPLS}")
        self.m = modeling
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = torch.bfloat16 if dtype is None else dtype
        self.verbose = verbose
        self.resident = resident
        self.attn = attn
        self.gemm_guard = gemm_guard
        self.arch = GemmaArch.from_model_dir(self.model_dir)
        self.index = ShardIndex.load(self.model_dir)
        if not self.index.weight_map:
            raise RuntimeError(f"no safetensors found under {self.model_dir}")
        self.config = Gemma4UnifiedConfig.from_pretrained(str(self.model_dir))
        self.text_config = self.config.get_text_config()
        force_eager(self.config)
        self.text_config._attn_implementation = attn
        self.max_layers = max_layers
        self.n_layers = self.arch.n_layers if max_layers is None else max_layers
        self.layer_types = list(self.arch.layer_types)
        self.vocab_size = self.arch.vocab
        self.rotary = modeling.Gemma4UnifiedTextRotaryEmbedding(self.text_config).to(device=device)
        self.final_norm_w = self.index.get_tensor(FINAL_NORM_NAME).to(device=device, dtype=self.dtype)
        self._layers: dict[int, object] = {}
        # The (tied) embedding table: 262144 x 3840 bf16 = 1.9 GB, on the device when resident.
        table = self.index.get_tensor(EMBED_NAME).to(dtype=self.dtype)
        if table.shape != (self.arch.vocab, self.arch.hidden):
            raise RuntimeError(f"{EMBED_NAME} shape {tuple(table.shape)} != config "
                               f"{(self.arch.vocab, self.arch.hidden)}")
        self.table = table.to(device) if resident else table
        self.embed_scale = torch.tensor(self.arch.hidden ** 0.5).to(self.dtype)  # bf16(sqrt(H)): 62.0 for 3840

    # -- weights ------------------------------------------------------------------------------

    def build_layer(self, i: int):
        import torch

        with torch.device("meta"):
            layer = self.m.Gemma4UnifiedTextDecoderLayer(self.text_config, i)
        prefix = self.arch.layer_prefix(i)
        state = {k: self.index.get_tensor(prefix + k).to(device=self.device, dtype=self.dtype)
                 for k in layer.state_dict()}
        layer.load_state_dict(state, assign=True)
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)
        if self.gemm_guard is not None:
            install_gemm_guard(layer, self.gemm_guard, f"L{i:02d}")
        return layer

    def layer(self, i: int):
        if i in self._layers:
            return self._layers[i]
        layer = self.build_layer(i)
        if self.resident:
            self._layers[i] = layer
        return layer

    def segment_meta(self) -> dict:
        return {"n_layers": self.n_layers, "debug_max_layers": self.max_layers}

    # -- forward ------------------------------------------------------------------------------

    def embed(self, token_ids) -> "torch.Tensor":
        import torch

        ids = torch.as_tensor([int(t) for t in token_ids], dtype=torch.long)
        rows = self.table[ids.to(self.table.device)].to(self.device)
        return rows * self.embed_scale.to(self.device)  # HF: embedding(ids) * embed_scale.to(weight.dtype)

    def forward_hidden(self, token_ids, record=None) -> "torch.Tensor":
        """Hidden states `[T, hidden]` after the final norm (bf16, on device), from position 0, no
        cache. `record(layer_index, hidden)` (if given) sees every layer's output."""
        import torch

        with torch.no_grad():
            h = self.embed(token_ids).unsqueeze(0)
            T = h.shape[1]
            pos = torch.arange(T, device=self.device).unsqueeze(0)
            pe = {lt: self.rotary(h, pos, lt) for lt in sorted(set(self.layer_types))}
            masks = {FULL: additive_mask(T, T, 0, None, self.device, self.dtype),
                     SLIDING: additive_mask(T, T, 0, self.arch.window, self.device, self.dtype)}
            shared = UserDict()
            for i in range(self.n_layers):
                layer = self.layer(i)
                lt = self.layer_types[i]
                h = layer(h, shared_kv_states=shared, position_embeddings=pe[lt], attention_mask=masks[lt],
                          position_ids=pos, past_key_values=None)
                if record is not None:
                    record(i, h)
                if not self.resident:
                    del layer
            return self.final_norm(h)[0]

    def final_norm(self, h):
        import torch

        x = h.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.arch.eps, -0.5)
        return (x * self.final_norm_w.float()).to(h.dtype)

    # -- logits -------------------------------------------------------------------------------

    def logits_rows(self, hidden, mode: str = "fp32", chunk: int = 32768):
        """Softcapped logits `[rows, V]` in fp32 for `hidden` `[rows, hidden]` (see module docstring)."""
        import torch

        if mode not in LOGITS_MODES:
            raise ValueError(f"logits mode {mode!r}: expected one of {LOGITS_MODES}")
        cap = self.arch.softcap
        out = torch.empty(hidden.shape[0], self.vocab_size, dtype=torch.float32, device=self.device)
        hf = hidden.float() if mode == "fp32" else hidden
        for s in range(0, self.vocab_size, chunk):
            e = min(s + chunk, self.vocab_size)
            w = self.table[s:e].to(self.device)
            if mode == "fp32":
                blk = hf @ w.float().T
                if cap:
                    blk = torch.tanh(blk / cap) * cap
            else:
                blk = torch.nn.functional.linear(hf, w)  # bf16, as nn.Linear
                if cap:
                    blk = torch.tanh(blk / cap) * cap     # in bf16, as HF
                blk = blk.float()
            out[:, s:e] = blk
            del w
        return out

    def logprobs_to_file(self, token_ids, f, mode: str = "fp32", chunk: int = 32768, row_block: int = 128,
                         record=None):
        """Writes rows 0..T-2 of log_softmax(logits) (fp32, clamped at LOGPROB_CLAMP, fp16) to the
        binary file `f`. Returns {"nll", "worst_lse", "last_top_logits": (values, ids) of position T-1,
        "seconds_stack", "seconds_lm_head"}."""
        import numpy as np
        import torch

        T = len(token_ids)
        t0 = time.perf_counter()
        hidden = self.forward_hidden(token_ids, record=record)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_stack = time.perf_counter() - t0
        t1 = time.perf_counter()
        nll_sum, worst = 0.0, 0.0
        for s in range(0, T - 1, row_block):
            e = min(s + row_block, T - 1)
            lg = self.logits_rows(hidden[s:e], mode, chunk)
            lp = torch.log_softmax(lg, dim=-1).clamp_min(LOGPROB_CLAMP).to(torch.float16)
            worst = max(worst, torch.logsumexp(lp.float(), dim=-1).abs().max().item())
            tgt = torch.as_tensor([int(t) for t in token_ids[s + 1:e + 1]], device=lp.device)
            nll_sum -= lp[torch.arange(e - s, device=lp.device), tgt].float().sum().item()
            f.write(np.ascontiguousarray(lp.cpu().numpy()).tobytes())
            del lg, lp
        last = self.logits_rows(hidden[-1:], mode, chunk)[0]
        top = torch.topk(last, min(5, last.numel()))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return {"nll": nll_sum / (T - 1), "worst_lse": worst, "seconds_stack": t_stack,
                "seconds_lm_head": time.perf_counter() - t1,
                "last_top": [(float(v), int(i)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
                "last_probs": torch.softmax(last, dim=-1)[top.indices].tolist()}

    def next_logits(self, token_ids, mode: str = "fp32", chunk: int = 32768):
        """Softcapped fp32 logits `[V]` of the position after `token_ids` (full recompute, no cache)."""
        hidden = self.forward_hidden(token_ids)
        return self.logits_rows(hidden[-1:], mode, chunk)[0]


# --------------------------------------------------------------------------------------------
# Tiny random-config checkpoint (the CPU smoke path of every script in this package)
# --------------------------------------------------------------------------------------------


def tiny_text_config(window: int = 4, layers: int = 4, vocab: int = 128):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedConfig,
        Gemma4UnifiedTextConfig,
    )

    types = [FULL if (i + 1) % 2 == 0 else SLIDING for i in range(layers)]  # s F s F ...
    text = Gemma4UnifiedTextConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=window, layer_types=types,
        final_logit_softcapping=30.0, max_position_embeddings=256, use_bidirectional_attention="vision",
        bos_token_id=2, pad_token_id=0, eos_token_id=1,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
    cfg = Gemma4UnifiedConfig(text_config=text)
    force_eager(cfg)
    return cfg


def build_tiny_checkpoint(out_dir: Path | str, seed: int = 0, window: int = 4, layers: int = 4,
                          vocab: int = 128) -> Path:
    """Writes a tiny random bf16 gemma4_unified checkpoint (config.json, generation_config.json,
    model.safetensors with the real on-disk tensor names) into `out_dir` and returns it. Projection
    weights are N(0, 0.1), norm weights 1 + 0.05 N(0,1) (perturb_norm_weights) and layer_scalar is
    random in [0.5, 1.5], so a dropped weight / scalar changes every output."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration

    cfg = tiny_text_config(window, layers, vocab)
    torch.manual_seed(seed)
    model = Gemma4UnifiedForConditionalGeneration(cfg)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand(b.shape, generator=gen))
    perturb_norm_weights(model, seed + 2)
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(out), safe_serialization=True)
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 2, "do_sample": True, "eos_token_id": [1, 3], "pad_token_id": 0,
         "temperature": 1.0, "top_k": 8, "top_p": 0.95}), encoding="utf-8")
    return out


def selftest_tiny(tmp_dir: Path | str, seed: int = 0) -> dict:
    """CPU check (no GPU): GemmaReference's layer loop == the HF `Gemma4UnifiedForConditionalGeneration`
    forward on a tiny random checkpoint written with the real tensor names, in resident and streaming
    mode. Returns the measured differences; raises AssertionError when they are not bit-exact."""
    import torch
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration

    d = build_tiny_checkpoint(tmp_dir, seed)
    arch = GemmaArch.from_model_dir(d)
    names = set(ShardIndex.load(d).weight_map)
    missing = [n for n in arch.text_tensor_names() if n not in names]
    assert not missing, f"arch table names not in the tiny checkpoint: {missing[:5]}"
    dev = torch.device("cpu")
    ids = [2] + [int(x) for x in torch.randint(4, arch.vocab, (11,), generator=torch.Generator().manual_seed(7))]

    hf = Gemma4UnifiedForConditionalGeneration.from_pretrained(str(d), dtype=torch.bfloat16,
                                                               attn_implementation="eager")
    hf.eval()
    with torch.no_grad():
        want = hf(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()   # [T, V] softcapped, bf16 math
    res = {"layers": arch.n_layers, "layer_types": arch.summary()["layer_types"], "window": arch.window}
    for resident in (True, False):
        ref = GemmaReference(d, dev, resident=resident)
        h = ref.forward_hidden(ids)
        got = ref.logits_rows(h, "hf-bf16")
        diff = float((got - want).abs().max())
        res[f"max_abs_diff_{'resident' if resident else 'streaming'}"] = diff
        assert diff == 0.0, f"GemmaReference (resident={resident}) differs from HF forward by {diff}"
        f32 = ref.logits_rows(h, "fp32")
        res["fp32_vs_hfbf16_max_abs"] = float((f32 - got).abs().max())
    return res


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(selftest_tiny(td), indent=2))
