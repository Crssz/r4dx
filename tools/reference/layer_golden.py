"""tools/reference/layer_golden.py

Builds Qwen3_5 text-decoder layers, one at a time, straight from the reference `transformers`
install's `modeling_qwen3_5.py` -- real weights when the shard is downloaded, deterministic
seeded random init otherwise -- runs them on a fixed seeded hidden-state input, and dumps
inputs/outputs/intermediates to .safetensors + a manifest.json. This is the contract the r4dx C++
engine's tests compare their own kernel outputs against (docs/validation.md, rung 3).

Covers, per the task: one GDN layer (layer 0), one full-attention layer (layer 3), final norm +
lm_head on a tiny vocab slice, and the MTP block (random-init fallback -- see the "mtp" section of
the manifest and docs/validation.md: transformers 5.17.0 has no Qwen3_5 MTP forward implementation
to load real weights into, so this exercises the container/loader's tensor *shapes* only).

Usage:
    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\reference\\layer_golden.py ^
        --device cuda --out-dir tools\\reference\\golden_out

See tools/reference/README.md for the full option list and expected runtime.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    ShardIndex,
    force_eager,
    load_module_state,
    load_text_config,
    resolve_device,
    save_golden,
    set_seed,
    sha256_file,
    tensor_manifest_entry,
)

TOLERANCES = {
    "bf16_matmul_rel_err": 2e-2,
    "fp32_elementwise_rel_err": 1e-4,
    "recurrent_state_rel_err": 5e-2,
    "note": "bf16_matmul_rel_err for anything that ends in a r4d_gemm_* call (attn qkv/o, GDN "
    "in/out proj, MLP, lm_head); fp32_elementwise_rel_err for norms/gates/activations computed "
    "in fp32 by the reference; recurrent_state_rel_err is looser because the GDN state accumulates "
    "64 (chunk) or more matmuls of drift before the golden's chunk boundary.",
}


def causal_additive_mask(q_len: int, kv_len: int, past_len: int, device, dtype) -> torch.Tensor:
    """[1, 1, q_len, kv_len] additive mask: 0 where key <= query position, dtype-min elsewhere."""
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + past_len
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = k_pos <= q_pos
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask.view(1, 1, q_len, kv_len)


def mrope_position_ids(total_len: int, device) -> torch.Tensor:
    """[3, 1, total_len] position ids. This golden is text-only (no vision tokens), so all three
    mrope sections (t, h, w) collapse to the same sequential index -- the standard fallback for a
    pure-text sequence (see Qwen3_5Model.get_rope_index for the general, image-aware case)."""
    idx = torch.arange(total_len, device=device)
    return idx.view(1, 1, total_len).expand(3, 1, total_len).clone()


#: Tensors `GdnCapture` guarantees on every successful GDN call (see `_wrap_rule` below). If the
#: `kernels`/`fla`/`causal_conv1d` optional hub packages are installed in the reference venv, they
#: can rebind `Qwen3_5GatedDeltaNet`'s module-level free functions to a real accelerated kernel
#: before this monkeypatch takes effect -- `run_text_layer` checks this set is a subset of
#: `cap.record` after every GDN call and raises rather than silently writing a golden with the GDN
#: intermediates missing (see finding 6 in the Opus review this fixes).
REQUIRED_GDN_KEYS = {
    "gdn_q", "gdn_k", "gdn_v", "gdn_g", "gdn_beta", "gdn_core_attn_out",
    "gdn_q_l2", "gdn_k_l2", "gdn_g_chunk_cumsum",
}


class GdnCapture:
    """Monkeypatches the module-level free functions Qwen3_5GatedDeltaNet.forward calls into
    (causal_conv1d_fn/update, torch_{chunk,recurrent}_gated_delta_rule) to record the
    intermediates transformers doesn't otherwise expose, in TWO forms per tensor:

    - the RAW form actually passed to/returned by the callee (`gdn_q`/`gdn_k`/`gdn_v`/`gdn_g`/
      `gdn_beta`/`gdn_conv_out_*`): pre-l2norm, pre-(1/sqrt(K))-scale, POST-repeat_interleave (i.e.
      GQA-expanded from Hg=16 key/query heads up to H=48 value heads) for q/k, and PER-TOKEN
      (not chunk-summed) for g. `Qwen3_5GatedDeltaNet` calls
      `torch_{chunk,recurrent}_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`, so l2norm AND
      the query scaling happen *inside* the callee, after this wrapper's clone -- these raw tensors
      are NOT normalized/scaled despite what an earlier version of this docstring claimed.
    - the r4d-shaped form (`gdn_q_l2`/`gdn_k_l2`/`gdn_g_chunk_cumsum`/`gdn_core_attn_out`): what
      `r4d_gdn_conv_prep_w4_h128_bf16` actually emits and `r4d_gdn_kkt_solve_k128_c64_bf16` /
      `r4d_gdn_chunk_scan_k128_v128_c64_bf16` actually consume, per r4d.h's layout comments (q,k
      `[T, Hg, K]` l2-normed but NOT scaled -- `r4d_gdn_chunk_scan_k128_v128_c64_bf16` takes the
      `1/sqrt(K)` scale as its own `scale` float argument, matching
      `torch_chunk_gated_delta_rule`'s `query = query * query.shape[-1] ** -0.5` applied after
      l2norm; g `[T, H]` fp32 already cumulatively summed within each 64-token chunk; o/core_attn_out
      `[T, H, V]`, the chunk-scan kernel's raw output BEFORE the gated RMSNorm).
      `gdn_q_l2`/`gdn_k_l2` are de-interleaved from the raw GQA-expanded form back to `Hg` heads by
      taking every `gqa_repeats`-th head (`repeat_interleave` duplicates each head's vector
      `gqa_repeats` times consecutively, and l2norm of two identical vectors is itself identical, so
      de-interleaving after l2norm is exactly equivalent to de-interleaving the original Hg-head
      tensor before l2norm/GQA-expansion -- no information is lost).

    Also records `gdn_conv_out_fn_full`/`gdn_conv_out_fn_trimmed` (from `causal_conv1d_fn`) and
    `gdn_conv_out_update` (from `causal_conv1d_update`, already exactly `stage_seq_len` columns).
    `causal_conv1d_fn` itself does NOT trim its cache-prepended input down to the new tokens --
    `Qwen3_5GatedDeltaNet.forward` does that with an extra `mixed_qkv[:, :, -seq_len:]` slice AFTER
    the call this wrapper intercepts (modeling_qwen3_5.py's GDN forward) -- so `..._full` is the raw,
    still-cache-prefixed return value and `..._trimmed` replicates that external slice using
    `self.stage_seq_len` (the caller MUST set this before invoking the layer). Only `..._trimmed` is
    the correct golden for `r4d_gdn_conv_update_w4_h128_bf16`'s output shape.

    Safe regardless of whether the `fla`/`causal_conv1d` optional packages are installed in the
    reference venv (they are not, as of this writing): the wrapper captures whatever the
    currently-bound implementation returns, it does not depend on which implementation that is --
    but see `REQUIRED_GDN_KEYS` above for what happens if one of those packages rebinds the
    functions out from under this monkeypatch.
    """

    PATCH_NAMES = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )

    def __init__(self, modeling_module, gqa_repeats: int):
        self.m = modeling_module
        self.gqa_repeats = gqa_repeats
        self.stage_seq_len: int | None = None  # caller MUST set this before invoking the layer
        self.originals: dict[str, object] = {}
        self.record: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for name in self.PATCH_NAMES:
            self.originals[name] = getattr(self.m, name)
        self.m.causal_conv1d_fn = self._wrap_conv(self.originals["causal_conv1d_fn"], "fn")
        self.m.causal_conv1d_update = self._wrap_conv(self.originals["causal_conv1d_update"], "update")
        self.m.torch_chunk_gated_delta_rule = self._wrap_rule(self.originals["torch_chunk_gated_delta_rule"])
        self.m.torch_recurrent_gated_delta_rule = self._wrap_rule(self.originals["torch_recurrent_gated_delta_rule"])
        return self

    def __exit__(self, *exc_info):
        for name, fn in self.originals.items():
            setattr(self.m, name, fn)
        return False

    def _wrap_conv(self, original, kind):
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            if kind == "fn":
                self.record["gdn_conv_out_fn_full"] = out.detach().clone()
                assert self.stage_seq_len is not None, "GdnCapture.stage_seq_len not set before layer() call"
                self.record["gdn_conv_out_fn_trimmed"] = out[:, :, -self.stage_seq_len :].detach().clone()
            else:  # "update" -- causal_conv1d_update already returns exactly stage_seq_len columns
                self.record["gdn_conv_out_update"] = out.detach().clone()
            return out

        return wrapped

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Matches modeling_qwen3_5.py's module-level `l2norm` exactly."""
        inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
        return x * inv_norm

    def _wrap_rule(self, original):
        def wrapped(query, key, value, g, beta, *args, **kwargs):
            self.record["gdn_q"] = query.detach().clone()
            self.record["gdn_k"] = key.detach().clone()
            self.record["gdn_v"] = value.detach().clone()
            self.record["gdn_g"] = g.detach().clone()
            self.record["gdn_beta"] = beta.detach().clone()

            # r4d-shaped q/k: l2-normed (matching use_qk_l2norm_in_kernel=True), NOT scaled by
            # 1/sqrt(K) (that's r4d_gdn_chunk_scan_k128_v128_c64_bf16's own `scale` argument), and
            # de-interleaved from the GQA-expanded H=48 heads back down to Hg=16.
            q_l2_full = self._l2norm(query.detach().to(torch.float32))
            k_l2_full = self._l2norm(key.detach().to(torch.float32))
            self.record["gdn_q_l2"] = q_l2_full[:, :, :: self.gqa_repeats, :].detach().clone()
            self.record["gdn_k_l2"] = k_l2_full[:, :, :: self.gqa_repeats, :].detach().clone()

            # r4d-shaped g: cumulatively summed (log-space decay) within each 64-token chunk, exactly
            # matching torch_chunk_gated_delta_rule's internal `cum_decay = decay.cumsum(dim=3)`
            # after reshaping into [.., num_chunks, chunk_size].
            chunk_size = 64
            g_bht = g.detach().to(torch.float32).transpose(1, 2)  # [B, H, T]
            total_len = g_bht.shape[-1]
            pad = (chunk_size - total_len % chunk_size) % chunk_size
            g_padded = torch.nn.functional.pad(g_bht, (0, pad))
            g_chunks = g_padded.reshape(g_padded.shape[0], g_padded.shape[1], -1, chunk_size)
            g_cumsum = g_chunks.cumsum(dim=3).reshape(g_padded.shape[0], g_padded.shape[1], -1)
            g_cumsum = g_cumsum[..., :total_len].transpose(1, 2)  # back to [B, T, H]
            self.record["gdn_g_chunk_cumsum"] = g_cumsum.detach().clone()

            # Always ask the real implementation for the final recurrent state, regardless of what
            # the caller (Qwen3_5GatedDeltaNet.forward) requested -- harmless when cache_params is
            # None (the caller just never reads it), and it's the only way to see the state produced
            # by a plain prefill call (no cache) at all.
            forced = dict(kwargs)
            forced["output_final_state"] = True
            out, state = original(query, key, value, g, beta, *args, **forced)
            self.record["gdn_core_attn_out"] = out.detach().clone()
            if state is not None:
                self.record["gdn_recurrent_state"] = state.detach().clone()
            return out, state

        return wrapped


def build_layer(m, layer_config, layer_idx: int, device, dtype):
    """Constructs one Qwen3_5DecoderLayer. Caller is responsible for seeding torch's RNG first --
    the layer's default (unseeded-here) init is exactly the deterministic random fallback we want
    when the checkpoint doesn't have this layer's weights."""
    layer = m.Qwen3_5DecoderLayer(layer_config, layer_idx)
    layer = layer.to(device=device, dtype=dtype)
    layer.eval()
    return layer


def perturb_norm_weights_for_smoke(module: torch.nn.Module, seed: int) -> bool:
    """In --force-random-init / weight-fallback mode every Qwen3_5RMSNorm(Gated) weight defaults
    to exactly zero (`nn.Parameter(torch.zeros(dim))`), and the norm's forward is zero-centered
    (`output * (1.0 + weight)`) -- so an all-zero weight makes that scale identically 1.0 and the
    random-init smoke path can never catch a missing/incorrect `(1 + w)` term or a dropped weight
    multiply, a classic port bug for this norm convention. Perturbs every RMSNorm-family weight
    found under `module` with small seeded noise so the scale path is actually exercised. Uses its
    own local `torch.Generator` (does not touch global RNG state), so it's safe to call at any point
    without disturbing later `torch.manual_seed`-based determinism elsewhere in the caller. Returns
    True if any weight was perturbed (i.e. there was something to perturb)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perturbed = False
    for sub in module.modules():
        if type(sub).__name__ in ("Qwen3_5RMSNorm", "Qwen3_5RMSNormGated") and hasattr(sub, "weight"):
            with torch.no_grad():
                noise = (torch.randn(sub.weight.shape, generator=gen) * 0.02).to(
                    dtype=sub.weight.dtype, device=sub.weight.device
                )
                sub.weight.copy_(noise)
            perturbed = True
    return perturbed


def run_text_layer(
    m,
    text_config,
    index: ShardIndex,
    device,
    dtype,
    layer_idx: int,
    layer_type: str,
    hf_prefix: str,
    seed: int,
    prefill_len: int,
    decode_len: int,
) -> dict:
    """Runs one decoder layer (GDN or full-attention) through a seeded prefill + a cached decode
    step, capturing every intermediate the task asks for. Returns a dict of {tensor_name: tensor}
    plus a "meta" sub-dict describing weight provenance -- the caller writes both out."""
    from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer

    set_seed(seed)
    layer_config = copy.deepcopy(text_config)
    layer_config.layer_types = list(text_config.layer_types)
    layer_config.layer_types[layer_idx] = layer_type
    layer = build_layer(m, layer_config, layer_idx, device, dtype)

    loaded, missing, error = load_module_state(layer, index, hf_prefix)
    weights_source = "safetensors" if loaded else f"random_init(seed={seed})"
    norm_weights_perturbed = False
    if not loaded:
        norm_weights_perturbed = perturb_norm_weights_for_smoke(layer, seed + 9000)

    gqa_repeats = 1
    if layer_type == "linear_attention":
        gqa_repeats = layer_config.linear_num_value_heads // layer_config.linear_num_key_heads

    hidden_size = text_config.hidden_size
    total_len = prefill_len + decode_len

    set_seed(seed)
    full_hidden = torch.randn(1, total_len, hidden_size, generator=torch.Generator(device="cpu").manual_seed(seed))
    full_hidden = full_hidden.to(device=device, dtype=dtype)
    prefill_hidden = full_hidden[:, :prefill_len, :]
    decode_hidden = full_hidden[:, prefill_len:, :]

    rotary = m.Qwen3_5TextRotaryEmbedding(text_config).to(device=device)
    pos_ids = mrope_position_ids(total_len, device)
    cos, sin = rotary(full_hidden, pos_ids)
    prefill_cos, prefill_sin = cos[:, :prefill_len], sin[:, :prefill_len]
    decode_cos, decode_sin = cos[:, prefill_len:], sin[:, prefill_len:]

    # Cache.update_conv_state/update_recurrent_state index `self.layers[layer_idx]` directly (no
    # lazy-fill, unlike the attention-style `.update()` path) -- pre-populate an explicit `layers`
    # list up to `layer_idx` so it exists. Only index `layer_idx` is ever touched by our
    # single-layer forward calls; the padding entries ahead of it are never used.
    if layer_type == "linear_attention":
        cache = Cache(layers=[LinearAttentionLayer() for _ in range(layer_idx + 1)])
        prefill_mask = None
        decode_mask = None
    else:
        cache = Cache(layers=[DynamicLayer() for _ in range(layer_idx + 1)])
        prefill_mask = causal_additive_mask(prefill_len, prefill_len, 0, device, dtype)
        decode_mask = causal_additive_mask(decode_len, total_len, prefill_len, device, dtype)

    tensors: dict[str, torch.Tensor] = {
        "input_hidden_states": full_hidden[0],
        "prefill_hidden_states": prefill_hidden[0],
        "decode_hidden_states": decode_hidden[0],
    }

    hook_handles = []
    hook_sink: dict[str, torch.Tensor] = {}

    def mk_hook(key):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            hook_sink[key] = out.detach().clone()

        return hook

    def mk_pre_hook(key):
        def hook(_module, inputs):
            hook_sink[key] = inputs[0].detach().clone()

        return hook

    def gate_sigmoid_hook(_module, _inputs, output):
        # Replicates Qwen3_5Attention.forward's own gate split exactly: q_proj's raw output is
        # [..., num_heads, 2*head_dim] per head (query first half, gate second half), and the gate
        # is later applied as attn_output * sigmoid(gate) -- captured here directly rather than by
        # patching torch.sigmoid globally.
        head_dim = layer.self_attn.head_dim
        reshaped = output.view(*output.shape[:-1], -1, head_dim * 2)
        _, gate = reshaped.chunk(2, dim=-1)
        hook_sink["attn_gate_sigmoid"] = torch.sigmoid(gate.detach().float()).clone()

    if layer_type == "full_attention":
        hook_handles = [
            layer.self_attn.q_proj.register_forward_hook(mk_hook("attn_qg_raw")),
            layer.self_attn.k_proj.register_forward_hook(mk_hook("attn_k_raw")),
            layer.self_attn.v_proj.register_forward_hook(mk_hook("attn_v_raw")),
            layer.self_attn.q_norm.register_forward_hook(mk_hook("attn_q_normed")),
            layer.self_attn.k_norm.register_forward_hook(mk_hook("attn_k_normed")),
            layer.self_attn.register_forward_hook(mk_hook("attention_output")),
            layer.self_attn.q_proj.register_forward_hook(gate_sigmoid_hook),
            layer.self_attn.o_proj.register_forward_pre_hook(mk_pre_hook("attn_post_gate")),
        ]
    common_hook_handles = [
        layer.input_layernorm.register_forward_hook(mk_hook("post_input_norm")),
        layer.post_attention_layernorm.register_forward_hook(mk_hook("post_attn_norm")),
        layer.mlp.register_forward_hook(mk_hook("mlp_output")),
    ]

    def run_stage(hidden, mask, cos_, sin_, stage: str):
        hook_sink.clear()
        with torch.no_grad():
            if layer_type == "linear_attention":
                with GdnCapture(m, gqa_repeats) as cap:
                    cap.stage_seq_len = hidden.shape[1]
                    out = layer(
                        hidden_states=hidden,
                        position_embeddings=(cos_, sin_),
                        attention_mask=mask,
                        position_ids=None,
                        past_key_values=cache,
                    )
                missing_gdn = REQUIRED_GDN_KEYS - set(cap.record)
                if missing_gdn:
                    raise RuntimeError(
                        f"GdnCapture failed to record {sorted(missing_gdn)} for stage={stage!r} -- "
                        "an optional hub kernel package (`kernels`, `fla`, `causal_conv1d`) may be "
                        "installed in this venv and rebound Qwen3_5GatedDeltaNet's module-level "
                        "functions before this monkeypatch could intercept them; see "
                        "tools/reference/README.md."
                    )
                for k, v in cap.record.items():
                    tensors[f"{stage}_{k}"] = v[0] if v.dim() > 0 and v.shape[0] == 1 else v
            else:
                # Reuse kv_calibrate.py's apply_rotary_pos_emb monkeypatch to capture post-rope q/k
                # -- transformers exposes no hook point between q_norm/k_norm (pre-rope) and the
                # attention call (post-rope), and the fp8 paged KV cache stores post-rope K.
                original_rope = m.apply_rotary_pos_emb

                def capture_rope(q, k, cos__, sin__, unsqueeze_dim=1):
                    q_embed, k_embed = original_rope(q, k, cos__, sin__, unsqueeze_dim)
                    hook_sink["attn_q_post_rope"] = q_embed.detach().clone()
                    hook_sink["attn_k_post_rope"] = k_embed.detach().clone()
                    return q_embed, k_embed

                m.apply_rotary_pos_emb = capture_rope
                try:
                    out = layer(
                        hidden_states=hidden,
                        position_embeddings=(cos_, sin_),
                        attention_mask=mask,
                        position_ids=None,
                        past_key_values=cache,
                    )
                finally:
                    m.apply_rotary_pos_emb = original_rope
        for k, v in hook_sink.items():
            tensors[f"{stage}_{k}"] = v[0] if v.dim() > 0 and v.shape[0] == 1 else v
        tensors[f"{stage}_layer_output"] = out[0]
        return out

    try:
        run_stage(prefill_hidden, prefill_mask, prefill_cos, prefill_sin, "prefill")
        run_stage(decode_hidden, decode_mask, decode_cos, decode_sin, "decode")
        status = "ok"
        run_error = None
    except Exception as exc:  # keep going -- a partial golden with the error recorded still helps
        status = "error"
        run_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        for h in hook_handles + common_hook_handles:
            h.remove()

    meta = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "hf_weight_prefix": hf_prefix,
        "weights_source": weights_source,
        "norm_weights_perturbed": norm_weights_perturbed,
        "missing_weight_keys": missing,
        "load_error": error,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "seed": seed,
        "status": status,
        "run_error": run_error,
        "note": (
            "decode uses a Cache continued from the prefill call, so the GDN branch takes the "
            "chunked-gated-delta-rule path with a carried initial_state (decode_len > 1), matching "
            "architecture.md's speculative-window note -- it does NOT exercise "
            "r4d_gdn_recurrent_update_k128_v128_bf16_fp32state's single-token path "
            "(decode_len == 1); see docs/validation.md open issues."
            if layer_type == "linear_attention"
            else "decode reuses the prefill Cache (paged-decode-equivalent growth of K/V)."
        ),
    }
    return tensors, meta


def run_final_norm_lm_head(m, text_config, index: ShardIndex, device, dtype, seed, tiny_vocab, prefill_len):
    set_seed(seed + 1000)
    hidden_size = text_config.hidden_size
    hidden = torch.randn(prefill_len, hidden_size, generator=torch.Generator(device="cpu").manual_seed(seed + 1000))
    hidden = hidden.to(device=device, dtype=dtype)

    final_norm = m.Qwen3_5RMSNorm(hidden_size, eps=text_config.rms_norm_eps).to(device=device, dtype=dtype)
    norm_loaded, norm_missing, norm_error = load_module_state(final_norm, index, "model.language_model.norm.")
    final_norm_weights_perturbed = False
    if not norm_loaded:
        final_norm_weights_perturbed = perturb_norm_weights_for_smoke(final_norm, seed + 9001)

    vocab_size = text_config.vocab_size
    lm_head_name = "lm_head.weight"
    lm_head_loaded = False
    lm_head_error = None
    if index.available(lm_head_name):
        try:
            lm_head_slice = index.get_row_slice(lm_head_name, 0, tiny_vocab).to(device=device, dtype=dtype)
            lm_head_loaded = True
        except Exception as exc:
            lm_head_error = f"{type(exc).__name__}: {exc}"
    if not lm_head_loaded:
        gen = torch.Generator(device="cpu").manual_seed(seed + 1001)
        lm_head_slice = (torch.randn(tiny_vocab, hidden_size, generator=gen) * 0.02).to(device=device, dtype=dtype)

    with torch.no_grad():
        normed = final_norm(hidden)
        logits = torch.nn.functional.linear(normed, lm_head_slice)

    tensors = {
        "hidden_states": hidden,
        "final_norm_output": normed,
        "lm_head_weight_slice": lm_head_slice,
        "logits": logits,
    }
    meta = {
        "tiny_vocab": tiny_vocab,
        "full_vocab_size": vocab_size,
        "prefill_len": prefill_len,
        "final_norm_weights_source": "safetensors" if norm_loaded else f"random_init(seed={seed + 1000})",
        "final_norm_weights_perturbed": final_norm_weights_perturbed,
        "final_norm_load_error": norm_error,
        "lm_head_weights_source": "safetensors[0:tiny_vocab]" if lm_head_loaded else f"random_init(seed={seed + 1001})",
        "lm_head_load_error": lm_head_error,
        "seed": seed,
        "status": "ok",
    }
    return tensors, meta


MTP_AUX_TENSORS = ("mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight")


def run_mtp(m, text_config, index: ShardIndex, device, dtype, seed, prefill_len, decode_len):
    # transformers 5.17.0 has no Qwen3_5 MTP module to run (Qwen3_5PreTrainedModel
    # `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` -- it's dropped on load, not implemented).
    # Empirically (checked directly against this checkpoint's model.safetensors.index.json), the
    # real tensor set is NOT a flat "mtp.*" mirror of one text layer as docs/container-format.md's
    # prose summary implies -- it's `mtp.layers.0.*` (one full_attention-type Qwen3_5DecoderLayer,
    # confirmed by the presence of self_attn.* and absence of linear_attn.* keys) plus four
    # top-level tensors that fuse the previous hidden state with the next token's embedding before
    # feeding the layer: `mtp.pre_fc_norm_hidden`/`mtp.pre_fc_norm_embedding` (RMSNorm weights),
    # `mtp.fc.weight` ([hidden, 2*hidden], presumably `fc(cat(norm_h(hidden), norm_e(embed)))`,
    # Eagle/DeepSeek-MTP-style), and `mtp.norm.weight` (a final norm after the layer, before
    # whatever lm_head the MTP head reuses). This script exercises only the inner decoder layer
    # (real weights now that they're loadable) with a plain hidden-state input, same as the other
    # layer goldens; it does NOT run the fc-fusion path (no verified reference implementation to
    # copy the math from) -- the four aux tensors below are dumped raw, un-exercised, for the
    # converter to at least round-trip shapes/values against. See docs/validation.md.
    mtp_prefix_present = len(index.names_with_prefix("mtp.")) > 0
    mtp_config = copy.deepcopy(text_config)
    mtp_config.num_hidden_layers = 1
    mtp_config.layer_types = ["full_attention"]
    tensors, meta = run_text_layer(
        m, mtp_config, index, device, dtype,
        layer_idx=0, layer_type="full_attention", hf_prefix="mtp.layers.0.",
        seed=seed + 2000, prefill_len=prefill_len, decode_len=decode_len,
    )

    aux_loaded, aux_missing = [], []
    for name in MTP_AUX_TENSORS:
        key = name.rsplit(".", 1)[0].replace(".", "_")  # "mtp.fc.weight" -> "mtp_fc"
        if index.available(name):
            try:
                tensors[f"{key}_weight_raw"] = index.get_tensor(name).to(device=device, dtype=dtype)
                aux_loaded.append(name)
                continue
            except Exception:
                pass
        aux_missing.append(name)

    meta["mtp_weights_present_in_checkpoint"] = mtp_prefix_present
    meta["mtp_aux_tensors_loaded"] = aux_loaded
    meta["mtp_aux_tensors_missing"] = aux_missing
    meta["architecture_assumption"] = (
        "inner block = one full_attention-style Qwen3_5DecoderLayer (verified against this "
        "checkpoint's tensor names: mtp.layers.0.self_attn.* present, no mtp.layers.0.linear_attn.*); "
        "the fc/embedding-fusion path (mtp.fc, mtp.pre_fc_norm_*, mtp.norm) is NOT run here, only "
        "dumped raw -- transformers 5.17.0 has no reference forward to copy that math from"
    )
    return tensors, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "golden_out")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prefill-len", type=int, default=64)
    ap.add_argument("--decode-len", type=int, default=4)
    ap.add_argument("--tiny-vocab", type=int, default=256)
    ap.add_argument("--gdn-layer", type=int, default=0, help="layer index to treat as the GDN case")
    ap.add_argument("--attn-layer", type=int, default=3, help="layer index to treat as the full-attention case")
    ap.add_argument("--skip-mtp", action="store_true")
    ap.add_argument(
        "--force-random-init",
        action="store_true",
        help="skip the checkpoint entirely (CPU-friendly smoke mode, e.g. for tests/reference/test_manifest.py)",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # GDN's chunked scan (chunk_size=64) needs float64-capable triangular solves on CPU for
    # torch.linalg.solve_triangular in some torch builds; float32 is fine and matches the "random
    # init smoke test" use case (CPU, small T) this branch exists for.

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    if args.force_random_init or not (args.model_dir / "config.json").exists():
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

        full_config = Qwen3_5Config()
        text_config = full_config.text_config
        force_eager(text_config)
        config_sha256 = None
        index = ShardIndex(model_dir=args.model_dir, weight_map={})
    else:
        full_config, text_config = load_text_config(args.model_dir)
        config_sha256 = sha256_file(args.model_dir / "config.json")
        index = ShardIndex.load(args.model_dir)

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "config_sha256": config_sha256,
        "device": str(device),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "seed": args.seed,
        "tolerances": TOLERANCES,
        "components": {},
    }

    components = [
        ("layer_%03d_gdn" % args.gdn_layer, "gdn", args.gdn_layer, "linear_attention"),
        ("layer_%03d_full_attention" % args.attn_layer, "attn", args.attn_layer, "full_attention"),
    ]
    for out_name, _kind, layer_idx, layer_type in components:
        hf_prefix = f"model.language_model.layers.{layer_idx}."
        tensors, meta = run_text_layer(
            m, text_config, index, device, dtype,
            layer_idx=layer_idx, layer_type=layer_type, hf_prefix=hf_prefix,
            seed=args.seed, prefill_len=args.prefill_len, decode_len=args.decode_len,
        )
        out_path = args.out_dir / f"{out_name}.safetensors"
        save_golden(out_path, tensors, extra_meta={"component": out_name})
        meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
        meta["file"] = out_path.name
        manifest["components"][out_name] = meta
        print(f"[layer_golden] {out_name}: status={meta['status']} weights={meta['weights_source']}")

    tensors, meta = run_final_norm_lm_head(
        m, text_config, index, device, dtype, args.seed, args.tiny_vocab, args.prefill_len
    )
    out_path = args.out_dir / "final_norm_lm_head.safetensors"
    save_golden(out_path, tensors, extra_meta={"component": "final_norm_lm_head"})
    meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
    meta["file"] = out_path.name
    manifest["components"]["final_norm_lm_head"] = meta
    print(f"[layer_golden] final_norm_lm_head: status={meta['status']}")

    if not args.skip_mtp:
        tensors, meta = run_mtp(m, text_config, index, device, dtype, args.seed, args.prefill_len, args.decode_len)
        out_path = args.out_dir / "mtp.safetensors"
        save_golden(out_path, tensors, extra_meta={"component": "mtp"})
        meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
        meta["file"] = out_path.name
        manifest["components"]["mtp"] = meta
        print(f"[layer_golden] mtp: status={meta['status']} weights_present={meta['mtp_weights_present_in_checkpoint']}")

    manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[layer_golden] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
