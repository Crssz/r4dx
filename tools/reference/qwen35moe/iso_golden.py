"""Layer-isolated fp32 goldens: each decoder layer alone, in float32, over the bf16 golden's own `layer{L}.residual_in`.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\iso_golden.py --dir E:\\models\\r4dx\\q35moe-real4 [--golden golden_fp8] [--out golden_iso_f32]

Why: a bf16 HF run is itself a noisy reference (MoE outputs accumulate 8 routed experts plus the shared expert in bf16, one rounding
per add; r4dx combines in fp32 and rounds once), and the whole-model goldens carry the noise of the layers before. This builds
each layer of `<dir>/hf` (the tiny.py / real_prefix.py layout) as a float32 `Qwen3_5MoeDecoderLayer` (K / V rounded to fp8 e4m3 in
the attention layers, like r4dx's cache), feeds it `<dir>/<golden>/T{T}.safetensors`'s `layer{L}.residual_in`, and writes the layer's
outputs under the golden's tensor names (bf16 on store) to `<dir>/<out>/T{T}.safetensors`; `input_ids` and every `layer{L}.residual_in`
are copied. tests/model/qwen35moe/test_model_q35moe_tiny.cpp judges its layer-isolated runs against it when R4DX_Q35MOE_ISO_GOLDEN
names the directory. One fp32 layer of the real model is ~10 GB of weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen35moe import tiny_golden  # noqa: E402

PROMPTS = (7, 70, 300)


def build_layer(hf_dir: Path, text_cfg, layer: int):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    mod = Qwen3_5MoeDecoderLayer(text_cfg, layer).float()
    prefix = f"model.language_model.layers.{layer}."
    sd = {}
    with safe_open(str(hf_dir / "model-00001-of-00001.safetensors"), framework="pt") as f:
        for name in f.keys():
            if name.startswith(prefix):
                sd[name[len(prefix):]] = f.get_tensor(name).float()
    missing, unexpected = mod.load_state_dict(sd, strict=True), None
    mod.eval()
    return mod


def run_layer(mod, text_cfg, rotary, x_bf16: torch.Tensor, capture: dict, p: str):
    T = x_bf16.shape[0]
    x = x_bf16.float().unsqueeze(0)
    pos = torch.arange(T).view(1, 1, T).expand(3, 1, T)
    cos, sin = rotary(x, pos)
    handles = []

    def put(name, t, batched=True):
        t = t.detach()
        capture[p + name] = (t[0] if batched else t).clone()

    handles.append(mod.input_layernorm.register_forward_hook(lambda m, a, o: put("attn_in", o)))
    mixer = mod.linear_attn if mod.block_type == "linear_attention" else mod.self_attn
    handles.append(mixer.register_forward_hook(lambda m, a, o: put("attn_out", o if torch.is_tensor(o) else o[0])))
    handles.append(mod.post_attention_layernorm.register_forward_pre_hook(lambda m, a: put("residual_attn", a[0])))
    handles.append(mod.post_attention_layernorm.register_forward_hook(lambda m, a, o: put("mlp_in", o)))
    handles.append(mod.mlp.register_forward_hook(lambda m, a, o: put("moe_out", o if torch.is_tensor(o) else o[0])))

    def router(m, a, out):
        capture[p + "router_logits"] = out[0].detach().clone()
        capture[p + "topk_weights"] = out[1].detach().clone()
        capture[p + "topk_ids"] = out[2].detach().clone()

    handles.append(mod.mlp.gate.register_forward_hook(router))
    handles.append(mod.mlp.shared_expert.register_forward_hook(lambda m, a, o: put("shared_out", o, False)))
    handles.append(mod.mlp.shared_expert_gate.register_forward_hook(lambda m, a, o: put("shared_gate_logit", o, False)))
    try:
        with torch.no_grad():
            out = mod(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos[0])
        put("residual_out", out if torch.is_tensor(out) else out[0])
    finally:
        for h in handles:
            h.remove()


def main(argv=None) -> int:
    from transformers import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--golden", default="golden_fp8")
    ap.add_argument("--out", default="golden_iso_f32")
    ap.add_argument("--prompts", type=int, nargs="*", default=list(PROMPTS))
    args = ap.parse_args(argv)
    root = Path(args.dir)
    hf = root / "hf"
    cfg = Qwen3_5MoeConfig.from_pretrained(str(hf))
    text_cfg = cfg.text_config
    text_cfg._attn_implementation = tiny_golden.install_fp8_kv_attention()
    rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    goldens = {t: load_file(str(root / args.golden / f"T{t}.safetensors")) for t in args.prompts}
    results = {t: {"input_ids": goldens[t]["input_ids"]} for t in args.prompts}
    for layer in range(text_cfg.num_hidden_layers):
        mod = build_layer(hf, text_cfg, layer)
        print(f"[iso] layer {layer} ({mod.block_type}) built", flush=True)
        for t in args.prompts:
            p = f"layer{layer}."
            x = goldens[t][p + "residual_in"]
            results[t][p + "residual_in"] = x.clone()
            cap: dict[str, torch.Tensor] = {}
            run_layer(mod, text_cfg, rotary, x, cap, p)
            for k, v in cap.items():
                results[t][k] = v
            print(f"[iso]   T={t}: {len(cap)} tensors", flush=True)
        del mod
    for t in args.prompts:
        save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in results[t].items()},
                  str(out_dir / f"T{t}.safetensors"))
    # the tests also read the free runs; copy the bf16 golden's (unused by the isolated comparison)
    for t in args.prompts:
        src = root / args.golden / f"free_T{t}.json"
        if src.exists():
            (out_dir / src.name).write_text(src.read_text())
    print(f"[iso] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
