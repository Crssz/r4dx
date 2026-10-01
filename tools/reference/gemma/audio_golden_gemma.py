"""CPU-only golden for the Gemma 4 (gemma4_unified) audio path (docs/gemma4-plan.md M3-1, docs/gemma4-audio.md).

NO GPU: HIP/CUDA are hidden before torch is imported and the script asserts it. The only checkpoint bytes
read are the 4.9 MB `model.embed_audio.embedding_projection.weight` tensor (raw seek+read, never mmap).

What it checks against transformers' own gemma4_unified code, and dumps for the C++ tests:
  1. Feature extractor: Gemma4UnifiedAudioFeatureExtractor on synthetic 16 kHz waveforms of many lengths
     (1, 639, 640, 641, 16000, 480000 = the 750-token cap, 480001) == the plain "right zero-pad to a multiple
     of 640, reshape [n, 640]" rule; mask all True; n == ceil(len / 640) == Processor._compute_audio_num_tokens.
  2. Embedder: Gemma4UnifiedMultimodalEmbedder with the REAL checkpoint weight == the explicit recipe
     (bf16 cast of the fp32 frames, RMSNorm(640, no weight, eps 1e-6) with fp32 math and a bf16 result,
     Linear 640 -> 3840 no bias with fp32 accumulate and a bf16 result). The bf16 cast of the waveform BEFORE
     the norm is the one non-obvious step (embedding_projection.weight.dtype is bf16).
  3. Processor + chat template on the real tokenizer dir: a user turn with one input_audio part renders
     `<|audio|>` once, the processor expands it to `<|audio>` + n x `<|audio|>` + `<audio|>`
     (ids 256000, 258881 x n, 258883); ids and counts are dumped.
  4. Causality: a tiny random gemma4_unified model with an audio config; perturbing audio frame k leaves the
     hidden states of every position < (position of frame k) bit-identical, and the generate-path masks give
     audio tokens (mm_token_type_ids 3) a plain causal mask.
  5. Cap: the model does not enforce 750; only the processor's audio_seq_length does (recorded).

Outputs (default <out-dir> = tools/reference/golden_out, gitignored):
  <out>/gemma/audio_golden.safetensors   weight [3840,640] bf16, per case frames_<i> f32 [n,640] and
                                         expect_<i> bf16 [n,3840]; metadata lists the cases
  <out>/gemma/audio_semantics.json       the extractor / processor / causality findings

  D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\audio_golden_gemma.py
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert not torch.cuda.is_available(), "audio_golden_gemma.py is CPU-only; a GPU is visible"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import raw_safetensors_header, raw_safetensors_read, save_golden  # noqa: E402

MODEL_DIR = Path(os.environ.get("R4DX_MODEL_DIR", r"D:\models\Huihui-gemma-4-12B-it-abliterated"))
TOK_DIR = Path(os.environ.get("R4DX_TOKENIZER_DIR", r"D:\models\Huihui-gemma-4-12B-it-abliterated-tok"))
W_NAME = "model.embed_audio.embedding_projection.weight"
SPT = 640
BOA, AUDIO, EOA = 256000, 258881, 258883

LENGTHS = [1, 639, 640, 641, 1280, 16000, 480000, 480001]  # samples
EMBED_LENGTHS = [1, 641, 7680, 480000]  # samples with a dumped embedding: 1, 2, 12, 750 tokens


def synth(n, seed):
    """Deterministic float32 waveform in [-1, 1]: two tones plus noise, with a non-trivial amplitude."""
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64) / 16000.0
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1234.5 * t) + 0.05 * rng.randn(n)
    return np.clip(x, -1, 1).astype(np.float32)


def frames_plain(x):
    pad = (-len(x)) % SPT
    return np.pad(x, (0, pad)).reshape(-1, SPT).astype(np.float32)


def embed_recipe(frames, w):
    """bf16 cast -> RMSNorm fp32 math, bf16 out -> Linear (fp32 accumulate), bf16 out."""
    xb = torch.from_numpy(frames).to(torch.bfloat16)
    x32 = xb.float()
    ms = x32.pow(2).mean(-1, keepdim=True) + 1e-6
    normed = (x32 * torch.pow(ms, -0.5)).to(torch.bfloat16)
    return (normed.float() @ w.float().T).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "golden_out"))
    ap.add_argument("--skip-real-weight", action="store_true", help="use a seeded random weight (no checkpoint read)")
    ap.add_argument("--skip-processor", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir) / "gemma"
    out.mkdir(parents=True, exist_ok=True)
    findings = {"transformers": __import__("transformers").__version__}

    from transformers.models.gemma4_unified import modeling_gemma4_unified as M
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedAudioConfig, Gemma4UnifiedConfig, Gemma4UnifiedTextConfig)
    from transformers.models.gemma4_unified.feature_extraction_gemma4_unified import (
        Gemma4UnifiedAudioFeatureExtractor)

    # ---- 1. feature extractor ----------------------------------------------------------------------------
    pc = json.loads((TOK_DIR / "processor_config.json").read_text())
    fe_cfg = {k: v for k, v in pc["feature_extractor"].items() if k != "feature_extractor_type"}
    fe = Gemma4UnifiedAudioFeatureExtractor(**fe_cfg)
    rows = []
    for i, n in enumerate(LENGTHS):
        x = synth(n, 100 + i)
        got = fe(x, sampling_rate=16000, return_tensors="np")
        feats, mask = got["input_features"][0], got["input_features_mask"][0]
        want = frames_plain(x)
        ntok = math.ceil(n / SPT)
        assert feats.shape == want.shape == (ntok, SPT), (n, feats.shape, want.shape)
        assert np.array_equal(feats, want), f"extractor differs from pad+reshape at n={n}"
        assert mask.dtype == bool and mask.all() and mask.shape == (ntok,)
        rows.append({"samples": n, "tokens": ntok, "padded_samples": ntok * SPT})
    findings["extractor"] = {"cases": rows, "rule": "right zero-pad to multiple of 640, reshape [n,640], all tokens valid",
                             "sampling_rate": fe.sampling_rate, "padding_side": fe.padding_side,
                             "do_normalize": False, "note": "no amplitude normalisation or clipping: raw float samples"}
    # a batch of two different lengths pads to the longest, mask marks the shorter one's tail (batching is not used by r4dx)
    b = fe([synth(640, 1), synth(1280, 2)], sampling_rate=16000, return_tensors="np")
    assert b["input_features_mask"].tolist() == [[True, False], [True, True]]
    print("[1] extractor == pad+reshape for", LENGTHS)

    # ---- 2. embedder with the real weight ------------------------------------------------------------------
    if args.skip_real_weight:
        g = torch.Generator().manual_seed(7)
        w = (torch.randn(3840, SPT, generator=g) * 0.02).to(torch.bfloat16)
        findings["weight_source"] = "seeded random (--skip-real-weight)"
    else:
        path = MODEL_DIR / "model.safetensors"
        hdr = raw_safetensors_header(path)
        w = raw_safetensors_read(path, W_NAME, hdr)
        findings["weight_source"] = f"{path} :: {W_NAME}"
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (3840, SPT), (w.dtype, w.shape)
    acfg = Gemma4UnifiedAudioConfig(audio_embed_dim=SPT, rms_norm_eps=1e-6)
    tcfg = Gemma4UnifiedTextConfig(hidden_size=3840, num_hidden_layers=1, layer_types=["full_attention"],
                                   num_attention_heads=16, num_key_value_heads=8, vocab_size=262144)
    emb = M.Gemma4UnifiedMultimodalEmbedder(acfg, tcfg).to(torch.bfloat16)
    with torch.no_grad():
        emb.embedding_projection.weight.copy_(w)
    tensors = {"weight": w.contiguous()}
    cases = []
    for i, n in enumerate(EMBED_LENGTHS):
        frames = frames_plain(synth(n, 200 + i))
        with torch.no_grad():
            hf = emb(inputs_embeds=torch.from_numpy(frames)[None])[0]  # float32 frames in, like the model path
        assert hf.dtype == torch.bfloat16
        mine = embed_recipe(frames, w)
        # HF's bf16 matmul blocks/accumulates differently from "fp32 dot then one rounding", so a few outputs sit
        # one bf16 ulp apart: require every element within 1 ulp (2^-7 relative + a tiny absolute floor) and
        # report how many differ. The C++ CPU embedder is held to the same bound against the HF rows.
        d = (hf.float() - mine.float()).abs()
        tol = hf.float().abs() * 2.0 ** -7 + 1e-6
        assert bool((d <= tol).all()), f"embedder recipe differs from HF at n={n}: max {d.max()}"
        n_diff = int((d > 0).sum())
        # control: skipping the bf16 cast of the waveform before the norm changes the result (so the step matters)
        x32 = torch.from_numpy(frames)
        nrm = (x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)).to(torch.bfloat16)
        alt = (nrm.float() @ w.float().T).to(torch.bfloat16)
        cases.append({"samples": n, "tokens": frames.shape[0],
                      "no_precast_differs": bool(not torch.equal(alt, hf)),
                      "elements_not_bit_equal_to_recipe": n_diff, "elements": int(hf.numel()),
                      "rms_of_output": float(hf.float().pow(2).mean().sqrt())})
        tensors[f"frames_{i}"] = torch.from_numpy(frames)
        tensors[f"expect_{i}"] = hf.contiguous()
    findings["embedder"] = {"recipe": "frames f32 -> bf16 -> RMSNorm(640, no weight, eps 1e-6; fp32 math, bf16 result) "
                                      "-> Linear(640->3840, no bias; fp32 accumulate, bf16 result); NOT scaled by sqrt(3840)",
                            "cases": cases}
    print("[2] embedder == explicit recipe (bf16 cast first) for", EMBED_LENGTHS)

    # ---- 3. processor + chat template on the real tokenizer ------------------------------------------------
    if not args.skip_processor:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(TOK_DIR))
        tok = proc.tokenizer
        ids = {"boa": tok.convert_tokens_to_ids("<|audio>"), "audio": tok.convert_tokens_to_ids("<|audio|>"),
               "eoa": tok.convert_tokens_to_ids("<audio|>")}
        assert ids == {"boa": BOA, "audio": AUDIO, "eoa": EOA}, ids
        proc_rows = []
        for n in [1, 640, 641, 16000, 480000]:
            wav = synth(n, 300)
            msgs = [{"role": "user", "content": [{"type": "text", "text": "Transcribe:"},
                                                 {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert text.count("<|audio|>") == 1 and "<|audio>" not in text, text
            enc = proc(text=text, audio=[wav], return_tensors="np")
            idl = enc["input_ids"][0].tolist()
            ntok = math.ceil(n / SPT)
            i0 = idl.index(BOA)
            assert idl[i0:i0 + ntok + 2] == [BOA] + [AUDIO] * ntok + [EOA], "expansion is boa + n*audio + eoa"
            assert idl.count(AUDIO) == ntok
            assert enc["input_features"].shape[-2:] == (ntok, SPT)
            assert proc._compute_audio_num_tokens(wav, 16000) == ntok
            # the same ids from the template text tokenized alone, with the one `<|audio|>` expanded by hand
            raw = tok(text, add_special_tokens=False)["input_ids"]
            a = raw.index(AUDIO)
            manual = raw[:a] + [BOA] + [AUDIO] * ntok + [EOA] + raw[a + 1:]
            assert manual == idl, "template ids + manual expansion != processor ids"
            proc_rows.append({"samples": n, "tokens": ntok, "prompt_len": len(idl), "boa_at": i0,
                              "prefix_ids": idl[:i0], "suffix_ids": idl[i0 + ntok + 2:]})
        findings["processor"] = {"ids": ids, "audio_seq_length": proc.audio_seq_length,
                                 "audio_ms_per_token": proc.audio_ms_per_token, "cases": proc_rows,
                                 "template_text_example": text,
                                 "rule": "template emits one <|audio|> per type==audio part (HF normalizes input_audio -> audio; the r4dx engine must map it itself); processor replaces it with "
                                         "<|audio> + n*<|audio|> + <audio|>, n = ceil(samples/640)",
                                 "cap": "750 is only audio_seq_length (an upper bound the processor never enforces: "
                                        "480001 samples -> 751 tokens expand fine); r4dx enforces it as a 400"}
        n751 = proc(text=text, audio=[synth(480001, 1)], return_tensors="np")["input_ids"][0].tolist().count(AUDIO)
        assert n751 == 751
        print("[3] processor/chat template: ids", ids, "; 751 tokens not capped by HF")

    # ---- 4. causality --------------------------------------------------------------------------------------
    text = Gemma4UnifiedTextConfig(
        vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True,
        sliding_window=4, layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=30.0,
        max_position_embeddings=64, use_bidirectional_attention="vision",
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
    cfg = Gemma4UnifiedConfig(text_config=text, audio_config=Gemma4UnifiedAudioConfig(audio_embed_dim=8),
                              audio_token_id=250, boa_token_id=251, eoa_token_index=252, image_token_id=249)
    cfg._attn_implementation = "eager"
    cfg.get_text_config()._attn_implementation = "eager"
    torch.manual_seed(0)
    model = M.Gemma4UnifiedModel(cfg).eval()
    # tokens: 5 text, boa, 6 audio, eoa, 3 text   (audio frames at positions 6..11)
    ids = torch.tensor([[3, 4, 5, 6, 7, 251] + [250] * 6 + [252, 8, 9, 10]])
    T = ids.shape[1]
    feats = torch.randn(1, 6, 8)
    mm = torch.zeros(1, T, dtype=torch.long)
    mm[0, 6:12] = 3  # audio is type 3: never a vision block
    fmask = torch.ones(1, 6, dtype=torch.bool)

    def run(f):
        with torch.no_grad():
            o = model(input_ids=ids, input_features=f, input_features_mask=fmask, mm_token_type_ids=mm,
                      attention_mask=torch.ones(1, T, dtype=torch.long), output_hidden_states=True)
        return o.last_hidden_state[0]

    base = run(feats)
    res = []
    for k in range(6):
        f2 = feats.clone()
        f2[0, k] += 1.0
        h2 = run(f2)
        pos = 6 + k
        before_same = bool(torch.equal(base[:pos], h2[:pos]))
        at_diff = bool((base[pos] != h2[pos]).any())
        after_diff = bool((base[pos + 1:] != h2[pos + 1:]).any())
        assert before_same and at_diff and after_diff, (k, before_same, at_diff, after_diff)
        res.append({"perturbed_frame": k, "position": pos, "earlier_positions_bit_identical": before_same,
                    "own_and_later_positions_change": at_diff and after_diff})
    full_cfg = cfg
    msk = M.Gemma4UnifiedForConditionalGeneration.create_masks_for_generate(
        full_cfg, torch.zeros(1, T, 32), torch.ones(1, T, dtype=torch.long), None, torch.arange(T)[None],
        mm_token_type_ids=mm)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for name, m in msk.items():
        mm_ = m[0, 0]
        allowed = (mm_ == 0) if mm_.dtype.is_floating_point else mm_
        if name == "sliding_attention":
            idx = torch.arange(T)
            allowed = allowed & True
            ref = causal & ((idx[:, None] - idx[None, :]) < text.sliding_window)
        else:
            ref = causal
        assert torch.equal(allowed, ref), f"{name}: audio tokens are not plain causal"
    findings["causality"] = {"perturb_frame": res, "masks": "generate-path masks with audio tokens (type 3) are the plain "
                             "causal (+window) masks for sliding and full layers"}
    print("[4] audio is causal: perturbing frame k leaves earlier positions bit-identical (6 frames)")

    # ---- write ---------------------------------------------------------------------------------------------
    save_golden(out / "audio_golden.safetensors", tensors,
                {"cases": [{"samples": c["samples"], "tokens": c["tokens"]} for c in cases], "spt": SPT})
    (out / "audio_semantics.json").write_text(json.dumps(findings, indent=1))
    print("wrote", out / "audio_golden.safetensors", "and", out / "audio_semantics.json")


if __name__ == "__main__":
    main()
