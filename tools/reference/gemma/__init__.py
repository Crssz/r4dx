"""Gemma 4 (gemma4_unified) reference tooling for r4dx (docs/gemma4-plan.md section 6.3).

One package shared by the golden, KL, quantization and validation scripts:

  common_gemma  paths, the device rule (R4DX_REF_ALLOWED_DEVICES), the tokenizer, shared helpers
  arch          the architecture table (layer types, per-layer geometry, checkpoint tensor names)
  ref           GemmaReference (resident or streaming bf16 text stack), tiny random-config builder

The Qwen tools next to this directory stay frozen; the pieces of tools/reference/common.py that are
model-agnostic (ShardIndex, save_golden, sha256_file, ...) are re-exported by common_gemma.

Import it from a script with
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from gemma import common_gemma
(tools/reference must be on sys.path so that `gemma` and `common` both resolve).
"""
