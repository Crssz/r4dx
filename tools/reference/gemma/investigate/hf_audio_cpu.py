"""CPU-only HF greedy answer to one audio + text prompt, to compare against r4dx-server's audio path.

  D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\hf_audio_cpu.py \\
      --wav <models root>\\r4dx\\huihui-gemma\\dry\\smoke\\tone.wav --text "Describe what you hear in this audio clip in one sentence."
"""
import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

assert not torch.cuda.is_available(), "this script must not see a GPU"

MODELS_ROOT = os.environ.get("R4DX_MODELS_ROOT", r"E:\models")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(MODELS_ROOT, "Huihui-gemma-4-12B-it-abliterated"))
ap.add_argument("--proc", default=os.path.join(MODELS_ROOT, "Huihui-gemma-4-12B-it-abliterated-tok"))
ap.add_argument("--wav", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--max-new", type=int, default=60)
args = ap.parse_args()

with wave.open(args.wav, "rb") as wf:
    assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

proc = AutoProcessor.from_pretrained(args.proc)
# Text first, then the clip: the same part order the r4dx-server smoke request used.
msgs = [{"role": "user", "content": [{"type": "text", "text": args.text}, {"type": "audio", "audio": audio}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print("prompt tokens:", inputs["input_ids"].shape[1], "keys:", sorted(inputs.keys()))
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False, eos_token_id=[1, 106, 50])
print("HF:", proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
