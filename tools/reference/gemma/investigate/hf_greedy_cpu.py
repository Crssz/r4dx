"""CPU-only HF greedy generation for a single chat prompt, to compare against r4dx-server output.

  D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\hf_greedy_cpu.py \\
      --prompt "Is 91 prime? Answer briefly." --think --max-new 400
"""
import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

assert not torch.cuda.is_available(), "this script must not see a GPU"

MODELS_ROOT = os.environ.get("R4DX_MODELS_ROOT", r"E:\models")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(MODELS_ROOT, "Huihui-gemma-4-12B-it-abliterated"))
ap.add_argument("--tok", default=os.path.join(MODELS_ROOT, "Huihui-gemma-4-12B-it-abliterated-tok"))
ap.add_argument("--prompt", required=True)
ap.add_argument("--think", action="store_true")
ap.add_argument("--max-new", type=int, default=400)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tok)
ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True,
                              enable_thinking=args.think, return_tensors="pt", return_dict=True)["input_ids"]
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False, eos_token_id=[1, 106, 50])
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))
