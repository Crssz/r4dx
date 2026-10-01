"""CPU-only check of kl_corpus/tokens_gemma.json and tokens_gemma_long.json (M0-7): every segment
is `[BOS=2] + AutoTokenizer ids of its corpus file`, truncated to max_tokens; the long file has
>= 1536-token segments (so the 1024 sliding ring wraps); neither shares a segment with the Qwen
tokens.json. Regenerate with the commands in tools/reference/make_tokens_json.py's docstring.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\gemma\\test_tokens_gemma.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pytest  # noqa: E402

from gemma import common_gemma as cg  # noqa: E402
from make_tokens_json import token_ids_sha256  # noqa: E402

CORPUS = cg.KL_CORPUS_DIR
FILES = {"tokens_gemma.json": {"cpp_source": "cpp_source.txt", "english_prose": "english_prose.txt",
                               "python_source": "python_source.txt", "thai_prose": "thai_prose.txt"},
         "tokens_gemma_long.json": {"python_source_long": "python_source.txt",
                                    "cpp_source_long": "cpp_source.txt"}}

pytestmark = pytest.mark.skipif(not (cg.DEFAULT_TOKENIZER_DIR / "tokenizer.json").is_file(),
                                reason="assembled Gemma tokenizer dir not on this machine")


@pytest.fixture(scope="module")
def autotok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(cg.DEFAULT_TOKENIZER_DIR))


@pytest.mark.parametrize("fname", sorted(FILES))
def test_ids_match_autotokenizer_with_bos(fname, autotok):
    doc = json.loads((CORPUS / fname).read_text(encoding="utf-8"))
    assert doc["tokenizer_arch"] == "gemma4" and doc["add_bos"] is True and doc["bos_token_id"] == 2
    assert sorted(s["name"] for s in doc["segments"]) == sorted(FILES[fname])
    for seg in doc["segments"]:
        text = (CORPUS / FILES[fname][seg["name"]]).read_text(encoding="utf-8")
        want = [autotok.bos_token_id] + list(autotok(text, add_special_tokens=False)["input_ids"])
        got = seg["token_ids"]
        assert got[0] == 2 and got[1] != 2, "exactly one BOS"
        assert got == want[: len(got)], seg["name"]
        assert len(got) == doc["max_tokens"], (seg["name"], len(got))
        assert max(got) < 262144


def test_long_segments_wrap_the_ring():
    doc = json.loads((CORPUS / "tokens_gemma_long.json").read_text(encoding="utf-8"))
    assert all(len(s["token_ids"]) >= 1536 for s in doc["segments"])


def test_disjoint_from_qwen_ids_and_stable_hashes():
    qwen = json.loads((CORPUS / "tokens.json").read_text(encoding="utf-8"))
    gem = json.loads((CORPUS / "tokens_gemma.json").read_text(encoding="utf-8"))
    q = {token_ids_sha256(s["token_ids"]) for s in qwen["segments"]}
    g = {token_ids_sha256(s["token_ids"]) for s in gem["segments"]}
    assert not (q & g)
    assert gem["tokenizer_provenance"]["tokenizer_json_sha256"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
