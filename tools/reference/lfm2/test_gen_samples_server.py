"""CPU unit test of tools/reference/lfm2/gen_samples_server.py against a fake OpenAI server (python http.server).

Needs only the LFM tokenizer files (tokenizer.json, chat_template.jinja; E:\\models\\LiquidAI\\LFM2.5-24B-A2B or
R4DX_MODEL_DIR) -- no weights, no GPU. Skipped when they are absent.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I -m unittest discover -s tools\\reference\\lfm2 -p test_gen_samples_server.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REF / "lfm2"))

import gen_samples_server as gs  # noqa: E402
from lfm2.common_lfm2 import DEFAULT_MODEL_DIR, IM_END_ID, Lfm2RefTokenizer  # noqa: E402

gc = gs.gc
HAVE_TOK = (DEFAULT_MODEL_DIR / "tokenizer.json").is_file() and (DEFAULT_MODEL_DIR / "chat_template.jinja").is_file()
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = Lfm2RefTokenizer(DEFAULT_MODEL_DIR)
    return _TOK


EN_KL = (REF / "kl_corpus" / "english_prose.txt").read_text(encoding="utf-8")

PROMPTS = {
    "format": gc.PROMPTS_FORMAT, "version": gc.PROMPTS_VERSION,
    "samples": [
        {"id": "english_prose/001", "category": "english_prose", "format": "raw", "enable_thinking": False,
         "max_tokens": 1000, "messages": [{"role": "user", "content": "Write two paragraphs about tides."}]},
        {"id": "chat/001", "category": "chat", "format": "chat", "enable_thinking": True, "max_tokens": 1200,
         "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why is the sky blue?"}],
         "followups": ["And at sunset?"]},
        {"id": "code/001", "category": "code", "format": "raw", "enable_thinking": False, "max_tokens": 1000,
         "messages": [{"role": "user", "content": "Write a function that sorts a list."}]},
    ]}


class Fake:
    """Behaviour knobs + a log of the request bodies."""

    def __init__(self):
        self.bodies = []
        self.prompt_delta = 0
        self.fail_first_post = False
        self.kl_for = None  # sample id (first message text) whose answer copies KL text
        self.posts = 0


def make_server(fake: Fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(200, {"data": [{"id": "fake-lfm2"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.posts += 1
            if fake.fail_first_post and fake.posts == 1:
                self._send(503, {"error": {"message": "busy"}})
                return
            fake.bodies.append(body)
            msgs = body["messages"]
            n_prompt = len(tok().encode_chat(msgs, add_generation_prompt=True)) + fake.prompt_delta
            first = msgs[-1]["content"]
            if "sorts a list" in msgs[0]["content"]:  # truncated thought: no close tag, length
                choice = {"message": {"role": "assistant", "content": "", "reasoning_content": "I should think a lot "
                                      "about sorting algorithms and their stability properties, then more."},
                          "finish_reason": "length"}
            elif fake.kl_for and fake.kl_for in first:
                copied = " ".join(EN_KL.split())[200:400]
                choice = {"message": {"role": "assistant", "content": copied, "reasoning_content": "Hmm."},
                          "finish_reason": "stop"}
            else:
                n = sum(1 for m in msgs if m["role"] == "assistant") + 1
                choice = {"message": {"role": "assistant", "content": f"Answer number {n} to: {first[:30]}. " * 3,
                                      "reasoning_content": f"Let me think about turn {n}. The user wants things."},
                          "finish_reason": "stop"}
            n_comp = 40
            self._send(200, {"model": "fake-lfm2", "choices": [{"index": 0, **choice}],
                             "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_comp}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@unittest.skipUnless(HAVE_TOK, "LFM tokenizer files not found")
class GenSamplesServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.prompts = self.dir / "prompts.json"
        self.prompts.write_text(json.dumps(PROMPTS), encoding="utf-8")
        self.fake = Fake()
        self.srv = make_server(self.fake)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.tmp.cleanup()

    def run_gen(self, *extra, out=None):
        out = out or self.dir / "samples.jsonl"
        argv = ["--server", self.url, "--prompts", str(self.prompts), "--out", str(out), "--retry-wait", "0.01",
                *extra]
        return gs.main(argv, tok=tok()), out

    def records(self, out):
        return {r["id"]: r for r in (json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x)}

    def test_stream_ids_and_requests(self):
        code, out = self.run_gen()
        self.assertEqual(code, 0)
        recs = self.records(out)
        self.assertEqual(sorted(recs), ["chat/001", "code/001", "english_prose/001"])
        t = tok()
        # request bodies: sampling, seed from id+turn, no chat_template_kwargs, reasoning replayed on turn 2
        sent = {(b["messages"][0]["content"], len(b["messages"])): b for b in self.fake.bodies}
        for b in self.fake.bodies:
            self.assertEqual((b["temperature"], b["top_p"], b["top_k"], b["repetition_penalty"]),
                             (0.7, 0.95, 50, 1.05))
            self.assertNotIn("chat_template_kwargs", b)
        r = recs["english_prose/001"]
        self.assertEqual(r["turn_seeds"], [gs.sample_seed("english_prose/001", 0)])
        self.assertEqual(r["max_tokens"], 1000)
        self.assertFalse(r["rejected"], r["reject_reason"])
        # token stream == the full template render of the conversation minus the trailing newline
        full = t.render_chat([{"role": m["role"], "content": m["content"],
                               **({"reasoning_content": m["reasoning_content"]} if m["role"] == "assistant" else {})}
                              for m in r["messages"]])
        self.assertEqual(r["token_ids"], t.encode(full[:-1]))
        self.assertEqual(r["token_ids"][0], t.bos_id)
        self.assertEqual(r["token_ids"][-1], IM_END_ID)
        self.assertIn(124901, r["token_ids"])  # <think>
        self.assertIn(124902, r["token_ids"])  # </think>
        # two-turn chat: stream is the FINAL turn's prompt (turn-1 thought stripped) + its completion
        c = recs["chat/001"]
        self.assertEqual(len(c["turns"]), 2)
        self.assertEqual(c["turn_seeds"], [gs.sample_seed("chat/001", 0), gs.sample_seed("chat/001", 1)])
        self.assertEqual([m["role"] for m in c["messages"]], ["system", "user", "assistant", "user", "assistant"])
        full = t.render_chat([{"role": m["role"], "content": m["content"],
                               **({"reasoning_content": m["reasoning_content"]} if m["role"] == "assistant" else {})}
                              for m in c["messages"]])
        self.assertEqual(c["token_ids"], t.encode(full[:-1]))
        self.assertNotIn("Let me think about turn 1", t.decode(c["token_ids"]))
        self.assertIn("Let me think about turn 2", t.decode(c["token_ids"]))
        self.assertEqual(c["turns"][1]["prompt_len"], c["turns"][1]["prompt_tokens_server"])
        # truncated thought: kept, no </think>, no <|im_end|>
        z = recs["code/001"]
        self.assertFalse(z["rejected"], z["reject_reason"])
        self.assertEqual(z["truncated_thought_turn"], 1)
        self.assertNotIn(124902, z["token_ids"])
        self.assertNotEqual(z["token_ids"][-1], IM_END_ID)
        self.assertEqual(sent and len(sent) > 0, True)
        # manifest + capture contract
        man = json.loads((self.dir / "gen_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["summary"]["samples"], 3)
        self.assertEqual(man["sampling"]["temperature"], 0.7)
        from gemma.capture import load_token_samples

        usable, rejected = load_token_samples(out)
        self.assertEqual((len(usable), len(rejected)), (3, 0))

    def test_resume_and_changes(self):
        code, out = self.run_gen("--limit", "2")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.records(out)), 2)
        n = len(self.fake.bodies)
        code, _ = self.run_gen(out=out)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.records(out)), 3)
        self.assertGreaterEqual(len(self.fake.bodies) - n, 1)
        before = len(self.fake.bodies)
        code, _ = self.run_gen(out=out)  # complete: nothing to do
        self.assertEqual((code, len(self.fake.bodies)), (0, before))
        # a torn last line is truncated and regenerated
        raw = out.read_bytes()
        out.write_bytes(raw[:-40])
        code, _ = self.run_gen(out=out)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.records(out)), 3)
        # a changed prompt stops the run
        doc = json.loads(self.prompts.read_text(encoding="utf-8"))
        doc["samples"][0]["messages"][0]["content"] += " Be vivid."
        self.prompts.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_gen(out=out)

    def test_prompt_mismatch_aborts_first_request(self):
        self.fake.prompt_delta = 3
        code, out = self.run_gen()
        self.assertEqual(code, 3)
        self.assertFalse(out.is_file() and out.read_text(encoding="utf-8").strip())
        code, out = self.run_gen("--no-verify-prompt", out=self.dir / "b.jsonl")
        self.assertEqual(code, 0)
        recs = self.records(out)
        self.assertTrue(all(r["rejected"] and r["reject_reason"].startswith("prompt id mismatch")
                            for r in recs.values()))

    def test_passes_and_retry(self):
        self.fake.fail_first_post = True
        code, out = self.run_gen("--passes", "2")
        self.assertEqual(code, 0)
        recs = self.records(out)
        self.assertEqual(len(recs), 6)
        self.assertIn("english_prose/001#1", recs)
        self.assertNotEqual(recs["english_prose/001"]["seed"], recs["english_prose/001#1"]["seed"])
        self.assertEqual(recs["english_prose/001#1"]["base_id"], "english_prose/001")
        self.assertEqual(recs["english_prose/001#1"]["prompt_sha256"], recs["english_prose/001"]["prompt_sha256"])
        # merge: split into two files, merge back, require-complete
        a, b = self.dir / "a.jsonl", self.dir / "b.jsonl"
        lines = out.read_text(encoding="utf-8").splitlines()
        a.write_text("\n".join(x for x in lines if '"pass": 0' in x) + "\n", encoding="utf-8")
        b.write_text("\n".join(x for x in lines if '"pass": 1' in x) + "\n", encoding="utf-8")
        m = self.dir / "m.jsonl"
        argv = ["--merge", str(a), str(b), "--out", str(m), "--prompts", str(self.prompts), "--passes", "2",
                "--require-complete"]
        self.assertEqual(gs.main(argv, tok=tok()), 0)
        self.assertEqual(len(self.records(m)), 6)
        with self.assertRaises(SystemExit):
            gs.main(["--merge", str(a), "--out", str(self.dir / "n.jsonl"), "--prompts", str(self.prompts),
                     "--passes", "2", "--require-complete"], tok=tok())

    def test_kl_overlap_rejected(self):
        self.fake.kl_for = "two paragraphs about tides"
        code, out = self.run_gen()
        self.assertEqual(code, 0)
        r = self.records(out)["english_prose/001"]
        self.assertTrue(r["rejected"])
        self.assertTrue(r["reject_reason"].startswith("kl overlap"), r["reject_reason"])

    def test_dry_run_selfcheck(self):
        code = gs.main(["--dry-run", "--passes", "3"], tok=tok())
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
