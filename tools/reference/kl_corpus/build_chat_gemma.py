"""tools/reference/kl_corpus/build_chat_gemma.py

Builds `chat_gemma.json`, the PRIMARY corpus of the Gemma M1 KL gate (docs/gemma4-plan.md 9.9-9.10):
ten single-turn conversations rendered with the Gemma chat template of the tokenizer dir (8 thinking
off, 2 thinking on), each a user turn followed by a natural model answer. Scoring is on the model-turn
tokens only (`score_mask`), so the template scaffolding and the user turn do not dilute the KL.

Provenance: every user prompt and every model answer below was written for this corpus by the
repo author's assistant (original text, no third-party material; released with the repo). They are
disjoint from the calibration prompts (tools/quant2/corpus_v2_prompts.json) and from the raw KL
segments (kl_corpus/*.txt); `check_disjoint` verifies it with a word 6-gram test.

Segment layout (token ids; BOS appears exactly once, at index 0, as the template's own `<bos>`):

    <bos><|turn>user\\n{user}<turn|>\\n<|turn>model\\n[<|channel>thought\\n<channel|>]   <- prompt (mask 0)
    {answer}<turn|>                                                                   <- mask 1
      (thinking on: the model turn starts right after `<|turn>model\\n`: the scored text is
       `<|channel>thought\\n{thought}<channel|>{answer}<turn|>`, thought included)

Verification: the prompt part of `token_ids` equals `AutoTokenizer.apply_chat_template(...,
add_generation_prompt=True, tokenize=True)` for every segment.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\kl_corpus\\build_chat_gemma.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SAMPLES: list[dict] = []


def add(name, group, thinking, user, answer, thought=None):
    SAMPLES.append({"name": name, "group": group, "thinking": thinking, "user": user,
                    "answer": answer.strip("\n"), "thought": thought})


add("en_bread", "english", False,
    "Why does bread dough need to rest before baking?",
    """Resting does a few different jobs, depending on which rest you mean.

The first rest, right after mixing, lets the flour finish absorbing water. Dry flour takes a while to hydrate fully, and while it does, enzymes in the flour start breaking some of the starch into sugars and the gluten proteins link up into a stretchy network. That is why a dough that felt shaggy and sticky right after mixing is smooth and elastic half an hour later, even though you hardly touched it. It also means you can knead less.

The long rise is when the yeast does its work. It eats the sugars and releases carbon dioxide, which the gluten network traps in thousands of tiny bubbles, and it also produces alcohols and acids that give bread most of its flavor. A slow, cool rise, for example overnight in the fridge, gives more flavor than a fast, warm one.

The last rest, after you shape the loaf, is about relaxing the gluten. Shaping tightens the dough, and if you bake it right away it tends to tear or spring unevenly. Fifteen to sixty minutes lets it loosen up and puff again so it can expand in the oven.

A simple test for when it is ready: press a floured finger about a centimeter into the dough. If the dent springs back slowly and only part of the way, it is ready to bake. If it springs back immediately, give it more time. If it does not spring back at all, bake it soon.""")

add("en_email", "english", False,
    "Write a short, polite email declining a meeting invitation because my week is overloaded, but offering an alternative.",
    """Subject: Re: Planning sync on Thursday

Hi Maria,

Thanks for the invitation. I would like to be part of the planning discussion, but this week is full: I have the release review on Wednesday and two deadlines on Friday, and I do not think I could give the meeting the attention it deserves.

Would it work if I sent you my notes on the roadmap items beforehand, so you can raise them on my behalf? I am also free next Tuesday or Wednesday afternoon if you would rather have a short follow-up with just the two of us once the dust has settled.

If it helps, I can also take the action items for the integration work, since that is mostly on my side anyway. Just let me know what the team would find most useful and I will make it happen after Friday.

Thanks for understanding, and sorry for the short notice.

Best,
Daniel""")

add("en_tides", "english", False,
    "Can you explain what causes ocean tides in plain language?",
    """Tides come mostly from the Moon's gravity, with the Sun adding a smaller contribution.

The Moon pulls on everything on Earth, but not equally. The side of the planet facing the Moon is closer to it and feels a slightly stronger pull than the center of the Earth does, while the far side feels a slightly weaker one. The difference between these pulls stretches the oceans into a shape with two bulges, one facing the Moon and one on the opposite side. As the Earth turns beneath these bulges, a given coastline passes through both of them, which is why most places get two high tides and two low tides in a little over a day. The cycle takes about 24 hours and 50 minutes rather than 24 because the Moon is also moving along its orbit.

The Sun does the same thing, but it is so much farther away that its effect is less than half of the Moon's. When the Sun and Moon line up, at new moon and full moon, their pulls add together and we get spring tides, which are unusually high and low. When they are at right angles, at the quarter moons, they partly cancel and we get the smaller neap tides.

The local picture is messier. The shape of the coastline, the depth of the water and the size of the basin all change how the water sloshes around, which is why the Bay of Fundy sees tides of more than ten meters while some places barely notice them at all.""")

add("th_rain", "thai", False,
    "ช่วยอธิบายหน่อยว่าฝนเกิดขึ้นได้อย่างไร",
    """ฝนเกิดจากวัฏจักรของน้ำที่หมุนเวียนอยู่ในธรรมชาติตลอดเวลา เริ่มจากความร้อนของดวงอาทิตย์ทำให้น้ำในทะเล แม่น้ำ และพื้นดินระเหยกลายเป็นไอน้ำลอยขึ้นสู่ท้องฟ้า นอกจากนี้ต้นไม้ก็คายน้ำออกมาเป็นไอด้วยเช่นกัน

เมื่ออากาศที่มีไอน้ำลอยสูงขึ้น อุณหภูมิจะค่อย ๆ ลดลง ไอน้ำจึงเริ่มควบแน่นเป็นหยดน้ำเล็ก ๆ เกาะอยู่บนอนุภาคฝุ่นหรือเกลือขนาดเล็กในอากาศ หยดน้ำจำนวนมหาศาลรวมตัวกันเป็นก้อนเมฆที่เรามองเห็น

หยดน้ำในเมฆเล็กมากจนลอยอยู่ได้ แต่เมื่อมันชนและรวมกับหยดอื่นไปเรื่อย ๆ จนมีขนาดใหญ่และหนักพอ แรงโน้มถ่วงก็จะดึงให้ตกลงมาเป็นเม็ดฝน ถ้าชั้นบนของเมฆเย็นมากก็อาจเกิดเป็นเกล็ดน้ำแข็งก่อน แล้วละลายกลายเป็นฝนระหว่างตกลงมา

สำหรับประเทศไทย ฝนส่วนใหญ่มาจากลมมรสุมตะวันตกเฉียงใต้ที่พัดพาความชื้นจากมหาสมุทรอินเดียเข้ามาในช่วงเดือนพฤษภาคมถึงตุลาคม ส่วนฝนที่ตกหนักในช่วงบ่ายหน้าร้อนมักเกิดจากอากาศร้อนใกล้พื้นดินลอยตัวขึ้นอย่างรวดเร็วจนเกิดเมฆฝนฟ้าคะนอง

หลังจากนั้นน้ำฝนจะไหลลงแม่น้ำ ซึมลงดิน และกลับสู่ทะเลอีกครั้ง วัฏจักรจึงเริ่มต้นใหม่อย่างไม่มีวันสิ้นสุด""")

add("th_plants", "thai", False,
    "ฉันเพิ่งเริ่มปลูกต้นไม้ในคอนโด มีคำแนะนำสำหรับมือใหม่ไหม",
    """ยินดีด้วยที่เริ่มปลูกต้นไม้นะคะ การปลูกต้นไม้ในคอนโดทำได้ไม่ยาก ถ้าเข้าใจหลักง่าย ๆ ไม่กี่ข้อ

1. เลือกต้นไม้ให้เหมาะกับแสง ก่อนซื้อให้สังเกตก่อนว่าระเบียงหรือห้องของคุณได้รับแสงแดดกี่ชั่วโมงต่อวัน ถ้าแสงน้อยให้เลือกพวกพลูด่าง ซานเซเวียเรีย หรือเศรษฐีเรือนใน ถ้าได้แดดมากให้เลือกกระบองเพชร ชบาแคระ หรือสมุนไพรอย่างโหระพาและกะเพรา

2. อย่ารดน้ำมากเกินไป ต้นไม้ในบ้านส่วนใหญ่ตายเพราะน้ำมากกว่าขาดน้ำ ให้ลองใช้นิ้วจิ้มลงไปในดินลึกประมาณหนึ่งข้อนิ้ว ถ้าดินยังชื้นอยู่ก็ยังไม่ต้องรด ถ้าแห้งแล้วค่อยรดให้ทั่วจนน้ำไหลออกทางรูก้นกระถาง

3. ใช้กระถางที่มีรูระบายน้ำ และไม่ควรวางจานรองที่มีน้ำขังไว้นาน เพราะรากจะเน่าได้ง่าย

4. ใช้ดินปลูกที่ร่วนซุย ผสมเพอร์ไลต์หรือเปลือกไม้สับเล็กน้อยเพื่อให้ระบายน้ำได้ดี

5. ให้ปุ๋ยแต่น้อย เดือนละครั้งก็เพียงพอ และให้เฉพาะช่วงที่ต้นไม้กำลังเติบโต

เริ่มจากต้นไม้ที่ทนง่าย ๆ สัก 2-3 ต้นก่อน แล้วลองสังเกตว่าใบเปลี่ยนไปอย่างไร ใบเหลืองมักแปลว่าน้ำมากเกินไป ใบเหี่ยวและกรอบมักแปลว่าขาดน้ำหรือแดดแรงเกินไป เมื่อเริ่มจับทางได้แล้วค่อยขยับไปลองต้นที่ดูแลยากขึ้น""")

add("py_intervals", "python", False,
    "Write a Python function that merges overlapping intervals, e.g. [[1,3],[2,6],[8,10]] -> [[1,6],[8,10]].",
    """The usual approach is to sort the intervals by start, then sweep through them once, extending the last merged interval whenever the next one overlaps it.

```python
def merge_intervals(intervals):
    \"\"\"Merge overlapping or touching [start, end] intervals.\"\"\"
    merged = []
    for start, end in sorted(intervals, key=lambda iv: iv[0]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


print(merge_intervals([[1, 3], [2, 6], [8, 10]]))   # [[1, 6], [8, 10]]
print(merge_intervals([[1, 4], [4, 5]]))            # [[1, 5]]
print(merge_intervals([]))                          # []
```

Sorting guarantees that any interval that can overlap the current one comes right after it, so a single pass is enough. The `max` matters: an interval such as `[1, 10]` followed by `[2, 3]` must not shrink the merged end to 3.

Intervals that merely touch, like `[1, 4]` and `[4, 5]`, are merged here because of the `<=`. If you want them kept separate, change it to `<`.

The cost is O(n log n) for the sort and O(n) for the sweep, with O(n) extra space for the result. The inner lists are created fresh, so the input is not modified.""")

add("cpp_queue", "cpp", False,
    "Show me a minimal thread-safe blocking queue in C++17.",
    """Here is a small one built on `std::mutex` and `std::condition_variable`. It supports multiple producers and consumers, and a `close()` call so consumers can shut down cleanly.

```cpp
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template <typename T>
class BlockingQueue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return;
            items_.push(std::move(value));
        }
        ready_.notify_one();
    }

    // Blocks until an item is available. Returns std::nullopt once the
    // queue is closed and drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) return std::nullopt;
        T value = std::move(items_.front());
        items_.pop();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<T> items_;
    bool closed_ = false;
};
```

A consumer loops with `while (auto item = queue.pop()) { ... }` and exits when `pop()` returns `nullopt`. If you need back-pressure, add a capacity and a second condition variable for producers.""")

add("py_decorator", "python", False,
    "What is a Python decorator? Give me a practical example.",
    """A decorator is a function that takes another function and returns a replacement for it, usually one that wraps the original with some extra behavior. The `@name` syntax above a definition is just shorthand: this

```python
@timed
def work(): ...
```

means exactly `work = timed(work)`.

A practical example is a decorator that measures how long a function takes:

```python
import functools
import time


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.3f}s")
    return wrapper


@timed
def load_data(n):
    return sum(i * i for i in range(n))


load_data(2_000_000)   # prints: load_data took 0.12s
```

Three details are worth remembering. `*args, **kwargs` lets the wrapper accept whatever the original does. `functools.wraps` copies the original name and docstring onto the wrapper, so debugging and help text still make sense. And the `try/finally` makes sure the timing is printed even if the function raises.

Decorators that take arguments, such as `@retry(times=3)`, add one more layer: a function that returns the decorator.""")

add("en_prime_think", "english", True,
    "Is 221 a prime number? Explain briefly.",
    """No, 221 is not prime. It equals 13 times 17.

To check, you only need to test primes up to the square root of 221, which is a little under 15. It is odd, so 2 is out. Its digits add up to 5, so 3 is out, and it does not end in 0 or 5, so 5 is out. Dividing by 7 gives 31 with a remainder of 4, and by 11 gives 20 with a remainder of 1. Then 13 times 17 is 221 exactly, so it factors.

There is also a neat shortcut for numbers that sit just below a perfect square. Since 221 is 225 minus 4, it is 15 squared minus 2 squared, and the difference of two squares factors as (15 minus 2) times (15 plus 2), which is 13 times 17.""",
    thought="""The user wants to know whether 221 is prime. The square root of 221 is about 14.9, so I only need to test the primes 2, 3, 5, 7, 11 and 13. It is odd. The digit sum is 2+2+1=5, not divisible by 3. It does not end in 0 or 5. For 7: 7 times 31 is 217, remainder 4. For 11: 11 times 20 is 220, remainder 1. For 13: 13 times 17 is 221, since 13 times 17 is 130 plus 91. So it is composite. I will give the factorization first and then the short check.""")

add("th_budget_think", "thai", True,
    "ผมมีเงิน 12,000 บาท อยากเก็บเป็นเงินสำรองฉุกเฉินครึ่งหนึ่ง ส่วนที่เหลือแบ่งเป็นเงินเที่ยวกับเงินลงทุนในอัตรา 1 ต่อ 2 จะได้อย่างละเท่าไหร่",
    """ได้ดังนี้ครับ

- เงินสำรองฉุกเฉิน: 6,000 บาท (ครึ่งหนึ่งของ 12,000)
- เงินเที่ยว: 2,000 บาท
- เงินลงทุน: 4,000 บาท

วิธีคิดคือ หลังเก็บเงินสำรองแล้วจะเหลือ 6,000 บาท เมื่อแบ่งในอัตรา 1 ต่อ 2 จะมีทั้งหมด 3 ส่วน ส่วนละ 2,000 บาท เงินเที่ยวได้ 1 ส่วนคือ 2,000 บาท และเงินลงทุนได้ 2 ส่วนคือ 4,000 บาท

รวมกันแล้ว 6,000 + 2,000 + 4,000 = 12,000 บาทพอดีครับ""",
    thought="""12,000 baht. Half goes to the emergency fund, so 6,000. The remaining 6,000 is split 1 to 2 between travel and investing. That is 3 parts of 2,000 each, so travel is 2,000 and investing is 4,000. Check: 6,000 + 2,000 + 4,000 = 12,000. The user writes in Thai, so I will answer in Thai with a short list and the working.""")


# --------------------------------------------------------------------------------------------


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def ngrams(text: str, n: int = 6) -> set[tuple]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def check_disjoint() -> list[str]:
    """Word 6-gram overlap of every sample (user + thought + answer) with the calibration prompts and the
    raw KL text files. Thai has no spaces so `\\w+` yields long runs: for Thai fall back to 24-char shingles."""
    problems = []
    other = []
    pj = HERE.parents[1] / "quant2" / "corpus_v2_prompts.json"
    doc = json.loads(pj.read_text(encoding="utf-8"))
    blob = json.dumps(doc["samples"], ensure_ascii=False)
    other.append(("corpus_v2_prompts.json", blob))
    for p in sorted(HERE.glob("*.txt")):
        other.append((p.name, p.read_text(encoding="utf-8")))
    for s in SAMPLES:
        text = " ".join([s["user"], s["thought"] or "", s["answer"]])
        for oname, otext in other:
            shared = ngrams(text) & ngrams(otext)
            if shared:
                problems.append(f"{s['name']} shares {len(shared)} word 6-grams with {oname}: {sorted(shared)[:2]}")
            tt = re.sub(r"\s+", "", text)
            ot = re.sub(r"\s+", "", otext)
            shingles = {tt[i:i + 24] for i in range(0, len(tt) - 23)}
            hits = [sh for sh in shingles if sh in ot]
            if hits and any("\u0e00" <= ch <= "\u0e7f" for ch in tt):
                problems.append(f"{s['name']} shares {len(hits)} 24-char shingles with {oname}: {hits[:1]}")
    return problems


def main() -> int:
    from gemma.common_gemma import BOS_ID, DEFAULT_TOKENIZER_DIR, load_tokenizer

    tok = load_tokenizer(None)
    hf = tok.hf
    problems = check_disjoint()
    if problems:
        print("\n".join(problems))
        raise SystemExit("corpus is not disjoint from the calibration/raw corpora")
    segs = []
    for s in SAMPLES:
        msgs = [{"role": "user", "content": s["user"]}]
        think = s["thinking"]
        prompt_text = tok.render_chat(msgs, add_generation_prompt=True, enable_thinking=think)
        ref_ids = hf.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think, tokenize=True)
        if not isinstance(ref_ids, list):
            ref_ids = list(ref_ids["input_ids"])
        prompt_ids = tok.encode(prompt_text)
        assert prompt_ids == list(ref_ids), f"{s['name']}: prompt ids differ from apply_chat_template"
        assert prompt_ids[0] == BOS_ID and prompt_ids.count(BOS_ID) == 1
        if think:
            reply = "<|channel>thought\n" + s["thought"] + "<channel|>" + s["answer"] + "<turn|>"
        else:
            reply = s["answer"] + "<turn|>"
        reply_ids = tok.encode(reply)
        joint = tok.encode(prompt_text + reply)
        ids = prompt_ids + reply_ids
        assert tok.raw.encode(prompt_text + reply, add_special_tokens=False).ids == joint, "HF != tokenizers.json"
        assert ids.count(BOS_ID) == 1
        assert ids[-1] == 106, f"{s['name']}: last id {ids[-1]} is not <turn|>"
        mask = [0] * len(prompt_ids) + [1] * len(reply_ids)
        # answer tokens only (excluding thought span) for reporting
        n_ans = len(tok.encode(s["answer"] + "<turn|>"))
        segs.append({"name": s["name"], "group": s["group"], "thinking": think, "token_ids": ids,
                     "score_mask": mask, "prompt_len": len(prompt_ids), "answer_tokens": n_ans,
                     "joint_tokenization_equals_concat": joint == ids,
                     "user": s["user"], "text": prompt_text + reply})
        print(f"{s['name']:14s} group={s['group']:8s} think={think!s:5s} prompt={len(prompt_ids):3d} "
              f"scored={len(reply_ids):3d} (answer {n_ans}) total={len(ids)} joint==concat {joint == ids}")
        if not 150 <= n_ans <= 400:
            print(f"  WARNING answer length {n_ans} outside 150-400")
    doc = {
        "format": "r4dx-kl-chat-v1",
        "tokenizer": tok.describe(), "tokenizer_arch": "gemma4", "tokenizer_mode": "canonical",
        "tokenizer_provenance": tok.provenance(), "add_bos": True, "bos_token_id": BOS_ID,
        "chat_template": True, "score_mask_semantics":
            "score_mask[i]=1 for ids that are model-turn tokens (thinking on: thought span + answer; thinking off: "
            "answer; both incl. the closing <turn|>). Row i of a logprob file predicts token i+1, so row i is "
            "scored when score_mask[i+1]=1.",
        "provenance": "All user prompts, thoughts and answers were written for this corpus (original text; no "
                      "third-party material). Disjoint from tools/quant2/corpus_v2_prompts.json and kl_corpus/*.txt "
                      "(word 6-gram / Thai 24-char shingle check in build_chat_gemma.py).",
        "segments": segs,
    }
    out = HERE / "chat_gemma.json"
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({sum(len(s['token_ids']) for s in segs)} tokens, "
          f"{sum(sum(s['score_mask']) for s in segs)} scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
