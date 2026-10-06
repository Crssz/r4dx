# tools/ngram: would an n-gram drafter pay?

The question (2026-10-03 research): standalone prompt-lookup gave 1.3-1.8 tok/round against DFlash k=7's
3.4-4.2 on benchmark prompts, and real agent traffic gets DFlash about 2.96 tok/round. A hybrid
(n-gram when a long match exists, DFlash otherwise) might still win on agent traffic, where the model
quotes files, diffs and tool output it has just been shown. This directory measures that on the real
requests instead of guessing. The verify window is capped at 8 rows (`kMaxUnsplitDraftK` = 7), so the
gate for building it is:

- standalone n-gram at k=7 reaches **>= 2.5 tok/round**, or
- **more than 50%** of the generated tokens sit in copy runs of **10+ tokens**.

## 1. Capture: run oh-my-pi against r4dx-server with logging on

`--request-log-tokens` records every request's whole prompt, the generated tokens and DFlash's accepted
draft tokens per round (docs/server.md "Request log" > "Token capture"). Off by default; it changes no
output. **The file holds your prompts and the model's answers as token ids, which decode back to text:
keep it on the local disk and delete it when the study is done.**

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
New-Item -ItemType Directory -Force E:\models\r4dx\ngram | Out-Null
.\build\win-hip\src\server\r4dx-server.exe `
    --model E:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx --layout trellis `
    --dflash E:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx --dflash-k 7 `
    --host 127.0.0.1 --port 8080 `
    --request-log E:\models\r4dx\ngram\requests.jsonl --request-log-tokens
```

That is the production launch line of docs/usage.md ("Run the OpenAI-compatible server") plus the two
log flags; add whatever else the usual omp setup passes (`--think on`, `--max-ctx`, ...) and use DFlash
k=7 as in the study: the logged acceptance is the baseline the hybrid estimate is compared with. Point
omp's OpenAI-compatible provider at `http://127.0.0.1:8080/v1`. The file is appended to across restarts
and can be followed while the server runs (`Get-Content E:\models\r4dx\ngram\requests.jsonl -Wait -Tail 1`
prints enormous lines; `(Get-Item ...).Length` and `Measure-Object -Line` are kinder). A few days of
agent work is a few hundred MB at most, since a request after the first stores only its new tokens.

## 2. Replay

```powershell
python tools\ngram\sim_ngram.py E:\models\r4dx\ngram\requests.jsonl
python tools\ngram\sim_ngram.py E:\models\r4dx\ngram\requests.jsonl --greedy-only --rows 0 --json E:\models\r4dx\ngram\sim.json
```

Plain `python` (stdlib only; numpy is not used). Options: `--min-n/--max-n` (suffix lengths, default
2..5), `--k` (draft lengths, default `7,9`; 9 is a what-if beyond the 8-row window), `--hybrid-t`
(match-length thresholds for the hybrid, default `3,4,5`), `--dflash-tpr` (tok/round for requests
whose log has no DFlash rounds, default 2.96), `--window N` (only match sources within the last N
tokens), `--min-gen` (skip tiny generations), `--greedy-only`, `--rows` (per-request rows printed:
30 by default, 0 none, -1 all).

What it does, per request: rebuilds the prompt from `prompt_shared` + `prompt_ids`, then walks the
generation round by round. A round proposes the tokens that followed the **most recent** earlier
occurrence of the **longest** suffix of the context (prompt + tokens generated so far, 2..5 tokens),
up to k of them; the verify accepts the longest prefix equal to what was actually generated next, plus
one bonus token. No match = a plain 1-token step.

Reported:

| number | meaning |
|---|---|
| tok/round | generated tokens / rounds (the engine's own `tok/round` convention) for the log's DFlash, for n-gram alone at each k, and for the hybrid |
| round hit | rounds whose proposal had at least one accepted token |
| token hit, run >= 10 | share of tokens after the first inside a copy run (the match's continuation keeps equalling the output), and inside runs of 10+ tokens; plus the run-length histogram |
| hybrid | per round: n-gram if its match is >= T tokens, else a DFlash round. A DFlash round emits what the log's DFlash round covering that position emitted (`positional`, exact when the rounds line up, optimistic after an n-gram stretch) or the request's mean logged tok/round (`mean`) |
| gate | the two conditions above, PASS/FAIL, and a verdict line |

Caveats. Greedy requests (temperature 0) are replayed exactly. A sampled request is one sampled
trajectory: the replay is an unbiased but noisy estimate (a deterministic proposal is accepted with
probability p(token)); use `--greedy-only` to see the exact subset. An n-gram round is counted as one
round like a DFlash round although it skips the drafter forward, so the hybrid's tok/round understates
its speed-up; the standalone numbers need no such correction. Requests that failed (non-200) are
skipped.

## 3. Without real data

```powershell
python tools\ngram\sim_ngram.py --selftest                 # synthetic log, cross-checks the proposer against a brute-force definition
python tools\ngram\sim_ngram.py tools\ngram\example_requests.jsonl --rows -1   # the 3 synthetic requests, in the server's file format
python tools\ngram\sim_ngram.py --write-example tools\ngram\example_requests.jsonl   # regenerate it
```

`example_requests.jsonl` is made of random token ids (a quoted block, a fresh reply, a sampled reply
with a repeated phrase) and carries only the fields the replay reads, not the server's full line: it
shows the token fields and the `prompt_shared` encoding and runs the report, and says nothing about
real traffic.
