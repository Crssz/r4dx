#requires -Version 5.1
<#
.SYNOPSIS
  Integration smoke test for r4dx-server: starts the server, hits /v1/models and a non-streaming
  and a streaming /v1/chat/completions, checks the JSON/SSE shapes.

.DESCRIPTION
  Only HIP device 1 may be used (project GPU rule) -- sets HIP_VISIBLE_DEVICES=1 before starting
  r4dx-server.exe; the one exception is -Tp 2 in real mode, which uses both GPUs by design (docs/tp.md
  9.2) under a TDR watch (see -Tp). Defaults to the 4-layer test container (qwen38-27b-l4-bf16.r4dx, --layout
  w4a16, the group-64 copy in <models-root>\r4dx\g64 -- see -Model) -- that model's
  text is nonsense (4 of 64 layers, arbitrary quantized-layout weights on a model that was never actually trained/converted for real use at 4 layers), so this
  script only checks response/SSE *shapes* and token counts, never the generated text itself.
  Pass -Model/-Layout to point at the real 64-layer container instead for a real-answer smoke run
  (docs/server.md's "Real-answer smoke run" section records one such run's output).

.PARAMETER Model
  Path to a .r4dx container. Default: the 4-layer test container qwen38-27b-l4-bf16.r4dx packed at
  w4a16 group 64, <models-root>\r4dx\g64\qwen38-27b-l4-bf16.r4dx (tools\r4dx_containers.ps1);
  R4DX_TEST_CONTAINER_DIR, when set, overrides the directory exactly as it does for ctest
  (tests/model/test_container_path.h). For a real-answer run pass the production container:
  <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx (the Huihui abliterated trellis mix4.5m,
  with -Layout trellis).

.PARAMETER Layout
  Body layout. Default: w4a16 (the 4-layer default container). The 64-layer production container is
  a trellis container and needs -Layout trellis.

.PARAMETER Port
  Port to run r4dx-server on. Default: 8091 (unlikely to collide with a dev server).

.PARAMETER Layers
  Passed to r4dx-server's --layers (r4dx::model::ModelOptions::layer_limit): the test container
  only physically carries 4 layers even though its (verbatim-copied) config.json still declares
  64, so this must be 4 for the default -Model. Pass -1 (or omit --layers entirely) for a real,
  full-size container.

.PARAMETER Mtp
  Passed to r4dx-server's --mtp (r4dx::model::ModelOptions::mtp_draft_k). 0 (default) disables MTP
  entirely. >0 requires -Model to be an MTP-converted container (docs/mtp.md's mtp.* weights) --
  e.g. <models-root>\r4dx\g64\qwen38-27b-l4-mtp.r4dx (4-layer test container) or the real 64-layer
  container with
  -Mtp 3 (this stage's own required verification runs).

.PARAMETER Dflash
  Path to a DFlash2 draft container, passed to r4dx-server's --dflash (docs/dflash2.md, Milestone 5
  stage S3 item 7). Empty (default) disables it. Mutually exclusive with -Mtp > 0 (server_args.h's
  own check rejects both at once) -- needs -Model pointed at a container with enough layers to
  cover the draft's own target_layers (the real 64-layer container for every shipped DFlash2
  container, whose target_layers reach layer 62).

.PARAMETER Vision
  Exercises image content parts end to end (docs/vision.md, docs/server.md's "Images") against a
  REAL vision-capable container -- pass -Model/-Layout/-Layers -1 pointed at one (e.g.
  <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx with -Layout trellis on the default build). Generates its own tiny synthetic PNGs
  with System.Drawing (no files committed to the repo): a shapes image for a description check, a rendered-text image for
  an OCR check, then two-images-in-one-request, image+tools, image+thinking, streaming, multi-turn
  prefix reuse (turn 2 must NOT re-encode: no `timings.image_n` key), different-image-same-text
  (must NOT reuse the prefix), and a battery of bad-input 400s. Off by default -- the DEFAULT run
  (no -Vision, the 4-layer test container) instead asserts the one thing that must ALWAYS hold: an
  image content part against a container with no vision tower is a clean 400 naming that, never a
  crash or a silent no-op.

.PARAMETER ToolRoundTrip
  Exercises a real tool call/result/answer multi-turn round trip (docs/server.md's "Tool calls"):
  offers a `get_current_weather` tool definition, sends the server's own parsed
  `message.tool_calls` back as a `role: "tool"` follow-up message, and checks the final answer
  comes back 200 with real prose. Also streams the same tool-offering turn and checks the live gate
  still delivers one complete `delta.tool_calls` batch without leaking any `<tool_call>` markup into
  a content delta. Off by default -- the 4-layer test container's nonsense output cannot reliably be
  coaxed into emitting a well-formed `<tool_call>` block, so this only produces a meaningful check
  against a real container (pass -Layers -1 with a real -Model).

.PARAMETER Tp
  Tensor parallelism (docs/tp.md 9.1, P5): 1 (default) is the single-device server on HIP device 1,
  exactly as before this parameter existed. 2 starts r4dx-server with `--tp 2` (and `--tp-mode`, see
  -TpMode) and passes the same `--tp 2 --tp-mode <mode>` to every r4dx-cli comparison run this script
  makes, so server and CLI always run the same engine. In real mode (the default) HIP_VISIBLE_DEVICES
  is REMOVED for the whole run (both GPUs: rank 0 = device 1, rank 1 = device 0, the desktop card),
  the script refuses to start while any r4dx-server is already running (docs/tp.md 9.2), and the run
  is wrapped in the device-0 TDR check (tools\tp\tdr_check.ps1, Appendix B N55/N64): every 20 s while
  the server runs (a TDR stops the server at once, no retry), once more before the r4dx-cli
  comparison run starts, then 30 s after the end, plus a scan of the server's and that r4dx-cli run's
  stderr for HIP error 719. Any TDR fails the run. When that sampled comparison fails under -Tp 2,
  the same seeded request also runs through r4dx-cli at --tp 1 (device 1), speculative and plain, and
  the script says whether the mismatch is also there at TP=1 (the known class, docs/tp.md Appendix B
  N29 / N80) or TP-specific; the check itself stays FAILED either way.

.PARAMETER TpMode
  With -Tp 2: real (default), emulate or noop -- passed to `--tp-mode`. emulate and noop keep
  HIP_VISIBLE_DEVICES=1 (both ranks, or the one noop rank, on device 1), so they touch no device 0 and
  run no TDR check. noop's tokens are meaningless (a no-op all-reduce, timing only): only shapes hold.

.PARAMETER TpFault
  With -Tp 2 (real or emulate): the recovery check of docs/tp.md 8.4 / gate G11. The server starts
  with R4DX_TP_FAULT=<-TpFaultSpec> (default 1:3000:1: rank 1 stalls 700 ms before its 3000th
  all-reduce after warm-up, so rank 0's all-reduce times out -- on device 1, the headless card, in real
  mode). Right after the server is ready, before every other check: the reference request (the
  standard smoke prompt, the first chat request below, greedy) is sent while the fault is still
  pending; then request A, a long greedy generation that crosses the armed all-reduce (HTTP 500 or 200
  are both accepted for A itself -- it is the fault request -- but the server log must show the fault
  fired while A ran); then request B = the standard smoke prompt again, which must return 200 with
  text equal to the reference, through a Reset() that recovered the group (the request log line's
  `tp_recovery=yes`). The rest of the smoke then runs on the recovered server as usual.

.PARAMETER TpFaultSpec
  The R4DX_TP_FAULT value -TpFault uses ("<rank>:<n>:<kind>", docs/tp.md 9.1). Default 1:3000:1.
  On the 4-layer default container the reference request uses 72 all-reduces (measured: 8 per
  forward, one prefill chunk + 8 decode steps), so request A reaches #3000 about 366 tokens in; on
  the 64-layer container (64 layers x 2 = 128 per forward) that is roughly 15 tokens in (computed,
  not measured). A request A that ends before #3000 fails the "fault fired while request A ran" check.
  In real mode a stall on rank 0 (0:<n>:1) is refused before anything starts: rank 1's all-reduce would
  spin on HIP device 0, the desktop card, until it timed out (docs/tp.md Appendix B N61).

.PARAMETER Pp
  Pipeline-parallel prefill (docs/pp-tp2-hybrid.md): 1 (default) leaves everything as it was. 2 needs -Tp 2 in
  real mode and starts the HYBRID serving mode, `--tp 2 --pp 2` (PP-2 prefill on stage Models of both cards, one
  reshard into the TP ranks, TP-2 decode; every prefill of >= --pp-min-rows = 1024 new tokens is pipelined, shorter
  ones take the TP=2 prefill). The server is started with --max-ctx 16384 (the stage-KV capacity S must be >= 16384,
  so every context_length check below expects 16384, not 4096), `--pp-split` (see -PpSplit), and the same flags go
  to the r4dx-cli comparison run. The server logs a "[stats] hybrid: ..." line after every request (engine.cpp:
  TpModel::HybridStatsLine, the line `r4dx-cli --stats` prints); this script parses its cumulative counters and, after
  the ordinary checks, runs the HYBRID MATRIX (all cases need prompts of >= 1024 new tokens, built from a
  deterministic synthetic log, see -HybridFillerLines; each asserts both the answer and the counters):
    - load: "[r4dx-hybrid] engaged in ..." and no "hybrid mode OFF" line; the short requests of the ordinary checks
      ran the TP prefill (0 pipelined calls before the matrix, unless -TpFault's long request A ran);
    - plain multi-turn: a cold long turn (pipelined, position 0), warm turns 2 and 3 of >= 1024 new tokens
      (pipelined, position > 0 when the server reused the prefix), a short turn 4 (TP prefill), a long turn 5 after
      TP-only work (pipelined again: the warm gather), a regenerate of turn 5 twice (a cold, identical call: the two
      answers must be equal);
    - thinking on: a long thinking turn and its replay without reasoning_content (checkpoint restore, then a warm
      hybrid call), with the dispatch counted per prefill call (the server splits a thinking prompt in two calls);
    - -Mtp > 0 / -Dflash: the long turns' request log lines show the speculative path (` mtp: ` / ` dflash: `),
      timings.draft_n > 0, and with -Dflash every pipelined call grew the DFlash tail rows;
    - -ToolRoundTrip: a long conversation with a tool call, then a LONG tool result (a warm hybrid call after a
      TP-decoded tool-call turn) and the final answer;
    - -Vision: image first / image after >= 3000 tokens of text (inside the DFlash tail window) / image first with
      3000 tokens after it (outside), then a warm follow-up turn (no image re-encode, pipelined, with -Dflash the
      tail rope rows after the mrope carry); -VisionStraddle sweeps the text after the image so the 2048-row tail
      window start passes through the image;
    - -TpFault: request A gets a long prompt, so the all-reduce fault fires in the decode AFTER a hybrid prefill, and the
      first long request after the recovery must be pipelined again (Reset() resets the stage Models).
  The case list is the design's P4 row (docs/pp-tp2-hybrid.md 9); the pure identity of the state lives in
  test_hybrid_real_identity / test_tp_hybrid_identity, this script checks the product path and the counters.

.PARAMETER PpSplit
  With -Pp 2: the split layer k (`--pp-split`). 0 (default): the server's own default (29, 30 with a drafter) for a
  full container, layers/2 for a partial container (-Layers N, N >= 0).

.PARAMETER HybridFillerLines
  With -Pp 2: the number of lines of synthetic log text per long user message (default 50, about 30 tokens each:
  1500 tokens, comfortably above the 1024-row threshold). A case that does not reach --pp-min-rows says so; raise it.

.PARAMETER VisionStraddle
  With -Pp 2 -Vision: also sweep the length of the text after an image so the DFlash tail window (the last 2048 rows)
  starts inside the image (about 20 extra requests, ~1 min). Off by default.

.EXAMPLE
  .\tools\server\smoke.ps1
.EXAMPLE
  .\tools\server\smoke.ps1 -Tp 2                 # both GPUs, the 4-layer default container
.EXAMPLE
  .\tools\server\smoke.ps1 -Tp 2 -Pp 2           # the hybrid, 4-layer default container (split 2): the counters and the dispatch
.EXAMPLE
  .\tools\server\smoke.ps1 -Tp 2 -Pp 2 -Model <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx -Layout trellis -Layers -1 -Dflash <models-root>\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx -ToolRoundTrip -Vision
.EXAMPLE
  .\tools\server\smoke.ps1 -Tp 2 -TpFault        # recovery after an injected all-reduce timeout
.EXAMPLE
  .\tools\server\smoke.ps1 -Tp 2 -Model <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx -Layout trellis -Layers -1 -Dflash <models-root>\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx -TpFault
.EXAMPLE
  .\tools\server\smoke.ps1 -Model <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx -Layout trellis -Layers -1
.EXAMPLE
  .\tools\server\smoke.ps1 -Model <models-root>\r4dx\g64\qwen38-27b-l4-mtp.r4dx -Layout w4a16 -Mtp 3
.EXAMPLE
  .\tools\server\smoke.ps1 -Model <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx -Layout trellis -Layers -1 -Mtp 3
.EXAMPLE
  .\tools\server\smoke.ps1 -Model <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx -Layout trellis -Layers -1 -ToolRoundTrip
#>
[CmdletBinding()]
param(
    [string]$Model = "",  # "" = the group-64 4-layer test container (see .PARAMETER Model)
    [string]$Layout = "w4a16",
    [int]$Port = 8091,
    [int]$Layers = 4,
    [int]$Mtp = 0,
    [string]$Dflash = "",
    [switch]$ToolRoundTrip,
    [switch]$Vision,
    [string]$Preset = "win-hip",
    [ValidateSet(1, 2)][int]$Tp = 1,
    [ValidateSet("", "real", "emulate", "noop")][string]$TpMode = "",
    [switch]$TpFault,
    [string]$TpFaultSpec = "1:3000:1",
    [ValidateSet(1, 2)][int]$Pp = 1,
    [int]$PpSplit = 0,
    [int]$HybridFillerLines = 50,
    [switch]$VisionStraddle
)

Add-Type -AssemblyName System.Drawing

# Synthetic test images (docs/vision.md's own "generate inside the script" preference over
# committing binary fixtures): built with System.Drawing, never touching disk longer than one PNG
# write/read round trip, freed immediately after. `DrawText` renders a KNOWN string the model is
# expected to read back verbatim under greedy decoding (docs/vision.md's own real-container OCR
# check used the same "render text, read text back" idea with "R4DX7391"; this stage's own smoke
# uses a different string so the two are never confused if compared side by side).
# Deterministic: the same arguments give byte-identical PNGs. -Square draws a square where the circle
# is -- a DIFFERENT picture of the same size (so the same grid and placeholder run).
function New-SyntheticShapesImageBase64 {
    param([switch]$Square)
    $bmp = New-Object System.Drawing.Bitmap 256, 256
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $rect = New-Object System.Drawing.Rectangle 0, 0, 256, 256
        $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
            $rect, [System.Drawing.Color]::FromArgb(20, 20, 200), [System.Drawing.Color]::FromArgb(220, 220, 20),
            [System.Drawing.Drawing2D.LinearGradientMode]::ForwardDiagonal)
        $g.FillRectangle($brush, $rect)
        $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::Black, 4)
        if ($Square) {
            $g.FillRectangle([System.Drawing.Brushes]::White, 78, 78, 100, 100)
            $g.DrawRectangle($pen, 78, 78, 100, 100)
        } else {
            $g.FillEllipse([System.Drawing.Brushes]::White, 78, 78, 100, 100)
            $g.DrawEllipse($pen, 78, 78, 100, 100)
        }
    } finally { $g.Dispose() }
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    [System.Convert]::ToBase64String($ms.ToArray())
}

function New-OcrImageBase64 {
    param([string]$Text = "R4DXVSN9")
    $bmp = New-Object System.Drawing.Bitmap 320, 128
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $g.Clear([System.Drawing.Color]::White)
        $font = New-Object System.Drawing.Font("Consolas", 36, [System.Drawing.FontStyle]::Bold)
        $g.DrawString($Text, $font, [System.Drawing.Brushes]::Black, 10, 40)
        $font.Dispose()
    } finally { $g.Dispose() }
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    [System.Convert]::ToBase64String($ms.ToArray())
}

function Image-DataUri { param([string]$Base64) "data:image/png;base64,$Base64" }

# -Pp 2: deterministic synthetic text for the long (>= 1024 new tokens) user messages of the hybrid matrix and of
# -TpFault's request A. $Seed makes each message different (so no turn is a repeat of an earlier one); every line
# is about 30 tokens (digits are single tokens), so the default 50 lines are about 1500.
function New-HybridFiller {
    param([int]$Seed, [int]$Lines = 50)
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $Lines; $i++) {
        $n = $Seed * 1000 + $i
        [void]$sb.AppendLine("Log entry $n : the lighthouse keeper recorded wind $(($n * 7) % 90) knots, tide $(($n * 3) % 12) feet and visibility $(($n % 9) + 1) miles at dock $(($n * 11) % 40).")
    }
    $sb.ToString()
}

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..

$RepoRoot = (Get-Location).Path
$ServerExe = Join-Path $RepoRoot "build\$Preset\src\server\r4dx-server.exe"
if (-not (Test-Path $ServerExe)) { throw "r4dx-server.exe not found at $ServerExe -- run .\build.ps1 first" }
# Default container from tools\r4dx_containers.ps1; -Model wins.
. (Join-Path $RepoRoot "tools\r4dx_containers.ps1")
if (-not $Model) {
    $Model = Get-R4dxTestContainer -Name "qwen38-27b-l4-bf16.r4dx"
}
if (-not (Test-Path $Model)) { throw "model container not found: $Model" }

$BaseUrl = "http://127.0.0.1:$Port"
$script:Failures = 0

function Check {
    param([bool]$Condition, [string]$Message)
    if ($Condition) {
        Write-Output "  [PASS] $Message"
    } else {
        Write-Output "  [FAIL] $Message"
        $script:Failures++
    }
}

# "<n> chars, <m> UTF-8 bytes" for a check message. .Length counts UTF-16 characters, not bytes.
# Printed alone and labelled "bytes", a 283-vs-286 mismatch (the same 286 UTF-8 bytes on the wire,
# one side decoded as Latin-1) looked like 3 bytes lost in the stream assembly (docs/server.md's
# "Response shapes").
function Text-Size { param([string]$Text) "$($Text.Length) chars, $([System.Text.Encoding]::UTF8.GetByteCount($Text)) UTF-8 bytes" }

# The server's log line for one request ("request <id>: prompt=... new=... [reset=|restore=] ckpt=...",
# engine.cpp; <id> is the response's own `id`). Written just after the response goes out, so it is
# polled for briefly. "" when it never appears.
function Get-RequestLogLine {
    param([string]$Id)
    for ($i = 0; $i -lt 20; $i++) {
        $hit = @(Select-String -Path $ServerErrLog -Pattern "request ${Id}:" -SimpleMatch -ErrorAction SilentlyContinue)
        if ($hit.Count -gt 0) { return $hit[-1].Line }
        Start-Sleep -Milliseconds 250
    }
    return ""
}

# ---- -Pp 2: the hybrid's counters (docs/pp-tp2-hybrid.md 15) ---------------------------------------------------
# After every request the server logs "[stats] hybrid: <TpModel::HybridStatsLine> [request <id>]" (engine.cpp; only when
# the hybrid mode is engaged). The counters are cumulative since the load; a request's own effect is the difference
# between the line before it and the line carrying its id.
function ConvertFrom-HybridStatsLine {
    param([string]$Line)
    $m = [regex]::Match($Line, 'hybrid: split k=(\d+), S=(\d+) tokens, min rows (\d+); (\d+) pipelined / (\d+) TP-prefill calls \(below min rows (\d+), over S (\d+), gather budget (\d+), DFlash frontier (\d+), not engaged (\d+)\); last pipelined call (\d+) rows at position (\d+): ')
    if (-not $m.Success) { return $null }
    $t = [regex]::Match($Line, 'tail rows (\d+)')
    [pscustomobject]@{
        Split = [int]$m.Groups[1].Value; S = [int64]$m.Groups[2].Value; MinRows = [int64]$m.Groups[3].Value
        Pipelined = [int64]$m.Groups[4].Value; TpPrefill = [int64]$m.Groups[5].Value
        BelowMin = [int64]$m.Groups[6].Value; OverS = [int64]$m.Groups[7].Value; GatherBudget = [int64]$m.Groups[8].Value
        DflashFrontier = [int64]$m.Groups[9].Value; NotEngaged = [int64]$m.Groups[10].Value
        LastRows = [int64]$m.Groups[11].Value; LastP0 = [int64]$m.Groups[12].Value
        TailRows = $(if ($t.Success) { [int64]$t.Groups[1].Value } else { 0 })
        Line = $Line
    }
}

# The newest "[stats] hybrid:" line of the server log, parsed ($null when there is none yet).
function Get-HybridStats {
    $hit = @(Select-String -Path $ServerErrLog -Pattern "[stats] hybrid:" -SimpleMatch -ErrorAction SilentlyContinue)
    if ($hit.Count -eq 0) { return $null }
    ConvertFrom-HybridStatsLine $hit[-1].Line
}

# The stats line a request left behind (written just after its "request <id>:" line, so polled like Get-RequestLogLine).
function Wait-HybridStats {
    param([string]$Id)
    for ($i = 0; $i -lt 40; $i++) {
        $hit = @(Select-String -Path $ServerErrLog -Pattern "[request $Id]" -SimpleMatch -ErrorAction SilentlyContinue |
                 Where-Object { $_.Line.Contains("[stats] hybrid:") })
        if ($hit.Count -gt 0) { return (ConvertFrom-HybridStatsLine $hit[-1].Line) }
        Start-Sleep -Milliseconds 250
    }
    return $null
}

# One greedy, non-streaming chat request with the stats line before and after it.
function Invoke-HybridChat {
    param([object[]]$Messages, [int]$MaxTokens = 24, [hashtable]$Extra = @{})
    $before = Get-HybridStats
    if ($null -eq $before) { $before = ConvertFrom-HybridStatsLine "hybrid: split k=0, S=0 tokens, min rows 0; 0 pipelined / 0 TP-prefill calls (below min rows 0, over S 0, gather budget 0, DFlash frontier 0, not engaged 0); last pipelined call 0 rows at position 0: " }
    $body = @{ messages = $Messages; max_tokens = $MaxTokens; temperature = 0; stream = $false }
    foreach ($k in $Extra.Keys) { $body[$k] = $Extra[$k] }
    $status = 0
    $chat = $null
    try {
        $resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" `
            -Body ($body | ConvertTo-Json -Depth 12) -UseBasicParsing -TimeoutSec 900
        $status = [int]$resp.StatusCode
        $chat = $resp.Content | ConvertFrom-Json
    } catch {
        if ($null -eq $_.Exception.Response) { throw }
        $status = [int]$_.Exception.Response.StatusCode.value__
    }
    $after = $null
    if ($null -ne $chat) { $after = Wait-HybridStats $chat.id }
    $text = if ($null -ne $chat -and $chat.choices.Count -gt 0) { [string]$chat.choices[0].message.content } else { "" }
    [pscustomobject]@{ Status = $status; Chat = $chat; Before = $before; After = $after; Text = $text }
}

# Asserts what the hybrid did with the prefill call(s) of one request (Invoke-HybridChat's result):
#   - exactly -Calls prefill calls were counted (a thinking prompt is split in two by the prompt checkpoint, engine.cpp);
#   - the dispatch rule: the call is PIPELINED iff its new rows >= the line's "min rows" and the whole context <= S,
#     otherwise it ran the TP=2 prefill (the counters must agree with timings.prompt_n whatever the case wanted);
#   - -Want pipe / tp: the case is live (a long case that did not reach min rows, or a short case that did, fails);
#     'tp' is not enforced against a partial container (-Layers N), whose nonsense replies need not round-trip, so a
#     'short' turn may legitimately re-prefill the whole conversation;
#   - -Cold: the pipelined call started at position 0; -Warm: when the server reused the prefix (timings.prompt_n <
#     usage.prompt_tokens) it started at position > 0 (a warm gather);
#   - with -Dflash every pipelined call grew the DFlash tail rows (the drafters' windows were fed).
function Test-HybridDispatch {
    param($R, [string]$Label, [ValidateSet("pipe", "tp", "any")][string]$Want = "any", [int]$Calls = 1, [switch]$Cold, [switch]$Warm)
    Check ($R.Status -eq 200) "hybrid: $Label returns 200 (got $($R.Status))"
    if ($R.Status -ne 200) { return }
    $a = $R.After
    $b = $R.Before
    Check ($null -ne $a) "hybrid: $Label : the server log carries a '[stats] hybrid:' line for this request"
    if ($null -eq $a) { return }
    $promptN = [int64]$R.Chat.timings.prompt_n
    $total = [int64]$R.Chat.usage.prompt_tokens
    $dP = $a.Pipelined - $b.Pipelined
    $dT = $a.TpPrefill - $b.TpPrefill
    # The call the rule sees: a split thinking prompt feeds its head (all but the last token) first.
    $headRows = if ($Calls -eq 2) { $promptN - 1 } else { $promptN }
    $expectPipe = ($headRows -ge $a.MinRows) -and ($total -le $a.S)
    $declined = "declined: below min rows $($a.BelowMin), over S $($a.OverS), gather budget $($a.GatherBudget), DFlash frontier $($a.DflashFrontier), not engaged $($a.NotEngaged)"
    Check (($dP + $dT) -eq $Calls) "hybrid: $Label : $Calls prefill call(s) counted (pipelined +$dP, TP prefill +$dT)"
    if ($expectPipe) {
        Check ($dP -eq 1) "hybrid: $Label : $headRows new rows >= min rows $($a.MinRows), context $total <= S $($a.S) -> the call was pipelined (+$dP; $declined)"
    } else {
        Check ($dP -eq 0) "hybrid: $Label : $headRows new rows (min rows $($a.MinRows), context $total, S $($a.S)) -> the TP=2 prefill, not the pipeline (pipelined +$dP)"
    }
    if ($Want -eq "pipe") {
        Check $expectPipe "hybrid: $Label : the case is live ($headRows new rows >= min rows $($a.MinRows); raise -HybridFillerLines if not)"
    } elseif ($Want -eq "tp" -and $Layers -lt 0) {
        Check (-not $expectPipe) "hybrid: $Label : a short turn ($headRows new rows < min rows $($a.MinRows)) stays on the TP=2 prefill"
    }
    if ($dP -eq 1) {
        Check ($a.LastRows -ge $a.MinRows) "hybrid: $Label : the pipelined call fed $($a.LastRows) rows at position $($a.LastP0)"
        if ($Cold) { Check ($a.LastP0 -eq 0) "hybrid: $Label : a cold call (position 0, got $($a.LastP0))" }
        if ($Warm -and $promptN -lt $total) {
            Check ($a.LastP0 -gt 0) "hybrid: $Label : a warm call (prefix reused: position $($a.LastP0) > 0, $promptN of $total tokens fed)"
        } elseif ($Warm) {
            Write-Output "  [info] hybrid: $Label : the prefix was not reused ($promptN of $total tokens fed): a cold call, warm assertion skipped"
        }
        if ($Dflash -ne "") {
            Check ($a.TailRows -gt $b.TailRows) "hybrid: $Label : the DFlash tail was injected (tail rows $($b.TailRows) -> $($a.TailRows))"
        }
    }
}

# -Mtp / -Dflash: the request decoded speculatively (after a hybrid prefill, or a warm hybrid turn's).
function Test-HybridSpec {
    param($R, [string]$Label)
    if ($Mtp -le 0 -and $Dflash -eq "") { return }
    if ($R.Status -ne 200) { return }
    Check ($R.Chat.timings.draft_n -gt 0) "hybrid: $Label : decoded speculatively after the prefill (timings.draft_n $($R.Chat.timings.draft_n) > 0)"
    $line = Get-RequestLogLine $R.Chat.id
    Check ($line -match ' (mtp|dflash): ') "hybrid: $Label : the request log line shows the speculative path"
}

# True when a reply begins or ends with whitespace, which the chat template's |trim drops when the
# client replays it -- so the replayed turn cannot re-tokenize to what was generated.
function Test-EdgeWhitespace { param([string]$Text) $Text -match '^\s' -or $Text -match '\s$' }

# POSTs a streaming request and returns every "data: ..." line WITH the millisecond offset at which
# it actually arrived. Invoke-WebRequest (used everywhere else in this script) buffers the whole
# body before returning, which is fine for checking SSE framing but cannot tell live streaming apart
# from a server that buffered the generation and dumped it at the end -- exactly the difference the
# live tool-call stream gate checks below are about (docs/server.md's "Tool calls"). HttpWebRequest
# hands back the response as soon as the headers land, and StreamReader.ReadLine returns per chunk
# as it arrives, so the offsets below are real arrival times.
function Invoke-SseStream {
    param([string]$Uri, [string]$Body, [int]$TimeoutSec = 600)
    [System.Net.ServicePointManager]::Expect100Continue = $false
    $req = [System.Net.HttpWebRequest]::Create($Uri)
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Proxy = $null  # never route a 127.0.0.1 request through a system proxy
    $req.Timeout = $TimeoutSec * 1000
    $req.ReadWriteTimeout = $TimeoutSec * 1000
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
    $req.ContentLength = $bytes.Length
    $reqStream = $req.GetRequestStream()
    $reqStream.Write($bytes, 0, $bytes.Length)
    $reqStream.Close()

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = $req.GetResponse()
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
    $events = New-Object System.Collections.ArrayList
    try {
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ($line -like "data: *") {
                [void]$events.Add([pscustomobject]@{
                    Data = $line.Substring(6)
                    Ms   = $sw.Elapsed.TotalMilliseconds
                })
            }
        }
    } finally {
        $reader.Close()
        $resp.Close()
    }
    [pscustomobject]@{ Events = $events; TotalMs = $sw.Elapsed.TotalMilliseconds }
}

# ---- tensor parallel (-Tp 2, docs/tp.md P5) ----------------------------------------------------
if ($Tp -eq 1 -and ($TpMode -ne "" -or $TpFault)) { throw "-TpMode and -TpFault need -Tp 2" }
$TpModeResolved = if ($TpMode) { $TpMode } else { "real" }
if ($TpFault -and $TpModeResolved -eq "noop") {
    throw "-TpFault needs -TpMode real or emulate (noop's all-reduce moves nothing, so there is nothing to fault)"
}
# Every r4dx-server and r4dx-cli this script starts runs the same engine.
$TpArgs = @()
if ($Tp -eq 2) { $TpArgs = @("--tp", "2", "--tp-mode", $TpModeResolved) }
# -Pp 2: the hybrid serving mode (docs/pp-tp2-hybrid.md): `--tp 2 --pp 2` for the server AND the r4dx-cli comparison
# run. The hybrid needs two real devices, and a stage-KV capacity S >= 16384 (so --max-ctx >= 16384, below).
if ($Pp -eq 2) {
    if ($Tp -ne 2 -or $TpModeResolved -ne "real") {
        throw "-Pp 2 is the hybrid mode (--tp 2 --pp 2): it needs -Tp 2 in real mode (the plain --pp 2 server is not covered by this script)"
    }
    $TpArgs += @("--pp", "2")
    $splitK = $PpSplit
    if ($splitK -le 0 -and $Layers -ge 0) { $splitK = [Math]::Max(1, [int][Math]::Floor($Layers / 2)) }
    if ($splitK -gt 0) { $TpArgs += @("--pp-split", "$splitK") }
} elseif ($PpSplit -ne 0 -or $VisionStraddle) {
    throw "-PpSplit and -VisionStraddle need -Pp 2"
}
if ($VisionStraddle -and -not $Vision) { throw "-VisionStraddle needs -Vision" }
# Real mode is the one configuration that touches HIP device 0 (the desktop card): both GPUs, so no
# other r4dx-server may be running (docs/tp.md 9.2), and the run is TDR-watched (see -Tp).
$UsesDevice0 = ($Tp -eq 2 -and $TpModeResolved -eq "real")
if ($UsesDevice0) {
    $already = @(Get-Process r4dx-server -ErrorAction SilentlyContinue)
    if ($already.Count -gt 0) {
        throw ("an r4dx-server is already running (pid $(($already | ForEach-Object { $_.Id }) -join ', ')) -- " +
               "stop it first: -Tp 2 uses both GPUs (docs/tp.md 9.2)")
    }
}
$TdrCheck = Join-Path $RepoRoot "tools\tp\tdr_check.ps1"
$TdrMarker = "$env:TEMP\r4dx-server-smoke.tdr.txt"
# -TpFaultSpec (docs/tp.md 9.1's R4DX_TP_FAULT, parsed again by TpModel::Load): a stall (kind 1) makes
# the OTHER rank's all-reduce kernel spin until it times out, so in real mode it must not be armed on
# rank 0, whose peer is rank 1 on HIP device 0 -- the desktop card, which cannot preempt compute
# (docs/tp.md Appendix B N61, N80).
if ($TpFault) {
    if ($TpFaultSpec -notmatch '^(\d+):(\d+):(\d+)$') { throw "-TpFaultSpec must be <rank>:<n>:<kind>, got '$TpFaultSpec'" }
    if ($UsesDevice0 -and [int]$Matches[1] -eq 0 -and [int]$Matches[3] -eq 1) {
        throw ("-TpFaultSpec $TpFaultSpec stalls rank 0, so rank 1's all-reduce would spin on HIP device 0 (the desktop " +
               "card) until it times out; real-mode stalls go on rank 1 (docs/tp.md Appendix B N61)")
    }
}
$RunStart = Get-Date
# Restored in the finally below: -Tp 2 real removes HIP_VISIBLE_DEVICES and -TpFault sets R4DX_TP_FAULT,
# and a caller that runs this in its own session must be left with neither change. Every environment
# change, the server start and the TDR watch happen INSIDE that try, so no failure between them can
# skip the restore, the server stop or the final TDR check (N80).
$SavedHipVisible = $env:HIP_VISIBLE_DEVICES
$SavedTpFault = $env:R4DX_TP_FAULT
$ServerErrLog = "$env:TEMP\r4dx-server-smoke.err.log"
$CliErrLog = "$env:TEMP\r4dx-server-smoke.cli.err.log"  # the -Tp 2 r4dx-cli comparison run's stderr
Remove-Item -LiteralPath $CliErrLog -ErrorAction SilentlyContinue
# 4096 (raised from 512, model-metadata/reasoning_content pass): the reasoning_content checks below
# send `max_tokens: 1024` on the real container so a real thinking span has room to close and still
# leave room for an answer -- 512 was too tight once thinking is on.
$MaxCtx = if ($Pp -eq 2) { 16384 } else { 4096 }  # -Pp 2: S must be >= 16384 (docs/pp-tp2-hybrid.md 2), S <= --max-ctx
$ServerArgList = @(
    "--model", $Model, "--layout", $Layout, "--host", "127.0.0.1", "--port", "$Port",
    "--max-ctx", "$MaxCtx", "--max-tokens-default", "16"
)
if ($Layers -ge 0) { $ServerArgList += @("--layers", "$Layers") }
if ($Mtp -gt 0) { $ServerArgList += @("--mtp", "$Mtp") }
if ($Dflash -ne "") { $ServerArgList += @("--dflash", "$Dflash") }
# The default (non--Vision) run's own "image on a non-vision container -> 400" check (below) assumes
# this server has no vision tower loaded. That was always true before the vision milestone (every
# -Model here, including the real 64-layer container, had no vision.* tensors), but the real
# container now ships them, and --vision defaults to "auto" (load iff present) -- so without this,
# pointing -Model at the real container without -Vision would auto-load vision and the "clean 400"
# check below would wrongly fail (not a server bug: the server is correctly answering the image).
# -Vision itself passes no --vision override, so its own real-container run keeps the "auto" default.
if (-not $Vision) { $ServerArgList += @("--vision", "off") }
$ServerArgList += $TpArgs
$proc = $null
$tdrJob = $null

try {
    Write-Output ("[smoke] starting r4dx-server: model=$Model layout=$Layout port=$Port mtp=$Mtp" +
                  $(if ($Tp -eq 2) { " tp=2 tp-mode=$TpModeResolved" } else { "" }) +
                  $(if ($Pp -eq 2) { " pp=2 (hybrid, max-ctx $MaxCtx)" } else { "" }) +
                  $(if ($TpFault) { " R4DX_TP_FAULT=$TpFaultSpec" } else { "" }))
    if ($UsesDevice0) {
        Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        Write-Output ("[smoke] -Tp 2 real: HIP_VISIBLE_DEVICES removed (both GPUs); TDR watch from " +
                      "$($RunStart.ToString('yyyy-MM-dd HH:mm:ss')), every 20 s")
    } else {
        $env:HIP_VISIBLE_DEVICES = "1"
    }
    # Never inherit an armed fault; -TpFault arms one for the server process only (removed right after
    # it starts, so no r4dx-cli comparison run below sees it).
    Remove-Item env:R4DX_TP_FAULT -ErrorAction SilentlyContinue
    if ($TpFault) { $env:R4DX_TP_FAULT = $TpFaultSpec }
    $proc = Start-Process -FilePath $ServerExe -ArgumentList $ServerArgList -PassThru `
      -RedirectStandardError $ServerErrLog `
      -RedirectStandardOutput "$env:TEMP\r4dx-server-smoke.out.log"
    Remove-Item env:R4DX_TP_FAULT -ErrorAction SilentlyContinue

    # In-run TDR watch (-Tp 2 real; tools\tp\tdr_watch.psm1's Start-TdrWatchJob, Appendix B N64/N77): a
    # background job runs tdr_check every 20 s and, at the first TDR, writes $TdrMarker and stops the
    # server at once -- no retry. Every request after that fails, and the finally below reports the TDR.
    if ($UsesDevice0) {
        Import-Module (Join-Path $RepoRoot "tools\tp\tdr_watch.psm1") -Force
        $tdrJob = Start-TdrWatchJob -Since $RunStart -TargetPid $proc.Id -Marker $TdrMarker
    }

    # Wait for the model to load and the listener to come up (container load alone can take
    # several seconds even for the 4-layer test container -- see docs/perf.md's load-time table
    # for the real 64-layer container's much longer load).
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) {
            Write-Output "[smoke] server process exited early (code $($proc.ExitCode)); stderr:"
            Get-Content "$env:TEMP\r4dx-server-smoke.err.log" -ErrorAction SilentlyContinue
            throw "r4dx-server exited before becoming ready"
        }
        try {
            $resp = Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "server did not become ready within timeout" }
    Write-Output "[smoke] server ready"

    # ---- --prompt-checkpoint (on by default, docs/server.md "Prefix cache"): its VRAM, per rank ---
    $ckptVram = @(Select-String -Path $ServerErrLog -Pattern "prompt checkpoint VRAM:" -SimpleMatch -ErrorAction SilentlyContinue)
    Check ($ckptVram.Count -ge 1) "--prompt-checkpoint (default on): the load log reports the checkpoint's VRAM"
    foreach ($l in $ckptVram) { Write-Output "  [info] $($l.Line)" }

    # ---- --tp 2: the load reported every rank (engine.cpp's one VRAM line per rank) --------------
    if ($Tp -eq 2) {
        $rankLines = @(Select-String -Path $ServerErrLog -Pattern "[r4dx-server] tp rank " -SimpleMatch -ErrorAction SilentlyContinue)
        $wantRanks = if ($TpModeResolved -eq "noop") { 1 } else { 2 }
        Check ($rankLines.Count -eq $wantRanks) `
            "--tp 2 --tp-mode ${TpModeResolved}: the load log has one 'tp rank' VRAM line per rank thread (got $($rankLines.Count), want $wantRanks)"
        foreach ($l in $rankLines) { Write-Output "  [info] $($l.Line)" }
    }

    # ---- --pp 2 (hybrid): the load engaged it (a budget that leaves no room drops back to plain --tp 2 with a log line) --
    if ($Pp -eq 2) {
        $engaged = @(Select-String -Path $ServerErrLog -Pattern "[r4dx-hybrid] engaged in" -SimpleMatch -ErrorAction SilentlyContinue)
        $dropped = @(Select-String -Path $ServerErrLog -Pattern "hybrid mode OFF" -SimpleMatch -ErrorAction SilentlyContinue)
        Check ($engaged.Count -eq 1 -and $dropped.Count -eq 0) `
            ("--pp 2: the hybrid mode engaged at load ('[r4dx-hybrid] engaged in ...' x$($engaged.Count), 'hybrid mode OFF' x$($dropped.Count)" +
             $(if ($dropped.Count -gt 0) { ": $($dropped[0].Line)" } else { "" }) + ")")
        foreach ($l in @(Select-String -Path $ServerErrLog -Pattern "[r4dx-hybrid] " -SimpleMatch -ErrorAction SilentlyContinue)) {
            Write-Output "  [info] $($l.Line)"
        }
    }

    # ---- -TpFault: a request that hits an injected all-reduce fault, then recovery ---------------
    # docs/tp.md 8.4 / gate G11. The fault (R4DX_TP_FAULT, armed at load) fires once, at the n-th
    # all-reduce after warm-up; the reference below runs while it is still pending, request A crosses
    # it, and request B -- the same request as the reference -- must come back 200 with the reference's
    # text, through the Reset() that recovers the group (docs/tp.md 2.5). Runs first, so the fault's
    # position does not depend on how many all-reduces the checks further down use.
    if ($TpFault) {
        $faultRefBody = @{
            messages    = @(@{ role = "user"; content = "Say hello in one short sentence." })
            max_tokens  = 8
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 5
        $faultRefResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $faultRefBody -UseBasicParsing
        $faultRef = $faultRefResp.Content | ConvertFrom-Json
        Check ($faultRefResp.StatusCode -eq 200) "TpFault: the reference request (before the fault fires) returns 200"
        $firedBefore = @(Select-String -Path $ServerErrLog -Pattern "fault injection (kind" -SimpleMatch -ErrorAction SilentlyContinue).Count
        Check ($firedBefore -eq 0) "TpFault: the fault has not fired yet after the reference request"

        # Request A: long and greedy, so it crosses the armed all-reduce on the 4-layer container too
        # (see -TpFaultSpec). With -Pp 2 its PROMPT is long too (>= 1024 new tokens), so the pipelined prefill runs
        # first and the fault fires in the TP decode after it (design P4: "AR fault after hybrid prefill"); the
        # prefill itself adds no all-reduce, so the fault lands a few forwards later than in the plain case.
        $faultAPrefix = if ($Pp -eq 2) { (New-HybridFiller 90 $HybridFillerLines) + "`n" } else { "" }
        $faultABody = @{
            messages    = @(@{ role = "user"; content = $faultAPrefix + "Write a long story about a lighthouse keeper who finds a message in a bottle. Use at least 600 words." })
            max_tokens  = 512
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 5
        $faultAStatus = 0
        try {
            $faultAResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $faultABody -UseBasicParsing -TimeoutSec 900
            $faultAStatus = [int]$faultAResp.StatusCode
        } catch {
            if ($null -eq $_.Exception.Response) { throw }
            $faultAStatus = [int]$_.Exception.Response.StatusCode.value__
        }
        Check ($faultAStatus -eq 500 -or $faultAStatus -eq 200) "TpFault: request A (the fault request) returns 500 or 200 (got $faultAStatus)"
        $fired = $false
        for ($i = 0; $i -lt 20 -and -not $fired; $i++) {
            $fired = @(Select-String -Path $ServerErrLog -Pattern "fault injection (kind" -SimpleMatch -ErrorAction SilentlyContinue).Count -gt 0
            if (-not $fired) { Start-Sleep -Milliseconds 500 }
        }
        Check $fired "TpFault: the armed fault fired while request A ran (server log: 'fault injection (kind ...) at all-reduce #...')"
        foreach ($l in @(Select-String -Path $ServerErrLog -Pattern "fault injection \(kind|\[r4dx-tp\] rank \d+ \(dev|tp: group needs recovery|tp: fatal" -ErrorAction SilentlyContinue)) {
            Write-Output "  [info] $($l.Line)"
        }
        Check (-not $proc.HasExited) "TpFault: the server is still running after request A"

        $faultBStatus = 0
        $faultB = $null
        try {
            $faultBResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $faultRefBody -UseBasicParsing
            $faultBStatus = [int]$faultBResp.StatusCode
            $faultB = $faultBResp.Content | ConvertFrom-Json
        } catch {
            if ($null -eq $_.Exception.Response) { throw }
            $faultBStatus = [int]$_.Exception.Response.StatusCode.value__
        }
        Check ($faultBStatus -eq 200) "TpFault: request B (the standard smoke prompt, after the fault) returns 200 (got $faultBStatus)"
        $refText = [string]$faultRef.choices[0].message.content
        $bText = if ($null -ne $faultB) { [string]$faultB.choices[0].message.content } else { "" }
        Check ($null -ne $faultB -and $bText -ceq $refText -and
               $faultB.usage.completion_tokens -eq $faultRef.usage.completion_tokens -and
               $faultB.choices[0].finish_reason -eq $faultRef.choices[0].finish_reason) `
            ("TpFault: request B's text, token count and finish_reason equal the pre-fault reference " +
             "(reference='$refText' B='$bText')")
        $recovered = $false
        for ($i = 0; $i -lt 20 -and -not $recovered; $i++) {
            $recovered = @(Select-String -Path $ServerErrLog -Pattern " tp_recovery=yes" -SimpleMatch -ErrorAction SilentlyContinue).Count -gt 0
            if (-not $recovered) { Start-Sleep -Milliseconds 500 }
        }
        Check $recovered "TpFault: request B's log line shows its Reset() recovered the group (tp_recovery=yes)"
        foreach ($l in @(Select-String -Path $ServerErrLog -Pattern " tp_recovery=yes" -SimpleMatch -ErrorAction SilentlyContinue)) {
            Write-Output "  [info] $($l.Line)"
        }
    }

    # ---- GET /v1/models ------------------------------------------------------------------------
    $modelsResp = Invoke-WebRequest -Uri "$BaseUrl/v1/models" -UseBasicParsing
    $models = $modelsResp.Content | ConvertFrom-Json
    Check ($modelsResp.StatusCode -eq 200) "/v1/models returns 200"
    Check ($models.object -eq "list") "/v1/models: object == 'list'"
    Check ($models.data.Count -ge 1) "/v1/models: data has at least one entry"
    Check ([bool]$models.data[0].id) "/v1/models: data[0].id is set"

    # ---- Model metadata extension fields (docs/server.md's "Model metadata" section) -------------
    Check ($models.data[0].context_length -eq $MaxCtx) `
        "/v1/models: data[0].context_length ($($models.data[0].context_length)) == the launched --max-ctx ($MaxCtx)"
    Check ($models.data[0].max_model_len -eq $MaxCtx) "/v1/models: data[0].max_model_len == --max-ctx"
    Check ($models.data[0].meta.n_ctx -eq $MaxCtx) "/v1/models: data[0].meta.n_ctx == --max-ctx"
    Check ($models.data[0].meta.n_ctx_train -gt 0) "/v1/models: data[0].meta.n_ctx_train > 0"
    Check (($models.data[0].capabilities) -contains "reasoning") `
        "/v1/models: data[0].capabilities contains 'reasoning'"

    # ---- OpenRouter-shaped capability block (docs/server.md's "Model metadata") -------------------
    Check ($models.data[0].top_provider.context_length -eq $MaxCtx) `
        "/v1/models: data[0].top_provider.context_length == --max-ctx"
    Check ($models.data[0].top_provider.max_completion_tokens -eq $MaxCtx) `
        "/v1/models: data[0].top_provider.max_completion_tokens == --max-ctx"
    Check ($models.data[0].top_provider.is_moderated -eq $false) `
        "/v1/models: data[0].top_provider.is_moderated == false"
    # The server was launched without --think, so the advertised default must say thinking is off.
    Check ($models.data[0].reasoning.default_enabled -eq $false) `
        "/v1/models: data[0].reasoning.default_enabled == false (server launched without --think)"
    Check ($models.data[0].reasoning.mandatory -eq $false) "/v1/models: data[0].reasoning.mandatory == false"
    foreach ($effort in @("none", "low", "medium", "high")) {
        Check (($models.data[0].reasoning.supported_efforts) -contains $effort) `
            "/v1/models: data[0].reasoning.supported_efforts contains '$effort'"
    }
    foreach ($param in @("reasoning", "reasoning_effort", "include_reasoning", "enable_thinking", "thinking")) {
        Check (($models.data[0].supported_parameters) -contains $param) `
            "/v1/models: data[0].supported_parameters contains '$param'"
    }
    # ["text"] unless this container's vision tower is actually loaded (-Vision implies a real,
    # vision-capable container) -- docs/vision.md's one-line ModelInputModalities() switch.
    if ($Vision) {
        Check (($models.data[0].architecture.input_modalities) -contains "image") `
            "/v1/models: data[0].architecture.input_modalities contains 'image' (-Vision)"
    } else {
        Check (($models.data[0].architecture.input_modalities -join ",") -eq "text") `
            "/v1/models: data[0].architecture.input_modalities == ['text'] (no vision tower loaded)"
    }

    # ---- GET /v1/models/{id} (task item 1) ---------------------------------------------------------
    $modelId = $models.data[0].id
    $oneModelResp = Invoke-WebRequest -Uri "$BaseUrl/v1/models/$modelId" -UseBasicParsing
    Check ($oneModelResp.StatusCode -eq 200) "GET /v1/models/{id}: known id returns 200"
    $oneModel = $oneModelResp.Content | ConvertFrom-Json
    Check ($oneModel.id -eq $modelId) "GET /v1/models/{id}: response id matches"
    Check ($oneModel.context_length -eq $MaxCtx) "GET /v1/models/{id}: context_length == --max-ctx"
    try {
        Invoke-WebRequest -Uri "$BaseUrl/v1/models/no-such-model" -UseBasicParsing | Out-Null
        Check $false "GET /v1/models/{unknown id} returns 404"
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Check ($status -eq 404) "GET /v1/models/{unknown id} returns 404 (got $status)"
    }

    # ---- POST /v1/chat/completions, non-streaming -----------------------------------------------
    $chatBody = @{
        messages    = @(@{ role = "user"; content = "Say hello in one short sentence." })
        max_tokens  = 8
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 5
    $chatResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $chatBody -UseBasicParsing
    $chat = $chatResp.Content | ConvertFrom-Json
    Check ($chatResp.StatusCode -eq 200) "chat completion (non-streaming) returns 200"
    Check ($chat.object -eq "chat.completion") "chat completion: object == 'chat.completion'"
    Check ([bool]$chat.id) "chat completion: id is set"
    Check ($chat.choices.Count -eq 1) "chat completion: exactly one choice"
    Check ($chat.choices[0].message.role -eq "assistant") "chat completion: choices[0].message.role == 'assistant'"
    Check ($null -ne $chat.choices[0].message.content) "chat completion: choices[0].message.content is present"
    Check ([bool]$chat.choices[0].finish_reason) "chat completion: finish_reason is set"
    Check ($chat.usage.prompt_tokens -gt 0) "chat completion: usage.prompt_tokens > 0"
    Check ($chat.usage.total_tokens -ge $chat.usage.prompt_tokens) "chat completion: usage.total_tokens >= prompt_tokens"

    # ---- `timings` (docs/server.md's "timings" section) on the non-streaming response -----------
    Check ($null -ne $chat.timings) "chat completion: timings object is present"
    Check ($chat.timings.predicted_per_second -gt 0) "chat completion: timings.predicted_per_second > 0"
    Check ($chat.timings.predicted_n -eq $chat.usage.completion_tokens) "chat completion: timings.predicted_n == usage.completion_tokens"
    Check ($chat.timings.prompt_n -gt 0) "chat completion: timings.prompt_n > 0"

    # ---- `timings.prompt_n` reflects prefix reuse (docs/server.md's "prompt_n semantics under
    # prefix reuse"): a second request that just extends the first request's own conversation
    # (same messages plus the assistant's real reply plus one more user turn) should only prefill
    # the newly-appended tail, so its own timings.prompt_n comes back smaller than its own
    # usage.prompt_tokens (the whole growing conversation's token count). Run back to back with
    # nothing else in between, so nothing else can invalidate the prefix in the meantime. The
    # strict "smaller" assertion is only meaningful against a real container -- like -ToolRoundTrip
    # above, the 4-layer test container's nonsense generated text is not guaranteed to round-trip
    # byte-for-byte back through encode(decode(...)) (this tokenizer's own documented NFC-
    # normalizer gap, tokenizer.h), which can make request 2's re-render diverge from what was
    # actually committed to KV and force a Reset() instead of a prefix match -- not a server bug.
    $prefixReuseBody1 = @{
        messages    = @(@{ role = "user"; content = "What is 2 plus 2?" })
        max_tokens  = 8
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 5
    $prefixReuseResp1 = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $prefixReuseBody1 -UseBasicParsing
    $prefixReuseChat1 = $prefixReuseResp1.Content | ConvertFrom-Json

    $prefixReuseBody2 = @{
        messages    = @(
            @{ role = "user"; content = "What is 2 plus 2?" },
            @{ role = "assistant"; content = $prefixReuseChat1.choices[0].message.content },
            @{ role = "user"; content = "And what is 3 plus 3?" }
        )
        max_tokens  = 8
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 5
    $prefixReuseResp2 = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $prefixReuseBody2 -UseBasicParsing
    $prefixReuseChat2 = $prefixReuseResp2.Content | ConvertFrom-Json
    Check ($prefixReuseResp2.StatusCode -eq 200) "prefix reuse: extending-conversation request returns 200"
    if ($Layers -lt 0) {
        Check ($prefixReuseChat2.timings.prompt_n -lt $prefixReuseChat2.usage.prompt_tokens) `
            "prefix reuse: second request's timings.prompt_n ($($prefixReuseChat2.timings.prompt_n)) < usage.prompt_tokens ($($prefixReuseChat2.usage.prompt_tokens))"
    } else {
        Write-Output "  [SKIP] prefix reuse: prompt_n < usage.prompt_tokens (only checked against a real container, -Layers -1)"
    }

    # ---- prompt checkpoint: a whitespace-led reply (docs/server.md "Prefix cache") ---------------
    # 2026-09-26: turn 1 answered " yes" (leading space) then EOS. The client replays exactly that,
    # but chat_template.jinja renders a replayed assistant turn through |trim, so the re-rendered
    # history says "yes" -- a different token -- and the server re-prefilled the whole conversation
    # (the GDN state cannot be rewound to the end of turn 1's prompt). With --prompt-checkpoint (on by
    # default) it keeps the state saved right after turn 1's prompt and continues from there: turn 2
    # must prefill EXACTLY the tokens after turn 1's prompt -- prompt_n == prompt_tokens(turn 2) -
    # prompt_tokens(turn 1); matching the whole committed sequence would prefill fewer, a reset all of
    # them -- and its log line must show `restore=`. Deterministic: greedy, so a given container and
    # build answer each question the same way every run. The case takes the first question whose
    # answer has edge whitespace and FAILS (never skips) when none has, so it cannot pass without
    # taking the path. The questions run from the likeliest LEADING space (a completion-style prompt:
    # the model continues the quoted text, " Paris") to a TRAILING one that holds by construction:
    # this tokenizer splits "1 2" into "1", " ", "2", so a count cut at max_tokens 2 ends "1 ".
    # Real container only, like the reuse check above.
    if ($Layers -lt 0) {
        $wsQuestions = @(
            @{ q = 'Continue this text with exactly one word and nothing else: "The capital of France is"'; max = 4 },
            @{ q = "Is the sky blue on a clear day? Reply with exactly one word: yes or no."; max = 4 },
            @{ q = "Count from 1 to 10, separated by single spaces, and nothing else."; max = 2 }
        )
        $wsTurn1 = $null
        $wsQuestion = ""
        $wsReplies = @()
        foreach ($wq in $wsQuestions) {
            $q = $wq.q
            $qBody = @{ messages = @(@{ role = "user"; content = $q }); max_tokens = $wq.max; temperature = 0; stream = $false } |
                ConvertTo-Json -Depth 5
            $qChat = (Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $qBody -UseBasicParsing).Content | ConvertFrom-Json
            $qReply = [string]$qChat.choices[0].message.content
            $wsReplies += "'$qReply'"
            if (Test-EdgeWhitespace $qReply) { $wsTurn1 = $qChat; $wsQuestion = $q; break }
        }
        Check ($null -ne $wsTurn1) `
            "whitespace-led reply: a greedy turn 1 answered with leading/trailing whitespace -- the case is live (replies: $($wsReplies -join ', '))"
        if ($null -ne $wsTurn1) {
            $wsReply = [string]$wsTurn1.choices[0].message.content
            $wsTurn2Body = @{
                messages    = @(
                    @{ role = "user"; content = $wsQuestion },
                    @{ role = "assistant"; content = $wsReply },
                    @{ role = "user"; content = "Why? Answer in one short sentence." }
                )
                max_tokens  = 16
                temperature = 0
                stream      = $false
            } | ConvertTo-Json -Depth 5
            $wsTurn2Resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $wsTurn2Body -UseBasicParsing
            $wsTurn2 = $wsTurn2Resp.Content | ConvertFrom-Json
            $wsWant = $wsTurn2.usage.prompt_tokens - $wsTurn1.usage.prompt_tokens
            Check ($wsTurn2Resp.StatusCode -eq 200 -and [bool]$wsTurn2.choices[0].message.content) `
                "whitespace-led reply ('$wsReply'): turn 2 returns 200 with content"
            $wsLine = Get-RequestLogLine $wsTurn2.id
            # Under --mtp/--dflash a reply cut at max_tokens can end on a DISPLAYED token that was never
            # committed (the round's last token is fed only by the next round, docs/mtp.md): a trailing
            # " " then never reached the model, the committed sequence re-tokenizes, and turn 2 correctly
            # reuses all of it -- the full-sequence path, fewer tokens prefilled than the checkpoint's.
            $wsTrailingOnly = ($wsReply -notmatch '^\s')
            if (($Mtp -gt 0 -or $Dflash -ne "") -and $wsTrailingOnly -and $wsLine -notmatch ' restore=') {
                Check ($wsTurn2.timings.prompt_n -lt $wsWant -and $wsLine -notmatch ' reset=') `
                    ("whitespace-led reply ('$wsReply', speculative, the trailing token never committed): turn 2 reused the " +
                     "whole committed sequence -- timings.prompt_n $($wsTurn2.timings.prompt_n) < $wsWant, no reset")
            } else {
                Check ($wsTurn2.timings.prompt_n -eq $wsWant) `
                    ("whitespace-led reply: turn 2 prefilled exactly the $wsWant token(s) after turn 1's prompt " +
                     "(timings.prompt_n $($wsTurn2.timings.prompt_n) of usage.prompt_tokens $($wsTurn2.usage.prompt_tokens)) -- " +
                     "continued from the prompt checkpoint, not a full re-prefill")
                Check ($wsLine -match ' restore=') "whitespace-led reply: turn 2's log line shows the checkpoint restore"
            }
            Write-Output "  [info] $wsLine"
        }
    }

    # ---- prompt checkpoint: thinking on, reasoning dropped from the replay -----------------------
    # With thinking on the prompt ends "<think>" "\n"; a client that replays the answer WITHOUT its
    # reasoning_content (most do) renders "<think>\n" + "" + "\n</think>\n\n..." -- "\n\n" is one token,
    # so the replay diverges AT turn 1's last prompt token, whatever the model said. The server
    # checkpoints one token early for such a prompt (docs/server.md "Prefix cache"), so turn 2 must
    # prefill exactly prompt_tokens(turn 2) - (prompt_tokens(turn 1) - 1) tokens and log restore=.
    # Deterministic by construction on any container: it depends only on the template, not the reply
    # (a turn 1 cut at max_tokens inside its reasoning replays an empty answer, which diverges the same way).
    $thinkKwargs = @{ enable_thinking = $true; reasoning_effort = "low" }
    $thinkQ = "What is 2 plus 2? Answer with just the number."
    $thinkT1Body = @{ messages = @(@{ role = "user"; content = $thinkQ }); max_tokens = 64; temperature = 0
                      stream = $false; chat_template_kwargs = $thinkKwargs } | ConvertTo-Json -Depth 6
    $thinkT1 = (Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $thinkT1Body -UseBasicParsing).Content | ConvertFrom-Json
    $thinkAnswer = [string]$thinkT1.choices[0].message.content
    $thinkT2Body = @{
        messages             = @(@{ role = "user"; content = $thinkQ }, @{ role = "assistant"; content = $thinkAnswer },
                                 @{ role = "user"; content = "And 3 plus 3?" })
        max_tokens           = 16
        temperature          = 0
        stream               = $false
        chat_template_kwargs = $thinkKwargs
    } | ConvertTo-Json -Depth 6
    $thinkT2Resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $thinkT2Body -UseBasicParsing
    $thinkT2 = $thinkT2Resp.Content | ConvertFrom-Json
    $thinkWant = $thinkT2.usage.prompt_tokens - ($thinkT1.usage.prompt_tokens - 1)
    Check ($thinkT2Resp.StatusCode -eq 200) "thinking, reasoning dropped: turn 2 returns 200 (turn 1 answered '$thinkAnswer')"
    Check ($thinkT2.timings.prompt_n -eq $thinkWant) `
        ("thinking, reasoning dropped: turn 2 prefilled exactly the $thinkWant token(s) from one token before turn 1's " +
         "prompt end (timings.prompt_n $($thinkT2.timings.prompt_n) of usage.prompt_tokens $($thinkT2.usage.prompt_tokens))")
    $thinkLine = Get-RequestLogLine $thinkT2.id
    Check ($thinkLine -match ' restore=') "thinking, reasoning dropped: turn 2's log line shows the checkpoint restore"
    Write-Output "  [info] $thinkLine"

    # ---- POST /v1/chat/completions, streaming (SSE) ---------------------------------------------
    # Invoke-WebRequest buffers the whole body, but that's fine here: we only need to check the
    # final SSE framing/shape, not observe incremental delivery.
    $streamBody = @{
        messages    = @(@{ role = "user"; content = "Count to three." })
        max_tokens  = 8
        temperature = 0
        stream      = $true
    } | ConvertTo-Json -Depth 5
    $streamResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $streamBody -UseBasicParsing
    Check ($streamResp.StatusCode -eq 200) "chat completion (streaming) returns 200"
    Check ($streamResp.Headers["Content-Type"] -like "text/event-stream*") "chat completion (streaming): Content-Type is text/event-stream"
    $rawEvents = $streamResp.Content -split "`n`n" | Where-Object { $_.Trim().Length -gt 0 }
    Check ($rawEvents.Count -ge 2) "chat completion (streaming): at least 2 SSE events"
    # NOTE: -like treats [...] as a wildcard character class, so a literal "[DONE]" needs a plain
    # substring check (.Contains()/.EndsWith()), not -like.
    Check ($streamResp.Content.TrimEnd().EndsWith("data: [DONE]")) "chat completion (streaming): stream ends with 'data: [DONE]'"
    $firstEvent = $rawEvents[0] -replace "^data: ", ""
    $firstChunk = $firstEvent | ConvertFrom-Json
    Check ($firstChunk.object -eq "chat.completion.chunk") "chat completion (streaming): first event object == 'chat.completion.chunk'"
    Check ($firstChunk.choices[0].delta.role -eq "assistant") "chat completion (streaming): first event delta.role == 'assistant'"
    $sawFinishReason = $false
    foreach ($ev in $rawEvents) {
        if ($ev -eq "data: [DONE]") { continue }
        $c = ($ev -replace "^data: ", "") | ConvertFrom-Json
        if ($null -ne $c.choices[0].finish_reason) { $sawFinishReason = $true }
    }
    Check $sawFinishReason "chat completion (streaming): some event carries a non-null finish_reason"
    $sawTimings = $false
    foreach ($ev in $rawEvents) {
        if ($ev -eq "data: [DONE]") { continue }
        $c = ($ev -replace "^data: ", "") | ConvertFrom-Json
        if ($null -ne $c.timings) { $sawTimings = $true }
    }
    Check $sawTimings "chat completion (streaming): the finish_reason chunk carries a timings object"

    # ---- streaming WITHOUT stream_options: no chunk may carry a "usage" key at all ---------------
    # (docs/server.md's `stream_options.include_usage` contract -- byte-for-byte unchanged apart
    # from the new `timings` key on the finish_reason chunk, checked just above).
    $noUsageKeyAnywhere = $true
    foreach ($ev in $rawEvents) {
        if ($ev -eq "data: [DONE]") { continue }
        if (($ev -replace "^data: ", "") -match '"usage"') { $noUsageKeyAnywhere = $false }
    }
    Check $noUsageKeyAnywhere "chat completion (streaming, no stream_options): no chunk has a 'usage' key"

    # ---- streaming WITH stream_options.include_usage=true -----------------------------------------
    $streamUsageBody = @{
        messages       = @(@{ role = "user"; content = "Count to three." })
        max_tokens     = 8
        temperature    = 0
        stream         = $true
        stream_options = @{ include_usage = $true }
    } | ConvertTo-Json -Depth 5
    $streamUsageResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $streamUsageBody -UseBasicParsing
    Check ($streamUsageResp.StatusCode -eq 200) "chat completion (streaming, include_usage): returns 200"
    $usageEvents = $streamUsageResp.Content -split "`n`n" | Where-Object { $_.Trim().Length -gt 0 }
    Check ($usageEvents[-1] -eq "data: [DONE]") "chat completion (streaming, include_usage): last raw event is '[DONE]'"
    $usageChunks = $usageEvents | Where-Object { $_ -ne "data: [DONE]" } | ForEach-Object { ($_ -replace "^data: ", "") | ConvertFrom-Json }
    $lastDataChunk = $usageChunks[-1]
    Check ($lastDataChunk.choices.Count -eq 0) "chat completion (streaming, include_usage): the last data chunk before [DONE] has empty choices"
    Check ($null -ne $lastDataChunk.usage -and $lastDataChunk.usage.completion_tokens -gt 0) `
        "chat completion (streaming, include_usage): the last data chunk's usage.completion_tokens > 0"
    Check ($null -ne $lastDataChunk.timings) "chat completion (streaming, include_usage): the last data chunk carries timings"
    $earlierChunksAllNullUsage = $true
    for ($i = 0; $i -lt $usageChunks.Count - 1; $i++) {
        if (-not ($usageChunks[$i].PSObject.Properties.Name -contains "usage") -or $null -ne $usageChunks[$i].usage) {
            $earlierChunksAllNullUsage = $false
        }
    }
    Check $earlierChunksAllNullUsage "chat completion (streaming, include_usage): every earlier chunk has usage == null"

    # ---- Two consecutive DIFFERENT-prompt requests must Reset(), never reload the container ------
    # docs/server.md's "Reset cost": a prefix mismatch used to pay a full Model::Load() (~18.6s
    # against the real container, and -- worse -- prints Model::Load()'s own "[r4dx::model::Model]
    # VRAM breakdown" stderr line, which only Load() ever prints, never Model::Reset()). Two
    # different-prompt requests in a row (the chat requests above already left a committed prefix
    # behind) exercise exactly the mismatch path; this checks the server's own stderr log directly
    # rather than trusting wall-clock timing alone (a loaded/quiet machine could make an 18s reload
    # look deceptively fast, but it can never fake away a SECOND "VRAM breakdown" line).
    $vramBreakdownBefore = @(Select-String -Path $ServerErrLog -Pattern "VRAM breakdown" -SimpleMatch -ErrorAction SilentlyContinue).Count

    $promptABody = @{
        messages = @(@{ role = "user"; content = "What color is the sky?" })
        max_tokens = 8; temperature = 0; stream = $false
    } | ConvertTo-Json -Depth 5
    Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $promptABody -UseBasicParsing | Out-Null

    $promptBBody = @{
        messages = @(@{ role = "user"; content = "Name a fruit that is not an apple." })
        max_tokens = 8; temperature = 0; stream = $false
    } | ConvertTo-Json -Depth 5
    $respB = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $promptBBody -UseBasicParsing
    Check ($respB.StatusCode -eq 200) "second (different-prompt) request returns 200"

    Start-Sleep -Milliseconds 500  # let the worker thread's one-line-per-request log land
    $vramBreakdownAfter = @(Select-String -Path $ServerErrLog -Pattern "VRAM breakdown" -SimpleMatch -ErrorAction SilentlyContinue).Count
    Check ($vramBreakdownAfter -eq $vramBreakdownBefore) `
        "two different-prompt requests do not reload the container (VRAM breakdown lines: before=$vramBreakdownBefore after=$vramBreakdownAfter)"

    $resetLines = @(Select-String -Path $ServerErrLog -Pattern "reset=" -SimpleMatch -ErrorAction SilentlyContinue)
    Check ($resetLines.Count -ge 1) "at least one request log line shows a cheap Model::Reset() (reset=...ms), not a reload"

    if ($Tp -eq 2) {
        $tpLines = @(Select-String -Path $ServerErrLog -Pattern " tp=2" -SimpleMatch -ErrorAction SilentlyContinue)
        Check ($tpLines.Count -ge 1) "--tp 2: the per-request log lines carry tp=2 ($($tpLines.Count) so far)"
    }

    if ($Mtp -gt 0) {
        $mtpLines = @(Select-String -Path $ServerErrLog -Pattern " mtp: " -SimpleMatch -ErrorAction SilentlyContinue)
        Check ($mtpLines.Count -ge 1) "-Mtp ${Mtp}: at least one request log line shows the MTP path was taken"
    }

    if ($Dflash -ne "") {
        $dflashLines = @(Select-String -Path $ServerErrLog -Pattern " dflash: " -SimpleMatch -ErrorAction SilentlyContinue)
        Check ($dflashLines.Count -ge 1) "-Dflash: at least one request log line shows the DFlash2 path was taken"
    }

    # ---- timings.draft_n / draft_n_accepted (docs/server.md's "timings" section) -------------------
    # Only meaningful with a speculative path enabled (-Mtp or -Dflash) and a greedy request (both
    # paths are greedy-only, engine.cpp's own use_mtp/use_dflash gate).
    if ($Mtp -gt 0 -or $Dflash -ne "") {
        $draftReqBody = @{
            messages    = @(@{ role = "user"; content = "Write one short sentence about the ocean." })
            max_tokens  = 24
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 5
        $draftResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $draftReqBody -UseBasicParsing
        $draftChat = $draftResp.Content | ConvertFrom-Json
        Check ($draftResp.StatusCode -eq 200) "speculative path: greedy request returns 200"
        Check ($draftChat.timings.draft_n -gt 0) "speculative path: timings.draft_n > 0"
        Check ($draftChat.timings.draft_n_accepted -le $draftChat.timings.draft_n) "speculative path: timings.draft_n_accepted <= timings.draft_n"
    }

    # A minimal, always-valid tool definition, shared by every tool-related check below (the live
    # streaming checks, and the thinking+tools streaming check inside the reasoning_content block).
    # Declared up here because those blocks run in a different order than they read.
    $proseTool = @{
        type     = "function"
        function = @{
            name        = "get_current_weather"
            description = "Get the current weather for a location."
            parameters  = @{
                type       = "object"
                properties = @{ location = @{ type = "string"; description = "City and state" } }
                required   = @("location")
            }
        }
    }

    # ---- Real tool call / result / answer round trip (docs/server.md's "Tool calls") -------------
    if ($ToolRoundTrip) {
        $weatherTool = @{
            type     = "function"
            function = @{
                name        = "get_current_weather"
                description = "Get the current weather for a location."
                parameters  = @{
                    type       = "object"
                    properties = @{
                        location = @{ type = "string"; description = "City and state, e.g. Boston, MA" }
                        unit     = @{ type = "string"; enum = @("celsius", "fahrenheit") }
                    }
                    required = @("location")
                }
            }
        }
        $toolReqBody = @{
            messages    = @(@{ role = "user"; content = "What is the weather like in Boston, MA right now? Use the tool." })
            tools       = @($weatherTool)
            max_tokens  = 64
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 8
        $toolResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $toolReqBody -UseBasicParsing
        $toolChat = $toolResp.Content | ConvertFrom-Json
        Check ($toolResp.StatusCode -eq 200) "tool round trip: tool-offering request returns 200"
        $gotCall = ($null -ne $toolChat.choices[0].message.tool_calls) -and ($toolChat.choices[0].message.tool_calls.Count -ge 1)
        Check $gotCall "tool round trip: model emitted at least one structured tool_calls entry"

        if ($gotCall) {
            $call = $toolChat.choices[0].message.tool_calls[0]
            Check ($call.type -eq "function") "tool round trip: tool_calls[0].type == 'function'"
            Check ($call.function.name -eq "get_current_weather") "tool round trip: tool_calls[0].function.name == 'get_current_weather'"
            Check ($call.function.arguments -is [string]) "tool round trip: tool_calls[0].function.arguments is a JSON-encoded STRING, not an object"
            Check ($toolChat.choices[0].finish_reason -eq "tool_calls") "tool round trip: finish_reason == 'tool_calls'"
            # Parse the (string) arguments to confirm they really are valid JSON, per-OpenAI-shape.
            $parsedArgs = $call.function.arguments | ConvertFrom-Json
            Check ([bool]$parsedArgs.location) "tool round trip: parsed arguments carry a 'location' field"

            # Feed the call + a synthetic tool result back as a follow-up turn -- the full
            # call/result/answer round trip this section of docs/server.md documents.
            $followUpBody = @{
                messages = @(
                    @{ role = "user"; content = "What is the weather like in Boston, MA right now? Use the tool." },
                    @{ role = "assistant"; content = $null; tool_calls = @($call) },
                    @{ role = "tool"; tool_call_id = $call.id; content = '{"temperature_f": 68, "condition": "partly cloudy"}' }
                )
                max_tokens  = 64
                temperature = 0
                stream      = $false
            } | ConvertTo-Json -Depth 8
            $followUpResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $followUpBody -UseBasicParsing
            $followUpChat = $followUpResp.Content | ConvertFrom-Json
            Check ($followUpResp.StatusCode -eq 200) "tool round trip: role:'tool' follow-up request returns 200"
            Check ([bool]$followUpChat.choices[0].message.content) "tool round trip: follow-up answer has non-empty content"
        }

        # ---- the same tool-offering turn, STREAMED (docs/server.md's "Tool calls") ---------------
        # The live gate must still deliver a complete tool_calls batch and must never let any part
        # of the `<tool_call>` markup out as a content delta on the way there.
        $toolStreamBody = @{
            messages    = @(@{ role = "user"; content = "What is the weather like in Boston, MA right now? Use the tool." })
            tools       = @($weatherTool)
            max_tokens  = 64
            temperature = 0
            stream      = $true
        } | ConvertTo-Json -Depth 8
        $toolSse = Invoke-SseStream -Uri "$BaseUrl/v1/chat/completions" -Body $toolStreamBody
        $streamedCall = $null
        $streamedToolTagLeak = $false
        $streamedToolFinish = ""
        foreach ($ev in $toolSse.Events) {
            if ($ev.Data -eq "[DONE]") { continue }
            $c = $ev.Data | ConvertFrom-Json
            if ($c.choices.Count -eq 0) { continue }
            $delta = $c.choices[0].delta
            if ($null -ne $c.choices[0].finish_reason) { $streamedToolFinish = $c.choices[0].finish_reason }
            if ($null -eq $delta) { continue }
            if (($delta.PSObject.Properties.Name -contains "content") -and $null -ne $delta.content) {
                if ($delta.content.Contains("<tool_call") -or $delta.content.Contains("</tool_call")) {
                    $streamedToolTagLeak = $true
                }
            }
            if (($delta.PSObject.Properties.Name -contains "tool_calls") -and $null -ne $delta.tool_calls) {
                $streamedCall = $delta.tool_calls[0]
            }
        }
        Check ($null -ne $streamedCall) "tool round trip (streaming): a delta carries a complete tool_calls entry"
        Check (-not $streamedToolTagLeak) "tool round trip (streaming): no content delta contains '<tool_call'/'</tool_call'"
        if ($null -ne $streamedCall) {
            Check ($streamedCall.function.name -eq "get_current_weather") `
                "tool round trip (streaming): delta.tool_calls[0].function.name == 'get_current_weather'"
            Check ($streamedCall.function.arguments -is [string]) `
                "tool round trip (streaming): delta.tool_calls[0].function.arguments is a JSON-encoded STRING"
            Check ($streamedToolFinish -eq "tool_calls") `
                "tool round trip (streaming): finish_reason == 'tool_calls' (got '$streamedToolFinish')"
        }
    }

    # ---- Thinking controls (docs/server.md's "Thinking controls" section) ------------------------
    # Every wire spelling an OpenAI-compatible client uses to turn thinking on/off. These run
    # against BOTH containers, because they do not depend on the model producing a well-formed
    # "</think>" span. The probe is the response's own `usage.completion_tokens_details` object,
    # which `BuildUsageJson` (openai_types.cpp) attaches if and only if the request's RESOLVED
    # thinking was on and OMITS entirely when it was off -- so its presence is an exact,
    # purely-HTTP read of the value these fields are supposed to move, even when the model's text
    # is nonsense. (The per-request stderr log line's `thinking=yes|no` says the same thing, but a
    # redirected stderr is block-buffered, so a line can still be sitting in the CRT's buffer when
    # the response has already come back.) The text-level consequences (a real reasoning_content
    # span / its absence) are checked against the real container further below.
    function Invoke-ThinkProbe {
        param([hashtable]$Extra)
        $body = @{
            messages    = @(@{ role = "user"; content = "Say hi." })
            max_tokens  = 4
            temperature = 0
            stream      = $false
        }
        foreach ($k in $Extra.Keys) { $body[$k] = $Extra[$k] }
        $resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body ($body | ConvertTo-Json -Depth 6) -UseBasicParsing
        $parsed = $resp.Content | ConvertFrom-Json
        $hasDetails = $null -ne $parsed.usage -and
            ($parsed.usage.PSObject.Properties.Name -contains "completion_tokens_details")
        if ($hasDetails) { "yes" } else { "no" }
    }

    # Each row: label, request body fragment, expected resolved thinking.
    $thinkForms = @(
        @{ Label = "chat_template_kwargs.enable_thinking=true";  Extra = @{ chat_template_kwargs = @{ enable_thinking = $true } };  Want = "yes" },
        @{ Label = "chat_template_kwargs.enable_thinking=false"; Extra = @{ chat_template_kwargs = @{ enable_thinking = $false } }; Want = "no"  },
        @{ Label = "enable_thinking=true";                       Extra = @{ enable_thinking = $true };                              Want = "yes" },
        @{ Label = "enable_thinking=false";                      Extra = @{ enable_thinking = $false };                             Want = "no"  },
        @{ Label = "reasoning_effort='high'";                    Extra = @{ reasoning_effort = "high" };                            Want = "yes" },
        @{ Label = "reasoning_effort='medium'";                  Extra = @{ reasoning_effort = "medium" };                          Want = "yes" },
        @{ Label = "reasoning_effort='low'";                     Extra = @{ reasoning_effort = "low" };                             Want = "yes" },
        @{ Label = "reasoning_effort='none'";                    Extra = @{ reasoning_effort = "none" };                            Want = "no"  },
        @{ Label = "reasoning={enabled:true}";                   Extra = @{ reasoning = @{ enabled = $true } };                     Want = "yes" },
        @{ Label = "reasoning={enabled:false}";                  Extra = @{ reasoning = @{ enabled = $false } };                    Want = "no"  },
        @{ Label = "reasoning={effort:'high'}";                  Extra = @{ reasoning = @{ effort = "high" } };                     Want = "yes" },
        @{ Label = "reasoning={effort:'none'}";                  Extra = @{ reasoning = @{ effort = "none" } };                     Want = "no"  },
        @{ Label = "reasoning={enabled:true,exclude:true}";      Extra = @{ reasoning = @{ enabled = $true; exclude = $true } };    Want = "yes" },
        @{ Label = "thinking={type:'enabled'}";                  Extra = @{ thinking = @{ type = "enabled" } };                     Want = "yes" },
        @{ Label = "thinking={type:'disabled'}";                 Extra = @{ thinking = @{ type = "disabled" } };                    Want = "no"  },
        @{ Label = "include_reasoning=false (text only, thinking untouched)"; Extra = @{ include_reasoning = $false }; Want = "no" },
        @{ Label = "no thinking field at all (follows --think, off here)";    Extra = @{};                             Want = "no" }
    )
    foreach ($form in $thinkForms) {
        $got = Invoke-ThinkProbe -Extra $form.Extra
        Check ($got -eq $form.Want) `
            "thinking controls: $($form.Label) resolves to thinking=$($form.Want) (got '$got')"
    }

    # Precedence: chat_template_kwargs.enable_thinking outranks every other spelling.
    $precGot = Invoke-ThinkProbe -Extra @{
        chat_template_kwargs = @{ enable_thinking = $false }
        reasoning            = @{ enabled = $true }
        thinking             = @{ type = "enabled" }
        enable_thinking      = $true
        reasoning_effort     = "high"
    }
    Check ($precGot -eq "no") `
        "thinking controls: chat_template_kwargs.enable_thinking=false beats every other field (got '$precGot')"

    # Malformed shapes are a clean 400 with the standard error body, never a silent no-op.
    $badThinkBodies = @(
        @{ Label = "enable_thinking not a boolean";  Extra = @{ enable_thinking = "yes" } },
        @{ Label = "reasoning_effort not a string";  Extra = @{ reasoning_effort = 3 } },
        @{ Label = "reasoning not an object";        Extra = @{ reasoning = "high" } },
        @{ Label = "reasoning.enabled not a boolean";Extra = @{ reasoning = @{ enabled = "yes" } } },
        @{ Label = "reasoning.max_tokens negative";  Extra = @{ reasoning = @{ max_tokens = -1 } } },
        @{ Label = "thinking not an object";         Extra = @{ thinking = "enabled" } },
        @{ Label = "thinking.type unrecognized";     Extra = @{ thinking = @{ type = "on" } } },
        @{ Label = "include_reasoning not a boolean";Extra = @{ include_reasoning = "no" } }
    )
    foreach ($bad in $badThinkBodies) {
        $body = @{ messages = @(@{ role = "user"; content = "hi" }); max_tokens = 4 }
        foreach ($k in $bad.Extra.Keys) { $body[$k] = $bad.Extra[$k] }
        try {
            Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body ($body | ConvertTo-Json -Depth 6) -UseBasicParsing | Out-Null
            Check $false "thinking controls: $($bad.Label) is rejected with 400"
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            Check ($status -eq 400) "thinking controls: $($bad.Label) is rejected with 400 (got $status)"
        }
    }

    # ---- reasoning_content (docs/server.md's "reasoning_content" section) -------------------------
    # Real-container only (-Layers -1): the 4-layer test container's nonsense output cannot reliably
    # be coaxed into emitting a well-formed "</think>" close tag, same reasoning as -ToolRoundTrip.
    if ($Layers -lt 0) {
        $thinkBody = @{
            messages             = @(@{ role = "user"; content = "What is 12 plus 30? Show your reasoning." })
            chat_template_kwargs = @{ enable_thinking = $true }
            max_tokens           = 1024
            temperature          = 0
            stream               = $false
        } | ConvertTo-Json -Depth 5
        $thinkResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $thinkBody -UseBasicParsing
        Check ($thinkResp.StatusCode -eq 200) "reasoning_content (non-streaming): request returns 200"
        $thinkChat = $thinkResp.Content | ConvertFrom-Json
        $reasoningContent = $thinkChat.choices[0].message.reasoning_content
        Check ([bool]$reasoningContent) "reasoning_content (non-streaming): message.reasoning_content is non-empty"
        $answerContent = $thinkChat.choices[0].message.content
        Check ($null -eq $answerContent -or -not $answerContent.Contains("</think>")) `
            "reasoning_content (non-streaming): message.content contains no '</think>'"
        Check ($null -eq $answerContent -or -not $answerContent.StartsWith("`n")) `
            "reasoning_content (non-streaming): message.content has no leading newline"
        Check ($thinkChat.usage.completion_tokens_details.reasoning_tokens -gt 0) `
            "reasoning_content (non-streaming): usage.completion_tokens_details.reasoning_tokens > 0"

        # ---- streaming equivalent: reasoning_content deltas, then content deltas, never both ------
        $thinkStreamBody = @{
            messages             = @(@{ role = "user"; content = "What is 12 plus 30? Show your reasoning." })
            chat_template_kwargs = @{ enable_thinking = $true }
            max_tokens           = 1024
            temperature          = 0
            stream               = $true
        } | ConvertTo-Json -Depth 5
        $thinkStreamResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $thinkStreamBody -UseBasicParsing
        Check ($thinkStreamResp.StatusCode -eq 200) "reasoning_content (streaming): request returns 200"
        $thinkStreamEvents = $thinkStreamResp.Content -split "`n`n" | Where-Object { $_.Trim().Length -gt 0 }
        $sawReasoningDelta = $false
        $sawContentDelta = $false
        $sawBothKeysInOneDelta = $false
        $sawTagAnywhere = $false
        foreach ($ev in $thinkStreamEvents) {
            if ($ev -eq "data: [DONE]") { continue }
            if ($ev.Contains("</think>")) { $sawTagAnywhere = $true }
            $c = ($ev -replace "^data: ", "") | ConvertFrom-Json
            $delta = $c.choices[0].delta
            $hasReasoning = $null -ne $delta -and ($delta.PSObject.Properties.Name -contains "reasoning_content")
            $hasContent = $null -ne $delta -and ($delta.PSObject.Properties.Name -contains "content")
            if ($hasReasoning) { $sawReasoningDelta = $true }
            if ($hasContent) { $sawContentDelta = $true }
            if ($hasReasoning -and $hasContent) { $sawBothKeysInOneDelta = $true }
        }
        Check $sawReasoningDelta "reasoning_content (streaming): at least one delta carries reasoning_content"
        Check $sawContentDelta "reasoning_content (streaming): at least one delta carries content"
        Check (-not $sawBothKeysInOneDelta) "reasoning_content (streaming): no delta carries both reasoning_content and content"
        Check (-not $sawTagAnywhere) "reasoning_content (streaming): '</think>' never appears in any event"

        # ---- thinking + tools + streaming: reasoning streams LIVE in tool mode too -----------------
        # (docs/server.md's "Tool-call mode" under reasoning_content.) This used to be delivered as
        # ONE one-shot reasoning_content delta after the whole generation was buffered; it must now
        # arrive as many per-piece deltas, exactly like the no-tools streaming path, and a
        # "</think>" must still never reach the client.
        $thinkToolStreamBody = @{
            messages             = @(@{ role = "user"; content = "What is 12 plus 30? Show your reasoning." })
            tools                = @($proseTool)
            chat_template_kwargs = @{ enable_thinking = $true }
            max_tokens           = 1024
            temperature          = 0
            stream               = $true
        } | ConvertTo-Json -Depth 8
        $thinkToolSse = Invoke-SseStream -Uri "$BaseUrl/v1/chat/completions" -Body $thinkToolStreamBody
        $ttReasoningDeltas = 0
        $ttFirstReasoningMs = -1.0
        $ttContent = ""
        $ttSawTag = $false
        foreach ($ev in $thinkToolSse.Events) {
            if ($ev.Data -eq "[DONE]") { continue }
            if ($ev.Data.Contains("</think>") -or $ev.Data.Contains("<tool_call")) { $ttSawTag = $true }
            $c = $ev.Data | ConvertFrom-Json
            if ($c.choices.Count -eq 0) { continue }
            $delta = $c.choices[0].delta
            if ($null -eq $delta) { continue }
            if (($delta.PSObject.Properties.Name -contains "reasoning_content") -and $null -ne $delta.reasoning_content) {
                $ttReasoningDeltas++
                if ($ttFirstReasoningMs -lt 0) { $ttFirstReasoningMs = $ev.Ms }
            }
            if (($delta.PSObject.Properties.Name -contains "content") -and $null -ne $delta.content) {
                $ttContent += $delta.content
            }
        }
        Check ($ttReasoningDeltas -gt 5) `
            "thinking+tools (streaming): reasoning_content arrives as many live deltas, not one lump (got $ttReasoningDeltas)"
        Check (-not $ttSawTag) "thinking+tools (streaming): no event contains '</think>' or '<tool_call'"
        Check ($ttFirstReasoningMs -ge 0 -and $ttFirstReasoningMs -lt 0.5 * $thinkToolSse.TotalMs) `
            ("thinking+tools (streaming): first reasoning delta arrives well before the end " +
             "(first=$([math]::Round($ttFirstReasoningMs,1))ms total=$([math]::Round($thinkToolSse.TotalMs,1))ms)")

        $thinkToolNonStreamBody = @{
            messages             = @(@{ role = "user"; content = "What is 12 plus 30? Show your reasoning." })
            tools                = @($proseTool)
            chat_template_kwargs = @{ enable_thinking = $true }
            max_tokens           = 1024
            temperature          = 0
            stream               = $false
        } | ConvertTo-Json -Depth 8
        $thinkToolNonStreamResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $thinkToolNonStreamBody -UseBasicParsing
        $thinkToolNonStreamChat = $thinkToolNonStreamResp.Content | ConvertFrom-Json
        $ttNonStreamContent = [string]$thinkToolNonStreamChat.choices[0].message.content
        Check ($ttContent -ceq $ttNonStreamContent) `
            ("thinking+tools: streamed content concatenation == non-streaming message.content " +
             "(streamed $(Text-Size $ttContent); non-streaming $(Text-Size $ttNonStreamContent))")

        # ---- enable_thinking=false: no reasoning_content key anywhere ------------------------------
        $noThinkBody = @{
            messages             = @(@{ role = "user"; content = "Say hi." })
            chat_template_kwargs = @{ enable_thinking = $false }
            max_tokens           = 16
            temperature          = 0
            stream               = $false
        } | ConvertTo-Json -Depth 5
        $noThinkResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $noThinkBody -UseBasicParsing
        Check ($noThinkResp.StatusCode -eq 200) "enable_thinking=false: request returns 200"
        Check (-not $noThinkResp.Content.Contains("reasoning_content")) `
            "enable_thinking=false: no reasoning_content key in the non-streaming response"

        $noThinkStreamBody = @{
            messages             = @(@{ role = "user"; content = "Say hi." })
            chat_template_kwargs = @{ enable_thinking = $false }
            max_tokens           = 16
            temperature          = 0
            stream               = $true
        } | ConvertTo-Json -Depth 5
        $noThinkStreamResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $noThinkStreamBody -UseBasicParsing
        Check (-not $noThinkStreamResp.Content.Contains("reasoning_content")) `
            "enable_thinking=false: no reasoning_content key in any streamed chunk"

        # ---- Thinking controls, text level (docs/server.md's "Thinking controls") -----------------
        # The log-line checks above proved each spelling moves the RESOLVED answer; these prove the
        # generation really follows it. The thinking-OFF forms must additionally produce content
        # byte-identical to today's `chat_template_kwargs.enable_thinking: false` output -- the
        # regression guard for "a request that does not use thinking must behave exactly as before".
        $offBaselineContent = [string]($noThinkResp.Content | ConvertFrom-Json).choices[0].message.content
        function Invoke-ThinkChat {
            param([hashtable]$Extra, [string]$Prompt = "Say hi.", [int]$MaxTokens = 16)
            $body = @{
                messages    = @(@{ role = "user"; content = $Prompt })
                max_tokens  = $MaxTokens
                temperature = 0
                stream      = $false
            }
            foreach ($k in $Extra.Keys) { $body[$k] = $Extra[$k] }
            $r = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body ($body | ConvertTo-Json -Depth 6) -UseBasicParsing
            [pscustomobject]@{ Raw = $r.Content; Json = ($r.Content | ConvertFrom-Json) }
        }

        foreach ($offForm in @(
            @{ Label = "enable_thinking=false";      Extra = @{ enable_thinking = $false } },
            @{ Label = "reasoning_effort='none'";    Extra = @{ reasoning_effort = "none" } },
            @{ Label = "reasoning={enabled:false}";  Extra = @{ reasoning = @{ enabled = $false } } },
            @{ Label = "reasoning={effort:'none'}";  Extra = @{ reasoning = @{ effort = "none" } } },
            @{ Label = "thinking={type:'disabled'}"; Extra = @{ thinking = @{ type = "disabled" } } }
        )) {
            $off = Invoke-ThinkChat -Extra $offForm.Extra
            Check (-not $off.Raw.Contains("reasoning_content")) `
                "thinking off via $($offForm.Label): no reasoning_content key anywhere"
            Check (-not $off.Raw.Contains("completion_tokens_details")) `
                "thinking off via $($offForm.Label): usage carries no completion_tokens_details"
            $offContent = [string]$off.Json.choices[0].message.content
            Check ($offContent -ceq $offBaselineContent) `
                ("thinking off via $($offForm.Label): content byte-identical to the " +
                 "chat_template_kwargs baseline")
        }

        foreach ($onForm in @(
            @{ Label = "enable_thinking=true";       Extra = @{ enable_thinking = $true } },
            @{ Label = "reasoning_effort='high'";    Extra = @{ reasoning_effort = "high" } },
            @{ Label = "reasoning={enabled:true}";   Extra = @{ reasoning = @{ enabled = $true } } },
            @{ Label = "reasoning={effort:'medium'}";Extra = @{ reasoning = @{ effort = "medium" } } },
            @{ Label = "thinking={type:'enabled'}";  Extra = @{ thinking = @{ type = "enabled" } } }
        )) {
            $on = Invoke-ThinkChat -Extra $onForm.Extra `
                -Prompt "What is 12 plus 30? Show your reasoning." -MaxTokens 1024
            Check ([bool]$on.Json.choices[0].message.reasoning_content) `
                "thinking on via $($onForm.Label): message.reasoning_content is non-empty"
            Check ($on.Json.usage.completion_tokens_details.reasoning_tokens -gt 0) `
                "thinking on via $($onForm.Label): usage.completion_tokens_details.reasoning_tokens > 0"
        }

        # ---- reasoning.exclude / include_reasoning: think, but withhold the thought ---------------
        $thinkOnBaseline = Invoke-ThinkChat -Extra @{ enable_thinking = $true } `
            -Prompt "What is 12 plus 30? Show your reasoning." -MaxTokens 1024
        $thinkOnContent = [string]$thinkOnBaseline.Json.choices[0].message.content
        foreach ($excludeForm in @(
            @{ Label = "reasoning={enabled:true,exclude:true}"; Extra = @{ reasoning = @{ enabled = $true; exclude = $true } } },
            @{ Label = "enable_thinking=true + include_reasoning=false"; Extra = @{ enable_thinking = $true; include_reasoning = $false } }
        )) {
            $ex = Invoke-ThinkChat -Extra $excludeForm.Extra `
                -Prompt "What is 12 plus 30? Show your reasoning." -MaxTokens 1024
            Check (-not $ex.Raw.Contains("reasoning_content")) `
                "$($excludeForm.Label): no reasoning_content key in the response"
            Check ($ex.Json.usage.completion_tokens_details.reasoning_tokens -gt 0) `
                "$($excludeForm.Label): thinking still happened (reasoning_tokens > 0)"
            $exContent = [string]$ex.Json.choices[0].message.content
            Check ($exContent -ceq $thinkOnContent) `
                "$($excludeForm.Label): message.content equals the same thinking-on answer"
            Check (-not $exContent.Contains("</think>")) `
                "$($excludeForm.Label): message.content contains no '</think>'"
        }

        # ---- reasoning_effort really reaches chat_template.jinja ----------------------------------
        # The template injects a DIFFERENT (or no) `reasoning_instructions` sentence per level
        # (xhigh / medium / low), so the rendered prompt length is the observable proof that the
        # mapped effort was applied rather than silently dropped.
        $effortPromptTokens = @{}
        foreach ($level in @("low", "medium", "high")) {
            $e = Invoke-ThinkChat -Extra @{ reasoning_effort = $level } -MaxTokens 4
            $effortPromptTokens[$level] = [int]$e.Json.usage.prompt_tokens
        }
        Check ($effortPromptTokens["medium"] -ne $effortPromptTokens["high"]) `
            ("reasoning_effort: 'medium' and 'high' render different prompts " +
             "($($effortPromptTokens['medium']) vs $($effortPromptTokens['high']) prompt tokens)")
        Check ($effortPromptTokens["low"] -ne $effortPromptTokens["high"]) `
            ("reasoning_effort: 'low' and 'high' render different prompts " +
             "($($effortPromptTokens['low']) vs $($effortPromptTokens['high']) prompt tokens)")
        # An unrecognized level must still answer (template default effort), never 400 out of the
        # template's own `raise_exception('Unexpected reasoning effort ...')`.
        $invented = Invoke-ThinkChat -Extra @{ reasoning_effort = "ludicrous" } -MaxTokens 4
        Check ($invented.Json.choices.Count -eq 1) `
            "reasoning_effort: an unrecognized level still answers (template default effort), no 400"
    } else {
        Write-Output "  [SKIP] reasoning_content checks (only checked against a real container, -Layers -1)"
    }

    # ---- Live tool-call streaming (docs/server.md's "Tool calls" streaming decision) -------------
    # The bug this replaced: offering ANY `tools` used to buffer the whole generation, so a client
    # that attaches a tools array to every turn (Unsloth Studio does) saw nothing at all until
    # generation finished, even for a plain prose answer that never called a tool. These checks
    # assert the three things that must now hold: content really arrives incrementally, its
    # concatenation still equals the non-streaming `message.content` byte for byte, and no
    # `<tool_call>` tag ever leaks into a content delta.
    $prosePrompt = "Write three sentences about the sea. Do not call any tool."
    $proseStreamBody = @{
        messages    = @(@{ role = "user"; content = $prosePrompt })
        tools       = @($proseTool)
        max_tokens  = 96
        temperature = 0
        stream      = $true
    } | ConvertTo-Json -Depth 8
    $proseSse = Invoke-SseStream -Uri "$BaseUrl/v1/chat/completions" -Body $proseStreamBody

    $proseContent = ""
    $proseDeltaCount = 0
    $proseFirstDeltaMs = -1.0
    $proseLastDeltaMs = -1.0
    $proseSawTag = $false
    foreach ($ev in $proseSse.Events) {
        if ($ev.Data -eq "[DONE]") { continue }
        if ($ev.Data.Contains("<tool_call") -or $ev.Data.Contains("</tool_call")) { $proseSawTag = $true }
        $c = $ev.Data | ConvertFrom-Json
        if ($c.choices.Count -eq 0) { continue }
        $delta = $c.choices[0].delta
        if ($null -eq $delta -or -not ($delta.PSObject.Properties.Name -contains "content")) { continue }
        if ($null -eq $delta.content) { continue }
        $proseContent += $delta.content
        $proseDeltaCount++
        if ($proseFirstDeltaMs -lt 0) { $proseFirstDeltaMs = $ev.Ms }
        $proseLastDeltaMs = $ev.Ms
    }
    Check ($proseDeltaCount -gt 5) `
        "live tool stream: a tools-offering prose request produced many content deltas (got $proseDeltaCount)"
    Check (-not $proseSawTag) "live tool stream: no streamed event contains '<tool_call'/'</tool_call'"
    if ($Layers -lt 0) {
        # Real container only: the 4-layer container decodes fast enough that the entire stream can
        # land in a single socket read, which would make any arrival-time assertion meaningless.
        Check ($proseFirstDeltaMs -ge 0 -and $proseFirstDeltaMs -lt 0.5 * $proseSse.TotalMs) `
            ("live tool stream: first content delta arrives well before the end " +
             "(first=$([math]::Round($proseFirstDeltaMs,1))ms last=$([math]::Round($proseLastDeltaMs,1))ms " +
             "total=$([math]::Round($proseSse.TotalMs,1))ms)")
    } else {
        Write-Output ("  [SKIP] live tool stream: first-delta arrival time (only checked against a real " +
                      "container, -Layers -1)")
    }

    # Same request, same greedy settings, non-streaming: the streamed concatenation must match it.
    $proseNonStreamBody = @{
        messages    = @(@{ role = "user"; content = $prosePrompt })
        tools       = @($proseTool)
        max_tokens  = 96
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 8
    $proseNonStreamResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $proseNonStreamBody -UseBasicParsing
    $proseNonStreamChat = $proseNonStreamResp.Content | ConvertFrom-Json
    $proseNonStreamContent = [string]$proseNonStreamChat.choices[0].message.content
    # -ceq, not -eq: PowerShell's -eq is case-INSENSITIVE on strings, which would let a byte
    # difference through on exactly the check whose whole point is byte identity.
    Check ($proseContent -ceq $proseNonStreamContent) `
        ("live tool stream: streamed content concatenation == non-streaming message.content " +
         "(streamed $(Text-Size $proseContent); non-streaming $(Text-Size $proseNonStreamContent))")

    # ---- `tools` + `stop` together must not echo the stop text back in content -------------------
    # Regression check (review finding, 2026-09-20): EmitToken only trims what it STREAMS, never
    # `accumulated` itself -- the tool_mode block used to re-derive its response straight from
    # `accumulated`, so a stop string (and anything decoded in the same token after it) leaked into
    # `message.content` whenever a request combined `tools` and `stop`. Without `tools` the content
    # is correctly "" for a stop string matching the very first token (a single space); with
    # `tools` it previously came back as the stop string itself.
    $toolsStopBody = @{
        messages    = @(@{ role = "user"; content = "Say hello." })
        tools       = @(@{ type = "function"; function = @{ name = "noop"; description = "does nothing"; parameters = @{ type = "object"; properties = @{} } } })
        stop        = @(" ")
        max_tokens  = 8
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 8
    $toolsStopResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $toolsStopBody -UseBasicParsing
    $toolsStopChat = $toolsStopResp.Content | ConvertFrom-Json
    Check ($toolsStopResp.StatusCode -eq 200) "tools+stop: request returns 200"
    $toolsStopContent = $toolsStopChat.choices[0].message.content
    Check ([string]::IsNullOrEmpty($toolsStopContent) -or -not $toolsStopContent.Contains(" ")) `
        "tools+stop: message.content does not echo the matched stop string (got '$toolsStopContent')"

    # ---- role: "function" message is accepted and rendered, not a 500 ---------------------------
    # Regression check (review finding, 2026-09-20): chat_template.jinja has NO "function" branch
    # (only system/user/assistant/tool) even though ParseChatCompletionRequest accepts role
    # "function" and docs/server.md advertises it as a supported way to carry a tool result back --
    # previously any request containing one 500'd with "ChatTemplate::render: Unexpected message
    # role." Engine::RunRequest now remaps "function" to "tool" before rendering.
    $functionRoleBody = @{
        messages = @(
            @{ role = "user"; content = "What is the weather like in Boston, MA?" },
            @{ role = "function"; name = "get_current_weather"; content = '{"temperature_f": 68}' }
        )
        max_tokens  = 8
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 5
    $functionRoleResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $functionRoleBody -UseBasicParsing
    Check ($functionRoleResp.StatusCode -eq 200) "role:'function' message renders and returns 200 (was 500)"

    # ---- messages with no user turn (only role:"tool") is a clean 400, not a 500 ----------------
    # Regression check (review finding, 2026-09-20): chat_template.jinja's own multi-step-tool scan
    # raises 'No user query found in messages.' when no message.role == "user" exists anywhere --
    # newly reachable now that tool/function roles are accepted at the request-shape level. This is
    # a caller-shape problem (bad conversation shape), so it must come back as 400
    # invalid_request_error, not fall through to the engine's generic 500 handler.
    $noUserBody = @{
        messages = @(@{ role = "tool"; tool_call_id = "call_fake"; content = "some tool result" })
        max_tokens  = 8
        temperature = 0
        stream      = $false
    } | ConvertTo-Json -Depth 5
    try {
        Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $noUserBody -UseBasicParsing | Out-Null
        Check $false "messages with no user turn returns 400 (was 500)"
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Check ($status -eq 400) "messages with no user turn returns 400, not 500 (got $status)"
    }

    # ---- images (docs/vision.md, docs/server.md's "Images") --------------------------------------
    # A remote (never-fetched) image URL is a clean 400 regardless of whether the loaded container
    # has a vision tower at all -- this validation happens at request-PARSE time, before the engine
    # ever asks the Model anything (openai_types.cpp), so it is checked unconditionally here.
    $remoteImageBody = @{
        messages = @(@{ role = "user"; content = @(@{ type = "image_url"; image_url = @{ url = "http://x" } }) })
    } | ConvertTo-Json -Depth 5
    try {
        Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $remoteImageBody -UseBasicParsing | Out-Null
        Check $false "remote image_url returns 400 (never fetched)"
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Check ($status -eq 400) "remote image_url returns 400, never fetched (got $status)"
    }

    if (-not $Vision) {
        # The default run (this container has no vision.* tensors): a well-formed LOCAL image must
        # still be rejected, cleanly, naming the real reason -- not the old blanket "not
        # implemented" 400, and never a crash or a hang trying to load a tower that isn't there.
        $localImageBody = @{
            messages = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = (Image-DataUri (New-SyntheticShapesImageBase64)) } },
                @{ type = "text"; text = "Describe this image." }
            ) })
        } | ConvertTo-Json -Depth 6
        try {
            Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $localImageBody -UseBasicParsing | Out-Null
            Check $false "image on a non-vision container returns 400 'this model/container has no vision tower'"
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            $errBody = $null
            try { $errBody = ($_.ErrorDetails.Message | ConvertFrom-Json).error.message } catch {}
            Check ($status -eq 400 -and $errBody -like "*no vision tower*") `
                ("image on a non-vision container returns a clean 400 naming the reason " +
                 "(status=$status message='$errBody')")
        }
        Write-Output "  [SKIP] full vision suite (pass -Vision with a real vision-capable container, -Layers -1)"
    } else {
        # ---- -Vision: the full image suite, against a real vision-capable container ---------------
        if ($Layers -ge 0) {
            Write-Output "  [WARN] -Vision was passed with -Layers $Layers (not -1) -- the shapes/OCR " +
                          "checks below assume a real container and may not mean much on a partial one"
        }
        $shapesImg = Image-DataUri (New-SyntheticShapesImageBase64)
        $ocrText = "R4DXVSN9"
        $ocrImg = Image-DataUri (New-OcrImageBase64 -Text $ocrText)
        # A DIFFERENT picture of the same size (a square where the circle is). Until 2026-09-26 this was a
        # second call of the same drawing, which is byte-identical to $shapesImg -- the same image key.
        $otherShapesImg = Image-DataUri (New-SyntheticShapesImageBase64 -Square)

        # ---- describe the synthetic image -----------------------------------------------------
        $describeBody = @{
            messages    = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                @{ type = "text"; text = "Describe this image in one short sentence." }
            ) })
            max_tokens  = 64
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 6
        $describeResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $describeBody -UseBasicParsing
        $describeChat = $describeResp.Content | ConvertFrom-Json
        Check ($describeResp.StatusCode -eq 200) "vision: describe request returns 200"
        Check ([bool]$describeChat.choices[0].message.content) "vision: describe response has non-empty content"
        Check ($describeChat.usage.prompt_tokens -gt 50) `
            "vision: describe usage.prompt_tokens ($($describeChat.usage.prompt_tokens)) reflects real spliced image tokens (>50)"
        Check ($describeChat.timings.image_n -eq 1) "vision: describe timings.image_n == 1"
        Check ($describeChat.timings.image_ms -gt 0) "vision: describe timings.image_ms > 0"

        # ---- OCR: the model must read the known string back exactly ----------------------------
        $ocrBody = @{
            messages    = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $ocrImg } },
                @{ type = "text"; text = "Read the text in this image and reply with exactly that text, nothing else." }
            ) })
            max_tokens  = 16
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 6
        $ocrResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $ocrBody -UseBasicParsing
        $ocrChat = $ocrResp.Content | ConvertFrom-Json
        Check ($ocrResp.StatusCode -eq 200) "vision: OCR request returns 200"
        $ocrAnswer = [string]$ocrChat.choices[0].message.content
        Check ($ocrAnswer.Contains($ocrText)) `
            "vision: OCR response contains the rendered string '$ocrText' (got '$ocrAnswer')"

        # ---- two images in one request ----------------------------------------------------------
        $twoImgBody = @{
            messages    = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                @{ type = "image_url"; image_url = @{ url = $ocrImg } },
                @{ type = "text"; text = "How many images did I just show you?" }
            ) })
            max_tokens  = 32
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 6
        $twoImgResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $twoImgBody -UseBasicParsing
        $twoImgChat = $twoImgResp.Content | ConvertFrom-Json
        Check ($twoImgResp.StatusCode -eq 200) "vision: two-images-in-one-request returns 200"
        Check ([bool]$twoImgChat.choices[0].message.content) "vision: two-images response has non-empty content"
        Check ($twoImgChat.timings.image_n -eq 2) "vision: two-images timings.image_n == 2"

        # ---- image + tools ------------------------------------------------------------------------
        $imgToolBody = @{
            messages    = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                @{ type = "text"; text = "What is the weather like in Boston, MA? Use the tool if you need to." }
            ) })
            tools       = @($proseTool)
            max_tokens  = 64
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 8
        $imgToolResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $imgToolBody -UseBasicParsing
        Check ($imgToolResp.StatusCode -eq 200) "vision: image + tools returns 200"

        # ---- image + thinking on ------------------------------------------------------------------
        $imgThinkBody = @{
            messages             = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                @{ type = "text"; text = "Describe this image." }
            ) })
            chat_template_kwargs = @{ enable_thinking = $true }
            max_tokens           = 256
            temperature          = 0
            stream               = $false
        } | ConvertTo-Json -Depth 8
        $imgThinkResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $imgThinkBody -UseBasicParsing
        $imgThinkChat = $imgThinkResp.Content | ConvertFrom-Json
        Check ($imgThinkResp.StatusCode -eq 200) "vision: image + thinking returns 200"
        Check ([bool]$imgThinkChat.choices[0].message.reasoning_content) `
            "vision: image + thinking produces non-empty reasoning_content"

        # ---- streaming -------------------------------------------------------------------------
        $imgStreamBody = @{
            messages    = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                @{ type = "text"; text = "Describe this image in one short sentence." }
            ) })
            max_tokens  = 64
            temperature = 0
            stream      = $true
        } | ConvertTo-Json -Depth 6
        $imgSse = Invoke-SseStream -Uri "$BaseUrl/v1/chat/completions" -Body $imgStreamBody
        $imgStreamedContent = ""
        foreach ($ev in $imgSse.Events) {
            if ($ev.Data -eq "[DONE]") { continue }
            $c = $ev.Data | ConvertFrom-Json
            if ($c.choices.Count -eq 0) { continue }
            $delta = $c.choices[0].delta
            if ($null -ne $delta -and ($delta.PSObject.Properties.Name -contains "content") -and $null -ne $delta.content) {
                $imgStreamedContent += $delta.content
            }
        }
        Check ([bool]$imgStreamedContent) "vision: streaming image request produces non-empty streamed content"

        # ---- multi-turn reuse: turn 2 must NOT re-encode the image ------------------------------
        # A one-word yes/no answer, not a full described sentence: this check's own point is the
        # image-aware prefix-cache mechanism (PrefixState::ImageKey), not whether an arbitrary real
        # generation round-trips byte-for-byte back through re-tokenization once replayed as a
        # plain-string `assistant` message -- a real (if narrow) risk for ANY server that matches
        # prefixes by re-tokenizing client-supplied text, independent of images entirely. The
        # "vision multi-turn (free-form)" case below is the one that replays a real, non-ASCII
        # model answer and asserts reuse still holds, so nothing here is avoided by keeping this
        # first case small -- it is only kept small so a failure points at the image mechanism.
        #
        # NB every Invoke-WebRequest in this script sends `charset=utf-8` (2026-09-22). PowerShell
        # 5.1 encodes a -Body STRING with the content type's charset, and with a bare
        # "application/json" it falls back to Latin-1: every non-ASCII character in a replayed
        # answer (the en-dashes a real description is full of) reaches the server mangled, the
        # re-rendered prompt then genuinely differs from what was committed, and the prefix is
        # correctly NOT reused. That is a CLIENT bug, and without the charset this script would
        # measure it and blame the server. The RESPONSE side had the same artifact until 2026-09-25:
        # the server's bare "application/json" made Invoke-WebRequest decode every reply as Latin-1,
        # so a replayed non-ASCII answer was mojibake of the real one, however the request was sent.
        # The server now declares charset=utf-8 on every response (docs/server.md's "Response shapes").
        $turn1Body = @{
            messages    = @(@{ role = "user"; content = @(
                @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                @{ type = "text"; text = "Is there a circle in this image? Reply with exactly one word: yes or no." }
            ) })
            max_tokens  = 4
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 6
        $turn1Resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $turn1Body -UseBasicParsing
        $turn1Chat = $turn1Resp.Content | ConvertFrom-Json
        Check ($turn1Resp.StatusCode -eq 200) "vision multi-turn: turn 1 returns 200"
        Check ($turn1Chat.timings.image_n -eq 1) "vision multi-turn: turn 1 timings.image_n == 1 (real encode)"

        $turn2Body = @{
            messages    = @(
                @{ role = "user"; content = @(
                    @{ type = "image_url"; image_url = @{ url = $shapesImg } },
                    @{ type = "text"; text = "Is there a circle in this image? Reply with exactly one word: yes or no." }
                ) },
                @{ role = "assistant"; content = $turn1Chat.choices[0].message.content },
                @{ role = "user"; content = "Now tell me what color the background gradient is." }
            )
            max_tokens  = 32
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 8
        $turn2Resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $turn2Body -UseBasicParsing
        $turn2Chat = $turn2Resp.Content | ConvertFrom-Json
        Check ($turn2Resp.StatusCode -eq 200) "vision multi-turn: turn 2 (same image, new text) returns 200"
        Check (-not ($turn2Chat.timings.PSObject.Properties.Name -contains "image_n")) `
            "vision multi-turn: turn 2 timings carries NO image_n (the image was NOT re-encoded)"
        Check ($turn2Chat.timings.prompt_n -lt $turn2Chat.usage.prompt_tokens) `
            ("vision multi-turn: turn 2 timings.prompt_n ($($turn2Chat.timings.prompt_n)) < " +
             "usage.prompt_tokens ($($turn2Chat.usage.prompt_tokens)) -- only the new tail was prefilled")
        # The 2026-09-26 failure itself: a reply with edge whitespace (" yes") re-renders trimmed, so
        # turn 2 can only continue from turn 1's prompt checkpoint (see "whitespace-led reply" above).
        $turn1Reply = [string]$turn1Chat.choices[0].message.content
        if (Test-EdgeWhitespace $turn1Reply) {
            $turn2Want = $turn2Chat.usage.prompt_tokens - $turn1Chat.usage.prompt_tokens
            Check ($turn2Chat.timings.prompt_n -eq $turn2Want -and (Get-RequestLogLine $turn2Chat.id) -match ' restore=') `
                ("vision multi-turn: turn 1's reply ('$turn1Reply') re-renders trimmed, so turn 2 continued from the " +
                 "prompt checkpoint -- prefilled exactly the $turn2Want token(s) after turn 1's prompt " +
                 "(timings.prompt_n $($turn2Chat.timings.prompt_n)) and logged restore=")
        }

        # ---- different-image-same-text: must NOT reuse the prefix -------------------------------
        # Turn 2's exact conversation with turn 1's image swapped for a DIFFERENT picture of the same
        # size ($otherShapesImg: a square where the circle is). Every image placeholder is the same
        # token id (docs/vision.md), so the TOKENS equal turn 2's and only the image key can refuse.
        # Turn 1 is re-sent first so the last committed prompt is turn 1's again: this request's tokens
        # then strictly extend it, which is exactly what the prompt checkpoint (or, for a reply that
        # re-tokenizes, the whole committed sequence) reuses when the image matches -- as turn 2 did.
        # Until 2026-09-26 this ran right after turn 2, whose committed tokens it never extended, and
        # with a byte-identical "other" image, so it could not have caught an image-key bug.
        $turn1AgainResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $turn1Body -UseBasicParsing
        $turn1Again = $turn1AgainResp.Content | ConvertFrom-Json
        Check ($turn1AgainResp.StatusCode -eq 200 -and
               [string]$turn1Again.choices[0].message.content -ceq $turn1Reply) `
            "vision multi-turn (different image): turn 1 re-sent answers the same ('$($turn1Again.choices[0].message.content)')"
        $diffImgTurn2Body = @{
            messages    = @(
                @{ role = "user"; content = @(
                    @{ type = "image_url"; image_url = @{ url = $otherShapesImg } },
                    @{ type = "text"; text = "Is there a circle in this image? Reply with exactly one word: yes or no." }
                ) },
                @{ role = "assistant"; content = $turn1Chat.choices[0].message.content },
                @{ role = "user"; content = "Now tell me what color the background gradient is." }
            )
            max_tokens  = 32
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 8
        $diffImgResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $diffImgTurn2Body -UseBasicParsing
        $diffImgChat = $diffImgResp.Content | ConvertFrom-Json
        Check ($diffImgResp.StatusCode -eq 200) "vision multi-turn (different image): request returns 200"
        Check ($diffImgChat.usage.prompt_tokens -eq $turn2Chat.usage.prompt_tokens) `
            ("vision multi-turn (different image): the same $($turn2Chat.usage.prompt_tokens) prompt tokens as turn 2 " +
             "-- only the image differs")
        Check ($diffImgChat.timings.prompt_n -eq $diffImgChat.usage.prompt_tokens) `
            ("vision multi-turn (different image): prefix NOT reused -- timings.prompt_n " +
             "($($diffImgChat.timings.prompt_n)) == usage.prompt_tokens ($($diffImgChat.usage.prompt_tokens))")
        Check ($diffImgChat.timings.image_n -ge 1) `
            "vision multi-turn (different image): the new image WAS encoded this request (image_n >= 1)"
        Check ((Get-RequestLogLine $diffImgChat.id) -notmatch ' restore=') `
            "vision multi-turn (different image): no prompt-checkpoint restore in its log line"

        # ---- multi-turn with a REAL, free-form NON-ASCII turn-1 answer --------------------------
        # Added 2026-09-22 after a review pass reported that reuse collapses whenever the replayed
        # answer carries any non-ASCII character. Part of that was the Latin-1 client-encoding
        # artifact described above (fixed). Until 2026-09-25 this case also replayed the turn-1 answer
        # as the response-side mojibake above, not the answer itself, so it could only ever take the
        # "NOT reused" branch; with the server's charset=utf-8 it replays the real text (measured:
        # TP=1 v6, 28 non-ASCII chars, REUSED). With a correctly-encoded body the underlying
        # tokenizer round-trip gap is still real and STRING-DEPENDENT: measured on this build, a
        # 25-char Japanese sentence, a Korean one, emoji and em/en dashes all round-trip and reuse,
        # while an 84-char Japanese answer and a 104-char Thai one do not. See docs/server.md's
        # "Prefix cache, image-aware".
        #
        # So this case does NOT assert reuse -- that would be a coin flip on the exact sentence the
        # model happens to produce. It asserts what must hold either way: the turn answers
        # correctly, and the two outcomes stay CONSISTENT -- either the prefix was reused and no
        # image was re-encoded, or it was not and the image was re-encoded and the whole prompt
        # re-prefilled. The silent corruption this guards against is the third combination: a
        # reused prefix whose image rows were dropped, or a re-prefill that skipped the re-encode.
        # Asking for Japanese makes turn 1 non-ASCII by construction (an unconstrained "describe
        # this image" often comes back pure ASCII for these synthetic shapes, which would let the
        # case pass without exercising anything); the check below fails if it somehow does not.
        $freeQuestion = "Describe this image in one short sentence, in Japanese."
        $freeUser = @{ role = "user"; content = @(
            @{ type = "image_url"; image_url = @{ url = $shapesImg } },
            @{ type = "text"; text = $freeQuestion }
        ) }
        $freeTurn1Body = @{ messages = @($freeUser); max_tokens = 80; temperature = 0; stream = $false } |
            ConvertTo-Json -Depth 6
        $freeTurn1 = (Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $freeTurn1Body -UseBasicParsing).Content | ConvertFrom-Json
        $freeAnswer = $freeTurn1.choices[0].message.content
        $freeNonAscii = ($freeAnswer.ToCharArray() | Where-Object { [int]$_ -gt 127 }).Count
        Check ($freeTurn1.timings.image_n -eq 1) "vision multi-turn (free-form): turn 1 encoded the image"
        Check ($freeNonAscii -gt 0) `
            "vision multi-turn (free-form): turn 1 answered with non-ASCII text ($freeNonAscii char(s)) -- the case is live"

        $freeTurn2Body = @{
            messages    = @($freeUser, @{ role = "assistant"; content = $freeAnswer },
                             @{ role = "user"; content = "Now answer with the single word OK." })
            max_tokens  = 8
            temperature = 0
            stream      = $false
        } | ConvertTo-Json -Depth 8
        $freeTurn2Resp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $freeTurn2Body -UseBasicParsing
        $freeTurn2 = $freeTurn2Resp.Content | ConvertFrom-Json
        $freeReused = $freeTurn2.timings.prompt_n -lt $freeTurn2.usage.prompt_tokens
        $freeReencoded = $freeTurn2.timings.PSObject.Properties.Name -contains "image_n"
        Check ($freeTurn2Resp.StatusCode -eq 200) `
            "vision multi-turn (free-form): turn 2 returns 200 after a $freeNonAscii-non-ASCII-char reply"
        Check ([bool]$freeTurn2.choices[0].message.content) `
            "vision multi-turn (free-form): turn 2 produced non-empty content"
        if ($freeReused) {
            Check (-not $freeReencoded) `
                ("vision multi-turn (free-form): prefix REUSED (timings.prompt_n $($freeTurn2.timings.prompt_n) " +
                 "< usage.prompt_tokens $($freeTurn2.usage.prompt_tokens)) and the image was NOT re-encoded")
        } else {
            Check ($freeReencoded -and $freeTurn2.timings.image_n -ge 1) `
                ("vision multi-turn (free-form): prefix NOT reused (this answer did not survive the " +
                 "tokenizer round trip) and the image WAS re-encoded -- full re-prefill of " +
                 "$($freeTurn2.usage.prompt_tokens) token(s), image_n=$($freeTurn2.timings.image_n)")
        }

        # ---- bad inputs: unsupported format, corrupt data, oversize, too many images -----------
        try {
            Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" `
                -Body (@{ messages = @(@{ role = "user"; content = @(
                    @{ type = "image_url"; image_url = @{ url = "data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAA" } }) }) } | ConvertTo-Json -Depth 6) `
                -UseBasicParsing | Out-Null
            Check $false "vision bad input: unsupported format (webp) returns 400"
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            Check ($status -eq 400) "vision bad input: unsupported format (webp) returns 400 (got $status)"
        }
        try {
            Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" `
                -Body (@{ messages = @(@{ role = "user"; content = @(
                    @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,dGhpcyBpcyBub3QgYSBwbmc=" } }) }) } | ConvertTo-Json -Depth 6) `
                -UseBasicParsing | Out-Null
            Check $false "vision bad input: corrupt image data returns 400"
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            Check ($status -eq 400) "vision bad input: corrupt image data returns 400 (got $status)"
        }
        try {
            $tooMany = @()
            for ($i = 0; $i -lt 9; $i++) { $tooMany += @{ type = "image_url"; image_url = @{ url = $ocrImg } } }
            Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" `
                -Body (@{ messages = @(@{ role = "user"; content = $tooMany }) } | ConvertTo-Json -Depth 6) `
                -UseBasicParsing | Out-Null
            Check $false "vision bad input: 9 images in one request returns 400 (max 8)"
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            Check ($status -eq 400) "vision bad input: 9 images in one request returns 400 (got $status)"
        }

        # ---- /v1/models advertises the vision capability now -----------------------------------
        $visionModelsResp = Invoke-WebRequest -Uri "$BaseUrl/v1/models" -UseBasicParsing
        $visionModels = $visionModelsResp.Content | ConvertFrom-Json
        Check (($visionModels.data[0].architecture.input_modalities) -contains "image") `
            "vision: /v1/models architecture.input_modalities contains 'image'"
        Check (($visionModels.data[0].capabilities) -contains "image") `
            "vision: /v1/models capabilities contains 'image'"
    }

    # ---- -Pp 2: the HYBRID MATRIX (docs/pp-tp2-hybrid.md 9 P4; see -Pp) ------------------------------------
    # Everything above ran SHORT prompts, so on the hybrid server they took the TP=2 prefill (below --pp-min-rows).
    # The cases below use prompts of >= 1024 new tokens so the pipelined prefill engages, interleaved with short ones.
    # Each case asserts the answer AND the "[stats] hybrid:" counters (Test-HybridDispatch). The hybrid's exactness
    # (state byte identity, decode equality, KL) is test_hybrid_real_identity / test_tp_hybrid_identity / kl_hybrid.ps1;
    # this is the product path: HTTP, chat template, prefix cache, checkpoints, speculative decode, images, tools.
    if ($Pp -eq 2) {
        Write-Output "[smoke] hybrid matrix (--tp 2 --pp 2, long prompts of ~$HybridFillerLines lines of synthetic log text)"
        $hyErrBefore = @(Select-String -Path $ServerErrLog -Pattern "[r4dx-server][error]" -SimpleMatch -ErrorAction SilentlyContinue).Count
        $hyBase = Get-HybridStats
        Check ($null -ne $hyBase) "hybrid: the server logs a '[stats] hybrid:' line after a request (engine.cpp)"
        if ($null -ne $hyBase) {
            Write-Output "  [info] $($hyBase.Line)"
            Check ($hyBase.Split -gt 0 -and $hyBase.S -ge 16384 -and $hyBase.MinRows -ge 1) `
                "hybrid: split k=$($hyBase.Split), stage-KV capacity S=$($hyBase.S) (>= 16384), min rows $($hyBase.MinRows)"
            Check ($hyBase.TpPrefill -ge 1) "hybrid: the ordinary checks' short prompts took the TP=2 prefill ($($hyBase.TpPrefill) TP-prefill calls, $($hyBase.BelowMin) below min rows)"
            if (-not $TpFault) {
                Check ($hyBase.Pipelined -eq 0) "hybrid: no pipelined call before the matrix (the ordinary checks are all short): $($hyBase.Pipelined)"
            } else {
                Write-Output "  [info] -TpFault: request A's long prompt ran a pipelined prefill before the fault ($($hyBase.Pipelined) pipelined call(s) so far)"
            }
        }

        $hyAsk = "Reply with one short plain ASCII sentence."
        function New-HyUser { param([int]$Seed, [string]$Ask = $hyAsk) @{ role = "user"; content = (New-HybridFiller $Seed $HybridFillerLines) + "`n" + $Ask } }
        function Get-HyReply { param($R) if ($R.Text) { $R.Text } else { "OK" } }

        # -- plain multi-turn: cold, warm, warm, short (TP), long after TP-only work (warm gather) -------------
        $u1 = New-HyUser 1
        $r1 = Invoke-HybridChat -Messages @($u1)
        Test-HybridDispatch $r1 "plain turn 1 (cold, long prompt)" -Want pipe -Cold
        Test-HybridSpec $r1 "plain turn 1"
        $a1 = Get-HyReply $r1

        $u2 = New-HyUser 2
        $r2 = Invoke-HybridChat -Messages @($u1, @{ role = "assistant"; content = $a1 }, $u2)
        Test-HybridDispatch $r2 "plain turn 2 (warm, >= 1024 new tokens)" -Want pipe -Warm
        Test-HybridSpec $r2 "plain turn 2"
        $a2 = Get-HyReply $r2

        $u3 = New-HyUser 3
        $m3 = @($u1, @{ role = "assistant"; content = $a1 }, $u2, @{ role = "assistant"; content = $a2 }, $u3)
        $r3 = Invoke-HybridChat -Messages $m3
        Test-HybridDispatch $r3 "plain turn 3 (warm, after a warm hybrid turn)" -Want pipe -Warm
        Test-HybridSpec $r3 "plain turn 3"
        $a3 = Get-HyReply $r3

        $u4 = @{ role = "user"; content = "Say OK." }
        $m4 = $m3 + @(@{ role = "assistant"; content = $a3 }, $u4)
        $r4 = Invoke-HybridChat -Messages $m4 -MaxTokens 8
        Test-HybridDispatch $r4 "plain turn 4 (short, TP=2 prefill)" -Want tp -Warm
        $a4 = Get-HyReply $r4

        $u5 = New-HyUser 5
        $m5 = $m4 + @(@{ role = "assistant"; content = $a4 }, $u5)
        $r5 = Invoke-HybridChat -Messages $m5
        Test-HybridDispatch $r5 "plain turn 5 (long, after TP-only work: decode + a short TP prefill)" -Want pipe -Warm
        Test-HybridSpec $r5 "plain turn 5"

        # -- regenerate: the same request again, twice. An identical prompt does not strictly extend the committed
        # one, so the server Reset()s and re-prefills it cold (a pipelined call at position 0): both runs are the same
        # call sequence from the same state and must answer the same. (Not compared with turn 5, whose rows came from a
        # warm gather and TP decode: that is a numerics question for the identity tests, shown here as info.)
        $rg1 = Invoke-HybridChat -Messages $m5
        $rg1Line = if ($rg1.Chat) { Get-RequestLogLine $rg1.Chat.id } else { "" }
        Test-HybridDispatch $rg1 "regenerate 1 (turn 5 again)" -Want pipe
        $rg2 = Invoke-HybridChat -Messages $m5
        Test-HybridDispatch $rg2 "regenerate 2 (turn 5 again)" -Want pipe
        Check ($rg1.Status -eq 200 -and $rg2.Status -eq 200 -and $rg1.Text -ceq $rg2.Text) `
            "hybrid: regenerate: two identical cold requests answer the same ('$($rg1.Text)' vs '$($rg2.Text)')"
        if ($rg1Line -match ' reset=') {
            Check ($null -ne $rg1.After -and $rg1.After.LastP0 -eq 0) "hybrid: regenerate 1 re-prefilled cold (log line has reset=, pipelined call at position $($rg1.After.LastP0))"
        }
        Write-Output "  [info] turn 5 answered '$($r5.Text)', its regenerate '$($rg1.Text)' ($(if ($r5.Text -ceq $rg1.Text) { 'equal' } else { 'differs: warm-gather state vs a cold prefill' }))"

        # -- thinking on: the prompt ends "<think>\n"; a replay without reasoning_content diverges at its last token, so
        # the server saves its checkpoint one token early (docs/server.md "Prefix cache") and splits the prefill in two
        # calls: the head (pipelined when long) and the one held-back token (TP). Turn 2 restores that checkpoint and
        # then runs a warm hybrid call from it.
        $thinkKw = @{ chat_template_kwargs = @{ enable_thinking = $true; reasoning_effort = "low" } }
        $tu1 = New-HyUser 11 "What is 2 plus 2? Answer with just the number."
        $tr1 = Invoke-HybridChat -Messages @($tu1) -MaxTokens 96 -Extra $thinkKw
        Test-HybridDispatch $tr1 "thinking turn 1 (cold, long)" -Want pipe -Cold -Calls 2
        $tu2 = New-HyUser 12 "And 3 plus 3? Answer with just the number."
        $tr2 = Invoke-HybridChat -Messages @($tu1, @{ role = "assistant"; content = (Get-HyReply $tr1) }, $tu2) -MaxTokens 96 -Extra $thinkKw
        Test-HybridDispatch $tr2 "thinking turn 2 (reasoning dropped from the replay: checkpoint restore, then a warm hybrid call)" -Want pipe -Warm -Calls 2
        if ($tr2.Chat) {
            Check ((Get-RequestLogLine $tr2.Chat.id) -match ' restore=') "hybrid: thinking turn 2's log line shows the checkpoint restore"
        }

        # -- a short turn right after: the TP path still works on the state a pipelined call left -----------------
        $rs = Invoke-HybridChat -Messages @(@{ role = "user"; content = "Say hello in one short sentence." }) -MaxTokens 8
        Test-HybridDispatch $rs "short cold turn after the long ones (Reset + TP prefill)" -Want tp

        # -- tool call round trip with a long context and a LONG tool result (-ToolRoundTrip, real container) --------
        if ($ToolRoundTrip) {
            $toolAsk = "What is the weather like in Boston, MA right now? Use the tool."
            $tc1 = Invoke-HybridChat -Messages @(New-HyUser 21 $toolAsk) -MaxTokens 64 -Extra @{ tools = @($proseTool) }
            Test-HybridDispatch $tc1 "tool round trip: the tool-offering turn (cold, long)" -Want pipe -Cold
            Test-HybridSpec $tc1 "tool round trip, turn 1"
            $tcalls = $null
            if ($tc1.Chat) { $tcalls = $tc1.Chat.choices[0].message.tool_calls }
            $gotHyCall = ($null -ne $tcalls) -and ($tcalls.Count -ge 1)
            Check $gotHyCall "hybrid: tool round trip: the long-context turn still emitted a structured tool_calls entry"
            if ($gotHyCall) {
                $hyCall = $tcalls[0]
                # A long result (80 forecast entries, ~2000 tokens): the follow-up is a WARM pipelined call after a
                # TP-decoded tool-call turn, with the DFlash tail on top of the drafters' decode-time state.
                $days = foreach ($d in 0..79) { '{"day":' + $d + ',"high_f":' + (60 + $d % 20) + ',"low_f":' + (40 + $d % 15) + ',"condition":"partly cloudy","wind_mph":' + ($d % 17) + '}' }
                $toolResult = '{"city":"Boston","forecast":[' + ($days -join ',') + ']}'
                $tc2 = Invoke-HybridChat -Messages @(
                    (New-HyUser 21 $toolAsk),
                    @{ role = "assistant"; content = $null; tool_calls = @($hyCall) },
                    @{ role = "tool"; tool_call_id = $hyCall.id; content = $toolResult }) -MaxTokens 64
                Test-HybridDispatch $tc2 "tool round trip: the follow-up with a long tool result (warm, pipelined)" -Want pipe -Warm
                Test-HybridSpec $tc2 "tool round trip, follow-up"
                Check ([bool]$tc2.Text) "hybrid: tool round trip: the follow-up answer has non-empty content"
            }
        } else {
            Write-Output "  [SKIP] hybrid: tool round trip with a long context (pass -ToolRoundTrip with a real container, -Layers -1)"
        }

        # -- vision (-Vision, real container): images in pipelined calls; with -Dflash the tail's rope rows -----------
        if ($Vision) {
            $hyImg = Image-DataUri (New-SyntheticShapesImageBase64)
            $hyImgPart = @{ type = "image_url"; image_url = @{ url = $hyImg } }
            function New-HyText { param([string]$Text) @{ type = "text"; text = $Text } }
            # image first, ~1500 tokens after it: the whole prompt is inside the DFlash tail window (last 2048 rows)
            $v1 = Invoke-HybridChat -Messages @(@{ role = "user"; content = @($hyImgPart, (New-HyText ((New-HybridFiller 31 $HybridFillerLines) + "`nDescribe the picture in one short sentence."))) }) -MaxTokens 48
            Test-HybridDispatch $v1 "vision: image first, long text after (cold)" -Want pipe -Cold
            Check ($v1.Chat.timings.image_n -eq 1) "hybrid: vision: the image was encoded (timings.image_n == 1)"
            Check ([bool]$v1.Text) "hybrid: vision: image-first answer has non-empty content"
            Test-HybridSpec $v1 "vision, image first"
            # ~3000 tokens of text, then the image, then a question: the image sits inside the last 2048 rows
            $v2 = Invoke-HybridChat -Messages @(@{ role = "user"; content = @((New-HyText (New-HybridFiller 32 ($HybridFillerLines * 2))), $hyImgPart, (New-HyText "What shape is in the picture? Answer with one word.")) }) -MaxTokens 16
            Test-HybridDispatch $v2 "vision: ~3000 tokens of text, then the image (the image is inside the DFlash tail window)" -Want pipe -Cold
            Check ([bool]$v2.Text) "hybrid: vision: image-after-text answer has non-empty content"
            # image first, ~3000 tokens after it: the image is OUTSIDE the tail window (the rope rows of the tail are plain text)
            $v3user = @{ role = "user"; content = @($hyImgPart, (New-HyText ((New-HybridFiller 33 ($HybridFillerLines * 2)) + "`nDescribe the picture in one short sentence."))) }
            $v3 = Invoke-HybridChat -Messages @($v3user) -MaxTokens 48
            Test-HybridDispatch $v3 "vision: image first, ~3000 tokens after it (the image is outside the DFlash tail window)" -Want pipe -Cold
            Check ([bool]$v3.Text) "hybrid: vision: image-outside-window answer has non-empty content"
            # a warm follow-up turn: the image is in the reused prefix (no re-encode), the mrope delta carries over
            # the TP decode, and the next pipelined call's tail rope rows continue from it
            $v4 = Invoke-HybridChat -Messages @($v3user, @{ role = "assistant"; content = (Get-HyReply $v3) }, (New-HyUser 34 "Now say OK.")) -MaxTokens 16
            Test-HybridDispatch $v4 "vision: warm follow-up turn after the image (>= 1024 new tokens, mrope delta carried)" -Want pipe -Warm
            if ($v4.Chat -and $v4.Chat.timings.prompt_n -lt $v4.Chat.usage.prompt_tokens) {
                Check (-not ($v4.Chat.timings.PSObject.Properties.Name -contains "image_n")) "hybrid: vision: the warm follow-up did NOT re-encode the image (no timings.image_n)"
            }
            Check ([bool]$v4.Text) "hybrid: vision: warm follow-up answer has non-empty content"
            if ($VisionStraddle) {
                # The text after the image grows by 8 tokens per step ("ok" is one token); the tail window starts where
                # n - 2048 lands, so somewhere in this sweep it starts INSIDE the image's rows (the image is ~64-81
                # tokens: k in (2048 - image_tokens, 2048)). Answers are only checked for being produced, the pipelined
                # dispatch and the counters; the byte identity of this case is test_hybrid_real_identity's job.
                $okRun = 0
                $sweepN = 0
                for ($k = 1960; $k -le 2056; $k += 8) {
                    $sw = Invoke-HybridChat -Messages @(@{ role = "user"; content = @((New-HyText ((New-HybridFiller 35 ([int]($HybridFillerLines * 0.6))) + "`nLook at the next picture.")), $hyImgPart, (New-HyText ("Describe it in one word." + (" ok" * $k)))) }) -MaxTokens 8
                    $sweepN++
                    if ($sw.Status -eq 200 -and $null -ne $sw.After -and ($sw.After.Pipelined - $sw.Before.Pipelined) -eq 1 -and [bool]$sw.Text) { $okRun++ }
                    elseif ($null -ne $sw.After) { Write-Output "  [info] straddle k=${k}: status $($sw.Status), pipelined +$($sw.After.Pipelined - $sw.Before.Pipelined), answer '$($sw.Text)'" }
                }
                Check ($okRun -eq $sweepN) "hybrid: vision straddle sweep: all $sweepN requests (text after the image 1960..2056 tokens) were pipelined and answered ($okRun ok)"
            }
        } else {
            Write-Output "  [SKIP] hybrid: vision rows through the pipelined call (pass -Vision with a real vision-capable container)"
        }

        # -- the matrix left the server healthy ---------------------------------------------------------------------
        $hyFinal = Get-HybridStats
        if ($null -ne $hyFinal) {
            Write-Output "  [info] $($hyFinal.Line)"
            Check ($hyFinal.Pipelined -ge 1) "hybrid: the matrix ran $($hyFinal.Pipelined) pipelined and $($hyFinal.TpPrefill) TP-prefill calls in all"
        }
        $hyErrAfter = @(Select-String -Path $ServerErrLog -Pattern "[r4dx-server][error]" -SimpleMatch -ErrorAction SilentlyContinue).Count
        Check ($hyErrAfter -eq $hyErrBefore) "hybrid: no '[r4dx-server][error]' line during the matrix ($hyErrBefore before, $hyErrAfter after)"
        Check (-not $proc.HasExited) "hybrid: the server is still running after the matrix"
    }

    # ---- Sampled speculative decode (Milestone 6 stage S3, docs/sampling.md section 9/10): lifting
    # the temperature<=0 gate means a seeded, TEMPERATURE>0 request on a speculative-enabled server
    # must (a) still take the speculative path (timings.draft_n > 0), (b) be reproducible (same
    # seed -> identical text on repeat), and -- real container only, same gating reason as
    # -ToolRoundTrip/reasoning_content above (the 4-layer test container's MTP head never accepts a
    # draft at all, task's own container note, so a speculative-vs-plain text comparison there would
    # be meaningless) -- (c) emit the SAME text a plain (no --mtp/--dflash) sampled r4dx-cli run with
    # the identical seed/sampling flags does, per docs/sampling.md's losslessness gate
    # (tools/validate_spec_sampling.ps1 is the exhaustive version of this same check across layouts/
    # prompts/configs/seeds; this is the one-shot smoke-test confirmation that the SERVER's own
    # request path -- chat template render, prefix reuse, EmitToken -- doesn't undo it).
    #
    # Deliberately the LAST check in this script (checks (a)/(b) above are not, but (c) below stops
    # the server to run the CLI comparison, per the project's "never run two of your own GPU
    # processes at once" rule -- restarting the server afterward would cost a second full container
    # load for no coverage this stage's task asks for, so nothing here may depend on the server
    # being alive afterward).
    if ($Mtp -gt 0 -or $Dflash -ne "") {
        $seededSampledBody = @{
            messages    = @(@{ role = "user"; content = "Write one short sentence about the ocean." })
            max_tokens  = 24
            temperature = 0.7
            top_k       = 20
            top_p       = 0.8
            seed        = 12345
            stream      = $false
        } | ConvertTo-Json -Depth 5
        $seededResp1 = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $seededSampledBody -UseBasicParsing
        $seededChat1 = $seededResp1.Content | ConvertFrom-Json
        Check ($seededResp1.StatusCode -eq 200) "sampled speculative path: seeded temperature=0.7 request returns 200"
        Check ($seededChat1.timings.draft_n -gt 0) "sampled speculative path: timings.draft_n > 0"
        Check ($seededChat1.timings.draft_n_accepted -le $seededChat1.timings.draft_n) `
            "sampled speculative path: timings.draft_n_accepted <= timings.draft_n"

        $seededResp2 = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $seededSampledBody -UseBasicParsing
        $seededChat2 = $seededResp2.Content | ConvertFrom-Json
        Check ($seededResp2.StatusCode -eq 200) "sampled speculative path: repeat of the same seeded request returns 200"
        Check ($seededChat1.choices[0].message.content -eq $seededChat2.choices[0].message.content) `
            "sampled speculative path: same seed -> identical text on repeat ('$($seededChat1.choices[0].message.content)')"

        if ($Layers -lt 0) {
            Write-Output "[smoke] stopping server (pid $($proc.Id)) to run a same-seeded CLI comparison (project rule: one GPU process at a time)"
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                $proc.WaitForExit(10000) | Out-Null
            }
            $cliExe = Join-Path $RepoRoot "build\$Preset\src\cli\r4dx-cli.exe"
            # -Tp 2: the same --tp 2 --tp-mode as the server ($TpArgs), so both sides run one engine.
            $cliBaseArgs = @(
                "--model", $Model, "--layout", $Layout, "--max-tokens", 24, "--max-ctx", $MaxCtx,
                "--temperature", 0.7, "--top-k", 20, "--top-p", 0.8, "--seed", 12345,
                "--prompt", "Write one short sentence about the ocean."
            )
            $cliArgs = $cliBaseArgs + $TpArgs
            if ($UsesDevice0) {
                # The in-run watch targets the server's pid, which is gone: end it here (a TDR it saw
                # already stopped the server and left $TdrMarker). The CLI run is short (one load plus
                # 24 tokens); the finally's 30 s wait and tdr_check -Since $RunStart cover it, and its
                # stderr joins the HIP error 719 scan there.
                if ($null -ne $tdrJob) {
                    Stop-Job $tdrJob -ErrorAction SilentlyContinue
                    Remove-Job $tdrJob -Force -ErrorAction SilentlyContinue
                    $tdrJob = $null
                }
                if (Test-Path -LiteralPath $TdrMarker) { throw "the in-run TDR watch saw a TDR -- not starting the r4dx-cli comparison run" }
                # First TDR stops device-0 work: check before the next device-0 process starts.
                & powershell -NoProfile -ExecutionPolicy Bypass -File $TdrCheck -Since $RunStart.ToString('yyyy-MM-ddTHH:mm:ss') -Quiet | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "TDR since $RunStart -- not starting the r4dx-cli comparison run" }
            }
            $prevPref = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                if ($Tp -eq 2) {
                    $cliOut = ((& $cliExe @cliArgs 2>$CliErrLog) -join "`n").Trim()
                } else {
                    $cliOut = ((& $cliExe @cliArgs 2>$null) -join "`n").Trim()
                }
            } finally {
                $ErrorActionPreference = $prevPref
            }
            $serverText = $seededChat1.choices[0].message.content.Trim()
            Check ($cliOut -eq $serverText) `
                ("sampled speculative path: server text matches a same-seeded plain sampled r4dx-cli " +
                 "run (server='$serverText' cli='$cliOut')")
            if ($Tp -eq 2 -and $cliOut -ne $serverText) {
                # A -Tp 2 mismatch alone does not say TP caused it: seeded speculative text can still
                # differ from plain at --tp 1 once a verify window straddles a change of the attention
                # split's segment width (docs/mtp.md "Sampled rounds are bit-exact"; before 2026-09-25
                # it differed far more often, docs/tp.md Appendix B N29). The
                # control: the same seeded request through r4dx-cli at --tp 1 on device 1, speculative
                # and plain. It changes no check -- it says which class the FAIL above is in (N80).
                $env:HIP_VISIBLE_DEVICES = "1"
                $specArgs = if ($Mtp -gt 0) { @("--mtp", "$Mtp") } else { @("--dflash", $Dflash) }
                $prevPref = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                try {
                    $tp1Plain = ((& $cliExe @cliBaseArgs 2>$null) -join "`n").Trim()
                    $tp1Spec = ((& $cliExe @cliBaseArgs @specArgs 2>$null) -join "`n").Trim()
                } finally {
                    $ErrorActionPreference = $prevPref
                }
                if ($tp1Spec -ne $tp1Plain) {
                    Write-Output ("  [info] TP=1 control: speculative and plain differ at --tp 1 too" +
                                  $(if ($tp1Spec -eq $serverText -and $tp1Plain -eq $cliOut) { ", with exactly the --tp 2 texts" } else { "" }) +
                                  " -- the pre-existing class (N29 / batched verify), not a TP regression by itself " +
                                  "(tp1 spec='$tp1Spec' tp1 plain='$tp1Plain')")
                } else {
                    Write-Output ("  [info] TP=1 control: speculative == plain at --tp 1 ('$tp1Plain') -- the --tp 2 " +
                                  "mismatch is TP-specific: investigate")
                }
            }
        } else {
            Write-Output ("  [SKIP] sampled speculative path: text matches a same-seeded CLI plain sampled run " +
                          "(only checked against a real container, -Layers -1 -- the 4-layer test container's " +
                          "MTP head never accepts a draft at all, task's own container note)")
        }
    }

} finally {
    # The environment restore sits in its own finally, so no failure in the cleanup above it can skip
    # it; $proc / $tdrJob are null when the try failed before starting them (N80).
    try {
        if ($null -ne $proc) {
            Write-Output "[smoke] stopping server (pid $($proc.Id))"
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                $proc.WaitForExit(30000) | Out-Null
            }
        }
        # ---- -Tp 2 real: the device-0 TDR check (tools\tp\tdr_check.ps1, Appendix B N55/N64) ------
        if ($null -ne $tdrJob) {
            Stop-Job $tdrJob -ErrorAction SilentlyContinue
            Remove-Job $tdrJob -Force -ErrorAction SilentlyContinue
        }
        if ($UsesDevice0) {
            if (Test-Path -LiteralPath $TdrMarker) {
                Get-Content -LiteralPath $TdrMarker | ForEach-Object { Write-Output "  $_" }
                Check $false "tp: the in-run TDR watch saw no TDR (it stopped the server, see above)"
            }
            Write-Output ("[smoke] device 0 in use since $($RunStart.ToString('yyyy-MM-dd HH:mm:ss')): waiting 30 s for " +
                          "Windows Error Reporting, then tdr_check")
            Start-Sleep -Seconds 30
            $tdrOut = & powershell -NoProfile -ExecutionPolicy Bypass -File $TdrCheck -Since $RunStart.ToString('yyyy-MM-ddTHH:mm:ss') -Quiet
            $tdrRc = $LASTEXITCODE
            foreach ($l in @($tdrOut)) { Write-Output "  $l" }
            Check ($tdrRc -eq 0) "tp: no TDR since the run started (tdr_check -Since $($RunStart.ToString('yyyy-MM-dd HH:mm:ss')))"
            # The P3 TDR reached HIP as error 719 (N44): such an error in the server's log, or in the
            # r4dx-cli comparison run's (when there was one), is a suspected TDR.
            $scanLogs = @(@($ServerErrLog, $CliErrLog) | Where-Object { Test-Path -LiteralPath $_ })
            $hip719 = @()
            if ($scanLogs.Count -gt 0) {
                $hip719 = @(Select-String -LiteralPath $scanLogs -Pattern 'HIP error 719\b|unspecified launch failure')
            }
            Check ($hip719.Count -eq 0) ("tp: the server log" + $(if ($scanLogs.Count -gt 1) { " and the r4dx-cli run's" } else { "" }) +
                                         " report no HIP error 719 / unspecified launch failure (a suspected TDR)")
        }
    } finally {
        if ($null -ne $SavedHipVisible) { $env:HIP_VISIBLE_DEVICES = $SavedHipVisible }
        else { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
        if ($null -ne $SavedTpFault) { $env:R4DX_TP_FAULT = $SavedTpFault }
        else { Remove-Item env:R4DX_TP_FAULT -ErrorAction SilentlyContinue }
    }
}

Write-Output ""
if ($script:Failures -gt 0) {
    Write-Output "[smoke] $($script:Failures) check(s) FAILED"
    exit 1
}
Write-Output "[smoke] all checks passed"
exit 0
