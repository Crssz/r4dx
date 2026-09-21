#requires -Version 5.1
<#
.SYNOPSIS
  Integration smoke test for r4dx-server: starts the server, hits /v1/models and a non-streaming
  and a streaming /v1/chat/completions, checks the JSON/SSE shapes.

.DESCRIPTION
  Only HIP device 1 may be used (project GPU rule) -- sets HIP_VISIBLE_DEVICES=1 before starting
  r4dx-server.exe. Defaults to the 4-layer test container (D:\models\r4dx\qwen38-27b-l4-bf16.r4dx,
  --layout w4a16) -- that model's text is nonsense (4 of 64 layers, arbitrary quantized-layout
  weights on a model that was never actually trained/converted for real use at 4 layers), so this
  script only checks response/SSE *shapes* and token counts, never the generated text itself.
  Pass -Model/-Layout to point at the real 64-layer container instead for a real-answer smoke run
  (docs/server.md's "Real-answer smoke run" section records one such run's output).

.PARAMETER Model
  Path to a .r4dx container. Default: the 4-layer test container.

.PARAMETER Layout
  Body layout. Default: w4a16.

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
  e.g. D:\models\r4dx\qwen38-27b-l4-mtp.r4dx (4-layer test container) or the real 64-layer
  container with -Mtp 3 (this stage's own required verification runs).

.PARAMETER Dflash
  Path to a DFlash2 draft container, passed to r4dx-server's --dflash (docs/dflash2.md, Milestone 5
  stage S3 item 7). Empty (default) disables it. Mutually exclusive with -Mtp > 0 (server_args.h's
  own check rejects both at once) -- needs -Model pointed at a container with enough layers to
  cover the draft's own target_layers (the real 64-layer container for every shipped DFlash2
  container, whose target_layers reach layer 62).

.PARAMETER ToolRoundTrip
  Exercises a real tool call/result/answer multi-turn round trip (docs/server.md's "Tool calls"):
  offers a `get_current_weather` tool definition, sends the server's own parsed
  `message.tool_calls` back as a `role: "tool"` follow-up message, and checks the final answer
  comes back 200 with real prose. Off by default -- the 4-layer test container's nonsense output
  cannot reliably be coaxed into emitting a well-formed `<tool_call>` block, so this only produces
  a meaningful check against a real container (pass -Layers -1 with a real -Model).

.EXAMPLE
  .\tools\server\smoke.ps1
.EXAMPLE
  .\tools\server\smoke.ps1 -Model D:\models\r4dx\qwen38-27b.r4dx -Layout w4a16 -Layers -1
.EXAMPLE
  .\tools\server\smoke.ps1 -Model D:\models\r4dx\qwen38-27b-l4-mtp.r4dx -Layout w4a16 -Mtp 3
.EXAMPLE
  .\tools\server\smoke.ps1 -Model D:\models\r4dx\qwen38-27b.r4dx -Layout w4a16 -Layers -1 -Mtp 3
.EXAMPLE
  .\tools\server\smoke.ps1 -Model D:\models\r4dx\qwen38-27b-v3.r4dx -Layout w4a16 -Layers -1 -ToolRoundTrip
#>
[CmdletBinding()]
param(
    [string]$Model = "D:\models\r4dx\qwen38-27b-l4-bf16.r4dx",
    [string]$Layout = "w4a16",
    [int]$Port = 8091,
    [int]$Layers = 4,
    [int]$Mtp = 0,
    [string]$Dflash = "",
    [switch]$ToolRoundTrip,
    [string]$Preset = "win-hip"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..

$RepoRoot = (Get-Location).Path
$ServerExe = Join-Path $RepoRoot "build\$Preset\src\server\r4dx-server.exe"
if (-not (Test-Path $ServerExe)) { throw "r4dx-server.exe not found at $ServerExe -- run .\build.ps1 first" }
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

Write-Output "[smoke] starting r4dx-server: model=$Model layout=$Layout port=$Port mtp=$Mtp"
$env:HIP_VISIBLE_DEVICES = "1"
$ServerErrLog = "$env:TEMP\r4dx-server-smoke.err.log"
# 4096 (raised from 512, model-metadata/reasoning_content pass): the reasoning_content checks below
# send `max_tokens: 1024` on the real container so a real thinking span has room to close and still
# leave room for an answer -- 512 was too tight once thinking is on.
$MaxCtx = 4096
$ServerArgList = @(
    "--model", $Model, "--layout", $Layout, "--host", "127.0.0.1", "--port", "$Port",
    "--max-ctx", "$MaxCtx", "--max-tokens-default", "16"
)
if ($Layers -ge 0) { $ServerArgList += @("--layers", "$Layers") }
if ($Mtp -gt 0) { $ServerArgList += @("--mtp", "$Mtp") }
if ($Dflash -ne "") { $ServerArgList += @("--dflash", "$Dflash") }
$proc = Start-Process -FilePath $ServerExe -ArgumentList $ServerArgList -PassThru `
  -RedirectStandardError $ServerErrLog `
  -RedirectStandardOutput "$env:TEMP\r4dx-server-smoke.out.log"

try {
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
        -ContentType "application/json" -Body $chatBody -UseBasicParsing
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
        -ContentType "application/json" -Body $prefixReuseBody1 -UseBasicParsing
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
        -ContentType "application/json" -Body $prefixReuseBody2 -UseBasicParsing
    $prefixReuseChat2 = $prefixReuseResp2.Content | ConvertFrom-Json
    Check ($prefixReuseResp2.StatusCode -eq 200) "prefix reuse: extending-conversation request returns 200"
    if ($Layers -lt 0) {
        Check ($prefixReuseChat2.timings.prompt_n -lt $prefixReuseChat2.usage.prompt_tokens) `
            "prefix reuse: second request's timings.prompt_n ($($prefixReuseChat2.timings.prompt_n)) < usage.prompt_tokens ($($prefixReuseChat2.usage.prompt_tokens))"
    } else {
        Write-Output "  [SKIP] prefix reuse: prompt_n < usage.prompt_tokens (only checked against a real container, -Layers -1)"
    }

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
        -ContentType "application/json" -Body $streamBody -UseBasicParsing
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
        -ContentType "application/json" -Body $streamUsageBody -UseBasicParsing
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
        -ContentType "application/json" -Body $promptABody -UseBasicParsing | Out-Null

    $promptBBody = @{
        messages = @(@{ role = "user"; content = "Name a fruit that is not an apple." })
        max_tokens = 8; temperature = 0; stream = $false
    } | ConvertTo-Json -Depth 5
    $respB = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -ContentType "application/json" -Body $promptBBody -UseBasicParsing
    Check ($respB.StatusCode -eq 200) "second (different-prompt) request returns 200"

    Start-Sleep -Milliseconds 500  # let the worker thread's one-line-per-request log land
    $vramBreakdownAfter = @(Select-String -Path $ServerErrLog -Pattern "VRAM breakdown" -SimpleMatch -ErrorAction SilentlyContinue).Count
    Check ($vramBreakdownAfter -eq $vramBreakdownBefore) `
        "two different-prompt requests do not reload the container (VRAM breakdown lines: before=$vramBreakdownBefore after=$vramBreakdownAfter)"

    $resetLines = @(Select-String -Path $ServerErrLog -Pattern "reset=" -SimpleMatch -ErrorAction SilentlyContinue)
    Check ($resetLines.Count -ge 1) "at least one request log line shows a cheap Model::Reset() (reset=...ms), not a reload"

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
            -ContentType "application/json" -Body $draftReqBody -UseBasicParsing
        $draftChat = $draftResp.Content | ConvertFrom-Json
        Check ($draftResp.StatusCode -eq 200) "speculative path: greedy request returns 200"
        Check ($draftChat.timings.draft_n -gt 0) "speculative path: timings.draft_n > 0"
        Check ($draftChat.timings.draft_n_accepted -le $draftChat.timings.draft_n) "speculative path: timings.draft_n_accepted <= timings.draft_n"
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
            -ContentType "application/json" -Body $toolReqBody -UseBasicParsing
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
                -ContentType "application/json" -Body $followUpBody -UseBasicParsing
            $followUpChat = $followUpResp.Content | ConvertFrom-Json
            Check ($followUpResp.StatusCode -eq 200) "tool round trip: role:'tool' follow-up request returns 200"
            Check ([bool]$followUpChat.choices[0].message.content) "tool round trip: follow-up answer has non-empty content"
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
            -ContentType "application/json" -Body $thinkBody -UseBasicParsing
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
            -ContentType "application/json" -Body $thinkStreamBody -UseBasicParsing
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

        # ---- enable_thinking=false: no reasoning_content key anywhere ------------------------------
        $noThinkBody = @{
            messages             = @(@{ role = "user"; content = "Say hi." })
            chat_template_kwargs = @{ enable_thinking = $false }
            max_tokens           = 16
            temperature          = 0
            stream               = $false
        } | ConvertTo-Json -Depth 5
        $noThinkResp = Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json" -Body $noThinkBody -UseBasicParsing
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
            -ContentType "application/json" -Body $noThinkStreamBody -UseBasicParsing
        Check (-not $noThinkStreamResp.Content.Contains("reasoning_content")) `
            "enable_thinking=false: no reasoning_content key in any streamed chunk"
    } else {
        Write-Output "  [SKIP] reasoning_content checks (only checked against a real container, -Layers -1)"
    }

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
        -ContentType "application/json" -Body $toolsStopBody -UseBasicParsing
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
        -ContentType "application/json" -Body $functionRoleBody -UseBasicParsing
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
            -ContentType "application/json" -Body $noUserBody -UseBasicParsing | Out-Null
        Check $false "messages with no user turn returns 400 (was 500)"
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Check ($status -eq 400) "messages with no user turn returns 400, not 500 (got $status)"
    }

    # ---- 400 on an unsupported (image) content part ----------------------------------------------
    $badBody = @{
        messages = @(@{ role = "user"; content = @(@{ type = "image_url"; image_url = @{ url = "http://x" } }) })
    } | ConvertTo-Json -Depth 5
    try {
        Invoke-WebRequest -Uri "$BaseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json" -Body $badBody -UseBasicParsing | Out-Null
        Check $false "image content part returns 400"
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Check ($status -eq 400) "image content part returns 400 (got $status)"
    }

} finally {
    Write-Output "[smoke] stopping server (pid $($proc.Id))"
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Output ""
if ($script:Failures -gt 0) {
    Write-Output "[smoke] $($script:Failures) check(s) FAILED"
    exit 1
}
Write-Output "[smoke] all checks passed"
exit 0
