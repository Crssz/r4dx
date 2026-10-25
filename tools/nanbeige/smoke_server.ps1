#requires -Version 5.1
<#
.SYNOPSIS
  Server smoke test for Nanbeige4.3-33B (docs/nanbeige-plan.md N3-4): r4dx-server on the w8a16 container at --tp 2,
  the nanbeige dialect end to end over HTTP.

.DESCRIPTION
  Starts r4dx-server (both GPUs, --tp 2 --layout w8a16), waits for /health, then checks, all greedy:
    1. /health 200; /v1/models lists the model with n_ctx_train 524288.
    2. Chat, thinking on (default): reasoning_content non-empty, content non-empty, no "<think>" / "</think>" text
       left in either; finish_reason stop; usage.prompt_tokens > 0.
    3. Chat, thinking off (chat_template_kwargs.enable_thinking=false): no reasoning_content, a content answer.
    4. Streaming == non-streaming (same request, greedy): the concatenated reasoning / content deltas equal the
       non-streamed message, no delta carries both keys, a [DONE] terminator.
    5. Chinese: the answer is CJK text (UTF-8 intact), streamed == non-streamed.
    6. Tool call (XML <tool_call><function=..><parameter=..>): a get_weather call with a city argument,
       finish_reason tool_calls, arguments a JSON string; the round trip with a role:"tool" result answers in content.
    7. Prefix cache: turn 2 of a conversation prefills fewer tokens than its prompt (timings.prompt_n < usage).
    8. Raw /v1/completions: a continuation of a plain-text prompt (the dialect's BOS on raw prompts).
  Writes the server log next to the script's temp files and prints [PASS] / [FAIL] lines; exits 1 on any failure.
  Uses device 0 (the desktop card): run on an idle machine.

.EXAMPLE
  .\tools\nanbeige\smoke_server.ps1
#>
[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Preset = "win-hip",
    [int]$Port = 8093,
    [int]$MaxCtx = 8192
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..
$RepoRoot = (Get-Location).Path
$ServerExe = Join-Path $RepoRoot "build\$Preset\src\server\r4dx-server.exe"
if (-not (Test-Path $ServerExe)) { throw "r4dx-server.exe not found at $ServerExe -- run .\build.ps1 first" }
if (-not $Model) {
    $root = if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { "E:\models" }
    $Model = Join-Path $root "r4dx\nanbeige\nanbeige43-33b-w8a16-g128.r4dx"
}
if (-not (Test-Path $Model)) { throw "model container not found: $Model" }

$BaseUrl = "http://127.0.0.1:$Port"
$ErrLog = Join-Path $env:TEMP "r4dx-nanbeige-smoke.err.log"
$OutLog = Join-Path $env:TEMP "r4dx-nanbeige-smoke.out.log"
$script:Failures = 0
function Check {
    param([bool]$Condition, [string]$Message)
    if ($Condition) { Write-Output "  [PASS] $Message" } else { Write-Output "  [FAIL] $Message"; $script:Failures++ }
}

# POST JSON, UTF-8 both ways. Returns the parsed response object (non-streaming) or the SSE data lines (streaming).
function Invoke-Post {
    param([string]$Path, [hashtable]$Body, [switch]$Stream)
    $json = $Body | ConvertTo-Json -Depth 10 -Compress
    $req = [System.Net.HttpWebRequest]::Create("$BaseUrl$Path")
    $req.Method = "POST"
    $req.ContentType = "application/json; charset=utf-8"
    $req.Proxy = $null
    $req.Timeout = 600000
    $req.ReadWriteTimeout = 600000
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $req.ContentLength = $bytes.Length
    $s = $req.GetRequestStream(); $s.Write($bytes, 0, $bytes.Length); $s.Close()
    $resp = $req.GetResponse()
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
    try {
        if ($Stream) {
            $lines = New-Object System.Collections.ArrayList
            while (-not $reader.EndOfStream) {
                $l = $reader.ReadLine()
                if ($l -like "data: *") { [void]$lines.Add($l.Substring(6)) }
            }
            return , $lines
        }
        return ($reader.ReadToEnd() | ConvertFrom-Json)
    } finally { $reader.Close(); $resp.Close() }
}

# The streamed deltas of a chat completion folded into {reasoning, content, both_keys, done, finish}.
function Fold-Stream {
    param($Lines)
    $r = New-Object System.Text.StringBuilder; $c = New-Object System.Text.StringBuilder
    $both = $false; $done = $false; $finish = ""
    foreach ($l in $Lines) {
        if ($l -eq "[DONE]") { $done = $true; continue }
        $ev = $l | ConvertFrom-Json
        foreach ($ch in $ev.choices) {
            $d = $ch.delta
            $hasR = $null -ne $d.reasoning_content; $hasC = $null -ne $d.content
            if ($hasR -and $hasC) { $both = $true }
            if ($hasR) { [void]$r.Append($d.reasoning_content) }
            if ($hasC) { [void]$c.Append($d.content) }
            if ($ch.finish_reason) { $finish = $ch.finish_reason }
        }
    }
    [pscustomobject]@{ Reasoning = $r.ToString(); Content = $c.ToString(); BothKeys = $both; Done = $done; Finish = $finish }
}

function Has-Cjk { param([string]$t) return [regex]::IsMatch($t, '[\u4e00-\u9fff]') }

$proc = $null
try {
    Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $args = @("--model", $Model, "--layout", "w8a16", "--tp", "2", "--tp-mode", "real", "--host", "127.0.0.1",
              "--port", "$Port", "--max-ctx", "$MaxCtx", "--dialect", "nanbeige")
    Write-Output "[smoke] starting r4dx-server: $Model --tp 2 --layout w8a16 --max-ctx $MaxCtx port $Port"
    $proc = Start-Process -FilePath $ServerExe -ArgumentList $args -PassThru -RedirectStandardError $ErrLog -RedirectStandardOutput $OutLog
    $ready = $false
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) { Get-Content $ErrLog -Tail 20; throw "r4dx-server exited early (code $($proc.ExitCode))" }
        try { if ((Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ready = $true; break } } catch { }
    }
    if (-not $ready) { throw "server did not become ready" }
    Write-Output "[smoke] server ready"

    # 1. models
    $models = (Invoke-WebRequest -Uri "$BaseUrl/v1/models" -UseBasicParsing).Content | ConvertFrom-Json
    Check ($models.data.Count -ge 1) "/v1/models lists a model"
    Check ($models.data[0].meta.n_ctx_train -eq 524288) "/v1/models: n_ctx_train 524288 (from the container config; got $($models.data[0].meta.n_ctx_train))"

    # 2. thinking on (r4dx-server defaults to thinking off unless --think on or the request asks, as for Qwen)
    $q = @{ messages = @(@{ role = "user"; content = "What is 17 * 23? Answer briefly." }); max_tokens = 600; temperature = 0;
            chat_template_kwargs = @{ enable_thinking = $true } }
    $a = Invoke-Post "/v1/chat/completions" $q
    $m = $a.choices[0].message
    Check ([bool]$m.reasoning_content) "thinking on: reasoning_content is non-empty"
    Check ([bool]$m.content -and $m.content.Contains("391")) "thinking on: the answer contains 391 (got: $($m.content))"
    Check (-not ($m.content + $m.reasoning_content).Contains("<think>") -and -not ($m.content + $m.reasoning_content).Contains("</think>")) "thinking on: no <think> markup left in the message"
    Check ($a.choices[0].finish_reason -eq "stop") "thinking on: finish_reason stop"
    Check ($a.usage.prompt_tokens -gt 0 -and $a.usage.completion_tokens -gt 0) "thinking on: usage counted ($($a.usage.prompt_tokens) prompt, $($a.usage.completion_tokens) completion)"

    # 3. thinking off
    $qOff = @{ messages = $q.messages; max_tokens = 200; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } }
    # (thinking off is also the server default: a request with no enable_thinking at all must match it)
    $b = Invoke-Post "/v1/chat/completions" $qOff
    Check (-not [bool]$b.choices[0].message.reasoning_content) "thinking off: no reasoning_content"
    Check ($b.choices[0].message.content -and $b.choices[0].message.content.Contains("391")) "thinking off: the answer contains 391 (got: $($b.choices[0].message.content))"

    # 4. streaming == non-streaming
    $qs = $q.Clone(); $qs.stream = $true
    $f = Fold-Stream (Invoke-Post "/v1/chat/completions" $qs -Stream)
    Check ($f.Done) "streaming: [DONE] terminator"
    Check (-not $f.BothKeys) "streaming: no delta carries both reasoning_content and content"
    # Trimmed, as tools/server/smoke.ps1's LFM2 suite compares: the stream sends whitespace before "</think>" live (it
    # cannot know yet that it is trailing), the non-streamed message trims it.
    $streamSame = $f.Reasoning.Trim() -ceq ([string]$m.reasoning_content).Trim() -and $f.Content.Trim() -ceq ([string]$m.content).Trim()
    Check $streamSame "streaming: concatenated deltas == the non-streamed message (trimmed)"
    if (-not $streamSame) {
        Write-Output ("    stream reasoning: " + ($f.Reasoning | ConvertTo-Json) + "`n    plain  reasoning: " + ([string]$m.reasoning_content | ConvertTo-Json))
        Write-Output ("    stream content:   " + ($f.Content | ConvertTo-Json) + "`n    plain  content:   " + ([string]$m.content | ConvertTo-Json))
    }
    Check ($f.Finish -eq "stop") "streaming: finish_reason stop"

    # 5. Chinese
    # Code points, so the script stays ASCII (Windows PowerShell 5.1 reads a BOM-less script as ANSI):
    # "Introduce the Great Wall in one sentence." in Chinese.
    $zh = -join ([char[]](0x7528, 0x4E00, 0x53E5, 0x8BDD, 0x4ECB, 0x7ECD, 0x957F, 0x57CE, 0x3002))
    $qz = @{ messages = @(@{ role = "user"; content = $zh }); max_tokens = 600; temperature = 0 }
    $z = Invoke-Post "/v1/chat/completions" $qz
    Check (Has-Cjk $z.choices[0].message.content) "Chinese: the answer is CJK text (got: $($z.choices[0].message.content))"
    $qzs = $qz.Clone(); $qzs.stream = $true
    $fz = Fold-Stream (Invoke-Post "/v1/chat/completions" $qzs -Stream)
    Check ($fz.Content -eq $z.choices[0].message.content) "Chinese: streamed == non-streamed (UTF-8 intact across deltas)"

    # 6. tool call round trip (XML)
    $weather = @{ type = "function"; function = @{ name = "get_weather"; description = "Get the current weather for a city.";
        parameters = @{ type = "object"; properties = @{ city = @{ type = "string"; description = "City name" } }; required = @("city") } } }
    $qt = @{ messages = @(@{ role = "user"; content = "What is the weather in Beijing right now? Use the tool." }); tools = @($weather);
             max_tokens = 800; temperature = 0 }
    $t = Invoke-Post "/v1/chat/completions" $qt
    $calls = $t.choices[0].message.tool_calls
    Check ($null -ne $calls -and $calls.Count -ge 1) "tool call: at least one structured tool_calls entry"
    if ($null -ne $calls -and $calls.Count -ge 1) {
        $call = $calls[0]
        Check ($call.function.name -eq "get_weather") "tool call: function.name get_weather"
        Check ($call.function.arguments -is [string]) "tool call: arguments is a JSON string"
        $parsed = $call.function.arguments | ConvertFrom-Json
        Check ([bool]$parsed.city) "tool call: arguments carry city ($($parsed.city))"
        Check ($t.choices[0].finish_reason -eq "tool_calls") "tool call: finish_reason tool_calls"
        Check (-not ([string]$t.choices[0].message.content).Contains("<tool_call>")) "tool call: no <tool_call> markup in content"
        $qf = @{ messages = @($qt.messages[0], @{ role = "assistant"; content = ""; tool_calls = @($call) },
                              @{ role = "tool"; tool_call_id = $call.id; content = '{"temp_c": 21, "condition": "sunny"}' });
                 tools = @($weather); max_tokens = 600; temperature = 0 }
        $tf = Invoke-Post "/v1/chat/completions" $qf
        Check ([bool]$tf.choices[0].message.content) "tool call: the follow-up after the tool result answers in content ($($tf.choices[0].message.content))"
    }

    # 7. prefix cache across turns
    $t1 = @{ messages = @(@{ role = "user"; content = "Name three primary colors." }); max_tokens = 400; temperature = 0 }
    $r1 = Invoke-Post "/v1/chat/completions" $t1
    $t2 = @{ messages = @($t1.messages[0], @{ role = "assistant"; content = $r1.choices[0].message.content },
                          @{ role = "user"; content = "And three secondary colors?" }); max_tokens = 400; temperature = 0 }
    $r2 = Invoke-Post "/v1/chat/completions" $t2
    Check ($r2.timings.prompt_n -lt $r2.usage.prompt_tokens) "prefix cache: turn 2 prefilled $($r2.timings.prompt_n) of $($r2.usage.prompt_tokens) prompt tokens"

    # 8. raw completions
    $c = Invoke-Post "/v1/completions" @{ prompt = "The capital of France is"; max_tokens = 8; temperature = 0 }
    Check ($c.choices[0].text -match "Paris") "raw /v1/completions: continues with Paris (got: $($c.choices[0].text))"
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
Write-Output ""
Write-Output "[smoke] server log: $ErrLog"
if ($script:Failures -gt 0) { Write-Output "[smoke] $($script:Failures) check(s) FAILED"; exit 1 }
Write-Output "[smoke] all checks passed"
