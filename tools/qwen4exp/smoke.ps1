#requires -Version 5.1
<#
.SYNOPSIS
  Server smoke of a qwen4_exp container: starts r4dx-server (both cards) and checks health/models, chat, thinking efforts, SSE
  ordering, a tool-call round trip, multi-turn prefix reuse (warm vs cold text), a ~20k-token needle (QSA is active beyond
  2051 tokens), the over-max-ctx 400, back-to-back requests and a clean shutdown. PASS/FAIL table, exit code != 0 on any failure.

.DESCRIPTION
  The WHOLE smoke runs under ONE `tools\with_lock.ps1 -Name gpu` acquisition (this script re-launches itself with -Inner under the
  lock; lock wait timeout 240 min). The smoke itself (load excluded) is meant to take < 15 min. Both cards are visible
  (HIP_VISIBLE_DEVICES removed), R4DX_Q4_DEVICES / R4DX_Q4_SPLIT are set from -Devices / -Split.
  Server command: r4dx-server --model <Container> --layout trellis --tokenizer-dir <TokenizerDir> --host 127.0.0.1 --port <Port> --max-ctx <MaxCtx>
  Prefix reuse is read from the response's own `timings.prompt_n` (tokens actually prefilled) against `usage.prompt_tokens`
  (docs/server.md "timings"). Shutdown is a Ctrl-C delivered to the server's console (its SIGINT handler); a forced kill is a FAIL.
  -DryRun prints the lock command, the server command and every request of every check (the response-dependent fields show
  placeholders) and starts nothing.

.PARAMETER Container   the .r4dx container (required)
.PARAMETER Layout      default trellis
.PARAMETER Port        default 8080
.PARAMETER Devices     R4DX_Q4_DEVICES "a,b", default 0,1
.PARAMETER Split       R4DX_Q4_SPLIT (0 = the model's choice)
.PARAMETER MaxCtx      server --max-ctx, default 32768 (the needle prompt is ~20k tokens; the 400 check needs a prompt beyond it)
.PARAMETER TokenizerDir default E:\models\Huihui-Qwen3.8-27B-abliterated
.PARAMETER LoadTimeoutSec  wait for /health, default 1500
.PARAMETER ServerExe   another r4dx-server.exe (default build\win-hip\src\server\r4dx-server.exe)
.EXAMPLE
  .\tools\qwen4exp\smoke.ps1 -Container E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx -Port 8080
.EXAMPLE
  .\tools\qwen4exp\smoke.ps1 -Container E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Container,
    [string]$Layout = "trellis",
    [int]$Port = 8080,
    [string]$Devices = "0,1",
    [int]$Split = 0,
    [int]$MaxCtx = 32768,
    [string]$TokenizerDir = "E:\models\Huihui-Qwen3.8-27B-abliterated",
    [int]$LoadTimeoutSec = 1500,
    [int]$LockTimeoutMinutes = 240,
    [string]$ServerExe = "",
    [string]$DryRunArg = "false",
    [string]$Inner = "false"
)
# DryRun arrives as a string ("false" is truthy in `$DryRun -or ...`): keep a real boolean for the whole script.
$DryRun = ($DryRunArg -eq "true")
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
if ($Devices -notmatch '^\d+,\d+$') { throw "-Devices must be 'a,b' (HIP ordinals), got '$Devices'" }
if (-not $ServerExe) { $ServerExe = Join-Path $Repo "build\win-hip\src\server\r4dx-server.exe" }
$BaseUrl = "http://127.0.0.1:$Port"
$ErrLog = Join-Path $env:TEMP "r4dx-q4exp-smoke.err.log"
$OutLog = Join-Path $env:TEMP "r4dx-q4exp-smoke.out.log"
$ServerArgs = @("--model", $Container, "--layout", $Layout, "--tokenizer-dir", $TokenizerDir, "--host", "127.0.0.1", "--port", "$Port", "--max-ctx", "$MaxCtx")

# ---- outer: take the GPU lock once for everything ------------------------------------------------------------------------------
if ($Inner -ne "true" -and $DryRun -ne "true") {
    if (-not (Test-Path -LiteralPath $Container)) { throw "container not found: $Container" }
    if (-not (Test-Path -LiteralPath $ServerExe)) { throw "r4dx-server.exe not found at $ServerExe (build the r4dx-server target)" }
    $q = { param($s) if ($s -match '\s') { '\"' + $s + '\"' } else { $s } }
    $inner = "powershell -NoProfile -ExecutionPolicy Bypass -File $(& $q $PSCommandPath) -Container $(& $q $Container) -Layout $Layout -Port $Port -Devices $Devices -Split $Split -MaxCtx $MaxCtx -TokenizerDir $(& $q $TokenizerDir) -LoadTimeoutSec $LoadTimeoutSec -ServerExe $(& $q $ServerExe) -Inner true"
    Write-Output "[smoke] waiting for the gpu lock (up to $LockTimeoutMinutes min); the whole smoke runs inside one acquisition"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Repo "tools\with_lock.ps1") -Name gpu -TimeoutMinutes $LockTimeoutMinutes -Run $inner
    exit $LASTEXITCODE
}

# ---- helpers -----------------------------------------------------------------------------------------------------------------------
$script:Results = New-Object System.Collections.Generic.List[object]
function Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    $script:Results.Add([pscustomobject]@{ Name = $Name; Ok = $Ok; Detail = $Detail })
    if ($DryRun -eq "true") { return }
    Write-Output ("  [{0}] {1}{2}" -f $(if ($Ok) { "PASS" } else { "FAIL" }), $Name, $(if ($Detail) { "  -- $Detail" } else { "" }))
}
function Section { param([string]$T) Write-Output ""; Write-Output "[smoke] $T" }
function Short { param([string]$S, [int]$N = 100) if ($null -eq $S) { return "" }; $t = ($S -replace '\s+', ' ').Trim(); if ($t.Length -gt $N) { $t.Substring(0, $N) + "..." } else { $t } }

$FakeChat = @'
{"id":"chatcmpl-<id>","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"<assistant reply of this request>","reasoning_content":"<reasoning>","tool_calls":[{"id":"<call id>","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\":\"Paris\"}"}}]}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"completion_tokens_details":{"reasoning_tokens":0}},"timings":{"prompt_n":0,"predicted_n":0}}
'@
$FakeModels = '{"data":[{"id":"<model id>","context_length":0}]}'
$FakeStream = "data: {`"choices`":[{`"delta`":{`"role`":`"assistant`"}}]}`n`ndata: [DONE]`n`n"

# One HTTP call. Body is sent as raw UTF-8 bytes (docs/server.md: a PowerShell string body can be re-encoded). Returns Status/Text/Json/Headers/Sec.
function Invoke-Api {
    param([string]$Path, $Body = $null, [string]$Method = "POST", [int]$TimeoutSec = 900, [string]$Label = "")
    $json = if ($null -ne $Body) { ConvertTo-Json -InputObject $Body -Depth 16 -Compress } else { "" }
    if ($DryRun -eq "true") {
        $shown = if ($json.Length -gt 700) { $json.Substring(0, 500) + " ...[$($json.Length) chars total]... " + $json.Substring($json.Length - 150) } else { $json }
        Write-Host "  > $Method $BaseUrl$Path   # $Label"
        if ($shown) { Write-Host "    $shown" }
        $txt = if ($Path -eq "/v1/models") { $FakeModels } elseif ($Body -and $Body.stream) { $FakeStream } else { $FakeChat }
        return [pscustomobject]@{ Status = 200; Text = $txt; Json = ($(if ($txt.StartsWith("{")) { $txt | ConvertFrom-Json } else { $null })); Headers = @{}; Sec = 0.0 }
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $status = 0; $text = ""; $headers = @{}
    try {
        $args2 = @{ Uri = "$BaseUrl$Path"; Method = $Method; UseBasicParsing = $true; TimeoutSec = $TimeoutSec }
        if ($null -ne $Body) { $args2.ContentType = "application/json; charset=utf-8"; $args2.Body = [Text.Encoding]::UTF8.GetBytes($json) }
        $resp = Invoke-WebRequest @args2
        $status = [int]$resp.StatusCode
        $text = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray())
        $headers = $resp.Headers
    } catch {
        if ($null -eq $_.Exception.Response) { throw }
        $status = [int]$_.Exception.Response.StatusCode
        $text = "$($_.ErrorDetails.Message)"
        if (-not $text) { try { $text = (New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())).ReadToEnd() } catch { } }
    }
    $obj = $null
    if ($text.TrimStart().StartsWith("{")) { try { $obj = $text | ConvertFrom-Json } catch { } }
    [pscustomobject]@{ Status = $status; Text = $text; Json = $obj; Headers = $headers; Sec = $sw.Elapsed.TotalSeconds }
}
function Chat {
    param([object[]]$Messages, [hashtable]$Extra = @{}, [int]$MaxTokens = 64, [string]$Label = "", [int]$TimeoutSec = 900)
    $b = [ordered]@{ messages = $Messages; max_tokens = $MaxTokens; temperature = 0 }
    foreach ($k in $Extra.Keys) { $b[$k] = $Extra[$k] }
    Invoke-Api "/v1/chat/completions" $b -Label $Label -TimeoutSec $TimeoutSec
}
function Msg { param($R) if ($R.Json -and $R.Json.choices) { $R.Json.choices[0].message } else { $null } }
function Text { param($R) $m = Msg $R; if ($m -and $null -ne $m.content) { [string]$m.content } else { "" } }
function Reasoning { param($R) $m = Msg $R; if ($m -and $m.PSObject.Properties["reasoning_content"] -and $null -ne $m.reasoning_content) { [string]$m.reasoning_content } else { "" } }
function User { param([string]$T) @{ role = "user"; content = $T } }
$NoThink = @{ chat_template_kwargs = @{ enable_thinking = $false } }

# Filler lines of ~35 tokens each (digits are single tokens): deterministic, every line different.
function New-Filler {
    param([int]$Lines, [int]$NeedleAt = -1, [string]$Needle = "")
    $sb = New-Object Text.StringBuilder
    for ($i = 0; $i -lt $Lines; $i++) {
        if ($i -eq $NeedleAt) { [void]$sb.Append($Needle + "`n") }
        $n = 5000 + $i
        [void]$sb.Append("Log entry $n : the lighthouse keeper recorded wind $(($n * 7) % 90) knots, tide $(($n * 3) % 12) feet and visibility $(($n % 9) + 1) miles at dock $(($n * 11) % 40).`n")
    }
    $sb.ToString()
}

# Ctrl-C into the server's (hidden) console from a throwaway process, so this process keeps its own console.
$CtrlCSource = @'
using System; using System.Runtime.InteropServices;
public static class CtrlC {
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool AttachConsole(uint pid);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool FreeConsole();
  [DllImport("kernel32.dll")] static extern bool SetConsoleCtrlHandler(IntPtr h, bool add);
  [DllImport("kernel32.dll")] static extern bool GenerateConsoleCtrlEvent(uint ev, uint grp);
  // A parent started with CREATE_NEW_PROCESS_GROUP (CI runners, agent harnesses) has Ctrl-C ignored and its children inherit that:
  // clear it in THIS process before the server is started from it.
  public static void Enable() { SetConsoleCtrlHandler(IntPtr.Zero, false); }
  public static int Send(uint pid) {
    FreeConsole();
    if (!AttachConsole(pid)) return Marshal.GetLastWin32Error();
    SetConsoleCtrlHandler(IntPtr.Zero, true);
    bool ok = GenerateConsoleCtrlEvent(0, 0);
    System.Threading.Thread.Sleep(300);
    FreeConsole();
    return ok ? 0 : -1;
  }
}
'@
function Send-CtrlC {
    param([int]$ProcId)
    $cmd = "Add-Type -TypeDefinition @'`n$CtrlCSource`n'@; exit [CtrlC]::Send($ProcId)"
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
    $p = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $enc -Wait -PassThru -WindowStyle Hidden
    $p.ExitCode
}

# ---- plan banner ---------------------------------------------------------------------------------------------------------------------
$lockRun = "powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name gpu -TimeoutMinutes $LockTimeoutMinutes -Run `"powershell -NoProfile -ExecutionPolicy Bypass -File tools\qwen4exp\smoke.ps1 -Container $Container ... -Inner`""
Write-Output "[smoke] container=$Container layout=$Layout port=$Port devices=$Devices split=$Split max-ctx=$MaxCtx"
if ($DryRun -eq "true") {
    Write-Output "[smoke] DRY RUN: nothing is started. Plan:"
    Write-Output "[smoke] 0. lock (one acquisition for the whole smoke): $lockRun"
    Write-Output "[smoke] 1. env: HIP_VISIBLE_DEVICES removed (both cards), R4DX_Q4_DEVICES=$Devices$(if ($Split) { ", R4DX_Q4_SPLIT=$Split" })"
    Write-Output "[smoke] 2. server: $ServerExe $($ServerArgs -join ' ')   (hidden window; stderr -> $ErrLog, stdout -> $OutLog; wait <= $LoadTimeoutSec s for GET /health)"
} else {
    if (-not (Test-Path -LiteralPath $ServerExe)) { throw "r4dx-server.exe not found at $ServerExe" }
    try { Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing -TimeoutSec 2 | Out-Null; throw "something already answers on $BaseUrl" } catch { if ("$_" -like "something already*") { throw } }
}

# ---- run -----------------------------------------------------------------------------------------------------------------------------
$saved = @{ HIP = $env:HIP_VISIBLE_DEVICES; DEV = $env:R4DX_Q4_DEVICES; SPL = $env:R4DX_Q4_SPLIT }
$proc = $null
$t0 = Get-Date
try {
    if ($DryRun -ne "true") {
        Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        $env:R4DX_Q4_DEVICES = $Devices
        if ($Split) { $env:R4DX_Q4_SPLIT = "$Split" } else { Remove-Item Env:R4DX_Q4_SPLIT -ErrorAction SilentlyContinue }
        Remove-Item $ErrLog, $OutLog -ErrorAction SilentlyContinue
        Add-Type -TypeDefinition $CtrlCSource
        [CtrlC]::Enable()
        Write-Output "[smoke] starting r4dx-server (load can take minutes)"
        $proc = Start-Process -FilePath $ServerExe -ArgumentList $ServerArgs -PassThru -WindowStyle Hidden -RedirectStandardError $ErrLog -RedirectStandardOutput $OutLog
        $null = $proc.Handle   # keeps the handle so ExitCode is readable after WaitForExit
        $ready = $false
        for ($i = 0; $i -lt $LoadTimeoutSec; $i++) {
            Start-Sleep -Seconds 1
            if ($proc.HasExited) { Write-Output "[smoke] server exited early (code $($proc.ExitCode)); stderr tail:"; Get-Content $ErrLog -Tail 30 -ErrorAction SilentlyContinue; throw "r4dx-server exited before becoming ready" }
            try { if ((Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ready = $true; break } } catch { }
        }
        if (-not $ready) { throw "server not ready within $LoadTimeoutSec s" }
        Write-Output ("[smoke] server ready after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)
        foreach ($l in @(Select-String -Path $ErrLog -Pattern "dialect:|stage \d|VRAM|listening" -ErrorAction SilentlyContinue)) { Write-Output "  [info] $($l.Line)" }
    }
    $tChecks = Get-Date

    Section "1. health / models"
    $h = Invoke-Api "/health" -Method GET -Label "health"
    Check "GET /health is 200" ($h.Status -eq 200) "status $($h.Status)"
    $mdl = Invoke-Api "/v1/models" -Method GET -Label "models"
    $mid = if ($mdl.Json) { [string]$mdl.Json.data[0].id } else { "" }
    Check "GET /v1/models returns a model id" ($mdl.Status -eq 200 -and $mid.Length -gt 0) "id '$mid'"
    Check "/v1/models context_length == --max-ctx" ($DryRun -or [int]$mdl.Json.data[0].context_length -eq $MaxCtx) "context_length $($mdl.Json.data[0].context_length), --max-ctx $MaxCtx"

    Section "2. chat completion, thinking off"
    $france = Chat @(User "What is the capital of France?") $NoThink 64 "France, thinking off, temperature 0"
    $fText = Text $france
    Check "chat: 200 and finish_reason stop|length" ($france.Status -eq 200 -and $france.Json.choices[0].finish_reason -in @("stop", "length")) "status $($france.Status), finish_reason $($france.Json.choices[0].finish_reason)"
    Check "chat: non-empty English text naming Paris" ($fText.Trim().Length -gt 0 -and $fText -match "Paris" -and (($fText -replace '[^\x00-\x7F]', '').Length -ge 0.9 * $fText.Length)) "'$(Short $fText)'"
    Check "chat: thinking off -> no reasoning_content" ((Reasoning $france).Length -eq 0)

    Section "3. thinking on, reasoning_effort low / medium / xhigh"
    $ptok = @{}
    foreach ($e in @(@("low", 768), @("medium", 1024), @("xhigh", 1536))) {
        $r = Chat @(User "What is 17 times 23? Answer briefly.") @{ reasoning_effort = $e[0] } $e[1] "thinking on, effort $($e[0])"
        $rc = Reasoning $r
        $ptok[$e[0]] = [int]$r.Json.usage.prompt_tokens
        Check "effort $($e[0]): 200 and reasoning_content present" ($r.Status -eq 200 -and $rc.Trim().Length -gt 0) "reasoning $($rc.Length) chars, finish $($r.Json.choices[0].finish_reason), completion_tokens $($r.Json.usage.completion_tokens)"
        Check "effort $($e[0]): usage.completion_tokens_details.reasoning_tokens > 0" ($DryRun -or [int]$r.Json.usage.completion_tokens_details.reasoning_tokens -gt 0) "reasoning_tokens $($r.Json.usage.completion_tokens_details.reasoning_tokens)"
        if ($r.Json.choices[0].finish_reason -eq "stop") { Check "effort $($e[0]): the answer (391) follows the reasoning" ((Text $r) -match "391") "'$(Short (Text $r))'" }
    }
    Check "effort reaches the template (low and xhigh render different prompts)" ($DryRun -or $ptok["low"] -ne $ptok["xhigh"]) "prompt_tokens low $($ptok['low']), medium $($ptok['medium']), xhigh $($ptok['xhigh'])"
    $off = Chat @(User "What is 17 times 23? Answer briefly.") @{ reasoning_effort = "none" } 64 "reasoning_effort none = thinking off"
    Check "reasoning_effort none: no reasoning_content" ($off.Status -eq 200 -and (Reasoning $off).Length -eq 0) "'$(Short (Text $off))'"

    Section "4. streaming SSE ordering"
    $sb = [ordered]@{ messages = @(User "What is the capital of France?"); max_tokens = 64; temperature = 0; stream = $true; stream_options = @{ include_usage = $true }; chat_template_kwargs = @{ enable_thinking = $false } }
    $s = Invoke-Api "/v1/chat/completions" $sb -Label "stream, include_usage"
    function Parse-Sse { param([string]$T)
        $items = @(); $done = $false; $afterDone = $false
        foreach ($line in ($T -split "`n")) {
            $l = $line.TrimEnd("`r")
            if (-not $l.StartsWith("data:")) { continue }
            $d = $l.Substring(5).Trim()
            if ($done) { $afterDone = $true }
            if ($d -eq "[DONE]") { $done = $true; $items += [pscustomobject]@{ Done = $true; J = $null }; continue }
            $items += [pscustomobject]@{ Done = $false; J = ($d | ConvertFrom-Json) }
        }
        [pscustomobject]@{ Items = $items; Done = $done; AfterDone = $afterDone }
    }
    $p = Parse-Sse $s.Text
    $chunks = @($p.Items | Where-Object { -not $_.Done })
    $contentIdx = @(); $finishIdx = @(); $usageIdx = @(); $content = ""
    for ($i = 0; $i -lt $chunks.Count; $i++) {
        $j = $chunks[$i].J
        if ($j.choices -and $j.choices.Count -gt 0) {
            $c = $j.choices[0]
            if ($c.delta -and $c.delta.content) { $contentIdx += $i; $content += [string]$c.delta.content }
            if ($c.finish_reason) { $finishIdx += $i }
        } elseif ($j.usage) { $usageIdx += $i }
    }
    $ctype = "$($s.Headers['Content-Type'])"
    Check "stream: 200, Content-Type text/event-stream" ($DryRun -or ($s.Status -eq 200 -and $ctype -like "text/event-stream*")) "status $($s.Status), '$ctype'"
    Check "stream: first chunk carries delta.role = assistant" ($DryRun -or ($chunks.Count -gt 0 -and $chunks[0].J.choices[0].delta.role -eq "assistant"))
    Check "stream: content deltas, then exactly one finish_reason chunk after them" ($DryRun -or ($contentIdx.Count -gt 0 -and $finishIdx.Count -eq 1 -and $finishIdx[0] -ge $contentIdx[-1])) "content chunks $($contentIdx.Count), finish chunks $($finishIdx.Count)"
    Check "stream: include_usage chunk (choices [], usage) after the finish chunk, then [DONE] last" ($DryRun -or ($usageIdx.Count -eq 1 -and $finishIdx.Count -eq 1 -and $usageIdx[0] -gt $finishIdx[0] -and $p.Done -and -not $p.AfterDone -and $p.Items[-1].Done)) "usage chunks $($usageIdx.Count), [DONE] $($p.Done)"
    Check "stream: concatenated deltas == the non-streaming text of the same request" ($DryRun -or $content -ceq $fText) "stream '$(Short $content 60)' vs '$(Short $fText 60)'"
    $st = $chunks | Where-Object { $_.J.choices -and $_.J.choices.Count -gt 0 -and $_.J.choices[0].finish_reason } | Select-Object -First 1
    Check "stream: finish chunk carries timings" ($DryRun -or ($null -ne $st -and $null -ne $st.J.timings.prompt_n))
    $tb = [ordered]@{ messages = @(User "What is 17 times 23? Answer briefly."); max_tokens = 512; temperature = 0; stream = $true; reasoning_effort = "low" }
    $ts = Invoke-Api "/v1/chat/completions" $tb -Label "stream, thinking on (low)"
    $tp = Parse-Sse $ts.Text
    $firstR = -1; $firstC = -1; $lastR = -1; $k = 0
    foreach ($it in ($tp.Items | Where-Object { -not $_.Done })) {
        if ($it.J.choices -and $it.J.choices.Count -gt 0 -and $it.J.choices[0].delta) {
            $d = $it.J.choices[0].delta
            if ($d.reasoning_content) { if ($firstR -lt 0) { $firstR = $k }; $lastR = $k }
            if ($d.content) { if ($firstC -lt 0) { $firstC = $k } }
        }
        $k++
    }
    Check "stream, thinking on: reasoning_content deltas arrive live, all before the first content delta" ($DryRun -or ($firstR -ge 0 -and ($firstC -lt 0 -or $lastR -lt $firstC))) "first reasoning chunk $firstR, last reasoning $lastR, first content $firstC"

    Section "5. tool call round trip"
    $tools = @(@{ type = "function"; function = @{ name = "get_current_weather"; description = "Get the current weather in a given city";
        parameters = @{ type = "object"; properties = @{ location = @{ type = "string"; description = "City name, e.g. Paris" }; unit = @{ type = "string"; enum = @("celsius", "fahrenheit") } }; required = @("location") } } })
    $q1 = @(User "What is the weather in Paris right now? Use the tool.")
    $t1 = Chat $q1 (@{ tools = $tools; tool_choice = "auto" } + $NoThink) 256 "tool offered"
    $calls = @(); if ($t1.Json -and (Msg $t1).tool_calls) { $calls = @((Msg $t1).tool_calls) }
    Check "tool: finish_reason tool_calls with one parsed call to get_current_weather" ($t1.Status -eq 200 -and $t1.Json.choices[0].finish_reason -eq "tool_calls" -and $calls.Count -ge 1 -and $calls[0].function.name -eq "get_current_weather") "finish $($t1.Json.choices[0].finish_reason), calls $($calls.Count), text '$(Short (Text $t1))'"
    $argOk = $false; $argObj = $null
    if ($calls.Count -ge 1) { try { $argObj = [string]$calls[0].function.arguments | ConvertFrom-Json; $argOk = $true } catch { } }
    Check "tool: arguments are valid JSON naming Paris" ($DryRun -or ($argOk -and "$($argObj.location)" -match "Paris")) "arguments '$(if ($calls.Count) { $calls[0].function.arguments })'"
    if ($calls.Count -ge 1) {
        $assistant = @{ role = "assistant"; content = $(if ((Text $t1)) { Text $t1 } else { "" }); tool_calls = @(@{ id = [string]$calls[0].id; type = "function"; function = @{ name = [string]$calls[0].function.name; arguments = [string]$calls[0].function.arguments } }) }
        $toolMsg = @{ role = "tool"; tool_call_id = [string]$calls[0].id; content = '{"temperature_c": 18, "condition": "light rain"}' }
        $t2 = Chat ($q1 + @($assistant, $toolMsg)) (@{ tools = $tools } + $NoThink) 128 "tool result fed back"
        Check "tool: final answer after the tool result (200, stop, text uses the result)" ($t2.Status -eq 200 -and $t2.Json.choices[0].finish_reason -eq "stop" -and (Text $t2) -match "18|rain") "'$(Short (Text $t2) 140)'"
    } else {
        Check "tool: final answer after the tool result" $false "no tool call to feed back"
    }

    Section "6. multi-turn prefix reuse, warm turn 2 == cold turn 2"
    $para = New-Filler 8
    $m1 = @(User ("Remember this number: 4721. Here is some unrelated log text you can ignore:`n" + $para + "Reply with just OK."))
    $a1 = Chat $m1 $NoThink 16 "turn 1"
    $m2 = $m1 + @(@{ role = "assistant"; content = (Text $a1) }, (User "What number did I ask you to remember? Answer with just the number."))
    $a2 = Chat $m2 $NoThink 16 "turn 2 (extends turn 1)"
    $p1 = [int]$a1.Json.usage.prompt_tokens; $p2 = [int]$a2.Json.usage.prompt_tokens; $n2 = [int]$a2.Json.timings.prompt_n
    Check "turn 2: answers 4721" ($a2.Status -eq 200 -and (Text $a2) -match "4721") "'$(Short (Text $a2))'"
    Check "turn 2: prefix reused (timings.prompt_n << usage.prompt_tokens, only the new tokens prefilled)" ($DryRun -or ($n2 -gt 0 -and $n2 -le ($p2 - 0.8 * $p1))) "turn 1 prompt_tokens $p1; turn 2 prompt_tokens $p2, prompt_n $n2 (cached $($p2 - $n2))"
    $null = Chat @(User "Say hello in one word.") $NoThink 8 "unrelated request (forces a reset)"
    $c2 = Chat $m2 $NoThink 16 "turn 2 again, cold"
    $nc = [int]$c2.Json.timings.prompt_n; $pc = [int]$c2.Json.usage.prompt_tokens
    Check "cold run: the whole prompt was prefilled (prompt_n == prompt_tokens)" ($DryRun -or ($nc -eq $pc -and $pc -eq $p2)) "prompt_n $nc, prompt_tokens $pc (warm run: $p2)"
    Check "turn-2 text: warm (prefix reuse) == cold (temperature 0)" ($DryRun -or (Text $a2) -ceq (Text $c2)) "warm '$(Short (Text $a2) 50)' vs cold '$(Short (Text $c2) 50)'"

    Section "7. ~20k-token needle (QSA active beyond 2051 tokens)"
    $needleWord = "TANGERINE-4417"
    $lines = 575   # ~34.6 tokens per line with this tokenizer (measured: 640 lines = 22.2k tokens) -> ~20k
    $hay = New-Filler $lines ([int]($lines / 4)) "IMPORTANT: the secret code word is $needleWord. Remember it."
    $nd = Chat @(User ($hay + "`nWhat is the secret code word mentioned in the log above? Answer with just the code word.")) $NoThink 32 "needle at 25% depth, $lines filler lines" 1800
    $np = [int]$nd.Json.usage.prompt_tokens
    Check "needle: prompt is ~20k tokens" ($DryRun -or ($np -ge 17000 -and $np -le 24000)) "prompt_tokens $np"
    Check "needle: the code word is retrieved" ($nd.Status -eq 200 -and (Text $nd) -match $needleWord) "'$(Short (Text $nd))'; prefill $([math]::Round([double]$nd.Json.timings.prompt_per_second, 1)) tok/s, $([math]::Round($nd.Sec, 1)) s wall"

    Section "8. request beyond --max-ctx"
    $big = New-Filler ([int]($MaxCtx * 1.7 / 30))
    $ov = Chat @(User ($big + "`nSummarize.")) $NoThink 16 "prompt of well over $MaxCtx tokens" 300
    $emsg = if ($ov.Json -and $ov.Json.error) { [string]$ov.Json.error.message } else { "" }
    Check "over max-ctx: HTTP 400 invalid_request_error naming --max-ctx" ($ov.Status -eq 400 -and ($DryRun -or ($ov.Json.error.type -eq "invalid_request_error" -and $emsg -match "max-ctx"))) "status $($ov.Status), '$(Short $emsg 120)'"

    Section "9. back-to-back requests"
    $b1 = Chat @(User "What is the capital of France?") $NoThink 64 "back-to-back A (France)"
    $b2 = Chat @(User "What is the capital of Japan?") $NoThink 64 "back-to-back B (Japan)"
    $b3 = Chat @(User "What is the capital of France?") $NoThink 64 "back-to-back A again"
    Check "back-to-back: B answers Tokyo" ($b2.Status -eq 200 -and (Text $b2) -match "Tokyo") "'$(Short (Text $b2))'"
    Check "back-to-back: A, B, A (server alive after the 400; the second A equals the first A and the section-2 text)" ($DryRun -or ($b1.Status -eq 200 -and $b3.Status -eq 200 -and (Text $b1) -ceq (Text $b3) -and (Text $b1) -ceq $fText)) "'$(Short (Text $b1) 50)' / '$(Short (Text $b3) 50)'"

    Section "10. clean shutdown"
    if ($DryRun) {
        Write-Output "  > Ctrl-C (CTRL_C_EVENT via AttachConsole from a helper process) to the server's console, wait <= 60 s, expect exit code 0 and no '[r4dx-server][error]' / 'HIP error' line in $ErrLog"
        Check "shutdown: Ctrl-C ends the server with exit code 0" $true "(dry run)"
    } else {
        $errLines = @(Select-String -Path $ErrLog -Pattern "\[error\]|HIP error|hipError|access violation" -ErrorAction SilentlyContinue)
        Check "no error line in the server log during the run" ($errLines.Count -eq 0) "$($errLines.Count) line(s)$(if ($errLines.Count) { ': ' + (Short $errLines[0].Line 160) })"
        $rc = Send-CtrlC $proc.Id
        $exited = $proc.WaitForExit(60000)
        Check "shutdown: Ctrl-C delivered and the server exits within 60 s" ($rc -eq 0 -and $exited) "helper rc $rc, exited $exited"
        if (-not $exited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; $proc.WaitForExit(10000) | Out-Null }
        Check "shutdown: exit code 0" ($exited -and $proc.ExitCode -eq 0) "exit code $($proc.ExitCode)"
    }
}
catch {
    Check "smoke aborted" $false "$_"
}
finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; $proc.WaitForExit(10000) | Out-Null }
    $env:HIP_VISIBLE_DEVICES = $saved.HIP; $env:R4DX_Q4_DEVICES = $saved.DEV; $env:R4DX_Q4_SPLIT = $saved.SPL
}

# ---- table -----------------------------------------------------------------------------------------------------------------------------
Write-Output ""
if ($DryRun) {
    Write-Output ("[smoke] DRY RUN: {0} checks planned, nothing was started or sent:" -f $script:Results.Count)
    foreach ($r in $script:Results) { Write-Output "  - $($r.Name)" }
    exit 0
}
$fail = @($script:Results | Where-Object { -not $_.Ok })
Write-Output "| result | check | detail |"
Write-Output "|---|---|---|"
foreach ($r in $script:Results) { Write-Output ("| {0} | {1} | {2} |" -f $(if ($r.Ok) { "PASS" } else { "FAIL" }), $r.Name, ($r.Detail -replace '\|', '/')) }
Write-Output ""
Write-Output ("[smoke] {0}: {1} passed, {2} failed; checks took {3:N1} min; server log {4}" -f $(if ($fail.Count -eq 0) { "PASS" } else { "FAIL" }), ($script:Results.Count - $fail.Count), $fail.Count, ((Get-Date) - $tChecks).TotalMinutes, $ErrLog)
exit $(if ($fail.Count -eq 0) { 0 } else { 1 })
