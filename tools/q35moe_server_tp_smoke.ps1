<#
.SYNOPSIS
  r4dx-server on the real Ornith-1.5-122B container, layer split against --tp 2 (docs/qwen35moe-tp-design.md P3): the same greedy requests (plain chat,
  streaming, a tool call and its follow-up with the tool result) through both, a shape / sanity check on each answer, and optionally the fault-recovery check.

.DESCRIPTION
  Both servers need both cards (~27 GiB each), so they run one after the other. The tensor-parallel run is wrapped in the device-0 TDR watch
  (tools\tp\tdr_watch.psm1): a TDR stops the server at once, no retry, and fails the script. Takes the machine-wide mutexes `gpu` and `gpu-dev0` itself.

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\q35moe_server_tp_smoke.ps1 [-Modes split,tp2] [-Fault 1:3000:1]

  The TP outputs are NOT expected to equal the layer split's byte for byte (the all-reduce adds one bf16 rounding per sublayer; greedy text diverges at the
  first near-tie), so the checks per server are: every request returns 200 with non-empty text, the streaming reply has several chunks, the tool request
  returns a get_weather call for Paris and the follow-up (the call's result sent back) returns prose. The texts of both servers are written side by side.

  -Fault <rank>:<n>:<kind> (tp2 only; R4DX_TP_FAULT, docs/tp.md 9.1): rank <rank> stalls (kind 1) or throws (kind 0) at its <n>-th all-reduce after warm-up. The
  reference request runs first (the fault is pending: 12 tokens ~ 1300 all-reduces at 96 per forward), request A (a long generation) crosses the armed all-reduce (200 or 500 both accepted), request B =
  the reference prompt again must return 200 with text equal to the reference, through a recovered group.
#>
param(
    [string]$Model = "E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx",
    [string]$Tokenizer = "E:\models\r4dx\Ornith-1.5-122B",
    [string]$Dflash = "E:\models\r4dx\ornith-1.5-122b-dflash2-w4a16.r4dx",
    [int]$DflashK = 11,
    [string[]]$Modes = @("split", "tp2"),
    [string]$Fault = "",
    [int]$Port = 8099,
    [int]$MaxCtx = 8192,
    [int]$MaxTokens = 200,
    [string]$OutDir = "",
    [int]$TimeoutMinutes = 240
)
$ErrorActionPreference = "Stop"
$Modes = @($Modes | ForEach-Object { $_ -split ",\s*" })  # (powershell -File passes "split,tp2" as one string)
Set-Location (Split-Path -Parent $PSScriptRoot)
Import-Module (Join-Path (Get-Location) "tools\tp\tdr_watch.psm1") -Force
if ($OutDir -eq "") { $OutDir = Join-Path $env:TEMP "q35moe_server_tp_smoke" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$exe = "build\win-hip\src\server\r4dx-server.exe"
$Prompt = "Explain in three sentences how a binary search works."

function Post-Json([string]$Body) {
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $Body -TimeoutSec 900
}
function Chat-Body([string]$Text, [int]$Max) {
    @{ model = "x"; messages = @(@{ role = "user"; content = $Text }); temperature = 0; max_tokens = $Max } | ConvertTo-Json -Depth 6
}
function Text-Of($resp) { $m = $resp.choices[0].message; "$($m.reasoning_content)|$($m.content)" }

function Run-Server([string]$Tag, [string[]]$Extra, [bool]$Tp, [string]$FaultSpec) {
    $log = Join-Path $OutDir "$Tag.server.err"
    $args = @("--model", $Model, "--layout", "trellis", "--tokenizer-dir", $Tokenizer, "--port", "$Port", "--max-ctx", "$MaxCtx") + $Extra
    Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $env:R4DX_ALLOW_GPU0 = "1"
    if ($FaultSpec -ne "") { $env:R4DX_TP_FAULT = $FaultSpec } else { Remove-Item Env:\R4DX_TP_FAULT -ErrorAction SilentlyContinue }
    $start = Get-Date
    $p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardError $log -RedirectStandardOutput (Join-Path $OutDir "$Tag.server.out") -PassThru -WindowStyle Hidden
    $marker = Join-Path $OutDir "$Tag.tdr.marker"
    $job = $null
    if ($Tp) { $job = Start-TdrWatchJob -Since $start -TargetPid $p.Id -Marker $marker }
    $res = @{ ok = $true; why = @() }
    try {
        $ready = $false
        for ($i = 0; $i -lt 300 -and -not $p.HasExited; ++$i) {
            Start-Sleep -Seconds 2
            try { $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($r.status -eq "ok") { $ready = $true; break } } catch {}
        }
        if (-not $ready) { throw "$Tag server did not come up (see $log)" }
        Write-Host ("[{0}] ready after {1:N0} s" -f $Tag, ((Get-Date) - $start).TotalSeconds)

        if ($FaultSpec -ne "") {
            $ref = Text-Of (Post-Json (Chat-Body $Prompt 12))
            Write-Host "[$Tag] reference: $ref"
            $a_status = "200"
            try { $null = Post-Json (Chat-Body "Write a long story about a lighthouse keeper." 400) } catch { $a_status = "$($_.Exception.Response.StatusCode.value__)" }
            Write-Host "[$Tag] request A (crosses the armed all-reduce): HTTP $a_status"
            $b = Text-Of (Post-Json (Chat-Body $Prompt 12))
            $same = $b -ceq $ref
            Write-Host "[$Tag] request B (after the fault): equal to the reference: $same"
            if (-not $same) { $res.ok = $false; $res.why += "request B differs from the reference after the fault" }
            if (-not (Select-String -Path $log -Pattern "fault injection" -Quiet)) { $res.ok = $false; $res.why += "the server log does not show the fault firing" }
            if (-not (Select-String -Path $log -Pattern "recovered" -Quiet)) { $res.ok = $false; $res.why += "the server log does not show a recovery" }
        }

        # 1. plain non-streaming chat
        $t0 = Get-Date
        $resp = Post-Json (Chat-Body $Prompt $MaxTokens)
        $dt = ((Get-Date) - $t0).TotalSeconds
        $resp | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir "$Tag.chat.json") -Encoding utf8
        $res.chat = Text-Of $resp
        Write-Host ("[{0}] chat: {1:N1} s, {2} completion tokens, {3:N1} tok/s end to end" -f $Tag, $dt, $resp.usage.completion_tokens, ($resp.usage.completion_tokens / $dt))
        if ($res.chat.Length -lt 20) { $res.ok = $false; $res.why += "chat answer is empty" }

        # 2. streaming (SSE)
        $sbody = @{ model = "x"; messages = @(@{ role = "user"; content = "Say hello in five different languages." }); temperature = 0; max_tokens = 120; stream = $true } | ConvertTo-Json -Depth 6
        $req = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:$Port/v1/chat/completions")
        $req.Method = "POST"; $req.ContentType = "application/json"; $req.Timeout = 900000
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($sbody)
        $req.GetRequestStream().Write($bytes, 0, $bytes.Length)
        $rd = New-Object System.IO.StreamReader($req.GetResponse().GetResponseStream())
        $sb = New-Object System.Text.StringBuilder
        $chunks = 0
        while (($line = $rd.ReadLine()) -ne $null) {
            if (-not $line.StartsWith("data: ")) { continue }
            $payload = $line.Substring(6)
            if ($payload -eq "[DONE]") { break }
            $j = $payload | ConvertFrom-Json
            if ($j.choices.Count -gt 0) {
                $d = $j.choices[0].delta
                if ($d.reasoning_content) { [void]$sb.Append($d.reasoning_content) }
                if ($d.content) { [void]$sb.Append($d.content) }
                ++$chunks
            }
        }
        $rd.Close()
        $res.stream = $sb.ToString()
        Set-Content (Join-Path $OutDir "$Tag.stream.txt") $res.stream -Encoding utf8
        Write-Host ("[{0}] stream: {1} chunks, {2} chars" -f $Tag, $chunks, $res.stream.Length)
        if ($chunks -lt 5 -or $res.stream.Length -lt 10) { $res.ok = $false; $res.why += "streaming reply too short ($chunks chunks)" }

        # 3. a tool call, then its follow-up with the tool result
        $tools = @(@{ type = "function"; function = @{ name = "get_weather"; description = "Get the current weather for a city"; parameters = @{ type = "object"; properties = @{ city = @{ type = "string" } }; required = @("city") } } })
        $u = @{ role = "user"; content = "What is the weather like in Paris right now? Use the tool." }
        $tr = Post-Json (@{ model = "x"; messages = @($u); tools = $tools; temperature = 0; max_tokens = 400 } | ConvertTo-Json -Depth 10)
        $tr | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir "$Tag.tool.json") -Encoding utf8
        $tm = $tr.choices[0].message
        $calls = @($tm.tool_calls)
        $res.tool = (@($calls | ForEach-Object { "$($_.function.name)($($_.function.arguments))" }) -join ";")
        Write-Host ("[{0}] tool: finish_reason {1}, calls: {2}" -f $Tag, $tr.choices[0].finish_reason, $res.tool)
        if ($calls.Count -lt 1 -or $calls[0].function.name -ne "get_weather" -or $calls[0].function.arguments -notmatch "Paris") {
            $res.ok = $false; $res.why += "no get_weather(Paris) call"
        } else {
            $asst = @{ role = "assistant"; content = ""; tool_calls = @(@{ id = $calls[0].id; type = "function"; function = @{ name = $calls[0].function.name; arguments = $calls[0].function.arguments } }) }
            $tool = @{ role = "tool"; tool_call_id = $calls[0].id; content = '{"temperature_c": 18, "conditions": "light rain"}' }
            $fr = Post-Json (@{ model = "x"; messages = @($u, $asst, $tool); tools = $tools; temperature = 0; max_tokens = 200 } | ConvertTo-Json -Depth 10)
            $fr | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir "$Tag.followup.json") -Encoding utf8
            $res.followup = Text-Of $fr
            Write-Host ("[{0}] follow-up: {1}" -f $Tag, $res.followup.Substring(0, [Math]::Min(160, $res.followup.Length)))
            if ($res.followup.Length -lt 10) { $res.ok = $false; $res.why += "follow-up answer is empty" }
        }
    } catch {
        $res.ok = $false; $res.why += "$_"
    } finally {
        if ($job) { Stop-Job $job -ErrorAction SilentlyContinue; Remove-Job $job -Force -ErrorAction SilentlyContinue }
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
        $p.WaitForExit()
    }
    if ($Tp) {
        if (Test-Path $marker) { $res.ok = $false; $res.why += "TDR: " + ((Get-Content $marker) -join " ") }
        Start-Sleep -Seconds 30
        $c = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path "tools\tp" "tdr_check.ps1") -Since $start.ToString('yyyy-MM-ddTHH:mm:ss') -Quiet
        if ($LASTEXITCODE -ne 0) { $res.ok = $false; $res.why += "TDR (final check): $($c -join ' ')" }
        if (Select-String -Path $log -Pattern "HIP error 719|unspecified launch failure" -Quiet) { $res.ok = $false; $res.why += "HIP error 719 in the server log" }
    }
    return $res
}

if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw "stop the running r4dx-server first (this run uses both GPUs)" }
$held = @()
$overall = $true
try {
    foreach ($n in 'gpu', 'gpu-dev0') {
        $m = New-Object System.Threading.Mutex($false, "Global\r4dx-lock-$n")
        $got = $false
        try { $got = $m.WaitOne([TimeSpan]::FromMinutes($TimeoutMinutes)) } catch [System.Threading.AbandonedMutexException] { $got = $true }
        if (-not $got) { Write-Error "timed out waiting for lock '$n'"; exit 75 }
        $held += $m
    }
    $results = @{}
    foreach ($mode in $Modes) {
        if ($mode -eq "split") { $results[$mode] = Run-Server "split" @() $false "" }
        elseif ($mode -eq "tp2") { $results[$mode] = Run-Server "tp2" @("--tp", "2") $true $Fault }
        elseif ($mode -eq "tp2dflash") { $results[$mode] = Run-Server "tp2dflash" @("--tp", "2", "--dflash", $Dflash, "--dflash-k", "$DflashK", "--dflash-p-min", "0.8") $true "" }
        else { throw "unknown mode $mode" }
        $r = $results[$mode]
        Write-Host ("[{0}] {1}{2}" -f $mode, $(if ($r.ok) { "PASS" } else { "FAIL" }), $(if ($r.why.Count) { ": " + ($r.why -join "; ") } else { "" }))
        if (-not $r.ok) { $overall = $false }
    }
    if ($results.ContainsKey("split") -and $results.ContainsKey("tp2")) {
        foreach ($n in "chat", "stream") {
            Write-Host ("`n--- {0}: layer split`n{1}`n--- {0}: tp 2`n{2}" -f $n, $results["split"][$n], $results["tp2"][$n])
        }
    }
    if ($results.ContainsKey("tp2") -and $results.ContainsKey("tp2dflash")) {
        # DFlash2 is lossless inside one engine: the --tp 2 server with the drafter must answer byte for byte like the plain --tp 2 server
        # followup is informational: it resumes from the prefix cache the tool turn left, and the speculative server keeps one token fewer there
        # (prompt_n 32 vs 33), so its prefill is split differently and a near-tie greedy choice can flip
        foreach ($n in "chat", "stream", "tool", "followup") {
            $same = $results["tp2"][$n] -ceq $results["tp2dflash"][$n]
            Write-Host ("{0}: --tp 2 --dflash output byte-identical to the plain --tp 2 server's: {1}{2}" -f $n, $same, $(if ($n -eq "followup") { " (informational)" } else { "" }))
            if (-not $same -and $n -ne "followup") { $overall = $false }
        }
        Select-String -Path (Join-Path $OutDir "tp2dflash.server.err") -Pattern "dflash" | Select-Object -Last 4 | ForEach-Object { Write-Host $_.Line }
    }
    Select-String -Path (Join-Path $OutDir "tp2.server.err") -Pattern "tp: mode|\[stats\]|recovered|VRAM" -ErrorAction SilentlyContinue | Select-Object -Last 4 | ForEach-Object { Write-Host $_.Line }
} finally {
    for ($i = $held.Count - 1; $i -ge 0; $i--) { $held[$i].ReleaseMutex() }
}
if (-not $overall) { exit 1 }
