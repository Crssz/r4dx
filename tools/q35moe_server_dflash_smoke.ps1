<#
.SYNOPSIS
  r4dx-server on the real Ornith-1.5-122B container with and without --dflash: the same greedy requests (plain chat, streaming, a tool call)
  through both; every completion must be byte-identical and the dflash server's log must show that speculation happened
  (docs/qwen35moe-plan.md section 6).

.DESCRIPTION
  Both servers need both cards (~27 GiB each), so they run one after the other. Run it under the GPU lock:
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name gpu -Run "powershell -NoProfile -ExecutionPolicy Bypass -File tools\q35moe_server_dflash_smoke.ps1"
  Writes <OutDir>\{plain,dflash}.{chat,stream,tool}.json / .txt and the two servers' logs.
#>
param(
    [string]$Model = "E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx",
    [string]$Tokenizer = "E:\models\r4dx\Ornith-1.5-122B",
    [string]$Dflash = "E:\models\r4dx\ornith-1.5-122b-dflash2-w4a16.r4dx",
    [int]$K = 11,
    [double]$PMin = 0.8,
    [int]$Port = 8098,
    [int]$MaxCtx = 16384,
    [int]$MaxTokens = 300,
    [string]$OutDir = "",
    [string]$Prompt = "Write a Python function that parses a CSV file into a list of dictionaries without using the csv module, handling quoted fields with embedded commas and escaped quotes. Include docstrings and a short usage example."
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if ($OutDir -eq "") { $OutDir = Join-Path $env:TEMP "q35moe_server_smoke" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:R4DX_ALLOW_GPU0 = "1"
$exe = "build\win-hip\src\server\r4dx-server.exe"

function Post-Json([string]$Body) {
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $Body -TimeoutSec 900
}

function Run-Server([string]$Tag, [string[]]$Extra) {
    $log = Join-Path $OutDir "$Tag.server.err"
    $args = @("--model", $Model, "--layout", "trellis", "--tokenizer-dir", $Tokenizer, "--port", "$Port", "--max-ctx", "$MaxCtx") + $Extra
    $p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardError $log -RedirectStandardOutput (Join-Path $OutDir "$Tag.server.out") -PassThru -WindowStyle Hidden
    $res = @{}
    try {
        $ready = $false
        for ($i = 0; $i -lt 300 -and -not $p.HasExited; ++$i) {
            Start-Sleep -Seconds 2
            try { $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($r.status -eq "ok") { $ready = $true; break } } catch {}
        }
        if (-not $ready) { throw "$Tag server did not come up (see $log)" }

        # 1. plain non-streaming chat
        $body = @{ model = "x"; messages = @(@{ role = "user"; content = $Prompt }); temperature = 0; max_tokens = $MaxTokens } | ConvertTo-Json -Depth 6
        $t0 = Get-Date
        $resp = Post-Json $body
        $dt = ((Get-Date) - $t0).TotalSeconds
        $resp | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir "$Tag.chat.json") -Encoding utf8
        $m = $resp.choices[0].message
        $res.chat = "$($m.reasoning_content)|$($m.content)"
        Write-Host ("[{0}] chat: {1:N1} s, {2} completion tokens, {3:N1} tok/s end to end" -f $Tag, $dt, $resp.usage.completion_tokens, ($resp.usage.completion_tokens / $dt))

        # 2. streaming (SSE): concatenate every delta
        $sbody = @{ model = "x"; messages = @(@{ role = "user"; content = "Explain in three sentences how a binary search works." }); temperature = 0; max_tokens = 200; stream = $true } | ConvertTo-Json -Depth 6
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

        # 3. a tool call
        $tools = @(@{ type = "function"; function = @{ name = "get_weather"; description = "Get the current weather for a city"; parameters = @{ type = "object"; properties = @{ city = @{ type = "string" } }; required = @("city") } } })
        $tbody = @{ model = "x"; messages = @(@{ role = "user"; content = "What is the weather like in Paris right now? Use the tool." }); tools = $tools; temperature = 0; max_tokens = 400 } | ConvertTo-Json -Depth 10
        $tr = Post-Json $tbody
        $tr | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutDir "$Tag.tool.json") -Encoding utf8
        $tm = $tr.choices[0].message
        # the call ids are random per request; compare the function names and arguments
        $res.tool = "$($tm.reasoning_content)|$($tm.content)|" + (@($tm.tool_calls | ForEach-Object { "$($_.function.name)($($_.function.arguments))" }) -join ";")
        Write-Host ("[{0}] tool: finish_reason {1}, {2} tool call(s)" -f $Tag, $tr.choices[0].finish_reason, @($tm.tool_calls).Count)
    } finally {
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
        $p.WaitForExit()
    }
    return $res
}

$plain = Run-Server "plain" @()
$df = Run-Server "dflash" @("--dflash", $Dflash, "--dflash-k", "$K", "--dflash-p-min", "$PMin")
$ok = $true
foreach ($n in "chat", "stream", "tool") {
    $same = $plain[$n] -ceq $df[$n]
    Write-Host ("{0}: dflash server output byte-identical to the plain server's: {1}" -f $n, $same)
    if (-not $same) { $ok = $false }
}
Select-String -Path (Join-Path $OutDir "dflash.server.err") -Pattern "dflash" | Select-Object -Last 4 | ForEach-Object { Write-Host $_.Line }
Select-String -Path (Join-Path $OutDir "dflash.server.err") -Pattern "VRAM" | Select-Object -Last 2 | ForEach-Object { Write-Host $_.Line }
if (-not $ok) { exit 1 }
