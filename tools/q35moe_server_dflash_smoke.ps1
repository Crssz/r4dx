<#
.SYNOPSIS
  r4dx-server on the real Ornith-1.5-122B container with and without --dflash: the same greedy chat request through both, the
  completions must be byte-identical and the server's dflash log line must show speculation happened (docs/qwen35moe-plan.md section 6).

.DESCRIPTION
  Both servers need both cards (~27 GiB each), so they run one after the other. Run it under the GPU lock:
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name gpu -Run "powershell -NoProfile -ExecutionPolicy Bypass -File tools\q35moe_server_dflash_smoke.ps1"
  Writes <OutDir>\plain.json, dflash.json (the responses) and the two servers' stderr logs.
#>
param(
    [string]$Model = "E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx",
    [string]$Tokenizer = "E:\models\r4dx\Ornith-1.5-122B",
    [string]$Dflash = "E:\models\r4dx\ornith-1.5-122b-dflash2-bf16.r4dx",
    [int]$K = 7,
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

function Run-Server([string]$Tag, [string[]]$Extra) {
    $log = Join-Path $OutDir "$Tag.server.err"
    $args = @("--model", $Model, "--layout", "trellis", "--tokenizer-dir", $Tokenizer, "--port", "$Port", "--max-ctx", "$MaxCtx") + $Extra
    $p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardError $log -RedirectStandardOutput (Join-Path $OutDir "$Tag.server.out") -PassThru -WindowStyle Hidden
    try {
        $ready = $false
        for ($i = 0; $i -lt 240 -and -not $p.HasExited; ++$i) {
            Start-Sleep -Seconds 2
            try { $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($r.status -eq "ok") { $ready = $true; break } } catch {}
        }
        if (-not $ready) { throw "$Tag server did not come up (see $log)" }
        $body = @{ model = "x"; messages = @(@{ role = "user"; content = $Prompt }); temperature = 0; max_tokens = $MaxTokens } | ConvertTo-Json -Depth 5
        $t0 = Get-Date
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600
        $dt = ((Get-Date) - $t0).TotalSeconds
        $resp | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutDir "$Tag.json") -Encoding utf8
        $text = $resp.choices[0].message.content
        $reason = $resp.choices[0].message.reasoning_content
        Write-Host ("[{0}] {1:N1} s, {2} completion tokens, content {3} chars, reasoning {4} chars" -f $Tag, $dt, $resp.usage.completion_tokens, ("$text").Length, ("$reason").Length)
        return @{ Text = "$reason|$text"; Seconds = $dt; Tokens = $resp.usage.completion_tokens }
    } finally {
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
        $p.WaitForExit()
    }
}

$plain = Run-Server "plain" @()
$df = Run-Server "dflash" @("--dflash", $Dflash, "--dflash-k", "$K")
$same = $plain.Text -ceq $df.Text
Write-Host ("completions byte-identical: {0}" -f $same)
$line = Select-String -Path (Join-Path $OutDir "dflash.server.err") -Pattern "dflash" | Select-Object -Last 3
$line | ForEach-Object { Write-Host $_.Line }
if (-not $same) { exit 1 }
