#requires -Version 5.1
<#
.SYNOPSIS
  qwen4_exp throughput on both cards through r4dx-cli (greedy, --stats): prefill tok/s and TTFT at 256 / 2048 / 8192 / 32768
  prompt tokens, and the decode tok/s of 128 generated tokens after each. Prints a markdown table; no thresholds.

.DESCRIPTION
  One fresh r4dx-cli process per length (so each TTFT is a cold prefill after a load), each under its own
  `with_lock.ps1 -Name gpu` acquisition, both cards visible (HIP_VISIBLE_DEVICES removed), R4DX_Q4_DEVICES / R4DX_Q4_SPLIT from
  -Devices / -Split. The prompt is deterministic log-line filler (about 34.6 tokens per line with the Qwen3.8 tokenizer, measured)
  followed by an instruction that asks for a long answer, so the 128-token decode does not hit EOS early; the table reports the
  token counts the CLI really measured (prompt tokens include the chat template). TTFT = the CLI's prefill wall seconds
  ("[stats] prefill: N tok in Xs"): the first token is sampled from the prefill's last row.
  VRAM: the CLI's `VRAM:` figure is hipMemGetInfo of ONE device (the CLI's current device); qwen4_exp's per-stage Vram() is not
  printed by the CLI at --tp 1. So the table also carries (a) the per-stage weight GiB / device from the model's load log and
  (b) the peak "GPU Adapter Memory / Dedicated Usage" per Windows adapter during the run minus its value just before the run
  (Get-Counter, 2 s sampling; adapters are listed by LUID, which is not a HIP ordinal).

.PARAMETER Container   the .r4dx container (required)
.PARAMETER Layout      default trellis
.PARAMETER Devices     R4DX_Q4_DEVICES "a,b", default 0,1
.PARAMETER Split       R4DX_Q4_SPLIT (0 = the model's choice)
.PARAMETER Lengths     prompt token targets, default 256,2048,8192,32768
.PARAMETER DecodeTokens  --max-tokens, default 128
.PARAMETER TokenizerDir  passed as --tokenizer-dir when given (the CLI's own default for qwen4_exp is the 27B directory)
.PARAMETER Out         directory for the prompts, logs, bench.md and bench.json. Default E:\models\r4dx\q4exp\bench\<yyyyMMdd-HHmmss>
.PARAMETER Cli         r4dx-cli.exe to run (default build\win-hip\src\cli\r4dx-cli.exe)
.PARAMETER DryRun      print every command, run nothing
.EXAMPLE
  .\tools\qwen4exp\bench.ps1 -Container E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx
.EXAMPLE
  .\tools\qwen4exp\bench.ps1 -Container E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Container,
    [string]$Layout = "trellis",
    [string]$Devices = "0,1",
    [int]$Split = 0,
    [string[]]$Lengths = @("256", "2048", "8192", "32768"),   # "256,2048" is one string under powershell -File

    [int]$DecodeTokens = 128,
    [string]$TokenizerDir = "",
    [string]$Out = "",
    [string]$Cli = "build\win-hip\src\cli\r4dx-cli.exe",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Lengths = @($Lengths | ForEach-Object { $_ -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
if ($Devices -notmatch '^\d+,\d+$') { throw "-Devices must be 'a,b' (HIP ordinals), got '$Devices'" }
$WithLock = "tools\with_lock.ps1"
if (-not $Out) { $Out = Join-Path "E:\models\r4dx\q4exp\bench" (Get-Date -Format "yyyyMMdd-HHmmss") }
$TokPerLine = 34.6          # measured with the Qwen3.8 tokenizer on these lines (640 lines = 22,169 tokens)
$TemplateTokens = 40        # chat template + the closing instruction, approximately
function Q { param([string]$S) if ($S -match '\s') { '\"' + $S + '\"' } else { $S } }

function New-Filler {
    param([int]$Lines)
    $sb = New-Object Text.StringBuilder
    for ($i = 0; $i -lt $Lines; $i++) {
        $n = 5000 + $i
        [void]$sb.Append("Log entry $n : the lighthouse keeper recorded wind $(($n * 7) % 90) knots, tide $(($n * 3) % 12) feet and visibility $(($n % 9) + 1) miles at dock $(($n * 11) % 40).`n")
    }
    $sb.ToString()
}
$Closing = "Now write a very long and detailed essay about the history of lighthouses. Keep writing and do not stop."

function Get-AdapterGiB {
    $o = @{}
    try { foreach ($s in (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples) { $o[$s.InstanceName -replace '_phys_\d+$', ''] = [math]::Round($s.CookedValue / 1GB, 2) } } catch { }
    $o
}

Write-Output "[bench] container=$Container layout=$Layout devices=$Devices split=$Split lengths=$($Lengths -join ',') decode=$DecodeTokens out=$Out"
Write-Output "[bench] env per run: HIP_VISIBLE_DEVICES removed (both cards), R4DX_Q4_DEVICES=$Devices$(if ($Split) { ", R4DX_Q4_SPLIT=$Split" })"
if (-not $DryRun) {
    if (-not (Test-Path -LiteralPath $Container)) { throw "container not found: $Container" }
    if (-not (Test-Path -LiteralPath $Cli)) { throw "r4dx-cli.exe not found at $Cli (build the r4dx-cli target)" }
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
}
$saved = @{ HIP = $env:HIP_VISIBLE_DEVICES; DEV = $env:R4DX_Q4_DEVICES; SPL = $env:R4DX_Q4_SPLIT }
$rows = @()
try {
    if (-not $DryRun) {
        Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        $env:R4DX_Q4_DEVICES = $Devices
        if ($Split) { $env:R4DX_Q4_SPLIT = "$Split" } else { Remove-Item Env:R4DX_Q4_SPLIT -ErrorAction SilentlyContinue }
    }
    foreach ($len in $Lengths) {
        $lenInt = [int]$len
        $lines = [math]::Max(1, [int][math]::Round(($lenInt - $TemplateTokens) / $TokPerLine))
        $maxCtx = [int]([math]::Ceiling(($lenInt + [int]$DecodeTokens + 512) / 1024.0) * 1024)
        $promptFile = Join-Path $Out "prompt_$len.txt"
        $errLog = Join-Path $Out "run_$len.err.log"
        $outLog = Join-Path $Out "run_$len.out.log"
        $cliArgs = "--model $(Q $Container) --layout $Layout --prompt-file $(Q $promptFile) --max-ctx $maxCtx --temperature 0 --max-tokens $DecodeTokens --think off --stats"
        if ($TokenizerDir) { $cliArgs += " --tokenizer-dir $(Q $TokenizerDir)" }
        $run = "$Cli $cliArgs 1> $(Q $outLog) 2> $(Q $errLog)"
        $lockCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File $WithLock -Name gpu -Run `"$run`""
        Write-Output "[bench] $len tokens: $lines filler lines (~$([int]($lines * $TokPerLine + $TemplateTokens)) tokens) -> $promptFile"
        Write-Output "[bench]   > $lockCmd"
        if ($DryRun) { continue }
        Set-Content -LiteralPath $promptFile -Value ((New-Filler $lines) + $Closing) -Encoding UTF8 -NoNewline
        $base = Get-AdapterGiB
        $peak = @{}; foreach ($k in $base.Keys) { $peak[$k] = $base[$k] }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $WithLock, "-Name", "gpu", "-Run", "`"$run`"" -PassThru -WindowStyle Hidden
        $null = $p.Handle
        $started = $false
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 2
            foreach ($kv in (Get-AdapterGiB).GetEnumerator()) { if (-not $peak.ContainsKey($kv.Key) -or $kv.Value -gt $peak[$kv.Key]) { $peak[$kv.Key] = $kv.Value } }
        }
        $wall = $sw.Elapsed.TotalSeconds
        $err = if (Test-Path $errLog) { Get-Content -Raw $errLog } else { "" }
        $row = [ordered]@{ target_tokens = $len; exit = $p.ExitCode; wall_s = [math]::Round($wall, 1); log = $errLog }
        $m = [regex]::Match($err, '\[stats\] prefill: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\) \| decode: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\) \| eos=([^|]*)\| VRAM: ([\d.]+) GiB')
        if ($m.Success) {
            $row.prefill_tokens = [int]$m.Groups[1].Value; $row.ttft_s = [double]$m.Groups[2].Value; $row.prefill_tok_s = [double]$m.Groups[3].Value
            $row.decode_tokens = [int]$m.Groups[4].Value; $row.decode_s = [double]$m.Groups[5].Value; $row.decode_tok_s = [double]$m.Groups[6].Value
            $row.eos = $m.Groups[7].Value.Trim(); $row.cli_vram_gib = [double]$m.Groups[8].Value
        } else { Write-Output "[bench] $len : no '[stats] prefill: ... decode: ...' line (exit $($p.ExitCode)); see $errLog" }
        $row.stages = @([regex]::Matches($err, 'stage (\d+): layers \[(\d+), (\d+)\) on device (\d+), ([\d.]+) GiB weights') | ForEach-Object { "stage $($_.Groups[1].Value) layers $($_.Groups[2].Value)-$($_.Groups[3].Value) dev $($_.Groups[4].Value) $($_.Groups[5].Value) GiB weights" })
        $row.adapter_peak_delta_gib = @($peak.Keys | Sort-Object | ForEach-Object { "{0}: +{1:N2}" -f ($_ -replace '^luid_0x00000000_', ''), ($peak[$_] - $(if ($base.ContainsKey($_)) { $base[$_] } else { 0 })) })
        $rows += [pscustomobject]$row
        Write-Output ("[bench] {0} done in {1:N0} s (exit {2})" -f $len, $wall, $p.ExitCode)
    }
}
finally {
    $env:HIP_VISIBLE_DEVICES = $saved.HIP; $env:R4DX_Q4_DEVICES = $saved.DEV; $env:R4DX_Q4_SPLIT = $saved.SPL
}
if ($DryRun) { Write-Output "[bench] dry run: nothing was run"; exit 0 }

$md = New-Object Text.StringBuilder
[void]$md.AppendLine("| target | prompt tok | TTFT (s) = prefill s | prefill tok/s | decode tok | decode tok/s | CLI VRAM (GiB, 1 device) | adapter VRAM peak delta (GiB) |")
[void]$md.AppendLine("|---:|---:|---:|---:|---:|---:|---:|---|")
foreach ($r in $rows) {
    [void]$md.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $r.target_tokens, $r.prefill_tokens, $r.ttft_s, $r.prefill_tok_s, $r.decode_tokens, $r.decode_tok_s, $r.cli_vram_gib, ($r.adapter_peak_delta_gib -join "; ")))
}
[void]$md.AppendLine("")
foreach ($r in ($rows | Select-Object -First 1)) { foreach ($s in $r.stages) { [void]$md.AppendLine("* $s") } }
[void]$md.AppendLine("* container $Container, layout $Layout, R4DX_Q4_DEVICES=$Devices; greedy (--temperature 0), $DecodeTokens-token decode; decode ends early at EOS (see the decode tok column)")
Write-Output ""
Write-Output $md.ToString()
Set-Content -LiteralPath (Join-Path $Out "bench.md") -Value $md.ToString() -Encoding UTF8
$rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Out "bench.json") -Encoding UTF8
Write-Output "[bench] wrote $(Join-Path $Out 'bench.md'), $(Join-Path $Out 'bench.json')"
exit $(if (@($rows | Where-Object { $_.exit -ne 0 -or $null -eq $_.decode_tok_s }).Count -eq 0) { 0 } else { 1 })
