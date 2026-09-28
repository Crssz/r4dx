# tools/prefill/ttft_cli.ps1 -- cold prefill (TTFT) of the frozen long prompts through r4dx-cli, one
# fresh process per run, parsed from `--stats` ("[stats] prefill: N tok in Xs (Y tok/s)").
# -ProfilePrefill instead runs `--profile-prefill` (per-op-family hipEvent table of the whole
# prefill -- the attention-share profile M0 asks for; it prints the table and generates nothing).
#
#   .\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 8k,32k -Runs 2
#   .\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 32k -ProfilePrefill
#   .\tools\prefill\ttft_cli.ps1 -Tp 2 -Lengths 8k,32k,64k,128k
#
# Prompts: <TasksDir>\prompts\ttft_<len>.txt (build_tasks.py writes them: the first niah_single item
# of each length, a user message the CLI wraps in the chat template, thinking off). Output lines are
# appended to <OutDir>\ttft.jsonl; each run's full stderr goes to <OutDir>\ttft_<len>_<run>.log.
param(
  [int]$Device = 1,
  [ValidateSet(1, 2)][int]$Tp = 1,
  [string[]]$Lengths = @('8k', '32k'),
  [int]$Runs = 1,
  [string]$Model = 'D:\models\r4dx\qwen38-27b-trellis-mix45m.r4dx',
  [string]$Layout = 'trellis',
  [string]$TasksDir = 'D:\models\r4dx\prefill-m0\tasks',
  [string]$OutDir = 'D:\models\r4dx\prefill-m0\ttft',
  [string]$Cli = '',
  [string[]]$ExtraArgs = @(),
  [switch]$ProfilePrefill,
  [switch]$AllowOthers
)
$ErrorActionPreference = 'Stop'
[string]$LengthList = ($Lengths -join ',')  # a,b arrives as an array
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Cli) { $Cli = Join-Path $repo 'build\win-hip\src\cli\r4dx-cli.exe' }
$others = @(Get-Process r4dx-server, r4dx-cli, tool_teacher_forced_logprobs -ErrorAction SilentlyContinue)
if ($others.Count -gt 0 -and -not $AllowOthers) {
  throw "[ttft] other GPU jobs are running ($(($others | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', ')); pass -AllowOthers if they use another device"
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
if ($Tp -eq 2) { $env:HIP_VISIBLE_DEVICES = '0,1' } else { $env:HIP_VISIBLE_DEVICES = "$Device" }
$lenTokens = @{ '4k' = 4096; '8k' = 8192; '16k' = 16384; '32k' = 32768; '64k' = 65536; '128k' = 131072 }
$head = (git -C $repo rev-parse --short HEAD)
foreach ($len in $LengthList.Split(',')) {
  $len = $len.Trim()
  $prompt = Join-Path $TasksDir "prompts\ttft_$len.txt"
  if (-not (Test-Path $prompt)) { throw "[ttft] missing $prompt (run build_tasks.py)" }
  $maxCtx = [int]([math]::Ceiling(($lenTokens[$len] + 256) / 1024.0) * 1024)
  for ($r = 1; $r -le $Runs; $r++) {
    $a = @('--model', $Model, '--layout', $Layout, '--prompt-file', $prompt, '--max-ctx', "$maxCtx",
           '--vision', 'off', '--temperature', '0', '--max-tokens', '8', '--stats')
    if ($ProfilePrefill) { $a += '--profile-prefill' }
    if ($Tp -eq 2) { $a += @('--tp', '2') }
    $a += $ExtraArgs
    $log = Join-Path $OutDir ("ttft_{0}_{1}{2}.log" -f $len, $r, $(if ($ProfilePrefill) { '_profile' } else { '' }))
    Write-Output "[ttft] HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES) $len run $r -> $log"
    $t0 = Get-Date
    # Start-Process (not &): PowerShell 5.1 would wrap every stderr line in an ErrorRecord.
    $p = Start-Process -FilePath $Cli -ArgumentList $a -NoNewWindow -Wait -PassThru `
      -RedirectStandardError $log -RedirectStandardOutput ($log -replace '\.log$', '.out.txt')
    $code = $p.ExitCode
    $wall = ((Get-Date) - $t0).TotalSeconds
    $text = Get-Content $log -Raw
    $m = [regex]::Match($text, '\[stats\] prefill: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\)')
    $rec = [ordered]@{ length = $len; run = $r; tp = $Tp; exit = $code; wall_s = [math]::Round($wall, 2);
      profile = [bool]$ProfilePrefill; git = $head; extra = ($ExtraArgs -join ' ') }
    if ($m.Success) {
      $rec.prefill_tokens = [int]$m.Groups[1].Value; $rec.prefill_s = [double]$m.Groups[2].Value
      $rec.prefill_tok_s = [double]$m.Groups[3].Value
      Write-Output ("[ttft] {0}: {1} tok in {2:N2}s ({3:N1} tok/s), process wall {4:N1}s" -f $len, $rec.prefill_tokens, $rec.prefill_s, $rec.prefill_tok_s, $wall)
    } elseif ($ProfilePrefill) {
      Write-Output "[ttft] $len profile table in $log"
    } else {
      Write-Output "[ttft] $len : no [stats] prefill line (exit $code); see $log"
    }
    ($rec | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 (Join-Path $OutDir 'ttft.jsonl')
  }
}
