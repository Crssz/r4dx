# tools/prefill/run_tasks.ps1 -- runs the long-context task set (build_tasks.py output) through
# r4dx-server on ONE chosen GPU (or both with -Tp 2) and scores it.
#
# Starts r4dx-server with HIP_VISIBLE_DEVICES set explicitly (-Device N for TP=1; '0,1' for -Tp 2),
# --max-ctx sized from the task files (longest prompt + its max_tokens, rounded up to 1024), --vision
# off, waits for /health, runs run_tasks.py (resumable: results.jsonl in -OutDir), prints the score
# table, and always stops the server it started. Server stdout/stderr go to -OutDir\server.*.log.
#
#   .\tools\prefill\run_tasks.ps1 -Device 1 -Lengths 8k,32k -OutDir D:\models\r4dx\prefill-m0\runs\dense-8k32k
#   .\tools\prefill\run_tasks.ps1 -Tp 2 -Lengths 128k -OutDir D:\models\r4dx\prefill-m0\runs\dense-tp2-128k
#   .\tools\prefill\run_tasks.ps1 -Device 1 -Lengths 8k -Limit 1 -OutDir ...\smoke      (1 item per task)
#   .\tools\prefill\run_tasks.ps1 -Device 1 -Lengths 8k,32k,64k,128k -MaxTokens 1 -Task niah_single `
#       -OutDir ...\ttft     (a pure TTFT sweep: one generated token per item)
#
# Refuses to start when another r4dx-server / r4dx-cli / tool_teacher_forced_logprobs is running
# unless -AllowOthers (then it is on you that the other job uses the OTHER device).
param(
  [int]$Device = 1,
  [ValidateSet(1, 2)][int]$Tp = 1,
  [string[]]$Lengths = @('8k', '32k'),
  [string[]]$Task = @(),
  [int]$Limit = 0,
  [int]$MaxTokens = 0,
  [string]$Model = 'D:\models\r4dx\qwen38-27b-trellis-mix45m.r4dx',
  [string]$Layout = 'trellis',
  [string]$TasksDir = 'D:\models\r4dx\prefill-m0\tasks',
  [Parameter(Mandatory = $true)][string]$OutDir,
  [int]$Port = 8093,
  [string]$Server = '',
  [string[]]$ServerArgs = @(),
  [string]$Python = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe',
  [int]$MaxCtx = 0,
  [switch]$AllowOthers
)
$ErrorActionPreference = 'Stop'
[string]$LengthList = ($Lengths -join ','); [string]$TaskList = ($Task -join ',')  # a,b arrives as an array
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Server) { $Server = Join-Path $repo 'build\win-hip\src\server\r4dx-server.exe' }
if (-not (Test-Path $Server)) { throw "no server binary at $Server (run .\build.ps1)" }
if (-not (Test-Path $Model)) { throw "no container at $Model" }

$others = @(Get-Process r4dx-server, r4dx-cli, tool_teacher_forced_logprobs -ErrorAction SilentlyContinue)
if ($others.Count -gt 0) {
  $desc = ($others | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', '
  if (-not $AllowOthers) { throw "[run_tasks] other GPU jobs are running ($desc); stop them or pass -AllowOthers" }
  Write-Output "[run_tasks] WARNING: other GPU jobs running ($desc) -- make sure they use another device"
}

# --max-ctx from the frozen task files.
if ($MaxCtx -le 0) {
  $need = 0
  foreach ($len in $LengthList.Split(',')) {
    $f = Join-Path $TasksDir "tasks_$($len.Trim()).jsonl"
    if (-not (Test-Path $f)) { throw "missing $f (run tools\prefill\build_tasks.py)" }
    foreach ($line in [IO.File]::ReadLines($f)) {
      if (-not $line.Trim()) { continue }
      $pt = [int]([regex]::Match($line, '"prompt_tokens": (\d+)').Groups[1].Value)
      $mt = [int]([regex]::Match($line, '"max_tokens": (\d+)').Groups[1].Value)
      if ($pt + $mt -gt $need) { $need = $pt + $mt }
    }
  }
  $MaxCtx = [int]([math]::Ceiling(($need + 64) / 1024.0) * 1024)
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$argList = @('--model', $Model, '--layout', $Layout, '--port', "$Port", '--max-ctx', "$MaxCtx", '--vision', 'off')
if ($Tp -eq 2) {
  $env:HIP_VISIBLE_DEVICES = '0,1'
  $argList += @('--tp', '2')
} else {
  $env:HIP_VISIBLE_DEVICES = "$Device"
}
$argList += $ServerArgs
$outLog = Join-Path $OutDir 'server.out.log'
$errLog = Join-Path $OutDir 'server.err.log'
$runInfo = [ordered]@{
  started = (Get-Date).ToString('s'); server = $Server; args = ($argList -join ' ');
  hip_visible_devices = $env:HIP_VISIBLE_DEVICES; tp = $Tp; lengths = $LengthList; task = $TaskList; limit = $Limit;
  max_tokens_override = $MaxTokens; git_head = (git -C $repo rev-parse HEAD); git_dirty = [bool](git -C $repo status --porcelain)
}
$runInfo | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $OutDir 'run_info.json')
Write-Output "[run_tasks] HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES) $Server $($argList -join ' ')"
$proc = Start-Process -FilePath $Server -ArgumentList $argList -NoNewWindow -PassThru `
  -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$rc = 1
try {
  $t0 = Get-Date
  while ($true) {
    if ($proc.HasExited) { throw "[run_tasks] server exited early (code $($proc.ExitCode)); see $errLog" }
    try {
      $h = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$Port/health"
      if ($h.StatusCode -eq 200) { break }
    } catch { }
    if (((Get-Date) - $t0).TotalSeconds -gt 900) { throw '[run_tasks] server not healthy after 900 s' }
    Start-Sleep -Seconds 3
  }
  Write-Output ("[run_tasks] server up after {0:N0}s" -f ((Get-Date) - $t0).TotalSeconds)
  $py = @((Join-Path $PSScriptRoot 'run_tasks.py'), '--port', "$Port", '--tasks-dir', $TasksDir,
          '--lengths', $LengthList, '--out', $OutDir)
  if ($TaskList) { $py += @('--task', $TaskList) }
  if ($Limit -gt 0) { $py += @('--limit', "$Limit") }
  if ($MaxTokens -gt 0) { $py += @('--max-tokens', "$MaxTokens") }
  & $Python @py
  $rc = $LASTEXITCODE
} finally {
  if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -Confirm:$false }
  Write-Output "[run_tasks] server stopped; logs: $errLog"
}
exit $rc
