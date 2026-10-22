# Live progress table for the Ornith-1.5-122B (qwen3_5_moe) bring-up. Read-only; Ctrl-C to quit.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\q35_progress.ps1 [-Once] [-IntervalSec 5]
# Task rows come from <Root>\progress.json (kept up to date by the session driving the work);
# the "live" panel reads the machine: running r4dx / reference processes, GPU-lock waiters, disk, newest log.
param([string]$Root = 'E:\models\r4dx\q35moe', [int]$IntervalSec = 5, [switch]$Once)

$colors = @{ done = 'Green'; warn = 'Yellow'; running = 'Cyan'; queued = 'DarkGray'; blocked = 'Red'; stopped = 'DarkYellow' }
$marks = @{ done = '[done]'; warn = '[done*]'; running = '[RUN ]'; queued = '[wait]'; blocked = '[BLOCK]'; stopped = '[stop]' }

function Bar($pct, $w = 20) {
  if ($pct -lt 0) { return '[  ...working...    ]' }
  $n = [int][math]::Round($w * [math]::Min(100, $pct) / 100)
  return '[' + ('#' * $n) + ('.' * ($w - $n)) + ']'
}
function Trunc($s, $n) { if ($null -eq $s) { return '' }; if ($n -lt 2) { return '' }; if ($s.Length -le $n) { return $s }; return $s.Substring(0, $n - 1) + '~' }
function Width { try { return [math]::Max(100, [Console]::WindowWidth - 1) } catch { return 150 } }

function Render {
  $w = Width
  $out = New-Object System.Collections.Generic.List[object]
  $out.Add(@('white', ("Ornith-1.5-122B on r4dx   |   {0}   |   refresh {1}s" -f (Get-Date -Format 'ddd yyyy-MM-dd HH:mm:ss'), $IntervalSec)))
  $out.Add(@('white', ('=' * $w)))
  $pj = Join-Path $Root 'progress.json'
  $tasks = @()
  if (Test-Path $pj) { try { $tasks = @((Get-Content $pj -Raw -Encoding utf8 | ConvertFrom-Json).tasks) } catch { $out.Add(@('Red', "cannot read ${pj}: $_")) } }
  else { $out.Add(@('Red', "no $pj")) }
  $out.Add(@('white', ('{0,-3} {1,-7} {2,-44} {3,-22} {4}' -f '#', 'status', 'task', 'progress', 'detail')))
  foreach ($t in $tasks) {
    $st = [string]$t.status
    $pct = if ($null -ne $t.pct) { [int]$t.pct } else { -1 }
    $barTxt = Bar $pct
    $pctTxt = if ($pct -ge 0) { ('{0,3}%' -f $pct) } else { ' ...' }
    $line = '{0,-3} {1,-7} {2,-44} {3,-22} {4}' -f $t.id, $marks[$st], (Trunc $t.name 44), ($barTxt + $pctTxt), (Trunc $t.detail ($w - 82))
    $out.Add(@($colors[$st], $line))
  }
  $out.Add(@('white', ('-' * $w)))
  $out.Add(@('white', 'LIVE'))
  # processes
  $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.CommandLine -and ($_.CommandLine -match 'r4dx-cli|r4dx-server|ref_run|r4dx_dump|gate_q35|hessian_q35|oracle_q35|pipeline\.ps1|kv_calib|r4dx-convert|test_model_q35') -and ($_.ProcessId -ne $PID) -and ($_.Name -notmatch 'cmd|conhost')
    })
  $waiters = @($procs | Where-Object { $_.CommandLine -match 'with_lock\.ps1' -and $_.CommandLine -match '-Name gpu' }).Count
  $jobs = @($procs | Where-Object { $_.Name -match 'python|r4dx' })
  if ($jobs.Count -eq 0) { $out.Add(@('DarkGray', 'GPU jobs: none running')) }
  else {
    $out.Add(@('Cyan', ('GPU/CPU jobs running: {0}   (with_lock wrappers: {1})' -f $jobs.Count, $waiters)))
    foreach ($j in ($jobs | Select-Object -First 5)) {
      $age = [int]((Get-Date) - $j.CreationDate).TotalMinutes
      $cmd = $j.CommandLine -replace '\s+', ' '
      $short = if ($cmd -match '(r4dx-cli|r4dx-server|ref_run|r4dx_dump|gate_q35|hessian_q35|oracle_q35|r4dx-convert|test_model_q35\w*)[^ ]*(.*)$') { $Matches[1] + (Trunc $Matches[2] 70) } else { Trunc $cmd 80 }
      $out.Add(@('Cyan', ('  pid {0,-6} {1,4} min  {2}' -f $j.ProcessId, $age, $short)))
    }
  }
  $e = Get-PSDrive E -ErrorAction SilentlyContinue
  if ($e) {
    $gib = $e.Free / 1GB
    $out.Add(@($(if ($gib -lt 50) { 'Red' } elseif ($gib -lt 100) { 'Yellow' } else { 'white' }), ('Disk E: free {0:N1} GiB{1}' -f $gib, $(if ($gib -lt 100) { '   (low: reference dumps are ~9 GiB each)' } else { '' }))))
  }
  $failed = Join-Path $Root 'FAILED.txt'
  if (Test-Path $failed) { $out.Add(@('Red', ('FAILED.txt: ' + (Trunc ((Get-Content $failed -Raw) -replace '\s+', ' ') ($w - 14))))) }
  $newest = Get-ChildItem (Join-Path $Root 'logs') -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest) {
    $ago = [int]((Get-Date) - $newest.LastWriteTime).TotalMinutes
    $last = (Get-Content $newest.FullName -Tail 1 -ErrorAction SilentlyContinue)
    $out.Add(@('DarkGray', ('Newest log: {0} ({1} min ago)' -f $newest.Name, $ago)))
    $out.Add(@('DarkGray', ('  ' + (Trunc $last ($w - 4)))))
  }
  $done = @($tasks | Where-Object { $_.status -in 'done', 'warn' }).Count
  $out.Add(@('white', ('-' * $w)))
  $out.Add(@('white', ('{0} of {1} tasks finished   |   legend: [done] ok  [done*] finished with a caveat  [RUN] in progress  [wait] queued  [stop] stopped on purpose  [BLOCK] blocked' -f $done, $tasks.Count)))
  return $out
}

if ($Once) { foreach ($l in (Render)) { Write-Host $l[1] -ForegroundColor $(if ($l[0] -eq 'white') { 'Gray' } else { $l[0] }) }; return }
try { [Console]::CursorVisible = $false } catch {}
Clear-Host
while ($true) {
  $lines = Render
  [Console]::SetCursorPosition(0, 0)
  $w = Width
  foreach ($l in $lines) { Write-Host ((Trunc $l[1] $w).PadRight($w)) -ForegroundColor $(if ($l[0] -eq 'white') { 'Gray' } else { $l[0] }) }
  for ($i = 0; $i -lt 3; $i++) { Write-Host (' ' * $w) }
  Start-Sleep -Seconds $IntervalSec
}
