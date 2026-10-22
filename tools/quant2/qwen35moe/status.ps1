# Progress of the qwen3_5_moe quantization run (read-only): .\tools\quant2\qwen35moe\status.ps1 [-Root E:\models\r4dx\q35moe]
param([string]$Root = 'E:\models\r4dx\q35moe', [int]$Layers = 48)
$logs = Join-Path $Root 'logs'
$done = @{}
foreach ($f in @((Join-Path $Root 'run_all.out.log')) + @(Get-ChildItem $logs -Filter 'worker-*.out.log' -ErrorAction SilentlyContinue | ForEach-Object FullName)) {
  if (Test-Path $f) { Select-String -Path $f -Pattern 'verify-L(\d+) done in' | ForEach-Object { $done[[int]$_.Matches[0].Groups[1].Value] = 1 } }
}
$hess = @(Select-String -Path (Join-Path $Root 'run_all.out.log'), (Join-Path $logs 'worker-*.out.log') -Pattern 'hessian-L(\d+) done in' -ErrorAction SilentlyContinue | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Sort-Object -Unique)
Write-Host ("{0}   layers verified: {1} of {2}   Hessians done up to layer {3}" -f (Get-Date -Format 'ddd HH:mm'), $done.Count, $Layers, ($hess | Select-Object -Last 1))
$failed = Join-Path $Root 'FAILED.txt'
if (Test-Path $failed) { Write-Host "FAILED: $(Get-Content $failed -Raw)" -ForegroundColor Red } else { Write-Host 'no FAILED.txt (nothing has stopped)' }
foreach ($w in 'G1', 'G0') {
  $f = Join-Path $logs "worker-$w.out.log"
  if (Test-Path $f) { $l = Select-String -Path $f -Pattern 'worker \w+\] =====' | Select-Object -Last 1; Write-Host ("{0}: {1}" -f $w, $l.Line) }
}
Write-Host ("python processes running: {0}" -f @(Get-Process python -ErrorAction SilentlyContinue).Count)
$e = Get-PSDrive E; Write-Host ("E: free {0} GiB" -f [math]::Round($e.Free / 1GB, 1))
