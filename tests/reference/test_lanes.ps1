# tests/reference/test_lanes.ps1 -- the two-lane mode of tools\quant2\qwen35moe\pipeline.ps1
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\reference\test_lanes.ps1            # CPU only, ~1 min
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\reference\test_lanes.ps1 -Gpu       # + the tiny model through both lanes
#
# CPU part (the real functions are cut out of pipeline.ps1 with the PowerShell parser and run against stubs):
#   * Try-Verify runs verify EXACTLY once per layer when the two lanes finish a layer at the same time, in either order (race of
#     two jobs per layer, 12 layers), and not before both K are done;
#   * Check-Stop throws LANE_STOP only in lane mode with a STOP marker; Write-Failed in lane mode writes the FAILED line with the lane
#     name, keeps the FIRST failure and writes STOP;
#   * the plan (-Lanes 2 without -Run): lane A = hessian + check + K3 only, lane B = probe + K4 only, one verify per layer.
# GPU part (-Gpu): scratch root, the tiny checkpoint, layers 0..3, both lanes on HIP device 0 with the locks gpu-test / gpu-dev0 (the
# main run's `gpu` lock is not touched): every layer verified, both oracle dirs complete, alloc + convert at the end, no FAILED.txt.
param([switch]$Gpu, [string]$Scratch = (Join-Path $env:TEMP 'lanes_test'))
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pipeline = Join-Path $repo 'tools\quant2\qwen35moe\pipeline.ps1'
$script:checks = 0; $script:fails = @()
function Check([bool]$ok, [string]$what) { $script:checks++; if (-not $ok) { $script:fails += $what; Write-Host "FAIL: $what" } else { Write-Host "ok   $what" } }

# the functions under test, as text
$ast = [System.Management.Automation.Language.Parser]::ParseFile($pipeline, [ref]$null, [ref]$null)
$want = 'Lk', 'LTag', 'LMark', 'LHas', 'LaneStatus', 'Check-Stop', 'Try-Verify', 'Write-Failed'
$fnText = ($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $want -contains $n.Name }, $true) | ForEach-Object { $_.Extent.Text }) -join "`n"
Check (($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $want -contains $n.Name }, $true)).Count -eq $want.Count) 'all lane functions found in pipeline.ps1'

$tmp = Join-Path $env:TEMP ("lanes_unit_{0}" -f [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$prelude = @"
`$ErrorActionPreference = 'Stop'
`$Run = `$true; `$Lane = 'X'; `$kList = @(3, 4)
`$script:lanesDir = '$tmp\lanes'
`$script:failedMarker = '$tmp\FAILED.txt'
`$script:markerWritten = `$false
function HessianFiles([int]`$L) { `$true }
function Stage-Verify([int]`$L) { Add-Content -Path '$tmp\verify.calls' -Value "L`$L"; Start-Sleep -Milliseconds 200 }
$fnText
"@
$preludeFile = Join-Path $tmp 'prelude.ps1'
Set-Content -Path $preludeFile -Value $prelude -Encoding utf8

# 1. the verify race
New-Item -ItemType Directory -Force "$tmp\lanes" | Out-Null
$jobs = @()
for ($L = 0; $L -lt 12; $L++) {
  foreach ($k in 3, 4) {
    $jobs += Start-Job -ArgumentList $preludeFile, $L, $k -ScriptBlock {
      param($pf, $L, $k)
      . $pf
      $script:Lane = if ($k -eq 3) { 'A' } else { 'B' }
      Start-Sleep -Milliseconds (Get-Random -Minimum 0 -Maximum 150)
      LMark ("k$k-" + (LTag $L) + '.done') "lane $script:Lane"
      Try-Verify $L
    }
  }
}
$jobs | Wait-Job -Timeout 120 | Out-Null
$jobs | Receive-Job -ErrorAction SilentlyContinue | Out-Null
$calls = @(Get-Content "$tmp\verify.calls" -ErrorAction SilentlyContinue)
Check ($calls.Count -eq 12 -and (($calls | Sort-Object -Unique).Count -eq 12)) "verify ran exactly once for each of 12 layers under a race ($($calls.Count) calls, $(($calls | Sort-Object -Unique).Count) distinct)"
Check ((@(Get-ChildItem "$tmp\lanes" -Filter 'verify-*.done')).Count -eq 12) 'every layer has its verify-done marker'
$jobs | Remove-Job -Force -ErrorAction SilentlyContinue

# not before both K
Remove-Item -Recurse -Force "$tmp\lanes", "$tmp\verify.calls" -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -Command ". '$preludeFile'; LMark 'k3-L07.done'; Try-Verify 7" | Out-Null
Check (-not (Test-Path "$tmp\verify.calls") -and -not (Test-Path "$tmp\lanes\verify-L07.claim")) 'with only K3 done, verify is not started and not claimed'

# 2. stop and failure markers
$r = & powershell -NoProfile -ExecutionPolicy Bypass -Command ". '$preludeFile'; try { Check-Stop; 'no-stop' } catch { `$_.Exception.Message }; LMark 'STOP' 't'; try { Check-Stop; 'no-stop' } catch { `$_.Exception.Message }; `$Lane = ''; try { Check-Stop; 'single-lane-ignores' } catch { `$_.Exception.Message }"
$r = @($r | Where-Object { $_ -notmatch '^\[lane' })
Check ((@($r) -join ',') -eq 'no-stop,LANE_STOP,single-lane-ignores') "Check-Stop: only in lane mode with a STOP marker ($(@($r) -join ','))"
Remove-Item -Recurse -Force "$tmp\lanes" -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -Command ". '$preludeFile'; `$Lane = 'B'; Write-Failed 'oracle-k4-L5' 'exit 1: RuntimeError: first' 'C:\x.log'; `$Lane = 'A'; Write-Failed 'hessian-L6' 'exit 1: second' | Out-Null" | Out-Null
$line = Get-Content "$tmp\FAILED.txt" -Raw
Check ($line -match 'FAILED lane=B step=oracle-k4-L5: exit 1: RuntimeError: first' -and $line -notmatch 'second' -and (Test-Path "$tmp\lanes\STOP")) 'Write-Failed: lane named, the FIRST failure stays, STOP written'

# 3. the plan
$plan = & powershell -NoProfile -ExecutionPolicy Bypass -File $pipeline -Stage all -Lanes 2 -Layers 0..1 -Root (Join-Path $tmp 'planroot') 2>&1 | ForEach-Object { $_.ToString() }
$txt = $plan -join "`n"
$a = $txt.Substring($txt.IndexOf('[lane A] device'), $txt.IndexOf('[lane B] device') - $txt.IndexOf('[lane A] device'))
$b = $txt.Substring($txt.IndexOf('[lane B] device')); $b = $b.Substring(0, $b.IndexOf('===== alloc'))
Check ($a -match 'hessian_q35.py capture' -and $a -match 'oracle-k3-L1' -and $a -notmatch 'oracle-k4' -and $a -match '-Name gpu ' -and $a -match 'HIP_VISIBLE_DEVICES=1') 'plan, lane A: hessian + K3 only, device 1, lock gpu'
$bq = ($b -split "`n" | Where-Object { $_ -notmatch 'verify-L' }) -join "`n"   # the verify command names both oracle dirs
Check ($bq -match 'gpu-probe-B' -and $bq -match 'oracle-k4-L0' -and $bq -notmatch 'oracle-k3' -and $bq -notmatch 'hessian_q35.py capture' -and $bq -match '-Name gpu-dev0' -and $bq -match 'HIP_VISIBLE_DEVICES=0&& set R4DX_ALLOW_GPU0=1') 'plan, lane B: probe + K4 only, device 0 with R4DX_ALLOW_GPU0, lock gpu-dev0'
Check (([regex]::Matches($b, 'verify-L[01] CMD')).Count -eq 2 -and $a -notmatch 'verify-L') 'plan: verify once per layer (listed under the second lane)'

if ($Gpu) {
  $root = $Scratch
  Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $root | Out-Null
  $py = 'D:\venvs\r4dx-qwen4-ref\Scripts\python.exe'
  $env:HIP_VISIBLE_DEVICES = '-1'
  & $py (Join-Path $repo 'tools\quant2\qwen35moe\corpus_q35.py') random --tokens 12000 --vocab 4095 --out (Join-Path $root 'tokens_calib.json') | Out-Null
  Remove-Item Env:\HIP_VISIBLE_DEVICES
  $out = Join-Path $root 'supervisor.out.log'
  $pa = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $pipeline, '-Stage', 'all', '-Lanes', '2', '-Run', '-Layers', '0..3', '-Root', $root,
          '-Model', 'E:\models\r4dx\tiny-q35moe\hf', '-Tokens', (Join-Path $root 'tokens_calib.json'), '-Output', (Join-Path $root 'tiny-lanes.r4dx'),
          '-DeviceA', '0', '-DeviceB', '0', '-LockA', 'gpu-test', '-LockB', 'gpu-dev0', '-ProbeMinFreeGiB', '4', '-ChunkExperts', '8',
          '-Cohort', '8', '-PollSeconds', '5', '-MinFreeGiB', '20')
  $p = Start-Process -FilePath powershell.exe -ArgumentList $pa -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $out -RedirectStandardError (Join-Path $root 'supervisor.err.log')
  Check ($p.ExitCode -eq 0) "tiny model through two lanes: supervisor exit $($p.ExitCode)"
  Check (-not (Test-Path (Join-Path $root 'FAILED.txt'))) 'no FAILED.txt'
  Check ((@(Get-ChildItem (Join-Path $root 'lanes') -Filter 'verify-*.done')).Count -eq 4) 'all four layers verified'
  foreach ($k in 3, 4) {
    $j = & $py -c "import json,sys; m = json.load(open(sys.argv[1], encoding='utf-8')); print(m['complete'], len(m['layers_done']))" (Join-Path $root "oracle-k$k\weights_override.json") 2>$null
    Check ((@($j) -join '') -eq 'True 4') "oracle-k$k complete, 4 layers ($j)"
  }
  Check (Test-Path (Join-Path $root 'tiny-lanes.r4dx')) 'alloc + convert ran: tiny-lanes.r4dx'
  $left = @(Get-ChildItem (Join-Path $root 'hessian') -Recurse -Filter '*.hess' -ErrorAction SilentlyContinue).Count
  Check ($left -eq 0) "every layer's .hess files deleted by verify ($left left)"
}
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
if ($script:fails.Count -gt 0) { Write-Host "$($script:fails.Count) of $($script:checks) checks FAILED"; exit 1 }
Write-Host "OK ($($script:checks) checks)"
