#requires -Version 5.1
<#
.SYNOPSIS
  Pipeline-parallel prefill, Phase 0 (docs/pp-prefill.md section 8): measure on BOTH GPUs what the design needs
  before any two-GPU pipeline code is written, and write the projected TTFT speedup and the go rule to
  <models root>\r4dx\pp\phase0\summary.txt.

.DESCRIPTION
  GPU RUN -- the main session runs this, with the user's approval, and NEVER while an r4dx-* process exists (the
  pre-flight refuses). Nothing here is destructive: it only reads the container and writes under -OutDir.

  What it runs (tools built from branch `pp`: tests\model\tool_pp_stage_bench / tool_pp_hop_bench / tool_pp_project):

    1  pre-flight          refuses if an r4dx-* process runs; prints each card's PCIe link (idle).
    2  stage cost, dev 1   tool_pp_stage_bench on HIP device 1 (the headless card, decode's): the monolithic
                           prefill TTFT and the PP-emulate run (layers [0,k) as stage A, [k,N) as stage B, the
                           carry through pinned host memory, per-chunk stage times) at 8k, 32k, 64k.
    3  stage cost, dev 0   the same on HIP device 0 (the desktop card): device skew and stage A's cost there; plus the
                           design's literal probe on both cards: only layers [0,32) loaded (--layers 32), mono 8k / 32k.
    4  DFlash / MTP        --dflash at split 35 (8k, 32k) on both cards, --mtp 3 at split 33 (8k) on device 1:
                           the DFlash injection and MTP priming per 256-row chunk (the design's < 12 ms gate).
    5  hop bandwidth       tool_pp_hop_bench, both cards visible: D2H on one card -> pinned -> H2D on the other,
                           both orders, 2.5 / 7.5 / 12.75 MiB and 24 x 3 MiB, alone.
    6  both GPUs busy      (needs -UserPresent) two stage benches at the same time, one per card, behind a start
                           barrier, while the hop bench runs on top of them: the dual-GPU clock / power sag and
                           the hop bandwidth "concurrent with compute"; the PCIe link is re-read under load.
    7  clock probe         (unless -SkipClockProbe) R4DX_CLOCK_PROBE=1 on one 8k emulate run per card, alone and
                           together: the sustained shader clock inside the model (the probe perturbs the timing,
                           so it never shares a run with the numbers above).
    8  projection          tool_pp_project: the two-stage pipeline simulation over the measured per-chunk stage
                           times, the design's go rule (>= 1.6x projected at 8k AND 32k), the other gates.

  POWER: no software power sensor is reachable from this repo on Windows (C:\opt\rocm\bin has no amd-smi /
  rocm-smi). The 2026-09-28 power-off happened with both cards ramping (docs/prefill.md), so step 6 and the
  both-cards clock probe are opt-in (-UserPresent), bounded (about 25 s of dual load, 60 s with -Concurrent64k)
  and announced: start your overlay (Adrenalin metrics / HWiNFO) before the script prints "RELEASING THE BARRIER"
  (-PreGoSeconds apart) and read the board power off it. The summary carries the clocks and the link state; the
  power line says "not measured by this script".

  Output (all under -OutDir, default <models root>\r4dx\pp\phase0): stage_dev0.csv, stage_dev1.csv,
  dflash_dev0.csv, dflash_dev1.csv, mtp_dev1.csv, hop_alone.csv, conc_dev0.csv, conc_dev1.csv, hop_conc.csv,
  the logs of every process, notes.txt (hardware, links, clocks, power) and summary.txt. Exit code: 0 = GO,
  3 = STOP, 2 = inconclusive / a failed run.

  Time: about 15-25 minutes (two model loads per card for plain, one per card for --dflash, one for --mtp, two
  more per clock-probe run; the 64k runs are 36+ s each). -Resume reuses CSVs already in -OutDir; -Skip64k drops
  the 64k runs; -Repeats8k sets the repeats of the 8k runs (median).

.PARAMETER BuildDir
  The build tree holding tests\model\tool_pp_*.exe (default: build\win-hip of the checkout this script is in).
.PARAMETER UserPresent
  Allow the both-cards-busy phases (power / clock / concurrent hop). Without it they are skipped and the go rule
  is judged on each card measured alone.
#>
[CmdletBinding()]
param(
    [string]$BuildDir = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'build\win-hip'),
    [string]$Model = '',
    [string]$Drafter = '',
    [string]$OutDir = '',
    [int]$Split = 33,
    [int]$DflashSplit = 35,
    [int]$Repeats8k = 3,
    [int]$PreGoSeconds = 10,
    [switch]$UserPresent,
    [switch]$Concurrent64k,
    [switch]$Skip64k,
    [switch]$SkipDflash,
    [switch]$SkipMtp,
    [switch]$SkipClockProbe,
    [switch]$SkipConcurrent,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$modelsRoot = if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' }
if (-not $Model) { $Model = Join-Path $modelsRoot 'r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx' }
if (-not $Drafter) { $Drafter = Join-Path $modelsRoot 'r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx' }
if (-not $OutDir) { $OutDir = Join-Path $modelsRoot 'r4dx\pp\phase0' }
$tools = Join-Path $BuildDir 'tests\model'
$stageExe = Join-Path $tools 'tool_pp_stage_bench.exe'
$hopExe = Join-Path $tools 'tool_pp_hop_bench.exe'
$projExe = Join-Path $tools 'tool_pp_project.exe'

# ---- pre-flight ---------------------------------------------------------------------------------------------
foreach ($e in @($stageExe, $hopExe, $projExe)) {
    if (-not (Test-Path $e)) { throw "[pp-phase0] $e not found: build it (cmake --build $BuildDir --target tool_pp_stage_bench tool_pp_hop_bench tool_pp_project)" }
}
if (-not (Test-Path $Model)) { throw "[pp-phase0] model container not found: $Model" }
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'r4dx-*' })
if ($busy.Count -gt 0) {
    throw ("[pp-phase0] refusing to run: r4dx process(es) are using the GPUs: " +
           (($busy | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', ') + " (the production server holds ~28 GiB of device 1)")
}
$others = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'tool_*' -or $_.ProcessName -like 'test_*' })
if ($others.Count -gt 0) {
    Write-Warning ("[pp-phase0] other r4dx test/tool processes are running (" + (($others | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', ') + "): the measurements need the GPUs to themselves")
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$log = Join-Path $OutDir 'phase0.log'
function Say([string]$m) {
    $line = "[pp-phase0 $(Get-Date -Format 'HH:mm:ss')] $m"
    Write-Output $line
    Add-Content -Path $log -Value $line
}
Say "start; build $BuildDir; out $OutDir; split $Split (dflash $DflashSplit); user present: $UserPresent"

# ---- helpers ------------------------------------------------------------------------------------------------
function Format-Arg([string]$a) {
    if ($a -eq '') { return '""' }
    if ($a -notmatch '[\s"]') { return $a }
    $q = $a -replace '(\\*)"', '$1$1\"'
    $q = $q -replace '(\\+)$', '$1$1'
    return '"' + $q + '"'
}
# Starts $Exe with the environment variables in $EnvMap set for it only; stdout / stderr to <Prefix>.out.txt /
# <Prefix>.log. Start-Process (not &): PowerShell 5.1 would wrap every native stderr line in an ErrorRecord.
function Start-Tool([string]$Exe, [string[]]$ArgList, [string]$Prefix, [hashtable]$EnvMap) {
    $saved = @{}
    foreach ($k in $EnvMap.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, [string]$EnvMap[$k])
    }
    try {
        $argLine = (($ArgList | ForEach-Object { Format-Arg $_ }) -join ' ')
        return Start-Process -FilePath $Exe -ArgumentList $argLine -NoNewWindow -PassThru `
            -RedirectStandardOutput "$Prefix.out.txt" -RedirectStandardError "$Prefix.log"
    } finally {
        foreach ($k in $EnvMap.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}
function Wait-Tool($Proc, [string]$What) {
    $Proc.WaitForExit()
    if ($Proc.ExitCode -ne 0) { throw "[pp-phase0] $What failed (exit $($Proc.ExitCode)); see its .log in $OutDir" }
}
function Run-Stage([int]$Device, [string]$Csv, [string]$Runs, [string[]]$Extra, [hashtable]$ExtraEnv, [string]$LogName) {
    $out = Join-Path $OutDir $Csv
    if ($Resume -and (Test-Path $out)) { Say "resume: keeping $Csv"; return }
    $envMap = @{ HIP_VISIBLE_DEVICES = "$Device" }
    if ($ExtraEnv) { foreach ($k in $ExtraEnv.Keys) { $envMap[$k] = $ExtraEnv[$k] } }
    $toolArgs = @('--model', $Model, '--runs', $Runs, '--out', $out, '--device-note', "physical device $Device") + $Extra
    Say "stage bench on device ${Device}: $Runs $($Extra -join ' ') -> $Csv"
    $p = Start-Tool $stageExe $toolArgs (Join-Path $OutDir $LogName) $envMap
    Wait-Tool $p "tool_pp_stage_bench (device $Device, $Csv)"
}
function Get-PcieLinks([string]$Tag) {
    $lines = @()
    foreach ($d in @(Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'OK' -and $_.InstanceId -like 'PCI\VEN_1002*' })) {
        $p = Get-PnpDeviceProperty -InstanceId $d.InstanceId -ErrorAction SilentlyContinue
        $get = { param($n) ($p | Where-Object { $_.KeyName -eq "DEVPKEY_PciDevice_$n" } | Select-Object -First 1).Data }
        $bus = ($p | Where-Object { $_.KeyName -eq 'DEVPKEY_Device_BusNumber' } | Select-Object -First 1).Data
        $names = @{ 1 = '2.5'; 2 = '5'; 3 = '8'; 4 = '16'; 5 = '32'; 6 = '64' }  # DEVPKEY_PciDevice_*LinkSpeed codes, GT/s
        $cur = & $get 'CurrentLinkSpeed'; $max = & $get 'MaxLinkSpeed'
        $lines += ("PCIe link ($Tag): {0}, bus {1}: x{2} at {3} GT/s now (max x{4} at {5} GT/s)" -f $d.FriendlyName, $bus,
                   (& $get 'CurrentLinkWidth'), $names[[int]$cur], (& $get 'MaxLinkWidth'), $names[[int]$max])
    }
    return $lines
}

$notes = New-Object System.Collections.Generic.List[string]
$notes.Add("hardware / run notes")
$notes.Add("host: $env:COMPUTERNAME; build: $BuildDir; git: $((git -C (Split-Path -Parent (Split-Path -Parent $BuildDir)) rev-parse --short HEAD 2>$null))")
foreach ($l in (Get-PcieLinks 'idle')) { $notes.Add($l); Say $l }
$notes.Add("(an idle card may train its link down; the under-load line below is the one that matters)")

$k = $Split
$sizes8 = "8kx$Repeats8k"
$plainRuns = @("mono:$sizes8", "emu${k}:$sizes8", 'mono:32k', "emu${k}:32k")
if (-not $Skip64k) { $plainRuns += @('mono:64k', "emu${k}:64k") }
$plainRunsArg = ($plainRuns -join ',')

# ---- 2, 3: stage cost on each card ------------------------------------------------------------------------------
Run-Stage 1 'stage_dev1.csv' $plainRunsArg @() $null 'stage_dev1'
Run-Stage 0 'stage_dev0.csv' $plainRunsArg @() $null 'stage_dev0'
# the design's literal probe too: only layers [0, 32) loaded on each card (its TTFT against the full model's, and the skew)
Run-Stage 1 'half_dev1.csv' 'mono:8k,mono:32k' @('--layers', '32') $null 'half_dev1'
Run-Stage 0 'half_dev0.csv' 'mono:8k,mono:32k' @('--layers', '32') $null 'half_dev0'

# ---- 4: DFlash injection / MTP priming ---------------------------------------------------------------------------
if (-not $SkipDflash) {
    if (Test-Path $Drafter) {
        $dk = $DflashSplit
        foreach ($dev in 1, 0) {
            Run-Stage $dev "dflash_dev$dev.csv" "mono:8k,emu${dk}:8k,mono:32k,emu${dk}:32k" @('--dflash', $Drafter) $null "dflash_dev$dev"
        }
    } else {
        Say "no DFlash drafter at ${Drafter}: skipping the --dflash runs"; $SkipDflash = $true
    }
}
if (-not $SkipMtp) {
    Run-Stage 1 'mtp_dev1.csv' "mono:8k,emu${k}:8k" @('--mtp', '3') $null 'mtp_dev1'
}

# ---- 5: hop bandwidth, alone -------------------------------------------------------------------------------------
$hopAlone = Join-Path $OutDir 'hop_alone.csv'
if ($Resume -and (Test-Path $hopAlone)) {
    Say 'resume: keeping hop_alone.csv'
} else {
    Say 'hop bench (both cards idle)'
    $p = Start-Tool $hopExe @('--src', '0', '--dst', '1', '--reps', '60', '--mode', 'alone', '--out', $hopAlone) (Join-Path $OutDir 'hop_alone') @{ HIP_VISIBLE_DEVICES = '0,1' }
    Wait-Tool $p 'tool_pp_hop_bench (alone)'
}
foreach ($l in (Get-Content (Join-Path $OutDir 'hop_alone.log') -ErrorAction SilentlyContinue | Where-Object { $_ -match 'visible ordinal' })) { $notes.Add($l) }

# ---- 6: both GPUs busy -------------------------------------------------------------------------------------------
$haveConc = $false
if ($UserPresent -and -not $SkipConcurrent) {
    $haveConc = $true
    $concRuns = @("emu${k}:8k", "emu${k}:32k")
    if ($Concurrent64k) { $concRuns += "emu${k}:64k" }
    $concRunsArg = ($concRuns -join ',')
    $c0 = Join-Path $OutDir 'conc_dev0.csv'; $c1 = Join-Path $OutDir 'conc_dev1.csv'
    $hopConc = Join-Path $OutDir 'hop_conc.csv'
    if ($Resume -and (Test-Path $c0) -and (Test-Path $c1) -and (Test-Path $hopConc)) {
        Say 'resume: keeping conc_dev0/1.csv and hop_conc.csv'
    } else {
        $barrier = Join-Path $OutDir 'conc_barrier'
        Remove-Item "$barrier.ready", "$barrier.go" -ErrorAction SilentlyContinue
        Say 'BOTH-GPUS-BUSY phase: loading one model per card (each process waits at the barrier once loaded)'
        $pa = Start-Tool $stageExe @('--model', $Model, '--runs', $concRunsArg, '--out', $c0, '--barrier', "${barrier}0", '--device-note', 'physical device 0, both cards busy') (Join-Path $OutDir 'conc_dev0') @{ HIP_VISIBLE_DEVICES = '0' }
        $pb = Start-Tool $stageExe @('--model', $Model, '--runs', $concRunsArg, '--out', $c1, '--barrier', "${barrier}1", '--device-note', 'physical device 1, both cards busy') (Join-Path $OutDir 'conc_dev1') @{ HIP_VISIBLE_DEVICES = '1' }
        $deadline = (Get-Date).AddMinutes(15)
        while (-not ((Test-Path "${barrier}0.ready") -and (Test-Path "${barrier}1.ready"))) {
            if ($pa.HasExited -or $pb.HasExited) { throw 'a stage bench exited before the barrier; see conc_dev0.log / conc_dev1.log' }
            if ((Get-Date) -gt $deadline) { throw 'timeout waiting for both stage benches to load' }
            Start-Sleep -Milliseconds 500
        }
        Say "both cards loaded and idle. Start your power overlay NOW; load starts in $PreGoSeconds s (about 25 s of both cards at full load)"
        Start-Sleep -Seconds $PreGoSeconds
        Say 'RELEASING THE BARRIER'
        Set-Content "${barrier}0.go" 'go'; Set-Content "${barrier}1.go" 'go'
        # the hop bench rides on top of the busy cards, a few seconds in (past the first chunks)
        Start-Sleep -Seconds 8
        foreach ($l in (Get-PcieLinks 'under load')) { $notes.Add($l); Say $l }
        $ph = Start-Tool $hopExe @('--src', '0', '--dst', '1', '--reps', '30', '--duration', '6', '--mode', 'conc', '--out', $hopConc) (Join-Path $OutDir 'hop_conc') @{ HIP_VISIBLE_DEVICES = '0,1' }
        Wait-Tool $ph 'tool_pp_hop_bench (concurrent)'
        Wait-Tool $pa 'tool_pp_stage_bench (concurrent, device 0)'
        Wait-Tool $pb 'tool_pp_stage_bench (concurrent, device 1)'
        Say 'both-cards-busy load finished'
    }
} else {
    Say 'skipping the both-cards-busy phases (pass -UserPresent to run them; the power-off of 2026-09-28 happened with both cards ramping)'
    $notes.Add('both-GPUs-busy phase: NOT RUN (no -UserPresent): the go rule is judged on each card measured alone')
}

# ---- 7: clock probe ----------------------------------------------------------------------------------------------
$clockLines = @()
if (-not $SkipClockProbe) {
    $probeEnv = @{ R4DX_CLOCK_PROBE = '1' }
    foreach ($dev in 1, 0) {
        $csv = "clock_dev${dev}_alone.csv"
        Run-Stage $dev $csv "emu${k}:8k" @() $probeEnv "clock_dev${dev}_alone"
    }
    if ($UserPresent) {
        $barrier = Join-Path $OutDir 'clock_barrier'
        Remove-Item "${barrier}0.ready", "${barrier}0.go", "${barrier}1.ready", "${barrier}1.go" -ErrorAction SilentlyContinue
        Say 'clock probe, both cards at once'
        $pa = Start-Tool $stageExe @('--model', $Model, '--runs', "emu${k}:8k,emu${k}:8k", '--out', (Join-Path $OutDir 'clock_dev0_both.csv'), '--barrier', "${barrier}0") (Join-Path $OutDir 'clock_dev0_both') @{ HIP_VISIBLE_DEVICES = '0'; R4DX_CLOCK_PROBE = '1' }
        $pb = Start-Tool $stageExe @('--model', $Model, '--runs', "emu${k}:8k,emu${k}:8k", '--out', (Join-Path $OutDir 'clock_dev1_both.csv'), '--barrier', "${barrier}1") (Join-Path $OutDir 'clock_dev1_both') @{ HIP_VISIBLE_DEVICES = '1'; R4DX_CLOCK_PROBE = '1' }
        $deadline = (Get-Date).AddMinutes(15)
        while (-not ((Test-Path "${barrier}0.ready") -and (Test-Path "${barrier}1.ready"))) {
            if ($pa.HasExited -or $pb.HasExited) { throw 'a clock-probe stage bench exited before the barrier' }
            if ((Get-Date) -gt $deadline) { throw 'timeout waiting for the clock-probe benches' }
            Start-Sleep -Milliseconds 500
        }
        Start-Sleep -Seconds $PreGoSeconds
        Set-Content "${barrier}0.go" 'go'; Set-Content "${barrier}1.go" 'go'
        Wait-Tool $pa 'clock probe (both, device 0)'
        Wait-Tool $pb 'clock probe (both, device 1)'
    }
    $notes.Add('')
    $notes.Add('sustained shader clock inside the model (R4DX_CLOCK_PROBE, one wave after every Mlp; debug_probe.h), the [probe] clock lines:')
    foreach ($name in 'clock_dev1_alone', 'clock_dev0_alone', 'clock_dev0_both', 'clock_dev1_both') {
        $f = Join-Path $OutDir "$name.log"
        if (-not (Test-Path $f)) { continue }
        $sel = @(Get-Content $f | Where-Object { $_ -match '^\[probe\].*(clock \(probe|prefill +T=)' })
        $notes.Add("  $name :")
        foreach ($l in ($sel | Select-Object -First 8)) { $notes.Add('    ' + $l.TrimEnd()) }
        if ($sel.Count -eq 0) { $notes.Add('    (no probe summary in the log)') }
    }
}
$notes.Add('')
$notes.Add('power: NOT measured by this script (no software board-power sensor on this stack). Read it off the overlay during the')
$notes.Add('both-cards-busy phase and record it in docs/pp-prefill.md; the design wants a documented power limit if either card browns out.')
$notesFile = Join-Path $OutDir 'notes.txt'
Set-Content -Path $notesFile -Value ($notes -join "`r`n") -Encoding utf8

# ---- 8: projection -----------------------------------------------------------------------------------------------
$projArgs = @('--stage-a', (Join-Path $OutDir 'stage_dev0.csv'), '--stage-b', (Join-Path $OutDir 'stage_dev1.csv'),
              '--half-a', (Join-Path $OutDir 'half_dev0.csv'), '--half-b', (Join-Path $OutDir 'half_dev1.csv'),
              '--hop', $hopAlone, '--split', "$Split", '--dflash-split', "$DflashSplit", '--notes', $notesFile,
              '--out', (Join-Path $OutDir 'summary.txt'))
if ($haveConc) { $projArgs += @('--conc-a', (Join-Path $OutDir 'conc_dev0.csv'), '--conc-b', (Join-Path $OutDir 'conc_dev1.csv'), '--hop-conc', (Join-Path $OutDir 'hop_conc.csv')) }
if (-not $SkipDflash) { $projArgs += @('--dflash-a', (Join-Path $OutDir 'dflash_dev0.csv'), '--dflash-b', (Join-Path $OutDir 'dflash_dev1.csv')) }
if (-not $SkipMtp) { $projArgs += @('--mtp-b', (Join-Path $OutDir 'mtp_dev1.csv')) }
Say 'projection'
$pp = Start-Tool $projExe $projArgs (Join-Path $OutDir 'project') @{ HIP_VISIBLE_DEVICES = '-1' }
$pp.WaitForExit()
$code = $pp.ExitCode
$summary = Join-Path $OutDir 'summary.txt'
if (-not (Test-Path $summary)) {
    Set-Content -Path $summary -Value "PP phase 0: tool_pp_project failed (exit $code), see project.log in $OutDir" -Encoding utf8
}
Get-Content $summary | ForEach-Object { Write-Output $_ }
Say "done: exit $code (0 = GO, 3 = STOP, 2 = inconclusive); summary $summary"
exit $code

