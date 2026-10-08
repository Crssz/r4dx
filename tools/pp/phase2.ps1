#requires -Version 5.1
<#
.SYNOPSIS
  Pipeline-parallel prefill, Phase 2 gates (docs/pp-prefill.md section 8): the real two-GPU pipeline against the monolithic
  prefill -- byte identity (G2a), cold TTFT and unchanged decode (G2b), the soak (G2c) -- in one unattended run, with the
  verdict in <models root>\r4dx\pp\phase2\summary.txt.

.DESCRIPTION
  GPU RUN -- needs the user's approval, and NEVER while an r4dx-* process exists (the pre-flight refuses). Uses BOTH cards
  (HIP_VISIBLE_DEVICES unset for the pipeline -- PpModel puts decode on the last visible ordinal = physical device 1, the
  headless card; =1 for the baselines); the first long prefills load the desktop card for
  seconds at a time, so run it with the user present (docs/pp-prefill.md 4: power, TDR).

    1  identity   build\...\tests\model\test_pp_real_identity.exe (G2a): real PP == monolithic, bit for bit, on the
                  4-layer containers and the real one (plain / --mtp 3 / --dflash / image rows / warm turns / checkpoint /
                  threshold / negative controls). Exit 0 required (77 = skipped counts as a FAILED gate here).
    2  ttft       tools\prefill\ttft_cli.ps1 cold TTFT of the frozen long prompts at 8k, 32k (and 64k) -- TP=1 baseline on
                  device 1 vs -Pp -- median of -Runs; gate G2b: speedup >= 1.6 at 8k and 32k, >= 1.7 at 64k. With -Dflash the
                  same at 8k, 32k with the drafter (split 35).
    3  decode     r4dx-cli short greedy generation, baseline vs --pp 2 (decode must be unchanged within noise: the gate is
                  >= 0.97 of the baseline tok/s, median of 3), and the generated text must be identical.
    4  soak       tools\pp\soak.ps1 --minutes -SoakMinutes (G2c), unless -SkipSoak.

  Output under -OutDir (default <models root>\r4dx\pp\phase2): one log per process, ttft_base\ and ttft_pp\ (ttft.jsonl),
  soak.jsonl and summary.txt. Exit code 0 = every gate passed, 1 = a gate failed, 2 = the run itself failed.
#>
[CmdletBinding()]
param(
    [string]$BuildDir = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'build\win-hip'),
    [string]$Model = '',
    [string]$Drafter = '',
    [string]$OutDir = '',
    [int]$Runs = 2,
    [int]$SoakMinutes = 5,
    [switch]$Skip64k,
    [switch]$Dflash,
    [switch]$SkipIdentity,
    [switch]$SkipTtft,
    [switch]$SkipDecode,
    [switch]$SkipSoak
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$modelsRoot = if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' }
if (-not $Model) { $Model = Join-Path $modelsRoot 'r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx' }
if (-not $Drafter) { $Drafter = Join-Path $modelsRoot 'r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx' }
if (-not $OutDir) { $OutDir = Join-Path $modelsRoot 'r4dx\pp\phase2' }
$identityExe = Join-Path $BuildDir 'tests\model\test_pp_real_identity.exe'
$cli = Join-Path $BuildDir 'src\cli\r4dx-cli.exe'

foreach ($e in @($identityExe, $cli)) {
    if (-not (Test-Path $e)) { throw "[pp-phase2] $e not found: build it (cmake --build $BuildDir --target test_pp_real_identity tool_pp_soak r4dx-cli)" }
}
if (-not (Test-Path $Model)) { throw "[pp-phase2] model container not found: $Model" }
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'r4dx-*' })
if ($busy.Count -gt 0) {
    throw ("[pp-phase2] refusing to run: r4dx process(es) are using the GPUs: " +
           (($busy | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', '))
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$logFile = Join-Path $OutDir 'phase2.log'
function Say([string]$m) {
    $line = "[pp-phase2 $(Get-Date -Format 'HH:mm:ss')] $m"
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}
$verdicts = New-Object System.Collections.Generic.List[string]
$failed = $false
function Gate([string]$name, [bool]$ok, [string]$detail) {
    $script:verdicts.Add(("{0,-6} {1}: {2}" -f $(if ($ok) { 'PASS' } else { 'FAIL' }), $name, $detail))
    if (-not $ok) { $script:failed = $true }
    Say ("{0} {1}: {2}" -f $(if ($ok) { 'PASS' } else { 'FAIL' }), $name, $detail)
}

function Format-Arg([string]$a) {
    if ($a -eq '') { return '""' }
    if ($a -notmatch '[\s"]') { return $a }
    $q = $a -replace '(\\*)"', '$1$1\"'
    $q = $q -replace '(\\+)$', '$1$1'
    return '"' + $q + '"'
}
# Runs $Exe with HIP_VISIBLE_DEVICES=$Hip ('' = UNSET: both cards visible, which is what the pipeline wants -- an EMPTY value
# would hide every card), stdout / stderr to <Prefix>.out.txt / .log; returns the exit code.
function Invoke-Tool([string]$Exe, [string[]]$ArgList, [string]$Prefix, [string]$Hip, [hashtable]$ExtraEnv = @{}) {
    $saved = $env:HIP_VISIBLE_DEVICES
    if ($Hip) { $env:HIP_VISIBLE_DEVICES = $Hip } else { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    $savedExtra = @{}
    foreach ($k in $ExtraEnv.Keys) { $savedExtra[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $ExtraEnv[$k]) }
    try {
        $sp = @{ FilePath = $Exe; NoNewWindow = $true; PassThru = $true; RedirectStandardOutput = "$Prefix.out.txt"; RedirectStandardError = "$Prefix.log" }
        if ($ArgList.Count -gt 0) { $sp.ArgumentList = (($ArgList | ForEach-Object { Format-Arg $_ }) -join ' ') }
        $p = Start-Process @sp
        $null = $p.Handle
        $p.WaitForExit()
        return $p.ExitCode
    } finally {
        $env:HIP_VISIBLE_DEVICES = $saved
        foreach ($k in $savedExtra.Keys) { [Environment]::SetEnvironmentVariable($k, $savedExtra[$k]) }
    }
}
function Median([double[]]$v) {
    if ($v.Count -eq 0) { return [double]::NaN }
    $s = $v | Sort-Object
    return $s[[int][math]::Floor(($s.Count - 1) / 2)]
}

Say "start; build $BuildDir; out $OutDir; head $(git -C $repo rev-parse --short HEAD)"

# ---- 1 identity (G2a) ---------------------------------------------------------------------------------------------
if (-not $SkipIdentity) {
    Say "G2a: test_pp_real_identity (this takes a while: several model loads on two cards)"
    $code = Invoke-Tool $identityExe @() (Join-Path $OutDir 'identity') ''
    $tail = (Get-Content (Join-Path $OutDir 'identity.log') -Tail 4 -ErrorAction SilentlyContinue) -join ' | '
    $fails = @(Select-String -Path (Join-Path $OutDir 'identity.log') -Pattern '^FAIL ' -ErrorAction SilentlyContinue).Count
    Gate 'G2a identity' ($code -eq 0) "exit $code, $fails FAIL line(s); $tail"
}

# ---- 2 TTFT (G2b) -------------------------------------------------------------------------------------------------
function Run-Ttft([string]$Label, [string[]]$Lengths, [switch]$Pp, [string[]]$Extra) {
    $dir = Join-Path $OutDir "ttft_$Label"
    New-Item -ItemType Directory -Force $dir | Out-Null
    # In-process (an array-valued -ExtraArgs survives; a child powershell -File would flatten it to one string).
    $splat = @{ Lengths = $Lengths; Runs = $Runs; Model = $Model; OutDir = $dir; Cli = $cli; Device = 1 }
    if ($Pp) { $splat.Pp = $true }
    if ($Extra.Count -gt 0) { $splat.ExtraArgs = $Extra }
    $savedHip = $env:HIP_VISIBLE_DEVICES
    try { $out = & (Join-Path $repo 'tools\prefill\ttft_cli.ps1') @splat } finally { $env:HIP_VISIBLE_DEVICES = $savedHip }
    $out | Add-Content -Path $logFile
    $jsonl = Join-Path $dir 'ttft.jsonl'
    $res = @{}
    if (Test-Path $jsonl) {
        foreach ($line in Get-Content $jsonl) {
            $r = $line | ConvertFrom-Json
            if ($r.exit -eq 0 -and $r.prefill_s) {
                if (-not $res.ContainsKey($r.length)) { $res[$r.length] = @() }
                $res[$r.length] += [double]$r.prefill_s
            }
        }
    }
    return $res
}
if (-not $SkipTtft) {
    $lens = @('8k', '32k')
    if (-not $Skip64k) { $lens += '64k' }
    foreach ($variant in @(@{ name = 'plain'; extra = @(); lens = $lens }, @{ name = 'dflash'; extra = @('--dflash', $Drafter, '--dflash-k', '7'); lens = @('8k', '32k') })) {
        if ($variant.name -eq 'dflash' -and (-not $Dflash -or -not (Test-Path $Drafter))) { continue }
        Say "G2b: cold TTFT, $($variant.name): baseline (TP=1, device 1) then --pp 2"
        $base = Run-Ttft "base_$($variant.name)" $variant.lens -Extra $variant.extra
        $pp = Run-Ttft "pp_$($variant.name)" $variant.lens -Pp -Extra $variant.extra
        foreach ($len in $variant.lens) {
            if (-not $base.ContainsKey($len) -or -not $pp.ContainsKey($len)) { Gate "G2b ttft $($variant.name) $len" $false 'no measurement'; continue }
            $b = Median $base[$len]
            $p = Median $pp[$len]
            $need = if ($len -eq '64k') { 1.7 } else { 1.6 }
            Gate "G2b ttft $($variant.name) $len" (($b / $p) -ge $need) ("baseline {0:N2} s, pipelined {1:N2} s = x{2:N2} (need >= {3})" -f $b, $p, ($b / $p), $need)
        }
    }
}

# ---- 3 decode unchanged -------------------------------------------------------------------------------------------
if (-not $SkipDecode) {
    Say "G2b: decode tok/s and text, baseline vs --pp 2"
    $prompt = 'Write a short story about a lighthouse keeper who finds a message in a bottle.'
    $common = @('--model', $Model, '--layout', 'trellis', '--prompt', $prompt, '--max-ctx', '4096', '--vision', 'off',
                '--temperature', '0', '--max-tokens', '192', '--stats')
    $rates = @{ base = @(); pp = @() }
    $texts = @{}
    foreach ($mode in @('base', 'pp')) {
        for ($i = 1; $i -le 3; $i++) {
            $prefix = Join-Path $OutDir "decode_${mode}_$i"
            $args2 = $common + $(if ($mode -eq 'pp') { @('--pp', '2') } else { @() })
            $code = Invoke-Tool $cli $args2 $prefix $(if ($mode -eq 'pp') { '' } else { '1' })
            $text = Get-Content "$prefix.log" -Raw -ErrorAction SilentlyContinue
            $m = [regex]::Match($text, 'decode: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\)')
            if ($code -eq 0 -and $m.Success) { $rates[$mode] += [double]$m.Groups[3].Value }
            $texts["$mode$i"] = if (Test-Path "$prefix.out.txt") { Get-Content "$prefix.out.txt" -Raw } else { '' }
        }
    }
    $rb = Median $rates.base
    $rp = Median $rates.pp
    Gate 'G2b decode tok/s' ($rates.base.Count -gt 0 -and $rates.pp.Count -gt 0 -and ($rp / $rb) -ge 0.97) ("baseline {0:N1} tok/s, --pp 2 {1:N1} tok/s (x{2:N3}; need >= 0.97)" -f $rb, $rp, ($rp / $rb))
    Gate 'G2b decode text' (($texts['base1'] -ne '') -and ($texts['base1'] -eq $texts['pp1'])) 'the generated text of --pp 2 equals the baseline''s'
}

# ---- 4 soak (G2c) ---------------------------------------------------------------------------------------------------
if (-not $SkipSoak) {
    Say "G2c: soak, $SoakMinutes minutes"
    $soakJson = Join-Path $OutDir 'soak.jsonl'
    $out = & powershell -NoProfile -File (Join-Path $repo 'tools\pp\soak.ps1') '--model' $Model '--layout' 'trellis' '--minutes' "$SoakMinutes" '--max-ctx' '36864' '--ref-every' '2' '--json' $soakJson
    $out | Add-Content -Path $logFile
    Gate 'G2c soak' ($LASTEXITCODE -eq 0) (($out | Select-Object -Last 2) -join ' | ')
}

$summary = Join-Path $OutDir 'summary.txt'
$text = @("Pipeline-parallel prefill, Phase 2 gates ($(Get-Date -Format 'yyyy-MM-dd HH:mm')), head $(git -C $repo rev-parse --short HEAD)", '') + $verdicts + @('', $(if ($failed) { 'VERDICT: FAIL' } else { 'VERDICT: PASS' }))
$text | Set-Content -Path $summary -Encoding utf8
$text | ForEach-Object { Write-Output $_ }
if ($failed) { exit 1 }
exit 0
