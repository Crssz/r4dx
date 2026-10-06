# Gemma 4 rotated mix4.5m trellis pipeline (docs/gemma4-plan.md section 9 items 9-13, section 10).
#
# Stages, each runnable alone, in dependency order:
#
#   rotation   CPU  r4dx-convert --rotate q2ab --rotation-out   the fp32 rotation tensors + fingerprint (config.json only)
#   merge      CPU  gen_samples.py --merge                      shard0 + shard1 -> samples.jsonl (+ samples.merge.json)
#   kvcalib    GPU  gemma\kv_calibrate_full.py                  kvcalib.json: per-(layer, kv head) fp8 amax -> descales
#   hessian    GPU  hessian_capture.py --arch gemma4_unified    hessian-v1 (post-norm + weightless rms taps, lm_head)
#   oracle-k4  GPU  trellis_quant.py quantize-model --K 4 --rotation ... --hessian-basis matched
#   oracle-k5  GPU  the same at --K 5
#   mix        CPU  trellis_quant.py mix --bpw 4.5              Gemma ranking (gemma_mix.py: prior x proxy gap, --ranking auto), a manifest only
#   convert    CPU  r4dx-convert --trellis-from ... --rotate q2ab --kv-calib ...   the container, verified
#   gate       GPU+CPU  tool_teacher_forced_logprobs on chat_gemma.json, then kl_report.py --gate gemma-fp32 --base-dir
#
#   .\tools\gemma\trellis_pipeline.ps1 -Stage all                 # print every stage's exact commands, run nothing
#   .\tools\gemma\trellis_pipeline.ps1 -Stage hessian             # print; refuses (exit 3): a GPU stage needs -Run
#   .\tools\gemma\trellis_pipeline.ps1 -Stage hessian -Run        # really run it, on -Device (default 1)
#   .\tools\gemma\trellis_pipeline.ps1 -Stage oracle-k5 -Run -Device 0 -AllowGpu0   # device 0 only with the user's go-ahead
#
# Without -Run NOTHING executes (every stage only prints its commands and whether its inputs are ready). A GPU
# stage without -Run prints its commands and exits 3 ("refused"). With -Run a stage executes only if its
# prerequisites hold (otherwise it throws before touching anything), skips itself when its output is already
# complete (-Force redoes it), and logs to <Root>\logs\pipeline_<stage>_<time>.log.
#
# Device rule: GPU stages default to HIP device 1 (the headless R9700). Device 0 drives the desktop and is accepted
# only with -AllowGpu0, which also sets R4DX_ALLOW_GPU0 / R4DX_REF_ALLOWED_DEVICES for the tools that need them.
# hessian_capture.py accepts device 1 only. CPU stages run with HIP_VISIBLE_DEVICES=-1.
#
# Python: the reference venv (transformers with gemma4_unified) for kvcalib / merge / hessian / gate; the oracle
# (quantize-model) needs torch.linalg.cholesky on CUDA, which torch 2.9.1+rocmsdk (the reference venv) cannot do
# ("requires compiling PyTorch with MAGMA", <models-root>\r4dx\huihui\RECIPE.md addendum), so -OraclePython defaults to
# <models-root>\r4dx\huihui\venv-rocm10 (torch 2.13+rocm10.0.0) when it exists. The oracle imports no transformers.
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('rotation', 'merge', 'kvcalib', 'hessian', 'oracle-k4', 'oracle-k5', 'mix', 'convert', 'gate', 'all')]
  [string]$Stage,
  [switch]$Run,
  [switch]$Force,
  [string]$Device = '1',
  [switch]$AllowGpu0,
  [string]$Root = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui-gemma",
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\Huihui-gemma-4-12B-it-abliterated",
  [string]$Python = 'D:\venvs\r4dx-gemma-ref\Scripts\python.exe',
  [string]$OraclePython = '',
  [string]$Exe = '',                    # r4dx-convert; default: this checkout's build\win-hip-dry or build\win-hip
  [string]$TeacherForced = '',          # tool_teacher_forced_logprobs.exe; default: this checkout's build\win-hip
  [ValidateSet('q2a', 'q2ab')][string]$Rotate = 'q2ab',
  [string]$RotationSeed = '',           # '' = the converter's default 0x5EED2025; the SAME value is passed to every step
  [double]$Bpw = 4.5,
  [int]$KvSeqLen = 4096,
  [int]$HessianSeqLen = 4096,
  [ValidateSet('fp8', 'bf16_full', 'bf16')][string]$Kv = 'fp8',   # R4DX_GEMMA_KV of the gate dump
  [string]$BaseDump = '',               # gate: the r4dx bf16-container dump the quantized one is compared to
  [switch]$SkipDump,                    # gate: only the CPU report (the dump must exist)
  [string]$TestDump = '',               # gate: report this dump instead of <Root>\kl\r4dx-trellis-<name>
  [switch]$AllowPartialCorpus,          # merge: do not require every prompt (a shard still running)
  [int]$Threads = 32
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$env:PYTHONIOENCODING = 'utf-8'

# ---- paths ------------------------------------------------------------------------------------------
$tag      = "$($Bpw.ToString('0.0##', [Globalization.CultureInfo]::InvariantCulture))mr"        # 4.5mr: matched basis, rotated
$corpus   = Join-Path $Root 'corpus'
$shards   = @((Join-Path $corpus 'samples.shard0.jsonl'), (Join-Path $corpus 'samples.shard1.jsonl'))
$samples  = Join-Path $corpus 'samples.jsonl'
$kvcalib  = Join-Path $Root 'kvcalib.json'
$hess     = Join-Path $Root 'hessian-v1'
$tdir     = Join-Path $Root 'trellis'
$rotFile  = Join-Path $tdir ("rot-{0}.safetensors" -f $Rotate)
$qdir     = Join-Path $tdir 'q'
$k4       = Join-Path $qdir 'K4mr'
$k5       = Join-Path $qdir 'K5mr'
$mixdir   = Join-Path $qdir "mix$tag"
$container = Join-Path $Root ("huihui-gemma-trellis-mix{0}.r4dx" -f ($tag -replace '\.', ''))
$logdir   = Join-Path $Root 'logs'
$kvdir    = Join-Path $Root 'kl'
$dumpdir  = Join-Path $kvdir ("r4dx-trellis-mix{0}-kv{1}" -f ($tag -replace '\.', ''), $Kv)
if (-not $BaseDump) { $BaseDump = Join-Path $kvdir 'r4dx-bf16kv-f32res' }
if ($TestDump) { $dumpdir = $TestDump }
$truth    = Join-Path $kvdir 'fp32\truth'
$noise    = Join-Path $kvdir 'fp32\bf16sdpa'
$chatTok  = 'tools\reference\kl_corpus\chat_gemma.json'
$rawTok   = 'tools\reference\kl_corpus\tokens_gemma.json'
$venvRocm10 = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui\venv-rocm10\Scripts\python.exe"
if (-not $OraclePython) { $OraclePython = if (Test-Path $venvRocm10) { $venvRocm10 } else { $Python } }
if (-not $Exe) {
  foreach ($c in 'build\win-hip-dry\src\convert\r4dx-convert.exe', 'build\win-hip\src\convert\r4dx-convert.exe') {
    if (Test-Path (Join-Path $repo $c)) { $Exe = Join-Path $repo $c; break }
  }
  if (-not $Exe) { $Exe = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe' }
}
if (-not $TeacherForced) { $TeacherForced = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }

$script:refused = $false
$script:problems = @()

# ---- helpers ----------------------------------------------------------------------------------------
function Q([string]$a) { if ($a -match '^[A-Za-z0-9_\-\.\\:/=,]+$') { $a } else { "'" + $a.Replace("'", "''") + "'" } }
function Fmt([string]$exe, [string[]]$a) { (@((Q $exe)) + ($a | ForEach-Object { Q $_ })) -join ' ' }
function EnvPrefix([hashtable]$e) { ($e.GetEnumerator() | Sort-Object Name | ForEach-Object { "`$env:$($_.Name)='$($_.Value)'" }) -join '; ' }

function GpuEnv([string]$tool) {
  $e = @{ HIP_VISIBLE_DEVICES = $Device }
  if ($Device -ne '1') {
    $e['R4DX_ALLOW_GPU0'] = '1'
    $e['R4DX_REF_ALLOWED_DEVICES'] = '0,1'
  }
  return $e
}
$cpuEnv = @{ HIP_VISIBLE_DEVICES = '-1'; CUDA_VISIBLE_DEVICES = '-1' }

function PyJson([string]$path, [string]$expr) {
  # json via python: PS 5.1's ConvertFrom-Json refuses keys that differ only in case ('K' / 'k' in oracle manifests)
  $code = "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print($expr)"
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { $o = & $Python -c $code $path 2>$null } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { return $null }
  return ("$o").Trim()
}
function OracleComplete([string]$dir) {
  $m = Join-Path $dir 'weights_override.json'
  if (-not (Test-Path $m)) { return $false }
  return ((PyJson $m "d['complete']") -eq 'True')
}
function CorpusComplete {
  $m = Join-Path $corpus 'samples.merge.json'
  return ((Test-Path $samples) -and (Test-Path $m) -and ((PyJson $m "d['complete']") -eq 'True'))
}
function GenRunning {
  $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -match 'gen_samples\.py' -and $_.CommandLine -notmatch '--merge' }
  return @($p)
}
function BusyWith([string]$pattern) {
  $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -match $pattern }
  return @($p)
}
function FreeGiB([string]$path) {
  $root = [IO.Path]::GetPathRoot($path)
  return [math]::Round((Get-PSDrive $root.Substring(0, 1)).Free / 1GB, 1)
}
function Need([bool]$ok, [string]$msg) { if (-not $ok) { $script:problems += $msg } }

function Execute([string]$name, [hashtable]$e, [string]$exe, [string[]]$a, [bool]$gpu) {
  Write-Host ("[pipeline] {0} CMD: {1}; {2}" -f $name, (EnvPrefix $e), (Fmt $exe $a))
  if (-not $Run) {
    if ($gpu) {
      Write-Host "[pipeline] $name REFUSED: a GPU stage executes only with -Run (the command above is what it would run)"
      $script:refused = $true
    }
    return $true
  }
  New-Item -ItemType Directory -Force $logdir | Out-Null
  $log = Join-Path $logdir ("pipeline_{0}_{1:yyyyMMdd-HHmmss}.log" -f $name, (Get-Date))
  Write-Host "[pipeline] $name running, log $log"
  $saved = @{}
  foreach ($k in $e.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $e[$k]) }
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # PS 5.1: native stderr lines are error records
  $t0 = Get-Date
  try { & $exe @a *> $log } finally {
    $ErrorActionPreference = $prev
    foreach ($k in $e.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  }
  $rc = $LASTEXITCODE
  $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
  if ($rc -ne 0) { throw "[pipeline] $name failed (exit $rc) after $min min, see $log" }
  Write-Host "[pipeline] $name done in $min min"
  return $true
}

function Open-Stage([string]$name, [bool]$gpu, [string]$est) {
  $script:problems = @()
  Write-Host ""
  Write-Host ("[pipeline] ===== {0} ({1}) est. {2}" -f $name, $(if ($gpu) { "GPU, device $Device" } else { 'CPU' }), $est)
  if ($gpu) {
    if ($Device -notmatch '^[01]$') { $script:problems += "-Device must be 0 or 1, got '$Device'" }
    if ($Device -eq '0' -and -not $AllowGpu0) { $script:problems += 'device 0 drives the desktop: pass -AllowGpu0 only with the user''s go-ahead' }
    $gen = GenRunning
    if ($gen.Count -gt 0) { $script:problems += "corpus generation is still running (gen_samples.py pid $(($gen | ForEach-Object ProcessId) -join ',')): wait for it, merge, then run the GPU stages" }
  }
}

function Close-Stage([string]$name, [bool]$gpu) {
  if ($script:problems.Count -gt 0) {
    foreach ($p in $script:problems) { Write-Host "[pipeline] $name NOT READY: $p" }
    if ($Run) { throw "[pipeline] $name refused: $($script:problems.Count) prerequisite problem(s) above" }
  }
}

# ---- stages -----------------------------------------------------------------------------------------
function Stage-Rotation {
  Open-Stage 'rotation' $false 'seconds (reads config.json only)'
  $a = @('--input', $Model, '--rotate', $Rotate, '--rotation-out', $rotFile)
  if ($RotationSeed) { $a += @('--rotation-seed', $RotationSeed) }
  Need (Test-Path $Exe) "converter not found at $Exe (build target r4dx-convert from this checkout)"
  if ((Test-Path $rotFile) -and -not $Force) { Write-Host "[pipeline] rotation: $rotFile exists, skipping (-Force redoes it)"; Close-Stage 'rotation' $false; return }
  New-Item -ItemType Directory -Force $tdir -ErrorAction SilentlyContinue | Out-Null
  Close-Stage 'rotation' $false
  Execute 'rotation' $cpuEnv $Exe $a $false | Out-Null
}

function Stage-Merge {
  Open-Stage 'merge' $false 'seconds'
  $a = @('tools\reference\gemma\gen_samples.py', '--merge') + $shards + @('--out', $samples)
  if (-not $AllowPartialCorpus) { $a += '--require-complete' }
  foreach ($s in $shards) { Need (Test-Path $s) "missing shard $s" }
  $gen = GenRunning
  if ($gen.Count -gt 0 -and -not $AllowPartialCorpus) { $script:problems += "shard generation still running (pid $(($gen | ForEach-Object ProcessId) -join ',')): the merge needs every prompt (-AllowPartialCorpus for a partial one, NOT for the real run)" }
  if ((CorpusComplete) -and -not $Force) { Write-Host "[pipeline] merge: $samples is already merged and complete, skipping (-Force redoes it)"; Close-Stage 'merge' $false; return }
  Close-Stage 'merge' $false
  Execute 'merge' $cpuEnv $Python $a $false | Out-Null
}

function Stage-KvCalib {
  Open-Stage 'kvcalib' $true '5-15 min (450K tokens through the 48-layer stack, layer-major, one read of the 24 GB checkpoint)'
  $a = @('tools\reference\gemma\kv_calibrate_full.py', '--model-dir', $Model, '--gen-file', $samples,
         '--max-seq-len', "$KvSeqLen", '--device', 'cuda', '--out', $kvcalib)
  Need (CorpusComplete) "the merged complete corpus $samples is missing (run -Stage merge after shard 0 finishes)"
  if ((Test-Path $kvcalib) -and -not $Force) {
    $full = (PyJson "$kvcalib.summary.json" "d['complete']") -eq 'True'
    if ($full) { Write-Host "[pipeline] kvcalib: $kvcalib is complete, skipping (-Force redoes it)"; Close-Stage 'kvcalib' $true; return }
  }
  Close-Stage 'kvcalib' $true
  Execute 'kvcalib' (GpuEnv 'kv') $Python $a $true | Out-Null
}

function Stage-Hessian {
  Open-Stage 'hessian' $true '20-35 min (Qwen 27B, 835K tokens took 67 min; scaled by tokens x params, plus the K=15360 taps)'
  if ($Device -ne '1') { $script:problems += 'hessian_capture.py refuses every device but 1' }
  $a = @('tools\reference\hessian_capture.py', '--arch', 'gemma4_unified', '--model-dir', $Model, '--out-dir', $hess,
         '--rms-taps', '--gen-file', $samples, '--seq-len', "$HessianSeqLen")
  Need (CorpusComplete) "the merged complete corpus $samples is missing (run -Stage merge after shard 0 finishes)"
  Need ((FreeGiB $hess) -ge 32) "less than 32 GiB free for $hess (estimate 28.7 GiB, 1.1 x needed)"
  $hj = Join-Path $hess 'hessian.json'
  if ((Test-Path $hj) -and -not $Force) {
    $smoke = PyJson $hj "d.get('smoke')"; $stub = PyJson $hj "d.get('stub')"
    if ($smoke -ne 'True' -and $stub -ne 'True') { Write-Host "[pipeline] hessian: $hj exists (not a smoke/stub set), skipping (-Force redoes it)"; Close-Stage 'hessian' $true; return }
    $a += '--force'
  }
  Close-Stage 'hessian' $true
  Execute 'hessian' (GpuEnv 'hessian') $Python $a $true | Out-Null
}

function Stage-Oracle([int]$K) {
  $name = "oracle-k$K"
  $dir = if ($K -eq 4) { $k4 } else { $k5 }
  Open-Stage $name $true '30-45 min (Qwen K4m/K5m took 74 min for 24.3 G weights; Gemma has 10.9 G); K4 and K5 in parallel on both GPUs = the same wall time as one'
  $a = @('tools\reference\trellis_quant.py', 'quantize-model', '--arch', 'gemma4_unified', '--model-dir', $Model,
         '--device', 'cuda', '--K', "$K", '--hessian-basis', 'matched', '--hessian-dir', $hess,
         '--rotation', $rotFile, '--out-dir', $dir)
  $hj = Join-Path $hess 'hessian.json'
  Need (Test-Path $hj) "no Hessians at $hess (run -Stage hessian)"
  if (Test-Path $hj) {
    Need ((PyJson $hj "d.get('stub')") -ne 'True') "$hess is a header-only STUB (make_stub_hessian.py), not a capture"
    Need ((PyJson $hj "d.get('smoke')") -ne 'True') "$hess is a --layers smoke capture"
    Need ((PyJson $hj "'rms_keys' in d") -eq 'True') "$hess has no rms_keys (the rotated oracle needs the --rms-taps capture)"
  }
  Need (Test-Path $rotFile) "no rotation file $rotFile (run -Stage rotation)"
  Need ((FreeGiB $dir) -ge 20) "less than 20 GiB free next to $dir"
  Need (Test-Path $OraclePython) "no python at $OraclePython"
  if ($OraclePython -eq $Python) { Write-Host "[pipeline] WARNING: -OraclePython is the reference venv (torch 2.9.1): torch.linalg.cholesky on CUDA needs MAGMA there and will fail; use $venvRocm10" }
  $busy = BusyWith ([regex]::Escape($dir))
  if ($busy.Count -gt 0) { $script:problems += "another process already works on $dir (pid $(($busy | ForEach-Object ProcessId) -join ','))" }
  $other = BusyWith 'hessian_capture\.py|kv_calibrate_full\.py|trellis_quant\.py'
  $other = @($other | Where-Object { $_.CommandLine -notmatch [regex]::Escape($dir) })
  if ($other.Count -gt 0 -and $Device -eq '1') { $script:problems += "another reference GPU job runs (pid $(($other | ForEach-Object ProcessId) -join ',')); if it is on device 0 pass -Device 0 -AllowGpu0 for this stage instead, else wait" }
  if ((OracleComplete $dir) -and -not $Force) { Write-Host "[pipeline] ${name}: $dir is complete, skipping (-Force redoes it; a re-run resumes per layer)"; Close-Stage $name $true; return }
  Close-Stage $name $true
  Execute $name (GpuEnv 'oracle') $OraclePython $a $true | Out-Null
  if ($Run -and -not (OracleComplete $dir)) { throw "[pipeline] ${name}: $dir is not complete after quantize-model" }
}

function Stage-Mix {
  Open-Stage 'mix' $false 'seconds (a manifest; no requantization)'
  $a = @('tools\reference\trellis_quant.py', 'mix', '--bpw', "$Bpw", '--ranking', 'auto', '--src', $k4, '--src', $k5, '--out-dir', $mixdir)
  Need (OracleComplete $k4) "$k4 is not a complete oracle directory (run -Stage oracle-k4)"
  Need (OracleComplete $k5) "$k5 is not a complete oracle directory (run -Stage oracle-k5)"
  if ((OracleComplete $mixdir) -and -not $Force) { Write-Host "[pipeline] mix: $mixdir exists, skipping (-Force redoes it)"; Close-Stage 'mix' $false; return }
  Close-Stage 'mix' $false
  Execute 'mix' $cpuEnv $OraclePython $a $false | Out-Null
  if ($Run) {
    $cnt = PyJson (Join-Path $mixdir 'weights_override.json') "d['allocation']['tensors_per_K'], round(d['summary']['bpw'],4), d['allocation']['rank']"
    Write-Host "[pipeline] mix: tensors per K, measured bpw, rank: $cnt"
  }
}

function Stage-Convert {
  Open-Stage 'convert' $false '2-4 min (Qwen: 80 s) plus the full reconstruction check'
  $manifest = Join-Path $mixdir 'weights_override.json'
  $sha = '<sha256 of weights_override.json>'
  if (Test-Path $manifest) { $sha = (Get-FileHash $manifest -Algorithm SHA256).Hash.ToLower() }
  $a = @('--input', $Model, '--output', $container,
         '--trellis-from', $mixdir, '--trellis-manifest-sha256', $sha, '--trellis-verify', 'full',
         '--rotate', $Rotate)
  if ($RotationSeed) { $a += @('--rotation-seed', $RotationSeed) }
  $a += @('--kv-calib', $kvcalib, '--no-bf16', '--lm-head', 'w4a16', '--w4a16-group-rule', '^lm_head$=32',
          '--hessian-dir', $hess, '--ldlq', '^lm_head$', '--threads', "$Threads")
  Need (Test-Path $Exe) "converter not found at $Exe"
  Need (OracleComplete $mixdir) "$mixdir is not a complete mix manifest (run -Stage mix)"
  Need (Test-Path $kvcalib) "no $kvcalib (run -Stage kvcalib): without --kv-calib every descale is the placeholder 1.0"
  Need (Test-Path (Join-Path $hess 'lm_head.hess')) "no lm_head.hess in $hess (run -Stage hessian)"
  Need (Test-Path $rotFile) "no rotation file $rotFile (run -Stage rotation); its seed must equal this run's"
  Need ((FreeGiB $container) -ge 25) "less than 25 GiB free next to $container"
  if ((Test-Path $mixdir) -and (Test-Path $rotFile)) {
    $mrot = PyJson $manifest "d.get('rotation')"
    Write-Host "[pipeline] convert: oracle manifest rotation fingerprint $mrot (the converter checks it against its own --rotate/--rotation-seed)"
  }
  if ((Test-Path $container) -and -not $Force) { Write-Host "[pipeline] convert: $container exists, skipping (-Force redoes it)"; Close-Stage 'convert' $false; return }
  Close-Stage 'convert' $false
  Execute 'convert' $cpuEnv $Exe $a $false | Out-Null
  if ($Run) {
    & $Python (Join-Path $repo 'tools\gemma\check_kvcalib_container.py') $container $kvcalib
    if ($LASTEXITCODE -ne 0) { throw '[pipeline] convert: the container descales do not match kvcalib.json' }
    & $Python (Join-Path $repo 'tools\quant2\decode_bytes.py') $container --by-class
  }
}

function Stage-Gate {
  Open-Stage 'gate' (-not $SkipDump) '3-6 min dump (chat_gemma.json, 10 sequences, r4dx decode path) + 1 min CPU report'
  $dumpArgs = @('--model', $container, '--layout', 'trellis', '--tokens', $chatTok, '--out-dir', $dumpdir, '--max-ctx', '8192')
  $rep = @('tools\reference\kl_report.py', '--gate', 'gemma-fp32', '--truth-dir', $truth, '--noise-dir', $noise,
           '--test-dir', $dumpdir, '--base-dir', $BaseDump, '--tokens', $chatTok,
           '--out', (Join-Path $dumpdir 'gate.json'))
  Need (Test-Path $truth) "no fp32 truth dump at $truth"
  Need (Test-Path $noise) "no noise dump at $noise"
  Need (Test-Path $BaseDump) "no r4dx bf16 base dump at $BaseDump (the increment is measured against it)"
  if (-not $SkipDump) {
    Need (Test-Path $container) "no container $container (run -Stage convert)"
    Need (Test-Path $TeacherForced) "tool_teacher_forced_logprobs.exe not found at $TeacherForced (build it: it is a GPU exe)"
    Write-Host "[pipeline] gate NOTE: the base dump $BaseDump used R4DX_GEMMA_KV=bf16 and fp32 residual; this dump uses R4DX_GEMMA_KV=$Kv (the container's --kv-calib descales apply to the fp8 modes), so the increment includes the KV cost"
    $e = GpuEnv 'gate'; $e['R4DX_GEMMA_KV'] = $Kv
    if ($Run) { New-Item -ItemType Directory -Force $dumpdir | Out-Null }
    Close-Stage 'gate' $true
    Execute 'gate-dump' $e $TeacherForced $dumpArgs $true | Out-Null
  } else {
    Need (Test-Path $dumpdir) "no dump to report at $dumpdir"
    Close-Stage 'gate' $false
  }
  Execute 'gate-report' $cpuEnv $Python $rep $false | Out-Null
  Write-Host "[pipeline] gate: the verdict is the per-group table of kl_report.py (increment <= 0.01 in english, thai, code and chat-ALL; top-1 floors; raw groups report only)"
}

$order = @('rotation', 'merge', 'kvcalib', 'hessian', 'oracle-k4', 'oracle-k5', 'mix', 'convert', 'gate')
$todo = if ($Stage -eq 'all') { $order } else { @($Stage) }
if (-not $Run) { Write-Host '[pipeline] PLAN ONLY: no -Run, nothing is executed' }
foreach ($s in $todo) {
  switch ($s) {
    'rotation'  { Stage-Rotation }
    'merge'     { Stage-Merge }
    'kvcalib'   { Stage-KvCalib }
    'hessian'   { Stage-Hessian }
    'oracle-k4' { Stage-Oracle 4 }
    'oracle-k5' { Stage-Oracle 5 }
    'mix'       { Stage-Mix }
    'convert'   { Stage-Convert }
    'gate'      { Stage-Gate }
  }
}
if ($script:refused) { exit 3 }
