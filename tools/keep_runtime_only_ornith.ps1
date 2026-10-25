# Keep ONLY what running Ornith-1.5-122B needs; everything else under E:\models\r4dx that belongs to Ornith can go. Dry run by default.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\keep_runtime_only_ornith.ps1                        # list
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\keep_runtime_only_ornith.ps1 -Delete                # step A: fixtures, spare container, drafter sources
#   ... -Delete -IncludeQuantState      step B: E:\models\r4dx\q35moe (oracles, mix, KL dumps, corpus, logs): ~26 GPU hours of work, IRREVERSIBLE
#   ... -Delete -IncludeCheckpoint      step C: the 233 GiB bf16 checkpoint (re-downloadable, ~250 GB) after copying its small runtime files
#
# KEPT (runtime): ornith-1.5-122b-trellis-mix-kv.r4dx (text), ornith-1.5-122b-trellis-mix-kv-vision.r4dx (text + images),
#   ornith-1.5-122b-dflash2-w4a16.r4dx (DFlash2 drafter), and E:\models\r4dx\Ornith-1.5-122B-runtime (tokenizer + chat template + configs, ~22 MiB;
#   this script creates it and the commands then use --tokenizer-dir E:\models\r4dx\Ornith-1.5-122B-runtime).
# Step A deletes: the first (placeholder-KV) container, the bf16 DFlash2 drafter, the HF drafter folder, the test fixtures (tiny-q35moe, q35moe-tiny*,
#   q35moe-real4, q35moe-fix3, q35moe-pilot, q35moe-repro) and stray logs. After step A the unit tests that load those fixtures cannot run.
# After step B no new container can be built (r4dx-convert reads the oracles and the mix) and no re-allocation / w8a16 / shared-expert quantization is possible.
# After step C no conversion or HF reference run is possible without re-downloading the checkpoint.
# Other models (lfm2, g64, Huihui, Swift15, ...) are never touched.
param([switch]$Delete, [switch]$IncludeQuantState, [switch]$IncludeCheckpoint)
$root = 'E:\models\r4dx'
$ckpt = Join-Path $root 'Ornith-1.5-122B'
$runtimeDir = Join-Path $root 'Ornith-1.5-122B-runtime'
$small = 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'vocab.json', 'merges.txt', 'config.json',
         'preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'

$targets = @(
  'ornith-1.5-122b-trellis-mix.r4dx', 'ornith-1.5-122b-dflash2-bf16.r4dx', 'Ornith-1.5-122B-DFlash2',
  'tiny-q35moe', 'q35moe-tiny', 'q35moe-tiny-short', 'q35moe-real4', 'q35moe-fix3', 'q35moe-pilot', 'q35moe-repro',
  'q35moe-real4-iso.err', 'q35moe-real4-iso.log', 'ornith-dl.err'
) | ForEach-Object { Join-Path $root $_ }
if ($IncludeQuantState) { $targets += Join-Path $root 'q35moe' }
if ($IncludeCheckpoint) {
  if (-not $Delete) { Write-Host "(dry run) the checkpoint small files would be copied to $runtimeDir first" }
  else {
    New-Item -ItemType Directory -Force $runtimeDir | Out-Null
    foreach ($f in $small) { $src = Join-Path $ckpt $f; if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination $runtimeDir -Force } }
    $missing = @($small | Where-Object { (Test-Path -LiteralPath (Join-Path $ckpt $_)) -and -not (Test-Path -LiteralPath (Join-Path $runtimeDir $_)) })
    if ($missing.Count -gt 0 -or -not (Test-Path (Join-Path $runtimeDir 'tokenizer.json'))) { throw "runtime file copy incomplete ($($missing -join ', ')); checkpoint NOT deleted" }
    Write-Host "copied the runtime files to $runtimeDir"
  }
  $targets += $ckpt
} elseif ($Delete) {
  New-Item -ItemType Directory -Force $runtimeDir | Out-Null
  foreach ($f in $small) { $src = Join-Path $ckpt $f; if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination $runtimeDir -Force } }
  Write-Host "copied the runtime files to $runtimeDir (use --tokenizer-dir $runtimeDir from now on)"
}
$total = 0.0
foreach ($t in $targets) {
  if (-not (Test-Path -LiteralPath $t)) { Write-Host ("missing   {0}" -f $t); continue }
  $item = Get-Item -LiteralPath $t -Force
  $bytes = if ($item.PSIsContainer) { (Get-ChildItem -LiteralPath $t -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum } else { $item.Length }
  $gib = $bytes / 1GB
  $total += $gib
  if ($Delete) { Remove-Item -LiteralPath $t -Recurse -Force -Confirm:$false; Write-Host ("deleted   {0,7:N1} GiB  {1}" -f $gib, $t) }
  else { Write-Host ("would del {0,7:N1} GiB  {1}" -f $gib, $t) }
}
Write-Host ("{0} {1:N1} GiB;  E: free now {2:N1} GiB" -f $(if ($Delete) { 'freed' } else { 'would free' }), $total, ((Get-PSDrive E).Free / 1GB))
Write-Host "kept: ornith-1.5-122b-trellis-mix-kv.r4dx, ornith-1.5-122b-trellis-mix-kv-vision.r4dx, ornith-1.5-122b-dflash2-w4a16.r4dx, Ornith-1.5-122B-runtime"
