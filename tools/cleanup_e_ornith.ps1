# Reclaim E: space from Ornith leftovers. Dry run by default; deletes only with -Delete.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\cleanup_e_ornith.ps1                    # list the default set
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\cleanup_e_ornith.ps1 -Delete            # delete the default set
#   ... -Delete -IncludeRealPrefix -IncludeKlDumps                                                     # also the optional sets
# Default set (nothing a runtime or a new conversion reads):
#   q35moe-fix3        scratch of the quantization session
#   the FIRST container ornith-1.5-122b-trellis-mix.r4dx (placeholder KV; the -kv container has the same weights and real KV scales)
#   q35moe-pilot, q35moe-tiny-short   old pilot / short fixtures
#   q35moe\kl\r4dx_placeholder        KL dump of that first container (regenerable)
# -IncludeRealPrefix  q35moe-real4 (the 4-layer real-weight prefix + goldens; only the real-prefix tests use it; rebuilding takes hours)
# -IncludeKlDumps     q35moe\kl\hf_bf16, hf_fp32, r4dx_kv (the KL gate's references; ~8-12 min of GPU each to regenerate)
# NEVER touched: Ornith-1.5-122B (the bf16 checkpoint), q35moe\oracle-k3 / oracle-k4 and q35moe\mix (r4dx-convert reads them for ANY new
#   container: vision, shared experts, re-allocation), the -kv and -kv-vision containers, the DFlash2 drafters, the tiny-q35moe test fixtures.
param([switch]$Delete, [switch]$IncludeRealPrefix, [switch]$IncludeKlDumps)
$targets = @(
  'E:\models\r4dx\q35moe-fix3',
  'E:\models\r4dx\ornith-1.5-122b-trellis-mix.r4dx',
  'E:\models\r4dx\q35moe-pilot',
  'E:\models\r4dx\q35moe-tiny-short',
  'E:\models\r4dx\q35moe\kl\r4dx_placeholder'
)
if ($IncludeRealPrefix) { $targets += 'E:\models\r4dx\q35moe-real4' }
if ($IncludeKlDumps) { $targets += 'E:\models\r4dx\q35moe\kl\hf_bf16', 'E:\models\r4dx\q35moe\kl\hf_fp32', 'E:\models\r4dx\q35moe\kl\r4dx_kv' }
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
