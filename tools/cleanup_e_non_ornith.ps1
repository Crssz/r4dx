# Reclaim E: space from NON-Ornith leftovers that no model runtime needs. Dry run by default; deletes only with -Delete.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\cleanup_e_non_ornith.ps1                       # list what would go
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\cleanup_e_non_ornith.ps1 -Delete               # delete the default set
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\cleanup_e_non_ornith.ps1 -Delete -IncludeLfm2Variants
# Default set: E:\models\r4dx\lfm2\kl (reference KL dumps of the LFM2 track; a container does not read them).
# -IncludeLfm2Variants adds the four LFM2 quantization-sweep containers (trellis4.75, trellis45, trellis5.0, trellis5.0-expertsonly).
# NOT touched: every Ornith path, the production containers (Huihui / Swift15 27B, LFM2 bf16 / trellis-hybrid / w8a16-g128 / l6),
# E:\models\r4dx\g64 (the test-container directory the test suite and smoke scripts resolve), and E:\vid (personal files).
param([switch]$Delete, [switch]$IncludeLfm2Variants)
$targets = @('E:\models\r4dx\lfm2\kl')
if ($IncludeLfm2Variants) {
  $targets += 'E:\models\r4dx\lfm2\lfm2-trellis4.75.r4dx', 'E:\models\r4dx\lfm2\lfm2-trellis45.r4dx',
              'E:\models\r4dx\lfm2\lfm2-trellis5.0.r4dx', 'E:\models\r4dx\lfm2\lfm2-trellis5.0-expertsonly.r4dx'
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
