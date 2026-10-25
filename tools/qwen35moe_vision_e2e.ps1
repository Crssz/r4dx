<#
.SYNOPSIS
  End-to-end image check of the Ornith vision path on the real container (docs/qwen35moe-vision.md): r4dx-cli on the container
  with the tower, describing the two images of the model repo's assets/ (no download), plus text-only runs on the vision container
  and the old text-only container (greedy; the outputs must be byte-identical, the decode speed equal).
  GPU job: run it through   tools\with_lock.ps1 -Name gpu -Run "powershell -NoProfile -File tools\qwen35moe_vision_e2e.ps1"
  (both cards; R4DX_ALLOW_GPU0=1 is set here). One CLI process per case, ~2 min each (load 75-95 s).
#>
param(
  [string]$Vision = 'E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv-vision.r4dx',
  [string]$Text = 'E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx',
  [string]$Tok = 'E:\models\r4dx\Ornith-1.5-122B',
  [string]$OutDir = "$env:TEMP\q35v\e2e",
  [string]$Cases = 'logo,chart,text_vision,text_plain',
  [int]$MaxPixels = 1000000
)
$ErrorActionPreference = 'Continue'
$env:R4DX_ALLOW_GPU0 = '1'
Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $OutDir | Out-Null
$repo = Split-Path -Parent $PSScriptRoot
$cli = Join-Path $repo 'build\win-hip\src\cli\r4dx-cli.exe'
$assets = Join-Path $Tok 'assets'
$textPrompt = 'Write a haiku about GPUs, then explain what a GPU is in two sentences.'
$all = @{
  logo        = @($Vision, @('--image', "$assets\ornith_logo.png", '--prompt', 'Describe this image in two or three sentences, including any text you can read.'), 160)
  chart       = @($Vision, @('--image', "$assets\ornith_122b_eval.png", '--prompt', 'What is the title of this chart, and which benchmark shows the highest score for the first model in the legend? Give the number.'), 160)
  # DFlash2 with an image in the prompt: the drafter ropes at index + the mrope delta, the verify windows take explicit rope rows; greedy
  # output must equal the plain `chart` run token for token.
  chart_dflash = @($Vision, @('--image', "$assets\ornith_122b_eval.png", '--prompt', 'What is the title of this chart, and which benchmark shows the highest score for the first model in the legend? Give the number.',
                              '--dflash', 'E:\models\r4dx\ornith-1.5-122b-dflash2-w4a16.r4dx', '--dflash-k', '11', '--dflash-p-min', '0.8'), 160)
  text_vision = @($Vision, @('--prompt', $textPrompt), 96)
  text_plain  = @($Text, @('--prompt', $textPrompt), 96)
}
foreach ($name in $Cases.Split(',')) {
  $c = $all[$name]
  $args = @('--model', $c[0], '--layout', 'trellis', '--tokenizer-dir', $Tok, '--max-ctx', '8192', '--max-tokens', "$($c[2])",
            '--temperature', '0', '--think', 'off', '--stats', '--image-max-pixels', "$MaxPixels", '--dump-token-ids', (Join-Path $OutDir "$name.tokens.json")) + $c[1]
  Write-Host "[e2e] $name"
  $o = Join-Path $OutDir "$name.out.txt"; $e = Join-Path $OutDir "$name.err.txt"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $cli @args > $o 2> $e
  Write-Host "[e2e] $name exit $LASTEXITCODE in $([int]$sw.Elapsed.TotalSeconds) s"
}
