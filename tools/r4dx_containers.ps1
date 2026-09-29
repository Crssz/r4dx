# tools/r4dx_containers.ps1 -- the ONE place the tools/*.ps1 scripts get their default container
# paths from. Dot-source it:
#
#   . (Join-Path $PSScriptRoot "r4dx_containers.ps1")
#   if (-not $Model) { $Model = Get-R4dxProductionTarget }
#
# A container and the binaries that read it are a matched pair: `--layout w4a16` on a container
# packed at a different w4a16 group than the binary's (64) is refused at load
# (docs/build-windows.md "w4a16 group size"). Every script still takes an explicit -Model (and -Dflash
# where it has one), which bypasses all of this.
#
# Same resolution rules as the C++ tests' tests/model/test_container_path.h:
#   production pair (Get-R4dxProductionTarget / Get-R4dxProductionDrafter) and the target's body
#   layout (Get-R4dxProductionLayout), in D:\models\r4dx:
#     huihui-qwen38-27b-abl-trellis-mix45m.r4dx (layout trellis: the Huihui abliterated trellis
#     mix4.5m, docs/huihui.md) + qwen38-27b-dflash2-w4a16-g64.r4dx
#   the tokenizer / chat-template directory (Get-R4dxTokenizerDir): the Huihui HF checkpoint dir, whose
#   four tokenizer files are byte-identical to the base Qwen3.8-27B's (the base checkpoint was retired)
#   fixed test containers (Get-R4dxTestContainer -Name <basename>):
#     R4DX_TEST_CONTAINER_DIR\<basename> if that variable is set, else D:\models\r4dx\g64\<basename>

$script:R4dxModelRoot = "D:\models\r4dx"
$script:R4dxTokenizerDir = "D:\models\Huihui-Qwen3.8-27B-abliterated"

function Get-R4dxTokenizerDir { return $script:R4dxTokenizerDir }

function Get-R4dxProductionTarget {
    return Join-Path $script:R4dxModelRoot "huihui-qwen38-27b-abl-trellis-mix45m.r4dx"
}

# The body layout Get-R4dxProductionTarget's container loads with (a trellis container refuses any other).
function Get-R4dxProductionLayout { return "trellis" }

function Get-R4dxProductionDrafter {
    return Join-Path $script:R4dxModelRoot "qwen38-27b-dflash2-w4a16-g64.r4dx"
}

function Get-R4dxTestContainer {
    param([Parameter(Mandatory = $true)][string]$Name)
    if ($env:R4DX_TEST_CONTAINER_DIR) { return Join-Path $env:R4DX_TEST_CONTAINER_DIR $Name }
    return Join-Path (Join-Path $script:R4dxModelRoot "g64") $Name
}
