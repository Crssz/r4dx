# tools/r4dx_containers.ps1 -- the ONE place the tools/*.ps1 scripts get their default container
# paths from. Dot-source it:
#
#   . (Join-Path $PSScriptRoot "r4dx_containers.ps1")
#   if (-not $Model) { $Model = Get-R4dxProductionTarget -BuildDir "build\win-hip" }
#
# A container and the binaries that read it are a matched pair: `--layout w4a16` on a container
# packed at a different w4a16 group than the build's R4DX_W4A16_GROUP is refused at load
# (docs/build-windows.md "w4a16 group size"). A script cannot know that group from the source tree
# -- it is a per-build-directory CMake cache value -- so these functions read it from the
# CMakeCache.txt of the build directory whose r4dx-cli.exe / r4dx-server.exe the script is about to
# run, and pick the matching container. Every script still takes an explicit -Model (and -Dflash
# where it has one), which bypasses all of this.
#
# Same resolution rules as the C++ tests' tests/model/test_container_path.h:
#   production pair (Get-R4dxProductionTarget / Get-R4dxProductionDrafter), in D:\models\r4dx:
#     group 64  -> qwen38-27b-v6.r4dx + qwen38-27b-dflash2-w4a16-g64.r4dx
#     group 128 -> qwen38-27b-v3.r4dx + qwen38-27b-dflash2-w4a16.r4dx
#   fixed test containers (Get-R4dxTestContainer -Name <basename>):
#     R4DX_TEST_CONTAINER_DIR\<basename> if that variable is set, else
#     group 128 -> D:\models\r4dx\<basename>, any other -> D:\models\r4dx\g<group>\<basename>

$script:R4dxModelRoot = "D:\models\r4dx"

function Get-R4dxW4a16Group {
    param([Parameter(Mandatory = $true)][string]$BuildDir)
    $cache = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cache)) {
        throw "cannot pick a default container: $cache not found (run .\build.ps1 first, or pass -Model explicitly)"
    }
    $hit = Select-String -Path $cache -Pattern '^R4DX_W4A16_GROUP:[A-Z]*=(\d+)\s*$' | Select-Object -First 1
    if (-not $hit) {
        throw "cannot pick a default container: no R4DX_W4A16_GROUP entry in $cache (pass -Model explicitly)"
    }
    return [int]$hit.Matches[0].Groups[1].Value
}

function Get-R4dxProductionTarget {
    param([Parameter(Mandatory = $true)][string]$BuildDir)
    $name = if ((Get-R4dxW4a16Group -BuildDir $BuildDir) -eq 128) { "qwen38-27b-v3.r4dx" } else { "qwen38-27b-v6.r4dx" }
    return Join-Path $script:R4dxModelRoot $name
}

function Get-R4dxProductionDrafter {
    param([Parameter(Mandatory = $true)][string]$BuildDir)
    $name = if ((Get-R4dxW4a16Group -BuildDir $BuildDir) -eq 128) { "qwen38-27b-dflash2-w4a16.r4dx" } else { "qwen38-27b-dflash2-w4a16-g64.r4dx" }
    return Join-Path $script:R4dxModelRoot $name
}

function Get-R4dxTestContainer {
    param([Parameter(Mandatory = $true)][string]$BuildDir, [Parameter(Mandatory = $true)][string]$Name)
    if ($env:R4DX_TEST_CONTAINER_DIR) { return Join-Path $env:R4DX_TEST_CONTAINER_DIR $Name }
    $group = Get-R4dxW4a16Group -BuildDir $BuildDir
    if ($group -eq 128) { return Join-Path $script:R4dxModelRoot $Name }
    return Join-Path (Join-Path $script:R4dxModelRoot "g$group") $Name
}
