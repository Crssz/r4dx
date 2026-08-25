#!/usr/bin/env bash
# Build one variant of the 4-bit GEMM as a standalone importable module.
#   ./w4build.sh <name> [extra hipcc flags...]
# e.g. ./w4build.sh v1        ./w4build.sh nt -DW4_NT=1
set -euo pipefail
cd "$(dirname "$0")"
NAME=$1; shift
INC=$(python3 -m pybind11 --includes)
FLAGS="-O3 -std=c++17 -fPIC --offload-arch=${GFX_ARCH:-gfx1201} -ffp-contract=off -Wno-unused-result"
hipcc $FLAGS $INC "$@" -DTU_NAME=w4_${NAME} -DR4D_W4_TAG="\"${NAME}\"" \
      -c ../r4d_gemm_w4a16_nt_m64.hip -o /tmp/w4_${NAME}_k.o
hipcc $FLAGS $INC "$@" -DTU_NAME=w4_${NAME} -DR4D_W4_TAG="\"${NAME}\"" \
      -c w4mod.hip -o /tmp/w4_${NAME}_m.o
hipcc -shared --offload-arch="${GFX_ARCH:-gfx1201}" /tmp/w4_${NAME}_k.o /tmp/w4_${NAME}_m.o \
      -o "w4_${NAME}.so"
echo "[w4build] $(ls -la w4_${NAME}.so)"
