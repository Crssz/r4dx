#!/usr/bin/env bash
# Build one variant of the W4A8 GEMM as a standalone importable module.
#   ./w4a8build.sh <name> [extra hipcc flags...]
set -euo pipefail
cd "$(dirname "$0")"
NAME=$1; shift
INC=$(python3 -m pybind11 --includes)
FLAGS="-O3 -std=c++17 -fPIC --offload-arch=${GFX_ARCH:-gfx1201} -ffp-contract=off -Wno-unused-result"
hipcc $FLAGS $INC "$@" -DTU_NAME=a8_${NAME} -DR4D_W4_TAG="\"${NAME}\"" \
      -c ../r4d_gemm_w4a8_nt_m64.hip -o /tmp/a8_${NAME}_k.o
hipcc $FLAGS $INC "$@" -DTU_NAME=a8_${NAME} -DR4D_W4_TAG="\"${NAME}\"" \
      -c ../r4d_quant_act_i8.hip -o /tmp/a8_${NAME}_q.o
hipcc $FLAGS $INC "$@" -DTU_NAME=a8_${NAME} -DR4D_W4_TAG="\"${NAME}\"" \
      -c w4a8mod.hip -o /tmp/a8_${NAME}_m.o
hipcc -shared --offload-arch="${GFX_ARCH:-gfx1201}" /tmp/a8_${NAME}_k.o /tmp/a8_${NAME}_q.o /tmp/a8_${NAME}_m.o \
      -o "a8_${NAME}.so"
echo "[w4a8build] $(ls -la a8_${NAME}.so)"
