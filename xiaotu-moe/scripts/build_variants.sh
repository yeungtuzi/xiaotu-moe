#!/usr/bin/env bash
# Build the xiaotu-moe ISA-specific extension variants for Phase 4.
#
# One source (csrc/python_binding/binding.cpp) is compiled several times, once
# per ISA level; each gets a distinct pybind module name
# `_xiaotu_moe_C<SUFFIX>` (see binding.cpp's PYBIND11_MODULE token-paste) and a
# matching `_xiaotu_moe_C<SUFFIX>.cpython-*.so` on disk. The Python-side
# `_dynamic_loader` inspects the host CPU and imports the best supported
# variant at runtime.
#
# Variant ladder (higher = newer instructions, picked first if supported):
#   scalar          no SIMD (portable fallback)
#   avx2            -mavx2 -mfma
#   avx512_base     -mavx512f -mavx512bw -mavx512vl -mavx512dq
#   avx512_vnni     base + -mavx512vnni
#   avx512_bf16     base + -mavx512bf16   (fp32-accumulate; host EPYC 9654 HAS it)
#   avx512_amx      base + AMX            (SKIPPED: no AMX on this AMD host; the
#                                          runtime loader simply won't pick it)
#
# Usage:
#   PYTHON=/home/user/lvllm/.search-venv/bin/python scripts/build_variants.sh
#   PYTHON=/home/user/anaconda3/envs/lvllmds4-x/bin/python scripts/build_variants.sh
#
# License: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/csrc/python_binding/binding.cpp"
OUT_DIR="$ROOT/build"
mkdir -p "$OUT_DIR"

PYTHON="${PYTHON:-python3}"
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PYBIND11_INC="${PYBIND11_INC:-/home/user/lvllm/.search-venv/lib/python3.10/site-packages/pybind11/include}"

COMMON="-std=c++17 -shared -fPIC -O3 -ffast-math -fno-finite-math-only"

# name|extra-arch-flags (empty flags => scalar variant)
VARIANTS=(
  "scalar|"
  "avx2|-mavx2 -mfma"
  "avx512_base|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma"
  "avx512_vnni|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma"
  "avx512_bf16|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mfma"
)

for entry in "${VARIANTS[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  mod="_xiaotu_moe_C_${name}"
  OUT="$OUT_DIR/${mod}${PY_EXT}"
  echo ">> Building variant [$name] -> $OUT"
  g++ $COMMON $flags \
      -DXIAOTU_MOE_MODULE_NAME="$mod" \
      -I"$PY_INC" -I"$PYBIND11_INC" -I"$ROOT/csrc" \
      "$SRC" \
      -o "$OUT"
done

echo ">> Done. Variants in $OUT_DIR:"
ls -1 "$OUT_DIR"/_xiaotu_moe_C*"${PY_EXT}"
