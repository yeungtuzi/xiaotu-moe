#!/usr/bin/env bash
# Build the xiaotu-moe BF16 closed-loop pybind11 extension.
#
# Produces build/_xiaotu_moe_C.cpython-*.so
#
# License: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/csrc/python_binding/binding.cpp"
OUT_DIR="$ROOT/build"
mkdir -p "$OUT_DIR"

# Target Python interpreter (override to build for another env/ABI), e.g.
#   PYTHON=/home/user/anaconda3/envs/lvllmds4-x/bin/python scripts/build.sh
PYTHON="${PYTHON:-python3}"

# Python includes (target interpreter).
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

# pybind11 includes (header-only; Python-version-agnostic, supplied via pip into .search-venv).
PYBIND11_INC="${PYBIND11_INC:-/home/user/lvllm/.search-venv/lib/python3.10/site-packages/pybind11/include}"

# ISA flags: this host is AMD EPYC 9654 (avx512_bf16 / avx512_vnni / avx2, no AMX).
ISA_FLAGS="-march=native -O3 -ffast-math -fno-finite-math-only"

OUT="$OUT_DIR/_xiaotu_moe_C${PY_EXT}"

echo ">> Building $OUT"
g++ -std=c++17 -shared -fPIC \
    $ISA_FLAGS \
    -I"$PY_INC" -I"$PYBIND11_INC" -I"$ROOT/csrc" \
    "$SRC" \
    -o "$OUT"

echo ">> Done: $OUT"
