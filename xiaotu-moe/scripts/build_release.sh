#!/usr/bin/env bash
# Build the release wheel for xiaotu-moe.
#
# Assemblies the native extension libraries built by build_variants.sh (which
# writes into build/) into the self-contained in-package location
# (xiaotu_moe/build/), then builds a properly-tagged platform wheel into dist/.
#
# The wheel is NOT "pure": it bundles prebuilt cp312 / x86_64 .so files linked
# against glibc >= 2.34, so it is tagged cp312-cp312-linux_x86_64 (see setup.py).
#
# Usage:
#   scripts/build_release.sh            # build for the active interpreter env
#   PYTHON=/path/to/python scripts/build_release.sh
#
# License: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# 1. Native libs: build_bariants.sh must have run for this interpreter.
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
if ! ls "$ROOT"/build/*"$PY_EXT" >/dev/null 2>&1; then
  echo ">> No '$PY_EXT' variants in build/. Running build_variants.sh first..."
  (cd "$ROOT" && PYTHON="$PYTHON" scripts/build_variants.sh)
fi

# 2. Assemble self-contained in-package build/ (wheel layout).
rm -rf "$ROOT/xiaotu_moe/build"
mkdir -p "$ROOT/xiaotu_moe/build"
for f in "$ROOT"/build/*"$PY_EXT"; do
  cp "$f" "$ROOT/xiaotu_moe/build/"
done
echo ">> Bundled $(ls "$ROOT/xiaotu_moe/build" | wc -l) native variants for $PY_EXT"

# 3. Build the wheel (wd = repo root so pyproject.toml/setup.py are found).
rm -rf "$ROOT/dist" "$ROOT"/*.egg-info
(cd "$ROOT" && "$PYTHON" -m pip wheel . --no-deps -w dist)

echo ">> Wheel:"
ls -1 "$ROOT"/dist/*.whl
