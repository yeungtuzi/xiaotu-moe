#!/usr/bin/env python3
"""Verify every built ISA variant of xiaotu-moe produces correct results.

pybind11's type registry is process-global, so a Python process can hold only
ONE variant module. This driver therefore runs each variant in a separate
subprocess (`_run_one_variant.py`), which force-loads the variant via the
dynamic loader and runs representative numerical checks (MXFP4 / WNA16 /
NVFP4) using the same numpy reference as scripts/smoke_test_packed4.py.

The loader module is loaded STANDALONE by file path (not via `import xiaotu_moe`)
so the downstream subprocess's package __init__ never eagerly imports the
auto-chosen variant (which would collide with the forced one).

Run with the .search-venv interpreter:
    /home/user/lvllm/.search-venv/bin/python scripts/verify_variants.py
"""
import importlib.util
import os
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "xt_moe_loader", os.path.join(ROOT, "xiaotu_moe", "loader.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = ["scalar", "avx2", "avx512_base", "avx512_vnni", "avx512_bf16"]

_RUN_ONE = r"""
import importlib.util, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT); sys.path.insert(0, _here)
spec = importlib.util.spec_from_file_location("xt_moe_loader", os.path.join(ROOT, "xiaotu_moe", "loader.py"))
loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
import smoke_test_packed4 as smoke
suffix = sys.argv[1]
mod = loader.load(force=suffix)
smoke.m = mod
sub = [
    smoke.run_case("MXFP4", smoke.E2M1, 32, 128, 256, 4, 2, e8m0=True),
    smoke.run_case("MXFP4", smoke.E2M1, 32, 512, 256, 2, 1, e8m0=True),
    smoke.run_case("WNA16", smoke.INT4_CENTER8, 32, 128, 256, 4, 2),
    smoke.run_nvfp4(128, 256, 4, 2, gk=32),
    smoke.run_nvfp4(512, 256, 6, 3, gk=32),
]
print("RESULT " + suffix + " " + ("PASS" if all(sub) else "FAIL"))
sys.exit(0 if all(sub) else 1)
"""


def _write_runner() -> str:
    p = os.path.join(_here, "_run_one_variant.py")
    with open(p, "w") as f:
        f.write(_RUN_ONE)
    return p


def main():
    loader = _load_loader()
    build_dir = loader._variant_dir()
    present = loader._available_variants(build_dir)
    print("present variants:", sorted(present))

    runner = _write_runner()
    py = sys.executable

    all_ok = True
    for name in VARIANTS:
        suffix = f"_{name}"
        if suffix not in present:
            print(f"[skip] variant {name!r} not built for this interpreter")
            continue
        print(f"\n===== variant [{suffix}] =====")
        proc = subprocess.run([py, runner, suffix], capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        for ln in out.splitlines():
            if any(k in ln for k in ("[MXFP4]", "[WNA16]", "[NVFP4]", "RESULT", "Traceback", "Error")):
                print("  " + ln)
        ok = f"RESULT {suffix} PASS" in out.replace("\n", " ")
        all_ok = all_ok and ok
        print(f"===== variant [{suffix}] {'PASS' if ok else 'FAIL'} =====")

    print("\n" + ("ALL VARIANTS PASS" if all_ok else "SOME VARIANT FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
