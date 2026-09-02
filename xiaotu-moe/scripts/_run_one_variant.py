
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
