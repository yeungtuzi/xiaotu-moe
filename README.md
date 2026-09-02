# xiaotu-moe

**An open-source, byte-level reimplementation of the closed-source `lk_moe` CPU
MoE engine — a drop-in replacement for vLLM forks (Lvllm / Lvllmds4-x).**

> [**中文版**](README_CN.md) · English (default)

---

## What is this?

`lk_moe` is a closed-source CPU inference engine (proprietary license, PyPI
`lmoe-moe==2.3.x`) used by the vLLM forks **Lvllm**, **Lvllmds4-x**, and
**lsglang** (same author) to run the Mixture-of-Experts (MoE) forward on the
**CPU** instead of only on the GPU. This lets a model bigger than VRAM fit as
**VRAM + RAM** (the "1+1=2, 100% VRAM utilization" scheme) with GPU + NUMA
hybrid decode / prefill.

Although `lk_moe` is closed, its author ships `THIRD_PARTY_LICENSES` naming
three files derived from **KVCache.AI ktransformers (Apache-2.0)** — the NUMA
scheduler (`backend_numa.cpp`) and the MoE skeleton (`moe.cpp/h`). Only the
**quantized compute core** (`MOE_V2<WeightTraits, ActivationType>`) is
proprietary.

**`xiaotu-moe` ("little rabbit") is the open reimplementation**: from
Apache-2.0 building blocks, it provides an ABI-compatible CPU MoE engine with
quantized GEMM / dequantization kernels for FP8 / BF16 / FP16 / WNA16 / MXFP4 /
NVFP4, dropping into the fork as `self.lk_moe → xiaotu_moe`.

## Feature highlights

- ✅ **ABI-compatible drop-in** — implements the fork's
  `MOEConfigV2` + `cpu_decode` / `cpu_prefill` contracts; `moe_runner.py`
  dispatch is unchanged.
- ✅ **CPU MoE compute** — BF16/FP16, FP8, WNA16, MXFP4, NVFP4 weight traits
  behind one shared `WeightTraits` orchestration loop.
- ✅ **Pure CPU, no hand-written CUDA** — the GPU side fully reuses the fork's
  standard vLLM GPU MoE (no proprietary `.cu` copied).
- ✅ **Multi-ISA packaging** — 5 ISA variants (`scalar`/`avx2`/`avx512_base`/
  `avx512_vnni`/`avx512_bf16`) built from one source; `_dynamic_loader` picks the
  best at import.
- ✅ **No ggml / llama.cpp / libnuma dependency.**
- ✅ **Verified end-to-end** on the real DeepSeek-V4-Flash-0731 checkpoint with
  output **byte-identical to lk_moe**.

## Repository layout

```
.
├── README.md                 # this file (English)
├── README_CN.md              # 中文版
├── LK_MOE_REIMPLEMENTATION_PLAN.md    # phase plan (Chinese)
├── LK_MOE_INVESTIGATION_REPORT.md     # provenance / reverse-engineering (Chinese)
├── LK_MOE_ABI_SPEC.md        # ABI + layout spec (Chinese)
└── xiaotu-moe/               # the Python package + native source
    ├── xiaotu_moe/           #   python package (dynamic ISA loader)
    ├── csrc/                 #   native engine (moe_v2_*.hpp, kernels, binding)
    ├── scripts/              #   build + numeric A/B + bench
    ├── integration/          #   fork swap patches / clean swap file / runbook
    └── docs/                 #   bilingual project report
```

## Build

From `xiaotu-moe/`:

```bash
# build all ISA variants for the current Python ABI (cp310 & cp312 both verified)
PYTHON=$(which python) PYBIND11_INC=/path/to/pybind11/include \
  bash scripts/build_variants.sh
```

or a single variant:

```bash
PYTHON=$(which python) bash scripts/build.sh   # builds avx512_bf16 (or best supported)
```

The runtime loader (`xiaotu_moe/loader.py`) reads `/proc/cpuinfo` and imports the
best supported variant at import time.

## Usage (drop into the fork)

1. Build the extension (above) so `build/*.so` sits next to the `xiaotu_moe`
   package.
2. In the fork's
   `vllm/model_executor/layers/fused_moe/routed_experts.py`, swap the closed
   engine for ours. Use the provided clean swap file (`integration/
   routed_experts.swap_xiaotu.clean.py`), which rewrites every `lk_moe.*` →
   `xiaotu_moe.*` while keeping the `self.lk_moe` plumbing intact.
3. Run the server: `LVLLM_MOE_NUMA_ENABLED=1 ... vllm serve ...`.

See `integration/LVLLM_BRIDGE.md` for the full runbook and the `_cpu_decode`
bridge choices.

## Correctness

- Pure-CPU numeric A/B vs a numpy golden on real layer-1 routed experts:
  MXFP4 max rel ~3.4e-4, NVFP4 ~8.7e-6 (far better than lk_moe's ~22% tail).
- Full end-to-end on the real checkpoint: real prompt returns output
  **byte-identical to lk_moe**, 0 segfault.

## Performance notes

The current `forward_many` is bandwidth-bound and reads each expert block per
token×rank; two optimizations are in progress:

1. **Expert grouping** — process each active expert once (batched GEMM over all
   tokens routed to it), cutting DRAM traffic from `batch×top_k×12 MB` to
   `active_experts×12 MB` (up to ~7×).
2. **Backend_NUMA equivalent** — a persistent thread pool with NUMA-node
   affinity, work stealing, and NUMA-interleaved memory (open reference:
   Apache-2.0 ktransformers `backend_numa.cpp`).

## License

Apache-2.0 (building on Apache-2.0 ktransformers / lktransformers). No
proprietary lk_moe code is copied; usage is interoperability/understanding only.
See `LK_MOE_INVESTIGATION_REPORT.md` §6 for provenance and licensing detail.

## Documentation

- [Project report (EN)](xiaotu-moe/docs/XIAOTU_MOE_REPORT_en.md)
- [项目报告（中文）](xiaotu-moe/docs/XIAOTU_MOE_REPORT_cn.md)
- [Reimplementation plan（中文）](LK_MOE_REIMPLEMENTATION_PLAN.md)
- [Investigation report（中文）](LK_MOE_INVESTIGATION_REPORT.md)
