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

## System requirement: transparent hugepages must be `madvise`, not `always`

> **This is the root cause of the previously-reported ~554–657 GB native-memory peak
> and it is now fixed — but it is a HOST-SIDE setting, not something the code can
> override from userland.**

The native CPU MoE engine uses **NUMA-sharded weight placement** for speed: each worker reads
only its own node's pages (no cross-node traffic). On each shard region it calls
`madvise(MADV_HUGEPAGE)` but writes only its "owned rows" at full stride (~1/8 of every region).
If the kernel-wide THP mode is `always`, the kernel promotes those regions to **2 MB hugepages
regardless of the `madvise` hint**, and once *any* byte of a 2 MB page is touched the **whole
2 MB page becomes resident**. The strided owned-row writes therefore balloon resident memory to
≈5.5× the single weight copy — a **load-time footprint that is flat afterwards** (not a decode
leak): ~657 GB on the ~155 GB model vs the ~250 GB single-copy expectation.

**Fix (run once on the serving host, `root` required):**

```sh
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled   # [always madvise never] choices
```

Keep it at `madvise` and make sure the engine does *not* re-request hugepages on its shard
regions (the serve scripts export `XIAOTU_MOE_SHARD_HUGEPAGE=0` by default). Then each shard
faults only its owned 4 KB pages → a one-copy footprint (~157 GB at load, ~251 GB plateau) while
retaining node-local read speed. With `always` instead, the same run peaks ~657 GB and misses the
memory target — the knob **cannot be forced back to 4 KB hugepages from userland** while the
sysctl is `always`.

**Verified** on DeepSeek-V4-Flash-0731 (~155 GB MoE, dual-socket EPYC 9654, CPU MoE, TP1,
96 threads, ShareGPT 50 prompts @ conc 4, with `SHARD_HUGEPAGE=0` + `max-num-batched-tokens
4096` + `max-num-seqs 4`): peak VmRSS **244 GB**, total throughput **59.5 tok/s**, mean
TPOT **108 ms**, **50/50** correct — all at / under the 256 GB / 51 tok/s / 177 ms targets.
The same config under THP `always` peaks ~657 GB (memory target missed).

## Performance notes

Two optimizations were implemented and measured (see §10 of the [project
report](xiaotu-moe/docs/XIAOTU_MOE_REPORT_en.md)):

1. **Expert grouping** — `forward_many` walks each active expert once (adaptive
   grouped vs per-token dispatch). Honest measured result: **no speedup** — each
   gate_up/down still reads its full 12 MB expert block per assignment (block ≫
   L2, so nothing is reused), so grouping alone does not cut DRAM traffic. A real
   blocked/batched GEMM with register blocking is future work.
2. **Backend_NUMA equivalent** — `csrc/moe/numa_pool.hpp` (`NumaWorkPool`): a
   persistent thread pool with NUMA-node affinity (each worker pinned to a
   distinct physical core, spread across NUMA nodes via `sched_setaffinity`),
   dynamic work stealing, and NUMA-interleaved memory for the engine-owned weight
   snapshots via the raw `mbind` syscall — no libnuma dependency (open reference:
   Apache-2.0 ktransformers `backend_numa.cpp`). Its completion barrier was
   hardened with per-worker generation records to eliminate a cross-generation
   race.

## Performance (v0.12)

**2.2× breakthrough** — running the fork's vLLM with
`--compilation_config.cudagraph_mode FULL_DECODE_ONLY` removes the
per-decode-step host GPU-kernel launch/sync orchestration across the 43 CPU MoE
layers. This matches the production lk server's own config, and is the real
speeeed lever (thread count is secondary).

Benchmark: **DeepSeek-V4-Flash-0731** (all 43 MoE layers on the xiaotu CPU
engine), ShareGPT **50 prompts / concurrency 4 / max-model-len 8192**, dual-socket
EPYC 9654 (192 physical cores, SMT OFF) + A100/GPU2 (tensor-parallel-size 1).

| Config | Total (tok/s) | Med TPOT (ms) | 50/50 |
|---|---:|---:|---:|
| v0.11 baseline (168, cudagraph NONE) | 25.95 | 293.6 | 50/0 |
| **v0.12 @ 168 (84/socket · 7/CCD, mandated)** | **66.62** | **79.83** | 50/0 |
| v0.12 @ 120 (5/CCD guide) | 77.69 | 73.65 | 50/0 |
| v0.12 @ 96 | 77.75 | 67.29 | 50/0 |

> MoE decode is **memory-bandwidth-bound**: per-socket IOD DDR5 bandwidth is the
> hard ceiling. **4 cores/CCD already reaches the peak**; go to **5/CCD only if**
> memory bandwidth is higher (e.g. board runs DDR5-5600); never past 5/CCD. Full
> why-analysis: [`xiaotu-moe/docs/THREAD_GEOMETRY.md`](xiaotu-moe/docs/THREAD_GEOMETRY.md).

## Known issues

| Priority | Issue | Status |
|---:|---|---|
| 🔴 High | **Native-mode memory footprint** — peak VmRSS up to ~657 GB on the real large model in untrimmed runs, far above the ~155 GB weight working set and the 256 GB lk reference. **Root-caused & fixed: host THP must be `madvise` (system `always` inflated the NUMA shards ~5.5×).** With `madvise` + `XIAOTU_MOE_SHARD_HUGEPAGE=0` the peak drops to **244 GB**. See *System requirement* above. | Resolved |
| 🟡 Med | WNA16 (FP8) path has a pre-existing `packed4` packing bug (flat & sharded both fail) — not a shipping format, does not affect real BF16/MXFP4 inference. | Known |
| 🟡 Med | MXFP4 sharded pool occasional race (E=2 H=512 synthetic) — pre-existing, unrelated to fusion/threads/cudagraph, never hit the 50/50 real bench. | Known |

## Authors

**大河马 (dahema@me.com)** · assisted by **DeepSeek Harness**.

## License

Apache-2.0 (building on Apache-2.0 ktransformers / lktransformers). No
proprietary lk_moe code is copied; usage is interoperability/understanding only.
See `LK_MOE_INVESTIGATION_REPORT.md` §6 for provenance and licensing detail.

## Documentation

- [Project report (EN)](xiaotu-moe/docs/XIAOTU_MOE_REPORT_en.md)
- [项目报告（中文）](xiaotu-moe/docs/XIAOTU_MOE_REPORT_cn.md)
- [Reimplementation plan（中文）](LK_MOE_REIMPLEMENTATION_PLAN.md)
- [Investigation report（中文）](LK_MOE_INVESTIGATION_REPORT.md)
