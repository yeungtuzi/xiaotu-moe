# Changelog — xiaotu-moe

**作者 Author:** 大河马 (dahema@me.com)，由 DeepSeek Harness 辅助 / with DeepSeek Harness assistance

All notable changes to xiaotu-moe are documented in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/) conventions. License: Apache-2.0.

Versioning: `MAJOR.MINOR`. `0.11` is the first tagged performance-aligned release.

---

> ⚠️ **DEPRECATED — DO NOT USE 0.1.0**
> **0.1.0 在实际运行中存在错误，请勿使用。** 请使用 **0.11 或更高版本**。
> **0.1.0 has runtime errors and must NOT be used — use 0.11 or later.**
> PyPI cannot delete already-uploaded files, so 0.1.0 remains on PyPI but is
> deprecated; ignore it and install `xiaotu-moe>=0.11`. Its GitHub release and
> tag `v0.1.0` have been removed.

---

## [0.11] — 2026-09-04 — Alignment Iteration toward lk-moe / ktransformers

Core deliverable of the "keep aligning/optimizing xiaotu-moe so performance
catches the closed-source lk-moe" effort. Reference for optimizations is
**ktransformers / lktransformers** (`kt-kernel/operators/avx2/mxfp4-moe.hpp`,
Apache-2.0) — studied for *how* it computes (multiplier, alignment, prefetch),
not copied from lk-moe's own binary/code. Every change is A/B-verified on the
same box (168 cores + A100/GPU2), 50-prompts / conc-4 ShareGPT, 50/50 pass,
0 failed, 0 stall, across all iterations.

### Added
- **Persistent batched MoE kernel** (`csrc/moe/moe_v2.hpp`): `forward_many_nsliced`
  now handles the decode/prefill MoE with an expert-instance batching scheme —
  each routed expert's instances are gathered into contiguous rows so every
  weight row is decoded once per layer and amortized across `me` tokens + M-way
  ILP (ktransformers 4-token-block technique). Sub-divided into 5 phases
  (A gate+up, A2 gated-SiLU, B0 f32→bf16, B down, C weighted reduce), each
  parallelized across the NUMA pool.
- **Persistent per-expert scratch buffers** (`ExpBuf` members: `xg`/`both`/`act`/
  `abf16`/`down`/`ai_list`). Capacities persist across calls → the steady-state
  hot loop does **zero allocation** (previous local-vector version regressed
  ~25% from mmap/munmap churn every forward).
- **N-parallel GEMV slicing** (`nc_gu` over `inter`, `nc_d` over `hidden`):
  one token's big GEMV fans out across the entire 168-thread NUMA pool
  (ktransformers `split_range_n` technique) instead of a single worker thread.
- **Weight-row prefetch** in `matmul_packed4_group` (`moe_v2_packed4.hpp`):
  a ktransformers-aligned `_mm_prefetch(T0)` of the *next* weight row's 4 cache
  lines at the top of the N-row loop, hiding DRAM latency for the next row in
  the bandwidth-bound sequential weight stream.

### Changed
- Logging fix in the vLLM fork integration: "lk_moe module is available..." is a
  fixed informational log; the real backend is selected by `XIAOTU_MOE_BACKEND`
  (unset ⇒ xiaotu). Verified via `/proc/<pid>/environ`.
- `numa_pool.hpp`: added robust `numa_interleave_begin()/end()`
  (`set_mempolicy(MPOL_INTERLEAVE)`) alongside the fragile `numa_set_interleaved()`
  (`mbind`). Both present but **unused by default** (see Reverted).

### Reverted (measured regressions — recorded to avoid revisiting)
- **A2+B0 phase fusion** (#3): merging gated-SiLU and f32→bf16 into one job loop
  removed a barrier but worsened the instruction mix; the intermediate `act[]`
  buffer is tiny (L1-resident) so removal gained nothing → total 24.01, TPOT 340
  (vs 25.50 / 297). Reverted.
- **NUMA page-level interleave** (#5): `set_mempolicy(INTERLEAVE)` scattering the
  weight pages across all 8 NUMA nodes broke DRAM page locality of the
  sequential weight stream (worker threads bounce between 8 memory controllers
  within one expert block) → total 25.46, TPOT 332 (vs 25.95 / 294). Default
  "contiguous placement on the constructor's node" is better; the lk-moe gap is
  not cross-socket bandwidth but kernel depth.

### Performance (final = batching + persistent buffers + prefetch, default placement)
Machine: AMD EPYC 9654 dual (192 online cores / 168-pool), A100-40GB (GPU2),
one GPU2/TP1/port-8071 vLLM server, 50 prompts / conc-4 / ml 8192 / seed 0.

| Metric | baseline | 0.11 final (#4/#6) | lk-moe | xiaotu/lk |
|---|---|---:|---:|---:|
| Successful / Failed | 50 / 0 | 50 / 0 | 50 / 0 | — |
| Benchmark duration (s) | 965.5 | 860–902 | 431.0 | 2.0x slower |
| Total token throughput (tok/s) | 22.71 | **25.95 (#4)** | 51.00 | 0.509x |
| Output token throughput (tok/s) | 10.08 | **11.77 (#4)** | 22.70 | 0.519x |
| TTFT mean (ms) | 7909.5 | **5809.9 (#4)** | 2055.2 | 2.83x slower |
| TPOT mean (ms) | 329.3 | **293.6 (#4)** | 177.3 | 1.66x slower |
| ITL mean (ms) | 350.2 | **308.1 (#4)** | 162.8 | 1.89x slower |

#6 (final build confirmation): total 25.00, TPOT 294.1, TTFT 6102, ITL 315 —
consistent with #4, results stable.

Convergence vs baseline: total throughput +10–14%, output +16.6%, TPOT −10.7%,
TTFT mean −25%, ITL −12%. Gap vs lk-moe: total 0.445x→0.51x, TPOT 1.86x→1.66x,
TTFT 3.85x→2.83x.

> Numerically, total generated tokens is not reproducible across kernels
> (9729 → 10418 → 10120 → 10356): fp4/ue8m0 reduction-order drift between the
> 4-token blocked path and the single-row remainder path flips occasional greedy
> argmax over the 256-token horizon. Outputs remain valid / coherent (smoke
> "4\n\n4" correct, 50/50 pass). Throughput is computed per-run on its own token
> count; TPOT/ITL (per-token) are the clean stable comparators.

Full per-iteration detail: `/home/user/lvllm/results.txt` §7 and
`/home/user/lvllm/xiaotu_vs_lkmoe.md` §3.5.

---

## [0.1.0] — 2026-09-02 — Initial open-source reimplementation

- Full Apache-2.0 reimplementation of the closed-source `lk_moe.MOEV2` CPU MoE
  engine (BF16/FP8/WNA16/MXFP4/NVFP4), drop-in ABI-compatible.
- Bilingual README (EN + CN) and LICENSE.
- WNA16/NVFP4 path (scalar reference + avx2 base) and MXFP4 (E2M1) path with
  `_mm256_i32gather`-based decode (latency-bound → replaced in 0.11).
- Single per-layer engine, integration with the vLLM fork via
  `XIAOTU_MOE_BACKEND`.
- `perf: NUMA-aware persistent thread pool + adaptive expert grouping`
  (single process-wide pool mirroring lk_moe's `Backend_NUMA`).
