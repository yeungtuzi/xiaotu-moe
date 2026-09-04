# xiaotu-moe v0.11 — Release Notes

**Author:** 大河马 (dahema@me.com) · built with DeepSeek Harness assistance
**License:** Apache-2.0 · **PyPI:** `xiaotu-moe==0.11`
**Support:** Python ≥3.10 · cp312 manylinux_2_34_x86_64 wheel

---

## Summary

xiaotu-moe is an **open-source, Apache-2.0 CPU implementation** (BF16 / FP8 /
WNA16 / MXFP4 / NVFP4) of the *closed-source* `lk_moe.MOEV2` CPU MoE engine —
a **drop-in ABI-compatible replacement**. v0.11 is the first **performance
alignment** release: a full iteration whose goal is to make xiaotu-moe's
same-machine performance converge toward lk-moe, using the open-source
**ktransformers / lktransformers** kernels (Apache-2.0) as the reference for
*how* to compute (multiplier, alignment, prefetch) — never copying lk-moe's own
binary/code.

## What's new in 0.11

- **Persistent batched MoE kernel** — `forward_many_nsliced` gathers each
  routed expert's instances into contiguous rows so every weight row is decoded
  once per layer and amortized across multiple tokens + M-way ILP
  (ktransformers 4-token-block technique), fanning each big GEMV across the
  whole 168-thread NUMA pool.
- **Persistent per-expert scratch buffers** — capacities persist across calls →
  zero allocation in the steady-state hot loop (the earlier local-vector version
  regressed ~25% from per-forward mmap/munmap churn).
- **ktransformers-aligned weight-row prefetch** — `_mm_prefetch(T0)` of the next
  weight row's 4 cache lines at the top of the N-row loop, hiding DRAM latency
  in the bandwidth-bound sequential weight stream.

## Performance (168-core EPYC 9654 + A100/GPU2, 50 prompts / conc-4 / ml-8192)

| Metric | baseline | v0.11 | lk-moe | v0.11 / lk-moe |
|---|---:|---:|---:|---:|
| Total token throughput (tok/s) | 22.71 | **25.95** | 51.00 | 0.509x |
| Output token throughput (tok/s) | 10.08 | **11.77** | 22.70 | 0.519x |
| TTFT mean (ms) | 7909.5 | **5809.9** | 2055.2 | 2.83x slower |
| TPOT mean (ms) | 329.3 | **293.6** | 177.3 | 1.66x slower |
| ITL mean (ms) | 350.2 | **308.1** | 162.8 | 1.89x slower |

- Correctness maintained: **50/50 pass, 0 failed, 0 stall** throughout.
- Convergence vs baseline: total +10–14%, output +16.6%, TPOT −10.7%,
  TTFT mean −25%, ITL −12%.
- Gap vs lk-moe narrowed: total 0.445x → **0.51x**, TPOT 1.86x → **1.66x**,
  TTFT 3.85x → **2.83x**.

## Reverted experiments (measured regressions — documented)

- **A2+B0 phase fusion** — removed a barrier but degraded the instruction mix
  (total 24.01, TPOT 340).
- **NUMA page-level interleave** — fine-grained 4 KB scatter broke DRAM page
  locality of the sequential weight stream (total 25.46, TPOT 332); default
  contiguous placement is better, showing the gap is kernel depth, not
  cross-socket bandwidth.

## Correctness note

Total generated tokens varies slightly across kernel builds (9729 → 10356) from
fp4/ue8m0 reduction-order drift between the 4-token blocked path and the
single-row remainder path; outputs remain valid/coherent, and TPOT/ITL
(per-token) are the stable comparators.

## Changelog & ToDo

- Full changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Long-term optimization ToDo (toward catching lk-moe):
  [`docs/TODO_LONGTERM.md`](docs/TODO_LONGTERM.md)
- Detailed benchmark report: `/home/user/lvllm/results.txt` §7 and
  `/home/user/lvllm/xiaotu_vs_lkmoe.md` §3.5

## Install

```bash
pip install xiaotu-moe==0.11
```
