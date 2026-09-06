# xiaotu-moe v0.12 — Release Notes

**Author:** 大河马 (dahema@me.com) · built with DeepSeek Harness assistance
**License:** Apache-2.0 · **PyPI:** `xiaotu-moe==0.12`
**Support:** Python ≥3.10 · cp312 manylinux_2_34_x86_64 wheel

> ⚠️ **DEPRECATED — DO NOT USE 0.1.0** — 0.1.0 has runtime errors and must not
> be used; PyPI cannot delete it, so ignore the 0.1.0 listing and always install
> **0.12 or later**: `pip install "xiaotu-moe>=0.12"`. (0.1.0 GitHub release/tag
> removed.)

---

## Summary

xiaotu-moe is an **open-source, Apache-2.0 CPU implementation** (BF16 / FP8 /
WNA16 / MXFP4 / NVFP4) of the *closed-source* `lk_moe.MOEV2` CPU MoE engine —
a **drop-in ABI-compatible replacement**. v0.12 is the follow-on performance
release: a **2.2× end-to-end breakthrough** that takes the same-machine payload
past the previous lk-moe comparison bar, plus a corrected CPU hardware model and
thread-geometry calibration.

## The headline: cudagraph FULL_DECODE_ONLY — 2.2×

The real ~33 tok/s ceiling was **per-decode-step HOST GPU orchestration** — 43
CPU MoE layers each launching + syncing a GPU kernel once per layer per token.
Enabling `--compilation_config.cudagraph_mode FULL_DECODE_ONLY` (the exact config
the production lk server already uses) replays one breakable CUDA graph per
decode step and removes that host latency:

| Threads | layout | cudagraph | Total (tok/s) | Med TPOT (ms) |
|---:|---|---:|---:|---:|
| 168 | 84/socket · 7/CCD | NONE (prior default) | 32.00 | 213.1 |
| 168 | 84/socket · 7/CCD | **FULL_DECODE_ONLY** | **66.62** | **79.83** |
| 120 | 60/socket · 5/CCD | FULL_DECODE_ONLY | 77.69 | 73.65 |
| 96 | 48/socket · 4/CCD | FULL_DECODE_ONLY | 77.75 | 67.29 |

- Correctness maintained: **50/50 pass, 0 failed, 0 stall** at every point.
- 168 is the **user-mandated production layout** (84/socket · 7/CCD — saturates
  each socket's IOD DDR5 bandwidth and reserves 1 core/CCD for other host tasks)
  and clears the DoD (≥51 tok/s, TPOT ≤177 ms) by a wide margin.

## What else changed in 0.12

- **Corrected hardware model.** Verified from `/proc/cpuinfo` + `/sys` topology:
  *dual-socket* EPYC 9654, **192 physical cores** (96/socket), **SMT OFF**,
  12 CCDs/socket sharing one IOD / 12-channel DDR5. The earlier "192 logical /
  96 physical, SMT ON" reading was a misparse (per-socket `core_id` re-numbering)
  and its "96 avoids SMT contention" theory is retracted.
- **3-barrier phase fusion kept.** The sharded decode hot path runs exactly 3
  `parallel_for` barriers (A: gate+up+gated-SiLU+f32→bf16 fused; B: down; C:
  weighted reduce), matching the 3 `do_k_work_stealing_job` barriers in the
  lk-moe decompilation — numerically identical, neutrally fast.
- **Spin-based synchronization** in `numa_pool.hpp` (hot-restart + completion
  busy-spin) — replaces per-`parallel_for` mutex/condvar futex storms, mirroring
  lk-moe's lock-free atomic + busy-yield.
- **Single-copy NUMA sharding** (`shard_fill_w13/w2`): full-weight replicated
  per-SHMEM node-local shards bound with `MBIND`, node-local weight reads.
- **ILP-16 + byte-profiling** in the AVX-512 bf16 kernel — 4-token × 4-partial
  independent zmm chains, rounded for explicit stream bandwidth.

## The "5 cores / CCD" guide — validated

A circulating BIOS guide claims enabling **5 of each CCD's 8 cores** (disabling
the rest) gives the best lk-moe throughput on the 9654. Our dual-socket 120-thread
run (5/CCD = 60/socket) reproduces it exactly: **77.69 tok/s / 73.65 ms**, tied
with 4/CCD (96, 77.75/67.29) and well above the over-provisioned 7/CCD (168,
66.62/79.83). Rationale: a bandwidth-bound streaming MoE saturates each IOD's
DDR5 with ~4–5 cores/CCD; beyond that, extra cores only add barrier/queue
overhead. See [`docs/THREAD_GEOMETRY.md`](docs/THREAD_GEOMETRY.md).

## Known limitation (REOPENED hard blocker: memory)

During a short 50-prompt bench the EngineCore VmRSS reached a **peak ~554 GB**
(~24 GB/min through decode) — far above lk-moe's ~256 G and the ≤ ~300 G target.
**Performance DoD is met, but memory footprint is not yet equivalent** and must be
reduced in a follow-up (tracked with O(1) `/proc/PID/status` sampling).

## Changelog & ToDo

- Full changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Thread geometry / why-thread-count analysis:
  [`docs/THREAD_GEOMETRY.md`](docs/THREAD_GEOMETRY.md)
- Long-term optimization ToDo: [`docs/TODO_LONGTERM.md`](docs/TODO_LONGTERM.md)
- Raw benchmark record: `/home/user/lvllm/results.txt` (Sessions 16–20)

## Install

```bash
pip install xiaotu-moe==0.12
```
