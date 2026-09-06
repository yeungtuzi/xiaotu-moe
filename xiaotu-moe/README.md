<div align="center">

# xiaotu-moe

**完全开源（Apache-2.0）的 `lk_moe.MOEV2` CPU MoE 引擎的等价替代实现**
**Open-source (Apache-2.0) drop-in ABI-compatible replacement for the closed-source `lk_moe.MOEV2` CPU MoE engine**

**作者 Author:** 大河马 (dahema@me.com) · with DeepSeek Harness assistance
**PyPI:** `xiaotu-moe` · **支持:** Python ≥3.10 / cp312 manylinux_2_34_x86_64

</div>

---

> ⚠️ **切勿使用 0.1.0（DEPRECATED — DO NOT USE 0.1.0）**
>
> **0.1.0 在实际运行中存在错误，请勿安装或使用。** 请使用 **0.12 或更高版本**。
>
> **0.1.0 has runtime errors and must NOT be used.** Please use **0.12 or later**.
>
> ```
> pip install "xiaotu-moe>=0.12"      # 推荐 Recommended
> ```
> PyPI 不允许删除已上传的文件，0.1.0 在 PyPI 上仍会存在，但**请忽略它**；请始终安装 0.12 及以上。
> PyPI cannot delete already-uploaded files, so 0.1.0 remains listed there, but **please ignore it** and always install 0.12+.

---

## 简介 Overview

`xiaotu-moe` reimplements the **closed-source** `lk_moe` CPU MoE engine as an
open, Apache-2.0 package that is a **drop-in ABI-compatible replacement** with
the same `MOEV2` runtime entry points. It supports **BF16 / FP8 (WNA16) /
MXFP4 (E2M1) / NVFP4** weight formats, running the routed MoE layers on CPU
while the transformer (attention etc.) stays on GPU.

| | xiaotu-moe | lk_moe (closed) |
|---|---|---|
| License | Apache-2.0 (free to use/modify) | closed-source |
| Native kernels | avx2 / avx512_base / avx512_vnni / **avx512_bf16** / scalar | proprietary |
| Entry points | `MOEV2` (ABI-compatible) | `MOEV2` |
| CPU MoE compute | own reimplementation | reference only |

## 安装 Install

```bash
pip install "xiaotu-moe>=0.12"
```

## 性能 Performance summary (v0.12)

See [`CHANGELOG.md`](CHANGELOG.md), [`RELEASE_NOTES.md`](RELEASE_NOTES.md) and
[`docs/THREAD_GEOMETRY.md`](docs/THREAD_GEOMETRY.md).
Dual-socket EPYC 9654 (192 **physical** cores, SMT OFF) + A100/GPU2,
50 prompts / conc-4 / ml-8192, all 43 MoE layers on the xiaotu CPU engine:

| Config | Total (tok/s) | Med TPOT (ms) | 50/50 |
|---|---:|---:|---:|
| v0.11 baseline (168, cudagraph NONE) | 25.95 | 293.6 | 50/0 |
| v0.12 @ **168** (84/socket, 7/CCD — **mandated**) | **66.62** | **79.83** | 50/0 |
| v0.12 @ 120 (5/CCD guide) | 77.69 | 73.65 | 50/0 |
| v0.12 @ 96 | 77.75 | 67.29 | 50/0 |

> **The breakthrough is vLLM cudagraph `FULL_DECODE_ONLY`** (removes per-decode-step
> host GPU-kernel launch/sync orchestration across the 43 CPU MoE layers) — 2.2×,
> matching the production lk server's own config. 168 (the user-mandated
> 84/socket · 7/CCD layout) clears the DoD (≥51 tok/s, TPOT ≤177 ms) by a wide
> margin. The guide's "5 cores/CCD" point is validated: at 4–5 cores/CCD the
> bandwidth-saturation plateau measures ~77.7 tok/s, statistically tied; past
> 6–7/CCD (168) the extra cores only add orchestration overhead. For a
> bandwidth-bound MoE, **4 cores/CCD already reaches the peak — more buys
> nothing**; go to 5/CCD only if memory bandwidth itself is higher (e.g. a board
> running DDR5-5600 instead of 4800); never past 5/CCD.

Correctness: 50/50 pass, 0 failed, 0 stall. Long-term alignment ToDo:
[`docs/TODO_LONGTERM.md`](docs/TODO_LONGTERM.md).

## License

Apache-2.0. High-performance kernels reference
[ktransformers/lktransformers](https://github.com/kvcache-ai/ktransformers)
(also Apache-2.0). The closed-source lk_moe is used only as a reference for
behavior/approach, never copied.
