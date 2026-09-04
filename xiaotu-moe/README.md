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
> **0.1.0 在实际运行中存在错误，请勿安装或使用。** 请使用 **0.11 或更高版本**。
>
> **0.1.0 has runtime errors and must NOT be used.** Please use **0.11 or later**.
>
> ```
> pip install "xiaotu-moe>=0.11"      # 推荐 Recommended
> ```
> PyPI 不允许删除已上传的文件，0.1.0 在 PyPI 上仍会存在，但**请忽略它**；请始终安装 0.11 及以上。
> PyPI cannot delete already-uploaded files, so 0.1.0 remains listed there, but **please ignore it** and always install 0.11+.

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
pip install "xiaotu-moe>=0.11"
```

## 性能 Performance summary (v0.11)

See [`CHANGELOG.md`](CHANGELOG.md) and [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
168-core EPYC 9654 + A100/GPU2, 50 prompts / conc-4 / ml-8192:

| Metric | baseline | v0.11 | lk-moe | v0.11 / lk-moe |
|---|---:|---:|---:|---:|
| Total token throughput (tok/s) | 22.71 | **25.95** | 51.00 | 0.509x |
| TPOT mean (ms) | 329.3 | **293.6** | 177.3 | 1.66x slower |
| TTFT mean (ms) | 7909.5 | **5809.9** | 2055.2 | 2.83x slower |

50/50 pass, 0 failed, 0 stall. Long-term alignment ToDo:
[`docs/TODO_LONGTERM.md`](docs/TODO_LONGTERM.md).

## License

Apache-2.0. High-performance kernels reference
[ktransformers/lktransformers](https://github.com/kvcache-ai/ktransformers)
(also Apache-2.0). The closed-source lk_moe is used only as a reference for
behavior/approach, never copied.
