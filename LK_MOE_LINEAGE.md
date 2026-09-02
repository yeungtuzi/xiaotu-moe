# lk_moe 谱系与代码来源验证报告

> 结论依据：lk_moe 发行包内自带的 `THIRD_PARTY_LICENSES` 第三方授权文件 + 对该文件点名的源码逐份比对 + lk_moe .so 二进制字符串/符号验证。
> 验证目标：用户向作者询问后，作者确认二进制附带第三方授权文件点明部分代码源自 ktransformers。本报告核实并参照比较这些代码。

---

## 1. 第三方授权文件内容（关键）

文件位于解压 wheel 的 `lk_moe-2.3.3.dist-info/licenses/THIRD_PARTY_LICENSES`（26 行），作者署名 **guqiong9696@gmail.com（Qiong GU）**。文件明确声明以下三份文件**衍生自 KVCache.AI 的 ktransformers**（Apache-2.0），由 guqiong9696 于 2025 年修改：

| lk_moe 内路径 | ktransformers 原始来源（tag v0.2.3） | 修改说明 |
|---|---|---|
| `csrc/lk_moe/moe.h` | `ktransformers/ktransformers_ext/operators/llamafile/moe.h` | 大量重构以支持 NUMA 与 CPU-GPU 混合推理 |
| `csrc/lk_moe/moe.cpp` | `ktransformers/ktransformers_ext/operators/llamafile/moe.cpp` | 同上 |
| `csrc/lk_moe/backend_numa.cpp` | `ktransformers/ktransformers_ext/cpu_backend/backend.cpp` | 大量重构以实现 NUMA 感知后端 |

note 明确：原始 KVCache.AI 代码仍受 Apache-2.0 约束；所有修改/重构/新代码为 Qiong GU 的专有（proprietary）成果，按 lk_moe LICENSE 使用。

---

## 2. 源码逐份比对（ktransformers v0.2.3 ↔ 中间版 lktransformers ↔ .so）

已通过 `raw.githubusercontent.com/kvcache-ai/ktransformers` 的 **v0.2.3 tag** 抓取三份原始文件：

- `kt_v023/moe.h`（102 行）、`kt_v023/moe.cpp`（353 行）、`kt_v023/backend.cpp`（154 行）
- 与之前从 `guqiong96/lktransformers`（NUMA 分支，2025-07/08 提交）抓取的中间版 `lkt_src_moe.cpp/h`、`lkt_src_backend_numa.cpp/h` 对比。

### 2.1 文件头版权（三者一致，证明同源）
- `moe.h` / `moe.cpp` / `backend.cpp` 头部均为：**chenht2022，@Date 2024-07-22，@Copyright (c) 2024 by KVCache.AI**。
- 与 license 文件点名一致，落在 ktransformers 的 `operators/llamafile` + `cpu_backend` 经典路径上。

### 2.2 backend.cpp → backend_numa.cpp（调度器，**字节级命中**）
ktransformers v0.2.3 `Backend`（含 `do_work_stealing_job` / `process_tasks` / `worker_thread`）→ lktransformers 重构为 `Backend_NUMA`（新增 `init_cpu_info`、按 package/core 拓扑建 NUMA 线程、`bind_to_cpu`、`set_numa_mempolicy`、`allocate_aligned_numa`、`do_k_work_stealing_job` 等）。

lk_moe .so 中逐字节命中的字符串（均能在 lktransformers `backend_numa.cpp` 中找到原文）：
- `Backend_NUMA init suscess, numa nodes: `（**保留原作者拼写错误 "suscess"**）
- `Backend_NUMA::do_k_work_stealing_job: n_threads == 0`
- `do_k_work_stealing`（符号级出现）
- `Using LK_THREADS from environment: ` / `Using LK_POWER_SAVING from environment: `
- `sched_setaffinity failed`、`topology/core_id`、`topology/physical_package_id`

→ **调度器（Backend_NUMA）确系从 ktransformers backend.cpp 经 lktransformers 衍生而来，铁证成立。**

### 2.3 moe.h / moe.cpp（MoE 骨架，结构保留、内容重写）
- 保留 `MOE` 类骨架与 `MOEConfig` 结构、`forward/forward_one/forward_many` 方法族、ggml 类型辅助计算（`ggml_type_size`/`ggml_vec_dot` 概念）。
- 中间版 lktransformers 大幅重写内存布局为 NUMA/AMX 分块（`gate_numa_/up_numa_/down_numa_`、`NumaBlock`、`stride_*/amx_stride_*_bytes`），并引入 `#if defined(__AMX_INT8__) && defined(__AVX512VNNI__) #include "amx_gemm.hpp"` 条件编译。
- diff 规模：`moe.cpp` 约 1070 行差异（353→800 行），`moe.h` 亦大面积重构。

### 2.4 .so 二进制层最终状态（V2 重写）
- **旧 `MOE` 类成员已基本消失**：`forward_many_numa`、`dispatch`、`gate_proj_numa`、`do_work_stealing` 均 0 命中；仅 `forward_many` 字符串 40 处保留（复用命名）。
- **量化为新的 `MOE_V2<WeightTraits, ActivationType>` 模板**，RTTI typeinfo 见 10 个实例化：
  - `FP8WeightTraits` × {Float16, BFloat16}
  - `BF16WeightTraits` × BFloat16
  - `FP16WeightTraits` × Float16
  - `WNA16WeightTraits` × {Float16, BFloat16}
  - `NVFP4WeightTraits` × {Float16, BFloat16}
  - `MXFP4WeightTraits` × {Float16, BFloat16}
  - 与 `__init__.py` 导出的 `MOE_FP8/MOE_BF16/MOE_FP16/MOE_WNA16[_FP16]/MOE_NVFP4/MOE_MXFP4` 一一对应。
- **ggml / llamafile 依赖被剥离**：.so 中 `ggml`、`llamafile` 字符串均为 0（V2 重写去掉了上一代基于 llamafile/tinyBLAS 的 ggml 计算层）。
- **专有新源码树**（.so 内构建主机路径）：`/home/guqiong/Downloads/lk_moe/csrc/cuda/moe_v2_gpu_memory.cu`、`moe_v2_gpu_metadata.cu`、`moe_v2_gpu_prefill.cu` → 新增 GPU 端 MoE 内核，支撑 CPU-GPU 混合推理。

---

## 3. 与"用户假设"（AVX512/VNNI/AMX + Marlin 反向量化）的对照结论

| 用户假设 | 验证结果 |
|---|---|
| 调度/骨架源于 ktransformers 早期成果 | ✅ **成立且被作者授权文件正式确认**（backend_numa.cpp、moe.h/cpp） |
| AVX512 / VNNI / AMX 加速 | ⚠️ **部分**。ktransformers v0.2.3 本身**无** AMX/VNNI 条件编译；`amx_gemm.hpp` 与 `__AMX_INT8__/__AVX512VNNI__` 是在 **lktransformers 中间版**由 guqiong96 引入（基于 ktransformers 的 llamafile amx_gemm 技术储备）。llama.cpp ggml 层已在 V2 重写中剥离，AMX GEMM 主体由作者自己的反量化/GEMM 内核承担 |
| 调用 Marlin 反向量化 | ❌ 引擎 .so 内**无 Marlin 字符串**。Marlin 是独立的 GPU CUDA 扩展（vLLM 侧的 `WNA16`/WNA16 Marlin 反量化路径），位于 GPU 侧而非本 CPU-GPU 引擎 .so 内 |

---

## 4. 完整谱系链（最终版）

```
上游 ktransformers（KVCache.AI）  moe.cpp/h (chenht2022, 2024-07-22)  backend.cpp (chenht2022, 2024-07-22)
        │  Apache-2.0  v0.2.3 + 后续
        ▼
guqiong96 / lktransformers（NUMA 分支，2025-07~11）
        │  重构为 Backend_NUMA；新增 AMX/VNNI 条件编译、NUMA 分块内存布局
        ▼  闭源化（V2 提取 + 重写，剥离 ggml/llamafile，新增 csrc/cuda）
lk_moe（PyPI 2025-11-08 起，闭源二进制，仅 cp312 manylinux x86_64）
        │  MOE_V2<FP8/BF16/FP16/WNA16/NVFP4/MXFP4 × f16/bf16>
        ▼  集成
Lvllm / Lsglang / Lvllmds4 / Lvllmds4-x（本 fork）
        │  fused_moe / routed_experts.py + moe_runner.py 混合分发
        ▼
vLLM 侧 GPU 扩展（Marlin WNA16 等，独立于引擎 .so）
```

---

## 5. 一句话总结

作者授权文件**正式背书**了我们的推断：lk_moe 的 **NUMA 调度器（Backend_NUMA）与 MoE 骨架（MOE::）确实衍生自 ktransformers**（Apache-2.0，经 lktransformers 中间版），二进制字符串逐字节吻合（含 "suscess" 拼写错误）。但**量化计算内核已被 V2 重写**为专有的 `MOE_V2<WeightTraits>` 模板并剥离 ggml/llamafile，AMX/VNNI 加速是作者自己在 lktransformers 阶段引入的，而 **Marlin 反量化在 GPU 端、不在该引擎 .so 内**。
