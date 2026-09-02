# lk-moe 开源重实现方案（Reimplementation Plan）

> 状态：**方案设计**（仅分析与规划，未写任何代码）
> 依据：三棵代码树已全部拉齐 —— `repos/ktransformers/`（当前 main）、`repos/lktransformers/`（guqiong96 开源中间层）、`repos/Lvllm/`（当前 main），以及本地已解压的 `lk_moe-2.3.3` wheel。
> 前置文档：`LK_MOE_INVESTIGATION_REPORT.md`（调查）、`LK_MOE_LINEAGE.md`（谱系）。

本方案回答：**要用开源代码把 lk-moe 的 CPU-GPU 混合 MoE 引擎重新实现出来，需要做什么、从哪开始、代码从哪里来、工作量有多大、法律边界在哪。** 本文件只给路线图，不进入实现。

---

## 1. 目标与范围

### 1.1 重实现什么
lk-moe 是一套**闭源**的 CPU-GPU 混合 MoE 推理引擎，供 vLLM fork 使用：

- **CPU 侧**：把 MoE 层（gate/up/down 投影 + 激活 + 融合）卸载到 CPU 用 **AMX / AVX-512 VNNI / AVX2** 指令集计算，权重量化为 **FP8 / BF16 / FP16 / WNA16(Marlin 型) / NVFP4 / MXFP4** 六种格式。
- **GPU 侧**：GPU-resident 层 + CUDA-graph capture + GPU prefill（batch 超过阈值时）交给 `moe_v2_gpu_*.cu` 处理。
- **调度**：`Backend_NUMA`（从 lktransformers 的 `backend_numa.cpp` 直接编译而来，字节级吻合）做 NUMA 感知的 work-stealing 分块调度。
- **Python ABI**：暴露 `MOEConfigV2` + 10 个 `MOE_*` 类（5 种量化 × {BF16, FP16} 激活），vLLM 侧通过 `data_ptr()` 传指针直接调用。

### 1.2 重实现的目标形态
一个可 `pip install` 的独立包（对标 `lk-moe`），提供相同的 `MOEConfigV2` / `MOE_*` Python ABI，并让 `Lvllm` / `Lvllmds4-x` 的 `routed_experts.py` **无需修改**（或仅最小改动）即可通过 `import lk_moe` 换用。

### 1.3 明确不做
- 不复制 lk-moe 专用新代码（V2 模板 `MOE_V2<WeightTraits, ActivationType>`、`csrc/cuda/moe_v2_gpu_*.cu`、`_dynamic_loader.py` 的内部实现）。这些是 Qiong GU 专有（apache-2.0 note 明确），只能作为互操作/理解的参考。
- 不追求与 lk-moe **逐 bit 一致**，只求**功能等价 + ABI 兼容**。
- 不实现 lk-moe 不公开的内部细节（如特定性能微调参数），除非有明确必要。

---

## 2. 代码来源：重实现的"零件库"

所有可复用源码均来自 **Apache-2.0** 的 ktransformers 与 lktransformers。已逐文件核实存在于磁盘。

### 2.1 调度器骨架（来自 lktransformers —— 作者亲笔、最接近 lk-moe）
文件位于 `repos/lktransformers/csrc/ktransformers_ext/`：

| 文件 | 作用 | 在本重实现中的角色 |
|---|---|---|
| `cpu_backend/backend_numa.cpp/.h` | NUMA 感知 work-stealing 调度器 | **核心骨架**，几乎可直接复用（lk-moe 的调度器即由此编译，含 `do_k_work_stealing_job`、`"suscess"` typo 等字节级痕迹） |
| `cpu_backend/backend.cpp/.h` | 非 NUMA 回退调度器 | 备选骨架 |
| `cpu_backend/shared_mem_buffer.*`、`task_queue.*` | 跨线程/进程共享缓冲、任务队列 | 依赖项 |
| `operators/llamafile/moe.cpp/.h` | MoE 计算编排（forward_one/many/_numa/_m） | **核心编排逻辑**，NUMA 权重分块、AMX/llamafile 双路径切换、激活函数——重实现的主模板 |
| `operators/llamafile/amx_gemm.cpp/.h` | AMX 量化 GEMM（TILE 16×16×32、VNNI） | **AMX 计算路径**，作者在 lktransformers 引入的 AMX 内核入口 |
| `operators/llamafile/linear.cpp/.h`、`mlp.cpp/.h`、`conversion.h` | Linear/MLP 封装、类型转换 | 辅助 |

> 注意：这些是**旧版 ktransformers** 的结构（作者 fork 时的快照）。当前 ktransformers main 已大改（见 2.3），所以调度骨架应以 **lktransformers 为准**，不要从当前 main 找。

### 2.2 量化 GEMM / 反量化内核（来自 ktransformers kt-kernel，当前 main，Apache-2.0）
文件位于 `repos/ktransformers/kt-kernel/operators/`：

| 内核文件 | 对应 lk-moe 量化 | 说明 |
|---|---|---|
| `amx/bf16-moe.hpp` | `MOE_BF16` / `MOE_FP16` | BF16 权重 × BF16 激活 AMX 核 |
| `amx/fp8-moe.hpp`、`amx/fp8-perchannel-moe.hpp` | `MOE_FP8` / `MOE_FP8_FP16` | FP8 E4M3 权重 × BF16 激活 AMX 核（per-block / per-channel） |
| `amx/fp4-moe.hpp` | `MOE_MXFP4` / `MOE_MXFP4_FP16` | **MXFP4** E2M1 权重 × BF16 激活 AMX 核（原名 `mxfp4-moe.hpp`，main 中改名；旧 AVX2 版在 `avx2/mxfp4-moe.hpp`） |
| `amx/awq-moe.hpp` | （WNA16 参考） | AWQ 型 4-bit 分组量化 AMX 核，与 Marlin WNA16 布局近似，可作 WNA16 内核的起点 |
| `amx/mxfp8-moe.hpp`、`amx/sft_moe.hpp`、`amx/k2-moe.hpp` | — | 其他辅助/变体 |
| `moe_kernel/moe.hpp` + `mat_kernel/` | 通用 GEMM 框架 | ktransformers 新一代 MoE 内核框架 |
| `csrc/.../gptq_marlin/`、`custom_marlin/` | WNA16/Marlin 反量化参考 | GPU 侧 marlin 反量化可参考 |

> **重要缺口**：ktransformers 当前 main **没有 NVFP4** 内核（只有 MXFP4）。lk-moe 独有 `MOE_NVFP4`。NVFP4/FP4 的 GPU 侧（`moe_v2_gpu_*.cu`）是 lk-moe 专有。→ 见 3.3 的 NVFP4 策略。

### 2.3 当前 ktransformers main 与作者 fork 时的差异（已核实）
- 当前 main `llamafile/` 只剩 `conversion.h`（`moe.cpp/h` 已被移除/重构）。
- 当前 main `cpu_backend/` 改为 `cpuinfer.h` + `task_queue/worker_pool` 架构，**`backend.cpp/h`、`backend_numa.cpp/h` 均已移除**。
- 结论：**调度骨架不可从当前 main 取得，必须用 lktransformers（或更早的 ktransformers 历史版本）。** 量化内核（amx/ 下的 *-moe.hpp）在两层都可得，以当前 main 为准较新。

---

## 3. 架构设计（目标实现结构）

```
lk_moe (重实现包)
├── lk_moe/
│   ├── __init__.py            # 兼容 facade：导出 MOEConfigV2 + 10 个 MOE_* 类
│   ├── _dynamic_loader.py     # 按指令集选 .so（可自研，勿抄内部实现）
│   ├── _version.py
│   └── _lk_moe_C.<arch>.so    # 编译产物（实际为 CPython 扩展）
├── csrc/
│   ├── scheduler/
│   │   ├── backend_numa.cpp/.h    # ← lktransformers（复用）
│   │   ├── backend.cpp/.h         # ← lktransformers（回退）
│   │   └── shared_mem_buffer.*, task_queue.*   # ← lktransformers
│   ├── moe/
│   │   ├── moe.cpp/.h             # ← lktransformers 编排（改造为模板 V2）
│   │   ├── moe_v2.hpp             # ★ 自研：MOE_V2<WeightTraits, ActType> 模板
│   │   ├── kernels/               # ← ktransformers amx/*.hpp（量化 GEMM）
│   │   └── ...
│   ├── python_binding/            # ★ 自研：CPython 扩展导出 MOEConfigV2 + MOE_*
│   └── cuda/                      # ★ 自研：GPU 端（对应 moe_v2_gpu_*.cu）
│       ├── moe_v2_gpu_memory.cu
│       ├── moe_v2_gpu_metadata.cu
│       └── moe_v2_gpu_prefill.cu
├── setup.py / pyproject.toml
```

### 3.1 分层映射（旧 MOE ⟷ 新 MOE_V2）
lktransformers 的 `MOEConfig` + `MOE` 类是**单类单量化类型**（ggml_type 运行时分派，靠 llamafile_sgemm / amx_gemm_compute 的 ggml_type 参数选择内核）。

lk-moe 的 `MOE_V2<WeightTraits, ActivationType>` 是**编译期模板**：`WeightTraits` 在编译期决定每个权重的布局/反量化/打包格式，`ActivationType` 决定激活是 BF16 还是 FP16，从而通过 RTTI typeinfo 生成 10 个独立实例化。

重实现思路：**把 lktransformers 的 `MOE` 通用编排逻辑提出来，重构为模板类 `MOE_V2<WT, ACT>`**，其中：
- 计算内核从 `amx/*-moe.hpp` 按 `WT` 在编译期绑定（而非运行时 ggml_type 分派）。
- 保持 `forward_one/_many/_numa/_m` + `do_k_work_stealing_job` 的调度结构（复用 lktransformers 代码骨架）。

### 3.2 Python ABI（兼容层）
vLLM 侧 `routed_experts.py` 的使用方式（已核实行号 `Lvllmds4-x` 1340–1805）：
```python
self.lk_moe_config = lk_moe.MOEConfigV2()
self.lk_moe_config.num_processes = ...
self.lk_moe_config.expert_num = ...
...  # 设置字段
self.lk_moe = lk_moe.MOE_WNA16(self.lk_moe_config, w13_ptr, w2_ptr,
                               w13_scale_ptr, w2_scale_ptr, 0, 0)
```
重实现必须：
1. 提供同名 `MOEConfigV2`（字段布局/顺序必须与 ABI 对齐，见 3.3）。
2. 提供 10 个同名 `MOE_*` 类，构造函数签名与调用方式一致。
3. 支持 `data_ptr()` 纯指针传递（零拷贝语义）。

### 3.3 ABI 布局：MOEConfigV2 字段表（关键）
`MOEConfigV2` 是 C/C++ 与 Python 共享的 struct。为使 `import lk_moe` 全面兼容，字段**顺序与类型必须精确对齐**（已在 `LK_MOE_INVESTIGATION_REPORT.md` 记录完整字段集，此处列主要项）：

```
num_processes, process_id, gpu_id,
has_gate_proj,
expert_num, top_k, hidden_size, intermediate_size,
max_batch_size, max_num_seqs,
stride, group_min_len, group_max_len,
groupN, groupK,          # WNA16/Marlin 分组参数
activation_type,
swiglu_alpha, swiglu_limit,
use_gpu_prefill,
...（实测字段实测导出，以实现时以二进制 strings/结构体实际布局为准）
```

> **实现步骤**：重实现时从 `_lk_moe_C_avx512_amx.so` 的 strings / ctypes Structure 导出精确布局；本方案规划阶段先记录字段名清单，代码阶段再逐字段核定。

### 3.4 NVFP4 处理策略（lk-moe 特有、无开源参考）
ktransformers 无 NVFP4 内核，`MOE_NVFP4` 是 lk-moe 独有。三条路径（按推荐排序）：
- **A（推荐）**：复用 vLLM 自带的 NVFP4 语义 —— vLLM 有 `nvfp4` GPU 量化（`nvfp4_emulation_moe.py` 等）。CPU 侧先用 MXFP4 内核 + NVFP4 的 per-block scale 反量化适配；或声明 `MOE_NVFP4` 在无优化路径时回退 GPU。功能等价优先。
- **B**：基于 ktransformers `fp8-perchannel` / `awq-moe` 的分组反量化结构自研 NVFP4 反量化（工作量中等）。
- **C**：只做 ABI 兼容，`MOE_NVFP4` 直接桥接到 GPU 侧的 vLLM NVFP4 实现，不落 CPU。

---

## 4. 分阶段实施路线

### Phase 0 — 锁定参考版本与 ABI（纯分析，已完成大半）
- [x] 三棵树拉齐（ktransformers / lktransformers / Lvllm）
- [x] 谱系与授权文件核实
- [x] lktransformers 与早期 kt_src 差异核对（amx 改名、backend 重构已弄清）
- [ ] **导出 `MOEConfigV2` 精确二进制布局**（strings/ctypes 逐字段核定）
- [ ] 解构 5 个 `.so` 的导出符号表，核对 10 类与 `MOEConfigV2` 的实际 Python 绑定签名

### Phase 1 — CPU 调度 + BF16 路径（最小可用闭环）✅（2026-09-01 代码通过冒烟）
- [x] 从 lktransformers 拉起 `backend_numa` + `moe.cpp/h` + `amx_gemm` → 放入 `csrc/scheduler/`、`csrc/moe/`（参考）。
- [x] 自含 `MOE_V2<WT, ACT>` 模板（`csrc/moe/moe_v2.hpp`），首个 `WT=BF16`，内部 std::thread 池按 token 并行（**不依赖 ggml/llama.cpp/libnuma**，Phase 2/4 可再挂回 `backend_numa` 做 NUMA 亲和而无需改接口）。
- [x] CPython 扩展 `_xiaotu_moe_C` 导出 `MOEConfigV2`（19 字段按 ABI 顺序）+ `MOE_BF16` / `MOE_BF16_FP16` + `cpu_decode`。
- [x] 自含 BF16 GEMM（`csrc/kernels/bf16_gemm.hpp`）：AVX512 fp32-over-K / AVX2 / 标量三路径，处理 N 非 16 倍数的边界（首版 dpbf16 语义错 + 对齐装载段错误，已修为无转置的 16 宽 FMA 方案）。
- [x] 数值冒烟：与 NumPy 参考对齐（rel err ~1.6e-3，即 bf16 舍入），含 odd 尺寸（H=509/I=1027、H=17/I=33）与强并行 M=64 用例。
- [x] **A/B 对照真实闭源 lk_moe 2.3.3**（本机 conda env `lvllmds4-x`，`scripts/ab_test_bf16.py`）：同一随机 BF16 权重喂 `real.MOE_BF16.cpu_prefill`（全 CPU 缓冲）与 `xiaotu-moe.MOE_BF16`，4 组真实风格维度（standard / small-mult16 / llama-scale I=8192 / h36-mult），worst rel err **1.805e-3** PASS。**证实：w13 布局 `[E][2I][H]` gate-first、w2 `[E][H][I]`、gate→up→silu→down→加权累加完全一致，BF16 路径与真实引擎逐值兼容。**
  - 工程发现：真实 lk_moe 构造器即做 GPU/NUMA 初始化（需设备访问，本 agent 默认沙箱拦 `/dev/nvidia*`，`danger-full-access` 下 OK）；其 AVX512 CPU 路径对非 16 倍数维度段错误（真实模型维度均为 16/32 倍数，无碍）；输出为 fp32。
  - 真实签名（实测）：`MOE_BF16(cfg, w13, w2, w13_scale=0, w2_scale=0, w13_global_scale=0, w2_global_scale=0)`；`cpu_decode(stream, batch, k, input_gpu, expert_ids_gpu, weights_gpu, output_gpu)`（GPU 缓冲，decode 路径）；`cpu_prefill(qlen, k, expert_ids, weights, input, output)`（全 CPU，prefill 路径）；均吃整型 data_ptr。
- [ ] **待办**：在 Lvllm 上以 `LVLLM_MOE_CPU_LAYER=...` 跑通一个真实 BF16 MoE 层（需引入 `backend_numa` + libnuma 链接，且核对真实喂入的 `w13/w2` 权重布局——见 ABI spec / 集成验证；真实模型权重维度均为 16/32 倍数，可直接喂）。
- **验收**：单层数值正确性 + 性能。
- 注：Phase 1 以"干净自含闭环"实现；`backend_numa` 的 work-stealing/NUMA 亲和留待 Phase 2/4 接入（`csrc/scheduler/` 已放好，接口 `do_k_work_stealing_job(k, nth, init, compute, finalize)` + `allocate_aligned_numa`）。

### Phase 2 — 扩展量化类型（FP8 / WNA16 / MXFP4） ✅
- ✅ FP8（e4m3 per-tensor）`moe_v2_fp8.hpp` — A/B 通过（mine-vs-numpyref 1.7e-3）。
- ✅ MXFP4（E2M1，groupK=16）+ WNA16（int4 center-8，groupK=32）`moe_v2_packed4.hpp` — numpy-ref 通过（rel 3.3e-3 ~ 5.4e-3）。
- ✅ 编译期分派（CRTP `WeightTraitsBase`）+ 10 类实例化（`MOE_FP8`/`_FP16`、`MOE_MXFP4`/`_FP16`、`MOE_WNA16`/`_FP16`）。
- 注：WNA16 LUT 采用 ktransformers GPTQ 约定（nibble-8），与 vLLM-Marlin 的精确匹配需真实 WNA16 模型 A/B（Phase 4）。

### Phase 3 — NVFP4 + GPU 端 ✅（方向已定并落地）
- ✅ **Phase 3a — NVFP4（策略 A，复用 MXFP4 内核 + vLLM NVFP4 语义 + 每专家 global scale）**：`moe_v2_packed4.hpp` 以 `Packed4WeightTraitsBase<E2M1, NVFP4Tag>` 复用共享 E2M1 内核对齐 vLLM NVFP4 语义（`_process_nvfp4`：w13 pack [E,2I,H/2]、w2 [E,H,I/2]（沿 K 打包，低 nibble=偶 k），block scale [E][N/1][K/32]，global scale 独立 [E] 且 `need_reciprocal_global_scale` 时取 1/gs），构造 arg 5,6=globals。冒烟：MXFP4×2 / WNA16×2 / NVFP4×3 均 PASS（NVFP4 rel 3.6e-3~4.3e-3，含每专家 global scale）；pybind 冲突已用 `Tag` 模板参解决（MXFP4/NVFP4/WNA16 成为不同 C++ 类型）。**cp310 + cp312 均已重建并验证。**
- ✅ **Phase 3b — 决策：xiaotu-moe 纯 CPU，GPU 侧完全复用 fork 的 vLLM 标准 GPU MoE（不手写 CUDA）**。早已删除自写 `csrc/cuda/moe_v2_gpu_{memory,metadata,prefill}.cu`、`moe_v2_gpu_binding.cu`、`moe_v2_gpu.hpp`、`scripts/build_gpu.sh`、`build/_xiaotu_moe_Cgpu*.so`。
- **lk_moe CUDA 的两个角色（理解记录）**：
  1. **浅层 GPU 缓冲/流对接（必要）**：`cpu_decode(stream, input_gpu, output_gpu, ...)` 需要 GPU 侧临时缓冲（`RoutedExperts.output_gpu`）与当前 CUDA 流指针，仅作数据搬移/同步对接，供 decode 在 CUDAGraph capture 时把 CPU 结果写回 GPU。这片始终要用，xiaotu-moe 通过 Python 层的 `_cpu_decode` 契约（收 `stream`+data_ptr）即可对接，**无需 CUDA 核**。
  2. **深层自研 gpu_prefill CUDA 核（冗余，被 vLLM 覆盖）**：仅服务"非 gpu_resident 层 + 大批量 + `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE>0`"。默认变量为 `"0"` → **默认关闭**；且 CUDAGraph/stream capture 时被强制禁用。它是作者在"权重留在 CPU 省显存"架构下为大批量 prefill 自研的混合核，本质是 vLLM 标准 GPU MoE 的替代/补强。**正确的复用姿势是把该层标成 GPU-resident（`LVLLM_GPU_RESIDENT_MOE_LAYERS`）走 `forward_monolithic`/`forward_modular`，或不开 gpu-prefill 环境变量——两者都不碰闭源核。** xiaotu-moe 不实现、不复制该核。
- **对 xiaotu-moe 的含义**：我们只需服务 `_cpu_decode`（decode/CUDAGraph）与 `_cpu_prefill`（普通 prefill），把 `self.lk_moe` 换成 `xiaotu_moe` 实例即可；GPU 大批量 prefill 一律交给 fork 的 vLLM 标准 GPU MoE（标 GPU-resident）。
- **Phase 3b 补充评估：GPU mass-prefill 的"复用 vLLM 算子 + 薄 gather 层"方案（评估结论，已确认不落码）**：
  - **结论：可行，且是比"复刻 `moe_v2_gpu_prefill.cu`"更对的落地方式。** 核心判断——权重一旦进入 GPU HBM，矩阵乘/反量化/SiLU/top-k 加权累加的**数学与 GPU-resident 层逐位一致**；所谓"特有功能"不是计算，而是**数据搬移 + 缓冲管理**：routing 逐层独立，须按每层活跃专家子集把权重从 CPU(pinned)拷入 GPU scratch 再调 vLLM 核。这是一块很薄的 gather 层（纯 torch/copy），**不需要写任何新 CUDA 计算核**，因此完全不碰闭源 `.cu`、零版权风险。
  - **性能胜负手在"带宽 regime"，不在"会不会写核"**（本地 DeepSeek-V4-Flash-0731 量化：H=4096、2I=4096、E=256、4bit）：每层权重 ≈ w13 `256·4096·4096/2`≈2.15 GiB + w2 `256·4096·2048/2`≈1.07 GiB ≈ **~3.2 GiB/层**；A100-PCIe(Gen4 x16) ~32 GB/s → 整层搬 ≈105 ms/层、43 层≈4.5 s；而 CPU 本地 DRAM(EPYC 9654 12ch DDR5)~400 GB/s，比 PCIe 快一个数量级。
    - **带宽bound**(小/中 batch)：权重读取占主导，PCIe 搬运开销 > GPU 算力收益 → **CPU 赢**。
    - **FLOP bound**(真正大批量)：权重仅被流一遍、被大量 token 摊销，PCIe 一次性开销 < GPU ~40x FLOPs 收益 → **GPU 赢（须靠 prefetch/pipeline 隐藏 PCIe；fork 的 `LVLLM_GPU_PREFETCH_WINDOW` 即为此——算第 L 层时预取第 L+1 层权重，双层流水线）**。
  - 这正好解释了闭源实现为何做成"默认关闭（`LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` 默认 0）+ 大批量阈值 + prefetch"，而非默认全体走 GPU。
  - **落地形态（保留为可选 extras，不写码）**：核心纯 CPU 不变；如未来要接 `_gpu_prefill`，做一个薄适配器——CPU pinned 权重 + 逐层 gather 活跃专家进 GPU scratch + 调 vLLM 成熟算子（scaled_mm/marlin/fused_moe），并沿用 min_batch_size 闸门以免小 batch 更慢。

### Phase 4 — 打包与集成验证 ✅（构建+动态选择已完成；集成验证进行中）
- ✅ 多 `.so` 变体构建：`scripts/build_variants.sh` 从同一 `binding.cpp` 编译 5 个 ISA 变体（`scalar`/`avx2`/`avx512_base`/`avx512_vnni`/`avx512_bf16`），pybind 模块名用 `-DXIAOTU_MOE_MODULE_NAME` 单 token 区分（`_xiaotu_moe_C_<variant>`）；avx512_amx 因本机 AMD 无 AMX 跳过。修了一个坑：avx512 变体需显式 `-mfma`（否则 `__AVX2__` 定义但 `__FMA__` 未定义，`_mm256_fmadd_ps` 报 target mismatch）。
- ✅ `_dynamic_loader`：`xiaotu_moe/loader.py` 按 `/proc/cpuinfo` flags 最高优先选支持变体（阶梯 avx512_bf16→vnni→base→avx2→scalar），`xiaotu_moe/__init__.py` 导入即锁定期望变体并重导出全部 12 类。pybind 类型注册表是进程级全局 → **一个进程只能持一个变体**，验证脚本 `scripts/verify_variants.py` 逐变体在独立子进程跑（MXFP4/WNA16/NVFP4 各用例），cp310 + cp312 **ALL PASS**。
- ✅ **真实模型引擎级 A/B（DeepSeek-V4-Flash-0731 layer-1 路由专家，`scripts/_prep_real.py` + `scripts/ab_test_real_nvfp4.py`，纯 CPU）**：核实了该 checkpoint 的路由专家格式 —— **全尺寸 int8，每字节沿 K 打包 2 个 E2M1 码（低 nibble=偶 k、高 nibble=奇 k）+ fp8_e8m0（2 的幂）逐块 scale（块=32 沿 K，每输出行一个），反量化 = E2M1[nibble] × 2^(e8m0_byte−127)**；scale 形状 (2048,128)/(4096,64) 正对应 groupN=1、groupK=32。把这组真实权重按 MOE_NVFP4 契约喂 xiaotu-moe（block scale=2^(e8m0−127) fp32、global=1），与 numpy golden 对比：**max abs err 1.9e-6、有效格子(|golden|>0.05) max rel 8.7e-6 ≈ 逐位一致**。这把 **E2M1 LUT、沿 K 打包、e8m0→fp32 块 scale 解释、SiLU 门控 + 加权累加语义全部锁定到真实模型数据**，直接证实本模型（FP4 专家）正是 MOE_NVFP4 的适用场景、且 NVFP4 策略 A 端到端成立。
  - 注：WNA16（int4 center-8）LUT 属 GPTQ 型模型（非本 fp4 模型），无法用本模型校验；其与 vLLM-Marlin 的精确匹配仍留待真实 WNA16 模型（Phase 4 剩余项，仅当出现此类模型时）。
- [x] **集成验证**：真实模型端到端已通过（见下「Phase 4 集成验证完成」）。
- ✅ **Phase 4 集成前的 scale 契约决定性修复（本轮）**：经真实 lk_moe 实测确认 **此 fork 喂 MXFP4 的是「原始 fp8_e8m0fnu 一字节指数 scale」而非 fp32，且 groupK=32（不是 ABI spec 早先误写的 16）**——`Mxfp4MoEMethod` 建 `w13_weight_scale [E][2I][H/32]`、`w2_weight_scale [E][H][I/32]` 为 `scale_dtype=uint8`，`_process_mxfp4` 直接 `data_ptr()` 喂 lk_moe；真实 lk_moe 按 `2^(byte−127)` 解读（raw/golden 比率中位数 1.0001 ≈ 逐位一致；喂 fp32 反而 3e4 爆炸，证其按字节读）。据此给 `Packed4WeightTraitsBase` 加 `bool E8M0` 模板参 + 256 项 e8m0 查找表，`MOE_MXFP4` 用 `E8M0=true`（WNA16/NVFP4 保持 fp32）。修了一个字节偏移 bug（fp32 scale 分支 sbase 位移曾按 1 字节而非 4 字节，导致读错专家 scale）。cp310+cp312 × 5 变体 verify **ALL PASS**；真实 layer-1 A/B：`MOE_MXFP4`(raw e8m0) max rel **3.4e-4**、`MOE_NVFP4`(fp32+global) max rel 8.7e-6，均远优于 lk_moe 的 22% 尾部。**本模型的真实服务引擎是 MOE_MXFP4（非 NVFP4）**——集成时喂法已锁定。脚本：`scripts/_ab_real_lkmoe_mxfp4.py`、`scripts/ab_test_real_mxfp4.py`、更新 `scripts/smoke_test_packed4.py`/`scripts/verify_variants.py`。

### Phase 4 集成设计（已核对，补丁已就绪；落地与全量层回归留给"测试最后统一做"）
- **全模型规模性能排练（本轮，`scripts/bench_mxfp4.py`，合成 256 专家满尺寸 MOE_MXFP4，avx512_bf16 变体）**：每层权重 ~3.2GB，`cpu_prefill` 每层 CPU 时间 —— batch 1→87.8ms、8→78.7ms、16→79.3ms、32→81.6ms、64→89.7ms（≈带宽 bound，几乎与 batch 无关；每 token:batch64→1.4ms）。43 层 batch64 ≈ **3.9s/解码步** —— 对交互式解码过慢，印证该 3.2GB/层模型实际部署应走 **GPU-resident / vLLM 标准 GPU MoE**（或 deep_gemm），CPU xiaotu-moe 面向小 MoE 或指定离图层（CPU 侧只搬活跃专家子集）。**这就是集成性能回归的每层基准**（替换 `self.lk_moe` 后同场景对比 lk_moe vs xiaotu-moe 的 ms/层）。
  - 优化机会（集成后视需要评估）：当前 `forward_many` 对每 token×每 topk 逐个读专家权重；若改为**按专家分组**（同层内先把所有 token 归并到活跃专家集合，每个活跃专家只读一次、批量 GEMM），内存流量从 `batch×top_k×12MB` 降到 `min(活跃专家数,256)×12MB`，batch64 时 ~7 倍带宽节省。是否值得做由实测吞吐决定，留给集成性能优化。
- **签名 1:1 兼容（已核对 `binding.cpp` vs fork 调用）**：
  - `cpu_decode(stream, qlen, top_k, hidden, expert_ids, weights, out_gpu)` ↔ fork `_cpu_decode()`（`stream,batch,k,input_gpu,expert_ids_gpu,weights_gpu,output_gpu`）。
  - `cpu_prefill(qlen, top_k, expert_ids, weights, input, output)` ↔ fork `_cpu_prefill()`（`qlen,k,eids_cpu,wgt_cpu,x_cpu,out_cpu`）。
  - 构造签名 7 参 `(cfg, w13_weight, w2_weight, w13_scale, w2_scale, w13_global_scale, w2_global_scale)` 与 fork `_process_*` 的喂入一致；w13/w2 layout 已在 Phase 1 A/B 证实逐值兼容。**→ 把 `self.lk_moe` 替换为 `xiaotu_moe.load()` 实例即可，moe_runner 的分派无需改动。**
- **唯一开放的集成约束：`_cpu_decode` 在 CUDA graph capture 时传入的是 GPU 指针**；xiaotu-moe 纯 CPU 不能读 → fork 侧把 `_cpu_decode` 桥接成 `_cpu_prefill` 模式（GPU→CPU 拷→CPU 算→CPU→GPU 写回固定 `output_gpu` 缓冲）。
  - **✅ 补丁已就绪（本轮）**：`integration/routed_experts.swap_xiaotu.patch`（把 `lk_moe.MOE_*`/`MOEConfigV2` 模块级构造换成 `xiaotu_moe.*`，`self.lk_moe` 实例方法保留）+ `integration/routed_experts.cpu_decode.patch`（`_cpu_decode` 纯 CPU 桥接，GPU→CPU→固定 `output_gpu` 写回）+ `integration/apply_integration.sh` 一键应用；均 `git apply --check` 通过，组合后 `py_compile` 通过。完整 runbook `integration/LVLLM_BRIDGE.md`。
  - 更简单备选：纯 CPU 部署时让 fork 的 CPU 层**不进 CUDA-graph capture**（或对离图层关闭 cudagraph），decode 走 `_cpu_prefill` 型路径。**此项在最终集成测试会话内决定并实测。**

### Phase 4 集成验证完成 ✅（真实 DeepSeek-V4-Flash 全模型端到端，2026-09-02）
- **在真实 checkpoint 服务器集成下遭遇并定位了首个端到端 segfault**（EngineCore 崩溃指令 `vmovd (%r12,%rcx,1),%xmm0`——MXFP4 `gate_up` 内沿 Wrow 读权重，fault addr=R12）。逐层排除：
  - 排除并发/竞态：崩溃发生在 `bound`-thread 并发池内，但改**单线程 pre-scan** 直接读同一地址**照样崩** → 纯内存布局问题、非线程逻辑。
  - **根因（`/proc/self/maps` 铁证）**：vLLM 加载完成后调用 `clean_weights_after_loading` **`delattr(self, 'w13_weight')` 等参数** → torch 释放原张量 → 内存页被归还/置 `PROT_NONE`（maps 显示 layer-0 的 `---p` 匿名保护区恰好覆盖 `w13→w13+2.13GB`，即专家 254 块）。而 fork 在 `_process_*` 时把 **`data_ptr()` 裸指针**传给引擎；关内自带的 `_lm._lm` 若只存指针、前向时才去读 → **悬垂指针** → 挂。
  - **对照验证**：还原回 fork 原版（闭源 `import lk_moe`）+ 同一 checkpoint + 发真实 prompt → **正常输出 "Paris. The capital of Spain is Madrid"**（HTTP 200）。即 lk_moe 自带的引擎在**构造时就把权重 COPY 进自身内部缓冲**，故不依赖 vLLM 原始 param 张量的存活。
- **修复（与 lk_moe 行为对齐）**：`MOE_V2<WeightTraits,…>` 构造器改为**把 w13/w2/scale 拷贝进引擎独占 `unique_ptr<uint8_t[]>` 缓冲**，成员指针指向副本（`moe_v2.hpp`）。一次拷贝 3.2GB/层 ×43 ≈ 137GB，但 vLLM 随后 `clean_weights_after_loading` 释放原 param，净占与原部署相当（机器内存 1.3TB，实测空闲 ~1.18TB）。代价：每层构造慢 ~2.7s（内存复制带宽 bound），全模型加载 ~127s。
- **端到端最终验证（干净构建，无探针）：** avx512_bf16 变体 + `routed_experts.swap_xiaotu.clean.py` + 同一 checkpoint → `Application startup complete`、发 prompt `The capital of France is` → **HTTP 200，输出 " Paris. The capital of Spain is Madrid"（与 lk_moe 逐字节一致），0 segfault**。`max_tokens=60` 长生成（28 tokens 俳句）亦稳定。cp310+cp312 × 5 变体 `build_variants.sh` ALL 重建通过。
- **沉淀**：干净的模块替换文件 `integration/routed_experts.swap_xiaotu.clean.py`（fork 源 + 全量 `lk_moe.*→xiaotu_moe.*` swap，无调试探针）已安装到 fresh env site-packages。根因与复现链详见 `LK_MOE_INVESTIGATION_REPORT.md` §6。

---

## 5. 工作量评估

| 模块 | 来源 | 工作量 |
|---|---|---|
| 调度器（backend_numa） | 复用 lktransformers | 低（近乎直接搬） |
| MoE 编排（moe.cpp 逻辑） | 复用 lktransformers + 模板化 | 中低 |
| AMX 量化 GEMM（BF16/FP8/MXFP4） | 复用 ktransformers amx/*.hpp | 中低 |
| WNA16（Marlin 型） | 参照 ktransformers awq-moe + vLLM 语义 | 中 |
| NVFP4 | 复用 MXFP4 内核（策略 A） | 中高 |
| GPU 端（cuda 三件套） | ✅ 不重写——纯 CPU，GPU 复用 fork vLLM 标准 GPU MoE | 已消除 |
| Python 绑定 / ABI / 打包 | 自研 | 中 |

**总评估**：一个熟练工程师，BF16 最小可用闭环约 1–2 周；全量化类型 + GPU 端完整版约 6–10 周。**大部分工作量不在调度（可复用），而在量化 GEMM 内核的具体每种格式适配 + NVFP4/GPU 自研。**

---

## 6. 法律与授权边界（重要）

1. **可复用**：ktransformers、lktransformers 均为 **Apache-2.0**。调度骨架、MOE 编排、AMX 量化内核可直接作为基础。
2. **勿复制**：lk-moe 的 V2 模板、`csrc/cuda/moe_v2_gpu_*.cu`、`_dynamic_loader` 内部实现、`MOE_V2` RTTI 实例化代码——这些是 Qiong GU 专有。
3. **本仓库用途**：逆向（strings/symbols/diff）仅为**互操作与理解**，输出的是功能等价的开源实现，不逐字搬运专有代码。
4. **THIRD_PARTY_LICENSES 已核实**：作者自身点名 3 份文件（moe.h/moe.cpp/backend_numa.cpp）衍生自 KVCache.AI ktransformers（Apache-2.0），即为本方案复用范围的官方背书。
5. **发布命名**：已定名 **`xiaotu-moe`**（独立包名），**不要冒充** `lk-moe` 官方包，避免混淆与商标问题。

---

## 7. 风险与开放问题

| 风险/问题 | 说明 | 缓解 |
|---|---|---|
| MOEConfigV2 精确 ABI | 闭源，可能随版本变 | Phase 0 逐字段核定 + 版本锁定 2.3.3 |
| NVFP4 无开源内核 | ktransformers 只有 MXFP4 | 3.4 策略 A/B/C |
| GPU 端三件套重写 | ~✅ 已消除：xiaotu-moe 纯 CPU；GPU 复用 fork vLLM 标准 GPU MoE | 层标 GPU-resident 走 forward_monolithic/modular |
| gpu_prefill 深层核 | ✅ 已消除：默认关闭、被 vLLM 覆盖 | 不复制；交由 vLLM 标准 GPU MoE |
| WNA16/Marlin 布局 | 需精确匹配压缩权重格式 | 参照 vLLM `CompressedTensorsWNA16MarlinMoEMethod` 输入 |
| 性能对齐 | 开源内核与闭源优化可能有差距 | 以"不劣于 lktransformers 原版"为基准，不强行追闭源峰值 |

---

## 8. 下一步（进入代码前的最后确认）

1. **定 NVFP4 策略**（A/B/C）。
2. **导出 MOEConfigV2 精确布局**并锁定为 ABI 规范。
3. **确认包名/发布命名**。
4. 以上确认后，从 **Phase 1（BF16 最小闭环）** 开始写代码。

> 本方案目前不涉及任何代码编写。待你批准后，从 Phase 1 启动。
