# lk-moe 调查报告

> 适用范围：Lvllmds4-x 及其上游 lk_moe 闭源推理引擎的完整溯源与代码来源核实。
> 对象版本：lk_moe 2.3.3（Lvllmds4-x 锁定版本）；vLLM fork 基线上游 v0.23.1rc1。
> 调查日期：2026-09-01 前后。

---

## 0. 结论速览（TL;DR）

1. **Lvllmds4-x 是一个 vLLM 深度 fork**：在 yhfgyyf/vllm-deepseek-v4-sm89 + jasl/vllm SM12x 之上，把 `fused_moe` 的 GPU 前向替换为 **lk_moe 闭源引擎**的 CPU-GPU 混合分发调度。它比主线 vLLM 落后约 5~6 个 minor 版本（基线 v0.23.1rc1 vs 最新 0.28.0），且活跃开发大约停在 2026-07 前后。
2. **lk_moe 是闭源二进制**：PyPI 仅有 `cp312 manylinux_2_34 x86_64` wheel，无 sdist、无公开源码仓；许可证为专有"Proprietary Software License Agreement"（签署人 Qiong GU / guqiong9696@gmail.com，2025-11-08）。
3. **但 lk_moe 并非凭空闭源**。它派生自开源的 ktransformers 谱系。**作者随包提供了 THIRD_PARTY_LICENSES，正式点名三份衍生自 KVCache.AI 的代码**（详见 §3）。二进制层亦以逐字节字符串印证（含原作者拼写错误）。
4. **量化计算内核已被作者 V2 重写**为专有 `MOE_V2<WeightTraits, ActivationType>` 模板（10 个实例化），并剥离了上一代的 ggml/llamafile 依赖（.so 中 ggml/llamafile 字符串为 0）。
5. **因此"开源重实现 lk-moe"是可行的**：调度器（Backend_NUMA）与 MoE 骨架开源可查；真正需要自己写的是**量化 GEMM/反量化内核**（FP8/BF16/FP16/WNA16/NVFP4/MXFP4）。ktransformers / lktransformers 的相应内核可作参考。

---

## 1. Lvllmds4-x fork 在 vLLM 之上改了什么

### 1.1 分层
```
上游 vLLM
   └─ yhfgyyf/vllm-deepseek-v4-sm89  +  jasl/vllm（SM12x）
        └─ guqiong96/Lvllmds4-x（本次分析对象）
```
基准 tag `lvllmds4-x-v2.3.11`，底层 vLLM v0.23.1rc1；作者 guqiong96 在 fork 内约 14 commits。

### 1.2 核心改动点（对上层 vLLM 而言）
- **`vllm/model_executor/layers/fused_moe/routed_experts.py`** —— GPU 前向被替换：
  - `forward()` (L1191) 直接 `raise AssertionError`，要求走 `forward_modular` / `forward_monolithic`（即改由 lk_moe 引擎调度）。
  - `_zero_tensor` (L1234)、`process_weights_after_loading` (L1275，按 `[GPU]/[CPU]` 打印并 `_offload*` 权重)、`clean_weights_after_loading` (L1296，delattr 权重)。
  - 量化处理器：`_process_wna16`、`_process_awq`（抛错，不支持）、以及 fp8/nvfp4/mxfp4/bf16 处理器 → 构造 `MOEConfigV2` 并实例化 `MOE_NVFP4/MOE_MXFP4/MOE_FP8/MOE_BF16/MOE_WNA16[_FP16]`，把权重 `.data_ptr()` 传给引擎。
  - `_initialize_cuda_graph_buffers` (L1694) —— CUDA Graph 侧缓冲。
- **`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`** —— 混合分发 (L557-609)：
  - GPU-resident > CUDA-graph capture (`_cpu_decode`) > `gpu_prefill`（batch ≥ 阈值）> `_cpu_prefill`。
- **`vllm/envs.py` (L2193-2284)**：`LK_*` / `LVLLM_*` 环境变量与辅助函数。
- **`vllm/utils/numa_utils.py`**：numaactl 的 NUMA 交织覆盖。
- **`vllm/model_executor/model_loader/utils.py`**：权重加载钩子。

### 1.3 能力门控（`is_device_capability_family(80)`）
SM80/SM86（无 FP8 tensor core）时：vLLM 侧 Triton 无法表示 `fp8e4nv` → 走 `DECODE_E4M3`（uint8→bf16）；MoE FP4 回退为 Marlin WNA16。

### 1.4 其它
- **DSpark 投机解码**（DeepSeek-V4）+ NUMA 环境 fork。
- 权重数据布局（见 §2.2）为 CPU 引擎定制（stride 32、group 32、packed 位打包）。

### 1.5 修正：Lvllm 与 Lvllmds4-x 是两条平行演进线，非简单父子
> 早期表述将 Lvllm 视为父、Lvllmds4-x 视为直接子。经拉取 `guqiong96/Lvllm` 当前 main（tag `lvllm-v2.3.11`，commit `30b0f47`，2026-08-17）逐文件比对，两仓均为 v2.3.11、均锁定 `lk_moe==2.3.3`、lk_moe 集成机制完全一致，但 **`routed_experts.py` 存在实质差异**，且 **Lvllm 当前 main 反而更新/更全**。结论：

- **两者是共享 lk_moe 集成底座的同源平行分支**：挂载同版 lk_moe，量处理器都构造 `MOEConfigV2` 并实例化 10 个 `MOE_*` 类、`data_ptr()` 传指针。这部分逐字节对应。
- **差异点（已逐一核实，`routed_experts.py`）**：
  | 维度 | Lvllm 当前 main | Lvllmds4-x（fork） |
  |---|---|---|
  | 权重加载映射 | `ckpt_gate_proj_name/ckpt_down_proj_name/ckpt_up_proj_name` + `get_expert_mapping`/`build_expert_params_mapping` | 构造参数 `expert_mapping: list[tuple[str,str,int,str]]`（外部传入） |
  | 复制归一化 | `_to_scalar` / `_orient_fused_weight` | `_normalize_loaded_weight_for_copy`（e8m0→uint8 view 等） |
  | quant method 注册 | 仅 `CompressedTensorsWNA16MoEMethod` 识别 | 额外识别 `CompressedTensorsWNA16MarlinMoEMethod` |
  | 缓冲持久性 | buffers 带 `persistent=False` | 无 `persistent=False` |
  | fused_moe 其余文件 | 有 `aiter_mxfp8_moe.py / hpc_moe.py / trtllm_lora_moe.py / int4_emulation_moe.py` 等 | 这些文件缺失 |
- **`diff -rq` 显示大量 fused_moe 文件 "differ" 的原因**：两者基准点不同（Lvllm main 更新，含更多专家后端），而 fork **没有独有的 fused_moe 源码文件**（除 autotune 基准 `configs/*.json` 工件外）。即 fork 是 Lvllm 早期线的快照 + 少量自有改动（`expert_mapping` 参数化、WNA16Marlin 识别、e8m0 归一化），而非从 Lvllm main 直接分出。
- **对重实现的影响**：lk_moe 的 Python 调用 ABI 在两线上完全一致（`MOEConfigV2` + `data_ptr()` 指针传递），故重实现包可同时服务两线，无需为并行线做接口分化。

### 1.6 谱系定向补充（已核实）
- lktransformers 是从**旧版 ktransformers** fork 的（带 `llamafile/moe.cpp` + `cpu_backend/backend{,_numa}.cpp` + `amx_gemm.cpp`），而当前 ktransformers main 已重构（llamafile 只剩 `conversion.h`，`cpu_backend` 改 `cpuinfer/task_queue/worker_pool`，`backend*.cpp` 移除）。
- 因此重实现的**调度骨架应以 lktransformers 为准**，量化内核以当前 ktransformers main 为准（AMX 的 `mxfp4-moe.hpp` 在 main 已改名 `fp4-moe.hpp`）。详见 `LK_MOE_REIMPLEMENTATION_PLAN.md` §2。

---

## 2. lk_moe 引擎本体逆向

### 2.1 包结构（wheel 2.3.3，~21.7 MB，解压后）
- `__init__.py`：导出 `MOEConfigV2` + `MOE_BF16 / MOE_FP8 / MOE_WNA16 / MOE_NVFP4 / MOE_MXFP4 / MOE_FP16 / MOE_WNA16_FP16`。
- `_dynamic_loader.py`：读 `/proc/cpuinfo`，按 `avx512_amx > avx512_vnni > avx512_base > avx2` 选 `_lk_moe_C_<arch>.so`（5 个 ~22MB 变体，x86_64 专用）。
- 捆绑 `lk_moe.libs/libnuma.so`。

### 2.2 MOEConfigV2 字段与数据布局
- 字段：`num_processes, process_id, gpu_id, has_gate_proj, expert_num(本地), top_k, hidden_size, intermediate_size, max_batch_size, max_num_seqs, stride=32, group_min_len=10, group_max_len, groupN, groupK, activation_type, swiglu_alpha, swiglu_limit, use_gpu_prefill`。
- 布局（vLLM 侧打包后喂给引擎）：
  - **WNA16**：`w13_weight_packed.cpu().transpose(1,2).contiguous().view(uint8)`，2 权重/字节（packed_factor 8 / num_bits 4），group_size 32。
  - **FP8**：raw + `w13_weight_scale`（或 `_inv`），无全局 scale，pack_ratio 1。
  - **NVFP4**：packed + weight_scale + global_scale（可用倒数 1.0/gs），pack_ratio 2。
  - **MXFP4**：packed + per-block scale，无全局 scale，pack_ratio 2。
  - **BF16/FP16**：raw，无 scale。
  - `groupN/groupK` 由 `weight.shape` vs `scale.shape` 反推（`_get_quant_params`）：`groupN = shape[1]/scale.shape[1]`，`groupK = (shape[2]*unpack_factor)/scale.shape[2]`，`unpack_factor = 1 if pack_ratio==1 else 2`。

### 2.3 核心：`MOE_V2<WeightTraits, ActivationType>` 模板
RTTI typeinfo 给出 10 个实例化（与 `__init__.py` 导出一一对应）：

| WeightTraits | Float16 | BFloat16 |
|---|---|---|
| FP8WeightTraits | ✅ | ✅ |
| BF16WeightTraits | — | ✅ |
| FP16WeightTraits | ✅ | — |
| WNA16WeightTraits | ✅ | ✅ |
| NVFP4WeightTraits | ✅ | ✅ |
| MXFP4WeightTraits | ✅ | ✅ |

### 2.4 构建主机泄漏的源码树
.so 内字符串显示构建自：
```
/home/guqiong/Downloads/lk_moe/csrc/cuda/moe_v2_gpu_memory.cu
/home/guqiong/Downloads/lk_moe/csrc/cuda/moe_v2_gpu_metadata.cu
/home/guqiong/Downloads/lk_moe/csrc/cuda/moe_v2_gpu_prefill.cu
```
即作者在闭源仓内维护 `csrc/`（含 `csrc/lk_moe/*` 与 `csrc/cuda/*`）两套源码。

### 2.5 反向量化可行性与事实
- pybind11 .so 使用隐藏可见性，但 **RTTI typeinfo + 字符串表保留**，可逆向。
- 已确认**不含** ggml / llamafile / marlin 字符串（引擎本体）。

---

## 3. 第三方授权文件与代码来源核实（本次核心新增）

### 3.1 THIRD_PARTY_LICENSES 原文要点
`lk_moe-2.3.3.dist-info/licenses/THIRD_PARTY_LICENSES`（26 行）明确列出 3 份衍生自 **KVCache.AI ktransformers**（Apache-2.0）、由 **guqiong9696@gmail.com** 2025 年修改的文件：

| lk_moe 内 | ktransformers v0.2.3 原始 | 修改 |
|---|---|---|
| `csrc/lk_moe/moe.h` | `ktransformers_ext/operators/llamafile/moe.h` | NUMA + 混合推理重构 |
| `csrc/lk_moe/moe.cpp` | `ktransformers_ext/operators/llamafile/moe.cpp` | 同上 |
| `csrc/lk_moe/backend_numa.cpp` | `ktransformers_ext/cpu_backend/backend.cpp` | NUMA 后端重构 |

note：原始 KVCache.AI 代码仍受 Apache-2.0 约束；修改/新代码为 Qiong GU 专有。

### 3.2 源码逐份比对（v0.2.3 ↔ lktransformers ↔ .so）
- 三份 v0.2.3 原始文件头：作者 **chenht2022，2024-07-22，© 2024 KVCache.AI**，与授权文件点名吻合。
- **调度器二进制铁证**（.so 逐字节命中，均出自 `backend_numa.cpp`）：
  - `Backend_NUMA init suscess, numa nodes: `（保留拼写错误 "suscess"）
  - `Backend_NUMA::do_k_work_stealing_job: n_threads == 0`、`do_k_work_stealing`
  - `Using LK_THREADS from environment:` / `Using LK_POWER_SAVING from environment:`
  - `sched_setaffinity failed`、`topology/core_id`、`topology/physical_package_id`
- **MoE 骨架保留、算核重写**：`MOE` 类骨架与 `forward_*` 方法族保留；`forward_many_numa`/`dispatch`/`gate_proj_numa` 等旧成员 0 命中；量化层重写为 `MOE_V2<WeightTraits>`；ggml/llamafile 字符串 0。
- **中间版 lktransformers 加入 AMX/VNNI**：`#if defined(__AMX_INT8__) && defined(__AVX512VNNI__) #include "amx_gemm.hpp"` —— 这是 guqiong96 在 lktransformers 阶段引入的（ktransformers v0.2.3 本身无此条件编译）。

### 3.3 用户假设核对
| 假设 | 结论 |
|---|---|
| 调度/骨架源于 ktransformers | ✅ 成立，作者授权文件正式背书 |
| AVX512/VNNI/AMX 加速 | ⚠️ 部分：技术储备源自 ktransformers llamafile amx_gemm，但由 guqiong96 在 lktransformers 引入；V2 又重写剥离 ggml |
| 调用 Marlin 反向量化 | ❌ 引擎 .so 无 Marlin；Marlin 是 GPU 端独立扩展 |

---

## 4. 完整谱系链

```
ktransformers (KVCache.AI, Apache-2.0)  moe.cpp/h(2024-07-22) backend.cpp(2024-07-22)
        │  下游维护（chenht2022/kkk1nak0/oql…）
        ▼
guqiong96 / lktransformers（NUMA 分支，2025-07~11，开源）
        │  重构 Backend_NUMA；加 AMX/VNNI 条件编译；NUMA 分块内存
        ▼  闭源 V2 提取 + 重写（剥离 ggml/llamafile，新增 csrc/cuda）
lk_moe（PyPI 2025-11-08，专有闭源二进制）
        │  MOE_V2<FP8/BF16/FP16/WNA16/NVFP4/MXFP4 × f16/bf16>
        ▼
Lvllm / Lsglang / Lvllmds4 / Lvllmds4-x（fork 集成）
        │  fused_moe / routed_experts.py + moe_runner.py 混合分发
        ▼
vLLM 侧 GPU 扩展（Marlin WNA16 等，独立于引擎 .so）
```

---

## 5. 对"开源重实现 lk-moe"的意义

- **开源可复用**：调度器 `Backend_NUMA`、MoE 骨架 `MOE::`、NUMA 内存规划、AMX/VNNI 条件编译入口（来自 ktransformers/lktransformers）。
- **需自研**：量化 GEMM/反量化内核（FP8/BF16/FP16/WNA16/NVFP4/MXFP4 的 `MOE_V2` 计算核心）——这是工作量主体，可参考 ktransformers 的 fp8/mxfp4/bf16-moe 内核与 llamafile amx_gemm.cpp。
- **法律**：对 lk_moe 逆向仅用于互操作/理解（用户自持二进制）；重实现应以 Apache-2.0 的 ktransformers/lktransformers 为源码基础，避免复制 lk_moe 专有新代码（V2 模板与 csrc/cuda）。

---

## 6. 重实现引擎的首个端到端 bug：权重悬垂指针 → SIGSEGV（已定位并修复）

> 现象：用 xiaotu-moe 替换 `self.lk_moe` 后，DeepSeek-V4-Flash-0731 真实 checkpoint 在引擎加载完成、首次 CPU 前向时 EngineCore segfault。崩溃指令与根因如下。

### 6.1 崩溃现场
- 崩溃指令（MXFP4 `forward_many` 内联 `gate_up` 的沿 W 读）：`vmovd (%r12,%rcx,1),%xmm0`，R12=W 行基址、RCX=k/2。fault addr = R12 = `w13_ + 254×8MB`（layer-0 的专家 254 块）。该地址被持续、可复现地命中。
- `/proc/self/maps` 铁证：`w13_` 所在的 layer-0 权重区几乎全部落入一段 **`---p`（PROT_NONE）匿名保护区**（`7f9bf0000000–7f9c80000000`，2.25GB）——该内存读/写即 segfault。

### 6.2 根因链
```
vLLM 加载权重 → _process_mxfp4 把 w13_weight.data_ptr() 裸指针传给引擎（此时可读）
   → 加载完成 → vLLM 调 clean_weights_after_loading：delattr(self,'w13_weight')…
   → torch 释放原 param 张量 → 页归还/置 PROT_NONE（map 变 ---p）
   → 引擎若只存裸指针、前向时才读 → 读已变成 PROT_NONE 的内存 → SIGSEGV
```
- 排除竞态依据：改**单线程 pre-scan** 直接读同一地址仍崩（非并发逻辑问题）。
- 对照：还原 fork 原版（闭源 `import lk_moe`）+ 同一 checkpoint + 真实 prompt → 正常输出。**即 lk_moe 自带引擎在构造时就把权重 COPY 进自身缓冲**，故不受 vLLM param 释放影响 —— 这解释了"为什么 lk_moe 没事、我们一替换就崩"。

### 6.3 修复（与 lk_moe 行为对齐）
- `MOE_V2<WeightTraits,…>` 构造器不再保存裸指针，而是**把 w13/w2/scale 拷贝进引擎独占 `std::unique_ptr<uint8_t[]>` 缓冲**，成员指针指向副本（`csrc/moe/moe_v2.hpp`）。代价：每层构造多 ~2.7s（全模型加载 ~127s）、在 `clean_weights_after_loading` 释放原张量前临时多占 ~137GB（机器 1.3TB 内存足够，实测空闲 ~1.18TB）。
- 干净重建 cp310+cp312 × 5 ISA 变体，端到端验证通过（`Application startup complete` + 输出与 lk_moe 逐字节一致 + 0 segfault）。

### 6.4 对 ABI 理解的修正
- 此前 ABI 设计认为引擎持有裸指针即可；**实测表明 fork 期望引擎自己持有所有权（拷贝）**。这是闭源引擎行为的一个此前未记录的事实，已写回 `LK_MOE_REIMPLEMENTATION_PLAN.md` Phase 4 集成验证完成节。
