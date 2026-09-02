# xiaotu-moe：lk-moe 兼容层 ABI 规范（Phase 0 产物）

> 状态：**已锁定**（依据 `lk_moe-2.3.3` 的 `_lk_moe_C_avx512_amx.so` 静态二进制提取 + vLLM 侧调用交叉验证）
> 用途：作为 xiaotu-moe 重实现的唯一 ABI 契约，保证 `import lk_moe` 调用方可无改换用。

## 1. 绑定框架
`_lk_moe_C_<arch>.so` 是 **pybind11** 实现的 CPython 扩展（`PyInit__lk_moe_C`，D 符号含 `pybind11::detail::internals...`）。wheel 为 **cp312 / manylinux_2_34 / x86_64** 专用，仅此一平台。

## 2. 模块导出（__init__.py facade）
- `MOEConfigV2`（配置结构体）
- 10 个 `MOE_*` 类 = 5 种 WeightTraits × 2 种激活，各自构造器签名一致（见 §4）。

## 3. WeightTraits → MOE_* 映射（从 .so RTTI typeinfo 提取，mangled 长度前缀=I 后数字）
| WeightTraits（源码名） | 激活 BF16 | 激活 FP16 |
|---|---|---|
| `BF16WeightTraits` | `MOE_BF16` | —（仅 BF16） |
| `FP16WeightTraits` | —（仅 FP16） | `MOE_FP16` |
| `FP8WeightTraits` | `MOE_FP8` | `MOE_FP8_FP16` |
| `WNA16WeightTraits` | `MOE_WNA16` | `MOE_WNA16_FP16` |
| `NVFP4WeightTraits` | `MOE_NVFP4` | `MOE_NVFP4_FP16` |
| `MXFP4WeightTraits` | `MOE_MXFP4` | `MOE_MXFP4_FP16` |

> 观测到的完整 typeinfo 字符串（共 10 条，一一对应上表）：
> `MOE_V2I15FP8WeightTraits7Float16E` / `...8BFloat16E`、`MOE_V2I16BF16WeightTraits8BFloat16E`、`MOE_V2I16FP16WeightTraits7Float16E`、`MOE_V2I17MXFP4WeightTraits7Float16E`/`8BFloat16E`、`MOE_V2I17NVFP4WeightTraits7Float16E`/`8BFloat16E`、`MOE_V2I17WNA16WeightTraits7Float16E`/`8BFloat16E`
> （`I15/I16/I17` = 标识符名字符串长度，非类型编号。）
> 注意：`BF16WeightTraits` 无 FP16 激活变体、`FP16WeightTraits` 无 BF16 激活变体——仅定义 6+6 中的 10 个非退化组合。

## 4. MOEConfigV2 字段（顺序 = pybind11 `.def_readwrite` 注册序 = rodata 字符串序）
| # | 字段 | 类型（推断） | 说明 |
|---|---|---|---|
| 1 | `num_processes` | int | 并行进程数（TP 规模） |
| 2 | `process_id` | int | 本进程 id |
| 3 | `gpu_id` | int | 关联 GPU id |
| 4 | `has_gate_proj` | bool | 是否有 gate 投影 |
| 5 | `expert_num` | int | 本地专家数 |
| 6 | `top_k` | int | top-k 路由 |
| 7 | `hidden_size` | int | 隐藏维 |
| 8 | `intermediate_size` | int | FFN 中间维 |
| 9 | `max_batch_size` | int | 最大 batch（=max_num_batched_tokens） |
| 10 | `max_num_seqs` | int | 最大 seq 数 |
| 11 | `stride` | int | 分块步长（vLLM 恒设 32） |
| 12 | `group_min_len` | int | 分组下限（vLLM 设 10） |
| 13 | `group_max_len` | int | 分组上限（=max_num_group_batch_size） |
| 14 | `groupN` | int | WNA16/Marlin 分组 N（`_get_quant_params` 算出） |
| 15 | `groupK` | int | WNA16/Marlin 分组 K（同上） |
| 16 | `swiglu_alpha` | float | SwiGLU alpha（可选，默认无） |
| 17 | `swiglu_limit` | float | SwiGLU limit（可选，默认无） |
| 18 | `activation_type` | int（枚举） | 激活类型（BF16/FP16 等） |
| 19 | `use_gpu_prefill` | bool | 是否启用 GPU prefill |

> 顺序注意：`swiglu_alpha` / `swiglu_limit` 在 `activation_type` **之前**；这与早期报告中"activation_type 在 swiglu 前"的初判不同，已按 `.so` 实际注册序修正。

## 5. MOE_* 构造器签名（vLLM 侧调用核实——已逐处理器确认）
两类构造器：

**（a）6 参形**：BF16 / FP16 / FP8 / FP8_FP16 / WNA16 / WNA16_FP16 / MXFP4 / MXFP4_FP16
```python
self.lk_moe = lk_moe.MOE_MXFP4(          # WNA16/FP8/BF16 同理
    self.lk_moe_config,                  # MOEConfigV2
    w13_weight_ptr,                      # void* gate+up 权重
    w2_weight_ptr,                       # void* down 权重
    w13_weight_scale_ptr,                # void* gate+up scale
    w2_weight_scale_ptr,                 # void* down scale
    0,                                   # w13 global_scale（无则 0）
    0,                                   # w2  global_scale（无则 0）
)
```
> MXFP4 实传 `0, 0` 而非 global_scale 指针（`_process_mxfp4`, L1673/1684）。早期"MXFP4 也传 global scale"的初判有误，已修正。
>
> **MXFP4 的 block scale 是「原始 fp8_e8m0fnu 一字节指数」而非 fp32，groupK=32（不是 16）。** 决定性证据（`scripts/_ab_real_lkmoe_mxfp4.py`，真实 lk_moe MOE_MXFP4 喂 DeepSeek-V4-Flash layer-1 专家）：`Mxfp4MoEMethod` 建 `w13_weight_scale [E][2I][H/32]`、`w2_weight_scale [E][H][I/32]` 为 `scale_dtype=torch.uint8`（e8m0 原始字节），`_process_mxfp4` 直接 `data_ptr()` 喂 lk_moe。真实 lk_moe 按 `2^(byte−127)` 解读：raw/golden 比率中位数 1.0001、p10 0.9953/p90 1.0049(≈逐位一致，仅尾部 ±22% 是 lk_moe 自身怪癖，无需复刻)。喂 fp32 反而爆炸(3e4)，证明 lk_moe 按字节读 scale。`_get_quant_params` 算出 `groupK=(H/2*2)/(H/32)=32`、`groupN=1`。**xiaotu-moe 的 `MOE_MXFP4` 已实现 e8m0 解码（`Packed4WeightTraitsBase<E2M1, MXFP4Tag, E8M0=true>`），且比 lk_moe 更贴近真值（max rel 3.4e-4 vs lk_moe 22% 尾部）。**
> - 对照：**NVFP4**（`moe_quant_algo=NVFP4` 才走；本模型 `moe_quant_algo=None` → 走 MXFP4）用 fp32 block scale + per-expert global scale（下（b）。

**（b）8 参形**：NVFP4 / NVFP4_FP16（唯一带 global scale 指针的）
```python
self.lk_moe = lk_moe.MOE_NVFP4(
    self.lk_moe_config,
    w13_weight_ptr, w2_weight_ptr,
    w13_weight_scale_ptr, w2_weight_scale_ptr,
    w13_weight_global_scale_ptr,         # E8M0 单字节指数 scale
    w2_weight_global_scale_ptr,
)
```
> global_scale 来自 `w13_weight_global_scale`（或 `w13_weight_scale_2`），可选 `need_reciprocal_global_scale=True` 时取倒数。

**（c）引擎前向 API（关键——非单一 forward）**：
```python
self.lk_moe.cpu_decode(st, qlen, top_k, hidden.data_ptr(), ids.data_ptr(), wts.data_ptr(), out_gpu.data_ptr())
self.lk_moe.cpu_prefill(qlen, k, ids_cpu.data_ptr(), wts_cpu.data_ptr(), hidden_cpu.data_ptr(), out_cpu.data_ptr())
self.lk_moe.gpu_prefill(hidden.data_ptr(), out.data_ptr(), ids.data_ptr(), wts.data_ptr(), qlen, k, stream_ptr)
```
- `cpu_decode`：带 `stream_ptr`（CUDA stream），输出直接写 GPU buffer（CPU 算、GPU 落缓冲）。
- `cpu_prefill`：纯 CPU 侧输入输出，host 往返。
- `gpu_prefill`：GPU 侧批量 prefill 核。
- 无 CUDA-graph 特化方法——GPU decode 通过 vLLM 侧 `_initialize_cuda_graph_buffers` 的 capture 处理。
- 这三点是引擎真正要实现的**运行时 ABI**，xiaotu-moe 必须精确提供同名方法。

## 6. vLLM 侧喂给引擎的权重布局（供内核兼容）
- **WNA16**：`w13_weight_packed.cpu().transpose(1,2).contiguous().view(uint8)`，`packed_factor 8 / num_bits 4` → 2 权重/字节；group_size 32；分 `groupN/groupK`。
- **FP8**：`w13_weight.contiguous()`，per-tensor scale（无 global scale 时 `0`）；block_quant 时用 `w13_weight_scale_inv`。
- **NVFP4 / MXFP4**：权重 + scale + **global_scale**（E8M0 一字节指数）三件套；vLLM 侧对 e8m0 有 `_normalize_loaded_weight_for_copy` 归一化。
- 全部先 `.cpu()`（卸载到 CPU 再喂引擎），引擎内部再 NUMA 分块（stride 32）。
- **BF16 / FP16（确认，routed_experts.py `_process_bf6_fp16` L1494）**：直接喂原始 `self.w13_weight` / `self.w2_weight` 裸指针（构造 7 参 = cfg + w13 + w2 + 0 + 0 + 0 + 0，无 scale/global_scale）。`params_dtype==bfloat16 → MOE_BF16`，否则 `MOE_FP16`。
  - `w13_weight` 为 vLLM 标准 `[E, 2*I, H]` 折叠布局，**gate 在前、up 在后**（row `0..I-1`=w1/gate，`I..2I-1`=w3/up），与 xiaotu-moe `moe_v2.hpp` 的 `w13_=[expert][2][inter][hidden]`（Wg 块在前）一致，**无需 swap**。
  - `w2_weight` = `[E, H, I]`（down）。
  - `swap_w13_to_w31`（[gate;up]→[up;gate]）仅 FLASHINFER/CUTLASS oracle 后端用，**lk_moe/BF16 路径不用**。
  - config `intermediate_size = intermediate_size_per_partition`（=I，非 2I），引擎按 `2*I` 解算 w13 两块。

## 7. 引擎内部结构（来自符号，供架构对齐）
- 类模板 `MOE_V2<WeightTraits, ActivationType>`。
- 成员函数（符号可见）：`distribute_weights`、`decode_one`、`forward_one`、`forward_many_impl`、`forward_many`、`forward`、`warm_up`、`(GPU)` `moe_v2_gpu_{memory,metadata,prefill}`。
- 反量化/融合内核（符号可见，CUDA 端）：`moe_silu_fuse_kernel<T>`、`moe_swigluoai_fuse_kernel<T>(…, alpha, limit)`（`__half` 与 `__nv_bfloat16` 双实例）。
- 数据成员：`NumaBlock`、`std::vector<void*>`（gate/up/down 的 numa 缓冲）、大小向量——与 lktransformers `MOE` 一致（继承自旧版 ktransformers 骨架）。

## 8. 多 .so 变体（构建目标）
`_dynamic_loader.py` 按 CPU 能力降级选择：
`avx512_amx > avx512_vnni > avx512_base > avx2`（另有一份 `_lk_moe_C.cpython-312-...so`）。
x86_64 专用；绑定 `libnuma.so`。xiaotu-moe 重实现沿用同构的 4 态多 .so 交付。

---

## ABI 兼容验收清单（Phase 4）
- [ ] 上述 19 个字段名/顺序/类型与 `.so` 提取完全一致。
- [ ] 构造器签名：NVFP4 8 参（含 global_scale 指针）、其余 6 参（config + 4 ptr + 0,0），与 vLLM 调用匹配。
- [ ] 运行时 API `cpu_decode` / `cpu_prefill` / `gpu_prefill` 参数序与签名精确匹配。
- [ ] `import lk_moe` 兼容别名（xiaotu-moe 包侧提供 `lk_moe` 命名空间或安装别名）。
- [ ] 4 态多 .so 指令集选择逻辑可用。
