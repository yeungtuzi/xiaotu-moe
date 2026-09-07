# xiaotu-moe

**闭源 `lk_moe` CPU MoE 引擎的开源重实现——可作为 vLLM fork（Lvllm / Lvllmds4-x）的即插即用替代。**

> **中文版** · [**English**](README.md)（默认英文）

---

## 这是什么？

`lk_moe` 是闭源 CPU 推理引擎（专有许可，PyPI `lk-moe==2.3.x`），被（同作者的）vLLM fork **Lvllm**、**Lvllmds4-x**、**lsglang** 用来把混合专家（MoE）前向放到 **CPU** 上运行，而非只放 GPU。这让比显存更大的模型能以 **显存 + 内存**（"1+1=2、100% 显存利用"方案）装入，并支持 GPU + NUMA 混合解码 / 预填充。

虽然 `lk_moe` 闭源，作者随包发布了 `THIRD_PARTY_LICENSES`，正式点名三份衍生自 **KVCache.AI ktransformers（Apache-2.0）** 的文件——NUMA 调度器（`backend_numa.cpp`）与 MoE 骨架（`moe.cpp/h`）。只有**量化计算核心**（`MOE_V2<WeightTraits, ActivationType>`）是专有的。

**`xiaotu-moe`（小兔子）就是开源重实现**：基于 Apache-2.0 零件，提供 ABI 兼容的 CPU MoE 引擎，含 FP8 / BF16 / FP16 / WNA16 / MXFP4 / NVFP4 的量化 GEMM / 反量化内核，以 `self.lk_moe → xiaotu_moe` 直接挂进 fork。

## 特性亮点

- ✅ **ABI 兼容即插即用** — 实现 fork 的 `MOEConfigV2` + `cpu_decode` / `cpu_prefill` 契约；`moe_runner.py` 分派无需改动。
- ✅ **CPU MoE 计算** — BF16/FP16、FP8、WNA16、MXFP4、NVFP4 权重 traits，共用一个 `WeightTraits` 编排循环。
- ✅ **纯 CPU、不手写 CUDA** — GPU 侧完全复用 fork 的 vLLM 标准 GPU MoE（不复制专有 `.cu`）。
- ✅ **多 ISA 打包** — 从单一源码构建 5 个 ISA 变体（`scalar`/`avx2`/`avx512_base`/`avx512_vnni`/`avx512_bf16`）；`_dynamic_loader` 导入时自动选最佳。
- ✅ **不依赖 ggml / llama.cpp / libnuma。**
- ✅ **端到端验证** — 在真实 DeepSeek-V4-Flash-0731 checkpoint 上验证，输出与 lk_moe **逐字节一致**。

## 仓库结构

```
.
├── README.md                 # 本文件（英文，默认）
├── README_CN.md              # 中文版
├── LK_MOE_REIMPLEMENTATION_PLAN.md    # 阶段计划（中文）
├── LK_MOE_INVESTIGATION_REPORT.md     # 谱系溯源/逆向（中文）
├── LK_MOE_ABI_SPEC.md        # ABI 与布局规范（中文）
└── xiaotu-moe/               # Python 包 + 原生源码
    ├── xiaotu_moe/           #   python 包（动态 ISA 加载器）
    ├── csrc/                 #   原生引擎（moe_v2_*.hpp、内核、绑定）
    ├── scripts/              #   构建 + 数值 A/B + bench
    ├── integration/          #   fork 替换补丁 / 干净替换文件 / runbook
    └── docs/                 #   中英双语项目报告
```

## 构建

在 `xiaotu-moe/` 目录下：

```bash
# 为当前 Python ABI 构建全部 ISA 变体（cp310 与 cp312 均已验证）
PYTHON=$(which python) PYBIND11_INC=/path/to/pybind11/include \
  bash scripts/build_variants.sh
```

或构建单个变体：

```bash
PYTHON=$(which python) bash scripts/build.sh   # 构建 avx512_bf16（或最佳支持）
```

运行时加载器（`xiaotu_moe/loader.py`）读取 `/proc/cpuinfo`，在导入时自动加载最佳支持变体。

## 用法（挂进 fork）

1. 按上面构建扩展，使 `build/*.so` 与 `xiaotu_moe` 包同级。
2. 在 fork 的
   `vllm/model_executor/layers/fused_moe/routed_experts.py` 中，把闭源引擎换成我们的。推荐用仓库提供的干净替换文件（`integration/
   routed_experts.swap_xiaotu.clean.py`），它把每个 `lk_moe.*` 重写为 `xiaotu_moe.*`，同时保留 `self.lk_moe` 管线。
3. 启动服务：`LVLLM_MOE_NUMA_ENABLED=1 ... vllm serve ...`。

完整 runbook 与 `_cpu_decode` 桥接方案见 `integration/LVLLM_BRIDGE.md`。

## 正确性

- 纯 CPU 数值 A/B（真实 layer-1 路由专家 vs numpy golden）：MXFP4 max rel ~3.4e-4、NVFP4 ~8.7e-6（远优于 lk_moe 的 ~22% 尾部）。
- 真实 checkpoint 全端到端：真实 prompt 返回与 lk_moe **逐字节一致**，0 segfault。

## 系统要求：透明大页（THP）必须为 `madvise`，不能是 `always`

> **这就是此前报告的原生模式峰值 ~554–657 GB 内存的根因，现已修复——但它是宿主机设置，
> 无法由代码在用户态覆盖。**

原生 CPU MoE 引擎采用 **NUMA 分片权重放置**来提速：每个 worker 只读自己节点的页（无跨节点
流量）。它在每个分片区调用 `madvise(MADV_HUGEPAGE)`，却只以全 stride 写自己的"属主行"
（约为每个区域的 1/8）。若内核级 THP 模式为 `always`，内核会**无视 `madvise` 提示**，把这些
区域提升为 **2 MB 大页**；一旦 2 MB 页内*任意*字节被触碰，**整页 2 MB 都会常驻**。于是散布的
属主行写入把常驻内存放大到约 **5.5 倍单份权重**——是**加载期足迹、之后全程持平**（不是解码泄漏）：
~155 GB 模型实测 ~657 GB，而单份期望约 250 GB。

**修复（在服务宿主机执行一次，需 `root`）：**

```sh
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled   # [always madvise never] 选项
```

保持 `madvise`，并确保引擎不在分片区再次请求大页（serve 脚本默认导出 `XIAOTU_MOE_SHARD_HUGEPAGE=0`）。
这样每个分片只落属主行的 4 KB 页 → 单份足迹（加载 ~157 GB，平台 ~251 GB），同时保留 node-local 读取速度。
如果仍是 `always`，同样的运行峰值 ~657 GB，内存目标不达标——只要 sysctl 是 `always`，
这个开关就**无法从用户态强制回 4 KB 大页**。

**实测验证**（DeepSeek-V4-Flash-0731，~155 GB MoE，双路 EPYC 9654，CPU MoE，TP1，96 线程，
ShareGPT 50 请求 @ 并发 4，`SHARD_HUGEPAGE=0` + `max-num-batched-tokens 4096` + `max-num-seqs 4`）：
峰值 VmRSS **244 GB**、总吞吐 **59.5 tok/s**、平均 TPOT **108 ms**、**50/50** 正确——
均达到/低于 256 GB / 51 tok/s / 177 ms 目标。同一配置在 THP `always` 下峰值 ~657 GB（内存目标未达标）。

## 性能说明

两项优化已实现并实测（详见[项目报告](xiaotu-moe/docs/XIAOTU_MOE_REPORT_cn.md) §10）：

1. **专家归组** — `forward_many` 按活跃专家各走一次（自适应 grouped vs 逐 token 派发）。**诚实的实测结论：无提速**——每次 gate_up/down 仍按 assignment 读整块 12MB 专家块（块≫L2，跨 assignment 不缓存），所以仅归组并未真正降 DRAM 流量。真正"每块只读一次"需分块/批量 GEMM + register blocking，留作 future work。
2. **Backend_NUMA 等价物** — `csrc/moe/numa_pool.hpp`（`NumaWorkPool`）：持久线程池 + NUMA 节点亲和（每个 worker 用 `sched_setaffinity` 钉到不同物理核、跨 NUMA 节点轮转）+ 动态工作窃取 + 对引擎独占的权重快照缓冲做 NUMA 交织内存（raw `mbind` syscall，**不依赖 libnuma**；开源参考：Apache-2.0 的 ktransformers `backend_numa.cpp`）。其完成屏障已改为逐 worker generation 记录，消除跨代竞态。

## 性能（v0.12）

**2.2× 突破**：用 fork 的 vLLM 配 `--compilation_config.cudagraph_mode FULL_DECODE_ONLY`
消除了跨 43 个 CPU MoE 层每步解码的宿主 GPU 内核启动/同步编排。该配置与生产版 lk
服务器一致，是真正的提速杠杆（线程数只是次要因素）。

基准：**DeepSeek-V4-Flash-0731**（43 个 MoE 层全部走 xiaotu CPU 引擎），ShareGPT
**50 prompts / 并发 4 / max-model-len 8192**，双路 EPYC 9654（192 物理核，SMT 关闭）
+ A100/GPU2（tensor-parallel-size 1）。

| 配置 | 总吞吐 (tok/s) | 中位 TPOT (ms) | 50/50 |
|---|---:|---:|---:|
| v0.11 基线（168，cudagraph NONE） | 25.95 | 293.6 | 50/0 |
| **v0.12 @ 168（84/路 · 7/CCD，指定）** | **66.62** | **79.83** | 50/0 |
| v0.12 @ 120（5/CCD 指南） | 77.69 | 73.65 | 50/0 |
| v0.12 @ 96 | 77.75 | 67.29 | 50/0 |

> MoE 解码是**内存带宽受限**：每路 IOD DDR5 带宽是硬上限。**4 核/CCD 已达峰值**；
> 仅当内存带宽更高（如主板把 DDR5 跑到 5600）才**上 5 核/CCD**；切勿超过 5 核/CCD。
> 详细"为什么 96≈120 > 168"的分析见 [`xiaotu-moe/docs/THREAD_GEOMETRY.md`](xiaotu-moe/docs/THREAD_GEOMETRY.md)。

## 已知问题

| 优先级 | 问题 | 状态 |
|---:|---|---|
| 🔴 高 | **原生模式内存占用过高**：未裁剪时真实大模型进程峰值 VmRSS 可达 ~657 GB，远超 ~155 GB 权重工作集与 lk 参考的 256 GB。**已定位根因并修复：宿主机 THP 必须为 `madvise`（系统 `always` 会把 NUMA 分片放大 ~5.5×）。** 设为 `madvise` + `XIAOTU_MOE_SHARD_HUGEPAGE=0` 后峰值降至 **244 GB**。见上文*系统要求*。 | 已解决 |
| 🟡 中 | WNA16（FP8）路径存在既有 `packed4` 打包 bug（flat 与 sharded 均失败）——非主打格式，不影响真实 BF16/MXFP4 推理。 | 已知 |
| 🟡 中 | MXFP4 sharded 池偶发竞态（E=2 H=512 合成形状）——既有问题，与融合/线程/cudagraph 无关，从未影响 50/50 真实基准。 | 已知 |

## 作者

**大河马（dahema@me.com）** · **由 DeepSeek Harness 辅助**。

## 许可

Apache-2.0（构建于 Apache-2.0 的 ktransformers / lktransformers 之上）。不复制任何 lk_moe 专有代码，仅为互操作/理解之用。谱系与许可详见 `LK_MOE_INVESTIGATION_REPORT.md` §6。

## 文档

- [项目报告（中文）](xiaotu-moe/docs/XIAOTU_MOE_REPORT_cn.md)
- [Project report (EN)](xiaotu-moe/docs/XIAOTU_MOE_REPORT_en.md)
- [Reimplementation plan（中文）](LK_MOE_REIMPLEMENTATION_PLAN.md)
- [Investigation report（中文）](LK_MOE_INVESTIGATION_REPORT.md)
