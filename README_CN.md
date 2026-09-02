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

## 性能说明

两项优化已实现并实测（详见[项目报告](xiaotu-moe/docs/XIAOTU_MOE_REPORT_cn.md) §10）：

1. **专家归组** — `forward_many` 按活跃专家各走一次（自适应 grouped vs 逐 token 派发）。**诚实的实测结论：无提速**——每次 gate_up/down 仍按 assignment 读整块 12MB 专家块（块≫L2，跨 assignment 不缓存），所以仅归组并未真正降 DRAM 流量。真正"每块只读一次"需分块/批量 GEMM + register blocking，留作 future work。
2. **Backend_NUMA 等价物** — `csrc/moe/numa_pool.hpp`（`NumaWorkPool`）：持久线程池 + NUMA 节点亲和（每个 worker 用 `sched_setaffinity` 钉到不同物理核、跨 NUMA 节点轮转）+ 动态工作窃取 + 对引擎独占的权重快照缓冲做 NUMA 交织内存（raw `mbind` syscall，**不依赖 libnuma**；开源参考：Apache-2.0 的 ktransformers `backend_numa.cpp`）。其完成屏障已改为逐 worker generation 记录，消除跨代竞态。

## 作者

**大河马（dahema@me.com）** · **由 DeepSeek Harness 辅助**。

## 许可

Apache-2.0（构建于 Apache-2.0 的 ktransformers / lktransformers 之上）。不复制任何 lk_moe 专有代码，仅为互操作/理解之用。谱系与许可详见 `LK_MOE_INVESTIGATION_REPORT.md` §6。

## 文档

- [项目报告（中文）](xiaotu-moe/docs/XIAOTU_MOE_REPORT_cn.md)
- [Project report (EN)](xiaotu-moe/docs/XIAOTU_MOE_REPORT_en.md)
- [Reimplementation plan（中文）](LK_MOE_REIMPLEMENTATION_PLAN.md)
- [Investigation report（中文）](LK_MOE_INVESTIGATION_REPORT.md)
