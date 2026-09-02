#!/usr/bin/env python
"""A/B validation: real (closed-source) lk_moe.MOE_BF16 vs xiaotu-moe MOE_BF16.

Feeds IDENTICAL random BF16 weights/inputs to both engines and compares the
CPU decode outputs. This confirms the xiaotu-moe w13 layout assumption
([E][2I][H], gate-first) and the gate->up->silu->down math against the real
reference implementation.

Run under the lvllmds4-x conda env (has real lk_moe + torch + numpy).
Usage: PYTHONPATH=build conda run -n lvllmds4-x python scripts/ab_test_bf16.py
"""
import sys, os
import torch
import numpy as np
import lk_moe
import _xiaotu_moe_C as om

torch.manual_seed(0)

def bf16_bits(t):
    """torch bf16 tensor -> uint16 tensor of its raw bit patterns (cpu)."""
    return t.cpu().view(torch.uint16).contiguous()

def run_case(E, H, I, k, M, tag):
    # Random weights/inputs.
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16) * 0.3          # gate-first [E][2I][H]
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16) * 0.3              # [E][H][I]
    hidden = torch.randn(M, H, dtype=torch.bfloat16) * 0.3
    eids = torch.randint(0, E, (M, k), dtype=torch.int32)
    topw = torch.rand(M, k, dtype=torch.float32)

    w13_u = bf16_bits(w13); w2_u = bf16_bits(w2); hin_u = bf16_bits(hidden)

    # ---- config (mirror Lvllmds4-x routed_experts._process_bf6_fp16) ----
    def make_cfg(cfgcls):
        c = cfgcls()
        c.num_processes = 1; c.process_id = 0; c.gpu_id = 0
        c.has_gate_proj = 1; c.expert_num = E; c.top_k = k
        c.hidden_size = H; c.intermediate_size = I
        c.max_batch_size = 8192; c.max_num_seqs = 2; c.stride = 32
        c.group_min_len = 10; c.group_max_len = 4096
        c.groupN = 0; c.groupK = 0
        c.swiglu_alpha = 1.0; c.swiglu_limit = 0
        c.activation_type = 0; c.use_gpu_prefill = 0
        return c
    cfg_real = make_cfg(lk_moe.MOEConfigV2)
    cfg_mine = make_cfg(om.MOEConfigV2)

    # ---- real lk_moe: cpu_prefill (all-CPU buffers, no GPU) ----
    # (cpu_decode needs GPU tensors; cpu_prefill is the CPU-only path.)
    real = lk_moe.MOE_BF16(cfg_real,
                           w13_u.data_ptr(), w2_u.data_ptr(), 0, 0, 0, 0)
    out_real = torch.zeros(M, H, dtype=torch.float32)
    real.cpu_prefill(M, k,
                     eids.contiguous().data_ptr(),
                     topw.contiguous().data_ptr(),
                     hidden.data_ptr(),
                     out_real.data_ptr())

    # ---- xiaotu-moe ----
    mine = om.MOE_BF16(cfg_mine, w13_u.numpy(), w2_u.numpy())
    out_mine = np.zeros((M, H), dtype=np.float32)
    mine.cpu_decode(0, M, k,
                    hin_u.numpy(), eids.numpy(), topw.numpy(), out_mine)

    # ---- compare ----
    r = out_real.numpy()
    m = out_mine
    denom = np.abs(r).max() + 1e-9
    max_abs = np.abs(r - m).max()
    rel = max_abs / denom
    print(f"[{tag}] E={E} H={H} I={I} k={k} M={M}: "
          f"out_maxabs={denom:.6f}  diff_maxabs={max_abs:.6f}  rel={rel:.3e}")
    return rel

if __name__ == "__main__":
    cases = [
        (8, 512, 1536, 4, 16, "standard"),
        (16, 128, 384, 6, 64, "small-mult16"),
        (4, 2048, 8192, 4, 32, "llama-scale"),
        (2, 576, 1152, 3, 12, "h36-mult"),
    ]
    # NOTE: real lk_moe's AVX512/VNNI CPU path segfaults on dimensions that are
    # not multiples of 16 (e.g. I=1027/H=509), so the A/B uses only realistic
    # 16-multiple shapes (real model dims are all 16/32 multiples).
    worst = 0.0
    for E, H, I, k, M, tag in cases:
        worst = max(worst, run_case(E, H, I, k, M, tag))
    print("\nworst rel err:", f"{worst:.3e}")
    print("PASS" if worst < 1e-2 else "FAIL")
