#!/usr/bin/env python
"""Reproduce the integration segfault from the EXACT dumped server input.
Loads layer-0's 256 real MXFP4 experts into xiaotu_moe.MOE_MXFP4 (raw e8m0,
groupK=32) and runs the dumped 1024-token input (hidden fp32 -> bf16, real ids,
real weights) exactly once. If it segfaults here, the bug is reproduced and we
can bisect. CUDA_VISIBLE_DEVICES=2 (only GPU2 free). conda env.
"""
import glob
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H, I, GK, LAYER, K = 4096, 2048, 32, 0, 6


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def main():
    d = np.load("/tmp/xiaotu_crash_input.npz")
    hidden_f = d["hidden"]          # [M, 4096] fp32 (exact)
    ids = d["ids"]                  # [M, 6] int32
    weights = d["weights"]          # [M, 6] fp32
    M = ids.shape[0]
    xb = f32_to_bf16_bits(hidden_f)
    print(f"dumped input: M={M} K={K} ids max={ids.max()} min={ids.min()} "
          f"weight range=({weights.min():.3f},{weights.max():.3f})")

    shard = None
    for sh in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with safe_open(sh, "pt") as f:
            if f"layers.{LAYER}.ffn.experts.0.w1.weight" in f.keys():
                shard = sh
                break
    assert shard

    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{LAYER}.ffn.experts.{e}.{sfx}").cpu()
        E = None
        for e in range(512):
            if f"layers.{LAYER}.ffn.experts.{e}.w1.weight" not in f.keys():
                E = e
                break
        assert E
        w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
        w2 = np.empty((E, H, I // 2), dtype=np.uint8)
        s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
        s2 = np.empty((E, H, I // GK), dtype=np.uint8)
        for e in range(E):
            w13[e, 0:I] = get(e, "w1.weight").numpy()
            w13[e, I:2 * I] = get(e, "w3.weight").numpy()
            w2[e] = get(e, "w2.weight").numpy()
            s13[e, 0:I] = get(e, "w1.scale").view(torch.uint8).numpy()
            s13[e, I:2 * I] = get(e, "w3.scale").view(torch.uint8).numpy()
            s2[e] = get(e, "w2.scale").view(torch.uint8).numpy()

    sys.path.insert(0, REPO)
    import xiaotu_moe
    xm = xiaotu_moe.load()
    cfg = xm.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 1024; cfg.max_num_seqs = 3
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 1024 + 128
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    cfg.swiglu_limit = 10.0
    engine = xm.MOE_MXFP4(cfg, np.ascontiguousarray(w13), np.ascontiguousarray(w2),
                          np.ascontiguousarray(s13), np.ascontiguousarray(s2), 0, 0)
    print(f"engine built E={E} with dumped input, running forward...", flush=True)
    out = np.zeros((M, H), dtype=np.float32)
    engine.cpu_prefill(M, K, np.ascontiguousarray(ids), np.ascontiguousarray(weights),
                       np.ascontiguousarray(xb), out)
    print("completed without segfault!, finite=", np.isfinite(out).all(),
          " mean|out|=", np.abs(out).mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
