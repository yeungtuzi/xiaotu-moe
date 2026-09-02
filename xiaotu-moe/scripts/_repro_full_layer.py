#!/usr/bin/env python
"""Reproduce the integration segfault empirically: load ALL 256 routed experts of
DeepSeek-V4-Flash layer-1 into xiaotu_moe.MOE_MXFP4 (conda env, raw e8m0 feed,
groupK=32) and run progressively larger batches M to find where/if it crashes —
isolating an engine-at-scale bug from a vLLM-server coupling issue.

Pure CPU (GPU2 only for safety). CUDA_VISIBLE_DEVICES=2.
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
H, I, GK, LAYER, K = 4096, 2048, 32, 1, 6


def main():
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
        # discover E from config
        E = None
        for e in range(512):
            if not (
                f"layers.{LAYER}.ffn.experts.{e}.w1.weight" in f.keys()):
                E = e
                break
        assert E and E > 0
        print(f"layer {LAYER} has {E} experts")
        w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
        w2 = np.empty((E, H, I // 2), dtype=np.uint8)
        s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
        s2 = np.empty((E, H, I // GK), dtype=np.uint8)
        for e in range(E):
            w1 = get(e, "w1.weight").numpy()
            w2w = get(e, "w2.weight").numpy()
            w3 = get(e, "w3.weight").numpy()
            s1 = get(e, "w1.scale").view(torch.uint8).numpy()
            s2w = get(e, "w2.scale").view(torch.uint8).numpy()
            s3 = get(e, "w3.scale").view(torch.uint8).numpy()
            w13[e, 0:I] = w1.astype(np.uint8)
            w13[e, I:2 * I] = w3.astype(np.uint8)
            w2[e] = w2w.astype(np.uint8)
            s13[e, 0:I] = s1
            s13[e, I:2 * I] = s3
            s2[e] = s2w

    w13 = np.ascontiguousarray(w13)
    w2 = np.ascontiguousarray(w2)
    s13 = np.ascontiguousarray(s13)
    s2 = np.ascontiguousarray(s2)

    sys.path.insert(0, REPO)
    import xiaotu_moe
    xm = xiaotu_moe.load()
    cfg = xm.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 4096; cfg.max_num_seqs = 4096
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    cfg.swiglu_limit = 10.0
    engine = xm.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    print(f"engine built: E={E} H={H} I={I} gk={GK}")

    rng = np.random.default_rng(1)
    for M in (1, 2, 4, 16, 64, 256, 512, 1024, 2048):
        xb = rng.integers(0, 65536, (M, H), dtype=np.uint16)
        ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
        wts = rng.uniform(0, 1, size=(M, K)).astype(np.float32)
        out = np.zeros((M, H), dtype=np.float32)
        try:
            engine.cpu_prefill(M, K, ids, wts, xb, out)
            print(f"M={M:5d} OK  out finite={np.isfinite(out).all()} "
                  f"mean|out|={np.abs(out).mean():.4f}")
        except Exception as ex:
            print(f"M={M:5d} PYTHON EXCEPTION: {type(ex).__name__}: {ex}")
        # force flush so a native segfault is clearly the last line printed
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
