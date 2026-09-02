#!/usr/bin/env python
"""Engine-level A/B of xiaotu-moe MOE_NVFP4 against REAL DeepSeek-V4-Flash
checkpoint routed-expert weights (layer 1).

Checkpoint ground truth: int8 bytes each holding two E2M1 nibbles packed along
K, plus fp8_e8m0 per-block power-of-2 scales (block=32 along K, one per output
row) => dequant = E2M1[nibble] * 2^(e8m0_byte - 127).

We feed the identical packed bytes with block scale == 2^(e8m0-127) fp32 and
global scale 1 into MOE_NVFP4, so the kernel's E2M1 LUT + packing + SiLU +
weighted-sum must reproduce the intended semantics modulo fp accumulation noise.
This locks the FP4 LUT/packing to genuine model data. (The fork's own
re-quantization/global-scale folding is deferred to final integration testing.)

Pure CPU (third GPU not needed). Run with .search-venv python (numpy only).
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "real_layer1_model.npz")

K = 6  # top_k


def f32_to_bf16_bits(x):
    """Round-to-nearest-even fp32 -> bf16, returned as raw uint16 bits."""
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main():
    d = np.load(NPZ)
    w13, w2 = d["w13"], d["w2"]
    g13, g2 = d["g13"], d["g2"]
    gol13, gol2 = d["gol13"], d["gol2"]
    E, I, H = int(d["E"]), int(d["I"]), int(d["H"])

    sys.path.insert(0, REPO)
    import xiaotu_moe
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 256; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    engine = m.MOE_NVFP4(cfg,
                         w13.ctypes.data, w2.ctypes.data,
                         g13.ctypes.data, g2.ctypes.data,
                         0, 0)

    rng = np.random.default_rng(1234)
    M = 6
    x = rng.standard_normal((M, H)).astype(np.float32)
    x_bf16_bits = f32_to_bf16_bits(x)              # (M,H) uint16 bf16 storage
    x_bf16 = bf16_bits_to_f32(x_bf16_bits)         # fp32 holding bf16 values
    ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    out = np.zeros((M, H), dtype=np.float32)

    engine.cpu_prefill(M, K, ids, wts, x_bf16_bits, out)
    got = out

    # ---- golden ----
    golden = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        for r in range(K):
            e, w = ids[t, r], wts[t, r]
            if w == 0.0:
                continue
            g = gol13[e, 0:I] @ x_bf16[t]
            u = gol13[e, I:2 * I] @ x_bf16[t]
            act = u * (g / (1.0 + np.exp(-g)))
            act_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(act))
            down = gol2[e] @ act_bf16
            golden[t] += w * down

    abs_diff = np.abs(got - golden)
    rel = abs_diff / (np.abs(golden) + 1e-6)
    big = np.abs(golden) > 0.05            # exclude near-zero cancellation cells
    print(f"REAL DeepSeek-V4-Flash layer1 fp4 experts: E={E} M={M} K={K}")
    print(f"  max abs err : {abs_diff.max():.6e}")
    print(f"  mean rel err: {rel.mean():.6e}")
    print(f"  max rel err (|golden|>0.05): {rel[big].max():.6e}")
    tol = 2e-3
    ok = rel[big].max() < tol
    print(f"PASS ({'OK' if ok else 'FAIL'})  max rel(|g|>0.05) < {tol}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
