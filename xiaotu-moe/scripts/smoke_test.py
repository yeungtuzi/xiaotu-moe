#!/usr/bin/env python3
"""Phase 1 smoke test: validate xiaotu-moe BF16 closed loop against a NumPy ref."""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "build")
import numpy as np
import _xiaotu_moe_C as moe

def fp32_to_bf16(x):
    # RNE rounding to bf16 (keep top 16 bits of fp32); reinterpret bits, not values
    u = x.astype(np.float32).view(np.uint32)
    r = u + 0x7FFF + ((u >> 16) & 1)
    b = (r >> 16).astype(np.uint16)
    return b

def bf16_to_fp32(x):
    # bit-reinterpret bf16 (uint16) -> fp32
    return (x.astype(np.uint32) << 16).view(np.float32)

def silu(x):
    return x / (1.0 + np.exp(-x))

def ref_forward(cfg, w13, w2, hidden, ids, weights):
    H, I = cfg.hidden_size, cfg.intermediate_size
    E = cfg.expert_num
    out = np.zeros((M, H), dtype=np.float32)
    # w13: [E, 2, I, H], gate block idx 0, up block idx 1
    for t in range(M):
        x = bf16_to_fp32(hidden[t])
        for r in range(cfg.top_k):
            e = ids[t, r]; w = weights[t, r]
            Wg = bf16_to_fp32(w13[e, 0])
            Wu = bf16_to_fp32(w13[e, 1])
            Wd = bf16_to_fp32(w2[e])
            gate = Wg @ x
            up = Wu @ x
            act = up * silu(gate)
            down = Wd @ act
            out[t] += w * down
    return out

# ---------- small deterministic config ----------
np.random.seed(0)
E, H, I, top_k, M = 8, 32, 64, 2, 4
cfg = moe.MOEConfigV2()
cfg.expert_num = E
cfg.top_k = top_k
cfg.hidden_size = H
cfg.intermediate_size = I
cfg.stride = 32
cfg.activation_type = 0

w13 = fp32_to_bf16(np.random.randn(E, 2, I, H).astype(np.float32))
w2 = fp32_to_bf16(np.random.randn(E, H, I).astype(np.float32))
hidden = fp32_to_bf16(np.random.randn(M, H).astype(np.float32))
ids = np.random.randint(0, E, size=(M, top_k)).astype(np.uint32)
weights = np.random.randn(M, top_k).astype(np.float32)

engine = moe.MOE_BF16(cfg, w13, w2, None, None)
out = np.zeros((M, H), dtype=np.float32)
engine.cpu_decode(None, M, top_k, hidden, ids, weights, out)

ref = ref_forward(cfg, w13, w2, hidden, ids, weights)
err = np.abs(out - ref).max()
rel = np.abs(out - ref).sum() / (np.abs(ref).sum() + 1e-12)
print(f"config  E={E} H={H} I={I} top_k={top_k} M={M}")
print(f"max abs err = {err:.3e}")
print(f"rel err     = {rel:.3e}")
assert rel < 5e-3, f"BF16 close loop mismatch too large: rel={rel}"
print("PASS: xiaotu-moe BF16 closed loop matches NumPy reference")

# quick check the MOE_BF16_FP16 class also constructs
e2 = moe.MOE_BF16_FP16(cfg, w13, w2, None, None)
print("PASS: MOE_FP16 constructible")
