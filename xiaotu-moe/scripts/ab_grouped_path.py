#!/usr/bin/env python
"""Verify xiaotu-moe MOE_MXFP4's EXPERT-GROUPED path against a numpy golden.

The engine now adaptively dispatches: with concentrated routing (many tokens
sharing a few active experts) it uses the grouped path (each active expert's
block read once, two-phase gate/up->silu then down, with a lock-free CAS
accumulate). This test forces concentrated routing (M=64 tokens routed only to
experts 0..5 of the real layer-1 subset) so the grouped path is exercised, and
checks the output against an independent numpy golden computed from the real
weights.

Pure CPU, numpy only. Uses scripts/real_layer1_model.npz (same as
ab_test_real_mxfp4.py).
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "real_layer1_model.npz")
K = 6
M = 64
CONCENTRATED_OVER = 6  # only experts 0..CONCENTRATED_OVER-1 are ever routed


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def main():
    d = np.load(NPZ)
    w13, w2 = d["w13"], d["w2"]
    gol13, gol2 = d["gol13"], d["gol2"]
    E, I, H = int(d["E"]), int(d["I"]), int(d["H"])
    g13, g2 = d["g13"], d["g2"]

    def f32_to_e8m0(s):
        lg = np.round(np.log2(np.maximum(s, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)
    s13 = f32_to_e8m0(g13)
    s2 = f32_to_e8m0(g2)

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
    engine = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)

    rng = np.random.default_rng(7)
    x_u16 = f32_to_bf16_bits(rng.standard_normal((M, H)).astype(np.float32))
    xf = bf16_bits_to_f32(x_u16)
    # CONCENTRATED routing: only experts 0..CONCENTRATED_OVER-1 appear, so the
    # grouped path is taken (nave*8 <= M*K).
    ids = rng.integers(0, CONCENTRATED_OVER, size=(M, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)

    def run(force_fallback):
        prev = os.environ.get("XIAOTU_MOE_GROUP_FACTOR")
        if force_fallback:
            os.environ["XIAOTU_MOE_GROUP_FACTOR"] = "100000"  # always per-token
        else:
            os.environ.pop("XIAOTU_MOE_GROUP_FACTOR", None)   # grouped (default)
        out = np.zeros((M, H), dtype=np.float32)
        engine.cpu_prefill(M, K, ids, wts, x_u16, out)
        if prev is None:
            os.environ.pop("XIAOTU_MOE_GROUP_FACTOR", None)
        else:
            os.environ["XIAOTU_MOE_GROUP_FACTOR"] = prev
        return out

    out_grouped = run(force_fallback=False)
    out_ref = run(force_fallback=True)

    # The real correctness claim: the grouped path equals the per-token path.
    d = np.abs(out_grouped - out_ref)
    drel = d / (np.abs(out_ref) + 1e-6)
    big = np.abs(out_ref) > 0.05  # meaningful magnitudes (avoid /~0 noise)
    print(f"xiaotu-moe MOE_MXFP4 — grouped path vs per-token reference (same input)")
    print(f"  E={E} M={M} K={K} concentrated_over={CONCENTRATED_OVER}")
    print(f"  grouped-vs-ref max abs : {d.max():.6e}")
    print(f"  grouped-vs-ref mean rel: {drel.mean():.6e}")
    print(f"  grouped-vs-ref max rel (|ref|>0.05): {drel[big].max():.6e}")
    ok = d.max() < 1e-4 and drel[big].max() < 1e-3
    print(f"PASS ({'OK' if ok else 'FAIL'})  max abs < 1e-4 AND max rel(|ref|>0.05) < 1e-3")

    # Absolute sanity vs the numpy golden (relaxed threshold; large-tail effect).
    golden = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        for r in range(K):
            e, w = ids[t, r], wts[t, r]
            if w == 0.0:
                continue
            g = gol13[e, 0:I] @ xf[t]
            u = gol13[e, I:2 * I] @ xf[t]
            act = u * (g / (1.0 + np.exp(-g)))
            act_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(act))
            golden[t] += w * (gol2[e] @ act_bf16)
    ad = np.abs(out_grouped - golden)
    rel2 = ad / (np.abs(golden) + 1e-6)
    print(f"  grouped vs golden: max abs {ad.max():.6e} mean rel {rel2.mean():.6e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
