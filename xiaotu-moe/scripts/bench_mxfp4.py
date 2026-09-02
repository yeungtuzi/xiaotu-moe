#!/usr/bin/env python
"""Full-scale performance rehearsal of xiaotu-moe MOE_MXFP4 at DeepSeek-V4-Flash
real model dimensions (E=256, H=4096, I=2048, top_k=6, groupK=32, raw e8m0
scales). Times cpu_prefill per MoE layer for a range of decode batches — the
exact per-layer CPU latency the Lvllmds4-x integration perf regression needs.

Synthetic full-scale weights (random bytes) — perf does not need true weights.
Pure CPU, no GPU. Best ISA variant (avx512_bf16) is auto-selected.
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E, H, I, K, GK = 256, 4096, 2048, 6, 32

# ~3.2 GiB total, matches the real per-layer footprint (2.1GB w13 + 1.07GB w2)
SIZE_MB = (E * (2 * I * H // 2 + H * I // 2)) / 1e6
print(f"layer footprint ~{SIZE_MB:.0f} MB (w13+w2 packed + scales)")


def main():
    sys.path.insert(0, REPO)
    import xiaotu_moe as x
    m = x.load()
    print("variant:", x.__variant__)

    rng = np.random.default_rng(0)
    w13 = rng.integers(0, 256, (E, 2 * I, H // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (E, H, I // 2), dtype=np.uint8)
    s13 = rng.integers(100, 145, (E, 2 * I, H // GK), dtype=np.uint8)  # 2^-27..2^17
    s2 = rng.integers(100, 145, (E, H, I // GK), dtype=np.uint8)

    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 256; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    engine = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)

    print(f"\n{'batch':>5} | {'ms/layer':>9} | {'us/token':>9} | {'MoE tok/s':>10}")
    # Routing concentration: XIAOTU_MOE_BENCH_CONCENTRATE=N routes tokens only
    # among experts 0..N-1 (grouped path). Default = E -> diverse (per-token).
    conc = int(os.environ.get("XIAOTU_MOE_BENCH_CONCENTRATE", E))
    for B in (1, 8, 16, 32, 64):
        x_u16 = rng.integers(0, 65536, (B, H), dtype=np.uint16)
        ids = rng.integers(0, min(conc, E), (B, K)).astype(np.int32)
        wts = rng.uniform(0, 1, (B, K)).astype(np.float32)
        out = np.zeros((B, H), dtype=np.float32)
        # warmup
        engine.cpu_prefill(B, K, ids, wts, x_u16, out)
        iters = 5 if B <= 16 else 3
        t0 = time.perf_counter()
        for _ in range(iters):
            engine.cpu_prefill(B, K, ids, wts, x_u16, out)
        dt = (time.perf_counter() - t0) / iters
        ms = dt * 1e3
        us_per_tok = dt / B * 1e6
        print(f"{B:>6} | {ms:>9.2f} | {us_per_tok:>9.1f} | {B/dt:>10.0f}")
    print(f"(routing concentrated over {min(conc, E)} experts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
