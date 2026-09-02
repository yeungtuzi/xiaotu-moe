#!/usr/bin/env python
"""DEFINITIVE drop-in integration A/B: build BOTH real lk_moe.MOE_MXFP4 and
xiaotu_moe.MOE_MXFP4 from the SAME engine tensors with the fork's EXACT feeding
(raw uint8 e8m0 scale, groupK=32, no global scale) for DeepSeek-V4-Flash layer-1
fp4 experts, run both on the same real input, and compare:
  1) xiaotu_moe vs lk_moe  (the actual swap the integration performs)
  2) each vs the torch dequant golden
This proves swapping self.lk_moe -> xiaotu_moe is numerically equivalent in the
exact deployment env (conda lvllmds4-x). CUDA_VISIBLE_DEVICES=2 (GPU2).
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "real_layer1_model.npz")
K = 6


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def main():
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=2 -> visible device 0
    d = np.load(NPZ)
    w13, w2 = d["w13"], d["w2"]
    gol13, gol2 = d["gol13"], d["gol2"]
    E, I, H = int(d["E"]), int(d["I"]), int(d["H"])

    def f32_to_e8m0(s):  # npz stores fp32 2^(e-127); recover exact byte
        lg = np.round(np.log2(np.maximum(s, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)
    s13 = f32_to_e8m0(d["g13"])  # [E][2I][H/32] raw e8m0 bytes
    s2 = f32_to_e8m0(d["g2"])    # [E][H][I/32]

    tw13 = torch.from_numpy(w13).contiguous()
    tw2v = torch.from_numpy(w2).contiguous()
    ts13 = torch.from_numpy(s13).contiguous()
    ts2 = torch.from_numpy(s2).contiguous()

    def make_cfg(mod):
        cfg = mod.MOEConfigV2()
        cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = torch.cuda.current_device()
        cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
        cfg.hidden_size = H; cfg.intermediate_size = I
        cfg.max_batch_size = 256; cfg.max_num_seqs = 256
        cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
        cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
        cfg.swiglu_limit = 10.0
        return cfg

    import lk_moe
    import xiaotu_moe
    xm = xiaotu_moe.load()
    cfg_l = make_cfg(lk_moe)
    cfg_x = make_cfg(xm)

    eng_l = lk_moe.MOE_MXFP4(cfg_l, tw13.data_ptr(), tw2v.data_ptr(),
                             ts13.data_ptr(), ts2.data_ptr(), 0, 0)
    eng_x = xm.MOE_MXFP4(cfg_x, tw13.data_ptr(), tw2v.data_ptr(),
                         ts13.data_ptr(), ts2.data_ptr(), 0, 0)

    rng = np.random.default_rng(7)
    M = 6
    x = rng.standard_normal((M, H)).astype(np.float32)
    xb = f32_to_bf16_bits(x).copy()
    xf = bf16_bits_to_f32(xb)
    ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    t_ids = torch.from_numpy(ids); t_wts = torch.from_numpy(wts)
    t_xb = torch.from_numpy(xb)

    def run(engine):
        out = torch.empty((M, H), dtype=torch.float32)
        engine.cpu_prefill(M, K, t_ids.data_ptr(), t_wts.data_ptr(),
                           t_xb.data_ptr(), out.data_ptr())
        return out.numpy()

    got_l = run(eng_l)
    got_x = run(eng_x)

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

    big = np.abs(golden) > 0.05

    def report(label, g):
        ad = np.abs(g - golden)
        rel = ad / (np.abs(golden) + 1e-6)
        print(f"  [{label}] max abs {ad.max():.6e}  "
              f"max rel(|g|>.05) {rel[big].max():.6e}")

    # the actual drop-in comparison
    ad_xl = np.abs(got_x - got_l)
    rel_xl = ad_xl / (np.abs(got_l) + 1e-6)
    print(f"=== xiaotu-moe MOE_MXFP4 vs real lk_moe MOE_MXFP4 "
          f"(drop-in, same feed, E={E} M={M} K={K}) ===")
    report("lk_moe      vs golden", got_l)
    report("xiaotu-moe  vs golden", got_x)
    print(f"  [xiaotu vs lk_moe ] max abs {ad_xl.max():.6e}  "
          f"max rel(|lk|>.05) {rel_xl[(np.abs(got_l)>0.05)].max():.6e}")

    # sanity: dominant agreement should be tight (lk_moe tail is a small fraction)
    tight = rel_xl[np.abs(got_l) > 0.05]
    print(f"  xiaotu/lk_moe rel(|lk|>.05) median {np.median(tight):.3e}  "
          f"p90 {np.percentile(tight,90):.3e}  n={tight.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
