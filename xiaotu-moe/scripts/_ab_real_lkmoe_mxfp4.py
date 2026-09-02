#!/usr/bin/env python
"""Definitive scale-contract test: feed REAL lk_moe.MOE_MXFP4 with the EXACT
tensors the Lvllmds4-x fork feeds it for DeepSeek-V4-Flash (fp4 experts,
moe_quant_algo=None -> Mxfp4MoEMethod path):
  w13 uint8 [E][2I][H/2] packed E2M1
  w2  uint8 [E][H][I/2]
  w13_scale uint8 [E][2I][H/32] (RAW fp8_e8m0 bytes, NOT converted)
  w2_scale  uint8 [E][H][I/32]
Then compare lk_moe MOE_MXFP4.cpu_prefill output to the numpy golden
(dequant = E2M1[nibble] * 2^(e8m0-127)). This tells us whether lk_moe reads the
scale as raw e8m0 bytes (matching fork feeding) so xiaotu-moe must do the same.

Run with CONDDA env (has real lk_moe). CUDA_VISIBLE_DEVICES=2 (only GPU2 free).
"""
import glob
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H = 4096
I = 2048
GK = 32
LAYER = 1
E = 16
K = 6

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def unpack_dequant(w_packed, scale_exp):
    N, Kh = w_packed.shape
    Kd = Kh * 2
    ub = w_packed.astype(np.uint8)
    vals = np.empty((N, Kd), dtype=np.float32)
    vals[:, 0::2] = E2M1[ub & 0x0F]
    vals[:, 1::2] = E2M1[(ub >> 4) & 0x0F]
    scale = np.power(2.0, scale_exp.astype(np.float32) - 127.0)
    return vals * np.repeat(scale, GK, axis=1)[:, :Kd]


def main():
    # locate shard
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
        w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
        w2 = np.empty((E, H, I // 2), dtype=np.uint8)
        s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)  # raw e8m0
        s2 = np.empty((E, H, I // GK), dtype=np.uint8)
        gol13 = np.empty((E, 2 * I, H), dtype=np.float32)
        gol2 = np.empty((E, H, I), dtype=np.float32)
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
            gol13[e, 0:I] = unpack_dequant(w1, s1)
            gol13[e, I:2 * I] = unpack_dequant(w3, s3)
            gol2[e] = unpack_dequant(w2w, s2w)

    # torch tensors on GPU... feeding real lk_moe
    t_w13 = torch.from_numpy(w13).contiguous()
    t_w2 = torch.from_numpy(w2).contiguous()
    t_s13 = torch.from_numpy(s13).contiguous()
    t_s2 = torch.from_numpy(s2).contiguous()

    import lk_moe
    cfg = lk_moe.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = torch.cuda.current_device()
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 256; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = GK
    cfg.activation_type = 0
    cfg.swiglu_limit = 10.0

    def build(scale13, scale2):
        return lk_moe.MOE_MXFP4(
            cfg, t_w13.data_ptr(), t_w2.data_ptr(),
            scale13.data_ptr(), scale2.data_ptr(), 0, 0,
        )

    rng = np.random.default_rng(7)
    M = 6
    x = rng.standard_normal((M, H)).astype(np.float32)
    xb = f32_to_bf16_bits(x)
    xf = bf16_bits_to_f32(xb)
    ids = rng.integers(0, E, size=(M, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)

    ids_t = torch.from_numpy(ids)
    wts_t = torch.from_numpy(wts)
    xb_t = torch.from_numpy(xb.copy())  # uint16 bf16 storage

    def run(engine):
        out_t = torch.empty((M, H), dtype=torch.float32)
        engine.cpu_prefill(M, K, ids_t.data_ptr(), wts_t.data_ptr(),
                           xb_t.data_ptr(), out_t.data_ptr())
        return out_t.numpy()

    # variant 1: raw uint8 e8m0 scale (what fork _process_mxfp4 feeds literally)
    got_raw = run(build(t_s13, t_s2))
    # variant 2: fp32 scale = 2^(e8m0-127) (my xiaotu-moe / my A/B behavior)
    f32_13 = np.power(2.0, s13.astype(np.float32) - 127.0)
    f32_2 = np.power(2.0, s2.astype(np.float32) - 127.0)
    got_f32 = run(build(torch.from_numpy(np.ascontiguousarray(f32_13, np.float32)),
                        torch.from_numpy(np.ascontiguousarray(f32_2, np.float32))))

    # golden
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
    ratio = got_raw[big] / golden[big]
    print(f"raw/golden ratio stats on big cells: n={ratio.size}")
    print(f"  min {ratio.min():.4f}  p10 {np.percentile(ratio,10):.4f} "
          f"p50 {np.percentile(ratio,50):.4f} p90 {np.percentile(ratio,90):.4f} "
          f"max {ratio.max():.4f}")
    # is ratio near a power of 2 for most cells?
    log2r = np.log2(np.abs(ratio))
    print(f"  log2|ratio|: median {np.median(log2r):.3f}  "
          f"std {log2r.std():.3f}")

    def report(label, g):
        ad = np.abs(g - golden)
        rel = ad / (np.abs(golden) + 1e-6)
        print(f"  [{label}] max abs {ad.max():.6e}  max rel(|g|>.05) {rel[big].max():.6e}")

    print(f"REAL lk_moe.MOE_MXFP4 feed comparison (E={E} M={M} K={K})")
    report("raw uint8 e8m0 scale ", got_raw)
    report("fp32 2^(e8m0-127) scale", got_f32)
    return 0


if __name__ == "__main__":
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=2 -> visible device 0 = real GPU2
    sys.exit(main())
