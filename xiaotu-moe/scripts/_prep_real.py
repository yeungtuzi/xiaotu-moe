#!/usr/bin/env python
"""Prep (conda env, has torch+safetensors): read real DeepSeek-V4-Flash layer-1
routed experts, assemble xiaotu-moe MOE_NVFP4 contract tensors + golden
dequant weights, store as a single .npz for the numpy-only A/B runner."""
import glob
import os

import numpy as np
from safetensors import safe_open

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "real_layer1_model.npz")

H = 4096
I = 2048
GK = 32
LAYER = 1
E = 16

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)


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
    shard = None
    for sh in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with safe_open(sh, "pt") as f:
            if f"layers.{LAYER}.ffn.experts.0.w1.weight" in f.keys():
                shard = sh
                break
    assert shard

    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    g13 = np.empty((E, 2 * I, H // GK), dtype=np.float32)
    g2 = np.empty((E, H, I // GK), dtype=np.float32)
    gol13 = np.empty((E, 2 * I, H), dtype=np.float32)
    gol2 = np.empty((E, H, I), dtype=np.float32)

    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{LAYER}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w1 = get(e, "w1.weight").numpy()
            w2w = get(e, "w2.weight").numpy()
            w3 = get(e, "w3.weight").numpy()
            s1 = get(e, "w1.scale").view(torch.uint8).numpy()
            s2 = get(e, "w2.scale").view(torch.uint8).numpy()
            s3 = get(e, "w3.scale").view(torch.uint8).numpy()
            w13[e, 0:I] = w1.astype(np.uint8)
            w13[e, I:2 * I] = w3.astype(np.uint8)
            w2[e] = w2w.astype(np.uint8)
            g13[e, 0:I] = np.power(2.0, s1.astype(np.float32) - 127.0)
            g13[e, I:2 * I] = np.power(2.0, s3.astype(np.float32) - 127.0)
            g2[e] = np.power(2.0, s2.astype(np.float32) - 127.0)
            gol13[e, 0:I] = unpack_dequant(w1, s1)
            gol13[e, I:2 * I] = unpack_dequant(w3, s3)
            gol2[e] = unpack_dequant(w2w, s2)

    np.savez_compressed(OUT, w13=w13, w2=w2, g13=g13, g2=g2,
                        gol13=gol13, gol2=gol2, E=E, I=I, H=H, GK=GK)
    print("saved", OUT)
    print("w13", w13.shape, w13.dtype, "g13", g13.shape)
    print("gol13", gol13.shape, "gol2", gol2.shape)


if __name__ == "__main__":
    import torch  # noqa: F401  (for .view(torch.uint8))
    main()
