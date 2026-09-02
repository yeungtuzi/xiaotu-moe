#!/usr/bin/env python3
"""FP8 (e4m3) A/B test: feed identical per-tensor-quantized weights/scales to the
real lk_moe.MOE_FP8 and to xiaotu-moe MOE_FP8, then compare cpu_prefill output.

Layout (mirrors what the Lvllmds4-x vLLM fork feeds the real engine):
  w13 : [E, 2I, H]  uint8 (e4m3), gate rows [0,I), up rows [I,2I)
  w2  : [E, H, I]   uint8 (e4m3)
  w13_weight_scale : [E, 2]   fp32 per-tensor (gate block, up block)
  w2_weight_scale  : [E]      fp32 per-tensor
  config groupN = max(I, H), groupK = max(I, H)   (per _get_quant_params)
Run under a Python that can import the real lk_moe (lvllmds4-x env, cp312).
"""
import os, sys, ctypes
import numpy as np

def e4m3_to_fp32(v):
    v = int(v)
    if v == 0x7F or v == 0xFF:
        return 0.0
    s = (v >> 7) & 1
    e = (v >> 3) & 0xF
    m = v & 0x7
    if e == 0:
        val = (m / 8.0) * (2.0 ** -7) if m else 0.0
    else:
        val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -val if s else val

def encode_e4m3(x):
    # nearest e4m3 byte whose decode is closest to x (x already value/scale)
    best = 0; best_err = 1e30
    for b in range(256):
        d = e4m3_to_fp32(b)
        err = abs(d - x)
        if err < best_err:
            best_err = err; best = b
    return best

def quantize_weight(w, scale_vec):
    # w: (..., ) floats; scale_vec broadcastable; returns uint8 e4m3 bytes
    scaled = w / scale_vec
    flat = scaled.reshape(-1)
    out = np.array([encode_e4m3(x) for x in flat], dtype=np.uint8)
    return out.reshape(w.shape)

def silu(x):
    return x / (1.0 + np.exp(-x))

def fp32_to_bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    r = u + 0x7FFF + ((u >> 16) & 1)
    return (r >> 16).astype(np.uint16)

def bf16_to_fp32(x):
    return (x.astype(np.uint32) << 16).view(np.float32)

def numpy_ref(cfg, w13_bytes, w13_scale, w2_bytes, w2_scale, hidden, ids, weights):
    H, I = cfg.hidden_size, cfg.intermediate_size
    E, M, tk = len(w13_bytes), len(hidden), cfg.top_k
    gn, gk = cfg.groupN, cfg.groupK
    out = np.zeros((M, H), dtype=np.float32)
    for t in range(M):
        x = bf16_to_fp32(hidden[t])  # hidden is bf16 (uint16)
        for r in range(tk):
            e = ids[t, r]; w = weights[t, r]
            if w == 0.0: continue
            wg = np.zeros((I, H), np.float32)
            wu = np.zeros((I, H), np.float32)
            for n in range(I):
                for k in range(H):
                    wg[n, k] = e4m3_to_fp32(w13_bytes[e, n, k]) * w13_scale[e, 0]
                    wu[n, k] = e4m3_to_fp32(w13_bytes[e, I + n, k]) * w13_scale[e, 1]
            gate = wg @ x
            up = wu @ x
            act = up * silu(gate)
            wd = np.zeros((H, I), np.float32)
            for n in range(H):
                for k in range(I):
                    wd[n, k] = e4m3_to_fp32(w2_bytes[e, n, k]) * w2_scale[e]
            down = wd @ act
            out[t] += w * down
    return out

def run(realmod, mymod, label):
    np.random.seed(0)
    E, H, I, top_k, M = 4, 128, 256, 2, 4
    # per-tensor blocks (scaled to realistic activation magnitudes so the real
    # engine's silu stays finite; gate values then stay ~O(1))
    w13_block = (np.random.randn(E, 2, I, H) * 0.05).astype(np.float32)
    w2_block = (np.random.randn(E, H, I) * 0.05).astype(np.float32)
    # per-expert block scales
    w13_scale = np.array([[np.abs(w13_block[e, 0]).max(), np.abs(w13_block[e, 1]).max()] for e in range(E)], dtype=np.float32) + 1e-3
    w2_scale = (np.abs(w2_block).reshape(E, -1).max(axis=1) + 1e-3).astype(np.float32)
    # quantize to e4m3; engines read w13 as contiguous [E, 2I, H]
    w13_bytes = quantize_weight(w13_block, w13_scale[:, :, None, None]).reshape(E, 2 * I, H)
    w2_bytes = quantize_weight(w2_block, w2_scale[:, None, None])

    hidden = np.random.randn(M, H).astype(np.float32)
    ids = np.random.randint(0, E, size=(M, top_k)).astype(np.uint32)
    weights = np.random.randn(M, top_k).astype(np.float32)
    hidden_bf16 = fp32_to_bf16(hidden)  # engine input is bf16 (uint16)

    gn = max(I, H); gk = max(I, H)
    cfg_r = realmod.MOEConfigV2(); cfg_m = mymod.MOEConfigV2()
    for cfg in (cfg_r, cfg_m):
        cfg.expert_num = E; cfg.top_k = top_k
        cfg.hidden_size = H; cfg.intermediate_size = I
        cfg.groupN = gn; cfg.groupK = gk
        cfg.stride = 32; cfg.activation_type = 0
        cfg.gpu_id = 2  # free A100; real engine allocates GPU act buffers up front
        cfg.max_batch_size = M; cfg.max_num_seqs = M
        cfg.num_processes = 1; cfg.process_id = 0

    # int data_ptr for numpy
    def ptr(a): return ctypes.cast(a.ctypes.data, ctypes.c_void_p).value

    out_r = np.zeros((M, H), dtype=np.float32)
    out_m = np.zeros((M, H), dtype=np.float32)

    eng_r = realmod.MOE_FP8(cfg_r, ptr(w13_bytes), ptr(w2_bytes), ptr(w13_scale), ptr(w2_scale), 0, 0)
    eng_m = mymod.MOE_FP8(cfg_m, w13_bytes, w2_bytes, w13_scale, w2_scale, None, None)

    # cpu_prefill(self, qlen, k, expert_ids, weights, input, output)
    eng_r.cpu_prefill(M, top_k, ptr(ids), ptr(weights), ptr(hidden_bf16), ptr(out_r))
    eng_m.cpu_prefill(M, top_k, ids, weights, hidden_bf16, out_m)

    ref = numpy_ref(cfg_m, w13_bytes, w13_scale, w2_bytes, w2_scale, hidden_bf16, ids, weights)

    e_rm = np.abs(out_r - out_m).sum() / (np.abs(out_r).sum() + 1e-12)
    e_rm_max = np.abs(out_r - out_m).max()
    e_mr = np.abs(out_m - ref).sum() / (np.abs(ref).sum() + 1e-12)
    e_rr = np.abs(out_r - ref).sum() / (np.abs(ref).sum() + 1e-12)
    print(f"[{label}] E={E} H={H} I={I} top_k={top_k} M={M} groupN={gn} groupK={gk}")
    print(f"[{label}] real-vs-mine rel={e_rm:.3e} max={e_rm_max:.3e}")
    print(f"[{label}] mine-vs-numpyref rel={e_mr:.3e}  real-vs-numpyref rel={e_rr:.3e}")
    return e_rm, e_mr

if __name__ == "__main__":
    # real lk_moe (must be importable in this interpreter)
    import lk_moe as realmod
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
    import _xiaotu_moe_C as mymod
    run(realmod, mymod, "FP8-per-tensor")
    print("DONE")
