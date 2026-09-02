#!/usr/bin/env python3
"""xiaotu-moe: MXFP4 & WNA16 numpy-reference validation.

Quantizes synthetic float weights into the vLLM packed-4bit layout
(w13 [E][2I][H/2], w2 [E][H][I/2] uint8, group scales) and checks the CPU
engine's MOE_MXFP4 / MOE_WNA16 output against a pure numpy reference that
implements the exact same declared semantics.

Run with the .search-venv interpreter (the correct one for xiaotu-moe):
    /home/user/lvllm/.search-venv/bin/python scripts/smoke_test_packed4.py
"""
import os, sys
import numpy as np

# Native module is injected by callers (see scripts/verify_variants.py) to test
# a specific ISA variant. When run standalone, fall back to the auto-chosen one.
m = None

# ---- FP4 E2M1 dequant table (must mirror csrc packed4::E2M1) ----
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
INT4_CENTER8 = (np.arange(16, dtype=np.float32) - 8.0)

def pack4(mat):  # mat [R][C] float -> uint8 [R][C/2], low=even k
    R, C = mat.shape
    assert C % 2 == 0
    q = np.zeros((R, C // 2), dtype=np.uint8)
    lo = mat[:, 0::2]
    hi = mat[:, 1::2]
    lo_u = np.clip(lo, 0, 15).astype(np.uint8)
    hi_u = np.clip(hi, 0, 15).astype(np.uint8)
    q[:] = (hi_u << 4) | (lo_u & 0x0F)
    return q

def group_scales(mat, gk):
    # per-row group scale: max abs within each group of gk along K, /6 (max e2m1)
    R, C = mat.shape
    ng = (C + gk - 1) // gk
    s = np.ones((R, ng), dtype=np.float32)
    for r in range(R):
        for g in range(ng):
            seg = mat[r, g*gk:(g+1)*gk]
            if seg.size and np.abs(seg).max() > 0:
                s[r, g] = np.abs(seg).max() / 6.0
    return s

def quant_to_nib(mat, scales, gk, table):
    # quantize float matrix to nearest table value after dividing by group scale
    R, C = mat.shape
    nib = np.zeros((R, C), dtype=np.uint8)
    for r in range(R):
        for g in range(ng := (C + gk - 1) // gk):
            seg = mat[r, g*gk:(g+1)*gk]
            s = scales[r, g] if scales is not None else 1.0
            n = (seg / s)[:, None]
            n = np.abs(n - table[None, :])  # nearest (handles -0==0)
            n = n.argmin(1)
            nib[r, g*gk:(g+1)*gk] = n
    return nib

def ref_gate_up(x, w13_float, scales, gk, table):
    # x [H] bf16-as-uint16; w13_float [2I][H]; returns (gate[I], up[I])
    xf = x.view(np.float32).copy()  # placeholder; bf16 handled by caller
    dtype = np.uint16
    out = np.zeros(2 * w13_float.shape[0] // 1, dtype=np.float32)
    return out

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16_uint16(f32):
    # float32 array [.., C] -> uint16 bf16 (truncate to high 16 bits)
    f = np.ascontiguousarray(f32, dtype=np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)

def emul_mat(m, gk, table, scales, bf16_act):
    """m [R][C] nibble matrix (pre-quantized); scales [R][ng]; bf16_act [Q][C] uint16
    returns [Q][R] fp32 = bf16_act @ (table[nib]*scale)^T"""
    R, C = m.shape
    Q = bf16_act.shape[0]
    W = np.empty((R, C), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            W[r, c] = table[m[r, c]] * (scales[r, c // gk] if scales is not None else 1.0)
    return bf16_to_f32(bf16_act) @ W.T

def f32_to_e8m0(s):
    """Nearest fp8_e8m0 byte for a (positive) fp32 group scale: round(log2)+127."""
    lg = np.round(np.log2(np.maximum(s, 1e-30))).astype(np.int32) + 127
    return np.clip(lg, 0, 255).astype(np.uint8)


def e8m0_to_f32(b):
    return np.power(2.0, b.astype(np.float32) - 127.0)


def run_case(name, table, gk, H, I, E, topk, e8m0=False):
    """e8m0=True (MXFP4): group scales are raw fp8_e8m0 uint8 bytes (fork feeding);
    the numpy reference must use the same 2^(byte-127) reconstruction."""
    rng = np.random.default_rng(1234)
    # random float weights with realistic magnitude
    w13_f = rng.normal(0, 0.02, (E, 2 * I, H)).astype(np.float32)
    w2_f = rng.normal(0, 0.02, (E, H, I)).astype(np.float32)

    s13 = group_scales(w13_f.reshape(E * 2 * I, H), gk).reshape(E, 2 * I, -1)
    s2 = group_scales(w2_f.reshape(E * H, I), gk).reshape(E, H, -1)
    if e8m0:
        s13_feed = f32_to_e8m0(s13)   # uint8 e8m0 bytes fed to engine
        s2_feed = f32_to_e8m0(s2)
        s13_ref = e8m0_to_f32(s13_feed)  # engine's effective scale (reference truth)
        s2_ref = e8m0_to_f32(s2_feed)
    else:
        s13_feed = s13; s2_feed = s2
        s13_ref = s13; s2_ref = s2
    w13_nib = quant_to_nib(w13_f.reshape(E * 2 * I, H), s13.reshape(E * 2 * I, -1), gk, table).reshape(E, 2 * I, H)
    w2_nib = quant_to_nib(w2_f.reshape(E * H, I), s2.reshape(E * H, -1), gk, table).reshape(E, H, I)
    w13 = pack4(w13_nib.reshape(E * 2 * I, H)).reshape(E, 2 * I, H // 2)
    w2 = pack4(w2_nib.reshape(E * H, I)).reshape(E, H, I // 2)

    cfg = m.MOEConfigV2()
    cfg.expert_num = E; cfg.top_k = topk
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.groupN = 1
    cfg.groupK = gk
    engine = (m.MOE_MXFP4 if table is E2M1 else m.MOE_WNA16)(cfg, w13, w2, s13_feed, s2_feed)
    # tokens
    M = 4
    x_bf16 = (rng.normal(0, 1, (M, H))).astype(np.float32)
    x_u16 = f32_to_bf16_uint16(x_bf16)  # [M, H] uint16 bf16
    eids = rng.integers(0, E, (M, topk)).astype(np.uint32)
    wgt = rng.uniform(0, 1, (M, topk)).astype(np.float32)
    out = np.zeros((M, H), dtype=np.float32)
    engine.cpu_prefill(M, topk, eids, wgt, x_u16, out)

    # numpy reference
    ref = np.zeros((M, H), dtype=np.float32)
    for q in range(M):
        for r in range(topk):
            e = eids[q, r]; w = wgt[q, r]
            both = emul_mat(w13_nib[e], gk, table, s13_ref[e], x_u16[q:q+1])[0]  # [2I]
            gate, up = both[:I], both[I:]
            act = up * (gate / (1 + np.exp(-gate)))
            act_bf16 = f32_to_bf16_uint16(act[np.newaxis, :])  # [1, I] uint16
            d = emul_mat(w2_nib[e], gk, table, s2_ref[e], act_bf16)[0]  # [H]
            ref[q] += w * d

    # compare (mine quantizes hidden act to bf16; ref already bf16-ish) -> loose tol
    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
    ok = "PASS" if rel < 1e-2 else "FAIL"
    print(f"[{name}] E={E} H={H} I={I} gk={gk} topk={topk} rel={rel:.3e} {ok}")
    return ok == "PASS"

def run_nvfp4(H, I, E, topk, gk=32):
    """NVFP4: E2M1 nibbles + per-group block scales + per-expert global scale."""
    rng = np.random.default_rng(99)
    w13_f = rng.normal(0, 0.02, (E, 2 * I, H)).astype(np.float32)
    w2_f = rng.normal(0, 0.02, (E, H, I)).astype(np.float32)
    s13 = group_scales(w13_f.reshape(E * 2 * I, H), gk).reshape(E, 2 * I, -1)
    s2 = group_scales(w2_f.reshape(E * H, I), gk).reshape(E, H, -1)
    w13_nib = quant_to_nib(w13_f.reshape(E * 2 * I, H), s13.reshape(E * 2 * I, -1), gk, E2M1).reshape(E, 2 * I, H)
    w2_nib = quant_to_nib(w2_f.reshape(E * H, I), s2.reshape(E * H, -1), gk, E2M1).reshape(E, H, I)
    w13 = pack4(w13_nib.reshape(E * 2 * I, H)).reshape(E, 2 * I, H // 2)
    w2 = pack4(w2_nib.reshape(E * H, I)).reshape(E, H, I // 2)
    g13 = rng.uniform(0.5, 2.0, E).astype(np.float32)  # per-expert global scale
    g2 = rng.uniform(0.5, 2.0, E).astype(np.float32)

    cfg = m.MOEConfigV2()
    cfg.expert_num = E; cfg.top_k = topk
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.groupN = 1; cfg.groupK = gk
    engine = m.MOE_NVFP4(cfg, w13, w2, s13, s2, g13, g2)

    M = 4
    x_u16 = f32_to_bf16_uint16(rng.normal(0, 1, (M, H)).astype(np.float32))
    eids = rng.integers(0, E, (M, topk)).astype(np.uint32)
    wgt = rng.uniform(0, 1, (M, topk)).astype(np.float32)
    out = np.zeros((M, H), dtype=np.float32)
    engine.cpu_prefill(M, topk, eids, wgt, x_u16, out)

    ref = np.zeros((M, H), dtype=np.float32)
    for q in range(M):
        for r in range(topk):
            e = eids[q, r]; w = wgt[q, r]
            both = emul_mat(w13_nib[e], gk, E2M1, s13[e] * g13[e], x_u16[q:q+1])[0]
            gate, up = both[:I], both[I:]
            act = up * (gate / (1 + np.exp(-gate)))
            act_bf16 = f32_to_bf16_uint16(act[np.newaxis, :])
            d = emul_mat(w2_nib[e], gk, E2M1, s2[e] * g2[e], act_bf16)[0]
            ref[q] += w * d
    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
    ok = "PASS" if rel < 1e-2 else "FAIL"
    print(f"[NVFP4] E={E} H={H} I={I} gk={gk} topk={topk} rel={rel:.3e} {ok}")
    return ok == "PASS"

if __name__ == "__main__":
    if m is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import xiaotu_moe
        m = xiaotu_moe
    results = []
    # MXFP4: E2M1 table, raw e8m0 uint8 scales (fork DeepSeek-V4 feeding), gk=32
    results.append(run_case("MXFP4", E2M1, 32, 128, 256, 4, 2, e8m0=True))
    results.append(run_case("MXFP4", E2M1, 32, 512, 256, 2, 1, e8m0=True))
    # WNA16: int4 centered table, groupK=32
    results.append(run_case("WNA16", INT4_CENTER8, 32, 128, 256, 4, 2))
    results.append(run_case("WNA16", INT4_CENTER8, 32, 512, 256, 2, 1))
    # NVFP4: E2M1 + per-group block scale + per-expert global scale
    results.append(run_nvfp4(128, 256, 4, 2, gk=32))
    results.append(run_nvfp4(256, 512, 2, 1, gk=32))
    results.append(run_nvfp4(512, 256, 6, 3, gk=32))
    print("ALL PASS" if all(results) else "SOME FAIL")
    sys.exit(0 if all(results) else 1)
