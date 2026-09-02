// xiaotu-moe: packed 4-bit WeightTraits (MXFP4 / WNA16) for MOE_V2.
// Self-contained CPU kernels.
//
// Weight layout (mirrors what vLLM feeds the real lk_moe MOE_MXFP4 / MOE_WNA16
// engines after _process_mxfp4/_process_wna16):
//   w13  : [expert_num][2*inter][hidden/2]  uint8, 4-bit nibble-packed along K.
//          gate = rows [0, inter), up = rows [inter, 2*inter).
//   w2   : [expert_num][hidden][inter/2]    uint8, nibble-packed along K.
//   Each byte = 2 consecutive K elements: low nibble = even k, high nibble = odd k.
//   Scales (w13_g / w2_g):
//          [expert_num][N/groupN][K/groupK] fp32 per-group block scale.
//          N = 2*inter for w13_g, hidden for w2_g; K = hidden/inter respectively.
//   Dequant: W[n,k] = lut[nibble(W,n,k)] * scale[n/groupN, k/groupK]
//     - MXFP4 : lut = FP4 E2M1 table (no zero point).
//     - WNA16 : lut = int4 centered at 8  ((nibble - 8)), ktransformers GPTQ
//               convention. Exact vLLM-Marlin matching needs a real-model A/B.
//
// GEMM uses the deferred-hsum pattern: inside a K-group only unscaled values
// accumulate, at each group boundary one broadcast-FMA folds the group scale,
// one hsum per row, then an optional global scale is applied.
//
// Copyright: Apache-2.0. Dequantization approach and FP4 E2M1 table follow
// KVCache.AI ktransformers (/kt-kernel/operators/avx2/mxfp4-moe.hpp).

#ifndef XIAOTU_MOE_MOE_V2_PACKED4_HPP
#define XIAOTU_MOE_MOE_V2_PACKED4_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../kernels/bf16_gemm.hpp"

namespace xiaotu_moe {

namespace packed4 {

// Standard FP4 E2M1 dequant table (nibble -> value). Shared by MXFP4 & NVFP4.
// e2m1: 1 sign, 2 exponent (bias 1), 1 mantissa.
static constexpr float E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f /* -0 */, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// WNA16 int4 table, centered at 8 (ktransformers GPTQ convention: (nibble-8)).
static constexpr float INT4_CENTER8[16] = {
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
};

// nibble value of packed element (W packed [N][K/2], low nibble = even k).
static inline int nibble_of(const uint8_t* Wrow, int k) {
    const uint8_t b = Wrow[k / 2];
    return (k & 1) ? (b >> 4) : (b & 0x0F);
}

// fp8_e8m0fnu ("ue8m0") decode: scale = 2^(byte - 127). This is how the fork's
// Mxfp4MoEMethod feeds DeepSeek-V4-Flash MXFP4 scales (raw uint8 bytes, verified
// against real lk_moe MOE_MXFP4: median ratio 1.0001 vs torch truth). Lazy
// 256-entry table so indexing costs one load, no per-group powf.
inline float e8m0_scale_table_gen(int i) { return std::pow(2.0f, i - 127); }
static inline const float* e8m0_table() {
    static const float table[256] = {
        e8m0_scale_table_gen(0), e8m0_scale_table_gen(1), e8m0_scale_table_gen(2),
        e8m0_scale_table_gen(3), e8m0_scale_table_gen(4), e8m0_scale_table_gen(5),
        e8m0_scale_table_gen(6), e8m0_scale_table_gen(7), e8m0_scale_table_gen(8),
        e8m0_scale_table_gen(9), e8m0_scale_table_gen(10), e8m0_scale_table_gen(11),
        e8m0_scale_table_gen(12), e8m0_scale_table_gen(13), e8m0_scale_table_gen(14),
        e8m0_scale_table_gen(15), e8m0_scale_table_gen(16), e8m0_scale_table_gen(17),
        e8m0_scale_table_gen(18), e8m0_scale_table_gen(19), e8m0_scale_table_gen(20),
        e8m0_scale_table_gen(21), e8m0_scale_table_gen(22), e8m0_scale_table_gen(23),
        e8m0_scale_table_gen(24), e8m0_scale_table_gen(25), e8m0_scale_table_gen(26),
        e8m0_scale_table_gen(27), e8m0_scale_table_gen(28), e8m0_scale_table_gen(29),
        e8m0_scale_table_gen(30), e8m0_scale_table_gen(31), e8m0_scale_table_gen(32),
        e8m0_scale_table_gen(33), e8m0_scale_table_gen(34), e8m0_scale_table_gen(35),
        e8m0_scale_table_gen(36), e8m0_scale_table_gen(37), e8m0_scale_table_gen(38),
        e8m0_scale_table_gen(39), e8m0_scale_table_gen(40), e8m0_scale_table_gen(41),
        e8m0_scale_table_gen(42), e8m0_scale_table_gen(43), e8m0_scale_table_gen(44),
        e8m0_scale_table_gen(45), e8m0_scale_table_gen(46), e8m0_scale_table_gen(47),
        e8m0_scale_table_gen(48), e8m0_scale_table_gen(49), e8m0_scale_table_gen(50),
        e8m0_scale_table_gen(51), e8m0_scale_table_gen(52), e8m0_scale_table_gen(53),
        e8m0_scale_table_gen(54), e8m0_scale_table_gen(55), e8m0_scale_table_gen(56),
        e8m0_scale_table_gen(57), e8m0_scale_table_gen(58), e8m0_scale_table_gen(59),
        e8m0_scale_table_gen(60), e8m0_scale_table_gen(61), e8m0_scale_table_gen(62),
        e8m0_scale_table_gen(63), e8m0_scale_table_gen(64), e8m0_scale_table_gen(65),
        e8m0_scale_table_gen(66), e8m0_scale_table_gen(67), e8m0_scale_table_gen(68),
        e8m0_scale_table_gen(69), e8m0_scale_table_gen(70), e8m0_scale_table_gen(71),
        e8m0_scale_table_gen(72), e8m0_scale_table_gen(73), e8m0_scale_table_gen(74),
        e8m0_scale_table_gen(75), e8m0_scale_table_gen(76), e8m0_scale_table_gen(77),
        e8m0_scale_table_gen(78), e8m0_scale_table_gen(79), e8m0_scale_table_gen(80),
        e8m0_scale_table_gen(81), e8m0_scale_table_gen(82), e8m0_scale_table_gen(83),
        e8m0_scale_table_gen(84), e8m0_scale_table_gen(85), e8m0_scale_table_gen(86),
        e8m0_scale_table_gen(87), e8m0_scale_table_gen(88), e8m0_scale_table_gen(89),
        e8m0_scale_table_gen(90), e8m0_scale_table_gen(91), e8m0_scale_table_gen(92),
        e8m0_scale_table_gen(93), e8m0_scale_table_gen(94), e8m0_scale_table_gen(95),
        e8m0_scale_table_gen(96), e8m0_scale_table_gen(97), e8m0_scale_table_gen(98),
        e8m0_scale_table_gen(99), e8m0_scale_table_gen(100), e8m0_scale_table_gen(101),
        e8m0_scale_table_gen(102), e8m0_scale_table_gen(103), e8m0_scale_table_gen(104),
        e8m0_scale_table_gen(105), e8m0_scale_table_gen(106), e8m0_scale_table_gen(107),
        e8m0_scale_table_gen(108), e8m0_scale_table_gen(109), e8m0_scale_table_gen(110),
        e8m0_scale_table_gen(111), e8m0_scale_table_gen(112), e8m0_scale_table_gen(113),
        e8m0_scale_table_gen(114), e8m0_scale_table_gen(115), e8m0_scale_table_gen(116),
        e8m0_scale_table_gen(117), e8m0_scale_table_gen(118), e8m0_scale_table_gen(119),
        e8m0_scale_table_gen(120), e8m0_scale_table_gen(121), e8m0_scale_table_gen(122),
        e8m0_scale_table_gen(123), e8m0_scale_table_gen(124), e8m0_scale_table_gen(125),
        e8m0_scale_table_gen(126), e8m0_scale_table_gen(127), e8m0_scale_table_gen(128),
        e8m0_scale_table_gen(129), e8m0_scale_table_gen(130), e8m0_scale_table_gen(131),
        e8m0_scale_table_gen(132), e8m0_scale_table_gen(133), e8m0_scale_table_gen(134),
        e8m0_scale_table_gen(135), e8m0_scale_table_gen(136), e8m0_scale_table_gen(137),
        e8m0_scale_table_gen(138), e8m0_scale_table_gen(139), e8m0_scale_table_gen(140),
        e8m0_scale_table_gen(141), e8m0_scale_table_gen(142), e8m0_scale_table_gen(143),
        e8m0_scale_table_gen(144), e8m0_scale_table_gen(145), e8m0_scale_table_gen(146),
        e8m0_scale_table_gen(147), e8m0_scale_table_gen(148), e8m0_scale_table_gen(149),
        e8m0_scale_table_gen(150), e8m0_scale_table_gen(151), e8m0_scale_table_gen(152),
        e8m0_scale_table_gen(153), e8m0_scale_table_gen(154), e8m0_scale_table_gen(155),
        e8m0_scale_table_gen(156), e8m0_scale_table_gen(157), e8m0_scale_table_gen(158),
        e8m0_scale_table_gen(159), e8m0_scale_table_gen(160), e8m0_scale_table_gen(161),
        e8m0_scale_table_gen(162), e8m0_scale_table_gen(163), e8m0_scale_table_gen(164),
        e8m0_scale_table_gen(165), e8m0_scale_table_gen(166), e8m0_scale_table_gen(167),
        e8m0_scale_table_gen(168), e8m0_scale_table_gen(169), e8m0_scale_table_gen(170),
        e8m0_scale_table_gen(171), e8m0_scale_table_gen(172), e8m0_scale_table_gen(173),
        e8m0_scale_table_gen(174), e8m0_scale_table_gen(175), e8m0_scale_table_gen(176),
        e8m0_scale_table_gen(177), e8m0_scale_table_gen(178), e8m0_scale_table_gen(179),
        e8m0_scale_table_gen(180), e8m0_scale_table_gen(181), e8m0_scale_table_gen(182),
        e8m0_scale_table_gen(183), e8m0_scale_table_gen(184), e8m0_scale_table_gen(185),
        e8m0_scale_table_gen(186), e8m0_scale_table_gen(187), e8m0_scale_table_gen(188),
        e8m0_scale_table_gen(189), e8m0_scale_table_gen(190), e8m0_scale_table_gen(191),
        e8m0_scale_table_gen(192), e8m0_scale_table_gen(193), e8m0_scale_table_gen(194),
        e8m0_scale_table_gen(195), e8m0_scale_table_gen(196), e8m0_scale_table_gen(197),
        e8m0_scale_table_gen(198), e8m0_scale_table_gen(199), e8m0_scale_table_gen(200),
        e8m0_scale_table_gen(201), e8m0_scale_table_gen(202), e8m0_scale_table_gen(203),
        e8m0_scale_table_gen(204), e8m0_scale_table_gen(205), e8m0_scale_table_gen(206),
        e8m0_scale_table_gen(207), e8m0_scale_table_gen(208), e8m0_scale_table_gen(209),
        e8m0_scale_table_gen(210), e8m0_scale_table_gen(211), e8m0_scale_table_gen(212),
        e8m0_scale_table_gen(213), e8m0_scale_table_gen(214), e8m0_scale_table_gen(215),
        e8m0_scale_table_gen(216), e8m0_scale_table_gen(217), e8m0_scale_table_gen(218),
        e8m0_scale_table_gen(219), e8m0_scale_table_gen(220), e8m0_scale_table_gen(221),
        e8m0_scale_table_gen(222), e8m0_scale_table_gen(223), e8m0_scale_table_gen(224),
        e8m0_scale_table_gen(225), e8m0_scale_table_gen(226), e8m0_scale_table_gen(227),
        e8m0_scale_table_gen(228), e8m0_scale_table_gen(229), e8m0_scale_table_gen(230),
        e8m0_scale_table_gen(231), e8m0_scale_table_gen(232), e8m0_scale_table_gen(233),
        e8m0_scale_table_gen(234), e8m0_scale_table_gen(235), e8m0_scale_table_gen(236),
        e8m0_scale_table_gen(237), e8m0_scale_table_gen(238), e8m0_scale_table_gen(239),
        e8m0_scale_table_gen(240), e8m0_scale_table_gen(241), e8m0_scale_table_gen(242),
        e8m0_scale_table_gen(243), e8m0_scale_table_gen(244), e8m0_scale_table_gen(245),
        e8m0_scale_table_gen(246), e8m0_scale_table_gen(247), e8m0_scale_table_gen(248),
        e8m0_scale_table_gen(249), e8m0_scale_table_gen(250), e8m0_scale_table_gen(251),
        e8m0_scale_table_gen(252), e8m0_scale_table_gen(253), e8m0_scale_table_gen(254),
        e8m0_scale_table_gen(255),
    };
    return table;
}

// C[M,N] fp32 = A[M,K]bf16 x W[N,K/2]fp4^T, group scale along N (groupN) and K
// (groupK), optional global scale. W row major [N][K/2]. S=[N/gn][K/gk].
// lut: 16-entry float table mapping nibble -> weight value.
// E8M0: when true, S is a raw uint8 fp8_e8m0 byte buffer (fork MXFP4 feeding),
// decoded as 2^(byte-127); when false, S is fp32 values (WNA16/NVFP4 feeding).
template <bool E8M0 = false>
inline void matmul_packed4_group(const uint16_t* A, const uint8_t* W,
                                 const float* lut, const void* S,
                                 float global_scale, float* C,
                                 int M, int N, int K,
                                 int groupN, int groupK) {
    if (K <= 0 || (K & 1)) return;  // packed layout requires even K
    const int gn = groupN > 0 ? groupN : 1;
    const int gk = groupK > 0 ? groupK : 1;
    const float* Srow = static_cast<const float*>(S);
    auto scale_at = [&](int n, int kbase) -> float {
        const int idx = (n / gn) * ((K + gk - 1) / gk) + (kbase / gk);
        if constexpr (E8M0) return e8m0_table()[static_cast<const uint8_t*>(S)[idx]];
        return Srow[idx];
    };
#if defined(__AVX2__)
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            const uint8_t* Wrow = W + (size_t)j * (K / 2);
            __m256 total = _mm256_setzero_ps();
            int kbase = 0;
            for (; kbase < K; kbase += gk) {
                __m256 gacc = _mm256_setzero_ps();
                int k = kbase;
                int kend = (kbase + gk < K) ? (kbase + gk) : K;
                for (; k + 8 <= kend; k += 8) {
                    __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Arow + k));
                    __m256 av = _mm256_castsi256_ps(
                        _mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
                    // 8 elements = 4 packed bytes starting at byte k/2.
                    uint32_t raw;
                    std::memcpy(&raw, Wrow + (k / 2), 4);
                    __m128i b = _mm_cvtsi32_si128((int)raw);
                    __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                    __m128i nib = _mm_unpacklo_epi8(lo, hi);  // 8 nibble indices
                    __m256i idx = _mm256_cvtepu8_epi32(nib);
                    __m256 wv = _mm256_i32gather_ps(lut, idx, 4);
                    gacc = _mm256_fmadd_ps(av, wv, gacc);
                }
                float gscalar = 0.f;
                for (; k < kend; ++k)
                    gscalar += bf16::bf16_to_fp32(Arow[k]) * lut[nibble_of(Wrow, k)];
                float scale = scale_at(j, kbase);
                total = _mm256_fmadd_ps(gacc, _mm256_set1_ps(scale), total);
                if (gscalar != 0.f) {
                    float t[8]; _mm256_storeu_ps(t, total);
                    t[0] += gscalar * scale;
                    total = _mm256_loadu_ps(t);
                }
            }
            float tmp[8]; _mm256_storeu_ps(tmp, total);
            float acc = 0.f;
            for (int t = 0; t < 8; ++t) acc += tmp[t];
            Crow[j] = acc * global_scale;
        }
    }
#else
    (void)lut;
    for (int i = 0; i < M; ++i) {
        const uint16_t* Arow = A + (size_t)i * K;
        float* Crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            const uint8_t* Wrow = W + (size_t)j * (K / 2);
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += bf16::bf16_to_fp32(Arow[k]) *
                       lut[nibble_of(Wrow, k)] * scale_at(j, k - (k % gk));
            Crow[j] = acc * global_scale;
        }
    }
#endif
}

}  // namespace packed4

// Generic CRTP base for a packed-4bit quant type given a dequant LUT.
// The LM `w13_g`/`w2_g` are block-scale arrays and `w13_gs`/`w2_gs` are optional
// per-expert global scales (NVFP4). dequant = lut[nibble] * block_scale * global.
// `Tag` type parameter gives each *configuration* a distinct C++ type so that
// pybind11 can register MXFP4 and NVFP4 (which share the E2M1 LUT) as separate
// classes (pybind11 keys registered classes by C++ typeid).
template <const float (&LUT)[16], typename Tag = void, bool E8M0 = false>
struct Packed4WeightTraitsBase
    : WeightTraitsBase<Packed4WeightTraitsBase<LUT, Tag, E8M0>> {
    static constexpr bool kE8M0 = E8M0;   // scale stored as 1-byte e8m0 vs fp32
    static void gate_up_impl(const uint16_t* x, const void* w13, const void* w13_g,
                            const float* w13_gs, float* gate, float* up,
                            int inter, int hidden, size_t eid,
                            int groupN, int groupK) {
        // w13 : [E][2I][H/2] packed; scales [E][2I/gn][H/gk] (fp32 or e8m0 bytes).
        const int n2 = 2 * inter;
        const uint8_t* base = static_cast<const uint8_t*>(w13) + eid * (size_t)n2 * (hidden / 2);
        std::vector<float> both(n2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(n2 + gn - 1) / gn;
        const size_t kb = (size_t)(hidden + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w13_g) {
            // scale element is 1 byte (e8m0) or 4 bytes (fp32) -> byte offset
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w13_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};  // 2^(127-127)=1
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w13_gs ? w13_gs[eid] : 1.0f;
        packed4::matmul_packed4_group<E8M0>(x, base, LUT, sbase, gs, both.data(),
                                            1, n2, hidden, groupN, groupK);
        std::copy(both.begin(), both.begin() + inter, gate);
        std::copy(both.begin() + inter, both.end(), up);
    }

    static void down_impl(const uint16_t* act, const void* w2, const void* w2_g,
                          const float* w2_gs, float* down, int hidden, int inter,
                          size_t eid, int groupN, int groupK) {
        // w2 : [E][H][I/2] packed; scales [E][H/gn][I/gk] (fp32 or e8m0 bytes).
        const uint8_t* base = static_cast<const uint8_t*>(w2) + eid * (size_t)hidden * (inter / 2);
        const int gn = groupN > 0 ? groupN : 1;
        const int gk = groupK > 0 ? groupK : 1;
        const size_t nb = (size_t)(hidden + gn - 1) / gn;
        const size_t kb = (size_t)(inter + gk - 1) / gk;
        const void* sbase = nullptr;
        if (w2_g) {
            const size_t byte_off = eid * (nb * kb) * (E8M0 ? 1u : sizeof(float));
            sbase = static_cast<const char*>(w2_g) + byte_off;
        } else if constexpr (E8M0) {
            static const uint8_t ones[1] = {127};
            sbase = ones;
        } else {
            static const float ones[1] = {1.0f};
            sbase = ones;
        }
        const float gs = w2_gs ? w2_gs[eid] : 1.0f;
        packed4::matmul_packed4_group<E8M0>(act, base, LUT, sbase, gs, down,
                                            1, hidden, inter, groupN, groupK);
    }
};

// MXFP4: FP4 E2M1 weights, raw fp8_e8m0 uint8 per-block scales (fork feeding),
// no global scale. groupK=32 when fed by the fork.
struct MXFP4Tag {};
using MXFP4WeightTraits = Packed4WeightTraitsBase<packed4::E2M1, MXFP4Tag, true>;

// WNA16: int4 weights (LUT centered at 8), fp32 per-group scales, no global scale.
struct WNA16Tag {};
using WNA16WeightTraits = Packed4WeightTraitsBase<packed4::INT4_CENTER8, WNA16Tag>;

// NVFP4: FP4 E2M1 weights (same nibble layout as MXFP4), fp32 per-group scales
// PLUS a per-expert global scale. groupK is 32 when fed by vLLM.
struct NVFP4Tag {};
using NVFP4WeightTraits = Packed4WeightTraitsBase<packed4::E2M1, NVFP4Tag>;

}  // namespace xiaotu_moe

#endif  // XIAOTU_MOE_MOE_V2_PACKED4_HPP
