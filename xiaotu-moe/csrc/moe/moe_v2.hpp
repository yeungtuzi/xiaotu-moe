// xiaotu-moe: MOE_V2<WeightTraits, ActivationType> — BF16 + quantized closed loop
//
// Phase 1 minimal closed loop: decode a batch of tokens routed through top_k
// experts entirely on the CPU. The aggregated forward loop mirrors the semantics
// of lktransformers' llamafile/moe.cpp (gate -> up -> gated activation -> down ->
// weighted accumulate) but is self-contained: it depends only on bf16_gemm.hpp and
// a small internal std::thread pool, with no ggml/llama.cpp or libnuma dependency.
// The NUMA work-stealing scheduler (backend_numa.cpp) can be dropped back in later
// without changing this interface.
//
// Quantization is factored out behind a CRTP "WeightTraits" interface so the
// orchestration loop is shared verbatim across BF16 / FP8 / WNA16 / MXFP4 / NVFP4:
//   - gate_up_impl computes (gate, up) = x @ [Wg; Wu]^T from a per-expert weight
//     block (and optional scale), writing inter floats each.
//   - down_impl computes down = act @ Wd^T from the (bf16) activation.
// M=1 per token by default; a traits can override to batch tokens if it wants to.
//
// License: Apache-2.0. Structure inspired by KVCache.AI / Qiong GU open source.

#ifndef XIAOTU_MOE_MOE_V2_HPP
#define XIAOTU_MOE_MOE_V2_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../kernels/bf16_gemm.hpp"

namespace xiaotu_moe {

// Minimal config matching lk_moe.MOEConfigV2 field semantics (field order and
// names follow the ABI spec; see LK_MOE_ABI_SPEC.md).
struct MOEConfigV2 {
    int   num_processes = 1;
    int   process_id = 0;
    int   gpu_id = 0;
    bool  has_gate_proj = true;
    int   expert_num = 0;
    int   top_k = 0;
    int   hidden_size = 0;
    int   intermediate_size = 0;
    int   max_batch_size = 0;
    int   max_num_seqs = 0;
    int   stride = 32;
    int   group_min_len = 10;
    int   group_max_len = 4096;
    int   groupN = 0;
    int   groupK = 0;
    float swiglu_alpha = 0.f;   // unused for plain SiLU
    float swiglu_limit = 0.f;
    int   activation_type = 0;  // 0=SiLU, 1=SwiGLU(multiply by alpha, clamp by limit)
    bool  use_gpu_prefill = false;
};

namespace act {

// gated activation: out[i] = up[i] * silu(gate[i])
// Written as up * g / (1+exp(-g)) rather than up * (g*sig) so that a very large
// |g| (which drives sig -> 0) yields a finite 0 instead of inf*0 = NaN under
// -ffast-math -fno-finite-math-only.
inline void silu_gate(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        out[i] = up[i] * (g / (1.f + std::exp(-g)));
    }
}

// in-place SiLU: v[i] = v[i] / (1 + exp(-v[i]))
inline void silu_single(float* v, int n) {
    for (int i = 0; i < n; ++i)
        v[i] = v[i] / (1.f + std::exp(-v[i]));
}

} // namespace act

// ---- CRTP base: dispatches to Derived::*_impl ----
template <typename Derived>
struct WeightTraitsBase {
    static constexpr bool kE8M0 = false;  // default: raw fp32 scales; overridden by packed4
    // gate_up: gate, up [inter] fp32 = x [hidden] bf16 x per-expert W13 [2*inter][hidden]^T
    // w13 points at expert block eid; w13_g optional per-expert block scale
    // [N/groupN][K/groupK]; w13_gs optional per-expert global scale (array) used
    // by NVFP4; groupN/groupK are the scale's grouping (<=0 -> treat as 1).
    static void gate_up(const uint16_t* x, const void* w13, const void* w13_g,
                        const float* w13_gs, float* gate, float* up,
                        int inter, int hidden, size_t eid,
                        int groupN, int groupK) {
        Derived::gate_up_impl(x, w13, w13_g, w13_gs, gate, up, inter, hidden, eid, groupN, groupK);
    }
    // down: down [hidden] fp32 = act [inter] bf16 x per-expert W2 [hidden][inter]^T
    static void down(const uint16_t* act, const void* w2, const void* w2_g,
                     const float* w2_gs, float* down, int hidden, int inter,
                     size_t eid, int groupN, int groupK) {
        Derived::down_impl(act, w2, w2_g, w2_gs, down, hidden, inter, eid, groupN, groupK);
    }
};

// ---- WeightTraits: BF16 (no quantization) ----
struct BF16WeightTraits : WeightTraitsBase<BF16WeightTraits> {
    using weight_t = uint16_t;      // bf16 storage

    static void gate_up_impl(const uint16_t* x, const void* w13, const void* w13_g,
                             const float* w13_gs, float* gate, float* up,
                             int inter, int hidden, size_t eid,
                             int groupN, int groupK) {
        (void)w13_g; (void)w13_gs; (void)groupN; (void)groupK;
        // w13 : [E][2*I][H] bf16, gate block then up block.
        const uint16_t* base = static_cast<const uint16_t*>(w13) +
            eid * (size_t)2 * inter * hidden;
        bf16::matmul_bf16(x, base, gate, 1, inter, hidden);
        bf16::matmul_bf16(x, base + (size_t)inter * hidden, up, 1, inter, hidden);
    }

    static void down_impl(const uint16_t* act, const void* w2, const void* w2_g,
                          const float* w2_gs, float* down, int hidden, int inter,
                          size_t eid, int groupN, int groupK) {
        (void)w2_g; (void)w2_gs; (void)groupN; (void)groupK;
        // w2 : [E][H][I] bf16.
        const uint16_t* base = static_cast<const uint16_t*>(w2) +
            eid * (size_t)hidden * inter;
        bf16::matmul_bf16(act, base, down, 1, hidden, inter);
    }
};

// ---- ActivationType ----
struct BF16Activation {
    using act_t = uint16_t;         // bf16
    static constexpr bool is_fp16 = false;
};

struct FP16Activation {
    using act_t = uint16_t;         // fp16 storage (same width as bf16)
    static constexpr bool is_fp16 = true;
};

// Small parallel-for over a hardware-thread pool. Each worker processes a
// strided slice of [0, n). Used to split independent token work across cores.
class ThreadPool {
public:
    explicit ThreadPool(size_t n = 0)
        : nthreads_(n == 0 ? default_threads() : n) {}

    // run func(i) for i in [0, n). Thread-safe; spawns/joins per call.
    template <typename F>
    void parallel_for(size_t n, const F& func) const {
        if (nthreads_ <= 1 || n <= 1) {
            for (size_t i = 0; i < n; ++i) func(i);
            return;
        }
        size_t nw = std::min(nthreads_, n);
        std::vector<std::thread> pool;
        pool.reserve(nw);
        for (size_t t = 0; t < nw; ++t) {
            pool.emplace_back([&, t]() {
                for (size_t i = t; i < n; i += nw) func(i);
            });
        }
        for (auto& th : pool) th.join();
    }

private:
    static size_t default_threads() {
        unsigned hw = std::thread::hardware_concurrency();
        size_t nt = hw > 0 ? (size_t)hw : 1;
        // Allow capping worker count via env (helps isolate thread-count-related
        // faults in the full server; over-subscribing by spawning one thread per
        // core per forward call is wasteful anyway).
        if (const char* e = std::getenv("XIAOTU_MOE_THREADS")) {
            long v = std::atol(e);
            if (v > 0) nt = (size_t)v;
        }
        return nt;
    }
    size_t nthreads_;
};

// ---- MOE_V2 ----
template <typename WeightTraits, typename ActivationType>
class MOE_V2 {
public:
    using wt = WeightTraits;
    using act = ActivationType;

    // w13/g13: routed expert weights. Layout assumption (verified against the
    // real feed at integration time):
    //   w13 = [expert_num][2][intermediate_size][hidden_size] (gate block 0,
    //         up block 1), base pointer (bf16 or quantized bytes).
    //   w13_g = optional per-expert scale for the gate/up block (unused for BF16;
    //           kept for ABI parity).
    // w2/g2: routed expert weights:
    //   w2 = [expert_num][hidden_size][intermediate_size] (down_proj).
    //   w2_g = optional per-expert scale (unused for BF16).
    // w13_gs/w2_gs: optional per-expert global scale arrays (NVFP4), may be null.
    explicit MOE_V2(const MOEConfigV2& cfg,
                    const void* w13, const void* w2,
                    const void* w13_g, const void* w2_g,
                    const float* w13_gs = nullptr, const float* w2_gs = nullptr)
        : cfg_(cfg) {
        // Validate minimal dims so we never divide by zero downstream.
        if (cfg_.expert_num <= 0 || cfg_.top_k <= 0 ||
            cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0)
            throw std::runtime_error("MOE_V2: invalid config dims");

        // COPY the weight blocks into engine-owned buffers. The fork hands us
        // pointers into the vLLM parameter tensors (self.w13_weight etc.), but
        // after loading vLLM (`clean_weights_after_loading`) drops those params
        // and the backing pages can be reclaimed / parked PROT_NONE, leaving the
        // stored pointers dangling (=> SIGSEGV on the first real forward). The
        // authoritative lk_moe engine likewise snapshots its weights at
        // construction. Here we mirror that: read the bytes once while the source
        // is live and keep our own copy for the lifetime of the engine.
        const int H = cfg_.hidden_size;
        const int I = cfg_.intermediate_size;
        const int E = cfg_.expert_num;
        const int gn = cfg_.groupN > 0 ? cfg_.groupN : 1;
        const int gk = cfg_.groupK > 0 ? cfg_.groupK : 1;
        const bool e8m0 = WeightTraits::kE8M0;
        const size_t n2 = (size_t)2 * I;                       // gate+up rows
        const size_t w13_bytes = (size_t)E * n2 * (size_t)(H / 2);
        const size_t w2_bytes = (size_t)E * (size_t)H * (size_t)(I / 2);
        const size_t w13g_bytes = (size_t)E * ((n2 + gn - 1) / gn) * ((H + gk - 1) / gk) * (e8m0 ? 1u : sizeof(float));
        const size_t w2g_bytes = (size_t)E * ((H + gn - 1) / gn) * ((I + gk - 1) / gk) * (e8m0 ? 1u : sizeof(float));
        if (w13) { buf_w13_ = std::make_unique<uint8_t[]>(w13_bytes); std::memcpy(buf_w13_.get(), w13, w13_bytes); w13_ = buf_w13_.get(); }
        else     { w13_ = nullptr; }
        if (w2)  { buf_w2_ = std::make_unique<uint8_t[]>(w2_bytes); std::memcpy(buf_w2_.get(), w2, w2_bytes); w2_ = buf_w2_.get(); }
        else     { w2_ = nullptr; }
        if (w13_g) { buf_w13_g_ = std::make_unique<uint8_t[]>(w13g_bytes); std::memcpy(buf_w13_g_.get(), w13_g, w13g_bytes); w13_g_ = buf_w13_g_.get(); }
        else       { w13_g_ = nullptr; } // no per-expert scale for this block
        if (w2_g) { buf_w2_g_ = std::make_unique<uint8_t[]>(w2g_bytes); std::memcpy(buf_w2_g_.get(), w2_g, w2g_bytes); w2_g_ = buf_w2_g_.get(); }
        else       { w2_g_ = nullptr; } // no per-expert scale for this block
        // global scales are tiny; copy if provided
        if (w13_gs) { buf_w13_gs_ = std::make_unique<float[]>(E); std::memcpy(buf_w13_gs_.get(), w13_gs, E * sizeof(float)); w13_gs_ = buf_w13_gs_.get(); }
        else        { w13_gs_ = nullptr; }
        if (w2_gs) { buf_w2_gs_ = std::make_unique<float[]>(E); std::memcpy(buf_w2_gs_.get(), w2_gs, E * sizeof(float)); w2_gs_ = buf_w2_gs_.get(); }
        else       { w2_gs_ = nullptr; }

        if (!w13_ || !w2_) throw std::runtime_error("MOE_V2: null w13/w2");
    }

    // forward_many: M tokens, top_k=k. expert_ids/weights are [M][k] row-major.
    // input [M, hidden] bf16, output [M, hidden] fp32 (accumulated in place).
    void forward_many(int M, int k,
                      const uint32_t* expert_ids, const float* weights,
                      const uint16_t* input, float* output) {
        const int hidden = cfg_.hidden_size;
        const int inter = cfg_.intermediate_size;
        const int nel = cfg_.expert_num;
        const int groupN = cfg_.groupN;
        const int groupK = cfg_.groupK;

        // Per-token scratch (thread-local so parallel tokens don't clash).
        auto compute_token = [&](size_t t) {
            const uint16_t* xt = input + t * (size_t)hidden;
            float* out_t = output + t * (size_t)hidden;
            std::fill(out_t, out_t + hidden, 0.f);

            std::vector<float> gate_out(inter), up_out(inter), act_out(inter);
            std::vector<uint16_t> act_bf16(inter);
            std::vector<float> down_out(hidden);

            for (int r = 0; r < k; ++r) {
                uint32_t eid = expert_ids[t * (size_t)k + r];
                float w = weights[t * (size_t)k + r];
                if (eid >= (uint32_t)nel || w == 0.f) continue;

                // gate/up:  (1, inter) = x(1, hidden) x W13^T(2*inter, hidden)
                wt::gate_up(xt, w13_, w13_g_, w13_gs_, gate_out.data(), up_out.data(),
                            inter, hidden, eid, groupN, groupK);
                // gated activation
                ::xiaotu_moe::act::silu_gate(gate_out.data(), up_out.data(),
                                             act_out.data(), inter);
                // down: (1, hidden) = act(1, inter) x Wd^T(inter, hidden);
                // act re-quantized to the traits' activation dtype (bf16).
                bf16::convert_f32_to_bf16(act_out.data(), act_bf16.data(), (size_t)inter);
                wt::down(act_bf16.data(), w2_, w2_g_, w2_gs_, down_out.data(),
                         hidden, inter, eid, groupN, groupK);
                // accumulate weighted
                for (int h = 0; h < hidden; ++h)
                    out_t[h] += w * down_out[h];
            }
        };

        pool_.parallel_for((size_t)M, compute_token);
    }

    // forward_one: single token, single routed expert (for warm-up / tests).
    void forward_one(uint32_t expert_id, float weight,
                     const uint16_t* x, float* out) {
        uint32_t ids[1] = {expert_id};
        float w[1] = {weight};
        forward_many(1, 1, ids, w, x, out);
    }

    // Matches the runtime method name seen in the closed binary (no-op here).
    void warm_up() {}

    // Debug accessors (parity-check helpers; not part of the public API).
    const MOEConfigV2& config() const { return cfg_; }
    const void* debug_w13() const { return w13_; }
    const void* debug_w2() const { return w2_; }

private:
    MOEConfigV2 cfg_;
    std::unique_ptr<uint8_t[]> buf_w13_, buf_w2_, buf_w13_g_, buf_w2_g_;
    std::unique_ptr<float[]> buf_w13_gs_, buf_w2_gs_;
    const void* w13_;
    const void* w2_;
    const void* w13_g_;
    const void* w2_g_;
    const float* w13_gs_;
    const float* w2_gs_;
    ThreadPool pool_;
};

} // namespace xiaotu_moe

#endif // XIAOTU_MOE_MOE_V2_HPP
