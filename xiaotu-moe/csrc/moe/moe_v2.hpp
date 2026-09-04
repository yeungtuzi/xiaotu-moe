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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../kernels/bf16_gemm.hpp"
#include "numa_pool.hpp"   // persistent NUMA-aware worker pool

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
    // Storage bytes for the full [E] weight tensors. Each Derived provides
    // *_impl which accounts for its own element width and packing.
    static constexpr size_t w13_bytes(size_t E, size_t n2, size_t H) {
        return Derived::w13_bytes_impl(E, n2, H);
    }
    static constexpr size_t w2_bytes(size_t E, size_t H, size_t I) {
        return Derived::w2_bytes_impl(E, H, I);
    }
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
    // N-parallel capability flag. Packed4 (MXFP4/NVFP4) overrides this to true so
    // forward_many can split the (N-rows of the) GEMV across all worker threads
    // (ktransformers `split_range_n` technique). BF16/FP8 stay single-chunk.
    static constexpr bool kNParallel = false;
    // N-sliced variants used by the N-parallel dispatch. `both` is [2*inter]:
    //   gate in [0, inter), up in [inter, 2*inter).
    // `down` is [hidden], sliced over hidden. Slice is [n0, n1) over `inter`
    // (gate/up) or `hidden` (down). Derived provides *_slice_impl.
    static void gate_up_slice(const uint16_t* x, const void* w13, const void* w13_g,
                              const float* w13_gs, float* both, int inter, int hidden,
                              size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::gate_up_slice_impl(x, w13, w13_g, w13_gs, both, inter, hidden, eid, groupN, groupK, n0, n1);
    }
    static void down_slice(const uint16_t* act, const void* w2, const void* w2_g,
                           const float* w2_gs, float* down, int hidden, int inter,
                           size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::down_slice_impl(act, w2, w2_g, w2_gs, down, hidden, inter, eid, groupN, groupK, n0, n1);
    }
    // Batched slice variants: process `me` token-instances of ONE expert in a
    // single call so the packed kernel decodes each weight row once and
    // amortizes it over me accumulators (M-way FMA ILP + reuse of the decoded
    // weight across tokens — ktransformers 4-token block pattern). `xg` is
    // [me * hidden] (gate/up) contiguous input rows, `actg` [me * inter] (down);
    // `both_buf` [me*2*inter] / `down_buf` [me*hidden]. Slice is [n0,n1) over
    // `inter` (gate/up) or `hidden` (down), covering ALL me rows. Dispatches to
    // Derived::*_batch_impl; the base default loops per row over the
    // single-instance *_slice_impl, and packed4 overrides with a true M>1 kernel
    // (matmul_packed4_group's 4-token blocked path).
    static void gate_up_slice_batched(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                      const float* w13_gs, float* both_buf, int inter, int hidden,
                                      size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::gate_up_slice_batch_impl(me, xg, w13, w13_g, w13_gs, both_buf, inter, hidden,
                                          eid, groupN, groupK, n0, n1);
    }
    static void gate_up_slice_batch_impl(int me, const uint16_t* xg, const void* w13, const void* w13_g,
                                         const float* w13_gs, float* both_buf, int inter, int hidden,
                                         size_t eid, int groupN, int groupK, int n0, int n1) {
        for (int mi = 0; mi < me; ++mi)
            Derived::gate_up_slice_impl(xg + (size_t)mi * hidden, w13, w13_g, w13_gs,
                                        both_buf + (size_t)mi * (2 * inter),
                                        inter, hidden, eid, groupN, groupK, n0, n1);
    }
    static void down_slice_batched(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                   const float* w2_gs, float* down_buf, int hidden, int inter,
                                   size_t eid, int groupN, int groupK, int n0, int n1) {
        Derived::down_slice_batch_impl(me, actg, w2, w2_g, w2_gs, down_buf, hidden, inter,
                                       eid, groupN, groupK, n0, n1);
    }
    static void down_slice_batch_impl(int me, const uint16_t* actg, const void* w2, const void* w2_g,
                                      const float* w2_gs, float* down_buf, int hidden, int inter,
                                      size_t eid, int groupN, int groupK, int n0, int n1) {
        for (int mi = 0; mi < me; ++mi)
            Derived::down_slice_impl(actg + (size_t)mi * inter, w2, w2_g, w2_gs,
                                     down_buf + (size_t)mi * hidden,
                                     hidden, inter, eid, groupN, groupK, n0, n1);
    }
};

// ---- WeightTraits: BF16 (no quantization) ----
struct BF16WeightTraits : WeightTraitsBase<BF16WeightTraits> {
    using weight_t = uint16_t;      // bf16 storage
    static constexpr size_t w13_bytes_impl(size_t E, size_t n2, size_t H) {
        return E * n2 * H * sizeof(uint16_t);  // [E][2I][H] bf16
    }
    static constexpr size_t w2_bytes_impl(size_t E, size_t H, size_t I) {
        return E * H * I * sizeof(uint16_t);   // [E][H][I] bf16
    }

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

    size_t nthreads() const { return nthreads_; }

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

// Lock-free float accumulate via CAS on the IEEE-754 bit pattern. Used when
// several expert-grouping jobs contribute to the same token's output in
// parallel (x86 aligned 32-bit float is not natively atomic to RMW).
inline void atomic_add_f32(float* p, float v) {
    std::atomic<uint32_t>* a = reinterpret_cast<std::atomic<uint32_t>*>(p);
    uint32_t old = a->load(std::memory_order_relaxed);
    for (;;) {
        float cur;
        std::memcpy(&cur, &old, sizeof(cur));
        float nv = cur + v;
        uint32_t nbits;
        std::memcpy(&nbits, &nv, sizeof(nbits));
        if (a->compare_exchange_weak(old, nbits, std::memory_order_relaxed,
                                     std::memory_order_relaxed))
            break;
    }
}

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
        : cfg_(cfg), pool_(shared_numa_pool()) {
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
        // NOTE: byte sizes are per-weight-trait (BF16 2B, FP8 1B, packed4 H/2).
        // The old H/2 formula was only valid for packed4 and under-sized BF16/FP8
        // by 4x/2x -> out-of-bounds reads -> NaN. Fixed via trait helpers.
        const size_t w13_bytes = WeightTraits::w13_bytes(E, n2, H);
        const size_t w2_bytes = WeightTraits::w2_bytes(E, H, I);
        const size_t w13g_bytes = (size_t)E * ((n2 + gn - 1) / gn) * ((H + gk - 1) / gk) * (e8m0 ? 1u : sizeof(float));
        const size_t w2g_bytes = (size_t)E * ((H + gn - 1) / gn) * ((I + gk - 1) / gk) * (e8m0 ? 1u : sizeof(float));
        // NUMA-align the big weight snapshots: the MoE is bandwidth-bound and the
        // pool spans both sockets (84 threads each). Default allocation lands on
        // the constructor's node, so half the workers would read remote DRAM at
        // ~half bandwidth. Set the thread policy to INTERLEAVE for these
        // allocations (pages fault in already interleaved), then restore default.
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
        const size_t NASS = (size_t)M * (size_t)k;
        if (M <= 0 || k <= 0 || inter <= 0 || hidden <= 0) return;

        std::lock_guard<std::mutex> lock(mtx_);

        // N-parallel dispatch (packed4): split each expert's GEMV over its N rows
        // across all worker threads so a single token's 4k-row GEMV uses the whole
        // pool, not 1 thread (ktransformers `split_range_n`). BF16/FP8 take the
        // existing single-chunk paths below.
        if constexpr (wt::kNParallel) {
            forward_many_nsliced(M, k, expert_ids, weights, input, output);
            return;
        }

        // Zero the whole output once, up front (was per-token before).
        std::fill(output, output + (size_t)M * (size_t)hidden, 0.f);

        // --- Expert grouping -------------------------------
        // Old per-token loop read each expert block once per token×rank, so DRAM
        // traffic was ~ batch*top_k * (12 MB per expert block). Instead we group
        // every (token, rank) assignment by its expert and process each active
        // expert's block ONCE, over all tokens routed to it (the block stays hot
        // in cache across the expert's token sub-batch). Bandwidth drops to
        // ~ active_experts * 12 MB (up to ~7x for typical routing diversity).
        //
        // Assignment index ai = t*k + r reconstructs the token t=ai/k.

        // per_expert[e] = list of assignment indexes routed to expert e (w!=0).
        std::vector<std::vector<size_t>> per_expert((size_t)nel);
        std::vector<int> active;
        {
            std::vector<size_t> count((size_t)nel, 0);
            for (size_t ai = 0; ai < NASS; ++ai) {
                uint32_t eid = expert_ids[ai];
                if (eid < (uint32_t)nel && weights[ai] != 0.f) count[eid]++;
            }
            for (int e = 0; e < nel; ++e)
                if (count[e]) { per_expert[e].reserve(count[e]); active.push_back(e); }
            for (size_t ai = 0; ai < NASS; ++ai) {
                uint32_t eid = expert_ids[ai];
                if (eid < (uint32_t)nel && weights[ai] != 0.f) per_expert[eid].push_back(ai);
            }
        }

        // SiLU output scratch: one [inter] f32 block per assignment.
        act_scratch_.resize(NASS * (size_t)inter);
        // Down output scratch: one [hidden] f32 block per assignment.
        down_scratch_.resize(NASS * (size_t)hidden);

        // --- Adaptive dispatch ---------------------------------------------
        // Grouping pays when routing is concentrated (several tokens share an
        // active expert, so its 12 MB block is re-read from cache instead of
        // DRAM per token). With diverse routing (almost every expert active for a
        // handful of tokens) the two-phase + CAS overhead dominates, so we fall
        // back to the direct per-token loop. Threshold: group only when we have
        // >= 8 assignments on average per active expert.
        const size_t nave = active.size();
        // Grouping-factor threshold: group only when we have >= F assignments on
        // average per active expert. F defaults to 8; XIAOTU_MOE_GROUP_FACTOR
        // overrides (a large value forces the per-token path — used to A/B the
        // two paths on identical routing).
        static constexpr size_t kDefaultF = 8;
        size_t F = kDefaultF;
        if (const char* e = std::getenv("XIAOTU_MOE_GROUP_FACTOR")) {
            long v = std::atol(e);
            if (v > 0) F = (size_t)v;
        }
        if (nave * F <= NASS) {
            // --- Grouped path: each active expert's block read once. ---
            // Jobs split each expert's token list into chunks so fewer active
            // experts still keep all worker threads busy. Job = (eid, [ab, ae)).
            struct Job { int eid; size_t ab, ae; };
            std::vector<Job> jobs;
            {
                size_t total_assign = 0;
                for (int e : active) total_assign += per_expert[e].size();
                size_t nt = pool_.nthreads();
                size_t target_jobs = (nt > 4 ? nt * 4 : 16);
                size_t perjob = total_assign / (target_jobs ? target_jobs : 1);
                if (perjob < 8) perjob = 8;   // amortize per-job fixed cost
                if (perjob < 1) perjob = 1;
                for (int e : active) {
                    const auto& lst = per_expert[e];
                    for (size_t b = 0; b < lst.size(); b += perjob)
                        jobs.push_back({e, b, std::min(b + perjob, lst.size())});
                }
            }
            if (jobs.empty()) return;

            // Phase 1 (parallel over jobs): gate/up + SiLU -> act_scratch_.
            pool_.parallel_for(jobs.size(), [&](size_t ji) {
                const Job& job = jobs[ji];
                const auto& lst = per_expert[job.eid];
                std::vector<float> gate_buf(inter), up_buf(inter);
                float* act_base = act_scratch_.data();
                for (size_t it = job.ab; it < job.ae; ++it) {
                    size_t ai = lst[it];
                    size_t t = ai / (size_t)k;
                    const uint16_t* xt = input + t * (size_t)hidden;
                    wt::gate_up(xt, w13_, w13_g_, w13_gs_, gate_buf.data(), up_buf.data(),
                                inter, hidden, job.eid, groupN, groupK);
                    ::xiaotu_moe::act::silu_gate(gate_buf.data(), up_buf.data(),
                                                 act_base + ai * (size_t)inter, inter);
                }
            });

            // Phase 2 (parallel over jobs): bf16 -> down -> into down_scratch_.
            // (No cross-expert race on `output`: contributions are staged per
            // assignment and reduced token-wise in Phase 3.)
            pool_.parallel_for(jobs.size(), [&](size_t ji) {
                const Job& job = jobs[ji];
                const auto& lst = per_expert[job.eid];
                std::vector<uint16_t> act_bf16(inter);
                std::vector<float> down_buf(hidden);
                const float* act_base = act_scratch_.data();
                float* down_base = down_scratch_.data();
                for (size_t it = job.ab; it < job.ae; ++it) {
                    size_t ai = lst[it];
                    bf16::convert_f32_to_bf16(act_base + ai * (size_t)inter,
                                              act_bf16.data(), (size_t)inter);
                    wt::down(act_bf16.data(), w2_, w2_g_, w2_gs_, down_buf.data(),
                             hidden, inter, job.eid, groupN, groupK);
                    float* dst = down_base + ai * (size_t)hidden;
                    std::memcpy(dst, down_buf.data(), (size_t)hidden * sizeof(float));
                }
            });

            // Phase 3 (parallel over tokens): weighted reduce, per token, in rank
            // order — one thread per token, so no output contention.
            pool_.parallel_for((size_t)M, [&](size_t t) {
                const float* down_base = down_scratch_.data();
                float* out_t = output + t * (size_t)hidden;
                for (int r = 0; r < k; ++r) {
                    size_t ai = t * (size_t)k + r;
                    uint32_t eid = expert_ids[ai];
                    float w = weights[ai];
                    if (eid >= (uint32_t)nel || w == 0.f) continue;
                    const float* d = down_base + ai * (size_t)hidden;
                    for (int h = 0; h < hidden; ++h) out_t[h] += w * d[h];
                }
            });
            return;
        }

        // --- Fallback: direct per-token loop (diverse routing). ---
        pool_.parallel_for((size_t)M, [&](size_t t) {
            const uint16_t* xt = input + t * (size_t)hidden;
            float* out_t = output + t * (size_t)hidden;   // already zeroed above
            std::vector<float> gate_out(inter), up_out(inter), act_out(inter);
            std::vector<uint16_t> act_bf16(inter);
            std::vector<float> down_out(hidden);
            for (int r = 0; r < k; ++r) {
                size_t ai = t * (size_t)k + r;
                uint32_t eid = expert_ids[ai];
                float w = weights[ai];
                if (eid >= (uint32_t)nel || w == 0.f) continue;
                wt::gate_up(xt, w13_, w13_g_, w13_gs_, gate_out.data(), up_out.data(),
                            inter, hidden, eid, groupN, groupK);
                ::xiaotu_moe::act::silu_gate(gate_out.data(), up_out.data(),
                                             act_out.data(), inter);
                bf16::convert_f32_to_bf16(act_out.data(), act_bf16.data(), (size_t)inter);
                wt::down(act_bf16.data(), w2_, w2_g_, w2_gs_, down_out.data(),
                         hidden, inter, eid, groupN, groupK);
                for (int h = 0; h < hidden; ++h) out_t[h] += w * down_out[h];
            }
        });
    }

    // N-parallel forward_many (used when wt::kNParallel). Splits each active
    // expert's gate/up GEMV over chunks of the `inter` row dimension and the down
    // GEMV over chunks of `hidden`, so one token's big GEMV fans out across the
    // whole NUMA pool instead of a single worker thread (ktransformers
    // `split_range_n` technique). Race-free: every job writes disjoint N-slices.
    void forward_many_nsliced(int M, int k,
                              const uint32_t* expert_ids, const float* weights,
                              const uint16_t* input, float* output) {
        const int hidden = cfg_.hidden_size;
        const int inter = cfg_.intermediate_size;
        const int nel = cfg_.expert_num;
        const int groupN = cfg_.groupN;
        const int groupK = cfg_.groupK;
        const size_t NASS = (size_t)M * (size_t)k;
        if (M <= 0 || k <= 0 || inter <= 0 || hidden <= 0) return;

        std::fill(output, output + (size_t)M * (size_t)hidden, 0.f);

        // Per-expert instance bookkeeping (persistent exp_/active_/inst_idx_
        // members re-used every call). An "instance" = one (token, rank)
        // assignment routing to an expert. Instances of one expert are gathered
        // into CONTIGUOUS rows so each batched matmul runs M>1 (decode a weight
        // row once, amortize over me slots + M-way ILP — ktransformers
        // 4-token block).
        exp_.resize((size_t)nel);
        active_.clear();
        count_.assign((size_t)nel, 0);
        inst_idx_.assign(NASS, (size_t)0);
        for (size_t ai = 0; ai < NASS; ++ai) {
            uint32_t eid = expert_ids[ai];
            if (eid < (uint32_t)nel && weights[ai] != 0.f) count_[eid]++;
        }
        for (int e = 0; e < nel; ++e)
            if (count_[e]) { exp_[e].ai_list.clear(); exp_[e].ai_list.reserve(count_[e]); active_.push_back(e); }
        for (size_t ai = 0; ai < NASS; ++ai) {
            uint32_t eid = expert_ids[ai];
            if (eid < (uint32_t)nel && weights[ai] != 0.f) {
                inst_idx_[ai] = exp_[eid].ai_list.size();
                exp_[eid].ai_list.push_back(ai);
            }
        }
        if (active_.empty()) return;

        // Gather contiguous per-expert input rows and size the output buffers.
        // resize() only grows capacity; steady state reuses it (no allocation).
        for (int e : active_) {
            ExpBuf& g = exp_[e];
            size_t me = g.ai_list.size();
            g.xg.resize(me * (size_t)hidden);
            g.both.resize(me * (size_t)2 * (size_t)inter);
            g.act.resize(me * (size_t)inter);
            g.abf16.resize(me * (size_t)inter);
            g.down.resize(me * (size_t)hidden);
            for (size_t m = 0; m < me; ++m) {
                size_t t = g.ai_list[m] / (size_t)k;
                std::memcpy(g.xg.data() + m * (size_t)hidden, input + t * (size_t)hidden,
                            (size_t)hidden * sizeof(uint16_t));
            }
        }
        const size_t na = active_.size();

        // Number of N-chunks per active expert (~4x jobs/thread, coarse ~128 rows).
        int nc_gu = 1;
        if (pool_.nthreads() > 1) {
            size_t need = (pool_.nthreads() * 4) / std::max<size_t>(1, na);
            size_t maxc = (size_t)(inter / 128);
            if (maxc < 1) maxc = 1;
            if (need > maxc) need = maxc;
            if (need < 1) need = 1;
            nc_gu = (int)need;
        }
        int nc_d = 1;
        if (pool_.nthreads() > 1) {
            size_t need = (pool_.nthreads() * 4) / std::max<size_t>(1, na);
            size_t maxc = (size_t)(hidden / 128);
            if (maxc < 1) maxc = 1;
            if (need > maxc) need = maxc;
            if (need < 1) need = 1;
            nc_d = (int)need;
        }

        // Flattened job index for A2/B0 = sum over active experts of me*nc_gu.
        exp_off_.resize(na + 1);
        exp_off_[0] = 0;
        for (size_t e_idx = 0; e_idx < na; ++e_idx)
            exp_off_[e_idx + 1] = exp_off_[e_idx] + exp_[active_[e_idx]].ai_list.size() * (size_t)nc_gu;
        const size_t a2_total = exp_off_[na];

        // Phase A: batched gate+up slices, parallel over (expert, inter-chunk).
        pool_.parallel_for(active_.size() * (size_t)nc_gu, [&](size_t ji) {
            size_t e_idx = ji / (size_t)nc_gu;
            int c = (int)(ji % (size_t)nc_gu);
            int eid = active_[e_idx];
            ExpBuf& g = exp_[eid];
            int n0 = c * inter / nc_gu;
            int n1 = (c + 1) * inter / nc_gu;
            if (n1 > inter) n1 = inter;
            if (n0 >= n1) return;
            wt::gate_up_slice_batched((int)g.ai_list.size(), g.xg.data(), w13_, w13_g_, w13_gs_,
                                      g.both.data(), inter, hidden, (size_t)eid, groupN, groupK, n0, n1);
        });

        // Phase A2: gated-SiLU -> act, parallel over (expert, instance, chunk).
        pool_.parallel_for(a2_total, [&](size_t ji) {
            size_t e_idx = (size_t)(std::upper_bound(exp_off_.begin(), exp_off_.end(), ji)
                                    - exp_off_.begin()) - 1;
            int eid = active_[e_idx];
            ExpBuf& g = exp_[eid];
            size_t rem = ji - exp_off_[e_idx];
            size_t m = rem / (size_t)nc_gu;
            int c = (int)(rem % (size_t)nc_gu);
            int n0 = c * inter / nc_gu;
            int n1 = (c + 1) * inter / nc_gu;
            if (n1 > inter) n1 = inter;
            if (n0 >= n1) return;
            const float* b = g.both.data() + m * (size_t)2 * (size_t)inter;
            float* a = g.act.data() + m * (size_t)inter;
            for (int i = n0; i < n1; ++i) {
                float gv = b[i];
                a[i] = b[inter + i] * (gv / (1.f + std::exp(-gv)));
            }
        });

        // Phase B0: f32 act -> bf16, parallel over (expert, instance, chunk).
        pool_.parallel_for(a2_total, [&](size_t ji) {
            size_t e_idx = (size_t)(std::upper_bound(exp_off_.begin(), exp_off_.end(), ji)
                                    - exp_off_.begin()) - 1;
            int eid = active_[e_idx];
            ExpBuf& g = exp_[eid];
            size_t rem = ji - exp_off_[e_idx];
            size_t m = rem / (size_t)nc_gu;
            int c = (int)(rem % (size_t)nc_gu);
            int n0 = c * inter / nc_gu;
            int n1 = (c + 1) * inter / nc_gu;
            if (n1 > inter) n1 = inter;
            if (n0 >= n1) return;
            const float* a = g.act.data() + m * (size_t)inter;
            uint16_t* ab = g.abf16.data() + m * (size_t)inter;
            for (int i = n0; i < n1; ++i)
                ab[i] = bf16::fp32_to_bf16(a[i]);
        });

        // Phase B: batched down slices, parallel over (expert, h-chunk).
        pool_.parallel_for(active_.size() * (size_t)nc_d, [&](size_t ji) {
            size_t e_idx = ji / (size_t)nc_d;
            int c = (int)(ji % (size_t)nc_d);
            int eid = active_[e_idx];
            ExpBuf& g = exp_[eid];
            int n0 = c * hidden / nc_d;
            int n1 = (c + 1) * hidden / nc_d;
            if (n1 > hidden) n1 = hidden;
            if (n0 >= n1) return;
            wt::down_slice_batched((int)g.ai_list.size(), g.abf16.data(), w2_, w2_g_, w2_gs_,
                                   g.down.data(), hidden, inter, (size_t)eid, groupN, groupK, n0, n1);
        });

        // Phase C: weighted reduce per token (rank order) - no output contention.
        pool_.parallel_for((size_t)M, [&](size_t t) {
            float* out_t = output + t * (size_t)hidden;
            for (int r = 0; r < k; ++r) {
                size_t ai = t * (size_t)k + r;
                uint32_t eid = expert_ids[ai];
                float w = weights[ai];
                if (eid >= (uint32_t)nel || w == 0.f) continue;
                const float* d = exp_[eid].down.data() + inst_idx_[ai] * (size_t)hidden;
                for (int h = 0; h < hidden; ++h) out_t[h] += w * d[h];
            }
        });
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
    // Scratch shared by the two expert-grouping phases within one forward call.
    // Guarded by mtx_ so accidental concurrent forward_many on the same engine is
    // safe (the fork processes layers sequentially; concurrent calls are serialized).
    mutable std::mutex mtx_;
    // Persistent per-expert scratch for the N-parallel batched path. Defeats the
    // per-forward malloc churn of locals: capacities persist across calls, so
    // the steady-state hot loop does zero allocation (the prior local-vector
    // version regressed ~25% because it mmap/munmap'd every buffer every call).
    struct ExpBuf {
        std::vector<uint16_t> xg, abf16;   // me*hidden / me*inter (bf16 rows)
        std::vector<float>    both, act, down;  // me*2*inter / me*inter / me*hidden
        std::vector<size_t>   ai_list;
    };
    mutable std::vector<ExpBuf> exp_;       // sized to nel; only active experts used
    mutable std::vector<int> active_;       // reusable list of active expert ids
    mutable std::vector<size_t> count_;     // per-expert instance counts
    mutable std::vector<size_t> inst_idx_;  // ai -> instance rank within its expert
    mutable std::vector<size_t> exp_off_;   // prefix sums for A2/B0 job mapping
    std::vector<float> act_scratch_;
    std::vector<float> down_scratch_;
    std::vector<float> both_scratch_;          // N-parallel gate+up (2*inter/assign)
    std::vector<uint16_t> act_bf16_scratch_;   // N-parallel bf16 activation
    // Shared process-wide NUMA pool (one pool for ALL layers, mirroring lk_moe's
    // single Backend_NUMA engine). Per-layer pools would give ~61 x threads and
    // thrash the scheduler; a single pool keeps the total = XIAOTU_MOE_THREADS.
    NumaWorkPool& pool_;
};

} // namespace xiaotu_moe

#endif // XIAOTU_MOE_MOE_V2_HPP
