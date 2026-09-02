// xiaotu-moe: persistent NUMA-aware worker pool (Backend_NUMA equivalent).
//
// This is the CPU MoE thread backend, an open reimplementation of lk_moe's
// Backend_NUMA. It provides:
//   * persistent worker threads (created once, reused across forward calls —
//     no per-call thread spawn/join),
//   * NUMA-node-aware affinity: each worker pinned to a distinct physical core,
//     spread across NUMA nodes (cross-node communication minimized),
//   * dynamic scheduling via a shared atomic index (implicit work stealing:
//     workers pull the next index, so no worker starves while others run long),
//   * optional NUMA-interleaved allocation for engine-owned buffers (default on
//     for weight snapshots when >1 node), via the mbind syscall.
//
// IMPORTANT: builds WITHOUT libnuma (per project constraint). Topology comes from
// /proc/cpuinfo + /sys/devices/system/node; affinity via sched_setaffinity
// (glibc); memory interleave via the raw mbind syscall (<linux/mempolicy.h>).
//
// License: Apache-2.0.

#ifndef XIAOTU_MOE_NUMA_POOL_HPP
#define XIAOTU_MOE_NUMA_POOL_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <utility>
#include <unordered_map>

namespace xiaotu_moe {

// ---------------------------------------------------------------------------
// Topology discovery (no libnuma: /proc/cpuinfo + /sys/devices/system/node).
// ---------------------------------------------------------------------------

// parse a Linux cpulist ("0-3,8,11-15") into a sorted set of cpu ids.
inline std::set<int> parse_cpulist(const std::string& s) {
    std::set<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        auto dash = tok.find('-');
        if (dash == std::string::npos) {
            out.insert(std::atoi(tok.c_str()));
        } else {
            int a = std::atoi(tok.substr(0, dash).c_str());
            int b = std::atoi(tok.substr(dash + 1).c_str());
            if (a > b) std::swap(a, b);
            for (int c = a; c <= b; ++c) out.insert(c);
        }
    }
    return out;
}

struct NumaTopology {
    // For each configured NUMA node: the set of cpu ids in it.
    std::vector<std::vector<int>> node_cpus;
    // For each cpu id: node index.
    std::unordered_map<int, int> cpu_node;
    // For each cpu id: physical core id (from core id); used to avoid pinning
    // siblings of the same physical core.
    std::unordered_map<int, int> cpu_core;
};

inline NumaTopology discover_numa_topology() {
    NumaTopology t;
    // Nodes 0.. while nodeX/cpulist exists.
    for (int n = 0;; ++n) {
        std::string p = "/sys/devices/system/node/node" + std::to_string(n) + "/cpulist";
        std::ifstream f(p);
        if (!f.good()) { if (n == 0) return t; break; }
        std::string s; std::getline(f, s);
        std::vector<int> cpus;
        for (int c : parse_cpulist(s)) { cpus.push_back(c); t.cpu_node[c] = n; }
        if (!cpus.empty()) t.node_cpus.push_back(cpus);
    }
    // Physical core ids from /proc/cpuinfo (boarder.py reads the first block; we
    // use the standard "processor : N" / "core id : C" pairing).
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line; int cur_proc = -1; int cur_core = -1;
    auto flush = [&]() {
        if (cur_proc >= 0 && cur_core >= 0) t.cpu_core[cur_proc] = cur_core;
    };
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("processor", 0) == 0) { flush(); cur_proc = std::atoi(line.c_str() + 10); cur_core = -1; }
        else if (line.rfind("core id", 0) == 0) { cur_core = std::atoi(line.c_str() + 8); }
    }
    flush();
    return t;
}

// ---------------------------------------------------------------------------
// mbind-based NUMA-interleaved allocation (raw syscall, no libnuma).
// ---------------------------------------------------------------------------
inline void numa_set_interleaved(void* addr, size_t len) {
    static int nodes_initialized = []() {
        // gather configured nodes at first use
        NumaTopology t = discover_numa_topology();
        if (t.node_cpus.size() > 1) {
            unsigned long mask = (1UL << t.node_cpus.size()) - 1;
            return static_cast<int>(mask);
        }
        return 0;
    }();
    if (nodes_initialized <= 1) return;  // single node: default policy fine
    unsigned long nodemask = static_cast<unsigned long>(nodes_initialized);
    long rc = syscall(SYS_mbind, addr, len, MPOL_INTERLEAVE,
                      &nodemask, sizeof(nodemask) * 8, MPOL_MF_MOVE);
    (void)rc;
}

// ---------------------------------------------------------------------------
// NumaWorkPool
// ---------------------------------------------------------------------------
class NumaWorkPool {
public:
    explicit NumaWorkPool(size_t n = 0)
        : nt_(n == 0 ? default_threads() : n), stop_(false),
          current_gen_(0), counter_(0), n_(0) {
        // per-worker completion slots; each worker owns exactly one and only
        // ever writes its own — no shared accumulator is reset across
        // generations, which eliminates the cross-generation barrier race.
        worker_gen_.reset(new std::atomic<uint64_t>[nt_]);
        for (size_t w = 0; w < nt_; ++w) worker_gen_[w].store(0);
        start_workers();
    }

    ~NumaWorkPool() { stop(); }

    size_t nthreads() const { return nt_; }

    // run fn(i) for i in [0, n). Persistent workers; dynamic index scheduling.
    template <typename F>
    void parallel_for(size_t n, const F& fn) {
        if (nt_ <= 1 || n <= 1) {
            for (size_t i = 0; i < n; ++i) fn(i);
            return;
        }
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            task_ = std::function<void(size_t)>(fn);  // type-erased copy
            n_ = n;
            counter_.store(0);
            gen = ++current_gen_;        // open a new generation
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(done_mtx_);
            // Tripwire only: the per-worker barrier provably cannot early-return,
            // so the ONLY reason this would time out is a genuine deadlock (or a
            // worker starved for minutes). We keep the timeout generous (300s)
            // because on loaded hosts (e.g. this box with an unrelated vLLM server
            // at load>130) a sleeping worker can take ~5s just to be woken by cv_ —
            // a short timeout would mischaracterize that self-recovering slow wake
            // as a hang and abort the host process. Only abort on true starvation.
            if (!done_cv_.wait_for(lk, std::chrono::seconds(300), [&] {
                if (stop_) return true;
                for (size_t w = 0; w < nt_; ++w)
                    if (worker_gen_[w].load(std::memory_order_acquire) != gen) return false;
                return true;
            })) {
                fprintf(stderr, "[pool] WATCHDOG fired: gen=%llu n=%zu counter=%zu current_gen=%llu\n",
                    (unsigned long long)gen, n_, counter_.load(),
                    (unsigned long long)current_gen_.load());
                for (size_t w = 0; w < nt_; ++w)
                    fprintf(stderr, "  worker[%zu] slot=%llu\n",
                        w, (unsigned long long)worker_gen_[w].load());
                abort();
            }
        }
        if (std::getenv("XIAOTU_MOE_POOL_DEBUG")) {
            fprintf(stderr, "[pool] parallel_for(n=%zu) returned at gen=%llu\n",
                    n, (unsigned long long)gen);
        }
    }

    static size_t default_threads() {
        unsigned hw = std::thread::hardware_concurrency();
        size_t nt = hw > 0 ? (size_t)hw : 1;
        if (const char* e = std::getenv("XIAOTU_MOE_THREADS")) {
            long v = std::atol(e);
            if (v > 0) nt = (size_t)v;
        }
        return nt;
    }

private:
    void start_workers() {
        topo_ = discover_numa_topology();
        // Build a list of distinct physical cores (across all allowed cpus) to
        // pin workers to, interleaved by NUMA node.
        cores_ = {};
        std::set<std::pair<int, int>> seen_core; // (core_id, node) seen
        for (size_t ni = 0; ni < topo_.node_cpus.size(); ++ni) {
            for (int cpu : topo_.node_cpus[ni]) {
                int coreid = -1;
                auto it = topo_.cpu_core.find(cpu);
                if (it != topo_.cpu_core.end()) coreid = it->second;
                auto key = std::make_pair(coreid, (int)ni);
                if (coreid >= 0 && seen_core.count(key)) continue;
                seen_core.insert(key);
                cores_.push_back(cpu);
            }
        }
        if (cores_.empty()) {
            for (int c = 0; c < (int)nt_; ++c) cores_.push_back(c);
        }
        // pin workers round-robin across the physical-core list
        for (size_t w = 0; w < nt_; ++w) {
            int cpu = cores_[w % cores_.size()];
            workers_.emplace_back([this, cpu, w] {
                pin_to(cpu);
                worker_loop(w);
            });
        }
    }

    static void pin_to(int cpu) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);  // best effort
    }

    void worker_loop(size_t w) {
        uint64_t my_last_gen = 0;  // per-worker: generation this thread handled
        for (;;) {
            std::function<void(size_t)> local_task;
            size_t n;
            uint64_t gen;
            {
                std::unique_lock<std::mutex> lk(work_mtx_);
                cv_.wait(lk, [&] { return stop_ || current_gen_.load() != my_last_gen; });
                if (stop_) return;
                gen = current_gen_.load();
                my_last_gen = gen;
                local_task = task_;   // copy the type-erased function
                n = n_;
            }
            for (;;) {
                size_t i = counter_.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) break;
                local_task(i);
            }
            // mark THIS worker done for generation `gen`; no shared accumulator.
            worker_gen_[w].store(gen, std::memory_order_release);
            done_cv_.notify_all();
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& th : workers_) if (th.joinable()) th.join();
    }

    size_t nt_;
    NumaTopology topo_;
    std::vector<int> cores_;

    // work dispatch
    std::mutex work_mtx_;
    std::condition_variable cv_;
    std::function<void(size_t)> task_;
    size_t n_;
    std::atomic<size_t> counter_;
    std::atomic<uint64_t> current_gen_;
    // per-worker completion slots: worker[w] writes only worker_gen_[w].
    std::unique_ptr<std::atomic<uint64_t>[]> worker_gen_;
    bool stop_;

    // completion barrier
    std::mutex done_mtx_;
    std::condition_variable done_cv_;

    std::vector<std::thread> workers_;
};

} // namespace xiaotu_moe

#endif // XIAOTU_MOE_NUMA_POOL_HPP
