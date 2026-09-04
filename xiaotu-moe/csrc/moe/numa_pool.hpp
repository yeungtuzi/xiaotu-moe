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

// Signal-based native-stack dump (diagnostics for stuck-forward investigation).
// Registered in NumaWorkPool so every thread inherits the handler; on SIGUSR2
// the receiving thread appends its own glibc backtrace to a single file. The
// harness signals every /proc/<pid>/task/*/tid to get ALL threads' native
// stacks. Needs -g -fno-omit-frame-pointer for depth; still accurate with -O3.
#include <signal.h>
#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <atomic>

namespace xiaotu_moe {
namespace {
std::atomic<int> g_stack_dump_fd{-1};

void stack_dump_handler(int /*sig*/) {
    int fd = g_stack_dump_fd.load(std::memory_order_relaxed);
    if (fd < 0) return;
    void* bt[160];
    int n = backtrace(bt, 160);
    int tid = (int)syscall(SYS_gettid);
    char hdr[96];
    int hl = snprintf(hdr, sizeof(hdr), "\n===== native stack tid=%d n=%d =====", tid, n);
    if (hl > 0) (void)!write(fd, hdr, (size_t)hl);
    (void)!write(fd, "\n", 1);
    backtrace_symbols_fd(bt, n, fd);
    fdatasync(fd);
}

struct StackDumperRegistrar {
    StackDumperRegistrar() {
        char path[128];
        int pid = (int)getpid();
        snprintf(path, sizeof(path), "/tmp/xiaotu_stack_%d.txt", pid);
        int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, 0644);
        if (fd >= 0) {
            g_stack_dump_fd.store(fd, std::memory_order_relaxed);
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = stack_dump_handler;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;                     // no SA_RESTART -> interrupts futex wait
            sigaction(SIGUSR2, &sa, nullptr);
        }
    }
};
} // anonymous namespace
} // namespace xiaotu_moe

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
// Thread-policy based NUMA interleave: set the calling thread's memory policy
// to MPOL_INTERLEAVE so that its new allocations (operator new / mmap for the
// large weight snapshots) fault their pages in already interleaved across all
// NUMA nodes — the most reliable path (mbind-on-existing-pages with MF_MOVE is
// fragile: it can EINVAL on sub-ranges and only migrates existing pages). We
// bracket each weight copy: begin() sets interleave, end() restores default.
// lk-moe / ktransformers similarly place weights so both CPU sockets read their
// weight pages locally (the MoE is bandwidth-bound; with weights on one socket
// the other socket's 84 threads read remote DRAM at ~half bandwidth).
// ---------------------------------------------------------------------------
inline void numa_interleave_begin() {
    static int nmask = []() {
        NumaTopology t = discover_numa_topology();
        if (t.node_cpus.size() > 1) return (int)((1UL << t.node_cpus.size()) - 1);
        return 0;
    }();
    if (nmask > 1) {
        unsigned long mask = (unsigned long)nmask;
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &mask, sizeof(mask) * 8);
    }
}
inline void numa_interleave_end() {
    syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
}

// ---------------------------------------------------------------------------
// NumaWorkPool
// ---------------------------------------------------------------------------
class NumaWorkPool {
public:
    explicit NumaWorkPool(size_t n = 0)
        : nt_(n == 0 ? default_threads() : n), stop_(false),
          current_gen_(0), counter_(0), remaining_(0), n_(0) {
        // per-worker completion slots kept only for diagnostics; the completion
        // barrier no longer waits on them (see parallel_for).
        worker_gen_.reset(new std::atomic<uint64_t>[nt_]);
        for (size_t w = 0; w < nt_; ++w) worker_gen_[w].store(0);
        static StackDumperRegistrar registrar;  // SIGUSR2 native-stack dump
        start_workers();
    }

    ~NumaWorkPool() { stop(); }

    size_t nthreads() const { return nt_; }

    // run fn(i) for i in [0, n). Persistent workers; dynamic index scheduling.
    template <typename F>
    void parallel_for(size_t n, const F& fn) {
        // Serialize access to the shared generation/counter state. When the pool
        // is shared process-wide (one pool for all MoE layers, mirroring lk_moe's
        // single Backend_NUMA), different layers could otherwise race on
        // counter_/task_/current_gen_. vLLM-eager drives layers sequentially, so
        // this lock is uncontended in practice; it only orders genuinely-concurrent
        // forward_many calls issued from multiple driver threads.
        std::lock_guard<std::mutex> call_lock(call_mtx_);
        if (nt_ <= 1 || n <= 1) {
            for (size_t i = 0; i < n; ++i) fn(i);
            return;
        }
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            task_ = std::function<void(size_t)>(fn);  // type-erased copy
            n_ = n;
            // MONOTONIC ticket counter (never reset). Each call occupies the
            // ticket range [start_+0, start_+n). Workers bound to THIS call
            // compute  i = ticket - start_  and only run for i in [0,n); any
            // ticket >= start_+n means the worker's own range is exhausted and
            // it re-arms for the next generation.
            //
            // PUBLISH ORDER: current_gen_ is bumped BEFORE start_/remaining_ are
            // written. A worker fast-path sees current_gen_==its snapshot gen only
            // while the caller has not yet started the next call, so start_ still
            // equals its snapshot start -> a range-exhausted drop there is a TRUE
            // future-gap (safe, no lock). Once current_gen_ advances, workers
            // re-anchor under the lock and read a consistent (start_,n_) pair.
            gen = ++current_gen_;        // open a new generation (publish FIRST)
            start_ = counter_.load();
            remaining_.store(n);      // outstanding work items in this call
            // diagnostic processed-bitset for this call (only when debug/trace on)
            if (diag_active()) proc_vec_.assign(n, 0);
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(done_mtx_);
            auto t0 = std::chrono::steady_clock::now();
            const auto deadline = t0 + std::chrono::seconds(300);
            const char* tre = std::getenv("XIAOTU_MOE_POOL_TRACE");
            long rstep_s = (tre && std::atoi(tre) > 0) ? (long)std::atoi(tre) : 0;
            bool late = false;
            if (rstep_s > 0) {
                // STALL-TRACE mode: sliced plain waits so live state can be
                // printed while a call is outstanding (no abort). Only enabled
                // explicitly for diagnosing the CPU-MoE conc stall. The normal
                // path below uses an immediate predicate wait.
                int last_report = -1;
                for (;;) {
                    if (stop_ || remaining_.load(std::memory_order_acquire) == 0) break;
                    auto now = std::chrono::steady_clock::now();
                    if (now >= deadline) { late = true; break; }
                    auto nw = now + std::chrono::seconds(rstep_s);
                    if (nw > deadline) nw = deadline;
                    done_cv_.wait_until(lk, nw);   // plain wait to allow periodic wake
                    long es = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                    int rk = (int)(es / 1000 / rstep_s);
                    if (rk != last_report) {
                        last_report = rk;
                        if (remaining_.load(std::memory_order_acquire) > 0) {
                            std::string miss; size_t nmiss = 0;
                            size_t first_missing = (size_t)-1;
                            for (size_t z = 0; z < n_; ++z) {
                                if (!proc_vec_[z]) {
                                    if (first_missing == (size_t)-1) first_missing = z;
                                    if (nmiss < 16) miss += std::to_string(start_+z)+" ";
                                    ++nmiss;
                                }
                            }
                            fprintf(stderr,
                                "[pool] STALL t=%lds gen=%llu n=%zu start=%zu rem=%zu livegen=%llu dropped=%zu first=%zu missing=%zu {%s}\n",
                                es/1000, (unsigned long long)gen, n_, start_, remaining_.load(),
                                (unsigned long long)current_gen_.load(), dropped_.load(),
                                first_missing, nmiss, miss.c_str());
                        }
                    }
                }
            } else {
                // Completion countdown barrier (normal). We wait for the workers
                // that ACTUALLY claim an index; idle workers are never waited on.
                // predicate wait returns promptly on the last worker's notify.
                late = !done_cv_.wait_until(lk, deadline, [&] {
                    if (stop_) return true;
                    return remaining_.load(std::memory_order_acquire) == 0;
                });
            }
            long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            // Genuine hang: remaining>0 past the 300s deadline -> dump + abort.
            if (late) {
                fprintf(stderr, "[pool] WATCHDOG fired: gen=%llu n=%zu start=%zu end=%zu counter=%zu remaining=%zu current_gen=%llu dropped=%zu\n",
                    (unsigned long long)gen, n_, start_, start_+n_, counter_.load(),
                    remaining_.load(), (unsigned long long)current_gen_.load(),
                    dropped_.load());
                if (diag_active()) {
                    size_t miss=0; std::string s;
                    for (size_t z=0;z<n_;++z) if(!proc_vec_[z]){ s+=std::to_string(start_+z)+" "; ++miss;}
                    fprintf(stderr, "  missing %zu tickets: %s\n", miss, s.c_str());
                }
                for (size_t w = 0; w < nt_; ++w)
                    fprintf(stderr, "  worker[%zu] slot=%llu\n",
                        w, (unsigned long long)worker_gen_[w].load());
                abort();
            }
            long slow_ms = 2000;
            if (const char* e = std::getenv("XIAOTU_MOE_POOL_SLOW_MS")) {
                long v = std::atol(e); if (v >= 0) slow_ms = v;
            }
            if (elapsed_ms >= slow_ms) {
                fprintf(stderr,
                    "[pool] SLOW parallel_for: elapsed=%ldms gen=%llu n=%zu counter=%zu remaining=%zu current_gen=%llu\n",
                    elapsed_ms, (unsigned long long)gen, n_, counter_.load(),
                    remaining_.load(), (unsigned long long)current_gen_.load());
                size_t nlag = 0;
                for (size_t w = 0; w < nt_; ++w) {
                    uint64_t s = worker_gen_[w].load(std::memory_order_acquire);
                    if (s != gen) {
                        fprintf(stderr, "  LAGGARD worker[%zu] slot=%llu\n",
                            (unsigned long long)w, (unsigned long long)s);
                        ++nlag;
                    }
                }
                fprintf(stderr, "  laggards=%zu/%zu remaining=%zu\n", nlag, (size_t)nt_,
                        remaining_.load());
            }
        }
        if (std::getenv("XIAOTU_MOE_POOL_DEBUG")) {
            fprintf(stderr, "[pool] parallel_for(n=%zu) returned at gen=%llu\n",
                    n, (unsigned long long)gen);
        }
    }

    static size_t default_threads() {        unsigned hw = std::thread::hardware_concurrency();
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
            size_t n = 0, start = 0;
            uint64_t gen = 0;
            {
                std::unique_lock<std::mutex> lk(work_mtx_);
                cv_.wait(lk, [&] { return stop_ || current_gen_.load() != my_last_gen; });
                if (stop_) return;
                gen = current_gen_.load();
                my_last_gen = gen;
                worker_gen_[w].store(gen, std::memory_order_release);
                local_task = task_;   // copy the type-erased function
                n = n_;
                start = start_;       // this call's first ticket
            }
            for (;;) {
                size_t t = counter_.fetch_add(1, std::memory_order_relaxed);
                size_t i = t - start;
                uint64_t g = current_gen_.load(std::memory_order_acquire);
                if (g == gen) {
                    // Snapshot is authoritative: caller has not started the next
                    // call (publish-first), so start_/n_/task_ unchanged.
                    if (i >= n) break;   // true future-gap -> safe lock-free drop
                    local_task(i);
                    if (diag_active()) proc_vec_[i] = 1;
                    // Last worker to reach 0 notifies the completion condvar.
                    // It must hold done_mtx_ (the same mutex the caller's
                    // predicate-wait runs under) so the notify can never fall in
                    // the caller's pred-check -> condvar-wait window (lost wakeup).
                    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> gl(done_mtx_);
                        done_cv_.notify_all();
                    }
                    continue;
                }
                // Generation advanced: re-anchor to the current LIVE call under
                // the lock (consistent start_/n_/task_), then evaluate t against
                // the authoritative live range.
                {
                    std::function<void(size_t)> nt;
                    size_t nn, ns; uint64_t ng;
                    {
                        std::lock_guard<std::mutex> lk(work_mtx_);
                        if (stop_) return;
                        ng = current_gen_.load();
                        nt = task_;
                        nn = n_;
                        ns = start_;
                    }
                    gen = ng;
                    my_last_gen = ng;
                    worker_gen_[w].store(ng, std::memory_order_release);
                    local_task = std::move(nt);
                    n = nn; start = ns;
                    i = t - start;
                    if (i >= n) {               // genuinely beyond live range -> re-arm
                        dropped_.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    local_task(i);
                    if (diag_active()) proc_vec_[i] = 1;
                    // Lock-protected notify (same reasoning as the fast path): hold
                    // done_mtx_ so the completion notify can't be missed by the
                    // caller's pred-check -> condvar-wait transition.
                    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> gl(done_mtx_);
                        done_cv_.notify_all();
                    }
                }
            }
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

    static bool pool_dbg() {
        static const int v = []{ const char* e = std::getenv("XIAOTU_MOE_POOL_DBG");
            return e && std::atoi(e) > 0 ? 1 : 0; }();
        return v != 0;
    }

    // True when per-call processed-bitset should be maintained (pool debug or
    // stall-trace mode). Cheap enough for diagnosis; off in production.
    static bool diag_active() {
        if (pool_dbg()) return true;
        static const int v = []{ const char* e = std::getenv("XIAOTU_MOE_POOL_TRACE");
            return e && std::atoi(e) > 0 ? 1 : 0; }();
        return v != 0;
    }

    // work dispatch
    std::mutex work_mtx_;
    std::condition_variable cv_;
    std::mutex call_mtx_;  // serializes parallel_for entry (shared-pool safety)
    std::function<void(size_t)> task_;
    size_t n_;
    std::atomic<size_t> counter_;
    size_t start_ = 0;                 // first ticket index of current call (monotonic)
    std::atomic<size_t> remaining_;  // outstanding work items in current call (countdown barrier)
    std::atomic<uint64_t> current_gen_;
    std::atomic<size_t> dropped_{0};  // diagnostic: tickets dropped (future-gap)
    std::vector<unsigned char> proc_vec_;  // diagnostic processed-bitset
    // per-worker completion slots: worker[w] writes only worker_gen_[w].
    std::unique_ptr<std::atomic<uint64_t>[]> worker_gen_;
    bool stop_;

    // completion barrier
    std::mutex done_mtx_;
    std::condition_variable done_cv_;

    std::vector<std::thread> workers_;
};

// ---------------------------------------------------------------------------
// Shared process-wide NUMA pool.
//
// One persistent worker pool for the whole process, shared by every MOE_V2
// layer. This mirrors lk_moe's architecture: a single shared CPU MoE engine
// (Backend_NUMA, LK_THREADS workers) is reused across all 61 layers, instead
// of giving each layer its own pool (which would balloon to ~61 x threads and
// thrash the scheduler). The C++11 magic-static guarantees thread-safe
// one-time construction; the worker count is set by XIAOTU_MOE_THREADS (or the
// hardware concurrency) exactly once.
// ---------------------------------------------------------------------------
inline NumaWorkPool& shared_numa_pool() {
    static NumaWorkPool pool;
    return pool;
}

} // namespace xiaotu_moe

#endif // XIAOTU_MOE_NUMA_POOL_HPP
