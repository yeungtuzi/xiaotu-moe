// xiaotu-moe Python binding (pybind11)
//
// Exposes the MOE_V2 family plus MOEConfigV2, mirroring the public surface of
// lk_moe's own pybind11 module (see LK_MOE_ABI_SPEC.md) but written fresh and
// self-contained. Constructors follow the real ABI: every engine takes
// (cfg, w13_weight, w2_weight, w13_scale=0, w2_scale=0, w13_global_scale=0,
//  w2_global_scale=0) where the weight/scale args are integer data pointers
// (numpy arrays are also accepted for convenience; their data() is used).
//
// Phase 1 ships BF16 (MOE_BF16 / MOE_FP16); Phase 2 adds FP8
// (MOE_FP8 / MOE_FP8_FP16) then WNA16 / MXFP4. Quantized traits each implement
// gate_up/down and never touch the orchestration loop.
//
// License: Apache-2.0.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <csignal>
#include <cstdio>
#include <cstring>
#include <execinfo.h>
#define _GNU_SOURCE
#include <ucontext.h>
#include <sys/ucontext.h>

#include "../moe/moe_v2.hpp"
#include "../moe/moe_v2_fp8.hpp"
#include "../moe/moe_v2_packed4.hpp"

namespace py = pybind11;
using namespace xiaotu_moe;

// ---- SIGSEGV diagnostics (debug build): print faulting addr + stack ----
namespace {
struct sigaction g_prev_segv {};
volatile sig_atomic_t g_in_handler = 0;

void xt_sigsegv_handler(int signo, siginfo_t* si, void* ctx) {
    fprintf(stderr, "\n[XTSIG] SIGSEGV at faulting address %p signal %d\n",
            (void*)(si ? si->si_addr : nullptr), signo);
    ucontext_t* uc = (ucontext_t*)ctx;
    if (uc) {
        auto& g = uc->uc_mcontext.gregs;
        fprintf(stderr,
                "[XTSIG] RIP=%p RSP=%p RBP=%p\n"
                "[XTSIG] RAX=%p RBX=%p RCX=%p RDX=%p\n"
                "[XTSIG] RSI=%p RDI=%p R8=%p R9=%p R10=%p R11=%p\n"
                "[XTSIG] R12=%p R13=%p R14=%p R15=%p\n",
                (void*)g[REG_RIP], (void*)g[REG_RSP], (void*)g[REG_RBP],
                (void*)g[REG_RAX], (void*)g[REG_RBX], (void*)g[REG_RCX], (void*)g[REG_RDX],
                (void*)g[REG_RSI], (void*)g[REG_RDI], (void*)g[REG_R8], (void*)g[REG_R9],
                (void*)g[REG_R10], (void*)g[REG_R11],
                (void*)g[REG_R12], (void*)g[REG_R13], (void*)g[REG_R14], (void*)g[REG_R15]);
    }
    void* bt[48]; int n = backtrace(bt, 48);
    backtrace_symbols_fd(bt, n, 2);
    fflush(stderr);
    // restore default so a second fault is a clean abort
    if (g_prev_segv.sa_handler == SIG_DFL || g_prev_segv.sa_handler == SIG_IGN)
        std::signal(SIGSEGV, SIG_DFL);
    else if (g_prev_segv.sa_sigaction)
        sigaction(SIGSEGV, &g_prev_segv, nullptr);
    std::raise(SIGSEGV);
}

void maybe_install_sigsegv_handler() {
    struct sigaction sa {};
    sa.sa_sigaction = xt_sigsegv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_prev_segv);
}
}  // namespace

// Extract a raw const pointer from an argument that may be an integer (data_ptr),
// a numpy array (.data()), or None/0 (-> nullptr).
static const void* as_ptr(const py::object& o) {
    if (o.is_none()) return nullptr;
    if (py::isinstance<py::int_>(o)) {
        long long v = py::cast<long long>(o);
        return reinterpret_cast<const void*>(v);
    }
    if (py::isinstance<py::array>(o))
        return py::cast<py::array>(o).data();
    return nullptr;
}

// Typed pointer extractors so cpu_prefill/cpu_decode accept BOTH numpy arrays
// and the integer `data_ptr()` values the Lvllmds4-x fork passes (mirroring the
// real lk_moe binding), exactly matching the fork's _cpu_prefill/_cpu_decode.
static const uint32_t* as_u32(const py::object& o) {
    return reinterpret_cast<const uint32_t*>(as_ptr(o));
}
static const float* as_f32(const py::object& o) {
    return reinterpret_cast<const float*>(as_ptr(o));
}
static const uint16_t* as_u16(const py::object& o) {
    return reinterpret_cast<const uint16_t*>(as_ptr(o));
}
static float* as_f32m(const py::object& o) {  // mutable output
    return reinterpret_cast<float*>(const_cast<void*>(as_ptr(o)));
}

template <typename WT, typename ACT>
static void bind_moe_class(py::module& m, const char* name) {
    using MOE = MOE_V2<WT, ACT>;
    py::class_<MOE>(m, name)
        .def(py::init([](const MOEConfigV2& cfg,
                         py::object w13, py::object w2,
                         py::object w13_g, py::object w2_g,
                         py::object w13_global, py::object w2_global) {
            return new MOE(cfg, as_ptr(w13), as_ptr(w2), as_ptr(w13_g), as_ptr(w2_g),
                           static_cast<const float*>(as_ptr(w13_global)),
                           static_cast<const float*>(as_ptr(w2_global)));
        }), py::arg("cfg"), py::arg("w13_weight"), py::arg("w2_weight"),
            py::arg("w13_scale") = py::int_(0), py::arg("w2_scale") = py::int_(0),
            py::arg("w13_global_scale") = py::int_(0), py::arg("w2_global_scale") = py::int_(0))
        .def("cpu_decode", [](MOE& self,
                              py::object stream_unused,  // placeholder (GPU stream ptr)
                              int qlen, int top_k,
                              py::object hidden, py::object expert_ids,
                              py::object weights, py::object out_gpu) {
            // hidden: [qlen, hidden] bf16 (uint16 storage)
            // expert_ids/weights: [qlen, top_k]
            // out_gpu: [qlen, hidden] fp32
            self.forward_many(qlen, top_k,
                              as_u32(expert_ids),
                              as_f32(weights),
                              as_u16(hidden),
                              as_f32m(out_gpu));
        }, py::arg("stream"), py::arg("qlen"), py::arg("top_k"),
           py::arg("hidden"), py::arg("expert_ids"), py::arg("weights"),
           py::arg("out_gpu"))
        .def("cpu_prefill", [](MOE& self,
                              int qlen, int top_k,
                              py::object expert_ids, py::object weights,
                              py::object input, py::object output) {
            self.forward_many(qlen, top_k,
                              as_u32(expert_ids),
                              as_f32(weights),
                              as_u16(input),
                              as_f32m(output));
        }, py::arg("qlen"), py::arg("top_k"), py::arg("expert_ids"),
           py::arg("weights"), py::arg("input"), py::arg("output"))
        .def("debug_first", [](MOE& self, int n) {
            return py::make_tuple(
                py::array(py::buffer_info(
                    const_cast<uint16_t*>(static_cast<const uint16_t*>(self.debug_w13())), sizeof(uint16_t),
                    py::format_descriptor<uint16_t>::format(), 1, { (size_t)n }, { sizeof(uint16_t) })),
                py::array(py::buffer_info(
                    const_cast<uint16_t*>(static_cast<const uint16_t*>(self.debug_w2())), sizeof(uint16_t),
                    py::format_descriptor<uint16_t>::format(), 1, { (size_t)n }, { sizeof(uint16_t) })));
        });
}

// The exported module name is parameterized so one source can be compiled into
// several ISA-specific variants (avx2 / avx512_base / avx512_vnni / ...) that
// coexist on disk and are selected at runtime by the _dynamic_loader. The base
// (no-suffix) build keeps the canonical name `_xiaotu_moe_C`.
//
// Build with e.g.  -DXIAOTU_MOE_MODULE_NAME=_xiaotu_moe_C_avx512_vnni  to emit
// `_xiaotu_moe_C_avx512_vnni`. This must be a single token (PYBIND11_MODULE
// token-pastes its first argument with the init-function name).
#ifndef XIAOTU_MOE_MODULE_NAME
#define XIAOTU_MOE_MODULE_NAME _xiaotu_moe_C
#endif

PYBIND11_MODULE(XIAOTU_MOE_MODULE_NAME, m) {
    m.doc() = "xiaotu-moe CPU MoE engine (BF16 + FP8 closed loops)";
    maybe_install_sigsegv_handler();

    py::class_<MOEConfigV2>(m, "MOEConfigV2")
        .def(py::init<>())
        .def_readwrite("num_processes", &MOEConfigV2::num_processes)
        .def_readwrite("process_id", &MOEConfigV2::process_id)
        .def_readwrite("gpu_id", &MOEConfigV2::gpu_id)
        .def_readwrite("has_gate_proj", &MOEConfigV2::has_gate_proj)
        .def_readwrite("expert_num", &MOEConfigV2::expert_num)
        .def_readwrite("top_k", &MOEConfigV2::top_k)
        .def_readwrite("hidden_size", &MOEConfigV2::hidden_size)
        .def_readwrite("intermediate_size", &MOEConfigV2::intermediate_size)
        .def_readwrite("max_batch_size", &MOEConfigV2::max_batch_size)
        .def_readwrite("max_num_seqs", &MOEConfigV2::max_num_seqs)
        .def_readwrite("stride", &MOEConfigV2::stride)
        .def_readwrite("group_min_len", &MOEConfigV2::group_min_len)
        .def_readwrite("group_max_len", &MOEConfigV2::group_max_len)
        .def_readwrite("groupN", &MOEConfigV2::groupN)
        .def_readwrite("groupK", &MOEConfigV2::groupK)
        .def_readwrite("swiglu_alpha", &MOEConfigV2::swiglu_alpha)
        .def_readwrite("swiglu_limit", &MOEConfigV2::swiglu_limit)
        .def_readwrite("activation_type", &MOEConfigV2::activation_type)
        .def_readwrite("use_gpu_prefill", &MOEConfigV2::use_gpu_prefill);

    bind_moe_class<BF16WeightTraits, BF16Activation>(m, "MOE_BF16");
    bind_moe_class<BF16WeightTraits, FP16Activation>(m, "MOE_FP16");
    // Backward-compatible alias from Phase 1 (real engine also exports MOE_BF16_FP16).
    // pybind11 forbids registering one C++ type twice, so alias the class object.
    m.attr("MOE_BF16_FP16") = m.attr("MOE_FP16");

    bind_moe_class<FP8WeightTraits, BF16Activation>(m, "MOE_FP8");
    bind_moe_class<FP8WeightTraits, FP16Activation>(m, "MOE_FP8_FP16");

    bind_moe_class<MXFP4WeightTraits, BF16Activation>(m, "MOE_MXFP4");
    bind_moe_class<MXFP4WeightTraits, FP16Activation>(m, "MOE_MXFP4_FP16");
    bind_moe_class<WNA16WeightTraits, BF16Activation>(m, "MOE_WNA16");
    bind_moe_class<WNA16WeightTraits, FP16Activation>(m, "MOE_WNA16_FP16");

    bind_moe_class<NVFP4WeightTraits, BF16Activation>(m, "MOE_NVFP4");
    bind_moe_class<NVFP4WeightTraits, FP16Activation>(m, "MOE_NVFP4_FP16");
}
