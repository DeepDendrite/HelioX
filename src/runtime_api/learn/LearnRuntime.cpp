#include "runtime_api/learn/LearnRuntime.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include "cuda_utils.h"

namespace {

static uint64_t fnv1a64_update_u64(uint64_t h, uint64_t v) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (int i = 0; i < 8; ++i) {
        const uint8_t b = static_cast<uint8_t>((v >> (i * 8)) & 0xFFU);
        h ^= static_cast<uint64_t>(b);
        h *= kPrime;
    }
    return h;
}

static uint64_t signature_i32_array(const int32_t* data, int len) {
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a64_update_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(len)));
    for (int i = 0; i < len; ++i) {
        h = fnv1a64_update_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(data[i])));
    }
    return h;
}

} // namespace

// CUDA batch kernels (compiled into gpulib).
extern "C" void launch_batch_copy_doubles(double** d_gpu_ptrs, const double* d_cpu_values, int count);
extern "C" void launch_batch_gather_floats(double** d_gpu_ptrs, float* d_out, int count);
extern "C" void launch_batch_scatter_add_f32(const float* d_values,
                                             const int32_t* d_dest,
                                             const float* d_scale,
                                             float* d_out,
                                             int count);

// CUDA replay kernels (compiled into gpulib, implemented in runtime_api/learn/kernels/replay_dw_dx.cu).
extern "C" void launch_replay_base_block_kernel(const float* it_lr,
                                                int N,
                                                int T,
                                                int t_lr,
                                                int t_window,
                                                const float* K_block,
                                                int bn,
                                                int K_len,
                                                int start,
                                                float dt,
                                                float grad_scale,
                                                float* dvtdw);
extern "C" void launch_replay_base_blocks_kernel(const float* it_lr,
                                                 int N,
                                                 int T,
                                                 int t_lr,
                                                 int t_window,
                                                 float** K_blocks,
                                                 const int* block_starts,
                                                 const int* block_bn,
                                                 const int* block_elem_off,
                                                 int n_blocks,
                                                 int total_elems,
                                                 int K_len,
                                                 float dt,
                                                 float grad_scale,
                                                 float* dvtdw);
extern "C" void launch_replay_base_block_ring_kernel(const float* it_ring,
                                                     const int* sig_win_idx,
                                                     int N,
                                                     int t_window,
                                                     const float* K_block,
                                                     int bn,
                                                     int K_len,
                                                     int start,
                                                     float dt,
                                                     float grad_scale,
                                                     float* dvtdw);
extern "C" void launch_replay_corr_allcols_kernel(const float* dV_hist,
                                                  const float* ditdv_lr,
                                                  const float* ditdvpre_lr,
                                                  const int* dV_win_idx,
                                                  int t_window,
                                                  int N,
                                                  int T,
                                                  int t_lr,
                                                  const int32_t* pre_of_col,
                                                  float** K_blocks,
                                                  const int* block_starts,
                                                  const int* block_bn,
                                                  const int32_t* col_block_id,
                                                  int K_len,
                                                  float dt,
                                                  float* dvtdw);
extern "C" void launch_replay_corr_allcols_ring_kernel(const float* dV_hist,
                                                       const float* ditdv_ring,
                                                       const float* ditdvpre_ring,
                                                       const int* dV_win_idx,
                                                       const int* sig_win_idx,
                                                       int t_window,
                                                       int N,
                                                       const int32_t* pre_of_col,
                                                       float** K_blocks,
                                                       const int* block_starts,
                                                       const int* block_bn,
                                                       const int32_t* col_block_id,
                                                       int K_len,
                                                       float dt,
                                                       float* dvtdw);
extern "C" void launch_replay_base_block_tmajor_kernel(const float* it_tn,
                                                       int N,
                                                       int T,
                                                       int t_lr,
                                                       int t_window,
                                                       const float* K_block,
                                                       int bn,
                                                       int K_len,
                                                       int start,
                                                       float dt,
                                                       float grad_scale,
                                                       float* dvtdw);
extern "C" void launch_replay_corr_allcols_tmajor_kernel(const float* dV_hist,
                                                         const float* ditdv_tn,
                                                         const float* ditdvpre_tn,
                                                         const int* dV_win_idx,
                                                         int t_window,
                                                         int N,
                                                         int T,
                                                         int t_lr,
                                                         const int32_t* pre_of_col,
                                                         float** K_blocks,
                                                         const int* block_starts,
                                                         const int* block_bn,
                                                         const int32_t* col_block_id,
                                                         int K_len,
                                                         float dt,
                                                         float* dvtdw);
extern "C" void launch_replay_compute_w_tick_kernel(const float* dvtdw,
                                                    int N,
                                                    const int32_t* poutput,
                                                    int n_output,
                                                    const float* dLtdv,
                                                    float* w_tick,
                                                    float* dw_accum);
extern "C" void launch_replay_dx_kernel(const float* dvtdw,
                                        int N,
                                        const int32_t* pinput,
                                        int n_input,
                                        const float* w_tick,
                                        float eps,
                                        int t_lr,
                                        int T_lr,
                                        float* dx_out,
                                        float* s_tmp,
                                        float* b_tmp);
extern "C" void launch_replay_scale_f32(float* data, size_t n, float scale);
extern "C" void launch_replay_nan_to_num_f32(float* data, size_t n);
extern "C" int launch_replay_frobenius_f32(const float* data,
                                           size_t n,
                                           double* partial_sums,
                                           int* partial_flags,
                                           double* out_sum,
                                           int* out_flag);
extern "C" void launch_transpose_tn_to_nt_f32(const float* src_tn, float* dst_nt, int T, int N);

namespace neurong::runtime_api::learn {

LearnRuntime::LearnRuntime(core::SimRuntimeCore& core) : core_(core) {}

LearnRuntime::~LearnRuntime() {
    clear_dense_blocks();
}

bool LearnRuntime::get_cached_pointers_or_print_(int handle, double*& cpu_ptr, double*& gpu_ptr) const {
    cpu_ptr = nullptr;
    gpu_ptr = nullptr;
    if (!core_.get_cached_pointers(handle, cpu_ptr, gpu_ptr)) {
        printf("Invalid handle: %d\n", handle);
        return false;
    }
    return true;
}

bool LearnRuntime::resolve_replay_dt_ms_(const char* fn_name, double dt_ms, double& out_dt_ms) const {
    Simulate* sim = core_.sim();
    if (sim == nullptr) {
        printf("%s: Simulate not initialized\n", fn_name);
        return false;
    }
    double effective_dt_ms = dt_ms;
    if (!(effective_dt_ms > 0.0)) {
        effective_dt_ms = sim->dt;
    }
    if (!(effective_dt_ms > 0.0) || !std::isfinite(effective_dt_ms)) {
        printf("%s: invalid dt (dt_ms=%f fallback_sim_dt=%f)\n", fn_name, dt_ms, sim->dt);
        return false;
    }
    out_dt_ms = effective_dt_ms;
    return true;
}

bool LearnRuntime::validate_replay_metadata_once_(const char* fn_name,
                                                  int N,
                                                  int n_output,
                                                  const int32_t* poutput,
                                                  int n_input,
                                                  const int32_t* pinput,
                                                  const int32_t* pre_of_col,
                                                  bool need_dx) {
    uint64_t signature = 1469598103934665603ULL;
    signature = fnv1a64_update_u64(signature, static_cast<uint64_t>(static_cast<uint32_t>(N)));
    signature = fnv1a64_update_u64(signature, static_cast<uint64_t>(static_cast<uint32_t>(n_output)));
    signature = fnv1a64_update_u64(signature, static_cast<uint64_t>(static_cast<uint32_t>(n_input)));
    signature = fnv1a64_update_u64(signature, static_cast<uint64_t>(need_dx ? 1 : 0));
    signature ^= signature_i32_array(poutput, n_output);
    signature ^= signature_i32_array(pre_of_col, N);
    if (need_dx && n_input > 0) {
        signature ^= signature_i32_array(pinput, n_input);
    }

    if (replay_metadata_cache_valid_ && replay_meta_N_ == N && replay_meta_n_output_ == n_output &&
        replay_meta_n_input_ == n_input && replay_meta_need_dx_ == need_dx &&
        replay_meta_signature_ == signature) {
        return true;
    }

    for (int i = 0; i < n_output; ++i) {
        const int idx = poutput[i];
        if (idx < 0 || idx >= N) {
            printf("%s: poutput out of range at %d (%d)\n", fn_name, i, idx);
            return false;
        }
    }
    if (need_dx) {
        for (int i = 0; i < n_input; ++i) {
            const int idx = pinput[i];
            if (idx < 0 || idx >= N) {
                printf("%s: pinput out of range at %d (%d)\n", fn_name, i, idx);
                return false;
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        const int idx = pre_of_col[i];
        if (idx != -1 && (idx < 0 || idx >= N)) {
            printf("%s: pre_of_col out of range at %d (%d)\n", fn_name, i, idx);
            return false;
        }
    }

    replay_metadata_cache_valid_ = true;
    replay_meta_N_ = N;
    replay_meta_n_output_ = n_output;
    replay_meta_n_input_ = n_input;
    replay_meta_need_dx_ = need_dx;
    replay_meta_signature_ = signature;
    return true;
}

void LearnRuntime::invalidate_replay_metadata_cache_() {
    replay_metadata_cache_valid_ = false;
    replay_meta_N_ = 0;
    replay_meta_n_output_ = 0;
    replay_meta_n_input_ = 0;
    replay_meta_need_dx_ = false;
    replay_meta_signature_ = 0;
}

int LearnRuntime::set_dense_blocks_f32(std::span<const DenseBlockHostView> blocks) {
    clear_dense_blocks();
    if (blocks.empty()) {
        dense_block_k_len_ = 0;
        return 0;
    }
    Simulate* sim = core_.sim();
    if (sim == nullptr) {
        printf("set_dense_blocks_f32: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("set_dense_blocks_f32: only supported on GPU\n");
        return -1;
    }

    int k_len = -1;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const DenseBlockHostView& b = blocks[i];
        if (b.data == nullptr || b.bn <= 0 || b.k_len <= 0) {
            printf("set_dense_blocks_f32: invalid block at %zu\n", i);
            clear_dense_blocks();
            return -1;
        }
        if (k_len < 0) {
            k_len = b.k_len;
        } else if (b.k_len != k_len) {
            printf("set_dense_blocks_f32: K_len mismatch in block %zu (%d vs %d)\n", i, b.k_len, k_len);
            clear_dense_blocks();
            return -1;
        }

        float* d_block = nullptr;
        const size_t bytes = static_cast<size_t>(b.bn) * static_cast<size_t>(b.bn) * static_cast<size_t>(b.k_len) * sizeof(float);
        if (bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
            printf("set_dense_blocks_f32: block too large\n");
            clear_dense_blocks();
            return -1;
        }
        gpu_mem_allocate((void**)&d_block, bytes);
        mem_copy_cpu2gpu_sync(d_block, b.data, static_cast<int>(bytes));
        dense_blocks_device_.push_back(d_block);
        dense_block_bn_.push_back(b.bn);
    }

    dense_block_k_len_ = k_len;
    return dense_block_k_len_;
}

void LearnRuntime::clear_dense_blocks() {
    clear_replay_dw_dx_buffers_();
    clear_capture_signal_buffers_();
    for (float* ptr : dense_blocks_device_) {
        if (ptr != nullptr) {
            gpu_mem_free((void**)&ptr);
        }
    }
    dense_blocks_device_.clear();
    dense_block_bn_.clear();
    dense_block_k_len_ = 0;
}

int LearnRuntime::simulate_output_vs_into(float* output_vs_tn,
                                         int t_steps_plus1,
                                         int n_output,
                                         std::span<const int> output_v_handles,
                                         double tstop_ms,
                                         double v_init) {
    if (output_vs_tn == nullptr) {
        printf("simulate_output_vs_into: output buffer is null\n");
        return -1;
    }
    Simulate* sim = core_.sim();
    if (sim == nullptr) {
        printf("simulate_output_vs_into: Simulate not initialized\n");
        return -1;
    }
    if (sim->dt <= 0.0) {
        printf("simulate_output_vs_into: invalid dt\n");
        return -1;
    }
    if (n_output <= 0 || static_cast<int>(output_v_handles.size()) != n_output) {
        printf("simulate_output_vs_into: output handles mismatch\n");
        return -1;
    }
    if (t_steps_plus1 < 2) {
        printf("simulate_output_vs_into: invalid steps\n");
        return -1;
    }
    const int total_steps = t_steps_plus1 - 1;
    const double tstop_from_steps = static_cast<double>(total_steps) * sim->dt;
    if (tstop_ms > 0.0 && std::abs(tstop_ms - tstop_from_steps) > (sim->dt * 0.5)) {
        printf("simulate_output_vs_into: tstop mismatch (tstop_ms=%f, steps*dt=%f)\n",
               tstop_ms, tstop_from_steps);
        return -1;
    }

    // Ensure any pending CPU-side writes are visible before running.
    core_.flush_dirty_variables();

    if (sim->mode == CPU) {
        sim->finitialize(v_init);
        for (int o = 0; o < n_output; ++o) {
            double* cpu_ptr = nullptr;
            double* gpu_ptr = nullptr;
            if (!get_cached_pointers_or_print_(output_v_handles[o], cpu_ptr, gpu_ptr) || cpu_ptr == nullptr) {
                output_vs_tn[o] = 0.0f;
            } else {
                output_vs_tn[o] = static_cast<float>(*cpu_ptr);
            }
        }
        for (int step = 0; step < total_steps; ++step) {
            sim->fadvance();
            const int t_idx = step + 1;
            float* row = output_vs_tn + static_cast<size_t>(t_idx) * static_cast<size_t>(n_output);
            for (int o = 0; o < n_output; ++o) {
                double* cpu_ptr = nullptr;
                double* gpu_ptr = nullptr;
                if (!get_cached_pointers_or_print_(output_v_handles[o], cpu_ptr, gpu_ptr) || cpu_ptr == nullptr) {
                    row[o] = 0.0f;
                } else {
                    row[o] = static_cast<float>(*cpu_ptr);
                }
            }
        }
        return 0;
    }

    if (sim->mode != GPU) {
        printf("simulate_output_vs_into: unsupported mode\n");
        return -1;
    }

    float* d_output_vs = nullptr;
    double** d_output_ptrs = nullptr;
    gpu_mem_allocate((void**)&d_output_vs,
                     static_cast<int>(static_cast<size_t>(t_steps_plus1) * static_cast<size_t>(n_output) * sizeof(float)));
    gpu_mem_allocate((void**)&d_output_ptrs, n_output * static_cast<int>(sizeof(double*)));

    auto cleanup = [&]() {
        if (d_output_vs != nullptr) gpu_mem_free((void**)&d_output_vs);
        if (d_output_ptrs != nullptr) gpu_mem_free((void**)&d_output_ptrs);
        d_output_vs = nullptr;
        d_output_ptrs = nullptr;
    };

    std::vector<double*> ptrs_host(n_output);
    for (int i = 0; i < n_output; ++i) {
        double* cpu_ptr = nullptr;
        double* gpu_ptr = nullptr;
        if (!get_cached_pointers_or_print_(output_v_handles[i], cpu_ptr, gpu_ptr) || gpu_ptr == nullptr) {
            printf("simulate_output_vs_into: invalid output handle\n");
            cleanup();
            return -1;
        }
        ptrs_host[i] = gpu_ptr;
    }
    mem_copy_cpu2gpu_sync(d_output_ptrs, ptrs_host.data(), n_output * static_cast<int>(sizeof(double*)));

    sim->finitialize(v_init);
    launch_batch_gather_floats(d_output_ptrs, d_output_vs, n_output);
    for (int step = 0; step < total_steps; ++step) {
        sim->fadvance();
        const int t_idx = step + 1;
        launch_batch_gather_floats(d_output_ptrs, d_output_vs + static_cast<size_t>(t_idx) * n_output, n_output);
    }

    mem_copy_gpu2cpu(output_vs_tn, d_output_vs,
                     static_cast<int>(static_cast<size_t>(t_steps_plus1) * static_cast<size_t>(n_output) * sizeof(float)));
    cleanup();
    return 0;
}

// ---- Not yet migrated: keep stubs for now (SimWrapper still owns old implementations) ----

int LearnRuntime::simulate_and_capture_mapped_signals_cached(float* output_vs_tn,
                                                            int total_steps_plus1,
                                                            int n_output,
                                                            std::span<const int> output_v_handles,
                                                            std::span<const int> pure_i_handles,
                                                            std::span<const int32_t> pure_i_dest,
                                                            std::span<const float> pure_i_scale,
                                                            std::span<const int> didv_handles,
                                                            std::span<const int32_t> didv_dest,
                                                            std::span<const float> didv_scale,
                                                            std::span<const int> didvpre_handles,
                                                            std::span<const int32_t> didvpre_dest,
                                                            std::span<const float> didvpre_scale,
                                                            double tstop_ms,
                                                            int k_mul,
                                                            bool percise,
                                                            double v_init) {
    Simulate* sim = core_.sim();
    if (sim == nullptr) {
        printf("simulate_and_capture_mapped_signals_cached: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("simulate_and_capture_mapped_signals_cached: only supported on GPU\n");
        return -1;
    }
    if (k_mul <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: k_mul must be positive\n");
        return -1;
    }
    if (sim->dt <= 0.0) {
        printf("simulate_and_capture_mapped_signals_cached: invalid dt\n");
        return -1;
    }
    if (output_vs_tn == nullptr) {
        printf("simulate_and_capture_mapped_signals_cached: output_vs_tn is null\n");
        return -1;
    }
    if (n_output <= 0 || static_cast<int>(output_v_handles.size()) != n_output) {
        printf("simulate_and_capture_mapped_signals_cached: output_v_handles size mismatch\n");
        return -1;
    }
    if (total_steps_plus1 < 2) {
        printf("simulate_and_capture_mapped_signals_cached: total_steps_plus1 must be >= 2\n");
        return -1;
    }
    const int total_steps = total_steps_plus1 - 1;
    const int ksteps_total = total_steps / k_mul;
    if (ksteps_total <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: ksteps_total must be positive\n");
        return -1;
    }

    // Infer N from mapping destinations.
    int N = 0;
    for (size_t i = 0; i < pure_i_dest.size(); ++i) {
        const int32_t d = pure_i_dest[i];
        if (d + 1 > N) {
            N = d + 1;
        }
    }
    if (N <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: inferred N is non-positive\n");
        return -1;
    }

    if (static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_dest.size()) ||
        static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_scale.size())) {
        printf("simulate_and_capture_mapped_signals_cached: pure_i handles/dest/scale size mismatch\n");
        return -1;
    }
    if (percise) {
        if (static_cast<int>(didv_handles.size()) != static_cast<int>(didv_dest.size()) ||
            static_cast<int>(didv_handles.size()) != static_cast<int>(didv_scale.size())) {
            printf("simulate_and_capture_mapped_signals_cached: didv handles/dest/scale size mismatch\n");
            return -1;
        }
        if (static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_dest.size()) ||
            static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_scale.size())) {
            printf("simulate_and_capture_mapped_signals_cached: didvpre handles/dest/scale size mismatch\n");
            return -1;
        }
    }

    // Validate destinations.
    for (size_t i = 0; i < pure_i_dest.size(); ++i) {
        const int32_t d = pure_i_dest[i];
        if (d < 0 || d >= N) {
            printf("simulate_and_capture_mapped_signals_cached: pure_i_dest out of range at %zu (%d)\n", i, d);
            return -1;
        }
    }
    if (percise) {
        for (size_t i = 0; i < didv_dest.size(); ++i) {
            const int32_t d = didv_dest[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_cached: didv_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
        for (size_t i = 0; i < didvpre_dest.size(); ++i) {
            const int32_t d = didvpre_dest[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_cached: didvpre_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
    }

    // tstop sanity check.
    const double tstop_from_steps = static_cast<double>(total_steps) * sim->dt;
    if (tstop_ms > 0.0 && std::abs(tstop_ms - tstop_from_steps) > (sim->dt * 0.5)) {
        printf("simulate_and_capture_mapped_signals_cached: tstop mismatch (tstop_ms=%f, steps*dt=%f)\n",
               tstop_ms, tstop_from_steps);
        return -1;
    }

    core_.flush_dirty_variables();

    const int T = ksteps_total + 1;
    const size_t cap_bytes = static_cast<size_t>(T) * static_cast<size_t>(N) * sizeof(float);
    if (cap_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("simulate_and_capture_mapped_signals_cached: capture buffer too large\n");
        return -1;
    }

    // Reconfigure capture buffers if needed.
    if (capture_N_ != N || capture_T_ != T || capture_k_mul_ != k_mul || capture_percise_ != percise) {
        clear_capture_signal_buffers_();
        capture_N_ = N;
        capture_T_ = T;
        capture_k_mul_ = k_mul;
        capture_percise_ = percise;

        gpu_mem_allocate((void**) &capture_it_tn_device_, static_cast<int>(cap_bytes));
        gpu_mem_allocate((void**) &capture_it_nt_device_, static_cast<int>(cap_bytes));
        capture_it_bytes_ = cap_bytes;
        capture_it_nt_bytes_ = cap_bytes;

        if (percise) {
            gpu_mem_allocate((void**) &capture_ditdv_tn_device_, static_cast<int>(cap_bytes));
            gpu_mem_allocate((void**) &capture_ditdvpre_tn_device_, static_cast<int>(cap_bytes));
            gpu_mem_allocate((void**) &capture_ditdv_nt_device_, static_cast<int>(cap_bytes));
            gpu_mem_allocate((void**) &capture_ditdvpre_nt_device_, static_cast<int>(cap_bytes));
            capture_ditdv_bytes_ = cap_bytes;
            capture_ditdvpre_bytes_ = cap_bytes;
            capture_ditdv_nt_bytes_ = cap_bytes;
            capture_ditdvpre_nt_bytes_ = cap_bytes;
        }
    }

    // Allocate device output buffer.
    float* d_output_vs = nullptr;
    const size_t out_bytes = static_cast<size_t>(total_steps_plus1) * static_cast<size_t>(n_output) * sizeof(float);
    if (out_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("simulate_and_capture_mapped_signals_cached: output buffer too large\n");
        return -1;
    }
    gpu_mem_allocate((void**) &d_output_vs, static_cast<int>(out_bytes));

    auto cleanup_output = [&]() {
        if (d_output_vs != nullptr) gpu_mem_free((void**) &d_output_vs);
        d_output_vs = nullptr;
    };

    // Pointer arrays on device for batched gather.
    double** d_output_ptrs = nullptr;
    double** d_pure_i_ptrs = nullptr;
    double** d_didv_ptrs = nullptr;
    double** d_didvpre_ptrs = nullptr;
    float* d_pure_i_values = nullptr;
    float* d_didv_values = nullptr;
    float* d_didvpre_values = nullptr;
    int32_t* d_pure_i_dest = nullptr;
    float* d_pure_i_scale = nullptr;
    int32_t* d_didv_dest = nullptr;
    float* d_didv_scale = nullptr;
    int32_t* d_didvpre_dest = nullptr;
    float* d_didvpre_scale = nullptr;

    auto cleanup_aux = [&]() {
        if (d_output_ptrs != nullptr) gpu_mem_free((void**) &d_output_ptrs);
        if (d_pure_i_ptrs != nullptr) gpu_mem_free((void**) &d_pure_i_ptrs);
        if (d_didv_ptrs != nullptr) gpu_mem_free((void**) &d_didv_ptrs);
        if (d_didvpre_ptrs != nullptr) gpu_mem_free((void**) &d_didvpre_ptrs);
        if (d_pure_i_values != nullptr) gpu_mem_free((void**) &d_pure_i_values);
        if (d_didv_values != nullptr) gpu_mem_free((void**) &d_didv_values);
        if (d_didvpre_values != nullptr) gpu_mem_free((void**) &d_didvpre_values);
        if (d_pure_i_dest != nullptr) gpu_mem_free((void**) &d_pure_i_dest);
        if (d_pure_i_scale != nullptr) gpu_mem_free((void**) &d_pure_i_scale);
        if (d_didv_dest != nullptr) gpu_mem_free((void**) &d_didv_dest);
        if (d_didv_scale != nullptr) gpu_mem_free((void**) &d_didv_scale);
        if (d_didvpre_dest != nullptr) gpu_mem_free((void**) &d_didvpre_dest);
        if (d_didvpre_scale != nullptr) gpu_mem_free((void**) &d_didvpre_scale);
        d_output_ptrs = nullptr;
        d_pure_i_ptrs = nullptr;
        d_didv_ptrs = nullptr;
        d_didvpre_ptrs = nullptr;
        d_pure_i_values = nullptr;
        d_didv_values = nullptr;
        d_didvpre_values = nullptr;
        d_pure_i_dest = nullptr;
        d_pure_i_scale = nullptr;
        d_didv_dest = nullptr;
        d_didv_scale = nullptr;
        d_didvpre_dest = nullptr;
        d_didvpre_scale = nullptr;
    };

    // Output pointers.
    {
        std::vector<double*> ptrs_host(static_cast<size_t>(n_output));
        for (int i = 0; i < n_output; ++i) {
            double* cpu_ptr = nullptr;
            double* gpu_ptr = nullptr;
            if (!get_cached_pointers_or_print_(output_v_handles[static_cast<size_t>(i)], cpu_ptr, gpu_ptr) || gpu_ptr == nullptr) {
                printf("simulate_and_capture_mapped_signals_cached: invalid output handle\n");
                cleanup_aux();
                cleanup_output();
                return -1;
            }
            ptrs_host[static_cast<size_t>(i)] = gpu_ptr;
        }
        gpu_mem_allocate((void**) &d_output_ptrs, n_output * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(d_output_ptrs, ptrs_host.data(), n_output * static_cast<int>(sizeof(double*)));
    }

    const int pure_i_count = static_cast<int>(pure_i_handles.size());
    if (pure_i_count > 0) {
        std::vector<double*> ptrs_host(static_cast<size_t>(pure_i_count));
        for (int i = 0; i < pure_i_count; ++i) {
            double* cpu_ptr = nullptr;
            double* gpu_ptr = nullptr;
            if (!get_cached_pointers_or_print_(pure_i_handles[static_cast<size_t>(i)], cpu_ptr, gpu_ptr) || gpu_ptr == nullptr) {
                printf("simulate_and_capture_mapped_signals_cached: invalid pure_i handle\n");
                cleanup_aux();
                cleanup_output();
                return -1;
            }
            ptrs_host[static_cast<size_t>(i)] = gpu_ptr;
        }
        gpu_mem_allocate((void**) &d_pure_i_ptrs, pure_i_count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(d_pure_i_ptrs, ptrs_host.data(), pure_i_count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**) &d_pure_i_values, pure_i_count * static_cast<int>(sizeof(float)));
        gpu_mem_allocate((void**) &d_pure_i_dest, pure_i_count * static_cast<int>(sizeof(int32_t)));
        gpu_mem_allocate((void**) &d_pure_i_scale, pure_i_count * static_cast<int>(sizeof(float)));
        mem_copy_cpu2gpu_sync(d_pure_i_dest, pure_i_dest.data(), pure_i_count * static_cast<int>(sizeof(int32_t)));
        mem_copy_cpu2gpu_sync(d_pure_i_scale, pure_i_scale.data(), pure_i_count * static_cast<int>(sizeof(float)));
    }

    const int didv_count = percise ? static_cast<int>(didv_handles.size()) : 0;
    if (percise && didv_count > 0) {
        std::vector<double*> ptrs_host(static_cast<size_t>(didv_count));
        for (int i = 0; i < didv_count; ++i) {
            double* cpu_ptr = nullptr;
            double* gpu_ptr = nullptr;
            if (!get_cached_pointers_or_print_(didv_handles[static_cast<size_t>(i)], cpu_ptr, gpu_ptr) || gpu_ptr == nullptr) {
                printf("simulate_and_capture_mapped_signals_cached: invalid didv handle\n");
                cleanup_aux();
                cleanup_output();
                return -1;
            }
            ptrs_host[static_cast<size_t>(i)] = gpu_ptr;
        }
        gpu_mem_allocate((void**) &d_didv_ptrs, didv_count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(d_didv_ptrs, ptrs_host.data(), didv_count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**) &d_didv_values, didv_count * static_cast<int>(sizeof(float)));
        gpu_mem_allocate((void**) &d_didv_dest, didv_count * static_cast<int>(sizeof(int32_t)));
        gpu_mem_allocate((void**) &d_didv_scale, didv_count * static_cast<int>(sizeof(float)));
        mem_copy_cpu2gpu_sync(d_didv_dest, didv_dest.data(), didv_count * static_cast<int>(sizeof(int32_t)));
        mem_copy_cpu2gpu_sync(d_didv_scale, didv_scale.data(), didv_count * static_cast<int>(sizeof(float)));
    }

    const int didvpre_count = percise ? static_cast<int>(didvpre_handles.size()) : 0;
    if (percise && didvpre_count > 0) {
        std::vector<double*> ptrs_host(static_cast<size_t>(didvpre_count));
        for (int i = 0; i < didvpre_count; ++i) {
            double* cpu_ptr = nullptr;
            double* gpu_ptr = nullptr;
            if (!get_cached_pointers_or_print_(didvpre_handles[static_cast<size_t>(i)], cpu_ptr, gpu_ptr) || gpu_ptr == nullptr) {
                printf("simulate_and_capture_mapped_signals_cached: invalid didvpre handle\n");
                cleanup_aux();
                cleanup_output();
                return -1;
            }
            ptrs_host[static_cast<size_t>(i)] = gpu_ptr;
        }
        gpu_mem_allocate((void**) &d_didvpre_ptrs, didvpre_count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(d_didvpre_ptrs, ptrs_host.data(), didvpre_count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**) &d_didvpre_values, didvpre_count * static_cast<int>(sizeof(float)));
        gpu_mem_allocate((void**) &d_didvpre_dest, didvpre_count * static_cast<int>(sizeof(int32_t)));
        gpu_mem_allocate((void**) &d_didvpre_scale, didvpre_count * static_cast<int>(sizeof(float)));
        mem_copy_cpu2gpu_sync(d_didvpre_dest, didvpre_dest.data(), didvpre_count * static_cast<int>(sizeof(int32_t)));
        mem_copy_cpu2gpu_sync(d_didvpre_scale, didvpre_scale.data(), didvpre_count * static_cast<int>(sizeof(float)));
    }

    sim->finitialize(v_init);
    launch_batch_gather_floats(d_output_ptrs, d_output_vs, n_output);

    cudaMemset(capture_it_tn_device_, 0, cap_bytes);
    if (percise) {
        cudaMemset(capture_ditdv_tn_device_, 0, cap_bytes);
        cudaMemset(capture_ditdvpre_tn_device_, 0, cap_bytes);
    }

    for (int step = 0; step < total_steps; ++step) {
        sim->fadvance();
        const int t_idx = step + 1;
        launch_batch_gather_floats(d_output_ptrs, d_output_vs + static_cast<size_t>(t_idx) * static_cast<size_t>(n_output), n_output);

        if ((t_idx % k_mul) == 0) {
            const int lr_idx = t_idx / k_mul;
            float* it_col = capture_it_tn_device_ + static_cast<size_t>(lr_idx) * static_cast<size_t>(N);
            if (pure_i_count > 0) {
                launch_batch_gather_floats(d_pure_i_ptrs, d_pure_i_values, pure_i_count);
                launch_batch_scatter_add_f32(d_pure_i_values, d_pure_i_dest, d_pure_i_scale, it_col, pure_i_count);
            }

            if (percise) {
                float* didv_col = capture_ditdv_tn_device_ + static_cast<size_t>(lr_idx) * static_cast<size_t>(N);
                float* didvpre_col = capture_ditdvpre_tn_device_ + static_cast<size_t>(lr_idx) * static_cast<size_t>(N);
                if (didv_count > 0) {
                    launch_batch_gather_floats(d_didv_ptrs, d_didv_values, didv_count);
                    launch_batch_scatter_add_f32(d_didv_values, d_didv_dest, d_didv_scale, didv_col, didv_count);
                }
                if (didvpre_count > 0) {
                    launch_batch_gather_floats(d_didvpre_ptrs, d_didvpre_values, didvpre_count);
                    launch_batch_scatter_add_f32(d_didvpre_values, d_didvpre_dest, d_didvpre_scale, didvpre_col, didvpre_count);
                }
            }
        }
    }

    // Convert to replay-friendly row-major (N,T) layout.
    launch_transpose_tn_to_nt_f32(capture_it_tn_device_, capture_it_nt_device_, capture_T_, capture_N_);
    if (percise) {
        launch_transpose_tn_to_nt_f32(capture_ditdv_tn_device_, capture_ditdv_nt_device_, capture_T_, capture_N_);
        launch_transpose_tn_to_nt_f32(capture_ditdvpre_tn_device_, capture_ditdvpre_nt_device_, capture_T_, capture_N_);
    }

    mem_copy_gpu2cpu(output_vs_tn, d_output_vs, static_cast<int>(out_bytes));

    cleanup_aux();
    cleanup_output();
    return 0;
}

int LearnRuntime::replay_compute_dw_dx_from_cached_signals_impl_(const float* dLtdv_lr_to,
                                                                int ksteps_total,
                                                                int n_output,
                                                                const int32_t* poutput,
                                                                const int32_t* pinput,
                                                                int n_input,
                                                                const int32_t* pre_of_col,
                                                                float* dw_out_n,
                                                                float* dx_lr_it,
                                                                double dt_ms,
                                                                bool percise,
                                                                double grad_scale,
                                                                double eps,
                                                                double grad_l2norm_threshold,
                                                                int clip_strategy,
                                                                int clip_check_every,
                                                                bool need_dx) {
    const char* fn_name =
        need_dx ? "replay_compute_dw_dx_from_cached_signals_into" : "replay_compute_dw_from_cached_signals_into";
    Simulate* sim = core_.sim();
    if (sim == nullptr) {
        printf("%s: Simulate not initialized\n", fn_name);
        return -1;
    }
    if (sim->mode != GPU) {
        printf("%s: only supported on GPU\n", fn_name);
        return -1;
    }
    if (dense_blocks_device_.empty() || dense_block_k_len_ <= 0) {
        printf("%s: dense blocks not set (call set_dense_blocks_f32)\n", fn_name);
        return -1;
    }
    if (capture_it_nt_device_ == nullptr || capture_N_ <= 0 || capture_T_ < 2) {
        printf("%s: cached capture buffers not initialized\n", fn_name);
        return -1;
    }
    if (percise && (!capture_percise_ || capture_ditdv_nt_device_ == nullptr || capture_ditdvpre_nt_device_ == nullptr)) {
        printf("%s: percise requested but cached didv buffers missing\n", fn_name);
        return -1;
    }
    if (dLtdv_lr_to == nullptr || poutput == nullptr || pre_of_col == nullptr || dw_out_n == nullptr ||
        (need_dx && (pinput == nullptr || dx_lr_it == nullptr))) {
        printf("%s: null input/output pointer\n", fn_name);
        return -1;
    }

    const int N = capture_N_;
    const int T = capture_T_;
    const int K_len = dense_block_k_len_;
    if (ksteps_total != (T - 1)) {
        printf("%s: ksteps_total mismatch (%d vs %d)\n", fn_name, ksteps_total, T - 1);
        return -1;
    }
    if (n_output <= 0 || (need_dx && n_input <= 0)) {
        printf("%s: empty poutput/pinput\n", fn_name);
        return -1;
    }
    double effective_dt_ms = 0.0;
    if (!resolve_replay_dt_ms_(fn_name, dt_ms, effective_dt_ms)) {
        return -1;
    }
    if (!validate_replay_metadata_once_(fn_name,
                                        N,
                                        n_output,
                                        poutput,
                                        n_input,
                                        pinput,
                                        pre_of_col,
                                        need_dx)) {
        return -1;
    }


    auto ensure_buf = [&](void** p, size_t* cur_bytes, size_t want_bytes) -> bool {
        if (want_bytes == 0) {
            return true;
        }
        if (*p != nullptr && *cur_bytes >= want_bytes) {
            return true;
        }
        if (*p != nullptr) {
            gpu_mem_free((void**) p);
            *p = nullptr;
            *cur_bytes = 0;
        }
        gpu_mem_allocate(p, want_bytes);
        if (*p == nullptr) {
            return false;
        }
        *cur_bytes = want_bytes;
        return true;
    };

    const size_t dLtdv_bytes = static_cast<size_t>(ksteps_total) * static_cast<size_t>(n_output) * sizeof(float);
    const size_t dvtdw_elems = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t dvtdw_bytes = dvtdw_elems * sizeof(float);
    const size_t dV_hist_bytes = static_cast<size_t>(K_len) * dvtdw_bytes;
    const size_t w_tick_bytes = static_cast<size_t>(N) * sizeof(float);
    const size_t dw_accum_bytes = static_cast<size_t>(N) * sizeof(float);
    const size_t dx_bytes = static_cast<size_t>(n_input) * static_cast<size_t>(ksteps_total) * sizeof(float);
    const size_t sb_bytes = static_cast<size_t>(n_input) * sizeof(float);
    const size_t idx_bytes = static_cast<size_t>(K_len) * sizeof(int);


    if (!ensure_buf((void**) &replay_dLtdv_device_, &replay_dLtdv_bytes_, dLtdv_bytes) ||
        !ensure_buf((void**) &replay_dvtdw_device_, &replay_dvtdw_bytes_, dvtdw_bytes) ||
        !ensure_buf((void**) &replay_dV_hist_device_, &replay_dV_hist_bytes_, dV_hist_bytes) ||
        !ensure_buf((void**) &replay_w_tick_device_, &replay_w_tick_bytes_, w_tick_bytes) ||
        !ensure_buf((void**) &replay_dw_accum_device_, &replay_dw_accum_bytes_, dw_accum_bytes) ||
        !ensure_buf((void**) &replay_dx_device_, &replay_dx_bytes_, dx_bytes) ||
        !ensure_buf((void**) &replay_s_tmp_device_, &replay_s_tmp_bytes_, sb_bytes) ||
        !ensure_buf((void**) &replay_b_tmp_device_, &replay_b_tmp_bytes_, sb_bytes) ||
        !ensure_buf((void**) &replay_dV_win_idx_device_, &replay_idx_bytes_, idx_bytes)) {
        printf("%s: allocation failed (workspace)\n", fn_name);
        return -1;
    }

    if (clip_strategy != 0) {
        constexpr size_t kClipThreads = 256;
        const int n_partials = static_cast<int>((dvtdw_elems + kClipThreads - 1) / kClipThreads);
        const size_t partial_sums_bytes = static_cast<size_t>(n_partials) * sizeof(double);
        const size_t partial_flags_bytes = static_cast<size_t>(n_partials) * sizeof(int);
        if (!ensure_buf((void**) &replay_norm_partial_sums_device_, &replay_norm_partial_sums_bytes_, partial_sums_bytes) ||
            !ensure_buf((void**) &replay_norm_partial_flags_device_, &replay_norm_partial_flags_bytes_, partial_flags_bytes) ||
            !ensure_buf((void**) &replay_norm_sum_device_, &replay_norm_sum_bytes_, sizeof(double)) ||
            !ensure_buf((void**) &replay_norm_flag_device_, &replay_norm_flag_bytes_, sizeof(int))) {
            printf("%s: allocation failed (norm clip buffers)\n", fn_name);
            return -1;
        }
    }

    const int n_blocks = static_cast<int>(dense_blocks_device_.size());
    if (replay_N_ != N || replay_K_len_ != K_len || replay_n_blocks_ != n_blocks) {
        const size_t starts_bytes = static_cast<size_t>(n_blocks) * sizeof(int);
        const size_t bn_bytes = static_cast<size_t>(n_blocks) * sizeof(int);
        const size_t elem_off_bytes = static_cast<size_t>(n_blocks + 1) * sizeof(int);
        const size_t col_id_bytes = static_cast<size_t>(N) * sizeof(int32_t);
        const size_t kptr_bytes = static_cast<size_t>(n_blocks) * sizeof(float*);
        if (!ensure_buf((void**) &replay_block_starts_device_, &replay_block_starts_bytes_, starts_bytes) ||
            !ensure_buf((void**) &replay_block_bn_device_, &replay_block_bn_bytes_, bn_bytes) ||
            !ensure_buf((void**) &replay_block_elem_off_device_, &replay_block_elem_off_bytes_, elem_off_bytes) ||
            !ensure_buf((void**) &replay_col_block_id_device_, &replay_col_block_id_bytes_, col_id_bytes) ||
            !ensure_buf((void**) &replay_K_blocks_ptrs_device_, &replay_K_blocks_ptrs_bytes_, kptr_bytes)) {
            printf("%s: allocation failed (block meta)\n", fn_name);
            return -1;
        }

        std::vector<int> block_starts_host(static_cast<size_t>(n_blocks));
        std::vector<int> block_bn_host(static_cast<size_t>(n_blocks));
        std::vector<int> block_elem_off_host(static_cast<size_t>(n_blocks) + 1);
        std::vector<int32_t> col_block_id_host(static_cast<size_t>(N));
        int start = 0;
        int elem_off = 0;
        block_elem_off_host[0] = 0;
        for (int b = 0; b < n_blocks; ++b) {
            const int bn = dense_block_bn_[static_cast<size_t>(b)];
            block_starts_host[static_cast<size_t>(b)] = start;
            block_bn_host[static_cast<size_t>(b)] = bn;
            elem_off += bn * bn;
            block_elem_off_host[static_cast<size_t>(b) + 1] = elem_off;
            for (int col = start; col < start + bn; ++col) {
                col_block_id_host[static_cast<size_t>(col)] = static_cast<int32_t>(b);
            }
            start += bn;
        }
        if (start != N) {
            printf("%s: internal block start mismatch\n", fn_name);
            return -1;
        }
        std::vector<float*> kptrs_host(static_cast<size_t>(n_blocks));
        for (int b = 0; b < n_blocks; ++b) {
            kptrs_host[static_cast<size_t>(b)] = dense_blocks_device_[static_cast<size_t>(b)];
        }

        mem_copy_cpu2gpu_sync(replay_block_starts_device_, block_starts_host.data(), static_cast<int>(starts_bytes));
        mem_copy_cpu2gpu_sync(replay_block_bn_device_, block_bn_host.data(), static_cast<int>(bn_bytes));
        mem_copy_cpu2gpu_sync(replay_block_elem_off_device_, block_elem_off_host.data(), static_cast<int>(elem_off_bytes));
        mem_copy_cpu2gpu_sync(replay_col_block_id_device_, col_block_id_host.data(), static_cast<int>(col_id_bytes));
        mem_copy_cpu2gpu_sync(replay_K_blocks_ptrs_device_, kptrs_host.data(), static_cast<int>(kptr_bytes));

        replay_N_ = N;
        replay_K_len_ = K_len;
        replay_n_blocks_ = n_blocks;
        replay_block_elem_total_ = elem_off;
    }

    mem_copy_cpu2gpu_sync(replay_dLtdv_device_, dLtdv_lr_to, static_cast<int>(dLtdv_bytes));

    int32_t* d_poutput = nullptr;
    int32_t* d_pinput = nullptr;
    int32_t* d_pre_of_col = nullptr;
    auto cleanup_small = [&]() {
        if (d_poutput) gpu_mem_free((void**) &d_poutput);
        if (d_pinput) gpu_mem_free((void**) &d_pinput);
        if (d_pre_of_col) gpu_mem_free((void**) &d_pre_of_col);
        d_poutput = nullptr;
        d_pinput = nullptr;
        d_pre_of_col = nullptr;
    };
    const size_t poutput_bytes = static_cast<size_t>(n_output) * sizeof(int32_t);
    const size_t pre_bytes = static_cast<size_t>(N) * sizeof(int32_t);
    gpu_mem_allocate((void**) &d_poutput, poutput_bytes);
    if (need_dx) {
        const size_t pinput_bytes = static_cast<size_t>(n_input) * sizeof(int32_t);
        gpu_mem_allocate((void**) &d_pinput, pinput_bytes);
        mem_copy_cpu2gpu_sync(d_pinput, pinput, static_cast<int>(pinput_bytes));
    }
    gpu_mem_allocate((void**) &d_pre_of_col, pre_bytes);
    mem_copy_cpu2gpu_sync(d_poutput, poutput, static_cast<int>(poutput_bytes));
    mem_copy_cpu2gpu_sync(d_pre_of_col, pre_of_col, static_cast<int>(pre_bytes));

    cudaMemset(replay_dV_hist_device_, 0, dV_hist_bytes);
    cudaMemset(replay_dw_accum_device_, 0, dw_accum_bytes);
    if (need_dx && dx_bytes > 0) {
        cudaMemset(replay_dx_device_, 0, dx_bytes);
    }

    const float dt = static_cast<float>(effective_dt_ms);
    float grad_scale_f = static_cast<float>(grad_scale);
    const float eps_f = static_cast<float>(eps);
    const float grad_l2_th_f = static_cast<float>(grad_l2norm_threshold);

    int dV_pos = 1;
    std::vector<int> dV_win_idx_host;
    dV_win_idx_host.reserve(static_cast<size_t>(K_len));

    for (int t_lr = 1; t_lr <= ksteps_total; ++t_lr) {
        const int t_window = (t_lr >= K_len) ? K_len : t_lr;
        float* dvtdw_cur = replay_dvtdw_device_;

        cudaMemset(dvtdw_cur, 0, dvtdw_bytes);
        launch_replay_base_blocks_kernel(capture_it_nt_device_, N, T, t_lr, t_window, replay_K_blocks_ptrs_device_,
                                         replay_block_starts_device_, replay_block_bn_device_, replay_block_elem_off_device_, n_blocks,
                                         replay_block_elem_total_, K_len, dt, grad_scale_f, dvtdw_cur);

        if (percise) {
            dV_win_idx_host.clear();
            for (int t = 0; t < t_window; ++t) {
                int idx = dV_pos - t_window + t;
                idx %= K_len;
                if (idx < 0) idx += K_len;
                dV_win_idx_host.push_back(idx);
            }
            mem_copy_cpu2gpu_sync(replay_dV_win_idx_device_, dV_win_idx_host.data(),
                                  static_cast<int>(static_cast<size_t>(t_window) * sizeof(int)));
            launch_replay_corr_allcols_kernel(replay_dV_hist_device_, capture_ditdv_nt_device_, capture_ditdvpre_nt_device_,
                                              replay_dV_win_idx_device_, t_window, N, T, t_lr, d_pre_of_col, replay_K_blocks_ptrs_device_,
                                              replay_block_starts_device_, replay_block_bn_device_, replay_col_block_id_device_, K_len, dt,
                                              dvtdw_cur);
        }

        if (clip_strategy != 0) {
            const bool do_check = (clip_strategy == 1) || ((t_lr % clip_check_every) == 0);
            if (do_check) {
                launch_replay_frobenius_f32(dvtdw_cur, dvtdw_elems, replay_norm_partial_sums_device_,
                                            replay_norm_partial_flags_device_, replay_norm_sum_device_, replay_norm_flag_device_);
                double sumsq = 0.0;
                int has_nonfinite = 0;
                mem_copy_gpu2cpu(&sumsq, replay_norm_sum_device_, static_cast<int>(sizeof(double)));
                mem_copy_gpu2cpu(&has_nonfinite, replay_norm_flag_device_, static_cast<int>(sizeof(int)));
                if (has_nonfinite) {
                    launch_replay_nan_to_num_f32(dvtdw_cur, dvtdw_elems);
                    launch_replay_frobenius_f32(dvtdw_cur, dvtdw_elems, replay_norm_partial_sums_device_,
                                                replay_norm_partial_flags_device_, replay_norm_sum_device_, replay_norm_flag_device_);
                    mem_copy_gpu2cpu(&sumsq, replay_norm_sum_device_, static_cast<int>(sizeof(double)));
                }
                const double l2 = sqrt(std::max(0.0, sumsq));
                if (l2 > static_cast<double>(grad_l2_th_f) && std::isfinite(l2)) {
                    const float scaler = static_cast<float>(static_cast<double>(grad_l2_th_f) / l2);
                    launch_replay_scale_f32(dvtdw_cur, dvtdw_elems, scaler);
                    launch_replay_scale_f32(replay_dV_hist_device_, static_cast<size_t>(K_len) * dvtdw_elems, scaler);
                    launch_replay_scale_f32(replay_dw_accum_device_, static_cast<size_t>(N), scaler);
                    launch_replay_scale_f32(replay_dx_device_, static_cast<size_t>(n_input) * static_cast<size_t>(ksteps_total), scaler);
                    grad_scale_f *= scaler;
                }
            }
        }

        const float* dLtdv_vec = replay_dLtdv_device_ + static_cast<size_t>(t_lr - 1) * static_cast<size_t>(n_output);
        launch_replay_compute_w_tick_kernel(dvtdw_cur, N, d_poutput, n_output, dLtdv_vec, replay_w_tick_device_,
                                            replay_dw_accum_device_);
        if (need_dx) {
            const int dx_t_idx = t_lr - 1;
            launch_replay_dx_kernel(dvtdw_cur, N, d_pinput, n_input, replay_w_tick_device_, eps_f, dx_t_idx, ksteps_total,
                                    replay_dx_device_, replay_s_tmp_device_, replay_b_tmp_device_);
        }

        float* dV_slot = replay_dV_hist_device_ +
                         static_cast<size_t>(dV_pos) * static_cast<size_t>(N) * static_cast<size_t>(N);
        mem_copy_gpu2gpu(dV_slot, dvtdw_cur, static_cast<int>(dvtdw_bytes), nullptr);
        dV_pos = (dV_pos + 1) % K_len;
    }

    mem_copy_gpu2cpu(dw_out_n, replay_dw_accum_device_, static_cast<int>(dw_accum_bytes));
    const float inv_t = 1.0f / static_cast<float>(ksteps_total);
    for (int i = 0; i < N; ++i) {
        dw_out_n[i] *= inv_t;
    }
    if (need_dx) {
        mem_copy_gpu2cpu(dx_lr_it, replay_dx_device_, static_cast<int>(dx_bytes));
    }

    cleanup_small();
    return 0;
}

int LearnRuntime::replay_compute_dw_dx_from_cached_signals_into(const float* dLtdv_lr_to,
                                                               int ksteps_total,
                                                               int n_output,
                                                               const int32_t* poutput,
                                                               const int32_t* pinput,
                                                               int n_input,
                                                               const int32_t* pre_of_col,
                                                               float* dw_out_n,
                                                               float* dx_lr_it,
                                                               double dt_ms,
                                                               bool percise,
                                                               double grad_scale,
                                                               double eps,
                                                               double grad_l2norm_threshold,
                                                               int clip_strategy,
                                                               int clip_check_every) {
    return replay_compute_dw_dx_from_cached_signals_impl_(dLtdv_lr_to,
                                                         ksteps_total,
                                                         n_output,
                                                         poutput,
                                                         pinput,
                                                         n_input,
                                                         pre_of_col,
                                                         dw_out_n,
                                                         dx_lr_it,
                                                         dt_ms,
                                                         percise,
                                                         grad_scale,
                                                         eps,
                                                         grad_l2norm_threshold,
                                                         clip_strategy,
                                                         clip_check_every,
                                                         true);
}

int LearnRuntime::replay_compute_dw_from_cached_signals_into(const float* dLtdv_lr_to,
                                                            int ksteps_total,
                                                            int n_output,
                                                            const int32_t* poutput,
                                                            const int32_t* pre_of_col,
                                                            float* dw_out_n,
                                                            double dt_ms,
                                                            bool percise,
                                                            double grad_scale,
                                                            double eps,
                                                            double grad_l2norm_threshold,
                                                            int clip_strategy,
                                                            int clip_check_every) {
    return replay_compute_dw_dx_from_cached_signals_impl_(dLtdv_lr_to,
                                                         ksteps_total,
                                                         n_output,
                                                         poutput,
                                                         nullptr,
                                                         0,
                                                         pre_of_col,
                                                         dw_out_n,
                                                         nullptr,
                                                         dt_ms,
                                                         percise,
                                                         grad_scale,
                                                         eps,
                                                         grad_l2norm_threshold,
                                                         clip_strategy,
                                                         clip_check_every,
                                                         false);
}

void LearnRuntime::clear_replay_dw_dx_buffers_() {
    invalidate_replay_metadata_cache_();
    auto free_ptr = [](auto** p) {
        if (*p != nullptr) {
            gpu_mem_free((void**) p);
            *p = nullptr;
        }
    };
    free_ptr(&replay_it_device_);
    free_ptr(&replay_ditdv_device_);
    free_ptr(&replay_ditdvpre_device_);
    free_ptr(&replay_dLtdv_device_);
    free_ptr(&replay_dvtdw_device_);
    free_ptr(&replay_dV_hist_device_);
    free_ptr(&replay_w_tick_device_);
    free_ptr(&replay_dw_accum_device_);
    free_ptr(&replay_dx_device_);
    free_ptr(&replay_s_tmp_device_);
    free_ptr(&replay_b_tmp_device_);
    free_ptr(&replay_dV_win_idx_device_);
    free_ptr(&replay_sig_win_idx_device_);
    free_ptr(&replay_it_ring_device_);
    free_ptr(&replay_ditdv_ring_device_);
    free_ptr(&replay_ditdvpre_ring_device_);
    free_ptr(&replay_block_starts_device_);
    free_ptr(&replay_block_bn_device_);
    free_ptr(&replay_block_elem_off_device_);
    free_ptr(&replay_col_block_id_device_);
    free_ptr(&replay_K_blocks_ptrs_device_);
    free_ptr(&replay_norm_partial_sums_device_);
    free_ptr(&replay_norm_partial_flags_device_);
    free_ptr(&replay_norm_sum_device_);
    free_ptr(&replay_norm_flag_device_);

    replay_N_ = 0;
    replay_T_ = 0;
    replay_K_len_ = 0;
    replay_n_input_ = 0;
    replay_n_output_ = 0;
    replay_n_blocks_ = 0;
    replay_it_bytes_ = 0;
    replay_ditdv_bytes_ = 0;
    replay_ditdvpre_bytes_ = 0;
    replay_dLtdv_bytes_ = 0;
    replay_dvtdw_bytes_ = 0;
    replay_dV_hist_bytes_ = 0;
    replay_w_tick_bytes_ = 0;
    replay_dw_accum_bytes_ = 0;
    replay_dx_bytes_ = 0;
    replay_s_tmp_bytes_ = 0;
    replay_b_tmp_bytes_ = 0;
    replay_idx_bytes_ = 0;
    replay_sig_idx_bytes_ = 0;
    replay_it_ring_bytes_ = 0;
    replay_ditdv_ring_bytes_ = 0;
    replay_ditdvpre_ring_bytes_ = 0;
    replay_block_starts_bytes_ = 0;
    replay_block_bn_bytes_ = 0;
    replay_block_elem_off_bytes_ = 0;
    replay_col_block_id_bytes_ = 0;
    replay_K_blocks_ptrs_bytes_ = 0;
    replay_block_elem_total_ = 0;
    replay_norm_partial_sums_bytes_ = 0;
    replay_norm_partial_flags_bytes_ = 0;
    replay_norm_sum_bytes_ = 0;
    replay_norm_flag_bytes_ = 0;
}

void LearnRuntime::clear_capture_signal_buffers_() {
    auto free_ptr = [](auto** p) {
        if (*p != nullptr) {
            gpu_mem_free((void**) p);
            *p = nullptr;
        }
    };
    free_ptr(&capture_it_tn_device_);
    free_ptr(&capture_it_nt_device_);
    free_ptr(&capture_ditdv_tn_device_);
    free_ptr(&capture_ditdvpre_tn_device_);
    free_ptr(&capture_ditdv_nt_device_);
    free_ptr(&capture_ditdvpre_nt_device_);
    capture_N_ = 0;
    capture_T_ = 0;
    capture_k_mul_ = 0;
    capture_percise_ = false;
    capture_it_bytes_ = 0;
    capture_ditdv_bytes_ = 0;
    capture_ditdvpre_bytes_ = 0;
    capture_it_nt_bytes_ = 0;
    capture_ditdv_nt_bytes_ = 0;
    capture_ditdvpre_nt_bytes_ = 0;
}

}  // namespace neurong::runtime_api::learn
