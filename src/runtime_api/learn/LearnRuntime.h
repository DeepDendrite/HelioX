#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "runtime_api/core/SimRuntimeCore.h"

namespace heliox::runtime_api::learn {

// Learning-oriented runtime helpers that are still "backend" code (CUDA buffers + kernels),
// but are not part of the generic simulation API.
//
// Design goals:
// - No Python/nanobind dependency (pure C++).
// - Keep heavy CUDA logic out of the Python binding layer.
// - Reuse SimRuntimeCore for handle->pointer mapping and runtime configuration.
class LearnRuntime final {
public:
    struct DenseBlockHostView {
        const float* data = nullptr; // host pointer (CPU)
        int bn = 0;                  // block size (bn x bn x K_len)
        int k_len = 0;
    };

    explicit LearnRuntime(core::SimRuntimeCore& core);
    ~LearnRuntime();

    LearnRuntime(const LearnRuntime&) = delete;
    LearnRuntime& operator=(const LearnRuntime&) = delete;
    LearnRuntime(LearnRuntime&&) = delete;
    LearnRuntime& operator=(LearnRuntime&&) = delete;

    int set_dense_blocks_f32(std::span<const DenseBlockHostView> blocks);
    void clear_dense_blocks();
    int dense_block_k_len() const { return dense_block_k_len_; }
    const std::vector<float*>& dense_blocks_device() const { return dense_blocks_device_; }
    const std::vector<int>& dense_block_bn() const { return dense_block_bn_; }

    int simulate_output_vs_into(float* output_vs_tn,
                                int t_steps_plus1,
                                int n_output,
                                std::span<const int> output_v_handles,
                                double tstop_ms,
                                double v_init);

    int replay_compute_dw_dx_from_signals_into(const float* it_lr_nt,
                                               const float* ditdv_lr_nt,
                                               const float* ditdvpre_lr_nt,
                                               int N,
                                               int T,
                                               const float* dLtdv_lr_to,
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
                                               int clip_check_every);

    int simulate_and_replay_dw_dx_streaming_into(const float* dLtdv_lr_to,
                                                 int ksteps_total,
                                                 int n_output,
                                                 const int32_t* poutput,
                                                 const int32_t* pinput,
                                                 int n_input,
                                                 int N,
                                                 const int32_t* pre_of_col,
                                                 float* dw_out_n,
                                                 float* dx_lr_it,
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
                                                 double v_init,
                                                 double dt_ms,
                                                 double grad_scale,
                                                 double eps,
                                                 double grad_l2norm_threshold,
                                                 int clip_strategy,
                                                 int clip_check_every);

    int simulate_and_capture_mapped_signals_into(float* output_vs_tn,
                                                 int total_steps_plus1,
                                                 int n_output,
                                                 float* it_lr_tn,
                                                 float* ditdv_lr_tn,
                                                 float* ditdvpre_lr_tn,
                                                 int ksteps_total_plus1,
                                                 int N,
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
                                                 double v_init);

    // Variant that caches it/ditdv/ditdvpre on-device (N,T) for later replay.
    // output_vs_tn: (total_steps+1, n_output)
    // - dLtdv is not captured here.
    int simulate_and_capture_mapped_signals_cached(float* output_vs_tn,
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
                                                   double v_init);

    int replay_compute_dw_dx_from_cached_signals_into(const float* dLtdv_lr_to,
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
                                                      int clip_check_every);

    // Replay dw only using cached signal buffers (no input gradient path).
    //
    // This exists to support tasks that do not train inputs (n_input == 0), while
    // keeping the hot replay path fully inside the HELIOX backend.
    int replay_compute_dw_from_cached_signals_into(const float* dLtdv_lr_to,
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
                                                   int clip_check_every);

private:
    core::SimRuntimeCore& core_;

    bool get_cached_pointers_or_print_(int handle, double*& cpu_ptr, double*& gpu_ptr) const;

    int replay_compute_dw_dx_from_cached_signals_impl_(const float* dLtdv_lr_to,
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
                                                       bool need_dx);

    void clear_replay_dw_dx_buffers_();
    void clear_capture_signal_buffers_();

    // Dense blocks on device.
    std::vector<float*> dense_blocks_device_;
    std::vector<int> dense_block_bn_;
    int dense_block_k_len_ = 0;

    // Replay (dw/dx) workspace on device (allocated lazily and reused).
    int replay_N_ = 0;
    int replay_T_ = 0;
    int replay_K_len_ = 0;
    int replay_n_input_ = 0;
    int replay_n_output_ = 0;
    int replay_n_blocks_ = 0;
    size_t replay_it_bytes_ = 0;
    size_t replay_ditdv_bytes_ = 0;
    size_t replay_ditdvpre_bytes_ = 0;
    size_t replay_dLtdv_bytes_ = 0;
    size_t replay_dvtdw_bytes_ = 0;
    size_t replay_dV_hist_bytes_ = 0;
    size_t replay_w_tick_bytes_ = 0;
    size_t replay_dw_accum_bytes_ = 0;
    size_t replay_dx_bytes_ = 0;
    size_t replay_s_tmp_bytes_ = 0;
    size_t replay_b_tmp_bytes_ = 0;
    size_t replay_idx_bytes_ = 0;
    size_t replay_sig_idx_bytes_ = 0;
    size_t replay_it_ring_bytes_ = 0;
    size_t replay_ditdv_ring_bytes_ = 0;
    size_t replay_ditdvpre_ring_bytes_ = 0;
    size_t replay_block_starts_bytes_ = 0;
    size_t replay_block_bn_bytes_ = 0;
    size_t replay_block_elem_off_bytes_ = 0;
    size_t replay_col_block_id_bytes_ = 0;
    size_t replay_K_blocks_ptrs_bytes_ = 0;
    float* replay_it_device_ = nullptr;
    float* replay_ditdv_device_ = nullptr;
    float* replay_ditdvpre_device_ = nullptr;
    float* replay_dLtdv_device_ = nullptr;
    float* replay_dvtdw_device_ = nullptr;
    float* replay_dV_hist_device_ = nullptr;
    float* replay_w_tick_device_ = nullptr;
    float* replay_dw_accum_device_ = nullptr;
    float* replay_dx_device_ = nullptr;
    float* replay_s_tmp_device_ = nullptr;
    float* replay_b_tmp_device_ = nullptr;
    int* replay_dV_win_idx_device_ = nullptr;
    int* replay_sig_win_idx_device_ = nullptr;
    float* replay_it_ring_device_ = nullptr;
    float* replay_ditdv_ring_device_ = nullptr;
    float* replay_ditdvpre_ring_device_ = nullptr;
    int* replay_block_starts_device_ = nullptr;
    int* replay_block_bn_device_ = nullptr;
    int* replay_block_elem_off_device_ = nullptr;
    int32_t* replay_col_block_id_device_ = nullptr;
    float** replay_K_blocks_ptrs_device_ = nullptr;

    int replay_block_elem_total_ = 0;

    // Cached capture buffers (device).
    int capture_N_ = 0;
    int capture_T_ = 0;
    int capture_k_mul_ = 0;
    bool capture_percise_ = false;
    float* capture_it_tn_device_ = nullptr;
    float* capture_it_nt_device_ = nullptr;
    float* capture_ditdv_tn_device_ = nullptr;
    float* capture_ditdvpre_tn_device_ = nullptr;
    float* capture_ditdv_nt_device_ = nullptr;
    float* capture_ditdvpre_nt_device_ = nullptr;
    size_t capture_it_bytes_ = 0;
    size_t capture_ditdv_bytes_ = 0;
    size_t capture_ditdvpre_bytes_ = 0;
    size_t capture_it_nt_bytes_ = 0;
    size_t capture_ditdv_nt_bytes_ = 0;
    size_t capture_ditdvpre_nt_bytes_ = 0;

    // Norm-clip buffers.
    double* replay_norm_partial_sums_device_ = nullptr;
    int* replay_norm_partial_flags_device_ = nullptr;
    double* replay_norm_sum_device_ = nullptr;
    int* replay_norm_flag_device_ = nullptr;
    size_t replay_norm_partial_sums_bytes_ = 0;
    size_t replay_norm_partial_flags_bytes_ = 0;
    size_t replay_norm_sum_bytes_ = 0;
    size_t replay_norm_flag_bytes_ = 0;
};

}  // namespace heliox::runtime_api::learn
