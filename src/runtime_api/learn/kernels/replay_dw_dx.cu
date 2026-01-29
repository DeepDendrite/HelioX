#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr int kThreads = 256;
}

__global__ void replay_scale_f32_kernel(float* data, size_t n, float scale) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) {
        return;
    }
    data[idx] *= scale;
}

__global__ void replay_nan_to_num_f32_kernel(float* data, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float v = data[idx];
    if (!isfinite(v)) {
        data[idx] = 0.0f;
    }
}

// Pass 1: compute per-block partial sum of squares and a nonfinite flag.
__global__ void replay_frobenius_partial_kernel(const float* data, size_t n, double* partial_sums, int* partial_flags) {
    __shared__ double s_sum[kThreads];
    __shared__ int s_flag[kThreads];
    const size_t tid = static_cast<size_t>(threadIdx.x);
    const size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    const size_t idx = base + tid;
    double acc = 0.0;
    int flag = 0;
    if (idx < n) {
        const float v = data[idx];
        if (isfinite(v)) {
            const double dv = static_cast<double>(v);
            acc = dv * dv;
        } else {
            flag = 1;
        }
    }
    s_sum[tid] = acc;
    s_flag[tid] = flag;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_flag[tid] |= s_flag[tid + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = s_sum[0];
        partial_flags[blockIdx.x] = s_flag[0];
    }
}

// Pass 2: reduce partial arrays into one sum and one flag.
__global__ void replay_frobenius_finalize_kernel(const double* partial_sums,
                                                 const int* partial_flags,
                                                 int n_partials,
                                                 double* out_sum,
                                                 int* out_flag) {
    __shared__ double s_sum[kThreads];
    __shared__ int s_flag[kThreads];
    const int tid = threadIdx.x;
    double acc = 0.0;
    int flag = 0;
    for (int i = tid; i < n_partials; i += blockDim.x) {
        acc += partial_sums[i];
        flag |= partial_flags[i];
    }
    s_sum[tid] = acc;
    s_flag[tid] = flag;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_flag[tid] |= s_flag[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_sum = s_sum[0];
        *out_flag = s_flag[0];
    }
}

// Compute w_tick[i] = sum_o dvtdw[i, poutput[o]] * dLtdv[o].
__global__ void replay_compute_w_tick_kernel(const float* dvtdw,
                                             int N,
                                             const int32_t* poutput,
                                             int n_output,
                                             const float* dLtdv,
                                             float* w_tick,
                                             float* dw_accum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    float acc = 0.0f;
    const int row_off = i * N;
    for (int o = 0; o < n_output; ++o) {
        const int col = static_cast<int>(poutput[o]);
        acc += dvtdw[row_off + col] * dLtdv[o];
    }
    w_tick[i] = acc;
    dw_accum[i] += acc;
}

// Per-block diagonal base term:
// dv[start+i, start+j] = dt*grad_scale * sum_t K[i,j, K_len-tw+t] * It[start+i, t_lr-tw+1+t]
__global__ void replay_base_block_kernel(const float* it_lr,
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
                                         float* dvtdw) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = bn * bn;
    if (idx >= total) {
        return;
    }
    const int bi = idx / bn;
    const int bj = idx - bi * bn;
    const int row = start + bi;
    const int col = start + bj;

    float sum = 0.0f;
    const int it_t0 = t_lr - t_window + 1;
    const int k_t0 = K_len - t_window;
    const int it_row_off = row * T;
    const int k_off = (bi * bn + bj) * K_len + k_t0;
    for (int t = 0; t < t_window; ++t) {
        sum += K_block[k_off + t] * it_lr[it_row_off + (it_t0 + t)];
    }
    dvtdw[row * N + col] = sum * dt * grad_scale;
}

// Time-major variant of base term:
// it_tn layout: (T, N) time-major, where the per-tick vector is contiguous.
__global__ void replay_base_block_tmajor_kernel(const float* it_tn,
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
                                                float* dvtdw) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = bn * bn;
    if (idx >= total) {
        return;
    }
    const int bi = idx / bn;
    const int bj = idx - bi * bn;
    const int row = start + bi;
    const int col = start + bj;

    float sum = 0.0f;
    const int it_t0 = t_lr - t_window + 1;
    const int k_t0 = K_len - t_window;
    const int k_off = (bi * bn + bj) * K_len + k_t0;
    for (int t = 0; t < t_window; ++t) {
        const int it_t = it_t0 + t;
        sum += K_block[k_off + t] * it_tn[it_t * N + row];
    }
    dvtdw[row * N + col] = sum * dt * grad_scale;
}

// Ring-buffer variant of base term.
//
// it_ring layout: (K_len, N) time-major, where slot = t_lr % K_len.
// sig_win_idx: (t_window) ring slot indices in chronological order matching K's window.
__global__ void replay_base_block_ring_kernel(const float* it_ring,
                                              const int* sig_win_idx,
                                              int N,
                                              int t_window,
                                              const float* K_block,
                                              int bn,
                                              int K_len,
                                              int start,
                                              float dt,
                                              float grad_scale,
                                              float* dvtdw) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = bn * bn;
    if (idx >= total) {
        return;
    }
    const int bi = idx / bn;
    const int bj = idx - bi * bn;
    const int row = start + bi;
    const int col = start + bj;

    float sum = 0.0f;
    const int k_t0 = K_len - t_window;
    const int k_off = (bi * bn + bj) * K_len + k_t0;
    for (int t = 0; t < t_window; ++t) {
        const int slot = sig_win_idx[t];
        sum += K_block[k_off + t] * it_ring[slot * N + row];
    }
    dvtdw[row * N + col] = sum * dt * grad_scale;
}

// Fused base term across all diagonal blocks (reduces one kernel launch per block per LR tick).
//
// Layout assumptions:
// - Blocks partition [0..N) contiguously with starts[]/bn[].
// - Each block has its own K_block (bn x bn x K_len), stored in K_blocks[bid].
// - block_elem_off is an exclusive prefix sum of bn^2, length n_blocks+1, with total_elems=block_elem_off[n_blocks].
__global__ void replay_base_blocks_kernel(const float* it_lr,
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
                                          float* dvtdw) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) {
        return;
    }

    // Find owning block via binary search in prefix sums.
    int lo = 0;
    int hi = n_blocks;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (idx >= block_elem_off[mid]) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int b = lo;
    const int bn = block_bn[b];
    const int start = block_starts[b];
    const int local = idx - block_elem_off[b];
    const int bi = local / bn;
    const int bj = local - bi * bn;
    const int row = start + bi;
    const int col = start + bj;

    float sum = 0.0f;
    const int it_t0 = t_lr - t_window + 1;
    const int k_t0 = K_len - t_window;
    const int it_row_off = row * T;
    const int k_off = (bi * bn + bj) * K_len + k_t0;
    float* K_block = K_blocks[b];
    for (int t = 0; t < t_window; ++t) {
        sum += K_block[k_off + t] * it_lr[it_row_off + (it_t0 + t)];
    }
    dvtdw[row * N + col] = sum * dt * grad_scale;
}

// Precise correction term (adds into dvtdw):
// For each output element (i, col):
//   bid = col_block_id[col], bn=block_bn[bid], start=block_start[bid], j=col-start
//   dv_corr = sum_{k,t} K_block[j,k, K_len-tw+t] * (dV_hist[i, col_k, dV_idx[t]]*dItdv[col_k, it_t]
//                                                 + dV_hist[i, pre_of_col[col_k], dV_idx[t]]*dItdvpre[col_k, it_t])
//   dvtdw[i,col] += dv_corr * dt
__global__ void replay_corr_allcols_kernel(const float* dV_hist,  // (K_len, N, N) time-major
                                           const float* ditdv_lr, // (N, T_lr+1)
                                           const float* ditdvpre_lr,
                                           const int* dV_win_idx, // (t_window)
                                           int t_window,
                                           int N,
                                           int T,
                                           int t_lr,
                                           const int32_t* pre_of_col, // (N)
                                           float** K_blocks,
                                           const int* block_starts,
                                           const int* block_bn,
                                           const int32_t* col_block_id, // (N)
                                           int K_len,
                                           float dt,
                                           float* dvtdw) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || i >= N) {
        return;
    }

    const int32_t bid = col_block_id[col];
    const int start = block_starts[bid];
    const int bn = block_bn[bid];
    const int j = col - start;
    float* K_block = K_blocks[bid];

    const int it_t0 = t_lr - t_window + 1;
    const int k_t0 = K_len - t_window;
    const int row_off = i * N;

    float acc = 0.0f;
    for (int k = 0; k < bn; ++k) {
        const int col_k = start + k;
        const int pre_col = static_cast<int>(pre_of_col[col_k]);
        const int dit_row_off = col_k * T;
        const int k_off = (j * bn + k) * K_len + k_t0;
        for (int t = 0; t < t_window; ++t) {
            const int hv = dV_win_idx[t];
            const float* dV_mat = dV_hist + static_cast<size_t>(hv) * static_cast<size_t>(N) * static_cast<size_t>(N);
            const float dV = dV_mat[row_off + col_k];
            const float dVpre = (pre_col >= 0) ? dV_mat[row_off + pre_col] : 0.0f;
            const float dItdv = ditdv_lr[dit_row_off + (it_t0 + t)];
            const float dItdvpre = ditdvpre_lr[dit_row_off + (it_t0 + t)];
            const float dItdw = dV * dItdv + dVpre * dItdvpre;
            acc += dItdw * K_block[k_off + t];
        }
    }
    dvtdw[row_off + col] += acc * dt;
}

// Time-major variant of precise correction term:
// ditdv_tn/ditdvpre_tn layout: (T, N) time-major.
__global__ void replay_corr_allcols_tmajor_kernel(const float* dV_hist,      // (K_len, N, N) time-major
                                                  const float* ditdv_tn,     // (T, N)
                                                  const float* ditdvpre_tn,  // (T, N)
                                                  const int* dV_win_idx,     // (t_window)
                                                  int t_window,
                                                  int N,
                                                  int T,
                                                  int t_lr,
                                                  const int32_t* pre_of_col, // (N)
                                                  float** K_blocks,
                                                  const int* block_starts,
                                                  const int* block_bn,
                                                  const int32_t* col_block_id, // (N)
                                                  int K_len,
                                                  float dt,
                                                  float* dvtdw) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || i >= N) {
        return;
    }

    const int32_t bid = col_block_id[col];
    const int start = block_starts[bid];
    const int bn = block_bn[bid];
    const int j = col - start;
    float* K_block = K_blocks[bid];

    const int it_t0 = t_lr - t_window + 1;
    const int k_t0 = K_len - t_window;
    const int row_off = i * N;

    float acc = 0.0f;
    for (int k = 0; k < bn; ++k) {
        const int col_k = start + k;
        const int pre_col = static_cast<int>(pre_of_col[col_k]);
        const int k_off = (j * bn + k) * K_len + k_t0;
        for (int t = 0; t < t_window; ++t) {
            const int hv = dV_win_idx[t];
            const float* dV_mat = dV_hist + static_cast<size_t>(hv) * static_cast<size_t>(N) * static_cast<size_t>(N);
            const float dV = dV_mat[row_off + col_k];
            const float dVpre = (pre_col >= 0) ? dV_mat[row_off + pre_col] : 0.0f;
            const int it_t = it_t0 + t;
            const float dItdv = ditdv_tn[it_t * N + col_k];
            const float dItdvpre = ditdvpre_tn[it_t * N + col_k];
            const float dItdw = dV * dItdv + dVpre * dItdvpre;
            acc += dItdw * K_block[k_off + t];
        }
    }
    dvtdw[row_off + col] += acc * dt;
}

// Ring-buffer variant of precise correction term.
//
// ditdv_ring/ditdvpre_ring layout: (K_len, N) time-major (slot = t_lr % K_len).
// sig_win_idx: (t_window) ring slot indices for ditdv/ditdvpre in chronological order.
__global__ void replay_corr_allcols_ring_kernel(const float* dV_hist,       // (K_len, N, N) time-major
                                                const float* ditdv_ring,    // (K_len, N) time-major
                                                const float* ditdvpre_ring, // (K_len, N) time-major
                                                const int* dV_win_idx,      // (t_window)
                                                const int* sig_win_idx,     // (t_window)
                                                int t_window,
                                                int N,
                                                const int32_t* pre_of_col, // (N)
                                                float** K_blocks,
                                                const int* block_starts,
                                                const int* block_bn,
                                                const int32_t* col_block_id, // (N)
                                                int K_len,
                                                float dt,
                                                float* dvtdw) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || i >= N) {
        return;
    }

    const int32_t bid = col_block_id[col];
    const int start = block_starts[bid];
    const int bn = block_bn[bid];
    const int j = col - start;
    float* K_block = K_blocks[bid];

    const int k_t0 = K_len - t_window;
    const int row_off = i * N;

    float acc = 0.0f;
    for (int k = 0; k < bn; ++k) {
        const int col_k = start + k;
        const int pre_col = static_cast<int>(pre_of_col[col_k]);
        const int k_off = (j * bn + k) * K_len + k_t0;
        for (int t = 0; t < t_window; ++t) {
            const int hv = dV_win_idx[t];
            const float* dV_mat = dV_hist + static_cast<size_t>(hv) * static_cast<size_t>(N) * static_cast<size_t>(N);
            const float dV = dV_mat[row_off + col_k];
            const float dVpre = (pre_col >= 0) ? dV_mat[row_off + pre_col] : 0.0f;
            const int slot = sig_win_idx[t];
            const float dItdv = ditdv_ring[slot * N + col_k];
            const float dItdvpre = ditdvpre_ring[slot * N + col_k];
            const float dItdw = dV * dItdv + dVpre * dItdvpre;
            acc += dItdw * K_block[k_off + t];
        }
    }
    dvtdw[row_off + col] += acc * dt;
}

// Reduce per-input dx scalars:
// s[j] = sum_i w_tick[i] / (dvtdw[i, pinput[j]] + eps)
// b[j] = sum_i dvtdw[i, pinput[j]]
__global__ void replay_dx_reduce_kernel(const float* dvtdw,
                                        int N,
                                        const int32_t* pinput,
                                        int n_input,
                                        const float* w_tick,
                                        float eps,
                                        float* s_out,
                                        float* b_out) {
    const int j = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_input) {
        return;
    }
    const int col = static_cast<int>(pinput[j]);

    float s = 0.0f;
    float b = 0.0f;
    if (i < N) {
        const float v = dvtdw[i * N + col];
        b = v;
        s = w_tick[i] / (v + eps);
    }

    // block reduce s and b
    __shared__ float s_shared[kThreads];
    __shared__ float b_shared[kThreads];
    s_shared[threadIdx.x] = s;
    b_shared[threadIdx.x] = b;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_shared[threadIdx.x] += s_shared[threadIdx.x + stride];
            b_shared[threadIdx.x] += b_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_out[j], s_shared[0]);
        atomicAdd(&b_out[j], b_shared[0]);
    }
}

__global__ void replay_dx_finalize_kernel(const float* s,
                                          const float* b,
                                          int n_input,
                                          int t_lr,
                                          int T_lr,
                                          float* dx_out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_input) {
        return;
    }
    dx_out[j * T_lr + t_lr] = s[j] * b[j];
}

__global__ void transpose_tn_to_nt_f32_kernel(const float* src_tn, float* dst_nt, int T, int N) {
    constexpr int kTile = 32;
    __shared__ float tile[kTile][kTile + 1]; // +1 avoids shared-memory bank conflicts

    const int n = blockIdx.x * kTile + threadIdx.x; // src col (N), dst row
    const int t = blockIdx.y * kTile + threadIdx.y; // src row (T), dst col
    if (n < N && t < T) {
        tile[threadIdx.y][threadIdx.x] = src_tn[t * N + n];
    }
    __syncthreads();

    const int t2 = blockIdx.y * kTile + threadIdx.x; // dst col (T)
    const int n2 = blockIdx.x * kTile + threadIdx.y; // dst row (N)
    if (n2 < N && t2 < T) {
        dst_nt[n2 * T + t2] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void launch_transpose_tn_to_nt_f32(const float* src_tn, float* dst_nt, int T, int N) {
    if (T <= 0 || N <= 0) {
        return;
    }
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    transpose_tn_to_nt_f32_kernel<<<grid, block>>>(src_tn, dst_nt, T, N);
}

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
                                                float* dvtdw) {
    const int total = bn * bn;
    const int grid = (total + kThreads - 1) / kThreads;
    replay_base_block_kernel<<<grid, kThreads>>>(it_lr, N, T, t_lr, t_window, K_block, bn, K_len, start, dt, grad_scale, dvtdw);
}

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
                                                 float* dvtdw) {
    if (n_blocks <= 0 || total_elems <= 0) {
        return;
    }
    const int grid = (total_elems + kThreads - 1) / kThreads;
    replay_base_blocks_kernel<<<grid, kThreads>>>(it_lr, N, T, t_lr, t_window, K_blocks, block_starts, block_bn, block_elem_off, n_blocks,
                                                  total_elems, K_len, dt, grad_scale, dvtdw);
}

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
                                                       float* dvtdw) {
    const int total = bn * bn;
    const int grid = (total + kThreads - 1) / kThreads;
    replay_base_block_tmajor_kernel<<<grid, kThreads>>>(it_tn, N, T, t_lr, t_window, K_block, bn, K_len, start, dt, grad_scale, dvtdw);
}

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
                                                     float* dvtdw) {
    const int total = bn * bn;
    const int grid = (total + kThreads - 1) / kThreads;
    replay_base_block_ring_kernel<<<grid, kThreads>>>(it_ring, sig_win_idx, N, t_window, K_block, bn, K_len, start, dt, grad_scale, dvtdw);
}

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
                                                  float* dvtdw) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    replay_corr_allcols_kernel<<<grid, block>>>(dV_hist, ditdv_lr, ditdvpre_lr, dV_win_idx, t_window, N, T, t_lr, pre_of_col, K_blocks,
                                                block_starts, block_bn, col_block_id, K_len, dt, dvtdw);
}

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
                                                         float* dvtdw) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    replay_corr_allcols_tmajor_kernel<<<grid, block>>>(dV_hist, ditdv_tn, ditdvpre_tn, dV_win_idx, t_window, N, T, t_lr, pre_of_col,
                                                       K_blocks, block_starts, block_bn, col_block_id, K_len, dt, dvtdw);
}

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
                                                       float* dvtdw) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    replay_corr_allcols_ring_kernel<<<grid, block>>>(dV_hist, ditdv_ring, ditdvpre_ring, dV_win_idx, sig_win_idx, t_window, N, pre_of_col,
                                                     K_blocks, block_starts, block_bn, col_block_id, K_len, dt, dvtdw);
}

extern "C" void launch_replay_compute_w_tick_kernel(const float* dvtdw,
                                                    int N,
                                                    const int32_t* poutput,
                                                    int n_output,
                                                    const float* dLtdv,
                                                    float* w_tick,
                                                    float* dw_accum) {
    const int grid = (N + kThreads - 1) / kThreads;
    replay_compute_w_tick_kernel<<<grid, kThreads>>>(dvtdw, N, poutput, n_output, dLtdv, w_tick, dw_accum);
}

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
                                        float* b_tmp) {
    cudaMemset(s_tmp, 0, static_cast<size_t>(n_input) * sizeof(float));
    cudaMemset(b_tmp, 0, static_cast<size_t>(n_input) * sizeof(float));

    dim3 block(kThreads, 1);
    dim3 grid((N + kThreads - 1) / kThreads, n_input);
    replay_dx_reduce_kernel<<<grid, block>>>(dvtdw, N, pinput, n_input, w_tick, eps, s_tmp, b_tmp);

    const int grid2 = (n_input + kThreads - 1) / kThreads;
    replay_dx_finalize_kernel<<<grid2, kThreads>>>(s_tmp, b_tmp, n_input, t_lr, T_lr, dx_out);
}

extern "C" void launch_replay_scale_f32(float* data, size_t n, float scale) {
    if (n == 0) {
        return;
    }
    const size_t grid = (n + static_cast<size_t>(kThreads) - 1) / static_cast<size_t>(kThreads);
    replay_scale_f32_kernel<<<static_cast<unsigned int>(grid), kThreads>>>(data, n, scale);
}

extern "C" void launch_replay_nan_to_num_f32(float* data, size_t n) {
    if (n == 0) {
        return;
    }
    const size_t grid = (n + static_cast<size_t>(kThreads) - 1) / static_cast<size_t>(kThreads);
    replay_nan_to_num_f32_kernel<<<static_cast<unsigned int>(grid), kThreads>>>(data, n);
}

extern "C" int launch_replay_frobenius_f32(const float* data,
                                           size_t n,
                                           double* partial_sums,
                                           int* partial_flags,
                                           double* out_sum,
                                           int* out_flag) {
    if (n == 0) {
        return 0;
    }
    const int blocks = static_cast<int>((n + static_cast<size_t>(kThreads) - 1) / static_cast<size_t>(kThreads));
    replay_frobenius_partial_kernel<<<blocks, kThreads>>>(data, n, partial_sums, partial_flags);
    // One-block finalize.
    replay_frobenius_finalize_kernel<<<1, kThreads>>>(partial_sums, partial_flags, blocks, out_sum, out_flag);
    return blocks;
}
