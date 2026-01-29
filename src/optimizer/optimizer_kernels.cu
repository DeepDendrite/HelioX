#include <cuda_runtime.h>
#include <math.h>
#include "cuda_utils.h"

namespace {
constexpr int THREADS_PER_BLOCK = 256;

__device__ double block_reduce_sum(double val) {
    __shared__ double shared[THREADS_PER_BLOCK];
    shared[threadIdx.x] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}
}  // namespace

__global__ void sgd_update_kernel(double** weight_ptrs,
                                  double** grad_ptrs,
                                  int* batch_sizes,
                                  int* batch_offsets,
                                  double* impedances,
                                  int param_count,
                                  double learning_rate,
                                  double inv_record_steps) {
    int param_idx = blockIdx.x;
    if (param_idx >= param_count) {
        return;
    }
    int batch_size = batch_sizes[param_idx];
    if (batch_size <= 0) {
        return;
    }
    int offset = batch_offsets[param_idx];

    double accum = 0.0;
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        double* g_ptr = grad_ptrs[offset + i];
        if (g_ptr) {
            accum += *g_ptr;
        }
    }
    double sum_grad = block_reduce_sum(accum);

    __shared__ double shared_weight;
    if (threadIdx.x == 0) {
        double avg_grad = sum_grad / static_cast<double>(batch_size);
        double scaled_grad = avg_grad * impedances[param_idx] * inv_record_steps;
        double* base_weight = weight_ptrs[offset];
        double updated_weight = base_weight ? (*base_weight - learning_rate * scaled_grad) : 0.0;
        if (base_weight) {
            *base_weight = updated_weight;
        }
        shared_weight = updated_weight;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        if (i > 0) {
            double* w_ptr = weight_ptrs[offset + i];
            if (w_ptr) {
                *w_ptr = shared_weight;
            }
        }
        double* g_ptr = grad_ptrs[offset + i];
        if (g_ptr) {
            *g_ptr = 0.0;
        }
    }
}

__global__ void sgd_momentum_kernel(double** weight_ptrs,
                                    double** grad_ptrs,
                                    double* velocities,
                                    int* batch_sizes,
                                    int* batch_offsets,
                                    double* impedances,
                                    int param_count,
                                    double learning_rate,
                                    double inv_record_steps,
                                    double momentum) {
    int param_idx = blockIdx.x;
    if (param_idx >= param_count) {
        return;
    }
    int batch_size = batch_sizes[param_idx];
    if (batch_size <= 0) {
        return;
    }
    int offset = batch_offsets[param_idx];

    double accum = 0.0;
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        double* g_ptr = grad_ptrs[offset + i];
        if (g_ptr) {
            accum += *g_ptr;
        }
    }
    double sum_grad = block_reduce_sum(accum);

    __shared__ double shared_weight;
    if (threadIdx.x == 0) {
        double avg_grad = sum_grad / static_cast<double>(batch_size);
        double scaled_grad = avg_grad * impedances[param_idx] * inv_record_steps;

        double velocity = velocities[param_idx];
        velocity = momentum * velocity - learning_rate * scaled_grad;
        velocities[param_idx] = velocity;

        double* base_weight = weight_ptrs[offset];
        double updated_weight = base_weight ? (*base_weight + velocity) : 0.0;
        if (base_weight) {
            *base_weight = updated_weight;
        }
        shared_weight = updated_weight;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        if (i > 0) {
            double* w_ptr = weight_ptrs[offset + i];
            if (w_ptr) {
                *w_ptr = shared_weight;
            }
        }
        double* g_ptr = grad_ptrs[offset + i];
        if (g_ptr) {
            *g_ptr = 0.0;
        }
    }
}

__global__ void adam_update_kernel(double** weight_ptrs,
                                   double** grad_ptrs,
                                   double* m_array,
                                   double* v_array,
                                   int* batch_sizes,
                                   int* batch_offsets,
                                   double* impedances,
                                   int param_count,
                                   double learning_rate,
                                   double inv_record_steps,
                                   double beta1,
                                   double beta2,
                                   double epsilon,
                                   long long step_count) {
    int param_idx = blockIdx.x;
    if (param_idx >= param_count) {
        return;
    }
    int batch_size = batch_sizes[param_idx];
    if (batch_size <= 0) {
        return;
    }
    int offset = batch_offsets[param_idx];

    double accum = 0.0;
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        double* g_ptr = grad_ptrs[offset + i];
        if (g_ptr) {
            accum += *g_ptr;
        }
    }
    double sum_grad = block_reduce_sum(accum);

    __shared__ double shared_weight;
    if (threadIdx.x == 0) {
        double avg_grad = sum_grad / static_cast<double>(batch_size);
        double scaled_grad = avg_grad * impedances[param_idx] * inv_record_steps;

        double m = m_array[param_idx];
        double v = v_array[param_idx];
        m = beta1 * m + (1.0 - beta1) * scaled_grad;
        v = beta2 * v + (1.0 - beta2) * scaled_grad * scaled_grad;

        double beta1_pow = pow(beta1, static_cast<double>(step_count));
        double beta2_pow = pow(beta2, static_cast<double>(step_count));
        double m_hat = m / (1.0 - beta1_pow);
        double v_hat = v / (1.0 - beta2_pow);

        double* base_weight = weight_ptrs[offset];
        double updated_weight = base_weight
                                    ? (*base_weight - learning_rate * m_hat / (sqrt(v_hat) + epsilon))
                                    : 0.0;
        if (base_weight) {
            *base_weight = updated_weight;
        }
        shared_weight = updated_weight;

        m_array[param_idx] = m;
        v_array[param_idx] = v;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        if (i > 0) {
            double* w_ptr = weight_ptrs[offset + i];
            if (w_ptr) {
                *w_ptr = shared_weight;
            }
        }
        double* g_ptr = grad_ptrs[offset + i];
        if (g_ptr) {
            *g_ptr = 0.0;
        }
    }
}

void launch_sgd_update_kernel(double** weight_ptrs,
                              double** grad_ptrs,
                              int* batch_sizes,
                              int* batch_offsets,
                              double* impedances,
                              int param_count,
                              double learning_rate,
                              double inv_record_steps,
                              cudaStream_t stream) {
    if (param_count <= 0) {
        return;
    }
    dim3 blocks(param_count);
    int threads = THREADS_PER_BLOCK;
    sgd_update_kernel<<<blocks, threads, 0, stream>>>(
        weight_ptrs,
        grad_ptrs,
        batch_sizes,
        batch_offsets,
        impedances,
        param_count,
        learning_rate,
        inv_record_steps);
    CUDA_CHECK_ERR();
}

void launch_sgd_momentum_kernel(double** weight_ptrs,
                                double** grad_ptrs,
                                double* velocities,
                                int* batch_sizes,
                                int* batch_offsets,
                                double* impedances,
                                int param_count,
                                double learning_rate,
                                double inv_record_steps,
                                double momentum,
                                cudaStream_t stream) {
    if (param_count <= 0) {
        return;
    }
    dim3 blocks(param_count);
    int threads = THREADS_PER_BLOCK;
    sgd_momentum_kernel<<<blocks, threads, 0, stream>>>(
        weight_ptrs,
        grad_ptrs,
        velocities,
        batch_sizes,
        batch_offsets,
        impedances,
        param_count,
        learning_rate,
        inv_record_steps,
        momentum);
    CUDA_CHECK_ERR();
}

void launch_adam_update_kernel(double** weight_ptrs,
                               double** grad_ptrs,
                               double* m_array,
                               double* v_array,
                               int* batch_sizes,
                               int* batch_offsets,
                               double* impedances,
                               int param_count,
                               double learning_rate,
                               double inv_record_steps,
                               double beta1,
                               double beta2,
                               double epsilon,
                               long long step_count,
                               cudaStream_t stream) {
    if (param_count <= 0) {
        return;
    }
    dim3 blocks(param_count);
    int threads = THREADS_PER_BLOCK;
    adam_update_kernel<<<blocks, threads, 0, stream>>>(
        weight_ptrs,
        grad_ptrs,
        m_array,
        v_array,
        batch_sizes,
        batch_offsets,
        impedances,
        param_count,
        learning_rate,
        inv_record_steps,
        beta1,
        beta2,
        epsilon,
        step_count);
    CUDA_CHECK_ERR();
}
