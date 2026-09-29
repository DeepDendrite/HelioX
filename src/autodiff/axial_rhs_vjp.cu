#include "autodiff/axial_rhs_vjp.hpp"

#include "cuda_utils.h"
#include "utils.h"

namespace neurong::autodiff {

__global__ void accumulate_axial_rhs_vjp_kernel(
    int len,
    int ncell,
    const int* parent_index,
    const double* a_t,
    const double* b_t,
    const double* grad_rhs_t,
    double* grad_v_t) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < ncell || i >= len) {
        return;
    }
    const int parent = parent_index[i];
    const double grad_rhs_parent = grad_rhs_t[parent];
    const double grad_rhs_child = grad_rhs_t[i];
    atomicAdd(&grad_v_t[parent], a_t[i] * grad_rhs_parent - b_t[i] * grad_rhs_child);
    atomicAdd(&grad_v_t[i], b_t[i] * grad_rhs_child - a_t[i] * grad_rhs_parent);
}

void accumulate_axial_rhs_vjp_gpu(
    int len,
    int ncell,
    const int* parent_index,
    const double* a_t,
    const double* b_t,
    const double* grad_rhs_t,
    double* grad_v_t) {
    if (len <= 0) {
        return;
    }
    const int block_num = (len + nthread_per_block - 1) / nthread_per_block;
    accumulate_axial_rhs_vjp_kernel<<<block_num, nthread_per_block>>>(
        len, ncell, parent_index, a_t, b_t, grad_rhs_t, grad_v_t);
}

}  // namespace neurong::autodiff
