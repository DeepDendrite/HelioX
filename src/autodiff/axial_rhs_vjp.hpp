#pragma once

namespace neurong::autodiff {

// GPU path for axial coupling accumulation:
// grad_v += J_axial^T * grad_rhs.
void accumulate_axial_rhs_vjp_gpu(
    int len,
    int ncell,
    const int* parent_index,
    const double* a_t,
    const double* b_t,
    const double* grad_rhs_t,
    double* grad_v_t);

}  // namespace neurong::autodiff
