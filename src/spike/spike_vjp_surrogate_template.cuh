#pragma once

#include <cuda_runtime.h>

#include "spike_vjp_surrogate.h"

namespace neurong::spike_vjp {

constexpr int kSpikeVjpSurrogateThreads = 128;

template <typename Surrogate>
__global__ void spike_vjp_surrogate_kernel(
    SpikeVjpSurrogateRunParam param,
    SpikeVjpSurrogateConfig config) {
    const int ipre = blockIdx.x * blockDim.x + threadIdx.x;
    if (ipre >= param.npre) {
        return;
    }
    double* pending_step = param.pending_pre_spike_adj;
    const double adj_spike = pending_step[ipre];
    if (adj_spike == 0.0) {
        return;
    }
    const double x = param.pre_v_tape[ipre] - param.threshold[ipre];
    const double phi = Surrogate::eval(x, config);
    if (phi != 0.0) {
        atomicAdd(&param.carry_v[param.pre_node_indices[ipre]], adj_spike * phi);
    }
    pending_step[ipre] = 0.0;
}

template <typename Surrogate>
void run_spike_vjp_surrogate_cpu(
    const SpikeVjpSurrogateRunParam& param,
    const SpikeVjpSurrogateConfig& config) {
    if (param.npre <= 0 || param.pending_pre_spike_adj == nullptr) {
        return;
    }
    for (int ipre = 0; ipre < param.npre; ++ipre) {
        const double adj_spike = param.pending_pre_spike_adj[ipre];
        if (adj_spike == 0.0) {
            continue;
        }
        const double x = param.pre_v_tape[ipre] - param.threshold[ipre];
        const double phi = Surrogate::eval(x, config);
        if (phi != 0.0) {
            param.carry_v[param.pre_node_indices[ipre]] += adj_spike * phi;
        }
        param.pending_pre_spike_adj[ipre] = 0.0;
    }
}

template <typename Surrogate>
void launch_spike_vjp_surrogate_gpu(
    const SpikeVjpSurrogateRunParam& param,
    const SpikeVjpSurrogateConfig& config) {
    if (param.npre <= 0 || param.pending_pre_spike_adj == nullptr) {
        return;
    }
    const int block_num = (param.npre + kSpikeVjpSurrogateThreads - 1) / kSpikeVjpSurrogateThreads;
    spike_vjp_surrogate_kernel<Surrogate><<<block_num, kSpikeVjpSurrogateThreads>>>(param, config);
}

template <typename Surrogate>
SpikeVjpSurrogateOps make_spike_vjp_surrogate_ops(
    const char* name,
    const char* source_file,
    const char* source_symbol) {
    return SpikeVjpSurrogateOps{
        .name = name,
        .source_file = source_file,
        .source_symbol = source_symbol,
        .run_cpu = &run_spike_vjp_surrogate_cpu<Surrogate>,
        .launch_gpu = &launch_spike_vjp_surrogate_gpu<Surrogate>,
    };
}

}  // namespace neurong::spike_vjp

#ifndef NEURONG_CONCAT_INNER
#define NEURONG_CONCAT_INNER(a, b) a##b
#endif
#ifndef NEURONG_CONCAT
#define NEURONG_CONCAT(a, b) NEURONG_CONCAT_INNER(a, b)
#endif

#define REGISTER_SPIKE_VJP_SURROGATE(CLASS, ...) \
    static const bool NEURONG_CONCAT(_spike_vjp_surrogate_registered_, __COUNTER__) = []() { \
        auto ops = ::neurong::spike_vjp::make_spike_vjp_surrogate_ops<CLASS>( \
            CLASS::name, __FILE__, #CLASS); \
        ::neurong::spike_vjp::SpikeVjpSurrogateRegistry::getInstance().registerSurrogate( \
            ops, {CLASS::name __VA_OPT__(,) __VA_ARGS__}); \
        return true; \
    }()
