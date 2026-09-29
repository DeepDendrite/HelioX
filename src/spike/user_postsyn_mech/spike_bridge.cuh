#pragma once

#include "postsyn_template.cuh"

#include <array>
#include <cstddef>
#include <cmath>

namespace SpikeBridge {

struct MechTrait {
    enum class VarNames {
        g,
        e,
        x,
        i,
        x_cur,
        drive,
    };
};

__global__ void spike_bridge_spike_vjp_kernel(
    int nnode,
    double* adj_x,
    const uint32_t* spk_vec_idx,
    const double* weights,
    const int* delay_steps,
    double* pending_pre_spike_adj,
    int pre_spike_adj_count,
    int step_count,
    int arrival_step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) {
        return;
    }
    const double local_adj = adj_x[i];
    if (local_adj != 0.0) {
        const uint32_t src_idx = spk_vec_idx[i];
        const int delay_step = delay_steps != nullptr ? delay_steps[i] : 0;
        const int emit_step = arrival_step - (delay_step > 0 ? delay_step : 0);
        if (src_idx < static_cast<uint32_t>(pre_spike_adj_count) &&
            emit_step >= 0 && emit_step < step_count) {
            const std::size_t dst =
                static_cast<std::size_t>(emit_step) * static_cast<std::size_t>(pre_spike_adj_count) +
                static_cast<std::size_t>(src_idx);
            atomicAdd(&pending_pre_spike_adj[dst], local_adj * weights[i]);
        }
        adj_x[i] = 0.0;
    }
}

class SpikeBridge_Templated final
        : public PostSynTemplate<SpikeBridge_Templated, MechTrait> {
    using Base = PostSynTemplate<SpikeBridge_Templated, MechTrait>;
    using enum MechTrait::VarNames;

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS | ENABLE_CURRENT_VJP;
    static constexpr auto LearnableVars = std::array{g};
    static constexpr auto VjpAdjointVars = std::array{x};
    static constexpr auto CurrentVjpTapeVars = std::array{x_cur, drive};

    explicit SpikeBridge_Templated(MechInitParams& param) : PostSynTemplate(param) {
        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({e, 1});
        var_in_coredata_idx.insert({x, 2});
        var_in_coredata_idx.insert({i, 3});
        var_in_coredata_idx.insert({x_cur, 4});
        var_in_coredata_idx.insert({drive, 5});

        init_values.insert({g, 1.0});
        init_values.insert({e, 0.0});
        init_values.insert({x, 0.0});
        init_values.insert({i, 0.0});
        init_values.insert({x_cur, 0.0});
        init_values.insert({drive, 0.0});
    }

    static consteval int x_adjoint_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::x>();
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& param, VarAccessor<MechTrait>& vars) {
        (void)param;
        vars(x) = 0.0;
        vars(i) = 0.0;
        vars(x_cur) = 0.0;
        vars(drive) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait>& vars) {
        const double local_x = vars(x);
        const double local_drive = param.volt - vars(e);
        const double current = vars(g) * local_x * local_drive;
        if (param.updateIon) {
            vars(i) = current;
            vars(x_cur) = local_x;
            vars(drive) = local_drive;
        }
        return current;
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& param, VarAccessor<MechTrait>& vars) {
        (void)param;
        vars(x) = 0.0;
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait> vars) {
        vars.idx = param.idx;
        const double grad_i = param.grad_mech_current;
        const double x_t = tape_ref<x_cur>(param, vars);
        const double drive_t = tape_ref<drive>(param, vars);

        mechAtomAdd(&adjoint_ref<x>(param, vars), grad_i * vars(g) * drive_t);
        mechAtomAdd(&param.grad_v[param.node_index], grad_i * vars(g) * x_t);
        mechAtomAdd(&grad_ref<g>(param, vars), grad_i * x_t * drive_t);
    }

    DUAL_EXEC void net_receive_single_node(PostSynTempRecvParam& recv, VarAccessor<MechTrait>& vars) {
        vars(x) += recv.weight;
    }

    bool supports_spike_vjp() const override { return true; }

    void spike_vjp_cpu(SimPostSynSpikeVJPParam& param) override {
        constexpr int slot = x_adjoint_slot();
        static_assert(slot >= 0, "SpikeBridge.x must be registered as a VJP adjoint var");
        this->ensure_default_vjp_adjoint_storage_();
        double* adj_x = this->vjp_adjoint_data_[static_cast<std::size_t>(slot)].get_cpu_data();
        uint32_t* spk_vec_idx = this->vecdata_spk_vec_idx->get_cpu_data();
        double* weights = this->vecdata_weights->get_cpu_data();
        int* delay_steps = this->vecdata_delay_steps->get_cpu_data();
        for (int i = 0; i < this->nnode; ++i) {
            const double local_adj = adj_x[i];
            if (local_adj != 0.0) {
                const uint32_t src_idx = spk_vec_idx[i];
                const int delay_step = delay_steps != nullptr ? delay_steps[i] : 0;
                const int emit_step = param.step_index - (delay_step > 0 ? delay_step : 0);
                if (src_idx < static_cast<uint32_t>(param.pre_spike_adj_count) &&
                    emit_step >= 0 && emit_step < param.step_count) {
                    const std::size_t dst =
                        static_cast<std::size_t>(emit_step) *
                            static_cast<std::size_t>(param.pre_spike_adj_count) +
                        static_cast<std::size_t>(src_idx);
                    param.pending_pre_spike_adj[dst] += local_adj * weights[i];
                }
                adj_x[i] = 0.0;
            }
        }
    }

    void spike_vjp_gpu(SimPostSynSpikeVJPParam& param) override {
        if (this->nnode <= 0) {
            return;
        }
        constexpr int slot = x_adjoint_slot();
        static_assert(slot >= 0, "SpikeBridge.x must be registered as a VJP adjoint var");
        this->ensure_default_vjp_adjoint_storage_();
        double* adj_x = this->vjp_adjoint_data_[static_cast<std::size_t>(slot)].get_gpu_data();
        uint32_t* spk_vec_idx = this->vecdata_spk_vec_idx->get_gpu_data();
        double* weights = this->vecdata_weights->get_gpu_data();
        int* delay_steps = this->vecdata_delay_steps->get_gpu_data();
        const int block_num = (this->nnode + nthread_per_block - 1) / nthread_per_block;
        cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(this->cuda_stream);
        spike_bridge_spike_vjp_kernel<<<block_num, nthread_per_block, 0, stream>>>(
            this->nnode,
            adj_x,
            spk_vec_idx,
            weights,
            delay_steps,
            param.pending_pre_spike_adj,
            param.pre_spike_adj_count,
            param.step_count,
            param.step_index);
    }
};

REGISTER_POSTSYN("SpikeBridge", SpikeBridge_Templated, 1);

} // namespace SpikeBridge
