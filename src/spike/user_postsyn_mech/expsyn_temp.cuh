#include "postsyn_template.cuh"

#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace ExpSynTemplated {
struct MechTrait {
    enum class VarNames {
        tau,
        e,
        g,
        i,
        g_cur,
        drive,
        event_weight_pending,
        event_weight_cur
    };
};

__global__ void expsyn_spike_vjp_kernel(
    int nnode,
    const double* adj_g,
    const double* event_weight_tape,
    int event_tape_step,
    int event_tape_stride,
    const uint32_t* spk_vec_idx,
    const int* delay_steps,
    double* pending_pre_spike_adj,
    int pre_spike_adj_count,
    int step_count,
    int arrival_step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) {
        return;
    }
    const double event_weight =
        event_weight_tape[event_tape_step * event_tape_stride + i];
    if (event_weight == 0.0) {
        return;
    }
    const double local_adj = adj_g[i];
    if (local_adj == 0.0) {
        return;
    }
    const uint32_t src_idx = spk_vec_idx[i];
    const int delay_step = delay_steps != nullptr ? delay_steps[i] : 0;
    const int emit_step = arrival_step - (delay_step > 0 ? delay_step : 0);
    if (src_idx < static_cast<uint32_t>(pre_spike_adj_count) &&
        emit_step >= 0 && emit_step < step_count) {
        const std::size_t dst =
            static_cast<std::size_t>(emit_step) *
                static_cast<std::size_t>(pre_spike_adj_count) +
            static_cast<std::size_t>(src_idx);
        atomicAdd(&pending_pre_spike_adj[dst], local_adj * event_weight);
    }
}

class ExpSyn_Templated : public PostSynTemplate<ExpSyn_Templated, MechTrait> {
    using Base = PostSynTemplate<ExpSyn_Templated, MechTrait>;
    using enum MechTrait::VarNames;

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS | ENABLE_CURRENT_VJP;
    static constexpr auto VjpAdjointVars = std::array{g};
    static constexpr auto CurrentVjpTapeVars = std::array{g_cur, drive, event_weight_cur};

    ExpSyn_Templated(MechInitParams& params) : PostSynTemplate(params) {
        var_in_coredata_idx.insert({tau, 0});
        var_in_coredata_idx.insert({e, 1});
        init_values.insert({g, 0.0});
        init_values.insert({i, 0.0});
        init_values.insert({g_cur, 0.0});
        init_values.insert({drive, 0.0});
        init_values.insert({event_weight_pending, 0.0});
        init_values.insert({event_weight_cur, 0.0});
    }

    static consteval int g_adjoint_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::g>();
    }

    static consteval int event_weight_tape_slot() {
        return Base::template current_vjp_tape_var_index_constexpr_<MechTrait::VarNames::event_weight_cur>();
    }

    DUAL_EXEC void init_single_node(MechTempInitParam param, VarAccessor<MechTrait> vars) {
        (void)param;
        vars(g) = 0.0;
        vars(i) = 0.0;
        vars(g_cur) = 0.0;
        vars(drive) = 0.0;
        vars(event_weight_pending) = 0.0;
        vars(event_weight_cur) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait> vars) {
        const double local_g = vars(g);
        const double local_drive = param.volt - vars(e);
        const double current = local_g * local_drive;
        if (param.updateIon) {
            vars(i) = current;
            vars(g_cur) = local_g;
            vars(drive) = local_drive;
            vars(event_weight_cur) = vars(event_weight_pending);
            vars(event_weight_pending) = 0.0;
        }
        return current;
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& param, VarAccessor<MechTrait>& vars) {
        vars(g) *= exp(-param.dt / vars(tau));
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait> vars) {
        vars.idx = param.idx;
        const double decay = exp(-param.dt / vars(tau));
        const double g_t = tape_ref<g_cur>(param, vars);
        const double drive_t = tape_ref<drive>(param, vars);
        double& adj_g = adjoint_ref<g>(param, vars);
        adj_g = adj_g * decay + param.grad_mech_current * drive_t;
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * g_t);
    }

    DUAL_EXEC void net_receive_single_node(PostSynTempRecvParam& recv_param, VarAccessor<MechTrait>& vars) {
        vars(g) += recv_param.weight;
        vars(event_weight_pending) += recv_param.weight;
    }

    bool supports_spike_vjp() const override { return true; }

    void spike_vjp_cpu(SimPostSynSpikeVJPParam& param) override {
        constexpr int adj_slot = g_adjoint_slot();
        constexpr int event_slot = event_weight_tape_slot();
        static_assert(adj_slot >= 0, "ExpSyn.g must be registered as a VJP adjoint var");
        static_assert(event_slot >= 0, "ExpSyn.event_weight_cur must be in CurrentVjpTapeVars");
        this->ensure_default_vjp_adjoint_storage_();
        if (static_cast<int>(this->current_vjp_tape_stores_.size()) <= event_slot) {
            throw std::runtime_error("ExpSyn spike_vjp_cpu: current VJP tape stores are not initialized");
        }
        const double* event_tape =
            this->current_vjp_tape_stores_[static_cast<std::size_t>(event_slot)].tape_cpu_data_for_backward();
        if (event_tape == nullptr) {
            throw std::runtime_error("ExpSyn spike_vjp_cpu: empty event-weight tape");
        }
        double* adj_g = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_slot)].get_cpu_data();
        uint32_t* spk_vec_idx = this->vecdata_spk_vec_idx->get_cpu_data();
        int* delay_steps = this->vecdata_delay_steps->get_cpu_data();
        const std::size_t base =
            static_cast<std::size_t>(param.step_index) * static_cast<std::size_t>(this->nnode);
        for (int i = 0; i < this->nnode; ++i) {
            const double event_weight = event_tape[base + static_cast<std::size_t>(i)];
            if (event_weight == 0.0 || adj_g[i] == 0.0) {
                continue;
            }
            const uint32_t src_idx = spk_vec_idx[i];
            const int delay_step = delay_steps != nullptr ? delay_steps[i] : 0;
            const int emit_step = param.step_index - (delay_step > 0 ? delay_step : 0);
            if (src_idx < static_cast<uint32_t>(param.pre_spike_adj_count) &&
                emit_step >= 0 && emit_step < param.step_count) {
                const std::size_t dst =
                    static_cast<std::size_t>(emit_step) *
                        static_cast<std::size_t>(param.pre_spike_adj_count) +
                    static_cast<std::size_t>(src_idx);
                param.pending_pre_spike_adj[dst] += adj_g[i] * event_weight;
            }
        }
    }

    void spike_vjp_gpu(SimPostSynSpikeVJPParam& param) override {
        if (this->nnode <= 0) {
            return;
        }
        constexpr int adj_slot = g_adjoint_slot();
        constexpr int event_slot = event_weight_tape_slot();
        static_assert(adj_slot >= 0, "ExpSyn.g must be registered as a VJP adjoint var");
        static_assert(event_slot >= 0, "ExpSyn.event_weight_cur must be in CurrentVjpTapeVars");
        this->ensure_default_vjp_adjoint_storage_();
        if (static_cast<int>(this->current_vjp_tape_stores_.size()) <= event_slot) {
            throw std::runtime_error("ExpSyn spike_vjp_gpu: current VJP tape stores are not initialized");
        }
        const auto replay =
            this->current_vjp_tape_stores_[static_cast<std::size_t>(event_slot)]
                .acquire_gpu_replay_view(param.step_index);
        double* adj_g = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_slot)].get_gpu_data();
        uint32_t* spk_vec_idx = this->vecdata_spk_vec_idx->get_gpu_data();
        int* delay_steps = this->vecdata_delay_steps->get_gpu_data();
        const int block_num = (this->nnode + nthread_per_block - 1) / nthread_per_block;
        cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(this->cuda_stream);
        expsyn_spike_vjp_kernel<<<block_num, nthread_per_block, 0, stream>>>(
            this->nnode,
            adj_g,
            replay.tape_gpu,
            replay.local_step_index,
            this->nnode,
            spk_vec_idx,
            delay_steps,
            param.pending_pre_spike_adj,
            param.pre_spike_adj_count,
            param.step_count,
            param.step_index);
    }
};

REGISTER_POSTSYN("ExpSyn", ExpSyn_Templated, 1);
}  // namespace ExpSynTemplated
