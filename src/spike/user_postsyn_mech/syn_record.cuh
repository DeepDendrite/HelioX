#pragma once
#include "postsyn_template.cuh"
#include <cmath>
#include <cassert>
#include <array>
#include <stdexcept>

namespace SynRecord {

__global__ void syn_record_spike_vjp_kernel(
    int nnode,
    const double* adj_ampa_a,
    const double* adj_ampa_b,
    const double* adj_nmda_a,
    const double* adj_nmda_b,
    const double* adj_gaba_a,
    const double* adj_gaba_b,
    const double* event_weight_tape,
    int event_tape_step,
    int event_tape_stride,
    const double* syn_w,
    const double* r_na,
    const double* ampa_factor,
    const double* nmda_factor,
    const double* gaba_factor,
    const uint32_t* spk_vec_idx,
    const int* delay_steps,
    double* pending_pre_spike_adj,
    int pre_spike_adj_count,
    int step_count,
    int arrival_step);

struct MechTrait {
    enum class VarNames {
        /* -------- parameter & assigned (0-28) -------- */
        AMPA_tau1,  AMPA_tau2,
        NMDA_tau1,  NMDA_tau2,
        GABA_tau1,  GABA_tau2,
        AMPA_e,     NMDA_e,     GABA_e,
        NMDA_C,     NMDA_rho,
        w, r_na, dv,
        Use_e, Use_i, Dep, Fac, u0,           // STP params（暂未用）
        i,                                     // 19
        AMPA_g, NMDA_g, GABA_g,                // 20-22
        v_prev, didv, pure_i, g_mul, dgdv, dgdg, // 23-28
        Use,                                   // 29 – 当前使用的 Use (简化)
        /* -------- state (30-35) -------- */
        AMPA_A, AMPA_B,
        NMDA_A, NMDA_B,
        GABA_A, GABA_B,
        /* -------- pre-computed factor (36-38) -------- */
        AMPA_factor, NMDA_factor, GABA_factor,
        event_weight_pending, event_weight_cur
    };
};

/*------------------------------------------------------------
 *  syn_record implementation
 *-----------------------------------------------------------*/
class SynRecord_Templated
        : public PostSynTemplate<SynRecord_Templated, MechTrait> {
    using Base = PostSynTemplate<SynRecord_Templated, MechTrait>;

    using enum MechTrait::VarNames;

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS | ENABLE_CURRENT_VJP;
    static constexpr auto LearnableVars = std::array{w};
    static constexpr auto VjpAdjointVars =
        std::array{AMPA_A, AMPA_B, NMDA_A, NMDA_B, GABA_A, GABA_B};
    static constexpr auto CurrentVjpTapeVars =
        std::array{v_prev, AMPA_g, NMDA_g, GABA_g, event_weight_cur};

    static consteval int w_learnable_slot() {
        return Base::template learnable_var_index_constexpr_<MechTrait::VarNames::w>();
    }
    static consteval int adj_ampa_a_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::AMPA_A>();
    }
    static consteval int adj_ampa_b_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::AMPA_B>();
    }
    static consteval int adj_nmda_a_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::NMDA_A>();
    }
    static consteval int adj_nmda_b_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::NMDA_B>();
    }
    static consteval int adj_gaba_a_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::GABA_A>();
    }
    static consteval int adj_gaba_b_slot() {
        return Base::template vjp_adjoint_var_index_constexpr_<MechTrait::VarNames::GABA_B>();
    }
    static consteval int event_weight_tape_slot() {
        return Base::template current_vjp_tape_var_index_constexpr_<MechTrait::VarNames::event_weight_cur>();
    }

    /* ---------------- ctor : register indexes ------------- */
    SynRecord_Templated(MechInitParams& param)
            : PostSynTemplate(param) {
        /* ---- 0‒18: parameters from model file ---- */
        var_in_coredata_idx.insert({AMPA_tau1, 0});
        var_in_coredata_idx.insert({AMPA_tau2, 1});
        var_in_coredata_idx.insert({NMDA_tau1, 2});
        var_in_coredata_idx.insert({NMDA_tau2, 3});
        var_in_coredata_idx.insert({GABA_tau1, 4});
        var_in_coredata_idx.insert({GABA_tau2, 5});
        var_in_coredata_idx.insert({AMPA_e, 6});
        var_in_coredata_idx.insert({NMDA_e, 7});
        var_in_coredata_idx.insert({GABA_e, 8});
        var_in_coredata_idx.insert({NMDA_C, 9});
        var_in_coredata_idx.insert({NMDA_rho, 10});
        var_in_coredata_idx.insert({w, 11});
        var_in_coredata_idx.insert({r_na, 12});
        var_in_coredata_idx.insert({dv, 13});
        var_in_coredata_idx.insert({Use_e, 14});
        var_in_coredata_idx.insert({Use_i, 15});
        var_in_coredata_idx.insert({Dep, 16});
        var_in_coredata_idx.insert({Fac, 17});
        var_in_coredata_idx.insert({u0, 18});
        /* ---- 19-28: assigned ---- */
        var_in_coredata_idx.insert({i,         19});
        var_in_coredata_idx.insert({AMPA_g,    20});
        var_in_coredata_idx.insert({NMDA_g,    21});
        var_in_coredata_idx.insert({GABA_g,    22});
        var_in_coredata_idx.insert({v_prev,    23});
        var_in_coredata_idx.insert({didv,      24});
        var_in_coredata_idx.insert({pure_i,    25});
        var_in_coredata_idx.insert({g_mul,     26});
        var_in_coredata_idx.insert({dgdv,      27});
        var_in_coredata_idx.insert({dgdg,      28});
        var_in_coredata_idx.insert({Use,       29});
        /* ---- 30-38: states + factors ---- */
        var_in_coredata_idx.insert({AMPA_A,       30});
        var_in_coredata_idx.insert({AMPA_B,       31});
        var_in_coredata_idx.insert({NMDA_A,       32});
        var_in_coredata_idx.insert({NMDA_B,       33});
        var_in_coredata_idx.insert({GABA_A,       34});
        var_in_coredata_idx.insert({GABA_B,       35});
        var_in_coredata_idx.insert({AMPA_factor,  36});
        var_in_coredata_idx.insert({NMDA_factor,  37});
        var_in_coredata_idx.insert({GABA_factor,  38});

        /* --- defaults identical to .mod --- */
        init_values.insert({w,     1.0});
        init_values.insert({r_na,  2.0});
        init_values.insert({dv,    1e-3});
        init_values.insert({Use_e, 1.0});
        init_values.insert({Use_i, 1.0});
        init_values.insert({Dep,   0.0});
        init_values.insert({Fac,   0.0});
        init_values.insert({u0,    0.0});
        init_values.insert({event_weight_pending, 0.0});
        init_values.insert({event_weight_cur, 0.0});
    }

    /* ----------------- helpers ----------------- */
    static __host__ __device__ __forceinline__ double sigma(double v, VarAccessor<MechTrait>& vars) {
        return 1.0 / (1.0 + vars(NMDA_C) * exp(-vars(NMDA_rho) * v));
    }

    static __host__ __device__ __forceinline__ double sigma_prime(double sigma_value,
                                                                  VarAccessor<MechTrait>& vars) {
        return vars(NMDA_rho) * sigma_value * (1.0 - sigma_value);
    }

    /* ----------------- INIT -------------------- */
    DUAL_EXEC void init_single_node(MechTempInitParam& param,
                                    VarAccessor<MechTrait>& vars) {
        using std::exp; using std::log;
        /* factor / tau sanity for each of the three exponentials */
        auto init_pair = [&](VarNames tau1, VarNames tau2,
                             VarNames A, VarNames B, VarNames factor) {
            double t1 = vars(tau1);
            double t2 = vars(tau2);
            if (t1 / t2 > 0.9999) { t1 = 0.9999 * t2; vars(tau1) = t1; }
            if (t1 / t2 < 1e-9)   { t1 = 1e-9  * t2; vars(tau1) = t1; }
            vars(A) = 0.0; vars(B) = 0.0;
            double tp = (t1 * t2) / (t2 - t1) * log(t2 / t1);
            vars(factor) = 1.0 / (-exp(-tp / t1) + exp(-tp / t2));
        };
        init_pair(AMPA_tau1, AMPA_tau2, AMPA_A, AMPA_B, AMPA_factor);
        init_pair(NMDA_tau1, NMDA_tau2, NMDA_A, NMDA_B, NMDA_factor);
        init_pair(GABA_tau1, GABA_tau2, GABA_A, GABA_B, GABA_factor);

        /* choose which Use to adopt (still simplified) */
        vars(Use) = (vars(w) > 0.0 ? vars(Use_e) : vars(Use_i));
        vars(i) = 0.0;
        vars(AMPA_g) = 0.0;
        vars(NMDA_g) = 0.0;
        vars(GABA_g) = 0.0;
        vars(v_prev) = param.volt;
        vars(didv) = 0.0;
        vars(pure_i) = 0.0;
        vars(g_mul) = 0.0;
        vars(dgdv) = 0.0;
        vars(dgdg) = 0.0;
        vars(event_weight_pending) = 0.0;
        vars(event_weight_cur) = 0.0;
    }

    /* ----------------- CURRENT ----------------- */
    DUAL_EXEC double current_single_node(MechTempCurParam& param,
                                         VarAccessor<MechTrait>& vars) {
        double v = param.volt;
        vars(v_prev) = v;                   // store for STATE step

        double current = 0.0;
        if (vars(w) > 0.0) {                // excitatory
            vars(AMPA_g) = vars(AMPA_B) - vars(AMPA_A);
            vars(NMDA_g) = vars(NMDA_B) - vars(NMDA_A);
            double s = SynRecord_Templated::sigma(v, vars);
            if (param.updateIon) {
                vars(pure_i) = vars(AMPA_g) * (v - vars(AMPA_e)) +
                               vars(NMDA_g) * s * (v - vars(NMDA_e));
                vars(event_weight_cur) = vars(event_weight_pending);
                vars(event_weight_pending) = 0.0;
            }
            current = fabs(vars(w)) *
                      (vars(AMPA_g) * (v - vars(AMPA_e)) +
                       vars(NMDA_g) * s * (v - vars(NMDA_e)));
        } else {                            // inhibitory (GABA)
            vars(GABA_g) = vars(GABA_B) - vars(GABA_A);
            if (param.updateIon) {
                vars(pure_i) = -vars(GABA_g) * (v - vars(GABA_e));
                vars(event_weight_cur) = vars(event_weight_pending);
                vars(event_weight_pending) = 0.0;
            }
            current = fabs(vars(w)) * vars(GABA_g) * (v - vars(GABA_e));
        }
        vars(i) = current;
        return current;
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param,
                                           VarAccessor<MechTrait> vars) {
        vars.idx = param.idx;
        constexpr int k_w_learnable_slot = w_learnable_slot();
        static_assert(k_w_learnable_slot >= 0, "syn_record.w must be registered as learnable");

        const double grad_i = param.grad_mech_current;
        const double v_t = tape_ref<v_prev>(param, vars);
        const double abs_w = fabs(vars(w));

        double& adj_ampa_a = adjoint_ref<AMPA_A>(param, vars);
        double& adj_ampa_b = adjoint_ref<AMPA_B>(param, vars);
        double& adj_nmda_a = adjoint_ref<NMDA_A>(param, vars);
        double& adj_nmda_b = adjoint_ref<NMDA_B>(param, vars);
        double& adj_gaba_a = adjoint_ref<GABA_A>(param, vars);
        double& adj_gaba_b = adjoint_ref<GABA_B>(param, vars);

        adj_ampa_a *= exp(-param.dt / vars(AMPA_tau1));
        adj_ampa_b *= exp(-param.dt / vars(AMPA_tau2));
        adj_nmda_a *= exp(-param.dt / vars(NMDA_tau1));
        adj_nmda_b *= exp(-param.dt / vars(NMDA_tau2));
        adj_gaba_a *= exp(-param.dt / vars(GABA_tau1));
        adj_gaba_b *= exp(-param.dt / vars(GABA_tau2));

        if (vars(w) > 0.0) {
            const double ampa_g_t = tape_ref<AMPA_g>(param, vars);
            const double nmda_g_t = tape_ref<NMDA_g>(param, vars);
            const double ampa_drive = v_t - vars(AMPA_e);
            const double nmda_drive = v_t - vars(NMDA_e);
            const double s = SynRecord_Templated::sigma(v_t, vars);
            const double ds = SynRecord_Templated::sigma_prime(s, vars);
            const double nmda_drive_eff = s * nmda_drive;
            const double pure = ampa_g_t * ampa_drive + nmda_g_t * nmda_drive_eff;

            adj_ampa_a += grad_i * abs_w * -ampa_drive;
            adj_ampa_b += grad_i * abs_w * ampa_drive;
            adj_nmda_a += grad_i * abs_w * -nmda_drive_eff;
            adj_nmda_b += grad_i * abs_w * nmda_drive_eff;
            mechAtomAdd(
                &param.grad_v[param.node_index],
                grad_i * abs_w * (ampa_g_t + nmda_g_t * (s + nmda_drive * ds)));
            mechAtomAdd(&grad_ref<w>(param, vars), grad_i * pure);
        } else {
            const double gaba_g_t = tape_ref<GABA_g>(param, vars);
            const double gaba_drive = v_t - vars(GABA_e);
            const double pure = -gaba_g_t * gaba_drive;

            adj_gaba_a += grad_i * abs_w * -gaba_drive;
            adj_gaba_b += grad_i * abs_w * gaba_drive;
            mechAtomAdd(&param.grad_v[param.node_index], grad_i * abs_w * gaba_g_t);
            mechAtomAdd(&grad_ref<w>(param, vars), grad_i * pure);
        }
    }

    /* ----------------- STATE ------------------- */
    DUAL_EXEC void state_single_node(MechTempStateParam& param,
                                     VarAccessor<MechTrait>& vars) {
        double dt = param.dt;
        auto decay = [&](VarNames X, VarNames tau) {
            vars(X) *= exp(-dt / vars(tau));
        };
        decay(AMPA_A, AMPA_tau1);  decay(AMPA_B, AMPA_tau2);
        decay(NMDA_A, NMDA_tau1);  decay(NMDA_B, NMDA_tau2);
        decay(GABA_A, GABA_tau1);  decay(GABA_B, GABA_tau2);

        /* 下列记录量仅用于梯度 / 诊断，可删减.
         * `pure_i` is intentionally left as the current-phase value written
         * during current_single_node(updateIon=true). Public monitor traces
         * should not be silently shifted to a state-phase
         * tape value here. */
        double v = param.volt;
        if (vars(w) > 0.0) {
            vars(didv)   = fabs(vars(w)) *
                           (vars(AMPA_g) +
                            vars(NMDA_g) * SynRecord_Templated::sigma(vars(v_prev), vars));
            vars(g_mul)  = fabs(vars(w)) *
                           vars(NMDA_g) * (v - vars(NMDA_e));
            vars(dgdv)   = (SynRecord_Templated::sigma(vars(v_prev) + vars(dv), vars) -
                            SynRecord_Templated::sigma(vars(v_prev), vars)) / vars(dv);
            vars(dgdg)   = 0.0;
        } else {
            vars(didv) = fabs(vars(w)) * vars(GABA_g);
            vars(g_mul) = vars(dgdv) = vars(dgdg) = 0.0;
        }
    }

    /* ----------------- NET_RECEIVE -------------- */
    DUAL_EXEC void net_receive_single_node(PostSynTempRecvParam& recv,
                                           VarAccessor<MechTrait>& vars) {
        /* —— simplified: ignore short-term plasticity —— */
        double w_event = recv.weight;           // NetCon weight (uS)
        vars(event_weight_pending) += w_event;
        if (vars(w) > 0.0) {                    // excitatory
            double ra = vars(r_na);
            double a_fac = w_event * vars(AMPA_factor) / (1.0 + ra);
            double n_fac = w_event * vars(NMDA_factor) * ra / (1.0 + ra);
            vars(AMPA_A) += a_fac;
            vars(AMPA_B) += a_fac;
            vars(NMDA_A) += n_fac;
            vars(NMDA_B) += n_fac;
        } else {                                // inhibitory
            double g_fac = w_event * vars(GABA_factor);
            vars(GABA_A) += g_fac;
            vars(GABA_B) += g_fac;
        }
    }

    bool supports_spike_vjp() const override { return true; }

    void spike_vjp_cpu(SimPostSynSpikeVJPParam& param) override {
        constexpr int event_slot = event_weight_tape_slot();
        static_assert(event_slot >= 0, "syn_record.event_weight_cur must be in CurrentVjpTapeVars");
        this->ensure_default_vjp_adjoint_storage_();
        if (static_cast<int>(this->current_vjp_tape_stores_.size()) <= event_slot) {
            throw std::runtime_error("syn_record spike_vjp_cpu: current VJP tape stores are not initialized");
        }
        const double* event_tape =
            this->current_vjp_tape_stores_[static_cast<std::size_t>(event_slot)].tape_cpu_data_for_backward();
        if (event_tape == nullptr) {
            throw std::runtime_error("syn_record spike_vjp_cpu: empty event-weight tape");
        }
        const std::size_t base =
            static_cast<std::size_t>(param.step_index) * static_cast<std::size_t>(this->nnode);

        double* adj_ampa_a = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_ampa_a_slot())].get_cpu_data();
        double* adj_ampa_b = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_ampa_b_slot())].get_cpu_data();
        double* adj_nmda_a = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_nmda_a_slot())].get_cpu_data();
        double* adj_nmda_b = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_nmda_b_slot())].get_cpu_data();
        double* adj_gaba_a = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_gaba_a_slot())].get_cpu_data();
        double* adj_gaba_b = this->vjp_adjoint_data_[static_cast<std::size_t>(adj_gaba_b_slot())].get_cpu_data();
        double* syn_w = this->var_struct[w]->get_cpu_data();
        double* r_na_data = this->var_struct[r_na]->get_cpu_data();
        double* ampa_factor_data = this->var_struct[AMPA_factor]->get_cpu_data();
        double* nmda_factor_data = this->var_struct[NMDA_factor]->get_cpu_data();
        double* gaba_factor_data = this->var_struct[GABA_factor]->get_cpu_data();
        uint32_t* spk_vec_idx = this->vecdata_spk_vec_idx->get_cpu_data();
        int* delay_steps = this->vecdata_delay_steps->get_cpu_data();

        for (int i = 0; i < this->nnode; ++i) {
            const double event_weight = event_tape[base + static_cast<std::size_t>(i)];
            if (event_weight == 0.0) {
                continue;
            }
            double local_adj = 0.0;
            if (syn_w[i] > 0.0) {
                const double ra = r_na_data[i];
                local_adj =
                    (adj_ampa_a[i] + adj_ampa_b[i]) * ampa_factor_data[i] / (1.0 + ra) +
                    (adj_nmda_a[i] + adj_nmda_b[i]) * nmda_factor_data[i] * ra / (1.0 + ra);
            } else {
                local_adj = (adj_gaba_a[i] + adj_gaba_b[i]) * gaba_factor_data[i];
            }
            if (local_adj == 0.0) {
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
                param.pending_pre_spike_adj[dst] += event_weight * local_adj;
            }
        }
    }

    void spike_vjp_gpu(SimPostSynSpikeVJPParam& param) override {
        if (this->nnode <= 0) {
            return;
        }
        constexpr int event_slot = event_weight_tape_slot();
        static_assert(event_slot >= 0, "syn_record.event_weight_cur must be in CurrentVjpTapeVars");
        this->ensure_default_vjp_adjoint_storage_();
        if (static_cast<int>(this->current_vjp_tape_stores_.size()) <= event_slot) {
            throw std::runtime_error("syn_record spike_vjp_gpu: current VJP tape stores are not initialized");
        }
        const auto replay =
            this->current_vjp_tape_stores_[static_cast<std::size_t>(event_slot)]
                .acquire_gpu_replay_view(param.step_index);
        const int block_num = (this->nnode + nthread_per_block - 1) / nthread_per_block;
        cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(this->cuda_stream);
        syn_record_spike_vjp_kernel<<<block_num, nthread_per_block, 0, stream>>>(
            this->nnode,
            this->vjp_adjoint_data_[static_cast<std::size_t>(adj_ampa_a_slot())].get_gpu_data(),
            this->vjp_adjoint_data_[static_cast<std::size_t>(adj_ampa_b_slot())].get_gpu_data(),
            this->vjp_adjoint_data_[static_cast<std::size_t>(adj_nmda_a_slot())].get_gpu_data(),
            this->vjp_adjoint_data_[static_cast<std::size_t>(adj_nmda_b_slot())].get_gpu_data(),
            this->vjp_adjoint_data_[static_cast<std::size_t>(adj_gaba_a_slot())].get_gpu_data(),
            this->vjp_adjoint_data_[static_cast<std::size_t>(adj_gaba_b_slot())].get_gpu_data(),
            replay.tape_gpu,
            replay.local_step_index,
            this->nnode,
            this->var_struct[w]->get_gpu_data(),
            this->var_struct[r_na]->get_gpu_data(),
            this->var_struct[AMPA_factor]->get_gpu_data(),
            this->var_struct[NMDA_factor]->get_gpu_data(),
            this->var_struct[GABA_factor]->get_gpu_data(),
            this->vecdata_spk_vec_idx->get_gpu_data(),
            this->vecdata_delay_steps->get_gpu_data(),
            param.pending_pre_spike_adj,
            param.pre_spike_adj_count,
            param.step_count,
            param.step_index);
    }
};

__global__ inline void syn_record_spike_vjp_kernel(
    int nnode,
    const double* adj_ampa_a,
    const double* adj_ampa_b,
    const double* adj_nmda_a,
    const double* adj_nmda_b,
    const double* adj_gaba_a,
    const double* adj_gaba_b,
    const double* event_weight_tape,
    int event_tape_step,
    int event_tape_stride,
    const double* syn_w,
    const double* r_na,
    const double* ampa_factor,
    const double* nmda_factor,
    const double* gaba_factor,
    const uint32_t* spk_vec_idx,
    const int* delay_steps,
    double* pending_pre_spike_adj,
    int pre_spike_adj_count,
    int step_count,
    int arrival_step) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned int>(nnode)) {
        return;
    }
    const double event_weight =
        event_weight_tape[static_cast<std::size_t>(event_tape_step) *
                              static_cast<std::size_t>(event_tape_stride) +
                          static_cast<std::size_t>(i)];
    if (event_weight == 0.0) {
        return;
    }
    double local_adj = 0.0;
    if (syn_w[i] > 0.0) {
        const double ra = r_na[i];
        local_adj =
            (adj_ampa_a[i] + adj_ampa_b[i]) * ampa_factor[i] / (1.0 + ra) +
            (adj_nmda_a[i] + adj_nmda_b[i]) * nmda_factor[i] * ra / (1.0 + ra);
    } else {
        local_adj = (adj_gaba_a[i] + adj_gaba_b[i]) * gaba_factor[i];
    }
    if (local_adj == 0.0) {
        return;
    }
    const uint32_t src_idx = spk_vec_idx[i];
    const int delay_step = delay_steps != nullptr ? delay_steps[i] : 0;
    const int emit_step = arrival_step - (delay_step > 0 ? delay_step : 0);
    if (src_idx < static_cast<uint32_t>(pre_spike_adj_count) &&
        emit_step >= 0 && emit_step < step_count) {
        const std::size_t dst =
            static_cast<std::size_t>(emit_step) * static_cast<std::size_t>(pre_spike_adj_count) +
            static_cast<std::size_t>(src_idx);
        atomicAdd(&pending_pre_spike_adj[dst], event_weight * local_adj);
    }
}

/* ----------- register with DeepDendrite ----------- */
REGISTER_POSTSYN("syn_record", SynRecord_Templated,5);

} // namespace SynRecord
