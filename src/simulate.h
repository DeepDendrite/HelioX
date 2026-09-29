#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <highfive/highfive.hpp>
#include "neuron.h"
#include "utils.h"
#include "variable_recorder.h"
#include "window_observer.h"
#include "optimizer/optimizer.h"
#include <unordered_map>
#include <unordered_set>
#include "autodiff/axial_rhs_vjp.hpp"

using namespace std;

// class that control simulation
// contains functions used during a simulation (e.g. finitialize, run)
class Simulate
{
    public:
        Simulate(Mode mode = CPU,
                 BufferEnable buf_enable = BufferEnable::HDF5,
                 int buffer_size = 1000,
                 bool recorder_use_fp32 = false);
        ~Simulate();
        void fadvance();
        void fadvance_cpu();
        void fadvance_gpu();
        void finitialize(double v_init);
        void run();
        void continue_run(double runtime);
        void output_spikes();
        std::map<VarDescriptor, int> init_monitor_data_sets(std::vector<VarDescriptor> &monitors);
        std::map<VarDescriptor, int> init_window_observers(std::vector<VarDescriptor>& observers);
        vector<NeuronGroupData*> neuron_group_list;
        // Maintained hand-written VJP lane:
        // inject local row gradients and run the transpose/VJP recurrence.
        void prepare_vjp(
            int group_index,
            const std::vector<int>& local_grad_rows,
            int step_count,
            const std::vector<std::string>& current_vjp_mechs);
        void run_vjp(
            const std::vector<std::vector<double>>& local_grad_row_history);
        void run_vjp_forward_grads_contiguous(
            const double* local_grad_forward,
            int local_count,
            int step_count);
        void set_spike_vjp_surrogate(const std::string& kind, double width_mv);

        pair<double*, double*> getVarPtr(VarDescriptor &descriptor, bool will_panic = false);

        // 以DC线性矩阵方式计算转移阻抗（节点索引为内部布局索引：已permute）。
        int compute_transfer_impedance_dc_nodes(int source_node_idx_internal,
                                                const std::vector<int>& target_node_idx_internal,
                                                double inject_amp,
                                                double v_init,
                                                std::vector<double>& out_impedance);

        // Optimizer 接口
        int create_optimizer(OptimizerType type);
        int register_optimizer_param(int optimizer_id,
                                     double* weight_cpu,
                                     double* grad_cpu,
                                     double* weight_gpu,
                                     double* grad_gpu,
                                     double impedance);
        int register_optimizer_param_batch(int optimizer_id,
                                           const std::vector<double*>& weight_cpu,
                                           const std::vector<double*>& grad_cpu,
                                           const std::vector<double*>& weight_gpu,
                                           const std::vector<double*>& grad_gpu,
                                           double impedance);
        int configure_optimizer(int optimizer_id, const OptimizerHyperParams& params);
        int optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt);
        int set_recorder_sampling(double start_time_ms, int stride_steps);
        int begin_window_observer_recording(const std::vector<int>& handles, int step_count);
        int end_window_observer_recording();
        int optimizer_reset_state(int optimizer_id);
        int optimizer_get_adam_state(int optimizer_id,
                                     long long& step_count,
                                     std::vector<double>& m,
                                     std::vector<double>& v,
                                     OptimizerHyperParams& params);
        int optimizer_set_adam_state(int optimizer_id,
                                     long long step_count,
                                     const std::vector<double>& m,
                                     const std::vector<double>& v,
                                     const OptimizerHyperParams& params);

        double t;
        double dt;
		double tstop;
        int permute_type;

        Mode mode;

        string output_folder;
        unique_ptr<HighFive::File> hdf5_file;
        VariableRecorder hdf5_manager;  // 变量记录器（旧名hdf5_manager保留兼容）
        WindowObserver window_observer;
        bool is_finitialized() const { return finitialize_done_; }

    private:
        bool finitialize_done_ = false;
        bool autodiff_record_vjp_tape_ = false;
        std::unordered_set<std::string> autodiff_current_vjp_mech_whitelist_;
        int autodiff_vjp_tape_recorded_steps_ = 0;
        int autodiff_vjp_tape_step_count_ = 0;
        std::string spike_vjp_surrogate_name_ = "triangle";
        neurong::spike_vjp::SpikeVjpSurrogateConfig spike_vjp_surrogate_config_;
        void arm_vjp_tape_recording_(int group_index);
        bool should_stage_vjp_tape_(const NeuronGroupData* group) const;
        void note_vjp_tape_staged_(const NeuronGroupData* group);
        void clear_current_vjp_tapes_();
        bool ensure_finitialized_(const char* fn_name) const;
        bool prepared_vjp_valid_ = false;
        int prepared_vjp_group_index_ = -1;
        void prepare_vjp_(
            int group_index,
            const std::vector<int>& local_grad_rows,
            int step_count,
            const std::vector<std::string>& current_vjp_mechs);
        void validate_vjp_cpu_solver_topology_(NeuronGroupData* group) const;
        void solve_vjp_matrix_cpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace);
        void run_vjp_step_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            const double* local,
            int local_count,
            int step);
        void run_vjp_step_forward_contiguous_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            const double* local_grad_forward,
            int local_count,
            int step_count,
            int step);
        void run_vjp_step_after_local_accumulation_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            int step);
        double* resolve_vjp_adjoint_ptr_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            const VarDescriptor& descriptor,
            Mode ptr_mode);
        void prepare_gap_vjp_routes_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace);
        void zero_vjp_adjoint_carries_(NeuronGroupData* group);
        void gap_vjp_transfer_cpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace);
        void gap_vjp_transfer_gpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace);
        void record_spike_vjp_presyn_tape_cpu_(NeuronGroupData* group);
        void record_spike_vjp_presyn_tape_gpu_(NeuronGroupData* group);
        void spike_vjp_deposit_cpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            int tape_step);
        void spike_vjp_consume_cpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            int tape_step);
        void spike_vjp_deposit_gpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            int tape_step);
        void spike_vjp_consume_gpu_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            int tape_step);
        void finalize_vjp_tape_for_backward_(
            NeuronGroupData::VoltageVjpWorkspace& workspace);
        void run_prepared_vjp_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            const std::vector<std::vector<double>>& local_grad_row_history);
        void run_prepared_vjp_forward_contiguous_(
            NeuronGroupData* group,
            NeuronGroupData::VoltageVjpWorkspace& workspace,
            const double* local_grad_forward,
            int local_count,
            int step_count);
        // Recorder sampling policy:
        // - start_time_ms: first time (inclusive) to start recording.
        // - stride_steps: 1 records every step, 2 records every other step, etc.
        double recorder_start_time_ms_ = 0.0;
        int recorder_stride_steps_ = 1;
        int64_t recorder_eligible_step_count_ = 0;
        bool should_record_now_();
        void record_if_needed_cpu_();
        void record_if_needed_gpu_();

        // Optimizer 管理
        int next_optimizer_id = 0;
        std::unordered_map<int, std::unique_ptr<OptimizerBase>> optimizers;
        
        vector<pair<double, int> > rec_spikes;
        void record_output_spikes_cpu(NeuronGroupData* p_neuron);
        void record_output_spikes_gpu(NeuronGroupData* p_neuron);

        void finitialize_cpu(double v_init);
        void spike_deliver_cpu();
        void setup_tree_matrix_cpu(NeuronGroupData* p_neuron);
        void solve_matrix_cpu(NeuronGroupData* p_neuron);
        void update_cpu(NeuronGroupData* p_neuron);
        void last_part_cpu();
        void nonvint_cpu(NeuronGroupData* p_neuron);
        void network_spike_send_cpu();
        void network_spike_receive_cpu();

        void finitialize_gpu(double v_init);
        void spike_deliver_gpu();
        void setup_tree_matrix_gpu(NeuronGroupData* p_neuron);
        void solve_matrix_gpu(NeuronGroupData* p_neuron);
        void update_gpu(NeuronGroupData* p_neuron);
        void last_part_gpu();
        void nonvint_gpu(NeuronGroupData* p_neuron);
        void network_spike_send_gpu();
        void network_spike_receive_gpu();
        void clearValidSpkFlags(VecData<SpikeFlag> *vecdata_spk_flags, bool cleanAll = false);

        void gap_transfer_cpu(NeuronGroupData *p_group);
        void gap_transfer_gpu(NeuronGroupData *p_group);

        //finitialize的时候调用，以清空之前的spike
        void clearAllSpikes_cpu();
        void clearAllSpikes_gpu();

};
