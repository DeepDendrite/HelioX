#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <highfive/highfive.hpp>
#include "neuron.h"
#include "utils.h"
#include "variable_recorder.h"
#include "optimizer/optimizer.h"
#include <unordered_map>

using namespace std;

// class that control simulation
// contains functions used during a simulation (e.g. finitialize, run)
class Simulate
{
    public:
        Simulate(Mode mode = CPU,BufferEnable buf_enable = BufferEnable::HDF5, int buffer_size = 1000);
        ~Simulate();
        void fadvance();
        void fadvance_cpu();
        void fadvance_gpu();
        void finitialize(double v_init);
        void run();
        void continue_run(double runtime);
        void output_spikes();
        std::map<VarDescriptor, int> init_monitor_data_sets(std::vector<VarDescriptor> &monitors);
        vector<HelioXroupData*> neuron_group_list;

        pair<double*, double*> getVarPtr(VarDescriptor &descriptor, bool will_panic = false);

        // Gap Junction 管理
        struct GapJunctionMeta {
            VarDescriptor source;           // 源变量
            vector<VarDescriptor> targets;  // 目标变量列表
        };

        // Gap Junction API
        int add_gap_source(const string& mech, const string& var, int idx, int sid = -1);  // sid=-1表示自动分配
        int add_gap_target(int sid, const string& mech, const string& var, int idx);
        int clear_all_gap_junctions();
        const map<int, GapJunctionMeta>& get_all_gap_junctions() const { return gap_junctions; }
        GapJunctionMeta* get_gap_junction(int sid);
        int get_next_available_sid() { return next_sid_counter++; }

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
        int optimizer_step_with_inv_record_steps(int optimizer_id, double learning_rate, double inv_record_steps);
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

    private:
        // Optional spike profiling (env: HELIOX_PROFILE_SPIKE=1).
        bool spike_profile_enabled_ = false;
        uint64_t net_receive_called_ = 0;
        uint64_t net_receive_skipped_ = 0;

        void maybe_init_spike_profile_();
        void print_spike_profile_summary_() const;

        // Gap Junction管理
        map<int, GapJunctionMeta> gap_junctions;  // sid -> gap元数据
        int next_sid_counter = 0;  // 下一个可用的sid，从0开始

        // Optimizer 管理
        int next_optimizer_id = 0;
        std::unordered_map<int, std::unique_ptr<OptimizerBase>> optimizers;
        
        vector<pair<double, int> > rec_spikes;
        void record_output_spikes_cpu(HelioXroupData* p_neuron);
        void record_output_spikes_gpu(HelioXroupData* p_neuron);

        void finitialize_cpu(double v_init);
        void spike_deliver_cpu();
        void setup_tree_matrix_cpu(HelioXroupData* p_neuron);
        void solve_matrix_cpu(HelioXroupData* p_neuron);
        void update_cpu(HelioXroupData* p_neuron);
        void last_part_cpu();
        void nonvint_cpu(HelioXroupData* p_neuron);
        void network_spike_send_cpu();
        void network_spike_receive_cpu();

        void finitialize_gpu(double v_init);
        void spike_deliver_gpu();
        void setup_tree_matrix_gpu(HelioXroupData* p_neuron);
        void solve_matrix_gpu(HelioXroupData* p_neuron);
        void update_gpu(HelioXroupData* p_neuron);
        void last_part_gpu();
        void nonvint_gpu(HelioXroupData* p_neuron);
        void network_spike_send_gpu();
        void network_spike_receive_gpu();
        void clearValidSpkFlags(VecData<SpikeFlag> *vecdata_spk_flags, bool cleanAll = false);

        void gap_transfer_cpu(HelioXroupData *p_group);
        void gap_transfer_gpu(HelioXroupData *p_group);

        //finitialize的时候调用，以清空之前的spike
        void clearAllSpikes_cpu();
        void clearAllSpikes_gpu();

};
