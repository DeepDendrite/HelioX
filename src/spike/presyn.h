#ifndef presyn_h
#define presyn_h

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <array>
#include "vecdata.h"
#include "spike.h"
#include "utils.h"
#include <vector>

// pre-synapse for realistic cells (those gid >= 0)
class PreSyn
{
    public:
        uint32_t npre;//真实presyn的个数
        PreSyn(uint32_t n, SpikeVector* vec);
        PreSyn(Mode mode, uint32_t n, SpikeVector* vec, 
               const vector<int> &pre_node_indices, 
               const vector<double> &threshold, 
               const vector<uint32_t> &spk_vec_offset,
               const vector<int> &pre_gids);

        ~PreSyn();
        VecData<int>* vecdata_gids;
        unordered_map<int, int>* gid_map;
        VecData<vector<double>>* vecdata_spk_output;
        VecData<int>* vecdata_pre_node_indices; // node index in vec_v,从coreDat的thvar_index来
        // VecData<uint32_t>* vecdata_post_syn_type; // type of all post synapses//ljf:useless now
        // VecData<uint32_t>* vecdata_post_synid; // index in post synapses (instance id) //ljf:useless now
        VecData<double> *vecdata_threshold;//从coreNeuron读来的
        // VecData<double> *vecdata_spike_deliver_time;//ljf useless now
        VecData<bool>* vecdata_pre_flags;//用来记录脉冲发放的

        SpikeVector* spk_vec_bkp; // 指向NeuronData中的 spike vector 
        // start index in spike vector
        VecData<uint32_t>* vecdata_spk_vec_offset;//在真实细胞中的index
        VecData<int> * vecdata_spk_idx_vec;//用于存放返回值，记录哪些index的presyn发生了Spk
        // number of post synapses
        // VecData<uint32_t>* vecdata_spk_vec_cnt;//ljf：useless now

        // detect whether the nodes fire,and add fired spikes to spike buffer.
        void threshold_detect_cpu(double* vec_v, SpikeFlag* spk_flags, double t, vector<pair<double, int>> &rec_spk);
        int threshold_detect_gpu(double* vec_v, VecData<SpikeFlag>* spk_flags, double t, vector<pair<double, int>> &rec_spk);

        // Optional profiling (disabled by default; keep overhead minimal when off).
        void set_spike_profile_enabled(bool enable) { spike_profile_enabled_ = enable; }
        bool is_spike_profile_enabled() const { return spike_profile_enabled_; }

        struct SpikeProfileStats {
            uint64_t steps = 0;
            uint64_t steps_with_presyn_spike = 0;
            uint64_t presyn_spike_total = 0;
            int presyn_spike_max = 0;
            // Histogram for presyn spikes per step:
            // bucket[i] = count of steps with exactly i spikes, for i in [0,63]
            // bucket[63] also accumulates ">=63" to keep it bounded.
            std::array<uint64_t, 64> presyn_spike_hist{};
        };

        const SpikeProfileStats& spike_profile_stats() const { return spike_profile_stats_; }
        void reset_spike_profile_stats() { spike_profile_stats_ = SpikeProfileStats{}; }

        // Spike detect (GPU) fast path:
        // - Use a non-blocking CUDA stream + event for spike detection work.
        // - Only copy back full spk flags / indices when spk_num_real > 0.
        // - Avoid device-wide synchronization in the hot loop.
        void* spike_stream = nullptr; // owns a cudaStream_t*
        void* spike_event = nullptr;  // owns a cudaEvent_t*

        int* cudaMappedMem = nullptr; // pinned host buffer for below 2 ints
        int* spk_num_real = nullptr;  // host-pinned: number of presyn spikes at current step
        int* spk_num_tot = nullptr;   // host-pinned: total spikes (legacy semantics)
        int* d_spk_num_real = nullptr; // device counter
        int* d_spk_num_tot = nullptr;  // device counter

    private:
        bool spike_profile_enabled_ = false;
        SpikeProfileStats spike_profile_stats_{};
};

#endif
