#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coredat_structs.h"
#include "neuron.h"
#include "read_coredat.h"
#include "simulate.h"

namespace heliox::runtime_api::core {

// Core runtime state + methods (simulation control + monitors + basic IO).
//
// Phase 2: This class is introduced to migrate "core" logic out of the Python binding layer.
// It must not depend on nanobind/Python headers.
class SimRuntimeCore final {
public:
    SimRuntimeCore();
    ~SimRuntimeCore();

    SimRuntimeCore(const SimRuntimeCore&) = delete;
    SimRuntimeCore& operator=(const SimRuntimeCore&) = delete;
    SimRuntimeCore(SimRuntimeCore&&) = default;
    SimRuntimeCore& operator=(SimRuntimeCore&&) = default;

    struct SimInitParam {
        Mode mode = GPU;
        int permute_type = -1;
        double dt = -1;
        std::string data_path;
        std::string output_dir;
        int user_mod_num = 1000;
        bool enable_hdf5 = false;

        // Pre-registered monitors (dedup via set).
        std::set<VarDescriptor> pre_registered_monitors;

        int get_permute_type() const;
        double get_dt() const;
    };

    // Access to underlying simulation (used by learn layer until migrated).
    Simulate* sim() { return sim_.get(); }
    const Simulate* sim() const { return sim_.get(); }

    // core: control/config
    int set_data_path(const std::string& path);
    int set_device(const std::string& dev);
    int set_output_dir(const std::string& dir);
    int set_permute_type(int type);
    int set_dt(double dt);
    double get_dt() const;
    void set_user_mod_num(int num);

    int load_model();

    // core: spike output flag (kept here for unified runtime config)
    int set_spike_output_enabled(bool enable);
    bool is_spike_output_enabled() const;

    // core: monitor registration
    int add_monitor(const std::string& mech, const std::string& var, int node_or_mech_idx);
    int add_monitor_with_array(const std::string& mech,
                               const std::string& var,
                               int node_or_mech_idx,
                               int array_index);

    // core: monitor handle mapping
    int get_monitor_handle(const std::string& mech, const std::string& var, int node_or_mech_idx);
    int get_monitor_handle_with_array(const std::string& mech,
                                      const std::string& var,
                                      int node_or_mech_idx,
                                      int array_index);

    // core: recorder IO
    int flush_recorders();
    std::vector<double> get_monitor_data(int handle);
    std::map<int, std::vector<double>> get_multiple_monitor_data(const std::vector<int>& handles);

    // core: VecPlay control (continuous playback)
    int add_vecplay(const std::string& mech_name,
                    const std::string& var_name,
                    int instance_id,
                    const std::vector<double>& tvec,
                    const std::vector<double>& yvec);
    int update_vecplay(const std::string& mech_name,
                       const std::string& var_name,
                       int instance_id,
                       const std::vector<double>& new_tvec,
                       const std::vector<double>& new_yvec);
    int remove_vecplay(const std::string& mech_name, const std::string& var_name, int instance_id);
    bool has_vecplay(const std::string& mech_name, const std::string& var_name, int instance_id);
    std::vector<std::vector<std::string>> get_vecplay_keys();

    // core: variable handle cache
    int get_variable_handle(const std::string& mech, const std::string& var, int node_or_mech_idx);
    int get_variable_handle_with_array(const std::string& mech,
                                       const std::string& var,
                                       int node_or_mech_idx,
                                       int array_index);
    double get_variable_by_handle(int handle);
    int set_variable_by_handle(int handle, double value);
    int set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values);
    int get_variables_by_handles_f32(const std::vector<int>& handles, float* out_cpu, int count);

    // core: direct descriptor value access (slow path)
    int set_variable_value(double val, const std::string& mech, const std::string& var, int node_or_mech_idx);
    int set_variable_value_with_array(
        double val, const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index);
    double get_variable_value(const std::string& mech, const std::string& var, int node_or_mech_idx);
    double get_variable_value_with_array(const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index);

    // core: expose pointers for internal glue (input batches/replay until migrated)
    bool get_cached_pointers(int handle, double*& cpu_ptr, double*& gpu_ptr) const;
    void flush_dirty_variables();

    // core: batch input stimulation (NetStim / VecStim)
    int register_netstim_batch(const std::vector<std::tuple<int, int, int>>& handle_triplets,
                               double interval_scale,
                               double start_base,
                               double epsilon,
                               double number);
    int register_vecstim_batch(const std::vector<int>& mech_indices,
                               double spike_scale,
                               double start_base,
                               double epsilon,
                               int spike_count);
    int set_input_batch_pixels(int batch_id, std::span<const double> pixels);

private:
    std::unique_ptr<Simulate> sim_;
    SimInitParam sim_param_;
    std::map<VarDescriptor, int> monitor_to_handle_;
    std::unique_ptr<coreneuron::CoreData*[]> coredata_arr_;
    bool spike_output_enabled_ = false;

    struct VarPointer {
        double* cpu_ptr = nullptr;
        double* gpu_ptr = nullptr;
        double cached_cpu_value = 0.0;
        bool is_dirty = false;
    };
    std::vector<VarPointer> var_pointer_cache_;
    std::unordered_map<double*, int> cpu_ptr_to_handle_;
    std::unordered_set<int> dirty_handles_;

    // Batched gather buffers (GPU) for fast reads by handle.
    double** gather_gpu_ptrs_device_ = nullptr;
    float* gather_values_device_ = nullptr;
    int gather_capacity_ = 0;

    // Input stimulation batches (used by both simulation and training frontends).
    struct NetStimEntry {
        int interval_handle = -1;
        int start_handle = -1;
        int number_handle = -1;
        double* interval_cpu = nullptr;
        double* interval_gpu = nullptr;
        double* start_cpu = nullptr;
        double* start_gpu = nullptr;
        double* number_cpu = nullptr;
        double* number_gpu = nullptr;
    };

    struct VecStimEntry {
        int mech_index = -1;
    };

    struct NetStimDeviceBuffers {
        double** interval_ptrs = nullptr;
        double** start_ptrs = nullptr;
        double** number_ptrs = nullptr;
        double* interval_values = nullptr;
        double* start_values = nullptr;
        double* number_values = nullptr;
    };

    struct InputStimBatch {
        enum class Type { NetStim, VecStim };
        Type type = Type::NetStim;
        struct NetParams {
            double interval_scale = 5.0;
            double start_base = 9.0;
            double epsilon = 0.01;
            double number = 100.0;
        } net_params;
        struct VecParams {
            double spike_scale = 5.0;
            double start_base = 9.0;
            double epsilon = 0.01;
            int spike_count = 20;
        } vec_params;
        std::vector<NetStimEntry> net_entries;
        std::vector<VecStimEntry> vec_entries;
        std::vector<double> net_interval_values;
        std::vector<double> net_start_values;
        std::vector<double> net_number_values;
        std::vector<double*> net_interval_gpu_ptrs;
        std::vector<double*> net_start_gpu_ptrs;
        std::vector<double*> net_number_gpu_ptrs;
        NetStimDeviceBuffers net_device;
        std::vector<double> vec_spike_buffer;
        Mode mode = Mode::CPU;
        size_t expected_size = 0;
    };

    int next_input_batch_id_ = 0;
    std::unordered_map<int, InputStimBatch> input_batches_;
    InputStimBatch* find_input_batch_(int batch_id);
    void release_input_batch_resources_(InputStimBatch& batch);
};

}  // namespace heliox::runtime_api::core
