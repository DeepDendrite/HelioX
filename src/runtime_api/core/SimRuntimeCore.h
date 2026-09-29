#pragma once

#include <map>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "celltemplate/celltemplate_builder.hpp"
#include "coredat_structs.h"
#include "neuron.h"
#include "read_coredat.h"
#include "runtime_api/core/MechPluginManager.h"
#include "simulate.h"

namespace neurong::runtime_api::core {

// Core runtime state + methods (simulation control + monitors + basic IO).
//
// Phase 2: This class is introduced to migrate "core" logic out of the Python binding layer.
// It must not depend on nanobind/Python headers.
class SimRuntimeCore final {
public:
    enum class ModelSource {
        Unset = 0,
        Neuron,
        Neurong,
    };

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
        int recorder_buffer_capacity = 1000;
        bool recorder_use_fp32 = false;
        double recorder_start_time_ms = 0.0;
        int recorder_stride_steps = 1;
        bool enable_hdf5 = false;

        // Pre-registered monitors (dedup via set).
        std::set<VarDescriptor> pre_registered_monitors;
        std::set<VarDescriptor> pre_registered_window_observers;

        int get_permute_type() const;
        double get_dt() const;
    };

    // Access to underlying simulation (used by learn layer until migrated).
    Simulate* sim() { return sim_.get(); }
    const Simulate* sim() const { return sim_.get(); }

    // core: control/config
    int set_data_path(const std::string& path);
    int set_device(const std::string& dev);
    int set_source(const std::string& source);
    std::string get_source() const;
    int set_output_dir(const std::string& dir);
    int set_permute_type(int type);
    int set_dt(double dt);
    double get_dt() const;
    int load_mech_library(const std::string& path);
    int register_ion_meta(const std::string& ion_name,
                          double charge,
                          double default_conci = 1.0,
                          double default_conco = 1.0);
    std::vector<std::string> get_loaded_mech_libraries() const;
    void set_user_mod_num(int num);
    int set_recorder_buffer_capacity(int capacity);
    int set_recorder_storage_dtype(const std::string& dtype);
    int set_recorder_start_time(double start_time_ms);
    int set_recorder_stride(int stride_steps);

    int load_model();
    std::vector<std::tuple<std::string, int, int>> load_celltemplate_morphology(
        const std::vector<neurong_celltemplate::CellTemplateMorphSpec>& templates,
        std::optional<std::uint64_t> random_seed = std::nullopt);
    std::vector<std::tuple<int, int, int>> apply_celltemplate_biophysics(
        const std::vector<neurong_celltemplate::CellTemplateBiophysSpec>& templates,
        double celsius,
        const std::vector<int>& record_gids);
    std::vector<std::tuple<std::string, int, int>> get_celltemplate_gid_ranges() const;
    const neurong_biophysical::CellTemplateMorphLayout* in_memory_morph_layout() const;
    bool is_in_memory_biophysics_applied() const;
    int resolve_node_index(int gid, const std::string& section_name, int seg_index) const;
    int resolve_segment_index_by_loc(int gid, const std::string& section_name, double loc) const;
    int resolve_node_index_by_loc(int gid, const std::string& section_name, double loc) const;
    int resolve_segment_node_index_by_loc(int gid, const std::string& section_name, double loc);
    int resolve_mech_index_by_loc(const std::string& mech,
                                  int gid,
                                  const std::string& section_name,
                                  double loc,
                                  int slot = 0);
    double resolve_segment_center_loc_by_loc(int gid, const std::string& section_name, double loc) const;
    double resolve_effective_loc_by_loc(const std::string& mech,
                                        int gid,
                                        const std::string& section_name,
                                        double loc) const;

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
    int get_monitor_sample_count(int handle);
    std::vector<double> get_monitor_data(int handle);
    std::map<int, std::vector<double>> get_multiple_monitor_data(const std::vector<int>& handles);
    int get_multiple_monitor_data_f32(const std::vector<int>& handles, float* out_cpu, int n_handle, int n_step);
    int get_multiple_monitor_data_f32_from(const std::vector<int>& handles,
                                           int start_step,
                                           float* out_cpu,
                                           int n_handle,
                                           int n_step);
    int add_window_observer(const std::string& mech, const std::string& var, int node_or_mech_idx);
    int add_window_observer_with_array(const std::string& mech,
                                       const std::string& var,
                                       int node_or_mech_idx,
                                       int array_index);
    int get_window_observer_handle(const std::string& mech, const std::string& var, int node_or_mech_idx);
    int get_window_observer_handle_with_array(const std::string& mech,
                                              const std::string& var,
                                              int node_or_mech_idx,
                                              int array_index);
    int begin_window_observer_recording(const std::vector<int>& handles, int step_count);
    int end_window_observer_recording();
    int get_window_observer_data_f32(const std::vector<int>& handles, float* out_cpu, int n_handle, int n_step);

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
    int get_variable_handle_with_array(const std::string& mech,
                                       const std::string& var,
                                       int index,
                                       int array_index = 0);
    int get_variable_handle_with_array_by_loc(const std::string& mech,
                                              const std::string& var,
                                              int gid,
                                              const std::string& section_name,
                                              double loc,
                                              int array_index = 0,
                                              int slot = 0);
    double get_handle_resolved_loc(int handle) const;
    int get_handle_resolved_segment_index(int handle) const;
    double get_variable_by_handle(int handle);
    int set_variable_by_handle(int handle, double value);
    int set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values);
    int get_variables_by_handles_f32(const std::vector<int>& handles, float* out_cpu, int count);
    int get_learnable_grad_handle_from_variable_handle(int variable_handle);
    std::vector<int> get_learnable_grad_handles_from_variable_handles(const std::vector<int>& variable_handles);

    // core: direct descriptor value access (slow path)
    int set_variable_value(double val, const std::string& mech, const std::string& var, int node_or_mech_idx);
    int set_variable_value_with_array(
        double val, const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index);
    double get_variable_value(const std::string& mech, const std::string& var, int node_or_mech_idx);
    double get_variable_value_with_array(const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index);

    // core: expose pointers for internal glue (input batches/replay until migrated)
    bool get_cached_pointers(int handle, double*& cpu_ptr, double*& gpu_ptr) const;
    int resolve_node_index_internal_from_original(int node_idx_original) const;
    int resolve_node_index_permuted_from_handle(int handle) const;
    int resolve_mech_index_permuted_from_handle(int handle) const;
    void flush_dirty_variables();
    std::tuple<double, int> debug_get_postsyn_delay_info(const std::string& mech, int instance_index) const;

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
    int register_vecstim_batch_by_locs(const std::string& mech,
                                       const std::vector<int>& gids,
                                       const std::vector<std::string>& section_names,
                                       const std::vector<double>& locs,
                                       double spike_scale,
                                       double start_base,
                                       double epsilon,
                                       int spike_count);
    int set_input_batch_pixels(int batch_id, std::span<const double> pixels);

private:
    int maybe_autoload_default_mech_library_();
    std::unique_ptr<Simulate> sim_;
    MechPluginManager mech_plugin_manager_;
    SimInitParam sim_param_;
    ModelSource model_source_ = ModelSource::Unset;
    bool default_mech_autoload_attempted_ = false;
    std::optional<neurong_biophysical::CellTemplateMorphLayout> neurong_morph_layout_;
    std::vector<std::tuple<std::string, int, int>> neurong_template_gid_ranges_;
    bool neurong_biophys_applied_ = false;
    int neurong_next_type_ = 0;
    std::map<VarDescriptor, int> monitor_to_handle_;
    std::map<VarDescriptor, int> window_observer_to_handle_;
    std::unique_ptr<coreneuron::CoreData*[]> coredata_arr_;
    bool spike_output_enabled_ = false;

    struct VarPointer {
        double* cpu_ptr = nullptr;
        double* gpu_ptr = nullptr;
        VarDescriptor descriptor{};
        double cached_cpu_value = 0.0;
        bool is_dirty = false;
        bool has_resolved_loc = false;
        double resolved_loc = std::numeric_limits<double>::quiet_NaN();
        int resolved_segment_index = -1;
    };
    std::vector<VarPointer> var_pointer_cache_;
    std::unordered_map<double*, int> cpu_ptr_to_handle_;
    std::vector<int> dirty_handles_;
    std::vector<double> flush_cpu_values_;
    std::vector<double*> flush_gpu_ptrs_;
    double* flush_values_device_ = nullptr;
    double** flush_gpu_ptrs_device_ = nullptr;
    int flush_capacity_ = 0;
    std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>> mech_permuted_node_to_inst_;

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
    int get_variable_handle_by_descriptor_internal_(const std::string& mech,
                                                    const std::string& var,
                                                    int index,
                                                    int array_index);
    bool build_mech_node_index_cache_(const std::string& mech);
    int resolve_mech_index_by_node_internal_(const std::string& mech,
                                             int node_idx_original,
                                             int slot);
    int resolve_segment_node_index_by_loc_internal_(int gid,
                                                    const std::string& section_name,
                                                    double loc);
    bool resolve_segment_info_by_loc_internal_(int gid,
                                               const std::string& section_name,
                                               double loc,
                                               int* out_seg_idx,
                                               int* out_nseg,
                                               int* out_node_idx_original = nullptr) const;
    int resolve_mech_index_by_loc_internal_(const std::string& mech,
                                            int gid,
                                            const std::string& section_name,
                                            double loc,
                                            int slot,
                                            double* out_resolved_loc = nullptr,
                                            int* out_resolved_segment_index = nullptr);
};

}  // namespace neurong::runtime_api::core
