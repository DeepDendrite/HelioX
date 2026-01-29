#include "runtime_api/core/SimRuntimeCore.h"

#include <algorithm>
#include <magic_enum/magic_enum.hpp>

#include "cuda_utils.h"
#include "coredat_to_innerdat.h"
#include "global_vars.h"
#include "permute_order.h"
#include "spike/vecevent.h"

namespace heliox::runtime_api::core {

extern "C" void launch_batch_copy_doubles(double** d_gpu_ptrs, const double* d_cpu_values, int count);
extern "C" void launch_batch_gather_floats(double** d_gpu_ptrs, float* d_out, int count);

SimRuntimeCore::SimRuntimeCore() = default;
SimRuntimeCore::~SimRuntimeCore() {
    for (auto& [id, batch] : input_batches_) {
        release_input_batch_resources_(batch);
    }
    input_batches_.clear();

    if (gather_gpu_ptrs_device_ != nullptr) {
        gpu_mem_free((void**)&gather_gpu_ptrs_device_);
        gather_gpu_ptrs_device_ = nullptr;
    }
    if (gather_values_device_ != nullptr) {
        gpu_mem_free((void**)&gather_values_device_);
        gather_values_device_ = nullptr;
    }
    gather_capacity_ = 0;
}

int SimRuntimeCore::SimInitParam::get_permute_type() const {
    if (permute_type == -1) {
        return mode == GPU ? 3 : 0;
    }
    return permute_type;
}

double SimRuntimeCore::SimInitParam::get_dt() const {
    if (dt < 0) {
        double dt_from_file = coreneuron::global_var_map.at("dt")[0];
        printf("dt not set, using dt from global.dat: %f\n", dt_from_file);
        return dt_from_file;
    }
    return dt;
}

int SimRuntimeCore::set_data_path(const std::string& path) {
    if (sim_ != nullptr) {
        printf("Cannot set data path after simulator is initialized\n");
        return -1;
    }
    sim_param_.data_path = path;
    printf("Data path: %s\n", sim_param_.data_path.c_str());
    return 0;
}

int SimRuntimeCore::set_device(const std::string& dev) {
    if (dev == "cpu") {
        sim_param_.mode = CPU;
    } else if (dev == "gpu") {
        sim_param_.mode = GPU;
    } else {
        printf("Unknown device: %s\n", dev.c_str());
        return -1;
    }
    printf("Device: %s\n", dev.c_str());
    return 0;
}

int SimRuntimeCore::set_output_dir(const std::string& dir) {
    if (sim_ != nullptr) {
        printf("Output dir must be set before loading model\n");
        return -1;
    }
    sim_param_.output_dir = dir;
    return 0;
}

int SimRuntimeCore::set_permute_type(int type) {
    sim_param_.permute_type = type;
    printf("Permute type: %d\n", sim_param_.permute_type);
    return 0;
}

int SimRuntimeCore::set_dt(double dt) {
    if (sim_ == nullptr) {
        sim_param_.dt = dt;
    } else {
        sim_->dt = dt;
    }
    return 0;
}

double SimRuntimeCore::get_dt() const {
    if (sim_ == nullptr) {
        return sim_param_.get_dt();
    }
    return sim_->dt;
}

void SimRuntimeCore::set_user_mod_num(int num) {
    sim_param_.user_mod_num = num;
}

int SimRuntimeCore::load_model() {
    using enum BufferEnable;
    if (sim_ != nullptr) {
        printf("Simulate already initialized\n");
        return -1;
    }
    if (sim_param_.data_path.empty()) {
        printf("Data path not set\n");
        return -1;
    }

    std::string filesdat = sim_param_.data_path + "/files.dat";
    permute_type = sim_param_.get_permute_type();
    int ngroup = read_coredat(
        coredata_arr_, sim_param_.data_path.c_str(), filesdat.c_str(), 0, false, sim_param_.user_mod_num);

    BufferEnable buffer_enable = IPC;  // default: IPC only
    if (sim_param_.enable_hdf5) {
        buffer_enable = buffer_enable | HDF5;
    }

    sim_ = std::make_unique<Simulate>(sim_param_.mode, buffer_enable);
    if (!sim_param_.output_dir.empty()) {
        sim_->output_folder = sim_param_.output_dir;
    }

    sim_->tstop = -1;
    sim_->dt = sim_param_.get_dt();
    sim_->permute_type = sim_param_.get_permute_type();
    data_format_trans(sim_->neuron_group_list, coredata_arr_, ngroup, sim_param_.mode, sim_->dt);

    printf("Mode =%s permute_type=%d\n", magic_enum::enum_name(sim_param_.mode).data(), sim_->permute_type);

    std::vector<VarDescriptor> pre_monitors(sim_param_.pre_registered_monitors.begin(),
                                           sim_param_.pre_registered_monitors.end());
    monitor_to_handle_ = sim_->init_monitor_data_sets(pre_monitors);
    printf("Model loaded with %zu pre-registered monitors\n", pre_monitors.size());
    return 0;
}

int SimRuntimeCore::set_spike_output_enabled(bool enable) {
    spike_output_enabled_ = enable;
    printf("Spike file output %s\n", enable ? "enabled" : "disabled");
    return 0;
}

bool SimRuntimeCore::is_spike_output_enabled() const {
    return spike_output_enabled_;
}

int SimRuntimeCore::add_monitor(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return add_monitor_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::add_monitor_with_array(const std::string& mech,
                                           const std::string& var,
                                           int node_or_mech_idx,
                                           int array_index) {
    VarDescriptor monitor;
    monitor.mech = mech;
    monitor.var = var;
    monitor.node_or_mech_idx = node_or_mech_idx;
    monitor.array_index = array_index;

    if (sim_ != nullptr) {
        auto [var_ptr_cpu, var_ptr_gpu] = sim_->getVarPtr(monitor, false);
        if (var_ptr_cpu == nullptr) {
            printf("Monitor variable not found: %s %s %d\n", monitor.mech.c_str(), monitor.var.c_str(),
                   monitor.node_or_mech_idx);
            return -1;
        }

        RecordPoint recordPoint;
        recordPoint.var_ptr_cpu = var_ptr_cpu;
        recordPoint.var_ptr_gpu = var_ptr_gpu;

        int handle = sim_->hdf5_manager.push_back(monitor, recordPoint);
        monitor_to_handle_[monitor] = handle;
        return handle;
    }

    // Model not loaded yet: preregister, dedup via set.
    if (sim_param_.pre_registered_monitors.find(monitor) != sim_param_.pre_registered_monitors.end()) {
        printf("Monitor already pre-registered: %s %s %d -> pending\n", monitor.mech.c_str(), monitor.var.c_str(),
               monitor.node_or_mech_idx);
        return -1;
    }
    sim_param_.pre_registered_monitors.insert(monitor);
    return -1;
}

int SimRuntimeCore::get_monitor_handle(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return get_monitor_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::get_monitor_handle_with_array(const std::string& mech,
                                                  const std::string& var,
                                                  int node_or_mech_idx,
                                                  int array_index) {
    VarDescriptor monitor;
    monitor.mech = mech;
    monitor.var = var;
    monitor.node_or_mech_idx = node_or_mech_idx;
    monitor.array_index = array_index;

    if (sim_ == nullptr) {
        printf("Model not loaded, cannot get monitor handle\n");
        return -1;
    }

    auto it = monitor_to_handle_.find(monitor);
    if (it != monitor_to_handle_.end()) {
        return it->second;
    }

    return -1;
}

int SimRuntimeCore::flush_recorders() {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    if (sim_->mode == CPU) {
        sim_->hdf5_manager.flush_cpu();
        return 0;
    }
    sim_->hdf5_manager.flush_gpu();
    return 0;
}

std::vector<double> SimRuntimeCore::get_monitor_data(int handle) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return {};
    }
    if (handle == -1) {
        printf("Handle is -1, cannot get data for pending monitor\n");
        return {};
    }
    auto search_result = sim_->hdf5_manager.get_single_irq_buffer(handle);
    if (search_result.has_value()) {
        auto& var_vector = search_result.value().get();
        return var_vector;
    }
    return {};
}

std::map<int, std::vector<double>> SimRuntimeCore::get_multiple_monitor_data(const std::vector<int>& handles) {
    std::map<int, std::vector<double>> result;
    for (int h : handles) {
        result[h] = get_monitor_data(h);
    }
    return result;
}

std::vector<std::vector<std::string>> SimRuntimeCore::get_vecplay_keys() {
    std::vector<std::vector<std::string>> result;
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot get vecplay keys\n");
        return result;
    }
    try {
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        auto keys = vecplay_continuous.getAllKeys();
        for (const auto& key : keys) {
            std::vector<std::string> key_str = {key.mech_name, key.var_name, std::to_string(key.instance_id)};
            result.push_back(key_str);
        }
        return result;
    } catch (const std::exception& e) {
        printf("Failed to get vecplay keys: %s\n", e.what());
        return result;
    }
}

static bool validate_vecplay_tvec(const std::vector<double>& tvec) {
    for (size_t i = 0; i < tvec.size(); i++) {
        if (tvec[i] < 0) {
            return false;
        }
        if (i > 0 && tvec[i] < tvec[i - 1]) {
            return false;
        }
    }
    return true;
}

int SimRuntimeCore::add_vecplay(const std::string& mech_name,
                                const std::string& var_name,
                                int instance_id,
                                const std::vector<double>& tvec,
                                const std::vector<double>& yvec) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot add vecplay\n");
        return -1;
    }
    if (tvec.size() != yvec.size()) {
        printf("tvec and yvec must have the same size\n");
        return -1;
    }
    if (tvec.empty()) {
        printf("tvec and yvec cannot be empty\n");
        return -1;
    }
    if (!validate_vecplay_tvec(tvec)) {
        printf("Invalid time vector for vecplay\n");
        return -1;
    }
    try {
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        vecplay_continuous.addVecPlay(mech_name, var_name, instance_id, tvec, yvec);
        printf("Added vecplay for %s.%s[%d] with %zu time points\n", mech_name.c_str(), var_name.c_str(), instance_id,
               tvec.size());
        return 0;
    } catch (const std::exception& e) {
        printf("Failed to add vecplay: %s\n", e.what());
        return -1;
    }
}

int SimRuntimeCore::update_vecplay(const std::string& mech_name,
                                   const std::string& var_name,
                                   int instance_id,
                                   const std::vector<double>& new_tvec,
                                   const std::vector<double>& new_yvec) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot update vecplay\n");
        return -1;
    }
    if (new_tvec.size() != new_yvec.size()) {
        printf("new_tvec and new_yvec must have the same size\n");
        return -1;
    }
    if (new_tvec.empty()) {
        printf("new_tvec and new_yvec cannot be empty\n");
        return -1;
    }
    if (!validate_vecplay_tvec(new_tvec)) {
        printf("Invalid time vector for vecplay\n");
        return -1;
    }
    try {
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        vecplay_continuous.updateVecPlay(key, new_tvec, new_yvec);
        printf("Updated vecplay for %s.%s[%d] with %zu time points\n", mech_name.c_str(), var_name.c_str(), instance_id,
               new_tvec.size());
        return 0;
    } catch (const std::exception& e) {
        printf("Failed to update vecplay: %s\n", e.what());
        return -1;
    }
}

int SimRuntimeCore::remove_vecplay(const std::string& mech_name, const std::string& var_name, int instance_id) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot remove vecplay\n");
        return -1;
    }
    try {
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        vecplay_continuous.removeVecPlay(key);
        return 0;
    } catch (const std::exception& e) {
        printf("Failed to remove vecplay: %s\n", e.what());
        return -1;
    }
}

bool SimRuntimeCore::has_vecplay(const std::string& mech_name, const std::string& var_name, int instance_id) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot check vecplay\n");
        return false;
    }
    try {
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        return vecplay_continuous.hasVecPlay(key);
    } catch (const std::exception& e) {
        printf("Failed to check vecplay: %s\n", e.what());
        return false;
    }
}

int SimRuntimeCore::get_variable_handle(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return get_variable_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::get_variable_handle_with_array(const std::string& mech,
                                                   const std::string& var,
                                                   int node_or_mech_idx,
                                                   int array_index) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    VarDescriptor desc;
    desc.mech = mech;
    desc.var = var;
    desc.node_or_mech_idx = node_or_mech_idx;
    desc.array_index = array_index;

    auto [cpu_var_ptr, gpu_var_ptr] = sim_->getVarPtr(desc, true);
    if (cpu_var_ptr == nullptr && gpu_var_ptr == nullptr) {
        printf("Variable not found: %s %s %d\n", mech.c_str(), var.c_str(), node_or_mech_idx);
        return -1;
    }
    auto it = cpu_ptr_to_handle_.find(cpu_var_ptr);
    if (it != cpu_ptr_to_handle_.end()) {
        return it->second;
    }

    VarPointer vp;
    vp.cpu_ptr = cpu_var_ptr;
    vp.gpu_ptr = gpu_var_ptr;
    vp.cached_cpu_value = 0.0;
    vp.is_dirty = false;

    int handle = static_cast<int>(var_pointer_cache_.size());
    var_pointer_cache_.push_back(vp);
    cpu_ptr_to_handle_[cpu_var_ptr] = handle;
    return handle;
}

bool SimRuntimeCore::get_cached_pointers(int handle, double*& cpu_ptr, double*& gpu_ptr) const {
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        return false;
    }
    const VarPointer& vp = var_pointer_cache_[handle];
    cpu_ptr = vp.cpu_ptr;
    gpu_ptr = vp.gpu_ptr;
    return true;
}

double SimRuntimeCore::get_variable_by_handle(int handle) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return 0.0;
    }
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("Invalid handle: %d\n", handle);
        return 0.0;
    }
    VarPointer& vp = var_pointer_cache_[handle];
    if (sim_->mode == CPU) {
        if (vp.cpu_ptr == nullptr) {
            printf("CPU pointer is null for handle %d\n", handle);
            return 0.0;
        }
        return *vp.cpu_ptr;
    }
    if (vp.is_dirty) {
        return vp.cached_cpu_value;
    }
    if (vp.gpu_ptr == nullptr) {
        printf("GPU pointer is null for handle %d\n", handle);
        return 0.0;
    }
    double val = 0.0;
    mem_copy_gpu2cpu(&val, vp.gpu_ptr, sizeof(double));
    return val;
}

int SimRuntimeCore::set_variable_by_handle(int handle, double value) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("Invalid handle: %d\n", handle);
        return -1;
    }
    VarPointer& vp = var_pointer_cache_[handle];
    if (sim_->mode == CPU) {
        if (vp.cpu_ptr == nullptr) {
            printf("CPU pointer is null for handle %d\n", handle);
            return -1;
        }
        *vp.cpu_ptr = value;
        return 0;
    }
    if (vp.cpu_ptr == nullptr) {
        printf("CPU pointer is null for handle %d\n", handle);
        return -1;
    }
    *vp.cpu_ptr = value;
    vp.cached_cpu_value = value;
    vp.is_dirty = true;
    dirty_handles_.insert(handle);
    return 0;
}

int SimRuntimeCore::set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values) {
    if (handles.size() != values.size()) {
        printf("set_variables_by_handles: handle/value size mismatch (%zu vs %zu)\n", handles.size(), values.size());
        return -1;
    }
    for (size_t i = 0; i < handles.size(); ++i) {
        if (set_variable_by_handle(handles[i], values[i]) < 0) {
            return -1;
        }
    }
    return 0;
}

void SimRuntimeCore::flush_dirty_variables() {
    if (sim_ == nullptr || sim_->mode != GPU || dirty_handles_.empty()) {
        return;
    }

    int count = static_cast<int>(dirty_handles_.size());
    std::vector<double> cpu_values;
    std::vector<double*> gpu_ptrs;
    cpu_values.reserve(count);
    gpu_ptrs.reserve(count);

    for (int handle : dirty_handles_) {
        VarPointer& vp = var_pointer_cache_[handle];
        cpu_values.push_back(vp.cached_cpu_value);
        gpu_ptrs.push_back(vp.gpu_ptr);
        vp.is_dirty = false;
    }

    double* d_cpu_values = nullptr;
    double** d_gpu_ptrs = nullptr;
    gpu_mem_allocate((void**)&d_cpu_values, count * sizeof(double));
    gpu_mem_allocate((void**)&d_gpu_ptrs, count * sizeof(double*));
    mem_copy_cpu2gpu_sync(d_cpu_values, cpu_values.data(), count * sizeof(double));
    mem_copy_cpu2gpu_sync(d_gpu_ptrs, gpu_ptrs.data(), count * sizeof(double*));
    launch_batch_copy_doubles(d_gpu_ptrs, d_cpu_values, count);
    gpu_mem_free((void**)&d_cpu_values);
    gpu_mem_free((void**)&d_gpu_ptrs);
    dirty_handles_.clear();
}

int SimRuntimeCore::get_variables_by_handles_f32(const std::vector<int>& handles, float* out_cpu, int count) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (count <= 0) {
        return 0;
    }
    if (sim_->mode != GPU) {
        // CPU fallback: fill by scalar reads.
        for (int i = 0; i < count; ++i) {
            out_cpu[i] = static_cast<float>(get_variable_by_handle(handles[i]));
        }
        return 0;
    }

    flush_dirty_variables();

    if (count > gather_capacity_) {
        if (gather_gpu_ptrs_device_ != nullptr) {
            gpu_mem_free((void**)&gather_gpu_ptrs_device_);
            gather_gpu_ptrs_device_ = nullptr;
        }
        if (gather_values_device_ != nullptr) {
            gpu_mem_free((void**)&gather_values_device_);
            gather_values_device_ = nullptr;
        }
        gpu_mem_allocate((void**)&gather_gpu_ptrs_device_, count * sizeof(double*));
        gpu_mem_allocate((void**)&gather_values_device_, count * sizeof(float));
        gather_capacity_ = count;
    }

    std::vector<double*> gpu_ptrs;
    gpu_ptrs.resize(count);
    for (int i = 0; i < count; ++i) {
        const int handle = handles[i];
        if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
            gpu_ptrs[i] = nullptr;
            continue;
        }
        VarPointer& vp = var_pointer_cache_[handle];
        gpu_ptrs[i] = vp.gpu_ptr;
    }

    mem_copy_cpu2gpu_sync(gather_gpu_ptrs_device_, gpu_ptrs.data(), count * sizeof(double*));
    launch_batch_gather_floats(gather_gpu_ptrs_device_, gather_values_device_, count);
    mem_copy_gpu2cpu(out_cpu, gather_values_device_, count * sizeof(float));
    for (int i = 0; i < count; ++i) {
        if (gpu_ptrs[i] == nullptr) {
            out_cpu[i] = 0.0f;
        }
    }
    return 0;
}

int SimRuntimeCore::set_variable_value(double val, const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return set_variable_value_with_array(val, mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::set_variable_value_with_array(
    double val, const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    VarDescriptor desc;
    desc.mech = mech;
    desc.var = var;
    desc.node_or_mech_idx = node_or_mech_idx;
    desc.array_index = array_index;
    auto [cpu_ptr, gpu_ptr] = sim_->getVarPtr(desc, false);
    if (cpu_ptr == nullptr) {
        printf("Variable not found: %s %s %d\n", mech.c_str(), var.c_str(), node_or_mech_idx);
        return -1;
    }
    *cpu_ptr = val;
    if (sim_->mode == GPU && gpu_ptr != nullptr) {
        mem_copy_cpu2gpu_sync(gpu_ptr, cpu_ptr, sizeof(double));
    }
    return 0;
}

double SimRuntimeCore::get_variable_value(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return get_variable_value_with_array(mech, var, node_or_mech_idx, 0);
}

double SimRuntimeCore::get_variable_value_with_array(
    const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return 0.0;
    }
    VarDescriptor desc;
    desc.mech = mech;
    desc.var = var;
    desc.node_or_mech_idx = node_or_mech_idx;
    desc.array_index = array_index;
    auto [cpu_ptr, gpu_ptr] = sim_->getVarPtr(desc, false);
    if (cpu_ptr == nullptr) {
        printf("Variable not found: %s %s %d\n", mech.c_str(), var.c_str(), node_or_mech_idx);
        return 0.0;
    }
    if (sim_->mode == GPU && gpu_ptr != nullptr) {
        double val = 0.0;
        mem_copy_gpu2cpu(&val, gpu_ptr, sizeof(double));
        return val;
    }
    return *cpu_ptr;
}

SimRuntimeCore::InputStimBatch* SimRuntimeCore::find_input_batch_(int batch_id) {
    auto it = input_batches_.find(batch_id);
    if (it == input_batches_.end()) {
        printf("Input batch %d not found\n", batch_id);
        return nullptr;
    }
    return &it->second;
}

void SimRuntimeCore::release_input_batch_resources_(InputStimBatch& batch) {
    if (batch.net_device.interval_ptrs != nullptr) {
        gpu_mem_free((void**)&batch.net_device.interval_ptrs);
        batch.net_device.interval_ptrs = nullptr;
    }
    if (batch.net_device.start_ptrs != nullptr) {
        gpu_mem_free((void**)&batch.net_device.start_ptrs);
        batch.net_device.start_ptrs = nullptr;
    }
    if (batch.net_device.number_ptrs != nullptr) {
        gpu_mem_free((void**)&batch.net_device.number_ptrs);
        batch.net_device.number_ptrs = nullptr;
    }
    if (batch.net_device.interval_values != nullptr) {
        gpu_mem_free((void**)&batch.net_device.interval_values);
        batch.net_device.interval_values = nullptr;
    }
    if (batch.net_device.start_values != nullptr) {
        gpu_mem_free((void**)&batch.net_device.start_values);
        batch.net_device.start_values = nullptr;
    }
    if (batch.net_device.number_values != nullptr) {
        gpu_mem_free((void**)&batch.net_device.number_values);
        batch.net_device.number_values = nullptr;
    }
}

int SimRuntimeCore::register_netstim_batch(const std::vector<std::tuple<int, int, int>>& handle_triplets,
                                          double interval_scale,
                                          double start_base,
                                          double epsilon,
                                          double number) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (handle_triplets.empty()) {
        printf("register_netstim_batch: handle list is empty\n");
        return -1;
    }

    InputStimBatch batch;
    batch.type = InputStimBatch::Type::NetStim;
    batch.net_params.interval_scale = interval_scale;
    batch.net_params.start_base = start_base;
    batch.net_params.epsilon = epsilon;
    batch.net_params.number = number;
    batch.mode = sim_->mode;
    batch.expected_size = handle_triplets.size();
    batch.net_entries.reserve(handle_triplets.size());

    if (sim_->mode == Mode::GPU) {
        batch.net_interval_values.resize(handle_triplets.size(), 0.0);
        batch.net_start_values.resize(handle_triplets.size(), 0.0);
        batch.net_number_values.resize(handle_triplets.size(), number);
        batch.net_interval_gpu_ptrs.reserve(handle_triplets.size());
        batch.net_start_gpu_ptrs.reserve(handle_triplets.size());
        batch.net_number_gpu_ptrs.reserve(handle_triplets.size());
    }

    for (size_t idx = 0; idx < handle_triplets.size(); ++idx) {
        auto [interval_h, start_h, number_h] = handle_triplets[idx];
        NetStimEntry entry;
        entry.interval_handle = interval_h;
        entry.start_handle = start_h;
        entry.number_handle = number_h;

        double* interval_cpu = nullptr;
        double* interval_gpu = nullptr;
        double* start_cpu = nullptr;
        double* start_gpu = nullptr;
        double* number_cpu = nullptr;
        double* number_gpu = nullptr;
        if (!get_cached_pointers(interval_h, interval_cpu, interval_gpu) ||
            !get_cached_pointers(start_h, start_cpu, start_gpu) ||
            !get_cached_pointers(number_h, number_cpu, number_gpu)) {
            printf("register_netstim_batch: invalid handle at index %zu\n", idx);
            release_input_batch_resources_(batch);
            return -1;
        }
        if (interval_cpu == nullptr || start_cpu == nullptr || number_cpu == nullptr) {
            printf("register_netstim_batch: missing CPU pointer at index %zu\n", idx);
            release_input_batch_resources_(batch);
            return -1;
        }

        entry.interval_cpu = interval_cpu;
        entry.interval_gpu = interval_gpu;
        entry.start_cpu = start_cpu;
        entry.start_gpu = start_gpu;
        entry.number_cpu = number_cpu;
        entry.number_gpu = number_gpu;

        batch.net_entries.push_back(entry);

        if (sim_->mode == Mode::GPU) {
            if (entry.interval_gpu == nullptr || entry.start_gpu == nullptr || entry.number_gpu == nullptr) {
                printf("register_netstim_batch: missing GPU pointer at index %zu\n", idx);
                release_input_batch_resources_(batch);
                return -1;
            }
            batch.net_interval_gpu_ptrs.push_back(entry.interval_gpu);
            batch.net_start_gpu_ptrs.push_back(entry.start_gpu);
            batch.net_number_gpu_ptrs.push_back(entry.number_gpu);
        }
    }

    if (sim_->mode == Mode::GPU) {
        int count = static_cast<int>(batch.net_entries.size());
        gpu_mem_allocate((void**)&batch.net_device.interval_ptrs, count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**)&batch.net_device.start_ptrs, count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**)&batch.net_device.number_ptrs, count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**)&batch.net_device.interval_values, count * static_cast<int>(sizeof(double)));
        gpu_mem_allocate((void**)&batch.net_device.start_values, count * static_cast<int>(sizeof(double)));
        gpu_mem_allocate((void**)&batch.net_device.number_values, count * static_cast<int>(sizeof(double)));

        mem_copy_cpu2gpu_sync(batch.net_device.interval_ptrs, batch.net_interval_gpu_ptrs.data(),
                              count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(batch.net_device.start_ptrs, batch.net_start_gpu_ptrs.data(),
                              count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(batch.net_device.number_ptrs, batch.net_number_gpu_ptrs.data(),
                              count * static_cast<int>(sizeof(double*)));
    }

    int batch_id = next_input_batch_id_++;
    input_batches_.emplace(batch_id, std::move(batch));
    return batch_id;
}

int SimRuntimeCore::register_vecstim_batch(const std::vector<int>& mech_indices,
                                          double spike_scale,
                                          double start_base,
                                          double epsilon,
                                          int spike_count) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (mech_indices.empty()) {
        printf("register_vecstim_batch: mech list is empty\n");
        return -1;
    }

    InputStimBatch batch;
    batch.type = InputStimBatch::Type::VecStim;
    batch.vec_params.spike_scale = spike_scale;
    batch.vec_params.start_base = start_base;
    batch.vec_params.epsilon = epsilon;
    batch.vec_params.spike_count = spike_count;
    batch.mode = sim_->mode;
    batch.expected_size = mech_indices.size();
    batch.vec_spike_buffer.resize(spike_count);
    batch.vec_entries.reserve(mech_indices.size());

    for (size_t idx = 0; idx < mech_indices.size(); ++idx) {
        if (mech_indices[idx] < 0) {
            printf("register_vecstim_batch: invalid mech index at %zu\n", idx);
            return -1;
        }
        VecStimEntry entry;
        entry.mech_index = mech_indices[idx];
        batch.vec_entries.push_back(entry);
    }

    int batch_id = next_input_batch_id_++;
    input_batches_.emplace(batch_id, std::move(batch));
    return batch_id;
}

int SimRuntimeCore::set_input_batch_pixels(int batch_id, std::span<const double> pixels) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    InputStimBatch* batch = find_input_batch_(batch_id);
    if (batch == nullptr) {
        return -1;
    }
    if (pixels.size() != batch->expected_size) {
        printf("set_input_batch_pixels: size mismatch batch=%zu, got=%zu\n",
               batch->expected_size, pixels.size());
        return -1;
    }

    if (batch->type == InputStimBatch::Type::NetStim) {
        const auto count = batch->net_entries.size();
        if (count == 0) {
            return 0;
        }
        for (size_t i = 0; i < count; ++i) {
            const double pixel = pixels[i];
            const double denom = std::max(pixel + batch->net_params.epsilon, batch->net_params.epsilon);
            const double interval = batch->net_params.interval_scale / denom;
            const double start = batch->net_params.start_base + interval;
            const double number = batch->net_params.number;

            NetStimEntry& entry = batch->net_entries[i];
            if (entry.interval_cpu == nullptr || entry.start_cpu == nullptr || entry.number_cpu == nullptr) {
                printf("set_input_batch_pixels: missing CPU pointer at index %zu\n", i);
                return -1;
            }

            *entry.interval_cpu = interval;
            *entry.start_cpu = start;
            *entry.number_cpu = number;

            if (batch->mode == Mode::GPU) {
                batch->net_interval_values[i] = interval;
                batch->net_start_values[i] = start;
                batch->net_number_values[i] = number;
            }
        }

        if (batch->mode == Mode::GPU) {
            const int count_i = static_cast<int>(count);
            mem_copy_cpu2gpu_sync(batch->net_device.interval_values, batch->net_interval_values.data(),
                                  count_i * static_cast<int>(sizeof(double)));
            mem_copy_cpu2gpu_sync(batch->net_device.start_values, batch->net_start_values.data(),
                                  count_i * static_cast<int>(sizeof(double)));
            mem_copy_cpu2gpu_sync(batch->net_device.number_values, batch->net_number_values.data(),
                                  count_i * static_cast<int>(sizeof(double)));
            launch_batch_copy_doubles(batch->net_device.interval_ptrs, batch->net_device.interval_values, count_i);
            launch_batch_copy_doubles(batch->net_device.start_ptrs, batch->net_device.start_values, count_i);
            launch_batch_copy_doubles(batch->net_device.number_ptrs, batch->net_device.number_values, count_i);
        }
        return 0;
    }

    if (VecEvent::getInstance() == nullptr) {
        printf("VecStim instance not ready\n");
        return -1;
    }

    const int spike_count = batch->vec_params.spike_count;
    batch->vec_spike_buffer.resize(spike_count);

    for (size_t i = 0; i < pixels.size(); ++i) {
        const double pixel = pixels[i];
        const double denom = std::max(pixel + batch->vec_params.epsilon, batch->vec_params.epsilon);
        const double step = batch->vec_params.spike_scale / denom;

        for (int k = 0; k < spike_count; ++k) {
            batch->vec_spike_buffer[k] = batch->vec_params.start_base + step * static_cast<double>(k + 1);
        }

        VecEvent::update_sequence(batch->mode,
                                  batch->vec_entries[i].mech_index,
                                  batch->vec_spike_buffer.data(),
                                  spike_count);
    }
    return 0;
}

}  // namespace heliox::runtime_api::core
