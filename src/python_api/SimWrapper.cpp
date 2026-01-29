#include "SimWrapper.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <cuda_runtime.h>

SimWrapper::SimWrapper() {
    printf("NeuronSimulator created\n");
}

SimWrapper::~SimWrapper() {
    clear_dense_blocks();
    printf("NeuronSimulator destroyed\n");
}

int SimWrapper::set_data_path(const string& path) {
    return core_.set_data_path(path);
}

int SimWrapper::set_device(const string& dev) {
    return core_.set_device(dev);
}

int SimWrapper::load_model() {
    int rc = core_.load_model();
    sim = core_.sim();
    return rc;
}

int SimWrapper::finitialize(double v_init) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    
    // 在初始化前同步脏变量到GPU
    core_.flush_dirty_variables();
    
    // printf("v_init: %f\n", v_init);
    sim->finitialize(v_init);
    // printf("finitialize finished\n");
    return 0;
}

int SimWrapper::run(double tstop) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    
    // 在运行前同步脏变量到GPU
    core_.flush_dirty_variables();
    
    sim->tstop = tstop;
    // printf("tstop: %f\n", tstop);

    auto start_time = clock();
    sim->run();
    auto end_time = clock();
    // printf("simulation finished, time:%fs\n", ((float)end_time - start_time) / CLOCKS_PER_SEC);

    if (core_.is_spike_output_enabled()) {
        sim->output_spikes();
    }
    return 0;
}

int SimWrapper::continue_run(double runtime) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    // 在运行前同步脏变量到GPU
    core_.flush_dirty_variables();

    sim->continue_run(runtime);

    if (core_.is_spike_output_enabled()) {
        sim->output_spikes();
    }
    return 0;
}

int SimWrapper::fadvance() {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    // step前同步脏变量到GPU（例如用户通过handle写入了参数）
    core_.flush_dirty_variables();
    sim->fadvance();
    return 0;
}

double SimWrapper::get_t() const {
    if (sim == nullptr) {
        return 0.0;
    }
    return sim->t;
}

int SimWrapper::flush_recorders() {
    return core_.flush_recorders();
}

int SimWrapper::set_spike_output_enabled(bool enable) {
    return core_.set_spike_output_enabled(enable);
}

bool SimWrapper::is_spike_output_enabled() const {
    return core_.is_spike_output_enabled();
}

int SimWrapper::set_permute_type(int type) {
    return core_.set_permute_type(type);
}

int SimWrapper::add_monitor(const string& mech, const string& var, int node_or_mech_idx) {
    return core_.add_monitor(mech, var, node_or_mech_idx);
}

int SimWrapper::add_monitor_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.add_monitor_with_array(mech, var, node_or_mech_idx, array_index);
}

int SimWrapper::set_dt(double dt) {
    return core_.set_dt(dt);
}

double SimWrapper::get_dt() {
    return core_.get_dt();
}

int SimWrapper::set_output_dir(const string& dir) {
    return core_.set_output_dir(dir);
}

vector<double> SimWrapper::get_monitor_data(int handle) {
    return core_.get_monitor_data(handle);
}

map<int, vector<double>> SimWrapper::get_multiple_monitor_data(const vector<int>& handles) {
    return core_.get_multiple_monitor_data(handles);
}


double SimWrapper::get_variable_value(const string& mech, const string& var, int node_or_mech_idx){
    return core_.get_variable_value(mech, var, node_or_mech_idx);
}

double SimWrapper::get_variable_value_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index){
    return core_.get_variable_value_with_array(mech, var, node_or_mech_idx, array_index);
}

int SimWrapper::create_optimizer(const string& optimizer_type) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    std::string lower = optimizer_type;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    OptimizerType type;
    if (lower == "sgd") {
        type = OptimizerType::SGD;
    } else if (lower == "momentum" || lower == "sgdmomentum" || lower == "sgd_momentum") {
        type = OptimizerType::Momentum;
    } else if (lower == "adam") {
        type = OptimizerType::Adam;
    } else {
        printf("Unsupported optimizer type: %s\n", optimizer_type.c_str());
        return -1;
    }
    return sim->create_optimizer(type);
}

int SimWrapper::optimizer_add_param(int optimizer_id, int weight_handle, int grad_handle, double impedance) {
    std::vector<int> weight_handles{weight_handle};
    std::vector<int> grad_handles{grad_handle};
    return optimizer_add_param_batch(optimizer_id, weight_handles, grad_handles, impedance);
}

int SimWrapper::optimizer_add_param_batch(int optimizer_id,
                                          const std::vector<int>& weight_handles,
                                          const std::vector<int>& grad_handles,
                                          double impedance) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (weight_handles.size() != grad_handles.size() || weight_handles.empty()) {
        printf("optimizer_add_param_batch: handle lists must be non-empty and have equal length\n");
        return -1;
    }

    std::vector<double*> weight_cpu;
    std::vector<double*> grad_cpu;
    std::vector<double*> weight_gpu;
    std::vector<double*> grad_gpu;
    weight_cpu.reserve(weight_handles.size());
    grad_cpu.reserve(weight_handles.size());
    weight_gpu.reserve(weight_handles.size());
    grad_gpu.reserve(weight_handles.size());

    for (size_t i = 0; i < weight_handles.size(); ++i) {
        int weight_handle = weight_handles[i];
        int grad_handle = grad_handles[i];

        double* weight_cpu_ptr = nullptr;
        double* weight_gpu_ptr = nullptr;
        double* grad_cpu_ptr = nullptr;
        double* grad_gpu_ptr = nullptr;
        if (!core_.get_cached_pointers(weight_handle, weight_cpu_ptr, weight_gpu_ptr)) {
            printf("optimizer_add_param_batch: invalid weight handle %d\n", weight_handle);
            return -1;
        }
        if (!core_.get_cached_pointers(grad_handle, grad_cpu_ptr, grad_gpu_ptr)) {
            printf("optimizer_add_param_batch: invalid grad handle %d\n", grad_handle);
            return -1;
        }

        if (weight_cpu_ptr == nullptr || grad_cpu_ptr == nullptr) {
            printf("optimizer_add_param_batch: null CPU pointer detected (index=%zu)\n", i);
            return -1;
        }

        weight_cpu.push_back(weight_cpu_ptr);
        grad_cpu.push_back(grad_cpu_ptr);

        if (sim->mode == GPU) {
            if (weight_gpu_ptr == nullptr || grad_gpu_ptr == nullptr) {
                printf("optimizer_add_param_batch: null GPU pointer detected in GPU mode (index=%zu)\n", i);
                return -1;
            }
            weight_gpu.push_back(weight_gpu_ptr);
            grad_gpu.push_back(grad_gpu_ptr);
        }
    }

    return sim->register_optimizer_param_batch(optimizer_id,
                                               weight_cpu,
                                               grad_cpu,
                                               weight_gpu,
                                               grad_gpu,
                                               impedance);
}

int SimWrapper::configure_optimizer(int optimizer_id, double momentum, double beta1, double beta2, double epsilon) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    OptimizerHyperParams params;
    params.momentum = momentum;
    params.beta1 = beta1;
    params.beta2 = beta2;
    params.epsilon = epsilon;
    return sim->configure_optimizer(optimizer_id, params);
}

int SimWrapper::optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt_step) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim->optimizer_step(optimizer_id, learning_rate, record_time, dt_step);
}

int SimWrapper::optimizer_step_with_inv_record_steps(int optimizer_id,
                                                     double learning_rate,
                                                     double inv_record_steps) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim->optimizer_step_with_inv_record_steps(optimizer_id, learning_rate, inv_record_steps);
}

int SimWrapper::optimizer_add_external_grads(int optimizer_id,
                                            const std::vector<int>& weight_handles,
                                            double impedance) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (weight_handles.empty()) {
        printf("optimizer_add_external_grads: empty weight_handles\n");
        return -1;
    }
    if (!(impedance > 0.0)) {
        printf("optimizer_add_external_grads: invalid impedance (%f)\n", impedance);
        return -1;
    }

    // Replace any existing external-grad binding for this optimizer_id.
    optimizer_external_grads_.erase(optimizer_id);
    auto [it, inserted] =
        optimizer_external_grads_.emplace(optimizer_id, OptimizerExternalGrads(sim->mode));
    (void)inserted;
    OptimizerExternalGrads& ext = it->second;
    ext.weight_handles = weight_handles;
    ext.impedance = impedance;
    ext.grads.resize(static_cast<int>(weight_handles.size()));

    // Zero-init CPU grads and sync to GPU if needed.
    std::fill_n(ext.grads.get_cpu_data(), ext.grads.size(), 0.0);
    if (sim->mode == GPU) {
        ext.grads.update_gpu_data_from_cpu();
    }

    // Register one optimizer param per weight handle (batch_size=1).
    for (size_t i = 0; i < weight_handles.size(); ++i) {
        const int weight_handle = weight_handles[i];
        double* weight_cpu_ptr = nullptr;
        double* weight_gpu_ptr = nullptr;
        if (!core_.get_cached_pointers(weight_handle, weight_cpu_ptr, weight_gpu_ptr)) {
            printf("optimizer_add_external_grads: invalid weight handle %d\n", weight_handle);
            return -1;
        }
        if (weight_cpu_ptr == nullptr) {
            printf("optimizer_add_external_grads: null CPU weight pointer (i=%zu)\n", i);
            return -1;
        }

        double* grad_cpu = ext.grads.get_cpu_data() + static_cast<int>(i);
        double* grad_gpu = nullptr;
        if (sim->mode == GPU) {
            if (weight_gpu_ptr == nullptr) {
                printf("optimizer_add_external_grads: null GPU weight pointer in GPU mode (i=%zu)\n", i);
                return -1;
            }
            grad_gpu = ext.grads.get_gpu_data() + static_cast<int>(i);
        }

        int rc = sim->register_optimizer_param(optimizer_id,
                                               weight_cpu_ptr,
                                               grad_cpu,
                                               weight_gpu_ptr,
                                               grad_gpu,
                                               impedance);
        if (rc < 0) {
            printf("optimizer_add_external_grads: register_optimizer_param failed (i=%zu)\n", i);
            return -1;
        }
    }

    return 0;
}

int SimWrapper::optimizer_set_external_grads_f32(int optimizer_id,
                                                nb::ndarray<float, nb::shape<-1>, nb::c_contig> grads) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    auto it = optimizer_external_grads_.find(optimizer_id);
    if (it == optimizer_external_grads_.end()) {
        printf("optimizer_set_external_grads_f32: optimizer %d has no external grads binding\n", optimizer_id);
        return -1;
    }
    OptimizerExternalGrads& ext = it->second;
    const int n = static_cast<int>(ext.weight_handles.size());
    if (n <= 0) {
        printf("optimizer_set_external_grads_f32: internal handle list is empty\n");
        return -1;
    }
    if (grads.ndim() != 1 || static_cast<int>(grads.shape(0)) != n) {
        printf("optimizer_set_external_grads_f32: grads shape mismatch (got %d expected %d)\n",
               static_cast<int>(grads.shape(0)), n);
        return -1;
    }

    double* out = ext.grads.get_cpu_data();
    const float* in = grads.data();
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<double>(in[i]);
    }
    if (sim->mode == GPU) {
        ext.grads.update_gpu_data_from_cpu();
    }
    return 0;
}

int SimWrapper::optimizer_clear_external_grads(int optimizer_id) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    auto it = optimizer_external_grads_.find(optimizer_id);
    if (it == optimizer_external_grads_.end()) {
        return 0;
    }
    OptimizerExternalGrads& ext = it->second;
    if (ext.grads.size() <= 0) {
        return 0;
    }
    std::fill_n(ext.grads.get_cpu_data(), ext.grads.size(), 0.0);
    if (sim->mode == GPU) {
        ext.grads.update_gpu_data_from_cpu();
    }
    return 0;
}

int SimWrapper::optimizer_reset_state(int optimizer_id) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim->optimizer_reset_state(optimizer_id);
}

std::tuple<long long, std::vector<double>, std::vector<double>, double, double, double> SimWrapper::optimizer_get_adam_state(
    int optimizer_id) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return {0, {}, {}, 0.9, 0.999, 1e-8};
    }
    long long step = 0;
    std::vector<double> m;
    std::vector<double> v;
    OptimizerHyperParams params;
    if (sim->optimizer_get_adam_state(optimizer_id, step, m, v, params) < 0) {
        return {0, {}, {}, 0.9, 0.999, 1e-8};
    }
    return {step, std::move(m), std::move(v), params.beta1, params.beta2, params.epsilon};
}

int SimWrapper::optimizer_set_adam_state(int optimizer_id,
                                        long long step_count,
                                        const std::vector<double>& m,
                                        const std::vector<double>& v,
                                        double beta1,
                                        double beta2,
                                        double epsilon) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    OptimizerHyperParams params;
    params.beta1 = beta1;
    params.beta2 = beta2;
    params.epsilon = epsilon;
    return sim->optimizer_set_adam_state(optimizer_id, step_count, m, v, params);
}

int SimWrapper::set_variable_value(double val, const string& mech, const string& var, int node_or_mech_idx) {
    return core_.set_variable_value(val, mech, var, node_or_mech_idx);
}

int SimWrapper::set_variable_value_with_array(double val, const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.set_variable_value_with_array(val, mech, var, node_or_mech_idx, array_index);
}

double SimWrapper::call_mech_func(const string& mech_name, const string& func_name, nb::args args) {
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    auto result = FunctionRegistry::getInstance().call(mech_name + "." + func_name, args_to_vector(sim->mode, args));
    return 0;
}

// int SimWrapper::vecevent_play(int mech_idx, const vector<double>& data) {
//     if (sim == nullptr) {
//         printf("Simulate not initialized\n");
//         return -1;
//     }
    
//     printf("vecevent_play: mech_idx=%d len=%zu\n", mech_idx, data.size());
//     sim->vecevent_play(mech_idx, data.size(), data.data());
//     return 0;
// }

void SimWrapper::set_user_mod_num(int num) {
    core_.set_user_mod_num(num);
}

vector<double>& SimWrapper::get_spk_by_gid(int gid) {
    return sim->neuron_group_list[0]->presyn->vecdata_spk_output->get_cpu_data()[gid];
}

int SimWrapper::get_monitor_handle(const string& mech, const string& var, int node_or_mech_idx) {
    return get_monitor_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimWrapper::get_monitor_handle_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.get_monitor_handle_with_array(mech, var, node_or_mech_idx, array_index);
}

// VecPlay相关方法实现
int SimWrapper::add_vecplay(const string& mech_name, const string& var_name, int instance_id,
                            const vector<double>& tvec, const vector<double>& yvec) {
    return core_.add_vecplay(mech_name, var_name, instance_id, tvec, yvec);
}

int SimWrapper::update_vecplay(const string& mech_name, const string& var_name, int instance_id,
                               const vector<double>& new_tvec, const vector<double>& new_yvec) {
    return core_.update_vecplay(mech_name, var_name, instance_id, new_tvec, new_yvec);
}

int SimWrapper::remove_vecplay(const string& mech_name, const string& var_name, int instance_id) {
    return core_.remove_vecplay(mech_name, var_name, instance_id);
}

bool SimWrapper::has_vecplay(const string& mech_name, const string& var_name, int instance_id) {
    return core_.has_vecplay(mech_name, var_name, instance_id);
}

vector<vector<string>> SimWrapper::get_all_vecplay_keys() {
    return core_.get_vecplay_keys();
}

// 变量handle缓存相关实现
int SimWrapper::get_variable_handle(const string& mech, const string& var, int node_or_mech_idx) {
    return core_.get_variable_handle(mech, var, node_or_mech_idx);
}

int SimWrapper::get_variable_handle_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index) {
    return core_.get_variable_handle_with_array(mech, var, node_or_mech_idx, array_index);
}

double SimWrapper::get_variable_by_handle(int handle) {
    return core_.get_variable_by_handle(handle);
}

int SimWrapper::set_dense_blocks_f32(nb::list blocks) {
    clear_dense_blocks();
    if (blocks.size() == 0) {
        return 0;
    }

    std::vector<heliox::runtime_api::learn::LearnRuntime::DenseBlockHostView> views;
    views.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        nb::handle item = blocks[i];
        nb::ndarray<float, nb::ndim<3>, nb::c_contig> arr =
            nb::cast<nb::ndarray<float, nb::ndim<3>, nb::c_contig>>(item);
        const int bn0 = static_cast<int>(arr.shape(0));
        const int bn1 = static_cast<int>(arr.shape(1));
        const int k0 = static_cast<int>(arr.shape(2));
        if (bn0 <= 0 || bn1 <= 0 || k0 <= 0) {
            printf("set_dense_blocks_f32: invalid shape in block %zu\n", i);
            clear_dense_blocks();
            return -1;
        }
        if (bn0 != bn1) {
            printf("set_dense_blocks_f32: block %zu is not square (%d vs %d)\n", i, bn0, bn1);
            clear_dense_blocks();
            return -1;
        }
        heliox::runtime_api::learn::LearnRuntime::DenseBlockHostView v;
        v.data = arr.data();
        v.bn = bn0;
        v.k_len = k0;
        views.push_back(v);
    }

    const int k_len = learn_.set_dense_blocks_f32(std::span<const heliox::runtime_api::learn::LearnRuntime::DenseBlockHostView>(
        views.data(), views.size()));
    if (k_len < 0) {
        clear_dense_blocks();
        return -1;
    }
    return k_len;
}

void SimWrapper::clear_dense_blocks() {
    learn_.clear_dense_blocks();
}

int SimWrapper::simulate_output_vs_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
    const std::vector<int>& output_v_handles,
    double tstop_ms,
    double v_init) {
    nb::gil_scoped_release release;

    if (output_vs_tn.ndim() != 2) {
        printf("simulate_output_vs_into: output_vs_tn must be 2D\n");
        return -1;
    }
    const int n_output = static_cast<int>(output_v_handles.size());
    if (n_output <= 0) {
        printf("simulate_output_vs_into: output_v_handles is empty\n");
        return -1;
    }
    if (static_cast<int>(output_vs_tn.shape(1)) != n_output) {
        printf("simulate_output_vs_into: output_vs_tn shape mismatch\n");
        return -1;
    }
    const int total_steps = static_cast<int>(output_vs_tn.shape(0)) - 1;
    if (total_steps <= 0) {
        printf("simulate_output_vs_into: total_steps must be positive\n");
        return -1;
    }
    return learn_.simulate_output_vs_into(output_vs_tn.data(),
                                         static_cast<int>(output_vs_tn.shape(0)),
                                         n_output,
                                         std::span<const int>(output_v_handles.data(), output_v_handles.size()),
                                         tstop_ms,
                                         v_init);
}

int SimWrapper::replay_compute_dw_dx_from_signals_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> it_lr_nt,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdv_lr_nt,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdvpre_lr_nt,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_to,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
    double dt_ms,
    bool percise,
    double grad_scale,
    double eps,
    double grad_l2norm_threshold,
    int clip_strategy,
    int clip_check_every) {
    nb::gil_scoped_release release;

    const int N = static_cast<int>(it_lr_nt.shape(0));
    const int T = static_cast<int>(it_lr_nt.shape(1));
    const int ksteps_total = T - 1;
    const int n_output = static_cast<int>(poutput.shape(0));
    const int n_input = static_cast<int>(pinput.shape(0));

    return learn_.replay_compute_dw_dx_from_signals_into(it_lr_nt.data(),
                                                        percise ? ditdv_lr_nt.data() : nullptr,
                                                        percise ? ditdvpre_lr_nt.data() : nullptr,
                                                        N,
                                                        T,
                                                        dLtdv_lr_to.data(),
                                                        ksteps_total,
                                                        n_output,
                                                        poutput.data(),
                                                        pinput.data(),
                                                        n_input,
                                                        pre_of_col.data(),
                                                        dw_out_n.data(),
                                                        dx_lr_it.data(),
                                                        dt_ms,
                                                        percise,
                                                        grad_scale,
                                                        eps,
                                                        grad_l2norm_threshold,
                                                        clip_strategy,
                                                        clip_check_every);
}

int SimWrapper::simulate_and_replay_dw_dx_streaming_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_to,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
    const std::vector<int>& pure_i_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
    const std::vector<int>& didv_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
    const std::vector<int>& didvpre_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
    double tstop_ms,
    int k_mul,
    bool percise,
    double v_init,
    double dt_ms,
    double grad_scale,
    double eps,
    double grad_l2norm_threshold,
    int clip_strategy,
    int clip_check_every) {
    nb::gil_scoped_release release;

    if (dLtdv_lr_to.ndim() != 2) {
        printf("simulate_and_replay_dw_dx_streaming_into: dLtdv_lr_to must be 2D\n");
        return -1;
    }
    const int ksteps_total = static_cast<int>(dLtdv_lr_to.shape(0));
    const int n_output = static_cast<int>(poutput.shape(0));
    const int n_input = static_cast<int>(pinput.shape(0));
    const int N = static_cast<int>(pre_of_col.shape(0));

    if (static_cast<int>(dLtdv_lr_to.shape(1)) != n_output) {
        printf("simulate_and_replay_dw_dx_streaming_into: dLtdv_lr_to shape mismatch\n");
        return -1;
    }
    if (dw_out_n.ndim() != 1 || static_cast<int>(dw_out_n.shape(0)) != N) {
        printf("simulate_and_replay_dw_dx_streaming_into: dw_out_n shape mismatch\n");
        return -1;
    }
    if (dx_lr_it.ndim() != 2 || static_cast<int>(dx_lr_it.shape(0)) != n_input ||
        static_cast<int>(dx_lr_it.shape(1)) != ksteps_total) {
        printf("simulate_and_replay_dw_dx_streaming_into: dx_lr_it shape mismatch\n");
        return -1;
    }
    if (static_cast<int>(pure_i_dest.shape(0)) != static_cast<int>(pure_i_handles.size()) ||
        static_cast<int>(pure_i_scale.shape(0)) != static_cast<int>(pure_i_handles.size())) {
        printf("simulate_and_replay_dw_dx_streaming_into: pure_i handles/dest/scale size mismatch\n");
        return -1;
    }
    if (percise) {
        if (static_cast<int>(didv_dest.shape(0)) != static_cast<int>(didv_handles.size()) ||
            static_cast<int>(didv_scale.shape(0)) != static_cast<int>(didv_handles.size())) {
            printf("simulate_and_replay_dw_dx_streaming_into: didv handles/dest/scale size mismatch\n");
            return -1;
        }
        if (static_cast<int>(didvpre_dest.shape(0)) != static_cast<int>(didvpre_handles.size()) ||
            static_cast<int>(didvpre_scale.shape(0)) != static_cast<int>(didvpre_handles.size())) {
            printf("simulate_and_replay_dw_dx_streaming_into: didvpre handles/dest/scale size mismatch\n");
            return -1;
        }
    }

    return learn_.simulate_and_replay_dw_dx_streaming_into(dLtdv_lr_to.data(),
                                                          ksteps_total,
                                                          n_output,
                                                          poutput.data(),
                                                          pinput.data(),
                                                          n_input,
                                                          N,
                                                          pre_of_col.data(),
                                                          dw_out_n.data(),
                                                          dx_lr_it.data(),
                                                          std::span<const int>(pure_i_handles.data(), pure_i_handles.size()),
                                                          std::span<const int32_t>(pure_i_dest.data(), pure_i_dest.shape(0)),
                                                          std::span<const float>(pure_i_scale.data(), pure_i_scale.shape(0)),
                                                          std::span<const int>(didv_handles.data(), didv_handles.size()),
                                                          std::span<const int32_t>(didv_dest.data(), didv_dest.shape(0)),
                                                          std::span<const float>(didv_scale.data(), didv_scale.shape(0)),
                                                          std::span<const int>(didvpre_handles.data(), didvpre_handles.size()),
                                                          std::span<const int32_t>(didvpre_dest.data(), didvpre_dest.shape(0)),
                                                          std::span<const float>(didvpre_scale.data(), didvpre_scale.shape(0)),
                                                          tstop_ms,
                                                          k_mul,
                                                          percise,
                                                          v_init,
                                                          dt_ms,
                                                          grad_scale,
                                                          eps,
                                                          grad_l2norm_threshold,
                                                          clip_strategy,
                                                          clip_check_every);
}

int SimWrapper::simulate_and_capture_mapped_signals_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> it_lr_tn,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdv_lr_tn,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdvpre_lr_tn,
    const std::vector<int>& output_v_handles,
    const std::vector<int>& pure_i_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
    const std::vector<int>& didv_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
    const std::vector<int>& didvpre_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
    double tstop_ms,
    int k_mul,
    bool percise,
    double v_init) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("simulate_and_capture_mapped_signals_into: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("simulate_and_capture_mapped_signals_into: only supported on GPU\n");
        return -1;
    }
    if (k_mul <= 0) {
        printf("simulate_and_capture_mapped_signals_into: k_mul must be positive\n");
        return -1;
    }
    if (sim->dt <= 0.0) {
        printf("simulate_and_capture_mapped_signals_into: invalid dt\n");
        return -1;
    }
    if (output_vs_tn.ndim() != 2 || it_lr_tn.ndim() != 2 || ditdv_lr_tn.ndim() != 2 || ditdvpre_lr_tn.ndim() != 2) {
        printf("simulate_and_capture_mapped_signals_into: outputs must be 2D arrays\n");
        return -1;
    }

    const int n_output = static_cast<int>(output_v_handles.size());
    if (n_output <= 0) {
        printf("simulate_and_capture_mapped_signals_into: output_v_handles is empty\n");
        return -1;
    }
    if (static_cast<int>(output_vs_tn.shape(1)) != n_output) {
        printf("simulate_and_capture_mapped_signals_into: output_vs_tn shape mismatch\n");
        return -1;
    }

    const int total_steps = static_cast<int>(output_vs_tn.shape(0)) - 1;
    if (total_steps <= 0) {
        printf("simulate_and_capture_mapped_signals_into: total_steps must be positive\n");
        return -1;
    }
    const int ksteps_total = static_cast<int>(total_steps / k_mul);
    const int N = static_cast<int>(it_lr_tn.shape(1));
    if (N <= 0) {
        printf("simulate_and_capture_mapped_signals_into: it_lr_tn second dim must be positive\n");
        return -1;
    }
    if (static_cast<int>(it_lr_tn.shape(0)) != (ksteps_total + 1)) {
        printf("simulate_and_capture_mapped_signals_into: it_lr_tn first dim mismatch\n");
        return -1;
    }
    if (static_cast<int>(ditdv_lr_tn.shape(0)) != (ksteps_total + 1) || static_cast<int>(ditdv_lr_tn.shape(1)) != N) {
        printf("simulate_and_capture_mapped_signals_into: ditdv_lr_tn shape mismatch\n");
        return -1;
    }
    if (static_cast<int>(ditdvpre_lr_tn.shape(0)) != (ksteps_total + 1) || static_cast<int>(ditdvpre_lr_tn.shape(1)) != N) {
        printf("simulate_and_capture_mapped_signals_into: ditdvpre_lr_tn shape mismatch\n");
        return -1;
    }

    if (static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_dest.shape(0)) ||
        static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_scale.shape(0))) {
        printf("simulate_and_capture_mapped_signals_into: pure_i handles/dest/scale size mismatch\n");
        return -1;
    }
    if (static_cast<int>(didv_handles.size()) != static_cast<int>(didv_dest.shape(0)) ||
        static_cast<int>(didv_handles.size()) != static_cast<int>(didv_scale.shape(0))) {
        printf("simulate_and_capture_mapped_signals_into: didv handles/dest/scale size mismatch\n");
        return -1;
    }
    if (static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_dest.shape(0)) ||
        static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_scale.shape(0))) {
        printf("simulate_and_capture_mapped_signals_into: didvpre handles/dest/scale size mismatch\n");
        return -1;
    }

    // Validate scatter destinations to avoid memory corruption.
    for (size_t i = 0; i < pure_i_dest.shape(0); ++i) {
        const int32_t d = pure_i_dest.data()[i];
        if (d < 0 || d >= N) {
            printf("simulate_and_capture_mapped_signals_into: pure_i_dest out of range at %zu (%d)\n", i, d);
            return -1;
        }
    }
    if (percise) {
        for (size_t i = 0; i < didv_dest.shape(0); ++i) {
            const int32_t d = didv_dest.data()[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_into: didv_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
        for (size_t i = 0; i < didvpre_dest.shape(0); ++i) {
            const int32_t d = didvpre_dest.data()[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_into: didvpre_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
    }

    // Basic sanity check for tstop_ms (we derive loop count from output buffer shape).
    const double tstop_from_steps = static_cast<double>(total_steps) * sim->dt;
    if (tstop_ms > 0.0 && std::abs(tstop_ms - tstop_from_steps) > (sim->dt * 0.5)) {
        printf("simulate_and_capture_mapped_signals_into: tstop mismatch (tstop_ms=%f, steps*dt=%f)\n",
               tstop_ms, tstop_from_steps);
        return -1;
    }

    // Ensure any pending CPU-side writes are visible on GPU before running.
    return learn_.simulate_and_capture_mapped_signals_into(output_vs_tn.data(),
                                                          total_steps + 1,
                                                          n_output,
                                                          it_lr_tn.data(),
                                                          ditdv_lr_tn.data(),
                                                          ditdvpre_lr_tn.data(),
                                                          ksteps_total + 1,
                                                          N,
                                                          std::span<const int>(output_v_handles.data(), output_v_handles.size()),
                                                          std::span<const int>(pure_i_handles.data(), pure_i_handles.size()),
                                                          std::span<const int32_t>(pure_i_dest.data(), pure_i_dest.shape(0)),
                                                          std::span<const float>(pure_i_scale.data(), pure_i_scale.shape(0)),
                                                          std::span<const int>(didv_handles.data(), didv_handles.size()),
                                                          std::span<const int32_t>(didv_dest.data(), didv_dest.shape(0)),
                                                          std::span<const float>(didv_scale.data(), didv_scale.shape(0)),
                                                          std::span<const int>(didvpre_handles.data(), didvpre_handles.size()),
                                                          std::span<const int32_t>(didvpre_dest.data(), didvpre_dest.shape(0)),
                                                          std::span<const float>(didvpre_scale.data(), didvpre_scale.shape(0)),
                                                          tstop_ms,
                                                          k_mul,
                                                          percise,
                                                          v_init);

}

int SimWrapper::simulate_and_capture_mapped_signals_cached(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
    const std::vector<int>& output_v_handles,
    const std::vector<int>& pure_i_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
    const std::vector<int>& didv_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
    const std::vector<int>& didvpre_handles,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
    double tstop_ms,
    int k_mul,
    bool percise,
    double v_init) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("simulate_and_capture_mapped_signals_cached: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("simulate_and_capture_mapped_signals_cached: only supported on GPU\n");
        return -1;
    }
    if (k_mul <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: k_mul must be positive\n");
        return -1;
    }
    if (sim->dt <= 0.0) {
        printf("simulate_and_capture_mapped_signals_cached: invalid dt\n");
        return -1;
    }
    if (output_vs_tn.ndim() != 2) {
        printf("simulate_and_capture_mapped_signals_cached: output_vs_tn must be 2D\n");
        return -1;
    }

    const int n_output = static_cast<int>(output_v_handles.size());
    if (n_output <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: output_v_handles is empty\n");
        return -1;
    }
    if (static_cast<int>(output_vs_tn.shape(1)) != n_output) {
        printf("simulate_and_capture_mapped_signals_cached: output_vs_tn shape mismatch\n");
        return -1;
    }
    const int total_steps = static_cast<int>(output_vs_tn.shape(0)) - 1;
    if (total_steps <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: total_steps must be positive\n");
        return -1;
    }
    const int ksteps_total = static_cast<int>(total_steps / k_mul);
    if (ksteps_total <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: ksteps_total must be positive\n");
        return -1;
    }

    // Infer N from the mapping destinations (must be compatible with training code).
    int N = 0;
    for (size_t i = 0; i < pure_i_dest.shape(0); ++i) {
        const int32_t d = pure_i_dest.data()[i];
        if (d + 1 > N) {
            N = d + 1;
        }
    }
    if (N <= 0) {
        printf("simulate_and_capture_mapped_signals_cached: inferred N is non-positive\n");
        return -1;
    }

    if (static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_dest.shape(0)) ||
        static_cast<int>(pure_i_handles.size()) != static_cast<int>(pure_i_scale.shape(0))) {
        printf("simulate_and_capture_mapped_signals_cached: pure_i handles/dest/scale size mismatch\n");
        return -1;
    }
    if (percise) {
        if (static_cast<int>(didv_handles.size()) != static_cast<int>(didv_dest.shape(0)) ||
            static_cast<int>(didv_handles.size()) != static_cast<int>(didv_scale.shape(0))) {
            printf("simulate_and_capture_mapped_signals_cached: didv handles/dest/scale size mismatch\n");
            return -1;
        }
        if (static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_dest.shape(0)) ||
            static_cast<int>(didvpre_handles.size()) != static_cast<int>(didvpre_scale.shape(0))) {
            printf("simulate_and_capture_mapped_signals_cached: didvpre handles/dest/scale size mismatch\n");
            return -1;
        }
    }

    // Validate scatter destinations to avoid memory corruption.
    for (size_t i = 0; i < pure_i_dest.shape(0); ++i) {
        const int32_t d = pure_i_dest.data()[i];
        if (d < 0 || d >= N) {
            printf("simulate_and_capture_mapped_signals_cached: pure_i_dest out of range at %zu (%d)\n", i, d);
            return -1;
        }
    }
    if (percise) {
        for (size_t i = 0; i < didv_dest.shape(0); ++i) {
            const int32_t d = didv_dest.data()[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_cached: didv_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
        for (size_t i = 0; i < didvpre_dest.shape(0); ++i) {
            const int32_t d = didvpre_dest.data()[i];
            if (d < 0 || d >= N) {
                printf("simulate_and_capture_mapped_signals_cached: didvpre_dest out of range at %zu (%d)\n", i, d);
                return -1;
            }
        }
    }

    // Basic sanity check for tstop_ms (we derive loop count from output buffer shape).
    const double tstop_from_steps = static_cast<double>(total_steps) * sim->dt;
    if (tstop_ms > 0.0 && std::abs(tstop_ms - tstop_from_steps) > (sim->dt * 0.5)) {
        printf("simulate_and_capture_mapped_signals_cached: tstop mismatch (tstop_ms=%f, steps*dt=%f)\n",
               tstop_ms, tstop_from_steps);
        return -1;
    }

    return learn_.simulate_and_capture_mapped_signals_cached(output_vs_tn.data(),
                                                            total_steps + 1,
                                                            n_output,
                                                            std::span<const int>(output_v_handles.data(), output_v_handles.size()),
                                                            std::span<const int>(pure_i_handles.data(), pure_i_handles.size()),
                                                            std::span<const int32_t>(pure_i_dest.data(), pure_i_dest.shape(0)),
                                                            std::span<const float>(pure_i_scale.data(), pure_i_scale.shape(0)),
                                                            std::span<const int>(didv_handles.data(), didv_handles.size()),
                                                            std::span<const int32_t>(didv_dest.data(), didv_dest.shape(0)),
                                                            std::span<const float>(didv_scale.data(), didv_scale.shape(0)),
                                                            std::span<const int>(didvpre_handles.data(), didvpre_handles.size()),
                                                            std::span<const int32_t>(didvpre_dest.data(), didvpre_dest.shape(0)),
                                                            std::span<const float>(didvpre_scale.data(), didvpre_scale.shape(0)),
                                                            tstop_ms,
                                                            k_mul,
                                                            percise,
                                                            v_init);

    
}

int SimWrapper::replay_compute_dw_dx_from_cached_signals_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_to,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
    double dt_ms,
    bool percise,
    double grad_scale,
    double eps,
    double grad_l2norm_threshold,
    int clip_strategy,
    int clip_check_every) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("replay_compute_dw_dx_from_cached_signals_into: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("replay_compute_dw_dx_from_cached_signals_into: only supported on GPU\n");
        return -1;
    }
    if (dLtdv_lr_to.ndim() != 2) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dLtdv_lr_to must be 2D\n");
        return -1;
    }
    const int ksteps_total = static_cast<int>(dLtdv_lr_to.shape(0));
    const int n_output = static_cast<int>(poutput.shape(0));
    const int n_input = static_cast<int>(pinput.shape(0));
    const int N = static_cast<int>(pre_of_col.shape(0));
    if (ksteps_total <= 0 || n_output <= 0 || n_input <= 0 || N <= 0) {
        printf("replay_compute_dw_dx_from_cached_signals_into: empty shapes\n");
        return -1;
    }
    if (static_cast<int>(dLtdv_lr_to.shape(1)) != n_output) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dLtdv_lr_to shape mismatch\n");
        return -1;
    }
    if (dw_out_n.ndim() != 1 || static_cast<int>(dw_out_n.shape(0)) != N) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dw_out_n shape mismatch\n");
        return -1;
    }
    if (dx_lr_it.ndim() != 2 || static_cast<int>(dx_lr_it.shape(0)) != n_input ||
        static_cast<int>(dx_lr_it.shape(1)) != ksteps_total) {
        printf("replay_compute_dw_dx_from_cached_signals_into: dx_lr_it shape mismatch\n");
        return -1;
    }

    return learn_.replay_compute_dw_dx_from_cached_signals_into(dLtdv_lr_to.data(),
                                                               ksteps_total,
                                                               n_output,
                                                               poutput.data(),
                                                               pinput.data(),
                                                               n_input,
                                                               pre_of_col.data(),
                                                               dw_out_n.data(),
                                                               dx_lr_it.data(),
                                                               dt_ms,
                                                               percise,
                                                               grad_scale,
                                                               eps,
                                                               grad_l2norm_threshold,
                                                               clip_strategy,
                                                               clip_check_every);

}

int SimWrapper::replay_compute_dw_from_cached_signals_into(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_to,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
    nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
    nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
    double dt_ms,
    bool percise,
    double grad_scale,
    double eps,
    double grad_l2norm_threshold,
    int clip_strategy,
    int clip_check_every) {
    nb::gil_scoped_release release;

    if (sim == nullptr) {
        printf("replay_compute_dw_from_cached_signals_into: Simulate not initialized\n");
        return -1;
    }
    if (sim->mode != GPU) {
        printf("replay_compute_dw_from_cached_signals_into: only supported on GPU\n");
        return -1;
    }
    if (dLtdv_lr_to.ndim() != 2) {
        printf("replay_compute_dw_from_cached_signals_into: dLtdv_lr_to must be 2D\n");
        return -1;
    }
    if (poutput.ndim() != 1 || pre_of_col.ndim() != 1) {
        printf("replay_compute_dw_from_cached_signals_into: poutput/pre_of_col must be 1D\n");
        return -1;
    }

    const int ksteps_total = static_cast<int>(dLtdv_lr_to.shape(0));
    const int n_output = static_cast<int>(poutput.shape(0));
    const int N = static_cast<int>(pre_of_col.shape(0));
    if (ksteps_total <= 0 || n_output <= 0 || N <= 0) {
        printf("replay_compute_dw_from_cached_signals_into: empty shapes\n");
        return -1;
    }
    if (static_cast<int>(dLtdv_lr_to.shape(1)) != n_output) {
        printf("replay_compute_dw_from_cached_signals_into: dLtdv_lr_to shape mismatch\n");
        return -1;
    }
    if (dw_out_n.ndim() != 1 || static_cast<int>(dw_out_n.shape(0)) != N) {
        printf("replay_compute_dw_from_cached_signals_into: dw_out_n shape mismatch\n");
        return -1;
    }

    return learn_.replay_compute_dw_from_cached_signals_into(dLtdv_lr_to.data(),
                                                             ksteps_total,
                                                             n_output,
                                                             poutput.data(),
                                                             pre_of_col.data(),
                                                             dw_out_n.data(),
                                                             dt_ms,
                                                             percise,
                                                             grad_scale,
                                                             eps,
                                                             grad_l2norm_threshold,
                                                             clip_strategy,
                                                             clip_check_every);
}

std::vector<double> SimWrapper::get_variables_by_handles(const std::vector<int>& handles) {
    std::vector<double> values;
    values.reserve(handles.size());
    for (int handle : handles) {
        values.push_back(get_variable_by_handle(handle));
    }
    return values;
}

int SimWrapper::get_variables_by_handles_f32_into(const std::vector<int>& handles,
                                                  nb::ndarray<float, nb::shape<-1>, nb::c_contig> out) {
    if (out.ndim() != 1) {
        printf("get_variables_by_handles_f32_into: out must be 1D\n");
        return -1;
    }
    const int count = static_cast<int>(handles.size());
    if (static_cast<int>(out.shape(0)) != count) {
        printf("get_variables_by_handles_f32_into: out length mismatch (%zd vs %d)\n",
               static_cast<ssize_t>(out.shape(0)), count);
        return -1;
    }
    if (count == 0) {
        return 0;
    }

    float* out_cpu = out.data();
    return core_.get_variables_by_handles_f32(handles, out_cpu, count);
}

int SimWrapper::set_variable_by_handle(int handle, double value) {
    return core_.set_variable_by_handle(handle, value);
}

int SimWrapper::set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values) {
    return core_.set_variables_by_handles(handles, values);
}

int SimWrapper::register_netstim_batch(const vector<tuple<int, int, int>>& handle_triplets,
                                       const NetStimBatchParams& params) {
    return core_.register_netstim_batch(handle_triplets,
                                        params.interval_scale,
                                        params.start_base,
                                        params.epsilon,
                                        params.number);
}

int SimWrapper::register_vecstim_batch(const vector<int>& mech_indices,
                                       const VecStimBatchParams& params) {
    return core_.register_vecstim_batch(mech_indices,
                                        params.spike_scale,
                                        params.start_base,
                                        params.epsilon,
                                        params.spike_count);
}

int SimWrapper::set_input_batch_pixels(int batch_id, std::span<const double> pixels) {
    return core_.set_input_batch_pixels(batch_id, pixels);
}

// Gap Junction 管理方法实现

map<int, map<string, nb::object>> SimWrapper::get_all_gap_junctions() {
    map<int, map<string, nb::object>> result;
    
    if (sim == nullptr) {
        printf("Simulator not initialized\n");
        return result;
    }
    
    // 从Simulate获取gap junction数据
    const auto& gap_junctions = sim->get_all_gap_junctions();
    
    for (const auto& [sid, gap_meta] : gap_junctions) {
        map<string, nb::object> gap_info;
        
        // 使用VarDescriptor的信息
        gap_info["src_mech"] = nb::cast(gap_meta.source.mech);
        gap_info["src_var"] = nb::cast(gap_meta.source.var);
        gap_info["src_idx"] = nb::cast(gap_meta.source.idx);
        gap_info["num_targets"] = nb::cast((int)gap_meta.targets.size());
        gap_info["is_dynamic"] = nb::cast(true);  // 所有手动添加的都是动态的
        
        result[sid] = gap_info;
    }
    
    return result;
}

map<string, nb::object> SimWrapper::get_gap_junction(int sid) {
    map<string, nb::object> result;
    
    if (sim == nullptr) {
        printf("Simulator not initialized\n");
        return result;
    }
    
    auto gap_meta = sim->get_gap_junction(sid);
    if (gap_meta == nullptr) {
        return result;  // 返回空map
    }
    
    // 简化：直接返回基本信息
    result["sid"] = nb::cast(sid);
    result["src_mech"] = nb::cast(gap_meta->source.mech);
    result["src_var"] = nb::cast(gap_meta->source.var);
    result["src_idx"] = nb::cast(gap_meta->source.idx);
    result["num_targets"] = nb::cast((int)gap_meta->targets.size());
    
    return result;
}

int SimWrapper::add_gap_source(int sid, const string& src_mech, const string& src_var, int src_idx) {
    if (sim == nullptr) {
        printf("Simulator not initialized\n");
        return -1;
    }
    // 调用Simulate层的方法，注意参数顺序变了（sid在最后）
    return sim->add_gap_source(src_mech, src_var, src_idx, sid);
}

int SimWrapper::add_gap_target(int sid, const string& tgt_mech, const string& tgt_var, int tgt_idx) {
    if (sim == nullptr) {
        printf("Simulator not initialized\n");
        return -1;
    }
    // 直接调用Simulate层的方法
    return sim->add_gap_target(sid, tgt_mech, tgt_var, tgt_idx);
}

int SimWrapper::clear_all_gap_junctions() {
    if (sim == nullptr) {
        printf("Simulator not initialized\n");
        return -1;
    }
    // 直接调用Simulate层的方法
    return sim->clear_all_gap_junctions();
}

int SimWrapper::get_next_available_sid() {
    if (sim == nullptr) {
        printf("Simulator not initialized\n");
        return -1;
    }
    // 调用Simulate层的方法
    return sim->get_next_available_sid();
}
