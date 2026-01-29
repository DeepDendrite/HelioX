#include "optimizer.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

OptimizerBase::OptimizerBase(Mode mode)
    : mode_(mode),
      weight_ptr_buffer_(mode == GPU ? GPU : CPU),
      grad_ptr_buffer_(mode == GPU ? GPU : CPU),
      impedance_buffer_(mode == GPU ? GPU : CPU),
      batch_size_buffer_(mode == GPU ? GPU : CPU),
      batch_offset_buffer_(mode == GPU ? GPU : CPU) {}

int OptimizerBase::add_param(const OptimizerParam& param) {
    int batch_size = static_cast<int>(param.weight_cpu.size());
    if (batch_size <= 0) {
        printf("Optimizer param requires at least one weight pointer\n");
        return -1;
    }
    if (param.grad_cpu.size() != static_cast<size_t>(batch_size)) {
        printf("Optimizer param grad_cpu size mismatch (expected %d, got %zu)\n",
               batch_size, param.grad_cpu.size());
        return -1;
    }
    if (mode_ == GPU) {
        if (param.weight_gpu.size() != static_cast<size_t>(batch_size) ||
            param.grad_gpu.size() != static_cast<size_t>(batch_size)) {
            printf("GPU optimizer param pointer count mismatch (batch=%d)\n", batch_size);
            return -1;
        }
    }

    for (int i = 0; i < batch_size; ++i) {
        if (param.weight_cpu[i] == nullptr || param.grad_cpu[i] == nullptr) {
            printf("Optimizer param requires valid CPU pointers (index=%d)\n", i);
            return -1;
        }
        if (mode_ == GPU) {
            if (param.weight_gpu[i] == nullptr || param.grad_gpu[i] == nullptr) {
                printf("GPU optimizer param requires valid GPU pointers (index=%d)\n", i);
                return -1;
            }
        }
    }

    params_.push_back({});
    ParamEntry& entry = params_.back();
    entry.weight_cpu = param.weight_cpu;
    entry.grad_cpu = param.grad_cpu;
    entry.impedance = param.impedance;
    entry.batch_size = batch_size;
    if (mode_ == GPU) {
        entry.weight_gpu = param.weight_gpu;
        entry.grad_gpu = param.grad_gpu;
    }

    int offset = weight_ptr_buffer_.size();
    batch_offset_buffer_.push_back(offset);
    batch_size_buffer_.push_back(batch_size);
    impedance_buffer_.push_back(entry.impedance);

    for (int i = 0; i < batch_size; ++i) {
        double* weight_ptr = (mode_ == GPU) ? entry.weight_gpu[i] : entry.weight_cpu[i];
        double* grad_ptr = (mode_ == GPU) ? entry.grad_gpu[i] : entry.grad_cpu[i];
        weight_ptr_buffer_.push_back(weight_ptr);
        grad_ptr_buffer_.push_back(grad_ptr);
    }

    on_param_added(params_.size() - 1);
    buffers_dirty_ = (mode_ == GPU);
    return static_cast<int>(params_.size() - 1);
}

void OptimizerBase::ensure_gpu_buffers_synced() {
    if (mode_ != GPU || !buffers_dirty_) {
        return;
    }
    weight_ptr_buffer_.update_gpu_data_from_cpu();
    grad_ptr_buffer_.update_gpu_data_from_cpu();
    impedance_buffer_.update_gpu_data_from_cpu();
    batch_size_buffer_.update_gpu_data_from_cpu();
    batch_offset_buffer_.update_gpu_data_from_cpu();
    buffers_dirty_ = false;
}

void OptimizerBase::step(double learning_rate, double inv_record_steps) {
    if (params_.empty()) {
        return;
    }
    if (mode_ == GPU) {
        step_gpu(learning_rate, inv_record_steps);
    } else {
        step_cpu(learning_rate, inv_record_steps);
    }
}

SGDOptimizer::SGDOptimizer(Mode mode)
    : OptimizerBase(mode) {}

void SGDOptimizer::step_cpu(double learning_rate, double inv_record_steps) {
    for (auto& entry : params_) {
        if (entry.batch_size <= 0 || entry.weight_cpu.empty()) {
            continue;
        }
        double grad_sum = 0.0;
        bool valid = true;
        for (double* grad_ptr : entry.grad_cpu) {
            if (!grad_ptr) {
                valid = false;
                break;
            }
            grad_sum += *grad_ptr;
        }
        if (!valid || entry.weight_cpu[0] == nullptr) {
            continue;
        }
        double avg_grad = grad_sum / static_cast<double>(entry.batch_size);
        double scaled_grad = avg_grad * entry.impedance * inv_record_steps;
        double updated_weight = *(entry.weight_cpu[0]) - learning_rate * scaled_grad;
        *(entry.weight_cpu[0]) = updated_weight;
        for (int i = 1; i < entry.batch_size; ++i) {
            if (entry.weight_cpu[i]) {
                *(entry.weight_cpu[i]) = updated_weight;
            }
        }
        for (double* grad_ptr : entry.grad_cpu) {
            if (grad_ptr) {
                *grad_ptr = 0.0;
            }
        }
    }
}

extern void launch_sgd_update_kernel(double** weight_ptrs,
                                     double** grad_ptrs,
                                     int* batch_sizes,
                                     int* batch_offsets,
                                     double* impedances,
                                     int param_count,
                                     double learning_rate,
                                     double inv_record_steps,
                                     cudaStream_t stream);

void SGDOptimizer::step_gpu(double learning_rate, double inv_record_steps) {
    ensure_gpu_buffers_synced();

    double** weight_gpu_ptrs = weight_ptr_buffer_.get_gpu_data();
    double** grad_gpu_ptrs = grad_ptr_buffer_.get_gpu_data();
    double* impedance_gpu = impedance_buffer_.get_gpu_data();
    int* batch_sizes_gpu = batch_size_buffer_.get_gpu_data();
    int* batch_offsets_gpu = batch_offset_buffer_.get_gpu_data();
    int param_count = static_cast<int>(params_.size());

    launch_sgd_update_kernel(weight_gpu_ptrs,
                             grad_gpu_ptrs,
                             batch_sizes_gpu,
                             batch_offsets_gpu,
                             impedance_gpu,
                             param_count,
                             learning_rate,
                             inv_record_steps,
                             0);
    cuda_sync_all();

    for (auto& entry : params_) {
        for (double* grad_ptr : entry.grad_cpu) {
            if (grad_ptr) {
                *grad_ptr = 0.0;
            }
        }
    }
}

SGDMomentumOptimizer::SGDMomentumOptimizer(Mode mode)
    : OptimizerBase(mode),
      velocity_(mode == GPU ? GPU : CPU) {}

void SGDMomentumOptimizer::configure(const OptimizerHyperParams& params) {
    OptimizerBase::configure(params);
    momentum_ = params.momentum;
}

void SGDMomentumOptimizer::on_param_added(size_t index) {
    (void)index;
    velocity_.push_back(0.0);
}

void SGDMomentumOptimizer::step_cpu(double learning_rate, double inv_record_steps) {
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        auto& entry = params_[idx];
        if (entry.batch_size <= 0 || entry.weight_cpu.empty()) {
            continue;
        }
        double grad_sum = 0.0;
        bool valid = true;
        for (double* grad_ptr : entry.grad_cpu) {
            if (!grad_ptr) {
                valid = false;
                break;
            }
            grad_sum += *grad_ptr;
        }
        if (!valid || entry.weight_cpu[0] == nullptr) {
            continue;
        }
        double avg_grad = grad_sum / static_cast<double>(entry.batch_size);
        double scaled_grad = avg_grad * entry.impedance * inv_record_steps;

        double& velocity = velocity_.get_cpu_data()[idx];
        velocity = momentum_ * velocity - learning_rate * scaled_grad;

        double updated_weight = *(entry.weight_cpu[0]) + velocity;
        *(entry.weight_cpu[0]) = updated_weight;
        for (int i = 1; i < entry.batch_size; ++i) {
            if (entry.weight_cpu[i]) {
                *(entry.weight_cpu[i]) = updated_weight;
            }
        }
        for (double* grad_ptr : entry.grad_cpu) {
            if (grad_ptr) {
                *grad_ptr = 0.0;
            }
        }
    }
}

extern void launch_sgd_momentum_kernel(double** weight_ptrs,
                                       double** grad_ptrs,
                                       double* velocities,
                                       int* batch_sizes,
                                       int* batch_offsets,
                                       double* impedances,
                                       int param_count,
                                       double learning_rate,
                                       double inv_record_steps,
                                       double momentum,
                                       cudaStream_t stream);

void SGDMomentumOptimizer::step_gpu(double learning_rate, double inv_record_steps) {
    ensure_gpu_buffers_synced();
    velocity_.update_gpu_data_from_cpu();

    double** weight_gpu_ptrs = weight_ptr_buffer_.get_gpu_data();
    double** grad_gpu_ptrs = grad_ptr_buffer_.get_gpu_data();
    double* impedance_gpu = impedance_buffer_.get_gpu_data();
    double* velocity_gpu = velocity_.get_gpu_data();
    int* batch_sizes_gpu = batch_size_buffer_.get_gpu_data();
    int* batch_offsets_gpu = batch_offset_buffer_.get_gpu_data();
    int param_count = static_cast<int>(params_.size());

    launch_sgd_momentum_kernel(weight_gpu_ptrs,
                               grad_gpu_ptrs,
                               velocity_gpu,
                               batch_sizes_gpu,
                               batch_offsets_gpu,
                               impedance_gpu,
                               param_count,
                               learning_rate,
                               inv_record_steps,
                               momentum_,
                               0);
    cuda_sync_all();

    velocity_.update_cpu_data_from_gpu();

    for (auto& entry : params_) {
        for (double* grad_ptr : entry.grad_cpu) {
            if (grad_ptr) {
                *grad_ptr = 0.0;
            }
        }
    }
}

void SGDMomentumOptimizer::reset_state() {
    if (velocity_.size() <= 0) {
        return;
    }
    std::fill_n(velocity_.get_cpu_data(), velocity_.size(), 0.0);
    if (mode_ == GPU) {
        velocity_.update_gpu_data_from_cpu();
    }
}

AdamOptimizer::AdamOptimizer(Mode mode)
    : OptimizerBase(mode),
      m_(mode == GPU ? GPU : CPU),
      v_(mode == GPU ? GPU : CPU) {}

void AdamOptimizer::configure(const OptimizerHyperParams& params) {
    OptimizerBase::configure(params);
    beta1_ = params.beta1;
    beta2_ = params.beta2;
    epsilon_ = params.epsilon;
}

void AdamOptimizer::on_param_added(size_t index) {
    (void)index;
    m_.push_back(0.0);
    v_.push_back(0.0);
}

void AdamOptimizer::reset_state() {
    step_count_ = 0;
    if (m_.size() > 0) {
        std::fill_n(m_.get_cpu_data(), m_.size(), 0.0);
    }
    if (v_.size() > 0) {
        std::fill_n(v_.get_cpu_data(), v_.size(), 0.0);
    }
    if (mode_ == GPU) {
        m_.update_gpu_data_from_cpu();
        v_.update_gpu_data_from_cpu();
    }
}

void AdamOptimizer::export_state(long long& step_count, std::vector<double>& m, std::vector<double>& v) {
    step_count = step_count_;
    const int n = static_cast<int>(params_.size());
    m.resize(n);
    v.resize(n);
    if (n > 0) {
        const double* m_ptr = m_.get_cpu_data();
        const double* v_ptr = v_.get_cpu_data();
        std::copy_n(m_ptr, n, m.begin());
        std::copy_n(v_ptr, n, v.begin());
    }
}

int AdamOptimizer::import_state(long long step_count, std::span<const double> m, std::span<const double> v) {
    const int n = static_cast<int>(params_.size());
    if (static_cast<int>(m.size()) != n || static_cast<int>(v.size()) != n) {
        printf("AdamOptimizer::import_state: size mismatch (expected %d, got m=%zu v=%zu)\n",
               n, m.size(), v.size());
        return -1;
    }
    step_count_ = step_count;
    if (n > 0) {
        std::copy_n(m.data(), n, m_.get_cpu_data());
        std::copy_n(v.data(), n, v_.get_cpu_data());
    }
    if (mode_ == GPU) {
        m_.update_gpu_data_from_cpu();
        v_.update_gpu_data_from_cpu();
    }
    return 0;
}

void AdamOptimizer::step_cpu(double learning_rate, double inv_record_steps) {
    ++step_count_;
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        auto& entry = params_[idx];
        if (entry.batch_size <= 0 || entry.weight_cpu.empty()) {
            continue;
        }
        double grad_sum = 0.0;
        bool valid = true;
        for (double* grad_ptr : entry.grad_cpu) {
            if (!grad_ptr) {
                valid = false;
                break;
            }
            grad_sum += *grad_ptr;
        }
        if (!valid || entry.weight_cpu[0] == nullptr) {
            continue;
        }
        double avg_grad = grad_sum / static_cast<double>(entry.batch_size);
        double scaled_grad = avg_grad * entry.impedance * inv_record_steps;

        double& m_ref = m_.get_cpu_data()[idx];
        double& v_ref = v_.get_cpu_data()[idx];

        m_ref = beta1_ * m_ref + (1.0 - beta1_) * scaled_grad;
        v_ref = beta2_ * v_ref + (1.0 - beta2_) * scaled_grad * scaled_grad;

        double beta1_pow = std::pow(beta1_, static_cast<double>(step_count_));
        double beta2_pow = std::pow(beta2_, static_cast<double>(step_count_));
        double m_hat = m_ref / (1.0 - beta1_pow);
        double v_hat = v_ref / (1.0 - beta2_pow);

        double updated_weight = *(entry.weight_cpu[0]) - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon_);
        *(entry.weight_cpu[0]) = updated_weight;
        for (int i = 1; i < entry.batch_size; ++i) {
            if (entry.weight_cpu[i]) {
                *(entry.weight_cpu[i]) = updated_weight;
            }
        }
        for (double* grad_ptr : entry.grad_cpu) {
            if (grad_ptr) {
                *grad_ptr = 0.0;
            }
        }
    }
}

extern void launch_adam_update_kernel(double** weight_ptrs,
                                      double** grad_ptrs,
                                      double* m_array,
                                      double* v_array,
                                      int* batch_sizes,
                                      int* batch_offsets,
                                      double* impedances,
                                      int param_count,
                                      double learning_rate,
                                      double inv_record_steps,
                                      double beta1,
                                      double beta2,
                                      double epsilon,
                                      long long step_count,
                                      cudaStream_t stream);

void AdamOptimizer::step_gpu(double learning_rate, double inv_record_steps) {
    ++step_count_;
    ensure_gpu_buffers_synced();
    m_.update_gpu_data_from_cpu();
    v_.update_gpu_data_from_cpu();

    double** weight_gpu_ptrs = weight_ptr_buffer_.get_gpu_data();
    double** grad_gpu_ptrs = grad_ptr_buffer_.get_gpu_data();
    double* impedance_gpu = impedance_buffer_.get_gpu_data();
    double* m_gpu = m_.get_gpu_data();
    double* v_gpu = v_.get_gpu_data();
    int* batch_sizes_gpu = batch_size_buffer_.get_gpu_data();
    int* batch_offsets_gpu = batch_offset_buffer_.get_gpu_data();
    int param_count = static_cast<int>(params_.size());

    launch_adam_update_kernel(weight_gpu_ptrs,
                              grad_gpu_ptrs,
                              m_gpu,
                              v_gpu,
                              batch_sizes_gpu,
                              batch_offsets_gpu,
                              impedance_gpu,
                              param_count,
                              learning_rate,
                              inv_record_steps,
                              beta1_,
                              beta2_,
                              epsilon_,
                              step_count_,
                              0);
    cuda_sync_all();

    m_.update_cpu_data_from_gpu();
    v_.update_cpu_data_from_gpu();

    for (auto& entry : params_) {
        for (double* grad_ptr : entry.grad_cpu) {
            if (grad_ptr) {
                *grad_ptr = 0.0;
            }
        }
    }
}
