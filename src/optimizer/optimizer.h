#pragma once

#include <memory>
#include <vector>
#include <span>
#include "utils.h"
#include "vecdata.h"

enum class OptimizerType {
    SGD = 0,
    Momentum = 1,
    Adam = 2,
};

struct OptimizerHyperParams {
    double momentum = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

struct OptimizerParam {
    std::vector<double*> weight_cpu;
    std::vector<double*> grad_cpu;
    std::vector<double*> weight_gpu;
    std::vector<double*> grad_gpu;
    double impedance = 1.0;
};

class OptimizerBase {
public:
    explicit OptimizerBase(Mode mode);
    virtual ~OptimizerBase() = default;

    int add_param(const OptimizerParam& param);
    size_t param_count() const { return params_.size(); }

    void step(double learning_rate, double inv_record_steps);
    virtual const char* name() const = 0;
    virtual void configure(const OptimizerHyperParams& params) { hyper_params_ = params; }
    virtual void reset_state() {}
    const OptimizerHyperParams& hyper_params() const { return hyper_params_; }

protected:
    struct ParamEntry {
        std::vector<double*> weight_cpu;
        std::vector<double*> grad_cpu;
        std::vector<double*> weight_gpu;
        std::vector<double*> grad_gpu;
        double impedance;
        int batch_size = 0;
    };

    Mode mode_;
    std::vector<ParamEntry> params_;
    VecData<double*> weight_ptr_buffer_;
    VecData<double*> grad_ptr_buffer_;
    VecData<double> impedance_buffer_;
    VecData<int> batch_size_buffer_;
    VecData<int> batch_offset_buffer_;
    bool buffers_dirty_ = false;

    OptimizerHyperParams hyper_params_;

    virtual void step_cpu(double learning_rate, double inv_record_steps) = 0;
    virtual void step_gpu(double learning_rate, double inv_record_steps) = 0;
    virtual void on_param_added(size_t /*index*/) {}

    void ensure_gpu_buffers_synced();
};

class SGDOptimizer final : public OptimizerBase {
public:
    explicit SGDOptimizer(Mode mode);
    const char* name() const override { return "SGD"; }

protected:
    void step_cpu(double learning_rate, double inv_record_steps) override;
    void step_gpu(double learning_rate, double inv_record_steps) override;
};

class SGDMomentumOptimizer final : public OptimizerBase {
public:
    explicit SGDMomentumOptimizer(Mode mode);
    const char* name() const override { return "SGDMomentum"; }
    void configure(const OptimizerHyperParams& params) override;
    void reset_state() override;

protected:
    void step_cpu(double learning_rate, double inv_record_steps) override;
    void step_gpu(double learning_rate, double inv_record_steps) override;
    void on_param_added(size_t index) override;

private:
    double momentum_ = 0.9;
    VecData<double> velocity_;
};

class AdamOptimizer final : public OptimizerBase {
public:
    explicit AdamOptimizer(Mode mode);
    const char* name() const override { return "Adam"; }
    void configure(const OptimizerHyperParams& params) override;
    void reset_state() override;
    void export_state(long long& step_count, std::vector<double>& m, std::vector<double>& v);
    int import_state(long long step_count, std::span<const double> m, std::span<const double> v);

protected:
    void step_cpu(double learning_rate, double inv_record_steps) override;
    void step_gpu(double learning_rate, double inv_record_steps) override;
    void on_param_added(size_t index) override;

private:
    double beta1_ = 0.9;
    double beta2_ = 0.999;
    double epsilon_ = 1e-8;
    long long step_count_ = 0;
    VecData<double> m_;
    VecData<double> v_;
};
