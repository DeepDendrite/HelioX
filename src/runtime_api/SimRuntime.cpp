#include "SimRuntime.h"

#include "simulate.h"

#include <utility>

struct SimRuntime::Impl {
    std::unique_ptr<Simulate> sim;

    // Phase 1: Keep minimal state here so the skeleton compiles cleanly.
    //
    // Important: This is NOT yet the full runtime implementation. The existing
    // python_api (SimWrapper) remains the source of truth for model loading,
    // monitor registration, etc. Later phases will migrate those behaviors
    // into src/runtime_api/{core,learn}.
    std::string data_path;
    std::string output_dir;
    int permute_type = -1;

    bool spike_output_enabled = false;

    Impl() : sim(std::make_unique<Simulate>()) {}
};

SimRuntime::SimRuntime() : impl_(std::make_unique<Impl>()) {}
SimRuntime::~SimRuntime() = default;

int SimRuntime::set_data_path(const std::string& path) {
    impl_->data_path = path;
    // Phase 1 skeleton: real "load from bbcore export path" is still implemented
    // in python_api/SimWrapper. This will be implemented in runtime_api/core later.
    return 0;
}

int SimRuntime::set_device(const std::string& dev) {
    if (dev == "gpu" || dev == "GPU") {
        impl_->sim->mode = GPU;
        return 0;
    }
    if (dev == "cpu" || dev == "CPU") {
        impl_->sim->mode = CPU;
        return 0;
    }
    return -1;
}

int SimRuntime::set_output_dir(const std::string& dir) {
    impl_->output_dir = dir;
    impl_->sim->output_folder = dir;
    return 0;
}

int SimRuntime::set_permute_type(int type) {
    impl_->permute_type = type;
    impl_->sim->permute_type = type;
    return 0;
}

int SimRuntime::set_dt(double dt) {
    impl_->sim->dt = dt;
    return 0;
}

double SimRuntime::get_dt() {
    return impl_->sim->dt;
}

int SimRuntime::set_spike_output_enabled(bool enable) {
    impl_->spike_output_enabled = enable;
    // Phase 1: spike output is still driven by python_api/SimWrapper.
    return 0;
}

bool SimRuntime::is_spike_output_enabled() const {
    return impl_->spike_output_enabled;
}

int SimRuntime::load_model() {
    // Phase 1 skeleton: model loading from export path is still implemented
    // in python_api/SimWrapper.
    return -1;
}

int SimRuntime::finitialize(double v_init) {
    impl_->sim->finitialize(v_init);
    return 0;
}

int SimRuntime::run(double tstop) {
    impl_->sim->tstop = tstop;
    impl_->sim->run();
    if (impl_->spike_output_enabled) {
        impl_->sim->output_spikes();
    }
    return 0;
}

int SimRuntime::continue_run(double runtime) {
    impl_->sim->continue_run(runtime);
    return 0;
}

int SimRuntime::fadvance() {
    impl_->sim->fadvance();
    return 0;
}

double SimRuntime::get_t() const {
    return impl_->sim->t;
}

int SimRuntime::flush_recorders() {
    // Phase 1 skeleton: record flushing is implemented in python_api/SimWrapper.
    return -1;
}
