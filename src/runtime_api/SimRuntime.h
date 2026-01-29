#pragma once

#include <memory>
#include <string>

// Phase 1: Skeleton Runtime API.
//
// Contract:
// - This layer must not depend on nanobind/Python headers.
// - It provides a stable C++ API surface that can be used by python_api bindings,
//   a future CLI, or other frontends.
//
// Implementation notes:
// - For now, SimRuntime is a thin wrapper that delegates to the existing simulation
//   implementation (`Simulate`) through an opaque internal state.
// - Later phases will migrate logic out of src/python_api/SimWrapper into
//   src/runtime_api/{core,learn}.

class SimRuntime final {
public:
    SimRuntime();
    ~SimRuntime();

    SimRuntime(const SimRuntime&) = delete;
    SimRuntime& operator=(const SimRuntime&) = delete;
    SimRuntime(SimRuntime&&) = default;
    SimRuntime& operator=(SimRuntime&&) = default;

    // core: control
    int set_data_path(const std::string& path);
    int set_device(const std::string& dev);
    int set_output_dir(const std::string& dir);
    int set_permute_type(int type);
    int set_dt(double dt);
    double get_dt();

    int set_spike_output_enabled(bool enable);
    bool is_spike_output_enabled() const;

    int load_model();
    int finitialize(double v_init);
    int run(double tstop);
    int continue_run(double runtime);
    int fadvance();
    double get_t() const;
    int flush_recorders();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

