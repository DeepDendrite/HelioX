#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include "device_dynamic_table.h"
#include "utils.h"
#include "vecdata.h"

struct WindowObservePoint {
    double* source_cpu = nullptr;
    double* source_gpu = nullptr;
};

struct WindowObserver {
    Mode mode;
    DynamicDeviceTable<WindowObservePoint, VarDescriptor> points;
    VecData<int> active_rows;
    std::unordered_map<int, int> active_slots;
    VecData<float> window;
    int step_count = 0;
    int cursor = 0;
    bool active = false;

    explicit WindowObserver(Mode mode = CPU) : mode(mode), points(mode), active_rows(mode), window(mode) {}

    int push_back(const VarDescriptor& desc, double* source_cpu, double* source_gpu) {
        return points.add_or_update(desc, WindowObservePoint{source_cpu, source_gpu});
    }

    void initialize() { points.update_gpu_from_cpu(); }
    void reset() {
        active = false;
        step_count = 0;
        cursor = 0;
        active_slots.clear();
        active_rows.resize(0);
    }

    int begin(const std::vector<int>& handles, int steps);
    void log_data_cpu();
    void log_data_gpu();
    int read_f32(const std::vector<int>& handles, float* out_cpu, int n_handle, int n_step);
};
