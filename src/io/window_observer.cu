#include "window_observer.h"

#include <algorithm>
#include <cstdio>

#include "utils.h"

int WindowObserver::begin(const std::vector<int>& handles, int steps) {
    if (steps <= 0) {
        std::printf("WindowObserver::begin: invalid step count %d\n", steps);
        return -1;
    }
    if (active) {
        std::printf("WindowObserver::begin: previous window is still active\n");
        return -1;
    }
    initialize();
    step_count = steps;
    cursor = 0;
    active = true;
    active_slots.clear();
    active_rows.resize(static_cast<int>(handles.size()));
    for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
        active_rows.cpu(i) = points.index_of_handle(handles[i]);
        active_slots[handles[i]] = i;
    }
    active_rows.update_gpu_data_from_cpu();
    window.resize(active_rows.size() * step_count);
    return 0;
}

void WindowObserver::log_data_cpu() {
    if (!active || cursor >= step_count) return;
    const int n = active_rows.size();
    const auto* point_data = points.get_cpu_data();
    for (int i = 0; i < n; ++i) {
        const auto& point = point_data[active_rows.cpu(i)];
        window.cpu(i * step_count + cursor) = point.source_cpu ? static_cast<float>(*point.source_cpu) : 0.0f;
    }
    ++cursor;
}

__global__ void window_observer_log_kernel(const WindowObservePoint* points,
                                           const int* active_rows,
                                           float* window,
                                           int n,
                                           int step_count,
                                           int cursor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double* src = points[active_rows[i]].source_gpu;
        window[i * step_count + cursor] = src ? static_cast<float>(*src) : 0.0f;
    }
}

void WindowObserver::log_data_gpu() {
    if (!active || cursor >= step_count) return;
    const int n = active_rows.size();
    if (n > 0) {
        int block_num = (n + nthread_per_block - 1) / nthread_per_block;
        window_observer_log_kernel<<<block_num, nthread_per_block>>>(
            points.get_gpu_data(), active_rows.get_gpu_data(), window.get_gpu_data(), n, step_count, cursor);
    }
    ++cursor;
}

int WindowObserver::read_f32(const std::vector<int>& handles, float* out_cpu, int n_handle, int n_step) {
    if (out_cpu == nullptr || static_cast<int>(handles.size()) != n_handle || n_step < 0) {
        return -1;
    }
    if (n_step > step_count || n_step > cursor) {
        std::printf("WindowObserver::read_f32: requested %d steps, recorded %d/%d\n", n_step, cursor, step_count);
        return -1;
    }
    if (mode == GPU) {
        window.update_cpu_data_from_gpu();
    }
    for (int i = 0; i < n_handle; ++i) {
        auto it = active_slots.find(handles[i]);
        if (it == active_slots.end()) return -1;
        const int slot = it->second;
        const float* src = window.get_cpu_data() + static_cast<size_t>(slot) * static_cast<size_t>(step_count);
        std::copy(src, src + n_step, out_cpu + static_cast<size_t>(i) * static_cast<size_t>(n_step));
    }
    return 0;
}
