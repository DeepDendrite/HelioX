#pragma once

#include <cassert>

#include "vecdata.h"

#ifdef __CUDACC__
#define STAGING_CHANNEL_HD __host__ __device__
#else
#define STAGING_CHANNEL_HD
#endif

// Semantics-free staged channel:
// - owns a small device/host ring buffer
// - samples from a source pointer pair
// - knows nothing about monitor names, HDF5, or IPC output
struct StagingChannel {
    double* source_cpu = nullptr;
    double* source_gpu = nullptr;
    bool use_fp32_storage = false;
    int sample_width = 1;
    int buffer_capacity = 0;
    int len = 0;
    VecData<double> buffer_f64;
    VecData<float> buffer_f32;

    StagingChannel(Mode mode,
                   int buffer_cap,
                   int sample_width,
                   double* source_cpu,
                   double* source_gpu,
                   bool use_fp32_storage)
        : source_cpu(source_cpu),
          source_gpu(source_gpu),
          use_fp32_storage(use_fp32_storage),
          sample_width(sample_width),
          buffer_capacity(buffer_cap),
          len(0),
          buffer_f64(mode, use_fp32_storage ? 0 : buffer_cap * sample_width),
          buffer_f32(mode, use_fp32_storage ? buffer_cap * sample_width : 0) {
        assert(buffer_cap >= 0);
        assert(sample_width > 0);
    }

    STAGING_CHANNEL_HD inline void stage_sample_single() {
        const int base = len * sample_width;
#ifdef __CUDA_ARCH__
        if (use_fp32_storage) {
            for (int i = 0; i < sample_width; ++i) {
                buffer_f32.get_dev_data()[base + i] = static_cast<float>(source_gpu[i]);
            }
        } else {
            for (int i = 0; i < sample_width; ++i) {
                buffer_f64.get_dev_data()[base + i] = source_gpu[i];
            }
        }
#else
        if (use_fp32_storage) {
            for (int i = 0; i < sample_width; ++i) {
                buffer_f32.get_dev_data()[base + i] = static_cast<float>(source_cpu[i]);
            }
        } else {
            for (int i = 0; i < sample_width; ++i) {
                buffer_f64.get_dev_data()[base + i] = source_cpu[i];
            }
        }
#endif
        len++;
    }

    STAGING_CHANNEL_HD inline void flush() { len = 0; }
    void sync_cpu_from_gpu();

    int staged_count() const { return len; }
    int staged_width() const { return sample_width; }
    int staged_capacity() const { return buffer_capacity; }
    int staged_storage_count() const { return buffer_capacity * sample_width; }

    double* mutable_f64_cpu_data() { return buffer_f64.get_cpu_data(); }
    float* mutable_f32_cpu_data() { return buffer_f32.get_cpu_data(); }
};

#undef STAGING_CHANNEL_HD
