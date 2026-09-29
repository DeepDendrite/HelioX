#include "staging_channel.h"

void StagingChannel::sync_cpu_from_gpu() {
    if (use_fp32_storage) {
        buffer_f32.update_cpu_data_from_gpu();
    } else {
        buffer_f64.update_cpu_data_from_gpu();
    }
}
