#include <cuda_runtime.h>
#include <stdint.h>

__global__ void batch_scatter_add_f32_kernel(const float* values,
                                             const int32_t* dest,
                                             const float* scale,
                                             float* out,
                                             int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const int32_t d = dest[idx];
        atomicAdd(&out[d], values[idx] * scale[idx]);
    }
}

extern "C" void launch_batch_scatter_add_f32(const float* d_values,
                                             const int32_t* d_dest,
                                             const float* d_scale,
                                             float* d_out,
                                             int count) {
    if (count <= 0) {
        return;
    }
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    batch_scatter_add_f32_kernel<<<grid_size, block_size>>>(d_values, d_dest, d_scale, d_out, count);
}
