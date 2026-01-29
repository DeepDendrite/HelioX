#include <cuda_runtime.h>

// Gather from a list of device pointers into a contiguous array.
// This is the reverse of batch_copy_doubles_kernel in batch_copy.cu.
__global__ void batch_gather_floats_kernel(double** gpu_ptrs, float* out, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = static_cast<float>(*gpu_ptrs[idx]);
    }
}

extern "C" void launch_batch_gather_floats(double** d_gpu_ptrs, float* d_out, int count) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    batch_gather_floats_kernel<<<grid_size, block_size>>>(d_gpu_ptrs, d_out, count);
}
