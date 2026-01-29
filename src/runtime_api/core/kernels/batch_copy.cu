#include <cuda_runtime.h>

// 批量拷贝kernel
__global__ void batch_copy_doubles_kernel(double** gpu_ptrs, const double* cpu_values, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        *gpu_ptrs[idx] = cpu_values[idx];
    }
}

// C++接口
extern "C" void launch_batch_copy_doubles(double** d_gpu_ptrs, const double* d_cpu_values, int count) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    batch_copy_doubles_kernel<<<grid_size, block_size>>>(d_gpu_ptrs, d_cpu_values, count);
    cudaDeviceSynchronize();
}
