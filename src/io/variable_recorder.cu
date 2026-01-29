#include "variable_recorder.h"
#include "utils.h"

void __host__ __device__ BufferItem::log_data_single(){
    #ifdef __CUDA_ARCH__
    // 在GPU上执行时，使用GPU端的指针
    buffer.get_dev_data()[len] = *var_ptr_gpu;
    #else
    // 在CPU上执行时，使用CPU端的指针
    buffer.get_dev_data()[len] = *var_ptr_cpu;
    #endif
    len++;
}
void __host__ __device__ BufferItem::flush() {
    // printf("Flushing buffer, len: %d\n", len);
    len = 0;
}
void VariableRecorder::initialize(Mode mode) {
    //如果没弄错的话，现在不再需要在这里初始化buffer了
    //因为buffer的初始化是在push_back的时候进行的

    bufferTable.update_gpu_from_cpu();
}

//kernel，通过调用wrapper内的log_data_single实现并行
__global__ void log_data_kernel(BufferItem *wrapper, int buffer_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < buffer_num){
        wrapper[idx].log_data_single();
    }
}

void VariableRecorder::log_data_gpu() {
    auto buffer_count = bufferTable.size();
    if(buffer_count <= 0)
        return;
    int block_num = (buffer_count + nthread_per_block - 1) / nthread_per_block;
    log_data_kernel<<<block_num, nthread_per_block>>>(bufferTable.get_gpu_data(), buffer_count);
    buffer_size++;
    if(buffer_size >= buffer_capacity){
        assert(buffer_size == buffer_capacity);
        flush_gpu();
    }
}

__global__ void flush_kernel(BufferItem *wrapper, int buffer_num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < buffer_num){
        wrapper[idx].flush();
    }
}

void VariableRecorder::flush_gpu() {
    auto buffer_count = bufferTable.size();
    if(buffer_count <= 0)
        return;
    //利用bufferTable的update_cpu_from_gpu函数来更新CPU端数据
    bufferTable.update_cpu_from_gpu();
    auto items = bufferTable.get_cpu_data();
    for (int i = 0; i < buffer_count; i++) {
        items[i].buffer.update_cpu_data_from_gpu();
    }
    int block_num = (buffer_count + nthread_per_block - 1) / nthread_per_block;
    flush_kernel<<<block_num,nthread_per_block>>>(bufferTable.get_gpu_data(), buffer_count);
    cudaDeviceSynchronize();
    put_data_to_hdf5();
    put_data_to_ipc_buf();
    buffer_size = 0;
}