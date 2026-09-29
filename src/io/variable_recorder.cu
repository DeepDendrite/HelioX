#include "utils.h"
#include "variable_recorder.h"

void VariableRecorder::initialize(Mode mode) {
    //如果没弄错的话，现在不再需要在这里初始化buffer了
    //因为buffer的初始化是在push_back的时候进行的

    bufferTable.update_gpu_from_cpu();
}

// kernel: staged channels are self-contained, so the device side only samples
// the current source pointer into the local ring buffer.
__global__ void log_data_kernel(StagingChannel *wrapper, int buffer_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < buffer_num){
        wrapper[idx].stage_sample_single();
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

__global__ void flush_kernel(StagingChannel *wrapper, int buffer_num)
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
        items[i].sync_cpu_from_gpu();
    }
    int block_num = (buffer_count + nthread_per_block - 1) / nthread_per_block;
    flush_kernel<<<block_num,nthread_per_block>>>(bufferTable.get_gpu_data(), buffer_count);
    cudaDeviceSynchronize();
    put_data_to_hdf5();
    put_data_to_ipc_buf();
    buffer_size = 0;
}
