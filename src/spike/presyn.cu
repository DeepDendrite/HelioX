#include "utils.h"
#include "presyn.cuh"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "spike.h"
//#include <helper_functions.h>
//#include <helper_cuda.h>

__global__ void detect_spikes(double* vec_v, SpikeFlag* spk_flags, int* pre_node_indices, uint32_t* spk_vec_offset, double* threshold,
 bool* pre_flags, double t, int len,/*int* d_flag,*/int *spk_count,int* spk_idx_vec,int tot_len, int *tot_spk_count)
{
    unsigned int ipre = blockIdx.x * blockDim.x + threadIdx.x;
    double pre_v, thresh;
    unsigned int pre_node_id;
    unsigned int offset;
    
    if (ipre < tot_len){
        if (ipre < len)
        {
            thresh = threshold[ipre];
            pre_node_id = pre_node_indices[ipre];
            pre_v = vec_v[pre_node_id];
            // d_flag[ipre] = 0;
            if (pre_v > thresh && !pre_flags[ipre])
            {
                offset = spk_vec_offset[ipre];
                spk_flags[offset] = SpikeFlag::NORMAL_EVENT;//原始输出

                pre_flags[ipre] = true;
                // d_flag[ipre] = 1;
                int result_idx = atomicAdd(spk_count, 1);
                spk_idx_vec[result_idx] = ipre;
            }
            else if (pre_v <= thresh)
            {
                pre_flags[ipre] = false;
            }
        }
        if(spk_flags[ipre] == SpikeFlag::NORMAL_EVENT){
            atomicAdd(tot_spk_count,1);
        }
    }

}

extern "C" void cuda_spike_send_async(double* vec_v,
                                      SpikeFlag* spk_flags,
                                      int* pre_node_indices,
                                      uint32_t* spk_vec_offset,
                                      double* threshold,
                                      bool* pre_flags,
                                      double t,
                                      int len,
                                      int* spk_idx_vec,
                                      int tot_len,
                                      int* d_spk_num_real,
                                      int* d_spk_num_tot,
                                      int* h_spk_num_real,
                                      int* h_spk_num_tot,
                                      void* cuda_stream,
                                      void* cuda_event)
{
    cudaStream_t stream = cuda_stream ? *reinterpret_cast<cudaStream_t*>(cuda_stream) : static_cast<cudaStream_t>(0);
    cudaEvent_t event = cuda_event ? *reinterpret_cast<cudaEvent_t*>(cuda_event) : static_cast<cudaEvent_t>(nullptr);

    if (len <= 0 || tot_len <= 0) {
        if (h_spk_num_real) {
            *h_spk_num_real = 0;
        }
        if (h_spk_num_tot) {
            *h_spk_num_tot = 0;
        }
        if (d_spk_num_real) {
            cudaMemsetAsync(d_spk_num_real, 0, sizeof(int), stream);
        }
        if (d_spk_num_tot) {
            cudaMemsetAsync(d_spk_num_tot, 0, sizeof(int), stream);
        }
        if (event != nullptr) {
            cudaEventRecord(event, stream);
        }
        return;
    }

    // Reset device counters asynchronously (no device-wide sync).
    cudaMemsetAsync(d_spk_num_real, 0, sizeof(int), stream);
    cudaMemsetAsync(d_spk_num_tot, 0, sizeof(int), stream);

    // Detect spikes on the provided stream.
    const int block_num = (nthread_per_block + len - 1) / nthread_per_block;
    detect_spikes<<<block_num, nthread_per_block, 0, stream>>>(
        vec_v,
        spk_flags,
        pre_node_indices,
        spk_vec_offset,
        threshold,
        pre_flags,
        t,
        len,
        d_spk_num_real,
        spk_idx_vec,
        tot_len,
        d_spk_num_tot);

    // Copy back only 2 small counters (host memory must be pinned for true async).
    cudaMemcpyAsync(h_spk_num_real, d_spk_num_real, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_spk_num_tot, d_spk_num_tot, sizeof(int), cudaMemcpyDeviceToHost, stream);

    if (event != nullptr) {
        cudaEventRecord(event, stream);
    }

#ifdef DEBUG
    // Keep lightweight error checking in Debug; do not synchronize device-wide.
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("cuda_spike_send_async launch error:%s\n", cudaGetErrorString(cuda_status));
    }
#endif
}
