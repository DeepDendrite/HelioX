#include "utils.h"
#include "capac.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void cap_cuda_initialize_kernel(double* vec_v, double* icap, int nnode)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnode)
    {
        icap[i] = 0.0;
    }
}

__global__ void cap_cuda_cap_current_kernel(double* vec_rhs, double* cm, double* icap, int* node_indices, double cfac, int nnode)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    int node_index;
    if (i < nnode)
    {
        node_index = node_indices[i];
        icap[i] = cfac * cm[i] * vec_rhs[node_index];
    }
} 

__global__ void cap_cuda_cap_jacob_kernel(double* vec_d, double* cm, int* node_indices, double cfac, int nnode)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    int node_index; 
    if (i < nnode)
    {
        node_index = node_indices[i];
        vec_d[node_index] += cfac * cm[i];
    }
}

void Capac::initialize_gpu(SimMechInitialParam &param)
{
    double* icap = this->vecdata_icap->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cap_cuda_initialize_kernel<<<block_num, nthread_per_block>>>(param.v, icap, nnode);
}

void Capac::current_gpu(SimMechCurrentParam &param)
{
    //do nothing
}

void Capac::cap_current_gpu(double cj, double* vec_rhs)
{
    double cfac = 0.001 * cj;
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* icap = this->vecdata_icap->get_gpu_data();
    double* cm = this->vecdata_cm->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cap_cuda_cap_current_kernel<<<block_num, nthread_per_block>>>(vec_rhs, cm, icap, node_indices, cfac, nnode);
}
void Capac::cap_jacob_gpu(double cj, double* vec_d)
{
    double cfac = 0.001 * cj;
    double* cm = this->vecdata_cm->get_gpu_data();
    int* node_indices = this->vecdata_node_indices->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cap_cuda_cap_jacob_kernel<<<block_num, nthread_per_block>>>(vec_d, cm, node_indices, cfac, nnode);

}
void Capac::sync_gpu(){
    cuda_stream_sync(cuda_stream);
}
void Capac::state_gpu(SimMechStateParam &param)
{
    //do nothing
}