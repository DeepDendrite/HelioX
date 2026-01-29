#include "utils.h"
#include "pas.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline double pas_cuda_cal_current(double v, double g_pas, double e_pas)
{
    return g_pas * (v - e_pas);;
}

__global__ void pas_cuda_current_kernel(double* vec_v, double* vec_d, double* vec_rhs, 
                                          double* i_mech, 
                                        double* g_pas, double* e_pas, int* node_indices, 
                                        int nnode, double t)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nnode) return;

    // double _v, _g, _rhs, _g_pas, _e_pas;
    int inode = node_indices[i];
    double _v = vec_v[inode];
    double _g_pas = g_pas[i];
    double _e_pas = e_pas[i];
    //printf("i:%d inode:%d ", i, inode);
    //printf("g_pas:%f ", _g_pas);
    //printf("e_pas:%f\n ", _e_pas);

/*
    _g = pas_cuda_cal_current(_v + 0.001, _g_pas, _e_pas);
    _rhs = pas_cuda_cal_current(_v, _g_pas, _e_pas);
    i_mech[i] = _rhs;
    _g = (_g - _rhs) / 0.001;
*/
    
    double _rhs = pas_cuda_cal_current(_v, _g_pas, _e_pas);
    i_mech[i] = _rhs;

    atomicAdd(vec_rhs + inode, -_rhs);
    atomicAdd(vec_d + inode, _g_pas);
    // atomicAdd(vec_d + inode, _g);
}

void Pas::initialize_gpu(SimMechInitialParam &param)
{
    //do nothing
}

void Pas::current_gpu(SimMechCurrentParam &param)
{
    double* vec_v = param.v;
    double* vec_d = param.d;
    double* vec_rhs = param.rhs;
    double t = param.t;
    double* i_mech = this->vecdata_i_mech->get_gpu_data();
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* g_pas = this->vecdata_g_pas->get_gpu_data();
    double* e_pas = this->vecdata_e_pas->get_gpu_data();

    static int minGridSize = 0;
    static int block_size = 0;
    if(block_size == 0){
        cudaOccupancyMaxPotentialBlockSize(&minGridSize,&block_size,pas_cuda_current_kernel,
            0,  // 动态共享内存大小
            0   // block size 上限
        );
    }
    int block_num = (nnode + block_size - 1) / block_size;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    pas_cuda_current_kernel<<<block_num, block_size, 0, stream>>>(vec_v, vec_d, vec_rhs,  i_mech, g_pas, e_pas, node_indices, nnode, t);

}

void Pas::state_gpu(SimMechStateParam &param)
{
    //do nothing
}
void Pas::sync_gpu(){
    cuda_stream_sync(cuda_stream);
}
