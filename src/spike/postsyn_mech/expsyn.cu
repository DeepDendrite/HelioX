#include "utils.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "expsyn.h"

__global__ void expsyn_cuda_initialize_kernel(double* g, int nnode)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nnode)
	{
		g[i] = 0;
	}
}

__device__ double expsyn_cuda_cal_current(double _v, int mech_index, double _e, double* g_mech, double* i_mech)
{
    double _current = 0.0;
    // g_mech[mech_index]  = _B - _A;
    i_mech[mech_index] = g_mech[mech_index] * (_v - _e);
    _current += i_mech[mech_index];
    return _current;
}

__global__ void expsyn_cuda_current_kernel(double* vec_v, double* vec_d, double* vec_rhs,    
                                            double*i_mech, double* g_mech, double* e, double* area, int* node_indices, int nnode, double t)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int inode;
    double _v, _g, _rhs, _mfact, _nd_area;
    if (i < nnode)
    {
        inode = node_indices[i];
        _v = vec_v[inode];
        _nd_area = area[inode];
        _g = expsyn_cuda_cal_current(_v + 0.001, i, e[i], g_mech, i_mech);
        _rhs = expsyn_cuda_cal_current(_v, i, e[i], g_mech, i_mech);
        _g = (_g - _rhs) / 0.001;
        _mfact = 1.0e2 / _nd_area;
        _g *= _mfact;
        _rhs *= _mfact;
        atomicAdd(vec_rhs + inode, -_rhs);
        atomicAdd(vec_d + inode, _g);
        //vec_rhs[inode] -= _rhs;
        //vec_d[inode] += _g;
    }
}

__global__ void expsyn_cuda_state_kernel(double* vec_v, double* tau, double* g_mech, int nnode, double dt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode)
    {
        g_mech[i] = g_mech[i] + (1. - exp(dt * ((-1.0) / tau[i]))) * (-(0.0) / ((-1.0) / tau[i]) - g_mech[i]);
    }
}

__global__ void expsyn_cuda_spike_receive_kernel(double* weight, double* g_mech, unsigned int* receive_idx_vec, int receive_count, double t)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int isyn;
    double _w;
    if (i < receive_count)
    {
        isyn = receive_idx_vec[i];
        _w = weight[isyn];
        g_mech[isyn] += _w;
    }
}

void ExpSyn::initialize_gpu(SimMechInitialParam &param)
{
    double* g = this->vecdata_g_mech->get_gpu_data();
    
    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
	expsyn_cuda_initialize_kernel <<<block_num, nthread_per_block >>>(g, nnode);
}
void ExpSyn::current_gpu(SimMechCurrentParam &param)
{
    double *vec_v = param.v;
    double *vec_d = param.d;
    double *vec_rhs = param.rhs;
    double t = param.t;
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* area = this->vecdata_area->get_gpu_data();
    double* i_mech = this->vecdata_i_mech->get_gpu_data();
    double* g_mech = this->vecdata_g_mech->get_gpu_data();
    double* e = this->vecdata_e->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    expsyn_cuda_current_kernel<<<block_num, nthread_per_block, 0, stream>>>(vec_v, vec_d, vec_rhs, i_mech, g_mech, e, area, node_indices, nnode, t);
}
void ExpSyn::state_gpu(SimMechStateParam &param)
{
    double *vec_v = param.v;
    double dt = param.dt;
    double* tau = this->vecdata_tau->get_gpu_data();
    double* g_mech = this->vecdata_g_mech->get_gpu_data();
    
    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    expsyn_cuda_state_kernel<<<block_num, nthread_per_block,0 , stream>>>(vec_v, tau, g_mech, nnode, dt);
}

bool ExpSyn::net_receive_gpu(double t)
{
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_gpu_data();
    double* weight = this->vecdata_weights->get_gpu_data();
    double* g_mech = this->vecdata_g_mech->get_gpu_data();
    
    if (receive_count > 0)
	{
		int block_num = (receive_count + nthread_per_block - 1) / nthread_per_block;
		expsyn_cuda_spike_receive_kernel <<<block_num, nthread_per_block >>>(weight, g_mech, receive_idx_vec, receive_count, t);
	}
    return false;
}

void ExpSyn::sync_gpu(){
    cuda_stream_sync(cuda_stream);
}
