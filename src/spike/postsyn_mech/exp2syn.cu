#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "utils.h"
#include "exp2syn.h"

__global__ void exp2syn_cuda_initialize_kernel(double* tau1, double* tau2, double* A, double* B, double* factor, int nnode)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	double tp;
	if (i < nnode)
	{
		if (tau1[i] / tau2[i] > 0.9999)
		{
			tau1[i] = 0.9999 * tau2[i];
		}
		if (tau1[i] / tau2[i] < 1e-9)
		{
			tau1[i] = tau2[i] * 1e-9;
		}
		A[i] = 0.0;
		B[i] = 0.0;
		tp = (tau1[i] * tau2[i]) / (tau2[i] - tau1[i]) * log(tau2[i] / tau1[i]);
		factor[i] = -exp(-tp / tau1[i]) + exp(-tp / tau2[i]);
		factor[i] = 1.0 / factor[i];
	}
}

__device__ double exp2syn_cuda_cal_current(double _v, int mech_index, double _e, double _A, double _B, double* g_mech, double* i_mech)
{
    double g = _B - _A;
    double i= g * (_v - _e);
    g_mech[mech_index] = g;
    i_mech[mech_index] = i;
    return i;
}
__device__ inline void exp2syn_cuda_cal_current_merge(double _v, int mech_index, double _e, double _A, double _B, double* g_mech, double* i_mech,
    double &out_g, double &out_i)
{
    out_g = _B - _A;
    out_i= out_g * (_v - _e);
    g_mech[mech_index] = out_g;
    i_mech[mech_index] = out_i;
}
__global__ void exp2syn_cuda_current_kernel(double* vec_v, double* vec_d, double* vec_rhs,    
                                            double*i_mech, double* g_mech, double* e, double* A, double* B, double* area, 
                                            int* node_indices, int nnode, double t)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int inode;
    double _v, _g, _rhs, _mfact, _nd_area;
    if (i < nnode)
    {
        inode = node_indices[i];
        _v = vec_v[inode];
        _nd_area = area[inode];

        // _g = exp2syn_cuda_cal_current(_v + 0.001, i, e[i], A[i], B[i], g_mech, i_mech);
        // _rhs = exp2syn_cuda_cal_current(_v, i, e[i], A[i], B[i], g_mech, i_mech);
        // _g = (_g - _rhs) / 0.001;
        exp2syn_cuda_cal_current_merge(_v, i, e[i], A[i], B[i], g_mech, i_mech,_g,_rhs);

        _mfact = 1.0e2 / _nd_area;
        _g *= _mfact;
        _rhs *= _mfact;
        atomicAdd(vec_rhs + inode, -_rhs);
        atomicAdd(vec_d + inode, _g);
    }
}

__global__ void exp2syn_cuda_state_kernel(double* vec_v, double* A, double* B, double* tau1, double* tau2, int* node_indices, int nnode, double dt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int node_index;
    // double _v;
    if (i < nnode)
    {
        // node_index = node_indices[i];
        // _v = vec_v[node_index];
        A[i] = A[i] + (1.0 - exp(dt * (-1.0 / tau1[i]))) * (-0.0 / (-1.0 / tau1[i]) - A[i]);
        B[i] = B[i] + (1.0 - exp(dt * (-1.0 / tau2[i]))) * (-0.0 / (-1.0 / tau2[i]) - B[i]);
    }
}

__global__ void exp2syn_cuda_spike_receive_kernel(double* weight, double* factor, double* A, double* B, unsigned int* receive_idx_vec, int receive_count, double t)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int isyn;
    double _w, _factor;
    if (i < receive_count)
    {
        isyn = receive_idx_vec[i];
        _w = weight[isyn];
        _factor = factor[isyn];
        A[isyn] = A[isyn] + _w * _factor;
        B[isyn] = B[isyn] + _w * _factor;
    }
}



void Exp2Syn::initialize_gpu(SimMechInitialParam &param)
{
    double* tau1 = this->vecdata_tau1->get_gpu_data();
    double* tau2 = this->vecdata_tau2->get_gpu_data();
    double* A = this->vecdata_A->get_gpu_data();
    double* B = this->vecdata_B->get_gpu_data();
    double* factor = this->vecdata_factor->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
	exp2syn_cuda_initialize_kernel <<<block_num, nthread_per_block >>>(tau1, tau2, A, B, factor, nnode);
}


void Exp2Syn::current_gpu(SimMechCurrentParam &param)
{
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* area = this->vecdata_area->get_gpu_data();
    double* g_mech = this->vecdata_g_mech->get_gpu_data();
    double* A = this->vecdata_A->get_gpu_data();
    double* B = this->vecdata_B->get_gpu_data();
    double* i_mech = this->vecdata_i_mech->get_gpu_data();
    double* e = this->vecdata_e->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    double *vec_v = param.v;
    double *vec_d = param.d;
    double *vec_rhs = param.rhs;
    double t = param.t;
    exp2syn_cuda_current_kernel<<<block_num, nthread_per_block, 0, stream>>>(vec_v, vec_d, vec_rhs, i_mech, g_mech, e, A, B, area, node_indices, nnode, t);
}


void Exp2Syn::state_gpu(SimMechStateParam &param)
{
    double dt = param.dt;
    double* vec_v = param.v;
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* A = this->vecdata_A->get_gpu_data();
    double* B = this->vecdata_B->get_gpu_data();
    double* tau1 = this->vecdata_tau1->get_gpu_data();
    double* tau2 = this->vecdata_tau2->get_gpu_data();
    
    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    exp2syn_cuda_state_kernel<<<block_num, nthread_per_block,0 , stream>>>(vec_v, A, B, tau1, tau2, node_indices, nnode, dt);
}


bool Exp2Syn::net_receive_gpu(double t)
{
    if (receive_count <= 0){
        return false;
    }
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_gpu_data();
    double* weight = this->vecdata_weights->get_gpu_data();
    double* factor = this->vecdata_factor->get_gpu_data();
    double* A = this->vecdata_A->get_gpu_data();
    double* B = this->vecdata_B->get_gpu_data();

    int block_num = (receive_count + nthread_per_block - 1) / nthread_per_block;
    exp2syn_cuda_spike_receive_kernel <<<block_num, nthread_per_block >>>(weight, factor, A, B, receive_idx_vec, receive_count, t);
    return false;
}


void Exp2Syn::sync_gpu(){
    cuda_stream_sync(cuda_stream);
}