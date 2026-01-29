#include "utils.h"
#include "iclamp.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void iclamp_cuda_initialize_kernel(double* i_iclamp, int nnode)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnode)
    {
        i_iclamp[i] = 0.0;
    }
}

__device__ double iclamp_cuda_cal_current(double _v, int mech_index, double _del, double _dur, double _amp, double* i_iclamp, double t)
{
    if (t < _del + _dur && t >= _del)
    {
        i_iclamp[mech_index] = _amp;
    }
    else
    {
        i_iclamp[mech_index] = 0;
    }
    
    return i_iclamp[mech_index];
}

__global__ void iclamp_cuda_current_kernel(double* vec_v, double* vec_d, double* vec_rhs, double* area, double* del, double* dur, double* amp, double* i_iclamp, int* node_indices, int nnode, double t)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    double _g, _rhs, _v, nd_area;
    int node_index;
    if (i < nnode)
    {
        node_index = node_indices[i];
        nd_area = area[node_index];
        _v = vec_v[node_index];
        _g = iclamp_cuda_cal_current(_v + 0.001, i, del[i], dur[i], amp[i], i_iclamp, t);
        _rhs = iclamp_cuda_cal_current(_v, i, del[i], dur[i], amp[i], i_iclamp, t);
        _g = (_g - _rhs) / 0.001;
        _g = _g * 1.e2 / nd_area;
        _rhs = _rhs * 1.e2 / nd_area;
        
        atomicAdd(vec_d + node_index, -_g);
        atomicAdd(vec_rhs + node_index, _rhs);
		/*if (t >= 200 && t < 210)
		{
			printf("iclamp gpu rhs:%f\n", _rhs);
		}*/
    }
}

void IClamp::initialize_gpu(SimMechInitialParam &param)
{
    double* i_iclamp = this->vecdata_i_iclamp->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    iclamp_cuda_initialize_kernel<<<block_num, nthread_per_block>>>(i_iclamp, nnode);

}

void IClamp::current_gpu(SimMechCurrentParam &param)
{
    double t = param.t;
    double* vec_v = param.v;
    double* vec_d = param.d;
    double* vec_rhs = param.rhs;
    double* area = this->vecdata_area->get_gpu_data();
    double* del = this->vecdata_del->get_gpu_data();
    double* dur = this->vecdata_dur->get_gpu_data();
    double* amp = this->vecdata_amp->get_gpu_data();
    double* i_iclamp = this->vecdata_i_iclamp->get_gpu_data();
    int* node_indices = this->vecdata_node_indices->get_gpu_data();

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    iclamp_cuda_current_kernel<<<block_num, nthread_per_block,0,stream>>>(vec_v, vec_d, vec_rhs, area, del, dur, amp, i_iclamp, node_indices, nnode, t);

}

void IClamp::state_gpu(SimMechStateParam &param)
{
    //do nothing
}

void IClamp::sync_gpu(){
    cuda_stream_sync(cuda_stream);
}
