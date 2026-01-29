#include "utils.h"
#include "hh.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <math.h>

__device__ double hh_cuda_cal_current(double _v, int mech_index, double _gnabar, double _gkbar, double _m, double _h,
                                      double _n, double _ena, double _ek, double _el, double* gna, double* ina, double* gk,
                                      double* ik, double* gl, double* il, double* i_mech)
{
    double _current = 0;
    gna[mech_index] = _gnabar * _m * _m * _m * _h;
    ina[mech_index] = gna[mech_index] * (_v - _ena);
    gk[mech_index] = _gkbar * _n * _n * _n * _n;
    ik[mech_index] = gk[mech_index] * (_v - _ek);
    il[mech_index] = gl[mech_index] * (_v - _el);
    i_mech[mech_index] = ina[mech_index] + ik[mech_index] + il[mech_index];
	//printf("i:%d v:%f gnabar:%f m:%f n:%f h:%f ina:%f ik:%f il:%f\n", mech_index, _v, _gnabar, _m, _n, _h, ina[mech_index], ik[mech_index], il[mech_index]);
    _current += i_mech[mech_index];
    return _current;
}

__global__ void hh_cuda_current_kernel(double* vec_v, double* vec_d, double* vec_rhs, 
                                         
                                       double* i_mech, double* gnabar, double* gkbar,
                                       double* m, double* h, double* n,
                                       double* ina, double* ik, double* il,
                                       double* ena, double* ek, double* el,
									   double* gna, double* gk, double* gl,
                                       int* node_indices, int nnode, double t)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double _v, _g, _rhs, _m, _h, _n, _ena, _ek, _el, _gnabar, _gkbar;
    //double _dina, _dik;
    int inode;
    if (i < nnode)
    {
        inode = node_indices[i];
        _v = vec_v[inode];
        _m = m[i];
        _h = h[i];
        _n = n[i];
        _ena = ena[i];
        _ek = ek[i];
        _el = el[i];
        _gnabar = gnabar[i];
        _gkbar = gkbar[i];
        _g = hh_cuda_cal_current(_v + 0.001, i, _gnabar, _gkbar, _m, _h, _n, _ena, _ek, _el, gna, ina, gk, ik, gl, il, i_mech);
        //_dina = ina[i];
        //_dik = ik[i];
        _rhs = hh_cuda_cal_current(_v, i, _gnabar, _gkbar, _m, _h, _n, _ena, _ek, _el, gna, ina, gk, ik, gl, il, i_mech);
		//printf("hh current gpu rhs[%d]:%f ", inode, vec_rhs[inode]);
        _g = (_g - _rhs) / 0.001;

        atomicAdd(vec_rhs + inode, -_rhs);
        atomicAdd(vec_d + inode, _g);
		//printf("i:%d nnode:%d _g:%f _rhs:%f rhs[%d]:%f\n", i, nnode, _g, _rhs, inode, vec_rhs[inode]);
    }
}

__device__ double hh_cuda_vtrap(double x, double y)
{
    if (fabs(x / y) < 1e-6)
    {
        return y * (1 - x / y / 2);
    }
    else
    {
        return x / (exp(x / y) - 1);
    }
}

__device__ void hh_cuda_rates(double _v, int i, double* minf, double* mtau, double* hinf, double* htau, double* ninf, double* ntau, double celsius)
{
    double q10, alpha, beta, sum;
    q10 = pow(3.0, (celsius - 6.3) / 10);
    alpha = 0.1 * hh_cuda_vtrap(-(_v + 40), 10);
    beta = 4 * exp(-(_v + 65) / 18);
    sum = alpha + beta;
    mtau[i] = 1 / (q10 * sum);
    minf[i] = alpha / sum; 

    alpha = 0.07 * exp(-(_v + 65) / 20);
    beta = 1 / (exp(-(_v + 35) / 10) + 1);
    sum = alpha + beta;
    htau[i] = 1 / (q10 * sum);
    hinf[i] = alpha / sum;

    alpha = 0.01 * hh_cuda_vtrap(-(_v + 55), 10);
    beta = 0.125 * exp(-(_v + 65) / 80);
    sum = alpha + beta;
    ntau[i] = 1 / (q10 * sum);
    ninf[i] = alpha / sum;
}

__global__ void hh_cuda_state_kernel(double* vec_v, double* m, double* h, double* n,
                                     double* minf, double* mtau, double* hinf, double* htau,
                                     double* ninf, double* ntau, int* node_indices,
                                     int nnode, double dt, double celsius)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double _v;
    int inode;
    if (i < nnode)
    {
        inode = node_indices[i];
        _v = vec_v[inode];
        hh_cuda_rates(_v, i, minf, mtau, hinf, htau, ninf, ntau, celsius);
		m[i] = m[i] + (1.0 - exp(dt*((((-1.0))) / mtau[i])))*(-(((minf[i])) / mtau[i]) / ((((-1.0))) / mtau[i]) - m[i]);
		h[i] = h[i] + (1.0 - exp(dt*((((-1.0))) / htau[i])))*(-(((hinf[i])) / htau[i]) / ((((-1.0))) / htau[i]) - h[i]);
		n[i] = n[i] + (1.0 - exp(dt*((((-1.0))) / ntau[i])))*(-(((ninf[i])) / ntau[i]) / ((((-1.0))) / ntau[i]) - n[i]);
    }
}

__global__ void hh_cuda_initialize_kernel(double* vec_v, double* m, double* h, double* n, double* minf, double* mtau,
                                          double* hinf, double* htau, double* ninf, double* ntau, int* node_indices,
                                          int nnode, double celsius)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double _v;
    int inode;
	//printf("hh init nnode:%d\n", nnode);
    if (i < nnode)
    {
        inode = node_indices[i];
        _v = vec_v[inode];
        hh_cuda_rates(_v, i, minf, mtau, hinf, htau, ninf, ntau, celsius);
        m[i] = minf[i];
        h[i] = hinf[i];
        n[i] = ninf[i];
    }
}

void HHMech::initialize_gpu(SimMechInitialParam &param)
{
    double* vec_v = param.v;

    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* m = this->vecdata_m->get_gpu_data();
    double* h = this->vecdata_h->get_gpu_data();
    double* n = this->vecdata_n->get_gpu_data();
    double* minf = this->vecdata_minf->get_gpu_data();
    double* hinf = this->vecdata_hinf->get_gpu_data();
    double* ninf = this->vecdata_ninf->get_gpu_data();
    double* mtau = this->vecdata_mtau->get_gpu_data();
    double* htau = this->vecdata_htau->get_gpu_data();
    double* ntau = this->vecdata_ntau->get_gpu_data();
    
    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    hh_cuda_initialize_kernel<<<block_num, nthread_per_block>>>(vec_v, m, h, n, minf, mtau, hinf, htau, ninf, ntau, node_indices, nnode, celsius);
    
}

void HHMech::current_gpu(SimMechCurrentParam &param)
{   
    double* vec_v = param.v;
    double* vec_d = param.d;
    double* vec_rhs = param.rhs;
    double t = param.t;
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* gna = this->vecdata_gna->get_gpu_data();
    double* gnabar = this->vecdata_gnabar->get_gpu_data();
    double* m = this->vecdata_m->get_gpu_data();
    double* h = this->vecdata_h->get_gpu_data();
    double* ina = this->vecdata_ina->get_gpu_data();
    double* ena = this->vecdata_ena->get_gpu_data();
    double* gk = this->vecdata_gk->get_gpu_data();
    double* gkbar = this->vecdata_gkbar->get_gpu_data();
    double* n = this->vecdata_n->get_gpu_data();
    double* ik = this->vecdata_ik->get_gpu_data();
    double* ek = this->vecdata_ek->get_gpu_data();
    double* il = this->vecdata_il->get_gpu_data();
    double* gl = this->vecdata_gl->get_gpu_data();
    double* el = this->vecdata_el->get_gpu_data();
    double* i_mech = this->vecdata_i_mech->get_gpu_data();
    
    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    hh_cuda_current_kernel<<<block_num, nthread_per_block, 0, stream>>>(vec_v, vec_d, vec_rhs, i_mech, gnabar, gkbar, m, h, n, 
                                                            ina, ik, il, ena, ek, el, gna, gk, gl, node_indices, nnode, t);

}

void HHMech::state_gpu(SimMechStateParam &param)
{
    double* vec_v = param.v;
    double dt = param.dt;
    int* node_indices = this->vecdata_node_indices->get_gpu_data();
    double* m = this->vecdata_m->get_gpu_data();
    double* h = this->vecdata_h->get_gpu_data();
    double* n = this->vecdata_n->get_gpu_data();
    double* mtau = this->vecdata_mtau->get_gpu_data();
    double* minf = this->vecdata_minf->get_gpu_data();
    double* hinf = this->vecdata_hinf->get_gpu_data();
    double* htau = this->vecdata_htau->get_gpu_data();
    double* ninf = this->vecdata_ninf->get_gpu_data();
    double* ntau = this->vecdata_ntau->get_gpu_data();
    //int node_index;

    int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
    cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(cuda_stream);
    hh_cuda_state_kernel<<<block_num, nthread_per_block,0,stream>>>(vec_v, m, h, n, minf, mtau, hinf, htau, ninf, ntau, node_indices, nnode, dt, celsius);
    
}


void HHMech::sync_gpu(){
    cuda_stream_sync(cuda_stream);
}
