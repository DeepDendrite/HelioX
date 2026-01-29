#include "iclamp.h"
#include <string.h>


IClamp::IClamp(MechInitParams &param) : Mechanism(param)
{
    need_area = true;
}

IClamp::~IClamp()
{
    try_delete(vecdata_i_iclamp);
    try_delete(vecdata_amp);
    try_delete(vecdata_del);
    try_delete(vecdata_dur);
}

void IClamp::reg_node_indices(MechInitParams &param)
{
    auto node_count = param.node_count;
    vecdata_amp = new VecData<double>(mode, 0.0, node_count);
    vecdata_del = new VecData<double>(mode, 0.0, node_count);
    vecdata_dur = new VecData<double>(mode, 0.0, node_count);
    vecdata_i_iclamp = new VecData<double>(mode, 0.0, node_count);
}

void IClamp::read_data_from_coredat(MechInitParams &param)
{
    auto n = param.node_count;
    auto param_size = param.data_size;
    auto data = param.data;

    double* del = this->vecdata_del->get_cpu_data(); //0
    double* dur = this->vecdata_dur->get_cpu_data(); //1
    double* amp = this->vecdata_amp->get_cpu_data(); //2
    printf_debug("IClamp: n=%d\n",n);
    for (int inode = 0; inode < n; inode++)
    {
        del[inode] = data[inode * param_size + 0];
        dur[inode] = data[inode * param_size + 1];
        amp[inode] = data[inode * param_size + 2];
        printf_debug("IClamp inode[%d]: del=%lf dur=%lf amp=%lf\n",inode,del[inode],dur[inode],amp[inode]);
    }

    if (mode == GPU)
    {
        this->vecdata_del->update_gpu_data_from_cpu();
        this->vecdata_dur->update_gpu_data_from_cpu();
        this->vecdata_amp->update_gpu_data_from_cpu();
    }
}

void IClamp::initialize_cpu(SimMechInitialParam &param)
{
    double* i_iclamp = this->vecdata_i_iclamp->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        i_iclamp[i] = 0;
    }
}


void IClamp::current_cpu(SimMechCurrentParam &param)
{
    double *v = param.v;
    double *vec_rhs = param.rhs;
    double *vec_d = param.d;
    double t = param.t;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* area = this->vecdata_area->get_cpu_data();
    double _g, _rhs, _v, nd_area;
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        nd_area = area[node_index];
        _v = v[node_index];
        _g = cal_current_cpu(_v+0.001, i, t);
        _rhs = cal_current_cpu(_v, i, t);
        _g = (_g - _rhs) / 0.001;
        _g *= 1.e2 / nd_area;
        _rhs *= 1.e2 / nd_area;
        vec_d[node_index] -= _g;
        vec_rhs[node_index] += _rhs;
		//if (t >= 200 && t < 210)
		//{
		//	printf("iclamp cpu rhs:%f\n", _rhs);
		//}
    }
}

void IClamp::state_cpu(SimMechStateParam &param)
{
    //do nothing
}

double IClamp::cal_current_cpu(double v, int nodeIdx, double t)
{
    double _current = 0;
    double* del = this->vecdata_del->get_cpu_data();
    double* dur = this->vecdata_dur->get_cpu_data();
    double* amp = this->vecdata_amp->get_cpu_data();
    double* i_iclamp = this->vecdata_i_iclamp->get_cpu_data();
    if (t < del[nodeIdx] + dur[nodeIdx] && t >= del[nodeIdx])
    {
        i_iclamp[nodeIdx] = amp[nodeIdx];
    }
    else
    {
        i_iclamp[nodeIdx] = 0;
    }
    _current += i_iclamp[nodeIdx];
    return _current;
}




