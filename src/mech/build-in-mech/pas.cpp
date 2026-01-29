#include "pas.h"
#include <math.h>
#include <string.h>

Pas::Pas(MechInitParams &param) : Mechanism(param)
{
}

Pas::~Pas()
{
    try_delete(vecdata_g_pas);
    try_delete(vecdata_e_pas); 
}

void Pas::reg_node_indices(MechInitParams &param)
{
    //e_pas = -60;
    //g_pas = 0.00016666;
    auto node_count = param.node_count;
    vecdata_e_pas = new VecData<double>(mode, -60, node_count);
    vecdata_g_pas = new VecData<double>(mode, 0.00016666, node_count);
}

void Pas::read_data_from_coredat(MechInitParams &param)
{
    auto param_size = param.data_size;
    auto data = param.data;
    auto n = param.node_count;
    
    double* g = this->vecdata_g_pas->get_cpu_data();
    double* e = this->vecdata_e_pas->get_cpu_data();
    for (int inode = 0; inode < n; inode++)
    {
        g[inode] = data[inode * param_size + 0];
        e[inode] = data[inode * param_size + 1];
    }

    if (mode == GPU)
    {
        this->vecdata_g_pas->update_gpu_data_from_cpu();
        this->vecdata_e_pas->update_gpu_data_from_cpu();
    }
}

void Pas::initialize_cpu(SimMechInitialParam &param)
{
    //do nothing
}


void Pas::current_cpu(SimMechCurrentParam &param)
{
    double *vec_v = param.v;
    double *vec_rhs = param.rhs;
    double *vec_d = param.d;
    double _rhs, _g, _v;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        _v = vec_v[node_index];
        _g = cal_current_cpu(_v + 0.001, i);
        _rhs = cal_current_cpu(_v, i);
        _g = (_g - _rhs) / 0.001;
        vec_rhs[node_index] -= _rhs;
        vec_d[node_index] += _g;
    }

}

void Pas::state_cpu(SimMechStateParam &param)
{
    //do nothing
}

double Pas::cal_current_cpu(double v, int node_index)
{
    double _current = 0;
    double* i_mech = this->vecdata_i_mech->get_cpu_data();
    double* g_pas = vecdata_g_pas->get_cpu_data();
    double* e_pas = vecdata_e_pas->get_cpu_data();
    i_mech[node_index] = g_pas[node_index] * (v - e_pas[node_index]);
    _current += i_mech[node_index];
    //_current += g_pas * (v - e_pas);
    return _current;
}
