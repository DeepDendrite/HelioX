#include "expsyn.h"
#include <cstring>
#include <cstdint>
#include "utils.h"


ExpSyn::ExpSyn(MechInitParams &param) : PostSyn(param)
{
    need_area = true;
}

ExpSyn::~ExpSyn()
{
    try_delete(vecdata_e);
}

void ExpSyn::reg_node_indices(MechInitParams &param)
{
    auto node_count = param.node_count;
    auto buf = param.nodeindices;
    if (nnode > 0)
    {
        PostSyn::reg_node_indices(param);
        vecdata_tau = new VecData<double>(mode, 0.1, node_count);
        vecdata_e = new VecData<double>(mode, 0.0, node_count);
    }
    

}

void ExpSyn::read_data_from_coredat(MechInitParams &param)
{
    /*#define t _nt->_t
#define dt _nt->_dt
#define tau _ml->template fpfield<0>(_iml)
#define tau_columnindex 0
#define e _ml->template fpfield<1>(_iml)
#define e_columnindex 1
#define i _ml->template fpfield<2>(_iml)
#define i_columnindex 2
#define g _ml->template fpfield<3>(_iml)
#define g_columnindex 3
#define Dg _ml->template fpfield<4>(_iml)
#define Dg_columnindex 4
#define v _ml->template fpfield<5>(_iml)
#define v_columnindex 5
#define _g _ml->template fpfield<6>(_iml)
#define _g_columnindex 6
#define _tsav _ml->template fpfield<7>(_iml)
#define _tsav_columnindex 7
#define _nd_area *_ml->dptr_field<0>(_iml)*/
    auto n = param.node_count;
    auto param_size = param.data_size;
    auto data = param.data;
    double* tau = this->vecdata_tau->get_cpu_data(); //0
    double* e = this->vecdata_e->get_cpu_data(); //1

    for (int inode = 0; inode < n; inode++)
    {
        tau[inode] = data[inode * param_size + 0];
        e[inode] = data[inode * param_size + 1];
    }

    if (mode == GPU)
    {
        this->vecdata_tau->update_gpu_data_from_cpu();
        this->vecdata_e->update_gpu_data_from_cpu();
    }

}

void ExpSyn::initialize_cpu(SimMechInitialParam &param)
{
    double* g = this->vecdata_g_mech->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        g[i] = 0;
    }

}


void ExpSyn::current_cpu(SimMechCurrentParam &param)
{
    double* vec_v = param.v;
    double* vec_rhs = param.rhs;
    double* vec_d = param.d;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* area = this->vecdata_area->get_cpu_data();
    double _rhs, _g, _v;
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        double nd_area = area[node_index];
        double mfact = 1.0e2 / nd_area;

        _v = vec_v[node_index];
        _g = cal_current_cpu(_v + 0.001, i);//I(V+deltaV)
        _rhs = cal_current_cpu(_v, i);//I(V)
        _g = (_g - _rhs) / 0.001;//R = DeltaI/DeltaV
        _g *= mfact;
        _rhs *= mfact;
        vec_rhs[node_index] -= _rhs;
        vec_d[node_index] += _g;//jacob
    }
}

double ExpSyn::cal_current_cpu(double v, int mech_index)
{
    double* i_mech = this->vecdata_i_mech->get_cpu_data();
    double* g_mech = this->vecdata_g_mech->get_cpu_data();
    double* e = this->vecdata_e->get_cpu_data();
    /*
    i = g*(v - e)
    */
    double _current = g_mech[mech_index] * (v - e[mech_index]);
    i_mech[mech_index] = _current;
    return _current;
}



void ExpSyn::state_cpu(SimMechStateParam &param)
{
    double dt = param.dt;
    double* tau = this->vecdata_tau->get_cpu_data();
    double* g_mech = this->vecdata_g_mech->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        g_mech[i] = g_mech[i] + (1. - exp(dt * ((-1.0) / tau[i]))) * (-(0.0) / ((-1.0) / tau[i]) - g_mech[i]);
    }
}



bool ExpSyn::net_receive_cpu(double t)
{
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    double* weight = this->vecdata_weights->get_cpu_data();
    double* g_mech = this->vecdata_g_mech->get_cpu_data();
    double _w;
    for (int i = 0; i < receive_count; i++)
    {
        int isyn = receive_idx_vec[i];
        _w = weight[isyn];
        g_mech[isyn] += _w;
        // printf("isyn:%d g:%f\n", isyn, g_mech[isyn]);
    }
    return false;
}
