#include "exp2syn.h"
#include <cstring>
#include <cstdint>


Exp2Syn::Exp2Syn(MechInitParams &param) : PostSyn(param)
{
    need_area = true;
}

Exp2Syn::~Exp2Syn()
{
    if (vecdata_tau1)
    {
        delete vecdata_tau1;
        vecdata_tau1 = nullptr;
    }
    if (vecdata_tau2)
    {
        delete vecdata_tau2;
        vecdata_tau2 = nullptr;
    }
    if (vecdata_e)
    {
        delete vecdata_e;
        vecdata_e = nullptr;
    }
    if (vecdata_factor)
    {
        delete vecdata_factor;
        vecdata_factor = nullptr;
    }
    if (vecdata_A)
    {
        delete vecdata_A;
        vecdata_A = nullptr;
    }
    if (vecdata_B)
    {
        delete vecdata_B;
        vecdata_B = nullptr;
    }
}

void Exp2Syn::reg_node_indices(MechInitParams &param)
{
    nnode = param.node_count;
    auto node_count = param.node_count;

    if (nnode > 0)
    {
        PostSyn::reg_node_indices(param);
        vecdata_tau1 = new VecData<double>(mode, 0.1, node_count);
        vecdata_tau2 = new VecData<double>(mode, 10.0, node_count);
        vecdata_e = new VecData<double>(mode, 0.0, node_count);
        vecdata_factor = new VecData<double>(mode, 0.0, node_count);
        vecdata_A = new VecData<double>(mode, 0.0, node_count);
        vecdata_B = new VecData<double>(mode, 0.0, node_count);
    }
    

}

void Exp2Syn::read_data_from_coredat(MechInitParams &param)
{
    auto param_size = param.data_size;
    auto data = param.data;
    auto n = param.node_count;
    double* tau1 = this->vecdata_tau1->get_cpu_data(); //0
    double* tau2 = this->vecdata_tau2->get_cpu_data(); //1
    double* e = this->vecdata_e->get_cpu_data(); //2
    double* A = this->vecdata_A->get_cpu_data(); //5
    double* B = this->vecdata_B->get_cpu_data(); //6
    double* factor = this->vecdata_factor->get_cpu_data(); //7
    for (int inode = 0; inode < n; inode++)
    {
        tau1[inode] = data[inode * param_size + 0];
        tau2[inode] = data[inode * param_size + 1];
        e[inode] = data[inode * param_size + 2];
        A[inode] = data[inode * param_size + 5];
        B[inode] = data[inode * param_size + 6];
        factor[inode] = data[inode * param_size + 7];
    }

    if (mode == GPU)
    {
        this->vecdata_tau1->update_gpu_data_from_cpu();
        this->vecdata_tau2->update_gpu_data_from_cpu();
        this->vecdata_e->update_gpu_data_from_cpu();
        this->vecdata_A->update_gpu_data_from_cpu();
        this->vecdata_B->update_gpu_data_from_cpu();
        this->vecdata_factor->update_gpu_data_from_cpu();
    }

}

void Exp2Syn::initialize_cpu(SimMechInitialParam &param)
{
    double* tau1 = this->vecdata_tau1->get_cpu_data();
    double* tau2 = this->vecdata_tau2->get_cpu_data();
    double* A = this->vecdata_A->get_cpu_data();
    double* B = this->vecdata_B->get_cpu_data();
    double* factor = this->vecdata_factor->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        double tp;
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



void Exp2Syn::current_cpu(SimMechCurrentParam &param)
{
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* area = this->vecdata_area->get_cpu_data();
    double _rhs, _g, _v;
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        double nd_area = area[node_index];
        _v = param.v[node_index];
        _g = cal_current_cpu(_v + 0.001, i);
        _rhs = cal_current_cpu(_v, i);
        _g = (_g - _rhs) / 0.001;
        double mfact = 1.0e2 / nd_area;
        _g *= mfact;
        _rhs *= mfact;
        param.rhs[node_index] -= _rhs;
        param.d[node_index] += _g;
    }
}

double Exp2Syn::cal_current_cpu(double v, int mech_index)
{
    double _current = 0.0;
    double* g_mech = this->vecdata_g_mech->get_cpu_data();
    double* A = this->vecdata_A->get_cpu_data();
    double* B = this->vecdata_B->get_cpu_data();
    double* i_mech = this->vecdata_i_mech->get_cpu_data();
    double* e = this->vecdata_e->get_cpu_data();
    g_mech[mech_index] = B[mech_index] - A[mech_index];
    i_mech[mech_index] = g_mech[mech_index] * (v - e[mech_index]);
    _current += i_mech[mech_index];
    return _current;
}

void Exp2Syn::state_cpu(SimMechStateParam &param)
{
    double dt = param.dt;
    double* vec_v = param.v;
    double _v;
    int node_index;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* A = this->vecdata_A->get_cpu_data();
    double* B = this->vecdata_B->get_cpu_data();
    double* tau1 = this->vecdata_tau1->get_cpu_data();
    double* tau2 = this->vecdata_tau2->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        node_index = node_indices[i];
        _v = vec_v[node_index];
        A[i] = A[i] + (1.0 - exp(dt * (-1.0 / tau1[i]))) * (-0.0 / (-1.0 / tau1[i]) - A[i]);
        B[i] = B[i] + (1.0 - exp(dt * (-1.0 / tau2[i]))) * (-0.0 / (-1.0 / tau2[i]) - B[i]);
    }
}

bool Exp2Syn::net_receive_cpu(double t)
{
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    double* weight = this->vecdata_weights->get_cpu_data();
    double* factor = this->vecdata_factor->get_cpu_data();
    double* A = this->vecdata_A->get_cpu_data();
    double* B = this->vecdata_B->get_cpu_data();
    double _w, _factor;
    for (int i = 0; i < receive_count; i++)
    {
        int isyn = receive_idx_vec[i];
        _w = weight[isyn];
        _factor = factor[isyn];
        A[isyn] = A[isyn] + _w * _factor;
        B[isyn] = B[isyn] + _w * _factor;
        // printf("isyn:%d A:%f B:%f\n", isyn, A[isyn], B[isyn]);
    }
    return false;
}


