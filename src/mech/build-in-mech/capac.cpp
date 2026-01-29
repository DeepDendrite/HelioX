#include "capac.h"
#include <string.h>
#include "utils.h"

Capac::Capac(MechInitParams &param) : Mechanism(param)
{
    
}

Capac::~Capac()
{
    try_delete(vecdata_cm);
    try_delete(vecdata_icap);
}

void Capac::reg_node_indices(MechInitParams &param)
{
    vecdata_cm = new VecData<double>(mode, 1.0, param.node_count);
    vecdata_icap = new VecData<double>(mode, 0.0, param.node_count);
}

void Capac::read_data_from_coredat(MechInitParams &param)
{
    auto param_size = param.data_size;
    auto data = param.data;
	double* cm = this->vecdata_cm->get_cpu_data();
	for (int inode = 0; inode < nnode; inode++)
	{
		cm[inode] = data[inode * param_size + 0];
	}

	if (mode == GPU)
	{
		this->vecdata_cm->update_gpu_data_from_cpu();
	}
}

void Capac::initialize_cpu(SimMechInitialParam &param)
{
    double* icap = this->vecdata_icap->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        icap[i] = 0;
    }
}



void Capac::current_cpu(SimMechCurrentParam &param)
{
    //do nothing
}

void Capac::cap_current_cpu(double cj, double* vec_rhs)
{
    double cfac = 0.001 * cj;
    double* icap = this->vecdata_icap->get_cpu_data();
    double* cm = this->vecdata_cm->get_cpu_data();
	int* node_indices = this->vecdata_node_indices->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        icap[i] = cfac * cm[i] * vec_rhs[node_index];
    }
}


void Capac::cap_jacob_cpu(double cj, double* vec_d)
{
    double cfac = 0.001 * cj;
    double* cm = this->vecdata_cm->get_cpu_data();
	int* node_indices = this->vecdata_node_indices->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        vec_d[node_index] += cfac * cm[i];
    }
}


void Capac::state_cpu(SimMechStateParam &param)
{
    //do nothing
}

