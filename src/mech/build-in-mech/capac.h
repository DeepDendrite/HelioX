#pragma once
#include "mechanism.h"

extern "C" {
	void cap_cuda_initialize(double* vec_v, double* icap, int nnode);
	void cap_cuda_cap_current(double* vec_rhs, double* cm, double* icap, int* node_indices, double cfac, int nnode);
	void cap_cuda_cap_jacob(double* vec_d, double* cm, int* node_indices, double cfac, int nnode);
}

class Capac : public Mechanism
{
    public:
        Capac(MechInitParams &param);
        virtual ~Capac();
        void reg_node_indices(MechInitParams &param);
        void initialize_cpu(SimMechInitialParam &param);
        void initialize_gpu(SimMechInitialParam &param);
        void current_cpu(SimMechCurrentParam &param); 
        void current_gpu(SimMechCurrentParam &param); 
        void state_cpu(SimMechStateParam &param);
        void state_gpu(SimMechStateParam &param);
        void sync_gpu();
        void cap_current_cpu(double cj, double *vec_rhs);
        void cap_current_gpu(double cj, double* vec_rhs);
        void cap_jacob_cpu(double cj, double* vec_d);
        void cap_jacob_gpu(double cj, double* vec_d);
		void read_data_from_coredat(MechInitParams &param);

        VecData<double>* vecdata_cm;
		VecData<double>* vecdata_icap;

        
};
