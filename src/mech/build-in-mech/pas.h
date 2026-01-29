#pragma once
#include "mechanism.h"
#include "vecdata.h"

class Pas : public Mechanism
{
    public:
        Pas(MechInitParams &param);
        virtual ~Pas();
        void reg_node_indices(MechInitParams &param);
        void initialize_cpu(SimMechInitialParam &param);
        void initialize_gpu(SimMechInitialParam &param);
        void current_cpu(SimMechCurrentParam &param); 
        void current_gpu(SimMechCurrentParam &param); 
        void state_cpu(SimMechStateParam &param);
        void state_gpu(SimMechStateParam &param);
        void sync_gpu();
        void read_data_from_coredat(MechInitParams &param);

    private:
        double cal_current_cpu(double v, int node_index);
        //double cal_current_gpu(double v, int node_index);
        VecData<double> *vecdata_g_pas, *vecdata_e_pas;
};


REGISTER_MECHANISM("pas-org",Pas);
