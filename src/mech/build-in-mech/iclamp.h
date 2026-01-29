#pragma once
#include "mechanism.h"

class IClamp : public Mechanism
{
    public:
        IClamp(MechInitParams &param);
        virtual ~IClamp();
        void reg_node_indices(MechInitParams &param);
        void initialize_cpu(SimMechInitialParam &param);
        void initialize_gpu(SimMechInitialParam &param);
        void current_cpu(SimMechCurrentParam &param); 
        void current_gpu(SimMechCurrentParam &param); 
        void state_cpu(SimMechStateParam &param);
        void state_gpu(SimMechStateParam &param);
        VecData<double> *vecdata_amp, *vecdata_del, *vecdata_dur;
        void read_data_from_coredat(MechInitParams &param);

    private:
        double cal_current_cpu(double v, int node_index, double t);
        void sync_gpu();
        VecData<double> *vecdata_i_iclamp; // current
};

REGISTER_MECHANISM("IClamp-org",IClamp);

