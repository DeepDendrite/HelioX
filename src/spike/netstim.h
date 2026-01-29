#pragma once

#include "articell.h"
#include "nrnran123.h"

class NetStim : public ArtiCell
{
    public:
        NetStim(MechInitParams &param);
        ~NetStim();
        vector<VecData<double>*> vec_ptr;
        void reg_node_indices(MechInitParams &param);
        void initialize_cpu(SimMechInitialParam &param);
        void initialize_gpu(SimMechInitialParam &param);
        void current_cpu(SimMechCurrentParam &param); 
        void current_gpu(SimMechCurrentParam &param); 
        void state_cpu(SimMechStateParam &param);
        void state_gpu(SimMechStateParam &param);
        void bbcore_read(int icnt, int dcnt, int* iArray, double* dArray, Mode mode);
        void read_data_from_coredat(MechInitParams &param);
        void set_seed(int seed);
        void sync_gpu();
        bool net_receive_cpu(double t);
        bool net_receive_gpu(double t);
        virtual double* getVarPtr(const VarDescriptor& descriptor, Mode mode) override;

    private:
        double invl(double interval, double noise);
        void next_invl();
        void init_sequence();
        double erand();

        VecData<double>* vecdata_interval; // 0
        VecData<double>* vecdata_number; // 1
        VecData<double>* vecdata_start; // 2
        VecData<double>* vecdata_noise; // 3
        VecData<double>* vecdata_event; // 4
        VecData<double>* vecdata_on; // 5
        VecData<double>* vecdata_ispike; // 6

        nrnran123_State* rng_state;
};

REGISTER_MECHANISM("NetStim",NetStim);
