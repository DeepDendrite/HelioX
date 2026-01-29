#pragma once

#include "postsyn.h"

class ExpSyn : public PostSyn
{
public:
    ExpSyn(MechInitParams &param);
    ~ExpSyn();
    void reg_node_indices(MechInitParams &param);
    void initialize_cpu(SimMechInitialParam &param);
    void initialize_gpu(SimMechInitialParam &param);
    void current_cpu(SimMechCurrentParam &param); 
    void current_gpu(SimMechCurrentParam &param); 
    void state_cpu(SimMechStateParam &param);
    void state_gpu(SimMechStateParam &param);
    void sync_gpu();
    bool net_receive_cpu(double t); //NET_RECEIVE part in .mod file
    bool net_receive_gpu(double t); //NET_RECEIVE part in .mod file
    void read_data_from_coredat(MechInitParams &param);


private:
    VecData<double>* vecdata_tau, * vecdata_e;
    double cal_current_cpu(double v, int mech_index);
};

REGISTER_POSTSYN("ExpSyn",ExpSyn,1);
