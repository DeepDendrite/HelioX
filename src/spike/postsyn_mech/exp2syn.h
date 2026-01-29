#pragma once

#include "postsyn.h"

class Exp2Syn : public PostSyn
{
public:
    Exp2Syn(MechInitParams &param);
    ~Exp2Syn();
    void reg_node_indices(MechInitParams &param);
    void initialize_cpu(SimMechInitialParam &param);
    void initialize_gpu(SimMechInitialParam &param);
    void current_cpu(SimMechCurrentParam &param); 
    void current_gpu(SimMechCurrentParam &param); 
    void state_cpu(SimMechStateParam &param);
    void state_gpu(SimMechStateParam &param);
    bool net_receive_cpu(double t); //NET_RECEIVE part in .mod file
    bool net_receive_gpu(double t); //NET_RECEIVE part in .mod file
    void sync_gpu();
    void read_data_from_coredat(MechInitParams &param);

private:
    VecData<double>* vecdata_tau1, * vecdata_tau2, * vecdata_e, * vecdata_factor;
    VecData<double>* vecdata_A, * vecdata_B;
    double cal_current_cpu(double v, int mech_index);
};

REGISTER_POSTSYN("Exp2Syn_org",Exp2Syn,1);