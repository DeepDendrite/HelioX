#pragma once

#include "articell.h"

class VecEvent : public ArtiCell
{
    public:
        VecEvent(MechInitParams &param);
        ~VecEvent();
        vector<VecData<double>*> vec_ptr;
        void reg_node_indices(MechInitParams &param);
        void initialize_cpu(SimMechInitialParam &param);
        void initialize_gpu(SimMechInitialParam &param);
        void current_cpu(SimMechCurrentParam &param); 
        void current_gpu(SimMechCurrentParam &param); 
        void state_cpu(SimMechStateParam &param);
        void state_gpu(SimMechStateParam &param);
        bool net_receive_cpu(double t); //NET_RECEIVE part in .mod file
        bool net_receive_gpu(double t); //NET_RECEIVE part in .mod file
        void bbcore_read(int icnt, int dcnt, int* iArray, double* dArray, Mode mode);
        void read_data_from_coredat(MechInitParams &param);

        void sync_gpu();

        // 单例模式相关方法
        static VecEvent* getInstance();
        static void setInstance(VecEvent* instance);
        static void destroyInstance();

        static void play(Mode mode, int mech_idx, vector<double> data_arr);
        static void update_sequence(Mode mode, int mech_idx, const double* data_ptr, int len);

    private:
        void element_cpu(int inode);
        VecData<double>* vecdata_etime;
        VecData<int>* vecdata_index;
        
        // 单例实例指针
        static VecEvent* instance;
};

REGISTER_MECHANISM("VecStim",VecEvent);
static bool VecEvent_play_registered = []() {
    FunctionRegistry::getInstance().register_func("VecStim", "play", std::function<void(Mode, int, vector<double>)>(&VecEvent::play));
    return true;
}();
