#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <map>
#include <cstdio>
#include "utils.h"
namespace coreneuron{

class NetCon;
    class CoreMechData
{
    public:
        CoreMechData();
        ~CoreMechData();
        int nmech_type;
        int *nrn_prop_param_size, *nrn_prop_dparam_size, *nrn_mech_data_layout;
        std::vector<int> nrn_has_net_event;
        std::vector<std::vector<int>> nrn_array_dims;//ljf 1.8兼容
        int* pnt_map; // map type id to point_process id, 0: not point_process
        std::vector<std::string> name_vec;//debug only
        short *pnt_receive_size, *nrn_is_artificial, *nrn_artcell_qindex;
        bool* nrn_bbcore_read;
        std::vector<bool> is_ion;
};

class CoreMech//对应TML
{
    public:
        CoreMech();
        ~CoreMech();
        int type, nodecount; //index: mech type
        int* nodeindices;
        int* pdata;
        double* _data;
        CoreMech* next;
        int icnt, dcnt; // size of dArray and iArray
        int* iArray;    //IArray和DArray好像是只给ArtiCell用的
        double* dArray;
        int* permute;

        //ljf:增加1.8的支持

        std::vector<int> pointer2type;//这个可能出现在文件中，直接读回来。或者，可能出现在relocation中（还没做）
        std::vector<uint32_t> nmodlrandom{};//同样，可能出现在文件中，有没有别的地方还没看
};

struct PointProcess
{
    int i_instance, cid; //cid: CoreData id that this PointProcess belongs to
    short type;
    void print(){
        printf_debug("PP: I_Instance=%d cid=%d type=%X\n",i_instance,cid,type);
    }
};

class NetCon//对应coreneuron/network/netcon.hpp的class NetCon: public DiscreteEvent 
{
    public:
        NetCon()
        {
            src_gid = -1000000;
            threshold = -1000000;
            delay = -1000000;
            pp = nullptr;
            active = false;
        }

        bool active;
        double threshold, delay;//Threshold不知道哪来的
        int weight_index;
        int src_gid;
        PointProcess* pp; //target point process

        void print(){
            printf_debug("NC: act[%d] Threshold[%lf] delay[%lf] weight_idx[%d] src_gid[%d]\n",active,threshold,delay,weight_index,src_gid);
            if(pp){
                pp->print();
            }else{
                printf_debug("NC has no PP!\n");
            }
        }
};

class CorePreSyn//对应netcon.hpp:102  class PreSyn
{
    public:
        CorePreSyn()
        {
            nc_index = -1;
            nc_cnt = 0;
            output_index = -1;
            gid = -1;
            thvar_index = -1;
            pntsrc = nullptr;
        }
        int nc_index; // index into global netcon_arr，即，指向该 PreSyn 的第一个 NetCon
        int nc_cnt; // how many netcon starting at nc_index
        int output_index;
        int gid;
        double threshold;
        int thvar_index; // >=0 points into _actual_v
        PointProcess* pntsrc; // source point process
};

class InputPreSyn
{
    public:
        int nc_index;
        int nc_cnt;
};
using namespace std;

enum class gap_idx_type { voltage = -1, i_membrane_ = -2 };
struct GapTransferInfo {
    int ntrans;
    std::unordered_map<int,int> src_sid_map;
    std::vector<int> src_sid;
    std::vector<int> src_type;
    std::vector<int> src_index;
    std::vector<int> tar_sid;
    std::vector<int> tar_type;
    std::vector<int> tar_index;
};
struct VecPlayContinuous_Core {
    constexpr static int PlayRecordType = 0;
    constexpr static int VecPlayContinuousType = 4;
    constexpr static int PlayRecordEventType = 21;
    
    int vtype=-1;
    int mtype=-1;//mech type
    int ix=-1;//变量在coredat所有变量中的idx，需要经过映射，找到对应变量的对应下标
    std::vector<double> yvec;
    std::vector<double> tvec;

    int last_index=0;
    int discon_index=0;
    int ubound_index=0;
    void print(){
        printf_debug("VecPlayContinuous: vtype[%d] mtype[%d] ix[%d] yvec.size[%d] tvec.size[%d] last_index[%d] discon_index[%d] ubound_index[%d]\n",vtype,mtype,ix,yvec.size(),tvec.size(),last_index,discon_index,ubound_index);
        for(int i=0;i<yvec.size();i++){
            printf_debug("yvec[%d]=%lf tvec[%d]=%lf\n",i,yvec[i],i,tvec[i]);
        }
    }
};
class CoreData
{
    
    
    public:
        CoreData(int cid);        
        ~CoreData();
        
        int id;//Group的编号，或者可以认为是thread id
        int have_gaps;//是否有gap junction
        int n_presyn, n_netcon;//presyn和netcon的数量，直接从文件中读回来
        int n_intput_presyn;//这个是在determine_inputpresyn中计算出来的
        int n_real_output, end, nidata, nvdata, n_weight, nmech, n_pntproc;//end == nnode
        int n_real_cell;
        int* gidgroups;
        int* imult;
        int *netcon_srcgid, *idata, *_v_parent_index, *output_gid;
        double *_actual_rhs, *_actual_d, *_actual_a, *_actual_b, *_actual_v, *_actual_area, *_actual_diam;
        double* weights;
        unique_ptr<void*[]> vdata;
        unique_ptr<unique_ptr<int[]>[]> pnt2presyn_ix;
        CoreMechData* mech_data;
        CoreMech* mech_list;
        CoreMech** map_type2mechptr;
        PointProcess* pntprocs;
        NetCon* netcon_arr;
        CorePreSyn* presyn_arr; // output spikes
        int* permute;//一个映射数组，把旧的VecData的index映射到新的index

        GapTransferInfo *gap_transfer;

        //ljf added:
        int patstim_index;
        std::vector<VecPlayContinuous_Core> vec_play_continuous_core;
        std::vector<int> preSynConditionEventFlags;
};

}


