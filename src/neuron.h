#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <highfive/highfive.hpp>
#include "mechanism.h"
#include "capac.h"
#include "vecdata.h"
#include "articell.h"
#include "presyn.h"
#include "magic_enum/magic_enum.hpp"
#include "vecplay.h"
#include "vecevent.h"

using namespace std;

class HelioXroupData:public VarMapAble
{
    public:
        HelioXroupData(Mode mode, double dt);
        ~HelioXroupData();
        Mode mode;
        int len;//node数量
        int ncell;
        double cj;
        //常数，从coreNeuron中直接读来，并且不会发生变化
        VecData<double>* vecdata_a;//上三角阵元素
        VecData<double>* vecdata_b;//下三角阵元素
        VecData<double>* vecdata_area;//似乎是表面积，在exp2sync和iclamp这两个mech里面会使用到
        VecData<int>* vecdata_parent_index;//用于记录父节点

        //变量，每次计算前清空重算
        VecData<double>* vecdata_d;//对角线元素
        VecData<double>* vecdata_rhs;//右手侧，在矩阵构建完的时候，是电流，解完之后，就是电压了
        // fast_imem support (NEURON i_membrane_ / FastIMemSavRHS):
        // - vecdata_sav_rhs / vecdata_sav_d store the membrane-only RHS/diagonal contribution
        //   (without axial terms), in the same units used by the solver (mA/cm2 and Jacobian units).
        // - vecdata_i_membrane_ stores the final i_membrane_ value in nA (after scaling by area).
        bool need_fast_imem = false;
        VecData<double>* vecdata_sav_rhs = nullptr;
        VecData<double>* vecdata_sav_d = nullptr;
        VecData<double>* vecdata_i_membrane_ = nullptr;

        //会被初始化，然后迭代更新
        VecData<double>* vecdata_v;//电压值，会被finitialize初始化成v_init，默认是-65.0

        PreSyn* presyn;//这部分是单个结构体，但是里面存放了很多信息，不是单个presyn，而postsyn在下面，是单独的vec
        Capac* mech_cap;//好像只有一个，指向膜电容这个mech，然后在lhs和update的时候会调用这个的函数，所以单独抽出来了

        VecEvent* mech_vecevent = nullptr;// vecevnet的指针，只可能有一个，用于play方法

        vector<Mechanism*> mech_write_state_ion_list;//有的Mech在计算State的时候会写离子浓度，这些需要提前计算
        vector<Mechanism*> mechanism_list;//所有机制的列表，不包含eion类型的，因为eion必须提前于所有的mech计算离子浓度
        vector<Mechanism*> mech_current_list;//保持CoreNEURON mechnism顺序，用于current计算
        vector<ArtiCell*> vec_articell; // 人工细胞的列表，articell同时继承了postsyn，所以也会加到下面postsyn列表里
        vector<PostSyn_trait*> vec_postsyn; // 如果是postsyn，那就会加到这列表里，包括上面的articell

        vector<Mechanism*> vec_eion;//这个是用来存储Eion类型的Mech,Eion没有state，因此不在mechanism_list里面也没事

        int* type2postsyn_ix; // map mech_type to index in vec_postsyn //每种类型的mech在group内只有一个，记录mech类型的位置
        int* type2articell_ix; // map mech_type to index in vec_articell

        SpikeVector* spk_vec; // all spikes at current time step ,presyn会用来存储要发射的spike，而postsyn会取出来
        VecData<SpikeFlag>* vecdata_spk_flags; // 用来标记是否发射, -1: not fired, 0: normal, 1: self-event



        //和CPU的permute有关
        int* permute = nullptr; //permute数组,若有，长度为len,即Node数量
        int nstride;
        VecData<int> *vecdata_cellsize,*vecdata_stride, *vecdata_firstnode, *vecdata_lastnode;

        //与permute有关，但是似乎没用上
        int nwarp, norder;
        int threads_num, nthread_each_cell;
        VecData<int> *vecdata_map_t2c;
        VecData<int> *vecdata_max_order_each_thread, *vecdata_min_order_each_thread;

        //gap 传输相关
        struct GapTransInfo{
            VecData<double*> src;
            VecData<double*> dst;
            bool pending;  // GPU模式下标记是否需要同步
            
            GapTransInfo(Mode mode = CPU) : src(mode), dst(mode), pending(false) {}
            
            // 获取传输数量
            int ntrans() const {
                return src.size();
            }
            
            // 添加一对gap junction连接
            void add_gap(double* src_ptr, double* dst_ptr) {
                src.push_back(src_ptr);
                dst.push_back(dst_ptr);
                pending = true;  // 添加后总是标记为pending，在sync_to_gpu时会根据mode决定是否真的同步
            }
            
            // GPU模式下同步数据
            void sync_to_gpu() {
                if (pending) {
                    src.update_gpu_data_from_cpu();
                    dst.update_gpu_data_from_cpu();
                    pending = false;
                }
            }
            
            // 清空但保留容量
            void clear() {
                src.clear();
                dst.clear();
                pending = false;
            }
        };
        //TODO：内存释放
        bool have_gap=false;
        GapTransInfo cpu_gap_trans_info;
        GapTransInfo gpu_gap_trans_info;

        //vecplay相关
        
        VecPlayContinuous vec_play_continuous;


        //提供变量名称和下标，到对应的变量的映射
        virtual double* getVarPtr(const VarDescriptor& descriptor, Mode mode) override;
};

