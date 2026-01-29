#pragma once
#include "coredat_structs.h"
#include "device_dynamic_table.h"
#include "vecdata.h"
#include <utility>
#include <functional>
#include <string>
#include <vector>

//TODO:有性能优化空间，把内存变成连续排布的
//TODO:内存释放
struct VecPlayContinuousItem {
    Mode mode;
    double *data_ptr_cpu;
    double *data_ptr_gpu;
    int len;
    VecData<double> yvec; //修改data的值
    VecData<double> tvec; //修改data的时间变量
    //TODO:非连续版的支持
    int last_index;
    // int discon_index;
    int next_idx;
    bool done;

    double next_time;
    VecPlayContinuousItem(Mode mode, const coreneuron::VecPlayContinuous_Core &vecplay_core, double* data_ptr_cpu, double* data_ptr_gpu);
    

    void play(double t);
    double interpolate(double t);
    void search(double t);
    double interp(double th, double x0, double x1);
    void continuous(double t);
    void reset();  // 重置VecPlay状态
};    

struct VecPlayContinuousKey {
    std::string mech_name;
    std::string var_name;
    int instance_id;
    
    bool operator==(const VecPlayContinuousKey& other) const {
        return mech_name == other.mech_name && var_name == other.var_name && instance_id == other.instance_id;
    }
};

// 为 VecPlayContinuousKey 提供哈希函数特化
namespace std {
    template<>
    struct hash<VecPlayContinuousKey> {
        size_t operator()(const VecPlayContinuousKey& key) const {
            return std::hash<std::string>()(key.mech_name) ^ 
                   (std::hash<std::string>()(key.var_name) << 1) ^
                   (std::hash<int>()(key.instance_id) << 2);
        }
    };
}
struct VecPlayContinuous{
    DynamicDeviceTable<VecPlayContinuousItem, VecPlayContinuousKey> vec_play_table;
    
    // 构造函数
    VecPlayContinuous(Mode mode = CPU) : vec_play_table(mode) {}
    void push_back(VecPlayContinuousKey key,VecPlayContinuousItem &&item){
        vec_play_table.add_or_update(key,std::move(item));
    }
    void play_cpu(double t){
        auto cpu_items = vec_play_table.get_cpu_data();
        auto lens = vec_play_table.size();
        for(int i=0;i<lens;i++){
            cpu_items[i].play(t);
        }   
    }

    void continuous_cpu(double t){
        auto cpu_items = vec_play_table.get_cpu_data();
        auto lens = vec_play_table.size();
        for(int i=0;i<lens;i++){
            cpu_items[i].continuous(t);
        }
    }

    void try_update_gpu();
    void play_gpu(double t);
    void continuous_gpu(double t);
    
    // 重置所有VecPlay状态
    void reset_all_cpu();
    void reset_all_gpu();
    
    // 新增的动态管理方法
    VecPlayContinuousKey addVecPlay(const std::string& mech_name, const std::string& var_name, int instance_id,
                                   const std::vector<double>& tvec, const std::vector<double>& yvec);
    void updateVecPlay(const VecPlayContinuousKey& key, const std::vector<double>& new_tvec, const std::vector<double>& new_yvec);
    void removeVecPlay(const VecPlayContinuousKey& key);
    std::vector<VecPlayContinuousKey> getAllKeys() const;
    bool hasVecPlay(const VecPlayContinuousKey& key) const;
    
private:
    // 辅助方法：根据mechanism name和variable name获取变量指针
    std::pair<double*, double*> getVarPtrByName(const std::string& mech_name, const std::string& var_name, int instance_id);
};

