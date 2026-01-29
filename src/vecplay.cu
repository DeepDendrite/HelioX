#include "vecplay.h"
#include "mechanism.h"

VecPlayContinuousItem::VecPlayContinuousItem(Mode _mode,const coreneuron::VecPlayContinuous_Core &vecplay_core, double* _data_ptr_cpu, double* _data_ptr_gpu)
:yvec(_mode,vecplay_core.yvec), tvec(_mode,vecplay_core.tvec) {
    this->mode = _mode;
    this->data_ptr_cpu = _data_ptr_cpu;
    this->data_ptr_gpu = _data_ptr_gpu;
    this->len = vecplay_core.yvec.size();

    
    this->last_index = 0;
    // this->discon_index = 0;
    this->next_idx = 0;
    this->done = false;
    this->next_time = vecplay_core.tvec[0];
}

void __host__ __device__ VecPlayContinuousItem::play(double t){
    auto *tvec_dev_ptr = tvec.get_dev_data();
    while(1){
        if(this->done || this->next_time > t){
            return;
        }
        last_index = next_idx;
        //double play_time = next_time;
        if(next_idx < len - 1){
            next_idx++;
            next_time = tvec_dev_ptr[next_idx];
        }else{
            done = true;
        }
    
        // continuous(play_time);//会被覆盖，所以只需要维护last_index的变化
        // if (t >= tvec_cpu[next_idx]) {//似乎不维护也行,因为在continuous中，做了基本完全一样的逻辑，先注释掉
        //     last_index = next_idx;

        // } else if (t <= tvec_cpu[0]) {
        //     last_index = 0;
        // } else {
        //     search(t);
        // }
    }
}
void __host__ __device__ VecPlayContinuousItem::continuous(double t){
    #ifdef __CUDA_ARCH__
    // 在GPU上运行时，直接使用data_ptr_gpu
    double* data = data_ptr_gpu;
    #else
    // 在CPU上运行时，使用data_ptr_cpu
    double* data = data_ptr_cpu;
    #endif
    *data = interpolate(t);
}

double __host__ __device__ VecPlayContinuousItem::interpolate(double tt) {
    
    auto *tvec_dev_ptr = tvec.get_dev_data();
    auto *yvec_dev_ptr = yvec.get_dev_data();
    if (tt >= tvec_dev_ptr[next_idx]) {
        last_index = next_idx;
        if (last_index == 0) {
            return yvec_dev_ptr[last_index];
        }
    } else if (tt <= tvec_dev_ptr[0]) {
        last_index = 0;
        return yvec_dev_ptr[0];
    } else {
        search(tt);
    }
    double x0 = yvec_dev_ptr[last_index - 1];
    double x1 = yvec_dev_ptr[last_index];
    double t0 = tvec_dev_ptr[last_index - 1];
    double t1 = tvec_dev_ptr[last_index];
    // printf("IvocVectRecorder::continuous tt=%g t0=%g t1=%g theta=%g x0=%g x1=%g\n", tt, t0, t1,
    // (tt - t0)/(t1 - t0), x0, x1);
    if (t0 == t1) {
        return (x0 + x1) / 2.;
    }
    return interp((tt - t0) / (t1 - t0), x0, x1);
}
void __host__ __device__ VecPlayContinuousItem::search(double t) {
    //	assert (tt > t_->elem(0) && tt < t_->elem(t_->size() - 1))
    auto *tvec_dev_ptr = tvec.get_dev_data();
    while (t < tvec_dev_ptr[last_index]) {
        --last_index;
    }
    while (t >= tvec_dev_ptr[last_index]) {
        ++last_index;
    }
}
double __host__ __device__ VecPlayContinuousItem::interp(double th, double x0, double x1) {
    return x0 + (x1 - x0) * th;
}

void __host__ __device__ VecPlayContinuousItem::reset() {
    // 重置VecPlay的内部状态到初始值
    last_index = 0;
    next_idx = 0;
    done = false;
    
    // 重置next_time到第一个时间点（如果有数据）
    if (len > 0) {
        auto *tvec_dev_ptr = tvec.get_dev_data();
        next_time = tvec_dev_ptr[0];
    } else {
        next_time = 0.0;
        done = true; // 如果没有数据，标记为完成
    }
}

void VecPlayContinuous::try_update_gpu(){
    vec_play_table.update_gpu_from_cpu();
}

__global__ void VecPlayContinuous_play_kernel(VecPlayContinuousItem* vec_play_vec_gpu, int n_vec_play_continuous, double t){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_vec_play_continuous){
        vec_play_vec_gpu[i].play(t);
    }
}
__global__ void VecPlayContinuous_continuous_kernel(VecPlayContinuousItem* vec_play_vec_gpu, int n_vec_play_continuous, double t){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_vec_play_continuous){
        vec_play_vec_gpu[i].continuous(t);
    }
}

void VecPlayContinuous::play_gpu(double t){
    auto lens = vec_play_table.size();
    if(lens == 0){
        return;
    }
    int block_num = (lens + nthread_per_block - 1) / nthread_per_block;
    VecPlayContinuous_play_kernel<<<block_num, nthread_per_block>>>(vec_play_table.get_gpu_data(), lens, t);
}

void VecPlayContinuous::continuous_gpu(double t){
    auto lens = vec_play_table.size();
    if(lens == 0){
        return;
    }
    int block_num = (lens + nthread_per_block - 1) / nthread_per_block;
    VecPlayContinuous_continuous_kernel<<<block_num, nthread_per_block>>>(vec_play_table.get_gpu_data(), lens, t);
}

// 辅助方法：根据mechanism name和variable name获取变量指针
std::pair<double*, double*> VecPlayContinuous::getVarPtrByName(const std::string& mech_name, const std::string& var_name, int instance_id) {
    auto& mechFactory = MechanismFactory::getInstance();
    auto varMapPtr = mechFactory.getVarMap(mech_name);

    if (varMapPtr == nullptr) {
        printf("VecPlay: mechanism '%s' not found\n", mech_name.c_str());
        return {nullptr, nullptr};
    }

    // 创建VarDescriptor
    VarDescriptor descriptor(mech_name, var_name, instance_id);

    double* var_ptr_cpu = varMapPtr->getVarPtr(descriptor, Mode::CPU);
    double* var_ptr_gpu = nullptr;

    if (var_ptr_cpu == nullptr) {
        printf("VecPlay: variable '%s' in mechanism '%s' at instance %d not found\n",
               var_name.c_str(), mech_name.c_str(), instance_id);
        return {nullptr, nullptr};
    }

    // 获取GPU指针（如果需要）
    if (vec_play_table.mode() == Mode::GPU) {
        var_ptr_gpu = varMapPtr->getVarPtr(descriptor, Mode::GPU);
    }

    return {var_ptr_cpu, var_ptr_gpu};
}

// 新增VecPlay
VecPlayContinuousKey VecPlayContinuous::addVecPlay(const std::string& mech_name, const std::string& var_name, int instance_id,
                                                   const std::vector<double>& tvec, const std::vector<double>& yvec) {
    VecPlayContinuousKey key{mech_name, var_name, instance_id};
    
    // 检查是否已存在
    if (hasVecPlay(key)) {
        printf("VecPlay: key {%s, %s, %d} already exists, use updateVecPlay instead\n", 
               mech_name.c_str(), var_name.c_str(), instance_id);
        return key;
    }
    
    // 获取变量指针
    auto [var_ptr_cpu, var_ptr_gpu] = getVarPtrByName(mech_name, var_name, instance_id);
    if (var_ptr_cpu == nullptr) {
        printf("VecPlay: failed to get variable pointer for {%s, %s, %d}\n", 
               mech_name.c_str(), var_name.c_str(), instance_id);
        return key;
    }
    
    // 创建伪造的CoreData结构用于构造VecPlayContinuousItem
    coreneuron::VecPlayContinuous_Core vecplay_core;
    vecplay_core.yvec = yvec;
    vecplay_core.tvec = tvec;
    
    // 创建VecPlayContinuousItem
    VecPlayContinuousItem item(vec_play_table.mode(), vecplay_core, var_ptr_cpu, var_ptr_gpu);
    
    // 添加到表中
    vec_play_table.add_or_update(key, std::move(item));
    
    printf("VecPlay: added vecplay for {%s, %s, %d} with %zu time points\n", 
           mech_name.c_str(), var_name.c_str(), instance_id, tvec.size());
    
    return key;
}

// 更新VecPlay
void VecPlayContinuous::updateVecPlay(const VecPlayContinuousKey& key, const std::vector<double>& new_tvec, const std::vector<double>& new_yvec) {
    if (!hasVecPlay(key)) {
        printf("VecPlay: key {%s, %s, %d} not found, use addVecPlay instead\n", 
               key.mech_name.c_str(), key.var_name.c_str(), key.instance_id);
        return;
    }
    
    // 获取现有的item
    auto* item = vec_play_table.get_item_ptr(key);
    if (item == nullptr) {
        printf("VecPlay: failed to get item for key {%s, %s, %d}\n", 
               key.mech_name.c_str(), key.var_name.c_str(), key.instance_id);
        return;
    }
    
    // 更新tvec和yvec
    item->tvec.resize(new_tvec.size());
    item->yvec.resize(new_yvec.size());
    
    // 更新数据
    std::copy(new_tvec.begin(), new_tvec.end(), item->tvec.get_cpu_data());
    std::copy(new_yvec.begin(), new_yvec.end(), item->yvec.get_cpu_data());
    
    // 重置状态
    item->len = new_tvec.size();
    item->last_index = 0;
    item->next_idx = 0;
    item->done = false;
    item->next_time = new_tvec.empty() ? 0.0 : new_tvec[0];
    
    // 同步到GPU
    if (vec_play_table.mode() == Mode::GPU) {
        item->tvec.update_gpu_data_from_cpu();
        item->yvec.update_gpu_data_from_cpu();
    }
    
    // printf("VecPlay: updated vecplay for {%s, %s, %d} with %zu time points\n", 
        //    key.mech_name.c_str(), key.var_name.c_str(), key.instance_id, new_tvec.size());
}

// 删除VecPlay
void VecPlayContinuous::removeVecPlay(const VecPlayContinuousKey& key) {
    if (!hasVecPlay(key)) {
        printf("VecPlay: key {%s, %s, %d} not found\n", 
               key.mech_name.c_str(), key.var_name.c_str(), key.instance_id);
        return;
    }
    
    vec_play_table.remove(key);
    
    printf("VecPlay: removed vecplay for {%s, %s, %d}\n", 
           key.mech_name.c_str(), key.var_name.c_str(), key.instance_id);
}

// 获取所有的key
std::vector<VecPlayContinuousKey> VecPlayContinuous::getAllKeys() const {
    return vec_play_table.get_all_keys();
}

// 检查是否存在指定key
bool VecPlayContinuous::hasVecPlay(const VecPlayContinuousKey& key) const {
    return vec_play_table.has_key(key);
}

// 重置所有VecPlay状态 - CPU版本
void VecPlayContinuous::reset_all_cpu() {
    auto cpu_items = vec_play_table.get_cpu_data();
    auto lens = vec_play_table.size();
    for(int i = 0; i < lens; i++) {
        cpu_items[i].reset();
    }
    vec_play_table.set_dirty(); // 标记为CPU数据已修改
}

// GPU kernel for resetting VecPlay states
__global__ void VecPlayContinuous_reset_kernel(VecPlayContinuousItem* vec_play_vec_gpu, int n_vec_play_continuous) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_vec_play_continuous) {
        vec_play_vec_gpu[i].reset();
    }
}

// 重置所有VecPlay状态 - GPU版本
void VecPlayContinuous::reset_all_gpu() {
    reset_all_cpu(); // 先在CPU上重置，确保数据一致性
    try_update_gpu(); // 确保GPU数据是最新的
}
