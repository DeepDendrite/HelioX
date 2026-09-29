#pragma once
#include "mech_template.cuh"
#include "postsyn.h"


struct PostSynTempRecvParam{
    double weight;
    double delay; 
    int delay_step;
};//预留，万一后面要添加什么参数


template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_net_receive_kernel(
    int receive_count,
    uint32_t *receive_idx_vec,
    double *weights,
    double *delays,
    int *delay_steps,
    VarAccessor<MechTrait> gpu_vars);
template <typename Derived, MechTraitType MechTrait>
class PostSynTemplate : public MechTemp<Derived, MechTrait>, public PostSyn_trait{
public:
    PostSynTemplate(MechInitParams &params) : MechTemp<Derived, MechTrait>(params), PostSyn_trait() {
        //只是调用父类的构造函数
    }
    virtual bool net_receive_cpu(double t) override{
        VarAccessor<MechTrait> cpu_vars = this->getCpuVarAccessor();
        PostSynTempRecvParam recv_param; // 预留参数，后面可能会用到
        uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();

        double *weights = this->vecdata_weights->get_cpu_data();
        double *delays = this->vecdata_delay->get_cpu_data();
        int *delay_steps = this->vecdata_delay_steps->get_cpu_data();

        for(int i = 0; i < this->receive_count; i++){
            auto mech_idx = receive_idx_vec[i];
            cpu_vars.idx = mech_idx; // set the mech index
            recv_param.weight = weights[mech_idx];
            recv_param.delay = delays[mech_idx];
            recv_param.delay_step = delay_steps[mech_idx];
            Derived::net_receive_single_node(recv_param,cpu_vars);
        }
        return 0;
    }
    virtual bool net_receive_gpu(double t) override{
        if(this->receive_count > 0) {
            uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_gpu_data();
            double *weights = this->vecdata_weights->get_gpu_data();
            double *delays = this->vecdata_delay->get_gpu_data();
            int *delay_steps = this->vecdata_delay_steps->get_gpu_data();
            
            int block_num = (this->receive_count + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(this->cuda_stream);
            
            VarAccessor<MechTrait> gpu_vars = this->getGpuVarAccessor();
            
            cuda_net_receive_kernel<Derived, MechTrait><<<block_num, nthread_per_block, 0, stream>>>(
                this->receive_count, receive_idx_vec, weights, delays, delay_steps, gpu_vars);
        }
        return 0;
    }


    virtual void reg_node_indices(MechInitParams &param) override{
        MechTemp<Derived, MechTrait>::reg_node_indices(param);
        PostSyn_trait::init_vecdatas(param);
    } 




    /////////////
    virtual void get_spike_from_vec_cpu(SpikeVector *spk_vec, SpikeFlag* spk_flags, double t) override{
        get_spike_from_vec_common(spk_vec, spk_flags, t, spike_buffer,
        vecdata_spk_vec_idx, vecdata_delay, this->nnode);
    }
    virtual void get_spike_from_vec_gpu(SpikeVector *spk_vec, SpikeFlag* spk_flags, double t) override{
        get_spike_from_vec_common(spk_vec, spk_flags, t, spike_buffer,
        vecdata_spk_vec_idx, vecdata_delay, this->nnode);
    }
    

};

// GPU内核函数实现
template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_net_receive_kernel(
    int receive_count,
    uint32_t *receive_idx_vec,
    double *weights,
    double *delays,
    int *delay_steps,
    VarAccessor<MechTrait> gpu_vars)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < receive_count) {
        auto mech_idx = receive_idx_vec[i];
        gpu_vars.idx = mech_idx;
        
        PostSynTempRecvParam recv_param;
        recv_param.weight = weights[mech_idx];
        recv_param.delay = delays[mech_idx];
        recv_param.delay_step = delay_steps[mech_idx];
        
        Derived::net_receive_single_node(recv_param, gpu_vars);
    }
}
