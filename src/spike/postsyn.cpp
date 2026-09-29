#include "postsyn.h"
#include <cstdint>

PostSyn_trait::PostSyn_trait()
{
    receive_count = 0;
    vecdata_receive_idx_vec = nullptr;
    vecdata_weights = nullptr;
    vecdata_delay = nullptr;
    vecdata_delay_steps = nullptr;
    vecdata_spike_flag = nullptr;
    vecdata_spk_vec_idx = nullptr;
}

PostSyn_trait::~PostSyn_trait()
{
    if (vecdata_receive_idx_vec)
    {
        delete vecdata_receive_idx_vec;
        vecdata_receive_idx_vec = nullptr;
    }
    if (vecdata_weights)
    {
        delete vecdata_weights;
        vecdata_weights = nullptr;
    }
    if (vecdata_delay)
    {
        delete vecdata_delay;
        vecdata_delay = nullptr;
    }
    if (vecdata_delay_steps)
    {
        delete vecdata_delay_steps;
        vecdata_delay_steps = nullptr;
    }
    if (vecdata_spike_flag)
    {
        delete vecdata_spike_flag;
        vecdata_spike_flag = nullptr;
    }
    if (vecdata_spk_vec_idx)
    {
        delete vecdata_spk_vec_idx;
        vecdata_spk_vec_idx = nullptr;
    }
}


/*
 * find the synapses that need to call spike_receive function
 * check all spikes in spike_buffer, if one spike reaches the time,
 * the post synapse should call spike_receive function in this timestep
 */
void PostSyn_trait::post_spike_receive_cpu(double t)
{
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    SpikeFlag* spike_flag = this->vecdata_spike_flag->get_cpu_data();
    receive_count = 0;
    Spike spk;
    while (!spike_buffer.empty())
    {
        spk = spike_buffer.top();
        if (spk.deliver_time > t)
        {
            break;
        }
        spike_buffer.pop();
        receive_idx_vec[receive_count] = spk.syn_id;
        spike_flag[receive_count] = spk.flag;
        receive_count++;
    }
}

void PostSyn_trait::post_spike_receive_gpu(double t)
{
    post_spike_receive_cpu(t);
    // Update the GPU data from CPU data
    if(receive_count > 0){
        this->vecdata_receive_idx_vec->update_gpu_data_from_cpu();
        this->vecdata_spike_flag->update_gpu_data_from_cpu();
    }
}

void PostSyn_trait::init_vecdatas(MechInitParams &param)
{
    auto node_count = param.node_count;
    auto mode = param.mode;
    vecdata_weights = new VecData<double>(mode, 0.0, node_count);
    vecdata_delay = new VecData<double>(mode, 0.0, node_count);
    vecdata_delay_steps = new VecData<int>(mode, 0, node_count);
    vecdata_spk_vec_idx = new VecData<uint32_t>(mode, (uint32_t)0, node_count);
    vecdata_spike_flag = new VecData<SpikeFlag>(mode, SpikeFlag::INVALID, node_count);
    vecdata_receive_idx_vec = new VecData<uint32_t>(mode, (uint32_t)0, node_count);
}

void PostSyn::reg_node_indices(MechInitParams &param)
{   
    init_vecdatas(param);
}

PostSyn::PostSyn(MechInitParams &param) : Mechanism(param), PostSyn_trait()
{

}

void get_spike_from_vec_common(SpikeVector* sv, SpikeFlag* spk_flags, double t, SpikeBuffer& spike_buffer,
    VecData<uint32_t>* vecdata_spk_vec_idx, VecData<double>* vecdata_delay, int nnode)
{
    if (nnode <= 0) return;

    uint32_t* spk_vec_idx = vecdata_spk_vec_idx->get_cpu_data();
    double* delay = vecdata_delay->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++)
    {
        uint32_t spk_idx = spk_vec_idx[inode];
        if (spk_flags[spk_idx] == SpikeFlag::NORMAL_EVENT) // has fired
        {
            Spike spk = sv->v[spk_idx];
            spk.syn_id = inode;
            spk.flag = spk_flags[spk_idx];
            spk.deliver_time += delay[inode];
            spike_buffer.push(spk);
        }
    }
    sv->hasNewSpk = false;
}


void PostSyn::get_spike_from_vec_cpu(SpikeVector* spk_vec, SpikeFlag* spk_flags, double t)
{
    get_spike_from_vec_common(spk_vec, spk_flags, t, spike_buffer,
        vecdata_spk_vec_idx, vecdata_delay, nnode);
    
}
void PostSyn::get_spike_from_vec_gpu(SpikeVector *spk_vec, SpikeFlag* spk_flags, double t)//和CPU版本完全一致
{
    get_spike_from_vec_common(spk_vec, spk_flags, t, spike_buffer,
        vecdata_spk_vec_idx, vecdata_delay, nnode);
    //spk vec是纯CPU的数据，GPU不需要额外的同步
}
