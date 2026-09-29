#ifndef postsyn_h
#define postsyn_h

#include <vector>
#include <queue>
#include <cmath>
#include <cstdlib>
#include "mechanism.h"
#include "spike.h"

using namespace std;

class PostSyn_trait
{
    public:
        PostSyn_trait();
        virtual ~PostSyn_trait();
        virtual bool net_receive_cpu(double t) = 0; //NET_RECEIVE part in .mod file
        virtual bool net_receive_gpu(double t) = 0; //NET_RECEIVE part in .mod file
        virtual bool supports_spike_vjp() const { return false; }
        virtual void spike_vjp_cpu(SimPostSynSpikeVJPParam& param) { (void)param; }
        virtual void spike_vjp_gpu(SimPostSynSpikeVJPParam& param) { (void)param; }
        ///下面这两个函数已经在trait中实现了，子类不需要再实现
        void post_spike_receive_cpu(double t);//从优先队列中弹出当前时间段可处理的spike,并放到receive_idx_vec中，并设置flag
        void post_spike_receive_gpu(double t);//当前版本会调用CPU版，只是多了把数据同步到GPU的操作

        virtual void get_spike_from_vec_cpu(SpikeVector *sv, SpikeFlag* spk_flags, double t) = 0;
        virtual void get_spike_from_vec_gpu(SpikeVector *sv, SpikeFlag* spk_flags, double t) = 0;

        int get_receive_count() const { return receive_count; }

        VecData<double> *vecdata_delay, *vecdata_weights;
        VecData<int>* vecdata_delay_steps;
        VecData<uint32_t>* vecdata_spk_vec_idx; // corresponding index in spike_vector, size: nodecount
        SpikeBuffer spike_buffer;

    protected:
        void init_vecdatas(MechInitParams &param);
        int receive_count; // number of synapses that need to call spike_receive()
        VecData<uint32_t>* vecdata_receive_idx_vec; // synapse index (note: not node index) that need to call spike_receive(), i.e. the instance order 
        //这个是PostSyn内部的索引，用于表示这个spike是哪个post synapse的实例收到的
        //每种PostSyn内部有一个，互不干扰
        VecData<SpikeFlag>* vecdata_spike_flag; // flags of received spikes, size: node_count
};
class PostSyn : public Mechanism, public PostSyn_trait
{
public:
    PostSyn(MechInitParams &param);
    virtual void reg_node_indices(MechInitParams &param) override;
    virtual void get_spike_from_vec_cpu(SpikeVector *sv, SpikeFlag* spk_flags, double t) override;
    virtual void get_spike_from_vec_gpu(SpikeVector *sv, SpikeFlag* spk_flags, double t) override;
};

//公用的函数，因为只有postsyn_trait和mech拼起来的时候，才能满足调用的参数，所以外置，等具体的函数调用
void get_spike_from_vec_common(SpikeVector* sv, SpikeFlag* spk_flags, double t, SpikeBuffer& spike_buffer,
    VecData<uint32_t>* vecdata_spk_vec_idx, VecData<double>* vecdata_delay, int nnode);
#endif
