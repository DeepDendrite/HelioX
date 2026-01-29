#include "presyn.h"
#include "presyn.cuh"
#include <cstdint>
#include <cuda_runtime.h>

using namespace std;

PreSyn::PreSyn(uint32_t n, SpikeVector* vec)
{
    // vecdata_spk_vec_cnt = nullptr;
    // vecdata_post_syn_type = nullptr;
    // vecdata_post_synid = nullptr;

    npre = n;
    spk_vec_bkp = vec;
    vecdata_pre_node_indices = nullptr;
    vecdata_threshold = nullptr;
    vecdata_pre_flags = nullptr;
    vecdata_spk_vec_offset = nullptr;//从coredat的spk_vec_offset拷贝过来的
    vecdata_gids = nullptr;
    // Pinned host counters for async D2H copy (do NOT use mapped memory here; atomicAdd into mapped host is slow).
    cpu_mem_allocate((void**)&cudaMappedMem, 2 * sizeof(int));
    spk_num_real = cudaMappedMem;
    spk_num_tot = cudaMappedMem + 1;
    *spk_num_real = 0;
    *spk_num_tot = 0;

    // Device-side counters.
    gpu_mem_allocate((void**)&d_spk_num_real, sizeof(int));
    gpu_mem_allocate((void**)&d_spk_num_tot, sizeof(int));

    // Non-blocking stream + event for spike detection; this avoids implicit sync with other streams.
    cudaStream_t* s = new cudaStream_t;
    cudaError_t err = cudaStreamCreateWithFlags(s, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create spike_stream: %s\n", cudaGetErrorString(err));
        delete s;
        s = nullptr;
    }
    spike_stream = reinterpret_cast<void*>(s);

    cudaEvent_t* ev = new cudaEvent_t;
    err = cudaEventCreateWithFlags(ev, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create spike_event: %s\n", cudaGetErrorString(err));
        delete ev;
        ev = nullptr;
    }
    spike_event = reinterpret_cast<void*>(ev);
}


PreSyn::PreSyn(Mode mode, uint32_t n, SpikeVector* vec,
               const vector<int> &pre_node_indices, 
               const vector<double> &threshold, 
               const vector<uint32_t> &spk_vec_offset,
               const vector<int> &pre_gids):PreSyn(n,vec)
{


    vecdata_pre_node_indices = new VecData<int>(mode, pre_node_indices);
    vecdata_threshold = new VecData<double>(mode, threshold);
    vecdata_pre_flags = new VecData<bool>(mode, false, n);
    vecdata_spk_vec_offset = new VecData<uint32_t>(mode, spk_vec_offset);
    vecdata_gids = new VecData<int>(mode, pre_gids);
    gid_map = new unordered_map<int, int>();
    for (int i = 0; i < pre_gids.size(); i++) {
        gid_map->insert({pre_gids[i], i});
        // printf("gid_map: %d %d\n", pre_gids[i], i);
    }
    vecdata_spk_output = new VecData<vector<double>>(mode, vector<double>(), vecdata_gids->size());
    vecdata_spk_idx_vec = new VecData<int>(mode, pre_gids);
}

PreSyn::~PreSyn()
{
    if (vecdata_pre_node_indices)
    {
        delete vecdata_pre_node_indices;
        vecdata_pre_node_indices = nullptr;
    }
    if (vecdata_threshold)
    {
        delete vecdata_threshold;
        vecdata_threshold = nullptr;
    }

    if (vecdata_spk_vec_offset)
    {
        delete vecdata_spk_vec_offset;
        vecdata_spk_vec_offset = nullptr;
    }

    if (vecdata_gids)
    {
        delete vecdata_gids;
        vecdata_gids = nullptr;
    }
    try_delete(vecdata_pre_flags);
    try_delete(vecdata_spk_idx_vec);
    
    if(cudaMappedMem){
        cpu_mem_free((void**)&cudaMappedMem);
    }

    if (d_spk_num_real) {
        gpu_mem_free((void**)&d_spk_num_real);
        d_spk_num_real = nullptr;
    }
    if (d_spk_num_tot) {
        gpu_mem_free((void**)&d_spk_num_tot);
        d_spk_num_tot = nullptr;
    }

    if (spike_event) {
        cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(spike_event);
        cudaEventDestroy(*ev);
        delete ev;
        spike_event = nullptr;
    }
    if (spike_stream) {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(spike_stream);
        cudaStreamDestroy(*s);
        delete s;
        spike_stream = nullptr;
    }
}

/*
 * check the voltage for all pre-synapse node, 
 * if it fires, push the spike to spike_buffer
 */
void PreSyn::threshold_detect_cpu(double* vec_v, SpikeFlag* spk_flags, double t, vector<pair<double, int>> &rec_spk)
{
    int* pre_node_indices = this->vecdata_pre_node_indices->get_cpu_data();
    uint32_t* spk_vec_offset = this->vecdata_spk_vec_offset->get_cpu_data();
    double* threshold = this->vecdata_threshold->get_cpu_data();
    bool* pre_flags = this->vecdata_pre_flags->get_cpu_data();
    int* gids = this->vecdata_gids->get_cpu_data();
    vector<double>* spk_output = this->vecdata_spk_output->get_cpu_data();
    int spk_count = 0;
    for (int ipre = 0; ipre < npre; ipre++)//遍历所有的presyn,pre会把spk给塞到spk_vec里面
    {
        int pre_node_id = pre_node_indices[ipre];
        double pre_v = vec_v[pre_node_id];
        if (pre_v > threshold[ipre] && !pre_flags[ipre])
        {
            // printf_debug("spk send at preNode[%d] t[%lf]\n",pre_node_id,t);
            //???spk_vec_offset[i]==i，这是在干啥呢
            Spike spk(gids[ipre], t, SpikeFlag::NORMAL_EVENT);    //pre_node_id,syn_type,syn_id都没设置
            spk_vec_bkp->set_spike(spk, spk_vec_offset[ipre]);//把新生成的spike给塞到vec里面
            pre_flags[ipre] = true;
            spk_flags[spk_vec_offset[ipre]] = SpikeFlag::NORMAL_EVENT;//0代表已发射
            rec_spk.emplace_back(t, gids[ipre]);
            spk_output[ipre].push_back(t);
            spk_count++;
        }
        else if (pre_v <= threshold[ipre])
        {
            pre_flags[ipre] = false;
        }
    }

    if (spike_profile_enabled_) {
        spike_profile_stats_.steps += 1;
        if (spk_count > 0) {
            spike_profile_stats_.steps_with_presyn_spike += 1;
            spike_profile_stats_.presyn_spike_total += static_cast<uint64_t>(spk_count);
            if (spk_count > spike_profile_stats_.presyn_spike_max) {
                spike_profile_stats_.presyn_spike_max = spk_count;
            }
        }
        const int bucket = spk_count >= 63 ? 63 : (spk_count < 0 ? 0 : spk_count);
        spike_profile_stats_.presyn_spike_hist[static_cast<size_t>(bucket)] += 1;
    }

}

int PreSyn::threshold_detect_gpu(double* vec_v, VecData<SpikeFlag>* vecdata_spk_flags, double t, vector<pair<double, int>> &rec_spk)
{
    int* pre_node_indices = this->vecdata_pre_node_indices->get_gpu_data();
    double* threshold = this->vecdata_threshold->get_gpu_data();
    bool* pre_flags = this->vecdata_pre_flags->get_gpu_data();
    SpikeFlag* spk_flags = vecdata_spk_flags->get_gpu_data();
    uint32_t* spk_vec_offset = this->vecdata_spk_vec_offset->get_gpu_data();
    int* spk_idx_vec = this->vecdata_spk_idx_vec->get_gpu_data();
    SpikeFlag* cpu_spk_flags = vecdata_spk_flags->get_cpu_data();
    int* cpu_spk_idx_vec = this->vecdata_spk_idx_vec->get_cpu_data();
    vector<double>* spk_output = this->vecdata_spk_output->get_cpu_data();

    int tot_len = vecdata_spk_flags->size();// 包含了articell
    cuda_spike_send_async(vec_v,
                          spk_flags,
                          pre_node_indices,
                          spk_vec_offset,
                          threshold,
                          pre_flags,
                          t,
                          this->npre,
                          spk_idx_vec,
                          tot_len,
                          d_spk_num_real,
                          d_spk_num_tot,
                          this->spk_num_real,
                          this->spk_num_tot,
                          spike_stream,
                          spike_event);

    // Wait only for the spike stream/event (non-blocking stream does not synchronize with other streams).
    if (spike_event) {
        cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(spike_event);
        cudaEventSynchronize(*ev);
    } else if (spike_stream) {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(spike_stream);
        cudaStreamSynchronize(*s);
    }

    const int spk_count = *(this->spk_num_real);
    if (spike_profile_enabled_) {
        spike_profile_stats_.steps += 1;
        if (spk_count > 0) {
            spike_profile_stats_.steps_with_presyn_spike += 1;
            spike_profile_stats_.presyn_spike_total += static_cast<uint64_t>(spk_count);
            if (spk_count > spike_profile_stats_.presyn_spike_max) {
                spike_profile_stats_.presyn_spike_max = spk_count;
            }
        }
        const int bucket = spk_count >= 63 ? 63 : (spk_count < 0 ? 0 : spk_count);
        spike_profile_stats_.presyn_spike_hist[static_cast<size_t>(bucket)] += 1;
    }
    if (spk_count == 0) {
        return *(this->spk_num_tot);
    }

    // Copy back only what we need (avoid VecData::update_cpu_data_from_gpu which uses sync cudaMemcpy).
    cudaStream_t stream = spike_stream ? *reinterpret_cast<cudaStream_t*>(spike_stream) : static_cast<cudaStream_t>(0);
    cudaMemcpyAsync(cpu_spk_flags, spk_flags, sizeof(SpikeFlag) * tot_len, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cpu_spk_idx_vec, spk_idx_vec, sizeof(int) * spk_count, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    uint32_t* cpu_spk_vec_offset = vecdata_spk_vec_offset->get_cpu_data();
    int* gids = this->vecdata_gids->get_cpu_data();

    for (int i = 0; i < spk_count; i++) {
        int ipre = cpu_spk_idx_vec[i];
        Spike spk(gids[ipre], t, SpikeFlag::NORMAL_EVENT);
        spk_vec_bkp->set_spike(spk, cpu_spk_vec_offset[ipre]);

        rec_spk.emplace_back(t, gids[ipre]); // write spk.out
        spk_output[ipre].push_back(t);
    }
    return *(this->spk_num_tot);
}
