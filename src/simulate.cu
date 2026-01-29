#include "simulate.h"
#include "utils.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

__global__ void cuda_finitialize_kernel(double* vec_v, double v_init, int len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
    {
        vec_v[i] = v_init;
        //printf("i:%d vec_v:%f v_init:%f\n", i, vec_v[i], v_init);
    }
}

__global__ void rhs_kernel(double* vec_v, double* vec_rhs, double* vec_d, double* vec_a, 
                           double* vec_b, int* parent_index, int ncell, int len)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    double dv;
    if (i >= ncell && i < len)
    {
        auto p_i = parent_index[i];
        dv = vec_v[p_i] - vec_v[i];
		atomicAdd(vec_rhs + i, -vec_b[i] * dv);
		atomicAdd(vec_rhs + p_i, vec_a[i] * dv);
    }
}

__global__ void lhs_kernel(double* vec_d, double* vec_a, double* vec_b, int* parent_index, int ncell, int len)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ncell && i < len)
    {
		atomicAdd(vec_d + i, -vec_b[i]);
		atomicAdd(vec_d + parent_index[i], -vec_a[i]);
    }
}
__global__ void lhs_and_rhs_kernel(double* vec_v, double* vec_rhs, double* vec_d, double* vec_a, 
                           double* vec_b, int* parent_index, int ncell, int len)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ncell && i < len)
    {
        auto p_i = parent_index[i];
        double dv = vec_v[p_i] - vec_v[i];
        auto b = vec_b[i];
        auto a = vec_a[i];
		atomicAdd(vec_d + i, -b);
		atomicAdd(vec_d + p_i, -a);

		atomicAdd(vec_rhs + i, -b * dv);
		atomicAdd(vec_rhs + p_i, a * dv);
    }
}
__global__ void solve_matrix_kernel(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index, int len, int ncell)
{
    double p;
    //triang
    for (int i = len - 1; i >= ncell; i--)
    {
        p = vec_a[i] / vec_d[i];
        vec_d[parent_index[i]] -= p * vec_b[i];
        vec_rhs[parent_index[i]] -= p * vec_rhs[i];
    }

    //bksub
    for (int i = 0; i < ncell; i++)
    {
        vec_rhs[i] /= vec_d[i];
    }
    for (int i = ncell; i < len; i++)
    {
        vec_rhs[i] -= vec_b[i] * vec_rhs[parent_index[i]];
        vec_rhs[i] /= vec_d[i];
    }
}

__global__ void solve_permute1_kernel(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, 
                                   int* parent_index, int nstride, int* stride, int* firstnode, 
                                   int* lastnode, int* cellsize, int ncell, int len)
{
    unsigned int tid;
    int i, icellsize;
    int istride, ip;
    double p;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ncell)
    {
        icellsize = cellsize[tid];
        i = lastnode[tid];
        for (istride = nstride - 1; istride >= 0; istride--)
        {
            if (istride < icellsize)
            {
                ip = parent_index[i];
                p = vec_a[i] / vec_d[i];
                vec_d[ip] -= p * vec_b[i];
                vec_rhs[ip] -= p * vec_rhs[i];
                i -= stride[istride];
            }
        }

        i = firstnode[tid];
        vec_rhs[tid] /= vec_d[tid];
        for (istride = 0; istride < icellsize; istride++)
        {
            ip = parent_index[i];
            vec_rhs[i] -= vec_b[i] * vec_rhs[ip];
            vec_rhs[i] /= vec_d[i];
            i += stride[istride + 1];
        }
    }
}
__global__ void cop_solve_kernel(
    const double* __restrict__ vec_a, 
    const double* __restrict__ vec_b, 
    double* __restrict__ vec_d, 
    double* __restrict__ vec_rhs, 
    const int* __restrict__ parent_index,
    const int* __restrict__ max_order_each_thread, 
    const int* __restrict__ min_order_each_thread, 
    const int* __restrict__ firstnode, 
    const int* __restrict__ lastnode,
    const int* __restrict__ stride, 
    const int* __restrict__ map_t2c, 
    int norder, 
    int ncell, 
    int nthread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthread) return;
    
    // 使用寄存器缓存常用变量，避免重复内存访问
    const int max_order = max_order_each_thread[tid];
    const int min_order = min_order_each_thread[tid];
    int i = lastnode[tid];
    const int offset = (tid >> 5) * (norder + 1) - 1;
    
    // 三角化阶段 - 展开循环以提高性能
    #pragma unroll 4
    for (int iorder = norder; iorder >= 0; iorder--) {
        if (iorder >= min_order && iorder <= max_order && i > -1) {
            // 预取数据到寄存器
            const int ip = parent_index[i];
            const double a_val = vec_a[i];
            const double d_val = vec_d[i];
            const double b_val = vec_b[i];
            const double rhs_val = vec_rhs[i];
            
            const double p = a_val / d_val;
            
            // 原子操作
            atomicAdd(&vec_d[ip], -p * b_val);
            atomicAdd(&vec_rhs[ip], -p * rhs_val);
            
            i -= stride[offset + iorder];
        }
    }
    
    // 回带阶段
    const int icell = map_t2c[tid];
    if (icell > -1) {
        vec_rhs[icell] /= vec_d[icell];
    }
    
    i = firstnode[tid];
    const int offset2 = offset + 1;
    
    #pragma unroll 4
    for (int iorder = 1; iorder <= norder; iorder++) {
        if (iorder >= min_order && iorder <= max_order && i > -1) {
            const int ip = parent_index[i];
            const double b_val = vec_b[i];
            const double rhs_parent = vec_rhs[ip];
            const double d_val = vec_d[i];
            const double rhs_val = vec_rhs[i];
            
            const double p = rhs_val - b_val * rhs_parent;
            vec_rhs[i] = p / d_val;
            
            i += stride[offset2 + iorder];
        }
    }
}


__global__ void update_kernel(double* vec_v, double* vec_rhs, int len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        vec_v[i] += vec_rhs[i];
    }
}

__global__ void negate_copy_kernel(double* dst, const double* src, int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(len)) {
        dst[i] = -src[i];
    }
}

__global__ void copy_kernel(double* dst, const double* src, int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(len)) {
        dst[i] = src[i];
    }
}

__global__ void fast_imem_kernel(double* out_imem,
                                const double* sav_d,
                                const double* dv,
                                const double* sav_rhs,
                                const double* area,
                                int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(len)) {
        out_imem[i] = (sav_d[i] * dv[i] + sav_rhs[i]) * area[i] * 0.01;
    }
}

__global__ void fast_imem_init_kernel(double* out_imem,
                                     const double* rhs,
                                     const double* sav_rhs,
                                     const double* area,
                                     int len) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(len)) {
        out_imem[i] = (rhs[i] + sav_rhs[i]) * area[i] * 0.01;
    }
}

static cudaStream_t stream[2] = {nullptr, nullptr};
static bool streams_initialized = false;

// Events marking completion of async memset for rhs/d (to avoid relying on legacy default-stream semantics).
static cudaEvent_t rhs_zeroed_event = nullptr;
static cudaEvent_t d_zeroed_event = nullptr;
// Event marking completion of default-stream work at the end of a timestep.
// Used to prevent non-blocking spike streams from racing ahead of default-stream kernels
// (e.g. VecPlay kernels) while still allowing overlap with non-default mech streams.
static cudaEvent_t default_stream_step_done_event = nullptr;

static void ensure_cuda_events() {
    if (rhs_zeroed_event == nullptr) {
        cudaEventCreateWithFlags(&rhs_zeroed_event, cudaEventDisableTiming);
    }
    if (d_zeroed_event == nullptr) {
        cudaEventCreateWithFlags(&d_zeroed_event, cudaEventDisableTiming);
    }
    if (default_stream_step_done_event == nullptr) {
        cudaEventCreateWithFlags(&default_stream_step_done_event, cudaEventDisableTiming);
        // Make it "already done" for the very first step before finitialize records a real one.
        cudaEventRecord(default_stream_step_done_event, 0);
    }
}

static void ensure_cuda_streams() {
    if (!streams_initialized) {
        for (auto& s : stream) {
            cudaStreamCreate(&s);
        }
        streams_initialized = true;
    }
    ensure_cuda_events();
}

void destroy_cuda_streams() {
    if (!streams_initialized) {
        return;
    }
    for (auto& s : stream) {
        if (s != nullptr) {
            cudaStreamDestroy(s);
            s = nullptr;
        }
    }
    if (rhs_zeroed_event != nullptr) {
        cudaEventDestroy(rhs_zeroed_event);
        rhs_zeroed_event = nullptr;
    }
    if (d_zeroed_event != nullptr) {
        cudaEventDestroy(d_zeroed_event);
        d_zeroed_event = nullptr;
    }
    if (default_stream_step_done_event != nullptr) {
        cudaEventDestroy(default_stream_step_done_event);
        default_stream_step_done_event = nullptr;
    }
    streams_initialized = false;
}

void cuda_finitialize(double* vec_v, double v_init, int len)
{
    ensure_cuda_streams();
    int block_num = (len + nthread_per_block - 1) / nthread_per_block;
    cuda_finitialize_kernel<<<block_num, nthread_per_block>>>(vec_v, v_init, len);

    #ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("finitialize error:%s\n", cudaGetErrorString(cuda_status));
	}
	else
	{
		// printf("finitialize success\n");
	}
    #endif  
}

void cuda_setup_matrix_init(HelioXroupData* p_neuron)
{
    ensure_cuda_streams();
    int len = p_neuron->len;
    double* vec_d = p_neuron->vecdata_d->get_gpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_gpu_data();
    cudaMemsetAsync(vec_rhs, 0, sizeof(double) * len,stream[0]);
    cudaMemsetAsync(vec_d, 0, sizeof(double) * len,stream[1]);

    // Record events so other streams can wait on matrix init completion without global sync.
    cudaEventRecord(rhs_zeroed_event, stream[0]);
    cudaEventRecord(d_zeroed_event, stream[1]);
}
void cuda_setup_matrix_init_wait_until_done(){
    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
}

void cuda_setup_matrix_rhs(double* vec_v, double* vec_rhs, double* vec_d, double* vec_a, 
                           double* vec_b, int* parent_index, int ncell, int len)
{
    int block_num = (len + nthread_per_block - 1) / nthread_per_block;
    rhs_kernel<<<block_num, nthread_per_block>>>(vec_v, vec_rhs, vec_d, vec_a, vec_b, parent_index, ncell, len);

    #ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("setup_rhs error:%s\n", cudaGetErrorString(cuda_status));
	}
    #endif 
}

void cuda_setup_matrix_lhs(double* vec_d, double* vec_a, double* vec_b, int* parent_index, int ncell, int len)
{
    int block_num = (len + nthread_per_block - 1) / nthread_per_block;
    lhs_kernel<<<block_num, nthread_per_block>>>(vec_d, vec_a, vec_b, parent_index, ncell, len);

    // #ifdef DEBUG
	// cudaDeviceSynchronize();
	// cudaError_t cuda_status = cudaGetLastError();
	// if (cuda_status != cudaSuccess)
	// {
	// 	printf("setup_lhs error:%s\n", cudaGetErrorString(cuda_status));
	// }
    // #endif 
}
cudaStream_t *lhs_rhs_stream = nullptr;
void cuda_setup_matrix_lhs_and_rhs(double* vec_v, double* vec_rhs, double* vec_d, double* vec_a, 
                           double* vec_b, int* parent_index, int ncell, int len)
{
    if(!lhs_rhs_stream){
        cuda_stream_initialize((void**)&lhs_rhs_stream);
    }
    static int minGridSize = 0;
    static int block_size = 0;
    if(block_size == 0){
        cudaOccupancyMaxPotentialBlockSize(&minGridSize,&block_size,lhs_and_rhs_kernel,
            0,  // 动态共享内存大小
            0   // block size 上限
        );
    }
    int block_num = (len + block_size - 1) / block_size;
    lhs_and_rhs_kernel<<<block_num, block_size,0,*lhs_rhs_stream>>>(vec_v, vec_rhs, vec_d, vec_a, vec_b, parent_index, ncell, len);

    #ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("lhs_and_rhs_kernel error:%s\n", cudaGetErrorString(cuda_status));
	}
    #endif 
}

void cuda_solve_matrix(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index, int len, int ncell)
{
    //int block_num = (len + nthread_per_block - 1) / nthread_per_block;
    //单线程
    solve_matrix_kernel<<<1, 1>>>(vec_a, vec_b, vec_d, vec_rhs, parent_index, len, ncell);

    #ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("solve error:%s\n", cudaGetErrorString(cuda_status));
	}
    #endif 
}


void cuda_solve_permute1(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index, int nstride, 
                         int* stride, int* firstnode, int* lastnode, int* cellsize, int ncell, int len)
{
    int block_num = (ncell + nthread_per_block - 1) / nthread_per_block;
    solve_permute1_kernel<<<block_num, nthread_per_block>>>(vec_a, vec_b, vec_d, vec_rhs, parent_index, nstride,
                                                            stride, firstnode, lastnode, cellsize, ncell, len);

    #ifdef DEBUG
    cudaDeviceSynchronize();
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("update error:%s\n", cudaGetErrorString(cuda_status));
	}
    #endif
}


void cuda_cop_solve(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index,
                 int* max_order_each_thread, int* min_order_each_thread, int* firstnode, int* lastnode,
                 int* stride, int* map_t2c, int norder, int ncell, int nthread)
{
    int block_num = (nthread + nthread_per_block - 1) / nthread_per_block;
    cop_solve_kernel<<<block_num, nthread_per_block>>>(vec_a, vec_b, vec_d, vec_rhs, parent_index,
                                                      max_order_each_thread, min_order_each_thread,
                                                      firstnode, lastnode, stride, map_t2c, norder,
                                                      ncell, nthread);
    #ifdef DEBUG
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("cop solve error:%s\n", cudaGetErrorString(cuda_status));
	}
    #endif 
}

void cuda_update(double* vec_v, double* vec_rhs, int len)
{
    int block_num = (len + nthread_per_block - 1) / nthread_per_block;
    update_kernel<<<block_num, nthread_per_block>>>(vec_v, vec_rhs, len);

    #ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
	{
		printf("update error:%s\n", cudaGetErrorString(cuda_status));
	}
    #endif
}




void Simulate::setup_tree_matrix_gpu(HelioXroupData* p_neuron)
{
    int ncell = p_neuron->ncell;
    int len = p_neuron->len;
    double* vec_v = p_neuron->vecdata_v->get_gpu_data();
    double* vec_a = p_neuron->vecdata_a->get_gpu_data();
    double* vec_b = p_neuron->vecdata_b->get_gpu_data();
    double* vec_d = p_neuron->vecdata_d->get_gpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_gpu_data();
    double* vec_sav_rhs = p_neuron->need_fast_imem ? p_neuron->vecdata_sav_rhs->get_gpu_data() : nullptr;
    double* vec_sav_d = p_neuron->need_fast_imem ? p_neuron->vecdata_sav_d->get_gpu_data() : nullptr;
    int* parent_index = p_neuron->vecdata_parent_index->get_gpu_data();

    //矩阵已经在fadvance启动时被清空了，这里不需要再清空
    SimMechCurrentParam param = {vec_v, vec_d, vec_rhs, t};
    // Ensure rhs/d async memset is complete before any mechanism stream starts atomicAdd into them.
    // This avoids relying on legacy default-stream synchronization rules.
    if (rhs_zeroed_event && d_zeroed_event) {
        auto wait_memset = [&](Mechanism* mech) {
            if (mech && mech->cuda_stream) {
                cudaStream_t s = *reinterpret_cast<cudaStream_t*>(mech->cuda_stream);
                cudaStreamWaitEvent(s, rhs_zeroed_event, 0);
                cudaStreamWaitEvent(s, d_zeroed_event, 0);
            }
        };
        for (auto p_ion : p_neuron->vec_eion) {
            wait_memset(p_ion);
        }
        for (auto p_mech : p_neuron->mechanism_list) {
            wait_memset(p_mech);
        }
    }

    // Phase 1: compute all EION currents first (erev depends on concentrations).
    // Historically this was enforced by a cudaDeviceSynchronize barrier; we replace it with
    // lightweight per-mechanism events so non-ion currents can wait without a device-wide sync.
    for (auto p_ion : p_neuron->vec_eion) {
        p_ion->current_gpu(param);
    }

    // Record completion of EION current kernels on their own streams.
    for (auto p_ion : p_neuron->vec_eion) {
        if (p_ion && p_ion->cuda_stream && p_ion->cuda_event) {
            cudaStream_t s = *reinterpret_cast<cudaStream_t*>(p_ion->cuda_stream);
            cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(p_ion->cuda_event);
            cudaEventRecord(*ev, s);
        }
    }

    // Phase 2: compute remaining mechanism currents. Each mechanism stream waits for all EION
    // events, ensuring it observes updated erev before contributing to rhs/d.
    for (auto p_mech : p_neuron->mechanism_list) {
        if (p_mech && p_mech->cuda_stream) {
            cudaStream_t s = *reinterpret_cast<cudaStream_t*>(p_mech->cuda_stream);
            for (auto p_ion : p_neuron->vec_eion) {
                if (p_ion && p_ion->cuda_event) {
                    cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(p_ion->cuda_event);
                    cudaStreamWaitEvent(s, *ev, 0);
                }
            }
        }
        p_mech->current_gpu(param);
    }

    // Ensure all current contributions are visible before building matrix and fast_imem buffers.
    // We avoid a device-wide synchronize by joining mechanism streams onto the (default) stream.
    for (auto p_ion : p_neuron->vec_eion) {
        if (p_ion && p_ion->cuda_event) {
            cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(p_ion->cuda_event);
            cudaStreamWaitEvent(0, *ev, 0);
        }
    }
    for (auto p_mech : p_neuron->mechanism_list) {
        if (p_mech && p_mech->cuda_stream && p_mech->cuda_event) {
            cudaStream_t s = *reinterpret_cast<cudaStream_t*>(p_mech->cuda_stream);
            cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(p_mech->cuda_event);
            cudaEventRecord(*ev, s);
            cudaStreamWaitEvent(0, *ev, 0);
        }
    }

    // fast_imem: save membrane-only RHS contribution (before axial terms).
    if (p_neuron->need_fast_imem) {
        int block_num = (len + nthread_per_block - 1) / nthread_per_block;
        negate_copy_kernel<<<block_num, nthread_per_block>>>(vec_sav_rhs, vec_rhs, len);
    }

    // Axial contribution to RHS.
    cuda_setup_matrix_rhs(vec_v, vec_rhs, vec_d, vec_a, vec_b, parent_index, ncell, len);

    // Capacitance contribution (membrane-only).
    p_neuron->mech_cap->cap_jacob_gpu(p_neuron->cj, vec_d);

    // fast_imem: save membrane-only diagonal contribution (after capacitance, before axial terms).
    if (p_neuron->need_fast_imem) {
        int block_num = (len + nthread_per_block - 1) / nthread_per_block;
        copy_kernel<<<block_num, nthread_per_block>>>(vec_sav_d, vec_d, len);
    }

    // Axial contribution to LHS.
    cuda_setup_matrix_lhs(vec_d, vec_a, vec_b, parent_index, ncell, len);

}


void Simulate::spike_deliver_gpu()
{
    tag_event("network_spike_send_gpu");
    network_spike_send_gpu();
    tag_event_end();
    
    tag_event("network_spike_receive_gpu");
    network_spike_receive_gpu();
    tag_event_end();
}

/*
 * for all synapse mechanism, call pre_spike_send() to 
 * put fired spikes into spike buffers
 */
void Simulate::network_spike_send_gpu()
{
    for (auto p_neuron:neuron_group_list)
    {
        PreSyn *ps = p_neuron->presyn;
        double* vec_v = p_neuron->vecdata_v->get_gpu_data();
        ps->threshold_detect_gpu(vec_v, p_neuron->vecdata_spk_flags, t, rec_spikes);
    }
}

/*
 * for all synapse mechanism, call post_spike_receive() to 
 * deal with fired spikes, if firetime + delay <= t, the 
 * NET_RECEIVE block in .mod file should be called 
 */
void Simulate::network_spike_receive_gpu()
{
    bool hasSent = false;
    do{
        hasSent = false;
        for (HelioXroupData* p_neuron : neuron_group_list)
        {
            if(p_neuron->spk_vec->hasNewSpk){
                SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
                for(auto postsyn : p_neuron->vec_postsyn)
                {
                    postsyn->get_spike_from_vec_gpu(p_neuron->spk_vec, spk_flags, t);
                }
                
                clearValidSpkFlags(p_neuron->vecdata_spk_flags);
                p_neuron->vecdata_spk_flags->update_gpu_data_from_cpu();
            }
            for (auto postsyn:p_neuron->vec_postsyn)
            {
                postsyn->post_spike_receive_gpu(t + dt / 2);
                const bool has_to_deliver = postsyn->get_receive_count() > 0;
                if (spike_profile_enabled_) {
                    if (has_to_deliver) {
                        net_receive_called_ += 1;
                    } else {
                        net_receive_skipped_ += 1;
                    }
                }
                if (has_to_deliver) {
                    hasSent |= postsyn->net_receive_gpu(t);
                }
            }
        }
    }while(hasSent);
}

void Simulate::solve_matrix_gpu(HelioXroupData* p_neuron)
{
    int len = p_neuron->len;
    int ncell = p_neuron->ncell;
    double* vec_a = p_neuron->vecdata_a->get_gpu_data();
    double* vec_b = p_neuron->vecdata_b->get_gpu_data();
    double* vec_d = p_neuron->vecdata_d->get_gpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_gpu_data();
    int* parent_index = p_neuron->vecdata_parent_index->get_gpu_data();

    if (permute_type == 0)
    {
        cuda_solve_matrix(vec_a, vec_b, vec_d, vec_rhs, parent_index, len, ncell);
    }
    else if (permute_type == 1)
    {
        int nstride = p_neuron->nstride;
        int* stride = p_neuron->vecdata_stride->get_gpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_gpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_gpu_data();
        int* cellsize = p_neuron->vecdata_cellsize->get_gpu_data();
        cuda_solve_permute1(vec_a, vec_b, vec_d, vec_rhs, parent_index, nstride, stride, firstnode,
                            lastnode, cellsize, ncell, len);
    }
    else if (permute_type == 3)
    {
        int norder = p_neuron->norder;
        int nthread = p_neuron->threads_num;
        int* stride = p_neuron->vecdata_stride->get_gpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_gpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_gpu_data();
        int* max_order_each_thread = p_neuron->vecdata_max_order_each_thread->get_gpu_data();
        int* min_order_each_thread = p_neuron->vecdata_min_order_each_thread->get_gpu_data();
        int* map_t2c = p_neuron->vecdata_map_t2c->get_gpu_data();
        cuda_cop_solve(vec_a, vec_b, vec_d, vec_rhs, parent_index, max_order_each_thread, 
                      min_order_each_thread, firstnode, lastnode, stride, map_t2c, norder, 
                      ncell, nthread);
        
    }
}
//vec_v[i] += vec_rhs[i];
void Simulate::update_gpu(HelioXroupData* p_group)
{
    int len = p_group->len;
    double* vec_v = p_group->vecdata_v->get_gpu_data();
    double* vec_rhs = p_group->vecdata_rhs->get_gpu_data();

    cuda_update(vec_v, vec_rhs, len);//vec_v[i] += vec_rhs[i];

	p_group->mech_cap->cap_current_gpu(p_group->cj, vec_rhs);//icap[i] = cfac * cm[i] * vec_rhs[node_index];
}

void Simulate::last_part_gpu()
{
    t += 0.5 * dt;
    int ncell = neuron_group_list.size();
    for (int i = 0; i < ncell; i++)
    {
        HelioXroupData* p_neuron = neuron_group_list[i];
        p_neuron->vec_play_continuous.continuous_gpu(t);
        nonvint_gpu(p_neuron);
    }
    hdf5_manager.log_data_gpu();
    // Mark default stream completion for this timestep so the next step's non-blocking spike stream
    // won't observe stale data from default-stream kernels (e.g. VecPlay).
    if (default_stream_step_done_event) {
        cudaEventRecord(default_stream_step_done_event, 0);
    }
}

void Simulate::nonvint_gpu(HelioXroupData* p_neuron)
{
    SimMechStateParam param = {p_neuron->vecdata_v->get_gpu_data(), dt, t};
    for(auto p_mech:p_neuron->mech_write_state_ion_list){
        p_mech->state_gpu(param);
    }

    for (auto p_mech:p_neuron->mechanism_list)
    {
        if(!p_mech->write_state_ion){
            p_mech->state_gpu(param);
        }
    }
    //结束了该步仿真，然后继续仿真的第一步，是在零号流上进行的，所以有隐式的同步，不需要继续同步
}

void Simulate::fadvance_gpu()
{
    for (HelioXroupData* p_neuron:neuron_group_list){
        // If the user added/updated VecPlay after finitialize(), the VecPlay table is dirty on CPU.
        // Sync it here so GPU kernels never see stale/null pointers.
        p_neuron->vec_play_continuous.try_update_gpu();
        //异步地memset清除rhs和d，由于spike deliver会使用0号流，因此可以保证在setup matrix前已经memset完毕
        //由于lastpart gpu也是异步的，并且不依赖rhs和d（只依赖v），因此，可以和未完成的last part一起调度
        cuda_setup_matrix_init(p_neuron);
    }

    // Ensure spike detection (on non-blocking per-PreSyn stream) does not race ahead of
    // default-stream work from the previous step. This preserves timestep semantics while still
    // allowing overlap with non-default mech streams (e.g. ion state updates).
    if (default_stream_step_done_event) {
        for (HelioXroupData* p_neuron : neuron_group_list) {
            if (p_neuron && p_neuron->presyn && p_neuron->presyn->spike_stream) {
                cudaStream_t s = *reinterpret_cast<cudaStream_t*>(p_neuron->presyn->spike_stream);
                cudaStreamWaitEvent(s, default_stream_step_done_event, 0);
            }
        }
    }

    int n_neuron = neuron_group_list.size();
	spike_deliver_gpu();
    t += 0.5 * dt;
    for (HelioXroupData* p_neuron:neuron_group_list)
    {
        p_neuron->vec_play_continuous.play_gpu(t);
        p_neuron->vec_play_continuous.continuous_gpu(t);
        setup_tree_matrix_gpu(p_neuron);
        solve_matrix_gpu(p_neuron);
        if (p_neuron->need_fast_imem) {
            const int len = p_neuron->len;
            int block_num = (len + nthread_per_block - 1) / nthread_per_block;
            fast_imem_kernel<<<block_num, nthread_per_block>>>(
                p_neuron->vecdata_i_membrane_->get_gpu_data(),
                p_neuron->vecdata_sav_d->get_gpu_data(),
                p_neuron->vecdata_rhs->get_gpu_data(),  // delta_v
                p_neuron->vecdata_sav_rhs->get_gpu_data(),
                p_neuron->vecdata_area->get_gpu_data(),
                len);
        }
        update_gpu(p_neuron);
        if(p_neuron->have_gap){
            gap_transfer_gpu(p_neuron);//TODO:目前是在流0上执行的，后续可以考虑优化
        }
    }
    last_part_gpu();
}


void Simulate::finitialize_gpu(double v_init)
{
    int n_neuron;
    n_neuron = neuron_group_list.size();

    // 同步所有pending的gap transfers
    for (auto p_group : neuron_group_list) {
        p_group->gpu_gap_trans_info.sync_to_gpu();
    }

        // 重置所有VecPlay状态
    for (int i = 0; i < n_neuron; i++)
    {
        HelioXroupData* p_neuron = this->neuron_group_list[i];
        p_neuron->vec_play_continuous.try_update_gpu();//如果用户修改了play_continuous需要更新
        p_neuron->vec_play_continuous.reset_all_gpu();
        CUDA_CHECK_ERR();
        p_neuron->vec_play_continuous.play_gpu(t);
        CUDA_CHECK_ERR();
        p_neuron->vec_play_continuous.continuous_gpu(t);
        CUDA_CHECK_ERR();
    }
    // 清空spike_buffer
    for (int i = 0; i < n_neuron; i++)
    {
        auto p_neuron = neuron_group_list[i];
        vector<PostSyn_trait*> postsyns = p_neuron->vec_postsyn;
        for (PostSyn_trait* postsyn: postsyns) {
            while (!postsyn->spike_buffer.empty()){
                postsyn->spike_buffer.pop();
            }
        }
    }
    for (auto p_neuron:neuron_group_list)
    {
        p_neuron->cj = 1.0 / dt;
        double* vec_v = p_neuron->vecdata_v->get_gpu_data();
        cuda_finitialize(vec_v, v_init, p_neuron->len);
        CUDA_CHECK_ERR();
        if (p_neuron->have_gap){
            gap_transfer_gpu(p_neuron);
        }
        SimMechInitialParam param = {vec_v,0};
        for(auto p_ion:p_neuron->vec_eion){
            p_ion->initialize_gpu(param);
            CUDA_CHECK_ERR();
        }

        // Ensure ion initialization (on per-mechanism streams) completes before launching
        // mechanism initialization on the default stream.
        for (auto p_ion : p_neuron->vec_eion) {
            if (p_ion && p_ion->cuda_stream && p_ion->cuda_event) {
                cudaStream_t s = *reinterpret_cast<cudaStream_t*>(p_ion->cuda_stream);
                cudaEvent_t* ev = reinterpret_cast<cudaEvent_t*>(p_ion->cuda_event);
                cudaEventRecord(*ev, s);
                cudaStreamWaitEvent(0, *ev, 0);
            }
        }

        for(auto p_mech:p_neuron->mechanism_list){
            p_mech->initialize_gpu(param);
            CUDA_CHECK_ERR();
        }
    }

    printf_debug("mech init done\n");

    hdf5_manager.finitialize();
    hdf5_manager.log_data_gpu();

    clearAllSpikes_gpu();
    
    spike_deliver_gpu();
    CUDA_CHECK_ERR();
    

    
    for (int i = 0; i < n_neuron; i++)
    {
        HelioXroupData* p_neuron = this->neuron_group_list[i];
        // Ensure rhs/d buffers are cleared before building the initial matrix.
        cuda_setup_matrix_init(p_neuron);
        setup_tree_matrix_gpu(p_neuron);
        CUDA_CHECK_ERR();
        if (p_neuron->need_fast_imem) {
            const int len = p_neuron->len;
            int block_num = (len + nthread_per_block - 1) / nthread_per_block;
            fast_imem_init_kernel<<<block_num, nthread_per_block>>>(
                p_neuron->vecdata_i_membrane_->get_gpu_data(),
                p_neuron->vecdata_rhs->get_gpu_data(),      // rhs before solve
                p_neuron->vecdata_sav_rhs->get_gpu_data(),  // membrane rhs contribution
                p_neuron->vecdata_area->get_gpu_data(),
                len);
            CUDA_CHECK_ERR();
        }
    }

    // Seed the default-stream "step done" event so the first fadvance() can safely wait on it.
    if (default_stream_step_done_event) {
        cudaEventRecord(default_stream_step_done_event, 0);
    }
}

__global__ void gap_transfer_kernel(double **dst, double **src, int ntrans) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ntrans) {
        *(dst[idx]) = *(src[idx]);
    }
}

void Simulate::gap_transfer_gpu(HelioXroupData *p_group) {
    auto &gap_info = p_group->gpu_gap_trans_info;
    int ntrans = gap_info.ntrans();
    if (ntrans == 0) return;
    
    int block_num = (ntrans + nthread_per_block - 1) / nthread_per_block;
    gap_transfer_kernel<<<block_num, nthread_per_block>>>(
        gap_info.dst.get_gpu_data(), 
        gap_info.src.get_gpu_data(), 
        ntrans
    );
}
