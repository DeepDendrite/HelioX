#include "simulate.h"
#include "cuda_utils.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <magic_enum/magic_enum.hpp>
#include "mechanism.h"
namespace fs = std::filesystem;
extern void destroy_cuda_streams();

static bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return false;
    // Accept: 1/true/yes/on (case-insensitive for first char)
    const char c = v[0];
    return c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y' || c == 'o' || c == 'O';
}

Simulate::Simulate(Mode _mode,BufferEnable buf_enable ,int buffer_size) : mode(_mode), hdf5_manager(mode,buffer_size,buf_enable)
{
    t = 0;
    dt = 0;
    hdf5_file = nullptr;
    tstop = 0;
    permute_type = 0;
    output_folder = "output";

    maybe_init_spike_profile_();
}

Simulate::~Simulate()
{
    int n = neuron_group_list.size();
    for (int i = 0; i < n; i++)
        delete neuron_group_list[i];
    neuron_group_list.clear();
    if (mode == GPU) {
        destroy_cuda_streams();
    }
}

void Simulate::maybe_init_spike_profile_() {
    spike_profile_enabled_ = env_flag_enabled("HELIOX_PROFILE_SPIKE");
    if (!spike_profile_enabled_) {
        return;
    }
    net_receive_called_ = 0;
    net_receive_skipped_ = 0;
    printf("Spike profiling enabled (HELIOX_PROFILE_SPIKE=1)\n");
}

void Simulate::print_spike_profile_summary_() const {
    if (!spike_profile_enabled_) return;
    uint64_t steps = 0;
    uint64_t steps_with_spike = 0;
    uint64_t presyn_total = 0;
    int presyn_max = 0;
    std::array<uint64_t, 64> hist{};

    for (auto* g : neuron_group_list) {
        if (!g || !g->presyn) continue;
        const auto& s = g->presyn->spike_profile_stats();
        steps += s.steps;
        steps_with_spike += s.steps_with_presyn_spike;
        presyn_total += s.presyn_spike_total;
        presyn_max = std::max(presyn_max, s.presyn_spike_max);
        for (size_t i = 0; i < hist.size(); ++i) {
            hist[i] += s.presyn_spike_hist[i];
        }
    }

    const uint64_t steps_no_spike = steps - steps_with_spike;
    const double avg_spk_per_step = steps ? (double)presyn_total / (double)steps : 0.0;
    const double avg_spk_per_spike_step = steps_with_spike ? (double)presyn_total / (double)steps_with_spike : 0.0;

    printf("\n=== Spike Profile Summary ===\n");
    printf("Steps: %llu\n", (unsigned long long)steps);
    printf("PreSyn spikes: total=%llu avg/step=%.6f avg/(spike-step)=%.6f max/step=%d\n",
           (unsigned long long)presyn_total, avg_spk_per_step, avg_spk_per_spike_step, presyn_max);
    printf("Steps with PreSyn spike: %llu (%.2f%%), no spike: %llu (%.2f%%)\n",
           (unsigned long long)steps_with_spike,
           steps ? 100.0 * (double)steps_with_spike / (double)steps : 0.0,
           (unsigned long long)steps_no_spike,
           steps ? 100.0 * (double)steps_no_spike / (double)steps : 0.0);
    printf("net_receive_gpu: called=%llu skipped=%llu (skip %.2f%%)\n",
           (unsigned long long)net_receive_called_,
           (unsigned long long)net_receive_skipped_,
           (net_receive_called_ + net_receive_skipped_) ?
               (100.0 * (double)net_receive_skipped_ / (double)(net_receive_called_ + net_receive_skipped_)) : 0.0);

    // Print a compact histogram for small spike counts (0..10) plus tail.
    printf("PreSyn spike count histogram (per step):\n");
    for (int i = 0; i <= 10; ++i) {
        printf("  %2d: %llu\n", i, (unsigned long long)hist[(size_t)i]);
    }
    uint64_t tail = 0;
    for (size_t i = 11; i < hist.size(); ++i) tail += hist[i];
    printf("  11+: %llu (bucket[63] includes >=63)\n", (unsigned long long)tail);
    printf("=== End Spike Profile ===\n\n");
}

void Simulate::finitialize(double v_init)
{
    t = 0;
    CUDA_CHECK_ERR();

    // Enable spike profiling on the fully constructed model (PreSyn exists only after data_format_trans()).
    if (spike_profile_enabled_) {
        for (auto* g : neuron_group_list) {
            if (g && g->presyn) {
                g->presyn->set_spike_profile_enabled(true);
                g->presyn->reset_spike_profile_stats();
            }
        }
        net_receive_called_ = 0;
        net_receive_skipped_ = 0;
    }
    
    // 同步所有pending的gap transfers（GPU模式）
    // 必须在finitialize_gpu之前同步，因为finitialize_gpu会调用gap_transfer_gpu
    if (this->mode == GPU) {
        for (auto p_group : neuron_group_list) {
            p_group->gpu_gap_trans_info.sync_to_gpu();
        }
    }
    
    if (this->mode == CPU)
    {
        finitialize_cpu(v_init);
        CUDA_CHECK_ERR();
    }
    else if (this->mode == GPU)
    {
        finitialize_gpu(v_init);
        CUDA_CHECK_ERR();
    }
}

pair<double*, double*> Simulate::getVarPtr(VarDescriptor &descriptor, bool will_panic){
    #define PANIC_OR_RETURN_NULL if(will_panic) { \
                                        assert(false); \
                                    } else { \
                                        return {nullptr, nullptr}; \
                                    }

    double *var_ptr_cpu = nullptr;
    double *var_ptr_gpu = nullptr;

    auto &mechFactory = MechanismFactory::getInstance();
    auto varMapPtr = mechFactory.getVarMap(descriptor.mech);
    if (varMapPtr == nullptr) {
        printf("mech:%s not found\n", descriptor.mech.c_str());
        PANIC_OR_RETURN_NULL;
    }

    // 简化：直接传递VarDescriptor
    var_ptr_cpu = varMapPtr->getVarPtr(descriptor, Mode::CPU);
    if (var_ptr_cpu == nullptr) {
        printf("mech:%s var:%s node_or_mech_idx:%d array_index:%d not found\n",
                    descriptor.mech.c_str(),
                    descriptor.var.c_str(),
                    descriptor.node_or_mech_idx,
                    descriptor.array_index);
        PANIC_OR_RETURN_NULL;
    }

    if(this->mode == GPU){
        var_ptr_gpu = varMapPtr->getVarPtr(descriptor, Mode::GPU);
        if (var_ptr_gpu == nullptr) {
            printf("mech:%s var:%s node_or_mech_idx:%d array_index:%d not found (GPU)\n",
                        descriptor.mech.c_str(),
                        descriptor.var.c_str(),
                        descriptor.node_or_mech_idx,
                        descriptor.array_index);
            PANIC_OR_RETURN_NULL;
        }
    }
    #undef PANIC_OR_RETURN_NULL
    return {var_ptr_cpu, var_ptr_gpu};
}
std::map<VarDescriptor, int> Simulate::init_monitor_data_sets(std::vector<VarDescriptor> &monitors) {
    std::map<VarDescriptor, int> monitor_to_handle;

    // 只在启用HDF5时创建HDF5文件
    if(hdf5_manager.isEnable(BufferEnable::HDF5)) {
        if(hdf5_file == nullptr) {
            if(!fs::exists(output_folder)) {
                fs::create_directories(output_folder);
            }
            // 创建 HDF5 文件
            hdf5_file = std::make_unique<HighFive::File>(output_folder + "/sim_out.h5",HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        }
    }
    
    int group_size = neuron_group_list.size();

    assert(group_size == 1); //目前只支持一个group

    // 只在启用HDF5时设置数据集属性
    HighFive::DataSetCreateProps prop;
    HighFive::Group hdf5_group;
    if(hdf5_manager.isEnable(BufferEnable::HDF5)) {
        prop.add(HighFive::Chunking({100})); 
        
        for (int i = 0; i < group_size; i++) {
            // 构造组名称，例如 "/group_0"
            std::string group_name = "/group_" + std::to_string(i);
            // 如果组不存在，则先创建组
            if (!hdf5_file->exist(group_name)) {
                hdf5_file->createGroup(group_name);
            }
            // 现在获取组不会出错
            hdf5_group = hdf5_file->getGroup(group_name);
            break; // 目前只支持一个group，所以直接break
        }
    }

        for (auto &monitorPoint : monitors) {
            // printf("adding monitor:%s %s %d\n", monitorPoint.mech.c_str(), monitorPoint.var.c_str(), monitorPoint.node_or_mech_idx);
            // 找到被监控的变量指针
            
            // 创建新的监控点
            RecordPoint watchPoint;
            auto [var_ptr_cpu, var_ptr_gpu] = getVarPtr(monitorPoint, true);
            watchPoint.var_ptr_cpu = var_ptr_cpu;
            watchPoint.var_ptr_gpu = var_ptr_gpu;

        // 只有在启用HDF5时才创建数据集
        if(hdf5_manager.isEnable(BufferEnable::HDF5)) {
            // 创建新的数据集路径：[group_i][mech_name][var_name][idx]
            std::string group_name = "/group_0"; // 目前只支持一个group
            std::string dataset_path = group_name + "/" + monitorPoint.mech + "/" + 
                                       monitorPoint.var + "/" + 
                                       std::to_string(monitorPoint.node_or_mech_idx);

            // 检查路径中的组是否存在，如果不存在则创建
            std::string mech_group_path = group_name + "/" + monitorPoint.mech;
            if (!hdf5_file->exist(mech_group_path)) {
                hdf5_group.createGroup(monitorPoint.mech);
            }

            std::string var_group_path = mech_group_path + "/" + monitorPoint.var;
            if (!hdf5_file->exist(var_group_path)) {
                hdf5_file->getGroup(mech_group_path).createGroup(monitorPoint.var);
            }

            // 创建数据集
            watchPoint.dataset = hdf5_file->createDataSet(
                dataset_path,
                HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
                HighFive::AtomicType<double>(),
                prop
            );
        }

        // 添加到监控点列表，返回的handle就是新的monitorId
        int handle = hdf5_manager.push_back(monitorPoint, watchPoint);
        monitor_to_handle[monitorPoint] = handle;
        // printf("Monitor added: %s %s %d -> handle %d\n", 
        //        monitorPoint.mech.c_str(), monitorPoint.var.c_str(), 
        //        monitorPoint.node_or_mech_idx, handle);
    }
    hdf5_manager.initialize(mode);
    return monitor_to_handle;
}

void Simulate::finitialize_cpu(double v_init)
{
    int n_group = neuron_group_list.size();

    // 重置所有VecPlay状态
    for (int i = 0; i < n_group; i++)
    {
        auto p_neuron = neuron_group_list[i];
        p_neuron->vec_play_continuous.reset_all_cpu();
        p_neuron->vec_play_continuous.play_cpu(t);          // TODO: 目前不确定是不是放在这里
        p_neuron->vec_play_continuous.continuous_cpu(t);
    }
    // 清空spike_buffer
    for (int i = 0; i < n_group; i++)
    {
        auto p_neuron = neuron_group_list[i];
        vector<PostSyn_trait*> postsyns = p_neuron->vec_postsyn;
        for (PostSyn_trait* postsyn: postsyns) {
            while (!postsyn->spike_buffer.empty()){
                postsyn->spike_buffer.pop();
            }
        }
    }
    for (int i = 0; i < n_group; i++)
    {
        auto p_group = neuron_group_list[i];
        p_group->cj = 1.0 / dt;
        double* vec_v = p_group->vecdata_v->get_cpu_data();
        for (int j = 0; j < p_group->len; j++)
        {
            vec_v[j] = v_init;
        }
        if (p_group->have_gap){
            gap_transfer_cpu(p_group);
        }
        SimMechInitialParam params;
        params.v = vec_v;
        params.dt = dt;
        for (auto p_mech : p_group->mech_current_list) {
            p_mech->initialize_cpu(params);
        }
    }
    hdf5_manager.finitialize();
    hdf5_manager.log_data_cpu();

    clearAllSpikes_cpu();

    spike_deliver_cpu();
    
    
    for (int i = 0; i < n_group; i++)
    {
        auto p_neuron = neuron_group_list[i];
        setup_tree_matrix_cpu(p_neuron);
        if (p_neuron->need_fast_imem) {
            // Fixed-step init semantics (CoreNEURON nrn_calc_fast_imem_init):
            // i_membrane_ = (rhs + sav_rhs) * area * 0.01
            const int len = p_neuron->len;
            double* vec_area = p_neuron->vecdata_area->get_cpu_data();
            double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();       // rhs before solve
            double* vec_sav_rhs = p_neuron->vecdata_sav_rhs->get_cpu_data();
            double* vec_imem = p_neuron->vecdata_i_membrane_->get_cpu_data();
            for (int j = 0; j < len; ++j) {
                vec_imem[j] = (vec_rhs[j] + vec_sav_rhs[j]) * vec_area[j] * 0.01;
            }
        }
    }

}


/*run simulate*/
void Simulate::run()
{
    int nstep = tstop / dt;
	t = 0.0;
    
    // 同步所有pending的gap transfers（GPU模式）
    if (this->mode == GPU) {
        for (auto p_group : neuron_group_list) {
            p_group->gpu_gap_trans_info.sync_to_gpu();
        }
    }
    
    if (this->mode == CPU){
        for (int iter = 0; iter < nstep; iter++)
        {
            fadvance_cpu();
        }
        hdf5_manager.flush_cpu();
    }else{
        for (int iter = 0; iter < nstep; iter++)
        {
            fadvance_gpu();
        }
        cuda_sync_all();
        hdf5_manager.flush_gpu();
    }

    print_spike_profile_summary_();
}

void Simulate::continue_run(double runtime)
{
    int nstep = runtime / dt;
    if (nstep <= 0) {
        return;
    }

    // 同步所有pending的gap transfers（GPU模式）
    if (this->mode == GPU) {
        for (auto p_group : neuron_group_list) {
            p_group->gpu_gap_trans_info.sync_to_gpu();
        }
    }

    if (this->mode == CPU) {
        for (int iter = 0; iter < nstep; iter++) {
            fadvance_cpu();
        }
        hdf5_manager.flush_cpu();
    } else {
        for (int iter = 0; iter < nstep; iter++) {
            fadvance_gpu();
        }
        cuda_sync_all();
        hdf5_manager.flush_gpu();
    }
}

void Simulate::fadvance()
{
    if (this->mode == CPU)
        fadvance_cpu();
    else
        fadvance_gpu();
}

/*one step in simulation*/
void Simulate::fadvance_cpu()
{

    int n_neuron = neuron_group_list.size();
	spike_deliver_cpu();
    

    t += 0.5 * dt;
    for (auto p_neuron : neuron_group_list)
    {
        p_neuron->vec_play_continuous.play_cpu(t);
        p_neuron->vec_play_continuous.continuous_cpu(t);
        setup_tree_matrix_cpu(p_neuron);
        solve_matrix_cpu(p_neuron);
        if (p_neuron->need_fast_imem) {
            const int len = p_neuron->len;
            double* vec_area = p_neuron->vecdata_area->get_cpu_data();
            double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();       // now holds delta_v
            double* vec_sav_rhs = p_neuron->vecdata_sav_rhs->get_cpu_data();
            double* vec_sav_d = p_neuron->vecdata_sav_d->get_cpu_data();
            double* vec_imem = p_neuron->vecdata_i_membrane_->get_cpu_data();
            for (int i = 0; i < len; ++i) {
                vec_imem[i] = (vec_sav_d[i] * vec_rhs[i] + vec_sav_rhs[i]) * vec_area[i] * 0.01;
            }
        }
        update_cpu(p_neuron);
        if(p_neuron->have_gap){
            gap_transfer_cpu(p_neuron);
        }
    }
    last_part_cpu();

}


void Simulate::spike_deliver_cpu()
{
    network_spike_send_cpu();
    network_spike_receive_cpu();
}

/*
 * setup the matrix of calble equation
 * compute mechanism current
 */
void Simulate::setup_tree_matrix_cpu(HelioXroupData* p_neuron)
{
    int len = p_neuron->len;
    int ncell = p_neuron->ncell;
    double* vec_v = p_neuron->vecdata_v->get_cpu_data();
    double* vec_a = p_neuron->vecdata_a->get_cpu_data();
    double* vec_b = p_neuron->vecdata_b->get_cpu_data();
    double* vec_d = p_neuron->vecdata_d->get_cpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();
    double* vec_sav_rhs = p_neuron->need_fast_imem ? p_neuron->vecdata_sav_rhs->get_cpu_data() : nullptr;
    double* vec_sav_d = p_neuron->need_fast_imem ? p_neuron->vecdata_sav_d->get_cpu_data() : nullptr;
    int* parent_index = p_neuron->vecdata_parent_index->get_cpu_data();

    /*rhs part*/
    std::fill(vec_rhs, vec_rhs + len, 0.0);
    std::fill(vec_d, vec_d + len, 0.0);

    SimMechCurrentParam params = {vec_v, vec_d, vec_rhs, t};

    const char* dbg_env = std::getenv("HELIOX_DEBUG_STEP_LOG");
    bool dbg_enabled = dbg_env && dbg_env[0] != '0';
    int dbg_node = 0;
    if (const char* node_env = std::getenv("HELIOX_DEBUG_NODE_INDEX")) {
        dbg_node = std::atoi(node_env);
        if (dbg_node < 0) dbg_node = 0;
        if (dbg_node >= len) dbg_node = len - 1;
    }
    double dbg_t_start = -1.0;
    double dbg_t_end = -1.0;
    if (const char* t_env = std::getenv("HELIOX_DEBUG_T_START")) {
        dbg_t_start = std::atof(t_env);
    }
    if (const char* t_env = std::getenv("HELIOX_DEBUG_T_END")) {
        dbg_t_end = std::atof(t_env);
    }
    const char* dbg_mechs = std::getenv("HELIOX_DEBUG_MECHS");
    auto dbg_mech_match = [&](const char* name) -> bool {
        if (!dbg_mechs || !dbg_mechs[0]) return true;
        return std::strstr(dbg_mechs, name) != nullptr;
    };
    auto dbg_time_match = [&](double tval) -> bool {
        if (dbg_t_start >= 0.0 && tval < dbg_t_start) return false;
        if (dbg_t_end >= 0.0 && tval > dbg_t_end) return false;
        return true;
    };
    static FILE* dbg_fp = nullptr;
    if (dbg_enabled && !dbg_fp) {
        const char* path = std::getenv("HELIOX_DEBUG_LOG_PATH");
        dbg_fp = (path && path[0]) ? std::fopen(path, "a") : stderr;
    }

    for (auto p_mech : p_neuron->mech_current_list) {
        if (dbg_enabled && dbg_fp && dbg_time_match(t) && dbg_mech_match(p_mech->name.c_str())) {
            double v_before = vec_v[dbg_node];
            double rhs_before = vec_rhs[dbg_node];
            double d_before = vec_d[dbg_node];
            std::fprintf(dbg_fp,
                         "DEBUG current pre t=%.6f mech=%s node=%d v=%.17g rhs=%.17g d=%.17g\n",
                         t, p_mech->name.c_str(), dbg_node, v_before, rhs_before, d_before);
        }
        p_mech->current_cpu(params);
        if (dbg_enabled && dbg_fp && dbg_time_match(t) && dbg_mech_match(p_mech->name.c_str())) {
            double v_after = vec_v[dbg_node];
            double rhs_after = vec_rhs[dbg_node];
            double d_after = vec_d[dbg_node];
            std::fprintf(dbg_fp,
                         "DEBUG current post t=%.6f mech=%s node=%d v=%.17g rhs=%.17g d=%.17g\n",
                         t, p_mech->name.c_str(), dbg_node, v_after, rhs_after, d_after);
        }
    }

    // fast_imem: save membrane-only RHS contribution (before axial terms).
    if (p_neuron->need_fast_imem) {
        for (int i = 0; i < len; ++i) {
            vec_sav_rhs[i] = -vec_rhs[i];
        }
    }

    // LHS (membrane-only): capacitance contribution (must happen after any possible cm changes by mechanisms).
    // Note: at this point vec_d contains only the membrane Jacobian from mechanisms; cap_jacob adds the capacitive part.
    p_neuron->mech_cap->cap_jacob_cpu(p_neuron->cj, vec_d);

    // fast_imem: save membrane-only diagonal contribution (after capacitance, before axial terms).
    if (p_neuron->need_fast_imem) {
        std::copy(vec_d, vec_d + len, vec_sav_d);
    }

    for (int i = ncell; i < len; i++)                    //前ncell个是根节点，没有父节点
    {
        double dv = vec_v[parent_index[i]] - vec_v[i];
        vec_rhs[i] -= vec_b[i] * dv;
        vec_rhs[parent_index[i]] += vec_a[i] * dv;
    }

    
    /*lhs part*/
    for (int i = ncell; i < len; i++)
    {
        vec_d[i] -= vec_b[i];
        vec_d[parent_index[i]] -= vec_a[i];
    }
}

void solve_serial(double* vec_a, double* vec_b, double* vec_d,
                  double* vec_rhs, int* parent_index, int ncell, 
                  int len)
{
    double p;
    //triang
    for (int i = len - 1; i >= ncell; i--)
    {
        p = vec_a[i] / vec_d[i];
        vec_d[parent_index[i]] -= p * vec_b[i];
        vec_rhs[parent_index[i]] -= p * vec_rhs[i];
    }

    //bksub//Note:这些都是根节点，因此没有parent_index
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

void cpu_solve_permute1(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, 
                        int* parent_index, int nstride, int* stride, int* firstnode,
                        int* lastnode, int* cellsize, int ncell, int len)
{
    /*
        这些是permute状态下，要传入的变量
        int nstride = p_neuron->nstride;
        int* stride = p_neuron->vecdata_stride->get_cpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_cpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_cpu_data();
        int* cellsize = p_neuron->vecdata_cellsize->get_cpu_data();
        */
    int tid;
    int i, icellsize;
    int istride, ip;
    double p;
    for (tid = 0; tid < ncell; tid++)//分成若干个小cell，就可以并行化了，在GPU那边是把这个for循环给并行了
    {
        /*
            处理firstnode[tid]到lastnode[tid]之间的方程
            同时，他是跳着来处理的，例如，1，3，6这几个单独处理，所以有一个stride来记录间隔
        */
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

void cpu_solve_permute3(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, 
                        int* parent_index, int* max_order_each_thread, int* min_order_each_thread,
                        int* firstnode, int* lastnode, int* stride, int* map_t2c, 
                        int norder, int ncell, int nthread)
{
    /*
     * CPU serial version of GPU permute3 cop_solve_kernel
     * Processes all threads sequentially to maintain same data layout compatibility
     */
    
    // Triangle phase - process in reverse order (norder down to 0)
    for (int iorder = norder; iorder >= 0; iorder--) {
        for (int tid = 0; tid < nthread; tid++) {
            const int max_order = max_order_each_thread[tid];
            const int min_order = min_order_each_thread[tid];
            
            if (iorder >= min_order && iorder <= max_order) {
                int i = lastnode[tid];
                const int offset = (tid >> 5) * (norder + 1) - 1;
                
                // Navigate to the correct node for this order
                for (int order_step = norder; order_step > iorder; order_step--) {
                    if (order_step >= min_order && order_step <= max_order && i > -1) {
                        i -= stride[offset + order_step];
                    }
                }
                
                if (i > -1) {
                    const int ip = parent_index[i];
                    const double a_val = vec_a[i];
                    const double d_val = vec_d[i];
                    const double b_val = vec_b[i];
                    const double rhs_val = vec_rhs[i];
                    
                    const double p = a_val / d_val;
                    
                    // Update parent node (no atomic operations needed in serial)
                    vec_d[ip] -= p * b_val;
                    vec_rhs[ip] -= p * rhs_val;
                }
            }
        }
    }
    
    // Backsubstitution phase - process root cells first
    for (int tid = 0; tid < nthread; tid++) {
        const int icell = map_t2c[tid];
        if (icell > -1) {
            vec_rhs[icell] /= vec_d[icell];
        }
    }
    
    // Backsubstitution phase - process remaining nodes in forward order
    for (int iorder = 1; iorder <= norder; iorder++) {
        for (int tid = 0; tid < nthread; tid++) {
            const int max_order = max_order_each_thread[tid];
            const int min_order = min_order_each_thread[tid];
            
            if (iorder >= min_order && iorder <= max_order) {
                int i = firstnode[tid];
                const int offset = (tid >> 5) * (norder + 1);
                
                // Navigate to the correct node for this order
                for (int order_step = 1; order_step < iorder; order_step++) {
                    if (order_step >= min_order && order_step <= max_order && i > -1) {
                        i += stride[offset + order_step];
                    }
                }
                
                if (i > -1) {
                    const int ip = parent_index[i];
                    const double b_val = vec_b[i];
                    const double rhs_parent = vec_rhs[ip];
                    const double d_val = vec_d[i];
                    const double rhs_val = vec_rhs[i];
                    
                    const double p = rhs_val - b_val * rhs_parent;
                    vec_rhs[i] = p / d_val;
                }
            }
        }
    }
}

/*
 * solve the cable equation
 */
void Simulate::solve_matrix_cpu(HelioXroupData* p_neuron)
{
    int len = p_neuron->len;
    int ncell = p_neuron->ncell;
    double* vec_a = p_neuron->vecdata_a->get_cpu_data();
    double* vec_b = p_neuron->vecdata_b->get_cpu_data();
    double* vec_d = p_neuron->vecdata_d->get_cpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();
    int* parent_index = p_neuron->vecdata_parent_index->get_cpu_data();

    //solve_serial(vec_a, vec_b, vec_d, vec_rhs, parent_index, ncell, len);
    if (permute_type == 0)
    {
        solve_serial(vec_a, vec_b, vec_d, vec_rhs, parent_index, ncell, len);
    }
    else if (permute_type == 1)
    {
        int nstride = p_neuron->nstride;
        int* stride = p_neuron->vecdata_stride->get_cpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_cpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_cpu_data();
        int* cellsize = p_neuron->vecdata_cellsize->get_cpu_data();
        cpu_solve_permute1(vec_a, vec_b, vec_d, vec_rhs, parent_index, nstride, 
                           stride, firstnode, lastnode, cellsize, ncell, len);
    }
    else if (permute_type == 3)
    {
        int norder = p_neuron->norder;
        int nthread = p_neuron->threads_num;
        int* stride = p_neuron->vecdata_stride->get_cpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_cpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_cpu_data();
        int* max_order_each_thread = p_neuron->vecdata_max_order_each_thread->get_cpu_data();
        int* min_order_each_thread = p_neuron->vecdata_min_order_each_thread->get_cpu_data();
        int* map_t2c = p_neuron->vecdata_map_t2c->get_cpu_data();
        cpu_solve_permute3(vec_a, vec_b, vec_d, vec_rhs, parent_index, max_order_each_thread, 
                          min_order_each_thread, firstnode, lastnode, stride, map_t2c, norder, 
                          ncell, nthread);
    }
}

void Simulate::update_cpu(HelioXroupData* p_group)
{
    int len = p_group->len;
    double* vec_v = p_group->vecdata_v->get_cpu_data();
    double* vec_rhs = p_group->vecdata_rhs->get_cpu_data();

    for (int i = 0; i < len; i++)
    {
        vec_v[i] += vec_rhs[i];
    }
    p_group->mech_cap->cap_current_cpu(p_group->cj, vec_rhs);
}

void Simulate::last_part_cpu()
{
    t += 0.5 * dt;

    int ncell = neuron_group_list.size();
    for (auto p_neuron : neuron_group_list)
    {
        p_neuron->vec_play_continuous.continuous_cpu(t);
        nonvint_cpu(p_neuron);
    }
}

void Simulate::nonvint_cpu(HelioXroupData* p_neuron)
{
    SimMechStateParam params = {p_neuron->vecdata_v->get_cpu_data(), dt, t};

    double* vec_v = p_neuron->vecdata_v->get_cpu_data();
    int len = p_neuron->len;
    const char* dbg_env = std::getenv("HELIOX_DEBUG_STEP_LOG");
    bool dbg_enabled = dbg_env && dbg_env[0] != '0';
    int dbg_node = 0;
    if (const char* node_env = std::getenv("HELIOX_DEBUG_NODE_INDEX")) {
        dbg_node = std::atoi(node_env);
        if (dbg_node < 0) dbg_node = 0;
        if (dbg_node >= len) dbg_node = len - 1;
    }
    double dbg_t_start = -1.0;
    double dbg_t_end = -1.0;
    if (const char* t_env = std::getenv("HELIOX_DEBUG_T_START")) {
        dbg_t_start = std::atof(t_env);
    }
    if (const char* t_env = std::getenv("HELIOX_DEBUG_T_END")) {
        dbg_t_end = std::atof(t_env);
    }
    const char* dbg_mechs = std::getenv("HELIOX_DEBUG_MECHS");
    auto dbg_mech_match = [&](const char* name) -> bool {
        if (!dbg_mechs || !dbg_mechs[0]) return true;
        return std::strstr(dbg_mechs, name) != nullptr;
    };
    auto dbg_time_match = [&](double tval) -> bool {
        if (dbg_t_start >= 0.0 && tval < dbg_t_start) return false;
        if (dbg_t_end >= 0.0 && tval > dbg_t_end) return false;
        return true;
    };
    static FILE* dbg_fp = nullptr;
    if (dbg_enabled && !dbg_fp) {
        const char* path = std::getenv("HELIOX_DEBUG_LOG_PATH");
        dbg_fp = (path && path[0]) ? std::fopen(path, "a") : stderr;
    }

    for (auto p_mech : p_neuron->mech_write_state_ion_list) {
        if (dbg_enabled && dbg_fp && dbg_time_match(t) && dbg_mech_match(p_mech->name.c_str())) {
            std::fprintf(dbg_fp,
                         "DEBUG state pre t=%.6f mech=%s node=%d v=%.17g\n",
                         t, p_mech->name.c_str(), dbg_node, vec_v[dbg_node]);
        }
        p_mech->state_cpu(params);
        if (dbg_enabled && dbg_fp && dbg_time_match(t) && dbg_mech_match(p_mech->name.c_str())) {
            std::fprintf(dbg_fp,
                         "DEBUG state post t=%.6f mech=%s node=%d v=%.17g\n",
                         t, p_mech->name.c_str(), dbg_node, vec_v[dbg_node]);
        }
    }
    for (auto p_mech : p_neuron->mechanism_list) {
        if (!p_mech->write_state_ion) {
            if (dbg_enabled && dbg_fp && dbg_time_match(t) && dbg_mech_match(p_mech->name.c_str())) {
                std::fprintf(dbg_fp,
                             "DEBUG state pre t=%.6f mech=%s node=%d v=%.17g\n",
                             t, p_mech->name.c_str(), dbg_node, vec_v[dbg_node]);
            }
            p_mech->state_cpu(params);
            if (dbg_enabled && dbg_fp && dbg_time_match(t) && dbg_mech_match(p_mech->name.c_str())) {
                std::fprintf(dbg_fp,
                             "DEBUG state post t=%.6f mech=%s node=%d v=%.17g\n",
                             t, p_mech->name.c_str(), dbg_node, vec_v[dbg_node]);
            }
        }
    }
    
    hdf5_manager.log_data_cpu();
}

/*
 * for all synapse mechanism, call pre_spike_send() to 
 * put fired spikes into spike buffers
 */
void Simulate::network_spike_send_cpu()
{
    int ngroup = neuron_group_list.size();
    for (int i = 0; i < ngroup; i++)
    {
        HelioXroupData* p_neuron = neuron_group_list[i];
        auto &ps = p_neuron->presyn;
        SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
        double* vec_v = p_neuron->vecdata_v->get_cpu_data();
        ps->threshold_detect_cpu(vec_v, spk_flags, t, rec_spikes);
    }
}

void Simulate::record_output_spikes_cpu(HelioXroupData* p_neuron)
{
    SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
    size_t buffer_size = p_neuron->spk_vec->v.size();
    for (int i = 0; i < buffer_size; i++)
    {
        int gid = p_neuron->spk_vec->v[i].gid;
        if (spk_flags[i] == SpikeFlag::NORMAL_EVENT && gid >= 0)
        {
            this->rec_spikes.emplace_back(t, gid);
        }
    }
}

bool comp(const pair<double, uint32_t> &p1, const pair<double, uint32_t> &p2)
{
    if (p1.first == p2.first)
        return p1.second < p2.second;
    return p1.first < p2.first;
}

void Simulate::output_spikes()
{
    if(!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    string outfile = output_folder + "/spk.dat";
    sort(rec_spikes.begin(), rec_spikes.end(), comp);    

    FILE *fp = fopen(outfile.c_str(), "w");
    for (int i = 0; i < rec_spikes.size(); i++)
    {
        if (rec_spikes[i].second > -1)
        {
            fprintf(fp, "%.8g\t%d\n", rec_spikes[i].first, rec_spikes[i].second);
        }
    }
    fclose(fp);
}

/*
 * for all synapse mechanism, call post_spike_receive() to 
 * deal with fired spikes, if firetime + delay <= t, the 
 * NET_RECEIVE block in .mod file should be called 
 */
void Simulate::network_spike_receive_cpu()
{
    bool hasSent = false;
    do {
        hasSent = false;
        for (HelioXroupData* p_neuron : neuron_group_list)
        {
            SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
            
            for(auto postsyn : p_neuron->vec_postsyn)
            {
                postsyn->get_spike_from_vec_cpu(p_neuron->spk_vec, spk_flags, t);
            }
            //因为有可能多个postsyn连到同一个pre-syn上，因此，需要把所有的postsyn遍历完，再恢复
            clearValidSpkFlags(p_neuron->vecdata_spk_flags);

            for(auto postsyn : p_neuron->vec_postsyn)
            {
                postsyn->post_spike_receive_cpu(t + dt / 2);
                hasSent |= postsyn->net_receive_cpu(t);
            }
        }
    }while(hasSent);
}



//clearAll是在finitialize的时候调用的，把所有的都清空
void Simulate::clearValidSpkFlags(VecData<SpikeFlag> *vecdata_spk_flags, bool cleanAll){
    auto cpu_vec_spk_flags = vecdata_spk_flags->get_cpu_data();
    int cpu_vec_spk_len = vecdata_spk_flags->size();
    for(int i = 0;i<cpu_vec_spk_len;i++){
        if(cpu_vec_spk_flags[i] == SpikeFlag::NORMAL_EVENT || cleanAll)
            cpu_vec_spk_flags[i] = SpikeFlag::INVALID; //清空spike标志
    }
}

void Simulate::gap_transfer_cpu(HelioXroupData *p_group){
    auto &gap_info = p_group->cpu_gap_trans_info;
    int ntrans = gap_info.ntrans();
    if (ntrans == 0) {
        return;
    }
    
    double** src = gap_info.src.get_cpu_data();
    double** dst = gap_info.dst.get_cpu_data();
    
    for(int i = 0; i < ntrans; i++){
        *(dst[i]) = *(src[i]);
    }
}

void Simulate::clearAllSpikes_cpu(){
    for(auto p_neuron : neuron_group_list){
        auto vec_spk = p_neuron->spk_vec;
        vec_spk->clean();

        auto vec_spk_flags = p_neuron->vecdata_spk_flags;
        clearValidSpkFlags(vec_spk_flags,true);
    }
}

void Simulate::clearAllSpikes_gpu(){
    clearAllSpikes_cpu();
    for(auto p_neuron : neuron_group_list){
        p_neuron->vecdata_spk_flags->update_gpu_data_from_cpu();
    }
}

// void Simulate::vecevent_play(int mech_idx, int len,const double *data_arr){
//     assert(neuron_group_list.size() == 1);
//     auto p_neuron = neuron_group_list[0];
//     auto p_vecevent = p_neuron->mech_vecevent;
//     if(p_vecevent == nullptr){
//         printf("vecevent is not initialized\n");
//         assert(false);
//     }
//     VecEvent::play(p_vecevent, this->mode, mech_idx, len, data_arr);
// }

// Gap Junction 管理实现

int Simulate::add_gap_source(const string& mech, const string& var, int idx, int sid) {
    // 自动分配sid
    if (sid < 0) {
        sid = next_sid_counter++;
        // 确保不冲突
        while (gap_junctions.find(sid) != gap_junctions.end()) {
            sid = next_sid_counter++;
        }
    } else {
        // 用户指定的sid，检查冲突
        if (gap_junctions.find(sid) != gap_junctions.end()) {
            printf("Gap junction with sid %d already exists\n", sid);
            return -1;
        }
        // 更新counter，保持它大于所有已使用的sid
        if (sid >= next_sid_counter) {
            next_sid_counter = sid + 1;
        }
    }
    
    // 创建Gap Junction元数据
    GapJunctionMeta meta;
    meta.source = VarDescriptor(mech, var, idx);
    gap_junctions[sid] = meta;
    
    return sid;
}

int Simulate::add_gap_target(int sid, const string& mech, const string& var, int idx) {
    // 检查源是否存在
    auto it = gap_junctions.find(sid);
    if (it == gap_junctions.end()) {
        printf("Gap junction with sid %d does not exist\n", sid);
        return -1;
    }
    
    if (neuron_group_list.empty()) {
        printf("add_gap_target: neuron_group_list is empty\n");
        return -1;
    }
    
    // 创建目标描述符
    VarDescriptor target(mech, var, idx);
    
    // 获取源和目标的指针
    VarDescriptor source_copy = it->second.source;  // 创建副本因为getVarPtr需要非const引用
    auto [src_cpu, src_gpu] = getVarPtr(source_copy, false);
    auto [tgt_cpu, tgt_gpu] = getVarPtr(target, false);
    
    if (src_cpu == nullptr || tgt_cpu == nullptr) {
        printf("Failed to resolve variable pointers\n");
        return -1;
    }
    
    // 添加到元数据
    it->second.targets.push_back(target);
    
    // 直接操作HelioXroupData的gap info
    auto p_group = neuron_group_list[0];
    
    // 使用封装的add_gap方法，让GapTransInfo自己处理细节
    if (mode == CPU) {
        p_group->cpu_gap_trans_info.add_gap(src_cpu, tgt_cpu);
    } else {
        // GPU模式：两个GapTransInfo都需要添加
        p_group->cpu_gap_trans_info.add_gap(src_cpu, tgt_cpu);
        p_group->gpu_gap_trans_info.add_gap(src_gpu, tgt_gpu);
    }
    
    // 更新状态
    p_group->have_gap = true;
    
    return 0;
}

int Simulate::clear_all_gap_junctions() {
    if (neuron_group_list.empty()) {
        printf("clear_all_gap_junctions: neuron_group_list is empty\n");
        return -1;
    }
    
    // 清空元数据
    gap_junctions.clear();
    next_sid_counter = 0;
    
    // 清空VecData
    auto p_group = neuron_group_list[0];
    p_group->cpu_gap_trans_info.clear();
    if (mode == GPU) {
        p_group->gpu_gap_trans_info.clear();
    }
    p_group->have_gap = false;
    
    return 0;
}

Simulate::GapJunctionMeta* Simulate::get_gap_junction(int sid) {
    auto it = gap_junctions.find(sid);
    if (it == gap_junctions.end()) {
        return nullptr;
    }
    return &it->second;
}

int Simulate::create_optimizer(OptimizerType type) {
    std::unique_ptr<OptimizerBase> optimizer;
    switch (type) {
        case OptimizerType::SGD:
            optimizer = std::make_unique<SGDOptimizer>(mode);
            break;
        case OptimizerType::Momentum:
            optimizer = std::make_unique<SGDMomentumOptimizer>(mode);
            break;
        case OptimizerType::Adam:
            optimizer = std::make_unique<AdamOptimizer>(mode);
            break;
        default:
            printf("Unsupported optimizer type\n");
            return -1;
    }
    int optimizer_id = next_optimizer_id++;
    optimizers.emplace(optimizer_id, std::move(optimizer));
    return optimizer_id;
}

int Simulate::register_optimizer_param(int optimizer_id,
                                       double* weight_cpu,
                                       double* grad_cpu,
                                       double* weight_gpu,
                                       double* grad_gpu,
                                       double impedance) {
    std::vector<double*> weight_cpu_vec;
    std::vector<double*> grad_cpu_vec;
    std::vector<double*> weight_gpu_vec;
    std::vector<double*> grad_gpu_vec;

    weight_cpu_vec.push_back(weight_cpu);
    grad_cpu_vec.push_back(grad_cpu);
    if (mode == GPU) {
        weight_gpu_vec.push_back(weight_gpu);
        grad_gpu_vec.push_back(grad_gpu);
    }

    return register_optimizer_param_batch(optimizer_id,
                                          weight_cpu_vec,
                                          grad_cpu_vec,
                                          weight_gpu_vec,
                                          grad_gpu_vec,
                                          impedance);
}

int Simulate::register_optimizer_param_batch(int optimizer_id,
                                             const std::vector<double*>& weight_cpu,
                                             const std::vector<double*>& grad_cpu,
                                             const std::vector<double*>& weight_gpu,
                                             const std::vector<double*>& grad_gpu,
                                             double impedance) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("register_optimizer_param: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    if (weight_cpu.empty()) {
        printf("register_optimizer_param_batch: empty weight list\n");
        return -1;
    }
    if (weight_cpu.size() != grad_cpu.size()) {
        printf("register_optimizer_param_batch: CPU pointer count mismatch (weight=%zu, grad=%zu)\n",
               weight_cpu.size(), grad_cpu.size());
        return -1;
    }
    if (mode == GPU) {
        if (weight_gpu.size() != weight_cpu.size() || grad_gpu.size() != weight_cpu.size()) {
            printf("register_optimizer_param_batch: GPU pointer count mismatch (batch=%zu)\n",
                   weight_cpu.size());
            return -1;
        }
    }

    OptimizerParam param;
    param.weight_cpu = weight_cpu;
    param.grad_cpu = grad_cpu;
    if (mode == GPU) {
        param.weight_gpu = weight_gpu;
        param.grad_gpu = grad_gpu;
    }
    param.impedance = impedance;
    return it->second->add_param(param);
}

int Simulate::configure_optimizer(int optimizer_id, const OptimizerHyperParams& params) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("configure_optimizer: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    it->second->configure(params);
    return 0;
}

int Simulate::optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt_step) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_step: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    if (record_time <= 0.0 || dt_step <= 0.0) {
        printf("optimizer_step: invalid record_time (%f) or dt (%f)\n", record_time, dt_step);
        return -1;
    }
    double inv_record_steps = dt_step / record_time;
    it->second->step(learning_rate, inv_record_steps);
    return 0;
}

int Simulate::optimizer_step_with_inv_record_steps(int optimizer_id,
                                                   double learning_rate,
                                                   double inv_record_steps) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_step_with_inv_record_steps: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    if (!(inv_record_steps > 0.0)) {
        printf("optimizer_step_with_inv_record_steps: invalid inv_record_steps (%f)\n", inv_record_steps);
        return -1;
    }
    it->second->step(learning_rate, inv_record_steps);
    return 0;
}

int Simulate::optimizer_reset_state(int optimizer_id) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_reset_state: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    it->second->reset_state();
    return 0;
}

int Simulate::optimizer_get_adam_state(int optimizer_id,
                                      long long& step_count,
                                      std::vector<double>& m,
                                      std::vector<double>& v,
                                      OptimizerHyperParams& params) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_get_adam_state: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    AdamOptimizer* adam = dynamic_cast<AdamOptimizer*>(it->second.get());
    if (adam == nullptr) {
        printf("optimizer_get_adam_state: optimizer %d is not Adam\n", optimizer_id);
        return -1;
    }
    params = it->second->hyper_params();
    adam->export_state(step_count, m, v);
    return 0;
}

int Simulate::optimizer_set_adam_state(int optimizer_id,
                                      long long step_count,
                                      const std::vector<double>& m,
                                      const std::vector<double>& v,
                                      const OptimizerHyperParams& params) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_set_adam_state: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    AdamOptimizer* adam = dynamic_cast<AdamOptimizer*>(it->second.get());
    if (adam == nullptr) {
        printf("optimizer_set_adam_state: optimizer %d is not Adam\n", optimizer_id);
        return -1;
    }
    it->second->configure(params);
    if (adam->import_state(step_count, std::span<const double>(m.data(), m.size()), std::span<const double>(v.data(), v.size())) < 0) {
        printf("optimizer_set_adam_state: import_state failed\n");
        return -1;
    }
    return 0;
}
