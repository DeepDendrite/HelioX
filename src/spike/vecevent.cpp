#include <iostream>
#include "vecevent.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

using namespace std;

REGISTER_MECHANISM("VecStim", VecEvent);
static bool VecEvent_play_registered = []() {
    FunctionRegistry::getInstance().register_func("VecStim", "play", std::function<void(Mode, int, vector<double>)>(&VecEvent::play));
    return true;
}();
static bool VecEvent_static_interface_registered = []() {
    StaticInterfaceRegistry::getInstance().register_property<std::string, std::string>(
        "VecStim",
        "mode",
        std::function<std::string()>([]() { return VecEvent::getPlayUpdateModeName(); }),
        std::function<void(std::string)>([](std::string mode) { VecEvent::setPlayUpdateModeByName(mode); })
    );
    return true;
}();

// 静态实例指针定义
VecEvent* VecEvent::instance = nullptr;
// 当前实现默认只有一个 sim 实例在使用 VecStim，因此这里仍保留全局静态状态。
// 如果以后要支持多个 sim 实例并存，这里的 current_time_/play_update_mode_
// 都需要改造成 per-sim 状态，否则不同实例之间会互相污染。
double VecEvent::current_time_ = 0.0;
VecEvent::PlayUpdateMode VecEvent::play_update_mode_ = VecEvent::PlayUpdateMode::Enhanced;

VecEvent::VecEvent(MechInitParams &param) : ArtiCell(param)
{
    vecdata_etime = nullptr;
    vecdata_index = nullptr;
    printf_debug("VecEvent Param: node_count: %d\n", param.node_count);
    
    // 设置为单例实例
    setInstance(this);
}

VecEvent::~VecEvent()
{
    if (vecdata_etime)
    {
        delete vecdata_etime;
        vecdata_etime = nullptr;    
    }
    if (vecdata_index)
    {
        delete vecdata_index;
        vecdata_index = nullptr;
    }
    for (int i = 0, len = vec_ptr.size(); i < len; i++)
    {
        if (vec_ptr[i])
        {
            delete vec_ptr[i];
            vec_ptr[i] = nullptr;
        }
    }
    
    // 清理单例实例指针
    if (instance == this) {
        instance = nullptr;
    }
}

// 单例模式实现
VecEvent* VecEvent::getInstance() {
    return instance;
}

void VecEvent::setInstance(VecEvent* inst) {
    instance = inst;
}

void VecEvent::destroyInstance() {
    if (instance) {
        delete instance;
        instance = nullptr;
    }
}

void VecEvent::setCurrentTime(double t) {
    current_time_ = t;
}

void VecEvent::setPlayUpdateModeByName(const std::string& mode) {
    if (mode == "legacy") {
        play_update_mode_ = PlayUpdateMode::Legacy;
        return;
    }
    if (mode == "enhanced") {
        play_update_mode_ = PlayUpdateMode::Enhanced;
        return;
    }
    throw std::runtime_error("VecStim.mode must be 'legacy' or 'enhanced', got: " + mode);
}

std::string VecEvent::getPlayUpdateModeName() {
    switch (play_update_mode_) {
        case PlayUpdateMode::Legacy:
            return "legacy";
        case PlayUpdateMode::Enhanced:
            return "enhanced";
    }
    return "legacy";
}

VecEvent::PlayUpdateMode VecEvent::getPlayUpdateMode() {
    return play_update_mode_;
}

void VecEvent::reg_node_indices(MechInitParams &param)
{
    auto node_count = param.node_count;
    nnode = node_count;
    if (nnode > 0)
    {
        printf_debug("VecEvent::reg_node_indices: nnode: %d\n", nnode);
        vecdata_weights = new VecData<double>(mode, 0.0, node_count);
        vecdata_delay = new VecData<double>(mode, 0.0, node_count);
        vecdata_delay_steps = new VecData<int>(mode, 0, node_count);
        vecdata_spk_vec_idx = new VecData<uint32_t>(mode, (uint32_t)0, node_count);
        vecdata_spike_flag = new VecData<SpikeFlag>(mode, SpikeFlag::INVALID, node_count);
        vecdata_receive_idx_vec = new VecData<uint32_t>(mode, (uint32_t)0, node_count);
        this->vecdata_index = new VecData<int>(mode, node_count);
        this->vecdata_etime = new VecData<double>(mode, node_count);
        this->vecdata_receive_idx_vec = new VecData<uint32_t>(mode, node_count);
    }

    // Ensure VecStim has a stable slot per instance even when bbcore export
    // does not provide VecStim vector data (icnt=0). This is required for the
    // "export-empty + inject spike times on NEURONG side" workflow.
    if (vec_ptr.size() < static_cast<size_t>(node_count)) {
        vec_ptr.resize(static_cast<size_t>(node_count), nullptr);
    }
    
    //if (node_count == 0)
    //{
    //    this->vecdata_receive_idx_vec = new VecData<int>(mode);
    //}
}

void VecEvent::initialize_cpu(SimMechInitialParam &param)
{
    int* index = this->vecdata_index->get_cpu_data();
    double* etime = this->vecdata_etime->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++)
    {
        index[inode] = 0;
        element_cpu(inode);
        if (index[inode] > 0)
        {
            net_send_cpu(inode, etime[inode], SpikeFlag::SELF_EVENT);
        }
    }
}

void VecEvent::initialize_gpu(SimMechInitialParam &param)
{
    int* index = this->vecdata_index->get_cpu_data();
    double* etime = this->vecdata_etime->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++)
    {
        index[inode] = 0;
        element_cpu(inode);
        if (index[inode] > 0)
        {
            net_send_cpu(inode, etime[inode], SpikeFlag::SELF_EVENT);
        }
    }
}

void VecEvent::element_cpu(int inode) 
{
    int* index = this->vecdata_index->get_cpu_data();
    double* etime = this->vecdata_etime->get_cpu_data();
    int i = index[inode];
    if (i >= 0)
    {
        if(inode >= vec_ptr.size())
        {
            cout << "inode: " << inode << " vec_ptr.size(): " << vec_ptr.size() << endl;
        }
        assert( inode< vec_ptr.size());
        VecData<double>* vv = this->vec_ptr[inode];
        if (vv)
        {
            int len = vv->size();
            double* px = vv->get_cpu_data();
            if (i < len)
            {
                etime[inode] = px[i];
                index[inode] += 1;
            }
            else
            {
                index[inode] = -1;
            }
        }
        else
        {
            index[inode] = -1;
        }

    }
}

void VecEvent::clear_pending_events_for_instance(int mech_idx)
{
    SpikeBuffer filtered;
    while (!spike_buffer.empty()) {
        Spike spk = spike_buffer.top();
        spike_buffer.pop();
        if (spk.syn_id == mech_idx && spk.syn_type == this->type) {
            continue;
        }
        filtered.push(spk);
    }
    spike_buffer = std::move(filtered);
}

void VecEvent::rearm_sequence(int mech_idx, double current_t)
{
    assert(mech_idx >= 0 && mech_idx < nnode);

    int* index = this->vecdata_index->get_cpu_data();
    double* etime = this->vecdata_etime->get_cpu_data();

    clear_pending_events_for_instance(mech_idx);

    VecData<double>* vv = this->vec_ptr[mech_idx];
    if (vv == nullptr || vv->size() <= 0) {
        index[mech_idx] = -1;
        etime[mech_idx] = 0.0;
        return;
    }

    const int len = vv->size();
    double* px = vv->get_cpu_data();
    int next_idx = 0;
    while (next_idx < len && px[next_idx] < current_t) {
        next_idx++;
    }

    if (next_idx >= len) {
        index[mech_idx] = -1;
        etime[mech_idx] = 0.0;
        return;
    }

    etime[mech_idx] = px[next_idx];
    index[mech_idx] = next_idx + 1;
    net_send_cpu(mech_idx, etime[mech_idx], SpikeFlag::SELF_EVENT);
}

bool VecEvent::net_receive_cpu(double t)
{
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    double* weight = this->vecdata_weights->get_cpu_data();
    SpikeFlag* spike_flag = this->vecdata_spike_flag->get_cpu_data();
    int* index = this->vecdata_index->get_cpu_data();
    double* etime = this->vecdata_etime->get_cpu_data();
    int _w;
    SpikeFlag _flag;
    int send_count = 0;
    for (int i = 0; i < receive_count; i++)
    {
        int isyn = receive_idx_vec[i];
        _w = weight[isyn];
        _flag = spike_flag[i];
        if (_flag == SpikeFlag::SELF_EVENT)
        {
            net_send_cpu(isyn, t, SpikeFlag::NORMAL_EVENT);
            element_cpu(isyn);
            if (index[isyn] > 0)
            {
                net_send_cpu(isyn, etime[isyn], SpikeFlag::SELF_EVENT);
            }
            send_count++;
        }
    }
    return send_count > 0;
}

bool VecEvent::net_receive_gpu(double t)
{
    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    double* weight = this->vecdata_weights->get_cpu_data();
    SpikeFlag* spike_flag = this->vecdata_spike_flag->get_cpu_data();
    int* index = this->vecdata_index->get_cpu_data();
    double* etime = this->vecdata_etime->get_cpu_data();
    int _w;
    SpikeFlag _flag;
    if(receive_count <= 0)
        return false;
    int send_count = 0;
    for (int i = 0; i < receive_count; i++)
    {
        int isyn = receive_idx_vec[i];
        _w = weight[isyn];
        _flag = spike_flag[i];
        if (_flag == SpikeFlag::SELF_EVENT)
        {
            net_send_cpu(isyn, t, SpikeFlag::NORMAL_EVENT);
            element_cpu(isyn);
            if (index[isyn] > 0)
            {
                net_send_cpu(isyn, etime[isyn], SpikeFlag::SELF_EVENT);
            }
            send_count++;
        }
    }
    spk_flags_bkp->update_gpu_data_from_cpu();
    return send_count > 0;
}

void VecEvent::bbcore_read(int icnt, int dcnt, int* iArray, double* dArray, Mode mod)
{
    // In the standard CoreNEURON export, VecStim provides:
    // - icnt: number of VecStim instances
    // - iArray[i]: length of spike time array for instance i
    // - dArray: concatenated spike times
    //
    // For some workflows we intentionally export VecStim with no vector payload
    // (icnt/dcnt may be 0) and inject spike times later from Python/CUDA.
    vec_ptr.clear();

    // Always keep at least nnode slots so initialize() can safely iterate.
    const int want_slots = std::max(icnt, nnode);
    if (want_slots > 0) {
        vec_ptr.resize(static_cast<size_t>(want_slots), nullptr);
    }

    if (icnt <= 0 || dcnt <= 0 || iArray == nullptr || dArray == nullptr) {
        printf_debug("VecEvent::bbcore_read: empty payload (icnt=%d dcnt=%d) slots=%zu\n",
                     icnt, dcnt, vec_ptr.size());
        return;
    }

    int offset = 0;
    for (int i = 0; i < icnt; i++) {
        const int dsize = iArray[i];
        if (dsize <= 0) {
            continue;
        }
        VecData<double>* ptr = new VecData<double>(mod, dArray + offset, dsize);
        vec_ptr[static_cast<size_t>(i)] = ptr;
        offset += dsize;
    }
    printf_debug("VecEvent::bbcore_read: slots=%zu offset=%d dcnt=%d\n",
                 vec_ptr.size(), offset, dcnt);
}

void VecEvent::current_cpu(SimMechCurrentParam &param)
{}

void VecEvent::current_gpu(SimMechCurrentParam &param)
{}


void VecEvent::state_cpu(SimMechStateParam &param)
{}

void VecEvent::state_gpu(SimMechStateParam &param)
{}

void VecEvent::read_data_from_coredat(MechInitParams &param)
{}

void VecEvent::sync_gpu(){
}





void VecEvent::play(Mode mode, int mech_idx, vector<double> data_arr){
    update_sequence(mode, mech_idx, data_arr.data(), static_cast<int>(data_arr.size()));
}

void VecEvent::update_sequence(Mode mode, int mech_idx, const double* data_ptr, int len) {
    VecEvent* mech = VecEvent::getInstance();
    if (mech == nullptr) {
        printf("VecEvent::update_sequence: instance not initialized\n");
        return;
    }
    if (mech_idx < 0 || static_cast<size_t>(mech_idx) >= mech->vec_ptr.size()) {
        printf("VecEvent::update_sequence: invalid mech_idx %d (size=%zu)\n",
               mech_idx,
               mech->vec_ptr.size());
        return;
    }
    if (len <= 0) {
        return;
    }
    if (data_ptr == nullptr) {
        printf("VecEvent::update_sequence: data_ptr is null for mech_idx %d\n", mech_idx);
        return;
    }

    VecData<double>* vecdata = mech->vec_ptr[mech_idx];
    if (vecdata == nullptr) {
        vecdata = new VecData<double>(mode, len);
        mech->vec_ptr[mech_idx] = vecdata;
    } else if (vecdata->capacity() < len) {
        delete vecdata;
        vecdata = new VecData<double>(mode, len);
        mech->vec_ptr[mech_idx] = vecdata;
    } else {
        vecdata->resize(len);
    }

    double* cpu_buf = vecdata->get_cpu_data();
    std::memcpy(cpu_buf, data_ptr, sizeof(double) * len);
    if (mode == Mode::GPU) {
        vecdata->update_gpu_data_from_cpu();
    }
    if (getPlayUpdateMode() == PlayUpdateMode::Enhanced) {
        mech->rearm_sequence(mech_idx, current_time_);
    }
}
