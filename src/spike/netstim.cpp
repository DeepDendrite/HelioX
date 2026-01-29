#include <iostream>
#include "netstim.h"
#include <cstdint>
#include "philox.h"
#include "nrnran123.h"
#include "magic_enum/magic_enum.hpp"

using namespace std;

NetStim::NetStim(MechInitParams &param) : ArtiCell(param)
{
    need_area = true;
    printf_debug("NetStim Param: node_count: %d\n", param.node_count);

}

NetStim::~NetStim()
{
    if (vecdata_interval)
    {
        delete vecdata_interval;
        vecdata_interval = nullptr;    
    }
    if (vecdata_number)
    {
        delete vecdata_number;
        vecdata_number = nullptr;
    }
    if (vecdata_start)
    {
        delete vecdata_start;
        vecdata_start = nullptr;
    }
    if (vecdata_noise)
    {
        delete vecdata_noise;
        vecdata_noise = nullptr;
    }
    if (vecdata_event)
    {
        delete vecdata_event;
        vecdata_event = nullptr;
    }
    if (vecdata_on)
    {
        delete vecdata_on;
        vecdata_on = nullptr;
    }
    if (vecdata_ispike)
    {
        delete vecdata_ispike;
        vecdata_ispike = nullptr;
    }
}

void NetStim::read_data_from_coredat(MechInitParams &param) {
    // printf("NetStim::read_data_from_coredat\n");
    auto param_size = param.data_size;
    auto data = param.data;
    double* interval = this->vecdata_interval->get_cpu_data();
    double* number = this->vecdata_number->get_cpu_data();
    double* start = this->vecdata_start->get_cpu_data();
    double* noise = this->vecdata_noise->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();
    double* on = this->vecdata_on->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++) {
        interval[inode] = data[inode * param_size + 0];
        // printf("interval[%d] = %f\n", inode, interval[inode]);
        number[inode] = data[inode * param_size + 1];
        // printf("number[%d] = %f\n", inode, number[inode]);
        start[inode] = data[inode * param_size + 2];
        // printf("start[%d] = %f\n", inode, start[inode]);
        noise[inode] = data[inode * param_size + 3];
        // printf("noise[%d] = %f\n", inode, noise[inode]);
        event[inode] = data[inode * param_size + 4];
        on[inode] = data[inode * param_size + 5];
        ispike[inode] = data[inode * param_size + 6];
    }
}

void NetStim::reg_node_indices(MechInitParams &param)
{
    auto node_count = param.node_count;
    nnode = node_count;
    if (nnode > 0)
    {
        // printf_debug("NetStim::reg_node_indices: nnode: %d\n", nnode);
        PostSyn::reg_node_indices(param);
        rng_state = new nrnran123_State();
        vecdata_interval = new VecData<double>(mode, 10.0, node_count);
        vecdata_number = new VecData<double>(mode, 10.0, node_count);
        vecdata_start = new VecData<double>(mode, 50.0, node_count);
        vecdata_noise = new VecData<double>(mode, 0.0, node_count);
        vecdata_event = new VecData<double>(mode, 0.0, node_count);
        vecdata_on = new VecData<double>(mode, 0.0, node_count); 
        vecdata_ispike = new VecData<double>(mode, 0.0, node_count);
    }
    if (mode == GPU) {
        vecdata_interval->update_gpu_data_from_cpu();
        vecdata_number->update_gpu_data_from_cpu();
        vecdata_start->update_gpu_data_from_cpu();
        vecdata_noise->update_gpu_data_from_cpu();
        vecdata_event->update_gpu_data_from_cpu();
        vecdata_on->update_gpu_data_from_cpu();
        vecdata_ispike->update_gpu_data_from_cpu();
    }
}

void NetStim::set_seed(int x) {
    nrnran123_setseq(rng_state, x);
}

double NetStim::erand() {
    return nrnran123_negexp(rng_state);
}

void NetStim::init_sequence() {
    double* number = this->vecdata_number->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    double* on = this->vecdata_on->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++) {
        if (number[inode] > 0.0) {
            event[inode] = 0.0;
            on[inode] = 1.0;
            ispike[inode] = 0.0;
        }
    }
}

double NetStim::invl(double interval, double noise) {
    if (interval <= 0.0)
    {
        interval = 0.01;
    }
    if (noise == 0)
    {
        return interval;
    }
    else
    {
        return (1.0 - noise) * interval + noise * interval * erand();
    }
}

void NetStim::next_invl() {
    double* number = this->vecdata_number->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    double* interval = this->vecdata_interval->get_cpu_data();
    double* noise = this->vecdata_noise->get_cpu_data();
    double* on = this->vecdata_on->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++) {
        if (number[inode] > 0) {
            event[inode] = invl(interval[inode], noise[inode]);
        }
        if (ispike[inode] >= number[inode]) {
            on[inode] = 0.0;
        }
    }
}

double* NetStim::getVarPtr(const VarDescriptor& descriptor, Mode mode) {
    int mech_idx = descriptor.node_or_mech_idx;
    if (mech_idx < 0 || mech_idx >= nnode) {
        printf("NetStim getVarPtr invalid mech_idx %d (nnode=%d)\n", mech_idx, nnode);
        return nullptr;
    }
    if (permute != nullptr) {
        mech_idx = permute[mech_idx];
    }

    VecData<double>* target = nullptr;
    const std::string& var_name = descriptor.var;
    if (var_name == "interval") {
        target = vecdata_interval;
    } else if (var_name == "number") {
        target = vecdata_number;
    } else if (var_name == "start") {
        target = vecdata_start;
    } else if (var_name == "noise") {
        target = vecdata_noise;
    } else if (var_name == "event") {
        target = vecdata_event;
    } else if (var_name == "on") {
        target = vecdata_on;
    } else if (var_name == "ispike") {
        target = vecdata_ispike;
    } else {
        return nullptr;
    }

    if (target == nullptr) {
        return nullptr;
    }

    if (mode == Mode::CPU) {
        double* base = target->get_cpu_data();
        return base ? base + mech_idx : nullptr;
    } else {
        double* base = target->get_gpu_data();
        return base ? base + mech_idx : nullptr;
    }
}

void NetStim::initialize_cpu(SimMechInitialParam &param)
{
    double* on = this->vecdata_on->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    double* noise = this->vecdata_noise->get_cpu_data();
    double* interval = this->vecdata_interval->get_cpu_data();
    double* start = this->vecdata_start->get_cpu_data();
    double* number = this->vecdata_number->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++)
    {
        on[inode] = 0.0;
        ispike[inode] = 0.0;
        if (noise[inode] < 0.0)
        {
            noise[inode] = 0.0;
        }
        if (noise[inode] > 1.0)
        {
            noise[inode] = 1.0;
        }
        if (start[inode] >= 0.0 && number[inode] > 0)
        {
            on[inode] = 1.0;
            event[inode] = start[inode] + invl(interval[inode], noise[inode]) - interval[inode] * (1.0 - noise[inode]);
            if (event[inode] < 0.0)
            {
                event[inode] = 0.0;
            }
            net_send_cpu(inode, event[inode], SpikeFlag::NETSTIM_INIT_EVENT);
        }
    }
}

void NetStim::initialize_gpu(SimMechInitialParam &param)
{
    // printf("NetStim::initialize_gpu\n");
    set_seed(0);
    double* on = this->vecdata_on->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    double* noise = this->vecdata_noise->get_cpu_data();
    double* interval = this->vecdata_interval->get_cpu_data();
    double* start = this->vecdata_start->get_cpu_data();
    double* number = this->vecdata_number->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();
    for (int inode = 0; inode < nnode; inode++)
    {
        on[inode] = 0.0;
        ispike[inode] = 0.0;
        if (noise[inode] < 0.0)
        {
            noise[inode] = 0.0;
        }
        if (noise[inode] > 1.0)
        {
            noise[inode] = 1.0;
        }
        if (start[inode] >= 0.0 && number[inode] > 0)
        {
            on[inode] = 1.0;
            event[inode] = start[inode] + invl(interval[inode], noise[inode]) - interval[inode] * (1.0 - noise[inode]);
            if (event[inode] < 0.0)
            {
                event[inode] = 0.0;
            }
            // printf("initialize_gpu %s, %d, %f\n", name.c_str(), inode, event[inode]);
            net_send_cpu(inode, event[inode], SpikeFlag::NETSTIM_INIT_EVENT);
        }
    }
}

bool NetStim::net_receive_cpu(double t)
{
    // printf("NetStim::net_receive_cpu\n");
    double* on = this->vecdata_on->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    double* noise = this->vecdata_noise->get_cpu_data();
    double* interval = this->vecdata_interval->get_cpu_data();
    double* start = this->vecdata_start->get_cpu_data();
    double* number = this->vecdata_number->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();

    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    double* weight = this->vecdata_weights->get_cpu_data();
    SpikeFlag* spike_flag = this->vecdata_spike_flag->get_cpu_data();
    int send_count = 0;
    for (int i = 0; i < receive_count; i++) {
        int isyn = receive_idx_vec[i];
        SpikeFlag flag = spike_flag[i];
        int w = weight[isyn];
        if (flag == SpikeFlag::NORMAL_EVENT) {
            if (w > 0 && on[isyn] == 0.0) {
                init_sequence();
                next_invl();
                event[isyn] -= interval[isyn] * (1. - noise[isyn]);
                net_send_cpu(isyn, t + event[isyn], SpikeFlag::SELF_EVENT);
                send_count++;
            } else if (w < 0) {
                on[isyn] = 0.0;
            }
        }
        if (flag == SpikeFlag::NETSTIM_INIT_EVENT) {
            if (on[isyn] == 1.0) {
                init_sequence();
                net_send_cpu(isyn, t + 0.0, SpikeFlag::SELF_EVENT);
                send_count++;
            }
        }
        if (flag == SpikeFlag::SELF_EVENT && on[isyn] == 1.0) {
            ispike[isyn] += 1.0;
            net_send_cpu(isyn, t, SpikeFlag::NORMAL_EVENT);
            next_invl();
            if (on[isyn] == 1.0) {
                net_send_cpu(isyn, t + event[isyn], SpikeFlag::SELF_EVENT);
                send_count++;
            }
        }
    }
    return send_count > 0;
}

bool NetStim::net_receive_gpu(double t)
{
    double* on = this->vecdata_on->get_cpu_data();
    double* ispike = this->vecdata_ispike->get_cpu_data();
    double* noise = this->vecdata_noise->get_cpu_data();
    double* interval = this->vecdata_interval->get_cpu_data();
    double* start = this->vecdata_start->get_cpu_data();
    double* number = this->vecdata_number->get_cpu_data();
    double* event = this->vecdata_event->get_cpu_data();

    uint32_t* receive_idx_vec = this->vecdata_receive_idx_vec->get_cpu_data();
    double* weight = this->vecdata_weights->get_cpu_data();
    SpikeFlag* spike_flag = this->vecdata_spike_flag->get_cpu_data();
    int send_count = 0;
    for (int i = 0; i < receive_count; i++) {
        int isyn = receive_idx_vec[i];
        SpikeFlag flag = spike_flag[i];
        int w = weight[isyn];
        if (flag == SpikeFlag::NORMAL_EVENT) {
            if (w > 0 && on[isyn] == 0.0) {
                init_sequence();
                next_invl();
                event[isyn] -= interval[isyn] * (1. - noise[isyn]);
                net_send_cpu(isyn, t + event[isyn], SpikeFlag::SELF_EVENT);
                send_count++;
            } else if (w < 0) {
                on[isyn] = 0.0;
            }
        }
        if (flag == SpikeFlag::NETSTIM_INIT_EVENT) {
            if (on[isyn] == 1.0) {
                init_sequence();
                net_send_cpu(isyn, t + 0.0, SpikeFlag::SELF_EVENT);
                send_count++;
            }
        }
        if (flag == SpikeFlag::SELF_EVENT && on[isyn] == 1.0) {
            ispike[isyn] += 1.0;
            net_send_cpu(isyn, t + 0.0, SpikeFlag::NORMAL_EVENT);
            next_invl();
            if (on[isyn] == 1.0) {
                net_send_cpu(isyn, t + event[isyn], SpikeFlag::SELF_EVENT);
            }
            send_count++;
        }
    }
    return send_count > 0;
}

void NetStim::current_cpu(SimMechCurrentParam& param) {
    
}

void NetStim::current_gpu(SimMechCurrentParam& param) {
    
}

void NetStim::sync_gpu() {
    
}

void NetStim::state_cpu(SimMechStateParam& param) {
    
}

void NetStim::state_gpu(SimMechStateParam& param) {
    
}

void NetStim::bbcore_read(int icnt, int dcnt, int* iArray, double* dArray, Mode mode) {
    
}
