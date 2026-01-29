#include "articell.h"
#include <iostream>
#include <cstdint>

using namespace std;

ArtiCell::ArtiCell(MechInitParams &param) : PostSyn(param)
{
}

ArtiCell::~ArtiCell()
{

}

void ArtiCell::net_send_cpu(uint32_t idx, double t, SpikeFlag flag)
{
    uint32_t stype, sid;
    uint32_t* spk_vec_offset = this->vecdata_spk_vec_offset->get_cpu_data();
    SpikeFlag* spk_flags = spk_flags_bkp->get_cpu_data();
    if (flag == SpikeFlag::SELF_EVENT || flag == SpikeFlag::NETSTIM_INIT_EVENT) // self-event
    {
        stype = this->type;
        sid = idx;
        Spike spk(stype, sid, t, flag);
        spike_buffer.push(spk);
    }
    else
    {
        Spike spk(stype, sid, t, flag);
        spk_vec_bkp->set_spike(spk, spk_vec_offset[idx]);
        spk_flags[spk_vec_offset[idx]] = flag;
    }
}
