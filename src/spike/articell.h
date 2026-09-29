#ifndef articell_h
#define articell_h

#include "postsyn.h"

// artificial cell, in addition to spike_receive 
// function, it has net_send() function to deliver 
// spikes to other synapses or artificial cells
class ArtiCell : public PostSyn
{
    public: 
        bool has_net_event = false;
        ArtiCell(MechInitParams &param);
        virtual ~ArtiCell();

        VecData<uint32_t>* vecdata_spk_vec_offset;
        SpikeVector* spk_vec_bkp; // spike vector pointer which points to spike vector object in NeuronGroupData
        VecData<SpikeFlag> *spk_flags_bkp; // spike flag pointer pointing to vecdata_spk_flags in NeuronGroupData
        // customized function for reading data
        virtual void bbcore_read(int icnt, int dcnt, int* iArray, double* dArray, Mode mode) = 0; 

        // deliver event at time t
        // flag: 0: normal event, 1: self-event
        void net_send_cpu(uint32_t idx, double t, SpikeFlag flag);
        
};

#endif
