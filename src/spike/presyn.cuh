#ifndef presyn_cuh
#define presyn_cuh
#include <iostream>
#include <cstdint>
#include "spike.h"

extern "C" {
	// Asynchronous spike detection:
	// - runs detect kernel on provided (non-blocking) stream
	// - copies back 2 small counters (spk_num_real/tot) to host-pinned memory
	// - records the provided event on the same stream
	// Caller decides when/how to wait (event/stream sync), avoiding device-wide sync.
	void cuda_spike_send_async(double* vec_v,
	                          SpikeFlag* spk_flags,
	                          int* pre_node_indices,
	                          uint32_t* spk_vec_offset,
	                          double* threshold,
	                          bool* pre_flags,
	                          double t,
	                          int len,
	                          int* spk_idx_vec,
	                          int tot_len,
	                          int* d_spk_num_real,
	                          int* d_spk_num_tot,
	                          int* h_spk_num_real,
	                          int* h_spk_num_tot,
	                          void* cuda_stream,
	                          void* cuda_event);
}

#endif
