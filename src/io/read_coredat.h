#pragma once
#include "neuron.h"
#include "coredat_structs.h"
#include "permute_order.h"


int read_coredat(unique_ptr<coreneuron::CoreData *[]> &coredata_arr, const char* datapath, const char* filesdat, int byte_swap, bool run_setup_cleanup,int user_mod_num);

void setup_permute_info(HelioXroupData* ndat, coreneuron::CoreData* cdat, PermuteInfo& pi, Mode mode);
void read_data_from_coredat(HelioXroupData* ndat, coreneuron::CoreData* cdat, Mode mode);
void load_data(vector<HelioXroupData*> &neuron_group_list, const char* filename, Mode mode);