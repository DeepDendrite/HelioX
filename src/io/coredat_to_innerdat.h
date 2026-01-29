#pragma once
#include "neuron.h"
#include "coredat_structs.h"

void data_format_trans(vector<HelioXroupData*> &neuron_group_list, unique_ptr<coreneuron::CoreData *[]> &coredata_arr, int ngroup, Mode mode,double dt);