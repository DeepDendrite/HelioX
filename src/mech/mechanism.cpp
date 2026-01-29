#include "mechanism.h"
#include <math.h>
#include <string.h>
#include <map>
#include <set>
#include "global_vars.h"

Mechanism::Mechanism(MechInitParams &param)
{
    name = param.name;
    mode = param.mode;
    type = param.type;
    permute = param.permute;
    need_area = false;
    write_state_ion = false;

    nnode = param.node_count;
    if(param.nodeindices){
        vecdata_node_indices = std::make_unique<VecData<int>>(mode, param.nodeindices, nnode);
    }else{
        printf_debug("Mech[%d] nodeindices is null please check it's arti cell\n",type);
    }
    vecdata_g_mech = std::make_unique<VecData<double>>(mode, 0.0, nnode);
    vecdata_i_mech = std::make_unique<VecData<double>>(mode, 0.0, nnode);

    celsius = coreneuron::global_var_map.at("celsius")[0];
    cuda_stream_initialize(&cuda_stream);
    cuda_event_initialize(&cuda_event);
}

Mechanism::~Mechanism()
{
    
}

void Mechanism::cleanUp()
{
    cuda_event_destroy(&cuda_event);
    cuda_stream_destroy(&cuda_stream);
}




