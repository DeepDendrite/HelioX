#include "mechanism.h"
#include <math.h>
#include <string.h>
#include <map>
#include <set>
#include "global_vars.h"
#include "utils.h"

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
    // CPU 模式下不应依赖 CUDA runtime 资源（否则在无 GPU/受限环境会失败，且没必要）。
    if (mode == GPU) {
        cuda_stream_initialize(&cuda_stream);
        cuda_event_initialize(&cuda_event);
    } else {
        cuda_stream = nullptr;
        cuda_event = nullptr;
    }
}

Mechanism::~Mechanism()
{
    
}

void Mechanism::cleanUp()
{
    if (mode == GPU) {
        cuda_event_destroy(&cuda_event);
        cuda_stream_destroy(&cuda_stream);
    } else {
        cuda_event = nullptr;
        cuda_stream = nullptr;
    }
}

const std::vector<int>* MechanismFactory::getPointerDparamSlots(const std::string& name) const {
    auto it = name2pointerDparamSlots_.find(name);
    if (it != name2pointerDparamSlots_.end()) {
        return &it->second;
    }
    return nullptr;
}

const std::vector<int>* MechanismFactory::getDparamSemantics(const std::string& name) const {
    auto it = name2dparamSemantics_.find(name);
    if (it != name2dparamSemantics_.end()) {
        return &it->second;
    }
    return nullptr;
}

