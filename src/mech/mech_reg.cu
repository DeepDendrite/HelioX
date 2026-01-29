#include "mechanism.h"

#include "eion.cuh"

// user-mech和worm-mech已移至独立.cu文件，使用whole-archive linking自动注册
// 无需手动include，添加新机制只需：
// 1. 在user-mech/或worm-mech/目录创建.cu文件（包含REGISTER_MECHANISM宏）
// 2. 重新编译即可，CMake会自动发现并编译

// 仍需include的机制（暂未迁移）：
#include "exp2syn_temp.cuh"
#include "syn_record.cuh"


// #include "CA_HVA_record.cuh"
// #include "Nap_Et2_record.cuh"
// #include "NaTa_t_record.cuh"
// #include "SKv3_1_record.cuh"
// #include "SK_E2_record.cuh"
// #include "K_Tst_record.cuh"
// #include "K_Pst_record.cuh"
// #include "Ca_LVAst_record.cuh"

void MechanismFactory::registerMechanism(const std::string& name, Creator creator) {
    registry_[name] = creator;
}

void MechanismFactory::registerVarMap(const std::string& name, VarMapAble* varMap) {
    name2varMap_[name] = varMap;
}

void MechanismFactory::registerPntReceiveSize(const std::string& name, int pnt_receive_size) {
    name2pntReceiveSize_[name] = pnt_receive_size;
}

void MechanismFactory::registerPointerDparamSlots(const std::string& name, std::vector<int> slots) {
    name2pointerDparamSlots_[name] = std::move(slots);
}

void MechanismFactory::registerDparamSemantics(const std::string& name, std::vector<int> semantics) {
    name2dparamSemantics_[name] = std::move(semantics);
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

Mechanism* MechanismFactory::createMechanism(const std::string& name, MechInitParams &initParam) {
    auto it = registry_.find(name);
    if (it != registry_.end()) {
        auto mech = it->second(initParam);            
        mech->reg_node_indices(initParam);
        mech->read_data_from_coredat(initParam);

        allocatedMechanisms_.push_back(mech);
        name2varMap_[name] = mech;
        type2name_[mech->type] = name;
        name2type_[name] = mech->type;

        return mech;
    }
    return nullptr;
}

VarMapAble* MechanismFactory::getVarMap(const std::string& name) {
    auto it = name2varMap_.find(name);
    if (it != name2varMap_.end()) {
        return it->second;
    }
    return nullptr;
}

int MechanismFactory::getPntReceiveSize(const std::string& name) {
    auto it = name2pntReceiveSize_.find(name);
    if (it != name2pntReceiveSize_.end()) {
        return it->second;
    }
    return -1; // 返回-1表示未找到
}
