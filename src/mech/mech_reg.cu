#include "mechanism.h"

#include "eion.cuh"

#include <cstdlib>
#include <unordered_set>

// 独立机制源码通过 standalone_mechs + whole-archive linking 自动注册
// 无需手动include，添加新机制只需：
// 1. 在对应机制源码目录创建.cu文件（包含REGISTER_MECHANISM宏）
// 2. 重新编译即可，CMake会自动发现并编译

// 仍需include的机制（暂未迁移）：
#include "expsyn_temp.cuh"
#include "exp2syn_temp.cuh"
#include "syn_record.cuh"
#include "spike_bridge.cuh"


// #include "CA_HVA_record.cuh"
// #include "Nap_Et2_record.cuh"
// #include "NaTa_t_record.cuh"
// #include "SKv3_1_record.cuh"
// #include "SK_E2_record.cuh"
// #include "K_Tst_record.cuh"
// #include "K_Pst_record.cuh"
// #include "Ca_LVAst_record.cuh"

void MechanismFactory::registerMechanism(const std::string& name,
                                         Creator creator,
                                         const char* source_file,
                                         const char* source_symbol) {
    auto it = registry_.find(name);
    if (it != registry_.end()) {
        const auto info_it = registry_info_.find(name);
        const RegistrationInfo existing = info_it != registry_info_.end()
                                              ? info_it->second
                                              : RegistrationInfo{"<unknown>", "<unknown>"};
        std::cerr << "FATAL: duplicate mechanism registration for \"" << name << "\"\n"
                  << "  existing: " << existing.source_symbol << " from " << existing.source_file << "\n"
                  << "  new:      " << (source_symbol ? source_symbol : "<unknown>") << " from "
                  << (source_file ? source_file : "<unknown>") << "\n";
        std::abort();
    }
    registry_[name] = creator;
    registry_info_[name] = RegistrationInfo{
        source_file ? source_file : "<unknown>",
        source_symbol ? source_symbol : "<unknown>",
    };
}

int MechanismFactory::registerMechanismBatch(const std::vector<PendingRegistration>& registrations,
                                             std::string* error_message) {
    std::unordered_set<std::string> batch_names;
    batch_names.reserve(registrations.size() * 2 + 1);
    for (const auto& reg : registrations) {
        if (reg.name.empty()) {
            if (error_message) {
                *error_message = "batch registration contains empty mechanism name";
            }
            return -1;
        }
        if (!reg.creator) {
            if (error_message) {
                *error_message = "batch registration contains null creator for mechanism '" + reg.name + "'";
            }
            return -1;
        }
        if (registry_.find(reg.name) != registry_.end()) {
            const auto info_it = registry_info_.find(reg.name);
            const RegistrationInfo existing = info_it != registry_info_.end()
                                                  ? info_it->second
                                                  : RegistrationInfo{"<unknown>", "<unknown>"};
            if (error_message) {
                *error_message = "duplicate mechanism registration for '" + reg.name + "'; existing: " +
                                 existing.source_symbol + " from " + existing.source_file + ", new: " +
                                 reg.source_symbol + " from " + reg.source_file;
            }
            return -1;
        }
        const auto [_, inserted] = batch_names.insert(reg.name);
        if (!inserted) {
            if (error_message) {
                *error_message = "batch registration contains duplicate mechanism name '" + reg.name + "'";
            }
            return -1;
        }
    }

    for (const auto& reg : registrations) {
        registry_[reg.name] = reg.creator;
        registry_info_[reg.name] = RegistrationInfo{
            reg.source_file.empty() ? "<unknown>" : reg.source_file,
            reg.source_symbol.empty() ? "<unknown>" : reg.source_symbol,
        };
        if (reg.has_pnt_receive_size) {
            name2pntReceiveSize_[reg.name] = reg.pnt_receive_size;
        }
        if (!reg.pointer_dparam_slots.empty()) {
            name2pointerDparamSlots_[reg.name] = reg.pointer_dparam_slots;
        }
        if (!reg.dparam_semantics.empty()) {
            name2dparamSemantics_[reg.name] = reg.dparam_semantics;
        }
    }
    return 0;
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

const MechanismFactory::RegistrationInfo* MechanismFactory::getRegistrationInfo(const std::string& name) const {
    auto it = registry_info_.find(name);
    if (it != registry_info_.end()) {
        return &it->second;
    }
    return nullptr;
}

Mechanism* MechanismFactory::createMechanism(const std::string& name, MechInitParams &initParam) {
    auto it = registry_.find(name);
    Mechanism* mech = nullptr;
    if (it != registry_.end()) {
        mech = it->second(initParam);
    } else if (initParam.create_as_generic_ion) {
        // Create generic ion mechanism from explicit caller intent.
        mech = new eiontemp::GenericIon(initParam);
    }
    if (!mech) {
        return nullptr;
    }

    mech->reg_node_indices(initParam);
    mech->read_data_from_coredat(initParam);

    allocatedMechanisms_.push_back(mech);
    name2varMap_[name] = mech;
    type2name_[mech->type] = name;
    name2type_[name] = mech->type;
    return mech;
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
