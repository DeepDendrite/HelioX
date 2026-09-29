#include "ion_table.h"
#include "var_struct.cuh"

#include <cmath>
#include <stdexcept>
using namespace std;
static unordered_map<string,EionData> EionTable;
static unordered_map<string, IonMeta> IonMetaTable;

namespace {
constexpr double kUndefinedIonCharge = -10000.0;
constexpr double kChargeTol = 1e-12;

void check_charge(const std::string& name, double charge) {
    if (!std::isfinite(charge) || std::abs(charge - kUndefinedIonCharge) < kChargeTol) {
        throw std::runtime_error("invalid valence for ion " + name);
    }
}

void check_builtin_charge(const std::string& name, double charge, double expected) {
    if (std::abs(charge - expected) > kChargeTol) {
        throw std::runtime_error(
            "conflicting valence for ion " + name + ": got " + std::to_string(charge) +
            ", expected " + std::to_string(expected));
    }
}
}  // namespace

void reg_ion(Mode mode, const std::string& name, double **cpu_data, double **gpu_data,int nnodes){
    auto &eionData = EionTable[normalize_ion_meta_name(name)];
    eionData.nnodes = nnodes;
    eionData.cpu_data = cpu_data;
    if(mode == Mode::GPU){
        int Eion_Var_Size = magic_enum::enum_count<EionVarNames>();
        if (eionData.gpu_data_oh_host) {
            delete[] eionData.gpu_data_oh_host;
            eionData.gpu_data_oh_host = nullptr;
        }
        eionData.gpu_data_oh_host = new double*[Eion_Var_Size];
        cudaMemcpy(eionData.gpu_data_oh_host,gpu_data,Eion_Var_Size*sizeof(double*),cudaMemcpyDeviceToHost);
    }

}
EionData& get_ion_vars(const std::string& name,bool must_exist){
    if(must_exist){
        assert(EionTable.find(normalize_ion_meta_name(name)) != EionTable.end());
    }
    return EionTable[normalize_ion_meta_name(name)];
}

std::string normalize_ion_meta_name(const std::string& name) {
    if (name.empty()) {
        throw std::runtime_error("empty ion name");
    }
    if (name.size() >= 4 && name.compare(name.size() - 4, 4, "_ion") == 0) {
        return name;
    }
    return name + "_ion";
}

IonMeta ion_meta_from_valence(const std::string& name, double charge) {
    const std::string ion_name = normalize_ion_meta_name(name);
    check_charge(ion_name, charge);

    if (ion_name == "na_ion") {
        check_builtin_charge(ion_name, charge, 1.0);
        return IonMeta{1.0, 10.0, 140.0};
    }
    if (ion_name == "k_ion") {
        check_builtin_charge(ion_name, charge, 1.0);
        return IonMeta{1.0, 140.0, 4.0};
    }
    if (ion_name == "ca_ion") {
        check_builtin_charge(ion_name, charge, 2.0);
        return IonMeta{2.0, 5.e-5, 2.0};
    }
    return IonMeta{charge, 1.0, 1.0};
}

void register_ion_meta(const std::string& name, const IonMeta& meta) {
    const std::string ion_name = normalize_ion_meta_name(name);
    if (!std::isfinite(meta.charge) ||
        !std::isfinite(meta.default_conci) ||
        !std::isfinite(meta.default_conco)) {
        throw std::runtime_error("non-finite ion metadata for ion " + ion_name);
    }
    if (std::abs(meta.charge) > kChargeTol) {
        if (ion_name == "na_ion" || ion_name == "k_ion") {
            check_builtin_charge(ion_name, meta.charge, 1.0);
        } else if (ion_name == "ca_ion") {
            check_builtin_charge(ion_name, meta.charge, 2.0);
        }
    }
    IonMetaTable[ion_name] = meta;
}

void register_ion_meta_from_valence(const std::string& name, double charge) {
    const std::string ion_name = normalize_ion_meta_name(name);
    IonMeta meta = ion_meta_from_valence(ion_name, charge);
    auto it = IonMetaTable.find(ion_name);
    if (it == IonMetaTable.end()) {
        IonMetaTable.emplace(ion_name, meta);
        return;
    }
    if (std::abs(it->second.charge) > kChargeTol &&
        std::abs(it->second.charge - meta.charge) > kChargeTol) {
        throw std::runtime_error(
            "conflicting valence for ion " + ion_name + ": existing " +
            std::to_string(it->second.charge) + ", imported " + std::to_string(meta.charge));
    }
    if (std::abs(it->second.charge) <= kChargeTol) {
        it->second.charge = meta.charge;
    }
}

bool has_ion_meta(const std::string& name) {
    return IonMetaTable.find(normalize_ion_meta_name(name)) != IonMetaTable.end();
}

IonMeta get_ion_meta(const std::string& name, bool must_exist) {
    const std::string ion_name = normalize_ion_meta_name(name);
    auto it = IonMetaTable.find(ion_name);
    if (it == IonMetaTable.end()) {
        if (must_exist) {
            throw std::runtime_error("ion meta not found for ion " + ion_name);
        }
        return IonMeta{};
    }
    return it->second;
}
