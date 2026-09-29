#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

enum Mode { CPU, GPU };

struct VarDescriptor {
    std::string mech;
    std::string var;
    union {
        int idx;
        int node_or_mech_idx;
    };
    int array_index = -1;

    VarDescriptor() : idx(0), array_index(-1) {}
    VarDescriptor(const std::string& m, const std::string& v, int i)
        : mech(m), var(v), idx(i), array_index(-1) {}
    VarDescriptor(const std::string& m, const std::string& v, int i, int arr_idx)
        : mech(m), var(v), idx(i), array_index(arr_idx) {}

    bool operator<(const VarDescriptor& other) const {
        return std::tie(mech, var, idx, array_index) <
               std::tie(other.mech, other.var, other.idx, other.array_index);
    }

    bool operator==(const VarDescriptor& other) const {
        return mech == other.mech && var == other.var && idx == other.idx &&
               array_index == other.array_index;
    }

    std::string to_string() const {
        return mech + "." + var + "[" + std::to_string(idx) + "][" +
               std::to_string(array_index) + "]";
    }
};

namespace std {
template <>
struct hash<VarDescriptor> {
    size_t operator()(const VarDescriptor& vd) const {
        return std::hash<std::string>()(vd.mech) ^
               (std::hash<std::string>()(vd.var) << 1) ^
               (std::hash<int>()(vd.idx) << 2) ^
               (std::hash<int>()(vd.array_index) << 3);
    }
};
}  // namespace std

template <typename T>
struct CoreVarInfo {
    T info;
    int array_size;

    CoreVarInfo(T info) : info(std::move(info)), array_size(1) {}
    CoreVarInfo(T info, int array_size) : info(std::move(info)), array_size(array_size) {}

    bool isArray() const {
        return array_size > 1;
    }
};

using CoreIdxInfo = CoreVarInfo<int>;
using CoreGlobalVarInfo = CoreVarInfo<std::string>;

struct MechInitParams {
    Mode mode;
    int type;
    std::string name;
    int node_count;
    int* nodeindices;

    int data_size;
    double* data;

    int pdata_size;
    int* pdata;

    int* permute;
    std::vector<int>* array_dims;

    const std::vector<int>* pointer2type = nullptr;
    const std::vector<int>* dparam_semantics = nullptr;
    const std::unordered_map<std::string, std::string>* ion_name_overrides{nullptr};
    bool create_as_generic_ion{false};
};

template <typename T>
void try_delete_arr(T*& ptr) {
    if (ptr) {
        delete[] ptr;
        ptr = nullptr;
    }
}

template <typename T>
void try_delete(T*& ptr) {
    if (ptr) {
        delete ptr;
        ptr = nullptr;
    }
}
