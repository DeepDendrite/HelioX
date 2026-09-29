#pragma once
#include <string>
#include <unordered_map>
#include "utils.h"
enum class EionVarNames
{//以na离子为例，对应着：
    erev, //e[na]
    conci,//[na]i
    conco,//[na]o
    cur,//i[na]
    dcurdv//di[na]_dv
};
struct EionTrait{
    using VarNames = EionVarNames;
};
struct EionData{
    // eion 变量在CPU/GPU上的指针数组，元素类型与机制内部一致（double，双精度）
    double **cpu_data{nullptr};
    double **gpu_data_oh_host{nullptr};
    int nnodes{0};//该eion的节点数
    std::unordered_map<int,int> idx_reverse_table;
};

struct IonMeta {
    double charge{0.0};
    double default_conci{0.0};
    double default_conco{0.0};
};

void reg_ion(Mode mode, const std::string& name, double **cpu_data, double **gpu_data,int nnodes);
EionData& get_ion_vars(const std::string& name,bool must_exist = false);

std::string normalize_ion_meta_name(const std::string& name);
IonMeta ion_meta_from_valence(const std::string& name, double charge);
void register_ion_meta(const std::string& name, const IonMeta& meta);
void register_ion_meta_from_valence(const std::string& name, double charge);
bool has_ion_meta(const std::string& name);
IonMeta get_ion_meta(const std::string& name, bool must_exist = false);
