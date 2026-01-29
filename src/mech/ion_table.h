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
    double **cpu_data;
    double **gpu_data_oh_host;
    int nnodes;//该eion的节点数
    std::unordered_map<int,int> idx_reverse_table;
};



void reg_ion(Mode mode, const std::string& name,double **cpu_data,double **gpu_data,int nnodes);
EionData& get_ion_vars(const std::string& name,bool must_exist = false);
