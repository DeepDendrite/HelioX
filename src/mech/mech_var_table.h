#pragma once

#include "vecdata.h"
#include <string>
#include <unordered_map>

using namespace std;

struct MechVarData{
    double *cpu_data = nullptr;
    double *gpu_data = nullptr;
    VecData<double>* vecdata = nullptr;
    int len = 0;
    string name;
};

using MechVarMap = unordered_map<int, MechVarData>;//cordDatIdx -> MechVarData
using MechVarTable = unordered_map<int, MechVarMap>;

extern MechVarTable mech_var_table;
