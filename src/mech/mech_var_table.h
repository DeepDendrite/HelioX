#pragma once

#include "vecdata.h"
#include <string>
#include <unordered_map>

using namespace std;

struct MechVarData{
    double *cpu_data;
    double *gpu_data;
    int len;
    string name;
};

using MechVarMap = unordered_map<int, MechVarData>;//cordDatIdx -> MechVarData
using MechVarTable = unordered_map<int, MechVarMap>;

inline MechVarTable mech_var_table;
