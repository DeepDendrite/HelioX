#pragma once

#include <unordered_map>
#include <string>
#include <vector>

namespace coreneuron
{
    extern int secondorder;
    extern std::unordered_map<std::string, std::vector<double>> global_var_map;
}


void read_global_dat(const char *datapath);

