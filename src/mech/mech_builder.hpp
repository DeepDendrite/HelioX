#pragma once

#include "simulate.h"
#include "neuron.h"

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

namespace neurong_mech {

enum class MechRole {
    Normal,
    IonSpecies,
    Eion,
};

Mechanism* create_mech(Simulate& sim,
                       NeuronGroupData& group,
                       int& next_type,
                       const std::string& name,
                       const std::vector<std::int32_t>& node_indices_i32,
                       int data_size,
                       const std::vector<double>& data,
                       int pdata_size,
                       const std::vector<int>& pdata,
                       const std::unordered_map<std::string, std::string>* ion_name_overrides = nullptr,
                       MechRole role = MechRole::Normal);

}  // namespace neurong_mech
