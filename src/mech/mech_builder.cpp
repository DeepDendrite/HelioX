#include "mech_builder.hpp"

#include <stdexcept>

namespace neurong_mech {

Mechanism* create_mech(Simulate& sim,
                       NeuronGroupData& group,
                       int& next_type,
                       const std::string& name,
                       const std::vector<std::int32_t>& node_indices_i32,
                       int data_size,
                       const std::vector<double>& data,
                       int pdata_size,
                       const std::vector<int>& pdata,
                       const std::unordered_map<std::string, std::string>* ion_name_overrides,
                       MechRole role) {
    std::vector<int> node_indices;
    node_indices.resize(node_indices_i32.size());
    for (std::size_t i = 0; i < node_indices.size(); ++i) {
        node_indices[i] = static_cast<int>(node_indices_i32[i]);
    }

    MechInitParams p{};
    p.mode = sim.mode;
    p.type = next_type++;
    p.name = name;
    p.node_count = static_cast<int>(node_indices.size());
    p.nodeindices = node_indices.data();
    p.data_size = data_size;
    p.data = data_size ? const_cast<double*>(data.data()) : nullptr;
    p.pdata_size = pdata_size;
    p.pdata = pdata_size ? const_cast<int*>(pdata.data()) : nullptr;
    p.permute = nullptr;
    p.array_dims = nullptr;
    p.ion_name_overrides = ion_name_overrides;
    p.create_as_generic_ion = (role == MechRole::Eion);

    auto& mech_factory = MechanismFactory::getInstance();
    Mechanism* mech = mech_factory.createMechanism(name, p);
    if (!mech) {
        throw std::runtime_error("Mechanism not registered: " + name);
    }
    if (mech->need_area) {
        mech->vecdata_area = group.vecdata_area;
    }

    if (name == "capacitance") {
        group.mech_cap = dynamic_cast<Capac*>(mech);
    }

    if (auto* arti = dynamic_cast<ArtiCell*>(mech)) {
        group.vec_articell.push_back(arti);
        if (group.spk_vec != nullptr) {
            arti->spk_vec_bkp = group.spk_vec;
        }
        if (group.vecdata_spk_flags != nullptr) {
            arti->spk_flags_bkp = group.vecdata_spk_flags;
        }
    }
    if (auto* postsyn = dynamic_cast<PostSyn_trait*>(mech)) {
        group.vec_postsyn.push_back(postsyn);
    }

    const bool is_ion = (role == MechRole::Eion);
    if (is_ion) {
        group.vec_eion.push_back(mech);
    } else {
        // Non-ion mechanisms (including ion_species models) go to the generic mech list.
        group.mechanism_list.push_back(mech);
    }
    group.mech_current_list.push_back(mech);

    // In in-memory builds, writer mechs are explicitly declared via ion_species.
    // This keeps ordering logic independent from low-level mechanism flags.
    if (!is_ion && role == MechRole::IonSpecies) {
        group.mech_write_state_ion_list.push_back(mech);
    }

    if (auto* ve = dynamic_cast<VecEvent*>(mech)) {
        group.mech_vecevent = ve;
    }

    return mech;
}

}  // namespace neurong_mech
