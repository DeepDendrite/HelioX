#include "network_builder.hpp"
#include "morph/section_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace neurong_network {
namespace {

[[nodiscard]] inline int map_template_node_to_original(
    const neurong_biophysical::CellTemplateMorphLayout& morph,
    int cell_id,
    std::int32_t template_node_index) {
    if (template_node_index < 0) {
        return -1;
    }
    if (template_node_index == 0) {
        return cell_id;
    }
    const auto cell_u = static_cast<std::size_t>(cell_id);
    const auto base = morph.cell_nonroot_base[cell_u];
    const auto idx = base + (static_cast<std::size_t>(template_node_index) - 1);
    return static_cast<int>(idx);
}

[[nodiscard]] inline int map_post_location_to_permuted_node(
    const neurong_biophysical::CellTemplateMorphLayout& morph,
    const std::vector<std::vector<std::string>>& section_names_by_index,
    const NeuronGroupData& group,
    int post_cell_id,
    const ConnectionSpec& conn) {
    const auto post_cell_u = static_cast<std::size_t>(post_cell_id);
    const auto tpl_u = morph.cell_template_id[post_cell_u];
    const auto& tpl = morph.templates[tpl_u];

    std::string section_name{};
    int section_index = -1;
    if (conn.postSectionIsIndex) {
        section_index = conn.postSectionIndex;
        if (tpl_u >= section_names_by_index.size()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " internal error: template index out of section-name cache range");
        }
        const auto& names = section_names_by_index[tpl_u];
        if (section_index < 0 || static_cast<std::size_t>(section_index) >= names.size()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " has out-of-range postSection index " + std::to_string(section_index));
        }
        section_name = names[static_cast<std::size_t>(section_index)];
        if (section_name.empty()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " has postSection index without resolvable section name");
        }
    } else {
        section_name = conn.postSection;
        if (section_name.empty()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) + " has empty postSection");
        }
    }

    const double loc = conn.postSectionLocation;
    if (!std::isfinite(loc) || loc < 0.0 || loc > 1.0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) +
            " has invalid postSectionLocation=" + std::to_string(loc));
    }

    std::int32_t tpl_node = -1;
    if (!neurong_morph::resolve_template_node_by_loc(tpl.section_name_to_id,
                                                     tpl.section_node_base_id,
                                                     tpl.section_nseg,
                                                     tpl.section_parent_node_id,
                                                     section_name,
                                                     loc,
                                                     &tpl_node) ||
        tpl_node < 0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) +
            " maps to invalid template node for postSection='" + section_name +
            "' loc=" + std::to_string(loc));
    }
    const int original = map_template_node_to_original(morph, post_cell_id, tpl_node);
    if (original < 0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) + " maps to invalid original node");
    }
    if (group.permute) {
        return group.permute[original];
    }
    return original;
}

[[nodiscard]] inline int map_pre_location_to_permuted_node(
    const neurong_biophysical::CellTemplateMorphLayout& morph,
    const NeuronGroupData& group,
    int pre_gid,
    const ConnectionSpec& conn) {
    if (pre_gid < 0 || pre_gid >= morph.num_cells_total) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) +
            " has invalid preGid=" + std::to_string(pre_gid));
    }
    const auto pre_cell_u = static_cast<std::size_t>(pre_gid);
    const auto tpl_u = morph.cell_template_id[pre_cell_u];
    const auto& tpl = morph.templates[tpl_u];
    const std::string section_name = conn.preSection;
    if (section_name.empty()) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) + " has empty preSection");
    }
    const double loc = conn.preSectionLocation;
    if (!std::isfinite(loc) || loc < 0.0 || loc > 1.0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) +
            " has invalid preSectionLocation=" + std::to_string(loc));
    }

    std::int32_t tpl_node = -1;
    if (!neurong_morph::resolve_template_node_by_loc(tpl.section_name_to_id,
                                                     tpl.section_node_base_id,
                                                     tpl.section_nseg,
                                                     tpl.section_parent_node_id,
                                                     section_name,
                                                     loc,
                                                     &tpl_node) ||
        tpl_node < 0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) +
            " maps to invalid template node for preSection='" + section_name +
            "' loc=" + std::to_string(loc));
    }
    const int original = map_template_node_to_original(morph, pre_gid, tpl_node);
    if (original < 0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(conn.id) + " maps to invalid pre original node");
    }
    if (group.permute) {
        return group.permute[original];
    }
    return original;
}

[[nodiscard]] inline std::int32_t quantize_delay_step(double delay_ms, double dt_ms) {
    if (!std::isfinite(delay_ms) || delay_ms < 0.0) {
        return 0;
    }
    if (!(dt_ms > 0.0) || !std::isfinite(dt_ms)) {
        dt_ms = 0.025;
    }
    // Fixed-step event delivery cannot observe a fractional-delay event before
    // the next delivery tick, so use ceil rather than nearest-step rounding.
    return static_cast<std::int32_t>(std::ceil(delay_ms / dt_ms - 1e-9));
}

[[nodiscard]] inline std::vector<std::vector<std::string>> build_section_names_by_index(
    const neurong_biophysical::CellTemplateMorphLayout& morph) {
    std::vector<std::vector<std::string>> out;
    out.resize(morph.templates.size());
    for (std::size_t tpl_i = 0; tpl_i < morph.templates.size(); ++tpl_i) {
        const auto& tpl = morph.templates[tpl_i];
        auto& names = out[tpl_i];
        names.resize(tpl.section_nseg.size());
        for (const auto& [name, section_id] : tpl.section_name_to_id) {
            const auto sec_u = static_cast<std::size_t>(section_id);
            if (sec_u >= names.size()) {
                throw std::runtime_error(
                    "template '" + tpl.name + "' has out-of-range section id for name '" + name + "'");
            }
            if (!names[sec_u].empty()) {
                throw std::runtime_error(
                    "template '" + tpl.name + "' has duplicate section id mapping for index " +
                    std::to_string(sec_u));
            }
            names[sec_u] = name;
        }
        for (std::size_t sec_u = 0; sec_u < names.size(); ++sec_u) {
            if (names[sec_u].empty()) {
                throw std::runtime_error(
                    "template '" + tpl.name + "' missing section name for index " + std::to_string(sec_u));
            }
        }
    }
    return out;
}

}  // namespace

void setup_spike_runtime(Mode mode,
                         NeuronGroupData& group,
                         const neurong_biophysical::CellTemplateMorphLayout& morph,
                         std::vector<ConnectionSpec>& connections) {
    if (group.presyn) {
        delete group.presyn;
        group.presyn = nullptr;
    }
    if (group.vecdata_spk_flags) {
        delete group.vecdata_spk_flags;
        group.vecdata_spk_flags = nullptr;
    }
    if (group.spk_vec) {
        delete group.spk_vec;
        group.spk_vec = nullptr;
    }

    std::vector<int> pre_node_indices;
    std::vector<double> pre_thresholds;
    std::vector<int> pre_gids;
    std::vector<int> pre_first_conn_ids;
    pre_node_indices.reserve(connections.size());
    pre_thresholds.reserve(connections.size());
    pre_gids.reserve(connections.size());
    pre_first_conn_ids.reserve(connections.size());

    std::unordered_map<int, int> pre_node_to_idx;
    pre_node_to_idx.reserve(connections.size());

    for (auto& conn : connections) {
        if (conn.preKind != PreEndpointKind::RealCell) {
            continue;
        }
        if (!std::isfinite(conn.threshold)) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " has non-finite threshold");
        }
        const int pre_permuted_node = map_pre_location_to_permuted_node(
            morph,
            group,
            conn.preGid,
            conn);
        auto [it, inserted] = pre_node_to_idx.emplace(pre_permuted_node,
                                                      static_cast<int>(pre_node_indices.size()));
        if (inserted) {
            pre_node_indices.push_back(pre_permuted_node);
            pre_thresholds.push_back(conn.threshold);
            pre_gids.push_back(conn.preGid);
            pre_first_conn_ids.push_back(conn.id);
            conn.preCellId = it->second;
            continue;
        }
        const int existing_idx = it->second;
        if (pre_gids[static_cast<std::size_t>(existing_idx)] != conn.preGid) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " maps to pre node that is already bound to another preGid=" +
                std::to_string(pre_gids[static_cast<std::size_t>(existing_idx)]) +
                " (node=" + std::to_string(pre_permuted_node) +
                ", this preGid=" + std::to_string(conn.preGid) + ")");
        }
        const double existing_threshold = pre_thresholds[static_cast<std::size_t>(existing_idx)];
        if (conn.threshold != existing_threshold) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " conflicts on real pre threshold for node=" + std::to_string(pre_permuted_node) +
                " (preGid=" + std::to_string(conn.preGid) +
                ", existing threshold=" + std::to_string(existing_threshold) +
                " from connection id=" +
                std::to_string(pre_first_conn_ids[static_cast<std::size_t>(existing_idx)]) +
                ", new threshold=" + std::to_string(conn.threshold) + ")");
        }
        conn.preCellId = existing_idx;
    }
    const int npre_real = static_cast<int>(pre_node_indices.size());

    int npre_arti = 0;
    for (auto* arti : group.vec_articell) {
        if (!arti) {
            continue;
        }
        npre_arti += std::max(0, arti->node_count());
    }

    // Artificial indices are resolved in runtime core as:
    //   preCellId = morph.num_cells_total + arti_local_index.
    // Remap them to runtime spike-vector space:
    //   preCellId = npre_real + arti_local_index.
    const int pre_arti_base = morph.num_cells_total;
    for (auto& conn : connections) {
        if (conn.preKind != PreEndpointKind::ArtificialByLoc) {
            continue;
        }
        const int arti_local_index = conn.preCellId - pre_arti_base;
        if (arti_local_index < 0 || arti_local_index >= npre_arti) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " has invalid artificial pre index (raw preCellId=" +
                std::to_string(conn.preCellId) + ", expected in [" +
                std::to_string(pre_arti_base) + ", " +
                std::to_string(pre_arti_base + npre_arti) + "))");
        }
        conn.preCellId = npre_real + arti_local_index;
    }

    const int nspk = npre_real + npre_arti;

    group.spk_vec = new SpikeVector(static_cast<std::uint32_t>(std::max(0, nspk)));
    group.vecdata_spk_flags = new VecData<SpikeFlag>(mode, SpikeFlag::INVALID, std::max(0, nspk));

    std::vector<std::uint32_t> spk_vec_offset;
    spk_vec_offset.resize(static_cast<std::size_t>(std::max(0, npre_real)));
    for (int i = 0; i < npre_real; ++i) {
        spk_vec_offset[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
    }

    group.presyn = new PreSyn(mode,
                              static_cast<std::uint32_t>(std::max(0, npre_real)),
                              group.spk_vec,
                              pre_node_indices,
                              pre_thresholds,
                              spk_vec_offset,
                              pre_gids);

    std::uint32_t next_offset = static_cast<std::uint32_t>(std::max(0, npre_real));
    for (auto* arti : group.vec_articell) {
        if (!arti) {
            continue;
        }
        const int ninst = std::max(0, arti->node_count());
        std::vector<std::uint32_t> offsets(static_cast<std::size_t>(ninst));
        for (int i = 0; i < ninst; ++i) {
            offsets[static_cast<std::size_t>(i)] = next_offset++;
        }
        if (arti->vecdata_spk_vec_offset) {
            delete arti->vecdata_spk_vec_offset;
            arti->vecdata_spk_vec_offset = nullptr;
        }
        arti->vecdata_spk_vec_offset = new VecData<std::uint32_t>(mode, offsets);
        arti->spk_vec_bkp = group.spk_vec;
        arti->spk_flags_bkp = group.vecdata_spk_flags;
    }
}

LoadResult load_connections(const LoadOptions& opt,
                            Mode mode,
                            NeuronGroupData& group,
                            const neurong_biophysical::CellTemplateMorphLayout& morph,
                            const std::vector<ConnectionSpec>& connections) {
    std::vector<ConnectionSpec> resolved_connections = connections;
    setup_spike_runtime(mode, group, morph, resolved_connections);

    LoadResult result{};
    result.n_presyn_real = group.presyn ? static_cast<int>(group.presyn->npre) : 0;
    for (auto* arti : group.vec_articell) {
        if (!arti) {
            continue;
        }
        result.n_presyn_arti += std::max(0, arti->node_count());
    }
    result.n_connections = static_cast<int>(resolved_connections.size());

    if (resolved_connections.empty()) {
        return result;
    }

    struct PostSynBindingState {
        PostSyn_trait* postsyn = nullptr;
        Mechanism* mech = nullptr;
        int npost_inst = 0;
        std::uint32_t* spk_vec_idx = nullptr;
        double* delay_vec = nullptr;
        int* delay_step_vec = nullptr;
        double* weight_vec = nullptr;
        std::unordered_map<int, std::vector<int>> node_to_instances;
    };
    std::unordered_map<std::string, PostSynBindingState> postsyn_bindings;
    postsyn_bindings.reserve(group.mechanism_list.size());
    for (auto* mech : group.mechanism_list) {
        if (!mech) {
            continue;
        }
        auto* postsyn = dynamic_cast<PostSyn_trait*>(mech);
        if (!postsyn) {
            continue;
        }
        const auto [it, inserted] = postsyn_bindings.emplace(
            mech->name, PostSynBindingState{postsyn, mech});
        if (!inserted) {
            throw std::runtime_error(
                "duplicate PostSyn mechanism name detected: '" + mech->name +
                "'; connection.postMech must be unique");
        }
        (void)it;
    }

    const int npre_total = result.n_presyn_real + result.n_presyn_arti;
    std::unordered_set<std::string> used_post_mechs;
    used_post_mechs.reserve(resolved_connections.size());
    for (const auto& conn : resolved_connections) {
        if (conn.preCellId < 0 || conn.preCellId >= npre_total) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " has invalid preCellId=" + std::to_string(conn.preCellId));
        }
        if (conn.postCellId < 0 || conn.postCellId >= morph.num_cells_total) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " has invalid postCellId=" + std::to_string(conn.postCellId));
        }
        if (!std::isfinite(conn.weight)) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) + " has non-finite weight");
        }
        if (!std::isfinite(conn.delay) || conn.delay < 0.0) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) + " has invalid delay");
        }
        if (conn.postMech.empty()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) + " has empty postMech");
        }
        used_post_mechs.insert(conn.postMech);
    }

    for (const auto& post_mech_name : used_post_mechs) {
        auto binding_it = postsyn_bindings.find(post_mech_name);
        if (binding_it == postsyn_bindings.end()) {
            throw std::runtime_error(
                "no PostSyn mechanism named '" + post_mech_name +
                "' found; insert that mechanism in celltemplate before loading network");
        }
        auto& binding = binding_it->second;
        if (!binding.postsyn->vecdata_spk_vec_idx || !binding.postsyn->vecdata_delay ||
            !binding.postsyn->vecdata_delay_steps || !binding.postsyn->vecdata_weights) {
            throw std::runtime_error(
                "selected postsyn '" + post_mech_name + "' is not initialized");
        }
        auto* node_indices = binding.mech->node_indices_cpu_data();
        binding.npost_inst = binding.mech->node_count();
        if (!node_indices || binding.npost_inst <= 0) {
            throw std::runtime_error(
                "selected postsyn '" + post_mech_name + "' has no instances");
        }
        binding.spk_vec_idx = binding.postsyn->vecdata_spk_vec_idx->get_cpu_data();
        binding.delay_vec = binding.postsyn->vecdata_delay->get_cpu_data();
        binding.delay_step_vec = binding.postsyn->vecdata_delay_steps->get_cpu_data();
        binding.weight_vec = binding.postsyn->vecdata_weights->get_cpu_data();
        for (int i = 0; i < binding.npost_inst; ++i) {
            binding.spk_vec_idx[i] = 0;
            binding.delay_vec[i] = 0.0;
            binding.delay_step_vec[i] = 0;
            binding.weight_vec[i] = 0.0;
        }
        binding.node_to_instances.clear();
        binding.node_to_instances.reserve(static_cast<std::size_t>(binding.npost_inst) * 2);
        for (int i = 0; i < binding.npost_inst; ++i) {
            binding.node_to_instances[node_indices[i]].push_back(i);
        }
    }

    const auto section_names_by_index = build_section_names_by_index(morph);
    for (const auto& conn : resolved_connections) {
        auto binding_it = postsyn_bindings.find(conn.postMech);
        if (binding_it == postsyn_bindings.end()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " references unknown postMech='" + conn.postMech + "'");
        }
        auto& binding = binding_it->second;

        const int post_permuted_node =
            map_post_location_to_permuted_node(morph, section_names_by_index, group, conn.postCellId, conn);

        auto it_inst = binding.node_to_instances.find(post_permuted_node);
        if (it_inst == binding.node_to_instances.end() || it_inst->second.empty()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(conn.id) +
                " cannot find free " + conn.postMech +
                " instance at post target node=" + std::to_string(post_permuted_node));
        }

        const int inst = it_inst->second.back();
        it_inst->second.pop_back();
        const std::int32_t delay_step = quantize_delay_step(conn.delay, opt.dt);
        const double quantized_delay = static_cast<double>(delay_step) * opt.dt;

        binding.spk_vec_idx[inst] = static_cast<std::uint32_t>(conn.preCellId);
        binding.delay_vec[inst] = quantized_delay;
        binding.delay_step_vec[inst] = delay_step;
        binding.weight_vec[inst] = conn.weight;

        result.n_bound_connections += 1;
    }

    if (mode == GPU) {
        for (const auto& post_mech_name : used_post_mechs) {
            auto binding_it = postsyn_bindings.find(post_mech_name);
            if (binding_it == postsyn_bindings.end()) {
                continue;
            }
            auto* postsyn = binding_it->second.postsyn;
            postsyn->vecdata_spk_vec_idx->update_gpu_data_from_cpu();
            postsyn->vecdata_delay->update_gpu_data_from_cpu();
            postsyn->vecdata_delay_steps->update_gpu_data_from_cpu();
            postsyn->vecdata_weights->update_gpu_data_from_cpu();
        }
    }
    return result;
}

}  // namespace neurong_network
