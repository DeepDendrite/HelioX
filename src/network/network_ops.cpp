#include "network/network_ops.hpp"

#include "runtime_api/core/SimRuntimeCore.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>

namespace neurong_network {
namespace {

std::string to_lower_ascii(std::string source) {
    std::transform(source.begin(), source.end(), source.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return source;
}

const char* pre_kind_to_string(PreEndpointKind kind) {
    switch (kind) {
    case PreEndpointKind::RealCell:
        return "real";
    case PreEndpointKind::ArtificialByLoc:
        return "artificial";
    default:
        return "unknown";
    }
}

int load_network_by_kind(neurong::runtime_api::core::SimRuntimeCore& core,
                         const std::vector<ConnectionSpec>& connections,
                         PreEndpointKind expected_kind,
                         const char* api_name) {
    auto* sim = core.sim();
    if (sim == nullptr) {
        printf("Simulate not initialized. Load morphology before %s().\n", api_name);
        return -1;
    }
    if (core.get_source() != "neurong") {
        printf("%s() is only available for source='neurong' in-memory models.\n", api_name);
        return -1;
    }
    const auto* morph_layout = core.in_memory_morph_layout();
    if (morph_layout == nullptr) {
        printf("Missing in-memory morphology layout. Call load_celltemplate_morphology() first.\n");
        return -1;
    }
    if (!core.is_in_memory_biophysics_applied()) {
        printf("Biophysics not applied. Call apply_celltemplate_biophysics() before %s().\n", api_name);
        return -1;
    }
    if (sim->neuron_group_list.empty() || sim->neuron_group_list[0] == nullptr) {
        printf("No neuron group available for network loading.\n");
        return -1;
    }
    if (!(sim->dt > 0.0)) {
        printf("Invalid dt=%g for network loading.\n", sim->dt);
        return -1;
    }

    LoadOptions opt{};
    opt.dt = sim->dt;

    try {
        std::vector<ConnectionSpec> resolved_connections = connections;
        for (auto& conn : resolved_connections) {
            if (conn.preKind != expected_kind) {
                throw std::runtime_error(
                    std::string(api_name) + " received wrong pre kind for connection id=" +
                    std::to_string(conn.id) + ": expected " + pre_kind_to_string(expected_kind) +
                    ", got " + pre_kind_to_string(conn.preKind));
            }
            if (expected_kind == PreEndpointKind::ArtificialByLoc) {
                const int resolved_arti_index = resolve_arti_presyn_index_by_loc(
                    core,
                    conn.preMech,
                    conn.preGid,
                    conn.preSection,
                    conn.preSectionLocation);
                if (resolved_arti_index < 0) {
                    throw std::runtime_error(
                        "connection id=" + std::to_string(conn.id) +
                        " failed to resolve artificial pre endpoint: mech='" + conn.preMech +
                        "' gid=" + std::to_string(conn.preGid) +
                        " section='" + conn.preSection +
                        "' loc=" + std::to_string(conn.preSectionLocation));
                }
                const int pre_real_cells = morph_layout->num_cells_total;
                conn.preCellId = pre_real_cells + resolved_arti_index;
            } else {
                conn.preCellId = conn.preGid;
            }
        }

        auto* group = sim->neuron_group_list[0];
        auto result = load_connections(opt, sim->mode, *group, *morph_layout, resolved_connections);
        printf("In-memory network loaded via %s: connections=%d bound=%d pre_real=%d pre_arti=%d\n",
               api_name,
               result.n_connections,
               result.n_bound_connections,
               result.n_presyn_real,
               result.n_presyn_arti);
        return 0;
    } catch (const std::exception& e) {
        printf("%s failed: %s\n", api_name, e.what());
        return -1;
    }
}

}  // namespace

int resolve_arti_presyn_index_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                                     const std::string& mech,
                                     int gid,
                                     const std::string& section_name,
                                     double loc) {
    const std::string mech_name = to_lower_ascii(mech);
    if (mech_name.empty() || mech_name == "global") {
        printf("resolve_arti_presyn_index_by_loc: mech must be non-global (got=%s)\n", mech.c_str());
        return -1;
    }
    return core.resolve_mech_index_by_loc(mech, gid, section_name, loc, 0);
}

std::vector<int> resolve_arti_presyn_indices_by_locs(
    neurong::runtime_api::core::SimRuntimeCore& core,
    const std::string& mech,
    const std::vector<int>& gids,
    const std::vector<std::string>& section_names,
    const std::vector<double>& locs) {
    if (gids.size() != section_names.size() || gids.size() != locs.size()) {
        printf("resolve_arti_presyn_indices_by_locs: size mismatch gids=%zu sections=%zu locs=%zu\n",
               gids.size(),
               section_names.size(),
               locs.size());
        return {};
    }
    std::vector<int> out;
    out.reserve(gids.size());
    for (std::size_t i = 0; i < gids.size(); ++i) {
        const int pre_cell_id = resolve_arti_presyn_index_by_loc(
            core,
            mech,
            gids[i],
            section_names[i],
            locs[i]);
        if (pre_cell_id < 0) {
            return {};
        }
        out.push_back(pre_cell_id);
    }
    return out;
}

int load_network_real(neurong::runtime_api::core::SimRuntimeCore& core,
                      const std::vector<ConnectionSpec>& connections) {
    return load_network_by_kind(core, connections, PreEndpointKind::RealCell, "load_network_real");
}

int load_network_artificial(neurong::runtime_api::core::SimRuntimeCore& core,
                            const std::vector<ConnectionSpec>& connections) {
    return load_network_by_kind(core, connections, PreEndpointKind::ArtificialByLoc, "load_network_artificial");
}

}  // namespace neurong_network
