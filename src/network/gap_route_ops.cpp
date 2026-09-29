#include "network/gap_route_ops.hpp"

#include "runtime_api/core/SimRuntimeCore.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>

namespace neurong_network {
namespace {

std::string to_lower_ascii(std::string source) {
    std::transform(source.begin(), source.end(), source.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return source;
}

int resolve_gap_index_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                             const std::string& mech,
                             const std::string& var,
                             int gid,
                             const std::string& section_name,
                             double loc,
                             int slot,
                             const char* api_name) {
    const std::string mech_name = to_lower_ascii(mech);
    const bool is_global = mech_name.empty() || mech_name == "global";
    if (is_global) {
        if (!std::isfinite(loc)) {
            printf("%s: loc must be finite (got=%g)\n", api_name, loc);
            return -1;
        }
        if (loc < 0.0 || loc > 1.0) {
            printf("%s: loc out of range (0 <= loc <= 1), got=%g\n", api_name, loc);
            return -1;
        }
        const std::string var_name = to_lower_ascii(var);
        if (var_name == "v" || var_name == "i_membrane_") {
            return core.resolve_node_index_by_loc(gid, section_name, loc);
        }
        return core.resolve_segment_node_index_by_loc(gid, section_name, loc);
    }
    return core.resolve_mech_index_by_loc(mech, gid, section_name, loc, slot);
}

}  // namespace

int add_gap_source_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                          GapRouteBuilder& gap_routes,
                          int sid,
                          const std::string& src_mech,
                          const std::string& src_var,
                          int gid,
                          const std::string& section_name,
                          double loc,
                          int array_index,
                          int slot) {
    if (core.sim() == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    const int resolved_index = resolve_gap_index_by_loc(
        core,
        src_mech,
        src_var,
        gid,
        section_name,
        loc,
        slot,
        "add_gap_source_by_loc");
    if (resolved_index < 0) {
        return -1;
    }
    return gap_routes.add_source(src_mech, src_var, resolved_index, sid, array_index);
}

int add_gap_target_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                          GapRouteBuilder& gap_routes,
                          int sid,
                          const std::string& tgt_mech,
                          const std::string& tgt_var,
                          int gid,
                          const std::string& section_name,
                          double loc,
                          int array_index,
                          int slot) {
    if (core.sim() == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    const int resolved_index = resolve_gap_index_by_loc(
        core,
        tgt_mech,
        tgt_var,
        gid,
        section_name,
        loc,
        slot,
        "add_gap_target_by_loc");
    if (resolved_index < 0) {
        return -1;
    }
    return gap_routes.add_target(sid, tgt_mech, tgt_var, resolved_index, array_index);
}

int finalize_gap_junctions(neurong::runtime_api::core::SimRuntimeCore& core,
                           GapRouteBuilder& gap_routes) {
    auto* sim = core.sim();
    if (sim == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return gap_routes.finalize(*sim);
}

int clear_all_gap_junctions(neurong::runtime_api::core::SimRuntimeCore& core,
                            GapRouteBuilder& gap_routes) {
    return gap_routes.clear(core.sim());
}

}  // namespace neurong_network
