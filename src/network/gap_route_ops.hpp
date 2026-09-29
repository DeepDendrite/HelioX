#pragma once

#include "network/gap_route_builder.hpp"

#include <string>

namespace neurong::runtime_api::core {
class SimRuntimeCore;
}

namespace neurong_network {

int add_gap_source_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                          GapRouteBuilder& gap_routes,
                          int sid,
                          const std::string& src_mech,
                          const std::string& src_var,
                          int gid,
                          const std::string& section_name,
                          double loc,
                          int array_index = -1,
                          int slot = 0);

int add_gap_target_by_loc(neurong::runtime_api::core::SimRuntimeCore& core,
                          GapRouteBuilder& gap_routes,
                          int sid,
                          const std::string& tgt_mech,
                          const std::string& tgt_var,
                          int gid,
                          const std::string& section_name,
                          double loc,
                          int array_index = -1,
                          int slot = 0);

int finalize_gap_junctions(neurong::runtime_api::core::SimRuntimeCore& core,
                           GapRouteBuilder& gap_routes);

int clear_all_gap_junctions(neurong::runtime_api::core::SimRuntimeCore& core,
                            GapRouteBuilder& gap_routes);

}  // namespace neurong_network
