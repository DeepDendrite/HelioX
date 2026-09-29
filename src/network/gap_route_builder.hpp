#pragma once

#include "neuron.h"

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Simulate;

namespace neurong_network {

struct GapJunctionMeta {
    VarDescriptor source;
    std::vector<VarDescriptor> targets;
};

class GapRouteBuilder final {
public:
    int add_source(const std::string& mech,
                   const std::string& var,
                   int idx,
                   int sid = -1,
                   int array_index = -1);
    int add_target(int sid,
                   const std::string& mech,
                   const std::string& var,
                   int idx,
                   int array_index = -1);

    int finalize(Simulate& sim);
    int clear(Simulate* sim = nullptr);

    const std::map<int, GapJunctionMeta>& get_all() const { return gap_junctions_; }
    const GapJunctionMeta* get(int sid) const;
    int get_next_available_sid();

private:
    std::map<int, GapJunctionMeta> gap_junctions_;
    int next_sid_counter_ = 0;
    std::vector<int> gap_source_sid_order_;
    std::vector<std::pair<int, VarDescriptor>> gap_target_order_;
    bool gap_routes_dirty_ = false;

    struct GapSetupTransferInfo {
        int ntrans = 0;
        std::unordered_map<int, int> src_sid_map;
        std::vector<int> src_sid;
        std::vector<VarDescriptor> src_desc;
        std::vector<int> tar_sid;
        std::vector<VarDescriptor> tar_desc;

        void clear() {
            ntrans = 0;
            src_sid_map.clear();
            src_sid.clear();
            src_desc.clear();
            tar_sid.clear();
            tar_desc.clear();
        }
    };
    GapSetupTransferInfo gap_setup_info_;
};

}  // namespace neurong_network
