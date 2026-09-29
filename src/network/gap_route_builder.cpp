#include "network/gap_route_builder.hpp"

#include "simulate.h"
#include "utils.h"

#include <cstdio>
#include <unordered_set>

namespace neurong_network {

int GapRouteBuilder::add_source(const std::string& mech,
                                const std::string& var,
                                int idx,
                                int sid,
                                int array_index) {
    if (sid < 0) {
        sid = next_sid_counter_++;
        while (gap_junctions_.find(sid) != gap_junctions_.end()) {
            sid = next_sid_counter_++;
        }
    } else {
        if (gap_junctions_.find(sid) != gap_junctions_.end()) {
            printf("Gap junction with sid %d already exists\n", sid);
            return -1;
        }
        if (sid >= next_sid_counter_) {
            next_sid_counter_ = sid + 1;
        }
    }

    GapJunctionMeta meta;
    meta.source = VarDescriptor(mech, var, idx, array_index);
    gap_junctions_[sid] = std::move(meta);
    gap_source_sid_order_.push_back(sid);
    gap_routes_dirty_ = true;
    return sid;
}

int GapRouteBuilder::add_target(int sid,
                                const std::string& mech,
                                const std::string& var,
                                int idx,
                                int array_index) {
    auto it = gap_junctions_.find(sid);
    if (it == gap_junctions_.end()) {
        printf("Gap junction with sid %d does not exist\n", sid);
        return -1;
    }

    VarDescriptor target(mech, var, idx, array_index);

    it->second.targets.push_back(target);
    gap_target_order_.push_back({sid, target});
    gap_routes_dirty_ = true;
    return 0;
}

int GapRouteBuilder::finalize(Simulate& sim) {
    if (!gap_routes_dirty_) {
        return 0;
    }
    if (sim.neuron_group_list.empty()) {
        printf("finalize_gap_junctions: neuron_group_list is empty\n");
        return -1;
    }

    auto* p_group = sim.neuron_group_list[0];
    p_group->cpu_gap_trans_info.clear();
    if (sim.mode == GPU) {
        p_group->gpu_gap_trans_info.clear();
    }
    p_group->have_gap = false;

    gap_setup_info_.clear();

    for (int sid : gap_source_sid_order_) {
        auto it = gap_junctions_.find(sid);
        if (it == gap_junctions_.end()) {
            continue;
        }
        if (gap_setup_info_.src_sid_map.find(sid) != gap_setup_info_.src_sid_map.end()) {
            continue;
        }
        const int src_id = static_cast<int>(gap_setup_info_.src_sid.size());
        gap_setup_info_.src_sid_map[sid] = src_id;
        gap_setup_info_.src_sid.push_back(sid);
        gap_setup_info_.src_desc.push_back(it->second.source);
    }

    for (const auto& [sid, meta] : gap_junctions_) {
        if (gap_setup_info_.src_sid_map.find(sid) != gap_setup_info_.src_sid_map.end()) {
            continue;
        }
        const int src_id = static_cast<int>(gap_setup_info_.src_sid.size());
        gap_setup_info_.src_sid_map[sid] = src_id;
        gap_setup_info_.src_sid.push_back(sid);
        gap_setup_info_.src_desc.push_back(meta.source);
    }

    for (const auto& [sid, desc] : gap_target_order_) {
        if (gap_setup_info_.src_sid_map.find(sid) == gap_setup_info_.src_sid_map.end()) {
            printf("finalize_gap_junctions: target references unknown sid=%d\n", sid);
            return -1;
        }
        gap_setup_info_.tar_sid.push_back(sid);
        gap_setup_info_.tar_desc.push_back(desc);
    }

    gap_setup_info_.ntrans = static_cast<int>(gap_setup_info_.tar_sid.size());
    const int ntrans = gap_setup_info_.ntrans;
    if (ntrans == 0) {
        gap_routes_dirty_ = false;
        return 0;
    }

    auto& cpu_gap = p_group->cpu_gap_trans_info;
    auto& gpu_gap = p_group->gpu_gap_trans_info;
    cpu_gap.src.reserve(static_cast<size_t>(ntrans));
    cpu_gap.dst.reserve(static_cast<size_t>(ntrans));
    cpu_gap.src_desc.reserve(static_cast<size_t>(ntrans));
    cpu_gap.dst_desc.reserve(static_cast<size_t>(ntrans));
    if (sim.mode == GPU) {
        gpu_gap.src.reserve(static_cast<size_t>(ntrans));
        gpu_gap.dst.reserve(static_cast<size_t>(ntrans));
        gpu_gap.src_desc.reserve(static_cast<size_t>(ntrans));
        gpu_gap.dst_desc.reserve(static_cast<size_t>(ntrans));
    }

    std::unordered_set<double*> dst_seen_cpu;
    std::unordered_set<double*> dst_seen_gpu;
    dst_seen_cpu.reserve(static_cast<size_t>(ntrans) * 2);
    if (sim.mode == GPU) {
        dst_seen_gpu.reserve(static_cast<size_t>(ntrans) * 2);
    }

    for (int i = 0; i < ntrans; ++i) {
        const int sid = gap_setup_info_.tar_sid[static_cast<size_t>(i)];
        auto src_it = gap_setup_info_.src_sid_map.find(sid);
        if (src_it == gap_setup_info_.src_sid_map.end()) {
            printf("finalize_gap_junctions: internal sid map missing sid=%d\n", sid);
            return -1;
        }
        const int src_id = src_it->second;
        VarDescriptor src_desc = gap_setup_info_.src_desc[static_cast<size_t>(src_id)];
        VarDescriptor dst_desc = gap_setup_info_.tar_desc[static_cast<size_t>(i)];

        auto [src_cpu, src_gpu] = sim.getVarPtr(src_desc, false);
        auto [dst_cpu, dst_gpu] = sim.getVarPtr(dst_desc, false);
        if (src_cpu == nullptr || dst_cpu == nullptr) {
            printf("finalize_gap_junctions: failed to resolve cpu pointers (sid=%d)\n", sid);
            return -1;
        }
        if (sim.mode == GPU && (src_gpu == nullptr || dst_gpu == nullptr)) {
            printf("finalize_gap_junctions: failed to resolve gpu pointers (sid=%d)\n", sid);
            return -1;
        }

        if (!dst_seen_cpu.insert(dst_cpu).second) {
            printf("finalize_gap_junctions: duplicated target pointer on CPU (sid=%d)\n", sid);
            return -1;
        }
        cpu_gap.add_gap(src_cpu, dst_cpu, src_desc, dst_desc);

        if (sim.mode == GPU) {
            if (!dst_seen_gpu.insert(dst_gpu).second) {
                printf("finalize_gap_junctions: duplicated target pointer on GPU (sid=%d)\n", sid);
                return -1;
            }
            gpu_gap.add_gap(src_gpu, dst_gpu, src_desc, dst_desc);
        }
    }

    p_group->have_gap = true;
    gap_routes_dirty_ = false;
    return 0;
}

int GapRouteBuilder::clear(Simulate* sim) {
    gap_junctions_.clear();
    next_sid_counter_ = 0;
    gap_source_sid_order_.clear();
    gap_target_order_.clear();
    gap_setup_info_.clear();
    gap_routes_dirty_ = false;

    if (sim == nullptr || sim->neuron_group_list.empty()) {
        return 0;
    }

    auto* p_group = sim->neuron_group_list[0];
    p_group->cpu_gap_trans_info.clear();
    if (sim->mode == GPU) {
        p_group->gpu_gap_trans_info.clear();
    }
    p_group->have_gap = false;
    return 0;
}

const GapJunctionMeta* GapRouteBuilder::get(int sid) const {
    const auto it = gap_junctions_.find(sid);
    if (it == gap_junctions_.end()) {
        return nullptr;
    }
    return &it->second;
}

int GapRouteBuilder::get_next_available_sid() {
    return next_sid_counter_++;
}

}  // namespace neurong_network
