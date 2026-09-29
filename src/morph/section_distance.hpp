#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace neurong_morph {

inline constexpr double kDisconnectedDistanceUm = 1e20;

// Minimal section-level topology/metric layout required for NEURON-like distance semantics.
struct SectionDistanceLayout {
    std::unordered_map<std::string, std::size_t> section_name_to_index{};
    std::vector<std::int32_t> section_node_base_id{};
    std::vector<std::int32_t> section_nseg{};
    std::vector<std::int32_t> section_parent_node_id{};
    std::vector<std::int32_t> section_parent_section_id{};
    std::vector<double> section_length_um{};
    std::vector<double> section_diam_um{};
    std::vector<std::int32_t> section_pt3d_offset{};
    std::vector<std::int32_t> section_pt3d_count{};
    std::vector<double> pt3d_arc_um{};
    std::vector<double> pt3d_diam_um{};
    // Set once at layout-build time after structural validation.
    bool valid{false};
};

struct SectionDistancePoint {
    std::size_t section_index{0};
    std::int32_t template_node{-1};
};

[[nodiscard]] bool resolve_segment_index_for_nseg(std::int32_t nseg,
                                                  double loc,
                                                  std::int32_t* out_segment_index);

template <typename SectionIndexMap>
[[nodiscard]] bool resolve_segment_index_by_loc(const SectionIndexMap& section_name_to_index,
                                                const std::vector<std::int32_t>& section_nseg,
                                                const std::string& section_name,
                                                double loc,
                                                std::int32_t* out_segment_index) {
    if (out_segment_index == nullptr) {
        return false;
    }
    const auto it = section_name_to_index.find(section_name);
    if (it == section_name_to_index.end()) {
        return false;
    }
    const auto sec_idx = static_cast<std::size_t>(it->second);
    if (sec_idx >= section_nseg.size()) {
        return false;
    }
    return resolve_segment_index_for_nseg(section_nseg[sec_idx], loc, out_segment_index);
}

template <typename SectionIndexMap>
[[nodiscard]] bool resolve_template_node_by_loc(const SectionIndexMap& section_name_to_index,
                                                const std::vector<std::int32_t>& section_node_base_id,
                                                const std::vector<std::int32_t>& section_nseg,
                                                const std::vector<std::int32_t>& section_parent_node_id,
                                                const std::string& section_name,
                                                double loc,
                                                std::int32_t* out_template_node,
                                                std::size_t* out_section_index = nullptr) {
    if (out_template_node == nullptr || !std::isfinite(loc)) {
        return false;
    }

    const auto it = section_name_to_index.find(section_name);
    if (it == section_name_to_index.end()) {
        return false;
    }
    const auto sec_idx = static_cast<std::size_t>(it->second);
    if (sec_idx >= section_node_base_id.size() ||
        sec_idx >= section_nseg.size() ||
        sec_idx >= section_parent_node_id.size()) {
        return false;
    }

    const int nseg = section_nseg[sec_idx];
    if (nseg <= 0) {
        return false;
    }

    std::int32_t tpl_node = -1;
    if (loc <= 0.0) {
        tpl_node = section_parent_node_id[sec_idx];
    } else if (loc >= 1.0) {
        tpl_node = section_node_base_id[sec_idx] + nseg;
    } else {
        int seg = static_cast<int>(std::floor(loc * static_cast<double>(nseg)));
        seg = std::clamp(seg, 0, nseg - 1);
        tpl_node = section_node_base_id[sec_idx] + seg;
    }
    if (tpl_node < 0) {
        return false;
    }

    *out_template_node = tpl_node;
    if (out_section_index != nullptr) {
        *out_section_index = sec_idx;
    }
    return true;
}

[[nodiscard]] bool resolve_segment_index_by_loc(const SectionDistanceLayout& layout,
                                                const std::string& section_name,
                                                double loc,
                                                std::int32_t* out_segment_index);

[[nodiscard]] bool resolve_template_node_by_loc(const SectionDistanceLayout& layout,
                                                const std::string& section_name,
                                                double loc,
                                                std::int32_t* out_template_node);

[[nodiscard]] bool resolve_distance_point(const SectionDistanceLayout& layout,
                                          const std::string& section_name,
                                          double loc,
                                          SectionDistancePoint* out);

[[nodiscard]] double distance_between_points_um(const SectionDistanceLayout& layout,
                                                const SectionDistancePoint& p1,
                                                const SectionDistancePoint& p2);

[[nodiscard]] double distance_between_locs_um(const SectionDistanceLayout& layout,
                                              const std::string& section_name1,
                                              double loc1,
                                              const std::string& section_name2,
                                              double loc2);

[[nodiscard]] bool resolve_diam_um_by_loc(const SectionDistanceLayout& layout,
                                          const std::string& section_name,
                                          double loc,
                                          double* out_diam_um);

}  // namespace neurong_morph
