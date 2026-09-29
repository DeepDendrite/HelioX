#pragma once

#include "iclamp_spec.hpp"
#include "biophys_prepare.hpp"
#include "section_to_node.hpp"
#include "section_label_layout.hpp"
#include "neuron.h"
#include "simulate.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace neurong_biophysical {

using neurong_morph::NodeBuildLayout;
using neurong_morph::section_id;

// Parsed biophysical insertion op:
//   - label: section label name (e.g. "soma") or "all"
//   - mech: mechanism name
//   - params: optional per-label parameter overrides for this mech
using BiophysInsertOp = InsertOp;

// Morphology/layout metadata for a population built from multiple cell templates.
//
// The model is represented as one NeuronGroupData (single group) containing:
//   - root nodes: indices [0 .. num_cells_total)
//   - non-root nodes: indices [num_cells_total .. nnode)
//
// Each cell belongs to exactly one template. Template-local node indices are mapped to
// global original node indices via:
//   - root: template_node_index == 0  -> global == cell_id
//   - non-root: template_node_index > 0 -> global == cell_nonroot_base[cell_id] + (template_node_index - 1)
struct CellTemplateInfo {
    std::string name{};
    int cell_base{0};   // starting global cell_id for this template
    int num_cells{0};   // number of instantiated cells for this template

    // Template metadata (single-cell, root-first with exactly 1 root).
    std::size_t roots_per_cell{1};
    std::size_t tpl_nodes_per_cell{0};
    std::size_t tpl_nonroots_per_cell{0};

    // Template node index of soma(0.5) used for recording/stim placement.
    std::int32_t tpl_soma_node{-1};

    // Template section/layout data for per-label expansion.
    std::vector<std::string> label_names{};
    std::unordered_map<std::string, std::size_t> label_index{};
    std::vector<std::vector<section_id>> sections_by_label{};
    std::unordered_map<std::string, section_id> section_name_to_id{};
    std::vector<std::int32_t> section_node_base_id{};
    std::vector<std::int32_t> section_nseg{};
    std::vector<std::int32_t> section_parent_node_id{};
    // Section topology in section-id space (-1 for roots).
    std::vector<std::int32_t> section_parent_section_id{};
    // Physical section lengths in micrometers, aligned with section indices.
    std::vector<double> section_length_um{};
    std::vector<double> section_ra{};
    std::vector<double> section_cm{};
    std::uint64_t random_seed{0};
    ScalarOrRandom v_init{};

    // Per-template biophys insert ops.
    Prepare prepare{};
};

struct CellTemplateMorphLayout {
    int num_cells_total{0};
    std::size_t nnode{0};  // population node count (after instantiation, before/after permute)

    std::vector<CellTemplateInfo> templates{};

    // Per-cell mapping (length == num_cells_total).
    std::vector<std::size_t> cell_template_id{};
    std::vector<std::size_t> cell_nonroot_base{};
};

struct NeuronGBiophysBuildResult {
    std::map<VarDescriptor, int> monitor_to_handle{};
    std::vector<std::tuple<int, int, int>> record_handles{};
};

// -----------------------------------------------------------------------------
// Label-based expansion helpers
//
// Expand NEURON-like "insert <mech> by label" operations into per-node instance
// lists aligned with the node layout from `build_nodes_neuron_compatible_with_layout()`.
// -----------------------------------------------------------------------------

// One-shot expansion output: per-node instance lists (SoA-friendly) for each inserted
// distributed mechanism, aligned with the node ids produced by the morph builder.
//
// `node_indices` includes only segment nodes (j in [0, nseg-1]) for each section
// (i.e., boundary nodes and synthetic roots are excluded).
//
// The build preserves label-major ordering, enabling fast per-label range updates:
//   instances for a given label are the contiguous range
//     [label_offset[label], label_offset[label] + label_count[label])
struct MechanismInstanceLayout {
    std::string mech{};
    std::vector<std::int32_t> node_indices{};
    std::vector<std::size_t> label_offset{};
    std::vector<std::size_t> label_count{};
    std::vector<std::uint8_t> inserted_by_label{};
    // Flattened per-label segment metadata for fast section/segment -> instance range mapping.
    // For a label `u`, segment entries are in:
    //   [label_segment_base[u], label_segment_base[u+1])
    // with local segment index:
    //   seg_local = section_offset_in_label + seg_index_in_section
    // and:
    //   seg_flat = label_segment_base[u] + seg_local
    //   local_instance_offset = label_segment_offset[seg_flat]
    //   instance_count_on_segment = label_segment_count[seg_flat]
    std::vector<std::size_t> label_segment_base{};
    std::vector<std::size_t> label_segment_offset{};
    std::vector<std::size_t> label_segment_count{};
    std::size_t instance_count{0};
};

struct BuildResult {
    std::vector<MechanismInstanceLayout> mechanisms{};
};

// Expand a label-based prepare into per-node mechanism instance layouts, using only:
//  - sections_by_label (built once during morphology construction)
//  - NodeBuildLayout (built once during Section->Node lowering)
//
// This avoids any extra "scan all sections" pass when inserting channels.
[[nodiscard]] inline BuildResult build_mechanism_instances(
    const std::unordered_map<std::string, std::size_t>& label_index,
    const std::vector<std::vector<section_id>>& sections_by_label,
    const std::unordered_map<std::string, section_id>& section_name_to_id,
    const NodeBuildLayout& layout,
    const Prepare& prepare,
    const neurong_morph::SectionLabelSegmentLayout& section_label_layout) {
    const auto& section_label_u_by_sec = section_label_layout.section_label_u_by_sec;
    const auto& section_offset_in_label_by_sec = section_label_layout.section_offset_in_label_by_sec;
    const auto& label_segment_count = section_label_layout.label_segment_count;
    const std::size_t max_label_plus_one = sections_by_label.size();
    const auto num_sections = layout.section_nseg.size();
    if (layout.section_node_base_id.size() != num_sections) {
        throw std::runtime_error("section layout size mismatch while building mechanism instances");
    }
    if (section_label_u_by_sec.size() != num_sections ||
        section_offset_in_label_by_sec.size() != num_sections) {
        throw std::runtime_error("section label mapping size mismatch while building mechanism instances");
    }
    if (label_segment_count.size() != max_label_plus_one) {
        throw std::runtime_error("label segment count size mismatch while building mechanism instances");
    }

    auto to_non_negative_integer = [](double value, std::string_view ctx) -> std::size_t {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error(std::string(ctx) + " must be a finite non-negative integer");
        }
        const double rounded = std::llround(value);
        if (std::fabs(value - rounded) > 1e-9) {
            throw std::runtime_error(std::string(ctx) + " must be an integer value");
        }
        return static_cast<std::size_t>(rounded);
    };

    auto build_one = [&](std::string_view mech_name) -> std::optional<MechanismInstanceLayout> {
        std::vector<std::vector<std::size_t>> segment_instances(max_label_plus_one);
        for (std::size_t label_u = 0; label_u < max_label_plus_one; ++label_u) {
            segment_instances[label_u].assign(label_segment_count[label_u], 0);
        }

        bool has_matching_op = false;

        for (const auto& op : prepare.inserts()) {
            if (op.mech != mech_name) {
                continue;
            }
            has_matching_op = true;

            const bool select_all = (op.label == AllLabels);
            std::optional<std::size_t> selected_label_u{};
            if (!select_all) {
                auto it = label_index.find(op.label);
                if (it == label_index.end()) {
                    throw std::runtime_error("unknown label name in insert: " + op.label);
                }
                selected_label_u = it->second;
            }

            const ParamValue* instances_value = nullptr;
            for (const auto& [k, v] : op.params) {
                if (k == "instances") {
                    instances_value = &v;
                    break;
                }
            }

            auto apply_scalar_instances = [&](std::size_t count, bool additive) {
                if (count == 0) {
                    return;
                }
                for (std::size_t label_u = 0; label_u < max_label_plus_one; ++label_u) {
                    if (!select_all && (!selected_label_u.has_value() || label_u != *selected_label_u)) {
                        continue;
                    }
                    auto& per_seg = segment_instances[label_u];
                    for (auto& x : per_seg) {
                        if (additive) {
                            x += count;
                        } else if (x < count) {
                            x = count;
                        }
                    }
                }
            };

            if (instances_value == nullptr) {
                // Default NEURON-like insertion: one instance per segment in selected labels.
                apply_scalar_instances(1, /*additive=*/false);
                continue;
            }

            if (std::holds_alternative<SectionParamMap>(*instances_value)) {
                const auto& section_map = std::get<SectionParamMap>(*instances_value);
                for (const auto& [sec_name, sec_entry] : section_map) {
                    auto it_sec = section_name_to_id.find(sec_name);
                    if (it_sec == section_name_to_id.end()) {
                        throw std::runtime_error("unknown section name in instances map: " + sec_name);
                    }
                    const auto sec_index = static_cast<std::size_t>(it_sec->second);
                    if (sec_index >= num_sections) {
                        throw std::runtime_error("section index out of range in instances map: " + sec_name);
                    }

                    const auto label_u_i32 = section_label_u_by_sec[sec_index];
                    if (label_u_i32 < 0) {
                        throw std::runtime_error("section has no label in instances map: " + sec_name);
                    }
                    const auto label_u = static_cast<std::size_t>(label_u_i32);
                    if (!select_all && (!selected_label_u.has_value() || label_u != *selected_label_u)) {
                        throw std::runtime_error("section " + sec_name + " not in selected label for instances map");
                    }

                    const auto nseg_i32 = layout.section_nseg[sec_index];
                    if (nseg_i32 < 0) {
                        throw std::runtime_error("invalid nseg for section in instances map: " + sec_name);
                    }
                    const auto nseg = static_cast<std::size_t>(nseg_i32);
                    const auto base = section_offset_in_label_by_sec[sec_index];
                    auto& per_seg = segment_instances[label_u];

                    if (std::holds_alternative<RandomParamSpec>(sec_entry)) {
                        throw std::runtime_error(
                            "instances map does not support random specs (section " + sec_name + ")");
                    }

                    const auto& values = std::get<std::vector<double>>(sec_entry);
                    if (nseg == 0) {
                        if (!values.empty()) {
                            throw std::runtime_error("instances map must be empty for nseg==0 section " + sec_name);
                        }
                        continue;
                    }

                    if (values.size() == 1) {
                        const auto count = to_non_negative_integer(values.front(), "instances scalar map entry");
                        for (std::size_t j = 0; j < nseg; ++j) {
                            per_seg[base + j] += count;
                        }
                        continue;
                    }

                    if (values.size() != nseg) {
                        throw std::runtime_error("instances map size mismatch for section " + sec_name);
                    }

                    for (std::size_t j = 0; j < nseg; ++j) {
                        const auto count = to_non_negative_integer(values[j], "instances per-segment map entry");
                        per_seg[base + j] += count;
                    }
                }
                continue;
            }

            if (std::holds_alternative<double>(*instances_value)) {
                const auto count = to_non_negative_integer(
                    std::get<double>(*instances_value), "instances");
                apply_scalar_instances(count, /*additive=*/true);
                continue;
            }

            throw std::runtime_error(
                "instances must be scalar or section map for mechanism " + std::string(mech_name));
        }

        if (!has_matching_op) {
            return std::nullopt;
        }

        std::vector<std::size_t> label_offset(max_label_plus_one, 0);
        std::vector<std::size_t> label_count(max_label_plus_one, 0);
        std::vector<std::uint8_t> inserted_by_label(max_label_plus_one, static_cast<std::uint8_t>(0));
        std::vector<std::size_t> label_segment_base(max_label_plus_one + 1, 0);
        std::vector<std::size_t> label_segment_offset{};
        std::vector<std::size_t> label_segment_count_flat{};

        std::size_t total_segments_flat = 0;
        for (const auto& per_seg : segment_instances) {
            total_segments_flat += per_seg.size();
        }
        label_segment_offset.reserve(total_segments_flat);
        label_segment_count_flat.reserve(total_segments_flat);

        std::size_t total_instances = 0;
        std::size_t seg_flat_cursor = 0;
        for (std::size_t label_u = 0; label_u < max_label_plus_one; ++label_u) {
            label_offset[label_u] = total_instances;
            label_segment_base[label_u] = seg_flat_cursor;

            std::size_t running = 0;
            const auto& per_seg = segment_instances[label_u];
            for (const auto seg_count : per_seg) {
                label_segment_offset.push_back(running);
                label_segment_count_flat.push_back(seg_count);
                running += seg_count;
            }
            seg_flat_cursor += per_seg.size();

            label_count[label_u] = running;
            inserted_by_label[label_u] = (running > 0) ? static_cast<std::uint8_t>(1)
                                                       : static_cast<std::uint8_t>(0);
            total_instances += running;
        }
        label_segment_base[max_label_plus_one] = seg_flat_cursor;

        if (total_instances == 0) {
            return std::nullopt;
        }

        MechanismInstanceLayout out{};
        out.mech = std::string(mech_name);
        out.instance_count = total_instances;
        out.node_indices.resize(total_instances);
        out.label_offset = std::move(label_offset);
        out.label_count = std::move(label_count);
        out.inserted_by_label = std::move(inserted_by_label);
        out.label_segment_base = std::move(label_segment_base);
        out.label_segment_offset = std::move(label_segment_offset);
        out.label_segment_count = std::move(label_segment_count_flat);

        // Fill node indices (label-major order, and within label segment-major with multiplicity).
        for (std::size_t label_u = 0; label_u < max_label_plus_one; ++label_u) {
            if (!out.inserted_by_label[label_u] || out.label_count[label_u] == 0) {
                continue;
            }
            std::size_t dst = out.label_offset[label_u];
            const auto& per_seg = segment_instances[label_u];
            for (const auto section : sections_by_label[label_u]) {
                const auto sec_index = static_cast<std::size_t>(section);
                const auto base = layout.section_node_base_id[sec_index];
                const auto nseg = layout.section_nseg[sec_index];
                const auto seg_offset = section_offset_in_label_by_sec[sec_index];
                for (std::int32_t j = 0; j < nseg; ++j) {
                    const auto multiplicity = per_seg[seg_offset + static_cast<std::size_t>(j)];
                    for (std::size_t rep = 0; rep < multiplicity; ++rep) {
                        out.node_indices[dst++] = base + j;
                    }
                }
            }
            if (dst != out.label_offset[label_u] + out.label_count[label_u]) {
                throw std::runtime_error("internal error while materializing mechanism instances");
            }
        }
        return out;
    };

    // Unique mechanisms in first-seen order from prepare ops.
    std::vector<std::string_view> mechs;
    std::unordered_set<std::string_view> seen;
    for (const auto& op : prepare.inserts()) {
        if (seen.insert(op.mech).second) {
            mechs.push_back(op.mech);
        }
    }
    BuildResult out{};
    for (const auto mech : mechs) {
        if (auto built = build_one(mech)) {
            out.mechanisms.push_back(std::move(*built));
        }
    }
    return out;
}

// Build a runnable NeuronG model from:
//  - an already-built morphology (nodes + permute + group)
//  - a list of NEURON-like biophys insert ops
//
// This function mutates `group` by creating ions/capacitance/distributed mechanisms
// and returns monitor handles for requested record gids.
//
// Note: the in-memory biophysics builder currently supports GPU mode only.
[[nodiscard]] NeuronGBiophysBuildResult build_neurong_biophysics(Simulate& sim,
                                                                 NeuronGroupData& group,
                                                                 const CellTemplateMorphLayout& morph,
                                                                 double celsius,
                                                                 const std::vector<IClampSpec>& iclamp,
                                                                 const std::vector<int>& record_gids,
                                                                 const std::vector<VarDescriptor>& extra_monitors,
                                                                 int& next_type);

}  // namespace neurong_biophysical
