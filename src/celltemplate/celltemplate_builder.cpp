#include "celltemplate_builder.hpp"

#include "mutable_section.hpp"
#include "permute_order.h"
#include "section_to_node.hpp"
#include "asc_to_section.hpp"
#include "swc_to_section.hpp"
#include "label_utils.hpp"
#include "biophysical/random_utils.hpp"

#include "mechanism.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace neurong_celltemplate {
namespace {

using neurong_morph::Morph;
using neurong_morph::NodeBuildConfig;
using neurong_morph::NodeBuildLayout;
using neurong_morph::NodeBuildResult;
using neurong_morph::NodeCoreSoA;
using neurong_morph::section_id;

[[nodiscard]] std::size_t checked_node_count(const NodeCoreSoA& nodes) {
    return nodes.parent_id.size();
}

[[nodiscard]] std::int32_t pick_soma_node_index(const std::vector<std::vector<section_id>>& sections_by_label,
                                                const std::unordered_map<std::string, std::size_t>& label_index,
                                                const NodeBuildLayout& layout) {
    auto it = label_index.find("soma");
    if (it == label_index.end()) {
        throw std::runtime_error("missing 'soma' label for template");
    }
    const auto soma_label_u = it->second;
    if (soma_label_u >= sections_by_label.size() || sections_by_label[soma_label_u].empty()) {
        throw std::runtime_error("empty 'soma' label for template");
    }
    const auto soma_sec = sections_by_label[soma_label_u].front();
    const auto sec_index = static_cast<std::size_t>(soma_sec);
    const auto base = layout.section_node_base_id[sec_index];
    const auto nseg = layout.section_nseg[sec_index];
    const auto j = std::min<std::int32_t>(nseg - 1, static_cast<std::int32_t>(static_cast<double>(nseg) * 0.5));
    return base + j;
}

struct SectionBuildPlan {
    std::vector<std::size_t> order{};
    std::unordered_map<std::string, std::size_t> name_to_spec{};
};

SectionBuildPlan prepare_section_build(const std::vector<SectionSpec>& sections) {
    SectionBuildPlan plan{};
    for (std::size_t i = 0; i < sections.size(); ++i) {
        const auto& sec = sections[i];
        const auto [it, inserted] = plan.name_to_spec.emplace(sec.name, i);
    }

    std::vector<std::uint8_t> state(sections.size(), 0);

    std::function<void(std::size_t)> dfs = [&](std::size_t idx) {
        const auto s = state[idx];
        if (s == 2) {
            return;
        }
        state[idx] = 1;
        const auto& sec = sections[idx];
        if (!sec.parent_name.empty()) {
            const auto it = plan.name_to_spec.find(sec.parent_name);
            dfs(it->second);
        }
        state[idx] = 2;
        plan.order.push_back(idx);
    };

    for (std::size_t i = 0; i < sections.size(); ++i) {
        dfs(i);
    }
    return plan;
}

Morph build_morph_from_sections(const std::vector<SectionSpec>& sections, SectionNseg& out_nseg) {
    const auto plan = prepare_section_build(sections);
    Morph morph{};
    out_nseg.clear();

    std::unordered_map<std::string, section_id> name_to_id;

    for (const auto idx : plan.order) {
        const auto& spec = sections[idx];
        const section_id parent = spec.parent_name.empty()
                                      ? neurong_morph::invalid_section_id
                                      : name_to_id.at(spec.parent_name);
        section_id new_id = neurong_morph::invalid_section_id;
        if (!spec.pt3d.empty()) {
            std::vector<neurong_morph::Pt3dPoint> points;
            for (const auto& p : spec.pt3d) {
                points.push_back(neurong_morph::Pt3dPoint{
                    .x_um = p.x_um,
                    .y_um = p.y_um,
                    .z_um = p.z_um,
                    .diam_um = p.diam_um,
                });
            }
            new_id = neurong_morph::append_section_with_pt3d(
                morph,
                spec.name,
                parent,
                spec.parentx,
                spec.label,
                std::span<const neurong_morph::Pt3dPoint>(points.data(), points.size()));
        } else {
            new_id = neurong_morph::append_cylindrical_section_no_pt3d(morph,
                                                                      spec.name,
                                                                      parent,
                                                                      spec.parentx,
                                                                      spec.label,
                                                                      spec.L_um,
                                                                      spec.diam_um);
        }
        name_to_id.emplace(spec.name, new_id);
        out_nseg.push_back(spec.nseg);
    }
    return morph;
}

void apply_section_nseg(Morph& morph, const SectionNseg& section_nseg) {
    // NEURON default: nseg==1 for every section unless explicitly overridden.
    if (section_nseg.empty()) {
        for (auto& sec : morph.sections) {
            sec.nseg = 1;
        }
        return;
    }
    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        const auto nseg = section_nseg[i];
        morph.sections[i].nseg = nseg;
    }
}

template <typename T>
[[nodiscard]] std::vector<T> permute_by_index(const std::vector<T>& values, const int* p, std::size_t nnode) {
    std::vector<T> out;
    out.resize(nnode);
    for (std::size_t i = 0; i < nnode; ++i) {
        const auto pi = static_cast<std::size_t>(p[i]);
        out[pi] = values[i];
    }
    return out;
}

[[nodiscard]] std::vector<int> permute_parent_indices(const std::vector<std::int32_t>& parent_old,
                                                      const int* p,
                                                      std::size_t nnode) {
    std::vector<int> parent_new;
    parent_new.resize(nnode);
    for (std::size_t i = 0; i < nnode; ++i) {
        const auto pi = static_cast<std::size_t>(p[i]);
        const std::int32_t parent = parent_old[i];
        if (parent < 0) {
            parent_new[pi] = -1;
        } else {
            parent_new[pi] = p[static_cast<std::size_t>(parent)];
        }
    }
    return parent_new;
}

struct BuiltTemplate {
    std::string name{};
    int num_cells{0};
    std::uint64_t random_seed{0};
    std::vector<std::string> label_names{};
    std::unordered_map<std::string, std::size_t> label_index{};
    std::vector<std::vector<section_id>> sections_by_label{};
    std::unordered_map<std::string, section_id> section_name_to_id{};
    std::vector<std::int32_t> section_parent_section_id{};
    std::vector<double> section_length_um{};
    std::vector<double> section_ra{};
    std::vector<double> section_cm{};
    NodeBuildResult tpl{};
    std::int32_t tpl_soma_node{-1};
};

struct PopulationLayout {
    int num_cells_total{0};
    std::vector<std::size_t> cell_template_id{};
    std::vector<std::size_t> cell_nonroot_base{};
};

[[nodiscard]] std::int32_t map_template_node_to_original(const PopulationLayout& layout,
                                                         std::size_t cell_id,
                                                         std::int32_t template_node_index) {
    if (template_node_index < 0) {
        return -1;
    }
    if (template_node_index == 0) {
        return static_cast<std::int32_t>(cell_id);
    }
    const auto base = layout.cell_nonroot_base[cell_id];
    const auto idx = base + (static_cast<std::size_t>(template_node_index) - 1);
    return static_cast<std::int32_t>(idx);
}

[[nodiscard]] double sample_template_v_init(const neurong_biophysical::ScalarOrRandom& spec,
                                            std::uint64_t tpl_seed,
                                            std::size_t global_cell_id) {
    if (std::holds_alternative<double>(spec)) {
        return std::get<double>(spec);
    }
    const auto& rule = std::get<neurong_biophysical::RandomParamSpec>(spec);
    std::uint64_t seed = neurong_biophysical::combine_seed_many(
        tpl_seed,
        neurong_biophysical::hash_string64("v_init"),
        static_cast<std::uint64_t>(global_cell_id));
    return neurong_biophysical::sample_random_rule(rule, seed);
}

}  // namespace

std::vector<SectionSpec> load_swc_sections(const std::string& swc_file) {
    Morph morph = neurong_morph::load_swc_morphology(swc_file);
    neurong_morph::build_pt3d_geometry(morph);

    std::vector<SectionSpec> out;
    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        const auto& sec = morph.sections[i];
        SectionSpec spec{};
        spec.name = sec.name;
        if (sec.parent_sec_id != neurong_morph::invalid_section_id) {
            spec.parent_name = morph.sections[static_cast<std::size_t>(sec.parent_sec_id)].name;
        }
        spec.parentx = sec.parentx;
        spec.label = sec.label;
        spec.nseg = 1;
        spec.L_um = sec.L_um;
        spec.diam_um = sec.diam_um;

        const bool implicit_cylinder = (sec.count == 1 && sec.L_um > 0.0 && sec.pt3d_count == 0);
        if (implicit_cylinder) {
            out.push_back(std::move(spec));
            continue;
        }

        const bool fix_first_diam =
            (!sec.wire_first && sec.parent_sec_id != neurong_morph::invalid_section_id &&
             neurong_labels::is_soma_label(morph.sections[sec.parent_sec_id].label) &&
             !neurong_labels::is_soma_label(sec.label));

        if (sec.count == 1) {
            const auto swc_id = morph.datas[sec.offset];
            const auto& s = morph.swc[static_cast<std::size_t>(swc_id)];
            const float d_um = static_cast<float>(s.d_um);
            const float y = static_cast<float>(s.xyz[1]);
            const float z = static_cast<float>(s.xyz[2]);
            const float half = 0.5f * d_um;
            const float x0 = static_cast<float>(s.xyz[0]) - half;
            const float x1 = static_cast<float>(s.xyz[0]);
            const float x2 = static_cast<float>(s.xyz[0]) + half;
            spec.pt3d.push_back(SectionPt3d{x0, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x1, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x2, y, z, d_um});
            out.push_back(std::move(spec));
            continue;
        }

        const auto first_id = morph.datas[sec.offset];
        const auto second_id = morph.datas[sec.offset + 1];
        const float first_diam = static_cast<float>(morph.swc[static_cast<std::size_t>(second_id)].d_um);

        for (std::size_t j = 0; j < sec.count; ++j) {
            const auto swc_id = morph.datas[sec.offset + j];
            const auto& s = morph.swc[static_cast<std::size_t>(swc_id)];
            float d_um = static_cast<float>(s.d_um);
            if (fix_first_diam && j == 0) {
                d_um = first_diam;
            }
            spec.pt3d.push_back(
                SectionPt3d{static_cast<float>(s.xyz[0]), static_cast<float>(s.xyz[1]), static_cast<float>(s.xyz[2]), d_um});
        }
        out.push_back(std::move(spec));
    }
    return out;
}

std::vector<SectionSpec> load_asc_sections(const std::string& asc_file) {
    Morph morph = neurong_morph::load_asc_morphology(asc_file);

    std::vector<SectionSpec> out;
    out.reserve(morph.sections.size());

    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        const auto& sec = morph.sections[i];
        SectionSpec spec{};
        spec.name = sec.name;
        if (sec.parent_sec_id != neurong_morph::invalid_section_id) {
            spec.parent_name = morph.sections[static_cast<std::size_t>(sec.parent_sec_id)].name;
        }
        spec.parentx = sec.parentx;
        spec.label = sec.label;
        spec.nseg = 1;
        spec.L_um = sec.L_um;
        spec.diam_um = sec.diam_um;

        if (sec.pt3d_count >= 2) {
            for (std::size_t j = 0; j < sec.pt3d_count; ++j) {
                const auto idx = sec.pt3d_offset + j;
                spec.pt3d.push_back(SectionPt3d{
                    morph.pt3d.x[idx],
                    morph.pt3d.y[idx],
                    morph.pt3d.z[idx],
                    morph.pt3d.d[idx],
                });
            }
        } else if (sec.pt3d_count == 1) {
            const auto idx = sec.pt3d_offset;
            const float d_um = morph.pt3d.d[idx];
            const float half = 0.5f * d_um;
            const float x = morph.pt3d.x[idx];
            const float y = morph.pt3d.y[idx];
            const float z = morph.pt3d.z[idx];
            spec.pt3d.push_back(SectionPt3d{x - half, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x + half, y, z, d_um});
        } else {
            if (!(spec.L_um > 0.0) || !std::isfinite(spec.L_um)) {
                spec.L_um = 1.0;
            }
            if (!(spec.diam_um > 0.0) || !std::isfinite(spec.diam_um)) {
                spec.diam_um = 1.0;
            }
        }

        out.push_back(std::move(spec));
    }

    return out;
}

std::vector<SectionSpec> sections_with_label(const std::vector<SectionSpec>& sections,
                                             const std::string& label) {
    if (sections.empty() || label.empty()) {
        return {};
    }

    std::unordered_map<std::string, const SectionSpec*> by_name;
    by_name.reserve(sections.size());
    for (const auto& sec : sections) {
        if (sec.name.empty()) {
            throw std::runtime_error("sections_with_label: section name must not be empty");
        }
        const auto [_, inserted] = by_name.emplace(sec.name, &sec);
        if (!inserted) {
            throw std::runtime_error("sections_with_label: duplicate section name: " + sec.name);
        }
    }

    SectionNseg nseg;
    Morph morph = build_morph_from_sections(sections, nseg);
    const auto sec_ids = neurong_morph::sections_with_label(morph, label);

    std::vector<SectionSpec> out;
    out.reserve(sec_ids.size());
    for (const auto sec_id : sec_ids) {
        const auto sec_u = static_cast<std::size_t>(sec_id);
        if (sec_u >= morph.sections.size()) {
            throw std::runtime_error("sections_with_label: internal section id out of range");
        }
        const auto& name = morph.sections[sec_u].name;
        const auto it = by_name.find(name);
        if (it == by_name.end() || it->second == nullptr) {
            throw std::runtime_error("sections_with_label: internal section name lookup failed: " + name);
        }
        out.push_back(*it->second);
    }
    return out;
}

neurong_morph::SectionDistanceLayout build_section_distance_layout(
    const std::vector<SectionSpec>& sections) {
    if (sections.empty()) {
        throw std::runtime_error("build_section_distance_layout requires non-empty sections");
    }

    SectionNseg section_nseg;
    Morph morph = build_morph_from_sections(sections, section_nseg);
    apply_section_nseg(morph, section_nseg);

    neurong_morph::SectionDistanceLayout out{};
    out.section_parent_section_id.resize(morph.sections.size(), -1);
    out.section_length_um.resize(morph.sections.size(), 0.0);
    out.section_diam_um.resize(morph.sections.size(), 0.0);
    out.section_pt3d_offset.resize(morph.sections.size(), -1);
    out.section_pt3d_count.resize(morph.sections.size(), 0);
    out.section_name_to_index.reserve(morph.sections.size());
    for (section_id sid = 0; sid < morph.sections.size(); ++sid) {
        const std::size_t sec_index = static_cast<std::size_t>(sid);
        const auto& sec = morph.sections[sec_index];
        out.section_name_to_index.emplace(sec.name, sec_index);
        if (sec.parent_sec_id != neurong_morph::invalid_section_id) {
            out.section_parent_section_id[sec_index] = static_cast<std::int32_t>(sec.parent_sec_id);
        }
        const double L_um = sec.L_um;
        out.section_length_um[sec_index] = (std::isfinite(L_um) && L_um > 0.0) ? L_um : 0.0;
        const double D_um = sec.diam_um;
        out.section_diam_um[sec_index] = (std::isfinite(D_um) && D_um > 0.0) ? D_um : 0.0;

        if (sec.pt3d_count > 0) {
            const std::size_t p_offset = sec.pt3d_offset;
            const std::size_t p_count = sec.pt3d_count;
            if (p_offset + p_count > morph.pt3d.arc.size() || p_offset + p_count > morph.pt3d.d.size()) {
                throw std::runtime_error("build_section_distance_layout: pt3d range out of bounds for section '" +
                                         sec.name + "'");
            }
            const auto out_offset = static_cast<std::int32_t>(out.pt3d_arc_um.size());
            out.section_pt3d_offset[sec_index] = out_offset;
            out.section_pt3d_count[sec_index] = static_cast<std::int32_t>(p_count);
            out.pt3d_arc_um.reserve(out.pt3d_arc_um.size() + p_count);
            out.pt3d_diam_um.reserve(out.pt3d_diam_um.size() + p_count);
            for (std::size_t k = 0; k < p_count; ++k) {
                const auto idx = p_offset + k;
                out.pt3d_arc_um.push_back(morph.pt3d.arc[idx]);
                out.pt3d_diam_um.push_back(static_cast<double>(morph.pt3d.d[idx]));
            }
        }
    }

    NodeBuildConfig cfg{};
    cfg.force_single_root = true;
    cfg.min_diam_um = 1e-6;
    auto built = build_nodes_neuron_compatible_with_layout(std::move(morph), cfg);
    out.section_node_base_id = std::move(built.layout.section_node_base_id);
    out.section_nseg = std::move(built.layout.section_nseg);
    out.section_parent_node_id = std::move(built.layout.section_parent_node_id);
    const std::size_t nsec = out.section_nseg.size();
    if (nsec == 0) {
        throw std::runtime_error("build_section_distance_layout: empty node layout");
    }
    if (out.section_node_base_id.size() != nsec ||
        out.section_parent_node_id.size() != nsec ||
        out.section_parent_section_id.size() != nsec ||
        out.section_length_um.size() != nsec) {
        throw std::runtime_error("build_section_distance_layout: inconsistent section layout vectors");
    }
    for (const auto& [_, sec_idx] : out.section_name_to_index) {
        if (sec_idx >= nsec) {
            throw std::runtime_error("build_section_distance_layout: section name index out of range");
        }
    }
    out.valid = true;
    return out;
}

MorphLoadResult load_celltemplate_morphology(const LoadOptions& opt,
                                             const std::vector<CellTemplateMorphSpec>& templates) {
    using enum BufferEnable;

    if (templates.empty()) {
        throw std::runtime_error("load_celltemplate_morphology requires at least one template");
    }

    // --------------------------------------------------------------------
    // Step 1: Morph blueprint build (parallel across templates).
    // --------------------------------------------------------------------
    std::vector<BuiltTemplate> built;
    built.resize(templates.size());
    const std::uint64_t sim_random_seed =
        opt.random_seed.value_or(neurong_biophysical::hash_string64("neurong_default_random_seed"));
    {
        std::unordered_set<std::string> unique_names;
        unique_names.reserve(templates.size());
        for (const auto& spec : templates) {
            if (spec.name.empty()) {
                throw std::runtime_error("celltemplate name must not be empty");
            }
            if (!unique_names.insert(spec.name).second) {
                throw std::runtime_error("duplicate celltemplate name: " + spec.name);
            }
            if (spec.sections.empty()) {
                throw std::runtime_error("template '" + spec.name + "' is missing sections");
            }
            if (spec.num_cells <= 0) {
                throw std::runtime_error("template '" + spec.name + "' has invalid num_cells");
            }
        }
    }

    const auto tpl_count = static_cast<std::int64_t>(templates.size());
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t i = 0; i < tpl_count; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const auto& spec = templates[idx];

        SectionNseg section_nseg;
        Morph morph = build_morph_from_sections(spec.sections, section_nseg);
        apply_section_nseg(morph, section_nseg);

        auto sections_by_label = std::move(morph.sections_by_label);
        auto label_names = std::move(morph.label_names);
        auto label_index = std::move(morph.label_index);
        std::unordered_map<std::string, section_id> section_name_to_id;
        std::vector<std::int32_t> section_parent_section_id(morph.sections.size(), -1);
        std::vector<double> section_length_um(morph.sections.size(), 0.0);
        for (section_id sid = 0; sid < morph.sections.size(); ++sid) {
            const std::size_t sec_index = static_cast<std::size_t>(sid);
            const auto& sec = morph.sections[sec_index];
            section_name_to_id.emplace(sec.name, sid);
            if (sec.parent_sec_id != neurong_morph::invalid_section_id) {
                section_parent_section_id[sec_index] = static_cast<std::int32_t>(sec.parent_sec_id);
            }
            const double sec_length_um = sec.L_um;
            section_length_um[sec_index] =
                (std::isfinite(sec_length_um) && sec_length_um > 0.0) ? sec_length_um : 0.0;
        }

        std::vector<double> section_ra(morph.sections.size(), 1.0);
        std::vector<double> section_cm(morph.sections.size(), 1.0);

        NodeBuildConfig cfg{};
        cfg.force_single_root = true;
        cfg.min_diam_um = 1e-6;

        BuiltTemplate out{};
        out.name = spec.name;
        out.num_cells = spec.num_cells;
        out.random_seed = neurong_biophysical::combine_seed_many(
            sim_random_seed,
            static_cast<std::uint64_t>(idx + 1),
            neurong_biophysical::hash_string64(spec.name));
        out.label_names = std::move(label_names);
        out.label_index = std::move(label_index);
        out.sections_by_label = std::move(sections_by_label);
        out.section_name_to_id = std::move(section_name_to_id);
        out.section_parent_section_id = std::move(section_parent_section_id);
        out.section_length_um = std::move(section_length_um);
        out.section_ra = std::move(section_ra);
        out.section_cm = std::move(section_cm);
        out.tpl = build_nodes_neuron_compatible_with_layout(std::move(morph), cfg);
        out.tpl_soma_node = pick_soma_node_index(out.sections_by_label, out.label_index, out.tpl.layout);

        built[idx] = std::move(out);
    }

    // --------------------------------------------------------------------
    // Step 2: Morph instantiation (serial copy).
    // --------------------------------------------------------------------
    PopulationLayout pop_layout{};
    for (const auto& t : built) {
        pop_layout.num_cells_total += t.num_cells;
    }
    const int num_cells_total = pop_layout.num_cells_total;

    std::vector<std::size_t> template_cell_base;
    template_cell_base.resize(built.size(), 0);

    pop_layout.cell_template_id.resize(static_cast<std::size_t>(num_cells_total));
    std::size_t next_cell = 0;
    for (std::size_t t = 0; t < built.size(); ++t) {
        template_cell_base[t] = next_cell;
        for (int c = 0; c < built[t].num_cells; ++c) {
            pop_layout.cell_template_id[next_cell++] = t;
        }
    }

    // Compute total node counts and per-cell nonroot base offsets.
    pop_layout.cell_nonroot_base.resize(static_cast<std::size_t>(num_cells_total));
    std::size_t total_nonroots = 0;
    for (const auto& t : built) {
        const std::size_t tpl_nodes = checked_node_count(t.tpl.nodes);
        total_nonroots += static_cast<std::size_t>(t.num_cells) * (tpl_nodes - 1);
    }

    const std::size_t nnode = static_cast<std::size_t>(num_cells_total) + total_nonroots;

    std::size_t next_nonroot = static_cast<std::size_t>(num_cells_total);
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(num_cells_total); ++cell) {
        const auto tpl_id = pop_layout.cell_template_id[cell];
        const std::size_t tpl_nodes = checked_node_count(built[tpl_id].tpl.nodes);
        pop_layout.cell_nonroot_base[cell] = next_nonroot;
        next_nonroot += (tpl_nodes - 1);
    }

    NodeCoreSoA pop_nodes{};
    pop_nodes.parent_id.resize(nnode);
    pop_nodes.area.resize(nnode);
    pop_nodes.a.resize(nnode);
    pop_nodes.b.resize(nnode);

    for (std::size_t cell = 0; cell < static_cast<std::size_t>(num_cells_total); ++cell) {
        const auto tpl_id = pop_layout.cell_template_id[cell];
        const auto& tpl_nodes = built[tpl_id].tpl.nodes;
        const std::size_t tpl_n = checked_node_count(tpl_nodes);
        const std::size_t base = pop_layout.cell_nonroot_base[cell];

        for (std::size_t tnode = 0; tnode < tpl_n; ++tnode) {
            const auto original_i32 =
                map_template_node_to_original(pop_layout, cell, static_cast<std::int32_t>(tnode));
            if (original_i32 < 0) {
                continue;
            }
            const auto original = static_cast<std::size_t>(original_i32);
            pop_nodes.parent_id[original] =
                map_template_node_to_original(pop_layout, cell, tpl_nodes.parent_id[tnode]);
            pop_nodes.area[original] = tpl_nodes.area[tnode];
            pop_nodes.a[original] = tpl_nodes.a[tnode];
            pop_nodes.b[original] = tpl_nodes.b[tnode];
        }
    }

    // --------------------------------------------------------------------
    // Step 3: One global permute (NeuronG expects permuted vectors + mapping table).
    // --------------------------------------------------------------------
    permute_type = opt.permute_type;
    if (permute_info_arr) {
        delete[] permute_info_arr;
        permute_info_arr = nullptr;
    }
    permute_info_arr = new PermuteInfo[1];

    std::vector<int> parent_tmp;
    parent_tmp.resize(nnode);
    for (std::size_t i = 0; i < nnode; ++i) {
        parent_tmp[i] = static_cast<int>(pop_nodes.parent_id[i]);
    }

    for (int c = 0; c < num_cells_total; ++c) {
        parent_tmp[static_cast<std::size_t>(c)] = -1;
    }

    int* node_permute =
        permute_order(/*ith=*/0, /*ncell=*/num_cells_total, /*nnode=*/static_cast<int>(nnode), parent_tmp.data());

    const auto parent_perm = permute_parent_indices(pop_nodes.parent_id, node_permute, nnode);
    const auto a_perm = permute_by_index(pop_nodes.a, node_permute, nnode);
    const auto b_perm = permute_by_index(pop_nodes.b, node_permute, nnode);
    const auto area_perm = permute_by_index(pop_nodes.area, node_permute, nnode);
    std::vector<double> v_init_override_perm(nnode, 0.0);

    // --------------------------------------------------------------------
    // Construct NeuronGroupData directly (skip CoreNEURON files).
    // --------------------------------------------------------------------
    auto sim = std::make_unique<Simulate>(GPU, opt.buffer_enable);
    if (!opt.output_dir.empty()) {
        sim->output_folder = opt.output_dir;
    }
    sim->tstop = -1;
    sim->dt = opt.dt;
    sim->permute_type = opt.permute_type;

    auto* group = new NeuronGroupData(GPU, sim->dt);
    sim->neuron_group_list.push_back(group);

    group->ncell = num_cells_total;
    group->len = static_cast<int>(nnode);

    group->vecdata_a = new VecData<double>(GPU, a_perm);
    group->vecdata_b = new VecData<double>(GPU, b_perm);
    group->vecdata_area = new VecData<double>(GPU, area_perm);
    group->vecdata_parent_index = new VecData<int>(GPU, parent_perm);

    group->vecdata_d = new VecData<double>(GPU, 0.0, group->len);
    group->vecdata_rhs = new VecData<double>(GPU, 0.0, group->len);
    group->vecdata_v = new VecData<double>(GPU, 0.0, group->len);
    group->vecdata_v_init_override = new VecData<double>(GPU, v_init_override_perm);

    group->permute = node_permute;

    // Keep minimal spike plumbing for in-memory builds (even without loaded synapses).
    // This preserves runtime execution shape with baseline/export path and avoids
    // front-end-specific scheduling divergence in spike-related pipeline stages.
    group->spk_vec = new SpikeVector(/*n=*/0);
    group->vecdata_spk_flags = new VecData<SpikeFlag>(GPU, SpikeFlag::INVALID, /*n=*/0);
    group->presyn = new PreSyn(GPU,
                               /*n=*/0,
                               group->spk_vec,
                               /*pre_node_indices=*/{},
                               /*threshold=*/{},
                               /*spk_vec_offset=*/{},
                               /*pre_gids=*/{});

    if (sim->permute_type == 1) {
        auto& pi = permute_info_arr[0];
        group->vecdata_firstnode = new VecData<int>(GPU, pi.firstnode, num_cells_total);
        group->vecdata_lastnode = new VecData<int>(GPU, pi.lastnode, num_cells_total);
        group->vecdata_cellsize = new VecData<int>(GPU, pi.cellsize, num_cells_total);
        group->vecdata_stride = new VecData<int>(GPU, pi.stride, pi.nstride + 1);
        group->nstride = pi.nstride;
    } else if (sim->permute_type == 3) {
        auto& pi = permute_info_arr[0];
        group->vecdata_firstnode = new VecData<int>(GPU, pi.firstnode, pi.threads_num);
        group->vecdata_lastnode = new VecData<int>(GPU, pi.lastnode, pi.threads_num);
        group->vecdata_max_order_each_thread = new VecData<int>(GPU, pi.max_order_each_thread, pi.threads_num);
        group->vecdata_min_order_each_thread = new VecData<int>(GPU, pi.min_order_each_thread, pi.threads_num);
        group->vecdata_map_t2c = new VecData<int>(GPU, pi.map_t2c, pi.threads_num);
        const int nwarps = (pi.threads_num + 31) / 32;
        group->vecdata_stride = new VecData<int>(GPU, pi.stride, nwarps * (pi.norder + 1));
        group->nstride = pi.nstride;
        group->norder = pi.norder;
        group->nwarp = nwarps;
        group->threads_num = pi.threads_num;
        group->nthread_each_cell = pi.nthread_each_cell;
    }

    MechanismFactory::getInstance().registerVarMap("global", group);

    neurong_biophysical::CellTemplateMorphLayout morph{};
    morph.num_cells_total = num_cells_total;
    morph.nnode = nnode;
    morph.cell_template_id = pop_layout.cell_template_id;
    morph.cell_nonroot_base = pop_layout.cell_nonroot_base;

    for (std::size_t t = 0; t < built.size(); ++t) {
        const auto& bt = built[t];
        neurong_biophysical::CellTemplateInfo info{};
        info.name = bt.name;
        info.cell_base = static_cast<int>(template_cell_base[t]);
        info.num_cells = bt.num_cells;
        info.tpl_nodes_per_cell = checked_node_count(bt.tpl.nodes);
        info.tpl_nonroots_per_cell = info.tpl_nodes_per_cell - 1;
        info.tpl_soma_node = bt.tpl_soma_node;
        info.label_names = bt.label_names;
        info.label_index = bt.label_index;
        info.sections_by_label = bt.sections_by_label;
        info.section_node_base_id = bt.tpl.layout.section_node_base_id;
        info.section_nseg = bt.tpl.layout.section_nseg;
        info.section_parent_node_id = bt.tpl.layout.section_parent_node_id;
        info.section_name_to_id = bt.section_name_to_id;
        info.section_parent_section_id = bt.section_parent_section_id;
        info.section_length_um = bt.section_length_um;
        info.section_ra = bt.section_ra;
        info.section_cm = bt.section_cm;
        info.random_seed = bt.random_seed;
        morph.templates.push_back(std::move(info));
    }

    MorphLoadResult out{};
    out.sim = std::move(sim);
    out.morph_layout = std::move(morph);
    out.gid_ranges.reserve(built.size());
    for (std::size_t t = 0; t < built.size(); ++t) {
        CellTemplateGidRange range{};
        range.name = built[t].name;
        range.gid_begin = static_cast<int>(template_cell_base[t]);
        range.gid_end_exclusive = range.gid_begin + built[t].num_cells;
        out.gid_ranges.push_back(std::move(range));
    }
    return out;
}

namespace {

[[nodiscard]] std::int32_t map_template_node_to_original(
    const neurong_biophysical::CellTemplateMorphLayout& layout,
    std::size_t cell_id,
    std::int32_t template_node_index) {
    if (template_node_index < 0) {
        return -1;
    }
    if (template_node_index == 0) {
        return static_cast<std::int32_t>(cell_id);
    }
    const auto base = layout.cell_nonroot_base[cell_id];
    const auto idx = base + (static_cast<std::size_t>(template_node_index) - 1);
    return static_cast<std::int32_t>(idx);
}

}  // namespace

BiophysApplyResult apply_celltemplate_biophysics(
    Simulate& sim,
    neurong_biophysical::CellTemplateMorphLayout& morph_layout,
    const std::vector<CellTemplateBiophysSpec>& templates,
    double celsius,
    const std::vector<int>& record_gids,
    const std::vector<VarDescriptor>& extra_monitors,
    int* next_type) {
    if (next_type == nullptr) {
        throw std::runtime_error("apply_celltemplate_biophysics requires a non-null next_type pointer");
    }
    if (sim.neuron_group_list.empty() || sim.neuron_group_list[0] == nullptr) {
        throw std::runtime_error("apply_celltemplate_biophysics requires a loaded morphology group");
    }
    auto* group = sim.neuron_group_list[0];
    if (group->len != static_cast<int>(morph_layout.nnode)) {
        throw std::runtime_error("morph layout node count does not match loaded neuron group");
    }
    if (morph_layout.templates.empty()) {
        throw std::runtime_error("apply_celltemplate_biophysics requires non-empty morphology templates");
    }

    if (morph_layout.num_cells_total <= 0) {
        throw std::runtime_error("invalid morphology layout: num_cells_total must be positive");
    }

    std::vector<int> gid_to_spec(static_cast<std::size_t>(morph_layout.num_cells_total), -1);
    auto assign_gid_to_spec = [&](int gid, int spec_idx) {
        if (gid < 0 || gid >= morph_layout.num_cells_total) {
            throw std::runtime_error("biophys selector contains out-of-range gid=" + std::to_string(gid));
        }
        int& slot = gid_to_spec[static_cast<std::size_t>(gid)];
        if (slot == -1 || slot == spec_idx) {
            slot = spec_idx;
            return;
        }
        throw std::runtime_error(
            "biophys selector overlap at gid=" + std::to_string(gid) +
            " between spec#" + std::to_string(slot) + " and spec#" + std::to_string(spec_idx));
    };

    std::unordered_map<std::string, std::pair<int, int>> template_name_to_gid_range;
    template_name_to_gid_range.reserve(morph_layout.templates.size());
    for (const auto& tpl : morph_layout.templates) {
        if (tpl.name.empty()) {
            throw std::runtime_error("morphology template name is empty");
        }
        const int gid_begin = tpl.cell_base;
        const int gid_end = tpl.cell_base + tpl.num_cells;
        const bool inserted = template_name_to_gid_range.emplace(
            tpl.name, std::make_pair(gid_begin, gid_end)).second;
        if (!inserted) {
            throw std::runtime_error("duplicate morphology template name: " + tpl.name);
        }
    }

    for (std::size_t spec_i = 0; spec_i < templates.size(); ++spec_i) {
        const auto& spec = templates[spec_i];
        const int spec_idx = static_cast<int>(spec_i);
        switch (spec.match_kind) {
        case BiophysMatchKind::TemplateName: {
            if (spec.template_name.empty()) {
                throw std::runtime_error("biophys template_name must be non-empty");
            }
            const auto it = template_name_to_gid_range.find(spec.template_name);
            if (it == template_name_to_gid_range.end()) {
                throw std::runtime_error(
                    "biophys template_name '" + spec.template_name + "' not found in loaded morphology templates");
            }
            const int begin = it->second.first;
            const int end = it->second.second;
            for (int gid = begin; gid < end; ++gid) {
                assign_gid_to_spec(gid, spec_idx);
            }
            break;
        }
        case BiophysMatchKind::GidSelector: {
            switch (spec.gid_selector.kind) {
            case GidSelectorKind::All: {
                for (int gid = 0; gid < morph_layout.num_cells_total; ++gid) {
                    assign_gid_to_spec(gid, spec_idx);
                }
                break;
            }
            case GidSelectorKind::Range: {
                const int begin = spec.gid_selector.gid_begin;
                const int end = spec.gid_selector.gid_end_exclusive;
                if (begin < 0 || end < begin || end > morph_layout.num_cells_total) {
                    throw std::runtime_error(
                        "biophys gid_selector range invalid: begin=" + std::to_string(begin) +
                        " end=" + std::to_string(end) +
                        " num_cells_total=" + std::to_string(morph_layout.num_cells_total));
                }
                for (int gid = begin; gid < end; ++gid) {
                    assign_gid_to_spec(gid, spec_idx);
                }
                break;
            }
            case GidSelectorKind::Explicit: {
                if (spec.gid_selector.gids.empty()) {
                    throw std::runtime_error("biophys gid_selector explicit gid list must be non-empty");
                }
                for (const int gid : spec.gid_selector.gids) {
                    assign_gid_to_spec(gid, spec_idx);
                }
                break;
            }
            default:
                throw std::runtime_error("unsupported biophys gid_selector kind");
            }
            break;
        }
        default:
            throw std::runtime_error("unsupported biophys match kind");
        }
    }

    struct TemplateRunInfo {
        std::size_t source_template_id{0};
        int source_local_begin{0};
        int source_local_count{0};
    };

    neurong_biophysical::CellTemplateMorphLayout biophys_layout{};
    biophys_layout.num_cells_total = morph_layout.num_cells_total;
    biophys_layout.nnode = morph_layout.nnode;
    biophys_layout.cell_nonroot_base = morph_layout.cell_nonroot_base;
    const std::size_t kUnassignedTpl = static_cast<std::size_t>(-1);
    biophys_layout.cell_template_id.assign(
        static_cast<std::size_t>(morph_layout.num_cells_total),
        kUnassignedTpl);

    std::vector<int> biophys_template_spec_idx{};
    std::vector<TemplateRunInfo> biophys_template_run_info{};
    biophys_layout.templates.reserve(morph_layout.templates.size());
    biophys_template_spec_idx.reserve(morph_layout.templates.size());
    biophys_template_run_info.reserve(morph_layout.templates.size());

    auto append_template_run = [&](std::size_t source_tpl_i,
                                   int run_gid_begin,
                                   int run_gid_end,
                                   int run_spec_idx,
                                   int run_part_index) {
        if (run_gid_end <= run_gid_begin) {
            throw std::runtime_error("internal error: invalid empty biophys run");
        }
        const auto& source_tpl = morph_layout.templates[source_tpl_i];
        auto run_tpl = source_tpl;
        run_tpl.cell_base = run_gid_begin;
        run_tpl.num_cells = run_gid_end - run_gid_begin;
        if (run_part_index >= 0) {
            run_tpl.name = source_tpl.name +
                           "__biophys_t" + std::to_string(source_tpl_i) +
                           "_part_" + std::to_string(run_part_index);
        }
        if (run_spec_idx < 0) {
            run_tpl.prepare = neurong_biophysical::Prepare{};
            run_tpl.v_init = neurong_biophysical::ScalarOrRandom{0.0};
        } else {
            const auto& spec = templates[static_cast<std::size_t>(run_spec_idx)];
            run_tpl.prepare = spec.inserts;
            run_tpl.v_init = spec.v_init;
        }

        const auto new_tpl_id = biophys_layout.templates.size();
        biophys_layout.templates.push_back(std::move(run_tpl));
        biophys_template_spec_idx.push_back(run_spec_idx);
        biophys_template_run_info.push_back(TemplateRunInfo{
            .source_template_id = source_tpl_i,
            .source_local_begin = run_gid_begin - source_tpl.cell_base,
            .source_local_count = run_gid_end - run_gid_begin,
        });
        for (int gid = run_gid_begin; gid < run_gid_end; ++gid) {
            biophys_layout.cell_template_id[static_cast<std::size_t>(gid)] = new_tpl_id;
        }
    };

    for (std::size_t tpl_i = 0; tpl_i < morph_layout.templates.size(); ++tpl_i) {
        const auto& tpl = morph_layout.templates[tpl_i];
        if (tpl.num_cells <= 0) {
            throw std::runtime_error("template '" + tpl.name + "' has invalid num_cells");
        }
        const int gid_begin = tpl.cell_base;
        const int gid_end = tpl.cell_base + tpl.num_cells;
        if (gid_begin < 0 || gid_end < gid_begin || gid_end > morph_layout.num_cells_total) {
            throw std::runtime_error(
                "template '" + tpl.name + "' gid range out of bounds: [" +
                std::to_string(gid_begin) + "," + std::to_string(gid_end) + ")");
        }

        int run_begin = gid_begin;
        int run_spec = gid_to_spec[static_cast<std::size_t>(gid_begin)];
        std::vector<std::tuple<int, int, int>> runs{};
        for (int gid = gid_begin + 1; gid < gid_end; ++gid) {
            const int spec_idx = gid_to_spec[static_cast<std::size_t>(gid)];
            if (spec_idx == run_spec) {
                continue;
            }
            runs.emplace_back(run_begin, gid, run_spec);
            run_begin = gid;
            run_spec = spec_idx;
        }
        runs.emplace_back(run_begin, gid_end, run_spec);

        const bool needs_split = (runs.size() > 1);
        for (std::size_t r = 0; r < runs.size(); ++r) {
            const auto [rb, re, rs] = runs[r];
            append_template_run(
                tpl_i,
                rb,
                re,
                rs,
                needs_split ? static_cast<int>(r) : -1);
        }
    }

    for (std::size_t gid = 0; gid < static_cast<std::size_t>(morph_layout.num_cells_total); ++gid) {
        if (biophys_layout.cell_template_id[gid] == kUnassignedTpl) {
            throw std::runtime_error("internal error: some gids are not assigned to biophys template runs");
        }
    }

    std::vector<neurong_biophysical::IClampSpec> merged_iclamp{};
    for (std::size_t tpl_i = 0; tpl_i < biophys_layout.templates.size(); ++tpl_i) {
        const auto& tpl = biophys_layout.templates[tpl_i];
        const int spec_idx = biophys_template_spec_idx[tpl_i];
        if (spec_idx < 0) {
            continue;
        }
        const auto& run = biophys_template_run_info[tpl_i];
        if (run.source_template_id >= morph_layout.templates.size()) {
            throw std::runtime_error("internal error: source template id out of range in biophys run info");
        }
        const auto& source_tpl = morph_layout.templates[run.source_template_id];
        const int source_local_begin = run.source_local_begin;
        const int source_local_end = run.source_local_begin + run.source_local_count;
        const bool split_run =
            (tpl.cell_base != source_tpl.cell_base) ||
            (tpl.num_cells != source_tpl.num_cells) ||
            (tpl.name != source_tpl.name);

        const auto& spec = templates[static_cast<std::size_t>(spec_idx)];
        for (const auto& stim_spec : spec.iclamp) {
            auto stim = stim_spec;
            if (!split_run) {
                stim.template_name = tpl.name;
                merged_iclamp.push_back(std::move(stim));
                continue;
            }

            stim.template_name = tpl.name;
            if (stim_spec.all_cells) {
                stim.all_cells = true;
                stim.cells.clear();
                merged_iclamp.push_back(std::move(stim));
                continue;
            }

            std::vector<int> local_cells{};
            local_cells.reserve(stim_spec.cells.size());
            for (const int local_src : stim_spec.cells) {
                if (local_src < source_local_begin || local_src >= source_local_end) {
                    continue;
                }
                local_cells.push_back(local_src - source_local_begin);
            }
            if (local_cells.empty()) {
                continue;
            }
            if (static_cast<int>(local_cells.size()) == tpl.num_cells) {
                stim.all_cells = true;
                stim.cells.clear();
            } else {
                stim.all_cells = false;
                stim.cells = std::move(local_cells);
            }
            merged_iclamp.push_back(std::move(stim));
        }
    }

    std::vector<double> v_init_override_original(biophys_layout.nnode, 0.0);
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(biophys_layout.num_cells_total); ++cell) {
        const auto tpl_id = biophys_layout.cell_template_id[cell];
        if (tpl_id >= biophys_layout.templates.size()) {
            throw std::runtime_error("invalid template id while applying biophysics");
        }
        const auto& tpl = biophys_layout.templates[tpl_id];
        const double cell_v_init = sample_template_v_init(
            tpl.v_init,
            tpl.random_seed,
            cell);
        const std::size_t tpl_nodes = tpl.tpl_nodes_per_cell;
        for (std::size_t tnode = 0; tnode < tpl_nodes; ++tnode) {
            const auto original_i32 =
                map_template_node_to_original(biophys_layout, cell, static_cast<std::int32_t>(tnode));
            if (original_i32 < 0) {
                continue;
            }
            const auto original = static_cast<std::size_t>(original_i32);
            v_init_override_original[original] = cell_v_init;
        }
    }

    std::vector<double> v_init_override_perm(biophys_layout.nnode, 0.0);
    if (group->permute == nullptr) {
        v_init_override_perm = v_init_override_original;
    } else {
        for (std::size_t original = 0; original < biophys_layout.nnode; ++original) {
            const auto permuted = static_cast<std::size_t>(group->permute[original]);
            v_init_override_perm[permuted] = v_init_override_original[original];
        }
    }

    if (group->vecdata_v_init_override == nullptr) {
        throw std::runtime_error("missing vecdata_v_init_override in neuron group");
    }
    if (group->vecdata_v_init_override->size() != biophys_layout.nnode) {
        throw std::runtime_error("v_init_override vector size mismatch");
    }
    auto* v_init_cpu = group->vecdata_v_init_override->get_cpu_data();
    for (std::size_t i = 0; i < biophys_layout.nnode; ++i) {
        v_init_cpu[i] = v_init_override_perm[i];
    }
    if (sim.mode == GPU) {
        group->vecdata_v_init_override->update_gpu_data_from_cpu();
    }

    auto biophys = neurong_biophysical::build_neurong_biophysics(sim,
                                                                 *group,
                                                                 biophys_layout,
                                                                 celsius,
                                                                 merged_iclamp,
                                                                 record_gids,
                                                                 extra_monitors,
                                                                 *next_type);
    BiophysApplyResult out{};
    out.monitor_to_handle = std::move(biophys.monitor_to_handle);
    out.record_handles = std::move(biophys.record_handles);
    return out;
}

}  // namespace neurong_celltemplate
