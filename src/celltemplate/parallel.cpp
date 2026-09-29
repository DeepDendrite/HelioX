// Module 4/4: Parallel helpers + population building utilities.

#include "parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace neurong_morph {

std::vector<NodeCoreSoA> build_nodes_neuron_compatible_many(std::vector<Morph>&& morphs,
                                                            const NodeBuildConfig& cfg,
                                                            std::size_t nthreads) {
    std::vector<NodeCoreSoA> out(morphs.size());
    const auto n = static_cast<std::int64_t>(morphs.size());
    if (nthreads > 0) {
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (std::int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] =
                build_nodes_neuron_compatible(std::move(morphs[static_cast<std::size_t>(i)]), cfg);
        }
    } else {
#pragma omp parallel for schedule(dynamic)
        for (std::int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] =
                build_nodes_neuron_compatible(std::move(morphs[static_cast<std::size_t>(i)]), cfg);
        }
    }
    return out;
}

namespace {

std::size_t checked_node_count(const NodeCoreSoA& nodes);

std::size_t template_root_count_per_cell(const NodeCoreSoA& template_nodes) {
    const auto per_cell = checked_node_count(template_nodes);
    if (per_cell == 0) {
        return 0;
    }

    std::size_t roots = 0;
    while (roots < per_cell && template_nodes.parent_id[roots] < 0) {
        ++roots;
    }
    return roots;
}

}  // namespace

std::vector<TemplatePopulationRootFirst> build_population_from_templates_root_first(
    const std::vector<NodeCoreSoA>& templates,
    const std::vector<std::size_t>& num_cells_per_template,
    std::size_t nthreads) {
    std::vector<TemplatePopulationRootFirst> out(templates.size());
    const auto n = static_cast<std::int64_t>(templates.size());
    if (nthreads > 0) {
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (std::int64_t i = 0; i < n; ++i) {
            TemplatePopulationRootFirst group{};
            group.num_cells = num_cells_per_template[static_cast<std::size_t>(i)];
            group.nodes_per_cell = checked_node_count(templates[static_cast<std::size_t>(i)]);
            if (group.nodes_per_cell == 0 || group.num_cells == 0) {
                out[static_cast<std::size_t>(i)] = std::move(group);
                continue;
            }
            group.roots_per_cell = template_root_count_per_cell(templates[static_cast<std::size_t>(i)]);
            group.nonroots_per_cell = group.nodes_per_cell - group.roots_per_cell;
            group.nodes = replicate_nodes_root_first_global_ids(templates[static_cast<std::size_t>(i)],
                                                                group.num_cells,
                                                                /*ignored_nthreads=*/0);
            out[static_cast<std::size_t>(i)] = std::move(group);
        }
    } else {
#pragma omp parallel for schedule(dynamic)
        for (std::int64_t i = 0; i < n; ++i) {
            TemplatePopulationRootFirst group{};
            group.num_cells = num_cells_per_template[static_cast<std::size_t>(i)];
            group.nodes_per_cell = checked_node_count(templates[static_cast<std::size_t>(i)]);
            if (group.nodes_per_cell == 0 || group.num_cells == 0) {
                out[static_cast<std::size_t>(i)] = std::move(group);
                continue;
            }
            group.roots_per_cell = template_root_count_per_cell(templates[static_cast<std::size_t>(i)]);
            group.nonroots_per_cell = group.nodes_per_cell - group.roots_per_cell;
            group.nodes = replicate_nodes_root_first_global_ids(templates[static_cast<std::size_t>(i)],
                                                                group.num_cells,
                                                                /*ignored_nthreads=*/0);
            out[static_cast<std::size_t>(i)] = std::move(group);
        }
    }
    return out;
}

std::vector<TemplatePopulationRootFirst> build_population_from_templates_root_first(
    const std::vector<NodeCoreSoA>& templates,
    std::size_t num_cells_per_template,
    std::size_t nthreads) {
    std::vector<std::size_t> num_cells(templates.size(), num_cells_per_template);
    return build_population_from_templates_root_first(templates, num_cells, nthreads);
}

namespace {

std::size_t checked_node_count(const NodeCoreSoA& nodes) {
    return nodes.parent_id.size();
}

}  // namespace

NodeCoreSoA replicate_nodes_root_first_global_ids(const NodeCoreSoA& template_nodes,
                                                  std::size_t num_cells,
                                                  [[maybe_unused]] std::size_t ignored_nthreads) {
    NodeCoreSoA out{};
    if (num_cells == 0) {
        return out;
    }

    const auto per_cell = checked_node_count(template_nodes);
    if (per_cell == 0) {
        return out;
    }
    const auto total = per_cell * num_cells;

    // Identify the number of root nodes in the template.
    // By convention (and as produced by build_nodes_neuron_compatible),
    // root nodes are contiguous and appear first.
    std::size_t roots = 0;
    while (roots < per_cell && template_nodes.parent_id[roots] < 0) {
        ++roots;
    }
    const auto nonroots = per_cell - roots;

    const auto root_total = roots * num_cells;

    // Ensure global node indices fit in int32, which is the on-disk / kernel format.
    const auto max_i32 = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());

    out.parent_id.resize(total);
    if (!template_nodes.area.empty()) {
        out.area.resize(total);
    }
    if (!template_nodes.a.empty()) {
        out.a.resize(total);
    }
    if (!template_nodes.b.empty()) {
        out.b.resize(total);
    }

    auto memcpy_subblock = [](auto& dst,
                              const auto& src,
                              std::size_t dst_base,
                              std::size_t src_base,
                              std::size_t count) {
        using T = std::remove_reference_t<decltype(dst)>::value_type;
        static_assert(std::is_trivially_copyable_v<T>, "memcpy_subblock requires trivially copyable T");
        if (dst.empty() || count == 0) {
            return;
        }
        std::memcpy(dst.data() + dst_base, src.data() + src_base, count * sizeof(T));
    };

    auto map_parent = [&](std::int32_t tp, std::size_t root_base, std::size_t nonroot_base) -> std::int32_t {
        if (tp < 0) {
            return -1;
        }
        const auto p = static_cast<std::size_t>(tp);
        if (p < roots) {
            return static_cast<std::int32_t>(root_base + p);
        }
        return static_cast<std::int32_t>(nonroot_base + (p - roots));
    };

    for (std::size_t ci = 0; ci < num_cells; ++ci) {
        const auto root_base = ci * roots;
        const auto nonroot_base = root_total + ci * nonroots;

        // Fast path for all non-parent fields: the template layout is identical
        // within the root and non-root blocks.
        memcpy_subblock(out.area, template_nodes.area, root_base, 0, roots);
        memcpy_subblock(out.a, template_nodes.a, root_base, 0, roots);
        memcpy_subblock(out.b, template_nodes.b, root_base, 0, roots);

        memcpy_subblock(out.area, template_nodes.area, nonroot_base, roots, nonroots);
        memcpy_subblock(out.a, template_nodes.a, nonroot_base, roots, nonroots);
        memcpy_subblock(out.b, template_nodes.b, nonroot_base, roots, nonroots);

        // parent_id requires rewriting.
        for (std::size_t i = 0; i < roots; ++i) {
            const auto dst_i = root_base + i;
            out.parent_id[dst_i] = map_parent(template_nodes.parent_id[i], root_base, nonroot_base);
        }
        for (std::size_t j = 0; j < nonroots; ++j) {
            const auto src_i = roots + j;
            const auto dst_i = nonroot_base + j;
            out.parent_id[dst_i] = map_parent(template_nodes.parent_id[src_i], root_base, nonroot_base);
        }
    }

    return out;
}

}  // namespace neurong_morph
