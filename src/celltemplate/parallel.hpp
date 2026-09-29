#pragma once

#include "section_to_node.hpp"

#include <cstddef>
#include <vector>

namespace neurong_morph {

// Grouped population build output for a single template morphology.
//
// Layout matches replicate_nodes_root_first_global_ids():
//   [all roots (num_cells*roots_per_cell)] [cell0 nonroots] [cell1 nonroots] ...
//
// Note: indices are "global" only within this group (0..nodes.size()).
struct TemplatePopulationRootFirst {
    NodeCoreSoA nodes{};
    std::size_t num_cells{0};
    std::size_t nodes_per_cell{0};
    std::size_t roots_per_cell{0};
    std::size_t nonroots_per_cell{0};
};

// Parallelize build_nodes_neuron_compatible() across multiple independent morphologies.
// The input vector is consumed (moved-from) to avoid extra copies.
std::vector<NodeCoreSoA> build_nodes_neuron_compatible_many(std::vector<Morph>&& morphs,
                                                            const NodeBuildConfig& cfg,
                                                            std::size_t nthreads = 0);

// Build a population for multiple templates:
//  - Parallelizes across templates (different morphologies).
//  - Replication within each template is intentionally single-threaded.
//
// This is a convenient building block for an Arbor-style "cell group per template"
// workflow: build templates once, then expand to many identical cells.
std::vector<TemplatePopulationRootFirst> build_population_from_templates_root_first(
    const std::vector<NodeCoreSoA>& templates,
    const std::vector<std::size_t>& num_cells_per_template,
    std::size_t nthreads = 0);

std::vector<TemplatePopulationRootFirst> build_population_from_templates_root_first(
    const std::vector<NodeCoreSoA>& templates,
    std::size_t num_cells_per_template,
    std::size_t nthreads = 0);

// Even faster path for "many identical neurons": directly generate a
// CoreNEURON/NeuronGPU-style "root-first" layout (all root nodes contiguous at
// the beginning, followed by per-cell non-root nodes), while rewriting
// parent_id to global node ids.
//
// This avoids an extra full-population permutation pass when the final consumer
// (e.g. NEURON/CoreNEURON/neurong Hines kernels) expects nodes [0..ncell) to be
// root nodes.
//
// Note: for a single template replicated many times, this is intentionally
// single-threaded (memory-bandwidth bound). Use parallelism across different
// templates/morphologies instead (see build_nodes_neuron_compatible_many()).
//
// ignored_nthreads is accepted for API symmetry with other helpers, but is
// currently ignored by design.
NodeCoreSoA replicate_nodes_root_first_global_ids(const NodeCoreSoA& template_nodes,
                                                  std::size_t num_cells,
                                                  std::size_t ignored_nthreads = 0);

}  // namespace neurong_morph
