#pragma once

#include "biophys_builder.hpp"
#include "section_distance.hpp"
#include "simulate.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace neurong_celltemplate {

// Per-section discretization (NEURON nseg).
// This is an internal helper for applying explicit nseg values after building a Morph.
using SectionNseg = std::vector<std::int32_t>;

struct SectionPt3d {
    float x_um{};
    float y_um{};
    float z_um{};
    float diam_um{};
};

// One complete section specification (topology + geometry + nseg).
//
// Geometry rules:
// - If pt3d is non-empty, it must contain >=2 points and (L_um, diam_um) are ignored.
// - If pt3d is empty, this section is a NEURON-style n3d==0 cylinder and requires
//   positive L_um and diam_um.
struct SectionSpec {
    SectionSpec() = default;
    explicit SectionSpec(std::string name_)
        : name(std::move(name_)) {}
    SectionSpec(std::string name_, std::string label_)
        : name(std::move(name_)),
          label(std::move(label_)) {
        if (label.empty()) {
            throw std::runtime_error("section label name is empty");
        }
        if (label == "all") {
            throw std::runtime_error("section label name 'all' is reserved");
        }
    }

    std::string name{};
    std::string parent_name{};  // empty means root
    double parentx{1.0};
    std::string label{};
    std::int32_t nseg{1};
    double L_um{0.0};
    double diam_um{0.0};
    std::vector<SectionPt3d> pt3d{};
};

struct CellTemplateMorphSpec {
    std::string name{};
    // Full section list (topology + geometry + per-section nseg).
    std::vector<SectionSpec> sections{};
    int num_cells{0};
};

enum class GidSelectorKind {
    All = 0,
    Range,
    Explicit,
};

struct GidSelector {
    GidSelectorKind kind{GidSelectorKind::All};
    int gid_begin{0};
    int gid_end_exclusive{0};
    std::vector<int> gids{};
};

enum class BiophysMatchKind {
    TemplateName = 0,
    GidSelector,
};

struct CellTemplateBiophysSpec {
    BiophysMatchKind match_kind{BiophysMatchKind::TemplateName};
    std::string template_name{};
    GidSelector gid_selector{};
    neurong_biophysical::ScalarOrRandom v_init{};
    neurong_biophysical::Prepare inserts{};
    std::vector<neurong_biophysical::IClampSpec> iclamp{};
};

struct LoadOptions {
    double dt{0.025};
    int permute_type{3};
    std::string output_dir{};
    BufferEnable buffer_enable{BufferEnable::IPC};
    std::optional<std::uint64_t> random_seed{};
};

struct CellTemplateGidRange {
    std::string name{};
    int gid_begin{0};
    int gid_end_exclusive{0};
};

struct MorphLoadResult {
    std::unique_ptr<Simulate> sim{};
    neurong_biophysical::CellTemplateMorphLayout morph_layout{};
    std::vector<CellTemplateGidRange> gid_ranges{};
};

struct BiophysApplyResult {
    std::map<VarDescriptor, int> monitor_to_handle{};
    std::vector<std::tuple<int, int, int>> record_handles{};
};

// Load a SWC morphology and return a full section list (topology + geometry).
[[nodiscard]] std::vector<SectionSpec> load_swc_sections(const std::string& swc_file);

// Load a Neurolucida ASC morphology and return a full section list (topology + geometry).
[[nodiscard]] std::vector<SectionSpec> load_asc_sections(const std::string& asc_file);

// Return sections matching a label name. Unknown labels return an empty list.
[[nodiscard]] std::vector<SectionSpec> sections_with_label(const std::vector<SectionSpec>& sections,
                                                           const std::string& label);

// Build morphology-only distance layout (no simulation state required).
[[nodiscard]] neurong_morph::SectionDistanceLayout build_section_distance_layout(
    const std::vector<SectionSpec>& sections);

[[nodiscard]] MorphLoadResult load_celltemplate_morphology(const LoadOptions& opt,
                                                           const std::vector<CellTemplateMorphSpec>& templates);

[[nodiscard]] BiophysApplyResult apply_celltemplate_biophysics(
    Simulate& sim,
    neurong_biophysical::CellTemplateMorphLayout& morph_layout,
    const std::vector<CellTemplateBiophysSpec>& templates,
    double celsius,
    const std::vector<int>& record_gids,
    const std::vector<VarDescriptor>& extra_monitors,
    int* next_type);

}  // namespace neurong_celltemplate
