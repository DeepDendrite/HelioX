#pragma once

#include "swc_to_section.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace neurong_morph {

// Pure section/label discretization mapping derived from morphology metadata.
//
// For each section index:
// - section_label_u_by_sec[sec] gives the owning label index in sections_by_label.
// - section_offset_in_label_by_sec[sec] gives the starting segment offset within that label.
//
// For each label index:
// - label_segment_count[label] is the total number of segments in that label.
struct SectionLabelSegmentLayout {
    std::vector<std::int32_t> section_label_u_by_sec{};
    std::vector<std::size_t> section_offset_in_label_by_sec{};
    std::vector<std::size_t> label_segment_count{};
};

// Build section<->label segment mapping once from morphology-only inputs.
//
// The function validates:
// - section indices referenced by labels are in range
// - each section appears in at most one label
// - section_nseg values are non-negative
[[nodiscard]] SectionLabelSegmentLayout build_section_label_segment_layout(
    const std::vector<std::vector<section_id>>& sections_by_label,
    const std::vector<std::int32_t>& section_nseg,
    const std::string& error_context = "");

}  // namespace neurong_morph

