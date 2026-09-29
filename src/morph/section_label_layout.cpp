#include "section_label_layout.hpp"

#include <stdexcept>
#include <string>

namespace neurong_morph {

SectionLabelSegmentLayout build_section_label_segment_layout(
    const std::vector<std::vector<section_id>>& sections_by_label,
    const std::vector<std::int32_t>& section_nseg,
    const std::string& error_context) {
    const std::size_t num_sections = section_nseg.size();
    SectionLabelSegmentLayout out{};
    out.section_label_u_by_sec.assign(num_sections, -1);
    out.section_offset_in_label_by_sec.assign(num_sections, 0);
    out.label_segment_count.assign(sections_by_label.size(), 0);

    auto with_ctx = [&](const std::string& msg) -> std::runtime_error {
        if (error_context.empty()) {
            return std::runtime_error(msg);
        }
        return std::runtime_error(error_context + ": " + msg);
    };

    for (std::size_t label_u = 0; label_u < sections_by_label.size(); ++label_u) {
        std::size_t offset_in_label = 0;
        for (const auto sec_id : sections_by_label[label_u]) {
            const auto sec_index = static_cast<std::size_t>(sec_id);
            if (sec_index >= num_sections) {
                throw with_ctx("section index out of range in label map");
            }
            if (out.section_label_u_by_sec[sec_index] >= 0) {
                throw with_ctx("section appears in multiple labels");
            }
            out.section_label_u_by_sec[sec_index] = static_cast<std::int32_t>(label_u);
            out.section_offset_in_label_by_sec[sec_index] = offset_in_label;

            const auto nseg = section_nseg[sec_index];
            if (nseg < 0) {
                throw with_ctx("section nseg must be non-negative");
            }
            offset_in_label += static_cast<std::size_t>(nseg);
        }
        out.label_segment_count[label_u] = offset_in_label;
    }

    return out;
}

}  // namespace neurong_morph

