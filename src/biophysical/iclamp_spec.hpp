#pragma once

#include <optional>
#include <string>
#include <vector>

namespace neurong_biophysical {

struct IClampSpec {
    // Target template name resolved from matched biophys template.
    std::string template_name{};
    // True -> apply to all cells in the selected template.
    bool all_cells{false};
    // Empty -> no stimulation (unless all_cells is true).
    std::vector<int> cells{};
    // Section selector: either section_name or section_id must be set.
    std::string section_name{};
    std::optional<int> section_id{};
    double x{0.5};
    double delay{0.0};
    double dur{0.0};
    double amp{0.0};
};

}  // namespace neurong_biophysical
