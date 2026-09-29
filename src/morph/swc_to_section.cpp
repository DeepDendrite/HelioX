// Module 1/4: SWC parsing + sectionify (SWC -> Morph::sections + Morph::datas)

#include "swc_to_section.hpp"

#include "label_utils.hpp"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stack>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

namespace neurong_morph {

namespace {

std::string swc_label_to_name_local(std::int32_t label) {
    switch (label) {
        case 1:
            return "soma";
        case 2:
            return "axon";
        case 3:
            return "dend";
        case 4:
            return "apic";
        default:
            return std::to_string(label);
    }
}

std::string label_prefix(std::string_view label) {
    if (neurong_labels::is_soma_label(label)) {
        return "soma";
    }
    if (neurong_labels::is_axon_label(label)) {
        return "axon";
    }
    if (neurong_labels::is_dend_label(label)) {
        return "dend";
    }
    if (neurong_labels::is_apic_label(label)) {
        return "apic";
    }
    return "sec";
}

void assign_default_section_names(Morph& morph) {
    std::unordered_map<std::string, int> counts;
    for (auto& sec : morph.sections) {
        if (!sec.name.empty()) {
            continue;
        }
        const std::string label_key = sec.label;
        auto& idx = counts[label_key];
        const std::string prefix = label_prefix(label_key);
        std::string name;
        if (prefix == "sec") {
            name = prefix + "_" + label_key + "_" + std::to_string(idx);
        } else {
            name = prefix + "_" + std::to_string(idx);
        }
        sec.name = std::move(name);
        ++idx;
    }
}

std::size_t ensure_label_index(Morph& morph, std::string_view label_name) {
    const auto key = std::string(label_name);
    if (key.empty()) {
        throw std::runtime_error("section label name is empty");
    }
    if (key == neurong_labels::kAllLabel) {
        throw std::runtime_error("section label name 'all' is reserved");
    }
    auto it = morph.label_index.find(key);
    if (it != morph.label_index.end()) {
        return it->second;
    }
    const auto idx = morph.label_names.size();
    morph.label_names.push_back(key);
    morph.label_index.emplace(key, idx);
    if (idx >= morph.sections_by_label.size()) {
        morph.sections_by_label.resize(idx + 1);
    }
    return idx;
}

}  // namespace

static inline bool is_ascii_space(char c) noexcept {
    // SWC is ASCII text; keep this fast and locale-independent.
    switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case '\f':
        case '\v':
            return true;
        default:
            return false;
    }
}

static bool is_blank_or_comment(const std::string& line) {
    for (char c : line) {
        if (is_ascii_space(c)) {
            continue;
        }
        return c == '#';
    }
    return true;
}

static inline void skip_spaces(const char*& p, const char* end) noexcept {
    while (p < end && is_ascii_space(*p)) {
        ++p;
    }
}

template <typename Int>
static bool parse_int(const char*& p, const char* end, Int& out) noexcept {
    static_assert(std::is_integral_v<Int>, "parse_int requires an integral type");
    skip_spaces(p, end);
    if (p >= end) {
        return false;
    }
    const auto res = std::from_chars(p, end, out);
    if (res.ec != std::errc{}) {
        return false;
    }
    p = res.ptr;
    return true;
}

static bool parse_float(const char*& p, const char* end, float& out) noexcept {
    skip_spaces(p, end);
    if (p >= end) {
        return false;
    }
    if constexpr (requires { std::from_chars(p, end, out, std::chars_format::general); }) {
        auto res = std::from_chars(p, end, out, std::chars_format::general);
        if (res.ec != std::errc{}) {
            return false;
        }
        p = res.ptr;
    } else {
        char* parsed_end = nullptr;
        out = std::strtof(p, &parsed_end);
        if (parsed_end == p) {
            return false;
        }
        p = parsed_end;
    }
    return true;
}

// Sectionify implementation notes:
// - SWC input is treated as an arbitrary forest: ids can be sparse and lines can be in any order.
// - We build an id->point map and parent/children adjacency, then do a MorphIO-style
//   DFS pre-order traversal that merges maximal same-label unary chains into Sections.
// - We intentionally do not sort samples by id or allocate dense id->index vectors.
// - We keep only minimal soma-connection behavior (e.g. single-point soma attaches at parentx=0.5)
//   so existing node-geometry/correctness checks keep working on the project reference SWCs.

Morph load_swc_morphology(const std::string& swc_path) {
    std::ifstream file(swc_path);

    std::vector<SwcData> swc_data;
    std::vector<origin_id> swc_origin_ids;
    std::vector<origin_id> swc_parent_origin_ids;
    std::string line;
    std::int32_t line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (is_blank_or_comment(line)) {
            continue;
        }
        origin_id origin{};
        origin_id parent_origin{};
        std::int32_t label{};
        float x{};
        float y{};
        float z{};
        float radius_um{};

        const char* p = line.c_str();
        const char* const end = p + line.size();
        parse_int(p, end, origin);
        parse_int(p, end, label);
        parse_float(p, end, x);
        parse_float(p, end, y);
        parse_float(p, end, z);
        parse_float(p, end, radius_um);
        parse_int(p, end, parent_origin);
        SwcData data{};
        data.xyz = {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
        data.d_um = static_cast<double>(radius_um) * 2.0;
        data.label = static_cast<std::int8_t>(label);
        swc_data.push_back(data);
        swc_origin_ids.push_back(origin);
        swc_parent_origin_ids.push_back(parent_origin);
    }

    Morph morph{};
    if (swc_data.empty()) {
        return morph;
    }

    const std::size_t num_points = swc_data.size();
    const auto num_points_id = static_cast<swcdata_id>(num_points);

    // Map SWC-origin ids (declared in file) to our internal swc point ids (swcdata_id).
    // Unlike NEURON's Import3d, we do NOT sort by id and we do NOT allocate a dense
    // id->index vector (which can be very slow / memory-heavy for sparse large ids).
    // This also allows forward references (a child line can appear before its parent line).
    std::unordered_map<origin_id, swcdata_id> origin2swc_id{};
    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        const origin_id origin = swc_origin_ids[static_cast<std::size_t>(point_id)];
        const auto [it, inserted] = origin2swc_id.emplace(origin, point_id);
    }

    // Resolve parent indices once up-front (to avoid repeated unordered_map lookups).
    // parent_of[i] == invalid_swcdata_id means "root".
    std::vector<swcdata_id> parent_of(num_points, invalid_swcdata_id);
    std::vector<swcdata_id> roots;

    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        const origin_id parent_origin = swc_parent_origin_ids[static_cast<std::size_t>(point_id)];
        if (parent_origin < 0) {
            roots.push_back(point_id);
            continue;
        }
        const auto parent_it = origin2swc_id.find(parent_origin);
        const auto parent_point_id = parent_it->second;
        parent_of[static_cast<std::size_t>(point_id)] = parent_point_id;
    }

    auto parent_swc_id = [&](swcdata_id point_id) -> swcdata_id {
        return parent_of[static_cast<std::size_t>(point_id)];
    };

    std::vector<std::uint32_t> nchild_soma(num_points, 0);
    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        if (swc_data[static_cast<std::size_t>(point_id)].label != 1) {
            continue;
        }
        const auto parent_point_id = parent_swc_id(point_id);
        if (parent_point_id != invalid_swcdata_id &&
            swc_data[static_cast<std::size_t>(parent_point_id)].label == 1) {
            ++nchild_soma[static_cast<std::size_t>(parent_point_id)];
        }
    }

    // DFS sectionify (MorphIO-style): build maximal same-label chains until a branch, leaf,
    // or label-change, then recurse on children. Unlike NEURON, we do not require SWC ids
    // to be sorted or parents to appear before children in the file.

    morph.sections.clear();
    morph.datas.clear();
    morph.label_names.clear();
    morph.label_index.clear();
    morph.sections_by_label.clear();

    std::vector<std::vector<swcdata_id>> swc_point_children(num_points);
    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        const auto parent_swc_point_id = parent_of[static_cast<std::size_t>(point_id)];
        if (parent_swc_point_id != invalid_swcdata_id) {
            swc_point_children[static_cast<std::size_t>(parent_swc_point_id)].push_back(point_id);
        }
    }

    std::vector<section_id> swc_point_to_section_id(num_points, invalid_section_id);
    std::stack<swcdata_id> dfs_stack;
    for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
        dfs_stack.push(*it);
    }

    std::vector<swcdata_id> chain;

    while (!dfs_stack.empty()) {
        const auto head_swc_point_id = dfs_stack.top();
        dfs_stack.pop();

        const auto parent_swc_point_id = parent_swc_id(head_swc_point_id);
        const bool is_root = (parent_swc_point_id == invalid_swcdata_id);

        chain.clear();
        chain.push_back(head_swc_point_id);
        swcdata_id tail_swc_point_id = head_swc_point_id;

        for (;;) {
            const auto& tail_child_points = swc_point_children[static_cast<std::size_t>(tail_swc_point_id)];
            if (tail_child_points.size() != 1) {
                break;
            }
            const auto next_swc_point_id = tail_child_points.front();
            if (swc_data[static_cast<std::size_t>(next_swc_point_id)].label !=
                swc_data[static_cast<std::size_t>(tail_swc_point_id)].label) {
                break;
            }
            tail_swc_point_id = next_swc_point_id;
            chain.push_back(tail_swc_point_id);
        }

        Section sec{};
        sec.head_swc_id = head_swc_point_id;
        sec.label = swc_label_to_name_local(swc_data[static_cast<std::size_t>(head_swc_point_id)].label);
        sec.offset = morph.datas.size();
        sec.rallbranch = 1.0;
        // `nseg` is assigned later by an explicit nseg policy stage (fixed/target_length).
        sec.nseg = 0;

        if (is_root) {
            sec.parent_sec_id = invalid_section_id;
            sec.parentx = 1.0;
            sec.wire_first = false;
            sec.parent_swc_id = invalid_swcdata_id;
            for (const auto chain_swc_point_id : chain) {
                morph.datas.push_back(chain_swc_point_id);
            }
        } else {
            sec.parent_swc_id = parent_swc_point_id;
            const section_id parent_section_id = swc_point_to_section_id[static_cast<std::size_t>(parent_swc_point_id)];
            sec.parent_sec_id = parent_section_id;
            sec.parentx = 1.0;
            sec.wire_first = false;

            const bool parent_section_is_soma = neurong_labels::is_soma_label(morph.sections[parent_section_id].label);
            const bool is_non_soma = !neurong_labels::is_soma_label(sec.label);

            if (morph.sections[parent_section_id].parent_sec_id == invalid_section_id) {
                const bool parent_section_is_single_point_soma = (morph.sections[parent_section_id].count == 1);
                if (parent_section_is_soma && is_non_soma && parent_section_is_single_point_soma) {
                    sec.parentx = 0.5;
                    if (chain.size() > 1) {
                        sec.wire_first = true;
                    }
                } else if (parent_swc_point_id == morph.sections[parent_section_id].head_swc_id) {
                    sec.parentx = 0.0;
                    if (is_non_soma && nchild_soma[static_cast<std::size_t>(parent_swc_point_id)] > 1) {
                        sec.wire_first = true;
                    }
                }
            }
            if (!sec.wire_first) {
                const auto& parent_pt = swc_data[static_cast<std::size_t>(parent_swc_point_id)];
                const auto& head_pt = swc_data[static_cast<std::size_t>(head_swc_point_id)];
                if (parent_pt.xyz != head_pt.xyz) {
                    morph.datas.push_back(parent_swc_point_id);
                }
            }
            for (const auto chain_swc_point_id : chain) {
                morph.datas.push_back(chain_swc_point_id);
            }
        }

        sec.count = morph.datas.size() - sec.offset;
        const auto new_section_id = static_cast<section_id>(morph.sections.size());
        morph.sections.push_back(std::move(sec));

        // Maintain a label->sections lookup table during the initial build, so
        // channel insertion planning ("for label in ...") can avoid an extra pass.
        const auto label_u = ensure_label_index(morph, morph.sections[new_section_id].label);
        morph.sections_by_label[label_u].push_back(new_section_id);

        for (const auto chain_swc_point_id : chain) {
            swc_point_to_section_id[static_cast<std::size_t>(chain_swc_point_id)] = new_section_id;
        }

        const auto& tail_child_points = swc_point_children[static_cast<std::size_t>(tail_swc_point_id)];
        for (auto it = tail_child_points.rbegin(); it != tail_child_points.rend(); ++it) {
            dfs_stack.push(*it);
        }
    }


    auto add_child_sorted = [&](section_id parent_section_id, section_id child_section_id) {
        auto& child_section_ids = morph.sections[parent_section_id].children;
        const double child_parentx = morph.sections[child_section_id].parentx;
        auto it = child_section_ids.begin();
        for (; it != child_section_ids.end(); ++it) {
            const double existing_parentx = morph.sections[*it].parentx;
            if (child_parentx <= existing_parentx) {
                child_section_ids.insert(it, child_section_id);
                return;
            }
        }
        child_section_ids.push_back(child_section_id);
    };

    for (section_id child_section_id = 0; child_section_id < morph.sections.size(); ++child_section_id) {
        const auto parent_section_id = morph.sections[child_section_id].parent_sec_id;
        if (parent_section_id != invalid_section_id) {
            add_child_sorted(parent_section_id, child_section_id);
        }
    }

    assign_default_section_names(morph);
    morph.swc = std::move(swc_data);

    return morph;
}

std::vector<section_id> sections_with_label(const Morph& morph, std::string_view label) {
    if (label.empty()) {
        return {};
    }
    const auto it = morph.label_index.find(std::string(label));
    if (it == morph.label_index.end()) {
        return {};
    }
    const auto label_u = it->second;
    if (label_u >= morph.sections_by_label.size()) {
        return {};
    }
    return morph.sections_by_label[label_u];
}

}  // namespace neurong_morph
