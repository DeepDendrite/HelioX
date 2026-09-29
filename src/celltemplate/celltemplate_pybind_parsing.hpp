#pragma once

#include "biophys_pybind_parsing.hpp"
#include "celltemplate_builder.hpp"
#include "label_utils.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace neurong_pybind {
namespace nb = nanobind;

inline void ensure_only_allowed_keys(const nb::dict& d,
                                     std::initializer_list<std::string_view> allowed,
                                     std::string_view context) {
    std::unordered_set<std::string_view> allowed_keys;
    allowed_keys.reserve(allowed.size());
    for (const auto key : allowed) {
        allowed_keys.insert(key);
    }

    std::vector<std::string> unknown;
    for (auto item : d) {
        nb::handle key_obj = item.first;
        if (!nb::isinstance<nb::str>(key_obj)) {
            throw std::runtime_error(std::string(context) + " keys must be strings");
        }
        const auto key = nb::cast<std::string>(key_obj);
        if (allowed_keys.find(key) == allowed_keys.end()) {
            unknown.push_back(key);
        }
    }

    if (!unknown.empty()) {
        std::sort(unknown.begin(), unknown.end());
        std::string msg = std::string(context) + " contains unexpected field(s): ";
        for (std::size_t i = 0; i < unknown.size(); ++i) {
            if (i != 0) {
                msg += ", ";
            }
            msg += "'" + unknown[i] + "'";
        }
        throw std::runtime_error(msg);
    }
}

[[nodiscard]] inline neurong_celltemplate::SectionSpec parse_section_spec(const nb::object& obj) {
    if (nb::isinstance<neurong_celltemplate::SectionSpec>(obj)) {
        return nb::cast<neurong_celltemplate::SectionSpec>(obj);
    }
    nb::dict d = nb::borrow<nb::dict>(obj);
    ensure_only_allowed_keys(
        d,
        {"name", "parent", "parentx", "nseg", "label", "pt3d", "L_um", "diam_um"},
        "section");

    neurong_celltemplate::SectionSpec out{};
    out.name = nb::cast<std::string>(d["name"]);

    if (d.contains("parent")) {
        nb::object parent_obj = d["parent"];
        if (parent_obj.is_none()) {
            out.parent_name.clear();
        } else if (nb::isinstance<neurong_celltemplate::SectionSpec>(parent_obj)) {
            const auto parent = nb::cast<neurong_celltemplate::SectionSpec>(parent_obj);
            out.parent_name = parent.name;
        } else {
            out.parent_name = nb::cast<std::string>(parent_obj);
        }
    }
    if (d.contains("parentx")) {
        out.parentx = nb::cast<double>(d["parentx"]);
    }
    if (d.contains("nseg")) {
        out.nseg = nb::cast<std::int32_t>(d["nseg"]);
    } else {
        out.nseg = 1;
    }
    out.label = nb::cast<std::string>(d["label"]);

    const bool has_pt3d = d.contains("pt3d") && !d["pt3d"].is_none();
    if (has_pt3d) {
        nb::handle pt3d_obj = d["pt3d"];
        nb::sequence pt3d_seq = nb::borrow<nb::sequence>(pt3d_obj);
        for (nb::handle item : pt3d_seq) {
            nb::sequence p = nb::borrow<nb::sequence>(item);
            neurong_celltemplate::SectionPt3d out_p{};
            out_p.x_um = nb::cast<float>(p[0]);
            out_p.y_um = nb::cast<float>(p[1]);
            out_p.z_um = nb::cast<float>(p[2]);
            out_p.diam_um = nb::cast<float>(p[3]);
            out.pt3d.push_back(out_p);
        }
    } else {
        out.L_um = nb::cast<double>(d["L_um"]);
        out.diam_um = nb::cast<double>(d["diam_um"]);
    }
    return out;
}

[[nodiscard]] inline std::vector<neurong_celltemplate::SectionSpec> parse_sections(const nb::list& sections) {
    std::vector<neurong_celltemplate::SectionSpec> out;
    for (nb::handle item : sections) {
        out.push_back(parse_section_spec(nb::borrow<nb::object>(item)));
    }
    return out;
}

[[nodiscard]] inline std::vector<neurong_biophysical::IClampSpec> parse_iclamp(const nb::object& obj);

[[nodiscard]] inline std::vector<neurong_celltemplate::CellTemplateMorphSpec> parse_celltemplate_morphology_specs(
    const nb::list& morph_templates) {
    const std::size_t n = nb::len(morph_templates);

    std::vector<neurong_celltemplate::CellTemplateMorphSpec> out;
    out.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        nb::dict d = nb::borrow<nb::dict>(morph_templates[i]);
        ensure_only_allowed_keys(
            d,
            {"name", "num_cells", "sections"},
            "morph template");

        neurong_celltemplate::CellTemplateMorphSpec spec{};
        if (d.contains("name")) {
            spec.name = nb::cast<std::string>(d["name"]);
        }
        if (d.contains("num_cells")) {
            spec.num_cells = nb::cast<int>(d["num_cells"]);
        } else {
            spec.num_cells = 0;
        }
        if (d.contains("sections") && !d["sections"].is_none()) {
            spec.sections = parse_sections(nb::borrow<nb::list>(d["sections"]));
        } else {
            spec.sections.clear();
        }

        out.push_back(std::move(spec));
    }
    return out;
}

[[nodiscard]] inline std::vector<neurong_celltemplate::CellTemplateBiophysSpec> parse_celltemplate_biophys_specs(
    const nb::list& biophys_templates) {
    const std::size_t n = nb::len(biophys_templates);

    std::vector<neurong_celltemplate::CellTemplateBiophysSpec> out;
    out.reserve(n);

    auto parse_gid_selector = [](const nb::object& selector_obj) -> neurong_celltemplate::GidSelector {
        neurong_celltemplate::GidSelector sel{};

        if (nb::isinstance<nb::str>(selector_obj)) {
            const std::string raw = nb::cast<std::string>(selector_obj);
            if (raw != "all") {
                throw std::runtime_error("gid_selector string must be 'all'");
            }
            sel.kind = neurong_celltemplate::GidSelectorKind::All;
            return sel;
        }

        if (nb::isinstance<nb::dict>(selector_obj)) {
            nb::dict d = nb::borrow<nb::dict>(selector_obj);
            ensure_only_allowed_keys(d, {"begin", "end"}, "gid_selector");
            if (!d.contains("begin") || !d.contains("end")) {
                throw std::runtime_error("gid_selector range dict requires 'begin' and 'end'");
            }
            sel.kind = neurong_celltemplate::GidSelectorKind::Range;
            sel.gid_begin = nb::cast<int>(d["begin"]);
            sel.gid_end_exclusive = nb::cast<int>(d["end"]);
            return sel;
        }

        if (nb::isinstance<nb::sequence>(selector_obj) && !nb::isinstance<nb::str>(selector_obj)) {
            nb::sequence seq = nb::borrow<nb::sequence>(selector_obj);
            sel.kind = neurong_celltemplate::GidSelectorKind::Explicit;
            for (nb::handle item : seq) {
                sel.gids.push_back(nb::cast<int>(item));
            }
            if (sel.gids.empty()) {
                throw std::runtime_error("gid_selector explicit list must be non-empty");
            }
            return sel;
        }

        throw std::runtime_error("gid_selector must be 'all', a {begin,end} dict, or a list of gids");
    };

    for (std::size_t i = 0; i < n; ++i) {
        nb::dict d = nb::borrow<nb::dict>(biophys_templates[i]);
        ensure_only_allowed_keys(
            d,
            {
                "template_name",
                "gid_selector",
                "v_init",
                "inserts",
                "ion_species",
                "specificCapacitance",
                "resistivity",
                "iclamp",
            },
            "biophys template");

        neurong_celltemplate::CellTemplateBiophysSpec spec{};
        const bool has_template_name = d.contains("template_name") && !d["template_name"].is_none();
        const bool has_gid_selector = d.contains("gid_selector") && !d["gid_selector"].is_none();
        if (!has_template_name && !has_gid_selector) {
            throw std::runtime_error("biophys template requires at least one of template_name or gid_selector");
        }

        if (has_template_name) {
            spec.template_name = nb::cast<std::string>(d["template_name"]);
            if (spec.template_name.empty()) {
                throw std::runtime_error("biophys template template_name must be non-empty");
            }
        }

        if (has_template_name && has_gid_selector) {
            const auto sel = parse_gid_selector(d["gid_selector"]);
            if (sel.kind != neurong_celltemplate::GidSelectorKind::All) {
                throw std::runtime_error(
                    "when both template_name and gid_selector are provided, gid_selector must be 'all' "
                    "(template-local all)");
            }
            spec.match_kind = neurong_celltemplate::BiophysMatchKind::TemplateName;
        } else if (has_template_name) {
            spec.match_kind = neurong_celltemplate::BiophysMatchKind::TemplateName;
        } else {
            spec.match_kind = neurong_celltemplate::BiophysMatchKind::GidSelector;
            spec.gid_selector = parse_gid_selector(d["gid_selector"]);
        }

        if (!d.contains("v_init") || d["v_init"].is_none()) {
            throw std::runtime_error("biophys template requires non-null v_init");
        }
        spec.v_init = parse_scalar_or_random(d["v_init"], "biophys template v_init");

        if (d.contains("inserts") && !d["inserts"].is_none()) {
            spec.inserts = parse_biophys_inserts(nb::borrow<nb::list>(d["inserts"]));
        }
        if (d.contains("ion_species") && !d["ion_species"].is_none()) {
            parse_biophys_ion_species(nb::borrow<nb::list>(d["ion_species"]), spec.inserts);
        }
        if (d.contains("specificCapacitance") && !d["specificCapacitance"].is_none()) {
            parse_biophys_specific_capacitance(nb::borrow<nb::list>(d["specificCapacitance"]), spec.inserts);
        }
        if (d.contains("resistivity") && !d["resistivity"].is_none()) {
            parse_biophys_resistivity(nb::borrow<nb::list>(d["resistivity"]), spec.inserts);
        }

        if (d.contains("iclamp") && !d["iclamp"].is_none()) {
            auto local_iclamp = parse_iclamp(d["iclamp"]);
            spec.iclamp = std::move(local_iclamp);
        }

        out.push_back(std::move(spec));
    }
    return out;
}

[[nodiscard]] inline neurong_biophysical::IClampSpec parse_iclamp_spec(const nb::object& obj) {
    nb::dict d = nb::borrow<nb::dict>(obj);
    ensure_only_allowed_keys(
        d,
        {"cells", "section", "x", "delay", "dur", "amp"},
        "iclamp");
    neurong_biophysical::IClampSpec out{};
    if (!d.contains("cells")) {
        throw std::runtime_error("iclamp requires 'cells'");
    }
    nb::handle cells_obj = d["cells"];
    if (cells_obj.is_none()) {
        throw std::runtime_error("iclamp 'cells' must be 'all', an int, or a sequence of ints");
    }
    if (nb::isinstance<nb::str>(cells_obj)) {
        const nb::str s = nb::borrow<nb::str>(cells_obj);
        if (std::string_view(s.c_str()) == "all") {
            out.all_cells = true;
        } else {
            throw std::runtime_error("iclamp 'cells' string must be 'all'");
        }
    } else if (nb::isinstance<nb::int_>(cells_obj)) {
        out.cells.push_back(nb::cast<int>(cells_obj));
    } else {
        nb::sequence seq = nb::borrow<nb::sequence>(cells_obj);
        for (nb::handle item : seq) {
            out.cells.push_back(nb::cast<int>(item));
        }
    }

    nb::handle section_obj = d["section"];
    if (nb::isinstance<nb::int_>(section_obj)) {
        out.section_id = nb::cast<int>(section_obj);
    } else {
        out.section_name = nb::cast<std::string>(section_obj);
    }

    out.x = nb::cast<double>(d["x"]);
    out.delay = nb::cast<double>(d["delay"]);
    out.dur = nb::cast<double>(d["dur"]);
    out.amp = nb::cast<double>(d["amp"]);

    return out;
}

[[nodiscard]] inline std::vector<neurong_biophysical::IClampSpec> parse_iclamp(const nb::object& obj) {
    if (obj.is_none()) {
        return {};
    }
    nb::sequence seq = nb::borrow<nb::sequence>(obj);
    std::vector<neurong_biophysical::IClampSpec> out;
    for (nb::handle item : seq) {
        out.push_back(parse_iclamp_spec(nb::borrow<nb::object>(item)));
    }
    return out;
}

}  // namespace neurong_pybind
