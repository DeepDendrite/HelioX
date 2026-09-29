#pragma once

#include "network/network_builder.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurong_pybind {
namespace nb = nanobind;

[[nodiscard]] inline bool has_non_none(const nb::dict& d, const char* key) {
    return d.contains(key) && !d[key].is_none();
}

enum class ParseConnectionMode {
    RealOnly = 0,
    ArtificialOnly = 1,
};

[[nodiscard]] inline neurong_network::ConnectionSpec parse_connection_spec(
    const nb::object& obj,
    int default_id,
    ParseConnectionMode mode) {
    nb::dict d = nb::borrow<nb::dict>(obj);

    const char* required[] = {
        "postGid",
        "postSection",
        "postSectionLocation",
        "postMech",
        "postParams",
    };
    for (const char* key : required) {
        if (!d.contains(key) || d[key].is_none()) {
            throw std::runtime_error(std::string("connection missing required field '") + key + "'");
        }
    }

    neurong_network::ConnectionSpec out{};
    if (has_non_none(d, "id")) {
        out.id = nb::cast<int>(d["id"]);
    } else {
        out.id = default_id;
    }

    const bool has_post_template_name = has_non_none(d, "postTemplateName");
    const bool has_post_template_index = has_non_none(d, "postTemplateIndex");
    if (has_post_template_name || has_post_template_index) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) +
            " uses deprecated postTemplateName/postTemplateIndex; only postGid is supported");
    }
    out.postCellId = nb::cast<int>(d["postGid"]);
    if (out.postCellId < 0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) +
            " has invalid postGid=" + std::to_string(out.postCellId));
    }

    const bool has_pre_mech = has_non_none(d, "preMech");
    const bool has_pre_gid = has_non_none(d, "preGid");
    const bool has_pre_section = has_non_none(d, "preSection");
    const bool has_pre_section_location = has_non_none(d, "preSectionLocation");
    const bool has_threshold = has_non_none(d, "threshold");
    const bool has_pre_slot = has_non_none(d, "preSlot");
    if (has_pre_mech) {
        if (has_threshold) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " sets threshold on artificial pre; threshold is only valid for real-cell pre");
        }
        if (has_pre_slot) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " sets deprecated preSlot; preSlot is no longer supported");
        }
        out.preKind = neurong_network::PreEndpointKind::ArtificialByLoc;
        out.preMech = nb::cast<std::string>(d["preMech"]);
        out.preGid = has_pre_gid ? nb::cast<int>(d["preGid"]) : out.postCellId;
        out.preCellId = out.preGid;
        if (out.preGid < 0) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " has invalid preGid=" + std::to_string(out.preGid));
        }
        if (has_pre_section != has_pre_section_location) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " artificial pre endpoint requires both preSection and preSectionLocation when either is set");
        }
        if (has_pre_section) {
            out.preSection = nb::cast<std::string>(d["preSection"]);
            out.preSectionLocation = nb::cast<double>(d["preSectionLocation"]);
        } else {
            // For artificial pre, allow omitting pre endpoint fields and infer from post endpoint.
            nb::object post_sec_obj = d["postSection"];
            if (nb::isinstance<nb::str>(post_sec_obj)) {
                out.preSection = nb::cast<std::string>(post_sec_obj);
            } else {
                out.preSection = "soma";
            }
            out.preSectionLocation = nb::cast<double>(d["postSectionLocation"]);
        }
        if (out.preMech.empty()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) + " has empty preMech");
        }
        if (out.preSection.empty()) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) + " has empty preSection");
        }
    } else {
        if (has_pre_slot) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " sets deprecated preSlot; preSlot is no longer supported");
        }
        if (!has_pre_gid) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " real pre endpoint requires preGid");
        }
        out.preGid = nb::cast<int>(d["preGid"]);
        if (out.preGid < 0) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " has invalid preGid=" + std::to_string(out.preGid));
        }
        if (has_pre_section != has_pre_section_location) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " real pre endpoint requires both preSection and preSectionLocation when either is set");
        }
        out.preKind = neurong_network::PreEndpointKind::RealCell;
        out.preMech.clear();
        if (has_pre_section) {
            out.preSection = nb::cast<std::string>(d["preSection"]);
            out.preSectionLocation = nb::cast<double>(d["preSectionLocation"]);
            if (out.preSection.empty()) {
                throw std::runtime_error(
                    "connection id=" + std::to_string(out.id) + " has empty preSection");
            }
        } else {
            // Keep baseline default source endpoint when caller does not specify one.
            out.preSection = "soma";
            out.preSectionLocation = 0.5;
        }
        if (!has_threshold) {
            throw std::runtime_error(
                "connection id=" + std::to_string(out.id) +
                " real pre endpoint requires threshold");
        }
        out.threshold = nb::cast<double>(d["threshold"]);
        out.preCellId = out.preGid;
    }

    nb::object sec_obj = d["postSection"];
    if (nb::isinstance<nb::int_>(sec_obj)) {
        out.postSectionIsIndex = true;
        out.postSectionIndex = nb::cast<int>(sec_obj);
    } else {
        out.postSection = nb::cast<std::string>(sec_obj);
        out.postSectionIsIndex = false;
    }

    out.postSectionLocation = nb::cast<double>(d["postSectionLocation"]);
    out.postMech = nb::cast<std::string>(d["postMech"]);
    if (out.postMech.empty()) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) + " has empty postMech");
    }

    nb::dict post_params;
    try {
        post_params = nb::cast<nb::dict>(d["postParams"]);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) + " field 'postParams' must be a dict");
    }

    bool has_weight = false;
    bool has_delay = false;
    for (auto item : post_params) {
        const std::string key = nb::cast<std::string>(item.first);
        if (key == "weight") {
            out.weight = nb::cast<double>(item.second);
            has_weight = true;
            continue;
        }
        if (key == "delay") {
            out.delay = nb::cast<double>(item.second);
            has_delay = true;
            continue;
        }
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) +
            " has unsupported postParams key '" + key +
            "' for postMech='" + out.postMech +
            "'; expected exactly {'weight', 'delay'}");
    }
    if (!has_weight || !has_delay || nb::len(post_params) != 2) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) +
            " postParams for postMech='" + out.postMech +
            "' must contain exactly {'weight', 'delay'}");
    }

    if (!std::isfinite(out.postSectionLocation) ||
        !std::isfinite(out.preSectionLocation) ||
        !std::isfinite(out.weight) ||
        !std::isfinite(out.delay) ||
        (out.preKind == neurong_network::PreEndpointKind::RealCell &&
         !std::isfinite(out.threshold))) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) + " has non-finite numeric fields");
    }
    if (out.delay < 0.0) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) + " has negative postParams.delay");
    }
    if (mode == ParseConnectionMode::RealOnly &&
        out.preKind != neurong_network::PreEndpointKind::RealCell) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) +
            " is artificial pre, but load_network_real expects real pre only");
    }
    if (mode == ParseConnectionMode::ArtificialOnly &&
        out.preKind != neurong_network::PreEndpointKind::ArtificialByLoc) {
        throw std::runtime_error(
            "connection id=" + std::to_string(out.id) +
            " is real pre, but load_network_artificial expects artificial pre only");
    }

    return out;
}

[[nodiscard]] inline std::vector<neurong_network::ConnectionSpec> parse_connections_with_mode(
    const nb::list& connections,
    ParseConnectionMode mode) {
    std::vector<neurong_network::ConnectionSpec> out;
    out.reserve(nb::len(connections));
    int idx = 0;
    for (nb::handle item : connections) {
        out.push_back(parse_connection_spec(nb::borrow<nb::object>(item), idx, mode));
        ++idx;
    }
    return out;
}

[[nodiscard]] inline std::vector<neurong_network::ConnectionSpec> parse_connections_real_only(
    const nb::list& connections) {
    return parse_connections_with_mode(connections, ParseConnectionMode::RealOnly);
}

[[nodiscard]] inline std::vector<neurong_network::ConnectionSpec> parse_connections_artificial_only(
    const nb::list& connections) {
    return parse_connections_with_mode(connections, ParseConnectionMode::ArtificialOnly);
}

}  // namespace neurong_pybind
