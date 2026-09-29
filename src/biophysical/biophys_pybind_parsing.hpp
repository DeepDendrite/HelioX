#pragma once

#include "biophys_builder.hpp"
#include "label_utils.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace neurong_pybind {
namespace nb = nanobind;

[[nodiscard]] inline neurong_biophysical::RandomParamSpec parse_random_param_spec(
    const nb::dict& d) {
    neurong_biophysical::RandomParamSpec spec{};

    if (!d.contains("distribution")) {
        throw std::runtime_error("random spec dict must contain key 'distribution'");
    }
    const std::string distribution = nb::cast<std::string>(d["distribution"]);

    if (distribution == "fixed") {
        spec.distribution = neurong_biophysical::RandomDist::Fixed;
        if (!d.contains("value")) {
            throw std::runtime_error("fixed random spec requires key 'value'");
        }
        spec.value = nb::cast<double>(d["value"]);
        return spec;
    }

    if (distribution == "uniform") {
        spec.distribution = neurong_biophysical::RandomDist::Uniform;
        if (!d.contains("low") || !d.contains("high")) {
            throw std::runtime_error("uniform random spec requires keys 'low' and 'high'");
        }
        spec.low = nb::cast<double>(d["low"]);
        spec.high = nb::cast<double>(d["high"]);
        if (spec.high < spec.low) {
            throw std::runtime_error("uniform random spec requires high >= low");
        }
        return spec;
    }

    if (distribution == "normal") {
        spec.distribution = neurong_biophysical::RandomDist::Normal;
        if (!d.contains("mean") || !d.contains("std")) {
            throw std::runtime_error("normal random spec requires keys 'mean' and 'std'");
        }
        spec.mean = nb::cast<double>(d["mean"]);
        spec.stddev = nb::cast<double>(d["std"]);
        if (spec.stddev < 0.0) {
            throw std::runtime_error("normal random spec requires std >= 0");
        }
        return spec;
    }

    throw std::runtime_error("random distribution must be one of: fixed, uniform, normal");
}

[[nodiscard]] inline neurong_biophysical::ScalarOrRandom parse_scalar_or_random(
    nb::handle value,
    std::string_view context) {
    double scalar{};
    if (nb::try_cast(value, scalar)) {
        return scalar;
    }
    if (nb::isinstance<nb::dict>(value)) {
        nb::dict d = nb::borrow<nb::dict>(value);
        return parse_random_param_spec(d);
    }
    throw std::runtime_error(std::string(context) + " must be scalar or random spec dict");
}

[[nodiscard]] inline std::vector<double> parse_param_vector(nb::handle value, std::string_view ctx) {
    nb::ndarray<const double, nb::shape<-1>, nb::c_contig> arr;
    try {
        if (nb::try_cast(value, arr)) {
            std::vector<double> out;
            const std::size_t n = arr.shape(0);
            out.assign(arr.data(), arr.data() + n);
            return out;
        }
    } catch (...) {
        // Fall back to sequence handling when nanobind throws on ndarray casts.
    }

    if (nb::isinstance<nb::sequence>(value) && !nb::isinstance<nb::str>(value)) {
        nb::sequence seq = nb::borrow<nb::sequence>(value);
        std::vector<double> out;
        for (nb::handle item : seq) {
            out.push_back(nb::cast<double>(item));
        }
        return out;
    }

    throw std::runtime_error(std::string(ctx) + " must be a 1D array or sequence of floats");
}

[[nodiscard]] inline neurong_biophysical::ParamValue parse_param_value(nb::handle value) {
    if (nb::isinstance<nb::str>(value)) {
        return nb::cast<std::string>(value);
    }

    double scalar{};
    if (nb::try_cast(value, scalar)) {
        return scalar;
    }

    if (nb::isinstance<nb::dict>(value)) {
        nb::dict d = nb::borrow<nb::dict>(value);
        if (d.contains("distribution")) {
            return parse_random_param_spec(d);
        }
        neurong_biophysical::SectionParamMap out{};
        for (auto kv : d) {
            const nb::str key = nb::borrow<nb::str>(kv.first);
            nb::handle v = kv.second;
            if (nb::isinstance<nb::dict>(v)) {
                nb::dict vd = nb::borrow<nb::dict>(v);
                out.emplace(std::string_view(key.c_str()), parse_random_param_spec(vd));
            } else {
                out.emplace(std::string_view(key.c_str()), parse_param_vector(v, "section param value"));
            }
        }
        return out;
    }

    throw std::runtime_error(
        "param value must be string, scalar, random spec dict, or dict[str, sequence|random spec dict]");
}

[[nodiscard]] inline neurong_biophysical::SectionScalarValue parse_section_scalar_value(nb::handle value,
                                                                                         std::string_view ctx) {
    double scalar{};
    if (nb::try_cast(value, scalar)) {
        return scalar;
    }

    if (nb::isinstance<nb::dict>(value)) {
        nb::dict d = nb::borrow<nb::dict>(value);
        if (d.contains("distribution")) {
            return parse_random_param_spec(d);
        }
        neurong_biophysical::SectionParamMap out{};
        for (auto kv : d) {
            const nb::str key = nb::borrow<nb::str>(kv.first);
            nb::handle v = kv.second;
            if (nb::isinstance<nb::dict>(v)) {
                nb::dict vd = nb::borrow<nb::dict>(v);
                out.emplace(std::string_view(key.c_str()), parse_random_param_spec(vd));
            } else {
                out.emplace(std::string_view(key.c_str()), parse_param_vector(v, "section scalar map value"));
            }
        }
        return out;
    }

    throw std::runtime_error(
        std::string(ctx) + " must be scalar, random spec dict, or dict[str, sequence|random spec dict]");
}

[[nodiscard]] inline neurong_biophysical::Prepare parse_biophys_inserts(const nb::list& inserts) {
    neurong_biophysical::Prepare out{};

    for (nb::handle item : inserts) {

        nb::sequence seq = nb::borrow<nb::sequence>(item);

        nb::handle label_obj = seq[0];
        const auto label_name = nb::cast<std::string>(label_obj);
        const auto mech = nb::cast<std::string>(seq[1]);

        nb::handle third = seq[2];
        neurong_biophysical::ParamList params{};
        nb::dict d = nb::borrow<nb::dict>(third);
        for (auto kv : d) {
            const nb::str key = nb::borrow<nb::str>(kv.first);
            params.emplace_back(std::string_view(key.c_str()), parse_param_value(kv.second));
        }

        out.insert(label_name, mech, std::move(params));
    }
    return out;
}

inline void parse_biophys_ion_species(const nb::list& species,
                                      neurong_biophysical::Prepare& out) {
    for (nb::handle item : species) {
        nb::sequence seq = nb::borrow<nb::sequence>(item);

        nb::handle label_obj = seq[0];
        const auto label_name = nb::cast<std::string>(label_obj);

        const auto ion_name = nb::cast<std::string>(seq[1]);
        const auto model_name = nb::cast<std::string>(seq[2]);

        nb::handle params_obj = seq[3];
        nb::dict d = nb::borrow<nb::dict>(params_obj);
        neurong_biophysical::ParamList params{};
        for (auto kv : d) {
            const nb::str key = nb::borrow<nb::str>(kv.first);
            params.emplace_back(std::string_view(key.c_str()), parse_param_value(kv.second));
        }

        out.ion_species(label_name, ion_name, model_name, std::move(params));
    }
}

inline void parse_biophys_specific_capacitance(const nb::list& items,
                                               neurong_biophysical::Prepare& out) {
    for (nb::handle item : items) {
        nb::sequence seq = nb::borrow<nb::sequence>(item);
        const auto label_name = nb::cast<std::string>(seq[0]);
        const auto value = parse_section_scalar_value(seq[1], "specificCapacitance value");
        out.specific_capacitance(label_name, std::move(value));
    }
}

inline void parse_biophys_resistivity(const nb::list& items,
                                      neurong_biophysical::Prepare& out) {
    for (nb::handle item : items) {
        nb::sequence seq = nb::borrow<nb::sequence>(item);
        const auto label_name = nb::cast<std::string>(seq[0]);
        const auto value = parse_section_scalar_value(seq[1], "resistivity value");
        out.resistivity(label_name, std::move(value));
    }
}

}  // namespace neurong_pybind
