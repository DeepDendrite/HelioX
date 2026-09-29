#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <variant>

#include "label_utils.hpp"
#include <vector>

namespace neurong_biophysical {

// Label selection helper:
// - Use label names like "soma"/"axon"/"dend"/"apic".
// - Use "all" to represent the "all labels" selection (NEURON-style `allsec`).
inline constexpr std::string_view AllLabels = "all";

// Prepare/build layer: record NEURON-like "insert <mech> by label" operations, then expand
// once to per-node mechanism instance indices aligned with the node layout from
// `build_nodes_neuron_compatible_with_layout()`.

// Param values can be either:
//  - scalar double (uniform over the selected label)
//  - per-section values keyed by section name, where each section entry can be:
//      * a per-segment vector<double>
//      * a random spec dict
//  - string (used for ion metadata like `ion` / `erev_type`)
//  - random distribution spec (sampled at build time)
enum class RandomDist {
    Fixed,
    Uniform,
    Normal,
};

struct RandomParamSpec {
    RandomDist distribution{RandomDist::Fixed};
    double value{0.0};   // fixed
    double low{0.0};     // uniform
    double high{0.0};    // uniform
    double mean{0.0};    // normal
    double stddev{0.0};  // normal

    [[nodiscard]] bool is_random() const noexcept {
        return distribution != RandomDist::Fixed;
    }
};

[[nodiscard]] inline bool operator==(const RandomParamSpec& a, const RandomParamSpec& b) noexcept {
    return a.distribution == b.distribution &&
           a.value == b.value &&
           a.low == b.low && a.high == b.high &&
           a.mean == b.mean && a.stddev == b.stddev;
}

using SectionParamEntry = std::variant<std::vector<double>, RandomParamSpec>;
using SectionParamMap = std::unordered_map<std::string, SectionParamEntry>;
using SectionScalarValue = std::variant<double, SectionParamMap, RandomParamSpec>;
using ScalarOrRandom = std::variant<double, RandomParamSpec>;
using ParamValue = std::variant<double, SectionParamMap, std::string, RandomParamSpec>;
using ParamList = std::vector<std::pair<std::string, ParamValue>>;

struct InsertOp {
    std::string label{};
    std::string mech{};
    ParamList params{};
};

struct IonSpeciesOp {
    std::string label{};
    std::string ion{};
    std::string model{};
    ParamList params{};
};

struct SectionPropertyOp {
    std::string label{};
    SectionScalarValue value{};
};

// Record-only prepare: users "insert" mechanisms by label with optional param overrides,
// then a final build step expands everything to per-node SoA.
class Prepare {
public:
    void insert(std::string_view label, std::string_view mech_name, ParamList params = {}) {
        const auto label_name = std::string(label);
        if (label_name.empty()) {
            throw std::runtime_error("insert label name is empty (use \"all\" to target all labels)");
        }
        inserts_.push_back(InsertOp{.label = label_name,
                                    .mech = std::string(mech_name),
                                    .params = std::move(params)});
    }

    void ion_species(std::string_view label,
                     std::string_view ion_name,
                     std::string_view model_name,
                     ParamList params = {}) {
        const auto label_name = std::string(label);
        if (label_name.empty()) {
            throw std::runtime_error("ion_species label name is empty (use \"all\" to target all labels)");
        }
        ion_species_.push_back(IonSpeciesOp{.label = label_name,
                                            .ion = std::string(ion_name),
                                            .model = std::string(model_name),
                                            .params = std::move(params)});
    }

    void specific_capacitance(std::string_view label, SectionScalarValue value) {
        const auto label_name = std::string(label);
        if (label_name.empty()) {
            throw std::runtime_error("specificCapacitance label name is empty (use \"all\" to target all labels)");
        }
        specific_capacitance_.push_back(SectionPropertyOp{
            .label = label_name,
            .value = std::move(value),
        });
    }

    void resistivity(std::string_view label, SectionScalarValue value) {
        const auto label_name = std::string(label);
        if (label_name.empty()) {
            throw std::runtime_error("resistivity label name is empty (use \"all\" to target all labels)");
        }
        resistivity_.push_back(SectionPropertyOp{
            .label = label_name,
            .value = std::move(value),
        });
    }

    [[nodiscard]] const std::vector<InsertOp>& inserts() const noexcept {
        return inserts_;
    }

    [[nodiscard]] const std::vector<IonSpeciesOp>& ion_species() const noexcept {
        return ion_species_;
    }

    [[nodiscard]] const std::vector<SectionPropertyOp>& specific_capacitance() const noexcept {
        return specific_capacitance_;
    }

    [[nodiscard]] const std::vector<SectionPropertyOp>& resistivity() const noexcept {
        return resistivity_;
    }

private:
    std::vector<InsertOp> inserts_{};
    std::vector<IonSpeciesOp> ion_species_{};
    std::vector<SectionPropertyOp> specific_capacitance_{};
    std::vector<SectionPropertyOp> resistivity_{};
};

}  // namespace neurong_biophysical
