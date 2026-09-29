#include "biophys_builder.hpp"

#include "cuda_utils.h"
#include "global_vars.h"
#include "ion_table.h"
#include "mech_var_table.h"
#include "mech_builder.hpp"
#include "random_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <memory>
#include <span>
#include <stdexcept>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace neurong_biophysical {
namespace {

template <typename T>
struct LabelOverrides {
    std::optional<T> all{};
    std::unordered_map<std::string, T> by_label{};

    void set(std::string_view label, T value) {
        if (label == neurong_biophysical::AllLabels) {
            all = std::move(value);
        } else {
            by_label[std::string(label)] = std::move(value);
        }
    }

    [[nodiscard]] std::optional<T> get(const std::string& label) const {
        if (label != neurong_biophysical::AllLabels) {
            auto it = by_label.find(label);
            if (it != by_label.end()) {
                return it->second;
            }
        }
        return all;
    }

    [[nodiscard]] std::optional<T> get(std::string_view label) const {
        return get(std::string(label));
    }
};

[[nodiscard]] inline std::string to_lower_ascii(std::string_view s) {
    std::string out;
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

[[nodiscard]] inline bool is_non_specific_ion(std::string_view name) {
    return name == "non_specific";
}

enum class ErevType {
    Fixed,
    Nernst,
};

struct IonSpec {
    ErevType erev_type{ErevType::Fixed};
    std::optional<ScalarOrRandom> erev{};
    std::optional<ScalarOrRandom> conci{};
    std::optional<ScalarOrRandom> conco{};
};

struct IonMetaSpec {
    bool has_charge{false};
    double charge{0.0};
};

struct IonSpecPatch {
    std::optional<ErevType> erev_type{};
    std::optional<ScalarOrRandom> erev{};
    std::optional<ScalarOrRandom> conci{};
    std::optional<ScalarOrRandom> conco{};
};

[[nodiscard]] inline bool ion_spec_equal(const IonSpec& a, const IonSpec& b) {
    return a.erev_type == b.erev_type &&
           a.erev == b.erev &&
           a.conci == b.conci &&
           a.conco == b.conco;
}

[[nodiscard]] inline std::optional<ScalarOrRandom> parse_scalar_or_random_ion(
    std::string_view ion_name,
    const std::optional<ParamValue>& value,
    std::string_view field) {
    if (!value.has_value()) {
        return std::nullopt;
    }
    if (std::holds_alternative<double>(*value)) {
        return std::get<double>(*value);
    }
    if (std::holds_alternative<RandomParamSpec>(*value)) {
        return std::get<RandomParamSpec>(*value);
    }
    throw std::runtime_error("ion " + std::string(ion_name) + " " +
                             std::string(field) + " must be scalar or random spec");
}

[[nodiscard]] inline IonSpecPatch parse_ion_patch(std::string_view ion_name,
                                                  const std::optional<std::string>& erev_type_raw,
                                                  const std::optional<ParamValue>& erev_value,
                                                  const std::optional<ParamValue>& ion_conci_value,
                                                  const std::optional<ParamValue>& ion_conco_value) {
    IonSpecPatch patch{};
    if (erev_type_raw.has_value()) {
        const std::string& type = *erev_type_raw;
        if (type == "fixed") {
            patch.erev_type = ErevType::Fixed;
        } else if (type == "nernst") {
            patch.erev_type = ErevType::Nernst;
        } else {
            throw std::runtime_error("unsupported erev_type for ion " + std::string(ion_name));
        }
    }

    if (erev_value.has_value()) {
        patch.erev = parse_scalar_or_random_ion(ion_name, erev_value, "erev");
        if (!patch.erev_type.has_value()) {
            patch.erev_type = ErevType::Fixed;
        }
    }

    patch.conci = parse_scalar_or_random_ion(ion_name, ion_conci_value, "ion_conci");
    patch.conco = parse_scalar_or_random_ion(ion_name, ion_conco_value, "ion_conco");
    return patch;
}

inline void merge_ion_patch(IonSpec& spec,
                            const IonSpecPatch& patch,
                            const std::string& ion_name) {
    if (patch.erev_type.has_value() && spec.erev_type != *patch.erev_type) {
        throw std::runtime_error("conflicting ion erev_type for ion " + ion_name);
    }
    if (patch.erev.has_value()) {
        if (spec.erev.has_value() && spec.erev != *patch.erev) {
            throw std::runtime_error("conflicting ion erev for ion " + ion_name);
        }
        spec.erev = *patch.erev;
    }
    if (patch.conci.has_value()) {
        if (spec.conci.has_value() && spec.conci != *patch.conci) {
            throw std::runtime_error("conflicting ion_conci for ion " + ion_name);
        }
        spec.conci = *patch.conci;
    }
    if (patch.conco.has_value()) {
        if (spec.conco.has_value() && spec.conco != *patch.conco) {
            throw std::runtime_error("conflicting ion_conco for ion " + ion_name);
        }
        spec.conco = *patch.conco;
    }
}

[[nodiscard]] inline IonSpec parse_ion_spec(std::string_view ion_name,
                                            const std::optional<std::string>& erev_type_raw,
                                            const std::optional<ParamValue>& erev_value,
                                            const std::optional<ParamValue>& ion_conci_value,
                                            const std::optional<ParamValue>& ion_conco_value) {
    IonSpec spec{};
    if (!erev_type_raw.has_value()) {
        spec.erev_type = ErevType::Fixed;
    } else {
        const std::string& type = *erev_type_raw;
        if (type == "fixed") {
            spec.erev_type = ErevType::Fixed;
        } else if (type == "nernst") {
            spec.erev_type = ErevType::Nernst;
        } else {
            throw std::runtime_error("unsupported erev_type for ion " + std::string(ion_name));
        }
    }

    if (spec.erev_type == ErevType::Fixed) {
        if (!erev_value.has_value()) {
            throw std::runtime_error("ion " + std::string(ion_name) +
                                     " requires erev when erev_type is fixed");
        }
        spec.erev = parse_scalar_or_random_ion(ion_name, erev_value, "erev");
    }

    spec.conci = parse_scalar_or_random_ion(ion_name, ion_conci_value, "ion_conci");
    spec.conco = parse_scalar_or_random_ion(ion_name, ion_conco_value, "ion_conco");
    return spec;
}

template <typename T>
inline void permute_blocks_inplace(std::vector<T>& data,
                                   const std::vector<int>& permute_old_to_new,
                                   std::size_t block_size) {
    const std::size_t n = permute_old_to_new.size();
    if (n < 2) {
        return;
    }

    std::vector<unsigned char> visited(n, 0);
    std::vector<T> tmp(block_size);
    T* base = data.data();

    for (std::size_t start = 0; start < n; ++start) {
        if (visited[start]) {
            continue;
        }
        const std::size_t start_new = static_cast<std::size_t>(permute_old_to_new[start]);
        if (start_new == start) {
            visited[start] = 1;
            continue;
        }

        std::size_t current = start;
        std::copy_n(base + current * block_size, block_size, tmp.data());

        while (true) {
            const std::size_t dest = static_cast<std::size_t>(permute_old_to_new[current]);
            visited[current] = 1;

            if (dest == start) {
                std::copy_n(tmp.data(), block_size, base + dest * block_size);
                break;
            }

            T* dest_ptr = base + dest * block_size;
            for (std::size_t j = 0; j < block_size; ++j) {
                std::swap(tmp[j], dest_ptr[j]);
            }
            current = dest;
        }
    }
}

template <typename IndexT>
[[nodiscard]] std::vector<int> stable_sort_node_indices_inplace(std::vector<IndexT>& node_indices) {
    static_assert(std::is_integral_v<IndexT>, "node index type must be integral");
    const std::size_t n = node_indices.size();
    std::vector<int> permute_old_to_new(n);
    if (n < 2) {
        for (std::size_t i = 0; i < n; ++i) {
            permute_old_to_new[i] = static_cast<int>(i);
        }
        return permute_old_to_new;
    }

    // Match CoreNEURON's nrn_index_sort_cmp: sort by node index, then by original instance index.
    std::vector<std::pair<IndexT, int>> vi;
    vi.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        vi[i].first = node_indices[i];
        vi[i].second = static_cast<int>(i);
    }
    std::sort(vi.begin(), vi.end(), [](const auto& a, const auto& b) {
        if (a.first < b.first) {
            return true;
        }
        if (a.first > b.first) {
            return false;
        }
        return a.second < b.second;
    });

    // vi[new_pos].second == old_pos; invert to get permute: old_pos -> new_pos.
    for (std::size_t new_pos = 0; new_pos < n; ++new_pos) {
        const int old_pos = vi[new_pos].second;
        permute_old_to_new[static_cast<std::size_t>(old_pos)] = static_cast<int>(new_pos);
    }

    permute_blocks_inplace(node_indices, permute_old_to_new, /*block_size=*/1);
    return permute_old_to_new;
}

[[nodiscard]] inline double sample_random_with_context(const RandomParamSpec& rule,
                                                       std::uint64_t base_seed,
                                                       std::uint64_t global_cell_id,
                                                       std::uint64_t label_u,
                                                       std::uint64_t section_u,
                                                       std::uint64_t segment_u) {
    (void)segment_u;
    // RandomScope has been removed from public and internal params.
    // Biophysical random sampling is fixed to one value per (cell, section).
    std::uint64_t seed = combine_seed_many(base_seed, global_cell_id, label_u, section_u);
    return sample_random_rule(rule, seed);
}

[[nodiscard]] inline double sample_scalar_or_random_with_context(const ScalarOrRandom& value,
                                                                 std::uint64_t base_seed,
                                                                 std::uint64_t global_cell_id,
                                                                 std::uint64_t label_u,
                                                                 std::uint64_t section_u,
                                                                 std::uint64_t segment_u) {
    if (std::holds_alternative<double>(value)) {
        return std::get<double>(value);
    }
    const auto& rule = std::get<RandomParamSpec>(value);
    return sample_random_with_context(rule, base_seed, global_cell_id, label_u, section_u, segment_u);
}

inline void set_global_scalar(std::string name, double value) {
    coreneuron::global_var_map[std::move(name)] = std::vector<double>{value};
}

}  // namespace

NeuronGBiophysBuildResult build_neurong_biophysics(Simulate& sim,
                                                   NeuronGroupData& group,
                                                   const CellTemplateMorphLayout& morph,
                                                   double celsius,
                                                   const std::vector<IClampSpec>& iclamp,
                                                   const std::vector<int>& record_gids,
                                                   const std::vector<VarDescriptor>& extra_monitors,
                                                   int& next_type) {

    const int num_cells = morph.num_cells_total;
    const std::size_t nnode = morph.nnode;

    auto map_template_node_to_original = [&](int cell_id, std::int32_t template_node_index) -> std::int32_t {
        if (template_node_index < 0) {
            return -1;
        }
        if (template_node_index == 0) {
            return static_cast<std::int32_t>(cell_id);
        }
        const auto cell_u = static_cast<std::size_t>(cell_id);
        const auto base = morph.cell_nonroot_base[cell_u];
        const auto idx = base + (static_cast<std::size_t>(template_node_index) - 1);
        return static_cast<std::int32_t>(idx);
    };

    struct ParsedTemplate {
        const CellTemplateInfo* tpl{nullptr};
        neurong_morph::NodeBuildLayout layout{};
        std::unordered_map<std::string, LabelOverrides<IonSpec>> ion_overrides{};
        std::unordered_map<std::string, std::vector<std::uint8_t>> ion_use_label{};
        neurong_morph::SectionLabelSegmentLayout section_label_layout{};
        std::vector<double> section_ra{};
        std::vector<double> section_cm{};
        std::uint64_t random_seed{0};
        Prepare prepare{};
        BuildResult built{};
    };

    std::vector<ParsedTemplate> parsed;
    parsed.resize(morph.templates.size());

    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> mech_ion_overrides;
    std::unordered_map<std::string, IonMetaSpec> ion_meta_overrides;
    std::unordered_map<std::string, bool> ion_needs_charge;
    std::unordered_set<std::string> ion_species_mechs;
    std::unordered_map<std::string, double> global_param_values;

    for (std::size_t t = 0; t < morph.templates.size(); ++t) {
        const auto& tpl = morph.templates[t];

        ParsedTemplate st{};
        st.tpl = &tpl;
        st.layout.section_node_base_id = tpl.section_node_base_id;
        st.layout.section_nseg = tpl.section_nseg;
        st.section_ra = tpl.section_ra;
        st.section_cm = tpl.section_cm;
        st.random_seed = combine_seed(tpl.random_seed, hash_string64(tpl.name));
        st.prepare = tpl.prepare;

        const std::size_t num_sections = st.layout.section_node_base_id.size();
        if (num_sections != st.layout.section_nseg.size()) {
            throw std::runtime_error("section layout size mismatch for template: " + tpl.name);
        }
        st.section_label_layout = neurong_morph::build_section_label_segment_layout(
            tpl.sections_by_label,
            st.layout.section_nseg,
            "template '" + tpl.name + "'");

        auto normalize_ion_name = [&](std::string_view name) -> std::string {
            std::string out = to_lower_ascii(name);
            if (out.ends_with("_ion")) {
                return out;
            }
            return out + "_ion";
        };

        auto label_index_for = [&](const std::string& label) -> std::size_t {
            auto it = tpl.label_index.find(label);
            if (it == tpl.label_index.end()) {
                throw std::runtime_error("unknown label name: " + label);
            }
            return it->second;
        };

        auto set_use_label = [&](std::vector<std::uint8_t>& use_label, const std::string& label) {
            if (label == neurong_biophysical::AllLabels) {
                for (auto& v : use_label) {
                    v = static_cast<std::uint8_t>(1);
                }
                return;
            }
            const auto label_u = label_index_for(label);
            if (label_u < use_label.size()) {
                use_label[label_u] = static_cast<std::uint8_t>(1);
            }
        };

        for (const auto& op : st.prepare.inserts()) {
            std::optional<std::string> ion_raw{};
            std::optional<std::string> erev_type_raw{};
            std::optional<ParamValue> erev_value{};
            std::optional<ParamValue> ion_conci_value{};
            std::optional<ParamValue> ion_conco_value{};
            std::optional<double> ion_charge_value{};

            for (const auto& [k, v] : op.params) {
                const std::string& key = k;
                if (key == "ion") {
                    if (!std::holds_alternative<std::string>(v)) {
                        throw std::runtime_error("ion param must be a string");
                    }
                    ion_raw = std::get<std::string>(v);
                } else if (key == "erev") {
                    erev_value = v;
                } else if (key == "erev_type") {
                    if (!std::holds_alternative<std::string>(v)) {
                        throw std::runtime_error("erev_type must be a string");
                    }
                    erev_type_raw = std::get<std::string>(v);
                } else if (key == "ion_conci") {
                    ion_conci_value = v;
                } else if (key == "ion_conco") {
                    ion_conco_value = v;
                } else if (key == "ion_charge") {
                    if (!std::holds_alternative<double>(v)) {
                        throw std::runtime_error("ion_charge must be a scalar");
                    }
                    ion_charge_value = std::get<double>(v);
                }
            }

            if (!ion_raw.has_value()) {
                if (erev_value.has_value() || erev_type_raw.has_value() || ion_conci_value.has_value() ||
                    ion_conco_value.has_value() || ion_charge_value.has_value()) {
                    throw std::runtime_error("ion metadata requires ion to be specified");
                }
                throw std::runtime_error("mechanism " + op.mech +
                                         " requires explicit ion (use ion=\"non_specific\" for nonspecific currents)");
            }

            if (is_non_specific_ion(*ion_raw)) {
                if (erev_value.has_value() || erev_type_raw.has_value() || ion_conci_value.has_value() ||
                    ion_conco_value.has_value() || ion_charge_value.has_value()) {
                    throw std::runtime_error("non_specific ion does not accept erev/erev_type/ion_conci/ion_conco/ion_charge");
                }
                continue;
            }

            const std::string ion_name = normalize_ion_name(*ion_raw);
            IonSpec spec = parse_ion_spec(ion_name, erev_type_raw, erev_value, ion_conci_value,
                                          ion_conco_value);

            auto& overrides = st.ion_overrides[ion_name];
            if (auto existing = overrides.get(op.label)) {
                if (!ion_spec_equal(*existing, spec)) {
                    throw std::runtime_error("conflicting ion erev spec for ion " + ion_name);
                }
            } else {
                overrides.set(op.label, spec);
            }

            auto& use_label = st.ion_use_label[ion_name];
            if (use_label.empty()) {
                use_label.assign(tpl.sections_by_label.size(), static_cast<std::uint8_t>(0));
            }
            set_use_label(use_label, op.label);

            const std::string ion_base = ion_name.substr(0, ion_name.size() - 4);
            const bool is_standard = (ion_base == "na" || ion_base == "k" || ion_base == "ca");
            if (!is_standard) {
                auto& mech_override = mech_ion_overrides[op.mech];
                if (!mech_override.empty()) {
                    auto it = mech_override.find("*");
                    if (it == mech_override.end() || it->second != ion_name) {
                        throw std::runtime_error("mechanism " + op.mech + " uses inconsistent ion overrides");
                    }
                } else {
                    mech_override.emplace("*", ion_name);
                }
            }

            if (ion_charge_value.has_value()) {
                auto& meta = ion_meta_overrides[ion_name];
                if (meta.has_charge && meta.charge != *ion_charge_value) {
                    throw std::runtime_error("conflicting ion_charge for ion " + ion_name);
                }
                meta.has_charge = true;
                meta.charge = *ion_charge_value;
            }

            if (spec.erev_type == ErevType::Nernst) {
                ion_needs_charge[ion_name] = true;
            }
        }

        for (const auto& op : st.prepare.ion_species()) {
            std::optional<std::string> erev_type_raw{};
            std::optional<ParamValue> erev_value{};
            std::optional<ParamValue> ion_conci_value{};
            std::optional<ParamValue> ion_conco_value{};
            std::optional<double> ion_charge_value{};

            for (const auto& [k, v] : op.params) {
                const std::string& key = k;
                if (key == "ion") {
                    throw std::runtime_error("ion_species does not accept ion param (use ion argument)");
                } else if (key == "erev") {
                    erev_value = v;
                } else if (key == "erev_type") {
                    if (!std::holds_alternative<std::string>(v)) {
                        throw std::runtime_error("erev_type must be a string");
                    }
                    erev_type_raw = std::get<std::string>(v);
                } else if (key == "ion_conci") {
                    ion_conci_value = v;
                } else if (key == "ion_conco") {
                    ion_conco_value = v;
                } else if (key == "ion_charge") {
                    if (!std::holds_alternative<double>(v)) {
                        throw std::runtime_error("ion_charge must be a scalar");
                    }
                    ion_charge_value = std::get<double>(v);
                }
            }

            if (is_non_specific_ion(op.ion)) {
                throw std::runtime_error("ion_species does not support non_specific ion");
            }
            const std::string ion_name = normalize_ion_name(op.ion);
            IonSpecPatch patch = parse_ion_patch(ion_name,
                                                 erev_type_raw,
                                                 erev_value,
                                                 ion_conci_value,
                                                 ion_conco_value);

            auto& overrides = st.ion_overrides[ion_name];
            if (auto existing = overrides.get(op.label)) {
                IonSpec merged = *existing;
                merge_ion_patch(merged, patch, ion_name);
                overrides.set(op.label, merged);
            } else {
                IonSpec spec{};
                if (patch.erev_type.has_value()) {
                    spec.erev_type = *patch.erev_type;
                }
                if (patch.erev.has_value()) {
                    spec.erev = *patch.erev;
                }
                if (patch.conci.has_value()) {
                    spec.conci = *patch.conci;
                }
                if (patch.conco.has_value()) {
                    spec.conco = *patch.conco;
                }
                overrides.set(op.label, spec);
            }

            auto& use_label = st.ion_use_label[ion_name];
            if (use_label.empty()) {
                use_label.assign(tpl.sections_by_label.size(), static_cast<std::uint8_t>(0));
            }
            set_use_label(use_label, op.label);

            const std::string ion_base = ion_name.substr(0, ion_name.size() - 4);
            const bool is_standard = (ion_base == "na" || ion_base == "k" || ion_base == "ca");
            if (!is_standard) {
                auto& mech_override = mech_ion_overrides[op.model];
                if (!mech_override.empty()) {
                    auto it = mech_override.find("*");
                    if (it == mech_override.end() || it->second != ion_name) {
                        throw std::runtime_error("mechanism " + op.model + " uses inconsistent ion overrides");
                    }
                } else {
                    mech_override.emplace("*", ion_name);
                }
            }

            if (ion_charge_value.has_value()) {
                auto& meta = ion_meta_overrides[ion_name];
                if (meta.has_charge && meta.charge != *ion_charge_value) {
                    throw std::runtime_error("conflicting ion_charge for ion " + ion_name);
                }
                meta.has_charge = true;
                meta.charge = *ion_charge_value;
            }

            if (patch.erev_type.has_value() && *patch.erev_type == ErevType::Nernst) {
                ion_needs_charge[ion_name] = true;
            }

            ion_species_mechs.insert(op.model);
        }

        Prepare build_prepare{};
        for (const auto& op : st.prepare.inserts()) {
            build_prepare.insert(op.label, op.mech, op.params);
        }
        for (const auto& op : st.prepare.ion_species()) {
            build_prepare.insert(op.label, op.model, op.params);
        }
        st.built = build_mechanism_instances(
            tpl.label_index,
            tpl.sections_by_label,
            tpl.section_name_to_id,
            st.layout,
            build_prepare,
            st.section_label_layout);
        parsed[t] = std::move(st);
    }

    // --------------------------------------------------------------------
    // Global vars (used by Mechanism ctor + global var reads during bbcore-like init)
    // --------------------------------------------------------------------
    coreneuron::global_var_map.clear();
    coreneuron::secondorder = 0;

    // Caller-provided overrides.
    set_global_scalar("celsius", celsius);
    set_global_scalar("dt", sim.dt);

    const auto node_permute = std::span<const int>(group.permute, static_cast<std::size_t>(group.len));

    auto apply_section_property_for_cell = [&](const ParsedTemplate& st,
                                               const std::vector<SectionPropertyOp>& property_ops,
                                               std::string_view property_name,
                                               std::size_t global_cell_id,
                                               std::vector<double>& section_values) {
        const auto& tpl = *st.tpl;
        if (section_values.size() != tpl.section_name_to_id.size()) {
            throw std::runtime_error("section value size mismatch for template: " + tpl.name);
        }

        for (std::size_t op_idx = 0; op_idx < property_ops.size(); ++op_idx) {
            const auto& op = property_ops[op_idx];
            const bool select_all = (op.label == AllLabels);
            std::optional<std::size_t> selected_label_u{};
            if (!select_all) {
                auto it = tpl.label_index.find(op.label);
                if (it == tpl.label_index.end()) {
                    throw std::runtime_error("unknown label name: " + op.label);
                }
                selected_label_u = it->second;
            }

            auto apply_value_to_label = [&](std::size_t label_u, double value) {
                if (label_u >= tpl.sections_by_label.size()) {
                    return;
                }
                for (const auto sec_id : tpl.sections_by_label[label_u]) {
                    const auto sec_index = static_cast<std::size_t>(sec_id);
                    section_values[sec_index] = value;
                }
            };

            auto is_label_selected = [&](std::size_t label_u) {
                return select_all || (selected_label_u.has_value() && *selected_label_u == label_u);
            };

            const auto& value = op.value;

            if (std::holds_alternative<double>(value)) {
                const double scalar = std::get<double>(value);
                if (select_all) {
                    for (std::size_t label_u = 0; label_u < tpl.sections_by_label.size(); ++label_u) {
                        apply_value_to_label(label_u, scalar);
                    }
                } else {
                    apply_value_to_label(*selected_label_u, scalar);
                }
                continue;
            }

            if (std::holds_alternative<RandomParamSpec>(value)) {
                const auto& rule = std::get<RandomParamSpec>(value);
                const std::uint64_t base_seed = combine_seed_many(
                    st.random_seed,
                    hash_string64("section"),
                    hash_string64(property_name),
                    op_idx);

                for (std::size_t label_u = 0; label_u < tpl.sections_by_label.size(); ++label_u) {
                    if (!is_label_selected(label_u)) {
                        continue;
                    }
                    for (const auto sec_id : tpl.sections_by_label[label_u]) {
                        const auto sec_index = static_cast<std::size_t>(sec_id);
                        const double sampled = sample_random_with_context(
                            rule,
                            base_seed,
                            global_cell_id,
                            label_u,
                            sec_index,
                            /*segment=*/0);
                        section_values[sec_index] = sampled;
                    }
                }
                continue;
            }

            const auto& map = std::get<SectionParamMap>(value);
            for (const auto& [sec_name, seq] : map) {
                const auto it = tpl.section_name_to_id.find(sec_name);
                if (it == tpl.section_name_to_id.end()) {
                    throw std::runtime_error("unknown section name in section map: " + sec_name);
                }
                const auto sec_id = it->second;
                const auto sec_index = static_cast<std::size_t>(sec_id);
                if (sec_index >= st.section_label_layout.section_label_u_by_sec.size()) {
                    throw std::runtime_error("section index out of range in section map: " + sec_name);
                }
                const auto label_u_i32 = st.section_label_layout.section_label_u_by_sec[sec_index];
                if (label_u_i32 < 0) {
                    throw std::runtime_error("section has no label: " + sec_name);
                }
                const auto label_u = static_cast<std::size_t>(label_u_i32);
                if (!is_label_selected(label_u)) {
                    throw std::runtime_error("section " + sec_name + " not in selected label");
                }

                if (std::holds_alternative<std::vector<double>>(seq)) {
                    const auto& seq_values = std::get<std::vector<double>>(seq);
                    if (seq_values.empty()) {
                        throw std::runtime_error("section map value list cannot be empty: " + sec_name);
                    }
                    const double scalar = seq_values.front();
                    for (double x : seq_values) {
                        if (x != scalar) {
                            throw std::runtime_error(
                                "section scalar param map expects per-section scalar values");
                        }
                    }
                    section_values[sec_index] = scalar;
                    continue;
                }

                const auto& rule = std::get<RandomParamSpec>(seq);
                const std::uint64_t base_seed = combine_seed_many(
                    st.random_seed,
                    hash_string64("section"),
                    hash_string64(property_name),
                    op_idx);
                section_values[sec_index] = sample_random_with_context(
                    rule,
                    base_seed,
                    global_cell_id,
                    label_u,
                    sec_index,
                    /*segment=*/0);
            }
        }
    };

    // --------------------------------------------------------------------
    // Axial coupling (Ra)
    //
    // Morph stage builds an Ra-free base (a/b does not include Ra yet). Apply Ra here by
    // scaling vecdata_a/vecdata_b once using per-section Ra values.
    // --------------------------------------------------------------------
    {

        auto* a_cpu = group.vecdata_a->get_cpu_data();
        auto* b_cpu = group.vecdata_b->get_cpu_data();

        for (const auto& st : parsed) {
            const auto& tpl = *st.tpl;

            if (st.section_ra.size() != st.layout.section_node_base_id.size() ||
                st.section_ra.size() != st.layout.section_nseg.size()) {
                throw std::runtime_error("section ra size mismatch for template: " + tpl.name);
            }

            for (int c = 0; c < tpl.num_cells; ++c) {
                std::vector<double> section_ra_cell = st.section_ra;
                const int cell_id = tpl.cell_base + c;
                apply_section_property_for_cell(st,
                                                st.prepare.resistivity(),
                                                "ra",
                                                static_cast<std::size_t>(cell_id),
                                                section_ra_cell);

                std::vector<double> inv_ra_tpl(tpl.tpl_nodes_per_cell, 1.0);
                for (std::size_t sec_index = 0; sec_index < section_ra_cell.size(); ++sec_index) {
                    const double ra_value = section_ra_cell[sec_index];
                    if (ra_value <= 0.0) {
                        throw std::runtime_error("section ra must be positive for template: " + tpl.name);
                    }
                    const double inv_ra = 1.0 / ra_value;
                    const auto base_i32 = st.layout.section_node_base_id[sec_index];
                    const auto nseg_i32 = st.layout.section_nseg[sec_index];
                    const std::size_t base = static_cast<std::size_t>(base_i32);
                    const std::size_t nseg = static_cast<std::size_t>(nseg_i32);
                    for (std::size_t j = 0; j <= nseg; ++j) {
                        inv_ra_tpl[base + j] = inv_ra;
                    }
                }

                // Apply scaling across the population. vecdata_a/b are permuted; map each original node through node_permute.
                for (std::size_t tnode = 0; tnode < tpl.tpl_nodes_per_cell; ++tnode) {
                    const std::int32_t original_i32 =
                        map_template_node_to_original(cell_id, static_cast<std::int32_t>(tnode));
                    if (original_i32 < 0) {
                        continue;
                    }
                    const auto original = static_cast<std::size_t>(original_i32);
                    const int perm_i = node_permute[original];
                    const auto perm = static_cast<std::size_t>(perm_i);
                    const double f = inv_ra_tpl[tnode];
                    a_cpu[perm] = static_cast<double>(a_cpu[perm]) * f;
                    b_cpu[perm] = static_cast<double>(b_cpu[perm]) * f;
                }
            }
        }

        group.vecdata_a->update_gpu_data_from_cpu();
        group.vecdata_b->update_gpu_data_from_cpu();
    }

    // --------------------------------------------------------------------
    // Ion table is global process state; clear any stale reverse-index maps from previous runs.
    // Otherwise, ion-variable mapping may accidentally reuse an old node->ion index.
    std::unordered_set<std::string> ion_names;
    for (const auto& st : parsed) {
        for (const auto& [ion_name, _] : st.ion_overrides) {
            ion_names.insert(ion_name);
        }
    }

    for (const auto& ion_name : ion_names) {
        auto& ion = get_ion_vars(ion_name, /*must_exist=*/false);
        ion.idx_reverse_table.clear();
        ion.nnodes = 0;
        ion.cpu_data = nullptr;
        if (ion.gpu_data_oh_host) {
            delete[] ion.gpu_data_oh_host;
            ion.gpu_data_oh_host = nullptr;
        }
    }

    constexpr int kEionDataSize = 5;
    constexpr int kCapDataSize = 1;

    // --------------------------------------------------------------------
    // Template section defaults (no implicit defaults) are parsed per-template.

    // --------------------------------------------------------------------
    // Ions first (required for ion-variable mapping during mech reg_node_indices).
    //
    // Important: instantiate ions only where required by inserted mechanisms,
    // rather than across the whole morphology. This matches NEURON/Arbor semantics
    // and avoids per-step ion overhead on segments that never reference the ion.
    // --------------------------------------------------------------------
    constexpr int kIonTypeFixed = 0;
    constexpr int kIonTypeNernst = 040 | 0100;

    auto build_ion_meta = [&](const std::string& ion_name) -> IonMeta {
        IonMeta meta{};
        bool has_charge = false;
        auto it_meta = ion_meta_overrides.find(ion_name);
        if (it_meta != ion_meta_overrides.end() && it_meta->second.has_charge) {
            meta.charge = it_meta->second.charge;
            has_charge = true;
        }
        if (ion_needs_charge[ion_name] && !has_charge) {
            throw std::runtime_error("ion " + ion_name + " requires ion_charge for nernst");
        }
        return meta;
    };

    auto build_and_create_ion = [&](const std::string& ion_name) {
        const IonMeta meta = build_ion_meta(ion_name);
        register_ion_meta(ion_name, meta);
        std::vector<std::int32_t> node_pop;
        std::vector<double> erev_pop;
        std::vector<double> conci_pop;
        std::vector<double> conco_pop;
        std::vector<int> iontype_pop;

        for (const auto& st : parsed) {
            const auto& tpl = *st.tpl;
            if (tpl.sections_by_label.empty()) {
                continue;
            }
            const auto it_overrides = st.ion_overrides.find(ion_name);
            const auto it_use = st.ion_use_label.find(ion_name);
            if (it_overrides == st.ion_overrides.end() || it_use == st.ion_use_label.end()) {
                continue;
            }
            const auto& overrides = it_overrides->second;
            const auto& use_label = it_use->second;
            if (use_label.empty()) {
                continue;
            }

            auto get_spec = [&](std::size_t label_u) -> IonSpec {
                if (label_u >= tpl.label_names.size()) {
                    throw std::runtime_error("label index out of range for ion " + ion_name);
                }
                const auto& label_name = tpl.label_names[label_u];
                auto spec_opt = overrides.get(label_name);
                if (!spec_opt.has_value()) {
                    throw std::runtime_error("missing ion spec for ion " + ion_name);
                }
                return *spec_opt;
            };

            auto sample_ion_param = [&](const std::optional<ScalarOrRandom>& value,
                                        std::string_view field_name,
                                        bool required,
                                        std::size_t global_cell_id,
                                        std::size_t label_u,
                                        std::size_t section_index,
                                        std::size_t segment_index) -> double {
                if (!value.has_value()) {
                    if (required) {
                        throw std::runtime_error("ion " + ion_name + " requires " +
                                                 std::string(field_name) + " for nernst");
                    }
                    return 0.0;
                }
                const std::uint64_t seed = combine_seed_many(
                    st.random_seed,
                    hash_string64(ion_name),
                    hash_string64(field_name),
                    label_u);
                return sample_scalar_or_random_with_context(
                    *value, seed, global_cell_id, label_u, section_index, segment_index);
            };

            for (int c = 0; c < tpl.num_cells; ++c) {
                const int cell_id = tpl.cell_base + c;
                const std::size_t global_cell_id = static_cast<std::size_t>(cell_id);
                for (std::size_t label_u = 0; label_u < use_label.size(); ++label_u) {
                    if (!use_label[label_u]) {
                        continue;
                    }
                    const auto spec = get_spec(label_u);
                    const int ion_type = (spec.erev_type == ErevType::Nernst) ? kIonTypeNernst : kIonTypeFixed;
                    for (const auto sec_id : tpl.sections_by_label[label_u]) {
                        const auto sec_index = static_cast<std::size_t>(sec_id);
                        const auto base = st.layout.section_node_base_id[sec_index];
                        const auto nseg = st.layout.section_nseg[sec_index];
                        for (std::int32_t j = 0; j < nseg; ++j) {
                            const auto tpl_node = base + j;
                            const std::int32_t original = map_template_node_to_original(cell_id, tpl_node);
                            const int perm = node_permute[static_cast<std::size_t>(original)];
                            node_pop.push_back(static_cast<std::int32_t>(perm));
                            if (spec.erev_type == ErevType::Fixed) {
                                erev_pop.push_back(sample_ion_param(
                                    spec.erev,
                                    "erev",
                                    /*required=*/true,
                                    global_cell_id,
                                    label_u,
                                    sec_index,
                                    static_cast<std::size_t>(j)));
                            } else {
                                erev_pop.push_back(0.0);
                            }
                            conci_pop.push_back(sample_ion_param(
                                spec.conci,
                                "ion_conci",
                                /*required=*/spec.erev_type == ErevType::Nernst,
                                global_cell_id,
                                label_u,
                                sec_index,
                                static_cast<std::size_t>(j)));
                            conco_pop.push_back(sample_ion_param(
                                spec.conco,
                                "ion_conco",
                                /*required=*/spec.erev_type == ErevType::Nernst,
                                global_cell_id,
                                label_u,
                                sec_index,
                                static_cast<std::size_t>(j)));
                            iontype_pop.push_back(ion_type);
                        }
                    }
                }
            }
        }

        if (node_pop.empty()) {
            return;
        }

        const auto inst_permute = stable_sort_node_indices_inplace(node_pop);
        permute_blocks_inplace(erev_pop, inst_permute, /*block_size=*/1);
        permute_blocks_inplace(conci_pop, inst_permute, /*block_size=*/1);
        permute_blocks_inplace(conco_pop, inst_permute, /*block_size=*/1);
        permute_blocks_inplace(iontype_pop, inst_permute, /*block_size=*/1);

        std::vector<double> data;
        data.assign(node_pop.size() * kEionDataSize, 0.0);
        for (std::size_t i = 0; i < node_pop.size(); ++i) {
            const std::size_t base = i * kEionDataSize;
            data[base + 0] = erev_pop[i];
            data[base + 1] = conci_pop[i];
            data[base + 2] = conco_pop[i];
            data[base + 3] = 0.0;  // cur
            data[base + 4] = 0.0;  // dcurdv
        }

        neurong_mech::create_mech(sim,
                                  group,
                                  next_type,
                                  ion_name,
                                  node_pop,
                                  kEionDataSize,
                                  data,
                                  /*pdata_size=*/1,
                                  iontype_pop,
                                  nullptr,
                                  neurong_mech::MechRole::Eion);
    };

    for (const auto& ion_name : ion_names) {
        build_and_create_ion(ion_name);
    }

    // --------------------------------------------------------------------
    // Capacitance (cm per segment)
    // --------------------------------------------------------------------
    {
        std::vector<std::int32_t> cap_nodes;
        std::vector<double> cm_values;

        for (const auto& st : parsed) {
            const auto& tpl = *st.tpl;
            if (st.section_cm.size() != st.layout.section_node_base_id.size() ||
                st.section_cm.size() != st.layout.section_nseg.size()) {
                throw std::runtime_error("section cm size mismatch for template: " + tpl.name);
            }
            for (int c = 0; c < tpl.num_cells; ++c) {
                std::vector<double> section_cm_cell = st.section_cm;
                const int cell_id = tpl.cell_base + c;
                apply_section_property_for_cell(st,
                                                st.prepare.specific_capacitance(),
                                                "cm",
                                                static_cast<std::size_t>(cell_id),
                                                section_cm_cell);
                for (std::size_t sec_index = 0; sec_index < section_cm_cell.size(); ++sec_index) {
                    const double cm_value = section_cm_cell[sec_index];
                    if (cm_value <= 0.0) {
                        throw std::runtime_error("section cm must be positive for template: " + tpl.name);
                    }
                    const auto base = st.layout.section_node_base_id[sec_index];
                    const auto nseg = st.layout.section_nseg[sec_index];
                    for (std::int32_t j = 0; j < nseg; ++j) {
                        const auto tpl_node = base + j;
                        const std::int32_t original = map_template_node_to_original(cell_id, tpl_node);
                        const int perm = node_permute[static_cast<std::size_t>(original)];
                        cap_nodes.push_back(static_cast<std::int32_t>(perm));
                        cm_values.push_back(cm_value);
                    }
                }
            }
        }


        const auto cap_inst_permute = stable_sort_node_indices_inplace(cap_nodes);
        permute_blocks_inplace(cm_values, cap_inst_permute, /*block_size=*/1);

        std::vector<double> cap_data;
        cap_data.assign(cap_nodes.size() * kCapDataSize, 0.0);
        for (std::size_t i = 0; i < cap_nodes.size(); ++i) {
            cap_data[i * kCapDataSize + 0] = cm_values[i];
        }
        neurong_mech::create_mech(sim, group, next_type, "capacitance", cap_nodes, kCapDataSize, cap_data,
                                  /*pdata_size=*/0, {}, nullptr);
    }

    // --------------------------------------------------------------------
    // Distributed mechanisms from the prepare ops.
    // --------------------------------------------------------------------
    auto set_global_var_value = [&](const std::string& mech_name, std::string_view var_name, double value) -> bool {
        VarDescriptor d{mech_name, std::string(var_name), /*idx=*/0, /*array_index=*/0};
        auto [cpu_ptr, gpu_ptr] = sim.getVarPtr(d, /*will_panic=*/false);
        const bool found = (cpu_ptr != nullptr) || (gpu_ptr != nullptr);
        if (!found) {
            return false;
        }
        const double v = value;
        if (cpu_ptr) {
            *cpu_ptr = v;
        }
        if (sim.mode == GPU && gpu_ptr) {
            mem_copy_cpu2gpu_sync(gpu_ptr, &v, sizeof(double));
        }
        return true;
    };

    // Unique mechanism names for this build, preserving first-seen order.
    std::vector<std::string> mech_names;
    std::unordered_set<std::string_view> seen_mech;
    for (const auto& st : parsed) {
        for (const auto& tpl_mech : st.built.mechanisms) {
            if (seen_mech.insert(tpl_mech.mech).second) {
                mech_names.push_back(tpl_mech.mech);
            }
        }
    }
    auto find_tpl_mech = [&](const BuildResult& built_res, std::string_view name) -> const MechanismInstanceLayout* {
        for (const auto& m : built_res.mechanisms) {
            if (m.mech == name) {
                return &m;
            }
        }
        return nullptr;
    };


    for (const auto& mech_name : mech_names) {
        std::vector<std::int32_t> node_pop;
        std::vector<std::optional<std::size_t>> tpl_base;
        std::vector<const MechanismInstanceLayout*> tpl_mech;
        tpl_base.resize(parsed.size());
        tpl_mech.resize(parsed.size(), nullptr);

        for (std::size_t t = 0; t < parsed.size(); ++t) {
            const auto& st = parsed[t];
            const auto* m = find_tpl_mech(st.built, mech_name);
            if (!m || m->instance_count == 0) {
                continue;
            }
            tpl_base[t] = node_pop.size();
            tpl_mech[t] = m;

            const auto& tpl = *st.tpl;
            for (int c = 0; c < tpl.num_cells; ++c) {
                const int cell_id = tpl.cell_base + c;
                for (const auto tpl_node : m->node_indices) {
                    const std::int32_t original = map_template_node_to_original(cell_id, tpl_node);
                    const int perm = node_permute[static_cast<std::size_t>(original)];
                    node_pop.push_back(static_cast<std::int32_t>(perm));
                }
            }
        }

        if (node_pop.empty()) {
            continue;
        }

        const auto inst_permute = stable_sort_node_indices_inplace(node_pop);
        const std::size_t total_instances = node_pop.size();

        std::vector<double> data;
        const std::unordered_map<std::string, std::string>* ion_override_ptr = nullptr;
        auto override_it = mech_ion_overrides.find(mech_name);
        if (override_it != mech_ion_overrides.end() && !override_it->second.empty()) {
            ion_override_ptr = &override_it->second;
        }

        const bool is_ion_species = (ion_species_mechs.find(mech_name) != ion_species_mechs.end());
        Mechanism* mech = neurong_mech::create_mech(sim,
                                                    group,
                                                    next_type,
                                                    mech_name,
                                                    node_pop,
                                                    /*data_size=*/0,
                                                    data,
                                                    /*pdata_size=*/0,
                                                    {},
                                                    ion_override_ptr,
                                                    is_ion_species ? neurong_mech::MechRole::IonSpecies
                                                                   : neurong_mech::MechRole::Normal);

        auto is_non_range_metadata_key = [](const std::string& key) -> bool {
            return key == "ion" || key == "erev" || key == "erev_type" ||
                   key == "ion_conci" || key == "ion_conco" || key == "ion_charge" ||
                   key == "instances";
        };

        bool want_overrides = false;
        for (const auto& st : parsed) {
            for (const auto& op : st.prepare.inserts()) {
                if (op.mech != mech_name) {
                    continue;
                }
                bool has_range_param = false;
                for (const auto& [k, v] : op.params) {
                    const std::string& key = k;
                    if (is_non_range_metadata_key(key)) {
                        continue;
                    }
                    if (std::holds_alternative<std::string>(v)) {
                        continue;
                    }
                    has_range_param = true;
                    break;
                }
                if (has_range_param) {
                    want_overrides = true;
                    break;
                }
            }
            if (!want_overrides) {
                for (const auto& op : st.prepare.ion_species()) {
                    if (op.model != mech_name) {
                        continue;
                    }
                    bool has_range_param = false;
                    for (const auto& [k, v] : op.params) {
                        const std::string& key = k;
                        if (is_non_range_metadata_key(key)) {
                            continue;
                        }
                        if (std::holds_alternative<std::string>(v)) {
                            continue;
                        }
                        has_range_param = true;
                        break;
                    }
                    if (has_range_param) {
                        want_overrides = true;
                        break;
                    }
                }
            }
            if (want_overrides) {
                break;
            }
        }
        if (!want_overrides) {
            continue;
        }

        auto it_table = mech_var_table.find(mech->type);
        if (it_table == mech_var_table.end()) {
            throw std::runtime_error(
                "mechanism '" + mech_name +
                "' does not expose in-memory range-parameter override table; "
                "remove range params from inserts for now");
        }

        std::unordered_map<std::string, int> var_to_coreidx;
        const std::string prefix = mech_name + "_";
        for (const auto& [core_idx, varData] : it_table->second) {
            const std::string& full = varData.name;
            if (!full.starts_with(prefix)) {
                continue;
            }
            var_to_coreidx.emplace(full.substr(prefix.size()), core_idx);
        }

        auto resolve_coreidx = [&](std::string_view name) -> std::optional<int> {
            // Strict input contract: require exact variable names as defined by the mechanism
            // implementation (VarNames/STATE/RANGE symbols in the corresponding .cu file).
            // No compatibility renaming (e.g. "g" -> "g_pas", "gbar" -> "gbar_<mech>").
            auto it = var_to_coreidx.find(std::string(name));
            if (it != var_to_coreidx.end()) {
                return it->second;
            }
            return std::nullopt;
        };

        std::unordered_set<int> touched_coreidx;
        bool any_update = false;

        auto apply_range_var = [&](std::size_t template_id,
                                   MechVarData& varData,
                                   const std::string& sel_label,
                                   std::string_view param_name,
                                   const ParamValue& value) {
            const auto& st = parsed[template_id];
            const auto& tpl = *st.tpl;
            const auto* tm = tpl_mech[template_id];
            const auto base_opt = tpl_base[template_id];
            if (tm == nullptr || !base_opt.has_value()) {
                throw std::runtime_error("internal error: missing mechanism layout base for range override");
            }
            const std::size_t base_global = *base_opt;
            const std::size_t tpl_n = tm->instance_count;

            const bool select_all = (sel_label == neurong_biophysical::AllLabels);
            std::optional<std::size_t> sel_label_u{};
            if (!select_all) {
                auto it = tpl.label_index.find(sel_label);
                if (it == tpl.label_index.end()) {
                    throw std::runtime_error("unknown label name: " + sel_label);
                }
                sel_label_u = it->second;
            }

            const std::uint64_t param_seed = combine_seed_many(
                st.random_seed,
                hash_string64(mech_name),
                hash_string64(param_name));

            auto segment_instance_range = [&](std::size_t label_u,
                                              std::size_t sec_index,
                                              std::size_t seg_local_index)
                                              -> std::pair<std::size_t, std::size_t> {
                if (label_u >= tm->label_offset.size() ||
                    label_u >= tm->label_count.size() ||
                    (label_u + 1) >= tm->label_segment_base.size()) {
                    throw std::runtime_error("label metadata out of range for mechanism " + mech_name);
                }
                if (sec_index >= st.section_label_layout.section_offset_in_label_by_sec.size()) {
                    throw std::runtime_error("section offset metadata out of range");
                }
                const std::size_t section_seg_base =
                    st.section_label_layout.section_offset_in_label_by_sec[sec_index];
                const std::size_t seg_flat_begin = tm->label_segment_base[label_u];
                const std::size_t seg_flat_end = tm->label_segment_base[label_u + 1];
                const std::size_t seg_flat = seg_flat_begin + section_seg_base + seg_local_index;
                if (seg_flat >= seg_flat_end ||
                    seg_flat >= tm->label_segment_offset.size() ||
                    seg_flat >= tm->label_segment_count.size()) {
                    throw std::runtime_error("segment metadata out of range for mechanism " + mech_name);
                }

                const std::size_t local_offset = tm->label_segment_offset[seg_flat];
                const std::size_t local_count = tm->label_segment_count[seg_flat];
                if (local_offset + local_count > tm->label_count[label_u]) {
                    throw std::runtime_error("segment instance range exceeds label instance span");
                }

                const std::size_t begin_tpl = tm->label_offset[label_u] + local_offset;
                return {begin_tpl, local_count};
            };

            if (std::holds_alternative<SectionParamMap>(value)) {
                const auto& section_values = std::get<SectionParamMap>(value);
                for (const auto& [sec_name, section_value] : section_values) {
                    const auto it = tpl.section_name_to_id.find(sec_name);
                    if (it == tpl.section_name_to_id.end()) {
                        throw std::runtime_error("unknown section name in param map: " + sec_name);
                    }
                    const auto sec_id = it->second;
                    const auto sec_index = static_cast<std::size_t>(sec_id);
                    if (sec_index >= st.section_label_layout.section_label_u_by_sec.size() ||
                        sec_index >= st.section_label_layout.section_offset_in_label_by_sec.size()) {
                        throw std::runtime_error("section index out of range in param map: " + sec_name);
                    }
                    const auto label_u_i32 = st.section_label_layout.section_label_u_by_sec[sec_index];
                    if (label_u_i32 < 0) {
                        throw std::runtime_error("section has no label: " + sec_name);
                    }
                    const std::size_t label_u = static_cast<std::size_t>(label_u_i32);
                    if (!select_all && (!sel_label_u.has_value() || label_u != *sel_label_u)) {
                        throw std::runtime_error("section " + sec_name + " not in selected label");
                    }
                    if (label_u >= tm->inserted_by_label.size() || !tm->inserted_by_label[label_u]) {
                        throw std::runtime_error("mechanism not inserted on section " + sec_name);
                    }
                    const auto nseg_i32 = st.layout.section_nseg[sec_index];
                    if (nseg_i32 < 0) {
                        throw std::runtime_error("negative nseg in section " + sec_name);
                    }
                    const auto nseg = static_cast<std::size_t>(nseg_i32);

                    if (std::holds_alternative<std::vector<double>>(section_value)) {
                        const auto& values = std::get<std::vector<double>>(section_value);
                        if (values.size() != nseg) {
                            throw std::runtime_error("param map size mismatch for section " + sec_name);
                        }
                        for (int c = 0; c < tpl.num_cells; ++c) {
                            for (std::size_t j = 0; j < values.size(); ++j) {
                                const auto [begin_tpl, count_tpl] = segment_instance_range(label_u, sec_index, j);
                                for (std::size_t rep = 0; rep < count_tpl; ++rep) {
                                    const std::size_t inst =
                                        base_global + static_cast<std::size_t>(c) * tpl_n + begin_tpl + rep;
                                    const auto inst_sorted = static_cast<std::size_t>(inst_permute[inst]);
                                    varData.cpu_data[inst_sorted] = static_cast<double>(values[j]);
                                }
                            }
                        }
                        continue;
                    }

                    const auto& rule = std::get<RandomParamSpec>(section_value);
                    for (int c = 0; c < tpl.num_cells; ++c) {
                        const int global_cell_id = tpl.cell_base + c;
                        for (std::size_t j = 0; j < nseg; ++j) {
                            const double sampled = sample_random_with_context(
                                rule,
                                param_seed,
                                static_cast<std::size_t>(global_cell_id),
                                label_u,
                                sec_index,
                                j);
                            const auto [begin_tpl, count_tpl] = segment_instance_range(label_u, sec_index, j);
                            for (std::size_t rep = 0; rep < count_tpl; ++rep) {
                                const std::size_t inst =
                                    base_global + static_cast<std::size_t>(c) * tpl_n + begin_tpl + rep;
                                const auto inst_sorted = static_cast<std::size_t>(inst_permute[inst]);
                                varData.cpu_data[inst_sorted] = sampled;
                            }
                        }
                    }
                }
                return;
            }

            ScalarOrRandom scalar_or_random{};
            if (std::holds_alternative<double>(value)) {
                scalar_or_random = std::get<double>(value);
            } else if (std::holds_alternative<RandomParamSpec>(value)) {
                scalar_or_random = std::get<RandomParamSpec>(value);
            } else {
                throw std::runtime_error("mechanism param must be scalar, random spec, or section map");
            }

            for (std::size_t label_u = 0; label_u < tm->inserted_by_label.size(); ++label_u) {
                if (!tm->inserted_by_label[label_u]) {
                    continue;
                }
                if (!select_all && (!sel_label_u.has_value() || label_u != *sel_label_u)) {
                    continue;
                }
                const auto& sections = tpl.sections_by_label[label_u];
                for (int c = 0; c < tpl.num_cells; ++c) {
                    const int global_cell_id = tpl.cell_base + c;
                    for (const auto sec_id : sections) {
                        const auto sec_index = static_cast<std::size_t>(sec_id);
                        const auto nseg_i32 = st.layout.section_nseg[sec_index];
                        if (nseg_i32 < 0) {
                            throw std::runtime_error("negative nseg in selected section");
                        }
                        const auto nseg = static_cast<std::size_t>(nseg_i32);
                        for (std::size_t j = 0; j < nseg; ++j) {
                            const double sampled = sample_scalar_or_random_with_context(
                                scalar_or_random,
                                param_seed,
                                static_cast<std::size_t>(global_cell_id),
                                label_u,
                                sec_index,
                                j);
                            const auto [begin_tpl, count_tpl] = segment_instance_range(label_u, sec_index, j);
                            for (std::size_t rep = 0; rep < count_tpl; ++rep) {
                                const std::size_t inst =
                                    base_global +
                                    static_cast<std::size_t>(c) * tpl_n +
                                    begin_tpl +
                                    rep;
                                const auto inst_sorted = static_cast<std::size_t>(inst_permute[inst]);
                                varData.cpu_data[inst_sorted] = sampled;
                            }
                        }
                    }
                }
            }
        };

        auto apply_one_param = [&](std::size_t template_id,
                                   const std::string& sel_label,
                                   std::string_view name,
                                   const ParamValue& value) {
            const auto maybe_coreidx = resolve_coreidx(name);
            if (!maybe_coreidx) {
                // Not a per-node range/state variable. Only allow known mechanism GLOBAL variables.
                if (!std::holds_alternative<double>(value)) {
                    throw std::runtime_error("global mechanism params must be scalar doubles (random not allowed)");
                }
                const double scalar = std::get<double>(value);
                const std::string key = mech_name + "::" + std::string(name);
                auto it = global_param_values.find(key);
                if (it != global_param_values.end()) {
                    if (it->second != scalar) {
                        throw std::runtime_error("conflicting global param " + key);
                    }
                } else {
                    global_param_values.emplace(key, scalar);
                }
                if (!set_global_var_value(mech_name, name, scalar)) {
                    throw std::runtime_error("unknown mechanism param " + std::string(name) +
                                             " for mechanism " + mech_name);
                }
                return;
            }

            const int coreidx = *maybe_coreidx;
            auto it_var = it_table->second.find(coreidx);
            MechVarData& varData = it_var->second;

            apply_range_var(template_id, varData, sel_label, name, value);
            any_update = true;
            touched_coreidx.insert(coreidx);
        };

        for (std::size_t template_id = 0; template_id < parsed.size(); ++template_id) {
            const auto& tpl_ops = parsed[template_id].prepare.inserts();
            if (!tpl_base[template_id].has_value()) {
                continue;
            }
            for (const auto& op : tpl_ops) {
                if (op.mech != mech_name) {
                    continue;
                }
                for (const auto& [name, value] : op.params) {
                    if (is_non_range_metadata_key(name)) {
                        continue;
                    }
                    if (std::holds_alternative<std::string>(value)) {
                        continue;
                    }
                    apply_one_param(template_id, op.label, name, value);
                }
            }

            const auto& tpl_species = parsed[template_id].prepare.ion_species();
            for (const auto& op : tpl_species) {
                if (op.model != mech_name) {
                    continue;
                }
                for (const auto& [name, value] : op.params) {
                    if (is_non_range_metadata_key(name)) {
                        continue;
                    }
                    if (std::holds_alternative<std::string>(value)) {
                        continue;
                    }
                    apply_one_param(template_id, op.label, name, value);
                }
            }
        }

                if (any_update) {
                    for (const int coreidx : touched_coreidx) {
                        auto it = it_table->second.find(coreidx);
                        if (it == it_table->second.end()) {
                            continue;
                        }
                        if (it->second.vecdata) {
                            it->second.vecdata->update_gpu_data_from_cpu(mech->cuda_stream);
                        }
                    }
                }
    }

    // --------------------------------------------------------------------
    // IClamp point-process mechanism (kept in biophys build path for in-memory model setup).
    // --------------------------------------------------------------------
    {
        constexpr int kIClampDataSize = 3;

        if (!iclamp.empty()) {
            std::unordered_map<std::string, std::size_t> template_name_to_id;
            for (std::size_t t = 0; t < morph.templates.size(); ++t) {
                const auto& name = morph.templates[t].name;
                if (!name.empty()) {
                    template_name_to_id.emplace(name, t);
                }
            }

            std::vector<std::int32_t> iclamp_nodes;
            std::vector<double> iclamp_data;

            auto resolve_section_id =
                [](const CellTemplateInfo& tpl,
                   const IClampSpec& spec) -> std::optional<neurong_morph::section_id> {
                if (spec.section_id.has_value()) {
                    return static_cast<neurong_morph::section_id>(*spec.section_id);
                }
                const auto it = tpl.section_name_to_id.find(spec.section_name);
                if (it == tpl.section_name_to_id.end()) {
                    return std::nullopt;
                }
                return it->second;
            };

            auto add_instance = [&](int permuted_node, double delay, double dur, double amp) {
                iclamp_nodes.push_back(static_cast<std::int32_t>(permuted_node));
                iclamp_data.push_back(delay);
                iclamp_data.push_back(dur);
                iclamp_data.push_back(amp);
            };

            for (const auto& spec : iclamp) {
                if (!(spec.dur > 0.0) || spec.amp == 0.0) {
                    continue;
                }
                if (spec.template_name.empty()) {
                    continue;
                }

                const auto it_tpl = template_name_to_id.find(spec.template_name);
                if (it_tpl == template_name_to_id.end()) {
                    continue;
                }
                const auto& tpl = morph.templates[it_tpl->second];
                const auto sec_id = resolve_section_id(tpl, spec);
                if (!sec_id.has_value()) {
                    continue;
                }
                if (!spec.all_cells && spec.cells.empty()) {
                    continue;
                }

                auto apply_to_cell = [&](int cell_id) {
                    const auto sec_index = static_cast<std::size_t>(*sec_id);
                    if (sec_index >= tpl.section_node_base_id.size() || sec_index >= tpl.section_nseg.size()) {
                        return;
                    }
                    const auto base = tpl.section_node_base_id[sec_index];
                    const auto nseg = tpl.section_nseg[sec_index];
                    if (nseg <= 0) {
                        return;
                    }
                    auto j = static_cast<std::int32_t>(static_cast<double>(nseg) * spec.x);
                    j = std::clamp(j, std::int32_t{0}, nseg);
                    const auto tpl_node = base + j;
                    const auto original = map_template_node_to_original(cell_id, tpl_node);
                    if (original < 0) {
                        return;
                    }
                    const int perm = node_permute[static_cast<std::size_t>(original)];
                    add_instance(perm, spec.delay, spec.dur, spec.amp);
                };

                if (spec.all_cells) {
                    for (int local = 0; local < tpl.num_cells; ++local) {
                        apply_to_cell(tpl.cell_base + local);
                    }
                    continue;
                }

                for (const int local : spec.cells) {
                    if (local < 0 || local >= tpl.num_cells) {
                        continue;
                    }
                    apply_to_cell(tpl.cell_base + local);
                }
            }

            if (!iclamp_nodes.empty()) {
                neurong_mech::create_mech(sim,
                                          group,
                                          next_type,
                                          "IClamp",
                                          iclamp_nodes,
                                          kIClampDataSize,
                                          iclamp_data,
                                          /*pdata_size=*/0,
                                          {},
                                          nullptr);
            }
        }
    }

    // --------------------------------------------------------------------
    // Pre-register monitors before recorder init (GPU recorder requires this ordering)
    // --------------------------------------------------------------------
    std::vector<std::tuple<int, int, VarDescriptor>> monitor_keys;
    std::set<VarDescriptor> uniq;
    for (const auto& d : extra_monitors) {
        uniq.insert(d);
    }
    for (int gid : record_gids) {
        const auto tpl_id = morph.cell_template_id[static_cast<std::size_t>(gid)];
        const auto& tpl = morph.templates[tpl_id];
        const int soma_node_original =
            static_cast<int>(map_template_node_to_original(gid, tpl.tpl_soma_node));
        VarDescriptor d("global", "v", soma_node_original, /*array_index=*/0);
        uniq.insert(d);
        monitor_keys.emplace_back(gid, soma_node_original, d);
    }

    std::vector<VarDescriptor> monitors(uniq.begin(), uniq.end());

    NeuronGBiophysBuildResult result{};
    result.monitor_to_handle = sim.init_monitor_data_sets(monitors);

    for (const auto& [gid, soma_node_original, desc] : monitor_keys) {
        auto it = result.monitor_to_handle.find(desc);
        result.record_handles.emplace_back(gid, soma_node_original, it->second);
    }

    return result;
}

}  // namespace neurong_biophysical
