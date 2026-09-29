#include "spike_vjp_surrogate.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace neurong::spike_vjp {

std::string normalize_spike_vjp_surrogate_name(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    name.erase(
        std::remove_if(name.begin(), name.end(), [](char c) {
            return c == '-' || c == '_';
        }),
        name.end());
    return name;
}

SpikeVjpSurrogateRegistry& SpikeVjpSurrogateRegistry::getInstance() {
    static SpikeVjpSurrogateRegistry registry;
    return registry;
}

void SpikeVjpSurrogateRegistry::registerSurrogate(
    SpikeVjpSurrogateOps ops,
    std::initializer_list<const char*> aliases) {
    if (ops.name == nullptr || ops.name[0] == '\0' || ops.run_cpu == nullptr || ops.launch_gpu == nullptr) {
        std::cerr << "FATAL: invalid spike VJP surrogate registration\n";
        std::abort();
    }
    ops_.push_back(ops);
    const SpikeVjpSurrogateOps* stored = &ops_.back();

    auto register_alias = [&](const char* alias) {
        if (alias == nullptr || alias[0] == '\0') {
            return;
        }
        const std::string normalized = normalize_spike_vjp_surrogate_name(alias);
        auto [it, inserted] = alias2ops_.emplace(normalized, stored);
        if (!inserted && it->second != stored) {
            std::cerr << "FATAL: duplicate spike VJP surrogate alias \"" << alias << "\"\n"
                      << "  existing: " << (it->second->source_symbol ? it->second->source_symbol : "<unknown>")
                      << " from " << (it->second->source_file ? it->second->source_file : "<unknown>") << "\n"
                      << "  new:      " << (ops.source_symbol ? ops.source_symbol : "<unknown>")
                      << " from " << (ops.source_file ? ops.source_file : "<unknown>") << "\n";
            std::abort();
        }
    };

    register_alias(ops.name);
    for (const char* alias : aliases) {
        register_alias(alias);
    }
}

const SpikeVjpSurrogateOps* SpikeVjpSurrogateRegistry::find(const std::string& name) const {
    const std::string normalized = normalize_spike_vjp_surrogate_name(name);
    const auto it = alias2ops_.find(normalized);
    return it == alias2ops_.end() ? nullptr : it->second;
}

const SpikeVjpSurrogateOps& SpikeVjpSurrogateRegistry::require(const std::string& name) const {
    if (const auto* ops = find(name)) {
        return *ops;
    }
    std::string available;
    for (const auto& ops : ops_) {
        if (!available.empty()) {
            available += ", ";
        }
        available += ops.name == nullptr ? "<unnamed>" : ops.name;
    }
    throw std::runtime_error(
        "unknown spike VJP surrogate kind: " + name +
        (available.empty() ? "" : "; available: " + available));
}

std::vector<std::string> SpikeVjpSurrogateRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(ops_.size());
    for (const auto& ops : ops_) {
        result.emplace_back(ops.name == nullptr ? "" : ops.name);
    }
    return result;
}

}  // namespace neurong::spike_vjp

