#pragma once

#include "biophys_prepare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace neurong_biophysical {

[[nodiscard]] inline std::uint64_t hash_string64(std::string_view s) noexcept {
    // Stable FNV-1a 64-bit hash for deterministic cross-run sampling.
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = x ^ (x >> 31);
    return x;
}

[[nodiscard]] inline std::uint64_t combine_seed(std::uint64_t seed, std::uint64_t value) noexcept {
    return splitmix64(seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2)));
}

template <typename... Ts>
[[nodiscard]] inline std::uint64_t combine_seed_many(std::uint64_t seed, Ts... values) noexcept {
    ((seed = combine_seed(seed, static_cast<std::uint64_t>(values))), ...);
    return seed;
}

[[nodiscard]] inline double uniform01_from_u64(std::uint64_t bits) noexcept {
    // Convert to [0,1) with 53-bit mantissa precision.
    constexpr double inv = 1.0 / static_cast<double>(1ull << 53);
    return static_cast<double>(bits >> 11) * inv;
}

[[nodiscard]] inline double sample_random_rule(const RandomParamSpec& rule, std::uint64_t seed) {
    switch (rule.distribution) {
    case RandomDist::Fixed:
        return rule.value;
    case RandomDist::Uniform: {
        return rule.low + (rule.high - rule.low) * uniform01_from_u64(splitmix64(seed));
    }
    case RandomDist::Normal: {
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        const double u1 = std::max(uniform01_from_u64(splitmix64(combine_seed(seed, 1))), 1e-12);
        const double u2 = uniform01_from_u64(splitmix64(combine_seed(seed, 2)));
        const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
        return rule.mean + rule.stddev * z;
    }
    }
    throw std::runtime_error("unsupported random distribution");
}

[[nodiscard]] inline double sample_scalar_or_random(const ScalarOrRandom& value, std::uint64_t seed) {
    if (std::holds_alternative<double>(value)) {
        return std::get<double>(value);
    }
    return sample_random_rule(std::get<RandomParamSpec>(value), seed);
}

[[nodiscard]] inline ScalarOrRandom to_scalar_or_random(const ParamValue& value,
                                                        std::string_view context) {
    if (std::holds_alternative<double>(value)) {
        return std::get<double>(value);
    }
    if (std::holds_alternative<RandomParamSpec>(value)) {
        return std::get<RandomParamSpec>(value);
    }
    throw std::runtime_error(std::string(context) + " must be scalar or random spec");
}

}  // namespace neurong_biophysical
