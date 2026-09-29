#pragma once

#include <cstdint>
#include <string>

// Phase 1: shared POD types for runtime_api/core.
//
// Note:
// - Keep this header independent from nanobind/Python.
// - Prefer simple POD structs and enums that can be used across frontends.

namespace neurong::runtime_api::core {

struct MonitorKey {
    std::string mech;
    std::string var;
    int32_t idx = -1;
    int32_t array_index = 0;
};

}  // namespace neurong::runtime_api::core

