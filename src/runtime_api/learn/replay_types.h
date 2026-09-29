#pragma once

#include <cstdint>

// Phase 1: shared POD types for runtime_api/learn replay APIs.
// This is only a container for configuration so that later phases can
// move replay logic out of python_api without changing signatures.

namespace neurong::runtime_api::learn {

struct ReplayConfig {
    double tstop_ms = 0.0;
    int32_t k_mul = 1;
    int32_t precise = 1;
    double v_init = -65.0;
    double dt_ms = 0.025;

    double grad_scale = 1.0;
    double eps = 1e-6;
    double grad_l2norm_threshold = 1e6;
    int32_t clip_strategy = 1;
    int32_t clip_check_every = 1;
};

}  // namespace neurong::runtime_api::learn

