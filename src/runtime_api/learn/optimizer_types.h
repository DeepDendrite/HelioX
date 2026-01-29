#pragma once

#include <cstdint>
#include <vector>

namespace heliox::runtime_api::learn {

struct AdamParams {
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

struct AdamState {
    int64_t step = 0;
    std::vector<double> m;
    std::vector<double> v;
    AdamParams params;
};

}  // namespace heliox::runtime_api::learn

