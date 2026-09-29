#include "spike_vjp_surrogate_template.cuh"

#include <cmath>

namespace neurong::spike_vjp {

struct RectangleSurrogate {
    static constexpr const char* name = "rectangle";

    __host__ __device__ __forceinline__
    static double eval(double voltage_minus_threshold, const SpikeVjpSurrogateConfig& config) {
        if (config.width_mv <= 0.0) {
            return 0.0;
        }
        return fabs(voltage_minus_threshold) < config.width_mv ? 0.5 / config.width_mv : 0.0;
    }
};

REGISTER_SPIKE_VJP_SURROGATE(RectangleSurrogate, "rect", "boxcar");

}  // namespace neurong::spike_vjp

