#include "spike_vjp_surrogate_template.cuh"

#include <cmath>

namespace neurong::spike_vjp {

struct FastSigmoidSurrogate {
    static constexpr const char* name = "fast_sigmoid";

    __host__ __device__ __forceinline__
    static double eval(double voltage_minus_threshold, const SpikeVjpSurrogateConfig& config) {
        if (config.width_mv <= 0.0) {
            return 0.0;
        }
        const double scaled = 1.0 + fabs(voltage_minus_threshold) / config.width_mv;
        return 0.5 / config.width_mv / (scaled * scaled);
    }
};

REGISTER_SPIKE_VJP_SURROGATE(FastSigmoidSurrogate, "fastsigmoid", "fastsig");

}  // namespace neurong::spike_vjp

