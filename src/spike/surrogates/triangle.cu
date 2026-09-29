#include "spike_vjp_surrogate_template.cuh"

#include <cmath>

namespace neurong::spike_vjp {

struct TriangleSurrogate {
    static constexpr const char* name = "triangle";

    __host__ __device__ __forceinline__
    static double eval(double voltage_minus_threshold, const SpikeVjpSurrogateConfig& config) {
        if (config.width_mv <= 0.0) {
            return 0.0;
        }
        const double scaled = fabs(voltage_minus_threshold) / config.width_mv;
        return scaled < 1.0 ? (1.0 - scaled) / config.width_mv : 0.0;
    }
};

REGISTER_SPIKE_VJP_SURROGATE(TriangleSurrogate, "triangular");

}  // namespace neurong::spike_vjp

