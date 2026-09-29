#pragma once

#include "mech_template.cuh"

#include <cmath>

DUAL_EXEC double l5pc_vtrap(double x, double y) {
    if (fabs(x / y) < 1e-6) {
        return y * (1 - x / y / 2);
    }
    return x / (exp(x / y) - 1);
}
