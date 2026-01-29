TITLE ptrdst2 (POINTER consumer)

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

NEURON {
    POINT_PROCESS ptrdst2_ng
    POINTER ipre, gpre
    RANGE g
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    g = 0.1 (uS)
}

ASSIGNED {
    v (mV)
    i (nA)
    ipre (nA)
    gpre (uS)
}

BREAKPOINT {
    : intentionally non-biophysical; just exercises POINTERs
    i = g * (v - ipre) + 1000.0 * gpre
}
