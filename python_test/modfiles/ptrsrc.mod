TITLE ptrsrc (POINTER target source)

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
}

NEURON {
    POINT_PROCESS ptrsrc_ng
    RANGE amp
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    amp = 0 (nA)
}

ASSIGNED {
    v (mV)
    i (nA)
}

BREAKPOINT {
    i = amp
}
