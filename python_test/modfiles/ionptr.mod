TITLE ionptr (ion variable POINTER consumer)

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

NEURON {
    POINT_PROCESS ionptr_ng
    POINTER enapre
    RANGE g
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    g = 0.05 (uS)
}

ASSIGNED {
    v (mV)
    i (nA)
    enapre (mV)
}

BREAKPOINT {
    : intentionally non-biophysical; just exercises ion-variable POINTERs
    i = g * (v - enapre)
}

