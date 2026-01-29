#pragma once

// NEURON/CoreNEURON dparam semantics encoding.
//
// Reference:
// - ~/nrn/src/coreneuron/mechanism/register_mech.cpp (hoc_register_dparam_semantics)
// - ~/nrn/src/nrnoc/ion_semantics.h (nrn_semantics_from_ion)
//
// This header exists to avoid scattering "magic numbers" like -5 across the codebase.

enum class DparamSemantics : int {
    area = -1,
    iontype = -2,
    cvodeieq = -3,
    netsend = -4,
    pointer = -5,
    pntproc = -6,
    bbcorepointer = -7,
    watch = -8,
    diam = -9,
    fornetcon = -10,
    random = -11,
};

inline constexpr int dpsem(DparamSemantics s) {
    return static_cast<int>(s);
}

inline constexpr bool dpsem_is_ion_or_ionstyle(int s) {
    return s >= 0;
}

// NEURON ion semantics: s = 2*ion_mech_type + style (style 0 for ion, 1 for ionstyle).
inline constexpr int dpsem_ion_mech_type(int s) {
    return s / 2;
}

inline constexpr int dpsem_ion_style(int s) {
    return s & 1;
}

