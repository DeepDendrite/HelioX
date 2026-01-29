// ptrsrc mechanism (templated, NEURON-compatible)
// Source mechanism for POINTER tests: exposes a NONSPECIFIC_CURRENT `i` that
// is simply set to a parameter `amp`.
//
// NMODL (nrnivmodl) layout for ptrsrc.mod:
//   amp -> fpfield<0>
//   i   -> fpfield<1>

#include "mech_template.cuh"

namespace PTRSRC {

// Shorthand for enum->int conversion in semantics registration.
#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames { amp, i };
};

class PtrSrc final : public MechTemp<PtrSrc, MechTrait> {
    using enum MechTrait::VarNames;

  public:
    constexpr static MechFlags flags = MechFlags::ENABLE_CURRENT | MechFlags::POINT_PROCESS;

    PtrSrc(MechInitParams& param) : MechTemp(param) {
        var_in_coredata_idx.insert({amp, 0});
        var_in_coredata_idx.insert({i, 1});

        init_values.insert({amp, 0.0});
        init_values.insert({i, 0.0});
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& /*param*/, VarAccessor<MechTrait> vars) {
        vars(i) = vars(amp);
        return vars(i);
    }
};

REGISTER_MECHANISM("ptrsrc_ng", PtrSrc);
// NEURON-generated dparam semantics for POINT_PROCESS without POINTER:
// 0: area, 1: pntproc
REGISTER_DPARAM_SEMANTICS("ptrsrc_ng", DPSEM(area), DPSEM(pntproc));

}  // namespace PTRSRC
