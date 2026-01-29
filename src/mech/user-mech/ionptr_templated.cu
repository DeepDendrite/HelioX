// ionptr_ng mechanism (templated, NEURON-compatible)
// Exercises POINTER targets that live in ion mechanisms (e.g. na_ion.erev == ena).
//
// NMODL (nrnivmodl) layout for ionptr.mod:
//   g      -> fpfield<0>
//   i      -> fpfield<1>
//   enapre -> _ppvar[2]

#include "mech_template.cuh"

namespace IONPTR {

// Shorthand for enum->int conversion in semantics registration.
#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames { g, i };
    enum class PointerVarNames { enapre };
};

class IonPtr final : public MechTemp<IonPtr, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::PointerVarNames;

  public:
    constexpr static MechFlags flags = MechFlags::ENABLE_CURRENT | MechFlags::POINT_PROCESS;

    IonPtr(MechInitParams& param) : MechTemp(param) {
        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({i, 1});

        init_values.insert({g, 0.05});
        init_values.insert({i, 0.0});
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait> vars) {
        vars(i) = vars(g) * (param.volt - vars.Ptr(enapre));
        return vars(i);
    }
};

REGISTER_MECHANISM("ionptr_ng", IonPtr);
REGISTER_POINTER_DPARAM_SLOTS("ionptr_ng", 2);
// NEURON-generated dparam semantics for POINT_PROCESS with one POINTER:
// 0: area, 1: pntproc, 2: pointer
REGISTER_DPARAM_SEMANTICS("ionptr_ng", DPSEM(area), DPSEM(pntproc), DPSEM(pointer));

}  // namespace IONPTR
