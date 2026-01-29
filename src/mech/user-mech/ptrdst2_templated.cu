// ptrdst2 mechanism (templated, NEURON-compatible)
// Consumes two POINTERs:
// - ipre: points to another mechanism's variable (e.g. ptrsrc.i)
// - gpre: points to a density mechanism variable (e.g. pas.g_pas)
//
// NMODL (nrnivmodl) layout for ptrdst2.mod:
//   g    -> fpfield<0>
//   i    -> fpfield<1>
//   ipre -> _ppvar[2]
//   gpre -> _ppvar[3]

#include "mech_template.cuh"

namespace PTRDST2 {

// Shorthand for enum->int conversion in semantics registration.
#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames { g, i };
    enum class PointerVarNames { ipre, gpre };
};

class PtrDst2 final : public MechTemp<PtrDst2, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::PointerVarNames;

  public:
    constexpr static MechFlags flags = MechFlags::ENABLE_CURRENT | MechFlags::POINT_PROCESS;

    PtrDst2(MechInitParams& param) : MechTemp(param) {
        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({i, 1});

        init_values.insert({g, 0.1});
        init_values.insert({i, 0.0});
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait> vars) {
        const double ipre_val = vars.Ptr(ipre);
        const double gpre_val = vars.Ptr(gpre);

        vars(i) = vars(g) * (param.volt - ipre_val) + 1000.0 * gpre_val;
        return vars(i);
    }
};

REGISTER_MECHANISM("ptrdst2_ng", PtrDst2);
REGISTER_POINTER_DPARAM_SLOTS("ptrdst2_ng", 2, 3);
// NEURON-generated dparam semantics for POINT_PROCESS with two POINTERs:
// 0: area, 1: pntproc, 2: pointer, 3: pointer
REGISTER_DPARAM_SEMANTICS("ptrdst2_ng", DPSEM(area), DPSEM(pntproc), DPSEM(pointer), DPSEM(pointer));

}  // namespace PTRDST2
