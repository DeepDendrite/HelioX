// gapjunction mechanism for eworm non-lr plugin.
#include "neurong_sdk/plugin_mechanism_api.hpp"
#include "mech_template.cuh"
#include "dparam_semantics.h"

namespace gapjunction_eworm_nonlr {

#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames { weight, i };
    enum class PointerVarNames { vpre };
};

class GapJunction : public MechTemp<GapJunction, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::PointerVarNames;
    constexpr static MechFlags flags = ENABLE_CURRENT | POINT_PROCESS;

    GapJunction(MechInitParams& param): MechTemp(param) {
        init_values.insert({weight, 0.0});
        var_in_coredata_idx.insert({weight, 0});
        var_in_coredata_idx.insert({i, 2});
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait>& vars) {
        vars(i) = vars(weight) * (param.volt - vars.Ptr(vpre));
        return vars(i);
    }
};

REGISTER_MECHANISM("gapjunction", GapJunction);
REGISTER_POINTER_DPARAM_SLOTS("gapjunction", 2);
REGISTER_DPARAM_SEMANTICS("gapjunction", DPSEM(area), DPSEM(pntproc), DPSEM(pointer));

}  // namespace gapjunction_eworm_nonlr

#undef DPSEM
