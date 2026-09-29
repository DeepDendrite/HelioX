// External SDK smoke example for MechTemp authoring.
#include "neurong_sdk/mech_template.cuh"

#include <array>
#include <cassert>

namespace sdk_template_gap {

#define MECH_CLASS_NAME SdkTemplateGap
static const char* MECH_NAME_TO_REG = "SdkTemplateGap";

struct MechTrait {
    enum class VarNames {
        g,
        vgap,
        i,
        drive,
        _g,
    };
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT | ENABLE_CURRENT_VJP;

    static constexpr auto LearnableVars = std::array{MechTrait::VarNames::g};
    static constexpr auto VjpCarryVars = std::array{MechTrait::VarNames::vgap};
    static constexpr auto CurrentVjpTapeVars = std::array{MechTrait::VarNames::drive};

    using enum MechTrait::VarNames;

    explicit MECH_CLASS_NAME(MechInitParams& param) : MechTemp(param) {
        need_area = true;
        init_values.insert({g, 0.0});
        init_values.insert({drive, 0.0});

        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({vgap, 1});
        var_in_coredata_idx.insert({i, 2});
        var_in_coredata_idx.insert({_g, 4});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& param, VarAccessor<MechTrait>& vars) {
        (void)param;
        vars(i) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait>& vars) {
        const double gap_drive = vars(vgap) - param.volt;
        if (param.updateIon) {
            vars(drive) = gap_drive;
        }
        const double current = gap_drive * vars(g);
        vars(i) = current;
        return current;
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait> vars) {
        vars.idx = param.idx;
        const double grad_i = param.grad_mech_current;
        mechAtomAdd(&adjoint_ref<vgap>(param, vars), grad_i * vars(g));
        mechAtomAdd(&param.grad_v[param.node_index], -grad_i * vars(g));
        mechAtomAdd(&grad_ref<g>(param, vars), grad_i * tape_ref<drive>(param, vars));
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

#undef MECH_CLASS_NAME

}  // namespace sdk_template_gap
