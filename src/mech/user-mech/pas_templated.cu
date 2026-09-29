// pas_templated mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <array>

namespace PAS{

struct MechTrait{
    enum class VarNames{
        i,e_pas,g_pas
    };
};

class PAS_Templated:public MechTemp<PAS_Templated,MechTrait>{

    using enum MechTrait::VarNames;
public:
    constexpr static MechFlags flags =
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_CURRENT_VJP;
    static constexpr auto LearnableVars = std::array{g_pas, e_pas};
    PAS_Templated(MechInitParams &param):MechTemp(param){
        var_in_coredata_idx.insert({g_pas,0});
        var_in_coredata_idx.insert({e_pas,1});

        init_values.insert({e_pas,-60});
        init_values.insert({g_pas,0.00016666});

        printf_debug("PAS_Templated init_vars\n");
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param,VarAccessor<MechTrait> vars) {
        double current = vars(g_pas) * (param.volt - vars(e_pas));
        vars(i) = current;
        return current;
    };

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait> vars) {
        vars.idx = param.idx;
        // For pas: i = g_pas * (v - e_pas), and in non-electrode path rhs += -i.
        // MechTemp already converts grad_rhs to grad_mech_current, so dL/dv accumulates as grad_mech_current * di/dv.
        mechAtomAdd(&grad_ref<g_pas>(param, vars), param.grad_mech_current * (param.volt - vars(e_pas)));
        mechAtomAdd(&grad_ref<e_pas>(param, vars), param.grad_mech_current * (-vars(g_pas)));
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * vars(g_pas));
    }
};

REGISTER_MECHANISM("pas",PAS_Templated);
}//end of namespace PAS
