// exc_syn_advance mechanism for eworm non-lr plugin.
#include "neurong_sdk/plugin_mechanism_api.hpp"
#include "mech_template.cuh"
#include <cmath>

namespace exc_syn_advance_eworm_nonlr {

struct MechTrait {
    enum class VarNames { weight, conductance, delta, k, Vth, erev, vpre, i, s, inf, tau };
};

class ExcSynAdvance : public MechTemp<ExcSynAdvance, MechTrait> {
public:
    using enum MechTrait::VarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;

    ExcSynAdvance(MechInitParams& param): MechTemp(param) {
        init_values.insert({weight, 1.0});
        init_values.insert({conductance, 4.9e-4});
        init_values.insert({delta, 5.0});
        init_values.insert({k, 0.5});
        init_values.insert({Vth, 0.0});
        init_values.insert({erev, 30.0});
        var_in_coredata_idx.insert({weight, 0});
        var_in_coredata_idx.insert({conductance, 1});
        var_in_coredata_idx.insert({delta, 2});
        var_in_coredata_idx.insert({k, 3});
        var_in_coredata_idx.insert({Vth, 4});
        var_in_coredata_idx.insert({erev, 5});
        var_in_coredata_idx.insert({vpre, 6});
        var_in_coredata_idx.insert({i, 7});
        var_in_coredata_idx.insert({s, 8});
        var_in_coredata_idx.insert({inf, 9});
        var_in_coredata_idx.insert({tau, 10});
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& param, VarAccessor<MechTrait>& vars) {
        vars(s) = 0.0;
        rates(param, vars);
        vars(s) = vars(inf);
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait>& vars) {
        vars(i) = vars(weight) * vars(conductance) * vars(s) * (param.volt - vars(erev));
        return vars(i);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& param, VarAccessor<MechTrait>& vars) {
        rates(param, vars);
        double tau_val = vars(tau);
        double inf_val = vars(inf);
        double s_old = vars(s);
        vars(s) = s_old + (1.0 - exp(-param.dt / tau_val)) * (inf_val - s_old);
    }

private:
    DUAL_EXEC void rates(MechTempStateParam& param, VarAccessor<MechTrait>& vars) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - vars(vpre)) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }

    DUAL_EXEC void rates(MechTempInitParam& param, VarAccessor<MechTrait>& vars) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - vars(vpre)) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }
};

REGISTER_MECHANISM("exc_syn_advance", ExcSynAdvance);

}  // namespace exc_syn_advance_eworm_nonlr
