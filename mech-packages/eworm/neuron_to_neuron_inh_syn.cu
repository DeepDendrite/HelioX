// neuron_to_neuron_inh_syn mechanism for eworm non-lr plugin.
#include "neurong_sdk/plugin_mechanism_api.hpp"
#include "mech_template.cuh"
#include "dparam_semantics.h"
#include <cmath>

namespace neuron_to_neuron_inh_syn_eworm_nonlr {

#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames { weight, conductance, delta, k, Vth, erev, i, s, inf, tau };
    enum class PointerVarNames { vpre };
};

class NeuronToNeuronInhSyn : public MechTemp<NeuronToNeuronInhSyn, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::PointerVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;

    NeuronToNeuronInhSyn(MechInitParams& param): MechTemp(param) {
        init_values.insert({weight, 1.0});
        init_values.insert({conductance, 0.0002});
        init_values.insert({delta, 5.0});
        init_values.insert({k, 0.015});
        init_values.insert({Vth, -20.0});
        init_values.insert({erev, -70.0});
        var_in_coredata_idx.insert({weight, 0});
        var_in_coredata_idx.insert({conductance, 1});
        var_in_coredata_idx.insert({delta, 2});
        var_in_coredata_idx.insert({k, 3});
        var_in_coredata_idx.insert({Vth, 4});
        var_in_coredata_idx.insert({erev, 5});
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
        const double vpre_val = vars.Ptr(vpre);
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - vpre_val) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }

    DUAL_EXEC void rates(MechTempInitParam& param, VarAccessor<MechTrait>& vars) {
        const double vpre_val = vars.Ptr(vpre);
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - vpre_val) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }
};

REGISTER_MECHANISM("neuron_to_neuron_inh_syn", NeuronToNeuronInhSyn);
REGISTER_POINTER_DPARAM_SLOTS("neuron_to_neuron_inh_syn", 2);
REGISTER_DPARAM_SEMANTICS(
    "neuron_to_neuron_inh_syn", DPSEM(area), DPSEM(pntproc), DPSEM(pointer));

}  // namespace neuron_to_neuron_inh_syn_eworm_nonlr

#undef DPSEM
