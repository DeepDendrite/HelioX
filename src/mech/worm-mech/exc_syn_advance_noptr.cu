// exc_syn_advance_noptr mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace exc_syn_advance_noptr {

struct MechTrait {
    enum class VarNames {
        // Parameters
        weight, conductance, delta, k, Vth, erev,
        // Assigned variables
        vpre, i,
        // State variable
        s,
        // Intermediate variables
        inf, tau
    };
};

class ExcSynAdvanceNoPtr : public MechTemp<ExcSynAdvanceNoPtr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;
    
    ExcSynAdvanceNoPtr(MechInitParams &param) : MechTemp(param) {
        // Set default values
        init_values.insert({weight, 1.0});
        init_values.insert({conductance, 4.9e-4});
        init_values.insert({delta, 5.0});
        init_values.insert({k, 0.5});
        init_values.insert({Vth, 0.0});
        init_values.insert({erev, 30.0});
        
        // Register variable indices (按NEURON CPP中的顺序)
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
    
    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 根据NEURON CPP中的initmodel函数
        vars(s) = 0.0;  // s = s0; where s0 = 0
        rates(param, vars);
        vars(s) = vars(inf);  // s = inf;
    }
    
    // 电流计算函数
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(i) = vars(weight) * vars(conductance) * vars(s) * (param.volt - vars(erev));
        return vars(i);
    }
    
    // 状态更新函数 - 从NEURON CPP中的states函数提取
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param, vars);
        // 从CPP的states函数：s = s + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau)))*(- ( ( ( inf ) ) / tau ) / ( ( ( ( - 1.0 ) ) ) / tau ) - s) ;
        // 简化：s = s + (1. - exp(-dt/tau)) * (inf - s)
        double tau_val = vars(tau);
        double inf_val = vars(inf);
        double s_old = vars(s);
        vars(s) = s_old + (1.0 - exp(-param.dt / tau_val)) * (inf_val - s_old);
    }

private:
    // rates函数
    DUAL_EXEC void rates(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - vars(vpre)) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }
    
    DUAL_EXEC void rates(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - vars(vpre)) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }
};

REGISTER_MECHANISM("exc_syn_advance_noptr", ExcSynAdvanceNoPtr);

} // namespace exc_syn_advance_noptr