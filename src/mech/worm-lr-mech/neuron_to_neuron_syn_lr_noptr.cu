// neuron_to_neuron_syn_lr_noptr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace neuron_to_neuron_syn_lr_noptr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        w, g, delta, k, Vth, erev, dvpre,
        // ASSIGNED variables
        i, vpre, pure_i, didv, dsdvpre, didvpre,
        // STATE variables
        s, s_ds,
        // Internal computation variables
        inf, tau
    };
};

class NeuronToNeuronSynLrNoptr : public MechTemp<NeuronToNeuronSynLrNoptr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    
    // Point process mechanism with init, current, and state functions
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;
    
    NeuronToNeuronSynLrNoptr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({w, 1.0});
        init_values.insert({g, 4.9});
        init_values.insert({delta, 5.0});
        init_values.insert({k, 0.5});
        init_values.insert({Vth, -20.0});
        init_values.insert({erev, 30.0});
        init_values.insert({dvpre, 1e-3});
        
        // Register variable indices based on CPP setup_instance
        var_in_coredata_idx.insert({w, 0});
        var_in_coredata_idx.insert({g, 1});
        var_in_coredata_idx.insert({delta, 2});
        var_in_coredata_idx.insert({k, 3});
        var_in_coredata_idx.insert({Vth, 4});
        var_in_coredata_idx.insert({erev, 5});
        var_in_coredata_idx.insert({dvpre, 6});
        var_in_coredata_idx.insert({i, 7});
        var_in_coredata_idx.insert({vpre, 8});
        var_in_coredata_idx.insert({pure_i, 9});
        var_in_coredata_idx.insert({didv, 10});
        var_in_coredata_idx.insert({dsdvpre, 11});
        var_in_coredata_idx.insert({didvpre, 12});
        var_in_coredata_idx.insert({s, 13});
        var_in_coredata_idx.insert({s_ds, 14});
        var_in_coredata_idx.insert({inf, 15});
        var_in_coredata_idx.insert({tau, 16});
    }
    
    // Helper function for rates calculation
    DUAL_EXEC void rates(VarAccessor<MechTrait> &vars, double _lv) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - _lv) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }
    
    // Initialize function - corresponds to INITIAL block
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize state variables to 0 first (NEURON default)
        vars(s) = 0.0;
        vars(s_ds) = 0.0;
        
        // Execute INITIAL block logic from CPP nrn_init function
        rates(vars, vars(vpre));
        vars(s_ds) = vars(inf);
        vars(s) = vars(inf);
        
        // Conditional parameter setting based on w
        if (vars(w) > 0.0) {
            vars(g) = 4.9;
            vars(k) = 0.5;
            vars(erev) = 30.0;
        } else {
            vars(g) = 2.0;
            vars(k) = 0.015000001;
            vars(erev) = -70.0;
        }
    }
    
    // Current calculation function - corresponds to BREAKPOINT block
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // From nrn_current_neuron_to_neuron_syn_lr_noptr function
        vars(pure_i) = vars(g) * vars(s) * (param.volt - vars(erev));
        vars(i) = fabs(vars(w)) * vars(pure_i);
        vars(didv) = -fabs(vars(w)) * vars(g) * vars(s);
        vars(dsdvpre) = (vars(s_ds) - vars(s)) / vars(dvpre);
        vars(didvpre) = fabs(vars(w)) * vars(g) * (vars(erev) - param.volt) * vars(dsdvpre);
        
        return vars(i);
    }
    
    // State update function - corresponds to DERIVATIVE states with cnexp method
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Strictly follow the CPP nrn_state function with two rates calls
        // First, save current s value to s_ds
        vars(s_ds) = vars(s);
        
        // First rates call: rates(vpre + dvpre)
        rates(vars, vars(vpre) + vars(dvpre));
        // Update s_ds using cnexp integration - exact formula from CPP
        vars(s_ds) = vars(s_ds) + (1.0 - exp(param.dt * ((-1.0) / vars(tau)))) * (-(vars(inf) / vars(tau)) / ((-1.0) / vars(tau)) - vars(s_ds));
        
        // Second rates call: rates(vpre)
        rates(vars, vars(vpre));
        // Update s using cnexp integration - exact formula from CPP  
        vars(s) = vars(s) + (1.0 - exp(param.dt * ((-1.0) / vars(tau)))) * (-(vars(inf) / vars(tau)) / ((-1.0) / vars(tau)) - vars(s));
    }
};

REGISTER_MECHANISM("neuron_to_neuron_syn_lr_noptr", NeuronToNeuronSynLrNoptr);

} // namespace neuron_to_neuron_syn_lr_noptr