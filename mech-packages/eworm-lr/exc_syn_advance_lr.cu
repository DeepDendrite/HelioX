// exc_syn_advance_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace exc_syn_advance_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        w, g, delta, k, Vth, erev, dvpre,
        // ASSIGNED variables
        i, vpre, inf, tau, pure_i, didv, dsdvpre, didvpre,
        // STATE variables  
        s, s_ds
    };
};

class ExcSynAdvanceLr : public MechTemp<ExcSynAdvanceLr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;
    
    ExcSynAdvanceLr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({w, 1.0});
        init_values.insert({g, 4.9});
        init_values.insert({delta, 5.0});
        init_values.insert({k, 0.5});
        init_values.insert({Vth, 0.0});
        init_values.insert({erev, 30.0});
        init_values.insert({dvpre, 1e-3});
        
        // Register variable indices
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
        
        // This is a point process - no global or ion variables to register
    }
    
    // Helper function for rates procedure
    DUAL_EXEC void rates(VarAccessor<MechTrait> &vars, double v) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - v) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(s) = 0.0;
        vars(s_ds) = 0.0;
        
        // Execute INITIAL block logic
        rates(vars, vars(vpre));
        vars(s_ds) = vars(inf);
        vars(s) = vars(inf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block
        vars(pure_i) = vars(g) * vars(s) * (param.volt - vars(erev));
        vars(i) = vars(w) * vars(pure_i);
        vars(didv) = -vars(w) * vars(g) * vars(s);
        vars(dsdvpre) = (vars(s_ds) - vars(s)) / vars(dvpre);
        vars(didvpre) = vars(w) * vars(g) * (vars(erev) - param.volt) * vars(dsdvpre);
        
        // For point process, return i directly (not added to ion current)
        return vars(i);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Save current state value
        vars(s_ds) = vars(s);
        
        // First rates call with vpre + dvpre
        rates(vars, vars(vpre) + vars(dvpre));
        // Update s_ds using cnexp integration
        vars(s_ds) = vars(s_ds) + (1.0 - exp(param.dt * ((-1.0) / vars(tau)))) * 
                     (-(vars(inf) / vars(tau)) / ((-1.0) / vars(tau)) - vars(s_ds));
        
        // Second rates call with vpre
        rates(vars, vars(vpre));
        // Update s using cnexp integration
        vars(s) = vars(s) + (1.0 - exp(param.dt * ((-1.0) / vars(tau)))) * 
                  (-(vars(inf) / vars(tau)) / ((-1.0) / vars(tau)) - vars(s));
    }
};

REGISTER_MECHANISM("exc_syn_advance_lr", ExcSynAdvanceLr);

} // namespace exc_syn_advance_lr
