// unc2 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace unc2_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Parameters
        gbunc2,
        
        // Assigned variables
        minf, hinf, tm, th,
        
        // Ion reversal potential and current
        eca, ica
    };
    
    enum class IonVarNames {
        _ion_eca, _ion_ica
    };
};

class UNC2_Channel : public MechTemp<UNC2_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    UNC2_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gbunc2, 1.0});  // 1 nS/um2
        
        // Register variable indices (based on NEURON CPP field order)
        var_in_coredata_idx.insert({gbunc2, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({hinf, 2});
        var_in_coredata_idx.insert({tm, 3});
        var_in_coredata_idx.insert({th, 4});
        var_in_coredata_idx.insert({m, 5});
        var_in_coredata_idx.insert({h, 6});
        var_in_coredata_idx.insert({eca, 7});
        var_in_coredata_idx.insert({ica, 8});
        
        // Register ion channel variables
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
    }
    
    // Initialize state variables to steady-state values
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(minf);
        vars(h) = vars(hinf);
    }
    
    // Current calculation
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(eca) = vars(_ion_eca);
        
        // Calculate calcium current
        // Note: using fixed eca=60mV as in original mod file
        vars(ica) = vars(gbunc2) * vars(m) * vars(h) * (param.volt - 60.0);
        
        // Update global ion current (only when updateIon=true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
        }
        
        return vars(ica);
    }
    
    // State variable updates
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        // Update state variables using exponential integration
        vars(m) = vars(m) + (1.0 - exp(-param.dt / vars(tm))) * (vars(minf) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-param.dt / vars(th))) * (vars(hinf) - vars(h));
    }
    
private:
    // Helper function to calculate rates and steady-state values
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        // Parameters from mod file
        double vhm = -12.2, ka = 4.0;
        double vhh = -52.5, ki = 5.6;
        double atm = 1.5, btm = -8.2, ctm = 9.1, dtm = 15.4, etm = 0.1;
        double ath = 83.8, bth = 52.9, cth = -3.5, dth = 72.1, eth = 23.9, fth = -3.6;
        
        // Calculate steady-state values and time constants
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hinf) = 1.0 / (1.0 + exp((v - vhh) / ki));
        vars(tm) = atm / (exp(-(v - btm) / ctm) + exp((v - btm) / dtm)) + etm;
        vars(th) = ath / (1.0 + exp(-(v - bth) / cth)) + dth / (1.0 + exp((v - eth) / fth));
    }
};

REGISTER_MECHANISM("unc2", UNC2_Channel);

} // namespace unc2_worm