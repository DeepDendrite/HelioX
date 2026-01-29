// egl2 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace egl2_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m,
        
        // Parameters
        gbegl2,
        
        // Assigned variables
        minf, tm,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class EGL2_Channel : public MechTemp<EGL2_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    EGL2_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gbegl2, 1.0});  // 1 nS/um2
        
        // Register variable indices (based on NEURON CPP field order)
        var_in_coredata_idx.insert({gbegl2, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({tm, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({ik, 5});
        
        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    // Initialize state variables to steady-state values
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(minf);
    }
    
    // Current calculation
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);
        
        // Calculate potassium current
        // Note: using fixed ek=-80mV as in original mod file
        vars(ik) = vars(gbegl2) * vars(m) * (param.volt + 80.0);
        
        // Update global ion current (only when updateIon=true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    // State variable updates
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        // Update state variable using exponential integration
        vars(m) = vars(m) + (1.0 - exp(-param.dt / vars(tm))) * (vars(minf) - vars(m));
    }
    
private:
    // Helper function to calculate rates and steady-state values
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        // Parameters from mod file
        double vhm = -6.9, ka = 14.9;
        double atm = 1845.8, btm = -122.6, ctm = 13.8, dtm = 1517.74;
        
        // Calculate steady-state values and time constants
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(tm) = atm / (1.0 + exp((v - btm) / ctm)) + dtm;
    }
};

REGISTER_MECHANISM("egl2", EGL2_Channel);

} // namespace egl2_worm