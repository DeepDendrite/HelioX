// egl19 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace egl19_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Parameters
        gbegl19,
        
        // Assigned variables
        minf, hinf, tm, th,
        
        // Ion reversal potential and current
        eca, ica
    };
    
    enum class IonVarNames {
        _ion_eca, _ion_ica
    };
};

class EGL19_Channel : public MechTemp<EGL19_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    EGL19_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gbegl19, 1.0});  // 1 nS/um2
        
        // Register variable indices (based on NEURON CPP field order)
        var_in_coredata_idx.insert({gbegl19, 0});
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
        vars(ica) = vars(gbegl19) * vars(m) * vars(h) * (param.volt - 60.0);
        
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
        double vhm = 5.6, ka = 7.5;
        double vhh = 24.9, ki = 12.0;
        double vhhb = -20.5, kib = 8.1;
        double ahinf = 1.43, bhinf = 0.14;
        double chinf = 5.96, dhinf = 0.60;
        double atm = 2.9, btm = 5.2, ctm = 6.0, dtm = 1.9, etm = 1.4, ftm = 30.0, gtm = 2.3;
        double ath = 0.4, bth = 44.6, cth = -23.0, dth = 5.0, eth = 36.4, fth = 28.7, gth = 3.7, hth = 43.1;
        
        // Calculate steady-state values and time constants
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hinf) = (ahinf / (1.0 + exp(-(v - vhh) / ki)) + bhinf) * (chinf / (1.0 + exp((v - vhhb) / kib)) + dhinf);
        vars(tm) = atm * exp(-pow((v - btm) / ctm, 2)) + dtm * exp(-pow((v - etm) / ftm, 2)) + gtm;
        vars(th) = ath * ((bth / (1.0 + exp((v - cth) / dth))) + (eth / (1.0 + exp((v - fth) / gth))) + hth);
    }
};

REGISTER_MECHANISM("egl19", EGL19_Channel);

} // namespace egl19_worm