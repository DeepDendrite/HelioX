// shl1 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace shl1_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, hf, hs,
        
        // Parameters
        gbshl1,
        
        // Assigned variables
        minf, hfinf, hsinf, tm, thf, ths,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class SHL1_Channel : public MechTemp<SHL1_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    SHL1_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gbshl1, 1.0});  // 1 nS/um2
        
        // Register variable indices (based on NEURON CPP field order)
        var_in_coredata_idx.insert({gbshl1, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({hfinf, 2});
        var_in_coredata_idx.insert({hsinf, 3});
        var_in_coredata_idx.insert({tm, 4});
        var_in_coredata_idx.insert({thf, 5});
        var_in_coredata_idx.insert({ths, 6});
        var_in_coredata_idx.insert({m, 7});
        var_in_coredata_idx.insert({hf, 8});
        var_in_coredata_idx.insert({hs, 9});
        var_in_coredata_idx.insert({ek, 10});
        var_in_coredata_idx.insert({ik, 11});
        
        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    // Initialize state variables to steady-state values
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(minf);
        vars(hf) = vars(hfinf);
        vars(hs) = vars(hsinf);
    }
    
    // Current calculation
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);
        
        // Calculate potassium current
        // Note: using fixed ek=-80mV as in original mod file
        vars(ik) = vars(gbshl1) * vars(m) * vars(m) * vars(m) * (0.7 * vars(hf) + 0.3 * vars(hs)) * (param.volt + 80.0);
        
        // Update global ion current (only when updateIon=true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    // State variable updates
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        // Update state variables using exponential integration with scaling factors from DERIVATIVE block
        vars(m) = vars(m) + (1.0 - exp(-param.dt / (vars(tm) * 0.4))) * (vars(minf) - vars(m));
        vars(hf) = vars(hf) + (1.0 - exp(-param.dt / (vars(thf) * 0.08))) * (vars(hfinf) - vars(hf));
        vars(hs) = vars(hs) + (1.0 - exp(-param.dt / (vars(ths) * 0.3))) * (vars(hsinf) - vars(hs));
    }
    
private:
    // Helper function to calculate rates and steady-state values
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        // Parameters from mod file
        double vhm = 11.2, ka = 14.1;
        double vhh = -33.1, ki = 8.3;
        double atm = 13.8, btm = -17.5, ctm = 12.9, dtm = -3.7, etm = 6.5, ftm = 1.9;
        double athf = 539.2, bthf = -28.2, cthf = 4.9, dthf = 27.3;
        double aths = 8422.0, bths = -37.7, cths = 6.4, dths = 118.9;
        
        // Calculate steady-state values and time constants
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hfinf) = 1.0 / (1.0 + exp((v - vhh) / ki));
        vars(hsinf) = 1.0 / (1.0 + exp((v - vhh) / ki));
        vars(tm) = atm / (exp(-(v - btm) / ctm) + exp((v - dtm) / etm)) + ftm;
        vars(thf) = athf / (1.0 + exp((v - bthf) / cthf)) + dthf;
        vars(ths) = aths / (1.0 + exp((v - bths) / cths)) + dths;
    }
};

REGISTER_MECHANISM("shl1", SHL1_Channel);

} // namespace shl1_worm