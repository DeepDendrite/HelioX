// cca1 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace cca1_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Parameters
        gbcca1,
        
        // Assigned variables
        minf, hinf, tm, th,
        
        // Ion reversal potential and current
        eca, ica
    };
    
    enum class IonVarNames {
        _ion_eca, _ion_ica
    };
};

class CCA1_Channel : public MechTemp<CCA1_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    CCA1_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gbcca1, 1.0});  // 1 nS/um2
        
        // Register variable indices (based on NEURON CPP field order)
        var_in_coredata_idx.insert({gbcca1, 0});
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
        
        // Calculate calcium current - using performance optimization for repeated vars(m) access
        double m_val = vars(m);
        double h_val = vars(h);
        vars(ica) = vars(gbcca1) * m_val * m_val * h_val * (param.volt - 60.0);
        
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
        double dt = param.dt;
        double tm_val = vars(tm);
        double th_val = vars(th);
        
        vars(m) = vars(m) + (1.0 - exp(-dt / tm_val)) * (vars(minf) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-dt / th_val)) * (vars(hinf) - vars(h));
    }
    
private:
    // Helper function to calculate rates and steady-state values
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        // Parameters from mod file
        double vhm = -43.32, ka = 7.6;
        double vhh = -58.0, ki = 7.0;
        double atm = 40.0, btm = -62.5, ctm = -12.6, dtm = 0.7;
        double ath = 280.0, bth = -60.7, cth = 8.5, dth = 19.8;
        
        // Calculate steady-state values and time constants
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hinf) = 1.0 / (1.0 + exp((v - vhh) / ki));
        vars(tm) = atm / (1.0 + exp(-(v - btm) / ctm)) + dtm;
        vars(th) = ath / (1.0 + exp((v - bth) / cth)) + dth;
    }
};

REGISTER_MECHANISM("cca1", CCA1_Channel);

} // namespace cca1_worm