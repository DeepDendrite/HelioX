// kqt3 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace kqt3_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        mf, ms, w, s,
        
        // Parameters
        gbkqt3,
        
        // Assigned variables
        minf, tmf, tms, winf, sinf, tw, ts,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class KQT3_Channel : public MechTemp<KQT3_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    KQT3_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gbkqt3, 1.0});  // 1 nS/um2
        
        // Register variable indices (based on NEURON CPP field order)
        var_in_coredata_idx.insert({gbkqt3, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({tmf, 2});
        var_in_coredata_idx.insert({tms, 3});
        var_in_coredata_idx.insert({winf, 4});
        var_in_coredata_idx.insert({sinf, 5});
        var_in_coredata_idx.insert({tw, 6});
        var_in_coredata_idx.insert({ts, 7});
        var_in_coredata_idx.insert({mf, 8});
        var_in_coredata_idx.insert({ms, 9});
        var_in_coredata_idx.insert({w, 10});
        var_in_coredata_idx.insert({s, 11});
        var_in_coredata_idx.insert({ek, 12});
        var_in_coredata_idx.insert({ik, 13});
        
        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    // Initialize state variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(mf) = 0.0;  // As specified in INITIAL block
        vars(ms) = 0.0;  // As specified in INITIAL block
        vars(w) = vars(winf);
        vars(s) = vars(sinf);
    }
    
    // Current calculation
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);
        
        // Calculate potassium current
        vars(ik) = vars(gbkqt3) * (0.3 * vars(mf) + 0.7 * vars(ms)) * vars(w) * vars(s) * (param.volt - vars(ek));
        
        // Update global ion current (only when updateIon=true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    // State variable updates
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        // Update state variables using exponential integration
        vars(mf) = vars(mf) + (1.0 - exp(-param.dt / vars(tmf))) * (vars(minf) - vars(mf));
        vars(ms) = vars(ms) + (1.0 - exp(-param.dt / vars(tms))) * (vars(minf) - vars(ms));
        vars(w) = vars(w) + (1.0 - exp(-param.dt / vars(tw))) * (vars(winf) - vars(w));
        vars(s) = vars(s) + (1.0 - exp(-param.dt / vars(ts))) * (vars(sinf) - vars(s));
    }
    
private:
    // Helper function to calculate rates and steady-state values
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        // Parameters from mod file
        double vhm = -12.6726, ka = 15.8008;
        double vhw = -1.084, kiw = 28.78, aw = 0.49, bw = 0.51;
        double vhs = -45.3, kis = 12.3, as = 0.34, bs = 0.66;
        double atmf = 395.3, btmf = 38.1, ctmf = 33.59;
        double atms = 5503.0, btms = -5345.4, ctms = 0.02827, dtms = -23.9;
        double etms = -4590.6, ftms = 0.0357, gtms = 14.15;
        double atw = 0.544, btw = 29.2, ctw = -48.09, dtw = 48.83;
        double ats = 500e3;
        
        // Calculate steady-state values and time constants
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(tmf) = atmf / (1.0 + pow((v + btmf) / ctmf, 2));
        vars(tms) = atms + btms / (1.0 + pow(10.0, -ctms * (dtms - v))) + etms / (1.0 + pow(10.0, -ftms * (gtms + v)));
        vars(winf) = aw + bw / (1.0 + exp((v - vhw) / kiw));
        vars(sinf) = as + bs / (1.0 + exp((v - vhs) / kis));
        vars(tw) = atw + btw / (1.0 + pow((v - ctw) / dtw, 2));
        vars(ts) = ats;
    }
};

REGISTER_MECHANISM("kqt3", KQT3_Channel);

} // namespace kqt3_worm