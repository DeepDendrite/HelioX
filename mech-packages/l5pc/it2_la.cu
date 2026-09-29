// it2_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace it2_la {

// Mechanism class name and registration name
#define MECH_CLASS_NAME IT2_LA
static const char *MECH_NAME_TO_REG = "it2_la";

// MechTrait structure defining variables
struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Computed variables
        gca, ica,
        
        // Parameters
        gcabar,
        
        // Activation and inactivation variables
        minf, hinf, mtau, htau, mexp, hexp,
        
        // Reversal potential
        eca,
        
        // Internal variable for NEURON compatibility
        _g
    };
    
    enum class GlobalVarNames {
        // Global parameters
        celsius,
        vshift, vmin, vmax,
        v12m, v12h, vwm, vwh,
        am, ah,
        vm1, vm2, vh1, vh2,
        wm1, wm2, wh1, wh2,
        cao, cai
    };
    
    enum class IonVarNames {
        _ion_eca, _ion_ica, _ion_dicadv
    };
};

// Main mechanism class
class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // Define flags: this mechanism requires init, current, and state functions
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    // Use enum values directly
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    // Constructor for initialization
    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values
        init_values.insert({gcabar, 0.0008}); // Default value from .mod file
        
        // Register variables with their columnindex values from CPP file
        var_in_coredata_idx.insert({gcabar, 0});
        var_in_coredata_idx.insert({gca, 1});
        var_in_coredata_idx.insert({minf, 2});
        var_in_coredata_idx.insert({hinf, 3});
        var_in_coredata_idx.insert({mexp, 4});
        var_in_coredata_idx.insert({hexp, 5});
        var_in_coredata_idx.insert({mtau, 6});
        var_in_coredata_idx.insert({htau, 7});
        var_in_coredata_idx.insert({m, 8});
        var_in_coredata_idx.insert({h, 9});
        var_in_coredata_idx.insert({ica, 10});
        var_in_coredata_idx.insert({eca, 11});
        var_in_coredata_idx.insert({_g, 15});
        
        // Register global variables with the exact names from NEURON
        global_info_map.insert({celsius, {"celsius"}});
        global_info_map.insert({vshift, {"vshift_it2_la"}});
        global_info_map.insert({vmin, {"vmin_it2_la"}});
        global_info_map.insert({vmax, {"vmax_it2_la"}});
        global_info_map.insert({v12m, {"v12m_it2_la"}});
        global_info_map.insert({v12h, {"v12h_it2_la"}});
        global_info_map.insert({vwm, {"vwm_it2_la"}});
        global_info_map.insert({vwh, {"vwh_it2_la"}});
        global_info_map.insert({am, {"am_it2_la"}});
        global_info_map.insert({ah, {"ah_it2_la"}});
        global_info_map.insert({vm1, {"vm1_it2_la"}});
        global_info_map.insert({vm2, {"vm2_it2_la"}});
        global_info_map.insert({vh1, {"vh1_it2_la"}});
        global_info_map.insert({vh2, {"vh2_it2_la"}});
        global_info_map.insert({wm1, {"wm1_it2_la"}});
        global_info_map.insert({wm2, {"wm2_it2_la"}});
        global_info_map.insert({wh1, {"wh1_it2_la"}});
        global_info_map.insert({wh2, {"wh2_it2_la"}});
        global_info_map.insert({cao, {"cao_it2_la"}});
        global_info_map.insert({cai, {"cai_it2_la"}});
        
        // Register ion variables
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dicadv, {"ca_ion", EionVarNames::dcurdv}});
        
        // Verify name matches expected
        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }
    
    // Calculate rates function (from rates in the original mod file)
    DUAL_EXEC void rates(double v_, VarAccessor<MechTrait> &vars) {
        // Calculate activation (m) and inactivation (h) steady-state values
        vars(minf) = 1.0 / (1.0 + exp(-(v_ + vars(v12m)) / vars(vwm)));
        vars(hinf) = 1.0 / (1.0 + exp((v_ + vars(v12h)) / vars(vwh)));
        
        // Calculate time constants for activation and inactivation
        vars(mtau) = (vars(am) + 1.0 / (exp((v_ + vars(vm1)) / vars(wm1)) + exp(-(v_ + vars(vm2)) / vars(wm2))));
        vars(htau) = (vars(ah) + 1.0 / (exp((v_ + vars(vh1)) / vars(wh1)) + exp(-(v_ + vars(vh2)) / vars(wh2))));
    }
    
    // Calculate trates function (similar to trates in the original mod file)
    DUAL_EXEC void trates(double v, double dt, VarAccessor<MechTrait> &vars) {
        // Call rates to update minf, hinf, mtau, htau
        rates(v, vars);
        
        // Calculate mexp and hexp based on time constants
        double tinc = -dt;
        vars(mexp) = 1.0 - exp(tinc / vars(mtau));
        vars(hexp) = 1.0 - exp(tinc / vars(htau));
    }
    
    // Initialize variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Get the reversal potential from the ion
        vars(eca) = vars(_ion_eca);
        
        // Calculate rates with voltage shift
        double v_shifted = param.volt + vars(vshift);
        trates(v_shifted, param.dt, vars);
        
        // Initialize state variables to steady-state values
        vars(m) = vars(minf);
        vars(h) = vars(hinf);
    }
    
    // Calculate current
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Update eca from ion
        vars(eca) = vars(_ion_eca);
        
        // Calculate conductance and current
        vars(gca) = vars(gcabar) * vars(m) * vars(m) * vars(h);
        double current = vars(gca) * (param.volt - vars(eca));
        vars(ica) = current;
        
        // Update ion current if required
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
        }
        
        return current;
    }
    
    // Update state variables
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Calculate updated rates with voltage shift
        double v_shifted = param.volt + vars(vshift);
        trates(v_shifted, param.dt, vars);
        
        // Update state variables using the expressions from states() in the mod file
        vars(m) = vars(m) + vars(mexp) * (vars(minf) - vars(m));
        vars(h) = vars(h) + vars(hexp) * (vars(hinf) - vars(h));
    }
};

// Register the mechanism with DeepDendrite
REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// Clean up macro definitions
#undef MECH_CLASS_NAME

} // namespace it2_la
