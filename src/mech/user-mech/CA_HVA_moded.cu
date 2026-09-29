// CA_HVA_moded mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace Ca_HVA_moded {

// Mechanism class name and registration name
#define MECH_CLASS_NAME Ca_HVA_moded
static const char *MECH_NAME_TO_REG = "Ca_HVA_moded";

// MechTrait structure defining variables
struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Computed variables
        g, ica,
        
        // Parameters
        gbar, // This is an array parameter in NEURON
        
        // Activation and inactivation variables
        mInf, mTau, mAlpha, mBeta,
        hInf, hTau, hAlpha, hBeta,
        
        // Reversal potential
        eca,
        
        // Internal variable for NEURON compatibility
        _g
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
    using enum MechTrait::IonVarNames;
    
    // Constructor for initialization
    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // Register variables with their columnindex values from CPP file
        var_in_coredata_idx.insert({gbar, {0, 2}}); // This is an array with 2 elements
        var_in_coredata_idx.insert({ica, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({eca, 5});
        var_in_coredata_idx.insert({mInf, 6});
        var_in_coredata_idx.insert({mTau, 7});
        var_in_coredata_idx.insert({mAlpha, 8});
        var_in_coredata_idx.insert({mBeta, 9});
        var_in_coredata_idx.insert({hInf, 10});
        var_in_coredata_idx.insert({hTau, 11});
        var_in_coredata_idx.insert({hAlpha, 12});
        var_in_coredata_idx.insert({hBeta, 13});
        var_in_coredata_idx.insert({_g, 17});
        
        // Register ion variables
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dicadv, {"ca_ion", EionVarNames::dcurdv}});
        
        // Verify name matches expected
        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }
    
    // Utility function for handling division by zero in rate equations
    DUAL_EXEC double vtrap(double x, double y) {
        if (fabs(x / y) < 1e-6) {
            return y * (1.0 - x / y / 2.0);
        } else {
            return x / (exp(x / y) - 1.0);
        }
    }
    
    // Calculate rates function (from rates in the original mod file)
    DUAL_EXEC void rates(VarAccessor<MechTrait> &vars, double v) {
        // Calculate activation (m) parameters
        vars(mAlpha) = 0.055 * vtrap(-27.0 - v, 3.8);
        vars(mBeta) = 0.94 * exp((-75.0 - v) / 17.0);
        vars(mInf) = vars(mAlpha) / (vars(mAlpha) + vars(mBeta));
        vars(mTau) = 1.0 / (vars(mAlpha) + vars(mBeta));
        
        // Calculate inactivation (h) parameters
        vars(hAlpha) = 0.000457 * exp((-13.0 - v) / 50.0);
        vars(hBeta) = 0.0065 / (exp((-v - 15.0) / 28.0) + 1.0);
        vars(hInf) = vars(hAlpha) / (vars(hAlpha) + vars(hBeta));
        vars(hTau) = 1.0 / (vars(hAlpha) + vars(hBeta));
    }
    
    // Initialize variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Get the reversal potential from the ion
        vars(eca) = vars(_ion_eca);
        
        // Calculate rates
        rates(vars, param.volt);
        
        // Initialize state variables to steady-state values
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
    }
    
    // Calculate current
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Update eca from ion
        vars(eca) = vars(_ion_eca);
        
        // Calculate conductance and current as per BREAKPOINT in the mod file
        vars(g) = vars.Arr(gbar, 0) * vars(m) * vars(m) * vars(h);
        vars(g) = vars.Arr(gbar, 1) + vars(g);
        double current = vars(g) * (param.volt - vars(eca));
        vars(ica) = current;
        
        // Update ion current if required
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
        }
        
        return current;
    }
    
    // Update state variables
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Calculate rates
        rates(vars, param.volt);
        
        // Update state variables using the cnexp method (as specified in the mod file)
        double dt = param.dt;
        
        // Implementation of the cnexp method for m and h
        vars(m) = vars(m) + (1.0 - exp(dt * (-1.0 / vars(mTau)))) * 
                  ((vars(mInf) / vars(mTau)) / (1.0 / vars(mTau)) - vars(m));
        
        vars(h) = vars(h) + (1.0 - exp(dt * (-1.0 / vars(hTau)))) * 
                  ((vars(hInf) / vars(hTau)) / (1.0 / vars(hTau)) - vars(h));
    }
};

// Register the mechanism with DeepDendrite
REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// Clean up macro definitions
#undef MECH_CLASS_NAME

} // namespace Ca_HVA_moded