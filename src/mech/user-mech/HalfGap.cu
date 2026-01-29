// HalfGap mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace HalfGap {

// Mechanism class name and registration name
#define MECH_CLASS_NAME HalfGap
static const char *MECH_NAME_TO_REG = "HalfGap";

// MechTrait structure defining variables
struct MechTrait {
    enum class VarNames {
        // Parameters
        g,
        
        // Assigned variables
        vgap, i,
        
        // Internal variable for NEURON compatibility
        _g
    };
};

// Main mechanism class
class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // Define flags: this mechanism is a point process with electrode current that only needs current calculation
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT;
    
    // Use enum values directly
    using enum MechTrait::VarNames;
    
    // Constructor for initialization
    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // Set need_area to true for point process
        need_area = true;
        
        // Set default parameter values
        init_values.insert({g, 0.0}); // Default value from .mod file
        
        // Register variables with their columnindex values from CPP file
        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({vgap, 1});
        var_in_coredata_idx.insert({i, 2});
        var_in_coredata_idx.insert({_g, 4});
        
        // Verify name matches expected
        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }
    
    // Initialize variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize i to zero
        vars(i) = 0.0;
    }
    
    // Calculate current
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Calculate gap junction current: i = (vgap - v) * g
        double current = (vars(vgap) - param.volt) * vars(g);
        vars(i) = current;
        
        // Return current - for electrode currents, this value will be added to the RHS
        return current;
    }
};

// Register the mechanism with HelioX
REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// Clean up macro definitions
#undef MECH_CLASS_NAME

} // namespace HalfGap