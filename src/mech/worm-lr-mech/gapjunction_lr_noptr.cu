// gapjunction_lr_noptr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace gapjunction_lr_noptr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        g, w,
        // ASSIGNED variables  
        i, vpre, pure_i, didv, didvpre
    };
};

class GapjunctionLrNoptr : public MechTemp<GapjunctionLrNoptr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    
    // Point process mechanism with current calculation
    constexpr static MechFlags flags = ENABLE_CURRENT | POINT_PROCESS;
    
    GapjunctionLrNoptr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({g, 1.0});  // 1. (uS)
        init_values.insert({w, 0.0});  // 0.
        
        // Register variable indices based on CPP file data layout
        // From setup_instance: inst->g = ml->data+0*pnodecount; etc.
        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({w, 1});  
        var_in_coredata_idx.insert({i, 2});
        var_in_coredata_idx.insert({vpre, 3});
        var_in_coredata_idx.insert({pure_i, 4});
        var_in_coredata_idx.insert({didv, 5});
        var_in_coredata_idx.insert({didvpre, 6});
    }
    
    // Current calculation function - corresponds to BREAKPOINT block
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // From nrn_current_gapjunction_lr_noptr function:
        vars(pure_i) = vars(g) * (param.volt - vars(vpre));
        vars(i) = vars(w) * vars(pure_i);
        vars(didv) = -vars(w) * vars(g);
        vars(didvpre) = vars(w) * vars(g);
        
        return vars(i);
    }
};

REGISTER_MECHANISM("gapjunction_lr_noptr", GapjunctionLrNoptr);

} // namespace gapjunction_lr_noptr