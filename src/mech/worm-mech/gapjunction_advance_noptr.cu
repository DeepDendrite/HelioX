// gapjunction_advance_noptr mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace gapjunction_advance_noptr {

struct MechTrait {
    enum class VarNames {
        // Parameters
        weight,
        // Assigned variables
        vpre, i
    };
};

class GapJunctionAdvanceNoPtr : public MechTemp<GapJunctionAdvanceNoPtr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    
    constexpr static MechFlags flags = ENABLE_CURRENT | POINT_PROCESS;
    
    GapJunctionAdvanceNoPtr(MechInitParams &param) : MechTemp(param) {
        // Set default values
        init_values.insert({weight, 0.0});
        
        // Register variable indices (按NEURON CPP中的顺序)
        var_in_coredata_idx.insert({weight, 0});
        var_in_coredata_idx.insert({vpre, 1});
        var_in_coredata_idx.insert({i, 2});
    }
    
    // 电流计算函数
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(i) = vars(weight) * (param.volt - vars(vpre));
        return vars(i);
    }
};

REGISTER_MECHANISM("gapjunction_advance_noptr", GapJunctionAdvanceNoPtr);

} // namespace gapjunction_advance_noptr