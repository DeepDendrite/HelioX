// gapjunction_noptr mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace gapjunction_noptr {

struct MechTrait {
    enum class VarNames {
        // Parameters
        weight,
        // Assigned variables
        vpre, i
    };
};

class GapJunctionNoPtr : public MechTemp<GapJunctionNoPtr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    
    constexpr static MechFlags flags = ENABLE_CURRENT | POINT_PROCESS;
    
    GapJunctionNoPtr(MechInitParams &param) : MechTemp(param) {
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

REGISTER_MECHANISM("gapjunction_noptr", GapJunctionNoPtr);

} // namespace gapjunction_noptr