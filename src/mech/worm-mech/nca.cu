// nca mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace nca_worm {

struct MechTrait {
    enum class VarNames {
        // Parameters
        gbnca,
        
        // Ion reversal potential and current
        ena, ina
    };
    
    enum class IonVarNames {
        _ion_ena, _ion_ina
    };
};

class NCA_Channel : public MechTemp<NCA_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_CURRENT;
    
    NCA_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbnca, 1.0});
        
        var_in_coredata_idx.insert({gbnca, 0});
        var_in_coredata_idx.insert({ena, 1});
        var_in_coredata_idx.insert({ina, 2});
        
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        
        // Passive sodium current with fixed ena=30mV
        vars(ina) = vars(gbnca) * (param.volt - 30.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
        }
        
        return vars(ina);
    }
};

REGISTER_MECHANISM("nca", NCA_Channel);

} // namespace nca_worm