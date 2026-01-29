// nca_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace nca_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbnca, ena_tmp, w,
        // ASSIGNED variables
        i, ina_tmp, pure_i, didv,
        // Ion variables (local copies)
        ena, ina
    };
    
    enum class IonVarNames {
        _ion_ena, _ion_ina, _ion_dinadv
    };
};

class NcaLr : public MechTemp<NcaLr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_CURRENT;
    
    NcaLr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbnca, 1.0});
        init_values.insert({ena_tmp, 30.0});
        init_values.insert({w, 1.0});
        
        // Register variable indices based on pattern
        var_in_coredata_idx.insert({gbnca, 0});
        var_in_coredata_idx.insert({ena_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({i, 3});
        var_in_coredata_idx.insert({ina_tmp, 4});
        var_in_coredata_idx.insert({pure_i, 5});
        var_in_coredata_idx.insert({didv, 6});
        var_in_coredata_idx.insert({ena, 7});
        var_in_coredata_idx.insert({ina, 8});
        
        // Register ion variables (sodium)
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dinadv, {"na_ion", EionVarNames::dcurdv}});
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);
        
        // Following BREAKPOINT block
        vars(ina_tmp) = vars(gbnca) * (param.volt - vars(ena_tmp));
        vars(pure_i) = vars(ina_tmp);
        vars(i) = vars(w) * vars(pure_i);
        vars(ina) = vars(i);
        vars(didv) = -vars(w) * vars(gbnca);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
            mechAtomAdd(&vars(_ion_dinadv), vars(didv));
        }
        
        return vars(ina);
    }

};

REGISTER_MECHANISM("nca_lr", NcaLr);

} // namespace nca_lr
