// egl36_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace egl36_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbegl36, ek_tmp, w, dv,
        // ASSIGNED variables
        i, ik_tmp, mfinf, mminf, msinf, tmf, tmm, tms, ik_di, i_di, pure_i, didv,
        // STATE variables  
        mf, mm, ms, mf_dmf, mm_dmm, ms_dms,
        // Ion variables (local copies)
        ek, ik
    };
    
    enum class GlobalVarNames {
        vhm, ka, atms, atmm, atmf
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_dikdv
    };
};

class Egl36Lr : public MechTemp<Egl36Lr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    Egl36Lr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbegl36, 1.0});
        init_values.insert({ek_tmp, -80.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices
        var_in_coredata_idx.insert({gbegl36, 0});
        var_in_coredata_idx.insert({ek_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({dv, 3});
        var_in_coredata_idx.insert({i, 4});
        var_in_coredata_idx.insert({ik_tmp, 5});
        var_in_coredata_idx.insert({mfinf, 6});
        var_in_coredata_idx.insert({mminf, 7});
        var_in_coredata_idx.insert({msinf, 8});
        var_in_coredata_idx.insert({tmf, 9});
        var_in_coredata_idx.insert({tmm, 10});
        var_in_coredata_idx.insert({tms, 11});
        var_in_coredata_idx.insert({ik_di, 12});
        var_in_coredata_idx.insert({i_di, 13});
        var_in_coredata_idx.insert({pure_i, 14});
        var_in_coredata_idx.insert({didv, 15});
        var_in_coredata_idx.insert({mf, 16});
        var_in_coredata_idx.insert({mm, 17});
        var_in_coredata_idx.insert({ms, 18});
        var_in_coredata_idx.insert({mf_dmf, 19});
        var_in_coredata_idx.insert({mm_dmm, 20});
        var_in_coredata_idx.insert({ms_dms, 21});
        var_in_coredata_idx.insert({ek, 22});
        var_in_coredata_idx.insert({ik, 23});
        
        // Register global variables
        global_info_map.insert({vhm, {"vhm_egl36_lr"}});
        global_info_map.insert({ka, {"ka_egl36_lr"}});
        global_info_map.insert({atms, {"atms_egl36_lr"}});
        global_info_map.insert({atmm, {"atmm_egl36_lr"}});
        global_info_map.insert({atmf, {"atmf_egl36_lr"}});
        
        // Register ion variables (potassium)
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        // All three inf values are the same
        vars(mfinf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        vars(mminf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        vars(msinf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        // Time constants are constant
        vars(tmf) = vars(atmf);
        vars(tmm) = vars(atmm);
        vars(tms) = vars(atms);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(mf) = 0.0;
        vars(mm) = 0.0;
        vars(ms) = 0.0;
        vars(mf_dmf) = 0.0;
        vars(mm_dmm) = 0.0;
        vars(ms_dms) = 0.0;
        
        // Execute INITIAL block logic
        setparames(vars, param.volt);
        vars(mf_dmf) = vars(mfinf);
        vars(mm_dmm) = vars(mminf);
        vars(ms_dms) = vars(msinf);
        vars(mf) = vars(mfinf);
        vars(mm) = vars(mminf);
        vars(ms) = vars(msinf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block with weighted sum
        vars(ik_di) = vars(gbegl36) * (0.33 * vars(mf_dmf) + 0.36 * vars(mm_dmm) + 0.39 * vars(ms_dms)) * 
                      (param.volt + vars(dv) - vars(ek_tmp));
        vars(i_di) = vars(w) * vars(ik_di);
        vars(ik_tmp) = vars(gbegl36) * (0.33 * vars(mf) + 0.36 * vars(mm) + 0.39 * vars(ms)) * 
                       (param.volt - vars(ek_tmp));
        vars(pure_i) = vars(ik_tmp);
        vars(i) = vars(w) * vars(pure_i);
        vars(ik) = vars(i);
        vars(didv) = -(vars(i_di) - vars(i)) / vars(dv);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
            mechAtomAdd(&vars(_ion_dikdv), vars(didv));
        }
        
        return vars(ik);
    }

    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);
        
        // Save current state values
        vars(mf_dmf) = vars(mf);
        vars(mm_dmm) = vars(mm);
        vars(ms_dms) = vars(ms);
        
        // First setparames call with v + dv
        setparames(vars, param.volt + vars(dv));
        // Update _d variables using cnexp integration
        vars(mf_dmf) = vars(mf_dmf) + (1.0 - exp(param.dt * ((-1.0) / vars(tmf)))) * 
                       (-(vars(mfinf) / vars(tmf)) / ((-1.0) / vars(tmf)) - vars(mf_dmf));
        vars(mm_dmm) = vars(mm_dmm) + (1.0 - exp(param.dt * ((-1.0) / vars(tmm)))) * 
                       (-(vars(mminf) / vars(tmm)) / ((-1.0) / vars(tmm)) - vars(mm_dmm));
        vars(ms_dms) = vars(ms_dms) + (1.0 - exp(param.dt * ((-1.0) / vars(tms)))) * 
                       (-(vars(msinf) / vars(tms)) / ((-1.0) / vars(tms)) - vars(ms_dms));
        
        // Second setparames call with v
        setparames(vars, param.volt);
        // Update main variables using cnexp integration
        vars(mf) = vars(mf) + (1.0 - exp(param.dt * ((-1.0) / vars(tmf)))) * 
                   (-(vars(mfinf) / vars(tmf)) / ((-1.0) / vars(tmf)) - vars(mf));
        vars(mm) = vars(mm) + (1.0 - exp(param.dt * ((-1.0) / vars(tmm)))) * 
                   (-(vars(mminf) / vars(tmm)) / ((-1.0) / vars(tmm)) - vars(mm));
        vars(ms) = vars(ms) + (1.0 - exp(param.dt * ((-1.0) / vars(tms)))) * 
                   (-(vars(msinf) / vars(tms)) / ((-1.0) / vars(tms)) - vars(ms));
    }
};

REGISTER_MECHANISM("egl36_lr", Egl36Lr);

} // namespace egl36_lr
