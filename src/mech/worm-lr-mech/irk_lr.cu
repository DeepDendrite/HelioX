// irk_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace irk_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbirk, ek_tmp, w, dv,
        // ASSIGNED variables
        i, ik_tmp, minf, tm, ik_di, i_di, pure_i, didv,
        // STATE variables  
        m, m_dm,
        // Ion variables (local copies)
        ek, ik
    };
    
    enum class GlobalVarNames {
        vhm, ka, atm, btm, ctm, dtm, etm, ftm
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_dikdv
    };
};

class IrkLr : public MechTemp<IrkLr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    IrkLr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbirk, 1.0});
        init_values.insert({ek_tmp, -80.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices (following pattern of other lr mechanisms)
        var_in_coredata_idx.insert({gbirk, 0});
        var_in_coredata_idx.insert({ek_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({dv, 3});
        var_in_coredata_idx.insert({i, 4});
        var_in_coredata_idx.insert({ik_tmp, 5});
        var_in_coredata_idx.insert({minf, 6});
        var_in_coredata_idx.insert({tm, 7});
        var_in_coredata_idx.insert({ik_di, 8});
        var_in_coredata_idx.insert({i_di, 9});
        var_in_coredata_idx.insert({pure_i, 10});
        var_in_coredata_idx.insert({didv, 11});
        var_in_coredata_idx.insert({m, 12});
        var_in_coredata_idx.insert({m_dm, 13});
        var_in_coredata_idx.insert({ek, 14});
        var_in_coredata_idx.insert({ik, 15});
        
        // Register global variables
        global_info_map.insert({vhm, {"vhm_irk_lr"}});
        global_info_map.insert({ka, {"ka_irk_lr"}});
        global_info_map.insert({atm, {"atm_irk_lr"}});
        global_info_map.insert({btm, {"btm_irk_lr"}});
        global_info_map.insert({ctm, {"ctm_irk_lr"}});
        global_info_map.insert({dtm, {"dtm_irk_lr"}});
        global_info_map.insert({etm, {"etm_irk_lr"}});
        global_info_map.insert({ftm, {"ftm_irk_lr"}});
        
        // Register ion variables (potassium)
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        vars(minf) = 1.0 / (1.0 + exp((v - vars(vhm)) / vars(ka)));
        vars(tm) = vars(atm) / (exp(-(v - vars(btm)) / vars(ctm)) + exp((v - vars(dtm)) / vars(etm))) + vars(ftm);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(m) = 0.0;
        vars(m_dm) = 0.0;
        
        // Execute INITIAL block logic
        setparames(vars, param.volt);
        vars(m_dm) = vars(minf);
        vars(m) = vars(minf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block
        vars(ik_di) = vars(gbirk) * vars(m_dm) * (param.volt + vars(dv) - vars(ek_tmp));
        vars(i_di) = vars(w) * vars(ik_di);
        vars(ik_tmp) = vars(gbirk) * vars(m) * (param.volt - vars(ek_tmp));
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
        
        // Save current state value
        vars(m_dm) = vars(m);
        
        // First setparames call with v + dv
        setparames(vars, param.volt + vars(dv));
        // Update m_dm using cnexp integration
        vars(m_dm) = vars(m_dm) + (1.0 - exp(param.dt * ((-1.0) / vars(tm)))) * 
                     (-(vars(minf) / vars(tm)) / ((-1.0) / vars(tm)) - vars(m_dm));
        
        // Second setparames call with v
        setparames(vars, param.volt);
        // Update m using cnexp integration
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(tm)))) * 
                  (-(vars(minf) / vars(tm)) / ((-1.0) / vars(tm)) - vars(m));
    }
};

REGISTER_MECHANISM("irk_lr", IrkLr);

} // namespace irk_lr
