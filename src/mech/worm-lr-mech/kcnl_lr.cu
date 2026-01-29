// kcnl_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace kcnl_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbkcnl, w, dv,
        // ASSIGNED variables
        i, ik_di, i_di, pure_i, didv, cai, minf, tm, ik_tmp,
        // STATE variables  
        m, m_dm,
        // Ion variables (local copies)
        ek, ik
    };
    
    enum class GlobalVarNames {
        kca, a, ek_tmp
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_dikdv, _ion_cai
    };
};

class KcnlLr : public MechTemp<KcnlLr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    KcnlLr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbkcnl, 1.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices based on CPP setup_instance
        var_in_coredata_idx.insert({gbkcnl, 0});
        var_in_coredata_idx.insert({w, 1});
        var_in_coredata_idx.insert({dv, 2});
        var_in_coredata_idx.insert({i, 3});
        var_in_coredata_idx.insert({ik_di, 4});
        var_in_coredata_idx.insert({i_di, 5});
        var_in_coredata_idx.insert({pure_i, 6});
        var_in_coredata_idx.insert({didv, 7});
        var_in_coredata_idx.insert({cai, 8});
        var_in_coredata_idx.insert({minf, 9});
        var_in_coredata_idx.insert({tm, 10});
        var_in_coredata_idx.insert({m, 11});
        var_in_coredata_idx.insert({m_dm, 12});
        var_in_coredata_idx.insert({ek, 13});
        var_in_coredata_idx.insert({ik, 14});
        var_in_coredata_idx.insert({ik_tmp, 15});
        
        // Register global variables
        global_info_map.insert({kca, {"kca_kcnl_lr"}});
        global_info_map.insert({a, {"a_kcnl_lr"}});
        global_info_map.insert({ek_tmp, {"ek_tmp_kcnl_lr"}});
        
        // Register ion variables (k and ca)
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        vars(minf) = vars(cai) / (vars(kca) + vars(cai));
        vars(tm) = vars(a);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(m) = 0.0;
        vars(m_dm) = 0.0;
        
        // Read calcium concentration
        vars(cai) = vars(_ion_cai);
        
        // Execute INITIAL block logic
        setparames(vars, param.volt);
        vars(m) = vars(minf);
        vars(m_dm) = vars(minf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block
        vars(ik_di) = vars(gbkcnl) * vars(m_dm) * (param.volt + vars(dv) - vars(ek_tmp));
        vars(i_di) = vars(w) * vars(ik_di);
        vars(ik_tmp) = vars(gbkcnl) * vars(m) * (param.volt - vars(ek_tmp));
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
        // Read ion values
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);
        
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

REGISTER_MECHANISM("kcnl_lr", KcnlLr);

} // namespace kcnl_lr
