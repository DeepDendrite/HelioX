// shl1_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace shl1_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbshl1, ek_tmp, w, dv,
        // ASSIGNED variables
        i, ik_tmp, minf, hfinf, hsinf, tm, thf, ths, ik_di, i_di, pure_i, didv,
        // STATE variables  
        m, hf, hs, m_dm, hf_dhf, hs_dhs,
        // Ion variables (local copies)
        ek, ik
    };
    
    enum class GlobalVarNames {
        vhm, ka, vhh, ki, atm, btm, ctm, dtm, etm, ftm, athf, bthf, cthf, dthf, aths, bths, cths, dths
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_dikdv
    };
};

class Shl1Lr : public MechTemp<Shl1Lr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    Shl1Lr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbshl1, 1.0});
        init_values.insert({ek_tmp, -80.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices
        var_in_coredata_idx.insert({gbshl1, 0});
        var_in_coredata_idx.insert({ek_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({dv, 3});
        var_in_coredata_idx.insert({i, 4});
        var_in_coredata_idx.insert({ik_tmp, 5});
        var_in_coredata_idx.insert({minf, 6});
        var_in_coredata_idx.insert({hfinf, 7});
        var_in_coredata_idx.insert({hsinf, 8});
        var_in_coredata_idx.insert({tm, 9});
        var_in_coredata_idx.insert({thf, 10});
        var_in_coredata_idx.insert({ths, 11});
        var_in_coredata_idx.insert({ik_di, 12});
        var_in_coredata_idx.insert({i_di, 13});
        var_in_coredata_idx.insert({pure_i, 14});
        var_in_coredata_idx.insert({didv, 15});
        var_in_coredata_idx.insert({m, 16});
        var_in_coredata_idx.insert({hf, 17});
        var_in_coredata_idx.insert({hs, 18});
        var_in_coredata_idx.insert({m_dm, 19});
        var_in_coredata_idx.insert({hf_dhf, 20});
        var_in_coredata_idx.insert({hs_dhs, 21});
        var_in_coredata_idx.insert({ek, 22});
        var_in_coredata_idx.insert({ik, 23});
        
        // Register global variables
        global_info_map.insert({vhm, {"vhm_shl1_lr"}});
        global_info_map.insert({ka, {"ka_shl1_lr"}});
        global_info_map.insert({vhh, {"vhh_shl1_lr"}});
        global_info_map.insert({ki, {"ki_shl1_lr"}});
        global_info_map.insert({atm, {"atm_shl1_lr"}});
        global_info_map.insert({btm, {"btm_shl1_lr"}});
        global_info_map.insert({ctm, {"ctm_shl1_lr"}});
        global_info_map.insert({dtm, {"dtm_shl1_lr"}});
        global_info_map.insert({etm, {"etm_shl1_lr"}});
        global_info_map.insert({ftm, {"ftm_shl1_lr"}});
        global_info_map.insert({athf, {"athf_shl1_lr"}});
        global_info_map.insert({bthf, {"bthf_shl1_lr"}});
        global_info_map.insert({cthf, {"cthf_shl1_lr"}});
        global_info_map.insert({dthf, {"dthf_shl1_lr"}});
        global_info_map.insert({aths, {"aths_shl1_lr"}});
        global_info_map.insert({bths, {"bths_shl1_lr"}});
        global_info_map.insert({cths, {"cths_shl1_lr"}});
        global_info_map.insert({dths, {"dths_shl1_lr"}});
        
        // Register ion variables (potassium)
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        vars(minf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        vars(hfinf) = 1.0 / (1.0 + exp((v - vars(vhh)) / vars(ki)));
        vars(hsinf) = 1.0 / (1.0 + exp((v - vars(vhh)) / vars(ki)));
        vars(tm) = vars(atm) / (exp(-(v - vars(btm)) / vars(ctm)) + exp((v - vars(dtm)) / vars(etm))) + vars(ftm);
        vars(thf) = vars(athf) / (1.0 + exp((v - vars(bthf)) / vars(cthf))) + vars(dthf);
        vars(ths) = vars(aths) / (1.0 + exp((v - vars(bths)) / vars(cths))) + vars(dths);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(m) = 0.0;
        vars(hf) = 0.0;
        vars(hs) = 0.0;
        vars(m_dm) = 0.0;
        vars(hf_dhf) = 0.0;
        vars(hs_dhs) = 0.0;
        
        // Execute INITIAL block logic
        setparames(vars, param.volt);
        vars(m_dm) = vars(minf);
        vars(hf_dhf) = vars(hfinf);
        vars(hs_dhs) = vars(hsinf);
        vars(m) = vars(minf);
        vars(hf) = vars(hfinf);
        vars(hs) = vars(hsinf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block
        vars(ik_di) = vars(gbshl1) * vars(m_dm) * vars(m_dm) * vars(m_dm) * 
                      (0.7 * vars(hf_dhf) + 0.3 * vars(hs_dhs)) * (param.volt + vars(dv) - vars(ek_tmp));
        vars(i_di) = vars(w) * vars(ik_di);
        vars(ik_tmp) = vars(gbshl1) * vars(m) * vars(m) * vars(m) * 
                       (0.7 * vars(hf) + 0.3 * vars(hs)) * (param.volt - vars(ek_tmp));
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
        vars(m_dm) = vars(m);
        vars(hf_dhf) = vars(hf);
        vars(hs_dhs) = vars(hs);
        
        // First setparames call with v + dv
        setparames(vars, param.volt + vars(dv));
        // Update _d variables using cnexp integration with modified time constants
        vars(m_dm) = vars(m_dm) + (1.0 - exp(param.dt * ((-1.0) / (vars(tm) * 0.4)))) * 
                     (-(vars(minf) / (vars(tm) * 0.4)) / ((-1.0) / (vars(tm) * 0.4)) - vars(m_dm));
        vars(hf_dhf) = vars(hf_dhf) + (1.0 - exp(param.dt * ((-1.0) / (vars(thf) * 0.08)))) * 
                       (-(vars(hfinf) / (vars(thf) * 0.08)) / ((-1.0) / (vars(thf) * 0.08)) - vars(hf_dhf));
        vars(hs_dhs) = vars(hs_dhs) + (1.0 - exp(param.dt * ((-1.0) / (vars(ths) * 0.3)))) * 
                       (-(vars(hsinf) / (vars(ths) * 0.3)) / ((-1.0) / (vars(ths) * 0.3)) - vars(hs_dhs));
        
        // Second setparames call with v
        setparames(vars, param.volt);
        // Update main variables using cnexp integration with modified time constants
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / (vars(tm) * 0.4)))) * 
                  (-(vars(minf) / (vars(tm) * 0.4)) / ((-1.0) / (vars(tm) * 0.4)) - vars(m));
        vars(hf) = vars(hf) + (1.0 - exp(param.dt * ((-1.0) / (vars(thf) * 0.08)))) * 
                   (-(vars(hfinf) / (vars(thf) * 0.08)) / ((-1.0) / (vars(thf) * 0.08)) - vars(hf));
        vars(hs) = vars(hs) + (1.0 - exp(param.dt * ((-1.0) / (vars(ths) * 0.3)))) * 
                   (-(vars(hsinf) / (vars(ths) * 0.3)) / ((-1.0) / (vars(ths) * 0.3)) - vars(hs));
    }
};

REGISTER_MECHANISM("shl1_lr", Shl1Lr);

} // namespace shl1_lr
