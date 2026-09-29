// kqt3_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace kqt3_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbkqt3, ek_tmp, w, dv,
        // ASSIGNED variables
        i, ik_tmp, minf, tmf, tms, winf, sinf, tw, ts, ik_di, i_di, pure_i, didv,
        // STATE variables  
        mf, ms, ww, s, mf_dmf, ms_dms, ww_dww, s_ds,
        // Ion variables (local copies)
        ek, ik
    };
    
    enum class GlobalVarNames {
        vhm, ka, vhw, kiw, aw, bw, vhs, kis, as, bs,
        atmf, btmf, ctmf, atms, btms, ctms, dtms, etms, ftms, gtms,
        atw, btw, ctw, dtw, ats
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_dikdv
    };
};

class Kqt3Lr : public MechTemp<Kqt3Lr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    Kqt3Lr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbkqt3, 1.0});
        init_values.insert({ek_tmp, -80.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices (following pattern of other lr mechanisms)
        var_in_coredata_idx.insert({gbkqt3, 0});
        var_in_coredata_idx.insert({ek_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({dv, 3});
        var_in_coredata_idx.insert({i, 4});
        var_in_coredata_idx.insert({ik_tmp, 5});
        var_in_coredata_idx.insert({minf, 6});
        var_in_coredata_idx.insert({tmf, 7});
        var_in_coredata_idx.insert({tms, 8});
        var_in_coredata_idx.insert({winf, 9});
        var_in_coredata_idx.insert({sinf, 10});
        var_in_coredata_idx.insert({tw, 11});
        var_in_coredata_idx.insert({ts, 12});
        var_in_coredata_idx.insert({ik_di, 13});
        var_in_coredata_idx.insert({i_di, 14});
        var_in_coredata_idx.insert({pure_i, 15});
        var_in_coredata_idx.insert({didv, 16});
        var_in_coredata_idx.insert({mf, 17});
        var_in_coredata_idx.insert({ms, 18});
        var_in_coredata_idx.insert({ww, 19});
        var_in_coredata_idx.insert({s, 20});
        var_in_coredata_idx.insert({mf_dmf, 21});
        var_in_coredata_idx.insert({ms_dms, 22});
        var_in_coredata_idx.insert({ww_dww, 23});
        var_in_coredata_idx.insert({s_ds, 24});
        var_in_coredata_idx.insert({ek, 25});
        var_in_coredata_idx.insert({ik, 26});
        
        // Register global variables
        global_info_map.insert({vhm, {"vhm_kqt3_lr"}});
        global_info_map.insert({ka, {"ka_kqt3_lr"}});
        global_info_map.insert({vhw, {"vhw_kqt3_lr"}});
        global_info_map.insert({kiw, {"kiw_kqt3_lr"}});
        global_info_map.insert({aw, {"aw_kqt3_lr"}});
        global_info_map.insert({bw, {"bw_kqt3_lr"}});
        global_info_map.insert({vhs, {"vhs_kqt3_lr"}});
        global_info_map.insert({kis, {"kis_kqt3_lr"}});
        global_info_map.insert({as, {"as_kqt3_lr"}});
        global_info_map.insert({bs, {"bs_kqt3_lr"}});
        global_info_map.insert({atmf, {"atmf_kqt3_lr"}});
        global_info_map.insert({btmf, {"btmf_kqt3_lr"}});
        global_info_map.insert({ctmf, {"ctmf_kqt3_lr"}});
        global_info_map.insert({atms, {"atms_kqt3_lr"}});
        global_info_map.insert({btms, {"btms_kqt3_lr"}});
        global_info_map.insert({ctms, {"ctms_kqt3_lr"}});
        global_info_map.insert({dtms, {"dtms_kqt3_lr"}});
        global_info_map.insert({etms, {"etms_kqt3_lr"}});
        global_info_map.insert({ftms, {"ftms_kqt3_lr"}});
        global_info_map.insert({gtms, {"gtms_kqt3_lr"}});
        global_info_map.insert({atw, {"atw_kqt3_lr"}});
        global_info_map.insert({btw, {"btw_kqt3_lr"}});
        global_info_map.insert({ctw, {"ctw_kqt3_lr"}});
        global_info_map.insert({dtw, {"dtw_kqt3_lr"}});
        global_info_map.insert({ats, {"ats_kqt3_lr"}});
        
        // Register ion variables (potassium)
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        vars(minf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        vars(tmf) = vars(atmf) / (1.0 + pow((v + vars(btmf)) / vars(ctmf), 2.0));
        vars(tms) = vars(atms) + vars(btms) / (1.0 + pow(10.0, -vars(ctms) * (vars(dtms) - v))) + 
                    vars(etms) / (1.0 + pow(10.0, -vars(ftms) * (vars(gtms) + v)));
        vars(winf) = vars(aw) + vars(bw) / (1.0 + exp((v - vars(vhw)) / vars(kiw)));
        vars(sinf) = vars(as) + vars(bs) / (1.0 + exp((v - vars(vhs)) / vars(kis)));
        vars(tw) = vars(atw) + vars(btw) / (1.0 + pow((v - vars(ctw)) / vars(dtw), 2.0));
        vars(ts) = vars(ats);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(mf) = 0.0;
        vars(ms) = 0.0;
        vars(ww) = 0.0;
        vars(s) = 0.0;
        vars(mf_dmf) = 0.0;
        vars(ms_dms) = 0.0;
        vars(ww_dww) = 0.0;
        vars(s_ds) = 0.0;
        
        // Execute INITIAL block logic
        setparames(vars, param.volt);
        vars(mf_dmf) = 0.0;
        vars(ms_dms) = 0.0;
        vars(ww_dww) = vars(winf);
        vars(s_ds) = vars(sinf);
        vars(mf) = 0.0;
        vars(ms) = 0.0;
        vars(ww) = vars(winf);
        vars(s) = vars(sinf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block - using ek from ion instead of ek_tmp for actual current
        vars(ek) = vars(_ion_ek);
        
        vars(ik_di) = vars(gbkqt3) * (0.3 * vars(mf_dmf) + 0.7 * vars(ms_dms)) * vars(ww_dww) * vars(s_ds) * 
                      (param.volt + vars(dv) - vars(ek));
        vars(i_di) = vars(w) * vars(ik_di);
        
        vars(ik_tmp) = vars(gbkqt3) * (0.3 * vars(mf) + 0.7 * vars(ms)) * vars(ww) * vars(s) * (param.volt - vars(ek));
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
        vars(ms_dms) = vars(ms);
        vars(ww_dww) = vars(ww);
        vars(s_ds) = vars(s);
        
        // First setparames call with v + dv
        setparames(vars, param.volt + vars(dv));
        // Update _d variables using cnexp integration
        vars(mf_dmf) = vars(mf_dmf) + (1.0 - exp(param.dt * ((-1.0) / vars(tmf)))) * 
                       (-(vars(minf) / vars(tmf)) / ((-1.0) / vars(tmf)) - vars(mf_dmf));
        vars(ms_dms) = vars(ms_dms) + (1.0 - exp(param.dt * ((-1.0) / vars(tms)))) * 
                       (-(vars(minf) / vars(tms)) / ((-1.0) / vars(tms)) - vars(ms_dms));
        vars(ww_dww) = vars(ww_dww) + (1.0 - exp(param.dt * ((-1.0) / vars(tw)))) * 
                       (-(vars(winf) / vars(tw)) / ((-1.0) / vars(tw)) - vars(ww_dww));
        vars(s_ds) = vars(s_ds) + (1.0 - exp(param.dt * ((-1.0) / vars(ts)))) * 
                     (-(vars(sinf) / vars(ts)) / ((-1.0) / vars(ts)) - vars(s_ds));
        
        // Second setparames call with v
        setparames(vars, param.volt);
        // Update main variables using cnexp integration
        vars(mf) = vars(mf) + (1.0 - exp(param.dt * ((-1.0) / vars(tmf)))) * 
                   (-(vars(minf) / vars(tmf)) / ((-1.0) / vars(tmf)) - vars(mf));
        vars(ms) = vars(ms) + (1.0 - exp(param.dt * ((-1.0) / vars(tms)))) * 
                   (-(vars(minf) / vars(tms)) / ((-1.0) / vars(tms)) - vars(ms));
        vars(ww) = vars(ww) + (1.0 - exp(param.dt * ((-1.0) / vars(tw)))) * 
                   (-(vars(winf) / vars(tw)) / ((-1.0) / vars(tw)) - vars(ww));
        vars(s) = vars(s) + (1.0 - exp(param.dt * ((-1.0) / vars(ts)))) * 
                  (-(vars(sinf) / vars(ts)) / ((-1.0) / vars(ts)) - vars(s));
    }
};

REGISTER_MECHANISM("kqt3_lr", Kqt3Lr);

} // namespace kqt3_lr
