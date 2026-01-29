// unc2_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace unc2_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbunc2, eca_tmp, w, dv,
        // ASSIGNED variables
        i, ica_tmp, minf, hinf, tm, th, ica_di, i_di, pure_i, didv,
        // STATE variables  
        m, h, m_dm, h_dh,
        // Ion variables (local copies)
        eca, ica
    };
    
    enum class GlobalVarNames {
        vhm, ka, vhh, ki, atm, btm, ctm, dtm, etm,
        ath, bth, cth, dth, eth, fth
    };
    
    enum class IonVarNames {
        _ion_eca, _ion_ica, _ion_dicadv
    };
};

class Unc2Lr : public MechTemp<Unc2Lr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    Unc2Lr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbunc2, 1.0});
        init_values.insert({eca_tmp, 60.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices based on CPP setup_instance
        var_in_coredata_idx.insert({gbunc2, 0});
        var_in_coredata_idx.insert({eca_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({dv, 3});
        var_in_coredata_idx.insert({i, 4});
        var_in_coredata_idx.insert({ica_tmp, 5});
        var_in_coredata_idx.insert({minf, 6});
        var_in_coredata_idx.insert({hinf, 7});
        var_in_coredata_idx.insert({tm, 8});
        var_in_coredata_idx.insert({th, 9});
        var_in_coredata_idx.insert({ica_di, 10});
        var_in_coredata_idx.insert({i_di, 11});
        var_in_coredata_idx.insert({pure_i, 12});
        var_in_coredata_idx.insert({didv, 13});
        var_in_coredata_idx.insert({m, 14});
        var_in_coredata_idx.insert({h, 15});
        var_in_coredata_idx.insert({m_dm, 16});
        var_in_coredata_idx.insert({h_dh, 17});
        var_in_coredata_idx.insert({eca, 18});
        var_in_coredata_idx.insert({ica, 19});
        
        // Register global variables
        global_info_map.insert({vhm, {"vhm_unc2_lr"}});
        global_info_map.insert({ka, {"ka_unc2_lr"}});
        global_info_map.insert({vhh, {"vhh_unc2_lr"}});
        global_info_map.insert({ki, {"ki_unc2_lr"}});
        global_info_map.insert({atm, {"atm_unc2_lr"}});
        global_info_map.insert({btm, {"btm_unc2_lr"}});
        global_info_map.insert({ctm, {"ctm_unc2_lr"}});
        global_info_map.insert({dtm, {"dtm_unc2_lr"}});
        global_info_map.insert({etm, {"etm_unc2_lr"}});
        global_info_map.insert({ath, {"ath_unc2_lr"}});
        global_info_map.insert({bth, {"bth_unc2_lr"}});
        global_info_map.insert({cth, {"cth_unc2_lr"}});
        global_info_map.insert({dth, {"dth_unc2_lr"}});
        global_info_map.insert({eth, {"eth_unc2_lr"}});
        global_info_map.insert({fth, {"fth_unc2_lr"}});
        
        // Register ion variables
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dicadv, {"ca_ion", EionVarNames::dcurdv}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        vars(minf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        vars(hinf) = 1.0 / (1.0 + exp((v - vars(vhh)) / vars(ki)));
        vars(tm) = vars(atm) / (exp(-(v - vars(btm)) / vars(ctm)) + exp((v - vars(btm)) / vars(dtm))) + vars(etm);
        vars(th) = vars(ath) / (1.0 + exp(-(v - vars(bth)) / vars(cth))) + vars(dth) / (1.0 + exp((v - vars(eth)) / vars(fth)));
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(m) = 0.0;
        vars(h) = 0.0;
        vars(m_dm) = 0.0;
        vars(h_dh) = 0.0;
        
        // Execute INITIAL block logic
        setparames(vars, param.volt);
        vars(m_dm) = vars(minf);
        vars(h_dh) = vars(hinf);
        vars(m) = vars(minf);
        vars(h) = vars(hinf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        if (!param.updateIon) {
            // Store I(v+0.001) for finite-difference dcurdv to match NEURON.
            vars(ica_tmp) = vars(gbunc2) * vars(m) * vars(h) * (param.volt - vars(eca_tmp));
            vars(pure_i) = vars(ica_tmp);
            vars(i) = vars(w) * vars(pure_i);
            vars(ica) = vars(i);
            vars(i_di) = vars(i);
            return vars(ica);
        }

        double i_vplus = vars(i_di);
        vars(ica_di) = vars(gbunc2) * vars(m_dm) * vars(h_dh) * (param.volt + vars(dv) - vars(eca_tmp));
        vars(i_di) = vars(w) * vars(ica_di);
        vars(ica_tmp) = vars(gbunc2) * vars(m) * vars(h) * (param.volt - vars(eca_tmp));
        vars(pure_i) = vars(ica_tmp);
        vars(i) = vars(w) * vars(pure_i);
        vars(ica) = vars(i);
        vars(didv) = (i_vplus - vars(i)) / 0.001;
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
            mechAtomAdd(&vars(_ion_dicadv), vars(didv));
        }
        
        return vars(ica);
    }

    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(eca) = vars(_ion_eca);
        
        // Save current state values
        vars(m_dm) = vars(m);
        vars(h_dh) = vars(h);
        
        // First setparames call with v + dv
        setparames(vars, param.volt + vars(dv));
        // Update m_dm and h_dh using cnexp integration
        vars(m_dm) = vars(m_dm) + (1.0 - exp(param.dt * ((-1.0) / vars(tm)))) * 
                     (-(vars(minf) / vars(tm)) / ((-1.0) / vars(tm)) - vars(m_dm));
        vars(h_dh) = vars(h_dh) + (1.0 - exp(param.dt * ((-1.0) / vars(th)))) * 
                     (-(vars(hinf) / vars(th)) / ((-1.0) / vars(th)) - vars(h_dh));
        
        // Second setparames call with v
        setparames(vars, param.volt);
        // Update m and h using cnexp integration
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(tm)))) * 
                  (-(vars(minf) / vars(tm)) / ((-1.0) / vars(tm)) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(param.dt * ((-1.0) / vars(th)))) * 
                  (-(vars(hinf) / vars(th)) / ((-1.0) / vars(th)) - vars(h));
    }
};

REGISTER_MECHANISM("unc2_lr", Unc2Lr);

} // namespace unc2_lr
