// slo2_egl19_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace slo2_egl19_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        gbslo2, ek_tmp, w, dv,
        // ASSIGNED variables
        i, ik_tmp, minf, tm, mcavinf, tmcav, hcavinf, thcav, ik_di, i_di, pure_i, didv,
        // STATE variables  
        m, hcav, mcav, m_dm, hcav_dhcav, mcav_dmcav,
        // Ion variables (local copies)
        ek, ik
    };
    
    enum class GlobalVarNames {
        wyx, wxy, wom, wop, kxy, nxy, kyx, nyx, canci,
        vhm, ka, vhh, ki, vhhb, kib, ahinf, bhinf, chinf, dhinf,
        atm, btm, ctm, dtm, etm, ftm, gtm,
        ath, bth, cth, dth, eth, fth, gth, hth,
        FARADAY, gsc, r, dca, kb, btot
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_dikdv
    };
};

class Slo2Egl19Lr : public MechTemp<Slo2Egl19Lr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    Slo2Egl19Lr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gbslo2, 1.0});
        init_values.insert({ek_tmp, -80.0});
        init_values.insert({w, 1.0});
        init_values.insert({dv, 1e-3});
        
        // Register variable indices
        var_in_coredata_idx.insert({gbslo2, 0});
        var_in_coredata_idx.insert({ek_tmp, 1});
        var_in_coredata_idx.insert({w, 2});
        var_in_coredata_idx.insert({dv, 3});
        var_in_coredata_idx.insert({i, 4});
        var_in_coredata_idx.insert({ik_tmp, 5});
        var_in_coredata_idx.insert({minf, 6});
        var_in_coredata_idx.insert({tm, 7});
        var_in_coredata_idx.insert({mcavinf, 8});
        var_in_coredata_idx.insert({tmcav, 9});
        var_in_coredata_idx.insert({hcavinf, 10});
        var_in_coredata_idx.insert({thcav, 11});
        var_in_coredata_idx.insert({ik_di, 12});
        var_in_coredata_idx.insert({i_di, 13});
        var_in_coredata_idx.insert({pure_i, 14});
        var_in_coredata_idx.insert({didv, 15});
        var_in_coredata_idx.insert({m, 16});
        var_in_coredata_idx.insert({hcav, 17});
        var_in_coredata_idx.insert({mcav, 18});
        var_in_coredata_idx.insert({m_dm, 19});
        var_in_coredata_idx.insert({hcav_dhcav, 20});
        var_in_coredata_idx.insert({mcav_dmcav, 21});
        var_in_coredata_idx.insert({ek, 22});
        var_in_coredata_idx.insert({ik, 23});
        
        // Register global variables
        global_info_map.insert({wyx, {"wyx_slo2_egl19_lr"}});
        global_info_map.insert({wxy, {"wxy_slo2_egl19_lr"}});
        global_info_map.insert({wom, {"wom_slo2_egl19_lr"}});
        global_info_map.insert({wop, {"wop_slo2_egl19_lr"}});
        global_info_map.insert({kxy, {"kxy_slo2_egl19_lr"}});
        global_info_map.insert({nxy, {"nxy_slo2_egl19_lr"}});
        global_info_map.insert({kyx, {"kyx_slo2_egl19_lr"}});
        global_info_map.insert({nyx, {"nyx_slo2_egl19_lr"}});
        global_info_map.insert({canci, {"canci_slo2_egl19_lr"}});
        global_info_map.insert({vhm, {"vhm_slo2_egl19_lr"}});
        global_info_map.insert({ka, {"ka_slo2_egl19_lr"}});
        global_info_map.insert({vhh, {"vhh_slo2_egl19_lr"}});
        global_info_map.insert({ki, {"ki_slo2_egl19_lr"}});
        global_info_map.insert({vhhb, {"vhhb_slo2_egl19_lr"}});
        global_info_map.insert({kib, {"kib_slo2_egl19_lr"}});
        global_info_map.insert({ahinf, {"ahinf_slo2_egl19_lr"}});
        global_info_map.insert({bhinf, {"bhinf_slo2_egl19_lr"}});
        global_info_map.insert({chinf, {"chinf_slo2_egl19_lr"}});
        global_info_map.insert({dhinf, {"dhinf_slo2_egl19_lr"}});
        global_info_map.insert({atm, {"atm_slo2_egl19_lr"}});
        global_info_map.insert({btm, {"btm_slo2_egl19_lr"}});
        global_info_map.insert({ctm, {"ctm_slo2_egl19_lr"}});
        global_info_map.insert({dtm, {"dtm_slo2_egl19_lr"}});
        global_info_map.insert({etm, {"etm_slo2_egl19_lr"}});
        global_info_map.insert({ftm, {"ftm_slo2_egl19_lr"}});
        global_info_map.insert({gtm, {"gtm_slo2_egl19_lr"}});
        global_info_map.insert({ath, {"ath_slo2_egl19_lr"}});
        global_info_map.insert({bth, {"bth_slo2_egl19_lr"}});
        global_info_map.insert({cth, {"cth_slo2_egl19_lr"}});
        global_info_map.insert({dth, {"dth_slo2_egl19_lr"}});
        global_info_map.insert({eth, {"eth_slo2_egl19_lr"}});
        global_info_map.insert({fth, {"fth_slo2_egl19_lr"}});
        global_info_map.insert({gth, {"gth_slo2_egl19_lr"}});
        global_info_map.insert({hth, {"hth_slo2_egl19_lr"}});
        global_info_map.insert({FARADAY, {"FARADAY_slo2_egl19_lr"}});
        global_info_map.insert({gsc, {"gsc_slo2_egl19_lr"}});
        global_info_map.insert({r, {"r_slo2_egl19_lr"}});
        global_info_map.insert({dca, {"dca_slo2_egl19_lr"}});
        global_info_map.insert({kb, {"kb_slo2_egl19_lr"}});
        global_info_map.insert({btot, {"btot_slo2_egl19_lr"}});
        
        // Register ion variables (potassium)
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        double alpha, beta, wm, wp, fm, fp, kom, kop, kcm, cain;
        
        // egl19 model
        vars(mcavinf) = 1.0 / (1.0 + exp(-(v - vars(vhm)) / vars(ka)));
        vars(hcavinf) = (vars(ahinf) / (1.0 + exp(-(v - vars(vhh)) / vars(ki))) + vars(bhinf)) * 
                        (vars(chinf) / (1.0 + exp((v - vars(vhhb)) / vars(kib))) + vars(dhinf));
        vars(tmcav) = vars(atm) * exp(-pow((v - vars(btm)) / vars(ctm), 2.0)) + 
                      vars(dtm) * exp(-pow((v - vars(etm)) / vars(ftm), 2.0)) + vars(gtm);
        vars(thcav) = vars(ath) * ((vars(bth) / (1.0 + exp((v - vars(cth)) / vars(dth)))) + 
                                   (vars(eth) / (1.0 + exp((v - vars(fth)) / vars(gth)))) + vars(hth));
        
        // cain
        if (v < 60.0) {
            cain = -vars(gsc) * (v - 60.0) * pow(10.0, 9.0) / (8.0 * 3.1415926 * vars(r) * vars(dca) * vars(FARADAY)) * 
                   exp(-vars(r) / sqrt(vars(dca) / (vars(kb) * vars(btot))));
        } else {
            cain = 0.0001;
        }
        
        // slo2-egl19 model
        alpha = vars(mcavinf) / vars(tmcav);
        beta = 1.0 / vars(tmcav) - alpha;
        wm = vars(wom) * exp(-vars(wyx) * v);
        wp = vars(wop) * exp(-vars(wxy) * v);
        fm = 1.0 / (1.0 + pow(cain / vars(kyx), vars(nyx)));
        fp = 1.0 / (1.0 + pow(vars(kxy) / cain, vars(nxy)));
        kom = wm * fm;
        kop = wp * fp;
        kcm = wm * 1.0 / (1.0 + pow(vars(canci) / vars(kyx), vars(nyx)));
        vars(minf) = vars(mcav) * kop * (alpha + beta + kcm) / ((kop + kom) * (kcm + alpha) + beta * kcm);
        vars(tm) = (alpha + beta + kcm) / ((kop + kom) * (kcm + alpha) + beta * kcm);
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Initialize STATE variables to 0 first (NEURON default)
        vars(m) = 0.0;
        vars(hcav) = 0.0;
        vars(mcav) = 0.0;
        vars(m_dm) = 0.0;
        vars(hcav_dhcav) = 0.0;
        vars(mcav_dmcav) = 0.0;
        
        // Execute INITIAL block logic - note different from slo1 (m_dm = minf)
        setparames(vars, param.volt);
        vars(hcav_dhcav) = vars(hcavinf);
        vars(mcav_dmcav) = vars(mcavinf);
        vars(m_dm) = vars(minf);
        vars(hcav) = vars(hcavinf);
        vars(mcav) = vars(mcavinf);
        vars(m) = vars(minf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Following BREAKPOINT block
        vars(ik_di) = vars(gbslo2) * vars(m_dm) * vars(hcav_dhcav) * (param.volt + vars(dv) - vars(ek_tmp));
        vars(i_di) = vars(w) * vars(ik_di);
        vars(ik_tmp) = vars(gbslo2) * vars(m) * vars(hcav) * (param.volt - vars(ek_tmp));
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
        vars(mcav_dmcav) = vars(mcav);
        vars(hcav_dhcav) = vars(hcav);
        vars(m_dm) = vars(m);
        
        // First setparames call with v + dv
        setparames(vars, param.volt + vars(dv));
        // Update _d variables using cnexp integration
        vars(mcav_dmcav) = vars(mcav_dmcav) + (1.0 - exp(param.dt * ((-1.0) / vars(tmcav)))) * 
                           (-(vars(mcavinf) / vars(tmcav)) / ((-1.0) / vars(tmcav)) - vars(mcav_dmcav));
        vars(hcav_dhcav) = vars(hcav_dhcav) + (1.0 - exp(param.dt * ((-1.0) / vars(thcav)))) * 
                           (-(vars(hcavinf) / vars(thcav)) / ((-1.0) / vars(thcav)) - vars(hcav_dhcav));
        vars(m_dm) = vars(m_dm) + (1.0 - exp(param.dt * ((-1.0) / vars(tm)))) * 
                     (-(vars(minf) / vars(tm)) / ((-1.0) / vars(tm)) - vars(m_dm));
        
        // Second setparames call with v
        setparames(vars, param.volt);
        // Update main variables using cnexp integration
        vars(mcav) = vars(mcav) + (1.0 - exp(param.dt * ((-1.0) / vars(tmcav)))) * 
                     (-(vars(mcavinf) / vars(tmcav)) / ((-1.0) / vars(tmcav)) - vars(mcav));
        vars(hcav) = vars(hcav) + (1.0 - exp(param.dt * ((-1.0) / vars(thcav)))) * 
                     (-(vars(hcavinf) / vars(thcav)) / ((-1.0) / vars(thcav)) - vars(hcav));
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(tm)))) * 
                  (-(vars(minf) / vars(tm)) / ((-1.0) / vars(tm)) - vars(m));
    }
};

REGISTER_MECHANISM("slo2_egl19_lr", Slo2Egl19Lr);

} // namespace slo2_egl19_lr
