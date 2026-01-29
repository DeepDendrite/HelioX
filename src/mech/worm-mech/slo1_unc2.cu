// slo1_unc2 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace slo1_unc2_worm {

// Similar structure to slo1_egl19 but with unc2 parameters
struct MechTrait {
    enum class VarNames {
        m, hcav, mcav, gbslo1, minf, tm, mcavinf, tmcav, hcavinf, thcav, ek, ik
    };
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class SLO1_UNC2_Channel : public MechTemp<SLO1_UNC2_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    SLO1_UNC2_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbslo1, 1.0});
        
        // Variable indexing
        var_in_coredata_idx.insert({gbslo1, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({tm, 2});
        var_in_coredata_idx.insert({mcavinf, 3});
        var_in_coredata_idx.insert({tmcav, 4});
        var_in_coredata_idx.insert({hcavinf, 5});
        var_in_coredata_idx.insert({thcav, 6});
        var_in_coredata_idx.insert({m, 7});
        var_in_coredata_idx.insert({hcav, 8});
        var_in_coredata_idx.insert({mcav, 9});
        var_in_coredata_idx.insert({ek, 10});
        var_in_coredata_idx.insert({ik, 11});
        
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(hcav) = vars(hcavinf);
        vars(mcav) = vars(mcavinf);
        vars(m) = 0.0;
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(ik) = vars(gbslo1) * vars(m) * vars(hcav) * (param.volt + 80.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return vars(ik);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        double dt = param.dt;
        vars(mcav) = vars(mcav) + (1.0 - exp(-dt / vars(tmcav))) * (vars(mcavinf) - vars(mcav));
        vars(hcav) = vars(hcav) + (1.0 - exp(-dt / vars(thcav))) * (vars(hcavinf) - vars(hcav));
        vars(m) = vars(m) + (1.0 - exp(-dt / vars(tm))) * (vars(minf) - vars(m));
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        // slo1 parameters (same as slo1_egl19)
        double wyx = 0.013, wxy = -0.028, wom = 3.15, wop = 0.16;
        double kxy = 55.73, nxy = 1.30, kyx = 34.34, nyx = 0.0001, canci = 0.05;
        
        // unc2 parameters (different from egl19)
        double vhm = -12.2, ka = 4.0, vhh = -52.5, ki = 5.6;
        double atm = 1.5, btm = -8.2, ctm = 9.1, dtm = 15.4, etm = 0.1;
        double ath = 83.8, bth = 52.9, cth = -3.5, dth = 72.1, eth = 23.9, fth = -3.6;
        
        // calcium parameters
        double FARADAY = 96485.0, gsc = 0.04, r = 0.013, dca = 250.0, kb = 500.0, btot = 30.0;
        
        // unc2 model
        vars(mcavinf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hcavinf) = 1.0 / (1.0 + exp((v - vhh) / ki));
        vars(tmcav) = atm / (exp(-(v - btm) / ctm) + exp((v - btm) / dtm)) + etm;
        vars(thcav) = ath / (1.0 + exp(-(v - bth) / cth)) + dth / (1.0 + exp((v - eth) / fth));
        
        // calcium concentration
        double cain = (v < 60.0) ? -gsc * (v - 60.0) * 1e9 / (8.0 * 3.1415926 * r * dca * FARADAY) * exp(-r / sqrt(dca / (kb * btot))) : 0.0001;
        
        // slo1-unc2 model
        double mcavinf_val = vars(mcavinf), tmcav_val = vars(tmcav);
        double alpha = mcavinf_val / tmcav_val, beta = 1.0 / tmcav_val - alpha;
        double wm = wom * exp(-wyx * v), wp = wop * exp(-wxy * v);
        double fm = 1.0 / (1.0 + pow(cain / kyx, nyx)), fp = 1.0 / (1.0 + pow(kxy / cain, nxy));
        double kom = wm * fm, kop = wp * fp, kcm = wm * 1.0 / (1.0 + pow(canci / kyx, nyx));
        double denom = (kop + kom) * (kcm + alpha) + beta * kcm;
        
        vars(minf) = vars(mcav) * kop * (alpha + beta + kcm) / denom;
        vars(tm) = (alpha + beta + kcm) / denom;
    }
};

REGISTER_MECHANISM("slo1_unc2", SLO1_UNC2_Channel);

} // namespace slo1_unc2_worm