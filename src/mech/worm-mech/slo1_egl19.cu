// slo1_egl19 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace slo1_egl19_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, hcav, mcav,
        
        // Parameters
        gbslo1,
        
        // Assigned variables
        minf, tm, mcavinf, tmcav, hcavinf, thcav,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class SLO1_EGL19_Channel : public MechTemp<SLO1_EGL19_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    SLO1_EGL19_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbslo1, 1.0});
        
        // Variable indexing based on RANGE order
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
        
        double m_val = vars(m);
        double hcav_val = vars(hcav);
        vars(ik) = vars(gbslo1) * m_val * hcav_val * (param.volt + 80.0);
        
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
        // slo1 parameters
        double wyx = 0.013, wxy = -0.028;
        double wom = 3.15, wop = 0.16;
        double kxy = 55.73, nxy = 1.30;
        double kyx = 34.34, nyx = 0.0001;
        double canci = 0.05;
        
        // egl19 parameters
        double vhm = 5.6, ka = 7.5, vhh = 24.9, ki = 12.0;
        double vhhb = -20.5, kib = 8.1;
        double ahinf = 1.43, bhinf = 0.14, chinf = 5.96, dhinf = 0.60;
        double atm = 2.9, btm = 5.2, ctm = 6.0, dtm = 1.9, etm = 1.4, ftm = 30.0, gtm = 2.3;
        double ath = 0.4, bth = 44.6, cth = -23.0, dth = 5.0, eth = 36.4, fth = 28.7, gth = 3.7, hth = 43.1;
        
        // calcium parameters
        double FARADAY = 96485.0;
        double gsc = 0.04, r = 0.013, dca = 250.0, kb = 500.0, btot = 30.0;
        
        // egl19 model
        vars(mcavinf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hcavinf) = (ahinf / (1.0 + exp(-(v - vhh) / ki)) + bhinf) * (chinf / (1.0 + exp((v - vhhb) / kib)) + dhinf);
        vars(tmcav) = atm * exp(-pow((v - btm) / ctm, 2)) + dtm * exp(-pow((v - etm) / ftm, 2)) + gtm;
        vars(thcav) = ath * ((bth / (1.0 + exp((v - cth) / dth))) + (eth / (1.0 + exp((v - fth) / gth))) + hth);
        
        // calcium concentration calculation
        double cain;
        if (v < 60.0) {
            cain = -gsc * (v - 60.0) * 1e9 / (8.0 * 3.1415926 * r * dca * FARADAY) * exp(-r / sqrt(dca / (kb * btot)));
        } else {
            cain = 0.0001;
        }
        
        // slo1-egl19 model - performance optimized
        double mcavinf_val = vars(mcavinf);
        double tmcav_val = vars(tmcav);
        double alpha = mcavinf_val / tmcav_val;
        double beta = 1.0 / tmcav_val - alpha;
        double wm = wom * exp(-wyx * v);
        double wp = wop * exp(-wxy * v);
        double fm = 1.0 / (1.0 + pow(cain / kyx, nyx));
        double fp = 1.0 / (1.0 + pow(kxy / cain, nxy));
        double kom = wm * fm;
        double kop = wp * fp;
        double kcm = wm * 1.0 / (1.0 + pow(canci / kyx, nyx));
        
        double mcav_val = vars(mcav);
        double denom = (kop + kom) * (kcm + alpha) + beta * kcm;
        vars(minf) = mcav_val * kop * (alpha + beta + kcm) / denom;
        vars(tm) = (alpha + beta + kcm) / denom;
    }
};

REGISTER_MECHANISM("slo1_egl19", SLO1_EGL19_Channel);

} // namespace slo1_egl19_worm