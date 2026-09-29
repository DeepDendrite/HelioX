// shk1 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace shk1_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Parameters
        gbshk1,
        
        // Assigned variables
        minf, hinf, tm, th,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class SHK1_Channel : public MechTemp<SHK1_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    SHK1_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbshk1, 1.0});
        
        var_in_coredata_idx.insert({gbshk1, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({hinf, 2});
        var_in_coredata_idx.insert({tm, 3});
        var_in_coredata_idx.insert({th, 4});
        var_in_coredata_idx.insert({m, 5});
        var_in_coredata_idx.insert({h, 6});
        var_in_coredata_idx.insert({ek, 7});
        var_in_coredata_idx.insert({ik, 8});
        
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(minf);
        vars(h) = vars(hinf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        
        double m_val = vars(m);
        double h_val = vars(h);
        vars(ik) = vars(gbshk1) * m_val * h_val * (param.volt + 80.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        double dt = param.dt;
        vars(m) = vars(m) + (1.0 - exp(-dt / vars(tm))) * (vars(minf) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-dt / vars(th))) * (vars(hinf) - vars(h));
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        double vhm = 20.4, ka = 7.7, vhh = -7.0, ki = 5.8;
        double atm = 26.6, btm = -33.7, ctm = 15.8, dtm = -33.7, etm = 15.4, ftm = 2.0;
        double ath = 1400.0;
        
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(hinf) = 1.0 / (1.0 + exp((v - vhh) / ki));
        vars(tm) = atm / (exp(-(v - btm) / ctm) + exp((v - dtm) / etm)) + ftm;
        vars(th) = ath;
    }
};

REGISTER_MECHANISM("shk1", SHK1_Channel);

} // namespace shk1_worm