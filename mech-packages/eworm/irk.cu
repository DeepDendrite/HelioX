// irk mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace irk_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m,
        
        // Parameters
        gbirk,
        
        // Assigned variables
        minf, tm,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class IRK_Channel : public MechTemp<IRK_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    IRK_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbirk, 1.0});
        
        var_in_coredata_idx.insert({gbirk, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({tm, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({ik, 5});
        
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(minf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(ik) = vars(gbirk) * vars(m) * (param.volt + 80.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(m) + (1.0 - exp(-param.dt / vars(tm))) * (vars(minf) - vars(m));
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        double vhm = -82.0, ka = 13.0;
        double atm = 17.1, btm = -17.8, ctm = 20.3, dtm = -43.4, etm = 11.2, ftm = 3.8;
        
        vars(minf) = 1.0 / (1.0 + exp((v - vhm) / ka));
        vars(tm) = atm / (exp(-(v - btm) / ctm) + exp((v - dtm) / etm)) + ftm;
    }
};

REGISTER_MECHANISM("irk", IRK_Channel);

} // namespace irk_worm