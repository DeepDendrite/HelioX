// kcnl mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace kcnl_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m,
        
        // Parameters
        gbkcnl,
        
        // Assigned variables
        minf, tm,
        
        // Ion reversal potential and current
        ek, ik, cai
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik, _ion_cai
    };
};

class KCNL_Channel : public MechTemp<KCNL_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    KCNL_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbkcnl, 1.0});
        
        // 按照NEURON CPP中的确切索引顺序
        var_in_coredata_idx.insert({gbkcnl, 0});
        var_in_coredata_idx.insert({m, 1});
        var_in_coredata_idx.insert({ek, 2});
        var_in_coredata_idx.insert({ik, 3});
        var_in_coredata_idx.insert({cai, 4});
        var_in_coredata_idx.insert({minf, 5});
        var_in_coredata_idx.insert({tm, 6});
        
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(minf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);
        
        vars(ik) = vars(gbkcnl) * vars(m) * (param.volt + 80.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(m) = vars(m) + (1.0 - exp(param.dt*(( ( ( - 1.0 ) ) ) / vars(tm))))*(- ( ( ( vars(minf) ) ) / vars(tm) ) / ( ( ( ( - 1.0 ) ) ) / vars(tm) ) - vars(m));
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        double kca = 0.33;  // uM/um2
        double a = 6.3;     // ms
        
        vars(cai) = vars(_ion_cai);  // Read calcium concentration
        vars(minf) = vars(cai) / (kca + vars(cai));
        vars(tm) = a;
    }
};

REGISTER_MECHANISM("kcnl", KCNL_Channel);

} // namespace kcnl_worm