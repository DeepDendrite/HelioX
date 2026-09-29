// kvs1 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace kvs1_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        m, h,
        
        // Parameters
        gbkvs1,
        
        // Assigned variables
        minf, hinf, tm, th,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class GlobalVarNames {
        vhm, ka, vhh, ki, atm, btm, ctm, dtm, ath, bth, cth, dth
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

DEFINE_HAS_ENUM(GlobalVarNames);
DEFINE_ENUM_MAP_ALIAS(GlobalVarNames, CoreGlobalVarInfo, GlobalVarInfoMap);

class KVS1_Channel : public MechTemp<KVS1_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    KVS1_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbkvs1, 1.0});
        
        var_in_coredata_idx.insert({gbkvs1, 0});
        var_in_coredata_idx.insert({minf, 1});
        var_in_coredata_idx.insert({hinf, 2});
        var_in_coredata_idx.insert({tm, 3});
        var_in_coredata_idx.insert({th, 4});
        var_in_coredata_idx.insert({m, 5});
        var_in_coredata_idx.insert({h, 6});
        var_in_coredata_idx.insert({ek, 7});
        var_in_coredata_idx.insert({ik, 8});
        
        // 注册全局变量 - 后面的string需要和与NEURON完全一致
        global_info_map.insert({vhm, {"vhm_kvs1"}});
        global_info_map.insert({ka, {"ka_kvs1"}});
        global_info_map.insert({vhh, {"vhh_kvs1"}});
        global_info_map.insert({ki, {"ki_kvs1"}});
        global_info_map.insert({atm, {"atm_kvs1"}});
        global_info_map.insert({btm, {"btm_kvs1"}});
        global_info_map.insert({ctm, {"ctm_kvs1"}});
        global_info_map.insert({dtm, {"dtm_kvs1"}});
        global_info_map.insert({ath, {"ath_kvs1"}});
        global_info_map.insert({bth, {"bth_kvs1"}});
        global_info_map.insert({cth, {"cth_kvs1"}});
        global_info_map.insert({dth, {"dth_kvs1"}});
        
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
        vars(ik) = vars(gbkvs1) * m_val * h_val * (param.volt + 80.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        double dt = param.dt;
        double tm_val = vars(tm);
        double th_val = vars(th);
        vars(m) = vars(m) + (1.0 - exp(dt * (-1.0 / tm_val))) * (vars(minf) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(dt * (-1.0 / th_val))) * (vars(hinf) - vars(h));
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        double vhm_val = vars(vhm);
        double ka_val = vars(ka);
        double vhh_val = vars(vhh);
        double ki_val = vars(ki);
        double atm_val = vars(atm);
        double btm_val = vars(btm);
        double ctm_val = vars(ctm);
        double dtm_val = vars(dtm);
        double ath_val = vars(ath);
        double bth_val = vars(bth);
        double cth_val = vars(cth);
        double dth_val = vars(dth);
        
        vars(minf) = 1.0 / (1.0 + exp(-(v - vhm_val) / ka_val));
        vars(hinf) = 1.0 / (1.0 + exp((v - vhh_val) / ki_val));
        vars(tm) = atm_val / (1.0 + exp((v - btm_val) / ctm_val)) + dtm_val;
        vars(th) = ath_val / (1.0 + exp((v - bth_val) / cth_val)) + dth_val;
    }
};

REGISTER_MECHANISM("kvs1", KVS1_Channel);

} // namespace kvs1_worm