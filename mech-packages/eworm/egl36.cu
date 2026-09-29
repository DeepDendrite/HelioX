// egl36 mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace egl36_worm {

struct MechTrait {
    enum class VarNames {
        // State variables
        mf, mm, ms,
        
        // Parameters
        gbegl36,
        
        // Assigned variables
        mfinf, mminf, msinf, tmf, tmm, tms,
        
        // Ion reversal potential and current
        ek, ik
    };
    
    enum class IonVarNames {
        _ion_ek, _ion_ik
    };
};

class EGL36_Channel : public MechTemp<EGL36_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    
    EGL36_Channel(MechInitParams &param) : MechTemp(param) {
        init_values.insert({gbegl36, 1.0});
        
        // Variable indexing - need to check CPP file for exact order
        var_in_coredata_idx.insert({gbegl36, 0});
        var_in_coredata_idx.insert({mfinf, 1});
        var_in_coredata_idx.insert({mminf, 2});
        var_in_coredata_idx.insert({msinf, 3});
        var_in_coredata_idx.insert({tmf, 4});
        var_in_coredata_idx.insert({tmm, 5});
        var_in_coredata_idx.insert({tms, 6});
        var_in_coredata_idx.insert({mf, 7});
        var_in_coredata_idx.insert({mm, 8});
        var_in_coredata_idx.insert({ms, 9});
        var_in_coredata_idx.insert({ek, 10});
        var_in_coredata_idx.insert({ik, 11});
        
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        vars(mf) = vars(mfinf);
        vars(mm) = vars(mminf);
        vars(ms) = vars(msinf);
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        
        // Using performance optimization for multiple variable access
        double mf_val = vars(mf);
        double mm_val = vars(mm);
        double ms_val = vars(ms);
        vars(ik) = vars(gbegl36) * (0.33 * mf_val + 0.36 * mm_val + 0.39 * ms_val) * (param.volt + 80.0);
        
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        return vars(ik);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        
        double dt = param.dt;
        vars(mf) = vars(mf) + (1.0 - exp(-dt / vars(tmf))) * (vars(mfinf) - vars(mf));
        vars(mm) = vars(mm) + (1.0 - exp(-dt / vars(tmm))) * (vars(mminf) - vars(mm));
        vars(ms) = vars(ms) + (1.0 - exp(-dt / vars(tms))) * (vars(msinf) - vars(ms));
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        double vhm = 63.0, ka = 28.5;
        double atms = 355.0, atmm = 63.0, atmf = 13.0;
        
        double inf_val = 1.0 / (1.0 + exp(-(v - vhm) / ka));
        vars(mfinf) = inf_val;
        vars(mminf) = inf_val;
        vars(msinf) = inf_val;
        vars(tmf) = atmf;
        vars(tmm) = atmm;
        vars(tms) = atms;
    }
};

REGISTER_MECHANISM("egl36", EGL36_Channel);

} // namespace egl36_worm