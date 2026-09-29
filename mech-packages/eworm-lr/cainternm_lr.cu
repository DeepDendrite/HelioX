// cainternm_lr mechanism (worm-lr) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cmath>

namespace cainternm_lr {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        vcell,
        // ASSIGNED variables  
        pure_i, didv, ica, alpha, cai
    };
    
    enum class GlobalVarNames {
        f, tca, caeq
    };
    
    enum class IonVarNames {
        _ion_ica, _ion_cai
    };
};

class CainternmLr : public MechTemp<CainternmLr, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    // Mechanism with init and state functions, writes to ion concentration
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | WRITE_EION_IN_STATE;
    
    CainternmLr(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({vcell, 31.16});
        
        // Register variable indices based on CPP setup_instance  
        var_in_coredata_idx.insert({vcell, 0});
        var_in_coredata_idx.insert({pure_i, 1});
        var_in_coredata_idx.insert({didv, 2});
        var_in_coredata_idx.insert({ica, 3});
        var_in_coredata_idx.insert({alpha, 4});
        var_in_coredata_idx.insert({cai, 5});
        
        // Register global variables
        global_info_map.insert({f, {"f_cainternm_lr"}});
        global_info_map.insert({tca, {"tca_cainternm_lr"}});
        global_info_map.insert({caeq, {"caeq_cainternm_lr"}});
        
        // Register ion variables
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
    }
    
    // Helper function for setparames procedure
    DUAL_EXEC void setparames(VarAccessor<MechTrait> &vars, double v) {
        const double FARADAY = 96485.0;
        if (v <= 60.0) { // eca = 60 mV
            vars(alpha) = (-vars(f) * vars(ica) * pow(10.0, 6.0) / (2.0 * FARADAY * vars(vcell))) - 
                         ((vars(cai) - vars(caeq)) / vars(tca));
        } else {
            vars(alpha) = -(vars(cai) - vars(caeq)) / vars(tca);
        }
    }
    
    // Initialize function - corresponds to INITIAL block
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Match NEURON init: read ion cai/ica, then call setparames
        vars(ica) = vars(_ion_ica);
        vars(cai) = vars(_ion_cai);
        setparames(vars, param.volt);
    }
    
    // State update function - match NEURON derivimplicit/Newton in nrn_newton_thread
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read current ion values (match NEURON/CoreNEURON ordering)
        vars(ica) = vars(_ion_ica);
        vars(cai) = vars(_ion_cai);

        const double old_cai = vars(cai);
        double current_cai = vars(cai);

        // Newton parameters (match NEURON/CoreNEURON)
        const int MAX_ITER = 50;
        const double CONVERGE = 1e-6;
        const double ZERO = 1e-8;

        for (int iter = 0; iter < MAX_ITER; iter++) {
            vars(cai) = current_cai;
            setparames(vars, param.volt);
            const double alpha_val = vars(alpha);

            const double F = (-current_cai + alpha_val * param.dt + old_cai) / param.dt;
            const double tca_val = vars(tca);
            const double dalpha_dcai = -1.0 / tca_val;
            const double J = (-1.0 + dalpha_dcai * param.dt) / param.dt;

            const double delta = F / J;
            current_cai -= delta;

            const double change = (fabs(old_cai) > ZERO) ? fabs(delta / old_cai) : fabs(delta);
            const double max_dev = fabs(F);
            if (change <= CONVERGE && max_dev <= ZERO) {
                break;
            }
        }

        vars(cai) = current_cai;
        vars(pure_i) = 0.0;
        vars(didv) = 0.0;
        vars(_ion_cai) = vars(cai);
    }

};

REGISTER_MECHANISM("cainternm_lr", CainternmLr);

} // namespace cainternm_lr
