// Ca_LVAst mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace Ca_LVAst {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables (index 0)
        gCa_LVAstbar,    // 0
        // ASSIGNED variables
        ica,             // 1
        gCa_LVAst,       // 2
        // STATE variables
        m,               // 3
        h,               // 4
        // Local copies of ion variables and intermediate values
        eca,             // 5
        mInf,            // 6
        mTau,            // 7
        hInf,            // 8
        hTau             // 9
    };

    enum class IonVarNames {
        _ion_eca,
        _ion_ica,
        _ion_dicadv
    };
};

class Ca_LVAst_Channel : public MechTemp<Ca_LVAst_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has INIT, CURRENT, and STATE functions
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    Ca_LVAst_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default values from MOD file
        init_values.insert({gCa_LVAstbar, 0.00001});

        // Register variable indices (must match CPP setup_instance order)
        var_in_coredata_idx.insert({gCa_LVAstbar, 0});
        var_in_coredata_idx.insert({ica, 1});
        var_in_coredata_idx.insert({gCa_LVAst, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({eca, 5});
        var_in_coredata_idx.insert({mInf, 6});
        var_in_coredata_idx.insert({mTau, 7});
        var_in_coredata_idx.insert({hInf, 8});
        var_in_coredata_idx.insert({hTau, 9});

        // Register ion channel variables
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dicadv, {"ca_ion", EionVarNames::dcurdv}});

        printf_debug("Ca_LVAst Mechanism constructed with name=%s\n", param.name.c_str());
    }

    // rates function (helper)
    DUAL_EXEC void rates(double v_param, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, ((34.0 - 21.0) / 10.0));
        double v = v_param + 10.0;
        vars(mInf) = 1.0 / (1.0 + exp((v - (-30.0)) / (-6.0)));
        vars(mTau) = (5.0 + 20.0 / (1.0 + exp((v - (-25.0)) / 5.0))) / qt;
        vars(hInf) = 1.0 / (1.0 + exp((v - (-80.0)) / 6.4));
        vars(hTau) = (20.0 + 50.0 / (1.0 + exp((v - (-40.0)) / 7.0))) / qt;
    }

    // Initialize STATE variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(eca) = vars(_ion_eca);

        // Initialize STATE variables to 0 first (NEURON default)
        vars(m) = 0.0;
        vars(h) = 0.0;

        // Calculate rates at initial voltage
        rates(param.volt, vars);

        // Set STATE variables to steady state values (from nrn_init)
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
    }

    // Calculate current
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(eca) = vars(_ion_eca);

        // Calculate conductance and current
        vars(gCa_LVAst) = vars(gCa_LVAstbar) * vars(m) * vars(m) * vars(h);
        vars(ica) = vars(gCa_LVAst) * (param.volt - vars(eca));

        // Update ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
        }

        return vars(ica);
    }

    // Update STATE variables - CRITICAL: use exact formula from CPP line 440-441
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(eca) = vars(_ion_eca);

        // Calculate rates at current voltage
        rates(param.volt, vars);

        // Update STATE variables using cnexp integration (from CPP nrn_state function)
        // Line 440: inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->mTau[id]))) * ( -(((inst->mInf[id])) / inst->mTau[id]) / (((( -1.0))) / inst->mTau[id]) - inst->m[id]);
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(mTau)))) * (-(vars(mInf) / vars(mTau)) / ((-1.0) / vars(mTau)) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(param.dt * ((-1.0) / vars(hTau)))) * (-(vars(hInf) / vars(hTau)) / ((-1.0) / vars(hTau)) - vars(h));
    }
};

// Register mechanism with the name from MOD file SUFFIX
REGISTER_MECHANISM("Ca_LVAst", Ca_LVAst_Channel);

} // namespace Ca_LVAst
