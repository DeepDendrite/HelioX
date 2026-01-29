// K_Pst mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace K_Pst {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables (index 0)
        gK_Pstbar,       // 0
        // ASSIGNED variables
        ik,              // 1
        gK_Pst,          // 2
        // STATE variables
        m,               // 3
        h,               // 4
        // Local copies of ion variables and intermediate values
        ek,              // 5
        mInf,            // 6
        mTau,            // 7
        hInf,            // 8
        hTau             // 9
    };

    enum class IonVarNames {
        _ion_ek,
        _ion_ik,
        _ion_dikdv
    };
};

class K_Pst_Channel : public MechTemp<K_Pst_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has INIT, CURRENT, and STATE functions
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    K_Pst_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default values from MOD file
        init_values.insert({gK_Pstbar, 0.00001});

        // Register variable indices (must match CPP setup_instance order)
        var_in_coredata_idx.insert({gK_Pstbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({gK_Pst, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({ek, 5});
        var_in_coredata_idx.insert({mInf, 6});
        var_in_coredata_idx.insert({mTau, 7});
        var_in_coredata_idx.insert({hInf, 8});
        var_in_coredata_idx.insert({hTau, 9});

        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});

        printf_debug("K_Pst Mechanism constructed with name=%s\n", param.name.c_str());
    }

    // rates function (helper)
    DUAL_EXEC void rates(double v_param, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, ((34.0 - 21.0) / 10.0));
        double v = v_param + 10.0;

        vars(mInf) = (1.0 / (1.0 + exp(-(v + 1.0) / 12.0)));

        if (v < -50.0) {
            vars(mTau) = (1.25 + 175.03 * exp(-v * (-0.026))) / qt;
        } else {
            vars(mTau) = ((1.25 + 13.0 * exp(-v * 0.026))) / qt;
        }

        vars(hInf) = 1.0 / (1.0 + exp(-(v + 54.0) / (-11.0)));
        vars(hTau) = (360.0 + (1010.0 + 24.0 * (v + 55.0)) * exp(-pow(((v + 75.0) / 48.0), 2.0))) / qt;
    }

    // Initialize STATE variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

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
        vars(ek) = vars(_ion_ek);

        // Calculate conductance and current
        vars(gK_Pst) = vars(gK_Pstbar) * vars(m) * vars(m) * vars(h);
        vars(ik) = vars(gK_Pst) * (param.volt - vars(ek));

        // Update ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        return vars(ik);
    }

    // Update STATE variables - CRITICAL: use exact formula from CPP line 452-453
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Calculate rates at current voltage
        rates(param.volt, vars);

        // Update STATE variables using cnexp integration (from CPP nrn_state function)
        // Line 452: inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->mTau[id]))) * ( -(((inst->mInf[id])) / inst->mTau[id]) / (((( -1.0))) / inst->mTau[id]) - inst->m[id]);
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(mTau)))) * (-(vars(mInf) / vars(mTau)) / ((-1.0) / vars(mTau)) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(param.dt * ((-1.0) / vars(hTau)))) * (-(vars(hInf) / vars(hTau)) / ((-1.0) / vars(hTau)) - vars(h));
    }
};

// Register mechanism with the name from MOD file SUFFIX
REGISTER_MECHANISM("K_Pst", K_Pst_Channel);

} // namespace K_Pst
