// Im mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace Im {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables (index 0)
        gImbar,          // 0
        // ASSIGNED variables
        ik,              // 1
        gIm,             // 2
        // STATE variables
        m,               // 3
        // Local copies of ion variables and intermediate values
        ek,              // 4
        mInf,            // 5
        mTau,            // 6
        mAlpha,          // 7
        mBeta            // 8
    };

    enum class IonVarNames {
        _ion_ek,
        _ion_ik,
        _ion_dikdv
    };
};

class Im_Channel : public MechTemp<Im_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has INIT, CURRENT, and STATE functions
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    Im_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default values from MOD file
        init_values.insert({gImbar, 0.00001});

        // Register variable indices (must match CPP setup_instance order)
        var_in_coredata_idx.insert({gImbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({gIm, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({mInf, 5});
        var_in_coredata_idx.insert({mTau, 6});
        var_in_coredata_idx.insert({mAlpha, 7});
        var_in_coredata_idx.insert({mBeta, 8});

        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});

        printf_debug("Im Mechanism constructed with name=%s\n", param.name.c_str());
    }

    // rates function (helper)
    DUAL_EXEC void rates(double v, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, ((34.0 - 21.0) / 10.0));
        vars(mAlpha) = 3.3e-3 * exp(2.5 * 0.04 * (v - (-35.0)));
        vars(mBeta) = 3.3e-3 * exp(-2.5 * 0.04 * (v - (-35.0)));
        vars(mInf) = vars(mAlpha) / (vars(mAlpha) + vars(mBeta));
        vars(mTau) = (1.0 / (vars(mAlpha) + vars(mBeta))) / qt;
    }

    // Initialize STATE variables
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Initialize STATE variable to 0 first (NEURON default)
        vars(m) = 0.0;

        // Calculate rates at initial voltage
        rates(param.volt, vars);

        // Set STATE variable to steady state value (from nrn_init)
        vars(m) = vars(mInf);
    }

    // Calculate current
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Calculate conductance and current
        vars(gIm) = vars(gImbar) * vars(m);
        vars(ik) = vars(gIm) * (param.volt - vars(ek));

        // Update ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        return vars(ik);
    }

    // Update STATE variables - CRITICAL: use exact formula from CPP line 424
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Calculate rates at current voltage
        rates(param.volt, vars);

        // Update STATE variable using cnexp integration (from CPP nrn_state function)
        // Line 424: inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->mTau[id]))) * ( -(((inst->mInf[id])) / inst->mTau[id]) / (((( -1.0))) / inst->mTau[id]) - inst->m[id]);
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(mTau)))) * (-(vars(mInf) / vars(mTau)) / ((-1.0) / vars(mTau)) - vars(m));
    }
};

// Register mechanism with the name from MOD file SUFFIX
REGISTER_MECHANISM("Im", Im_Channel);

} // namespace Im
