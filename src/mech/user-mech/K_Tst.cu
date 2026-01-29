// K_Tst mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace K_Tst {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables (index 0)
        gK_Tstbar,       // 0
        // ASSIGNED variables
        ik,              // 1
        gK_Tst,          // 2
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

class K_Tst_Channel : public MechTemp<K_Tst_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has INIT, CURRENT, and STATE functions
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    K_Tst_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default values from MOD file
        init_values.insert({gK_Tstbar, 0.00001});

        // Register variable indices (must match CPP setup_instance order)
        var_in_coredata_idx.insert({gK_Tstbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({gK_Tst, 2});
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

        printf_debug("K_Tst Mechanism constructed with name=%s\n", param.name.c_str());
    }

    // rates function (helper)
    DUAL_EXEC void rates(double v_param, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, ((34.0 - 21.0) / 10.0));
        double v = v_param + 10.0;

        vars(mInf) = 1.0 / (1.0 + exp(-(v + 0.0) / 19.0));
        vars(mTau) = (0.34 + 0.92 * exp(-pow(((v + 71.0) / 59.0), 2.0))) / qt;
        vars(hInf) = 1.0 / (1.0 + exp(-(v + 66.0) / (-10.0)));
        vars(hTau) = (8.0 + 49.0 * exp(-pow(((v + 73.0) / 23.0), 2.0))) / qt;
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

        // Calculate conductance and current (note: m^4 from MOD file line 40)
        vars(gK_Tst) = vars(gK_Tstbar) * pow(vars(m), 4.0) * vars(h);
        vars(ik) = vars(gK_Tst) * (param.volt - vars(ek));

        // Update ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        return vars(ik);
    }

    // Update STATE variables - CRITICAL: use exact formula from CPP line 440-441
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Calculate rates at current voltage
        rates(param.volt, vars);

        // Update STATE variables using cnexp integration (from CPP nrn_state function)
        // Line 440: inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->mTau[id]))) * ( -(((inst->mInf[id])) / inst->mTau[id]) / (((( -1.0))) / inst->mTau[id]) - inst->m[id]);
        vars(m) = vars(m) + (1.0 - exp(param.dt * ((-1.0) / vars(mTau)))) * (-(vars(mInf) / vars(mTau)) / ((-1.0) / vars(mTau)) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(param.dt * ((-1.0) / vars(hTau)))) * (-(vars(hInf) / vars(hTau)) / ((-1.0) / vars(hTau)) - vars(h));
    }
};

// Register mechanism with the name from MOD file SUFFIX
REGISTER_MECHANISM("K_Tst", K_Tst_Channel);

} // namespace K_Tst
