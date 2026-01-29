// SK_E2 mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace SK_E2 {

struct MechTrait {
    enum class VarNames {
        // Parameters (RANGE variables from MOD file, matching CPP index)
        gSK_E2bar,    // 0
        // ASSIGNED variables
        ik,           // 1
        gSK_E2,       // 2
        // STATE variables
        z,            // 3
        // Local copies of ion variables
        ek,           // 4
        cai,          // 5
        // Intermediate variables
        zInf          // 6
    };

    enum class IonVarNames {
        _ion_ek,      // k_ion erev
        _ion_ik,      // k_ion current
        _ion_dikdv,   // k_ion derivative
        _ion_cai      // ca_ion internal concentration
    };

    enum class GlobalVarNames {
        zTau          // Global parameter from MOD file (default 1.0 ms)
    };
};

class SK_E2_Channel : public MechTemp<SK_E2_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    using enum MechTrait::GlobalVarNames;

    // This mechanism has STATE variables, so needs INIT, CURRENT, and STATE
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    SK_E2_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gSK_E2bar, 0.000001});

        // Register variable indices - must match CPP file's setup_instance
        var_in_coredata_idx.insert({gSK_E2bar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({gSK_E2, 2});
        var_in_coredata_idx.insert({z, 3});

        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});

        // Register global variable (from CPP line 95: {"zTau_SK_E2", &SK_E2_global.zTau})
        global_info_map.insert({zTau, {"zTau_SK_E2"}});
    }

    // Helper function to compute rates (equivalent to rates() procedure in MOD)
    DUAL_EXEC void compute_rates(double ca, VarAccessor<MechTrait> &vars) {
        // Avoid very small calcium values
        double ca_safe = ca;
        if (ca_safe < 1e-7) {
            ca_safe = ca_safe + 1e-07;
        }
        // From CPP line 296: inst->zInf[id] = 1.0 / (1.0 + pow((0.00043 / _lca), 4.8));
        vars(zInf) = 1.0 / (1.0 + pow((0.00043 / ca_safe), 4.8));
    }

    // Initialize STATE variables (from INITIAL block in MOD and nrn_init in CPP)
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion concentrations and reversal potentials
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);

        // Compute rates at initial calcium concentration
        compute_rates(vars(cai), vars);

        // Initialize STATE variable to steady state (from CPP line 340)
        vars(z) = vars(zInf);
    }

    // Compute current (from BREAKPOINT block in MOD and nrn_current in CPP)
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion concentrations and reversal potentials
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);

        // Calculate conductance: gSK_E2 = gSK_E2bar * z
        vars(gSK_E2) = vars(gSK_E2bar) * vars(z);

        // Calculate current: ik = gSK_E2 * (v - ek)
        vars(ik) = vars(gSK_E2) * (param.volt - vars(ek));

        // Update global ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        return vars(ik);
    }

    // Update STATE variables (from DERIVATIVE block with cnexp method)
    // CRITICAL: Use exact integration formula from CPP file line 428
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion concentrations and reversal potentials
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);

        // Compute rates at current calcium concentration
        compute_rates(vars(cai), vars);

        // Get global zTau value
        double zTau_val = vars(zTau);

        // Update STATE variable using cnexp integration from CPP
        // From CPP line 428: inst->z[id] = inst->z[id] + (1.0 - exp(nt->_dt * ((((-1.0))) / inst->global->zTau))) *
        //                                 (-(((inst->zInf[id])) / inst->global->zTau) / ((((-1.0))) / inst->global->zTau) - inst->z[id]);
        vars(z) = vars(z) + (1.0 - exp(param.dt * (-1.0 / zTau_val))) *
                  (-(vars(zInf) / zTau_val) / (-1.0 / zTau_val) - vars(z));
    }
};

// Register mechanism with name matching MOD file SUFFIX
REGISTER_MECHANISM("SK_E2", SK_E2_Channel);

} // namespace SK_E2
