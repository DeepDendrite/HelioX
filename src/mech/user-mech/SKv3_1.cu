// SKv3_1 mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace SKv3_1 {

struct MechTrait {
    enum class VarNames {
        // Parameters (RANGE variables from MOD file, matching CPP index)
        gSKv3_1bar,   // 0
        // ASSIGNED variables
        ik,           // 1
        gSKv3_1,      // 2
        // STATE variables
        m,            // 3
        // Local copies of ion variables
        ek,           // 4
        // Intermediate variables
        mInf,         // 5
        mTau          // 6
    };

    enum class IonVarNames {
        _ion_ek,      // k_ion erev
        _ion_ik,      // k_ion current
        _ion_dikdv    // k_ion derivative
    };
};

class SKv3_1_Channel : public MechTemp<SKv3_1_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has STATE variables, so needs INIT, CURRENT, and STATE
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    SKv3_1_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gSKv3_1bar, 0.00001});

        // Register variable indices - must match CPP file's setup_instance
        var_in_coredata_idx.insert({gSKv3_1bar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({gSKv3_1, 2});
        var_in_coredata_idx.insert({m, 3});

        // Register ion channel variables
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dikdv, {"k_ion", EionVarNames::dcurdv}});
    }

    // Helper function to compute rates (equivalent to rates() procedure in MOD)
    DUAL_EXEC void compute_rates(double v, VarAccessor<MechTrait> &vars) {
        // From CPP line 284-285:
        // inst->mInf[id] = 1.0 / (1.0 + exp(((v - (18.700)) / (-9.700))));
        // inst->mTau[id] = 0.2 * 20.0 / (1.0 + exp(((v - (-46.560)) / (-44.140))));
        vars(mInf) = 1.0 / (1.0 + exp((v - 18.700) / (-9.700)));
        vars(mTau) = 0.2 * 20.0 / (1.0 + exp((v - (-46.560)) / (-44.140)));
    }

    // Initialize STATE variables (from INITIAL block in MOD and nrn_init in CPP)
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Compute rates at initial voltage
        compute_rates(param.volt, vars);

        // Initialize STATE variable to steady state (from CPP line 324)
        vars(m) = vars(mInf);
    }

    // Compute current (from BREAKPOINT block in MOD and nrn_current in CPP)
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Calculate conductance: gSKv3_1 = gSKv3_1bar * m
        vars(gSKv3_1) = vars(gSKv3_1bar) * vars(m);

        // Calculate current: ik = gSKv3_1 * (v - ek)
        vars(ik) = vars(gSKv3_1) * (param.volt - vars(ek));

        // Update global ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        return vars(ik);
    }

    // Update STATE variables (from DERIVATIVE block with cnexp method)
    // CRITICAL: Use exact integration formula from CPP file line 406
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ek) = vars(_ion_ek);

        // Compute rates at current voltage
        compute_rates(param.volt, vars);

        // Update STATE variable using cnexp integration from CPP
        // From CPP line 406: inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * ((((-1.0))) / inst->mTau[id]))) *
        //                                 (-(((inst->mInf[id])) / inst->mTau[id]) / ((((-1.0))) / inst->mTau[id]) - inst->m[id]);
        vars(m) = vars(m) + (1.0 - exp(param.dt * (-1.0 / vars(mTau)))) *
                  (-(vars(mInf) / vars(mTau)) / (-1.0 / vars(mTau)) - vars(m));
    }
};

// Register mechanism with name matching MOD file SUFFIX
REGISTER_MECHANISM("SKv3_1", SKv3_1_Channel);

} // namespace SKv3_1
