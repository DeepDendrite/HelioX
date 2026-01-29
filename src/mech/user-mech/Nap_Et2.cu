// Nap_Et2 mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace Nap_Et2 {

struct MechTrait {
    enum class VarNames {
        // Parameters (RANGE variables from MOD file, matching CPP index)
        gNap_Et2bar,  // 0
        // ASSIGNED variables
        ina,          // 1
        gNap_Et2,     // 2
        // STATE variables
        m,            // 3
        h,            // 4
        // Local copies of ion variables
        ena,          // 5
        // Intermediate variables
        mInf,         // 6
        mTau,         // 7
        mAlpha,       // 8
        mBeta,        // 9
        hInf,         // 10
        hTau,         // 11
        hAlpha,       // 12
        hBeta         // 13
    };

    enum class IonVarNames {
        _ion_ena,     // na_ion erev
        _ion_ina,     // na_ion current
        _ion_dinadv   // na_ion derivative
    };
};

class Nap_Et2_Channel : public MechTemp<Nap_Et2_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has STATE variables, so needs INIT, CURRENT, and STATE
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    Nap_Et2_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gNap_Et2bar, 0.00001});

        // Register variable indices - must match CPP file's setup_instance
        var_in_coredata_idx.insert({gNap_Et2bar, 0});
        var_in_coredata_idx.insert({ina, 1});
        var_in_coredata_idx.insert({gNap_Et2, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});

        // Register ion channel variables
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dinadv, {"na_ion", EionVarNames::dcurdv}});
    }

    // Helper function to compute rates (equivalent to rates() procedure in MOD)
    DUAL_EXEC void compute_rates(double v, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, ((34.0 - 21.0) / 10.0));

        // mInf calculation
        vars(mInf) = 1.0 / (1.0 + exp((v - (-52.6)) / (-4.6)));

        // Avoid division by zero for mAlpha/mBeta
        double v_m = v;
        if (v_m == -38.0) {
            v_m = v_m + 0.0001;
        }

        vars(mAlpha) = (0.182 * (v_m - (-38.0))) / (1.0 - exp(-(v_m - (-38.0)) / 6.0));
        vars(mBeta) = (0.124 * (-v_m - 38.0)) / (1.0 - exp(-(-v_m - 38.0) / 6.0));
        vars(mTau) = 6.0 * (1.0 / (vars(mAlpha) + vars(mBeta))) / qt;

        // Avoid division by zero for hAlpha/hBeta
        double v_h1 = v;
        if (v_h1 == -17.0) {
            v_h1 = v_h1 + 0.0001;
        }
        double v_h2 = v;
        if (v_h2 == -64.4) {
            v_h2 = v_h2 + 0.0001;
        }

        vars(hInf) = 1.0 / (1.0 + exp((v - (-48.8)) / 10.0));
        vars(hAlpha) = -2.88e-6 * (v_h1 + 17.0) / (1.0 - exp((v_h1 + 17.0) / 4.63));
        vars(hBeta) = 6.94e-6 * (v_h2 + 64.4) / (1.0 - exp(-(v_h2 + 64.4) / 2.63));
        vars(hTau) = (1.0 / (vars(hAlpha) + vars(hBeta))) / qt;
    }

    // Initialize STATE variables (from INITIAL block in MOD and nrn_init in CPP)
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);

        // Compute rates at initial voltage
        compute_rates(param.volt, vars);

        // Initialize STATE variables to steady state (from CPP line 385-386)
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
    }

    // Compute current (from BREAKPOINT block in MOD and nrn_current in CPP)
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);

        // Calculate conductance: gNap_Et2 = gNap_Et2bar * m^3 * h
        vars(gNap_Et2) = vars(gNap_Et2bar) * vars(m) * vars(m) * vars(m) * vars(h);

        // Calculate current: ina = gNap_Et2 * (v - ena)
        vars(ina) = vars(gNap_Et2) * (param.volt - vars(ena));

        // Update global ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
        }

        return vars(ina);
    }

    // Update STATE variables (from DERIVATIVE block with cnexp method)
    // CRITICAL: Use exact integration formula from CPP file line 485-486
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);

        // Compute rates at current voltage
        compute_rates(param.volt, vars);

        // Update STATE variables using cnexp integration from CPP
        // From CPP line 485-486: exact same formula as NaTg but with Nap_Et2 parameters
        vars(m) = vars(m) + (1.0 - exp(param.dt * (-1.0 / vars(mTau)))) *
                  (-(vars(mInf) / vars(mTau)) / (-1.0 / vars(mTau)) - vars(m));

        vars(h) = vars(h) + (1.0 - exp(param.dt * (-1.0 / vars(hTau)))) *
                  (-(vars(hInf) / vars(hTau)) / (-1.0 / vars(hTau)) - vars(h));
    }
};

// Register mechanism with name matching MOD file SUFFIX
REGISTER_MECHANISM("Nap_Et2", Nap_Et2_Channel);

} // namespace Nap_Et2
