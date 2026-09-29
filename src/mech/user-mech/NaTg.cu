// NaTg mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace NaTg {

struct MechTrait {
    enum class VarNames {
        // Parameters (RANGE variables from MOD file, matching CPP index)
        gNaTgbar,     // 0
        vshifth,      // 1
        vshiftm,      // 2
        slopeh,       // 3
        slopem,       // 4
        // ASSIGNED variables
        ina,          // 5
        gNaTg,        // 6
        // STATE variables
        m,            // 7
        h,            // 8
        // Local copies of ion variables
        ena,          // 9
        // Intermediate variables
        mInf,         // 10
        mTau,         // 11
        mAlpha,       // 12
        mBeta,        // 13
        hInf,         // 14
        hTau,         // 15
        hAlpha,       // 16
        hBeta         // 17
    };

    enum class IonVarNames {
        _ion_ena,     // na_ion erev
        _ion_ina,     // na_ion current
        _ion_dinadv   // na_ion derivative
    };
};

class NaTg_Channel : public MechTemp<NaTg_Channel, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // This mechanism has STATE variables, so needs INIT, CURRENT, and STATE
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::ENABLE_STATE;

    NaTg_Channel(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values from MOD file
        init_values.insert({gNaTgbar, 0.00001});
        init_values.insert({vshifth, 0.0});
        init_values.insert({vshiftm, 0.0});
        init_values.insert({slopeh, 6.0});
        init_values.insert({slopem, 6.0});

        // Register variable indices - must match CPP file's _nrn_mechanism_register_data_fields
        // From CPP setup_instance: inst->gNaTgbar = ml->data+0*pnodecount, etc.
        var_in_coredata_idx.insert({gNaTgbar, 0});
        var_in_coredata_idx.insert({vshifth, 1});
        var_in_coredata_idx.insert({vshiftm, 2});
        var_in_coredata_idx.insert({slopeh, 3});
        var_in_coredata_idx.insert({slopem, 4});
        var_in_coredata_idx.insert({ina, 5});
        var_in_coredata_idx.insert({gNaTg, 6});
        var_in_coredata_idx.insert({m, 7});
        var_in_coredata_idx.insert({h, 8});

        // Register ion channel variables
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_dinadv, {"na_ion", EionVarNames::dcurdv}});
    }

    // Helper function to compute rates (equivalent to rates() procedure in MOD)
    DUAL_EXEC void compute_rates(double v, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, ((34.0 - 21.0) / 10.0));

        // Avoid division by zero
        double v_m = v;
        if (v_m == -38.0) {
            v_m = v_m + 0.0001;
        }

        vars(mAlpha) = (0.182 * (v_m - (-38.0 + vars(vshiftm)))) /
                       (1.0 - exp(-(v_m - (-38.0 + vars(vshiftm))) / vars(slopem)));
        vars(mBeta) = (0.124 * (-v_m + (-38.0 + vars(vshiftm)))) /
                      (1.0 - exp(-(-v_m + (-38.0 + vars(vshiftm))) / vars(slopem)));
        vars(mTau) = (1.0 / (vars(mAlpha) + vars(mBeta))) / qt;
        vars(mInf) = vars(mAlpha) / (vars(mAlpha) + vars(mBeta));

        double v_h = v;
        if (v_h == -66.0) {
            v_h = v_h + 0.0001;
        }

        vars(hAlpha) = (-0.015 * (v_h - (-66.0 + vars(vshifth)))) /
                       (1.0 - exp((v_h - (-66.0 + vars(vshifth))) / vars(slopeh)));
        vars(hBeta) = (-0.015 * (-v_h + (-66.0 + vars(vshifth)))) /
                      (1.0 - exp((-v_h + (-66.0 + vars(vshifth))) / vars(slopeh)));
        vars(hTau) = (1.0 / (vars(hAlpha) + vars(hBeta))) / qt;
        vars(hInf) = vars(hAlpha) / (vars(hAlpha) + vars(hBeta));
    }

    // Initialize STATE variables (from INITIAL block in MOD and nrn_init in CPP)
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);

        // Compute rates at initial voltage
        compute_rates(param.volt, vars);

        // Initialize STATE variables to steady state (from CPP line 395-396)
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
    }

    // Compute current (from BREAKPOINT block in MOD and nrn_current in CPP)
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);

        // Calculate conductance: gNaTg = gNaTgbar * m^3 * h
        vars(gNaTg) = vars(gNaTgbar) * vars(m) * vars(m) * vars(m) * vars(h);

        // Calculate current: ina = gNaTg * (v - ena)
        vars(ina) = vars(gNaTg) * (param.volt - vars(ena));

        // Update global ion current (only when updateIon is true)
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
        }

        return vars(ina);
    }

    // Update STATE variables (from DERIVATIVE block with cnexp method)
    // CRITICAL: Use exact integration formula from CPP file line 492-493
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read ion reversal potential
        vars(ena) = vars(_ion_ena);

        // Compute rates at current voltage
        compute_rates(param.volt, vars);

        // Update STATE variables using cnexp integration from CPP
        // From CPP line 492: inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * ((((-1.0))) / inst->mTau[id]))) *
        //                                 (-(((inst->mInf[id])) / inst->mTau[id]) / ((((-1.0))) / inst->mTau[id]) - inst->m[id]);
        vars(m) = vars(m) + (1.0 - exp(param.dt * (-1.0 / vars(mTau)))) *
                  (-(vars(mInf) / vars(mTau)) / (-1.0 / vars(mTau)) - vars(m));

        vars(h) = vars(h) + (1.0 - exp(param.dt * (-1.0 / vars(hTau)))) *
                  (-(vars(hInf) / vars(hTau)) / (-1.0 / vars(hTau)) - vars(h));
    }
};

// Register mechanism with name matching MOD file SUFFIX
REGISTER_MECHANISM("NaTg_origin", NaTg_Channel);

} // namespace NaTg
