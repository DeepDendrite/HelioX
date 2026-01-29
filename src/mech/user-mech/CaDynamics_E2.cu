// CaDynamics_E2 mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace CaDynamics_E2 {

//-------------------------------------
// 1) MechTrait definition
//-------------------------------------
struct MechTrait {
    enum class VarNames {
        // PARAMETER variables (RANGE in NEURON)
        gamma,      // 0 - percent of free calcium (not buffered)
        decay,      // 1 - rate of removal of calcium (ms)
        depth,      // 2 - depth of shell (um)
        minCai,     // 3 - minimum calcium concentration (mM)

        // ASSIGNED variables
        ica,        // 4 - calcium current (mA/cm2)

        // STATE variables
        cai         // 5 - internal calcium concentration (mM)
    };

    enum class IonVarNames {
        _ion_ica,   // calcium current from ion mechanism
        _ion_cai    // internal calcium concentration (to be written)
    };
};

//-------------------------------------
// 2) CaDynamics_E2 class definition
//-------------------------------------
class CaDynamics_E2_Mechanism : public MechTemp<CaDynamics_E2_Mechanism, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    // Flags: ENABLE_INIT (for STATE), ENABLE_STATE (for dynamics),
    //        WRITE_EION_IN_STATE (because we write to ion_cai in state function)
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_STATE |
        MechFlags::WRITE_EION_IN_STATE;

    //-------------------------------------
    // Constructor
    //-------------------------------------
    CaDynamics_E2_Mechanism(MechInitParams &param) : MechTemp(param) {
        // Set default parameter values (from MOD file PARAMETER block)
        init_values.insert({gamma, 0.05});
        init_values.insert({decay, 80.0});
        init_values.insert({depth, 0.1});
        init_values.insert({minCai, 1e-4});

        // Register variable indices (must match CPP file _nrn_mechanism_register_data_fields)
        // See CPP lines 196-204
        var_in_coredata_idx.insert({gamma, 0});
        var_in_coredata_idx.insert({decay, 1});
        var_in_coredata_idx.insert({depth, 2});
        var_in_coredata_idx.insert({minCai, 3});
        var_in_coredata_idx.insert({ica, 4});
        var_in_coredata_idx.insert({cai, 5});

        // Register ion variables
        // See CPP lines 314-316 and 346-349
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
    }

    //-------------------------------------
    // 3) INIT function
    //    Since there's no INITIAL block in MOD file,
    //    initialize STATE variable cai to 0.0
    //-------------------------------------
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // Read initial calcium current from ion mechanism
        vars(ica) = vars(_ion_ica);

        // Read initial internal calcium concentration from ion mechanism
        vars(cai) = vars(_ion_cai);

        // Write back to ion mechanism (as per CPP line 316)
        vars(_ion_cai) = vars(cai);
    }

    //-------------------------------------
    // 4) STATE function
    //    Update cai using the exact integration formula from CPP
    //    ⚠️ CRITICAL: This formula is extracted from CPP line 348
    //-------------------------------------
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // Read current calcium current from ion mechanism
        vars(ica) = vars(_ion_ica);

        // Read current internal calcium concentration from ion mechanism
        vars(cai) = vars(_ion_cai);

        // FARADAY constant (from CPP line 36)
        constexpr double FARADAY = 96485.3321233100184;

        // Extract the exact integration formula from CPP line 348
        // Original CPP formula:
        // inst->cai[id] = inst->cai[id] + (1.0 - exp(nt->_dt * (( -((1.0)) / inst->decay[id]))))
        //                 * ( -(( -(10000.0)) * ((((inst->ica[id]) * (inst->gamma[id])) / (2.0 * FARADAY * inst->depth[id])))
        //                 - ((( -inst->minCai[id]))) / inst->decay[id]) / (( -((1.0)) / inst->decay[id])) - inst->cai[id]);

        // Simplify the formula step by step:
        // Let tau = decay
        // dcai/dt = -(10000) * (ica * gamma / (2 * FARADAY * depth)) - (cai - minCai) / tau
        //         = -(10000) * (ica * gamma / (2 * FARADAY * depth)) - cai/tau + minCai/tau
        //
        // For cnexp integration:
        // cai_new = cai + (1 - exp(-dt/tau)) * (steady_state - cai)
        // where steady_state = tau * drive_ica / (1) + minCai
        //       drive_ica = -(10000) * (ica * gamma / (2 * FARADAY * depth))

        double gamma_val = vars(gamma);
        double decay_val = vars(decay);
        double depth_val = vars(depth);
        double minCai_val = vars(minCai);
        double ica_val = vars(ica);
        double cai_val = vars(cai);

        // Calculate the driving term for calcium influx
        double drive_ica = -(10000.0) * (ica_val * gamma_val) / (2.0 * FARADAY * depth_val);

        // cnexp integration formula (from CPP line 348)
        // Reorganized for clarity:
        double exp_factor = exp(param.dt * (-1.0 / decay_val));
        double steady_term = -(drive_ica - (-minCai_val) / decay_val) / (-1.0 / decay_val);

        vars(cai) = cai_val + (1.0 - exp_factor) * (steady_term - cai_val);

        // Write updated cai back to ion mechanism
        vars(_ion_cai) = vars(cai);
    }
};

// Register the mechanism with the name matching MOD file SUFFIX
REGISTER_MECHANISM("CaDynamics_E2", CaDynamics_E2_Mechanism);

} // namespace CaDynamics_E2
