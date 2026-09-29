// SK mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cassert>
#include <cmath>

namespace SK_mech {

static const char *MECH_NAME_TO_REG = "SK";

struct MechTrait {
    enum class VarNames { gbar, ik, g, z, ek, cai, zInf, Dz, v, _g };
    enum class IonVarNames { _ion_ek, _ion_ik, _ion_cai, _ion_cao };
};

class SK final : public MechTemp<SK, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC double zinf_from_cai(double cai_mM) {
        double ca = cai_mM;
        if (ca < 1e-7) {
            ca += 1e-7;
        }
        return 1.0 / (1.0 + pow((0.00043 / ca), 4.8));
    }

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    explicit SK(MechInitParams &param) : MechTemp(param) {
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
        ion_var_map.insert({_ion_cao, {"ca_ion", EionVarNames::conco}});

        // Field order from NEURON-generated run_network/x86_64/SK.cpp.
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({z, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({cai, 5});
        var_in_coredata_idx.insert({zInf, 6});
        var_in_coredata_idx.insert({Dz, 7});
        var_in_coredata_idx.insert({v, 8});
        var_in_coredata_idx.insert({_g, 9});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        (void)param;
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);
        vars(zInf) = zinf_from_cai(vars(cai));
        vars(z) = vars(zInf);
        vars(Dz) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(z);
        vars(ik) = vars(g) * (param.volt - vars(ek));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return vars(ik);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // zTau is a PARAMETER (not RANGE) in the original MOD file, with default 1ms.
        // Neuron471085845_modified does not override it, so we hardcode the default.
        constexpr double zTau_ms = 1.0;

        vars(cai) = vars(_ion_cai);
        vars(zInf) = zinf_from_cai(vars(cai));
        if (zTau_ms > 0.0) {
            double exp_z = exp(-param.dt / zTau_ms);
            double dz = (1.0 - exp_z) * (vars(zInf) - vars(z));
            vars(z) += dz;
            vars(Dz) = dz / param.dt;
        } else {
            vars(Dz) = 0.0;
        }
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, SK);

} // namespace SK_mech
