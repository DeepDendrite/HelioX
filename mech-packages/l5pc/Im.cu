// Legacy L5PC-compatible Im mechanism.
#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Im_mech {

static const char *MECH_NAME_TO_REG = "Im";

struct MechTrait {
    enum class VarNames {
        gbar, ik, g, m, ek,
        mInf, mTau, mAlpha, mBeta, Dm,
        v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Im final : public MechTemp<Im, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 21.0) / 10.0);
        vars(mAlpha) = 3.3e-3 * exp(2.5 * 0.04 * (volt - (-35.0)));
        vars(mBeta) = 3.3e-3 * exp(-2.5 * 0.04 * (volt - (-35.0)));
        double sum = vars(mAlpha) + vars(mBeta);
        vars(mInf) = vars(mAlpha) / sum;
        vars(mTau) = (1.0 / sum) / qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    explicit Im(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({mInf, 5});
        var_in_coredata_idx.insert({mTau, 6});
        var_in_coredata_idx.insert({mAlpha, 7});
        var_in_coredata_idx.insert({mBeta, 8});
        var_in_coredata_idx.insert({Dm, 9});
        var_in_coredata_idx.insert({v, 10});
        var_in_coredata_idx.insert({_g, 11});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(Dm) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(m);
        vars(ik) = vars(g) * (param.volt - vars(ek));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return vars(ik);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);
        double tau = vars(mTau);
        if (tau <= 0.0) {
            vars(Dm) = 0.0;
            return;
        }
        double exp_term = exp(-param.dt / tau);
        double dm = (1.0 - exp_term) * (vars(mInf) - vars(m));
        vars(m) += dm;
        vars(Dm) = dm / param.dt;
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, Im);

} // namespace l5pc_Im_mech
