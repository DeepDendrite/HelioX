#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_K_P_mech {

static const char *MECH_NAME_TO_REG = "K_P";

struct MechTrait {
    enum class VarNames {
        gbar, ik, g, m, h, ek,
        mInf, mTau, hInf, hTau,
        Dm, Dh, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class K_P final : public MechTemp<K_P, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 21.0) / 10.0);
        constexpr double vshift = 0.0;
        constexpr double tauF = 1.0;

        vars(mInf) = 1.0 / (1.0 + exp(-(volt - (-14.3 + vshift)) / 14.6));
        if (volt < -50.0 + vshift) {
            vars(mTau) = tauF * (1.25 + 175.03 * exp(-(volt - vshift) * -0.026)) / qt;
        } else {
            vars(mTau) = tauF * (1.25 + 13.0 * exp(-(volt - vshift) * 0.026)) / qt;
        }
        vars(hInf) = 1.0 / (1.0 + exp(-(volt - (-54.0 + vshift)) / -11.0));
        vars(hTau) = (360.0 + (1010.0 + 24.0 * (volt - (-55.0 + vshift))) *
                                exp(-pow((volt - (-75.0 + vshift)) / 48.0, 2.0))) /
                    qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    explicit K_P(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({ek, 5});
        var_in_coredata_idx.insert({mInf, 6});
        var_in_coredata_idx.insert({mTau, 7});
        var_in_coredata_idx.insert({hInf, 8});
        var_in_coredata_idx.insert({hTau, 9});
        var_in_coredata_idx.insert({Dm, 10});
        var_in_coredata_idx.insert({Dh, 11});
        var_in_coredata_idx.insert({v, 12});
        var_in_coredata_idx.insert({_g, 13});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
        vars(Dm) = 0.0;
        vars(Dh) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(m) * vars(m) * vars(h);
        vars(ik) = vars(g) * (param.volt - vars(ek));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return vars(ik);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);
        if (vars(mTau) > 0.0) {
            double exp_m = exp(-param.dt / vars(mTau));
            double dm = (1.0 - exp_m) * (vars(mInf) - vars(m));
            vars(m) += dm;
            vars(Dm) = dm / param.dt;
        } else {
            vars(Dm) = 0.0;
        }

        if (vars(hTau) > 0.0) {
            double exp_h = exp(-param.dt / vars(hTau));
            double dh = (1.0 - exp_h) * (vars(hInf) - vars(h));
            vars(h) += dh;
            vars(Dh) = dh / param.dt;
        } else {
            vars(Dh) = 0.0;
        }
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, K_P);

} // namespace l5pc_K_P_mech
