#include "l5pc_common.cuh"
#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Kv2like_mech {

static const char *MECH_NAME_TO_REG = "Kv2like";

struct MechTrait {
    enum class VarNames {
        gbar, ik, g, m, h1, h2, ek,
        mInf, mAlpha, mBeta, mTau,
        hInf, h1Tau, h2Tau,
        Dm, Dh1, Dh2, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Kv2like final : public MechTemp<Kv2like, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void update_gates(double volt,
                                double m_t,
                                double h1_t,
                                double h2_t,
                                VarAccessor<MechTrait> &vars,
                                double dt,
                                double& m_next,
                                double& h1_next,
                                double& h2_next) {
        rates(volt, vars);
        if (vars(mTau) > 0.0 && std::isfinite(vars(mTau))) {
            const double exp_m = exp(-dt / vars(mTau));
            m_next = m_t + (1.0 - exp_m) * (vars(mInf) - m_t);
        } else {
            m_next = m_t;
        }
        if (vars(h1Tau) > 0.0 && std::isfinite(vars(h1Tau))) {
            const double exp_h1 = exp(-dt / vars(h1Tau));
            h1_next = h1_t + (1.0 - exp_h1) * (vars(hInf) - h1_t);
        } else {
            h1_next = h1_t;
        }
        if (vars(h2Tau) > 0.0 && std::isfinite(vars(h2Tau))) {
            const double exp_h2 = exp(-dt / vars(h2Tau));
            h2_next = h2_t + (1.0 - exp_h2) * (vars(hInf) - h2_t);
        } else {
            h2_next = h2_t;
        }
    }

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 21.0) / 10.0);
        vars(mAlpha) = 0.12 * l5pc_vtrap(-(volt - 43.0), 11.0);
        vars(mBeta) = 0.02 * exp(-(volt + 1.27) / 120.0);
        double sum = vars(mAlpha) + vars(mBeta);
        vars(mInf) = vars(mAlpha) / sum;
        vars(mTau) = 2.5 * (1.0 / (qt * sum));

        vars(hInf) = 1.0 / (1.0 + exp((volt + 58.0) / 11.0));
        vars(h1Tau) = (360.0 + (1010.0 + 23.7 * (volt + 54.0)) *
                                 exp(-pow((volt + 75.0) / 48.0, 2.0))) /
                     qt;
        vars(h2Tau) = (2350.0 + 1380.0 * exp(-0.011 * volt) - 210.0 * exp(-0.03 * volt)) / qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    explicit Kv2like(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h1, 4});
        var_in_coredata_idx.insert({h2, 5});
        var_in_coredata_idx.insert({ek, 6});
        var_in_coredata_idx.insert({mInf, 7});
        var_in_coredata_idx.insert({mAlpha, 8});
        var_in_coredata_idx.insert({mBeta, 9});
        var_in_coredata_idx.insert({mTau, 10});
        var_in_coredata_idx.insert({hInf, 11});
        var_in_coredata_idx.insert({h1Tau, 12});
        var_in_coredata_idx.insert({h2Tau, 13});
        var_in_coredata_idx.insert({Dm, 14});
        var_in_coredata_idx.insert({Dh1, 15});
        var_in_coredata_idx.insert({Dh2, 16});
        var_in_coredata_idx.insert({v, 17});
        var_in_coredata_idx.insert({_g, 18});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(h1) = vars(hInf);
        vars(h2) = vars(hInf);
        vars(Dm) = vars(Dh1) = vars(Dh2) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(m) * vars(m) * (0.5 * vars(h1) + 0.5 * vars(h2));
        vars(ik) = vars(g) * (param.volt - vars(ek));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return vars(ik);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars) {
        const double g = vars(gbar) * vars(m) * vars(m) * (0.5 * vars(h1) + 0.5 * vars(h2));
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * g);
    }



    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);

        if (vars(mTau) > 0.0 && std::isfinite(vars(mTau))) {
            double exp_m = exp(-param.dt / vars(mTau));
            double dm = (1.0 - exp_m) * (vars(mInf) - vars(m));
            vars(m) += dm;
            vars(Dm) = dm / param.dt;
        } else {
            vars(Dm) = 0.0;
        }

        if (vars(h1Tau) > 0.0 && std::isfinite(vars(h1Tau))) {
            double exp_h1 = exp(-param.dt / vars(h1Tau));
            double dh1 = (1.0 - exp_h1) * (vars(hInf) - vars(h1));
            vars(h1) += dh1;
            vars(Dh1) = dh1 / param.dt;
        } else {
            vars(Dh1) = 0.0;
        }

        if (vars(h2Tau) > 0.0 && std::isfinite(vars(h2Tau))) {
            double exp_h2 = exp(-param.dt / vars(h2Tau));
            double dh2 = (1.0 - exp_h2) * (vars(hInf) - vars(h2));
            vars(h2) += dh2;
            vars(Dh2) = dh2 / param.dt;
        } else {
            vars(Dh2) = 0.0;
        }
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, Kv2like);

} // namespace l5pc_Kv2like_mech
