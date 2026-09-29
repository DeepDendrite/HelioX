#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Kd_mech {

static const char *MECH_NAME_TO_REG = "Kd";

struct MechTrait {
    enum class VarNames {
        gbar, ik, g, m, h, ek,
        mInf, mTau, hInf, hTau,
        Dm, Dh, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Kd final : public MechTemp<Kd, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void update_gates(double volt,
                                double m_t,
                                double h_t,
                                VarAccessor<MechTrait> &vars,
                                double dt,
                                double& m_next,
                                double& h_next) {
        rates(volt, vars);
        if (vars(mTau) > 0.0 && std::isfinite(vars(mTau))) {
            const double exp_m = exp(-dt / vars(mTau));
            m_next = m_t + (1.0 - exp_m) * (vars(mInf) - m_t);
        } else {
            m_next = m_t;
        }
        if (vars(hTau) > 0.0 && std::isfinite(vars(hTau))) {
            const double exp_h = exp(-dt / vars(hTau));
            h_next = h_t + (1.0 - exp_h) * (vars(hInf) - h_t);
        } else {
            h_next = h_t;
        }
    }

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        (void)volt;
        (void)vars;
        vars(mInf) = 1.0 - 1.0 / (1.0 + exp((vars(v) - (-43.0)) / 8.0));
        vars(mTau) = 1.0;
        vars(hInf) = 1.0 / (1.0 + exp((vars(v) - (-67.0)) / 7.3));
        vars(hTau) = 1500.0;
    }

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    explicit Kd(MechInitParams &param) : MechTemp(param) {
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
        vars(v) = param.volt;
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
        vars(Dm) = 0.0;
        vars(Dh) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ek) = vars(_ion_ek);
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(m) * vars(h);
        vars(ik) = vars(g) * (param.volt - vars(ek));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return vars(ik);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars) {
        const double g = vars(gbar) * vars(m) * vars(h);
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * g);
    }



    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        vars(v) = param.volt;
        rates(param.volt, vars);
        if (vars(mTau) > 0.0 && std::isfinite(vars(mTau))) {
            double exp_m = exp(-param.dt / vars(mTau));
            double dm = (1.0 - exp_m) * (vars(mInf) - vars(m));
            vars(m) += dm;
            vars(Dm) = dm / param.dt;
        } else {
            vars(Dm) = 0.0;
        }

        if (vars(hTau) > 0.0 && std::isfinite(vars(hTau))) {
            double exp_h = exp(-param.dt / vars(hTau));
            double dh = (1.0 - exp_h) * (vars(hInf) - vars(h));
            vars(h) += dh;
            vars(Dh) = dh / param.dt;
        } else {
            vars(Dh) = 0.0;
        }
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, Kd);

} // namespace l5pc_Kd_mech
