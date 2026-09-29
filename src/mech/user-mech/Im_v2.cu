// Im_v2 mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cassert>
#include <cmath>

namespace Im_v2_mech {

static const char *MECH_NAME_TO_REG = "Im_v2";

struct MechTrait {
    enum class VarNames {
        gbar, ik, g, m, ek, mInf, mTau, mAlpha, mBeta, Dm, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Im_v2 final : public MechTemp<Im_v2, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC double update_m(double volt, double m_t, VarAccessor<MechTrait> &vars, double dt) {
        rates(volt, vars);
        const double tau = vars(mTau);
        if (!(tau > 0.0) || !std::isfinite(tau)) {
            return m_t;
        }
        const double decay = exp(-dt / tau);
        return m_t + (1.0 - decay) * (vars(mInf) - m_t);
    }

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 30.0) / 10.0);
        vars(mAlpha) = 0.007 * exp((6.0 * 0.4 * (volt - (-48.0))) / 26.12);
        vars(mBeta) = 0.007 * exp((-6.0 * (1.0 - 0.4) * (volt - (-48.0))) / 26.12);
        double sum = vars(mAlpha) + vars(mBeta);
        vars(mInf) = sum > 0.0 ? (vars(mAlpha) / sum) : 0.0;
        vars(mTau) = (15.0 + (sum > 0.0 ? (1.0 / sum) : 0.0)) / qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    explicit Im_v2(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        // Field order from NEURON-generated run_network/x86_64/Im_v2.cpp.
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

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars) {
        const double g = vars(gbar) * vars(m);
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
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, Im_v2);

} // namespace Im_v2_mech
