#include "l5pc_common.cuh"
#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Ih_mech {

static const char *MECH_NAME_TO_REG = "Ih";

struct MechTrait {
    enum class VarNames {
        gbar, ihcn, g,
        m, mInf, mTau, mAlpha, mBeta, Dm,
        v, _g
    };
};

class Ih final : public MechTemp<Ih, MechTrait> {
    using enum MechTrait::VarNames;

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
        vars(v) = volt;
        vars(mAlpha) = 0.001 * 6.43 * l5pc_vtrap(volt + 154.9, 11.9);
        vars(mBeta) = 0.001 * 193.0 * exp(volt / 33.1);
        double sum = vars(mAlpha) + vars(mBeta);
        vars(mInf) = vars(mAlpha) / sum;
        vars(mTau) = 1.0 / sum;
    }

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    explicit Ih(MechInitParams &param) : MechTemp(param) {
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ihcn, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({mInf, 4});
        var_in_coredata_idx.insert({mTau, 5});
        var_in_coredata_idx.insert({mAlpha, 6});
        var_in_coredata_idx.insert({mBeta, 7});
        var_in_coredata_idx.insert({Dm, 8});
        var_in_coredata_idx.insert({v, 9});
        var_in_coredata_idx.insert({_g, 10});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(Dm) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        constexpr double ehcn = -45.0;
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(m);
        vars(ihcn) = vars(g) * (param.volt - ehcn);
        return vars(ihcn);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars) {
        const double g = vars(gbar) * vars(m);
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * g);
    }



    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);
        double tau = vars(mTau);
        if (!(tau > 0.0) || !std::isfinite(tau)) {
            vars(Dm) = 0.0;
            return;
        }
        double exp_term = exp(-param.dt / tau);
        double dm = (1.0 - exp_term) * (vars(mInf) - vars(m));
        vars(m) += dm;
        vars(Dm) = dm / param.dt;
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, Ih);

} // namespace l5pc_Ih_mech
