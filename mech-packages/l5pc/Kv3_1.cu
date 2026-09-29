#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Kv3_1_mech {

static const char *MECH_NAME_TO_REG = "Kv3_1";

struct MechTrait {
    enum class VarNames { gbar, ik, g, m, ek, mInf, mTau, Dm, v, _g };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Kv3_1 final : public MechTemp<Kv3_1, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void update_m(double volt, double m_t, VarAccessor<MechTrait> &vars, double dt, double& m_next) {
        rates(volt, vars);
        if (vars(mTau) > 0.0 && std::isfinite(vars(mTau))) {
            const double decay = exp(-dt / vars(mTau));
            m_next = m_t + (1.0 - decay) * (vars(mInf) - m_t);
        } else {
            m_next = m_t;
        }
    }

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        constexpr double vshift = 0.0;
        vars(mInf) = 1.0 / (1.0 + exp(((volt - (18.700 + vshift)) / (-9.700))));
        vars(mTau) = 0.2 * 20.0 / (1.0 + exp(((volt - (-46.560 + vshift)) / (-44.140))));
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    explicit Kv3_1(MechInitParams &param) : MechTemp(param) {
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({mInf, 5});
        var_in_coredata_idx.insert({mTau, 6});
        var_in_coredata_idx.insert({Dm, 7});
        var_in_coredata_idx.insert({v, 8});
        var_in_coredata_idx.insert({_g, 9});

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

REGISTER_MECHANISM(MECH_NAME_TO_REG, Kv3_1);

} // namespace l5pc_Kv3_1_mech
