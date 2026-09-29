#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Ca_LVA_mech {

static const char *MECH_NAME_TO_REG = "Ca_LVA";

struct MechTrait {
    enum class VarNames {
        gbar, ica, g, m, h, eca,
        mInf, mTau, hInf, hTau,
        Dm, Dh, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_eca, _ion_ica };
};

class Ca_LVA final : public MechTemp<Ca_LVA, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 21.0) / 10.0);
        double vshifted = volt + 10.0;
        vars(mInf) = 1.0 / (1.0 + exp((vshifted - (-30.0)) / -6.0));
        vars(mTau) = (5.0 + 20.0 / (1.0 + exp((vshifted - (-25.0)) / 5.0))) / qt;
        vars(hInf) = 1.0 / (1.0 + exp((vshifted - (-80.0)) / 6.4));
        vars(hTau) = (20.0 + 50.0 / (1.0 + exp((vshifted - (-40.0)) / 7.0))) / qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    explicit Ca_LVA(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ica, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({eca, 5});
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
        vars(eca) = vars(_ion_eca);
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
        vars(Dm) = vars(Dh) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(eca) = vars(_ion_eca);
        vars(v) = param.volt;
        vars(g) = vars(gbar) * vars(m) * vars(m) * vars(h);
        vars(ica) = vars(g) * (param.volt - vars(eca));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
        }
        return vars(ica);
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

REGISTER_MECHANISM(MECH_NAME_TO_REG, Ca_LVA);

} // namespace l5pc_Ca_LVA_mech
