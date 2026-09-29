#include "l5pc_common.cuh"
#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_NaTs_mech {

static const char *MECH_NAME_TO_REG = "NaTs";

struct MechTrait {
    enum class VarNames {
        gbar, ina, g, m, h, ena,
        mInf, mTau, mAlpha, mBeta,
        hInf, hTau, hAlpha, hBeta,
        Dm, Dh, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ena, _ion_ina };
};

class NaTs final : public MechTemp<NaTs, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 23.0) / 10.0);
        constexpr double malphaF = 0.182;
        constexpr double mbetaF = 0.124;
        constexpr double mvhalf = -40.0;
        constexpr double mk = 6.0;
        constexpr double halphaF = 0.015;
        constexpr double hbetaF = 0.015;
        constexpr double hvhalf = -66.0;
        constexpr double hk = 6.0;

        vars(mAlpha) = malphaF * l5pc_vtrap(-(volt - mvhalf), mk);
        vars(mBeta) = mbetaF * l5pc_vtrap((volt - mvhalf), mk);
        double msum = vars(mAlpha) + vars(mBeta);
        vars(mInf) = vars(mAlpha) / msum;
        vars(mTau) = (1.0 / msum) / qt;

        vars(hAlpha) = halphaF * l5pc_vtrap(volt - hvhalf, hk);
        vars(hBeta) = hbetaF * l5pc_vtrap(-(volt - hvhalf), hk);
        double hsum = vars(hAlpha) + vars(hBeta);
        vars(hInf) = vars(hAlpha) / hsum;
        vars(hTau) = (1.0 / hsum) / qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    explicit NaTs(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ina, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({ena, 5});
        var_in_coredata_idx.insert({mInf, 6});
        var_in_coredata_idx.insert({mTau, 7});
        var_in_coredata_idx.insert({mAlpha, 8});
        var_in_coredata_idx.insert({mBeta, 9});
        var_in_coredata_idx.insert({hInf, 10});
        var_in_coredata_idx.insert({hTau, 11});
        var_in_coredata_idx.insert({hAlpha, 12});
        var_in_coredata_idx.insert({hBeta, 13});
        var_in_coredata_idx.insert({Dm, 14});
        var_in_coredata_idx.insert({Dh, 15});
        var_in_coredata_idx.insert({v, 16});
        var_in_coredata_idx.insert({_g, 17});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        rates(param.volt, vars);
        vars(m) = vars(mInf);
        vars(h) = vars(hInf);
        vars(Dm) = vars(Dh) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        vars(v) = param.volt;
        double m2 = vars(m) * vars(m);
        vars(g) = vars(gbar) * m2 * vars(m) * vars(h);
        vars(ina) = vars(g) * (param.volt - vars(ena));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
        }
        return vars(ina);
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

REGISTER_MECHANISM(MECH_NAME_TO_REG, NaTs);

} // namespace l5pc_NaTs_mech
