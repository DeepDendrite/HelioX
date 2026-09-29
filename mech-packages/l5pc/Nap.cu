#include "l5pc_common.cuh"
#include "mech_template.cuh"

#include <cassert>
#include <cmath>

namespace l5pc_Nap_mech {

static const char *MECH_NAME_TO_REG = "Nap";

struct MechTrait {
    enum class VarNames {
        gbar, ina, g, h, ena,
        mInf, hInf, hTau, hAlpha, hBeta,
        Dh, v, _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ena, _ion_ina };
};

class Nap final : public MechTemp<Nap, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 21.0) / 10.0);
        vars(mInf) = 1.0 / (1.0 + exp((volt - (-52.6)) / -4.6));
        vars(hInf) = 1.0 / (1.0 + exp((volt - (-48.8)) / 10.0));
        vars(hAlpha) = 2.88e-6 * l5pc_vtrap(volt + 17.0, 4.63);
        vars(hBeta) = 6.94e-6 * l5pc_vtrap(-(volt + 64.4), 2.63);
        double sum = vars(hAlpha) + vars(hBeta);
        vars(hTau) = (1.0 / sum) / qt;
        vars(v) = volt;
    }

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    explicit Nap(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ina, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({h, 3});
        var_in_coredata_idx.insert({ena, 4});
        var_in_coredata_idx.insert({mInf, 5});
        var_in_coredata_idx.insert({hInf, 6});
        var_in_coredata_idx.insert({hTau, 7});
        var_in_coredata_idx.insert({hAlpha, 8});
        var_in_coredata_idx.insert({hBeta, 9});
        var_in_coredata_idx.insert({Dh, 10});
        var_in_coredata_idx.insert({v, 11});
        var_in_coredata_idx.insert({_g, 12});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        rates(param.volt, vars);
        vars(h) = vars(hInf);
        vars(Dh) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        // Match MOD BREAKPOINT semantics: after SOLVE states, Nap recomputes
        // rates() at the current voltage before forming g and ina.
        rates(param.volt, vars);
        vars(g) = vars(gbar) * vars(mInf) * vars(h);
        vars(ina) = vars(g) * (param.volt - vars(ena));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
        }
        return vars(ina);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);
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

REGISTER_MECHANISM(MECH_NAME_TO_REG, Nap);

} // namespace l5pc_Nap_mech
