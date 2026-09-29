// Human_L2_3_K_P (K_P) – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_K_P {

struct MechTrait {
    enum class VarNames {
        gbar, ik, g, m, h, ek,
        mInf, mTau, hInf, hTau
    };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class K_P : public MechTemp<K_P, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    K_P(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
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

        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        assert(param.name == std::string("K_P"));
    }

    DUAL_EXEC void rates(double v, double& mInf, double& mTau, double& hInf, double& hTau) {
        const double qt = pow(2.3, (34.0 - 21.0) / 10.0);
        double vv = v + 10.0;
        mInf = 1.0 / (1.0 + exp(-(vv + 1.0) / 12.0));
        if (vv < -50.0) {
            mTau = (1.25 + 175.03 * exp(-vv * -0.026)) / qt;
        } else {
            mTau = (1.25 + 13.0 * exp(-vv * 0.026)) / qt;
        }
        hInf = 1.0 / (1.0 + exp(-(vv + 54.0) / -11.0));
        hTau = (360.0 + (1010.0 + 24.0 * (vv + 55.0)) * exp(-pow((vv + 75.0) / 48.0, 2.0))) / qt;
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        double _mInf, _mTau, _hInf, _hTau;
        rates(p.volt, _mInf, _mTau, _hInf, _hTau);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(hInf) = _hInf; vars(hTau) = _hTau;
        vars(m) = _mInf; vars(h) = _hInf;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        vars(g) = vars(gbar) * vars(m) * vars(m) * vars(h);
        vars(ik) = vars(g) * (p.volt - vars(ek));
        if (p.updateIon) { mechAtomAdd(&vars(_ion_ik), vars(ik)); }
        return vars(ik);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        double _mInf, _mTau, _hInf, _hTau;
        rates(p.volt, _mInf, _mTau, _hInf, _hTau);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(hInf) = _hInf; vars(hTau) = _hTau;
        const double dt = p.dt;
        vars(m) = vars(m) + (1.0 - exp(-dt / _mTau)) * (_mInf - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-dt / _hTau)) * (_hInf - vars(h));
    }
};

REGISTER_MECHANISM("K_P", K_P);

} // namespace Human_L2_3_K_P
