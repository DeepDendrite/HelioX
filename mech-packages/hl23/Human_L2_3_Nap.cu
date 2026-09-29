// Human_L2_3_Nap – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_Nap {

struct MechTrait {
    enum class VarNames {
        gbar, ina, g, m, h, ena,
        mInf, mTau, mAlpha, mBeta,
        hInf, hTau, hAlpha, hBeta
    };
    enum class IonVarNames { _ion_ena, _ion_ina };
};

class Nap : public MechTemp<Nap, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    Nap(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
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

        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
        assert(param.name == std::string("Nap"));
    }

    DUAL_EXEC void rates(double v,
                                double& mInf, double& mTau, double& mAlpha, double& mBeta,
                                double& hInf, double& hTau, double& hAlpha, double& hBeta) {
        const double qt = pow(2.3, (34.0 - 21.0) / 10.0);
        mInf = 1.0 / (1.0 + exp((v - -52.6) / -4.6));
        if (v == -38.0) v += 1e-4;
        mAlpha = (0.182 * (v - -38.0)) / (1.0 - exp(-(v - -38.0) / 6.0));
        mBeta  = (0.124 * (-v - 38.0)) / (1.0 - exp(-(-v - 38.0) / 6.0));
        mTau   = 6.0 * (1.0 / (mAlpha + mBeta)) / qt;
        if (v == -17.0) v += 1e-4;
        if (v == -64.4) v += 1e-4;
        hInf = 1.0 / (1.0 + exp((v - -48.8) / 10.0));
        hAlpha = -2.88e-6 * (v + 17.0) / (1.0 - exp((v + 17.0) / 4.63));
        hBeta  =  6.94e-6 * (v + 64.4) / (1.0 - exp(-(v + 64.4) / 2.63));
        hTau   = (1.0 / (hAlpha + hBeta)) / qt;
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(ena) = vars(_ion_ena);
        double _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB;
        rates(p.volt, _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(mAlpha) = _mA; vars(mBeta) = _mB;
        vars(hInf) = _hInf; vars(hTau) = _hTau; vars(hAlpha) = _hA; vars(hBeta) = _hB;
        vars(m) = _mInf; vars(h) = _hInf;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& p, VarAccessor<MechTrait>& vars) {
        vars(ena) = vars(_ion_ena);
        vars(g) = vars(gbar) * vars(m) * vars(m) * vars(m) * vars(h);
        vars(ina) = vars(g) * (p.volt - vars(ena));
        if (p.updateIon) { mechAtomAdd(&vars(_ion_ina), vars(ina)); }
        return vars(ina);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        vars(ena) = vars(_ion_ena);
        double _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB;
        rates(p.volt, _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(mAlpha) = _mA; vars(mBeta) = _mB;
        vars(hInf) = _hInf; vars(hTau) = _hTau; vars(hAlpha) = _hA; vars(hBeta) = _hB;
        const double dt = p.dt;
        vars(m) = vars(m) + (1.0 - exp(-dt / _mTau)) * (_mInf - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-dt / _hTau)) * (_hInf - vars(h));
    }
};

REGISTER_MECHANISM("Nap", Nap);

} // namespace Human_L2_3_Nap
