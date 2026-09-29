// Human_L2_3_NaTg – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_NaTg {

struct MechTrait {
    enum class VarNames {
        gbar, vshifth, vshiftm, slopeh, slopem,
        ina, g, m, h, ena,
        mInf, mTau, mAlpha, mBeta,
        hInf, hTau, hAlpha, hBeta
    };
    enum class IonVarNames { _ion_ena, _ion_ina };
};

class NaTg : public MechTemp<NaTg, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    NaTg(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
        init_values.insert({vshifth, 0.0});
        init_values.insert({vshiftm, 0.0});
        init_values.insert({slopeh, 6.0});
        init_values.insert({slopem, 6.0});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({vshifth, 1});
        var_in_coredata_idx.insert({vshiftm, 2});
        var_in_coredata_idx.insert({slopeh, 3});
        var_in_coredata_idx.insert({slopem, 4});
        var_in_coredata_idx.insert({ina, 5});
        var_in_coredata_idx.insert({g, 6});
        var_in_coredata_idx.insert({m, 7});
        var_in_coredata_idx.insert({h, 8});
        var_in_coredata_idx.insert({ena, 9});
        var_in_coredata_idx.insert({mInf, 10});
        var_in_coredata_idx.insert({mTau, 11});
        var_in_coredata_idx.insert({mAlpha, 12});
        var_in_coredata_idx.insert({mBeta, 13});
        var_in_coredata_idx.insert({hInf, 14});
        var_in_coredata_idx.insert({hTau, 15});
        var_in_coredata_idx.insert({hAlpha, 16});
        var_in_coredata_idx.insert({hBeta, 17});

        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
        assert(param.name == std::string("NaTg"));
    }

    DUAL_EXEC void rates(double v, double vshiftm, double vshifth, double slopem, double slopeh,
                                double& mInf, double& mTau, double& mAlpha, double& mBeta,
                                double& hInf, double& hTau, double& hAlpha, double& hBeta) {
        const double qt = pow(2.3, (34.0 - 21.0) / 10.0);
        if (v == (-38.0 + vshiftm)) v += 1e-4;
        mAlpha = (0.182 * (v - (-38.0 + vshiftm))) / (1.0 - exp(-(v - (-38.0 + vshiftm)) / slopem));
        mBeta  = (0.124 * (-v + (-38.0 + vshiftm))) / (1.0 - exp(-(-v + (-38.0 + vshiftm)) / slopem));
        mTau   = (1.0 / (mAlpha + mBeta)) / qt;
        mInf   = mAlpha / (mAlpha + mBeta);

        if (v == (-66.0 + vshifth)) v += 1e-4;
        hAlpha = (-0.015 * (v - (-66.0 + vshifth))) / (1.0 - exp((v - (-66.0 + vshifth)) / slopeh));
        hBeta  = (-0.015 * (-v + (-66.0 + vshifth))) / (1.0 - exp((-v + (-66.0 + vshifth)) / slopeh));
        hTau   = (1.0 / (hAlpha + hBeta)) / qt;
        hInf   = hAlpha / (hAlpha + hBeta);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(ena) = vars(_ion_ena);
        double _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB;
        rates(p.volt, vars(vshiftm), vars(vshifth), vars(slopem), vars(slopeh),
              _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB);
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
        rates(p.volt, vars(vshiftm), vars(vshifth), vars(slopem), vars(slopeh),
              _mInf, _mTau, _mA, _mB, _hInf, _hTau, _hA, _hB);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(mAlpha) = _mA; vars(mBeta) = _mB;
        vars(hInf) = _hInf; vars(hTau) = _hTau; vars(hAlpha) = _hA; vars(hBeta) = _hB;
        const double dt = p.dt;
        vars(m) = vars(m) + (1.0 - exp(-dt / _mTau)) * (_mInf - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-dt / _hTau)) * (_hInf - vars(h));
    }
};

REGISTER_MECHANISM("NaTg", NaTg);

} // namespace Human_L2_3_NaTg
