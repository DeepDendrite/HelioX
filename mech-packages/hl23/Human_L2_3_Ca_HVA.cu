// Human_L2_3_Ca_HVA – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_Ca_HVA {

struct MechTrait {
    enum class VarNames {
        gbar, ica, g, m, h, eca,
        mInf, mTau, mAlpha, mBeta,
        hInf, hTau, hAlpha, hBeta
    };
    enum class IonVarNames { _ion_eca, _ion_ica };
};

class Ca_HVA : public MechTemp<Ca_HVA, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    Ca_HVA(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ica, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({h, 4});
        var_in_coredata_idx.insert({eca, 5});
        var_in_coredata_idx.insert({mInf, 6});
        var_in_coredata_idx.insert({mTau, 7});
        var_in_coredata_idx.insert({mAlpha, 8});
        var_in_coredata_idx.insert({mBeta, 9});
        var_in_coredata_idx.insert({hInf, 10});
        var_in_coredata_idx.insert({hTau, 11});
        var_in_coredata_idx.insert({hAlpha, 12});
        var_in_coredata_idx.insert({hBeta, 13});

        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        assert(param.name == std::string("Ca_HVA"));
    }

    DUAL_EXEC void rates(double v,
                                double& mInf, double& mTau, double& mAlpha, double& mBeta,
                                double& hInf, double& hTau, double& hAlpha, double& hBeta) {
        if (v == -27.0) v += 1e-4;
        mAlpha = 0.055 * (-27.0 - v) / (exp((-27.0 - v) / 3.8) - 1.0);
        mBeta  = 0.94 * exp((-75.0 - v) / 17.0);
        mInf = mAlpha / (mAlpha + mBeta);
        mTau = 1.0 / (mAlpha + mBeta);
        hAlpha = 0.000457 * exp((-13.0 - v) / 50.0);
        hBeta  = 0.0065 / (exp((-v - 15.0) / 28.0) + 1.0);
        hInf = hAlpha / (hAlpha + hBeta);
        hTau = 1.0 / (hAlpha + hBeta);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(eca) = vars(_ion_eca);
        double _mInf,_mTau,_mA,_mB,_hInf,_hTau,_hA,_hB;
        rates(p.volt, _mInf,_mTau,_mA,_mB,_hInf,_hTau,_hA,_hB);
        vars(mInf)=_mInf;vars(mTau)=_mTau;vars(mAlpha)=_mA;vars(mBeta)=_mB;
        vars(hInf)=_hInf;vars(hTau)=_hTau;vars(hAlpha)=_hA;vars(hBeta)=_hB;
        vars(m)=_mInf;vars(h)=_hInf;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& p, VarAccessor<MechTrait>& vars) {
        vars(eca) = vars(_ion_eca);
        vars(g) = vars(gbar) * vars(m) * vars(m) * vars(h);
        vars(ica) = vars(g) * (p.volt - vars(eca));
        if (p.updateIon) { mechAtomAdd(&vars(_ion_ica), vars(ica)); }
        return vars(ica);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        vars(eca) = vars(_ion_eca);
        double _mInf,_mTau,_mA,_mB,_hInf,_hTau,_hA,_hB;
        rates(p.volt, _mInf,_mTau,_mA,_mB,_hInf,_hTau,_hA,_hB);
        vars(mInf)=_mInf;vars(mTau)=_mTau;vars(mAlpha)=_mA;vars(mBeta)=_mB;
        vars(hInf)=_hInf;vars(hTau)=_hTau;vars(hAlpha)=_hA;vars(hBeta)=_hB;
        vars(m) = vars(m) + (1.0 - exp(-p.dt / _mTau)) * (_mInf - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-p.dt / _hTau)) * (_hInf - vars(h));
    }
};

REGISTER_MECHANISM("Ca_HVA", Ca_HVA);

} // namespace Human_L2_3_Ca_HVA
