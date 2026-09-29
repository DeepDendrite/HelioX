// Human_L2_3_Im – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_Im {

struct MechTrait {
    enum class VarNames { gbar, ik, g, m, ek, mInf, mTau, mAlpha, mBeta };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Im : public MechTemp<Im, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    Im(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({mInf, 5});
        var_in_coredata_idx.insert({mTau, 6});
        var_in_coredata_idx.insert({mAlpha, 7});
        var_in_coredata_idx.insert({mBeta, 8});

        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        assert(param.name == std::string("Im"));
    }

    DUAL_EXEC void rates(double v, double& mInf, double& mTau, double& mAlpha, double& mBeta) {
        const double qt = pow(2.3, (34.0 - 21.0) / 10.0);
        mAlpha = 3.3e-3 * exp(2.5 * 0.04 * (v - -35.0));
        mBeta  = 3.3e-3 * exp(-2.5 * 0.04 * (v - -35.0));
        mInf = mAlpha / (mAlpha + mBeta);
        mTau = (1.0 / (mAlpha + mBeta)) / qt;
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        double _mInf, _mTau, _mA, _mB;
        rates(p.volt, _mInf, _mTau, _mA, _mB);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(mAlpha) = _mA; vars(mBeta) = _mB;
        vars(m) = _mInf;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        vars(g) = vars(gbar) * vars(m);
        vars(ik) = vars(g) * (p.volt - vars(ek));
        if (p.updateIon) { mechAtomAdd(&vars(_ion_ik), vars(ik)); }
        return vars(ik);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        double _mInf, _mTau, _mA, _mB;
        rates(p.volt, _mInf, _mTau, _mA, _mB);
        vars(mInf) = _mInf; vars(mTau) = _mTau; vars(mAlpha) = _mA; vars(mBeta) = _mB;
        vars(m) = vars(m) + (1.0 - exp(-p.dt / _mTau)) * (_mInf - vars(m));
    }
};

REGISTER_MECHANISM("Im", Im);

} // namespace Human_L2_3_Im
