// Human_L2_3_Ca_LVA – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_Ca_LVA {

struct MechTrait {
    enum class VarNames { gbar, ica, g, m, h, eca, mInf, mTau, hInf, hTau };
    enum class IonVarNames { _ion_eca, _ion_ica };
};

class Ca_LVA : public MechTemp<Ca_LVA, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    Ca_LVA(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
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

        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        assert(param.name == std::string("Ca_LVA"));
    }

    DUAL_EXEC void rates(double v, double& mInf, double& mTau, double& hInf, double& hTau) {
        const double qt = pow(2.3, (34.0 - 21.0) / 10.0);
        double vv = v + 10.0;
        mInf = 1.0 / (1.0 + exp((vv - -30.0) / -6.0));
        mTau = (5.0 + 20.0 / (1.0 + exp((vv - -25.0) / 5.0))) / qt;
        hInf = 1.0 / (1.0 + exp((vv - -80.0) / 6.4));
        hTau = (20.0 + 50.0 / (1.0 + exp((vv - -40.0) / 7.0))) / qt;
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(eca) = vars(_ion_eca);
        double _mInf,_mTau,_hInf,_hTau;
        rates(p.volt, _mInf,_mTau,_hInf,_hTau);
        vars(mInf)=_mInf;vars(mTau)=_mTau;vars(hInf)=_hInf;vars(hTau)=_hTau;
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
        double _mInf,_mTau,_hInf,_hTau;
        rates(p.volt, _mInf,_mTau,_hInf,_hTau);
        vars(mInf)=_mInf;vars(mTau)=_mTau;vars(hInf)=_hInf;vars(hTau)=_hTau;
        vars(m) = vars(m) + (1.0 - exp(-p.dt / _mTau)) * (_mInf - vars(m));
        vars(h) = vars(h) + (1.0 - exp(-p.dt / _hTau)) * (_hInf - vars(h));
    }
};

REGISTER_MECHANISM("Ca_LVA", Ca_LVA);

} // namespace Human_L2_3_Ca_LVA
