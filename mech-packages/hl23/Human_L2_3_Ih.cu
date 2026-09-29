// Human_L2_3_Ih – translated from NEURON (NONSPECIFIC_CURRENT ihcn)
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_Ih {

struct MechTrait {
    enum class VarNames {
        gbar, shift1, shift2, shift3, shift4, shift5, shift6,
        ihcn, g, m, mInf, mTau, mAlpha, mBeta
    };
    enum class GlobalVarNames { ehcn };
};

class Ih : public MechTemp<Ih, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    Ih(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
        init_values.insert({shift1, 154.9});
        init_values.insert({shift2, 11.9});
        init_values.insert({shift3, 0.0});
        init_values.insert({shift4, 33.1});
        init_values.insert({shift5, 6.43});
        init_values.insert({shift6, 193.0});

        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({shift1, 1});
        var_in_coredata_idx.insert({shift2, 2});
        var_in_coredata_idx.insert({shift3, 3});
        var_in_coredata_idx.insert({shift4, 4});
        var_in_coredata_idx.insert({shift5, 5});
        var_in_coredata_idx.insert({shift6, 6});
        var_in_coredata_idx.insert({ihcn, 7});
        var_in_coredata_idx.insert({g, 8});
        var_in_coredata_idx.insert({m, 9});
        var_in_coredata_idx.insert({mInf, 10});
        var_in_coredata_idx.insert({mTau, 11});
        var_in_coredata_idx.insert({mAlpha, 12});
        var_in_coredata_idx.insert({mBeta, 13});

        global_info_map.insert({ehcn, {"ehcn_Ih"}});
        // Match NEURON mod default for this GLOBAL parameter.
        if (!coreneuron::global_var_map.contains("ehcn_Ih")) {
            coreneuron::global_var_map["ehcn_Ih"] = std::vector<double>{-45.0};
        }
        assert(param.name == std::string("Ih"));
    }

    DUAL_EXEC void rates(double v, double s1, double s2, double s3, double s4, double s5, double s6,
                                double& mInf, double& mTau, double& mAlpha, double& mBeta) {
        if (v == -s1) v += 1e-4;
        if (s4 == 0.0) s4 += 1e-4;
        if (s2 == 0.0) s2 += 1e-4;
        mAlpha = 0.001 * s5 * (v + s1) / (exp((v + s1) / s2) - 1.0);
        mBeta  = 0.001 * s6 * exp((v + s3) / s4);
        mInf = mAlpha / (mAlpha + mBeta);
        mTau = 1.0 / (mAlpha + mBeta);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        double _mInf,_mTau,_mA,_mB;
        rates(p.volt, vars(shift1), vars(shift2), vars(shift3), vars(shift4), vars(shift5), vars(shift6),
              _mInf,_mTau,_mA,_mB);
        vars(mInf)=_mInf;vars(mTau)=_mTau;vars(mAlpha)=_mA;vars(mBeta)=_mB;
        vars(m)=_mInf;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& p, VarAccessor<MechTrait>& vars) {
        vars(g) = vars(gbar) * vars(m);
        vars(ihcn) = vars(g) * (p.volt - vars(ehcn));
        return vars(ihcn);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        double _mInf,_mTau,_mA,_mB;
        rates(p.volt, vars(shift1), vars(shift2), vars(shift3), vars(shift4), vars(shift5), vars(shift6),
              _mInf,_mTau,_mA,_mB);
        vars(mInf)=_mInf;vars(mTau)=_mTau;vars(mAlpha)=_mA;vars(mBeta)=_mB;
        vars(m) = vars(m) + (1.0 - exp(-p.dt / _mTau)) * (_mInf - vars(m));
    }
};

REGISTER_MECHANISM("Ih", Ih);

} // namespace Human_L2_3_Ih
