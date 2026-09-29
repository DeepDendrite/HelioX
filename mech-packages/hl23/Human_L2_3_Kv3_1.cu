// Human_L2_3_Kv3_1 – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_Kv3_1 {

struct MechTrait {
    enum class VarNames { gbar, ik, g, m, ek, mInf, mTau };
    enum class GlobalVarNames { vshift };
    enum class IonVarNames { _ion_ek, _ion_ik };
};

class Kv3_1 : public MechTemp<Kv3_1, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    Kv3_1(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-05});
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({m, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({mInf, 5});
        var_in_coredata_idx.insert({mTau, 6});

        global_info_map.insert({vshift, {"vshift_Kv3_1"}});
        // Match NEURON mod default for this GLOBAL parameter.
        if (!coreneuron::global_var_map.contains("vshift_Kv3_1")) {
            coreneuron::global_var_map["vshift_Kv3_1"] = std::vector<double>{0.0};
        }
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});
        assert(param.name == std::string("Kv3_1"));
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        double vs = vars(vshift);
        vars(mInf) = 1.0 / (1.0 + exp((p.volt - (18.700 + vs)) / (-9.700)));
        vars(mTau) = 0.2 * 20.0 / (1.0 + exp((p.volt - (-46.560 + vs)) / (-44.140)));
        vars(m) = vars(mInf);
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
        double vs = vars(vshift);
        double mInf_ = 1.0 / (1.0 + exp((p.volt - (18.700 + vs)) / (-9.700)));
        double mTau_ = 0.2 * 20.0 / (1.0 + exp((p.volt - (-46.560 + vs)) / (-44.140)));
        vars(mInf) = mInf_; vars(mTau) = mTau_;
        vars(m) = vars(m) + (1.0 - exp(-p.dt / mTau_)) * (mInf_ - vars(m));
    }
};

REGISTER_MECHANISM("Kv3_1", Kv3_1);

} // namespace Human_L2_3_Kv3_1
