// Human_L2_3_SK – translated from NEURON (Ca-activated K current)
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_SK {

struct MechTrait {
    enum class VarNames { gbar, ik, g, z, ek, cai, zInf };
    enum class GlobalVarNames { zTau };
    enum class IonVarNames { _ion_ek, _ion_ik, _ion_cai };
};

class SK : public MechTemp<SK, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    SK(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gbar, 1e-06});
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({g, 2});
        var_in_coredata_idx.insert({z, 3});
        var_in_coredata_idx.insert({ek, 4});
        var_in_coredata_idx.insert({cai, 5});
        var_in_coredata_idx.insert({zInf, 6});

        global_info_map.insert({zTau, {"zTau_SK"}});
        // Match NEURON mod default for this GLOBAL parameter.
        if (!coreneuron::global_var_map.contains("zTau_SK")) {
            coreneuron::global_var_map["zTau_SK"] = std::vector<double>{1.0};
        }
        ion_var_map.insert({_ion_ek,  {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik,  {"k_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
        assert(param.name == std::string("SK"));
    }

    DUAL_EXEC void rates(double cai, double& zInf) {
        double zInf_d;
        if (cai < 1e-4) { zInf_d = 0.0; }
        else { zInf_d = 1.0 / (1.0 + pow(0.00043 / cai, 4.8)); }
        zInf = zInf_d;
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);
        rates(vars(cai), vars(zInf));
        vars(z) = vars(zInf);
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);
        vars(g) = vars(gbar) * vars(z);
        vars(ik) = vars(g) * (p.volt - vars(ek));
        if (p.updateIon) { mechAtomAdd(&vars(_ion_ik), vars(ik)); }
        return vars(ik);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        vars(ek) = vars(_ion_ek);
        vars(cai) = vars(_ion_cai);
        rates(vars(cai), vars(zInf));
        const double zTau_ = vars(zTau);
        vars(z) = vars(z) + (1.0 - exp(-p.dt / zTau_)) * (vars(zInf) - vars(z));
    }
};

REGISTER_MECHANISM("SK", SK);

} // namespace Human_L2_3_SK
