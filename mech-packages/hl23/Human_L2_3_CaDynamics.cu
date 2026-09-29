// Human_L2_3_CaDynamics – translated from NEURON
#include "mech_template.cuh"
#include <cmath>

namespace Human_L2_3_CaDynamics {

struct MechTrait {
    enum class VarNames { gamma, decay, depth, minCai, ica, cai };
    enum class IonVarNames { _ion_ica, _ion_cai };
};

class CaDynamics : public MechTemp<CaDynamics, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    // Writes cai in STATE
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | WRITE_EION_IN_STATE;

    CaDynamics(MechInitParams& param) : MechTemp(param) {
        init_values.insert({gamma, 0.05});
        init_values.insert({decay, 80.0});
        init_values.insert({depth, 0.1});
        init_values.insert({minCai, 1e-4});

        var_in_coredata_idx.insert({gamma, 0});
        var_in_coredata_idx.insert({decay, 1});
        var_in_coredata_idx.insert({depth, 2});
        var_in_coredata_idx.insert({minCai, 3});
        var_in_coredata_idx.insert({ica, 4});
        var_in_coredata_idx.insert({cai, 5});

        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
        assert(param.name == std::string("CaDynamics"));
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& p, VarAccessor<MechTrait>& vars) {
        vars(cai) = vars(minCai);
        vars(_ion_cai) = vars(cai);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& p, VarAccessor<MechTrait>& vars) {
        vars(ica) = vars(_ion_ica);
        vars(cai) = vars(_ion_cai);
        const double FARADAY = 96485.33212; // coulomb/mol
        const double gamma_ = vars(gamma);
        const double decay_ = vars(decay);
        const double depth_ = vars(depth);
        const double minCai_ = vars(minCai);
        // cnexp form of: cai' = -(10000)*(ica*gamma/(2*FARADAY*depth)) - (cai - minCai)/decay
        const double a = -1.0 / decay_;
        const double b = -(10000.0) * ((vars(ica) * gamma_) / (2.0 * FARADAY * depth_)) - ( -minCai_ ) / decay_;
        vars(cai) = vars(cai) + (1.0 - exp(p.dt * a)) * ( -b / a - vars(cai) );
        vars(_ion_cai) = vars(cai);
    }
};

REGISTER_MECHANISM("CaDynamics", CaDynamics);

} // namespace Human_L2_3_CaDynamics
