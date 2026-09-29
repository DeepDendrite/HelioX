#include "mech_template.cuh"

#include <cassert>

namespace l5pc_cad2_mech {

static const char *MECH_NAME_TO_REG = "cad2";

struct MechTrait {
    enum class VarNames { ca, cai, Dca, ica, drive_channel, v, _g };
    enum class GlobalVarNames { depth, taur, cainf };
    enum class IonVarNames { _ion_ica, _ion_cai };
};

class cad2 final : public MechTemp<cad2, MechTrait> {
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | WRITE_EION_IN_STATE;

    explicit cad2(MechInitParams &param) : MechTemp(param) {
        var_in_coredata_idx.insert({ca, 0});
        var_in_coredata_idx.insert({cai, 1});
        var_in_coredata_idx.insert({Dca, 2});
        var_in_coredata_idx.insert({ica, 3});
        var_in_coredata_idx.insert({drive_channel, 4});
        var_in_coredata_idx.insert({v, 5});
        var_in_coredata_idx.insert({_g, 6});

        global_info_map.insert({depth, {"depth_cad2"}});
        global_info_map.insert({taur, {"taur_cad2"}});
        global_info_map.insert({cainf, {"cainf_cad2"}});

        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &, VarAccessor<MechTrait> &vars) {
        vars(ca) = vars(cainf);
        vars(cai) = vars(ca);
        vars(_ion_cai) = vars(ca);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        vars(ica) = vars(_ion_ica);
        constexpr double FARADAY = 96485.3321233100184;
        vars(drive_channel) = -(10000.0) * vars(ica) / (2.0 * FARADAY * vars(depth));
        if (vars(drive_channel) <= 0.0) {
            vars(drive_channel) = 0.0;
        }

        double tau_r = vars(taur);
        if (tau_r <= 0.0) {
            return;
        }
        vars(ca) = (vars(ca) + param.dt * (vars(drive_channel) + vars(cainf) / tau_r)) /
                   (1.0 + param.dt / tau_r);
        vars(cai) = vars(ca);
        vars(_ion_cai) = vars(ca);
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, cad2);

} // namespace l5pc_cad2_mech
