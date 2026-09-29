// Legacy L5PC-compatible IClamp_replay_gpu point process.
#include "mech_template.cuh"

#include <cassert>

namespace l5pc_IClamp_replay_gpu_mech {

static const char *MECH_NAME_TO_REG = "IClamp_replay_gpu";

struct MechTrait {
    enum class VarNames { i, index, i_len, t_len, v, _g };
};

class IClamp_replay_gpu final : public MechTemp<IClamp_replay_gpu, MechTrait> {
    using enum MechTrait::VarNames;

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT;

    explicit IClamp_replay_gpu(MechInitParams &param) : MechTemp(param) {
        var_in_coredata_idx.insert({i, 0});
        var_in_coredata_idx.insert({index, 1});
        var_in_coredata_idx.insert({i_len, 2});
        var_in_coredata_idx.insert({t_len, 3});
        var_in_coredata_idx.insert({v, 6});
        var_in_coredata_idx.insert({_g, 7});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &, VarAccessor<MechTrait> &vars) {
        vars(i) = 0.0;
        vars(index) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &, VarAccessor<MechTrait> &vars) {
        // Minimal compatible behavior: keep i as-is; external wrappers may drive it.
        return vars(i);
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, IClamp_replay_gpu);

} // namespace l5pc_IClamp_replay_gpu_mech
