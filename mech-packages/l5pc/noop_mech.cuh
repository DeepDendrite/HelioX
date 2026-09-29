#pragma once

#include "mech_template.cuh"

#include <cassert>

#define REGISTER_L5PC_NOOP_MECH(MECH_STR, NAMESPACE_NAME)                                \
    namespace NAMESPACE_NAME {                                                           \
    static const char *MECH_NAME_TO_REG = MECH_STR;                                      \
    struct MechTrait {                                                                   \
        enum class VarNames { dummy };                                                   \
    };                                                                                   \
    class Noop final : public MechTemp<Noop, MechTrait> {                                \
    public:                                                                              \
        constexpr static MechFlags flags = ENABLE_CURRENT;                               \
        explicit Noop(MechInitParams &param) : MechTemp(param) {                         \
            assert(param.name == MECH_NAME_TO_REG);                                      \
        }                                                                                \
        DUAL_EXEC double current_single_node(MechTempCurParam &, VarAccessor<MechTrait> &) { \
            return 0.0;                                                                  \
        }                                                                                \
    };                                                                                   \
    REGISTER_MECHANISM(MECH_NAME_TO_REG, Noop);                                          \
    }                                                                                    \
    static_assert(true)
