// gapjunction_lr mechanism (worm-lr, NEURON-compatible)
// Source MOD:
//   $HOME/path/to/eworm_learn/components/mechanism/modfile/gapjunction_lr.mod
//
// NEURON:
//   POINT_PROCESS gapjunction_lr
//   POINTER vpre
//   NONSPECIFIC_CURRENT i
//
// Notes:
// - `vpre` is a POINTER and is resolved from bbcore-exported pdata + pointer2type.
// - We expose the RANGE variables (`pure_i`, `didv`, `didvpre`) for compatibility/recording.
// - We intentionally do not expose NEURON's internal `_g` field.

#include "mech_template.cuh"

namespace gapjunction_lr_wormlr {

#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames {
        g,
        w,
        i,
        pure_i,
        didv,
        didvpre,
    };

    enum class PointerVarNames {
        vpre,
    };
};

class GapjunctionLr final : public MechTemp<GapjunctionLr, MechTrait> {
  public:
    using enum MechTrait::VarNames;
    using enum MechTrait::PointerVarNames;

    constexpr static MechFlags flags = MechFlags::ENABLE_CURRENT | MechFlags::POINT_PROCESS;

    explicit GapjunctionLr(MechInitParams& param) : MechTemp(param) {
        // Defaults from MOD.
        init_values.insert({g, 1.0});
        init_values.insert({w, 0.0});

        // Data field order from nrnivmodl-generated C++:
        // g:0, w:1, i:2, pure_i:3, didv:4, didvpre:5, _g:6
        var_in_coredata_idx.insert({g, 0});
        var_in_coredata_idx.insert({w, 1});
        var_in_coredata_idx.insert({i, 2});
        var_in_coredata_idx.insert({pure_i, 3});
        var_in_coredata_idx.insert({didv, 4});
        var_in_coredata_idx.insert({didvpre, 5});
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait>& vars) {
        const double vpre_val = vars.Ptr(vpre);
        vars(pure_i) = vars(g) * (param.volt - vpre_val);
        vars(i) = vars(w) * vars(pure_i);
        vars(didv) = -vars(w) * vars(g);
        vars(didvpre) = vars(w) * vars(g);
        return vars(i);
    }
};

REGISTER_MECHANISM("gapjunction_lr", GapjunctionLr);
REGISTER_POINTER_DPARAM_SLOTS("gapjunction_lr", 2);
// NEURON-generated dparam semantics for POINT_PROCESS with one POINTER:
// 0: area, 1: pntproc, 2: pointer
REGISTER_DPARAM_SEMANTICS("gapjunction_lr", DPSEM(area), DPSEM(pntproc), DPSEM(pointer));

#undef DPSEM

}  // namespace gapjunction_lr_wormlr
