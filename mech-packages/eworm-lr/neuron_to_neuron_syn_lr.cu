// neuron_to_neuron_syn_lr mechanism (worm-lr, NEURON-compatible)
// Source MOD:
//   /home/ausar/BAAIWORM/eworm_learn/components/mechanism/modfile/neuron_to_neuron_syn_lr.mod
//
// NEURON:
//   POINT_PROCESS neuron_to_neuron_syn_lr
//   POINTER vpre
//   NONSPECIFIC_CURRENT i
//
// Notes:
// - `vpre` is a POINTER and is resolved from bbcore-exported pdata + pointer2type.
// - This is a cnexp-updated 2-state mechanism (`s`, `s_ds`) driven by presynaptic voltage.
// - We intentionally do not expose NEURON's internal `_g` / `Ds` / `Ds_ds` fields.

#include "mech_template.cuh"
#include <cmath>

namespace neuron_to_neuron_syn_lr_wormlr {

#define DPSEM(x) dpsem(DparamSemantics::x)

struct MechTrait {
    enum class VarNames {
        // PARAMETER
        w,
        g,
        delta,
        k,
        Vth,
        erev,
        dvpre,

        // ASSIGNED/RANGE
        i,
        pure_i,
        didv,
        dsdvpre,
        didvpre,

        // STATE
        s,
        s_ds,

        // ASSIGNED (internal but recordable in NEURON)
        inf,
        tau,
    };

    enum class PointerVarNames {
        vpre,
    };
};

class NeuronToNeuronSynLr final : public MechTemp<NeuronToNeuronSynLr, MechTrait> {
  public:
    using enum MechTrait::VarNames;
    using enum MechTrait::PointerVarNames;

    constexpr static MechFlags flags = MechFlags::ENABLE_INIT | MechFlags::ENABLE_CURRENT |
                                      MechFlags::ENABLE_STATE | MechFlags::POINT_PROCESS;

    explicit NeuronToNeuronSynLr(MechInitParams& param) : MechTemp(param) {
        // Defaults from MOD.
        init_values.insert({w, 1.0});
        init_values.insert({g, 4.9});
        init_values.insert({delta, 5.0});
        init_values.insert({k, 0.5});
        init_values.insert({Vth, -20.0});
        init_values.insert({erev, 30.0});
        init_values.insert({dvpre, 1e-3});

        // Data field order from nrnivmodl-generated C++:
        // w:0 g:1 delta:2 k:3 Vth:4 erev:5 dvpre:6
        // i:7 pure_i:8 didv:9 dsdvpre:10 didvpre:11
        // s:12 s_ds:13 inf:14 tau:15 Ds:16 Ds_ds:17 _g:18
        var_in_coredata_idx.insert({w, 0});
        var_in_coredata_idx.insert({g, 1});
        var_in_coredata_idx.insert({delta, 2});
        var_in_coredata_idx.insert({k, 3});
        var_in_coredata_idx.insert({Vth, 4});
        var_in_coredata_idx.insert({erev, 5});
        var_in_coredata_idx.insert({dvpre, 6});

        var_in_coredata_idx.insert({i, 7});
        var_in_coredata_idx.insert({pure_i, 8});
        var_in_coredata_idx.insert({didv, 9});
        var_in_coredata_idx.insert({dsdvpre, 10});
        var_in_coredata_idx.insert({didvpre, 11});

        var_in_coredata_idx.insert({s, 12});
        var_in_coredata_idx.insert({s_ds, 13});
        var_in_coredata_idx.insert({inf, 14});
        var_in_coredata_idx.insert({tau, 15});
    }

    DUAL_EXEC void rates(VarAccessor<MechTrait>& vars, double v_mV) {
        vars(inf) = 1.0 / (1.0 + exp((vars(Vth) - v_mV) / vars(delta)));
        vars(tau) = (1.0 - vars(inf)) / vars(k);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam& /*param*/, VarAccessor<MechTrait>& vars) {
        // Match NEURON default for STATE variables when INITIAL exists: initialize explicitly.
        vars(s) = 0.0;
        vars(s_ds) = 0.0;

        const double vpre_val = vars.Ptr(vpre);
        rates(vars, vpre_val);
        vars(s_ds) = vars(inf);
        vars(s) = vars(inf);

        if (vars(w) > 0.0) {
            vars(g) = 4.9;
            vars(k) = 0.5;
            vars(erev) = 30.0;
        } else {
            vars(g) = 2.0;
            vars(k) = 0.015000001;
            vars(erev) = -70.0;
        }
    }

    DUAL_EXEC double current_single_node(MechTempCurParam& param, VarAccessor<MechTrait>& vars) {
        vars(pure_i) = vars(g) * vars(s) * (param.volt - vars(erev));
        vars(i) = fabs(vars(w)) * vars(pure_i);
        vars(didv) = -fabs(vars(w)) * vars(g) * vars(s);
        vars(dsdvpre) = (vars(s_ds) - vars(s)) / vars(dvpre);
        vars(didvpre) = fabs(vars(w)) * vars(g) * (vars(erev) - param.volt) * vars(dsdvpre);
        return vars(i);
    }

    DUAL_EXEC void state_single_node(MechTempStateParam& param, VarAccessor<MechTrait>& vars) {
        const double vpre_val = vars.Ptr(vpre);

        // Advance s_ds using vpre+dvpre, then advance s using vpre (cnexp).
        vars(s_ds) = vars(s);

        rates(vars, vpre_val + vars(dvpre));
        {
            const double tau_local = vars(tau);
            const double inf_local = vars(inf);
            const double exp_term = exp(-param.dt / tau_local);
            vars(s_ds) = inf_local + (vars(s_ds) - inf_local) * exp_term;
        }

        rates(vars, vpre_val);
        {
            const double tau_local = vars(tau);
            const double inf_local = vars(inf);
            const double exp_term = exp(-param.dt / tau_local);
            vars(s) = inf_local + (vars(s) - inf_local) * exp_term;
        }
    }
};

REGISTER_MECHANISM("neuron_to_neuron_syn_lr", NeuronToNeuronSynLr);
REGISTER_POINTER_DPARAM_SLOTS("neuron_to_neuron_syn_lr", 2);
// bbcore export dparam semantics (fixed-step path):
// 0: area, 1: pntproc, 2: pointer
REGISTER_DPARAM_SEMANTICS("neuron_to_neuron_syn_lr", DPSEM(area), DPSEM(pntproc), DPSEM(pointer));

#undef DPSEM

}  // namespace neuron_to_neuron_syn_lr_wormlr
