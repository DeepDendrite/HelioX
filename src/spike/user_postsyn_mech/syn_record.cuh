#pragma once
#include "postsyn_template.cuh"
#include <cmath>
#include <cassert>

namespace SynRecord {

struct MechTrait {
    enum class VarNames {
        /* -------- parameter & assigned (0-28) -------- */
        AMPA_tau1,  AMPA_tau2,
        NMDA_tau1,  NMDA_tau2,
        GABA_tau1,  GABA_tau2,
        AMPA_e,     NMDA_e,     GABA_e,
        NMDA_C,     NMDA_rho,
        w, r_na, dv,
        Use_e, Use_i, Dep, Fac, u0,           // STP params（暂未用）
        i,                                     // 19
        AMPA_g, NMDA_g, GABA_g,                // 20-22
        v_prev, didv, pure_i, g_mul, dgdv, dgdg, // 23-28
        Use,                                   // 29 – 当前使用的 Use (简化)
        /* -------- state (30-35) -------- */
        AMPA_A, AMPA_B,
        NMDA_A, NMDA_B,
        GABA_A, GABA_B,
        /* -------- pre-computed factor (36-38) -------- */
        AMPA_factor, NMDA_factor, GABA_factor
    };
};

/*------------------------------------------------------------
 *  syn_record implementation
 *-----------------------------------------------------------*/
class SynRecord_Templated
        : public PostSynTemplate<SynRecord_Templated, MechTrait> {

    using enum MechTrait::VarNames;

public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;

    /* ---------------- ctor : register indexes ------------- */
    SynRecord_Templated(MechInitParams& param) : PostSynTemplate(param) {
        /* ---- 0‒18: parameters from model file ---- */
        var_in_coredata_idx.insert({AMPA_tau1, 0});
        var_in_coredata_idx.insert({AMPA_tau2, 1});
        var_in_coredata_idx.insert({NMDA_tau1, 2});
        var_in_coredata_idx.insert({NMDA_tau2, 3});
        var_in_coredata_idx.insert({GABA_tau1, 4});
        var_in_coredata_idx.insert({GABA_tau2, 5});
        var_in_coredata_idx.insert({AMPA_e, 6});
        var_in_coredata_idx.insert({NMDA_e, 7});
        var_in_coredata_idx.insert({GABA_e, 8});
        var_in_coredata_idx.insert({NMDA_C, 9});
        var_in_coredata_idx.insert({NMDA_rho, 10});
        var_in_coredata_idx.insert({w, 11});
        var_in_coredata_idx.insert({r_na, 12});
        var_in_coredata_idx.insert({dv, 13});
        var_in_coredata_idx.insert({Use_e, 14});
        var_in_coredata_idx.insert({Use_i, 15});
        var_in_coredata_idx.insert({Dep, 16});
        var_in_coredata_idx.insert({Fac, 17});
        var_in_coredata_idx.insert({u0, 18});
        /* ---- 19-28: assigned ---- */
        var_in_coredata_idx.insert({i,         19});
        var_in_coredata_idx.insert({AMPA_g,    20});
        var_in_coredata_idx.insert({NMDA_g,    21});
        var_in_coredata_idx.insert({GABA_g,    22});
        var_in_coredata_idx.insert({v_prev,    23});
        var_in_coredata_idx.insert({didv,      24});
        var_in_coredata_idx.insert({pure_i,    25});
        var_in_coredata_idx.insert({g_mul,     26});
        var_in_coredata_idx.insert({dgdv,      27});
        var_in_coredata_idx.insert({dgdg,      28});
        var_in_coredata_idx.insert({Use,       29});
        /* ---- 30-38: states + factors ---- */
        var_in_coredata_idx.insert({AMPA_A,       30});
        var_in_coredata_idx.insert({AMPA_B,       31});
        var_in_coredata_idx.insert({NMDA_A,       32});
        var_in_coredata_idx.insert({NMDA_B,       33});
        var_in_coredata_idx.insert({GABA_A,       34});
        var_in_coredata_idx.insert({GABA_B,       35});
        var_in_coredata_idx.insert({AMPA_factor,  36});
        var_in_coredata_idx.insert({NMDA_factor,  37});
        var_in_coredata_idx.insert({GABA_factor,  38});

        /* --- defaults identical to .mod --- */
        init_values.insert({w,     1.0});
        init_values.insert({r_na,  2.0});
        init_values.insert({dv,    1e-3});
        init_values.insert({Use_e, 1.0});
        init_values.insert({Use_i, 1.0});
        init_values.insert({Dep,   0.0});
        init_values.insert({Fac,   0.0});
        init_values.insert({u0,    0.0});
    }

    /* ----------------- helpers ----------------- */
    static __host__ __device__ __forceinline__ double sigma(double v, VarAccessor<MechTrait>& vars) {
        return 1.0 / (1.0 + vars(NMDA_C) * exp(-vars(NMDA_rho) * v));
    }

    /* ----------------- INIT -------------------- */
    DUAL_EXEC void init_single_node(MechTempInitParam& param,
                                    VarAccessor<MechTrait>& vars) {
        using std::exp; using std::log;
        /* factor / tau sanity for each of the three exponentials */
        auto init_pair = [&](VarNames tau1, VarNames tau2,
                             VarNames A, VarNames B, VarNames factor) {
            double t1 = vars(tau1);
            double t2 = vars(tau2);
            if (t1 / t2 > 0.9999) { t1 = 0.9999 * t2; vars(tau1) = t1; }
            if (t1 / t2 < 1e-9)   { t1 = 1e-9  * t2; vars(tau1) = t1; }
            vars(A) = 0.0; vars(B) = 0.0;
            double tp = (t1 * t2) / (t2 - t1) * log(t2 / t1);
            vars(factor) = 1.0 / (-exp(-tp / t1) + exp(-tp / t2));
        };
        init_pair(AMPA_tau1, AMPA_tau2, AMPA_A, AMPA_B, AMPA_factor);
        init_pair(NMDA_tau1, NMDA_tau2, NMDA_A, NMDA_B, NMDA_factor);
        init_pair(GABA_tau1, GABA_tau2, GABA_A, GABA_B, GABA_factor);

        /* choose which Use to adopt (still simplified) */
        vars(Use) = (vars(w) > 0.0 ? vars(Use_e) : vars(Use_i));
    }

    /* ----------------- CURRENT ----------------- */
    DUAL_EXEC double current_single_node(MechTempCurParam& param,
                                         VarAccessor<MechTrait>& vars) {
        double v = param.volt;
        vars(v_prev) = v;                   // store for STATE step

        double current = 0.0;
        if (vars(w) > 0.0) {                // excitatory
            vars(AMPA_g) = vars(AMPA_B) - vars(AMPA_A);
            vars(NMDA_g) = vars(NMDA_B) - vars(NMDA_A);
            double s = SynRecord_Templated::sigma(v, vars);
            current = fabs(vars(w)) *
                      (vars(AMPA_g) * (v - vars(AMPA_e)) +
                       vars(NMDA_g) * s * (v - vars(NMDA_e)));
        } else {                            // inhibitory (GABA)
            vars(GABA_g) = vars(GABA_B) - vars(GABA_A);
            current = fabs(vars(w)) * vars(GABA_g) * (v - vars(GABA_e));
        }
        vars(i) = current;
        return current;
    }

    /* ----------------- STATE ------------------- */
    DUAL_EXEC void state_single_node(MechTempStateParam& param,
                                     VarAccessor<MechTrait>& vars) {
        double dt = param.dt;
        auto decay = [&](VarNames X, VarNames tau) {
            vars(X) *= exp(-dt / vars(tau));
        };
        decay(AMPA_A, AMPA_tau1);  decay(AMPA_B, AMPA_tau2);
        decay(NMDA_A, NMDA_tau1);  decay(NMDA_B, NMDA_tau2);
        decay(GABA_A, GABA_tau1);  decay(GABA_B, GABA_tau2);

        /* 下列记录量仅用于梯度 / 诊断，可删减 */
        double v = param.volt;
        if (vars(w) > 0.0) {
            vars(didv)   = fabs(vars(w)) *
                           (vars(AMPA_g) +
                            vars(NMDA_g) * SynRecord_Templated::sigma(vars(v_prev), vars));
            vars(pure_i) = vars(AMPA_g) * (v - vars(AMPA_e)) +
                           vars(NMDA_g) * SynRecord_Templated::sigma(vars(v_prev), vars) *
                           (v - vars(NMDA_e));
            vars(g_mul)  = fabs(vars(w)) *
                           vars(NMDA_g) * (v - vars(NMDA_e));
            vars(dgdv)   = (SynRecord_Templated::sigma(vars(v_prev) + vars(dv), vars) -
                            SynRecord_Templated::sigma(vars(v_prev), vars)) / vars(dv);
            vars(dgdg)   = 0.0;
        } else {
            vars(didv) = fabs(vars(w)) * vars(GABA_g);
            vars(pure_i) = -vars(GABA_g) * (v - vars(GABA_e));
            vars(g_mul) = vars(dgdv) = vars(dgdg) = 0.0;
        }
    }

    /* ----------------- NET_RECEIVE -------------- */
    DUAL_EXEC void net_receive_single_node(PostSynTempRecvParam& recv,
                                           VarAccessor<MechTrait>& vars) {
        /* —— simplified: ignore short-term plasticity —— */
        double w_event = recv.weight;           // NetCon weight (uS)
        if (vars(w) > 0.0) {                    // excitatory
            double ra = vars(r_na);
            double a_fac = w_event * vars(AMPA_factor) / (1.0 + ra);
            double n_fac = w_event * vars(NMDA_factor) * ra / (1.0 + ra);
            vars(AMPA_A) += a_fac;
            vars(AMPA_B) += a_fac;
            vars(NMDA_A) += n_fac;
            vars(NMDA_B) += n_fac;
        } else {                                // inhibitory
            double g_fac = w_event * vars(GABA_factor);
            vars(GABA_A) += g_fac;
            vars(GABA_B) += g_fac;
        }
    }
};

/* ----------- register with HelioX ----------- */
REGISTER_POSTSYN("syn_record", SynRecord_Templated,5);

} // namespace SynRecord
