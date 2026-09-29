// NaV_modified mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cassert>
#include <cmath>

namespace NaV_modified_mech {

static const char *MECH_NAME_TO_REG = "NaV_modified";

struct MechTrait {
    enum class VarNames {
        gbar,
        g,
        C1,
        C2,
        C3,
        C4,
        C5,
        I1,
        I2,
        I3,
        I4,
        I5,
        O,
        I6,
        f01,
        f02,
        f03,
        f04,
        f0O,
        f11,
        f12,
        f13,
        f14,
        f1n,
        fi1,
        fi2,
        fi3,
        fi4,
        fi5,
        fin,
        b01,
        b02,
        b03,
        b04,
        b0O,
        b11,
        b12,
        b13,
        b14,
        b1n,
        bi1,
        bi2,
        bi3,
        bi4,
        bi5,
        bin,
        ena,
        ina,
        DC1,
        DC2,
        DC3,
        DC4,
        DC5,
        DI1,
        DI2,
        DI3,
        DI4,
        DI5,
        DO,
        DI6,
        v,
        _g
    };
    enum class GlobalVarNames { celsius };
    enum class IonVarNames { _ion_ena, _ion_ina };
};

class NaV_modified final : public MechTemp<NaV_modified, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    static constexpr double Con = 0.01;
    static constexpr double Coff = 40.0;
    static constexpr double Oon = 8.0;
    static constexpr double Ooff = 0.05;
    static constexpr double alpha = 400.0;
    static constexpr double beta = 12.0;
    static constexpr double gamma = 250.0;
    static constexpr double delta = 60.0;
    static constexpr double alfac = 2.51;
    static constexpr double btfac = 5.32;
    static constexpr double x1 = 24.0;
    static constexpr double x2 = -24.0;

    static constexpr auto LearnableVars = std::array{gbar};

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars) {
        double qt = pow(2.3, (vars(celsius) - 37.0) / 10.0);

        double exp1 = exp(volt / x1);
        double exp2 = exp(volt / x2);

        vars(f01) = qt * 4.0 * alpha * exp1;
        vars(f02) = qt * 3.0 * alpha * exp1;
        vars(f03) = qt * 2.0 * alpha * exp1;
        vars(f04) = qt * 1.0 * alpha * exp1;
        vars(f0O) = qt * gamma;
        vars(f11) = qt * 4.0 * alpha * alfac * exp1;
        vars(f12) = qt * 3.0 * alpha * alfac * exp1;
        vars(f13) = qt * 2.0 * alpha * alfac * exp1;
        vars(f14) = qt * 1.0 * alpha * alfac * exp1;
        vars(f1n) = qt * gamma;
        vars(fi1) = qt * Con;
        vars(fi2) = qt * Con * alfac;
        vars(fi3) = qt * Con * (alfac * alfac);
        vars(fi4) = qt * Con * (alfac * alfac * alfac);
        vars(fi5) = qt * Con * (alfac * alfac * alfac * alfac);
        vars(fin) = qt * Oon;

        vars(b01) = qt * 1.0 * beta * exp2;
        vars(b02) = qt * 2.0 * beta * exp2;
        vars(b03) = qt * 3.0 * beta * exp2;
        vars(b04) = qt * 4.0 * beta * exp2;
        vars(b0O) = qt * delta;
        vars(b11) = qt * 1.0 * beta * exp2 / btfac;
        vars(b12) = qt * 2.0 * beta * exp2 / btfac;
        vars(b13) = qt * 3.0 * beta * exp2 / btfac;
        vars(b14) = qt * 4.0 * beta * exp2 / btfac;
        vars(b1n) = qt * delta;
        vars(bi1) = qt * Coff;
        vars(bi2) = qt * Coff / btfac;
        vars(bi3) = qt * Coff / (btfac * btfac);
        vars(bi4) = qt * Coff / (btfac * btfac * btfac);
        vars(bi5) = qt * Coff / (btfac * btfac * btfac * btfac);
        vars(bin) = qt * Ooff;
    }

    DUAL_EXEC double clamp_nonneg(double x) { return x < 0.0 ? 0.0 : x; }

    // Solve a 12x12 linear system using Gauss-Jordan elimination with partial pivoting.
    // This is used to mimic NEURON's KINETIC sparse solver behavior for NaV_modified.
    DUAL_EXEC bool solve_linear_12(double a[12][12], double b[12]) {
        for (int col = 0; col < 12; col++) {
            int pivot = col;
            double pivot_abs = fabs(a[col][col]);
            for (int row = col + 1; row < 12; row++) {
                double v = fabs(a[row][col]);
                if (v > pivot_abs) {
                    pivot_abs = v;
                    pivot = row;
                }
            }
            if (pivot_abs < 1e-30) {
                return false;
            }
            if (pivot != col) {
                for (int k = col; k < 12; k++) {
                    double tmp = a[col][k];
                    a[col][k] = a[pivot][k];
                    a[pivot][k] = tmp;
                }
                double tmpb = b[col];
                b[col] = b[pivot];
                b[pivot] = tmpb;
            }

            double diag = a[col][col];
            double inv = 1.0 / diag;
            for (int k = col; k < 12; k++) {
                a[col][k] *= inv;
            }
            b[col] *= inv;

            for (int row = 0; row < 12; row++) {
                if (row == col) {
                    continue;
                }
                double factor = a[row][col];
                if (factor == 0.0) {
                    continue;
                }
                for (int k = col; k < 12; k++) {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
        return true;
    }

    // Build generator matrix A for the 12-state Markov chain.
    // State order: C1,C2,C3,C4,C5,O,I1,I2,I3,I4,I5,I6
    DUAL_EXEC void build_generator(double A[12][12], VarAccessor<MechTrait> &vars) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                A[i][j] = 0.0;
            }
        }

        auto add_edge = [&](int from, int to, double k) {
            // from -> to with rate k
            A[to][from] += k;
            A[from][from] -= k;
        };

        // C1<->C2
        add_edge(0, 1, vars(f01));
        add_edge(1, 0, vars(b01));
        // C2<->C3
        add_edge(1, 2, vars(f02));
        add_edge(2, 1, vars(b02));
        // C3<->C4
        add_edge(2, 3, vars(f03));
        add_edge(3, 2, vars(b03));
        // C4<->C5
        add_edge(3, 4, vars(f04));
        add_edge(4, 3, vars(b04));
        // C5<->O
        add_edge(4, 5, vars(f0O));
        add_edge(5, 4, vars(b0O));
        // O<->I6
        add_edge(5, 11, vars(fin));
        add_edge(11, 5, vars(bin));
        // I1<->I2
        add_edge(6, 7, vars(f11));
        add_edge(7, 6, vars(b11));
        // I2<->I3
        add_edge(7, 8, vars(f12));
        add_edge(8, 7, vars(b12));
        // I3<->I4
        add_edge(8, 9, vars(f13));
        add_edge(9, 8, vars(b13));
        // I4<->I5
        add_edge(9, 10, vars(f14));
        add_edge(10, 9, vars(b14));
        // I5<->I6
        add_edge(10, 11, vars(f1n));
        add_edge(11, 10, vars(b1n));
        // C1<->I1
        add_edge(0, 6, vars(fi1));
        add_edge(6, 0, vars(bi1));
        // C2<->I2
        add_edge(1, 7, vars(fi2));
        add_edge(7, 1, vars(bi2));
        // C3<->I3
        add_edge(2, 8, vars(fi3));
        add_edge(8, 2, vars(bi3));
        // C4<->I4
        add_edge(3, 9, vars(fi4));
        add_edge(9, 3, vars(bi4));
        // C5<->I5
        add_edge(4, 10, vars(fi5));
        add_edge(10, 4, vars(bi5));
    }

    DUAL_EXEC void load_state_from_array(const double* x, VarAccessor<MechTrait>& vars) {
        vars(C1) = x[0];
        vars(C2) = x[1];
        vars(C3) = x[2];
        vars(C4) = x[3];
        vars(C5) = x[4];
        vars(O) = x[5];
        vars(I1) = x[6];
        vars(I2) = x[7];
        vars(I3) = x[8];
        vars(I4) = x[9];
        vars(I5) = x[10];
        vars(I6) = x[11];
    }

    DUAL_EXEC void store_state_to_array(double* x, VarAccessor<MechTrait>& vars) {
        x[0] = vars(C1);
        x[1] = vars(C2);
        x[2] = vars(C3);
        x[3] = vars(C4);
        x[4] = vars(C5);
        x[5] = vars(O);
        x[6] = vars(I1);
        x[7] = vars(I2);
        x[8] = vars(I3);
        x[9] = vars(I4);
        x[10] = vars(I5);
        x[11] = vars(I6);
    }

    DUAL_EXEC void evaluate_state_update(double volt, double dt, const double* x_in, double* x_out, VarAccessor<MechTrait>& vars) {
        load_state_from_array(x_in, vars);
        rates(volt, vars);

        double A[12][12];
        build_generator(A, vars);

        double M[12][12];
        double b[12];
        for (int i = 0; i < 12; i++) {
            b[i] = x_in[i];
            for (int j = 0; j < 12; j++) {
                M[i][j] = (i == j ? 1.0 : 0.0) - dt * A[i][j];
            }
        }

        for (int j = 0; j < 12; j++) {
            M[11][j] = 1.0;
        }
        b[11] = 1.0;

        bool ok = solve_linear_12(M, b);
        if (ok) {
            for (int i = 0; i < 12; i++) {
                x_out[i] = b[i];
            }
        } else {
            for (int i = 0; i < 12; i++) {
                x_out[i] = x_in[i];
            }
        }
    }

    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    explicit NaV_modified(MechInitParams &param) : MechTemp(param) {
        global_info_map.insert({celsius, {"celsius"}});
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});

        // Field order from NEURON-generated run_network/x86_64/NaV_modified.cpp.
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({g, 1});
        var_in_coredata_idx.insert({C1, 2});
        var_in_coredata_idx.insert({C2, 3});
        var_in_coredata_idx.insert({C3, 4});
        var_in_coredata_idx.insert({C4, 5});
        var_in_coredata_idx.insert({C5, 6});
        var_in_coredata_idx.insert({I1, 7});
        var_in_coredata_idx.insert({I2, 8});
        var_in_coredata_idx.insert({I3, 9});
        var_in_coredata_idx.insert({I4, 10});
        var_in_coredata_idx.insert({I5, 11});
        var_in_coredata_idx.insert({O, 12});
        var_in_coredata_idx.insert({I6, 13});
        var_in_coredata_idx.insert({f01, 14});
        var_in_coredata_idx.insert({f02, 15});
        var_in_coredata_idx.insert({f03, 16});
        var_in_coredata_idx.insert({f04, 17});
        var_in_coredata_idx.insert({f0O, 18});
        var_in_coredata_idx.insert({f11, 19});
        var_in_coredata_idx.insert({f12, 20});
        var_in_coredata_idx.insert({f13, 21});
        var_in_coredata_idx.insert({f14, 22});
        var_in_coredata_idx.insert({f1n, 23});
        var_in_coredata_idx.insert({fi1, 24});
        var_in_coredata_idx.insert({fi2, 25});
        var_in_coredata_idx.insert({fi3, 26});
        var_in_coredata_idx.insert({fi4, 27});
        var_in_coredata_idx.insert({fi5, 28});
        var_in_coredata_idx.insert({fin, 29});
        var_in_coredata_idx.insert({b01, 30});
        var_in_coredata_idx.insert({b02, 31});
        var_in_coredata_idx.insert({b03, 32});
        var_in_coredata_idx.insert({b04, 33});
        var_in_coredata_idx.insert({b0O, 34});
        var_in_coredata_idx.insert({b11, 35});
        var_in_coredata_idx.insert({b12, 36});
        var_in_coredata_idx.insert({b13, 37});
        var_in_coredata_idx.insert({b14, 38});
        var_in_coredata_idx.insert({b1n, 39});
        var_in_coredata_idx.insert({bi1, 40});
        var_in_coredata_idx.insert({bi2, 41});
        var_in_coredata_idx.insert({bi3, 42});
        var_in_coredata_idx.insert({bi4, 43});
        var_in_coredata_idx.insert({bi5, 44});
        var_in_coredata_idx.insert({bin, 45});
        var_in_coredata_idx.insert({ena, 46});
        var_in_coredata_idx.insert({ina, 47});
        var_in_coredata_idx.insert({DC1, 48});
        var_in_coredata_idx.insert({DC2, 49});
        var_in_coredata_idx.insert({DC3, 50});
        var_in_coredata_idx.insert({DC4, 51});
        var_in_coredata_idx.insert({DC5, 52});
        var_in_coredata_idx.insert({DI1, 53});
        var_in_coredata_idx.insert({DI2, 54});
        var_in_coredata_idx.insert({DI3, 55});
        var_in_coredata_idx.insert({DI4, 56});
        var_in_coredata_idx.insert({DI5, 57});
        var_in_coredata_idx.insert({DO, 58});
        var_in_coredata_idx.insert({DI6, 59});
        var_in_coredata_idx.insert({v, 60});
        var_in_coredata_idx.insert({_g, 61});

        assert(param.name == MECH_NAME_TO_REG);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        vars(v) = param.volt;
        rates(param.volt, vars);

        // Pre-computed equilibrium initials (fixed at celsius=34, finitialize at -80mV).
        vars(C1) = 0.7943030506073554;
        vars(C2) = 0.13478838594701364;
        vars(C3) = 0.008582578752122788;
        vars(C4) = 0.00024682560448255635;
        vars(C5) = 4.331284969747251e-06;
        vars(I1) = 0.00013580298382367433;
        vars(I2) = 0.0002679953909560066;
        vars(I3) = 0.00019703724552117086;
        vars(I4) = 0.00705657380170769;
        vars(I5) = 0.0062431942819651375;
        vars(O) = 5.130833868392259e-05;
        vars(I6) = 0.048122915761398474;

        vars(g) = vars(gbar) * vars(O);
        vars(ina) = 0.0;

        vars(DC1) = vars(DC2) = vars(DC3) = vars(DC4) = vars(DC5) = 0.0;
        vars(DI1) = vars(DI2) = vars(DI3) = vars(DI4) = vars(DI5) = 0.0;
        vars(DO) = vars(DI6) = 0.0;
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        vars(ena) = vars(_ion_ena);
        vars(v) = param.volt;

        vars(g) = vars(gbar) * vars(O);
        vars(ina) = vars(g) * (param.volt - vars(ena));
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
        }
        return vars(ina);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars) {
        vars.idx = param.idx;
        const double volt = param.volt;
        const double ena_local = vars(_ion_ena);
        const double o = vars(O);
        const double g = vars(gbar) * o;
        if (!std::isfinite(g) || !std::isfinite(o)) {
            return;
        }

        // Non-electrode current: rhs += -i. The current baseline does not
        // propagate dynamic-D tape.
        const double dloss_dv = param.grad_mech_current * g;
        if (std::isfinite(dloss_dv)) {
            mechAtomAdd(&param.grad_v[param.node_index], dloss_dv);
        }

        const double dloss_dgbar = param.grad_mech_current * (o * (volt - ena_local));
        if (std::isfinite(dloss_dgbar)) {
            mechAtomAdd(&grad_ref<gbar>(param, vars), dloss_dgbar);
        }
    }



    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        rates(param.volt, vars);

        double x0[12] = {
            vars(C1), vars(C2), vars(C3), vars(C4), vars(C5), vars(O),
            vars(I1), vars(I2), vars(I3), vars(I4), vars(I5), vars(I6),
        };

        double A[12][12];
        build_generator(A, vars);

        // Solve (I - dt*A) x1 = x0 with conservation sum(x1)=1 enforced by replacing last row.
        double M[12][12];
        double b[12];
        for (int i = 0; i < 12; i++) {
            b[i] = x0[i];
            for (int j = 0; j < 12; j++) {
                M[i][j] = (i == j ? 1.0 : 0.0) - param.dt * A[i][j];
            }
        }

        // Replace last equation with conservation.
        for (int j = 0; j < 12; j++) {
            M[11][j] = 1.0;
        }
        b[11] = 1.0;

        bool ok = solve_linear_12(M, b);
        double x1[12];
        if (ok) {
            for (int i = 0; i < 12; i++) {
                x1[i] = b[i];
            }
        } else {
            // Fallback: keep previous state (better than exploding).
            for (int i = 0; i < 12; i++) {
                x1[i] = x0[i];
            }
        }

        // Assign states back.
        vars(C1) = x1[0];
        vars(C2) = x1[1];
        vars(C3) = x1[2];
        vars(C4) = x1[3];
        vars(C5) = x1[4];
        vars(O) = x1[5];
        vars(I1) = x1[6];
        vars(I2) = x1[7];
        vars(I3) = x1[8];
        vars(I4) = x1[9];
        vars(I5) = x1[10];
        vars(I6) = x1[11];

        // Derivative-like outputs (not used for current, but exported/recordable).
        vars(DC1) = (x1[0] - x0[0]) / param.dt;
        vars(DC2) = (x1[1] - x0[1]) / param.dt;
        vars(DC3) = (x1[2] - x0[2]) / param.dt;
        vars(DC4) = (x1[3] - x0[3]) / param.dt;
        vars(DC5) = (x1[4] - x0[4]) / param.dt;
        vars(DO) = (x1[5] - x0[5]) / param.dt;
        vars(DI1) = (x1[6] - x0[6]) / param.dt;
        vars(DI2) = (x1[7] - x0[7]) / param.dt;
        vars(DI3) = (x1[8] - x0[8]) / param.dt;
        vars(DI4) = (x1[9] - x0[9]) / param.dt;
        vars(DI5) = (x1[10] - x0[10]) / param.dt;
        vars(DI6) = (x1[11] - x0[11]) / param.dt;
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, NaV_modified);

} // namespace NaV_modified_mech
