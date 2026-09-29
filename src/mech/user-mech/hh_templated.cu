// hh_templated mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>


namespace HH_Templated{

//用户请修改以下宏定义
#define MECH_CLASS_NAME HH_Templated
static const char *MECH_NAME_TO_REG = "hh";

struct MechTrait{
    static constexpr bool MechSupportsTable = true;
    enum class VarNames{//可以自由添加或修改变量
        m, h, n,
        ina, ik, il,
        gnabar, gkbar, gl, gna, gk,
        minf, hinf, ninf, mtau, htau, ntau,
        ena, ek, el
    };
    enum class GlobalVarNames{
        celsius, usetable_hh
    };

    enum class IonVarNames{
        _ion_ena,_ion_ina,_ion_ek,_ion_ik
    };


};



class MECH_CLASS_NAME:public MechTemp<MECH_CLASS_NAME,MechTrait>{
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
public:
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;
    struct TableSpec {
        static constexpr int kTableBins = 200;
        static constexpr int kOutputCount = 6;
        static constexpr bool dep_dt = false;
        static constexpr auto dep_global_vars = std::array{
            MechTrait::GlobalVarNames::celsius,
        };
        static constexpr std::array<MechTrait::VarNames, 0> dep_range_vars{};

        DUAL_EXEC MechTrait::VarNames output_var(int out_idx) {
            switch (out_idx) {
                case 0: return MechTrait::VarNames::minf;
                case 1: return MechTrait::VarNames::mtau;
                case 2: return MechTrait::VarNames::hinf;
                case 3: return MechTrait::VarNames::htau;
                case 4: return MechTrait::VarNames::ninf;
                default: return MechTrait::VarNames::ntau;
            }
        }

        DUAL_EXEC bool use_table(VarAccessor<MechTrait>& vars) {
            return vars(MechTrait::GlobalVarNames::usetable_hh) != 0.0;
        }

        DUAL_EXEC double table_min(VarAccessor<MechTrait>& /*vars*/) {
            return -100.0;
        }

        DUAL_EXEC double table_max(VarAccessor<MechTrait>& /*vars*/) {
            return 100.0;
        }

        DUAL_EXEC double vtrap(double x, double y) {
            if (fabs(x / y) < 1e-6)
            {
                return y * (1 - x / y / 2);
            }
            return x / (exp(x / y) - 1);
        }

        DUAL_EXEC void compute_table_outputs_at_point(double v, double dt, VarAccessor<MechTrait>& vars) {
            (void)dt;
            double q10, alpha, beta, sum;
            q10 = pow(3.0, (vars(MechTrait::GlobalVarNames::celsius) - 6.3) / 10);

            alpha = 0.1 * vtrap(-(v + 40), 10);
            beta = 4 * exp(-(v + 65) / 18);
            sum = alpha + beta;
            vars(MechTrait::VarNames::mtau) = 1.0 / (q10 * sum);
            vars(MechTrait::VarNames::minf) = alpha / sum;

            alpha = 0.07 * exp(-(v + 65) / 20);
            beta = 1.0 / (exp(-(v + 35) / 10) + 1);
            sum = alpha + beta;
            vars(MechTrait::VarNames::htau) = 1.0 / (q10 * sum);
            vars(MechTrait::VarNames::hinf) = alpha / sum;

            alpha = 0.01 * vtrap(-(v + 55), 10);
            beta = 0.125 * exp(-(v + 65) / 80);
            sum = alpha + beta;
            vars(MechTrait::VarNames::ntau) = 1.0 / (q10 * sum);
            vars(MechTrait::VarNames::ninf) = alpha / sum;
        }
    };

    MECH_CLASS_NAME(MechInitParams &param):MechTemp(param){
        // need_area = false;
        // var_in_coredata_idx.insert({your_var_name,0});

        init_values.insert({gnabar,0.25});
        init_values.insert({gl,0.00016666});
        init_values.insert({el,-60.0});
        init_values.insert({gkbar,0.036});
        init_values.insert({ena,50});
        init_values.insert({ek,-77});

        global_info_map.insert({celsius,{"celsius"}});
        global_info_map.insert({usetable_hh,{"usetable_hh"}});
        if (!coreneuron::global_var_map.contains("usetable_hh")) {
            coreneuron::global_var_map["usetable_hh"] = std::vector<double>{0.0};
        }
        var_in_coredata_idx.insert({gnabar,0});
        var_in_coredata_idx.insert({gkbar,1});
        var_in_coredata_idx.insert({gl,2});
        var_in_coredata_idx.insert({el,3});
        var_in_coredata_idx.insert({gna,4});
        var_in_coredata_idx.insert({gk,5});
        var_in_coredata_idx.insert({il,6});
        var_in_coredata_idx.insert({minf,7});
        var_in_coredata_idx.insert({hinf,8});
        var_in_coredata_idx.insert({ninf,9});
        var_in_coredata_idx.insert({mtau,10});
        var_in_coredata_idx.insert({htau,11});
        var_in_coredata_idx.insert({ntau,12});
        var_in_coredata_idx.insert({ena,19});
        var_in_coredata_idx.insert({ek,20});

        ion_var_map.insert({_ion_ek,{"k_ion",EionVarNames::erev}});
        ion_var_map.insert({_ion_ik,{"k_ion",EionVarNames::cur}});
        ion_var_map.insert({_ion_ina,{"na_ion",EionVarNames::cur}});
        ion_var_map.insert({_ion_ena,{"na_ion",EionVarNames::erev}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n",param.name.c_str());
    }

private:
    DUAL_EXEC double update_gate_cnexp(double x_t, double x_inf, double tau, double dt) {
        const double decay = std::exp(dt * (-1.0 / tau));
        return x_t + (1.0 - decay) * (x_inf - x_t);
    }

    DUAL_EXEC void eval_gate_targets(double volt,
                                     VarAccessor<MechTrait>& vars,
                                     double& out_minf,
                                     double& out_mtau,
                                     double& out_hinf,
                                     double& out_htau,
                                     double& out_ninf,
                                     double& out_ntau) {
        trates(volt, vars);
        out_minf = vars(minf);
        out_mtau = vars(mtau);
        out_hinf = vars(hinf);
        out_htau = vars(htau);
        out_ninf = vars(ninf);
        out_ntau = vars(ntau);
    }

public:

    DUAL_EXEC double vtrap(double x, double y){
        if (fabs(x / y) < 1e-6)
        {
            return y * (1 - x / y / 2);
        }
        else
        {
            return x / (exp(x / y) - 1);
        }
    }

    DUAL_EXEC void rates(double volt, VarAccessor<MechTrait> &vars)
    {
        double q10, alpha, beta, sum;
        q10 = pow(3.0, (vars(celsius) - 6.3) / 10);

        alpha = 0.1 * vtrap(-(volt + 40), 10);
        beta = 4 * exp(-(volt + 65) / 18);
        sum = alpha + beta;
        vars(mtau) = 1.0 / (q10 * sum);
        vars(minf) = alpha / sum;

        alpha = 0.07 * exp(-(volt + 65) / 20);
        beta = 1.0 / (exp(-(volt + 35) / 10) + 1);
        sum = alpha + beta;
        vars(htau) = 1.0 / (q10 * sum);
        vars(hinf) = alpha / sum;

        alpha = 0.01 * vtrap(-(volt + 55), 10);
        beta = 0.125 * exp(-(volt + 65) / 80);
        sum = alpha + beta;
        vars(ntau) = 1.0 / (q10 * sum);
        vars(ninf) = alpha / sum;
    }

    DUAL_EXEC void trates(double volt, VarAccessor<MechTrait> &vars)
    {
        if (table_try_lookup<TableSpec>(volt, vars))
        {
            return;
        }
        rates(volt, vars);
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars){
        vars(ena) = vars(_ion_ena);
        vars(ek) = vars(_ion_ek);

        trates(param.volt, vars);

        //printf("celsius: %f\n", vars(celsius));

        vars(m) = vars(minf);
        vars(h) = vars(hinf);
        vars(n) = vars(ninf);
    }


    DUAL_EXEC double hh_cal_current(double volt,
                                    double _gnabar,
                                    double _gkbar,
                                    double _gl,
                                    double _m,
                                    double _h,
                                    double _n,
                                    double _ena,
                                    double _ek,
                                    double _el,
                                    double& gna,
                                    double& ina,
                                    double& gk,
                                    double& ik,
                                    double& il)
    {
        gna = _gnabar * _m * _m * _m * _h;
        ina = gna * (volt - _ena);
        gk = _gkbar * _n * _n * _n * _n;
        ik = gk * (volt - _ek);
        il = _gl * (volt - _el);
        return ina + ik + il;
    }

    DUAL_EXEC void eval_matrix_terms(double volt,
                                     double _gnabar,
                                     double _gkbar,
                                     double _gl,
                                     double _m,
                                     double _h,
                                     double _n,
                                     double _ena,
                                     double _ek,
                                     double _el,
                                     double& rhs_matrix,
                                     double& d_matrix) {
        double gna0, ina0, gk0, ik0, il0;
        double gna1, ina1, gk1, ik1, il1;
        const double i0 = hh_cal_current(
            volt, _gnabar, _gkbar, _gl, _m, _h, _n, _ena, _ek, _el, gna0, ina0, gk0, ik0, il0);
        const double i1 = hh_cal_current(
            volt + 1e-3, _gnabar, _gkbar, _gl, _m, _h, _n, _ena, _ek, _el, gna1, ina1, gk1, ik1, il1);
        rhs_matrix = -i0;
        d_matrix = (i1 - i0) / 1e-3;
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam &param, VarAccessor<MechTrait> &vars) {
        const double _gnabar = vars(gnabar);
        const double _gkbar = vars(gkbar);
        const double _m = vars(m);
        const double _h = vars(h);
        const double _n = vars(n);
        const double gna = _gnabar * _m * _m * _m * _h;
        const double gk = _gkbar * _n * _n * _n * _n;
        const double glocal = vars(gl);
        // MechTemp converts matrix-domain grad_rhs to current-domain grad_mech_current.
        // The current baseline does not propagate dynamic-D tape.
        const double di_dv = gna + gk + glocal;
        const double dloss_dv = param.grad_mech_current * di_dv;
        mechAtomAdd(&param.grad_v[param.node_index], dloss_dv);
    }


    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars){
        vars(ena) = vars(_ion_ena);
        vars(ek) = vars(_ion_ek);
        double current = hh_cal_current(param.volt,
            vars(gnabar), vars(gkbar), vars(gl), vars(m), vars(h), vars(n),
            vars(ena), vars(ek), vars(el),
            vars(gna), vars(ina), vars(gk), vars(ik), vars(il));
        if(param.updateIon){
            mechAtomAdd(&vars(_ion_ina), vars(ina));
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return current;
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars){
        trates(param.volt, vars);
        double _minf = vars(minf);
        double _mtau = vars(mtau);
        double _hinf = vars(hinf);
        double _htau = vars(htau);
        double _ninf = vars(ninf);
        double _ntau = vars(ntau);

        auto dt = param.dt;
        vars(m) = vars(m) + (1.0 - exp(dt * (-1.0 / _mtau))) * ((_minf / _mtau) / (1.0 / _mtau) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(dt * (-1.0 / _htau))) * ((_hinf / _htau) / (1.0 / _htau) - vars(h));
        vars(n) = vars(n) + (1.0 - exp(dt * (-1.0 / _ntau))) * ((_ninf / _ntau) / (1.0 / _ntau) - vars(n));
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG,MECH_CLASS_NAME);

//清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

}
