// hh3_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>
#include <array>

namespace HH3_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME HH3
static const char *MECH_NAME_TO_REG = "hh3_la";

struct MechTrait {
    enum class VarNames {
        // 状态变量
        m, h, n, s, n2,

        // 当前
        ina, ik, il,

        // 电导
        gnabar, gkbar, gkbar2, gl,

        // 特征电压
        el, vshift,

        // 中间变量
        minf, hinf, ninf, sinf, n2inf,
        mtau, htau, ntau, stau, n2tau,

        // 离子反转电势 - 存在本地变量中以便访问
        ena, ek,

        // 内部使用的_g
        _g
    };

    enum class GlobalVarNames {
        // 全局变量
        celsius,
        taum, tauh, taus, tausv, tausd,
        taun, taun2, mN, nN, sN, tausb
    };

    enum class IonVarNames {
        // 离子通道相关变量
        _ion_ena, _ion_ina, _ion_ek, _ion_ik
    };
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // 需要实现INIT, CURRENT和STATE函数
    // 根据.mod文件中的定义，此机制需要实现CURRENT和STATE功能
    // mod文件没有INITIAL块，但为了读取ena和ek，我们需要保留INIT功能
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    static constexpr auto LearnableVars = std::array{gnabar, gkbar, gkbar2, gl, el};

    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // 设置默认初始值
        init_values.insert({gnabar, 0.20});
        init_values.insert({gkbar, 0.12});
        init_values.insert({gkbar2, 0.12});
        init_values.insert({gl, 0.0001});
        init_values.insert({el, -70.0});
        init_values.insert({vshift, 0.0});

        // 在coredata中的变量索引
        var_in_coredata_idx.insert({gnabar, 0});
        var_in_coredata_idx.insert({gkbar, 1});
        var_in_coredata_idx.insert({gkbar2, 2});
        var_in_coredata_idx.insert({gl, 3});
        var_in_coredata_idx.insert({el, 4});
        var_in_coredata_idx.insert({vshift, 5});
        var_in_coredata_idx.insert({il, 6});
        var_in_coredata_idx.insert({m, 7});
        var_in_coredata_idx.insert({h, 8});
        var_in_coredata_idx.insert({n, 9});
        var_in_coredata_idx.insert({s, 10});
        var_in_coredata_idx.insert({n2, 11});
        var_in_coredata_idx.insert({ena, 12});
        var_in_coredata_idx.insert({ek, 13});
        var_in_coredata_idx.insert({_g, 22});

        // 注册全局变量 - 名称需与NEURON中一致
        global_info_map.insert({celsius, {"celsius"}});
        global_info_map.insert({taum, {"taum_hh3_la"}});
        global_info_map.insert({tauh, {"tauh_hh3_la"}});
        global_info_map.insert({taus, {"taus_hh3_la"}});
        global_info_map.insert({tausv, {"tausv_hh3_la"}});
        global_info_map.insert({tausd, {"tausd_hh3_la"}});
        global_info_map.insert({taun, {"taun_hh3_la"}});
        global_info_map.insert({taun2, {"taun2_hh3_la"}});
        global_info_map.insert({mN, {"mN_hh3_la"}});
        global_info_map.insert({nN, {"nN_hh3_la"}});
        global_info_map.insert({sN, {"sN_hh3_la"}});
        global_info_map.insert({tausb, {"tausb_hh3_la"}});

        // 注册离子通道变量
        ion_var_map.insert({_ion_ena, {"na_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ina, {"na_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // 初始化状态变量（使 finitialize() 可重复/幂等）
    //
    // 这个 .mod 没有 INITIAL，也没有为 STATE 声明初值；在 NEURON 里这类 STATE
    // 在 finitialize() 后会被置为 0，并在第一个时间步通过 states() 逐步收敛。
    //
    // 如果这里不做 INIT（不加 ENABLE_INIT），那么第二次及之后 finitialize() 时这些 STATE
    // 会保留上一次 run() 的值，从而造成多次 run 的轨迹漂移（强刺激时会被放大到 mV 级）。
    DUAL_EXEC void init_single_node(MechTempInitParam& param, VarAccessor<MechTrait>& vars) {
        (void)param;
        vars(m) = 0.0;
        vars(h) = 0.0;
        vars(n) = 0.0;
        vars(s) = 0.0;
        vars(n2) = 0.0;
    }

    // 计算电流
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ena) = vars(_ion_ena);
        vars(ek) = vars(_ion_ek);

        // 计算电导和电流 - 按照原始mod文件中的公式计算
        // ina = gnabar*h*s^sN*(v - ena)*m^mN
        // ik = gkbar*(v - ek)*n^nN+gkbar2*(v - ek)*n2^nN
        // il = gl*(v - el)
        vars(ina) = vars(gnabar) * vars(h) * pow(vars(s), vars(sN)) * (param.volt - vars(ena)) * pow(vars(m), vars(mN));
        vars(ik) = vars(gkbar) * (param.volt - vars(ek)) * pow(vars(n), vars(nN)) +
                   vars(gkbar2) * (param.volt - vars(ek)) * pow(vars(n2), vars(nN));
        vars(il) = vars(gl) * (param.volt - vars(el));

        // 如果需要更新离子通道中的电流
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ina), vars(ina));
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        // 返回总电流
        return vars(ina) + vars(ik) + vars(il);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam &param, VarAccessor<MechTrait> &vars) {
        vars.idx = param.idx;
        const double _m = vars(m);
        const double _h = vars(h);
        const double _n = vars(n);
        const double _s = vars(s);
        const double _n2 = vars(n2);
        const double gna = vars(gnabar) * _h * pow(_s, vars(sN)) * pow(_m, vars(mN));
        const double gk = vars(gkbar) * pow(_n, vars(nN));
        const double gk2 = vars(gkbar2) * pow(_n2, vars(nN));
        const double glocal = vars(gl);
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * (gna + gk + gk2 + glocal));
    }


    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 直接按照原始mod文件中的PROCEDURE states()实现
        // 计算sigmoid因子
        double sigmas = 1.0 / (1.0 + exp((param.volt + vars(tausv) + vars(vshift)) / vars(tausd)));

        // 更新门控变量，完全按照原始公式:
        // m = m + (1 - exp(-dt/taum))*(1 / (1 + exp((v + 40+vshift)/(-3)))  - m)
        // h = h + (1 - exp(-dt/tauh))*(1 / (1 + exp((v + 45+vshift)/3))  - h)
        // s = s + (1 - exp(-dt/(taus*sigmas+tausb)))*(1 / (1 + exp((v + 44+vshift)/3))  - s)
        // n = n + (1 - exp(-dt/taun))*(1 / (1 + exp((v + 40+vshift)/(-3)))  - n)
        // n2 = n2 + (1 - exp(-dt/taun2))*(1 / (1 + exp((v + 40+vshift)/(-3)))  - n2)

        vars(m) = vars(m) + (1.0 - exp(-param.dt / vars(taum))) *
                 (1.0 / (1.0 + exp((param.volt + 40.0 + vars(vshift)) / (-3.0))) - vars(m));

        vars(h) = vars(h) + (1.0 - exp(-param.dt / vars(tauh))) *
                 (1.0 / (1.0 + exp((param.volt + 45.0 + vars(vshift)) / 3.0)) - vars(h));

        vars(s) = vars(s) + (1.0 - exp(-param.dt / (vars(taus) * sigmas + vars(tausb)))) *
                 (1.0 / (1.0 + exp((param.volt + 44.0 + vars(vshift)) / 3.0)) - vars(s));

        vars(n) = vars(n) + (1.0 - exp(-param.dt / vars(taun))) *
                 (1.0 / (1.0 + exp((param.volt + 40.0 + vars(vshift)) / (-3.0))) - vars(n));

        vars(n2) = vars(n2) + (1.0 - exp(-param.dt / vars(taun2))) *
                  (1.0 / (1.0 + exp((param.volt + 40.0 + vars(vshift)) / (-3.0))) - vars(n2));
    }

};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace HH3_LA
