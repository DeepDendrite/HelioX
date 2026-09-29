// hd_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>
#include <array>

namespace HD_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME HD
static const char *MECH_NAME_TO_REG = "hd_la";

struct MechTrait {
    enum class VarNames {
        // 状态变量
        l,

        // 参数
        ghdbar, vhalfl, tau,

        // 计算变量
        i, linf, taul, ghd,

        // 导数变量
        Dl,

        // 内部使用的_g
        _g
    };

    enum class GlobalVarNames {
        // 全局变量
        celsius, ehd, kl, vhalft, a0t, zetat, gmt, q10, qtl
    };

    // 这个机制没有离子通道相关变量，因为它使用NONSPECIFIC_CURRENT
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // 需要实现INIT, CURRENT和STATE函数
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    static constexpr auto LearnableVars = std::array{ghdbar};

    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // 设置默认初始值
        init_values.insert({ghdbar, 0.0001});  // mho/cm2
        init_values.insert({vhalfl, -81.0});   // mV
        init_values.insert({tau, 1.0});        // ms

        // 在coredata中的变量索引
        var_in_coredata_idx.insert({ghdbar, 0});
        var_in_coredata_idx.insert({vhalfl, 1});
        var_in_coredata_idx.insert({tau, 2});
        var_in_coredata_idx.insert({i, 3});
        var_in_coredata_idx.insert({linf, 4});
        var_in_coredata_idx.insert({taul, 5});
        var_in_coredata_idx.insert({l, 6});
        var_in_coredata_idx.insert({Dl, 7});
        var_in_coredata_idx.insert({ghd, 8});
        var_in_coredata_idx.insert({_g, 10});

        // 注册全局变量 - 名称需与NEURON中一致
        global_info_map.insert({celsius, {"celsius"}});
        global_info_map.insert({ehd, {"ehd_hd_la"}});
        global_info_map.insert({kl, {"kl_hd_la"}});
        global_info_map.insert({vhalft, {"vhalft_hd_la"}});
        global_info_map.insert({a0t, {"a0t_hd_la"}});
        global_info_map.insert({zetat, {"zetat_hd_la"}});
        global_info_map.insert({gmt, {"gmt_hd_la"}});
        global_info_map.insert({q10, {"q10_hd_la"}});
        global_info_map.insert({qtl, {"qtl_hd_la"}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // alpt函数 - 计算指数因子a
    DUAL_EXEC double alpt(double v, VarAccessor<MechTrait> &vars) {
        return exp(0.0378 * vars(zetat) * (v - vars(vhalft)));
    }

    // bett函数 - 计算指数因子b
    DUAL_EXEC double bett(double v, VarAccessor<MechTrait> &vars) {
        return exp(0.0378 * vars(zetat) * vars(gmt) * (v - vars(vhalft)));
    }

    // rate函数 - 计算速率常数
    DUAL_EXEC void rate(double v, VarAccessor<MechTrait> &vars) {
        double a, qt;

        // 温度校正系数
        qt = pow(vars(q10), ((vars(celsius) - 33.0) / 10.0));

        // 计算门激活相关参数
        a = alpt(v, vars);
        vars(linf) = 1.0 / (1.0 + exp(-(v - vars(vhalfl)) / vars(kl)));
        vars(taul) = vars(tau) * bett(v, vars) / (vars(qtl) * qt * vars(a0t) * (1.0 + a));
    }

    DUAL_EXEC double l_next(double v, double l_t, VarAccessor<MechTrait> &vars, double dt) {
        rate(v, vars);
        const double alpha = 1.0 - exp(-dt / vars(taul));
        return l_t + alpha * (vars(linf) - l_t);
    }

    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 计算速率常数
        rate(param.volt, vars);

        // 初始化门控变量
        vars(l) = vars(linf);
    }

    // 计算电流
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 计算电导和电流
        vars(ghd) = vars(ghdbar) * vars(l);
        vars(i) = vars(ghd) * (param.volt - vars(ehd));

        // 返回总电流
        return vars(i);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam &param, VarAccessor<MechTrait> &vars) {
        vars.idx = param.idx;
        const double ghd = vars(ghdbar) * vars(l);
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * ghd);
    }


    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 计算速率常数
        rate(param.volt, vars);

        // 更新门控变量（使用指数欧拉法）
        vars(l) = vars(l) + (1.0 - exp(-param.dt / vars(taul))) * (vars(linf) - vars(l));
    }

};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace HD_LA
