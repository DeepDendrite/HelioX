// kap_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace KAP_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME KAP
static const char *MECH_NAME_TO_REG = "kap_la";

struct MechTrait {
    enum class VarNames {
        // 状态变量
        n, l,
        
        // 参数
        gkabar, lmin,
        
        // 计算变量
        ninf, linf, taul, taun, gka, ik, ek,
        
        // 导数变量
        Dn, Dl,
        
        // 内部使用的_g
        _g
    };

    enum class GlobalVarNames {
        // 全局变量
        celsius,
        vhalfn, vhalfl, a0l, a0n, zetan, zetal, gmn, gml, nmin,
        pw, tq, qq, q10, qtl
    };

    enum class IonVarNames {
        // 离子通道相关变量
        _ion_ek, _ion_ik
    };
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // 需要实现INIT, CURRENT和STATE函数
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // 设置默认初始值
        init_values.insert({gkabar, 0.008});  // mho/cm2
        init_values.insert({lmin, 2.0});      // mS
        
        // 在coredata中的变量索引
        var_in_coredata_idx.insert({gkabar, 0});
        var_in_coredata_idx.insert({lmin, 1});
        var_in_coredata_idx.insert({ninf, 2});
        var_in_coredata_idx.insert({linf, 3});
        var_in_coredata_idx.insert({taul, 4});
        var_in_coredata_idx.insert({taun, 5});
        var_in_coredata_idx.insert({gka, 6});
        var_in_coredata_idx.insert({n, 7});
        var_in_coredata_idx.insert({l, 8});
        var_in_coredata_idx.insert({ek, 9});
        var_in_coredata_idx.insert({Dn, 10});
        var_in_coredata_idx.insert({Dl, 11});
        var_in_coredata_idx.insert({ik, 12});
        var_in_coredata_idx.insert({_g, 14});

        // 注册全局变量 - 名称需与NEURON中一致
        global_info_map.insert({celsius, {"celsius"}});
        global_info_map.insert({vhalfn, {"vhalfn_kap_la"}});
        global_info_map.insert({vhalfl, {"vhalfl_kap_la"}});
        global_info_map.insert({a0l, {"a0l_kap_la"}});
        global_info_map.insert({a0n, {"a0n_kap_la"}});
        global_info_map.insert({zetan, {"zetan_kap_la"}});
        global_info_map.insert({zetal, {"zetal_kap_la"}});
        global_info_map.insert({gmn, {"gmn_kap_la"}});
        global_info_map.insert({gml, {"gml_kap_la"}});
        global_info_map.insert({nmin, {"nmin_kap_la"}});
        global_info_map.insert({pw, {"pw_kap_la"}});
        global_info_map.insert({tq, {"tq_kap_la"}});
        global_info_map.insert({qq, {"qq_kap_la"}});
        global_info_map.insert({q10, {"q10_kap_la"}});
        global_info_map.insert({qtl, {"qtl_kap_la"}});

        // 注册离子通道变量
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // alpn函数 - 计算n的激活率
    DUAL_EXEC double alpn(double v, VarAccessor<MechTrait> &vars) {
        double zeta = vars(zetan) + vars(pw) / (1.0 + exp((v - vars(tq)) / vars(qq)));
        return exp(1.e-3 * zeta * (v - vars(vhalfn)) * 9.648e4 / (8.315 * (273.16 + vars(celsius))));
    }

    // betn函数 - 计算n的失活率
    DUAL_EXEC double betn(double v, VarAccessor<MechTrait> &vars) {
        double zeta = vars(zetan) + vars(pw) / (1.0 + exp((v - vars(tq)) / vars(qq)));
        return exp(1.e-3 * zeta * vars(gmn) * (v - vars(vhalfn)) * 9.648e4 / (8.315 * (273.16 + vars(celsius))));
    }

    // alpl函数 - 计算l的激活率
    DUAL_EXEC double alpl(double v, VarAccessor<MechTrait> &vars) {
        return exp(1.e-3 * vars(zetal) * (v - vars(vhalfl)) * 9.648e4 / (8.315 * (273.16 + vars(celsius))));
    }

    // betl函数 - 计算l的失活率
    DUAL_EXEC double betl(double v, VarAccessor<MechTrait> &vars) {
        return exp(1.e-3 * vars(zetal) * vars(gml) * (v - vars(vhalfl)) * 9.648e4 / (8.315 * (273.16 + vars(celsius))));
    }

    // rates函数 - 计算速率常数
    DUAL_EXEC void rates(double v, VarAccessor<MechTrait> &vars) {
        double a, qt;
        
        // 温度校正系数
        qt = pow(vars(q10), ((vars(celsius) - 24.0) / 10.0));
        
        // 计算n门激活相关参数
        a = alpn(v, vars);
        vars(ninf) = 1.0 / (1.0 + a);
        vars(taun) = betn(v, vars) / (qt * vars(a0n) * (1.0 + a));
        if (vars(taun) < vars(nmin)) {
            vars(taun) = vars(nmin);
        }
        
        // 计算l门激活相关参数
        a = alpl(v, vars);
        vars(linf) = 1.0 / (1.0 + a);
        vars(taul) = 0.26 * (v + 50.0) / vars(qtl);
        if (vars(taul) < vars(lmin) / vars(qtl)) {
            vars(taul) = vars(lmin) / vars(qtl);
        }
    }

    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ek) = vars(_ion_ek);
        
        // 计算速率常数
        rates(param.volt, vars);
        
        // 初始化门控变量
        vars(n) = vars(ninf);
        vars(l) = vars(linf);
    }

    // 计算电流
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ek) = vars(_ion_ek);
        
        // 计算电导和电流
        vars(gka) = vars(gkabar) * vars(n) * vars(l);
        vars(ik) = vars(gka) * (param.volt - vars(ek));
        
        // 如果需要更新离子通道中的电流
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        // 返回总电流
        return vars(ik);
    }

    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 计算速率常数
        rates(param.volt, vars);
        
        // 更新门控变量（使用指数欧拉法）
        vars(n) = vars(n) + (1.0 - exp(-param.dt / vars(taun))) * (vars(ninf) - vars(n));
        vars(l) = vars(l) + (1.0 - exp(-param.dt / vars(taul))) * (vars(linf) - vars(l));
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace KAP_LA