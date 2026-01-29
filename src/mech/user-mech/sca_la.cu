// sca_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace SCA_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME SlowCa
static const char *MECH_NAME_TO_REG = "sca_la";

struct MechTrait {
    enum class VarNames {
        // 状态变量
        m, h,
        
        // 参数
        inactF, actF, gbar,
        
        // 计算变量
        gca, ica, eca,
        
        // 中间变量
        minf, hinf, mexp, hexp, mtau, htau,
        
        // 内部使用的_g
        _g
    };

    enum class GlobalVarNames {
        // 全局变量
        celsius, temp, q10, tadj,
        vshift, vmin, vmax,
        cao, cai, usetable
    };

    enum class IonVarNames {
        // 离子通道相关变量
        _ion_eca, _ion_ica
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
        init_values.insert({inactF, 3.0});
        init_values.insert({actF, 1.0});
        init_values.insert({gbar, 0.1});
        
        // 在coredata中的变量索引
        var_in_coredata_idx.insert({inactF, 0});
        var_in_coredata_idx.insert({actF, 1});
        var_in_coredata_idx.insert({gbar, 2});
        var_in_coredata_idx.insert({gca, 3});
        var_in_coredata_idx.insert({minf, 4});
        var_in_coredata_idx.insert({hinf, 5});
        var_in_coredata_idx.insert({mexp, 6});
        var_in_coredata_idx.insert({hexp, 7});
        var_in_coredata_idx.insert({mtau, 8});
        var_in_coredata_idx.insert({htau, 9});
        var_in_coredata_idx.insert({m, 10});
        var_in_coredata_idx.insert({h, 11});
        var_in_coredata_idx.insert({ica, 12});
        var_in_coredata_idx.insert({eca, 13});
        var_in_coredata_idx.insert({_g, 17});

        // 注册全局变量 - 名称需与NEURON中一致
        global_info_map.insert({celsius, {"celsius"}});
        global_info_map.insert({temp, {"temp_sca_la"}});
        global_info_map.insert({q10, {"q10_sca_la"}});
        global_info_map.insert({tadj, {"tadj_sca_la"}});
        global_info_map.insert({vshift, {"vshift_sca_la"}});
        global_info_map.insert({vmin, {"vmin_sca_la"}});
        global_info_map.insert({vmax, {"vmax_sca_la"}});
        global_info_map.insert({cao, {"cao_sca_la"}});
        global_info_map.insert({cai, {"cai_sca_la"}});
        global_info_map.insert({usetable, {"usetable_sca_la"}});

        // 注册离子通道变量
        ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // efun辅助函数
    DUAL_EXEC double efun(double z) {
        if (fabs(z) < 1e-4) {
            return 1.0 - z / 2.0;
        } else {
            return z / (exp(z) - 1.0);
        }
    }

    // rates函数 - 计算门控变量的稳态值和时间常数
    DUAL_EXEC void rates(double vm, VarAccessor<MechTrait> &vars) {
        double a, b;
        
        // m门激活动力学
        a = 0.055 * (-27.0 - vm) / (exp((-27.0 - vm) / 3.8) - 1.0) / vars(actF);
        b = 0.94 * exp((-75.0 - vm) / 17.0) / vars(actF);
        
        vars(mtau) = 1.0 / (a + b);
        vars(minf) = a * vars(mtau);
        
        // h门失活动力学
        a = 0.000457 * exp((-13.0 - vm) / 50.0) / vars(inactF);
        b = 0.0065 / (exp((-vm - 15.0) / 28.0) + 1.0) / vars(inactF);
        
        vars(htau) = 1.0 / (a + b);
        vars(hinf) = a * vars(htau);
    }

    // trates函数 - 计算指数因子
    DUAL_EXEC void trates(double v, VarAccessor<MechTrait> &vars, double dt) {
        // 计算速率常数
        rates(v, vars);
        
        // 计算指数因子
        double tinc = -dt * vars(tadj);
        vars(mexp) = 1.0 - exp(tinc / vars(mtau));
        vars(hexp) = 1.0 - exp(tinc / vars(htau));
    }

    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(eca) = vars(_ion_eca);
        
        // 计算速率常数和稳态值
        trates(param.volt + vars(vshift), vars, param.dt);
        
        // 初始化门控变量
        vars(m) = vars(minf);
        vars(h) = vars(hinf);
    }

    // 计算电流
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(eca) = vars(_ion_eca);
        
        // 计算电导和电流
        vars(gca) = vars(tadj) * vars(gbar) * vars(m) * vars(m) * vars(h);
        vars(ica) = (1e-4) * vars(gca) * (param.volt - vars(eca));
        
        // 如果需要更新离子通道中的电流
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ica), vars(ica));
        }
        
        // 返回总电流
        return vars(ica);
    }

    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 更新速率常数和指数因子
        trates(param.volt + vars(vshift), vars, param.dt);
        
        // 更新门控变量
        vars(m) = vars(m) + vars(mexp) * (vars(minf) - vars(m));
        vars(h) = vars(h) + vars(hexp) * (vars(hinf) - vars(h));
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace SCA_LA