// km_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace KM_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME KM
static const char *MECH_NAME_TO_REG = "km_la";

struct MechTrait {
    enum class VarNames {
        // 状态变量
        n,
        
        // 参数
        gbar,
        
        // 计算变量
        gk, ninf, nexp, ntau, a, b, ik, ek,
        
        // 导数变量
        Dn,
        
        // 内部使用的_g
        _g
    };

    enum class GlobalVarNames {
        // 全局变量
        celsius, temp, q10, tadj, vmin, vmax,
        tha, qa, Ra, Rb, usetable
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
        init_values.insert({gbar, 10.0});  // pS/um2
        
        // 在coredata中的变量索引
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({gk, 1});
        var_in_coredata_idx.insert({ninf, 2});
        var_in_coredata_idx.insert({nexp, 3});
        var_in_coredata_idx.insert({ntau, 4});
        var_in_coredata_idx.insert({n, 5});
        var_in_coredata_idx.insert({a, 6});
        var_in_coredata_idx.insert({b, 7});
        var_in_coredata_idx.insert({ik, 8});
        var_in_coredata_idx.insert({ek, 9});
        var_in_coredata_idx.insert({Dn, 10});
        var_in_coredata_idx.insert({_g, 12});

        // 注册全局变量 - 名称需与NEURON中一致
        global_info_map.insert({celsius, {"celsius"}});
        global_info_map.insert({temp, {"temp_km_la"}});
        global_info_map.insert({q10, {"q10_km_la"}});
        global_info_map.insert({tadj, {"tadj_km_la"}});
        global_info_map.insert({vmin, {"vmin_km_la"}});
        global_info_map.insert({vmax, {"vmax_km_la"}});
        global_info_map.insert({tha, {"tha_km_la"}});
        global_info_map.insert({qa, {"qa_km_la"}});
        global_info_map.insert({Ra, {"Ra_km_la"}});
        global_info_map.insert({Rb, {"Rb_km_la"}});
        global_info_map.insert({usetable, {"usetable_km_la"}});

        // 注册离子通道变量
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // rates函数 - 计算门控变量的速率常数
    DUAL_EXEC void rates(double v, VarAccessor<MechTrait> &vars) {
        // 计算激活和失活速率
        vars(a) = vars(Ra) * (v - vars(tha)) / (1.0 - exp(-(v - vars(tha)) / vars(qa)));
        vars(b) = -vars(Rb) * (v - vars(tha)) / (1.0 - exp((v - vars(tha)) / vars(qa)));
        
        // 计算时间常数和稳态值
        vars(ntau) = 1.0 / (vars(a) + vars(b));
        vars(ninf) = vars(a) * vars(ntau);
    }

    // trates函数 - 计算速率常数和指数因子
    DUAL_EXEC void trates(double v, VarAccessor<MechTrait> &vars, double dt) {
        // 计算速率常数
        rates(v, vars);
        
        // 计算指数因子
        double tinc = -dt * vars(tadj);
        vars(nexp) = 1.0 - exp(tinc / vars(ntau));
    }

    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ek) = vars(_ion_ek);
        
        // 计算速率常数
        trates(param.volt, vars, param.dt);
        
        // 初始化门控变量
        vars(n) = vars(ninf);
    }

    // 计算电流
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ek) = vars(_ion_ek);
        
        // 计算电导和电流
        vars(gk) = vars(tadj) * vars(gbar) * vars(n);
        vars(ik) = (1e-4) * vars(gk) * (param.volt - vars(ek));
        
        // 如果需要更新离子通道中的电流
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        
        // 返回总电流
        return vars(ik);
    }

    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 计算速率常数和指数因子
        trates(param.volt, vars, param.dt);
        
        // 更新门控变量
        vars(n) = vars(n) + vars(nexp) * (vars(ninf) - vars(n));
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace KM_LA