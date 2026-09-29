// kdr_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>
#include <array>

namespace KDR_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME KDR
static const char *MECH_NAME_TO_REG = "kdr_la";

struct MechTrait {
    enum class VarNames {
        // 参数
        gbar,

        // 状态变量
        m,

        // 计算变量
        ik, ek,

        // 中间变量
        minf, mtau,

        // 导数变量
        Dm,

        // 内部使用的_g
        _g
    };

    enum class IonVarNames {
        // 离子通道相关变量
        _ion_ek, _ion_ik
    };
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // 需要实现INIT, CURRENT和STATE函数
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE |
        ENABLE_CURRENT_VJP;

    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    static constexpr auto LearnableVars = std::array{gbar};

    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // 设置默认初始值
        init_values.insert({gbar, 0.0});  // mho/cm2

        // 在coredata中的变量索引 - 根据变量在mod文件中出现的顺序
        var_in_coredata_idx.insert({gbar, 0});
        var_in_coredata_idx.insert({ik, 1});
        var_in_coredata_idx.insert({minf, 2});
        var_in_coredata_idx.insert({mtau, 3});
        var_in_coredata_idx.insert({m, 4});
        var_in_coredata_idx.insert({Dm, 5});
        var_in_coredata_idx.insert({ek, 6});
        var_in_coredata_idx.insert({_g, 7});

        // 注册离子通道变量
        ion_var_map.insert({_ion_ek, {"k_ion", EionVarNames::erev}});
        ion_var_map.insert({_ion_ik, {"k_ion", EionVarNames::cur}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // settables函数 - 计算速率常数
    DUAL_EXEC void settables(double v, VarAccessor<MechTrait> &vars) {
        // 计算激活门的稳态值
        vars(minf) = 1.0 / (1.0 + exp((-v - 29.5) / 10.0));

        // 计算时间常数
        if (v < -10.0) {
            vars(mtau) = 0.25 + 4.35 * exp((v + 10.0) / 10.0);
        } else {
            vars(mtau) = 0.25 + 4.35 * exp((-v - 10.0) / 10.0);
        }
    }

    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ek) = vars(_ion_ek);

        // 计算速率常数
        settables(param.volt, vars);

        // 初始化门控变量 - 注意这里根据mod文件中的INITIAL块，先设置为minf，然后覆盖为0
        vars(m) = vars(minf);
        vars(m) = 0.0;
    }

    // 计算电流
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道读取反转电势
        vars(ek) = vars(_ion_ek);

        // 计算电流 - m^4 * gbar * (v - ek)
        double m_4 = vars(m) * vars(m) * vars(m) * vars(m);
        vars(ik) = vars(gbar) * m_4 * (param.volt - vars(ek));

        // 如果需要更新离子通道中的电流
        if (param.updateIon) {
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }

        // 返回总电流
        return vars(ik);
    }

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam &param, VarAccessor<MechTrait> &vars) {
        vars.idx = param.idx;
        const double m_t = vars(m);
        const double gk = vars(gbar) * pow(m_t, 4.0);
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * gk);
    }


    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 计算速率常数
        settables(param.volt, vars);

        // 更新门控变量（使用指数欧拉法）
        // m' = (minf - m) / mtau
        vars(m) = vars(m) + (1.0 - exp(-param.dt / vars(mtau))) * (vars(minf) - vars(m));
    }

};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace KDR_LA
