// km_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>
#include <array>

namespace KM_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME KM
static const char *MECH_NAME_TO_REG = "km_la";

struct MechTrait {
    static constexpr bool MechSupportsTable = true;

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
        tha, qa, Ra, Rb, usetable_km_la
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
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    static constexpr auto LearnableVars = std::array{gbar};

    struct TableSpec {
        static constexpr int kTableBins = 199;
        static constexpr int kOutputCount = 2;
        static constexpr bool dep_dt = true;
        static constexpr auto dep_global_vars = std::array{
            MechTrait::GlobalVarNames::celsius,
            MechTrait::GlobalVarNames::temp,
            MechTrait::GlobalVarNames::Ra,
            MechTrait::GlobalVarNames::Rb,
            MechTrait::GlobalVarNames::tha,
            MechTrait::GlobalVarNames::qa,
        };
        static constexpr std::array<MechTrait::VarNames, 0> dep_range_vars{};

        DUAL_EXEC MechTrait::VarNames output_var(int out_idx) {
            return (out_idx == 0) ? MechTrait::VarNames::ninf : MechTrait::VarNames::nexp;
        }

        DUAL_EXEC bool use_table(VarAccessor<MechTrait>& vars) {
            return vars(MechTrait::GlobalVarNames::usetable_km_la) != 0.0;
        }
        DUAL_EXEC double table_min(VarAccessor<MechTrait>& vars) {
            return vars(MechTrait::GlobalVarNames::vmin);
        }
        DUAL_EXEC double table_max(VarAccessor<MechTrait>& vars) {
            return vars(MechTrait::GlobalVarNames::vmax);
        }
        DUAL_EXEC void compute_table_outputs_at_point(double v, double dt, VarAccessor<MechTrait>& vars) {
            vars(MechTrait::VarNames::a) =
                vars(MechTrait::GlobalVarNames::Ra) * (v - vars(MechTrait::GlobalVarNames::tha)) /
                (1.0 - exp(-(v - vars(MechTrait::GlobalVarNames::tha)) /
                           vars(MechTrait::GlobalVarNames::qa)));
            vars(MechTrait::VarNames::b) =
                -vars(MechTrait::GlobalVarNames::Rb) * (v - vars(MechTrait::GlobalVarNames::tha)) /
                (1.0 - exp((v - vars(MechTrait::GlobalVarNames::tha)) /
                           vars(MechTrait::GlobalVarNames::qa)));
            vars(MechTrait::VarNames::ntau) =
                1.0 / (vars(MechTrait::VarNames::a) + vars(MechTrait::VarNames::b));
            vars(MechTrait::VarNames::ninf) =
                vars(MechTrait::VarNames::a) * vars(MechTrait::VarNames::ntau);

            const double tinc = -dt * vars(MechTrait::GlobalVarNames::tadj);
            vars(MechTrait::VarNames::nexp) =
                1.0 - exp(tinc / vars(MechTrait::VarNames::ntau));
        }
    };

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
        global_info_map.insert({usetable_km_la, {"usetable_km_la"}});
        if (!coreneuron::global_var_map.contains("usetable_km_la")) {
            coreneuron::global_var_map["usetable_km_la"] = std::vector<double>{0.0};
        }

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
        if (table_try_lookup<TableSpec>(v, vars)) {
            return;
        }
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

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam &param, VarAccessor<MechTrait> &vars) {
        vars.idx = param.idx;
        const double n_t = vars(n);
        const double gk = vars(tadj) * vars(gbar) * n_t;
        mechAtomAdd(&param.grad_v[param.node_index], param.grad_mech_current * gk * 1e-4);
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
