// BP_Syn_FullyConnected mechanism - Fully connected layer with backpropagation
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace BP_Syn_FullyConnected {

struct MechTrait {
    enum class VarNames {
        // PARAMETER变量
        lr_start, lr_end, w, g,
        // ASSIGNED变量
        i, grad_from_next, grad_to_prev, v_gap, PI, acc_grad,
        // 内部变量
        is_learning, v, _g
    };
};

class BP_Syn_FullyConnected_Mech : public MechTemp<BP_Syn_FullyConnected_Mech, MechTrait> {
public:
    using enum MechTrait::VarNames;

    // 根据MOD文件特点设置标志：POINT_PROCESS + ELECTRODE_CURRENT + 有acc_grad需要初始化
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT;

    BP_Syn_FullyConnected_Mech(MechInitParams &param) : MechTemp(param) {
        // 设置默认值（来自CPP文件的_parm_default）
        init_values.insert({lr_start, 0.0});
        init_values.insert({lr_end, 0.0});
        init_values.insert({w, 0.0});
        init_values.insert({g, 0.01});

        // 注册变量索引（按NEURON CPP中fpfield的顺序）
        var_in_coredata_idx.insert({lr_start, 0});      // lr_start_columnindex 0
        var_in_coredata_idx.insert({lr_end, 1});        // lr_end_columnindex 1
        var_in_coredata_idx.insert({w, 2});             // w_columnindex 2
        var_in_coredata_idx.insert({g, 3});             // g_columnindex 3
        var_in_coredata_idx.insert({i, 4});             // i_columnindex 4
        var_in_coredata_idx.insert({grad_from_next, 5}); // grad_from_next_columnindex 5
        var_in_coredata_idx.insert({grad_to_prev, 6});  // grad_to_prev_columnindex 6
        var_in_coredata_idx.insert({v_gap, 7});         // v_gap_columnindex 7
        var_in_coredata_idx.insert({PI, 8});            // PI_columnindex 8
        var_in_coredata_idx.insert({acc_grad, 9});      // acc_grad_columnindex 9
        var_in_coredata_idx.insert({is_learning, 10});  // is_learning_columnindex 10
        var_in_coredata_idx.insert({v, 11});            // v_columnindex 11
        var_in_coredata_idx.insert({_g, 12});           // _g_columnindex 12
    }

    // 初始化函数：对应MOD文件的INITIAL块
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 从MOD文件INITIAL块：
        // is_learning = 0
        // acc_grad = 0
        vars(is_learning) = 0.0;
        vars(acc_grad) = 0.0;
        vars(grad_from_next) = 0.0;
        vars(grad_to_prev) = 0.0;
        vars(PI) = 0.0;
        vars(i) = 0.0;
    }

    // 电流计算函数：对应MOD文件的BREAKPOINT块
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 更新电压
        vars(v) = param.volt;

        // 学习期间判断（从CPP文件的_nrn_current函数）
        if (vars(lr_start) < param.t && param.t < vars(lr_end)) {
            vars(is_learning) = 1.0;
        } else {
            vars(is_learning) = 0.0;
        }

        // 学习期间的梯度计算
        if (vars(is_learning) == 1.0) {
            if (vars(v_gap) > 0.0) {
                vars(grad_to_prev) = vars(grad_from_next) * vars(w);
                vars(PI) = vars(grad_from_next) * vars(v_gap);
            } else {
                vars(grad_to_prev) = 0.0;
                vars(PI) = 0.0;
            }
            vars(acc_grad) = vars(acc_grad) + vars(PI);
        }

        // 电流计算
        if (vars(v_gap) > 0.0) {
            vars(i) = vars(g) * vars(w) * vars(v_gap);
        } else {
            vars(i) = 0.0;
        }

        return vars(i);
    }
};

REGISTER_MECHANISM("BP_Syn_FullyConnected", BP_Syn_FullyConnected_Mech);

} // namespace BP_Syn_FullyConnected
