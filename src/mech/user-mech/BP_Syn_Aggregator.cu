// BP_Syn_Aggregator mechanism - Gradient aggregator for backpropagation
#include "mech_template.cuh"
#include <cstdio>

namespace BP_Syn_Aggregator {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        lr_start,        // learning start time (ms)
        lr_end,          // learning end time (ms)
        n_outputs,       // number of outputs

        // ASSIGNED variables
        grad_from_output, // 50-dimensional array for gradients from output layer
        aggregated_grad,  // aggregated gradient result
        is_learning      // learning state flag
    };
};

class BP_Syn_Aggregator_Mech : public MechTemp<BP_Syn_Aggregator_Mech, MechTrait> {
public:
    using enum MechTrait::VarNames;

    // 根据机制特点设置标志：POINT_PROCESS，且仅在 state 阶段运行
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | POINT_PROCESS;
    constexpr static int max_array_size = 256; // 最大数组大小
    BP_Syn_Aggregator_Mech(MechInitParams &param) : MechTemp(param) {
        // 设置默认值（从CPP文件中的_parm_default获取）
        init_values.insert({lr_start, 0.0});
        init_values.insert({lr_end, 0.0});
        init_values.insert({n_outputs, 10.0});
        init_values.insert({is_learning, 0.0});
        init_values.insert({aggregated_grad, 0.0});

        // 注册变量索引（按NEURON CPP中的顺序）
        // 从CPP文件中可以看到：
        // lr_start: fpfield<0>, lr_end: fpfield<1>, n_outputs: fpfield<2>
        // grad_from_output: data_array<3, 50>, aggregated_grad: fpfield<4>, is_learning: fpfield<5>
        var_in_coredata_idx.insert({lr_start, 0});
        var_in_coredata_idx.insert({lr_end, 1});
        var_in_coredata_idx.insert({n_outputs, 2});
        var_in_coredata_idx.insert({grad_from_output, {3, max_array_size}}); // 数组变量：索引3，长度50
        var_in_coredata_idx.insert({aggregated_grad, 4});
        var_in_coredata_idx.insert({is_learning, 5});
    }

    // 初始化函数：对应INITIAL块
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(is_learning) = 0.0;
        vars(aggregated_grad) = 0.0;

        // 初始化50维数组
        for (int i = 0; i < max_array_size; i++) {
            vars.Arr(grad_from_output, i) = 0.0;
        }
    }

    // 状态更新函数：对应BREAKPOINT块（NEURON中被编译到state阶段）
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 判断是否在学习时间窗口内
        if (param.t < vars(lr_start)) {
            vars(is_learning) = 0.0;
        } else if (param.t > vars(lr_end)) {
            vars(is_learning) = 0.0;
        } else {
            vars(is_learning) = 1.0;
        }

        // 聚合梯度
        vars(aggregated_grad) = 0.0;
        if (vars(is_learning) == 1.0) {
            int n_out = static_cast<int>(vars(n_outputs));
            for (int i = 0; i < n_out; i++) {
                vars(aggregated_grad) += vars.Arr(grad_from_output, i);
            }
        }
    }

    // 机制不产生电流，保留空实现以满足模板接口
    DUAL_EXEC double current_single_node(MechTempCurParam&, VarAccessor<MechTrait>&) {
        return 0.0;
    }
};

REGISTER_MECHANISM("BP_Syn_Aggregator", BP_Syn_Aggregator_Mech);

} // namespace BP_Syn_Aggregator
