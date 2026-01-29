// BP_Syn_SoftMax mechanism - SoftMax activation function with backpropagation
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace BP_Syn_SoftMax {

struct MechTrait {
    enum class VarNames {
        // PARAMETER variables
        lr_start,        // learning start time (ms)
        lr_end,          // learning end time (ms)
        tgt,             // target values array [10]
        n_outputs,       // number of outputs (default 10)

        // ASSIGNED variables
        i,               // current (nA)
        u,               // input values array [10]
        s_sum,           // sum for SoftMax normalization
        s,               // SoftMax output array [10]
        grad_to_prev,    // gradient to previous layer array [10]
        is_learning,     // learning state flag
        v,               // voltage (mV)
        _g               // conductance
    };
};

class BP_Syn_SoftMax_Mech : public MechTemp<BP_Syn_SoftMax_Mech, MechTrait> {
public:
    using enum MechTrait::VarNames;

    // 根据机制特点设置标志：POINT_PROCESS + ELECTRODE_CURRENT + ENABLE_INIT + ENABLE_CURRENT
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT;

    BP_Syn_SoftMax_Mech(MechInitParams &param) : MechTemp(param) {
        // 设置默认值（从CPP文件中的_parm_default获取）
        init_values.insert({lr_start, 0.0});
        init_values.insert({lr_end, 0.0});
        init_values.insert({n_outputs, 10.0});
        init_values.insert({i, 0.0});
        init_values.insert({s_sum, 0.0});
        init_values.insert({is_learning, 0.0});
        init_values.insert({v, 0.0});
        init_values.insert({_g, 0.0});

        // 注册变量索引（按NEURON CPP中的顺序）
        // 从CPP文件中可以看到变量索引：
        var_in_coredata_idx.insert({lr_start, 0});        // fpfield<0>
        var_in_coredata_idx.insert({lr_end, 1});          // fpfield<1>
        var_in_coredata_idx.insert({tgt, {2, 10}});       // data_array<2, 10>
        var_in_coredata_idx.insert({n_outputs, 3});       // fpfield<3>
        var_in_coredata_idx.insert({i, 4});               // fpfield<4>
        var_in_coredata_idx.insert({u, {5, 10}});         // data_array<5, 10>
        var_in_coredata_idx.insert({s_sum, 6});           // fpfield<6>
        var_in_coredata_idx.insert({s, {7, 10}});         // data_array<7, 10>
        var_in_coredata_idx.insert({grad_to_prev, {8, 10}}); // data_array<8, 10>
        var_in_coredata_idx.insert({is_learning, 9});     // fpfield<9>
        var_in_coredata_idx.insert({v, 10});              // fpfield<10>
        var_in_coredata_idx.insert({_g, 11});             // fpfield<11>
    }

    // 初始化函数：对应INITIAL块
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(is_learning) = 0.0;
    }

    // 电流计算函数：对应BREAKPOINT块
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 设置学习时间窗口
        if (param.t < vars(lr_start)) {
            vars(is_learning) = 0.0;
        } else if (param.t > vars(lr_end)) {
            vars(is_learning) = 0.0;
        } else {
            vars(is_learning) = 1.0;
        }

        // 计算SoftMax
        // 第一步：计算exp(u[j])的和
        vars(s_sum) = 0.0;
        int n_out = static_cast<int>(vars(n_outputs));
        for (int j = 0; j < n_out; j++) {
            vars(s_sum) += exp(vars.Arr(u, j));
        }

        // 第二步：计算SoftMax输出 s[j] = exp(u[j]) / s_sum
        for (int j = 0; j < n_out; j++) {
            vars.Arr(s, j) = exp(vars.Arr(u, j)) / vars(s_sum);
        }

        // 计算反向传播梯度（CrossEntropy + SoftMax的梯度）
        if (vars(is_learning) == 1.0) {
            for (int j = 0; j < n_out; j++) {
                // CrossEntropy + SoftMax 的梯度就是 s[j] - tgt[j]
                vars.Arr(grad_to_prev, j) = vars.Arr(s, j) - vars.Arr(tgt, j);
            }
        }

        // 设置电流为0（这是一个激活函数，不产生实际电流）
        vars(i) = 0.0;
        return vars(i);
    }
};

REGISTER_MECHANISM("BP_Syn_SoftMax", BP_Syn_SoftMax_Mech);

} // namespace BP_Syn_SoftMax