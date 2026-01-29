// BP_Syn mechanism - auto-registered via whole-archive linking

#include "mech_template.cuh"
#include <cmath>
#include <cstdio>

namespace BP_Syn {

//-------------------------------------
// 1) 先定义 Trait:
//    - VarNames 里把需要的所有变量都列出
//-------------------------------------
struct MechTrait {
    enum class VarNames {
        // 与 NEURON .cpp 中的下标顺序保持一致（见宏 #define lr_start _ml->fpfield<0>(_iml) 等）
        lr_start,      // 0
        lr_end,        // 1
        tgt_0,         // 2
        tgt_1,         // 3
        tgt_2,         // 4
        tgt_3,         // 5
        tgt_4,         // 6
        tgt_5,         // 7
        tgt_6,         // 8
        tgt_7,         // 9
        tgt_8,         // 10
        tgt_9,         // 11
        u_tgt,         // 12
        g_0,           // 13
        g_1,           // 14
        g_2,           // 15
        g_3,           // 16
        g_4,           // 17
        g_5,           // 18
        g_6,           // 19
        g_7,           // 20
        g_8,           // 21
        g_9,           // 22
        w,             // 23
        learning_rate, // 24
        g,             // 25
        layer_flag,    // 26

        // 以下是分配在 mech 中、可读可写的变量
        i,             // 27
        u_0,           // 28
        u_1,           // 29
        u_2,           // 30
        u_3,           // 31
        u_4,           // 32
        u_5,           // 33
        u_6,           // 34
        u_7,           // 35
        u_8,           // 36
        u_9,           // 37
        s_sum,         // 38
        s_u_soma,      // 39
        s_0,           // 40
        s_1,           // 41
        s_2,           // 42
        s_3,           // 43
        s_4,           // 44
        s_5,           // 45
        s_6,           // 46
        s_7,           // 47
        s_8,           // 48
        s_9,           // 49
        u_soma,        // 50
        v_gap,         // 51
        PI,            // 52
        delta_w,       // 53
        has_stdp,      // 54
        fa_error,      // 55
        v,             // 56
        _g             // 57 (HelioX 中并不一定必须用到，但这里为了对齐 NEURON 的下标，也保留)
    };

    // 本例无离子变量 IonVarNames，省略
    // 本例无全局变量 GlobalVarNames，省略
};

//-------------------------------------
// 2) 定义 BP_Syn 类，继承自模板 MechTemp<BP_Syn, MechTrait>
//-------------------------------------
class BP_Syn : public MechTemp<BP_Syn, MechTrait> {
public:
    // 这是一个点过程且有电极电流，并且需要 INIT、CURRENT 两个函数
    constexpr static MechFlags flags =
        MechFlags::ENABLE_INIT |
        MechFlags::ENABLE_CURRENT |
        MechFlags::POINT_PROCESS |
        MechFlags::ELECTRODE_CURRENT;

    // 把枚举名直接using方便在下面用
    using enum MechTrait::VarNames;

    // 构造函数
    BP_Syn(MechInitParams &param) : MechTemp(param) {
        // 对点过程 + 电极电流，往往需要 area
        need_area = true;

        //== 2.1) 给一些变量默认值 (对应 NEURON 里的 PARAMETER ... = xxx) ==
        // 例如 learning_rate=0.01, g=0.01, g_0~g_9=1, w=0等
        init_values.insert({learning_rate, 0.01});
        init_values.insert({g, 0.01});
        // g_0 ~ g_9 默认都是1
        init_values.insert({g_0, 1.0});
        init_values.insert({g_1, 1.0});
        init_values.insert({g_2, 1.0});
        init_values.insert({g_3, 1.0});
        init_values.insert({g_4, 1.0});
        init_values.insert({g_5, 1.0});
        init_values.insert({g_6, 1.0});
        init_values.insert({g_7, 1.0});
        init_values.insert({g_8, 1.0});
        init_values.insert({g_9, 1.0});
        // w 未设置默认值时，可令其为 0
        init_values.insert({w, 0.0});

        //== 2.2) 注册所有 var_in_coredata_idx (与 NEURON 中的下标对应) ==
        var_in_coredata_idx.insert({lr_start,       0});
        var_in_coredata_idx.insert({lr_end,         1});
        var_in_coredata_idx.insert({tgt_0,          2});
        var_in_coredata_idx.insert({tgt_1,          3});
        var_in_coredata_idx.insert({tgt_2,          4});
        var_in_coredata_idx.insert({tgt_3,          5});
        var_in_coredata_idx.insert({tgt_4,          6});
        var_in_coredata_idx.insert({tgt_5,          7});
        var_in_coredata_idx.insert({tgt_6,          8});
        var_in_coredata_idx.insert({tgt_7,          9});
        var_in_coredata_idx.insert({tgt_8,          10});
        var_in_coredata_idx.insert({tgt_9,          11});
        var_in_coredata_idx.insert({u_tgt,          12});
        var_in_coredata_idx.insert({g_0,            13});
        var_in_coredata_idx.insert({g_1,            14});
        var_in_coredata_idx.insert({g_2,            15});
        var_in_coredata_idx.insert({g_3,            16});
        var_in_coredata_idx.insert({g_4,            17});
        var_in_coredata_idx.insert({g_5,            18});
        var_in_coredata_idx.insert({g_6,            19});
        var_in_coredata_idx.insert({g_7,            20});
        var_in_coredata_idx.insert({g_8,            21});
        var_in_coredata_idx.insert({g_9,            22});
        var_in_coredata_idx.insert({w,              23});
        var_in_coredata_idx.insert({learning_rate,  24});
        var_in_coredata_idx.insert({g,              25});
        var_in_coredata_idx.insert({layer_flag,     26});
        var_in_coredata_idx.insert({has_stdp,       27});
        var_in_coredata_idx.insert({i,              28});
        var_in_coredata_idx.insert({u_0,            29});
        var_in_coredata_idx.insert({u_1,            30});
        var_in_coredata_idx.insert({u_2,            31});
        var_in_coredata_idx.insert({u_3,            32});
        var_in_coredata_idx.insert({u_4,            33});
        var_in_coredata_idx.insert({u_5,            34});
        var_in_coredata_idx.insert({u_6,            35});
        var_in_coredata_idx.insert({u_7,            36});
        var_in_coredata_idx.insert({u_8,            37});
        var_in_coredata_idx.insert({u_9,            38});
        var_in_coredata_idx.insert({s_sum,          39});
        var_in_coredata_idx.insert({s_u_soma,       40});
        var_in_coredata_idx.insert({s_0,            41});
        var_in_coredata_idx.insert({s_1,            42});
        var_in_coredata_idx.insert({s_2,            43});
        var_in_coredata_idx.insert({s_3,            44});
        var_in_coredata_idx.insert({s_4,            45});
        var_in_coredata_idx.insert({s_5,            46});
        var_in_coredata_idx.insert({s_6,            47});
        var_in_coredata_idx.insert({s_7,            48});
        var_in_coredata_idx.insert({s_8,            49});
        var_in_coredata_idx.insert({s_9,            50});
        var_in_coredata_idx.insert({fa_error,       51});
        var_in_coredata_idx.insert({u_soma,         52});
        var_in_coredata_idx.insert({v_gap,          53});
        var_in_coredata_idx.insert({PI,             54});
        var_in_coredata_idx.insert({delta_w,        55});
        var_in_coredata_idx.insert({v,             56});
        var_in_coredata_idx.insert({_g,             57});

        // 打印调试信息（可选）
        printf_debug("BP_Syn Mechanism constructed with name=%s\n", param.name.c_str());
    }

    //-------------------------------------
    // 3) INIT 函数(对应 INITIAL 块)
    //    在 NEURON 里:
    //    INITIAL {
    //      has_stdp = 0
    //      delta_w = 0
    //    }
    //-------------------------------------
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 与 NEURON INITIAL 块对应
        vars(has_stdp) = 0.0;
        vars(delta_w)  = 0.0;
    }

    //-------------------------------------
    // 4) CURRENT 函数(对应 BREAKPOINT 块)
    //    在 NEURON 里此机制无 STATE,
    //    所以所有更新都在 BREAKPOINT
    //-------------------------------------
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars) {
        // 由于 NEURON 那边要把节点电压 v 存进来，这里可自愿同步
        // 在 .mod 中 `v` 并没有参与计算(除了被记录到 assigned)，但保持一致性:
        vars(v) = param.volt;

        // replicate NEURON 里的 t < lr_start / t > lr_end => has_stdp=0, etc
        double tval = param.t;
        if (tval < vars(lr_start)) {
            vars(has_stdp) = 0.0;
        } else if (tval > vars(lr_end)) {
            vars(has_stdp) = 0.0;
        } else {
            vars(has_stdp) = 1.0;
        }

        // 先设一个局部
        double current = 0.0;

        // if layer_flag == 2 => softmax 计算 + i=0
        if (abs((vars(layer_flag)) - 2.0) < 1e-2) {
            double e0 = exp(vars(u_0));
            double e1 = exp(vars(u_1));
            double e2 = exp(vars(u_2));
            double e3 = exp(vars(u_3));
            double e4 = exp(vars(u_4));
            double e5 = exp(vars(u_5));
            double e6 = exp(vars(u_6));
            double e7 = exp(vars(u_7));
            double e8 = exp(vars(u_8));
            double e9 = exp(vars(u_9));

            double sum_e = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8 + e9;
            vars(s_sum) = sum_e;

            vars(s_0) = e0 / sum_e;
            vars(s_1) = e1 / sum_e;
            vars(s_2) = e2 / sum_e;
            vars(s_3) = e3 / sum_e;
            vars(s_4) = e4 / sum_e;
            vars(s_5) = e5 / sum_e;
            vars(s_6) = e6 / sum_e;
            vars(s_7) = e7 / sum_e;
            vars(s_8) = e8 / sum_e;
            vars(s_9) = e9 / sum_e;

            // 电流置0
            vars(i) = 0.0;
            current = 0.0;
        } else {
            // 其余情况
            // 如果 has_stdp=1 => 更新 delta_w
            if (abs(vars(has_stdp) - 1.0) < 1e-2) {
                // fa_error = 0
                vars(fa_error) = 0.0;

                // layer_flag == 1 => fa_error = u_tgt - s_u_soma
                if (abs(vars(layer_flag) - 1) < 1e-2) {
                    vars(fa_error) = (vars(u_tgt) - vars(s_u_soma));
                } else {
                    // layer_flag == 0 时, 需要看 u_soma>0 后再累加
                    if (vars(u_soma) > 0.0) {
                        vars(fa_error) += (vars(tgt_0) - vars(s_0)) * vars(g_0);
                        vars(fa_error) += (vars(tgt_1) - vars(s_1)) * vars(g_1);
                        vars(fa_error) += (vars(tgt_2) - vars(s_2)) * vars(g_2);
                        vars(fa_error) += (vars(tgt_3) - vars(s_3)) * vars(g_3);
                        vars(fa_error) += (vars(tgt_4) - vars(s_4)) * vars(g_4);
                        vars(fa_error) += (vars(tgt_5) - vars(s_5)) * vars(g_5);
                        vars(fa_error) += (vars(tgt_6) - vars(s_6)) * vars(g_6);
                        vars(fa_error) += (vars(tgt_7) - vars(s_7)) * vars(g_7);
                        vars(fa_error) += (vars(tgt_8) - vars(s_8)) * vars(g_8);
                        vars(fa_error) += (vars(tgt_9) - vars(s_9)) * vars(g_9);
                    }
                }

                // PI = fa_error * v_gap (如果 v_gap>0)
                if (vars(v_gap) > 0.0) {
                    vars(PI) = vars(fa_error) * vars(v_gap);
                } else {
                    vars(PI) = 0.0;
                }

                // delta_w = delta_w + learning_rate * PI
                // 只在 param.updateIon == true 时更新一次，避免被调用2次时翻倍
                // if (param.updateIon) {
                vars(delta_w) = vars(delta_w) + vars(learning_rate) * vars(PI);
                // }
            }

            // 计算电流 i = g * w * v_gap (若 v_gap>0)
            if (vars(v_gap) > 0.0) {
                vars(i) = vars(g) * vars(w) * vars(v_gap);
            } else {
                vars(i) = 0.0;
            }

            current = vars(i);
        }

        // 返回电流值，用于后续加到膜方程
        return current;
    }
};

// 最后注册本机理
REGISTER_MECHANISM("BP_Syn", BP_Syn);

} // end of namespace BP_Syn
