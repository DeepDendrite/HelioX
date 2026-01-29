// CA_Dynamic mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>
#include "debug_counter.cuh"

//注意：这个还不完善，还没实现离子通道机制，别用！
namespace CaDynamics_Templated {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME CaDynamics
static const char *MECH_NAME_TO_REG = "CaDynamics";

// 注意：NEURON mod 文件中 RANGE 变量顺序为：
//   0: gamma, 1: decay, 2: depth, 3: minCai, 4: ica, 5: cai
// 其中 ica 为读入的离子电流，cai 为状态变量。


struct MechTrait {
    enum class VarNames {
        gamma,   // 百分比（未被缓冲的）自由钙
        decay,   // 钙清除速率，单位 ms
        depth,   // 钙外壳厚度，单位 um
        minCai,  // 最低钙浓度，单位 mM
        ica,     // 读入的钙电流，单位 mA/cm2
        cai,     // 状态变量：细胞内钙浓度，单位 mM
        _g
    };
    enum class IonVarNames {
        _ion_ica, // 读入的钙电流
        _ion_cai  // 状态变量：细胞内钙浓度
    };
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // 此机制不产生电流，因此关闭 current 部分
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | WRITE_EION_IN_STATE;

    using enum MechTrait::VarNames;
    using enum MechTrait::IonVarNames;
    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // 设置默认参数
        init_values.insert({gamma, 0.05});
        init_values.insert({decay, 80.0});
        init_values.insert({depth, 0.1});
        init_values.insert({minCai, 0.0001});
        // ica 为离子变量，从离子机制获得，不在此设置

        // 按照 NEURON 生成代码中的顺序注册变量
        // 顺序： gamma (0), decay (1), depth (2), minCai (3), ica (4), cai (5)
        var_in_coredata_idx.insert({gamma, 0});
        var_in_coredata_idx.insert({decay, 1});
        var_in_coredata_idx.insert({depth, 2});
        var_in_coredata_idx.insert({minCai, 3});
        var_in_coredata_idx.insert({ica, 4});
        var_in_coredata_idx.insert({cai, 5});
        var_in_coredata_idx.insert({_g, 8});

        // 注册外部离子机制变量
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }


    // 初始化时将状态变量 cai 置为 minCai
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        vars(cai) = vars(minCai);
        vars(_ion_cai) = vars(cai);
    }

    // 状态更新：利用 cnexp 方法更新 cai
    // 公式： cai' = -10000*(ica*gamma/(2*FARADAY*depth)) - (cai - minCai)/decay
    // 可改写为： cai' = -(cai - cai_inf)/decay，其中
    //       cai_inf = minCai - 10000*decay*(ica*gamma/(2*FARADAY*depth))
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        vars(ica) = vars(_ion_ica);
        vars(cai) = vars(_ion_cai);
        double gamma_val  = vars(gamma);
        double decay_val  = vars(decay);
        double depth_val  = vars(depth);
        double minCai_val = vars(minCai);
        double ica_val    = vars(ica);
        double cai_val    = vars(cai);

        constexpr double FARADAY = 0x1.78e555060882cp+16;
        vars(cai) = cai_val + (1. - exp(param.dt * ((-((1.0)) / decay_val)))) * (-((-(10000.0)) * ((((ica_val) * (gamma_val)) / (2.0 * FARADAY * depth_val))) - (((-minCai_val))) / decay_val) / ((-((1.0)) / decay_val)) - cai_val);

        vars(_ion_cai) = vars(cai);
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义
#undef MECH_CLASS_NAME

} // namespace CaDynamics_Templated