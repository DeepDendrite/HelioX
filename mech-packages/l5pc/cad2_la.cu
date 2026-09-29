// cad2_la mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace CAD2_LA {

// 用户请修改以下宏定义
#define MECH_CLASS_NAME CAD2
static const char *MECH_NAME_TO_REG = "cad2_la";

struct MechTrait {
    enum class VarNames {
        // 状态变量
        ca,
        
        // 中间变量
        cai, ica, drive_channel, Dca,
        
        // 内部使用的_g
        _g
    };

    enum class GlobalVarNames {
        // 全局参数
        depth, taur, cainf
    };

    enum class IonVarNames {
        // 离子通道相关变量
        _ion_ica, _ion_cai, _ion_cao, _ion_ca_erev
    };
};

class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait> {
public:
    // 需要实现INIT和STATE函数，且在STATE函数中写入了离子浓度
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | WRITE_EION_IN_STATE;

    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;

    MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param) {
        // 在coredata中的变量索引
        var_in_coredata_idx.insert({ca, 0});
        var_in_coredata_idx.insert({cai, 1});
        var_in_coredata_idx.insert({Dca, 2});
        var_in_coredata_idx.insert({ica, 3});
        var_in_coredata_idx.insert({drive_channel, 4});
        var_in_coredata_idx.insert({_g, 6});

        // 注册全局变量 - 名称需与NEURON中一致
        global_info_map.insert({depth, {"depth_cad2_la"}});
        global_info_map.insert({taur, {"taur_cad2_la"}});
        global_info_map.insert({cainf, {"cainf_cad2_la"}});

        // 注册离子通道变量
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
        ion_var_map.insert({_ion_cao, {"ca_ion", EionVarNames::conco}});
        ion_var_map.insert({_ion_ca_erev, {"ca_ion", EionVarNames::erev}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
    }

    // 初始化函数
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        // 初始化ca为cainf
        vars(ca) = vars(cainf);
    }

    // 该机制没有current函数，因为它不产生电流
    // 只需要实现state函数

    // 更新状态变量
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 从离子通道获取当前钙电流
        vars(ica) = vars(_ion_ica);
        
        // 计算drive_channel
        constexpr double FARADAY = 96485.3321233100184; // 法拉第常数
        vars(drive_channel) = -(10000.0) * vars(ica) / (2.0 * FARADAY * vars(depth));
        
        // 不能向内泵入
        if (vars(drive_channel) <= 0.0) {
            vars(drive_channel) = 0.0;
        }
        
        // 使用隐式导数方法更新ca
        // ca' = drive_channel + (cainf-ca)/taur
        double ca_inf = vars(cainf);
        double tau_r = vars(taur);
        double dca_dt = vars(drive_channel) + (ca_inf - vars(ca)) / tau_r;
        
        // 隐式欧拉法更新ca
        // ca(t+dt) = ca(t) + dt*dca_dt
        // 当使用隐式方法时，需要考虑到dca_dt中包含ca(t+dt)
        // 可以改写为：ca(t+dt) = (ca(t) + dt*(drive_channel + ca_inf/tau_r)) / (1 + dt/tau_r)
        vars(ca) = (vars(ca) + param.dt * (vars(drive_channel) + ca_inf / tau_r)) / (1.0 + param.dt / tau_r);
        
        // 更新内部钙浓度并同步到全局钙离子浓度
        vars(cai) = vars(ca);
        vars(_ion_cai) = vars(ca);
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

} // namespace CAD2_LA
