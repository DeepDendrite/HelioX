// cainternm mechanism (worm) - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace cainternm_worm {

struct MechTrait {
    enum class VarNames {
        // Parameters
        vcell,
        
        // Assigned variables
        ica, alpha,
        
        // State variable
        cai
    };
    
    enum class GlobalVarNames {
        f, tca, caeq
    };
    
    enum class IonVarNames {
        _ion_ica, _ion_cai, _ion_cao, _ion_ca_erev
    };
};

DEFINE_HAS_ENUM(GlobalVarNames);
DEFINE_ENUM_MAP_ALIAS(GlobalVarNames, CoreGlobalVarInfo, GlobalVarInfoMap);

class CaInternM : public MechTemp<CaInternM, MechTrait> {
public:
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
    
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_STATE | WRITE_EION_IN_STATE;
    
    CaInternM(MechInitParams &param) : MechTemp(param) {
        init_values.insert({vcell, 31.16});
        
        var_in_coredata_idx.insert({vcell, 0});
        var_in_coredata_idx.insert({ica, 1});
        var_in_coredata_idx.insert({alpha, 2});
        var_in_coredata_idx.insert({cai, 3});
        
        // 注册全局变量
        global_info_map.insert({f, {"f_cainternm"}});
        global_info_map.insert({tca, {"tca_cainternm"}});
        global_info_map.insert({caeq, {"caeq_cainternm"}});
        
        ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});
        ion_var_map.insert({_ion_cai, {"ca_ion", EionVarNames::conci}});
        ion_var_map.insert({_ion_cao, {"ca_ion", EionVarNames::conco}});
        ion_var_map.insert({_ion_ca_erev, {"ca_ion", EionVarNames::erev}});
    }
    
    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars) {
        setparames(param.volt, vars);
        // cainternm没有明确的初始化，默认初始化为0或从离子浓度读取
        vars(cai) = vars(_ion_cai);
    }
    
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars) {
        // 1. 从全局离子变量读取最新值 (完全按照CoreNEURON)
        vars(ica) = vars(_ion_ica);
        vars(cai) = vars(_ion_cai);
        vars(cai) = vars(_ion_cai);  // CoreNEURON重复读取了两次
        
        // 2. 完全复刻CoreNEURON的牛顿法求解
        double old_cai = vars(cai);  // 保存初值
        double current_cai = vars(cai);  // 当前试探值
        
        // 牛顿法迭代 (完全按照NEURON/CoreNEURON的参数)
        const int MAX_ITER = 50;
        const double CONVERGE = 1e-6;  // NEURON的相对变化阈值
        const double ZERO = 1e-8;      // NEURON的绝对残差阈值
        // const double MAXCHANGE = 0.05; // 重新计算雅可比的阈值
        
        double change = 1.0;
        
        for (int iter = 0; iter < MAX_ITER; iter++) {
            // 用当前cai值计算alpha
            vars(cai) = current_cai;  // 设置当前试探值
            setparames(param.volt, vars);
            double alpha_val = vars(alpha);  // 避免名称冲突，每次循环重新计算
            
            // 计算残差函数: F(cai) = (-cai + alpha * dt + old_cai) / dt
            double F = (-current_cai + alpha_val * param.dt + old_cai) / param.dt;
            
            // 计算雅可比: dF/dcai = -1/dt
            double tca_val = vars(tca);
            double dalpha_dcai = -1.0 / tca_val;
            double J = (-1.0 + dalpha_dcai * param.dt) / param.dt;
            
            // 牛顿法更新: cai = cai - F/J
            double delta = F / J;
            current_cai -= delta;
            
            // NEURON风格的收敛判断
            // change = 相对变化, max_dev = 绝对残差
            if (fabs(old_cai) > ZERO) {
                change = fabs(delta / old_cai);
            } else {
                change = fabs(delta);
            }
            double max_dev = fabs(F);
            
            // NEURON的收敛条件: change <= CONVERGE && max_dev <= ZERO
            if (change <= CONVERGE && max_dev <= ZERO) {
                break;
            }
        }
        
        // 3. 设置最终结果
        vars(cai) = current_cai;
        
        // 4. 写回全局离子浓度 (完全按照CoreNEURON)
        vars(_ion_cai) = vars(cai);
    }
    
private:
    DUAL_EXEC void setparames(double v, VarAccessor<MechTrait> &vars) {
        const double FARADAY = 96485.0;
        
        double f_val = vars(f);
        double tca_val = vars(tca);
        double caeq_val = vars(caeq);
        double vcell_val = vars(vcell);
        double ica_val = vars(ica);
        double cai_val = vars(cai);
        
        if (v <= 60.0) { // eca=60 mV
            vars(alpha) = (-f_val * ica_val * pow(10.0, 6.0) / (2.0 * FARADAY * vcell_val)) - ((cai_val - caeq_val) / tca_val);
        }
        else {
            vars(alpha) = -(cai_val - caeq_val) / tca_val;
        }
    }
};

REGISTER_MECHANISM("cainternm", CaInternM);

} // namespace cainternm_worm