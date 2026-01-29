// Ca mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace Ca_Templated
{

    struct MechTrait
    {
        enum class VarNames
        {
            gCabar,  // 最大电导 (S/cm2) - index 0
            ica,     // 钙离子电流 (mA/cm2) - index 1
            gCa,     // 电导 g (S/cm2) - index 2
            m,       // 状态变量 m - index 3
            h,       // 状态变量 h - index 4
            eca,     // 钙离子反转电位 (mV) - index 5
            mInf,    // m 的稳态值 - index 6
            mTau,    // m 的时间常数 (ms) - index 7
            mAlpha,  // m 速率常数 α - index 8
            mBeta,   // m 速率常数 β - index 9
            hInf,    // h 的稳态值 - index 10
            hTau,    // h 的时间常数 (ms) - index 11
            hAlpha,  // h 速率常数 α - index 12
            hBeta    // h 速率常数 β - index 13
        };
        enum class IonVarNames
        {
            _ion_eca,
            _ion_ica
        };
    };

#define MECH_CLASS_NAME Ca
    static const char *MECH_NAME_TO_REG = "Ca";

    class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait>
    {
        using enum MechTrait::VarNames;
        using enum MechTrait::IonVarNames;

    public:
        constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

        MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param)
        {
            // 设置默认参数值
            init_values.insert({gCabar, 0.00001});  // 默认值来自 MOD 文件

            // 按照 NEURON CPP 文件中的顺序注册变量（_nrn_mechanism_register_data_fields）
            // 索引必须与 CPP 文件完全一致
            var_in_coredata_idx.insert({gCabar, 0});
            var_in_coredata_idx.insert({ica, 1});
            var_in_coredata_idx.insert({gCa, 2});
            var_in_coredata_idx.insert({m, 3});
            var_in_coredata_idx.insert({h, 4});
            var_in_coredata_idx.insert({eca, 5});
            var_in_coredata_idx.insert({mInf, 6});
            var_in_coredata_idx.insert({mTau, 7});
            var_in_coredata_idx.insert({mAlpha, 8});
            var_in_coredata_idx.insert({mBeta, 9});
            var_in_coredata_idx.insert({hInf, 10});
            var_in_coredata_idx.insert({hTau, 11});
            var_in_coredata_idx.insert({hAlpha, 12});
            var_in_coredata_idx.insert({hBeta, 13});

            // 注册离子通道变量
            ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
            ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});

            assert(param.name == MECH_NAME_TO_REG);
            printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
        }

        // rates 函数：根据当前膜电位计算门控变量速率及稳态值
        // 对应 CPP 文件中的 rates 函数（lines 333-345）
        DUAL_EXEC void rates(double &volt,
                             double &mInf, double &mTau, double &mAlpha, double &mBeta,
                             double &hInf, double &hTau, double &hAlpha, double &hBeta)
        {
            // 特殊处理：避免除零
            if (volt == -27.0)
            {
                volt = volt + 0.0001;
            }

            mAlpha = (0.055 * (-27.0 - volt)) / (exp((-27.0 - volt) / 3.8) - 1.0);
            mBeta = (0.94 * exp((-75.0 - volt) / 17.0));
            mInf = mAlpha / (mAlpha + mBeta);
            mTau = 1.0 / (mAlpha + mBeta);

            hAlpha = (0.000457 * exp((-13.0 - volt) / 50.0));
            hBeta = (0.0065 / (exp((-volt - 15.0) / 28.0) + 1.0));
            hInf = hAlpha / (hAlpha + hBeta);
            hTau = 1.0 / (hAlpha + hBeta);
        }

        // 初始化函数：对应 CPP 文件中的 initmodel 函数（lines 436-446）
        // 将状态变量 m、h 初始化为稳态值
        DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars)
        {
            vars(eca) = vars(_ion_eca);

            double volt = param.volt;
            double _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta;

            rates(volt, _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta);

            vars(mInf) = _mInf;
            vars(mTau) = _mTau;
            vars(mAlpha) = _mAlpha;
            vars(mBeta) = _mBeta;
            vars(hInf) = _hInf;
            vars(hTau) = _hTau;
            vars(hAlpha) = _hAlpha;
            vars(hBeta) = _hBeta;

            // 根据 CPP 的 initmodel：m = mInf; h = hInf;
            vars(m) = _mInf;
            vars(h) = _hInf;
        }

        // 电流计算函数：对应 CPP 文件中的 _nrn_current 函数（lines 469-478）
        // gCa = gCabar*m*m*h; ica = gCa*(v-eca)
        DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars)
        {
            vars(eca) = vars(_ion_eca);

            vars(gCa) = vars(gCabar) * vars(m) * vars(m) * vars(h);
            vars(ica) = vars(gCa) * (param.volt - vars(eca));

            // 更新全局离子电流（仅在 updateIon=true 时）
            if (param.updateIon)
            {
                mechAtomAdd(&vars(_ion_ica), vars(ica));
            }

            return vars(ica);
        }

        // 状态更新函数：对应 CPP 文件中的 states 函数（lines 325-331）
        // ⚠️ 关键：这里的积分公式必须从 CPP 文件提取，不能自行推导！
        DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars)
        {
            vars(eca) = vars(_ion_eca);

            double volt = param.volt;
            double _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta;

            rates(volt, _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta);

            vars(mInf) = _mInf;
            vars(mTau) = _mTau;
            vars(mAlpha) = _mAlpha;
            vars(mBeta) = _mBeta;
            vars(hInf) = _hInf;
            vars(hTau) = _hTau;
            vars(hAlpha) = _hAlpha;
            vars(hBeta) = _hBeta;

            // 从 CPP 文件提取的确切积分公式（cnexp 方法）：
            // CPP line 327: m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mTau)))*(- ( ( ( mInf ) ) / mTau ) / ( ( ( ( - 1.0 ) ) ) / mTau ) - m)
            // 简化后：m = m + (1. - exp(-dt/mTau)) * (mInf - m)
            double dt = param.dt;
            vars(m) = vars(m) + (1.0 - exp(-dt / _mTau)) * (_mInf - vars(m));
            vars(h) = vars(h) + (1.0 - exp(-dt / _hTau)) * (_hInf - vars(h));
        }
    };

    REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制的影响
#undef MECH_CLASS_NAME

} // namespace Ca_Templated
