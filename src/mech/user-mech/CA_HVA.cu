// CA_HVA mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>
#include <cmath>

namespace Ca_HVA_Templated
{

    struct MechTrait
    {
        enum class VarNames
        {
            i_mech, // 机制输出的总电流
            ica,    // 计算得到的 ica 电流（mA/cm2）
            g,      // 电导 g (S/cm2)
            m,      // 状态变量 m
            h,      // 状态变量 h
            eca,    // 钙离子反转电位 (mV)
            mInf,   // m 的稳态值
            mTau,   // m 的时间常数 (ms)
            mAlpha, // m 速率常数 α
            mBeta,  // m 速率常数 β
            hInf,   // h 的稳态值
            hTau,   // h 的时间常数 (ms)
            hAlpha, // h 速率常数 α
            hBeta,  // h 速率常数 β
            gbar    // 最大电导 (S/cm2)
        };
        enum class IonVarNames
        {
            _ion_ica,
            _ion_eca
        };
    };
// 用户请修改以下宏定义
#define MECH_CLASS_NAME Ca_HVA
    static const char *MECH_NAME_TO_REG = "Ca_HVA";

    class MECH_CLASS_NAME : public MechTemp<MECH_CLASS_NAME, MechTrait>
    {
        using enum MechTrait::VarNames;
        using enum MechTrait::IonVarNames;

    public:
        constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;

        MECH_CLASS_NAME(MechInitParams &param) : MechTemp(param)
        {
            // 设置默认参数值
            init_values.insert({gbar, 0.1});
            // 因为离子浓度机制尚未支持，这里用固定值；注意 NEURON 中 eca 来自离子变量
            init_values.insert({eca, 140.0});

            // 这里按照 NEURON 的顺序注册变量：
            // 0: gbar, 1: ica, 2: g, 3: m, 4: h, 5: eca,
            // 6: mInf, 7: mTau, 8: mAlpha, 9: mBeta, 10: hInf, 11: hTau, 12: hAlpha, 13: hBeta, 14: i_mech
            var_in_coredata_idx.insert({gbar, 0});
            var_in_coredata_idx.insert({ica, 1});
            var_in_coredata_idx.insert({g, 2});
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
            var_in_coredata_idx.insert({i_mech, 14});

            // 注册外部离子机制变量
            ion_var_map.insert({_ion_eca, {"ca_ion", EionVarNames::erev}});
            ion_var_map.insert({_ion_ica, {"ca_ion", EionVarNames::cur}});

            assert(param.name == MECH_NAME_TO_REG);
            printf_debug("MECH_CLASS_NAME(%s) init_vars\n", param.name.c_str());
        }

        DUAL_EXEC double vtrap(double _lx, double _ly)
        {
            double _lvtrap;
            if (fabs(_lx / _ly) < 1e-6)
            {
                _lvtrap = _ly * (1.0 - _lx / _ly / 2.0);
            }
            else
            {
                _lvtrap = _lx / (exp(_lx / _ly) - 1.0);
            }

            return _lvtrap;
        }

        // 根据当前膜电位 _v 计算门控变量速率及稳态值
        DUAL_EXEC void rates(double volt,
                             double &mInf, double &mTau, double &mAlpha, double &mBeta,
                             double &hInf, double &hTau, double &hAlpha, double &hBeta)
        {
            mAlpha = 0.055 * vtrap(-27.0 - volt, 3.8);
            mBeta = 0.94 * exp((-75.0 - volt) / 17.0);
            mInf = mAlpha / (mAlpha + mBeta);
            mTau = 1.0 / (mAlpha + mBeta);
            hAlpha = 0.000457 * exp((-13.0 - volt) / 50.0);
            hBeta = 0.0065 / (exp((-volt - 15.0) / 28.0) + 1.0);
            hInf = hAlpha / (hAlpha + hBeta);
            hTau = 1.0 / (hAlpha + hBeta);
        }

        // 每个节点的初始化：调用 rates() 后将状态 m、h 初始化为稳态值
        DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars)
        {
            vars(eca) = vars(_ion_eca);

            double _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta;

            rates(param.volt, _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta);

            vars(mInf) = _mInf;
            vars(mTau) = _mTau;
            vars(mAlpha) = _mAlpha;
            vars(mBeta) = _mBeta;
            vars(hInf) = _hInf;
            vars(hTau) = _hTau;
            vars(hAlpha) = _hAlpha;
            vars(hBeta) = _hBeta;

            vars(m) = _mInf;
            vars(h) = _hInf;
        }

        // 计算电流：根据 g = gbar*m*m*h 及 ica = g*(v-eca)
        DUAL_EXEC double ca_hva_cal_current(double volt,
                                            double _gbar, double _m, double _h, double _eca,
                                            double &g, double &ica, double &i_mech)
        {
            g = _gbar * _m * _m * _h;
            ica = g * (volt - _eca);
            i_mech = ica;
            return i_mech;
        }

        DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars)
        {
            vars(eca) = vars(_ion_eca);
            double current = ca_hva_cal_current(param.volt,
                                                vars(gbar), vars(m), vars(h), vars(eca),
                                                vars(g), vars(ica), vars(i_mech));
            if (param.updateIon)
            {
                mechAtomAdd(&vars(_ion_ica), vars(ica));
            }
            return current;
        }

        // 状态更新：使用 cnexp（指数欧拉）方法更新 m 与 h
        DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars)
        {
            vars(eca) = vars(_ion_eca);

            double _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta;

            rates(param.volt, _mInf, _mTau, _mAlpha, _mBeta, _hInf, _hTau, _hAlpha, _hBeta);

            vars(mInf) = _mInf;
            vars(mTau) = _mTau;
            vars(mAlpha) = _mAlpha;
            vars(mBeta) = _mBeta;
            vars(hInf) = _hInf;
            vars(hTau) = _hTau;
            vars(hAlpha) = _hAlpha;
            vars(hBeta) = _hBeta;

            auto dt = param.dt;
            vars(m) = vars(m) + (1.0 - exp(-dt / _mTau)) * (_mInf - vars(m));
            vars(h) = vars(h) + (1.0 - exp(-dt / _hTau)) * (_hInf - vars(h));
        }
    };

    REGISTER_MECHANISM(MECH_NAME_TO_REG, MECH_CLASS_NAME);

// 清理宏定义，防止对其他机制的影响
#undef MECH_CLASS_NAME

} // namespace Ca_HVA_Templated