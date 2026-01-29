// hh_templated mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>


namespace HH_Templated{

//用户请修改以下宏定义
#define MECH_CLASS_NAME HH_Templated
static const char *MECH_NAME_TO_REG = "hh";

struct MechTrait{
    enum class VarNames{//可以自由添加或修改变量
        m, h, n,
        ina, ik, il,
        gnabar, gkbar, gl, gna, gk,
        minf, hinf, ninf, mtau, htau, ntau,
        ena, ek, el
    };
    enum class GlobalVarNames{
        celsius
    };

    enum class IonVarNames{
        _ion_ena,_ion_ina,_ion_ek,_ion_ik
    };


};



class MECH_CLASS_NAME:public MechTemp<MECH_CLASS_NAME,MechTrait>{
    using enum MechTrait::VarNames;
    using enum MechTrait::GlobalVarNames;
    using enum MechTrait::IonVarNames;
public:
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE;
    MECH_CLASS_NAME(MechInitParams &param):MechTemp(param){
        // need_area = false;
        // var_in_coredata_idx.insert({your_var_name,0});
    
        init_values.insert({gnabar,0.25});
        init_values.insert({gl,0.00016666});
        init_values.insert({el,-60.0});
        init_values.insert({gkbar,0.036});
        init_values.insert({ena,50});
        init_values.insert({ek,-77});
        
        global_info_map.insert({celsius,{"celsius"}});
        var_in_coredata_idx.insert({gnabar,0});
        var_in_coredata_idx.insert({gkbar,1});
        var_in_coredata_idx.insert({gl,2});
        var_in_coredata_idx.insert({el,3});
        var_in_coredata_idx.insert({gna,4});
        var_in_coredata_idx.insert({gk,5});
        var_in_coredata_idx.insert({il,6});
        var_in_coredata_idx.insert({minf,7});
        var_in_coredata_idx.insert({hinf,8});
        var_in_coredata_idx.insert({ninf,9});
        var_in_coredata_idx.insert({mtau,10});
        var_in_coredata_idx.insert({htau,11});
        var_in_coredata_idx.insert({ntau,12});
        var_in_coredata_idx.insert({ena,19});
        var_in_coredata_idx.insert({ek,20});

        ion_var_map.insert({_ion_ek,{"k_ion",EionVarNames::erev}});
        ion_var_map.insert({_ion_ik,{"k_ion",EionVarNames::cur}});
        ion_var_map.insert({_ion_ina,{"na_ion",EionVarNames::cur}});
        ion_var_map.insert({_ion_ena,{"na_ion",EionVarNames::erev}});

        assert(param.name == MECH_NAME_TO_REG);
        printf_debug("MECH_CLASS_NAME(%s) init_vars\n",param.name.c_str());
    }

    DUAL_EXEC double vtrap(double x, double y){
        if (fabs(x / y) < 1e-6)
        {
            return y * (1 - x / y / 2);
        }
        else
        {
            return x / (exp(x / y) - 1);
        }
    }

    DUAL_EXEC void rates(double volt, double& minf, double& mtau, double& hinf, double& htau, double& ninf, double& ntau, double celsius)
    {
        double q10, alpha, beta, sum;
        q10 = pow(3.0, (celsius - 6.3) / 10);
        
        alpha = 0.1 * vtrap(-(volt + 40), 10);
        beta = 4 * exp(-(volt + 65) / 18);
        sum = alpha + beta;
        mtau = 1.0 / (q10 * sum);
        minf = alpha / sum; 

        alpha = 0.07 * exp(-(volt + 65) / 20);
        beta = 1.0 / (exp(-(volt + 35) / 10) + 1);
        sum = alpha + beta;
        htau = 1.0 / (q10 * sum);
        hinf = alpha / sum;

        alpha = 0.01 * vtrap(-(volt + 55), 10);
        beta = 0.125 * exp(-(volt + 65) / 80);
        sum = alpha + beta;
        ntau = 1.0 / (q10 * sum);
        ninf = alpha / sum;
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param, VarAccessor<MechTrait> &vars){
        vars(ena) = vars(_ion_ena);
        vars(ek) = vars(_ion_ek);

        rates(param.volt, vars(minf), vars(mtau), vars(hinf), vars(htau), vars(ninf), vars(ntau), vars(celsius));
            
        //printf("celsius: %f\n", vars(celsius));

        vars(m) = vars(minf);
        vars(h) = vars(hinf);
        vars(n) = vars(ninf);
    }


    DUAL_EXEC double hh_cal_current(double volt, 
                                    double _gnabar, double _gkbar, double _m, double _h,double _n, double _ena, double _ek, double _el, 
                                    double& gna, double& ina, double& gk,double& ik, double& gl, double& il)
    {
        gna = _gnabar * _m * _m * _m * _h;
        ina = gna * (volt - _ena);
        gk = _gkbar * _n * _n * _n * _n;
        ik = gk * (volt - _ek);
        il = gl * (volt - _el);
        return ina + ik + il;
    }
    
    DUAL_EXEC double current_single_node(MechTempCurParam &param, VarAccessor<MechTrait> &vars){
        vars(ena) = vars(_ion_ena);
        vars(ek) = vars(_ion_ek);
        double current = hh_cal_current(param.volt,
            vars(gnabar), vars(gkbar), vars(m), vars(h), vars(n),
            vars(ena), vars(ek), vars(el),
            vars(gna), vars(ina), vars(gk), vars(ik), vars(gl), vars(il));
        if(param.updateIon){
            mechAtomAdd(&vars(_ion_ina), vars(ina));
            mechAtomAdd(&vars(_ion_ik), vars(ik));
        }
        return current;
    }

    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars){
        rates(param.volt, vars(minf), vars(mtau), vars(hinf), vars(htau), vars(ninf), vars(ntau), vars(celsius));
        double _minf = vars(minf);
        double _mtau = vars(mtau);
        double _hinf = vars(hinf);
        double _htau = vars(htau);
        double _ninf = vars(ninf);
        double _ntau = vars(ntau);

        auto dt = param.dt;
        vars(m) = vars(m) + (1.0 - exp(dt * (-1.0 / _mtau))) * ((_minf / _mtau) / (1.0 / _mtau) - vars(m));
        vars(h) = vars(h) + (1.0 - exp(dt * (-1.0 / _htau))) * ((_hinf / _htau) / (1.0 / _htau) - vars(h));
        vars(n) = vars(n) + (1.0 - exp(dt * (-1.0 / _ntau))) * ((_ninf / _ntau) / (1.0 / _ntau) - vars(n));
    }
};

REGISTER_MECHANISM(MECH_NAME_TO_REG,MECH_CLASS_NAME);

//清理宏定义，防止对其他机制产生影响
#undef MECH_CLASS_NAME

}
