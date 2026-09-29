// iclamp_templated mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"

namespace ICLAMP{

    
struct MechTrait{
    enum class VarNames{
        i,amp,delay,dur
    };  
};
class ICLAMP_Templated:public MechTemp<ICLAMP_Templated,MechTrait>{
public:
using enum MechTrait::VarNames;
    constexpr static MechFlags flags =
        ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT | ENABLE_CURRENT_VJP;
    // NOTE: scalar learnable vars only for now; array learnable vars are intentionally unsupported.
    static constexpr auto LearnableVars = std::array{amp};

    ICLAMP_Templated(MechInitParams &param):MechTemp(param){
        need_area = true;
        var_in_coredata_idx.insert({delay,0});
        var_in_coredata_idx.insert({dur,1});
        var_in_coredata_idx.insert({amp,2});

        printf_debug("ICLAMP_Templated init_vars\n");
    }

    DUAL_EXEC void init_single_node(MechTempInitParam &param,VarAccessor<MechTrait> vars) {
        vars(i) = 0.0;
    }
    DUAL_EXEC double current_single_node(MechTempCurParam &param,VarAccessor<MechTrait> vars) {
        double _current = 0.0;
        double _dur = vars(dur);
        double _del = vars(delay);
        double t = param.t;
        if (t < _del + _dur && t >= _del)
        {
            _current = vars(amp);
        }
        vars(i) = _current;
        return _current;
    };

    DUAL_EXEC void current_vjp_single_node(MechTempCurVJPParam& param, VarAccessor<MechTrait> vars) {
        vars.idx = param.idx;
        const double del = vars(delay);
        const double dur_value = vars(dur);
        if (!(param.t >= del && param.t < del + dur_value)) {
            return;
        }
        // IClamp current law: i = amp (inside active window), so local pullback is identity.
        mechAtomAdd(&grad_ref<amp>(param, vars), param.grad_mech_current);
    }
};

REGISTER_MECHANISM("IClamp",ICLAMP_Templated);

}//end of namespace ICLAMP
