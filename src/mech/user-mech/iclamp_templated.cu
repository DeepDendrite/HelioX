// iclamp_templated mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"
#include <cstdio>

namespace ICLAMP{
    
struct MechTrait{
    enum class VarNames{
        i,amp,delay,dur
    };  
};
class ICLAMP_Templated:public MechTemp<ICLAMP_Templated,MechTrait>{
public:
using enum MechTrait::VarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | POINT_PROCESS | ELECTRODE_CURRENT;

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
};

REGISTER_MECHANISM("IClamp",ICLAMP_Templated);

}//end of namespace ICLAMP