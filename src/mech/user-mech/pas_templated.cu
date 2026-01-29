// pas_templated mechanism - auto-registered via whole-archive linking
#include "mech_template.cuh"

namespace PAS{

struct MechTrait{
    enum class VarNames{
        i,e_pas,g_pas
    };
};

class PAS_Templated:public MechTemp<PAS_Templated,MechTrait>{

    using enum MechTrait::VarNames;
public:
    constexpr static MechFlags flags = MechFlags::ENABLE_CURRENT;
    PAS_Templated(MechInitParams &param):MechTemp(param){
        var_in_coredata_idx.insert({g_pas,0});
        var_in_coredata_idx.insert({e_pas,1});

        init_values.insert({e_pas,-60});
        init_values.insert({g_pas,0.00016666});

        printf_debug("PAS_Templated init_vars\n");
    }

    DUAL_EXEC double current_single_node(MechTempCurParam &param,VarAccessor<MechTrait> vars) {
        double current = vars(g_pas) * (param.volt - vars(e_pas));
        vars(i) = current;
        return current;
    };
};

REGISTER_MECHANISM("pas",PAS_Templated);
}//end of namespace PAS