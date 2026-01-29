#include "postsyn_template.cuh"

namespace Exp2Syn {
struct MechTrait {
    enum class VarNames {
        tau1,
        tau2,
        e,
        factor,
        A,
        B,
        g,
        i
    };
};
class Exp2Syn_Templated : public PostSynTemplate<Exp2Syn_Templated, MechTrait> {
public:
using enum MechTrait::VarNames;
    constexpr static MechFlags flags = ENABLE_INIT | ENABLE_CURRENT | ENABLE_STATE | POINT_PROCESS;

    Exp2Syn_Templated(MechInitParams &params) : PostSynTemplate(params) {
        var_in_coredata_idx.insert({tau1, 0});
        var_in_coredata_idx.insert({tau2, 1});
        var_in_coredata_idx.insert({e, 2});
        var_in_coredata_idx.insert({A, 5});
        var_in_coredata_idx.insert({B, 6});
        var_in_coredata_idx.insert({factor, 7});
    }
    DUAL_EXEC void init_single_node(MechTempInitParam param, VarAccessor<MechTrait> vars) {
        auto _tau1 = vars(tau1);
        auto _tau2 = vars(tau2);
        if(_tau1 / _tau2 > 0.9999) {
            _tau1 = 0.9999 * _tau2;
            vars(tau1) = _tau1;
        }
        if(_tau1 / _tau2 < 1e-9) {
            _tau1 = _tau2 * 1e-9;
            vars(tau1) = _tau1;
        }
        vars(A) = 0.0;
        vars(B) = 0.0;
        double tp = (_tau1 * _tau2) / (_tau2 - _tau1) * log(_tau2 / _tau1);
        vars(factor) = 1.0 / (-exp(-tp / _tau1) + exp(-tp / _tau2));
    }
    DUAL_EXEC double current_single_node(MechTempCurParam &param,VarAccessor<MechTrait> vars){
        double _g = vars(B) - vars(A);
        vars(g) = _g;
        double _current = _g * (param.volt - vars(e));
        vars(i) = _current;
        return _current;
    }
    DUAL_EXEC void state_single_node(MechTempStateParam &param, VarAccessor<MechTrait> &vars){
        double _tau1 = vars(tau1);
        double _tau2 = vars(tau2);
        double _dt = param.dt;
        double _A = vars(A);
        double _B = vars(B);
        vars(A) = _A + (1.0 - exp(_dt * (-1.0 / _tau1))) * (-0.0 / (-1.0 / _tau1) - _A);
        vars(B) = _B + (1.0 - exp(_dt * (-1.0 / _tau2))) * (-0.0 / (-1.0 / _tau2) - _B);
    }

    DUAL_EXEC void net_receive_single_node(PostSynTempRecvParam &recv_param, VarAccessor<MechTrait> &vars) {
        double _factor = vars(factor);
        double _w = recv_param.weight;
        vars(A) = vars(A) + _w * _factor;
        vars(B) = vars(B) + _w * _factor;
    }
};

REGISTER_POSTSYN("Exp2Syn", Exp2Syn_Templated,1);
} // namespace Exp2Syn