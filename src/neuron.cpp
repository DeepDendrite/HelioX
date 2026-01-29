#include "neuron.h"

HelioXroupData::HelioXroupData(Mode mode, double dt):vec_play_continuous(mode), 
    cpu_gap_trans_info(CPU), gpu_gap_trans_info(mode)
{
    len = 0;
    ncell = 0;
    cj = 1/dt;
    vecdata_a = nullptr;
    vecdata_b = nullptr;
    vecdata_d = nullptr;
    vecdata_rhs = nullptr;
    vecdata_v = nullptr;
    vecdata_sav_rhs = nullptr;
    vecdata_sav_d = nullptr;
    vecdata_i_membrane_ = nullptr;
    vecdata_area = nullptr;
    vecdata_parent_index = nullptr;
    mech_cap = nullptr;
    spk_vec = nullptr;
    vecdata_stride = nullptr;
    vecdata_firstnode = nullptr;
    vecdata_lastnode = nullptr;
    vecdata_cellsize = nullptr;
    vecdata_map_t2c = nullptr;
    vecdata_max_order_each_thread = nullptr;
    vecdata_min_order_each_thread = nullptr;
    
    // 初始化gap transfer info
    have_gap = false;
}

HelioXroupData::~HelioXroupData()
{
    if (vecdata_a)
        delete vecdata_a;
    if (vecdata_b)
        delete vecdata_b;
    if (vecdata_d)
        delete vecdata_d;
    if (vecdata_rhs)
        delete vecdata_rhs;
    if (vecdata_v)
        delete vecdata_v;
    if (vecdata_sav_rhs)
        delete vecdata_sav_rhs;
    if (vecdata_sav_d)
        delete vecdata_sav_d;
    if (vecdata_i_membrane_)
        delete vecdata_i_membrane_;
    if (vecdata_area)
        delete vecdata_area;
    if (vecdata_parent_index)
        delete vecdata_parent_index;
    if (spk_vec)
    {
        delete spk_vec;
        spk_vec = nullptr;
    }

    if (vecdata_stride)
        delete vecdata_stride;
    if (vecdata_firstnode)
        delete vecdata_firstnode;
    if (vecdata_lastnode)
        delete vecdata_lastnode;
    if (vecdata_cellsize)
        delete vecdata_cellsize;
    if (vecdata_map_t2c)
        delete vecdata_map_t2c;
    if (vecdata_max_order_each_thread)
        delete vecdata_max_order_each_thread;
    if (vecdata_min_order_each_thread)
        delete vecdata_min_order_each_thread;
    try_delete(vecdata_spk_flags);
    try_delete_arr(type2postsyn_ix);
    try_delete_arr(type2articell_ix);
    try_delete(presyn);

	vecdata_a = nullptr;
	vecdata_b = nullptr;
	vecdata_d = nullptr;
	vecdata_rhs = nullptr;
	vecdata_v = nullptr;
    vecdata_sav_rhs = nullptr;
    vecdata_sav_d = nullptr;
    vecdata_i_membrane_ = nullptr;
	vecdata_area = nullptr;
	vecdata_parent_index = nullptr;
	mech_cap = nullptr;
    vecdata_stride = nullptr;
    vecdata_firstnode = nullptr;
    vecdata_lastnode = nullptr;
    vecdata_cellsize = nullptr;
    vecdata_map_t2c = nullptr;
    vecdata_max_order_each_thread = nullptr;
    vecdata_min_order_each_thread = nullptr;

	for (int i = 0, j = mechanism_list.size(); i < j; i++)
	{
        Mechanism* mech = mechanism_list[i];
        printf_debug("delete mech[%s]\n",mech->name.c_str());
        mech->cleanUp();
		delete mech;
		mechanism_list[i] = nullptr;
	}
    for (int i=0, j = vec_eion.size(); i < j; i++)
    {
        Mechanism* mech = vec_eion[i];
        printf_debug("delete eion[%s]\n",mech->name.c_str());
        mech->cleanUp();
        delete mech;
        vec_eion[i] = nullptr;
    }

    /*for (int i = 0, j = vec_articell.size(); i < j; i++)
    {
        ArtiCell* articell = vec_articell[i];
        delete articell;
    }*/

}

double* HelioXroupData::getVarPtr(const VarDescriptor& descriptor, Mode mode){
    const std::string& var_name = descriptor.var;
    int node_idx = descriptor.node_or_mech_idx;
    // Note: array_index is ignored for HelioXroupData variables since they are scalar values
    // If in the future some neuron variables become arrays, this logic would need to be updated

    VecData<double> *var_vec = nullptr;
    if(var_name == "v"){
        var_vec = vecdata_v;
    }else if(var_name == "d"){
        var_vec = vecdata_d;
    }else if(var_name == "rhs"){
        var_vec = vecdata_rhs;
    }else if(var_name == "i_membrane_"){
        var_vec = vecdata_i_membrane_;
    }else if(var_name == "a"){
        var_vec = vecdata_a;
    }else if(var_name == "b"){
        var_vec = vecdata_b;
    }else if(var_name == "area"){
        var_vec = vecdata_area;
    }

    if(var_vec == nullptr){
        return nullptr;
    }

    if(permute != nullptr){
        node_idx = permute[node_idx];
    }

    if(mode == CPU){
        return var_vec->get_cpu_data() + node_idx;
    }else{
        return var_vec->get_gpu_data() + node_idx;
    }
}
